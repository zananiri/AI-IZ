"""Polite HTTP for the corpus fetchers (scripts/legal_data/).

Every request goes through PoliteClient:
  * robots.txt is read for each host before its first request, and a disallowed URL raises
    RobotsDisallowed. A missing robots.txt (404/410) allows everything; one that answers
    401/403, 5xx or not at all is treated as "disallow all" and reported.
  * At most one request per `min_interval_s` per host (robots.txt fetches included), longer if
    robots.txt sets a Crawl-delay.
  * 429 and 5xx back off exponentially with jitter, honouring Retry-After.
  * 401/403, or an HTML block page ("Access Denied", "Request Rejected", a challenge page),
    raises AccessDenied. So does a host that keeps resetting the connection (HostUnavailable).
    The fetcher stops and reports: nothing here retries a block, changes identity or routes
    around one.
  * Redirects are followed by hand, so every hop's host gets the same checks.
  * The User-Agent names the project and the contact email from config (legal_data.contact_email);
    without one the client refuses to start.
Downloads stream to `<file>.part` (resumed with a Range request), are hashed while written,
verified against an expected hash when the source publishes one, and renamed into place.
The ledger (_manifests/_downloads.json) lets unchanged files be skipped without a request.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlsplit

import httpx

from aicodex.config import LegalDataConfig
from aicodex.legal_data.progress import Progress

USER_AGENT_NAME = "AI-Codex-legal-corpus/1.0"
USER_AGENT_TOKEN = "AI-Codex-legal-corpus"  # what a robots.txt group would name us by
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_BLOCK_PAGE_RE = re.compile(
    r"access denied|request rejected|the requested url was rejected|attention required|cf-browser-verification"
    r"|cf-chl-|captcha|you don't have permission|forbidden",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BLOCK_PAGE_MAX_CHARS = 20_000  # block pages are short; a long page merely mentioning "captcha" isn't one
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5
_CONNECTION_RETRIES = 2  # a host that keeps resetting the connection may be blocking us: stop, don't insist


class FetchStopped(Exception):
    """The fetcher must stop and report (never work around it)."""


class MissingContactEmail(FetchStopped):
    pass


class RobotsDisallowed(FetchStopped):
    pass


class AccessDenied(FetchStopped):
    pass


class HostUnavailable(FetchStopped):
    pass


class FetchError(Exception):
    """A single request failed (e.g. 404) -- the caller decides whether that is fatal."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class RobotsRules:
    """robots.txt with the matching rules of RFC 9309 (and Google's): `*` matches any run of
    characters and a trailing `$` anchors the end; the longest matching rule wins, Allow on a tie.
    The group for our own token is used if there is one, else the `*` group. The standard library's
    parser ignores wildcards -- and the BOE lists the documents it has withdrawn (right to be
    forgotten) exactly that way ("Disallow: /*BOE-A-2026-4920$")."""

    def __init__(self, text: str = "", agent_token: str = USER_AGENT_TOKEN) -> None:
        self.disallow_all = False
        groups: list[tuple[list[str], list[tuple[bool, str]], float | None]] = []
        agents: list[str] = []
        rules: list[tuple[bool, str]] = []
        delay: float | None = None
        in_rules = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            field, value = (part.strip() for part in line.split(":", 1))
            field = field.lower()
            if field == "user-agent":
                if in_rules:
                    groups.append((agents, rules, delay))
                    agents, rules, delay, in_rules = [], [], None, False
                agents.append(value.lower())
            elif field in ("allow", "disallow"):
                in_rules = True
                if value:
                    rules.append((field == "allow", value))
            elif field == "crawl-delay":
                in_rules = True
                try:
                    delay = float(value)
                except ValueError:
                    pass
        if agents:
            groups.append((agents, rules, delay))
        token = agent_token.lower()
        own = [g for g in groups if any(a != "*" and a in token for a in g[0])]
        chosen = own or [g for g in groups if "*" in g[0]]
        self.rules = [r for g in chosen for r in g[1]]
        delays = [g[2] for g in chosen if g[2] is not None]
        self.delay = max(delays) if delays else None

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        anchored = pattern.endswith("$")
        body = pattern[:-1] if anchored else pattern
        regex = "".join(".*" if c == "*" else re.escape(c) for c in body)
        return re.match(regex + ("$" if anchored else ""), path) is not None

    def can_fetch(self, user_agent: str, url: str) -> bool:
        if self.disallow_all:
            return False
        parts = urlsplit(url)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        best: tuple[int, bool] | None = None
        for allow, pattern in self.rules:
            if self._matches(pattern, path):
                candidate = (len(pattern), allow)
                if best is None or candidate > best:
                    best = candidate
        return True if best is None else best[1]

    def crawl_delay(self, user_agent: str) -> float | None:
        return self.delay


class DownloadLedger:
    """URL -> {path, sha256, bytes, etag, last_modified, downloaded_at}."""

    def __init__(self, path: Path, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only
        self._entries: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, url: str) -> dict | None:
        return self._entries.get(url)

    def record(self, url: str, **info) -> None:
        if self.read_only:
            return
        self._entries[url] = {**info, "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, json.dumps(self._entries, ensure_ascii=False, indent=1))


@dataclass
class DownloadResult:
    url: str
    path: Path
    bytes: int
    sha256: str
    status: Literal["downloaded", "skipped"]


class PoliteClient:
    def __init__(
        self,
        cfg: LegalDataConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        email_address = cfg.contact_email.strip()
        if not _EMAIL_RE.fullmatch(email_address):
            raise MissingContactEmail(
                "legal_data.contact_email is not set (config/config.yaml, or the "
                "AICODEX_LEGAL_DATA_CONTACT_EMAIL environment variable): the fetchers identify themselves "
                "with a contact address and won't send requests without one."
            )
        self.user_agent = f"{USER_AGENT_NAME} (offline Spanish legal research corpus; +mailto:{email_address})"
        self._cfg = cfg
        self._sleep = sleep
        self._clock = clock
        self._log = log or (lambda message: None)
        self._http = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=cfg.timeout_s,
            follow_redirects=False,
            transport=transport,
        )
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotsRules | None] = {}  # None = everything allowed
        self._crawl_delay: dict[str, float] = {}
        self.requests_per_host: Counter[str] = Counter()
        self.robots_log: list[dict] = []

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- pacing ---------------------------------------------------------------------------

    def _pace(self, host: str) -> None:
        interval = max(self._cfg.min_interval_s, self._crawl_delay.get(host, 0.0))
        last = self._last_request.get(host)
        if last is not None:
            wait = last + interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    def _backoff(self, attempt: int, retry_after: str | None, url: str, reason: str) -> None:
        delay = None
        if retry_after:
            if retry_after.strip().isdigit():
                delay = float(retry_after.strip())
            else:
                parsed = email.utils.parsedate_to_datetime(retry_after)
                if parsed is not None:
                    delay = (parsed - datetime.now(parsed.tzinfo or timezone.utc)).total_seconds()
        if delay is None or delay < 0:
            delay = self._cfg.backoff_base_s * 2 ** (attempt - 1) * (1 + random.random() * 0.25)
        delay = min(delay, self._cfg.backoff_max_s)
        self._log(f"backing off {delay:.0f}s after {reason} (attempt {attempt}): {url}")
        self._sleep(delay)

    # --- robots.txt -----------------------------------------------------------------------

    def _robots_for(self, scheme: str, host: str) -> RobotsRules | None:
        if host in self._robots:
            return self._robots[host]
        robots_url = f"{scheme}://{host}/robots.txt"
        parser: RobotsRules | None = RobotsRules()
        try:
            response = self._send("GET", robots_url)
            status = response.status_code
            if status == 200:
                parser = RobotsRules(response.text)
                decision = "rules"
            elif status in (401, 403) or status >= 500:
                parser.disallow_all = True
                decision = f"disallow all (robots.txt answered HTTP {status})"
            else:  # 404, 410 and other 4xx: no rules
                parser = None
                decision = f"allow all (robots.txt HTTP {status})"
        except (AccessDenied, HostUnavailable) as exc:
            parser.disallow_all = True
            decision = f"disallow all (robots.txt unreachable: {exc})"
        self._robots[host] = parser
        if parser is not None and not parser.disallow_all:
            delay = parser.crawl_delay(self.user_agent)
            if delay:
                self._crawl_delay[host] = float(delay)
        self.robots_log.append({"host": host, "url": robots_url, "decision": decision,
                                "crawl_delay": self._crawl_delay.get(host)})
        self._log(f"robots.txt {host}: {decision}")
        return parser

    def disallowed_tokens(self, url: str, pattern: str) -> set[str]:
        """Every match of `pattern` inside the Disallow rules of `url`'s host -- e.g. the BOE ids a
        robots.txt withdraws ("Disallow: /*BOE-A-2026-4920$"), checked once instead of matching
        thousands of rules against every catalogue entry."""
        parts = urlsplit(url)
        parser = self._robots_for(parts.scheme, parts.hostname or "")
        if parser is None:
            return set()
        regex = re.compile(pattern)
        return {m for allow, rule in parser.rules if not allow for m in regex.findall(rule)}

    def allowed(self, url: str) -> bool:
        """Whether robots.txt lets us fetch `url` (reads the host's robots.txt once; no other request)."""
        parts = urlsplit(url)
        parser = self._robots_for(parts.scheme, parts.hostname or "")
        return parser is None or parser.can_fetch(self.user_agent, url)

    def check_robots(self, url: str) -> None:
        parts = urlsplit(url)
        parser = self._robots_for(parts.scheme, parts.hostname or "")
        if parser is not None and not parser.can_fetch(self.user_agent, url):
            raise RobotsDisallowed(f"robots.txt of {parts.hostname} disallows {url}")

    # --- requests -------------------------------------------------------------------------

    def _send(self, method: str, url: str, headers: dict | None = None, stream: bool = False) -> httpx.Response:
        host = urlsplit(url).hostname or ""
        attempt = connection_failures = 0
        while True:
            self._pace(host)
            try:
                response = self._http.send(self._http.build_request(method, url, headers=headers), stream=stream)
            except httpx.TransportError as exc:
                connection_failures += 1
                if connection_failures > _CONNECTION_RETRIES:
                    raise HostUnavailable(
                        f"{url}: connection failed {connection_failures} times ({exc!r}). Stopping -- a host that "
                        "keeps resetting connections may be refusing us."
                    ) from exc
                self._backoff(connection_failures, None, url, type(exc).__name__)
                continue
            self.requests_per_host[host] += 1
            status = response.status_code
            if status in (401, 403):
                response.close()
                raise AccessDenied(f"{url}: HTTP {status} -- stopping, as required; no retry")
            if status in _RETRY_STATUSES:
                attempt += 1
                retry_after = response.headers.get("retry-after")
                response.close()
                if attempt > self._cfg.max_retries:
                    raise HostUnavailable(f"{url}: HTTP {status} after {self._cfg.max_retries} retries")
                self._backoff(attempt, retry_after, url, f"HTTP {status}")
                continue
            if not stream and _is_block_page(response):
                raise AccessDenied(f"{url}: the server returned an access-denied page -- stopping")
            return response

    def request(self, method: str, url: str, headers: dict | None = None, stream: bool = False,
                follow_redirects: bool = True) -> httpx.Response:
        for _ in range(_MAX_REDIRECTS + 1):
            self.check_robots(url)
            response = self._send(method, url, headers, stream)
            if follow_redirects and response.status_code in _REDIRECTS and "location" in response.headers:
                response.close()
                url = urljoin(url, response.headers["location"])
                continue
            return response
        raise HostUnavailable(f"{url}: more than {_MAX_REDIRECTS} redirects")

    def get(self, url: str, ok: tuple[int, ...] = (200,), headers: dict | None = None,
            follow_redirects: bool = True) -> httpx.Response:
        response = self.request("GET", url, headers=headers, follow_redirects=follow_redirects)
        if response.status_code not in ok:
            raise FetchError(f"{url}: HTTP {response.status_code}")
        return response

    def get_json(self, url: str, headers: dict | None = None) -> dict:
        return self.get(url, headers={"Accept": "application/json", **(headers or {})}).json()

    def head(self, url: str) -> httpx.Response | None:
        """HEAD for size estimates; None if the server doesn't support it."""
        response = self.request("HEAD", url)
        return response if response.status_code == 200 else None

    def stream(self, url: str) -> Iterator[bytes]:
        """The body of a GET as it arrives (sample runs read only a prefix of large files)."""
        response = self.request("GET", url, stream=True)
        try:
            if response.status_code != 200:
                raise FetchError(f"{url}: HTTP {response.status_code}")
            _refuse_block_page_stream(response, url)
            yield from response.iter_bytes(1 << 16)
        finally:
            response.close()

    def download(
        self,
        url: str,
        dest: Path,
        ledger: DownloadLedger,
        *,
        expected_sha256: str | None = None,
        expected_sha1: str | None = None,
        progress_label: str | None = None,
    ) -> DownloadResult:
        """Resumable, hash-verified download. An existing file whose hash matches the expected one
        (or, without one, the ledger's) is skipped without any request. With `progress_label`, a
        progress line (MB, %, speed, ETA) is logged every few seconds."""
        if dest.exists():
            known = ledger.get(url)
            if expected_sha1 and _file_sha1(dest) == expected_sha1:
                return DownloadResult(url, dest, dest.stat().st_size, file_sha256(dest), "skipped")
            if not expected_sha1:
                digest = file_sha256(dest)
                if digest == expected_sha256 or (not expected_sha256 and known and known.get("sha256") == digest):
                    return DownloadResult(url, dest, dest.stat().st_size, digest, "skipped")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else None
        response = self.request("GET", url, headers=headers, stream=True)
        sha256, sha1 = hashlib.sha256(), hashlib.sha1()
        try:
            if response.status_code == 200 and offset:
                offset = 0  # the server ignored the Range: start over
            elif response.status_code == 416 and offset:
                response.close()
                part.unlink()
                return self.download(url, dest, ledger, expected_sha256=expected_sha256, expected_sha1=expected_sha1,
                                     progress_label=progress_label)
            elif response.status_code not in (200, 206):
                raise FetchError(f"{url}: HTTP {response.status_code}")
            _refuse_block_page_stream(response, url)
            if offset:
                with open(part, "rb") as existing:
                    for block in iter(lambda: existing.read(1 << 20), b""):
                        sha256.update(block)
                        sha1.update(block)
            length = response.headers.get("content-length")
            progress = Progress(progress_label, total=offset + int(length) if length and length.isdigit() else None,
                                unit="bytes", log=self._log) if progress_label else None
            if progress:
                progress.set(offset)
            with open(part, "ab" if offset else "wb") as out:
                for block in response.iter_bytes(1 << 20):
                    out.write(block)
                    sha256.update(block)
                    sha1.update(block)
                    if progress:
                        progress.update(len(block))
            if progress:
                progress.done()
            etag, modified = response.headers.get("etag"), response.headers.get("last-modified")
        finally:
            response.close()
        if expected_sha256 and sha256.hexdigest() != expected_sha256 or expected_sha1 and sha1.hexdigest() != expected_sha1:
            part.unlink()
            raise FetchError(f"{url}: downloaded file doesn't match the published hash -- deleted, re-run to retry")
        part.replace(dest)
        size = dest.stat().st_size
        ledger.record(url, path=str(dest), sha256=sha256.hexdigest(), bytes=size, etag=etag, last_modified=modified)
        return DownloadResult(url, dest, size, sha256.hexdigest(), "downloaded")


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def looks_like_block_page(html: str) -> bool:
    title = _TITLE_RE.search(html[:4096])
    if title and _BLOCK_PAGE_RE.search(title.group(1)):
        return True
    return len(html) <= _BLOCK_PAGE_MAX_CHARS and bool(_BLOCK_PAGE_RE.search(html))


def _is_block_page(response: httpx.Response) -> bool:
    if "html" not in response.headers.get("content-type", ""):
        return False
    return looks_like_block_page(response.text)


def _refuse_block_page_stream(response: httpx.Response, url: str) -> None:
    """A binary download that comes back as an HTML page is a block page or an error page."""
    if "html" in response.headers.get("content-type", ""):
        html = response.read().decode("utf-8", errors="replace")
        if looks_like_block_page(html):
            raise AccessDenied(f"{url}: the server returned an access-denied page -- stopping")
        raise FetchError(f"{url}: expected a file, got an HTML page")
