"""Knesset OData V4 (https://knesset.gov.il/OdataV4/ParliamentInfo) through PoliteClient.

The Knesset firewall rejects any query containing ";" or an $apply=aggregate(...): every URL is
checked before it is sent, and a server-generated @odata.nextLink containing ";" switches the
listing to $orderby=Id&$skip paging instead. Delta syncs filter on LastUpdatedDate (inclusive:
rows are upserted by Id, so a re-seen row is harmless).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit

from docslides.legal_data.http import PoliteClient

_FORBIDDEN_RE = re.compile(r";|aggregate\s*\(", re.IGNORECASE)
_QUERY_SAFE = "$'(),:-"  # left unescaped in query values: OData literals read better, and none is ';'


class ForbiddenQuery(ValueError):
    pass


def guard(url: str) -> str:
    parts = urlsplit(url)
    if _FORBIDDEN_RE.search(parts.query) or ";" in parts.path:
        raise ForbiddenQuery(f"refusing to send {url}: the Knesset firewall rejects ';' and aggregate()")
    return url


def odata_datetime(value: str) -> str:
    """An ISO timestamp (any offset) as the UTC literal OData filters take: 2026-09-01T10:00:00Z."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class KnessetOData:
    def __init__(self, http: PoliteClient, base: str) -> None:
        self.http = http
        self.base = base.rstrip("/")

    def url(self, path: str, params: dict | None = None) -> str:
        query = "&".join(f"{key}={quote(str(value), safe=_QUERY_SAFE)}" for key, value in (params or {}).items())
        return guard(f"{self.base}/{path}" + (f"?{query}" if query else ""))

    def entity_sets(self) -> list[str]:
        data = self.http.get_json(guard(self.base + "/"))
        return [entry["name"] for entry in data.get("value", [])]

    @staticmethod
    def _filter(since: str | None) -> dict:
        return {"$filter": f"LastUpdatedDate ge {odata_datetime(since)}"} if since else {}

    def count(self, table: str, since: str | None = None) -> int:
        """Rows in a table (since a watermark). The service has no /$count path (404): ask for
        $count=true on a one-row page instead."""
        data = self.http.get_json(self.url(table, {**self._filter(since), "$count": "true", "$top": 1}))
        return int(data.get("@odata.count", len(data.get("value", []))))

    def rows(self, table: str, since: str | None = None, top: int | None = None) -> Iterator[dict]:
        params = self._filter(since)
        if top:
            params["$top"] = top
        url = self.url(table, params)
        fetched = 0
        while url:
            data = self.http.get_json(url)
            for row in data.get("value", []):
                yield row
                fetched += 1
                if top and fetched >= top:
                    return
            next_link = data.get("@odata.nextLink")
            if not next_link:
                return
            if _FORBIDDEN_RE.search(urlsplit(next_link).query):
                # Can't follow it. Start the listing over in Id order: the server's default order
                # may not be by Id, and rows seen twice are upserted by Id anyway.
                yield from self._rows_by_skip(table, params, top=top)
                return
            url = guard(next_link)

    def _rows_by_skip(self, table: str, params: dict, top: int | None) -> Iterator[dict]:
        """Paging without the server's nextLink, ordered by Id so pages don't overlap."""
        params = {k: v for k, v in params.items() if k != "$top"}
        skip = fetched = 0
        while True:
            page = self.http.get_json(self.url(table, {**params, "$orderby": "Id", "$skip": skip})).get("value", [])
            if not page:
                return
            for row in page:
                yield row
                fetched += 1
                if top and fetched >= top:
                    return
            skip += len(page)
