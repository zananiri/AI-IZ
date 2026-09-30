"""Parse a judgment's header block: court, panel, parties, and where the body starts.

A typical Supreme Court document opens:

    בבית המשפט העליון בשבתו כבית משפט גבוה לצדק
    בג"ץ 5856/03
    בפני: כבוד השופטת ד' דורנר
          כבוד השופטת מ' נאור
    העותרת: פלונית
    נ ג ד
    המשיבים: 1. שר הפנים
             2. ...
    בשם העותרת: עו"ד ...
    פסק-דין
    1. ...

Anything that doesn't fit leaves the field empty; the caller falls back to the dataset metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COURT_RE = re.compile(r"^\s*(?:ב?בית[\s\-]+(?:ה)?משפט[\s\-]+העליון|in the supreme court)", re.I)
COURT_SPLIT_RE = re.compile(r"^\s*ב?בית[\s\-]+(?:ה)?משפט\s*$")
SITTING_RE = re.compile(r"^\s*(?:בשבתו|sitting as)", re.I)
PANEL_RE = re.compile(r"^\s*(?:בפני|לפני|before)\s*:?\s*(.*)$", re.I)
VERSUS_RE = re.compile(r"^\s*[-–]?\s*(?:נ\s*ג\s*ד|נגד|נ'|v\.?|vs\.?|versus|against)\s*[-–]?\s*$", re.I)
ROLE_RE = re.compile(
    r"^\s*(?:ה?(?:עותר|מערער|מבקש|משיב|נאשם|תובע|נתבע|מאשים|מאשימה|עורר|נילון)(?:ת|ה|ים|ות)?"
    r"(?:\s+(?:ב|מס['׳]?\s*)?\d+)?|(?:the\s+)?(?:petitioners?|appellants?|applicants?|respondents?))\s*:?\s*", re.I)
ROLE_ONLY_RE = re.compile(ROLE_RE.pattern + r"$", re.I)
COUNSEL_RE = re.compile(r"^\s*(?:בשם|ב\"כ|ב״כ|ע\"י|ע״י|עו\"ד|עו״ד|על ידי|for the|on behalf of|תאריך|תאריכי|ישיבה|בקשה ל|ערעור על|עתירה ל|מ?לשכת)", re.I)
TITLE_RE = re.compile(
    r"^\s*(?:פסק[\s\-־]*דין|פ\s*ס\s*ק\s*[\-־]?\s*ד\s*י\s*ן|החלטה|ה\s*ח\s*ל\s*ט\s*ה|צו(?:\s+(?:ביניים|על[\s\-]+תנאי))?"
    r"|judgment|decision|order)\s*:?\s*$", re.I)
FIRST_PARA_RE = re.compile(r"^\s*1\.\s")
JUDGE_TITLE_RE = re.compile(
    r"כבוד|כב['׳]|ה?שופט(?:ת|ים)?|ה?נשיא(?:ה)?|המשנה\s+ל?נשיא(?:ה)?|ממלא(?:ת)?\s+מקום|ה?רשמ(?:ת|ים)?|"
    r"\bjustices?\b|\bpresident\b|\bdeputy\b|\bhon\.?\b|\bthe\b|\bregistrar\b|\(בדימ['׳]?\)|:", re.I)
# The case number under the court line, whole or split: 'בג"ץ 5856/03', or 'בג"ץ' then '5856/03 - י\''.
CITATION_LINE_RE = re.compile(r"^\s*(?:[א-ת]{1,4}[\"״][א-ת]{1,2}\s*(?:\d|$)|\d{1,6}/\d{2,4}\b)")
_ITEM_RE = re.compile(r"(?:^|\s)\d{1,2}\s*[.)]\s+")


@dataclass
class Header:
    court: str | None = None
    judges: list[str] = field(default_factory=list)
    parties: list[str] = field(default_factory=list)
    body_start: int = 0
    text: str = ""
    parsed: bool = False


def _lines_with_offsets(text: str, limit: int) -> list[tuple[int, str]]:
    out, pos = [], 0
    for line in text[:limit].split("\n"):
        out.append((pos, line))
        pos += len(line) + 1
    return out


def clean_judge(name: str) -> str:
    name = JUDGE_TITLE_RE.sub(" ", name)
    return " ".join(name.replace(",", " ").split()).strip(" -–.")


def _split_items(text: str) -> list[str]:
    parts = [p.strip(" ,;.-–") for p in _ITEM_RE.split(text)]
    return [p for p in parts if p]


def clean_parties(lines: list[str]) -> list[str]:
    names: list[str] = []
    for line in lines:
        if not line.strip() or COUNSEL_RE.match(line) or ROLE_ONLY_RE.match(line):
            continue
        line = ROLE_RE.sub("", line, count=1)
        names += _split_items(line)
    return [n for n in names if len(n) > 1][:30]


def _next_nonblank(lines: list[tuple[int, str]], i: int) -> int | None:
    for j in range(i + 1, min(i + 4, len(lines))):
        if lines[j][1].strip():
            return j
    return None


def parse_header(text: str, max_chars: int = 6000) -> Header:
    lines = _lines_with_offsets(text, max_chars)
    header = Header()
    court_idx = panel_idx = versus_idx = title_idx = None
    for i, (_, line) in enumerate(lines):
        if court_idx is None and (COURT_RE.match(line) or COURT_SPLIT_RE.match(line)):
            nxt = _next_nonblank(lines, i)
            if COURT_RE.match(line):
                header.court = line.strip()
            elif nxt is not None and lines[nxt][1].strip().startswith("העליון"):
                # The crawl's layout splits the court line: "בבית המשפט" / "העליון".
                header.court = f"{line.strip()} {lines[nxt][1].strip()}"
                nxt = _next_nonblank(lines, nxt)
            else:
                continue
            court_idx = i
            if nxt is not None and SITTING_RE.match(lines[nxt][1]):
                header.court += " " + lines[nxt][1].strip()
        elif panel_idx is None and PANEL_RE.match(line):
            panel_idx = i
        elif versus_idx is None and VERSUS_RE.match(line):
            versus_idx = i
        elif TITLE_RE.match(line) and (versus_idx is not None or panel_idx is not None):
            title_idx = i
            break
        elif (FIRST_PARA_RE.match(line) and len(line) > 80  # a party list "1. שר הפנים" is short
              and (versus_idx is not None or panel_idx is not None)):
            title_idx = i - 1  # no title line: the body starts at paragraph 1
            break

    if panel_idx is not None:
        first = PANEL_RE.match(lines[panel_idx][1]).group(1)
        panel = [first] if first.strip() else []
        panel_end = panel_idx + 1
        for j in range(panel_idx + 1, len(lines)):
            stripped = lines[j][1].strip()
            if not stripped:
                continue
            if JUDGE_TITLE_RE.search(stripped) and not ROLE_RE.match(stripped) and not VERSUS_RE.match(stripped):
                panel.append(stripped)
                panel_end = j + 1  # blank lines between judges don't shift where the parties start
            else:
                break
        header.judges = [j for j in (clean_judge(p) for p in panel) if j and len(j) < 60]
    else:
        panel_end = (court_idx + 1) if court_idx is not None else 0

    if versus_idx is not None:
        end = title_idx if title_idx is not None else min(versus_idx + 12, len(lines))
        side_a = [line for _, line in lines[panel_end:versus_idx] if not PANEL_RE.match(line)]
        # Skip the citation line(s) right under the court line (בג"ץ 5856/03).
        side_a = [line for line in side_a if not CITATION_LINE_RE.match(line) and "העליון" not in line]
        side_b = [line for _, line in lines[versus_idx + 1:end]]
        header.parties = clean_parties(side_a) + clean_parties(side_b)

    if title_idx is not None:
        body_line = title_idx + 1
        while body_line < len(lines) - 1 and not lines[body_line][1].strip():
            body_line += 1
        if body_line < len(lines):
            header.body_start = lines[body_line][0]
        header.text = text[:header.body_start].strip()
    header.parsed = bool(header.court and (header.judges or header.parties) and header.body_start > 0)
    return header
