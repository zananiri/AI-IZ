"""Decision-date resolution.

Order: meta_verdict_dt -> VerdictsDt -> VerdictDt -> Year. The date-only fields are preferred;
VerdictDt holds a local midnight shifted by a UTC offset ("2003-07-08T19:00" is 9 July), so a
naive VerdictDt with a time is rounded to the nearest midnight, and an offset-aware one is
converted to Israel time. A date before 1948 or before the case was opened
(meta_case_dt) is bogus (e.g. 1920 on a 2003 case): the next field is tried, and the fix logged."""

from __future__ import annotations

import datetime as dt
import re

try:
    from zoneinfo import ZoneInfo

    _IL = ZoneInfo("Asia/Jerusalem")
except Exception:  # noqa: BLE001 -- no tz database: fixed +02:00 (winter time)
    _IL = dt.timezone(dt.timedelta(hours=2))

_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$")
_DMY_RE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def parse_datetime(value) -> tuple[dt.datetime | None, bool, bool]:
    """(datetime, has_time, tz_aware) for the date forms seen in the dataset; (None, ...) if unparseable."""
    if value is None:
        return None, False, False
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        has_time = (value.hour, value.minute, value.second) != (0, 0, 0)
        return value, has_time, value.tzinfo is not None
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day), False, False
    s = str(value).strip()
    if not s or s.lower() in ("nan", "nat", "none", "null"):
        return None, False, False
    try:
        m = _ISO_RE.match(s)
        if m:
            y, mo, d, hh, mm, ss, tz = m.groups()
            out = dt.datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0))
            if tz:
                if tz == "Z":
                    out = out.replace(tzinfo=dt.timezone.utc)
                else:
                    sign = 1 if tz[0] == "+" else -1
                    digits = tz[1:].replace(":", "")
                    out = out.replace(tzinfo=dt.timezone(sign * dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))))
            return out, hh is not None and (int(hh), int(mm or 0)) != (0, 0), tz is not None
        m = _DMY_RE.match(s)
        if m:
            d, mo, y, hh, mm, ss = m.groups()
            out = dt.datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0))
            return out, hh is not None and (int(hh), int(mm or 0)) != (0, 0), False
        m = _COMPACT_RE.match(s)
        if m:
            return dt.datetime(int(m[1]), int(m[2]), int(m[3])), False, False
    except ValueError:
        return None, False, False
    return None, False, False


def to_date(value, shifted_midnight: bool = False) -> dt.date | None:
    """The calendar date. An offset-aware value is converted to Israel time first. With
    shifted_midnight=True (VerdictDt) a naive value with a time is a midnight stored with an offset,
    so it is rounded to the nearest midnight: 19:00 on the 8th is the 9th, 03:00 on the 9th the 9th."""
    parsed, has_time, aware = parse_datetime(value)
    if parsed is None:
        return None
    if aware:
        return parsed.astimezone(_IL).date()
    if shifted_midnight and has_time and parsed.hour >= 12:
        return parsed.date() + dt.timedelta(days=1)
    return parsed.date()


def to_year(value) -> int | None:
    try:
        year = int(str(value).strip()[:4])
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= 2999 else None


def resolve_decision_date(fields: dict, min_valid: dt.date) -> tuple[dt.date | None, str | None, list[str]]:
    """fields: meta_verdict_dt, VerdictsDt, VerdictDt, case_dt, year (raw values, any may be None).
    Returns (date, the field it came from, notes on the values rejected)."""
    notes: list[str] = []
    case_date = to_date(fields.get("case_dt"))
    if case_date is not None and case_date < min_valid:
        case_date = None  # a bogus case date can't veto a decision date
    for name in ("meta_verdict_dt", "VerdictsDt", "VerdictDt"):
        raw = fields.get(name)
        if raw is None:
            continue
        date = to_date(raw, shifted_midnight=(name == "VerdictDt"))
        if date is None:
            notes.append(f"{name}={raw!s} unparseable")
            continue
        if date < min_valid:
            notes.append(f"{name}={date} before {min_valid}")
            continue
        if case_date is not None and date < case_date:
            notes.append(f"{name}={date} before case date {case_date}")
            continue
        return date, name, notes
    year = to_year(fields.get("year"))
    if year is not None:
        if year < min_valid.year:
            notes.append(f"Year={year} before {min_valid.year}")
        elif case_date is not None and year < case_date.year:
            notes.append(f"Year={year} before case year {case_date.year}")
        else:
            return dt.date(year, 1, 1), "Year", notes
    return None, None, notes
