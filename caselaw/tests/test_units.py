import datetime as dt

from israeli_caselaw_ingest.chunk import Unit, pack
from israeli_caselaw_ingest.citations import extract_citations
from israeli_caselaw_ingest.dates import resolve_decision_date, to_date
from israeli_caselaw_ingest.encoding import normalize, repair_text
from israeli_caselaw_ingest.header import parse_header

MIN = dt.date(1948, 1, 1)


# -- encoding repair ------------------------------------------------------------------------------

def test_latin1_mojibake_is_repaired():
    fixed, method = repair_text("á áéú äîùôè äòìéåï")
    assert fixed == "ב בית המשפט העליון" and method in ("latin1", "cp1252")


def test_greek_mojibake_is_repaired():
    fixed, method = repair_text("α αιϊ δξωτθ δςμιεο")
    assert fixed == "ב בית המשפט העליון" and method == "cp1253"


def test_good_text_and_accented_names_are_left_alone():
    text = "בבית המשפט העליון\nראו Café de Paris v. Société Générale"
    assert repair_text(text) == (text, "none")


def test_only_the_garbled_lines_are_repaired():
    fixed, _ = repair_text("פסק-דין\ná áéú äîùôè")
    assert fixed == "פסק-דין\nב בית המשפט"


def test_normalize_keeps_paragraphs_and_drops_bidi_marks():
    assert normalize("א‏   ב‫\n\n\n\nג    ד  ") == "א ב\n\nג ד"


# -- dates ----------------------------------------------------------------------------------------

def test_date_only_field_wins():
    d, src, notes = resolve_decision_date({"meta_verdict_dt": "2003-07-09", "VerdictDt": "2003-07-08T19:00"}, MIN)
    assert (d, src, notes) == (dt.date(2003, 7, 9), "meta_verdict_dt", [])


def test_bogus_1920_date_falls_back():
    d, src, notes = resolve_decision_date(
        {"meta_verdict_dt": "1920-07-09", "VerdictDt": "2003-07-08T19:00", "case_dt": "2002-05-01", "year": 2003}, MIN)
    assert d == dt.date(2003, 7, 9) and src == "VerdictDt" and "before 1948" in notes[0]


def test_date_before_the_case_was_opened_falls_back_to_year():
    d, src, notes = resolve_decision_date({"meta_verdict_dt": "1999-01-01", "case_dt": "2003-02-01", "year": 2004}, MIN)
    assert (d, src) == (dt.date(2004, 1, 1), "Year") and "before case date" in notes[0]


def test_verdictdt_evening_is_the_next_day_and_formats_parse():
    assert to_date("2003-07-08T19:00", shifted_midnight=True) == dt.date(2003, 7, 9)
    assert to_date("2021-12-31T21:00", shifted_midnight=True) == dt.date(2022, 1, 1)
    assert to_date("2003-07-09T03:00", shifted_midnight=True) == dt.date(2003, 7, 9)
    assert to_date("09/07/2003") == dt.date(2003, 7, 9)
    assert to_date(dt.datetime(2003, 7, 9)) == dt.date(2003, 7, 9)


def test_unresolvable_date():
    assert resolve_decision_date({"meta_verdict_dt": "garbage"}, MIN)[0] is None


# -- header ---------------------------------------------------------------------------------------

HEADER = """בבית המשפט העליון בירושלים
בשבתו כבית משפט גבוה לצדק

בג"ץ 5856/03

בפני: כבוד השופטת ד' דורנר
          כבוד השופטת מ' נאור
          כבוד השופטת א' חיות

העותרת: פלונית אלמונית

נ ג ד

המשיבים: 1. שר הפנים
2. רשות האוכלוסין

בשם העותרת: עו"ד משה כהן

פסק-דין

1. העתירה שלפנינו עניינה בחובת ההנמקה."""


def test_header_parses_court_panel_parties_and_body_start():
    h = parse_header(HEADER)
    assert h.parsed
    assert h.court == "בבית המשפט העליון בירושלים בשבתו כבית משפט גבוה לצדק"
    assert h.judges == ["ד' דורנר", "מ' נאור", "א' חיות"]
    assert h.parties == ["פלונית אלמונית", "שר הפנים", "רשות האוכלוסין"]
    assert HEADER[h.body_start:].startswith("1. העתירה")
    assert h.text.endswith("פסק-דין")


def test_unparseable_header_falls_back():
    h = parse_header("טקסט ללא כותרת כלל.\n1. פסקה.")
    assert not h.parsed and h.body_start == 0


# -- citations ------------------------------------------------------------------------------------

def test_citation_regex():
    text = ('ראו ע"א 6821/93 בנק המזרחי, פ"ד מט(4) 221; בג״ץ 5856/03; ת"א (ת"א) 1234/98; '
            'ע"פ (מחוזי ב"ש) 7/20; רע"א 12345-06-15; פ"ד ד 123. סכום של 500 ש"ח ו-ס"ק 3/4.')
    assert extract_citations(text) == ['ע"א 6821/93', 'בג"ץ 5856/03', 'ת"א (ת"א) 1234/98', 'ע"פ (מחוזי ב"ש) 7/20',
                                       'רע"א 12345-06-15', 'פ"ד מט(4) 221', 'פ"ד ד 123']


# -- chunk packing --------------------------------------------------------------------------------

def _units(sizes, para_every=1):
    out, pos = [], 0
    for i, n in enumerate(sizes):
        out.append(Unit(start=pos, end=pos + 10, para_start=i % para_every == 0, tokens=n))
        pos += 11
    return out


def test_chunks_stay_under_max_and_overlap_whole_units():
    units = _units([60] * 20, para_every=3)
    chunks = pack(units, min_tokens=150, max_tokens=200, overlap_tokens=70)
    assert all(sum(u.tokens for u in c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        shared = [u for u in b if u in a]
        assert sum(u.tokens for u in shared) <= 70
        assert not shared or b[:len(shared)] == a[-len(shared):]  # overlap = the tail of the previous chunk
    covered = {id(u) for c in chunks for u in c}
    assert covered == {id(u) for u in units}


def test_holding_starts_a_fresh_chunk_without_overlap():
    units = _units([50] * 6)
    for u in units[4:]:
        u.holding = True
    units[4].holding_start = True
    chunks = pack(units, min_tokens=400, max_tokens=600, overlap_tokens=80)
    assert [len(c) for c in chunks] == [4, 2] and all(u.holding for u in chunks[1])
