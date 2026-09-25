"""The corpus fetchers without the network: polite HTTP (robots.txt, pacing, backoff, the stop on
403 / block pages, resumable downloads), the OData firewall guard and paging, the Open Law Book
parser and dump reader, registry joins, Hebrew repair, the Supreme Court keep rules, privacy
filter and anonymizer, the JSONL writer, and the guard that keeps ingest_legal_txt.py out of the
corpus folders."""

import bz2
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from docslides.config import LegalDataConfig, get_config
from docslides.legal_data import legislation, supreme_court, wikisource
from docslides.legal_data.hebrew import normalize_for_embedding, normalize_for_index, repair_text
from docslides.legal_data.http import (
    AccessDenied,
    DownloadLedger,
    HostUnavailable,
    MissingContactEmail,
    PoliteClient,
    RobotsDisallowed,
)
from docslides.legal_data.knesset_odata import ForbiddenQuery, KnessetOData, guard, odata_datetime
from docslides.legal_data.law_names import NameIndex, has_prefix, name_key
from docslides.legal_data.records import JsonlWriter, read_jsonl, record_hash

# --- polite HTTP ------------------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def make_client(handler, **overrides):
    cfg = LegalDataConfig(contact_email="corpus@example.org", min_interval_s=1.0, max_retries=2,
                          backoff_base_s=1.0, backoff_max_s=5.0, **overrides)
    clock = FakeClock()
    client = PoliteClient(cfg, transport=httpx.MockTransport(handler), sleep=clock.sleep, clock=clock.now)
    return client, clock


def no_robots(request, respond):
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    return respond(request)


def test_refuses_to_run_without_a_contact_email():
    with pytest.raises(MissingContactEmail):
        PoliteClient(LegalDataConfig(contact_email=""))


def test_user_agent_names_the_project_and_contact():
    client, _ = make_client(lambda r: httpx.Response(200))
    assert "AI-IZ-legal-corpus" in client.user_agent and "corpus@example.org" in client.user_agent


def test_at_most_one_request_per_second_per_host():
    client, clock = make_client(lambda r: no_robots(r, lambda _: httpx.Response(200, text="ok")))
    client.get("https://a.test/1")
    client.get("https://a.test/2")
    assert client.requests_per_host["a.test"] == 3  # robots.txt + 2
    assert clock.sleeps and all(s >= 1.0 for s in clock.sleeps)


def test_robots_txt_is_honoured():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="ok")

    client, _ = make_client(handler)
    assert client.get("https://a.test/public").text == "ok"
    with pytest.raises(RobotsDisallowed):
        client.get("https://a.test/private/x")
    assert client.robots_log[0]["decision"] == "rules"


def test_403_stops_without_a_retry():
    client, _ = make_client(lambda r: no_robots(r, lambda _: httpx.Response(403)))
    with pytest.raises(AccessDenied):
        client.get("https://a.test/x")
    assert client.requests_per_host["a.test"] == 2  # robots.txt + the one refused request


def test_an_access_denied_page_stops_but_a_long_page_mentioning_captcha_does_not():
    block = '<html><head><title>Request Rejected</title></head><body>support id 123</body></html>'
    long_page = "<html><head><title>Data files</title></head><body>" + "x" * 30_000 + " captcha form</body></html>"
    pages = {"/blocked": block, "/fine": long_page}
    client, _ = make_client(lambda r: no_robots(
        r, lambda req: httpx.Response(200, text=pages[req.url.path], headers={"content-type": "text/html"})))
    with pytest.raises(AccessDenied):
        client.get("https://a.test/blocked")
    assert "Data files" in client.get("https://a.test/fine").text


def test_5xx_backs_off_then_gives_up():
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else httpx.Response(200, text="ok")

    client, clock = make_client(lambda r: no_robots(r, flaky))
    assert client.get("https://a.test/x").text == "ok"
    assert max(clock.sleeps) >= 1.0

    client, _ = make_client(lambda r: no_robots(r, lambda _: httpx.Response(503)))
    with pytest.raises(HostUnavailable):
        client.get("https://a.test/x")


def test_download_resumes_verifies_and_skips_an_unchanged_file(tmp_path):
    content = bytes(range(256)) * 8

    def serve(request):
        start = int(request.headers["range"].split("=")[1].rstrip("-")) if "range" in request.headers else 0
        status = 206 if start else 200
        return httpx.Response(status, content=content[start:], headers={"content-type": "application/pdf"})

    client, _ = make_client(lambda r: no_robots(r, serve))
    ledger = DownloadLedger(tmp_path / "_downloads.json")
    dest = tmp_path / "pdf" / "1.pdf"
    dest.parent.mkdir()
    (dest.parent / "1.pdf.part").write_bytes(content[:400])

    result = client.download("https://a.test/1.pdf", dest, ledger,
                             expected_sha256=hashlib.sha256(content).hexdigest())
    assert result.status == "downloaded" and dest.read_bytes() == content
    before = client.requests_per_host["a.test"]
    assert client.download("https://a.test/1.pdf", dest, ledger).status == "skipped"
    assert client.requests_per_host["a.test"] == before  # no request at all


# --- Knesset OData ----------------------------------------------------------------------------------


def test_queries_with_a_semicolon_or_aggregate_are_never_sent():
    with pytest.raises(ForbiddenQuery):
        guard("https://knesset.test/OdataV4/ParliamentInfo/KNS_Law?$filter=a;b")
    with pytest.raises(ForbiddenQuery):
        guard("https://knesset.test/OdataV4/ParliamentInfo/KNS_Law?$apply=aggregate(Id with sum as n)")
    assert odata_datetime("2025-09-15T16:04:54.6+03:00") == "2025-09-15T13:04:54Z"


def test_rows_follow_next_links_and_fall_back_to_skip_paging():
    requested = []

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        requested.append(str(request.url))
        params = request.url.params
        if request.url.path.endswith("/Good"):
            if "$skiptoken" in params:
                return httpx.Response(200, json={"value": [{"Id": 3}]})
            return httpx.Response(200, json={"value": [{"Id": 1}, {"Id": 2}],
                                             "@odata.nextLink": "https://knesset.test/OdataV4/ParliamentInfo/Good?$skiptoken=2"})
        if "$skip" in params:  # the fallback, ordered by Id
            rows = [{"Id": i} for i in (1, 2, 3)][int(params["$skip"]):]
            return httpx.Response(200, json={"value": rows})
        return httpx.Response(200, json={"value": [{"Id": 1}, {"Id": 2}],
                                         "@odata.nextLink": "https://knesset.test/OdataV4/ParliamentInfo/Bad?$skiptoken=a;b"})

    client, _ = make_client(handler)
    odata = KnessetOData(client, "https://knesset.test/OdataV4/ParliamentInfo")
    assert [r["Id"] for r in odata.rows("Good")] == [1, 2, 3]
    assert {r["Id"] for r in odata.rows("Bad")} == {1, 2, 3}
    assert not any(";" in url for url in requested)


# --- the Open Law Book ------------------------------------------------------------------------------

WIKITEXT = """{{ח:התחלה}}
{{ח:כותרת|חוק החוזים (חלק כללי), תשל״ג–1973}}

{{ח:פתיח-התחלה}}
{{ח:מאגר|2000292}} {{ח:תיבה|ס״ח תשל״ג, 118|חוק החוזים (חלק כללי)|https://fs.knesset.gov.il/7/law/7_lsr_208381.pdf}}.
{{ח:סוגר}}
{{ח:מפריד}}

{{ח:קטע2||תוכן עניינים}}

<div class="law-toc">
<div class="law-toc-2">{{ח:פנימי|פרק א|פרק א׳: כריתת החוזה}}</div>
</div>

{{ח:קטע2|פרק א|פרק א׳: כריתת החוזה}}

{{ח:סעיף|1|כריתת חוזה – כיצד}}
{{ח:ת}} חוזה נכרת בדרך של הצעה וקיבול לפי הוראות {{ח:פנימי|פרק א|פרק זה}}.

{{ח:סעיף|3|חזרה מן ההצעה}}
{{ח:תת|(א)}} המציע רשאי לחזור בו מן ההצעה.
{{ח:תת|(ב)}} קבע המציע שהצעתו היא ללא חזרה, אין הוא רשאי לחזור בו.
{{ח:תתת|(1)}} פריט פנימי.

{{ח:קטע2|תוספת|התוספת הראשונה}}
{{ח:ת}} טופס בקשה.
{{ח:חתימות|גולדה מאיר<br>ראש הממשלה}}
{{ח:סוף}}
"""


def test_open_law_book_page_parses_into_sections_headings_and_registry_id():
    parsed = wikisource.parse_law_page(WIKITEXT, "חוק החוזים (חלק כללי)")
    assert parsed.full_title == "חוק החוזים (חלק כללי), תשל״ג–1973"
    assert parsed.registry_id == 2000292
    assert parsed.citations[0]["url"].endswith("7_lsr_208381.pdf")
    sections = [s for s in parsed.sections if s.kind == "section"]
    assert [s.number for s in sections] == ["1", "3"]
    assert sections[0].chapter == "פרק א׳: כריתת החוזה" and sections[0].intro.endswith("פרק זה.")
    assert [sub.label for sub in sections[1].subsections] == ["א", "ב"]
    assert "(1) פריט פנימי." in sections[1].subsections[1].text
    (schedule,) = [s for s in parsed.sections if s.kind == "schedule"]
    assert schedule.title == "התוספת הראשונה" and schedule.text == "טופס בקשה."
    assert "תוכן עניינים" not in parsed.text and "גולדה" not in parsed.text
    assert parsed.parsed and not parsed.repeal_marker and not parsed.unknown_templates


def test_dump_pages_stream_out_of_multistream_bz2_and_only_law_pages_count():
    xml = (
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"><siteinfo><sitename>x</sitename></siteinfo>'
        "<page><title>חוק א</title><ns>0</ns><id>5</id><revision><id>77</id>"
        "<timestamp>2026-01-01T00:00:00Z</timestamp><text>{{ח:התחלה}} {{ח:כותרת|חוק א}}</text></revision></page>"
        "<page><title>פורטל:משפט</title><ns>100</ns><id>6</id><revision><id>78</id>"
        "<timestamp>2026-01-01T00:00:00Z</timestamp><text>{{ח:התחלה}}</text></revision></page>"
        '<page><title>הפניה</title><ns>0</ns><id>7</id><redirect title="חוק א" /><revision><id>79</id>'
        "<timestamp>2026-01-01T00:00:00Z</timestamp><text>{{ח:התחלה}}</text></revision></page></mediawiki>"
    )
    data = bz2.compress(xml[:150].encode("utf-8")) + bz2.compress(xml[150:].encode("utf-8"))
    pages = list(wikisource.iter_pages([data[:40], data[40:]]))
    assert [p.title for p in pages] == ["חוק א", "פורטל:משפט", "הפניה"]
    assert [wikisource.is_law_book_page(p) for p in pages] == [True, False, False]
    assert pages[0].rev_id == 77 and pages[0].page_id == 5


# --- registry joins ------------------------------------------------------------------------------------


def test_law_names_match_across_quote_dash_and_year_spellings():
    assert name_key("חוק החוזים (חלק כללי), תשל״ג–1973") == name_key('חוק החוזים (חלק כללי), התשל"ג-1973')
    assert has_prefix("תקנות סדר הדין האזרחי, התשע״ט–2018", 'תקנות סדר הדין האזרחי, התשע"ט-2018')
    assert has_prefix('תקנות סדר הדין הפלילי (חקירת עדים), התש"ם-1980', "תקנות סדר הדין הפלילי")
    assert not has_prefix('תקנות סדר הדין האזרחי, התשמ"ד-1984', 'תקנות סדר הדין האזרחי, התשע"ט-2018')
    index = NameIndex()
    index.add("חוק א", 1)
    index.add("חוק א", 2)
    assert index.lookup("חוק א") is None  # ambiguous: never guessed


def test_titles_classify_into_laws_and_procedural_rules():
    assert legislation.classify_title("חוק־יסוד: הכנסת") == "basic_law"
    assert legislation.classify_title("חוק יסוד: כבוד האדם וחירותו") == "basic_law"
    assert legislation.classify_title("פקודת הנזיקין [נוסח חדש]") == "ordinance"
    assert legislation.classify_title("חוק החוזים (חלק כללי), תשל״ג–1973") == "law"
    assert legislation.classify_title("תקנות סדר הדין האזרחי, התשע״ט–2018") == "regulation"
    assert legislation.category_for("ordinance") == "laws" and legislation.category_for("regulation") == "procedural_rules"
    assert (legislation.law_status("תקף"), legislation.law_status("בטל"), legislation.law_status("")) == (
        "in_force", "repealed", "unknown")


def test_registry_file_paths_become_clean_knesset_urls_only():
    host = "fs.knesset.gov.il"
    assert legislation.knesset_file_url("https://fs.knesset.gov.il/\\20\\SecondaryLaw\\20_scl_bg_491805.pdf", host) == \
        "https://fs.knesset.gov.il/20/SecondaryLaw/20_scl_bg_491805.pdf"
    assert legislation.knesset_file_url("https://fs.knesset.gov.il/\\20\\x\\20___491804.docx", host) is None
    assert legislation.knesset_file_url("https://olaw.org.il/takanot/takanot-8085.pdf", host) is None


def _ctx():
    return SimpleNamespace(cfg=SimpleNamespace(sources=SimpleNamespace(wikisource_wiki="https://he.wikisource.org/wiki/")))


def test_a_law_joins_the_registry_by_its_page_id_then_by_name():
    page = wikisource.WikiPage("חוק החוזים (חלק כללי)", 0, 1234, 99, "2026-01-01T00:00:00Z", False, WIKITEXT)
    dump = wikisource.DumpFile("20260901", "x.xml.bz2", "https://dumps.test/x", 1, None)
    row = {"Id": 2000292, "Name": 'חוק החוזים (חלק כללי), התשל"ג-1973', "IsBasicLaw": False, "LawValidityDesc": "תקף",
           "PublicationDate": "1973-06-01T00:00:00+03:00", "ValidityStartDate": None,
           "LatestPublicationDate": "2026-01-05T00:00:00+02:00"}
    registry = legislation.Registry(laws={2000292: row})
    registry.law_names.add(row["Name"], 2000292)

    parsed = wikisource.parse_law_page(WIKITEXT, page.title)
    record = legislation.build_record(page, parsed, "law", registry, _ctx(), dump, {})
    assert (record.law_id, record.law_id_source, record.status) == (2000292, "wikisource_registry_id", "in_force")
    assert (record.effective_date, record.latest_amendment_date) == ("1973-06-01", "2026-01-05")
    assert record.section_numbers == ["1", "3"] and record.license == wikisource.LICENSE
    assert record.source_url.endswith("?oldid=99")

    without_id = wikisource.parse_law_page(WIKITEXT.replace("{{ח:מאגר|2000292}}", ""), page.title)
    record = legislation.build_record(page, without_id, "law", registry, _ctx(), dump, {})
    assert (record.law_id, record.law_id_source) == (2000292, "name_match")


# --- Hebrew -----------------------------------------------------------------------------------------------


def _as_cp1252_mojibake(text: str) -> str:
    return "".join(chr(b) if b in (0x81, 0x8D, 0x8F, 0x90, 0x9D) else bytes([b]).decode("cp1252")
                   for b in text.encode("utf-8"))


def test_mojibake_is_repaired_and_visual_order_is_reversed_and_noted():
    original = "חוק החוזים קובע את דרך כריתת החוזה"
    assert repair_text(_as_cp1252_mojibake(original))[:2] == (original, "repaired")
    assert repair_text(original.encode("cp1255").decode("latin-1"))[:2] == (original, "repaired")

    logical = "\n".join(["הצדדים הסכימו שהחוזים יהיו תקפים בכל המקרים והתנאים"] * 10)
    text, status, notes = repair_text("\n".join(line[::-1] for line in logical.splitlines()))
    assert (text, status) == (logical, "repaired") and any("visual-order" in n for n in notes)
    assert repair_text(logical)[1] == "ok"


def test_index_copy_folds_final_letters_but_the_embedded_copy_keeps_them():
    assert normalize_for_index("חוק־יסוד: שָׁלוֹם") == "חוק-יסוד: שלומ"
    assert normalize_for_embedding('שָׁלוֹם "תשל"ג"') == "שלום ״תשל״ג״"


# --- Supreme Court ------------------------------------------------------------------------------------------


def test_keep_rules_judgments_and_non_technical_decisions_only():
    types = ["פסק-דין"]
    assert supreme_court.classify({"Type": "פסק־דין"}, types) == ("judgment", "judgment")
    assert supreme_court.classify({"Type": "החלטה", "Technical": False}, types)[0] == "decision"
    assert supreme_court.classify({"Type": "החלטה", "Technical": True}, types) == (None, "decision_technical")
    assert supreme_court.classify({"Type": "החלטה", "meta_is_technical": "false"}, types)[0] == "decision"


def test_each_privacy_rule_catches_its_case():
    rules = supreme_court.PrivacyFilter(get_config().legal_data.privacy)
    assert rules.hits({"text": "הפרטים המזהים אסורים בפרסום"}) == ["publication_restriction"]
    assert rules.hits({"meta_case_nbr": "בע״מ 1234/20", "text": "ערעור"}) == ["family_case_type"]
    assert rules.hits({"CaseName": "פלוני נ' מדינת ישראל", "text": "ערעור"}) == ["anonymized_parties"]
    assert rules.hits({"text": "המערער הורשע בעבירות מין"}) == ["topic_keywords"]
    assert rules.hits({"CaseName": 'חברת א בע"מ נ\' פקיד שומה', "meta_case_nbr": 'ע"א 1/20',
                       "text": "ערעור על שומת מס הכנסה"}) == []


def test_anonymizer_replaces_people_and_keeps_the_state_and_organizations():
    anonymizer = supreme_court.Anonymizer(get_config().legal_data.privacy.public_body_patterns)
    parties = ["מדינת ישראל", "משה כהן", 'חברת החשמל לישראל בע"מ']
    text, case_name, replaced, private = anonymizer.apply(
        "המערער משה כהן טען. כהן משה חזר על טענתו. מדינת ישראל השיבה.", "משה כהן נ' מדינת ישראל", parties)
    assert text == "המערער [צד 1] טען. [צד 1] חזר על טענתו. מדינת ישראל השיבה."
    assert case_name == "[צד 1] נ' מדינת ישראל" and (replaced, private) == (3, 1)


def test_a_ruling_becomes_a_record_without_party_fields():
    row = {"Type": "פסק-דין", "meta_case_nbr": 'ע"א 1234/20', "CaseName": "כהן נ' לוי", "meta_judge": ["א' חיות"],
           "VerdictDt": "2021-03-04T00:00:00", "text": "פסק דין. הערעור נדחה.", "document_hash": "ab" * 32,
           "meta_side_nm": ["כהן", "לוי"], "Year": 2021}
    record, _ = supreme_court.to_record(row, "judgment", "LevMuchnik/SupremeCourtOfIsrael", "2022", None)
    assert record.id == "supreme_court:" + "ab" * 32
    assert (record.case_number, record.decision_date, record.judges, record.status) == (
        'ע"א 1234/20', "2021-03-04", ["א' חיות"], "unknown")
    assert "meta_side_nm" not in record.model_dump() and record.license.startswith("OpenRAIL")


# --- JSONL, the run wrapper, the ingest guard ---------------------------------------------------------------


def test_jsonl_shards_are_swapped_in_whole_and_stale_shards_removed(tmp_path):
    (tmp_path / "x-00009.jsonl").write_text("{}\n", encoding="utf-8")
    writer = JsonlWriter(tmp_path, "x", shard_size=2)
    for i in range(5):
        writer.write({"id": str(i), "text": "טקסט"})
    paths = writer.close()
    assert [p.name for p in paths] == ["x-00001.jsonl", "x-00002.jsonl", "x-00003.jsonl"]
    assert not (tmp_path / "x-00009.jsonl").exists() and not list(tmp_path.glob("*.tmp"))
    assert [r["id"] for p in paths for r in read_jsonl(p)] == ["0", "1", "2", "3", "4"]
    assert record_hash({"id": "1", "retrieved_at": "a"}) == record_hash({"id": "1", "retrieved_at": "b"})


def test_a_run_without_contact_email_stops_and_writes_a_manifest(tmp_path, monkeypatch):
    from docslides.legal_data import cli

    monkeypatch.setattr(get_config().legal_data, "contact_email", "")
    args = SimpleNamespace(dry_run=True, sample=None, output=str(tmp_path))
    assert cli.run("fetch_test", args, lambda ctx: None) == 2
    (manifest,) = (tmp_path / "_manifests").glob("*_fetch_test.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["status"] == "stopped" and "contact_email" in data["errors"][0]


def test_ingest_legal_txt_skips_the_corpus_folders(tmp_path, monkeypatch):
    from docslides.legal import folder_ingest

    monkeypatch.setattr(get_config().legal.ingestion, "legal_txt_dir", str(tmp_path))
    monkeypatch.setattr(get_config().legal.ingestion, "staging_dir", str(tmp_path / "staging"))
    for sub in ("laws/raw/pdf/1", "procedural_rules", "supreme_court", "_sample/laws"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
        (tmp_path / sub / "doc.txt").write_text("חוק לדוגמה", encoding="utf-8")
    assert folder_ingest.run(dry_run=True) == []


def test_progress_lines_show_percent_rate_and_eta_at_most_every_few_seconds():
    from docslides.legal_data.progress import Progress

    lines, now = [], [0.0]
    progress = Progress("rows", total=100, unit="rows", log=lines.append, every=5, clock=lambda: now[0])
    now[0] = 1
    progress.update(10)  # under 5 s since the start: silent
    now[0] = 10
    progress.update(40, stored=50)
    progress.done()
    assert len(lines) == 2
    assert "50/100 rows (50.0%)" in lines[0] and "5.0 rows/s" in lines[0] and "ETA 10s" in lines[0]
    assert "stored 50" in lines[0] and lines[1].endswith("done")
