"""The Spanish corpus fetchers (legal_data/): BOE consolidated texts, Tribunal Constitucional pages,
EUR-Lex XHTML, robots.txt wildcards, text normalization and law names -- on fixtures shaped like
the real responses, with no network."""

import httpx
import pytest

from aicodex.config import LegalDataConfig
from aicodex.legal_data import boe, eurlex, tc
from aicodex.legal_data.corpus_chunking import chunk_record
from aicodex.legal_data.http import PoliteClient, RobotsDisallowed, RobotsRules
from aicodex.legal_data.law_names import law_number_key, name_key, resolve_abbreviation
from aicodex.legal_data.records import record_hash
from aicodex.legal_data.spanish import fold_accents, normalize_for_embedding, normalize_for_index, repair_text

BOE_XML = """<?xml version="1.0" encoding="utf-8"?>
<response><status><code>200</code></status><data><texto>
  <bloque id="pr" tipo="preambulo"><version id_norma="BOE-A-1994-26003" fecha_publicacion="19941125" fecha_vigencia="19950101">
    <p class="parrafo">JUAN CARLOS I, REY DE ESPAÑA.</p></version></bloque>
  <bloque id="ti" tipo="encabezado" titulo="TÍTULO I"><version fecha_vigencia="19950101">
    <p class="titulo_num">TÍTULO I</p><p class="titulo_tit">Ámbito de la ley</p></version></bloque>
  <bloque id="a9" tipo="precepto" titulo="Artículo 9"><version fecha_vigencia="19950101">
    <p class="articulo">Artículo 9. Plazo mínimo.</p><p class="parrafo">1. La duración será de tres años.</p></version>
    <version fecha_vigencia="20190306">
    <p class="articulo">Artículo 9. Plazo mínimo.</p>
    <p class="parrafo">1. La duración del arrendamiento será libremente pactada por las partes.</p>
    <p class="parrafo_2">a) Primera letra.</p>
    <p class="parrafo">2. Se entenderán celebrados por un año los arrendamientos sin plazo.</p>
    <p class="nota_pie">Se modifica por el art. 1.4 del Real Decreto-ley 7/2019. Ref. BOE-A-2019-3244</p></version></bloque>
  <bloque id="s1" tipo="precepto" fecha_caducidad="19990101" titulo="Sección primera"><version fecha_vigencia="19950101">
    <p class="seccion">Sección primera. Derogada</p></version></bloque>
  <bloque id="a9bis" tipo="precepto" titulo="Artículo 9 bis"><version fecha_vigencia="20230526">
    <p class="articulo">Artículo 9 bis. Vivienda temporal.</p><p class="parrafo">Texto del 9 bis.</p></version></bloque>
  <bloque id="da" tipo="precepto" titulo="Disposición adicional primera"><version fecha_vigencia="19950101">
    <p class="articulo">Disposición adicional primera. Fianzas.</p><p class="parrafo">Las comunidades podrán exigir el depósito.</p>
    </version></bloque>
  <bloque id="ai" tipo="encabezado" titulo="ANEXO"><version fecha_vigencia="19950101">
    <p class="anexo_num">ANEXO I</p><p class="anexo_tit">Modelo de contrato</p></version></bloque>
  <bloque id="a" tipo="precepto" titulo="A"><version fecha_vigencia="19950101">
    <table><tr><td>Renta</td><td>500 euros</td></tr></table></version></bloque>
  <bloque id="fi" tipo="firma"><version fecha_vigencia="19950101"><p class="firma_rey">JUAN CARLOS R.</p></version></bloque>
</texto></data></response>""".encode()

CATALOGUE_ITEM = {
    "identificador": "BOE-A-1994-26003", "titulo": "Ley 29/1994, de 24 de noviembre, de Arrendamientos Urbanos.",
    "rango": {"codigo": "1300", "texto": "Ley"}, "ambito": {"codigo": "1", "texto": "Estatal"},
    "departamento": {"codigo": "7723", "texto": "Jefatura del Estado"}, "fecha_disposicion": "19941124",
    "fecha_publicacion": "19941125", "fecha_vigencia": "19950101", "vigencia_agotada": "N",
    "estado_consolidacion": {"codigo": "3", "texto": "Finalizado"}, "numero_oficial": "29/1994",
    "fecha_actualizacion": "20261008T101010Z", "url_eli": "https://www.boe.es/eli/es/l/1994/11/24/29",
    "url_html_consolidada": "https://www.boe.es/buscar/act.php?id=BOE-A-1994-26003", "diario": "Boletín Oficial del Estado",
}


def test_boe_consolidated_text_keeps_the_current_wording_and_structure():
    sections, stats = boe.parse_consolidated_text(BOE_XML)
    by_number = {s.number: s for s in sections}
    assert [s.kind for s in sections] == ["preamble", "section", "section", "disposition", "schedule"]
    nine = by_number["9"]
    assert nine.title == "Plazo mínimo" and nine.division == "TÍTULO I. Ámbito de la ley"
    assert nine.text.startswith("1. La duración del arrendamiento será libremente pactada")  # last version only
    assert [s.label for s in nine.subsections] == ["1", "2"]  # "a)" stays inside apartado 1
    assert nine.note.startswith("Se modifica por el art. 1.4") and "Se modifica" not in nine.text
    assert nine.version_date == "2019-03-06" and stats["latest_version"] == "2023-05-26"
    assert by_number["9 bis"].title == "Vivienda temporal"
    assert by_number["Disposición adicional primera"].title == "Fianzas"
    assert by_number["ANEXO I. Modelo de contrato"].text == "Renta | 500 euros"
    assert "Sección primera" not in str(sections) and stats["expired_blocks"] == 1  # fecha_caducidad: gone
    assert "JUAN CARLOS R." not in str(sections)  # signatures dropped


def test_boe_record_and_classification():
    assert boe.classify(CATALOGUE_ITEM) == ("laws", "law")
    assert boe.classify({**CATALOGUE_ITEM, "rango": {"texto": "Real Decreto"}, "titulo": "Real Decreto 1/2020, de x."}) \
        == ("regulations", "regulation")
    assert boe.classify({**CATALOGUE_ITEM, "rango": {"texto": "Real Decreto"},
                         "titulo": "Real Decreto de 24 de julio de 1889 por el que se publica el Código Civil."}) == ("laws", "law")
    assert boe.classify({**CATALOGUE_ITEM, "ambito": {"texto": "Autonómico"}, "rango": {"texto": "Ley Foral"}}) \
        == ("regional_laws", "regional_law")
    assert boe.classify({**CATALOGUE_ITEM, "rango": {"texto": "Constitución"}}) == ("laws", "constitution")
    sections, stats = boe.parse_consolidated_text(BOE_XML)
    record = boe.build_record(CATALOGUE_ITEM, "laws", "law", sections, stats)
    assert record.id == "boe:BOE-A-1994-26003" and record.status == "in_force" and record.effective_date == "1995-01-01"
    assert record.rank == "Ley" and record.official_number == "29/1994" and record.scope == "estatal"
    assert record.section_numbers == ["9", "9 bis"] and "Artículo 9. Plazo mínimo." in record.text
    assert "solo la publicación oficial en el BOE tiene validez jurídica" in record.attribution
    repealed = boe.build_record({**CATALOGUE_ITEM, "vigencia_agotada": "S"}, "laws", "law", sections, stats)
    assert repealed.status == "repealed"


def test_boe_record_chunks_carry_the_article_breadcrumb_and_scalar_metadata():
    sections, stats = boe.parse_consolidated_text(BOE_XML)
    record = boe.build_record(CATALOGUE_ITEM, "laws", "law", sections, stats).model_dump(mode="json")
    chunks = chunk_record(record, record_hash(record), 500, 64, False)
    texts = [c.text for c in chunks]
    assert any(t.startswith("Ley 29/1994, de 24 de noviembre, de Arrendamientos Urbanos > TÍTULO I. Ámbito de la ley > "
                            "Artículo 9 — Plazo mínimo") for t in texts)
    assert any("Disposición adicional primera — Fianzas" in t for t in texts)
    for chunk in chunks:
        assert all(isinstance(v, (str, int, float, bool)) for v in chunk.metadata.values())
        assert chunk.lexical_text == chunk.lexical_text.lower()
    assert len({c.chunk_id for c in chunks}) == len(chunks)


TC_HTML = """<html><head><title>Sistema HJ - Resoluci&#243;n: SENTENCIA 84/2025</title></head><body>
<ul><li id="resolucion-identifier">SENTENCIA 84/2025, de 7 de abril<br/>(BOE núm. 117, de 15 de mayo de 2025)<br/>
ECLI:ES:TC:2025:84</li></ul>
<div id="complete_resolucion"><p>La Sala Primera del Tribunal Constitucional, compuesta por el magistrado don Cándido
Conde-Pumpido Tourón, presidente, ha pronunciado</p><p>SENTENCIA</p>
<p>En el recurso de amparo núm. 6194-2020, promovido por don Miguel Ángel González Ortiz, representado por la
procuradora doña Ana Pérez Gil, y con intervención del magistrado don César Tolosa Tribiño.</p>
<p>II. Fundamentos jurídicos</p><p>1. Se estima el recurso.</p></div>
<div id="ficha-tecnica"><table>
<tr><th>Órgano</th><td>Sala Primera</td></tr>
<tr><th>Magistrados</th><td><p>Don Cándido Conde-Pumpido Tourón y doña Laura Díez Bueso.</p></td></tr>
<tr><th>Tipo y número de registro</th><td>Recurso de amparo 6194-2020</td></tr>
<tr><th>Fecha de resolución</th><td>07/04/2025</td></tr>
<tr><td>Síntesis Descriptiva</td><td>Promovido por don Miguel Ángel González Ortiz respecto de un acuerdo.</td></tr>
</table></div></body></html>"""


def test_tc_resolution_is_parsed_and_private_parties_are_anonymized():
    fields = tc.parse_resolution(TC_HTML, "https://hj.tribunalconstitucional.es/es/Resolucion/Show/31500", 31500, True)
    record = tc.to_record(fields)
    assert record.id == "tc:ECLI:ES:TC:2025:84" and record.doc_type == "SENTENCIA" and record.authority_level == "judgment"
    assert (record.resolution_number, record.year, record.chamber) == ("84/2025", 2025, "Sala Primera")
    assert record.decision_date == "2025-04-07" and record.boe_publication == "BOE núm. 117, de 15 de mayo de 2025"
    assert record.procedure_type == "Recurso de amparo" and record.case_number == "Recurso de amparo 6194-2020"
    assert record.judges == ["Cándido Conde-Pumpido Tourón", "Laura Díez Bueso"]
    assert "Miguel Ángel" not in record.text and "Ana Pérez" not in record.text and "Miguel Ángel" not in record.summary
    assert "don [persona 1]" in record.text and "doña [persona 2]" in record.text
    assert "Cándido Conde-Pumpido Tourón" in record.text  # a magistrate of the bench
    assert "César Tolosa Tribiño" in record.text  # a magistrate of the court, named as such
    kept = tc.to_record(tc.parse_resolution(TC_HTML, "u", 31500, False))
    assert "Miguel Ángel González Ortiz" in kept.text and not kept.anonymized
    assert tc.parse_resolution("<html><title>Error</title></html>", "u", 1, True) is None


EU_XHTML = """<html><body>
<p class="oj-doc-ti">REGLAMENTO (UE) 2016/679 DEL PARLAMENTO EUROPEO Y DEL CONSEJO</p>
<p class="oj-doc-ti">de 27 de abril de 2016</p>
<p class="oj-normal">Considerando lo siguiente:</p>
<p class="oj-ti-section-1"><span>CAPÍTULO I</span></p><p class="oj-ti-section-2">Disposiciones generales</p>
<p class="oj-ti-art">Artículo 1</p><p class="oj-sti-art">Objeto</p>
<p class="oj-normal">1.</p><p class="oj-normal">El presente Reglamento establece normas.</p>
<p class="oj-normal">2.</p><p class="oj-normal">Protege los derechos.</p>
<p class="oj-ti-section-1">CAPÍTULO III</p><p class="oj-ti-section-2">Derechos del interesado</p>
<p class="oj-ti-section-1">Sección 3</p><p class="oj-ti-section-2">Rectificación y supresión</p>
<p class="oj-ti-art">Artículo 17</p><p class="oj-sti-art">Derecho de supresión</p>
<p class="oj-normal">El interesado tendrá derecho a obtener la supresión.</p>
<p class="oj-doc-ti">ANEXO</p><p class="oj-normal">Tabla de correspondencias.</p>
</body></html>""".encode()


def test_eurlex_xhtml_articles_chapters_and_annex():
    title, sections = eurlex.parse_xhtml(EU_XHTML)
    assert title.startswith("REGLAMENTO (UE) 2016/679")
    by_number = {s.number: s for s in sections}
    assert by_number["1"].title == "Objeto" and by_number["1"].chapter == "CAPÍTULO I. Disposiciones generales"
    assert [s.label for s in by_number["1"].subsections] == ["1", "2"]  # "1." joined with its paragraph
    assert by_number["17"].chapter == "CAPÍTULO III. Derechos del interesado"
    assert by_number["17"].subchapter == "Sección 3. Rectificación y supresión"
    assert sections[0].kind == "preamble" and sections[-1].kind == "schedule"
    record = eurlex.build_record("32016R0679", "02016R0679-20160504", "u", title, sections)
    assert record.authority_level == "eu_regulation" and record.official_number == "2016/679"
    assert record.effective_date == "2016-05-04" and record.consolidated
    assert eurlex.consolidated_prefix("32016R0679") == "02016R0679" and eurlex.consolidated_prefix("12016P/TXT") is None


def test_robots_wildcards_and_withdrawn_documents():
    rules = RobotsRules("User-agent: *\nDisallow: /buscar/doc.php?*lang=en\nDisallow: /*BOE-A-2026-4920$\n"
                        "Disallow: /datos/pdfs/\nAllow: /datos/pdfs/ok/\n")
    assert not rules.can_fetch("x", "https://www.boe.es/buscar/doc.php?id=BOE-A-1&lang=en")
    assert rules.can_fetch("x", "https://www.boe.es/buscar/doc.php?id=BOE-A-1&lang=es")
    assert not rules.can_fetch("x", "https://www.boe.es/buscar/act.php?id=BOE-A-2026-4920")
    assert rules.can_fetch("x", "https://www.boe.es/buscar/act.php?id=BOE-A-2026-49201")  # $ anchors the end
    assert not rules.can_fetch("x", "https://www.boe.es/datos/pdfs/a.pdf")
    assert rules.can_fetch("x", "https://www.boe.es/datos/pdfs/ok/a.pdf")  # the longer Allow wins
    own = RobotsRules("User-agent: *\nDisallow: /\n\nUser-agent: AI-Codex-legal-corpus\nDisallow: /private/\nCrawl-delay: 3\n")
    assert own.can_fetch("x", "https://h/datos") and not own.can_fetch("x", "https://h/private/a") and own.delay == 3.0


def test_client_reads_withdrawn_ids_and_refuses_disallowed_urls():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /*BOE-A-2026-4920$\nDisallow: /secret/\n")
        return httpx.Response(200, json={"data": []})

    cfg = LegalDataConfig(contact_email="corpus@example.org", min_interval_s=0)
    client = PoliteClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert "AI-Codex-legal-corpus" in client.user_agent and "Spanish" in client.user_agent
    assert client.disallowed_tokens("https://www.boe.es/x", r"BOE-[A-Z]-\d{4}-\d+") == {"BOE-A-2026-4920"}
    assert client.get_json("https://www.boe.es/datosabiertos/api/x") == {"data": []}
    with pytest.raises(RobotsDisallowed):
        client.get("https://www.boe.es/secret/a")


def test_spanish_text_normalization_and_mojibake_repair():
    assert fold_accents("Artículo único: año, pingüino") == "Articulo unico: año, pinguino"  # ñ kept
    assert normalize_for_index("ARTÍCULO «Único»") == 'articulo "unico"'
    assert normalize_for_embedding("el  «consumidor»") == 'el "consumidor"'
    fixed, status, notes = repair_text("El artÃ­culo 9 del cÃ³digo establece el aÃ±o de duraciÃ³n.")
    assert fixed == "El artículo 9 del código establece el año de duración." and status == "repaired"
    assert repair_text("Texto correcto.")[1] == "ok"


def test_law_names_numbers_and_abbreviations():
    assert law_number_key("Ley 1/2000, de 7 de enero, de Enjuiciamiento Civil") == "ley 1/2000"
    assert law_number_key("Real Decreto Legislativo 2/2015, de 23 de octubre") == "real decreto legislativo 2/2015"
    assert name_key("Ley Orgánica 10/1995, de 23 de noviembre, del Código Penal") == name_key("Código Penal")
    assert name_key("Real Decreto de 24 de julio de 1889 por el que se publica el Código Civil.") == "codigo civil"
    assert resolve_abbreviation("LEC") == "BOE-A-2000-323" and resolve_abbreviation("Código Civil") == "BOE-A-1889-4763"
    assert resolve_abbreviation("XYZ") is None
