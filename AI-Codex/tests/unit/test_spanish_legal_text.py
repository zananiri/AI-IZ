"""Spanish legal text in the Legal tab (legal/): keyword search, numbers, amendments, insertions,
chunk breadcrumbs, query parsing and the embedding prompts."""

from aicodex.legal import amendments
from aicodex.legal.chunking import SourceMeta, chunk_sections, normalize_json_quotes
from aicodex.legal.insertions import find_insertion, join_spaced_section_numbers
from aicodex.legal.keyword import KeywordIndex, stem, terms
from aicodex.legal.numeric_check import unsupported_numbers
from aicodex.legal.pipeline import _looks_spanish
from aicodex.legal.retrieval import named_sections, question_parts
from aicodex.legal.structure import _section_key, extract_cross_references, parse_sections
from aicodex.rag.embedding import document_prompt, query_prompt


def test_keyword_terms_fold_accents_stem_and_keep_identifiers():
    assert stem("arrendamientos") == stem("arrendamiento") == "arrendamient"
    assert stem("obligadas") == stem("obligado")
    found = terms("Artículo 14 bis de la Ley 29/1994: los Arrendamientos URBANOS; usucapión")
    assert {"14bis", "29/1994", "usucapion", "arrendamient", "urban"} <= set(found)
    assert "de" not in found and "articulo" not in found  # stopwords
    index = KeywordIndex([("a", "El arrendamiento urbano de vivienda"), ("b", "La usucapión extraordinaria")])
    assert [cid for cid, _ in index.search("arrendamientos de viviendas", 2)] == ["a"]


def test_numbers_match_in_spanish_and_english_notation():
    evidence = ["33 días de salario por año de servicio, hasta 24 mensualidades; 2.500.000 euros; 20 por ciento; 1500"]
    answer = "La indemnización es de 33 días por año, hasta 24 mensualidades; 2,5 millones; el 20 %; 1.500 euros; 7 días."
    assert unsupported_numbers(answer, evidence) == ["7"]
    assert unsupported_numbers("un plazo de 1,5 años", ["1.5 años"]) == []


def test_amendment_targets_spanish_laws_and_their_articles():
    text = ("Se modifica el artículo 9 de la Ley 29/1994, de 24 de noviembre, de Arrendamientos Urbanos, que queda "
            "redactado como sigue:\n«Artículo 9. Plazo mínimo.\n1. La duración será libremente pactada.»")
    ref = amendments.extract_amendment(None, text, [])
    assert (ref.target_key, ref.sections, ref.temporary) == ("ley 29/1994", ["9"], False)
    titled = amendments.extract_amendment("Modificación del Código Civil", "Se añade un artículo 1902 bis con la "
                                          "siguiente redacción:\n«Artículo 1902 bis. Daños.\nTexto.»", [])
    assert titled.target_key == "codigo civil" and titled.sections == ["1902 bis"]
    assert amendments.extract_amendment(None, "Los arrendatarios podrán desistir del contrato.", []) is None
    assert amendments._base_section("14 bis(3)") == "14 bis" and amendments._base_section("62(2)") == "62"


def test_insertion_splits_quoted_articles_with_their_numbers():
    insertion = find_insertion("Se añaden dos nuevos artículos con la siguiente redacción:\n"
                               "«Artículo 14 bis. Uso temporal.\n1. Texto uno.\n"
                               "Artículo 14 ter. Otra cosa.\nTexto dos.»")
    assert [(p.number, p.title) for p in insertion.provisions] == [("14 bis", "Uso temporal"), ("14 ter", "Otra cosa")]
    assert insertion.provisions[1].lines == ["Texto dos."]
    apartado = find_insertion("Se modifica el apartado 2 del artículo 9, que queda redactado como sigue:\n«2. Nuevo texto.»")
    assert apartado.provisions[0].number == "9.2"
    assert find_insertion("Se sustituye la expresión «tres años» por «cinco años».") is None
    assert join_spaced_section_numbers("artículo 14bis") == "artículo 14 bis"


def test_structure_orders_suffixes_and_reads_cross_references():
    assert _section_key("14") < _section_key("14 bis") < _section_key("14 ter") < _section_key("15")
    refs = extract_cross_references("De conformidad con los artículos 3 a 5 y el artículo 14.2, sin perjuicio del "
                                    "artículo 1902 del Código Civil y del art. 24 CE.", "1")
    assert refs == ["3", "5", "4", "14"]  # a range's inner numbers follow its ends
    sections = parse_sections("Artículo 1. Objeto.\nTexto.\nArtículo 1. Repetido en una cita.\nArtículo 2.\nOtro.")
    assert [s.number for s in sections] == ["1", "2"]  # numbering must advance


def test_chunks_name_the_article_and_apartado_and_strip_the_promulgation():
    text = ("Artículo 1. Objeto.\nEsta ley regula \"algo\".\nArtículo 2. Ámbito.\n1. Se aplica a todas las personas físicas y jurídicas que residan en España.\n"
            "2. Salvo lo dispuesto en el artículo 1 para los supuestos especiales que esta ley prevé.\nDisposición final única. Entrada en vigor.\nMañana.\n"
            "Por tanto,\nMando a todos los españoles...\nFELIPE R.")
    meta = SourceMeta(law_id="boe:X", law_name="Ley 1/2000", effective_date_start="2000-01-01", status="current",
                      source_type="statute", source_origin="boe")
    chunks = chunk_sections(parse_sections(text), meta, "2026-10-16", budget=40)
    heads = [c.text.split("\n\n")[0] for c in chunks]
    assert heads == ["Ley 1/2000 > Artículo 1 — Objeto", "Ley 1/2000 > Artículo 2.1 — Ámbito",
                     "Ley 1/2000 > Artículo 2.2 — Ámbito", "Ley 1/2000 > Disposición final única — Entrada en vigor"]
    assert "«algo»" in chunks[0].text and "FELIPE" not in chunks[-1].text
    assert chunks[2].metadata.cross_references == ["boe:X@2000-01-01:1"]
    assert normalize_json_quotes('un "a" y otro "') == "un «a» y otro ”"


def test_questions_name_articles_and_split_into_parts():
    assert named_sections("¿Qué dice el artículo 1902 del CC y el art. 14.2 LAU, y el artículo 20 bis?") == [
        ("1902", None), ("14", "2"), ("20 bis", None)]
    assert question_parts("¿Cuál es el plazo para recurrir en apelación y qué tribunal resuelve el recurso?") == [
        "Cuál es el plazo para recurrir en apelación", "qué tribunal resuelve el recurso"]
    assert _looks_spanish("¿Puedo anular un contrato?") and _looks_spanish("cual es el plazo de la ley para el recurso")
    assert not _looks_spanish("What is the deadline for an appeal under the law?")


def test_embedding_prompts_follow_each_models_convention():
    prompt, name = query_prompt("Qwen/Qwen3-Embedding-0.6B")
    assert prompt.startswith("Instruct: ") and prompt.endswith("\nQuery:") and name is None
    assert query_prompt("BAAI/bge-m3") == (None, None) and document_prompt("BAAI/bge-m3") is None
    assert query_prompt("intfloat/multilingual-e5-large") == ("query: ", None)
    assert document_prompt("intfloat/multilingual-e5-large") == "passage: "

    class WithPrompts:
        prompts = {"query": "consulta: "}

    assert query_prompt("SINAI/some-legal-model", WithPrompts()) == (None, "query")
