"""Legal tab pipeline: grounded RAG over Israeli law.

    language routing -> retrieval -> Pass 0 analysis notes (thinking on)
    -> Pass A research memorandum -> validation gate
    -> Pass B draft -> citation verification (structural + entailment)
    -> unverified sentences removed -> wrong-script words repaired
    -> final citation check -> numeric grounding check -> audit log

One model (the orchestrator, Qwen) does all of it: research, drafting and
verification. The reply is always in the question's language. Every model call
-- with its reasoning -- is recorded in the audit entry (llm_calls) and the LLM
trace (llm/trace.py).

Nothing flagged is resolved silently. A memorandum that fails the gate is
revised, and if it still fails no draft is written: the turn escalates.
Citations that fail verification are sent back for a redraft. If they still
fail, a sentence whose source doesn't state it at all (or isn't a source the
memorandum established) is removed; if nothing cited is left, no answer is
given. A partly supported citation stays, marked unverified. Either way the
turn escalates. When several laws match the question, or it names a section
the index doesn't hold, the answer says so up front.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docslides.config import get_config
from docslides.ingestion.language_detect import detect_language
from docslides.legal import amendments, audit, prompts, script_check
from docslides.legal.chunking import normalize_hebrew_quotes
from docslides.legal.citations import (
    expand_citations,
    format_citation,
    parse_citations,
    remove_cited_sentences,
    render_with_footnotes,
    sentence_before,
    strip_citations,
)
from docslides.legal.models import ChunkMetadata
from docslides.legal.numeric_check import unsupported_numbers
from docslides.legal.retrieval import (
    RetrievalResult,
    RetrievedLegalChunk,
    amendment_index,
)
from docslides.legal.retrieval import retrieve_question as retrieve  # whole question + each clause
from docslides.legal.validation import (
    check_draft_citations,
    clean_memorandum,
    ground_memorandum,
    record_contrary_search_notes,
    validate_memorandum,
)
from docslides.llm import trace
from docslides.llm.client import (
    ChatMessage,
    LLMCallSite,
    QwenClient,
    SamplingParams,
    get_legal_orchestrator_client,
)
from docslides.llm.schemas import (
    EntailmentVerdict,
    LegalDraft,
    ReplyLanguage,
    ResearchMemorandum,
    ScriptRepair,
    grounded_memorandum_schema,
)
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

StatusFn = Callable[[str], Awaitable[None]]

_GATE_FAILED_NOTICE = {
    "en": "I couldn't produce a validated research memorandum for this question from the indexed sources, "
    "so no draft answer was written. The question has been flagged for review by a licensed attorney.",
    "he": "לא ניתן היה להפיק מזכר מחקר מאומת לשאלה זו מתוך המקורות שבמאגר, ולכן לא נוסחה טיוטת תשובה. "
    "השאלה סומנה לבדיקה של עורך דין מוסמך.",
    "ar": "تعذّر إعداد مذكرة بحث قانوني موثّقة لهذا السؤال من المصادر المفهرسة، لذلك لم تتم صياغة مسودة "
    "إجابة. تم تحويل السؤال لمراجعة محامٍ مرخّص.",
    "fr": "Je n'ai pas pu établir, à partir des sources indexées, un mémorandum de recherche validé pour cette "
    "question ; aucun projet de réponse n'a donc été rédigé. La question a été signalée pour examen par un "
    "avocat habilité.",
}

_NOT_STATED_NOTICE = {
    "en": "The indexed law does not state this: none of the retrieved provisions addresses what was asked. "
    "The question has been flagged for review by a licensed attorney.",
    "he": "החוק שבמאגר אינו קובע זאת: אף אחת מההוראות שנמצאו אינה עוסקת במה שנשאל. "
    "השאלה סומנה לבדיקה של עורך דין מוסמך.",
    "ar": "القانون المفهرس لا ينص على ذلك: لا يتناول أي من الأحكام التي تم العثور عليها ما سُئل عنه. "
    "تم تحويل السؤال لمراجعة محامٍ مرخّص.",
    "fr": "Le droit indexé ne le précise pas : aucune des dispositions trouvées ne traite de la question posée. "
    "La question a été signalée pour examen par un avocat habilité.",
}
_NOT_STATED_RE = re.compile(r"\bNOT STATED\b")

_UNVERIFIED_NOTICE = {
    "en": "The drafted answer could not be verified against the indexed sources, so it was withheld. "
    "The question has been flagged for review by a licensed attorney.",
    "he": "לא ניתן היה לאמת את טיוטת התשובה מול המקורות שבמאגר, ולכן היא לא נמסרה. "
    "השאלה סומנה לבדיקה של עורך דין מוסמך.",
    "ar": "تعذّر التحقق من مسودة الإجابة مقابل المصادر المفهرسة، لذلك لم يتم تقديمها. "
    "تم تحويل السؤال لمراجعة محامٍ مرخّص.",
    "fr": "Le projet de réponse n'a pas pu être vérifié au regard des sources indexées ; il n'a donc pas été "
    "communiqué. La question a été signalée pour examen par un avocat habilité.",
}


def _ambiguity_notice(
    reply_language: str, laws_in_play: list[str], uncovered: list[str], sections: dict[str, list[str]] | None = None
) -> str:
    """Said up front: the question matches several laws -- because a section it names is in
    several of them (`sections`), or on its wording -- and which laws the answer leaves out."""
    sections = sections or {}
    laws = list(dict.fromkeys(law for names in sections.values() for law in names)) or laws_in_play
    if reply_language == "he":
        if sections:
            which = (f"סעיף {next(iter(sections))} מופיע" if len(sections) == 1
                     else "הסעיפים " + ", ".join(sections) + " מופיעים")
            head = (f"השאלה אינה חד־משמעית: {which} ביותר מחוק אחד שבמאגר ({'; '.join(laws)}), "
                    "והשאלה אינה מציינת לאיזה מהם היא מתייחסת.")
        else:
            head = f"השאלה אינה חד־משמעית: הוראות מכמה חוקים שבמאגר מתאימות לה ({'; '.join(laws)})."
        if not uncovered:
            return f"{head} להלן מה שקובע כל אחד מהם."
        missing = f"ב{uncovered[0]}" if len(uncovered) == 1 else "בחוקים הבאים: " + "; ".join(uncovered)
        return f"{head} התשובה שלהלן אינה עוסקת {missing}."
    if sections:
        which = f"section {next(iter(sections))} appears" if len(sections) == 1 else \
            "sections " + ", ".join(sections) + " appear"
        head = (f"This question is ambiguous: {which} in more than one indexed law ({'; '.join(laws)}), "
                "and the question doesn't say which it means.")
    else:
        head = f"This question is ambiguous: provisions of several indexed laws match it ({'; '.join(laws)})."
    if not uncovered:
        return f"{head} What each of them provides follows."
    return f"{head} The answer below does not cover: {'; '.join(uncovered)}."


# The draft already opens by saying the question is ambiguous.
_FLAGS_AMBIGUITY_RE = re.compile(r"(?:אינ[הו]|לא)\s+חד[־-]?\s?משמעי|ambigu", re.IGNORECASE)


def _hebrew_safe(text: str) -> str:
    """`text` with ״ for ASCII quotes if it has Hebrew in it -- for anything handed back to
    the model that it might copy into a JSON string (legal/chunking.normalize_hebrew_quotes).
    A memo claim 'במקום "ה־40" יקראו "ה־43"' left the drafter citing empty lead-ins rather
    than write the numbers; a verifier's explanation quoting 'יו"ר' in a revision request
    cut the redraft off at 'יו'."""
    return normalize_hebrew_quotes(text) if re.search("[֐-׿]", text) else text


def _hebrew_safe_values(value):
    if isinstance(value, str):
        return _hebrew_safe(value)
    if isinstance(value, list):
        return [_hebrew_safe_values(v) for v in value]
    if isinstance(value, dict):
        return {k: _hebrew_safe_values(v) for k, v in value.items()}
    return value


def _memo_for_prompt(memo: ResearchMemorandum) -> str:
    """The memorandum as the drafter sees it -- every Hebrew string with ״ for ASCII quotes.
    The stored memorandum keeps what the model wrote."""
    return json.dumps(_hebrew_safe_values(memo.model_dump()), ensure_ascii=False, indent=1)


def _missing_sections_notice(reply_language: str, sections: list[str]) -> str:
    if reply_language == "he":
        which = f"סעיף {sections[0]}" if len(sections) == 1 else "הסעיפים " + ", ".join(sections)
        return f"נוסח {which} אינו נמצא במאגר, ולכן לא ניתן לומר מה נקבע בו."
    which = f"section {sections[0]}" if len(sections) == 1 else "sections " + ", ".join(sections)
    return f"The text of {which} is not in the index, so what it provides can't be stated from the indexed sources."


@dataclass
class CitationCheck:
    index: int
    claim_id: str
    source_id: str
    relation: str
    sentence: str
    structural_problems: list[str]
    verdict: str | None = None
    explanation: str | None = None

    @property
    def ok(self) -> bool:
        return not self.structural_problems and self.verdict == "entailed"


@dataclass
class LegalTurnResult:
    output: dict  # spec section 9 schema
    display_answer: str
    footnotes: list[dict]
    reply_language: str
    escalation_reasons: list[str]
    audit_path: str
    notes: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)  # chunk_id, source_id, text, distance, via
    analysis_notes: str = ""  # Pass 0's notes
    llm_calls: list[dict] = field(default_factory=list)  # every model call, with its reasoning (llm/trace.py)


# --- language ------------------------------------------------------------------


def _dominant_rtl_script(text: str) -> str | None:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    hebrew = sum("֐" <= c <= "׿" for c in letters)
    arabic = sum("؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ" for c in letters)
    if hebrew / len(letters) >= 0.5:
        return "he"
    if arabic / len(letters) >= 0.5:
        return "ar"
    return None


async def detect_reply_language(qwen: QwenClient, query: str) -> str:
    """Hebrew/Arabic are unambiguous by script. Anything else goes to Qwen,
    because the local detector is restricted to config.languages.supported
    and would force e.g. a Russian question into the nearest supported
    language -- the spec forbids defaulting away from the asker's language."""
    script = _dominant_rtl_script(query)
    if script:
        return script
    try:
        result = await qwen.complete_json(
            [ChatMessage("system", prompts.LANGUAGE_ID_PROMPT), ChatMessage("user", query)],
            LLMCallSite("legal_language_id"),
            schema=ReplyLanguage,
            sampling=SamplingParams(temperature=0.0, max_tokens=32),
        )
        code = result.language.strip().lower()[:2]
        if re.fullmatch(r"[a-z]{2}", code):
            return code
    except Exception as exc:  # noqa: BLE001 -- fall back to the local detector
        logger.warning("legal_language_id_failed", error=str(exc))
    return detect_language(query) or "en"


# --- the question as the model sees it ----------------------------------------------


_THIN_COVERAGE_NOTE = (
    "Retrieval note: none of the retrieved provisions was rated as directly answering this question. "
    "If the evidence does not state the answer, the correct finding is that the indexed law does not "
    "state it -- do not build an answer from loosely related provisions."
)


def _question_block(
    query: str,
    thin_coverage: bool = False,
    laws_in_play: list[str] | None = None,
    missing_sections: list[str] | None = None,
    ambiguous_sections: dict[str, list[str]] | None = None,
) -> str:
    # ״ for the ASCII quote, as in the evidence: a model that copies 'יו"ר' from the question
    # into its JSON answer unescaped cuts the answer off there (legal/chunking.py).
    if _dominant_rtl_script(query) == "he":
        query = normalize_hebrew_quotes(query)
    block = f"User's question:\n{query}"
    if thin_coverage:  # the reranker found nothing that directly answers (legal/retrieval.py)
        block += f"\n\n{_THIN_COVERAGE_NOTE}"
    if laws_in_play:  # provisions of several laws match and the question names none
        block += (
            "\n\nRetrieval note: provisions of several laws match this question -- "
            + "; ".join(laws_in_play)
            + ". Unless the question clearly refers to one of them, say that it is ambiguous and answer "
            "separately for each law, citing each."
        )
    for section, laws in (ambiguous_sections or {}).items():  # "סעיף 25" -- which law's?
        block += (
            f"\n\nRetrieval note: section {section}, which the question names, exists in several laws -- "
            + "; ".join(laws)
            + f". The question doesn't say which law it means: it is ambiguous. Say so first, then state what "
            f"section {section} of EACH of these laws provides, one paragraph per law, naming the law and citing it."
        )
    if missing_sections:  # named by the question, but the index doesn't hold their text
        block += (
            "\n\nRetrieval note: the index does not hold the text of "
            + ", ".join(f"section {s}" for s in missing_sections)
            + ", which the question names. Evidence may refer to it, but what it provides is not available: say "
            "that its text is not in the index, and do not describe its content."
        )
    return block


# A draft's own JSON fields copied into its answer text ("escalation_flag: true",
# "coverage_gaps: ...") -- the reader should never see them.
_ECHOED_FIELD_RE = re.compile(r"(?im)^[ \t]*(?:escalat\w*|coverage_gaps|answer_draft)[ \t]*:.*$\n?")


def _drop_echoed_fields(text: str) -> tuple[str, list[str]]:
    dropped: list[str] = []

    def drop(match: re.Match[str]) -> str:
        if "[[CITE" in match.group(0):  # never drop a citation: footnotes follow the tokens
            return match.group(0)
        dropped.append(match.group(0).strip())
        return ""

    text = _ECHOED_FIELD_RE.sub(drop, text)
    return (re.sub(r"\n{3,}", "\n\n", text).strip() if dropped else text), dropped


def _uncovered_laws(laws_in_play: list[str], source_ids, evidence: dict[str, ChunkMetadata]) -> list[str]:
    """Laws in play that none of `source_ids` belongs to."""
    cited = {evidence[s].law_name for s in source_ids if s in evidence}
    return [law for law in laws_in_play if law not in cited]


def _for_model(error: str) -> str:
    """A gate error in the words of the memo the model writes (GroundedMemorandum)."""
    return _hebrew_safe(error.replace("has no supporting_authority", "lists no source_ids"))


def _task_input(evidence_text: str, question: str, notes: str = "") -> str:
    text = f"<evidence_set>\n{evidence_text}\n</evidence_set>\n\n{question}"
    return f"{text}\n\n{prompts.analysis_block(notes)}" if notes else text


# --- Pass 0 ---------------------------------------------------------------------------


async def analyze_question(qwen: QwenClient, evidence_text: str, question: str) -> str:
    """Pass 0: with thinking on, the model reads the evidence against the question and
    writes notes -- what is asked, the decisive words, a direct answer (or NOT STATED /
    DELEGATED / AMBIGUOUS / NOT IN INDEX), the exceptions, the pitfalls -- that the
    memorandum and the draft both start from.

    Every other call returns grammar-constrained JSON with thinking off, so this is
    where the model reasons before it commits: the misreadings seen in evals -- the
    replaced number instead of the replacing one, a deeming provision read backwards,
    an answer built from a provision on a related topic -- happen before any JSON is
    written. Its reasoning goes to the trace. Returns "" when the pass is off or
    yields nothing; the turn then runs without notes."""
    cfg = get_config().legal.pipeline
    if not cfg.analysis_pass:
        return ""
    try:
        notes = await qwen.complete_text(
            [ChatMessage("system", prompts.ANALYSIS_PROMPT), ChatMessage("user", _task_input(evidence_text, question))],
            LLMCallSite("legal_analysis"),
            # Qwen3's recommended thinking-mode sampling: greedy decoding makes it loop.
            sampling=SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=cfg.analysis_max_tokens,
                                    seed=0),
        )
    except Exception as exc:  # noqa: BLE001 -- the notes help; the turn doesn't depend on them
        logger.warning("legal_analysis_failed", error=str(exc))
        return ""
    return _hebrew_safe(notes.strip())


# --- Pass A ----------------------------------------------------------------------------


async def research_memorandum(
    qwen: QwenClient,
    evidence_text: str,
    question: str,
    evidence: dict[str, ChunkMetadata],
    evidence_texts: dict[str, str],
    attempts_log: list,
    laws_in_play: list[str] | None = None,
    notes: str = "",
) -> tuple[ResearchMemorandum | None, list[str]]:
    """Returns (memorandum, gate errors still unresolved). When several laws are in
    play, a memorandum without a claim from each is revised too -- but that alone
    never fails the gate: the answer then says which laws it leaves out."""
    max_revisions = get_config().legal.pipeline.max_memo_revisions
    schema = grounded_memorandum_schema(list(evidence))
    messages = [
        ChatMessage("system", prompts.RESEARCH_MEMO_PROMPT),
        ChatMessage("user", _task_input(evidence_text, question, notes)),
    ]
    errors: list[str] = ["no memorandum produced"]
    memo: ResearchMemorandum | None = None
    best: tuple[ResearchMemorandum, list[str], list[str], str] | None = None  # memo, errors, uncovered, json
    for attempt in range(1 + max_revisions):
        try:
            grounded = await qwen.complete_json(
                messages,
                LLMCallSite("legal_research_memo"),
                schema=schema,
                sampling=SamplingParams(temperature=0.0, max_tokens=3072),  # same evidence, same memo
            )
        except Exception as exc:  # noqa: BLE001 -- schema failure after retries counts as a failed attempt
            errors = [f"memorandum generation failed: {exc}"]
            attempts_log.append({"attempt": attempt + 1, "memorandum": None, "errors": errors})
            continue
        memo, attached = ground_memorandum(grounded, evidence, evidence_texts)
        memo = clean_memorandum(memo)
        memo, auto_notes = record_contrary_search_notes(memo)
        errors = validate_memorandum(memo, evidence)
        uncovered = [
            f"the question is ambiguous -- provisions of several laws match it ({'; '.join(laws_in_play or [])}) -- "
            f"but no claim cites {law}: add a claim, with that law's source_ids, for what it provides on the question"
            for law in _uncovered_laws(laws_in_play or [], [a.source_id for a in memo.supporting_authority], evidence)
        ]
        attempts_log.append({
            "attempt": attempt + 1, "memorandum": memo.model_dump(), "errors": errors,
            "uncovered_laws": uncovered, "auto_recorded_notes": auto_notes, "auto_attached_sources": attached,
        })
        if not errors and not uncovered:
            return memo, []
        # A revision can come back worse than what it revised (a small model asked to add one
        # note may drop a claim's sources instead): always revise from, and fall back to,
        # the attempt with the fewest problems -- gate errors first.
        if best is None or (len(errors), len(uncovered)) < (len(best[1]), len(best[2])):
            best = (memo, errors, uncovered, grounded.model_dump_json())
        _, base_errors, base_uncovered, base_json = best
        messages = [
            *messages[:2],
            ChatMessage("assistant", base_json),
            ChatMessage(
                "user",
                "The memorandum failed validation and cannot go forward to drafting. Keep everything that is "
                "already correct -- in particular every claim's source_ids -- and fix only the problems "
                "below. Do not copy these problem descriptions into any field. Return the complete corrected "
                "memorandum:\n- " + "\n- ".join(_for_model(e) for e in [*base_errors, *base_uncovered]),
            ),
        ]
    if best is not None:
        attempts_log.append({"kept_attempt_with_fewest_problems": len(best[1]) + len(best[2])})
        return best[0], best[1]
    return memo, errors


# --- Pass B + verification ------------------------------------------------------


async def _entailment(
    qwen: QwenClient, check: CitationCheck, claim_text: str, parts: list[RetrievedLegalChunk], sem: asyncio.Semaphore
) -> None:
    source_text = _hebrew_safe("\n".join(p.text for p in parts))  # it quotes the source in its explanation
    async with sem:
        try:
            result = await qwen.complete_json(
                [
                    ChatMessage("system", prompts.ENTAILMENT_PROMPT),
                    ChatMessage(
                        "user",
                        f"Claim ({check.claim_id}): {_hebrew_safe(claim_text)}\n\nDraft sentence: {check.sentence}\n\n"
                        f"Asserted relation: {check.relation}\n\n<evidence source_id=\"{check.source_id}\">\n"
                        f"{source_text}\n</evidence>",
                    ),
                ],
                LLMCallSite("legal_citation_verification"),
                schema=EntailmentVerdict,
                sampling=SamplingParams(temperature=0.0, max_tokens=768),  # the explanation comes first
            )
            check.verdict, check.explanation = result.verdict, result.explanation
        except Exception as exc:  # noqa: BLE001 -- an unverifiable citation is a failed citation
            check.verdict, check.explanation = "not_entailed", f"verification call failed: {exc}"


_GROUNDED_SHARE = 0.8  # share of a sentence's words that must be in its source to count as a restatement
_MIN_GROUNDED_WORDS = 4


def _grounded(sentence: str, source_text: str) -> bool:
    """The sentence restates its source: every number in it is there, and nearly every word is (in
    some prefix form). The 8B verifier sometimes rejects exactly such sentences -- on 25 Sept it read
    "the Constitution Committee" as "not the Knesset" and removed q05's dates -- so a rejected
    restatement is kept, marked unverified, instead of removed."""
    from docslides.legal.keyword import terms

    source_terms = set(terms(source_text))
    words = [w for w in re.findall(r"\S+", sentence) if terms(w)]
    if len(words) < _MIN_GROUNDED_WORDS or unsupported_numbers(sentence, [source_text]):
        return False
    present = sum(1 for w in words if set(terms(w)) & source_terms)
    return present / len(words) >= _GROUNDED_SHARE


def _repair_source_ids(text: str, memo: ResearchMemorandum, evidence: dict[str, ChunkMetadata]) -> tuple[str, list[str]]:
    """A citation naming a source the evidence doesn't have ("source_id=S1") whose claim has exactly
    one source of that relation in the memorandum: that source. The memorandum made the pairing;
    the entailment check still verifies it."""
    pairs: dict[tuple[str, str], set[str]] = {}
    for relation, authorities in (("supports", memo.supporting_authority), ("contrary", memo.contrary_authority)):
        for authority in authorities:
            pairs.setdefault((authority.claim_id, relation), set()).add(authority.source_id)
    pieces, cursor, repaired = [], 0, []
    for citation in parse_citations(text):
        pieces.append(text[cursor : citation.start])
        cursor = citation.end
        relation = citation.relation or "supports"
        sources = pairs.get((citation.claim_id, relation), set())
        if citation.source_id not in evidence and len(sources) == 1:
            (source_id,) = sources
            repaired.append(f"{citation.claim_id}: {citation.source_id or '(none)'} -> {source_id}")
            pieces.append(format_citation(claim_id=citation.claim_id, source_id=source_id, relation=relation))
        else:
            pieces.append(citation.raw)
    pieces.append(text[cursor:])
    return "".join(pieces), repaired


async def verify_citations(
    qwen: QwenClient, draft: str, memo: ResearchMemorandum, retrieval: RetrievalResult, evidence: dict[str, ChunkMetadata]
) -> list[CitationCheck]:
    grouped = retrieval.by_source_id()
    claims = {c.claim_id: c.text for c in memo.governing_law}
    structural = check_draft_citations(draft, memo, evidence)
    checks = [
        CitationCheck(
            index=i,
            claim_id=c.claim_id,
            source_id=c.source_id,
            relation=c.relation,
            sentence=sentence_before(draft, c.start),
            structural_problems=structural.get(i, []),
        )
        for i, c in enumerate(parse_citations(draft))
    ]
    sem = asyncio.Semaphore(get_config().legal.pipeline.entailment_concurrency)
    await asyncio.gather(
        *(
            _entailment(qwen, check, claims[check.claim_id], grouped[check.source_id], sem)
            for check in checks
            if not check.structural_problems
        )
    )
    for check in checks:
        if (check.verdict == "not_entailed" and check.relation == "supports" and not check.structural_problems
                and _grounded(check.sentence, "\n".join(p.text for p in grouped[check.source_id]))):
            check.verdict = "partially_entailed"
            check.explanation = ("kept as unverified: its words and numbers are all in the cited source; "
                                 f"the verifier said: {check.explanation}")
    return checks


def _citation_failures(checks: list[CitationCheck]) -> list[str]:
    failures = []
    for check in checks:
        if check.ok:
            continue
        reasons = check.structural_problems or [f"{check.verdict}: {check.explanation}"]
        failures.append(f"citation #{check.index + 1} ({check.claim_id} -> {check.source_id}): {'; '.join(reasons)}")
    return failures


class DraftUnavailable(Exception):
    """No draft could be generated at all (every attempt malformed, nothing salvageable)."""


_DRAFT_OPENING_RE = re.compile(r'^\s*\{\s*"answer_draft"\s*:\s*"')
_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}


def _unterminated_json_string(body: str) -> str:
    """The value of a JSON string cut off before its closing quote."""
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch == '"':
            break
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(body):
            break
        escaped = body[i + 1]
        if escaped == "u":
            digits = body[i + 2 : i + 6]
            if len(digits) < 4 or not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                break
            out.append(chr(int(digits, 16)))
            i += 6
            continue
        out.append(_JSON_ESCAPES.get(escaped, escaped))
        i += 2
    return "".join(out)


def _salvage_draft(raw: str) -> LegalDraft | None:
    """A draft cut off at max_tokens -- typically a loop repeating its sentences until the
    limit -- kept up to its last complete citation token, each repeated sentence once.
    What survives goes through the same verification as any draft, and the turn escalates."""
    match = _DRAFT_OPENING_RE.match(raw)
    if not match:
        return None
    text = _unterminated_json_string(raw[match.end() :])
    end = text.rfind("]]")
    if end == -1:
        return None
    kept, seen = [], set()
    for piece in re.split(r"(?<=\]\])", text[: end + 2]):  # each piece ends with a citation token
        key = " ".join(strip_citations(piece).split())
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(piece)
    text = "".join(kept).strip()
    if not parse_citations(text):
        return None
    return LegalDraft(
        answer_draft=text,
        escalation_flag=True,
        escalation_reason="The draft ran past its length limit; it was kept up to its last complete citation",
    )


async def draft_answer(
    qwen: QwenClient,
    reply_language: str,
    evidence_text: str,
    question: str,
    memo: ResearchMemorandum,
    retrieval: RetrievalResult,
    evidence: dict[str, ChunkMetadata],
    attempts_log: list,
    laws_in_play: list[str] | None = None,
    notes: str = "",
) -> tuple[LegalDraft, list[CitationCheck], list[str]]:
    """Returns (draft, its citation checks, citation failures still unresolved after
    the last revision). A draft that leaves out a law in play the memorandum
    covers is revised too. A revision that can't be generated leaves the previous
    draft standing; if not even the first can be, raises DraftUnavailable."""
    max_revisions = get_config().legal.pipeline.max_draft_revisions
    memo_laws = {evidence[a.source_id].law_name for a in memo.supporting_authority if a.source_id in evidence}
    messages = [
        ChatMessage("system", prompts.draft_prompt(reply_language)),
        ChatMessage(
            "user",
            f"{_task_input(evidence_text, question, notes)}\n\n"
            f"Validated research memorandum:\n{_memo_for_prompt(memo)}",
        ),
    ]
    previous: tuple[LegalDraft, list[CitationCheck], list[str]] | None = None
    sampling = SamplingParams(temperature=0.0, max_tokens=2048)  # same memo, same draft
    for attempt in range(1 + max_revisions):
        try:
            draft = await qwen.complete_json(
                messages,
                LLMCallSite("legal_draft"),
                schema=LegalDraft,
                sampling=sampling,
                salvage=_salvage_draft,
            )
        except Exception as exc:  # noqa: BLE001 -- malformed after every retry and not salvageable
            attempts_log.append({"attempt": attempt + 1, "draft": None, "error": f"{type(exc).__name__}: {exc}"})
            if previous is not None:
                return previous
            raise DraftUnavailable(f"{type(exc).__name__}: {exc}") from exc
        as_written = draft.model_dump_json()  # short-form tokens, for the revision turn
        repaired_text, repaired = _repair_source_ids(draft.answer_draft, memo, evidence)
        draft.answer_draft = expand_citations(repaired_text, evidence)
        checks = await verify_citations(qwen, draft.answer_draft, memo, retrieval, evidence)
        failures = _citation_failures(checks)
        no_citations = not checks and bool(memo.supporting_authority)
        if no_citations:
            failures.append("the draft contains no [[CITE]] tokens although the memorandum has supporting authority")
        # A draft that stops after a lead-in ("... בכל אחד מהחוקים:") was most likely cut off by an
        # ASCII double quote inside a Hebrew word. At temperature 0 the redraft repeats the same text,
        # so it is sampled differently -- and told why.
        cut_off = no_citations or bool(re.search(r"[:,]\s*$", strip_citations(draft.answer_draft)))
        if cut_off:
            failures.append("the draft stops mid-answer: it was probably cut off by an ASCII double-quote character "
                            "inside a word -- write ״ instead, and finish every sentence with its citation token")
        uncovered = [law for law in _uncovered_laws(laws_in_play or [], [c.source_id for c in checks], evidence)
                     if law in memo_laws]
        attempts_log.append({"attempt": attempt + 1, "draft": draft.model_dump(), "repaired_source_ids": repaired,
                             "cut_off": cut_off, "citation_checks": [asdict(c) for c in checks],
                             "uncovered_laws": uncovered})
        if (not failures and not uncovered) or attempt == max_revisions:
            return draft, checks, failures
        previous = (draft, checks, failures)
        if cut_off:
            sampling = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048, seed=attempt + 1)
        problems = failures + [
            f"the question is ambiguous -- provisions of several laws match it ({'; '.join(laws_in_play or [])}) -- "
            f"and the memorandum has claims for {law}, but the draft doesn't cite it: say that the question is "
            "ambiguous and answer separately for each law, naming it and citing its claims"
            for law in uncovered
        ]
        messages = [
            *messages[:2],
            ChatMessage("assistant", as_written),
            ChatMessage(
                "user",
                "The draft failed verification. Revise it: fix each problem below, and if no source the memorandum "
                "lists actually states a sentence's proposition, remove that proposition rather than cite loosely. "
                "Do not add claims the memorandum doesn't establish.\n- "
                + "\n- ".join(_hebrew_safe(p) for p in problems),
            ),
        ]
    raise AssertionError("unreachable")


async def repair_foreign_words(
    qwen: QwenClient, text: str, reply_language: str, allowed: set[str]
) -> tuple[str, dict]:
    """Replaces words written in the wrong script (legal/script_check.py) with the
    reply-language words the model gives for them. Only the flagged words change:
    a replacement is applied by exact whole-word match, and only if it is itself
    clean. Returns (text, log with what was flagged, replaced and remains)."""
    words = script_check.foreign_words(text, reply_language, allowed)
    log: dict = {"flagged": words}
    if not words:
        return text, log
    try:
        result = await qwen.complete_json(
            [
                ChatMessage("system", prompts.script_repair_prompt(reply_language)),
                ChatMessage("user", "Words:\n- " + "\n- ".join(words) + "\n\nSentences:\n"
                            + "\n".join(script_check.sentences_with(text, words))),
            ],
            LLMCallSite("legal_script_repair"),
            schema=ScriptRepair,
            sampling=SamplingParams(temperature=0.0, max_tokens=512),
        )
        replacements = {
            r.word: r.replacement.strip() for r in result.repairs
            if r.word in words and not script_check.foreign_words(r.replacement, reply_language, allowed)
        }
    except Exception as exc:  # noqa: BLE001 -- an unrepaired word is reported, not fatal
        log["error"] = str(exc)
        replacements = {}
    text = script_check.replace_words(text, replacements)
    log.update(replacements=replacements, remaining=script_check.foreign_words(text, reply_language, allowed))
    return text, log


# --- assembly -------------------------------------------------------------------------


def _escalation_reasons(
    draft: LegalDraft | None,
    memo: ResearchMemorandum | None,
    retrieval: RetrievalResult,
    evidence: dict[str, ChunkMetadata],
    checks: list[CitationCheck],
    draft_failures: list[str] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if draft and draft.escalation_flag:
        reasons.append(draft.escalation_reason or "Flagged for escalation by the drafting model")
    if retrieval.low_relevance:
        reasons.append("Retrieved sources have low relevance or thin coverage for this question")
    if memo and memo.authority_conflicts:
        reasons.append("Unresolved conflict between authorities: " + "; ".join(memo.authority_conflicts))
    if memo and memo.supporting_authority:
        uploads = sum(evidence.get(a.source_id) is not None and evidence[a.source_id].source_type == "uploaded_document"
                      for a in memo.supporting_authority)
        if uploads * 2 > len(memo.supporting_authority):
            reasons.append("The analysis relies primarily on uploaded documents rather than official statutes or rulings")
    cited = {c.source_id for c in checks}
    stale = sorted({evidence[s].law_name + " " + evidence[s].section_number for s in cited
                    if s in evidence and evidence[s].status != "current"})
    if stale:
        reasons.append("Cites provisions marked amended or repealed: " + ", ".join(stale))
    failed = [c for c in checks if not c.ok]
    if failed:
        reasons.append(f"{len(failed)} citation(s) could not be verified against their sources")
    if draft_failures and not failed:
        reasons.append("The draft failed citation verification: " + "; ".join(draft_failures))
    if retrieval.rejected_chunk_ids:
        reasons.append("Some index entries failed signed-bundle verification and were excluded")
    return reasons


def _footnotes(final_text: str, evidence: dict[str, ChunkMetadata], checks: list[CitationCheck]) -> tuple[str, list[dict]]:
    display, citations, numbers = render_with_footnotes(final_text)
    notes: dict[int, dict] = {}
    for citation, number, check in zip(citations, numbers, checks):
        meta = evidence.get(citation.source_id)
        note = notes.setdefault(
            number,
            {
                "number": number,
                "source_id": citation.source_id,
                "law": meta.law_name if meta else citation.law,
                "section": meta.display_section if meta else citation.section,
                "breadcrumb": meta.breadcrumb if meta else "",
                "effective": f"{meta.effective_date_start} – {meta.effective_date_end or 'current'}" if meta else citation.effective,
                "status": meta.status if meta else "unknown",
                "source_type": meta.source_type if meta else citation.source_type,
                "source_origin": meta.source_origin if meta else "",
                "relations": [],
                "verified": True,
                "problems": [],
            },
        )
        if citation.relation not in note["relations"]:
            note["relations"].append(citation.relation)
        if not check.ok:
            note["verified"] = False
            note["problems"].extend(check.structural_problems or [f"{check.verdict}: {check.explanation}"])
    return display, [notes[n] for n in sorted(notes)]


def _amendment_notes(evidence: dict[str, ChunkMetadata]) -> dict[str, list]:
    """source_id -> later indexed amendments to that provision's law (legal/amendments.py)."""
    try:
        index = amendment_index()
    except Exception as exc:  # noqa: BLE001 -- a missing index must not block answering
        logger.warning("legal_amendment_index_unavailable", error=str(exc))
        return {}
    notes = {sid: amendments.notes_for(meta, index) for sid, meta in evidence.items()}
    return {sid: n for sid, n in notes.items() if n}


async def _localized_notice(qwen: QwenClient, reply_language: str, notices: dict[str, str]) -> str:
    if reply_language in notices:
        return notices[reply_language]
    try:
        return await qwen.complete_text(
            [
                ChatMessage("system", f"Translate the user's text into {prompts.language_name(reply_language)}. "
                                      "Return only the translation."),
                ChatMessage("user", notices["en"]),
            ],
            LLMCallSite("legal_draft"),
            sampling=SamplingParams(temperature=0.0, max_tokens=512),
        )
    except Exception:  # noqa: BLE001
        return notices["en"]


async def _no_answer(
    qwen: QwenClient,
    entry: dict,
    reply_language: str,
    notices: dict[str, str],
    reasons: list[str],
    memo: ResearchMemorandum | None,
    retrieved: list[dict],
    coverage_gaps: str | None = None,
) -> LegalTurnResult:
    """A turn that ends without an answer: a notice in the reply language, escalated."""
    notice = await _localized_notice(qwen, reply_language, notices)
    output = {
        "research_memorandum": memo.model_dump() if memo else None,
        "answer_draft": notice,
        "escalation_flag": True,
        "escalation_reason": "; ".join(reasons),
        "coverage_gaps": coverage_gaps,
    }
    entry.update(output=output)
    path = _write_audit(entry)
    return LegalTurnResult(output, notice, [], reply_language, reasons, str(path), retrieved_chunks=retrieved,
                           analysis_notes=entry.get("analysis_notes", ""))


def _write_audit(entry: dict) -> Path:
    """The audit entry, closed by every model call the turn made -- reasoning included."""
    entry["llm_calls"] = trace.current_calls() or []
    return audit.write_entry(entry)


async def run_legal_turn(query: str, job_id: str, status: StatusFn) -> LegalTurnResult:
    with trace.collect(job_id) as calls:
        result = await _legal_turn(query, job_id, status)
    result.llm_calls = calls
    return result


async def _legal_turn(query: str, job_id: str, status: StatusFn) -> LegalTurnResult:
    qwen = get_legal_orchestrator_client()
    entry: dict = {"job_id": job_id, "query": query, "orchestrator_model": qwen.model}

    await status("Detecting the question's language")
    reply_language = await detect_reply_language(qwen, query)
    entry["reply_language"] = reply_language

    await status("Searching the Israeli-law index")
    retrieval = await asyncio.to_thread(retrieve, query)
    grouped = retrieval.by_source_id()
    evidence = {source_id: parts[0].metadata for source_id, parts in grouped.items()}
    amendment_notes = _amendment_notes(evidence)
    evidence_text = prompts.format_evidence(grouped, amendment_notes, retrieval.indexed_law_keys)
    retrieved = [
        {"chunk_id": c.chunk_id, "source_id": c.metadata.source_id, "text": c.text, "distance": c.distance, "via": c.via}
        for c in retrieval.chunks
    ]
    question = _question_block(query, thin_coverage=retrieval.low_relevance, laws_in_play=retrieval.laws_in_play,
                               missing_sections=retrieval.missing_sections,
                               ambiguous_sections=retrieval.ambiguous_sections)
    entry["retrieval"] = {
        "query": query,
        "bundle_verification": retrieval.bundle_verification,
        "best_distance": retrieval.best_distance,
        "best_rerank_score": retrieval.best_rerank_score,
        "laws_in_play": retrieval.laws_in_play,
        "ambiguous_sections": retrieval.ambiguous_sections,
        "missing_sections": retrieval.missing_sections,
        "low_relevance": retrieval.low_relevance,
        "amendment_notes": {sid: [n.describe() for n in notes] for sid, notes in amendment_notes.items()},
        "rejected_chunk_ids": retrieval.rejected_chunk_ids,
        "duplicate_chunk_ids": retrieval.duplicate_chunk_ids,
        "trimmed_chunk_ids": retrieval.trimmed_chunk_ids,
        "chunks": [
            {"chunk_id": c.chunk_id, "source_id": c.metadata.source_id, "distance": c.distance, "via": c.via,
             "score": c.score}
            for c in retrieval.chunks
        ],
    }

    analysis_notes = ""
    if grouped:
        await status(f"Pass 0: reading {len(grouped)} source(s) against the question (thinking)")
        analysis_notes = await analyze_question(qwen, evidence_text, question)
        entry["analysis_notes"] = analysis_notes

    await status(f"Pass A: research memorandum over {len(grouped)} source(s)")
    memo_attempts: list = []
    evidence_texts = {source_id: "\n".join(p.text for p in parts) for source_id, parts in grouped.items()}
    memo, gate_errors = await research_memorandum(
        qwen, evidence_text, question, evidence, evidence_texts, memo_attempts, retrieval.laws_in_play,
        analysis_notes,
    )
    entry["memorandum_attempts"] = memo_attempts

    if gate_errors:
        reasons = ["The research memorandum failed validation after revision: " + "; ".join(gate_errors[:5])]
        reasons += _escalation_reasons(None, memo, retrieval, evidence, [])
        if memo is not None and not memo.governing_law and (_NOT_STATED_RE.search(analysis_notes)
                                                             or retrieval.low_relevance):
            # Nothing in the evidence states what was asked -- which is the answer (a count of cases, a
            # fine the law never set), not a failure to produce one.
            reasons.insert(0, "No retrieved provision states what was asked (Pass 0: NOT STATED)")
            return await _no_answer(qwen, entry, reply_language, _NOT_STATED_NOTICE, reasons, memo, retrieved)
        return await _no_answer(qwen, entry, reply_language, _GATE_FAILED_NOTICE, reasons, memo, retrieved)

    await status("Pass B: drafting the answer and verifying every citation")
    draft_attempts: list = []
    entry["draft_attempts"] = draft_attempts
    try:
        draft, checks, draft_failures = await draft_answer(
            qwen, reply_language, evidence_text, question, memo, retrieval, evidence, draft_attempts,
            retrieval.laws_in_play, analysis_notes,
        )
    except DraftUnavailable as exc:
        reasons = [f"No well-formed draft could be generated: {exc}"]
        reasons += _escalation_reasons(None, memo, retrieval, evidence, [])
        return await _no_answer(qwen, entry, reply_language, _UNVERIFIED_NOTICE, reasons, memo, retrieved)
    notes: list[str] = []
    final_text = draft.answer_draft

    # A sentence whose citation failed outright -- the source doesn't state it, or isn't one the
    # memorandum established for it -- is not shipped. A partly supported one stays, marked
    # unverified, and escalates.
    failed_outright = {c.index for c in checks if c.structural_problems or c.verdict == "not_entailed"}
    removed: set[int] = set()
    removed_failures: list[str] = []
    if failed_outright:
        final_text, removed = remove_cited_sentences(final_text, failed_outright)
        entry["removed_sentences"] = [
            {"sentence": checks[i].sentence, "claim_id": checks[i].claim_id, "source_id": checks[i].source_id,
             "problems": checks[i].structural_problems or [f"{checks[i].verdict}: {checks[i].explanation}"]}
            for i in sorted(removed)
        ]
        removed_failures = _citation_failures([checks[i] for i in sorted(removed)])
        checks = [c for c in checks if c.index not in removed]
        draft_failures = _citation_failures(checks)
    if memo.supporting_authority and not parse_citations(final_text):
        # Nothing verified is left to say, or the draft never cited anything: no answer.
        failures = removed_failures if removed else draft_failures
        reasons = ["The draft's citations failed verification, so no answer was given: " + "; ".join(failures[:5])]
        reasons += _escalation_reasons(draft, memo, retrieval, evidence, [])
        entry["withheld_draft"] = draft.answer_draft
        return await _no_answer(qwen, entry, reply_language, _UNVERIFIED_NOTICE, reasons, memo, retrieved,
                                draft.coverage_gaps)

    await status("Checking the answer's wording")
    final_text, echoed = _drop_echoed_fields(final_text)
    if echoed:
        entry["dropped_field_lines"] = echoed
    allowed = script_check.allowed_words([query, *evidence_texts.values()])
    final_text, script_log = await repair_foreign_words(qwen, final_text, reply_language, allowed)
    entry["script_check"] = script_log

    await status("Final citation check")
    final_structural = check_draft_citations(final_text, memo, evidence)
    entry["final_integrity"] = {
        "structural_problems": {f"citation #{i + 1}": p for i, p in final_structural.items()},
    }

    reasons = _escalation_reasons(draft, memo, retrieval, evidence, checks, draft_failures)
    amended_citations = sorted({
        f"{evidence[c.source_id].law_name} סעיף {evidence[c.source_id].section_number}: "
        + "; ".join(n.describe() for n in amendment_notes[c.source_id] if n.touches_section)
        for c in checks
        if any(n.touches_section for n in amendment_notes.get(c.source_id, []))
    })
    if amended_citations:
        reasons.append(
            "Cites provisions a later indexed law amended -- confirm which version applies to the facts' date: "
            + " | ".join(amended_citations)
        )

    # Numeric grounding: every number the answer states must be in what it cites.
    cited_ids = {c.source_id for c in parse_citations(final_text)}
    numeric_evidence = [
        c.text for c in retrieval.chunks if not cited_ids or c.metadata.source_id in cited_ids
    ]
    ungrounded = unsupported_numbers(strip_citations(final_text), numeric_evidence, question=query)
    entry["numeric_check"] = {"unsupported": ungrounded, "evidence_chunks": len(numeric_evidence)}
    if ungrounded:
        reasons.append("Numbers in the answer not found in the cited sources: " + ", ".join(ungrounded))
        notes.append("Check these figures against the law: " + ", ".join(ungrounded))

    if removed:
        reasons.append(f"Removed {len(entry['removed_sentences'])} statement(s) whose citations failed verification")
        notes.append(f"{len(entry['removed_sentences'])} statement(s) were removed from the draft because the cited "
                     "sources did not support them.")
    if script_log.get("remaining"):
        reasons.append("Words in another script remain in the answer: " + ", ".join(script_log["remaining"]))
        notes.append("Some words are not in the answer's language: " + ", ".join(script_log["remaining"]))

    # Said up front, whatever the draft says: sections the question names that the index lacks,
    # laws in play the answer leaves out, and a named section several laws have (unless the
    # draft already opens by saying the question is ambiguous).
    preface: list[str] = []
    if retrieval.missing_sections:
        reasons.append("The question names section(s) whose text is not in the index: "
                       + ", ".join(retrieval.missing_sections))
        preface.append(_missing_sections_notice(reply_language, retrieval.missing_sections))
    uncovered = _uncovered_laws(retrieval.laws_in_play, [c.source_id for c in parse_citations(final_text)], evidence)
    if uncovered:
        reasons.append("Several laws match the question and the answer leaves out: " + "; ".join(uncovered))
    opening = strip_citations(final_text)[:300]
    if uncovered or (retrieval.ambiguous_sections and not _FLAGS_AMBIGUITY_RE.search(opening)):
        preface.append(_ambiguity_notice(reply_language, retrieval.laws_in_play, uncovered,
                                         retrieval.ambiguous_sections))
    if preface:
        final_text = "\n\n".join([*preface, final_text])

    display, footnotes = _footnotes(final_text, evidence, checks)
    for note in footnotes:
        note["amended_by"] = [n.describe() for n in amendment_notes.get(note["source_id"], [])]
    output = {
        "research_memorandum": memo.model_dump(),
        "answer_draft": final_text,
        "escalation_flag": bool(reasons),
        "escalation_reason": "; ".join(reasons) or None,
        "coverage_gaps": draft.coverage_gaps,
    }
    entry.update(output=output, footnotes=footnotes)
    path = _write_audit(entry)
    return LegalTurnResult(output, display, footnotes, reply_language, reasons, str(path), notes,
                           retrieved_chunks=retrieved, analysis_notes=analysis_notes)
