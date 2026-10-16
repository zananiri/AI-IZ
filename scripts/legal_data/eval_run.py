#!/usr/bin/env python3
"""Answer, then judge, legal_txt/Evals/israeli_legal_eval/questions.jsonl (the 500-question
set) against the bulk legal corpus (data/legal_corpus_vectordb, scripts/legal_data/vectorize.py)
-- a separate, much broader index from the Legal tab's own signed bundle (src/docslides/legal/),
which this eval set is too broad to run against (five curated laws vs. the eval's eleven areas
of law).

    # 1) answer, always with retrieval over the corpus (there is no model-alone mode).
    #    Skips ids already in --out, so re-running after an interruption resumes.
    python scripts/legal_data/eval_run.py answer \
        --questions legal_txt/Evals/israeli_legal_eval/questions.jsonl \
        --out data/legal/eval/bulk500/answers_rag.jsonl

    # 2) build judge requests (unmodified israeli_legal_eval/score.py):
    python legal_txt/Evals/israeli_legal_eval/score.py prepare \
        --questions legal_txt/Evals/israeli_legal_eval/questions.jsonl \
        --gold legal_txt/Evals/israeli_legal_eval/gold.jsonl \
        --answers data/legal/eval/bulk500/answers_rag.jsonl \
        --out data/legal/eval/bulk500/judge_requests_rag.jsonl

    # 3) judge (fills judge_requests.jsonl's prompts in, writes judged.jsonl):
    python scripts/legal_data/eval_run.py judge \
        --requests data/legal/eval/bulk500/judge_requests_rag.jsonl \
        --out data/legal/eval/bulk500/judged_rag.jsonl

    # 4) report (unmodified score.py):
    python legal_txt/Evals/israeli_legal_eval/score.py report \
        --gold legal_txt/Evals/israeli_legal_eval/gold.jsonl \
        --answers data/legal/eval/bulk500/answers_rag.jsonl \
        --judged data/legal/eval/bulk500/judged_rag.jsonl \
        --json-out data/legal/eval/bulk500/report_rag.json

Per question: (0) a first reading flags questions outside the index -- foreign law, rulings or
statistics, a false premise, a law or section that doesn't exist, a harmful request -- and the answer
call gets a note on how to handle that case (legal_eval_scope); (1) the answering model lists the
issues and the laws/sections it believes govern them (legal_eval_plan, thinking off), and a question
about which version of a law applies also searches for commencement and transitional provisions; (2) retrieve_planned searches on the question and on
each issue (dense + BM25), looks the named sections up directly, drops out-of-scope records (West
Bank orders, drafts, repealed law), keeps at most two chunks per section and reranks with
legal.retrieval.reranker_model; (3) the answer call (ANSWER_SYSTEM); (4) an empty answer is
retried without thinking, one containing other scripts is rewritten into Hebrew (legal_eval_rewrite,
up to two passes), a כן/לא opener the instructions didn't ask for is dropped, a rule_conclusion answer
-- written explanation first, ending "מסקנה: כן/לא" -- gets that conclusion moved to the front (one with
no conclusion line and no label gets the label its explanation supports, legal_eval_label_check), and
leaked chat-template tokens are removed. The plan and any repairs
are recorded on the answers.jsonl row.

Variants (--variant, each a legal.corpus switch; all on by default, --variant <names> turns on only
the named ones and --variant baseline none, to measure one at a time on the v2 dev split): whole_sections (a split section retrieved whole), toc (the model picks sections from the
top laws' tables of contents), xref (sections a retrieved section refers to), reg_cap (at most 3
regulation excerpts unless the plan names one), grouped (context ordered by law and section),
extract (a first call quotes the governing provisions and lists their elements and exceptions; the
answer must cover them), completeness (extract, plus a check that the answer covers every element of
the provisions it cites, and a revision when it doesn't), doctrines (legal_txt/doctrine_cards.jsonl: case-law doctrines and
amendment timelines, labelled as notes, not statute text).

Reasoning: every LLM call is written in full -- prompt, reasoning ("thinking"), output, timing --
to <logging.llm_trace_dir>/<date>.jsonl (llm/trace.py), with job_id set to the question id, as
long as DOCSLIDES_CONFIG's logging.llm_trace_dir is set (see notebooks/kaggle_legal_eval_bulk500.ipynb).
scripts/llm_trace_report.py renders that file to Markdown, one section per question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal import caselaw, corpus_navigation
from docslides.legal.corpus_retrieval import (  # shared with the Legal tab's corpus path
    PLAN_MAX_TOKENS,
    PLAN_PROMPT,
    in_scope,
    retrieve_planned,
    warm_up_retrieval,
)
from docslides.legal.evaluation import get_judge_client
from docslides.legal_data.hebrew import normalize_for_embedding
from docslides.llm import trace
from docslides.llm.client import (
    ChatMessage,
    LLMCallSite,
    SamplingParams,
    get_legal_orchestrator_client,
)
from docslides.llm.schemas import BulkEvalJudgement, EvalIssue, EvalRetrievalPlan
from docslides.rag.embedding import embed_texts

ANSWER_SYSTEM = """You are a legal assistant answering questions about ISRAELI law, in Hebrew.

<context> holds excerpts retrieved from an index of Israeli legislation. Some are relevant and some \
are not.
- Base the answer on the excerpts that actually govern the question. Ignore excerpts about other \
subjects, and never cite a law just because it appears in <context>.
- If no excerpt governs the question, answer from the core Israeli statutes you are certain of and \
cite them by name and section.
- Cite only the few provisions the answer relies on, usually one to three. Copy law names and \
section numbers exactly as they appear in <context>; when a provision's text is in <context>, trust \
it over your memory. Never invent a section, subsection, number or date.
- When <context> shows that a section was changed or added by an amending law (תיקון), cite that \
amending law as well.
- Follow the question's <instructions> exactly. Open with כן or לא only when the <instructions> ask \
for a yes/no answer; then that first word answers the question exactly as asked, and the explanation \
must agree with it. Every other answer opens directly with the rule or the analysis.
- A law the question calls by its common short name is that law: חוק התרופות means חוק החוזים \
(תרופות בשל הפרת חוזה), חוק השכירות means חוק השכירות והשאילה. Answer about it under its full name; \
never say such a law does not exist or correct the user for the short name.
- If the question assumes a law, amendment, section or fact that does not exist or is wrong, say so \
plainly and give the correct rule. If a request asks for help deceiving, threatening, hiding assets \
or otherwise evading the law, refuse and briefly name the lawful alternative. If the answer depends \
on facts that were not given, say which facts are needed.
- Write only in Hebrew. Do not use words from any other language or script, and do not refer to \
the excerpts by number or mention <context>."""

ISSUE_SPOTTING_NOTE = """This is an issue-spotting question: list every distinct legal issue the \
facts raise, across all areas of law (civil, contracts, torts, labour, property, consumer, privacy, \
criminal, procedure), one per line, in the form: הסוגיה – החוק והסעיף."""

REWRITE_PROMPT = """The answer below contains words in other languages or scripts ({words}). \
Rewrite it entirely in Hebrew. Keep its content, structure, law names and section numbers exactly \
as they are. Return only the rewritten answer.

<answer>
{answer}
</answer>"""

# The 29 Sept review found Gemma 12B opening the answer with a label its own reasoning contradicts
# (IL-016, IL-214, IL-219). A second read of the explanation alone -- label removed, so it can't
# anchor -- decides which answer the reasoning supports.
LABEL_CHECK_PROMPT = """Below is a yes/no question about Israeli law and the explanation part of an \
answer to it. Read only the explanation: does it lead to "yes" or to "no" as the answer to the \
question exactly as asked? Judge what the explanation says, not whether it is correct.

<question>{question}</question>

<explanation>
{explanation}
</explanation>"""

# rule_conclusion answers: the explanation first, the label last. In the 29 Sept full 27B run, 15 of 72
# labels were wrong and 8 of those answers argued the right conclusion after a wrong opening word: a
# label written first commits the model before it has reasoned. A second model reading the explanation
# (check_label) broke more labels than it fixed, so here the answering model writes the conclusion
# itself, last, and place_conclusion moves it to the front where score.py reads it.
CONCLUSION_NOTE = """Write the explanation first. Then end the answer with one separate last line, \
exactly: מסקנה: כן  or  מסקנה: לא -- the answer to the question exactly as asked (if the question \
asks "can he ... without X?" and he cannot, the answer is לא). Do not open the answer with כן or לא: \
the conclusion line is moved to the front for you."""
_CONCLUSION_RE = re.compile(r"\n?[ \t>*_-]*מסקנה[ \t*_]*[:：][ \t*_]*(כן|לא)\b[^\n]*\s*$")

# A first reading of the question, before retrieval: questions the index can't or mustn't answer.
# The 29 Sept full 27B run answered 9 of 24 abstention items anyway (California leave law from
# memory, "section 999" with another section's text, an invented summary of a ruling).
SCOPE_PROMPT = """Classify this question put to a legal assistant whose only source is an index of \
ISRAELI legislation (laws and regulations -- no case law, news, statistics or personal data).

- in_scope: an ordinary question about Israeli law, even a hard or hypothetical one
- foreign_law: it asks about the law of another country
- not_in_legislation: it asks for court rulings or case numbers, news, speeches, statistics, or \
personal details of real people
- false_premise: it asserts a legal rule, period or amendment that is wrong, and asks a question \
built on it
- nonexistent_law_or_section: it names a law that does not exist, or a section number the named \
law does not have (for example section 250 of a law of about 60 sections). A real law called by its \
common short name is not this: חוק התרופות is חוק החוזים (תרופות בשל הפרת חוזה), חוק השכירות \
is חוק השכירות והשאילה
- harmful_request: it asks for help deceiving, threatening, forging, hiding assets or evading the law

Choose in_scope unless you are sure. In note, say in a few Hebrew words what is wrong (empty for \
in_scope).

<question>{question}</question>"""

SCOPE_NOTES = {
    "foreign_law": "The question appears to ask about foreign law, which the index does not cover. If so, "
                   "say so in one or two sentences and do not describe the foreign law from memory; you may "
                   "name the Israeli rule on the same subject if <context> states it.",
    "not_in_legislation": "The question appears to ask for material the index does not hold (rulings, case "
                          "numbers, news, statistics or personal details). If so, say plainly that it is not "
                          "available here and never invent a ruling, number or fact; add the relevant "
                          "statutory rule only if <context> states it.",
    "false_premise": "The question may rest on a wrong premise. If the premise is wrong, open by saying so "
                     "and give the correct rule from <context>; do not answer as if the premise were true.",
    "nonexistent_law_or_section": "The question may name a law or section that does not exist. If <context> "
                                  "does not contain it, say it does not exist in the index, do not describe "
                                  "its content, and point to the provision that does govern the subject if "
                                  "<context> has one. A real law called by its common short name (חוק "
                                  "התרופות) does exist: then answer about that law.",
    "harmful_request": "The request may ask for help deceiving, threatening or evading the law. If so, "
                       "refuse briefly and describe the lawful alternative.",
}

# Questions about which version of a law applies scored 25% in the 29 Sept full 27B run: the model
# answered from the current text or from memory ("manslaughter still exists", "Amendment 3 did not
# change s.25(a)"). Retrieval also looks for the commencement and transitional provisions.
TEMPORAL_RE = re.compile(r"תיקון מס|בתיקון|לפני התיקון|נוסח(?:ו)? (?:החדש|הקודם)|בוטל|הוראת השעה|הוראת שעה|"
                         r"נכנס(?:ה)? לתוקף|כיום|בעבר|תמיד היה|תמיד הייתה|תמיד ניתן")
TEMPORAL_ISSUE = "תחילה תחולה והוראות מעבר"
TEMPORAL_NOTE = """This question is about which version of a law applies, or what changed and when. Look \
in <context> for the commencement (תחילה), application (תחולה) and transitional (הוראות מעבר) provisions \
and for amending laws (תיקון). Say which version applies to the date or facts in the question and why. If \
<context> does not show when the provision changed, say so and name the version to check. Never state from \
memory that a law, offence or section still exists, or that an amendment left a section unchanged."""

_TEMPLATE_TOKEN_RE = re.compile(r"\s*</?(?:start|end)_of_turn>\s*")

# Extract, then answer (legal.corpus.extract_then_answer). In the 29 Sept full 27B run half the
# half-credit answers (49 of 91) had the governing section in <context> and still left out a
# condition, exception or qualifier. A first call quotes the provisions and lists every element;
# the answer call is then told to cover each one.
EXTRACT_PROMPT = """Read the question about Israeli law and the excerpts in <context>. Do not answer yet.

provisions: the excerpts that govern the question (at most four). Skip excerpts about other subjects, other laws the question doesn't raise, and other kinds of case. For each:
- law and section: the law's name and section number exactly as in <context>;
- quote: the words that decide the question, copied exactly (at most 60 words);
- elements: every condition, element, threshold, number, deadline and actor this provision sets that the answer must state, one short Hebrew phrase each;
- exceptions: every exception, proviso ("ואולם", "אלא אם", "בכפוף ל", "למעט") or special case in it that bears on the question.
If no excerpt governs the question, return an empty list.

<question>{question}</question>

<context>
{context}
</context>"""

EXTRACTED_NOTE = """<extracted> lists the provisions that govern this question and every element and exception they set, taken from <context> in an earlier step. Your answer must address each element and each exception that bears on the facts -- state it, or say why it does not apply -- and cite the provisions listed. If <extracted> is empty, answer as instructed above."""

COMPLETENESS_PROMPT = """Below are a question about Israeli law, the elements and exceptions that the governing provisions set, and an answer. List each element or exception that bears on the question and that the answer neither states nor explains away. Return an empty list if the answer covers them all. Judge coverage only, not style.

<question>{question}</question>

<elements>
{elements}
</elements>

<answer>
{answer}
</answer>"""

REVISE_PROMPT = """The answer below leaves out these points, which the provisions it cites set:
{missing}

The provisions, as quoted from the law:
{provisions}

Rewrite the answer so that it covers them, each in its right place, in the provision's own terms and citing it. Add nothing else: no other law, section or point. Keep everything else as it is: its content, structure, law names, section numbers, and any first word; a final "מסקנה:" line keeps its conclusion after the colon. No bold or other markup. Write only in Hebrew and return only the answer.

<question>{question}</question>

<answer>
{answer}
</answer>"""

# Retrieved text per question with whole or added sections: the Kaggle runs give Ollama 16,384 tokens
# for the prompt and the answer (up to 6,144), and twelve single chunks take about 5,000.
CONTEXT_BUDGET_TOKENS = 7000

# Letter answers are read by exact match: an added paragraph could only hurt.
NO_COMPLETENESS = {"mcq_bar"}

# --variant names for eval_run.py answer -> the legal.corpus settings they turn on.
VARIANTS = {
    "whole_sections": {"whole_sections": True},
    "toc": {"toc_navigation": True},
    "xref": {"cross_references": True},
    "reg_cap": {"regulation_cap": 3},
    "grouped": {"law_grouped_context": True},
    "extract": {"extract_then_answer": True},
    "completeness": {"extract_then_answer": True, "completeness_check": True},
    "doctrines": {"doctrine_cards_path": "legal_txt/doctrine_cards.jsonl"},
}


class Provision(BaseModel):
    law: str
    section: str
    quote: str
    elements: list[str] = []
    exceptions: list[str] = []


class Extraction(BaseModel):
    provisions: list[Provision] = []


class Completeness(BaseModel):
    missing: list[str] = []


# The value each switch takes when --variant leaves it out: the fix is off.
VARIANT_OFF = {"whole_sections": False, "toc_navigation": False, "cross_references": False, "regulation_cap": None,
               "law_grouped_context": False, "extract_then_answer": False, "completeness_check": False,
               "doctrine_cards_path": None}


def apply_variants(names: list[str]) -> dict:
    """Every fix is on by default; --variant turns on only the named ones ("baseline": none) and
    sets legal.corpus to match. Returns the settings changed ({} with no --variant)."""
    corpus_cfg = get_config().legal.corpus
    if not names:
        return {}
    changed: dict = dict(VARIANT_OFF)
    for name in names:
        if name == "baseline":
            continue
        if name not in VARIANTS:
            raise SystemExit(f"unknown --variant {name}; choose from baseline, {', '.join(VARIANTS)}")
        changed.update(VARIANTS[name])
    for key, value in changed.items():
        setattr(corpus_cfg, key, value)
    return changed


def render_extraction(extraction: Extraction) -> str:
    lines: list[str] = []
    for p in extraction.provisions:
        lines.append(f"- {p.law} סעיף {p.section}: \"{p.quote}\"")
        if p.elements:
            lines += ["  יסודות ותנאים:"] + [f"  - {e}" for e in p.elements]
        if p.exceptions:
            lines += ["  חריגים וסייגים:"] + [f"  - {e}" for e in p.exceptions]
    return "\n".join(lines)


_CITED_SPAN_RE = re.compile(r"סעי(?:ף|פים|פי)\s+([^.;:\n]{0,40})")
_SECTION_NUMBER_RE = re.compile(r"\d+[א-ת]{0,2}(?![א-ת])")
_LAW_YEAR_RE = re.compile(r"[,\s]+(?:ה?תש[א-ת]{0,2}[\"״'׳]|\d{4}|\[).*$")


def cited_sections(text: str) -> set[str]:
    """Section numbers the text cites ("סעיף 7(ב)", "סעיפים 12 ו-39" -> 7, 12, 39)."""
    return {n for span in _CITED_SPAN_RE.findall(text) for n in _SECTION_NUMBER_RE.findall(span)}


def cites(text: str, provision: Provision, sections: set[str]) -> bool:
    """Whether the answer relies on this provision: it cites the section number and names the law
    (its name without the year, or the first two words: "חוק החוזים")."""
    number = _SECTION_NUMBER_RE.match(provision.section.strip())
    if not number or number.group(0) not in sections:
        return False
    name = _LAW_YEAR_RE.sub("", provision.law).strip()
    return bool(name) and (name in text or " ".join(name.split()[:2]) in text)


def _norm(phrase: str) -> str:
    return re.sub(r"[\W_]+", " ", phrase).strip()


def listed(point: str, elements: list[str]) -> bool:
    """A point the checker names is one of the elements it was given, not a new one."""
    p = _norm(point)
    return bool(p) and any(p in _norm(e) or _norm(e) in p for e in elements if _norm(e))


_EMPTY_CONCLUSION_RE = re.compile(r"\n+\s*\**מסקנה:?\**\s*$")


def drop_empty_conclusion(text: str) -> str:
    """An answer ending in a "מסקנה:" heading with nothing after it loses the heading."""
    return _EMPTY_CONCLUSION_RE.sub("", text).rstrip()


async def extract_provisions(llm, qtext: str, context: str) -> Extraction | None:
    try:
        return await llm.complete_json(
            [ChatMessage("user", EXTRACT_PROMPT.format(question=qtext, context=context))],
            LLMCallSite("legal_eval_extract"), schema=Extraction,
            sampling=SamplingParams(temperature=0.0, max_tokens=2048), enable_thinking=False,
        )
    except Exception:  # noqa: BLE001 -- answer from the context alone
        return None


async def complete_answer(llm, qtext: str, extraction: Extraction, text: str) -> tuple[str, list[str]]:
    """(the answer, revised to cover what it missed; the points it missed) -- the answer unchanged
    when nothing is missing or a call fails.

    Only the elements of provisions the answer itself cites are checked: in the 30 Sept A/B the check
    ran on every extracted provision and pushed in points from laws the answer didn't rely on (the
    international sale law into a Contracts Remedies definition, electricity-supply rules into a
    tenancy case), and the revision then added wrong numbers. A point the checker names must be one
    of those elements."""
    sections = cited_sections(text)
    cited = [p for p in extraction.provisions if cites(text, p, sections)]
    elements = [e for p in cited for e in p.elements + p.exceptions]
    if not elements or not text:
        return text, []
    try:
        verdict = await llm.complete_json(
            [ChatMessage("user", COMPLETENESS_PROMPT.format(
                question=qtext, elements="\n".join(f"- {e}" for e in elements), answer=text))],
            LLMCallSite("legal_eval_completeness"), schema=Completeness,
            sampling=SamplingParams(temperature=0.0, max_tokens=512), enable_thinking=False,
        )
    except Exception:  # noqa: BLE001 -- keep the answer
        return text, []
    missing = [m for m in verdict.missing if m.strip() and listed(m, elements)][:5]
    if not missing:
        return text, []
    provisions = "\n".join(f"- {p.law} סעיף {p.section}: \"{p.quote}\"" for p in cited)
    revised = (await llm.complete_text(
        [ChatMessage("user", REVISE_PROMPT.format(missing="\n".join(f"- {m}" for m in missing),
                                                  provisions=provisions, question=qtext, answer=text))],
        LLMCallSite("legal_eval_completeness"), sampling=eval_sampling(ANSWER_MAX_TOKENS, False),
        enable_thinking=False,
    )).strip()
    revised = drop_empty_conclusion(revised)
    # A revision far shorter than the answer dropped content rather than adding it.
    return (revised, missing) if len(revised) >= 0.8 * len(text) else (text, missing)


class ScopeVerdict(BaseModel):
    scope: Literal["in_scope", "foreign_law", "not_in_legislation", "false_premise",
                   "nonexistent_law_or_section", "harmful_request"]
    note: str = ""


def place_conclusion(text: str) -> tuple[str, str | None]:
    """A rule_conclusion answer ending "מסקנה: כן/לא" -> (label first, explanation, no conclusion line),
    and the repair made. An opening label the model wrote anyway is dropped: the conclusion, written
    after the reasoning, wins. (text, None) when there is no conclusion line."""
    match = _CONCLUSION_RE.search(text)
    if not match:
        return text, None
    label, body = match.group(1), text[: match.start()].rstrip()
    opener, rest = split_yes_no_opener(body)
    if opener is not None:
        body = rest.lstrip()
    if not body:
        return text, None
    return f"{label}. {body}", ("conclusion_first" if opener in (None, label) else "opener_overruled")


def strip_template_tokens(text: str) -> str:
    return _TEMPLATE_TOKEN_RE.sub(" ", text).strip() if _TEMPLATE_TOKEN_RE.search(text) else text


async def check_scope(llm, qtext: str) -> ScopeVerdict:
    """in_scope on failure: a failed check must never turn an ordinary question into a refusal."""
    try:
        return await llm.complete_json(
            [ChatMessage("user", SCOPE_PROMPT.format(question=qtext))],
            LLMCallSite("legal_eval_scope"), schema=ScopeVerdict,
            sampling=SamplingParams(temperature=0.0, max_tokens=256), enable_thinking=False,
        )
    except Exception:  # noqa: BLE001
        return ScopeVerdict(scope="in_scope")


# Categories whose instructions don't ask for a yes/no answer: a bare כן/לא opener there (55 of 66
# of Gemma 12B's such answers) is dropped.
NO_YES_NO_OPENER = {"rule_recall", "rule_application", "interpretation", "citation_grounding",
                    "temporal_amendment", "issue_spotting"}
_YES_NO_OPENER_RE = re.compile(r"^\s*(?:\*\*)?\s*(כן|לא)\s*(?:\*\*)?\s*(?:[,.:;!\-–—]+|\n)\s*(?:\*\*)?\s*")
MAX_REWRITE_PASSES = 2

# Output budgets, reasoning included. The 26 Sept bulk500 run (16k context, never more than 6.1k
# of it used) stopped 8 of 500 answers at 3072 and 11 of 431 judge calls at 2048 while still
# thinking, 7 answers came back empty; the context window was never the limit.
ANSWER_MAX_TOKENS = 6144
ISSUE_SPOTTING_TOP_K = 16
JUDGE_MAX_TOKENS = 4096  # same as eval_cases.py


def eval_sampling(max_tokens: int, thinking: bool) -> SamplingParams:
    """Greedy without thinking (reproducible). With thinking, the recommended thinking-mode sampling
    (temperature 0.6, top_p 0.95, top_k 20) with a fixed seed: its model card warns that greedy
    decoding in thinking mode degrades answers and loops -- the 26 Sept trace has reasoning that
    repeats "Wait, no..." until the budget runs out (IL-283)."""
    if thinking:
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=0, max_tokens=max_tokens)
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_done(out: Path) -> dict[str, dict]:
    return {r["id"]: r for r in _load_jsonl(out)} if out.exists() else {}


def _append(out: Path, row: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question_text(q: dict) -> str:
    text = q["question"]
    if q.get("options"):
        text += "\n\n" + "\n".join(f"{letter}. {body}" for letter, body in q["options"].items())
    return text


def retrieve(query_text: str, categories: list[str], top_k: int) -> list[dict]:
    """Top `top_k` chunks (by distance, merged across `categories`) from the bulk corpus
    at config.legal.corpus.vectordb_dir. Each category is queried for `top_k` first, so a
    category with the single best hits isn't starved by one that returns weaker ones. Records
    corpus_retrieval.in_scope rules out (West Bank orders, drafts, repealed law) are skipped."""
    from docslides.legal_data.corpus_index import CorpusCollection

    legal_cfg = get_config().legal
    corpus_cfg = legal_cfg.corpus
    vector = embed_texts(
        legal_cfg.retrieval.embedding_model,  # same embedding model the corpus was built with -- see vectorize.py
        [normalize_for_embedding(query_text, corpus_cfg.fold_final_letters_for_embedding)],
        device=legal_cfg.retrieval.device,
    )[0]
    hits: list[dict] = []
    for category in categories:
        path = Path(corpus_cfg.vectordb_dir) / category
        if not path.exists():
            continue
        collection = CorpusCollection(path, f"{corpus_cfg.collection_prefix}_{category}")
        if collection.count() == 0:
            continue
        result = collection.query(vector, top_k * 2)
        kept = 0
        for distance, meta, document in zip(result["distances"][0], result["metadatas"][0], result["documents"][0]):
            if kept < top_k and in_scope(meta, []):
                hits.append({"category": category, "distance": distance, "meta": meta, "text": document})
                kept += 1
    hits.sort(key=lambda h: h["distance"])
    return hits[:top_k]


def render_context(hits: list[dict]) -> str:
    if not hits:
        return "(no matching context found)"
    blocks = []
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        section = meta.get("section_number") or meta.get("case_number") or ""
        status = meta.get("status", "")
        blocks.append(f"[{i}] {meta.get('title', '')} {section} ({status})\n{h['text']}")
    return "\n\n".join(blocks)


_FOREIGN_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+|[A-Za-z]{4,}")


def foreign_words(text: str) -> list[str]:
    return _FOREIGN_RE.findall(text)


def split_yes_no_opener(text: str) -> tuple[str | None, str]:
    """("כן" / "לא", the rest) for an answer opening with a bare label -- "לא.", "**כן**,", "כן\\n" --
    else (None, text). "לא ניתן ..." is a sentence, not a label, and stays whole."""
    match = _YES_NO_OPENER_RE.match(text)
    if not match or not text[match.end():].strip():
        return None, text
    return match.group(1), text[match.end():]


class LabelVerdict(BaseModel):
    answer: Literal["yes", "no", "unclear"]


async def check_label(llm, question: str, text: str) -> tuple[str, str | None]:
    """A rule_conclusion answer that opens with a label: one the model wrote is kept, a missing one
    is added from what the explanation (read on its own) supports. Returns (answer, repair or None).

    Swapping a label the model wrote was dropped: over the two 29 Sept Gemma 27B runs the checker
    fixed 2 labels (IL-219, IL-164) and broke 3 (IL-212, IL-409 twice) -- it misreads negated
    questions ("can he file without approval?" / "cannot file without approval" -> "yes")."""
    label, explanation = split_yes_no_opener(text)
    first_words = re.findall(r"[א-ת]+", text[:40])  # how score.py reads the label
    if label is not None or first_words[:1] in (["כן"], ["לא"]):
        return text, None
    try:
        verdict = await llm.complete_json(
            [ChatMessage("user", LABEL_CHECK_PROMPT.format(question=question, explanation=explanation.strip()))],
            LLMCallSite("legal_eval_label_check"), schema=LabelVerdict,
            sampling=SamplingParams(temperature=0.0, max_tokens=256), enable_thinking=False,
        )
    except Exception:  # noqa: BLE001 -- keep the answer as written
        return text, None
    if verdict.answer == "unclear":
        return text, None
    wanted = "כן" if verdict.answer == "yes" else "לא"
    return f"{wanted}. {explanation.lstrip()}", "label_added"


async def plan_issues(llm, q: dict, max_issues: int) -> list[EvalIssue]:
    """The answering model's own list of issues and governing laws (thinking off: a short,
    structured call). An empty list on failure, which leaves retrieval on the question alone."""
    try:
        plan = await llm.complete_json(
            [ChatMessage("user", PLAN_PROMPT.format(max_issues=max_issues, question=_question_text(q)))],
            LLMCallSite("legal_eval_plan"), schema=EvalRetrievalPlan,
            sampling=SamplingParams(temperature=0.0, max_tokens=PLAN_MAX_TOKENS), enable_thinking=False,
        )
    except Exception as exc:  # noqa: BLE001 -- retrieval still works without a plan
        _log(f"{q['id']}: plan failed ({type(exc).__name__}: {exc}); retrieving on the question alone")
        return []
    return [i for i in plan.issues if i.law.strip()][:max_issues]


async def answer_one(llm, q: dict, categories: list[str], top_k: int, thinking: bool = True,
                     max_tokens: int = ANSWER_MAX_TOKENS) -> dict:
    qtext = _question_text(q)
    issues: list[EvalIssue] = []
    repairs: list[str] = []
    with trace.collect(job_id=q["id"]):
        spotting = q.get("category") == "issue_spotting"
        conclusion = q.get("category") == "rule_conclusion"
        temporal = q.get("category") == "temporal_amendment" or bool(TEMPORAL_RE.search(q["question"]))
        scope = await check_scope(llm, qtext)
        issues = await plan_issues(llm, q, max_issues=6 if spotting else 3)
        search_issues = list(issues)
        if temporal and issues:
            search_issues.append(EvalIssue(issue=TEMPORAL_ISSUE, law=issues[0].law))
        hits = retrieve_planned(qtext, search_issues, categories, ISSUE_SPOTTING_TOP_K if spotting else top_k,
                                per_issue_slot=spotting)
        corpus_cfg = get_config().legal.corpus
        extra: list[dict] = []
        if corpus_cfg.toc_navigation and scope.scope == "in_scope":
            extra += await corpus_navigation.navigate_toc(llm, qtext, hits)
        if corpus_cfg.cross_references:
            extra += corpus_navigation.cross_reference_hits(hits + extra)
        if corpus_cfg.whole_sections or extra:
            # Whole and added sections are longer: keep the context inside CONTEXT_BUDGET_TOKENS, the
            # sections the model picked or that the top hits refer to first, then the rest best-first.
            extra = corpus_navigation.trim_to_budget(extra, CONTEXT_BUDGET_TOKENS // 2)
            spent = sum(count_tokens(h["text"]) for h in extra)
            hits = corpus_navigation.trim_to_budget(hits, CONTEXT_BUDGET_TOKENS - spent) + extra
        if corpus_cfg.law_grouped_context:
            hits = corpus_navigation.group_by_law(hits)
        context = render_context(hits)
        user = f"<instructions>{q['instructions']}</instructions>\n\n<question>{qtext}</question>\n\n" \
               f"<context>\n{context}\n</context>"
        cards: list[dict] = []
        if corpus_cfg.doctrine_cards_path:
            plan_text = " ".join(f"{i.law} {i.issue}" for i in issues)
            cards = corpus_navigation.match_doctrine_cards(
                f"{qtext} {plan_text}", corpus_navigation.load_doctrine_cards(corpus_cfg.doctrine_cards_path),
                corpus_cfg.doctrine_cards_max)
            if cards:
                user += "\n\n" + corpus_navigation.render_doctrine_cards(cards)
        case_hits: list[dict] = []
        if corpus_cfg.caselaw_dir and scope.scope == "in_scope" and (
                corpus_cfg.caselaw_question_categories is None
                or q.get("category") in corpus_cfg.caselaw_question_categories):
            case_hits = caselaw.search_caselaw(qtext, [i.issue for i in issues])
            if case_hits:
                user += "\n\n" + caselaw.render_caselaw(case_hits)
        extraction = None
        if corpus_cfg.extract_then_answer and scope.scope == "in_scope":
            extraction = await extract_provisions(llm, qtext, context)
            if extraction is not None and extraction.provisions:
                user += f"\n\n<extracted>\n{render_extraction(extraction)}\n</extracted>"
        notes = [ISSUE_SPOTTING_NOTE] if spotting else []
        if extraction is not None and extraction.provisions:
            notes.append(EXTRACTED_NOTE)
        if scope.scope != "in_scope":
            notes.append(SCOPE_NOTES[scope.scope] + (f" (First reading: {scope.note})" if scope.note else ""))
        if temporal:
            notes.append(TEMPORAL_NOTE)
        if conclusion:
            notes.append(CONCLUSION_NOTE)
        if notes:
            user += "\n\n" + "\n\n".join(notes)
        messages = [ChatMessage("system", ANSWER_SYSTEM), ChatMessage("user", user)]

        text = (await llm.complete_text(
            messages, LLMCallSite("legal_eval_baseline"), sampling=eval_sampling(max_tokens, thinking),
            enable_thinking=thinking,  # the point of this run: capture how the model reasons, for fine-tuning
        )).strip()
        sampling = eval_sampling(max_tokens, False)
        if not text and thinking:
            # The reasoning used the whole budget and no answer was written: answer without it.
            repairs.append("empty_answer_retry")
            text = (await llm.complete_text(messages, LLMCallSite("legal_eval_baseline"), sampling=sampling,
                                             enable_thinking=False)).strip()
        missing: list[str] = []
        if corpus_cfg.completeness_check and extraction is not None and q.get("category") not in NO_COMPLETENESS:
            revised, missing = await complete_answer(llm, qtext, extraction, text)
            if revised != text:
                repairs.append("completeness_revision")
                text = revised
        # Up to two passes: the 29 Sept review found one rewrite leaving script in 26 answers.
        for _ in range(MAX_REWRITE_PASSES):
            stray = foreign_words(text)
            if not stray:
                break
            repairs.append("hebrew_rewrite")
            rewritten = (await llm.complete_text(
                [ChatMessage("user", REWRITE_PROMPT.format(words=", ".join(dict.fromkeys(stray)), answer=text))],
                LLMCallSite("legal_eval_rewrite"), sampling=sampling, enable_thinking=False,
            )).strip()
            if not rewritten or len(foreign_words(rewritten)) >= len(stray):
                break
            text = rewritten
        if q.get("category") in NO_YES_NO_OPENER:
            label, rest = split_yes_no_opener(text)
            if label:
                repairs.append("stray_yes_no_opener")
                text = rest.lstrip()
        elif conclusion and text:
            text, repair = place_conclusion(text)
            if repair is None:  # no conclusion line: add a missing label as before
                text, repair = await check_label(llm, qtext, text)
            if repair:
                repairs.append(repair)
        trimmed = drop_empty_conclusion(text)
        if trimmed != text:
            repairs.append("empty_conclusion")
            text = trimmed
        cleaned = strip_template_tokens(text)
        if cleaned != text:
            repairs.append("template_token")
            text = cleaned
    return {
        "id": q["id"], "answer": text,
        "retrieved": [{"category": h["category"], "distance": round(h["distance"], 4),
                        "score": round(h["score"], 4) if "score" in h else None,
                        "title": h["meta"].get("title"), "section": h["meta"].get("section_number"),
                        "via": "".join(sorted({s[0] for s in h.get("sources", ())})),
                        "chunk_ids": h.get("chunk_ids") or [h.get("id")]} for h in hits],
        "plan": [i.model_dump() for i in issues],
        "scope": scope.model_dump(),
        "extraction": extraction.model_dump() if extraction is not None else None,
        "missing": missing,
        "doctrine_cards": [c["id"] for c in cards],
        "caselaw": caselaw.caselaw_record(case_hits),
        "repairs": repairs,
    }


async def cmd_answer(a) -> None:
    questions = _load_jsonl(Path(a.questions))
    if a.limit:
        questions = questions[: a.limit]
    out = Path(a.out)
    done = _load_done(out)
    categories = a.categories.split(",")
    variants = [v for v in (a.variant or "").split(",") if v]
    changed = apply_variants(variants)
    if changed:
        _log(f"variant {'+'.join(variants)}: legal.corpus {changed}")
    llm = get_legal_orchestrator_client()
    started = time.monotonic()
    device = warm_up_retrieval()
    _log(f"retrieval ready on {device} ({round(time.monotonic() - started)}s)")
    _log(f"{len(questions)} questions, {len(done)} already answered")
    for q in questions:
        if q["id"] in done:
            continue
        started = time.monotonic()
        try:
            row = await answer_one(llm, q, categories, a.top_k, thinking=a.thinking, max_tokens=a.max_tokens)
        except Exception as exc:  # noqa: BLE001 -- record and move on; --out is resumable
            row = {"id": q["id"], "answer": "", "error": f"{type(exc).__name__}: {exc}"}
        if variants:
            row["variant"] = variants
        _append(out, row)
        _log(f"{q['id']} ({q['category']}): {round(time.monotonic() - started)}s"
             f"{' -- ' + row['error'] if row.get('error') else ''}")


# score.py's prompt (unmodified, from the eval set) says "Grade ONLY against the reference
# material". gemma3:27b read that as "anything not in the reference is wrong" and scored correct
# answers 0 for accurate extra detail (27 Sept run, ~23 of 58 zeros). This says what it means.
JUDGE_SYSTEM = """You grade answers against a reference answer. How to read the grading instructions:
- "Grade only against the reference" means: check the reference's key points against the answer. \
It does not mean that everything outside the reference is wrong.
- Accurate extra detail, additional correct citations, or naming the law and section a rule comes \
from are NOT errors and never lower correctness. Lower it only for a key point that is missing or \
wrong, or for a statement that contradicts the reference.
- A citation to a different but real provision that states the same rule is not a hallucination. \
hallucination is for an invented law, section, number or date, or one that plainly does not exist.
- Judge meaning, not wording. The answer is usually in Hebrew."""


async def judge_one(llm, item: dict, thinking: bool = True, max_tokens: int = JUDGE_MAX_TOKENS) -> dict:
    with trace.collect(job_id=item["id"]):
        try:
            verdict = await llm.complete_json(
                [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", item["prompt"])],
                LLMCallSite("legal_eval_judge"),
                schema=BulkEvalJudgement,
                sampling=eval_sampling(max_tokens, thinking),
                enable_thinking=thinking,  # the judge's own reasoning matters just as much for calibration
            )
        except Exception as exc:  # noqa: BLE001 -- record and move on; --out is resumable
            return {"id": item["id"], "correctness": 0, "grounding": "n/a", "hallucination": False,
                    "key_points_hit": [], "needs_review": True, "note": f"judge failed: {exc}"}
    return {"id": item["id"], **verdict.model_dump()}


async def cmd_judge(a) -> None:
    requests = _load_jsonl(Path(a.requests))
    out = Path(a.out)
    done = _load_done(out)
    llm = get_judge_client()
    _log(f"{len(requests)} judge requests, {len(done)} already judged")
    for item in requests:
        if item["id"] in done:
            continue
        started = time.monotonic()
        row = await judge_one(llm, item, thinking=a.thinking, max_tokens=a.max_tokens)
        _append(out, row)
        _log(f"{item['id']}: correctness={row['correctness']} ({round(time.monotonic() - started)}s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("answer")
    pa.add_argument("--questions", default="legal_txt/Evals/israeli_legal_eval/questions.jsonl")
    pa.add_argument("--mode", choices=["rag"], default="rag", help="accepted for older commands; rag is the only mode")
    pa.add_argument("--categories", default="laws,procedural_rules")
    pa.add_argument("--top-k", type=int, default=12)
    pa.add_argument("--out", required=True)
    pa.add_argument("--limit", type=int, help="answer only the first N questions (smoke test)")
    pa.add_argument("--max-tokens", type=int, default=ANSWER_MAX_TOKENS,
                    help="output budget per answer, reasoning included (a model that always reasons needs more)")
    pa.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="reasoning on the answer call (--no-thinking for models without a thinking mode)")
    pa.add_argument("--variant", default="",
                    help="comma-separated legal.corpus switches to measure on the dev split: "
                         + ", ".join(VARIANTS))

    pj = sub.add_parser("judge")
    pj.add_argument("--requests", required=True, help="judge_requests.jsonl from score.py prepare")
    pj.add_argument("--out", required=True)
    pj.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    pj.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS,
                    help="output budget per judgement, reasoning included")

    a = p.parse_args()
    from docslides.llm.client import aclose_all_clients

    async def run() -> None:
        try:
            await {"answer": cmd_answer, "judge": cmd_judge}[a.cmd](a)
        finally:
            await aclose_all_clients()

    asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
