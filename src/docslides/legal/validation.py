"""Deterministic checks in the Legal pipeline -- no model judgment involved:

  * `validate_memorandum`: the gate between Pass A and Pass B (spec section 5).
    A memorandum that fails goes back to Qwen for revision, never forward to
    drafting.
  * `check_draft_citations`: structural checks on every [[CITE]] token in a
    draft (known claim ID, retrieved source, unchanged source_type, a
    claim/source pairing that Pass A actually recorded). The entailment check
    in legal/pipeline.py only runs on tokens that pass these.

`evidence` maps source_id -> ChunkMetadata for everything retrieved this turn.
A source_id not in it is by definition invented.
"""

from __future__ import annotations

import re

from docslides.legal.citations import parse_citations, sentence_before
from docslides.legal.models import ChunkMetadata
from docslides.llm.schemas import (
    ContraryAuthority,
    GoverningLawClaim,
    GroundedMemorandum,
    ResearchMemorandum,
    SupportingAuthority,
)

_MIN_EXPLANATION_WORDS = 3


def _mentions(texts: list[str], claim_id: str) -> bool:
    """True if some entry names `claim_id` AND says something about it -- a
    bare "C1" is not an explanation (small models do exactly that to get
    an unsupported claim past the gate)."""
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(claim_id)}(?![0-9])")
    return any(
        pattern.search(t) and len(pattern.sub(" ", t).split()) >= _MIN_EXPLANATION_WORDS for t in texts
    )


# Phrases only the gate's own error messages contain (see validate_memorandum):
# a small model asked to fix them sometimes pastes them into the memo instead.
_ECHOED_ERROR_RE = re.compile(
    r"has no supporting_authority|lists no source_ids|has no contrary_authority|contrary_search_performed must be true|"
    r"is not listed in unresolved_questions|never invent a source_id|must be carried through unchanged"
)


# An "authority conflict" that says there is none ("אין סתירה בין ...") -- small models fill the field
# anyway, and every entry escalates the turn as an unresolved conflict.
_NO_CONFLICT_RE = re.compile(
    r"^\s*(?:אין\s+(?:כל\s+)?סתירה|לא\s+(?:קיימת|נמצאה|קיימות|נמצאו)\s+סתיר|no\s+(?:real\s+|actual\s+)?(?:conflict|contradiction))",
    re.IGNORECASE,
)


def clean_memorandum(memo: ResearchMemorandum) -> ResearchMemorandum:
    """Drops free-text list entries with no letters or digits (e.g. "],"
    leaked from a malformed generation) so they can't surface as escalation
    reasons or count as explanations, and "conflicts" that deny any conflict."""
    def keep(values: list[str]) -> list[str]:
        return [v.strip() for v in values if re.search(r"\w", v) and not _ECHOED_ERROR_RE.search(v)]

    return memo.model_copy(
        update={
            "unresolved_questions": keep(memo.unresolved_questions),
            "temporal_issues": keep(memo.temporal_issues),
            "authority_conflicts": [c for c in keep(memo.authority_conflicts) if not _NO_CONFLICT_RE.search(c)],
        }
    )


AUTO_NOTE_MARK = "(auto-recorded)"
_MIN_QUOTED_WEIGHT = 4  # shared terms (numbers count double) for a claim to "quote" a source


def _authority_fields(claim_id: str, meta: ChunkMetadata) -> dict:
    return {
        "claim_id": claim_id,
        "source_id": meta.source_id,
        "law": meta.law_name,
        "section": meta.display_section,
        "effective": f"{meta.effective_date_start} to {meta.effective_date_end or 'current'}",
        "source_type": meta.source_type,
    }


def _quoted_source(claim_text: str, evidence_texts: dict[str, str]) -> str | None:
    """The one evidence item a claim's wording and numbers clearly come from,
    or None when no item stands out. Terms every item shares (the law's name
    in each breadcrumb) don't count."""
    from docslides.legal.keyword import terms

    claim = set(terms(claim_text))
    per_source = {source_id: set(terms(text)) for source_id, text in evidence_texts.items()}
    if not per_source:
        return None
    everywhere = set.intersection(*per_source.values()) if len(per_source) > 1 else set()

    def weight(source_id: str) -> int:
        return sum(2 if term[0].isdigit() else 1 for term in (claim & per_source[source_id]) - everywhere)

    ranked = sorted(per_source, key=weight, reverse=True)
    if weight(ranked[0]) < _MIN_QUOTED_WEIGHT:
        return None
    if len(ranked) > 1 and weight(ranked[1]) == weight(ranked[0]):
        return None  # no clear source
    return ranked[0]


def ground_memorandum(
    grounded: GroundedMemorandum, evidence: dict[str, ChunkMetadata], evidence_texts: dict[str, str]
) -> tuple[ResearchMemorandum, list[str]]:
    """The model's Pass A output as a ResearchMemorandum: each claim's
    source_ids become supporting_authority entries filled from the source's
    metadata, so the model never has to copy law, section or dates. A claim
    that names no retrieved source gets the evidence item it quotes, when one
    clearly stands out, marked attached_by="pipeline" -- Pass B's entailment
    check still verifies that pairing before anything is cited. Returns the
    memo and a note per attached source."""
    supporting: list[SupportingAuthority] = []
    attached: list[str] = []
    for claim in grounded.governing_law:
        source_ids = [s for s in dict.fromkeys(claim.source_ids) if s in evidence]
        by = "model"
        if not source_ids:
            quoted = _quoted_source(claim.text, evidence_texts)
            if quoted:
                source_ids, by = [quoted], "pipeline"
                attached.append(f"{claim.claim_id} -> {quoted}")
        supporting += [
            SupportingAuthority(**_authority_fields(claim.claim_id, evidence[s]), attached_by=by) for s in source_ids
        ]
    contrary = [
        ContraryAuthority(**_authority_fields(c.claim_id, evidence[c.source_id]), note=c.note)
        for c in grounded.contrary_authority
        if c.source_id in evidence
    ]
    memo = ResearchMemorandum(
        issues=grounded.issues,
        facts_relied_on=grounded.facts_relied_on,
        governing_law=[
            GoverningLawClaim(claim_id=c.claim_id, text=c.text, issue_id=c.issue_id, fact_ids=c.fact_ids)
            for c in grounded.governing_law
        ],
        supporting_authority=supporting,
        contrary_authority=contrary,
        contrary_search_performed=grounded.contrary_search_performed,
        unresolved_questions=grounded.unresolved_questions,
        temporal_issues=grounded.temporal_issues,
        authority_conflicts=grounded.authority_conflicts,
    )
    return memo, attached


def record_contrary_search_notes(memo: ResearchMemorandum) -> tuple[ResearchMemorandum, list[str]]:
    """The spec requires every claim without contrary authority to say so
    explicitly in unresolved_questions. When the model affirms it searched
    (contrary_search_performed) but left a supported claim's note out -- the
    single most common reason a small model's memo fails the gate -- record
    the note for it, marked auto-recorded, instead of burning a revision.
    Unsupported claims and a missing search are never papered over."""
    if not memo.contrary_search_performed:
        return memo, []
    supported = {a.claim_id for a in memo.supporting_authority}
    with_contrary = {a.claim_id for a in memo.contrary_authority}
    added = [
        f"{claim.claim_id}: the search of the retrieved evidence for limiting or contrary authority "
        f"found none {AUTO_NOTE_MARK}"
        for claim in memo.governing_law
        if claim.claim_id in supported
        and claim.claim_id not in with_contrary
        and not _mentions(memo.unresolved_questions, claim.claim_id)
    ]
    if not added:
        return memo, []
    return memo.model_copy(update={"unresolved_questions": [*memo.unresolved_questions, *added]}), added


def _duplicates(values: list[str]) -> list[str]:
    seen, dupes = set(), []
    for value in values:
        if value in seen and value not in dupes:
            dupes.append(value)
        seen.add(value)
    return dupes


def validate_memorandum(memo: ResearchMemorandum, evidence: dict[str, ChunkMetadata]) -> list[str]:
    errors: list[str] = []
    if not memo.contrary_search_performed:
        errors.append("contrary_search_performed must be true: search the evidence for limiting/contrary authority")
    if not memo.governing_law and not memo.unresolved_questions:
        errors.append("governing_law is empty and unresolved_questions doesn't explain why")

    issue_ids = [i.issue_id for i in memo.issues]
    fact_ids = {f.fact_id for f in memo.facts_relied_on}
    claim_ids = [c.claim_id for c in memo.governing_law]
    for dupe in _duplicates(issue_ids):
        errors.append(f"duplicate issue_id {dupe}")
    for dupe in _duplicates(claim_ids):
        errors.append(f"duplicate claim_id {dupe}")

    supported = {a.claim_id for a in memo.supporting_authority}
    with_contrary = {a.claim_id for a in memo.contrary_authority}
    for claim in memo.governing_law:
        if claim.issue_id not in issue_ids:
            errors.append(f"{claim.claim_id} references unknown issue_id {claim.issue_id}")
        for fact_id in claim.fact_ids:
            if fact_id not in fact_ids:
                errors.append(f"{claim.claim_id} references unknown fact_id {fact_id}")
        if claim.claim_id not in supported and not _mentions(memo.unresolved_questions, claim.claim_id):
            errors.append(
                f"{claim.claim_id} has no supporting_authority and is not listed in unresolved_questions"
            )
        if claim.claim_id not in with_contrary and not _mentions(memo.unresolved_questions, claim.claim_id):
            errors.append(
                f"{claim.claim_id} has no contrary_authority: if a genuine search found none, say so "
                f"explicitly in unresolved_questions (mentioning {claim.claim_id})"
            )

    for kind, authorities in (("supporting", memo.supporting_authority), ("contrary", memo.contrary_authority)):
        for authority in authorities:
            label = f"{kind}_authority {authority.claim_id} -> {authority.source_id}"
            if authority.claim_id not in claim_ids:
                errors.append(f"{label}: claim_id is not in governing_law")
            meta = evidence.get(authority.source_id)
            if meta is None:
                errors.append(f"{label}: source_id is not in the retrieved evidence (never invent a source_id)")
                continue
            if authority.source_type != meta.source_type:
                errors.append(
                    f"{label}: source_type '{authority.source_type}' must be carried through unchanged "
                    f"as '{meta.source_type}'"
                )
            if meta.section_number not in authority.section:
                errors.append(f"{label}: section '{authority.section}' doesn't match the source's section {meta.section_number}")
    return errors


_MIN_SENTENCE_WORDS = 2


def states_nothing(sentence: str) -> bool:
    """A citation's sentence that can't carry a claim: a bare connective ("ובנוסף,") or a
    lead-in ending in a colon ("שונו מספרים בהתאם לחוקים הבאים:"). A small model handed a
    claim it finds awkward to write (quoted numbers) cites these instead, and the
    entailment check, reading the claim, passed them."""
    sentence = sentence.strip()
    return len(re.findall(r"\w+", sentence)) < _MIN_SENTENCE_WORDS or sentence.endswith(":")


def check_draft_citations(
    draft: str, memo: ResearchMemorandum, evidence: dict[str, ChunkMetadata]
) -> dict[int, list[str]]:
    """Problems per citation, keyed by the citation's index in the draft."""
    claim_ids = {c.claim_id for c in memo.governing_law}
    supporting = {(a.claim_id, a.source_id) for a in memo.supporting_authority}
    contrary = {(a.claim_id, a.source_id) for a in memo.contrary_authority}

    problems: dict[int, list[str]] = {}
    sentence = ""
    for index, citation in enumerate(parse_citations(draft)):
        issues: list[str] = []
        if citation.claim_id not in claim_ids:
            issues.append(f"claim_id '{citation.claim_id}' was not established in Pass A")
        meta = evidence.get(citation.source_id)
        if meta is None:
            issues.append(f"source_id '{citation.source_id}' is not in the retrieved evidence")
        elif citation.source_type != meta.source_type:
            issues.append(f"source_type '{citation.source_type}' differs from the source's '{meta.source_type}'")
        if citation.relation == "supports":
            if (citation.claim_id, citation.source_id) not in supporting:
                issues.append("Pass A doesn't list this source as supporting this claim")
        elif citation.relation == "contrary":
            if (citation.claim_id, citation.source_id) not in contrary:
                issues.append("Pass A doesn't list this source as contrary authority for this claim")
        else:
            issues.append(f"relation must be 'supports' or 'contrary', got '{citation.relation}'")
        # A token right after another token shares that token's sentence.
        sentence = sentence_before(draft, citation.start) or sentence
        if states_nothing(sentence):
            issues.append(
                f"the sentence this citation is attached to ('{sentence}') doesn't state the claim: write the "
                "claim's content -- its rule, number or date -- in the sentence itself"
            )
        if issues:
            problems[index] = issues
    return problems
