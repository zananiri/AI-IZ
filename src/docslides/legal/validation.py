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

from docslides.legal.citations import parse_citations
from docslides.legal.models import ChunkMetadata
from docslides.llm.schemas import ResearchMemorandum

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
    r"has no supporting_authority|has no contrary_authority|contrary_search_performed must be true|"
    r"is not listed in unresolved_questions|never invent a source_id|must be carried through unchanged"
)


def clean_memorandum(memo: ResearchMemorandum) -> ResearchMemorandum:
    """Drops free-text list entries with no letters or digits (e.g. "],"
    leaked from a malformed generation) so they can't surface as escalation
    reasons or count as explanations."""
    def keep(values: list[str]) -> list[str]:
        return [v.strip() for v in values if re.search(r"\w", v) and not _ECHOED_ERROR_RE.search(v)]

    return memo.model_copy(
        update={
            "unresolved_questions": keep(memo.unresolved_questions),
            "temporal_issues": keep(memo.temporal_issues),
            "authority_conflicts": keep(memo.authority_conflicts),
        }
    )


AUTO_NOTE_MARK = "(auto-recorded)"


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


def check_draft_citations(
    draft: str, memo: ResearchMemorandum, evidence: dict[str, ChunkMetadata]
) -> dict[int, list[str]]:
    """Problems per citation, keyed by the citation's index in the draft."""
    claim_ids = {c.claim_id for c in memo.governing_law}
    supporting = {(a.claim_id, a.source_id) for a in memo.supporting_authority}
    contrary = {(a.claim_id, a.source_id) for a in memo.contrary_authority}

    problems: dict[int, list[str]] = {}
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
        if issues:
            problems[index] = issues
    return problems
