# An enhanced Israeli legal eval set: recommendations

What to change in `legal_txt/Evals/israeli_legal_eval` (500 questions, 10 cases), based on the
29 Sept 2026 results (`evals/gemma27b_full_judged/`) and published practice for legal RAG evaluation.

## What the current set does well

- LegalBench's reasoning types (issue spotting, rule recall/application/conclusion, interpretation),
  plus multiple choice, abstention, citation grounding and temporal questions.
- Gold key points and law + section citations per item; a confidence flag on the gold.
- Case files with a points rubric, deliberate traps and must-not lists.

## Where it falls short

| Gap | Effect in the 29 Sept runs | Practice it comes from |
|---|---|---|
| Gold names a law and section, not the chunk that proves it | Retrieval is only checked as "governing section retrieved" (84.5%); no recall@k or context precision, so retrieval and answer errors can't be separated | LegalBench-RAG: gold at the text-span level, precision@k / recall@k |
| Answers are graded as a whole | 67 of 336 graded answers state a correct rule on a wrong or neighbouring source, and that only shows up in a free-text note | Stanford legal-RAG study: a "hallucination" includes a true statement attributed to a source that doesn't support it; RAGAS faithfulness |
| The weakest categories are the smallest | Temporal has 16–20 items: one answer moves the category 6 points | Report per-category confidence intervals; ≥ 40 items per category |
| No comparison, multi-hop, aggregation or computation questions | The model's real failures (cross-referenced definitions, deadline arithmetic) are tested only inside the 10 cases | CRAG question types: comparison, multi-hop, set, post-processing, false premise |
| One wording per question | Can't tell whether a fix is real or fits one phrasing; negated questions ("can he … without …?") broke the label checker | Robustness via paraphrase sets |
| The same 500 items used for tuning and reporting | Prompts were fixed item by item after each review, so later scores overstate | Held-out test split |
| 168 medium-confidence gold items, none reviewed by a lawyer | 6 flagged items look like gold errors (e.g. insurance payout is s.27, not s.28) | The set's own README: lawyer review of 30–50 items, verify_gold.py against the index |
| Only the eval pipeline is tested | The app's Legal tab (memo → draft → citation verification) was never run on the set | Test the system users actually use |

## Recommended composition (600 items: 200 dev, 400 test)

| Type | Items | New or changed |
|---|--:|---|
| Rule recall | 60 | Fewer (the model already scores 89%) |
| Rule conclusion, yes/no | 70 | Add 20 negated or double-negative questions |
| Bar-exam multiple choice | 60 | Keep |
| Rule application (IRAC) | 70 | Key points split into issue / rule / application / conclusion |
| Issue spotting | 50 | Score recall of issues and precision (no invented issues) separately |
| Interpretation | 40 | Keep |
| Citation grounding | 30 | Keep |
| **Comparison** | 40 | New: two laws or two versions ("how does the deposit cap differ from…") |
| **Multi-hop / cross-reference** | 40 | New: a term defined in one law and applied in another; "subject to section X" chains |
| **Computation** | 30 | New: deadlines, caps, pro-rata amounts (notice days, deposit cap, limitation with minority) |
| **Temporal / versioned** | 50 | Expanded: every item carries an `as_of` date and the version in force then |
| Abstention | 50 | Expanded: foreign law, non-existent sections, case law, statistics, false premise, harmful, and **real but un-indexed** regulations |
| Paraphrase sets | +100 variants | 25 items × 4 wordings (formal, colloquial, with typos, asked in English or Arabic, answered in Hebrew) — report per-item consistency |

Keep the 10 cases and add 10 more, so each area of law has at least one. Give every case at
least two traps and one must-not.

## Fields to add to each gold item

```json
{
  "gold_chunks": ["contracts@1973-06-01:14#p1"],
  "as_of": "2026-09-01",
  "answer_type": "rule | yes_no | letter | number | date | list | refusal",
  "hops": 1,
  "negated": false,
  "abstain_reason": null,
  "paraphrase_group": null,
  "split": "dev | test",
  "reviewed_by_lawyer": false
}
```

- `gold_chunks` makes retrieval scoring deterministic: recall@k and precision@k per item, with no judge.
- `answer_type` "number" and "date" items are scored by exact match (after normalization), like the letters.
- `abstain_reason` separates "should refuse" from "should correct the premise".

## Scoring to add

1. **Retrieval**: recall@12 and MRR against `gold_chunks`, reported next to answer scores so a
   miss is attributed to retrieval or to the model.
2. **Claim-level grounding**: split each answer into sentences with a citation; a judge checks that
   the cited section supports the sentence (the Legal tab's entailment check already does this).
   Report "correct but misgrounded" separately from "wrong".
3. **Label consistency**: for yes/no items, check the label against the explanation as well as the gold.
4. **Confidence intervals**: bootstrap 95% intervals per category; don't call a change an
   improvement when the intervals overlap.
5. **Judge calibration**: a lawyer grades 50 test items; report judge–lawyer agreement, and grade
   the test split twice (or with two judge models) and flag disagreements.
6. **Both pipelines**: run the test split through `eval_run.py` and through the Legal tab's
   `run_legal_turn` (`scripts/eval_legal.py --phase after`), reporting escalation rate and the
   share of citations that pass verification.

## Order of work

1. Split the current 500 into dev (200) and test (300) now; stop tuning on test.
2. Lawyer review of the 168 medium-confidence items plus the 6 flagged in the 29 Sept review.
3. Add `gold_chunks` to the test split (verify_gold.py already finds most of them).
4. Write the new comparison, multi-hop, computation and expanded temporal/abstention items.
5. Add the paraphrase sets last; they multiply the items to maintain.

## Sources

- LegalBench (Guha et al., 2023): https://arxiv.org/abs/2308.11462
- LegalBench-RAG (Pipitone & Alami, 2024): https://arxiv.org/abs/2408.10343
- Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools (Magesh et al., Stanford, 2024/2025): https://arxiv.org/abs/2405.20362
- CRAG – Comprehensive RAG Benchmark (Meta, 2024): https://arxiv.org/abs/2406.04744
- Ragas metrics (faithfulness, context precision/recall): https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

## Status (29 Sept 2026)

Built: `legal_txt/Evals/israeli_legal_eval_v2` (see its README). 580 scored items (388 test, 192 dev) instead of
600: multi-hop has 30 new items rather than 40, and temporal has 20 new rather than 30, because only items whose
rule the author was sure of were written. Plus 75 paraphrases (robustness) and 115 reserve v1 items.
Next: run `notebooks/kaggle_legal_eval_v2_verify.ipynb`, then the lawyer review, then the first test run.
