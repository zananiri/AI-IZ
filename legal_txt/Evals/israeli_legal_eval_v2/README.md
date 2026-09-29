# Israeli legal eval v2

770 items built from the v1 set (`legal_txt/Evals/{questions,gold}.jsonl`) and 195 new items, following
`evals/ENHANCED_QUESTION_SET.md`. Rebuild with `python legal_txt/Evals/israeli_legal_eval_v2/src/build_v2.py`
(deterministic); edit items in `src/items_*.py`, never in the generated files.

| File | Send to the model? | Contents |
|---|---|---|
| `questions.jsonl` | yes | id, category, area, format, instructions, question (+ options, as_of, question_language) |
| `gold.jsonl` | no, judge only | the v1 gold fields plus split, source, gold_chunks, as_of, answer_type, hops, negated, abstain_reason, paraphrase_group, verified, reviewed_by_lawyer (and answer_value for computation) |
| `manifest.json` | – | counts by split, category and source |
| `src/` | – | the new items and the build script |

## Splits

| Split | Items | Use |
|---|--:|---|
| test | 388 | the only split to report; don't tune prompts on it |
| dev | 192 | tuning prompts and retrieval |
| robustness | 75 | 3 rewordings (colloquial, typos without punctuation, English) of 25 test items; same gold |
| reserve | 115 | v1 items dropped to rebalance (50 rule recall, 40 yes/no, 20 multiple choice, 5 citation) |

Test and dev together (580) by category: rule application 70, yes/no 70 (20 of them negated), multiple choice 60,
rule recall 60, issue spotting 50, abstention 50, temporal / versioned 40, comparison 40, computation 40,
interpretation 40, citation grounding 30, multi-hop 30.

## Before the first scored run

The new items were written from knowledge of the law, not from the statute text. They are built with
`confidence: medium` (the author's own certainty is in `author_confidence`) and `verified: null`.

1. Run `notebooks/kaggle_legal_eval_v2_verify.ipynb` (CPU, ~15 min). It checks every citation against the
   corpus, fills `gold_chunks`, and flags numbers the cited text doesn't contain.
2. Have a lawyer review the flagged items, the 168 medium-confidence v1 items and the 6 flagged in the
   29 Sept review (IL-068, IL-074, IL-090, IL-112, IL-131, IL-393).
3. Answer `test/questions.jsonl` (from step 1) with `scripts/legal_data/eval_run.py answer`, score with the
   set's unmodified `score.py`, then run `scripts/legal_data/eval_v2.py extras` for retrieval recall against
   `gold_chunks` and paraphrase consistency.

`score.py` needs no change: yes/no items use `label_then_judge`, multiple choice `exact_letter`, everything
else the judge. Computation items also carry `answer_value` for an exact check.
