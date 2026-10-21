# Israeli legal eval v3: 100 new questions + 5 new paralegal cases (for attorney review)

A fresh test built to the criteria of the 100-question sample used by the 29 Sept (Gemma 3 27B) and 1 Oct (Gemma 4 31B)
runs: the same nine categories in the same numbers, the same eleven areas in the same numbers, the same formats and
instructions, the same gold fields and the same scoring. Every question and case is new; a similarity check against the
965 v1/v2 items replaced the ones that repeated an earlier question. Rebuild with
`python legal_txt/Evals/israeli_legal_eval_v3/src/build_v3.py` (deterministic); edit `src/items.py`, `src/cases_gold_src.py`
and `cases/*.md`, never the generated files.

Run it on Kaggle with `notebooks/kaggle_legal_eval_attorney_review.ipynb` (Gemma 4 31B answers, gpt-oss:20b grades). The
notebook writes two reports with `scripts/legal_data/review_reports.py`: `technical_report.html` (to revise and enhance the
system) and `attorney_review.html` (Hebrew, no system or component names, with review controls and a CSV export).

| File | Send to the model? | Contents |
|---|---|---|
| `questions.jsonl` | yes | 100 items: id (`IL3-001`…), category, area, format, instructions, question, options for multiple choice |
| `gold.jsonl` | no, judge only | gold answer, key points, citations (law + section), confidence, scoring, answer_letter / answer_label, abstain_kind |
| `cases/00_TASK_INSTRUCTIONS.md` + `cases/case_1X_*.md` | yes | the shared paralegal task and 5 case files with fictional documents |
| `cases_gold.json` | no, judge only | per case: chronology, legal issues, deadlines with the calculation, red flags, missing information, deliverable checks, next step, must_not; the v1 100-point rubric and penalties |
| `score.py`, `judge_prompt.md` | – | the v1 files, unmodified (Prompt A for questions, Prompt B for cases) |
| `manifest.json` | – | counts |

## Questions

| Category | n | Scored by |
|---|--:|---|
| rule_recall | 22 | judge |
| rule_conclusion (yes/no, 9 כן / 9 לא) | 18 | label automatically, explanation by the judge |
| mcq_bar (4 per letter; 11 substantive, 5 procedural) | 16 | exact letter |
| rule_application (IRAC) | 12 | judge against key points |
| interpretation | 8 | judge |
| citation_grounding | 7 | judge |
| issue_spotting | 7 | judge (share of key points found) |
| abstention (2 false premise, fictional law, harmful request, out of scope, privacy) | 6 | judge |
| temporal_amendment | 4 | judge |

Areas: criminal 15, labor 13, constitutional / administrative / procedure 12, contracts 11, torts 10, remedies and sale 8,
family and succession 7, issue spotting 7, companies / consumer / privacy 6, abstention 6, property and tenancy 5.

## Cases

| Case | Subject | Deliberate traps |
|---|---|---|
| case_11 | wage cut, overtime, resigning as if dismissed | conflicting start dates, a general waiver form, the client still working under the new terms, wants to resign without warning |
| case_12 | a minor's injury in a municipal playground | conflicting birth years, the mother believes the claim expired (minority), a waiver signed for a minor without court approval, the parents' own claim did expire, the client is the now-adult daughter |
| case_13 | summons to an interrogation under caution | the SMS date differs from the client's, an amount conflict, a co-suspect asks for joint representation and coordinated versions, the client wants to delete chats, uses the employer's laptop |
| case_14 | intestate estate, minor heirs, a son from an earlier marriage | an undated photocopied note, the widow wants to exclude the son, the pension goes to a designated beneficiary, resource balancing before succession, sale needs court approval |
| case_15 | personal guarantee for a failing company's loan | the client asks to move the house to a relative, joint representation of spouse and company, the bank's 7 days have passed, unpaid wages |

## Before relying on the gold

The gold answers were written from knowledge of Israeli law as of October 2026 and have not been checked against the statute
text or by a lawyer. 71 items are `high` confidence and 29 `medium` (an exact section number, a period, a regulation or a
recent amendment). The attorney review page is meant for exactly that check: each item can be marked "gold answer right /
needs fixing" with a comment, and the export lists them. Items worth checking first: the sick-pay accrual (IL3-002), the
fair-rental repair periods and the tenant's self-help remedy (the burst-pipe application question), the 2026 Contracts Law amendment (temporal), the early resource
balance (section 5א), and the protected-guarantor rules in case_15.
