# gemma4_31b_caselaw judging: handoff (4 Oct 2026)

Task from the user: judge the new run in `evals/gemma4_31b_caselaw_results/`, build a report page (artifact)
with the total score, and compare it with the same 100-question test run with Qwen.

## Run facts
- Kaggle, 1 Oct 2026, commit 42f5d22, gemma4:31b (thinking off), retrieval on CPU, 2x T4.
- Arm `caselaw`: 100 questions + 5 cases, no errors. Arm `no_caselaw`: only 43 questions and 0 cases
  (it stopped, probably on the time guard), so the arms can only be compared on those 43.
- Unit tests 34 passed.

## Done
- All 84 judged questions graded by Claude with Prompt A (rubric in `score.py` / the eval zip
  `legal_txt/Evals/israeli_legal_eval.zip`). Grades and notes are in `build_judged.py`. Running it writes
  `judged_caselaw.jsonl` and `judged_no_caselaw.jsonl`. Reports come from the unmodified `score.py`:
  `report_caselaw.txt/json` and `report_no_caselaw.txt/json`.
- Calibrated against the earlier Claude grades (gemma27b, qwen3:14b) on about 30 borderline items.

### Question scores (score.py)
| | overall | MCQ | yes/no | recall | application | issue spotting | temporal | abstention |
|---|---|---|---|---|---|---|---|---|
| gemma4:31b caselaw (100) | **85.0** | 16/16 | 94.4 | 95.5 | 50.0 | 50.0 | 50.0 | 100 |
| gemma4:31b, the 43 shared questions | caselaw 88.4 / no_caselaw 87.2 | | | | | | | |
| qwen3:14b, 28 Sep, Claude-judged (same 100) | 68.5 | 93.8 | 75.0 | 79.5 | 37.5 | 21.4 | 37.5 | 83.3 |
| gemma3:27b + fixes, 29 Sep, Claude-judged | 75.8 | 14/16 | | | 42 | 50 | | |
| qwen3:32b, 27 Sep, **gemma3:27b judge** | 41.0 reported / ~58.5 adjusted | 14/16 | 16/18 | | | | | |

Qwen3:32b answers aren't in the repo (only `evals/bulk500_qwen32b_2026-09-27/REPORT_2026-09-27.md`), so
it can't be re-judged. Its numbers use a different judge and pre-PR #2 code. Qwen3:14b (same sample,
Claude-judged, `evals/corpus_qwen14b_gemma12b_judged/`) is the only like-for-like Qwen comparison.

### Case law effect
77 of 100 questions got judgments (2.57 each), and 7 answers cite one. Grades differ between arms on
only one shared item (IL-274: case law added the warn-the-employer rule, 2 vs 1). Small but real gain;
no harm seen.

## Cases (Prompt B, gold in `cases_gold.json`)
Qwen3:14b cases were re-graded in this session for the same-session rule. They came out close to the
earlier grades (23/21/49 then).

| case | gemma4:31b | qwen3:14b re-graded |
|---|---|---|
| case_01 | **49** (9/9/11/5/11/4/8/2, -5 wage s.12 wrong, -5 invented 3-year labour limitation; בג"ץ 7000/08 was retrieved, not invented) | 22 |
| case_02 | **72** (9/10/16/12/10/4/8/3, no penalties; misses the security cap 25י) | 22 |
| case_03 | **11** (truncated work file, see bug below) | 49 |
| case_04 | **52** (9/7/11/11/6/8/6/4, -5 LRS Ordinance s.85, -5 Sale (Apartments) Law s.2) | 3 (earlier 0) |
| case_05 | **66** (9/10/17/12/6/6/7/4, -5 s.25(b)(1) for handwritten will) | 22 (earlier 24) |
| mean | **50.0** (59.8 without case_03) | 23.6 (17.3) |

Per-case dumps (case file + gold + both work files) were built in a temp scratchpad; rebuild them from
`cases_gold.json`, the case markdown in the eval zip, and
`gemma4_31b/caselaw/cases/answers_rag.jsonl` / the qwen3_14b `cases/answers_rag.jsonl` under
`evals/corpus_qwen14b_gemma12b/`.

## Bug found (important)
case_03's case answer call (`legal_eval_baseline`) ran with **max_tokens 256** and stopped at length
(prompt 15,251 tokens). The `missing_sections` repair also got 256 tokens. The work file is 869
characters: half a summary and half a timeline. The other cases look complete. Check why the case
call got 256 instead of CASE_MAX_TOKENS 5120. It is likely a token budget clamp: context 16384 minus a
prompt of about 15.2k leaves about 1k, minus a reserve. Fix: cap the case prompt (CASE_TOP_K, case law
1500 tokens) so there's room to answer, or raise CONTEXT_LENGTH. llm_trace:
`gemma4_31b/caselaw/llm_trace/2026-10-01.jsonl`, job_id case_03.

## Other weaknesses seen
- "חוק התרופות" claimed not to exist (IL-061, IL-108). This is a pipeline or prompt artifact.
- Says the material lacks the answer when retrieval missed it (IL-198, IL-223, IL-327). IL-347 denies
  the premise.
- IL-306: wrong label (כן). It applied the Youth Law consult rule instead of age 12.
- Rule application and issue spotting are still about 50%: correct core, missing secondary issues.

## Status (4 Oct 2026): DONE
All 10 case grades in `judged_cases.jsonl`. Report page `legal_eval_report.html`, published at
https://claude.ai/artifact/L7ASEJVG5FMaLPH7GcKW5F. Still open: fix the case_03 token-budget bug and re-run.
