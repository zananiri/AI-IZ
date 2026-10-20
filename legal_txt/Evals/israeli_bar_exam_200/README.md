# Israeli bar-exam eval: 200 questions + 10 fictional cases

A test of the offline legal system against the law **as it stands in its corpus** (Hebrew Wikisource dump of
1 Sept 2026; case law to 2022), in the style of the Israel Bar Association licensing exam. Run it on Kaggle with
`notebooks/kaggle_legal_eval_bar_exam_200.ipynb`: it answers every item with the system, judges the answers
against the golden answers here, and writes `summary.md` and a `review.csv` for a lawyer's spot-check.

The questions are original, written in the exam's format and on its subjects; none is copied from a published
exam. All names, companies, institutions and ID numbers in the cases are fictional.

| File | Send to the model? | Contents |
|---|---|---|
| `questions.jsonl` | yes | id, category, area, format, instructions (with the corpus date), question, as_of (+ options for multiple choice) |
| `gold.jsonl` | **no, judge only** | gold_answer, key_points, citations (law + section), confidence, scoring, answer_letter / answer_label, abstain_reason |
| `cases/` | yes | `00_TASK_INSTRUCTIONS.md` and the 10 case files |
| `cases_gold.json` | **no, judge only** | per case: chronology, legal issues, deadlines (with the calculation), red flags, missing info, deliverable checks, next step, must-nots; the 100-point rubric and penalties |
| `score.py`, `judge_prompt.md` | – | unchanged from `israeli_legal_eval` (exact letter, label + judge, judge) |
| `manifest.json` | – | counts by category, area, confidence, answer letter |
| `src/` | – | the items and the build script |

Rebuild with `python legal_txt/Evals/israeli_bar_exam_200/src/build.py` (deterministic); edit `src/`, never the
generated files.

## What it covers

| Category | Items | Scoring |
|---|--:|---|
| `mcq_bar` – bar-exam multiple choice, four options | 120 | exact letter; each letter is right 30 times |
| `rule_conclusion` – yes/no fact patterns | 30 | label 50% + judge 50% (14 yes, 16 no) |
| `rule_application` – open IRAC questions | 15 | judge |
| `issue_spotting` | 10 | judge: share of the gold issues found |
| `temporal_amendment` – which version applies, what changed, the corpus date | 10 | judge |
| `abstention` – after the cutoff (4), false premise (3), made-up section or case (2), out of scope (2), improper request (3), missing facts (1) | 15 | judge |
| Paralegal cases | 10 | judge, 100-point rubric |

Areas (the exam's subjects): contracts (general part, remedies, special contracts), torts, defamation and road
accidents, land, succession / matrimonial property / capacity, companies and insolvency, civil procedure and
limitation, criminal law, criminal procedure and arrests, evidence, constitutional law, labour law, professional
ethics, consumer protection / unjust enrichment / privacy.

Cases: birth malpractice and limitation for a minor and the parents; minority-shareholder oppression; an
individual guarantor of a company in insolvency; immediate dismissal, overtime and a police summons; self-defence
and a pre-indictment hearing; a late handwritten will against a witnessed one; dissolving a co-owned inherited
flat; failed negotiations, an exclusivity clause and the 2026 interpretation amendment; retracting a gift, a
mistaken bank transfer and a humiliating video; a goods dispute with an arbitration clause. Reference dates are
in August 2026, before the corpus cutoff.

## The cutoff

Every question's instructions state the corpus date ("המאגר כולל חקיקה בנוסחה נכון ל-1.9.2026"). Items that
only the corpus can answer, because the models' own knowledge stops in January 2025: Contracts Law amendment
no. 3 (2026, the new section 25(a)), the 26th Knesset elections law (2026), the 7 October trials law (2026),
Privacy Law amendment 13 (August 2025). The `after_cutoff` items ask about 2027 events; the right answer says
the corpus can't know. If the corpus was built from a later dump, set `CUTOFF_OVERRIDE` in the notebook: the
instructions are rewritten, and the 2027 items still hold.

## Before relying on a score

The gold answers were written from knowledge of the law, not from the statute text. 161 items are `high`
confidence and 39 `medium`; `medium` marks a section number or detail to check. The notebook runs
`eval_v2.py verify` (every cited law and section looked up in the corpus, numbers compared) and
`check_corpus_coverage.py`, and writes `gold_verification.md`: have a lawyer look at the flagged items and the
`medium` ones before reporting a score. Known points to check: the CON26 commencement date (case 08 asks the model
to check it rather than assume it), the probate objection period (case 06), and the opposition period for
executing a promissory note (case 03).
