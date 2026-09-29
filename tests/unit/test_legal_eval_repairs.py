"""The eval scripts' answer repairs (scripts/legal_data/eval_run.py, eval_cases.py): a yes/no opener
the instructions didn't ask for, a rule_conclusion label its own explanation contradicts, and a
work file with missing sections or dates/amounts the case file doesn't hold."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "legal_data"))

import eval_cases
import eval_run


def test_a_bare_label_is_split_off_but_a_sentence_starting_with_lo_is_not():
    assert eval_run.split_yes_no_opener("לא. לפי סעיף 5 לחוק ...") == ("לא", "לפי סעיף 5 לחוק ...")
    assert eval_run.split_yes_no_opener("**כן**, המעסיק רשאי") == ("כן", "המעסיק רשאי")
    assert eval_run.split_yes_no_opener("כן\nהמעסיק רשאי") == ("כן", "המעסיק רשאי")
    assert eval_run.split_yes_no_opener("לא ניתן לפטר עובדת בהריון") == (None, "לא ניתן לפטר עובדת בהריון")
    assert eval_run.split_yes_no_opener("כן.") == (None, "כן.")  # nothing after it: leave it alone


class _Verdict:
    def __init__(self, answer):
        self.calls, self.answer = [], answer

    async def complete_json(self, messages, *_, **__):
        self.calls.append(messages[0].content)
        return eval_run.LabelVerdict(answer=self.answer)


def test_a_missing_label_is_added_but_one_the_model_wrote_is_kept():
    # The 29 Sept 27B runs: swapping the model's own label broke more answers than it fixed.
    checker = _Verdict("no")
    assert asyncio.run(eval_run.check_label(checker, "האם מותר?", "כן. המעסיק אינו רשאי לפטר, סעיף 9.")) == (
        "כן. המעסיק אינו רשאי לפטר, סעיף 9.", None)
    assert checker.calls == []  # no call needed when the answer has a label

    assert asyncio.run(eval_run.check_label(_Verdict("no"), "ש", "לא ניתן לפטר.")) == ("לא ניתן לפטר.", None)
    assert asyncio.run(eval_run.check_label(_Verdict("yes"), "ש", "המעסיק רשאי.")) == ("כן. המעסיק רשאי.", "label_added")
    assert asyncio.run(eval_run.check_label(_Verdict("unclear"), "ש", "אולי.")) == ("אולי.", None)


WORK_FILE = """## 1. תקציר עובדתי
העובדת פוטרה ב-10.9.2026.
## 2. ציר זמן
| 10.9.2026 | פיטורים | מסמך 3 |
## 3. סוגיות משפטיות
**4. מועדים ודחיפות**
הגשת תביעה עד 10.12.2026 (10.9.2026 + 90 יום). שכר 14,000 ש"ח, פיצוי 50,000 ש"ח.
5) סתירות וסימני אזהרה
### 6. מידע ומסמכים חסרים
- 1. חוזה העבודה
"""


def test_missing_sections_and_values_the_case_file_does_not_hold():
    assert eval_cases.missing_sections(WORK_FILE) == ["7. טיוטת מסמך", "8. המלצה לצעד הבא"]
    case_text = "תאריך קבלת התיק: 20.9.2026. פוטרה ב-10.9.2026. שכרה 14,000 ש\"ח."
    assert eval_cases.unsupported_values(WORK_FILE, case_text) == ["10.12", "50,000"]


def test_a_value_followed_by_a_comma_is_still_found_in_the_case_file():
    # case_02 on 29 Sept: every "3.2.2026, יוסי:" date was missed in the case file, so the work
    # file's correct copies of them were reported as unsupported.
    case_text = "- 3.2.2026, יוסי: נזילה.\n- 20.3.2026, אבי: בוצע. פיקדון 13,000, ערבות 19,500 ש\"ח."
    work_file = "| 3.2.2026 | נזילה |\n| 20.3.2026 | תיקון |\nפיקדון 13,000, ניכויים 8,000 ש\"ח."
    assert eval_cases.unsupported_values(work_file, case_text) == ["8,000"]
