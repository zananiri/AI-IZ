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


def test_the_conclusion_written_last_is_moved_to_the_front():
    text = "לפי סעיף 9(א) לחוק עבודת נשים, אין לפטר עובדת בהריון בלי היתר.\nמסקנה: לא"
    assert eval_run.place_conclusion(text) == (
        "לא. לפי סעיף 9(א) לחוק עבודת נשים, אין לפטר עובדת בהריון בלי היתר.", "conclusion_first")
    # an opening label the reasoning contradicts is overruled by the conclusion
    assert eval_run.place_conclusion("כן. המבטח איחר, חלפו 69 יום.\n**מסקנה:** כן") == (
        "כן. המבטח איחר, חלפו 69 יום.", "conclusion_first")
    assert eval_run.place_conclusion("לא, המבטח איחר.\n\nמסקנה: כן.") == ("כן. המבטח איחר.", "opener_overruled")
    assert eval_run.place_conclusion("המעסיק רשאי.") == ("המעסיק רשאי.", None)  # no conclusion line
    assert eval_run.place_conclusion("מסקנה: כן") == ("מסקנה: כן", None)  # nothing but the line


def test_leaked_template_tokens_are_removed():
    assert eval_run.strip_template_tokens("הפרת חוזה – סעיף 15. </start_of_turn>") == "הפרת חוזה – סעיף 15."
    assert eval_run.strip_template_tokens("תשובה רגילה") == "תשובה רגילה"


def test_amendment_questions_are_told_to_check_the_version_but_repair_questions_are_not():
    assert eval_run.TEMPORAL_RE.search("על אילו חוזים חל הנוסח החדש של סעיף 25(א) שנקבע בתיקון מס' 3?")
    assert eval_run.TEMPORAL_RE.search("האם תמיד היה ניתן לעשות צוואות הדדיות?")
    assert not eval_run.TEMPORAL_RE.search("תוך כמה זמן חייב משכיר לתקן ליקוי? מה דין תיקון על חשבון השוכר?")


class _Scope:
    def __init__(self, verdict=None):
        self.verdict = verdict

    async def complete_json(self, *_, **__):
        if self.verdict is None:
            raise RuntimeError("server error")
        return self.verdict


def test_a_failed_scope_check_never_turns_a_question_into_a_refusal():
    assert asyncio.run(eval_run.check_scope(_Scope(), "ש")).scope == "in_scope"
    flagged = eval_run.ScopeVerdict(scope="foreign_law", note="דין קליפורניה")
    assert asyncio.run(eval_run.check_scope(_Scope(flagged), "ש")) is flagged
    assert set(eval_run.SCOPE_NOTES) == set(eval_run.ScopeVerdict.model_fields["scope"].annotation.__args__) - {"in_scope"}
