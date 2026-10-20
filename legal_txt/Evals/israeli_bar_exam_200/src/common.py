"""Shared pieces of the bar-exam set: the corpus cutoff, the laws cited, the areas, and item builders.

Every item is built by one of the helpers below, which return (question_row, gold_row) in the v1 schema
(legal_txt/Evals/questions.jsonl, gold.jsonl), so score.py, eval_run.py and eval_v2.py read them unchanged."""

from __future__ import annotations

# The law corpus is built from the Hebrew Wikisource dump of 1 Sept 2026 (scripts/legal_data/README.md);
# the case-law index stops at 2022. The model's own knowledge stops in January 2025 (config/config.yaml).
CUTOFF = "2026-09-01"
CUTOFF_HE = "1.9.2026"
CASELAW_CUTOFF_HE = "2022"

LAWS = {
    "CON": 'חוק החוזים (חלק כללי), התשל"ג-1973',
    "CON26": 'חוק החוזים (חלק כללי) (תיקון מס\' 3), התשפ"ו-2026',
    "REM": 'חוק החוזים (תרופות בשל הפרת חוזה), התשל"א-1970',
    "SAL": 'חוק המכר, התשכ"ח-1968',
    "GIF": 'חוק המתנה, התשכ"ח-1968',
    "RNT": 'חוק השכירות והשאילה, התשל"א-1971',
    "AGN": 'חוק השליחות, התשכ"ה-1965',
    "GUA": 'חוק הערבות, התשכ"ז-1967',
    "INS": 'חוק חוזה הביטוח, התשמ"א-1981',
    "ARB": 'חוק הבוררות, התשכ"ח-1968',
    "TOR": "פקודת הנזיקין [נוסח חדש]",
    "DEF": 'חוק איסור לשון הרע, התשכ"ה-1965',
    "RAV": 'חוק הפיצויים לנפגעי תאונות דרכים, התשל"ה-1975',
    "LND": 'חוק המקרקעין, התשכ"ט-1969',
    "SUC": 'חוק הירושה, התשכ"ה-1965',
    "SPO": 'חוק יחסי ממון בין בני זוג, התשל"ג-1973',
    "CAP": 'חוק הכשרות המשפטית והאפוטרופסות, התשכ"ב-1962',
    "CMP": 'חוק החברות, התשנ"ט-1999',
    "INV": 'חוק חדלות פירעון ושיקום כלכלי, התשע"ח-2018',
    "LIM": 'חוק ההתיישנות, התשי"ח-1958',
    "CPR": 'תקנות סדר הדין האזרחי, התשע"ט-2018',
    "CRT": 'חוק בתי המשפט [נוסח משולב], התשמ"ד-1984',
    "PEN": 'חוק העונשין, התשל"ז-1977',
    "CRP": 'חוק סדר הדין הפלילי [נוסח משולב], התשמ"ב-1982',
    "ARR": 'חוק סדר הדין הפלילי (סמכויות אכיפה – מעצרים), התשנ"ו-1996',
    "OCT7": 'החוק בעניין העמדה לדין של מבצעי טבח 7 באוקטובר (ס"ח 3527), התשפ"ו-2026',
    "EVD": 'פקודת הראיות [נוסח חדש], התשל"א-1971',
    "BDL": "חוק-יסוד: כבוד האדם וחירותו",
    "BFO": "חוק-יסוד: חופש העיסוק",
    "BKN": "חוק-יסוד: הכנסת",
    "BJU": "חוק-יסוד: השפיטה",
    "ELC26": 'חוק הבחירות לכנסת העשרים ושש (הוראות מיוחדות ותיקוני חקיקה), התשפ"ו-2026',
    "SEV": 'חוק פיצויי פיטורים, התשכ"ג-1963',
    "NOT": 'חוק הודעה מוקדמת לפיטורים ולהתפטרות, התשס"א-2001',
    "WOM": 'חוק עבודת נשים, התשי"ד-1954',
    "EQU": 'חוק שוויון ההזדמנויות בעבודה, התשמ"ח-1988',
    "LCT": 'חוק בית הדין לעבודה, התשכ"ט-1969',
    "HRS": 'חוק שעות עבודה ומנוחה, התשי"א-1951',
    "WAG": 'חוק הגנת השכר, התשי"ח-1958',
    "BAR": 'חוק לשכת עורכי הדין, התשכ"א-1961',
    "ETH": 'כללי לשכת עורכי הדין (אתיקה מקצועית), התשמ"ו-1986',
    "CNS": 'חוק הגנת הצרכן, התשמ"א-1981',
    "UNJ": 'חוק עשיית עושר ולא במשפט, התשל"ט-1979',
    "PRV": 'חוק הגנת הפרטיות, התשמ"א-1981',
}

# Areas follow the subjects of the Israel Bar Association's licensing exam.
A_CON = "חוזים – חלק כללי"
A_REM = "תרופות בשל הפרת חוזה"
A_SPC = "חוזים מיוחדים (מכר, מתנה, שכירות, שליחות, ביטוח, בוררות)"
A_TOR = "נזיקין, לשון הרע ותאונות דרכים"
A_LND = "מקרקעין"
A_FAM = "ירושה, יחסי ממון וכשרות משפטית"
A_CMP = "חברות וחדלות פירעון"
A_CIV = "סדר דין אזרחי, בתי משפט והתיישנות"
A_PEN = "דיני עונשין"
A_CRP = "סדר דין פלילי ומעצרים"
A_EVD = "ראיות"
A_CST = "משפט חוקתי"
A_LAB = "דיני עבודה"
A_ETH = "אתיקה מקצועית"
A_CNS = "צרכנות, עשיית עושר ופרטיות"
A_GEN = "כללי – גבולות המאגר"

_CUTOFF_NOTE = f" המאגר כולל חקיקה בנוסחה נכון ל-{CUTOFF_HE}."
INSTRUCTIONS = {
    "mcq": "בחר תשובה אחת מבין א/ב/ג/ד. כתוב את האות בלבד בשורה הראשונה, ולאחר מכן נימוק קצר." + _CUTOFF_NOTE,
    "yesno": "ענה 'כן' או 'לא' במילה הראשונה, ולאחר מכן נמק בקצרה וציין את מקור החוק והסעיף." + _CUTOFF_NOTE,
    "open": "נתח לפי מבנה: סוגיה, כלל (חוק וסעיף), יישום על העובדות, מסקנה." + _CUTOFF_NOTE,
    "list": "מנה את כל הסוגיות המשפטיות שעולות מהעובדות, ולכל אחת ציין את החוק והסעיף." + _CUTOFF_NOTE,
    "short": "ענה בקצרה לפי הדין הישראלי וציין את מקור החוק והסעיף." + _CUTOFF_NOTE,
    "abstain": "ענה לפי הדין הישראלי ולפי המאגר בלבד. אם המידע אינו במאגר, אם הנחת השאלה שגויה או אם הבקשה "
               "אינה ראויה, אמור זאת במפורש ואל תמציא." + _CUTOFF_NOTE,
}
LETTERS = ["א", "ב", "ג", "ד"]


def cites(*pairs: tuple[str, str]) -> list[dict]:
    return [{"law_code": code, "law": LAWS[code], "section": section} for code, section in pairs]


def _base(area, category, fmt, scoring, question, gold_answer, key_points, citations, confidence, **extra):
    q = {"category": category, "area": area, "format": fmt, "instructions": INSTRUCTIONS[fmt], "question": question,
         "as_of": CUTOFF}
    g = {"category": category, "area": area, "format": fmt, "scoring": scoring, "gold_answer": gold_answer,
         "key_points": key_points, "citations": citations, "confidence": confidence, "as_of": CUTOFF,
         "source": "bar200", "split": "test", **extra}
    return q, g


def mcq(area, question, correct, wrong, explanation, citations, confidence="high", position=0):
    """`correct` goes to letter LETTERS[position]; the build cycles positions so each letter is right 30 times."""
    assert len(wrong) == 3, question
    options = list(wrong)
    options.insert(position, correct)
    letter = LETTERS[position]
    q, g = _base(area, "mcq_bar", "mcq", "exact_letter", question, f"{letter}. {correct} — {explanation}",
                 [explanation], citations, confidence, answer_letter=letter)
    q["options"] = dict(zip(LETTERS, options))
    q["bar_part"] = "bar_exam"
    return q, g


def yesno(area, question, label, explanation, citations, confidence="high"):
    assert label in ("כן", "לא")
    return _base(area, "rule_conclusion", "yesno", "label_then_judge", question, f"{label}. {explanation}",
                 [label, explanation], citations, confidence, answer_label=label)


def open_q(area, question, gold_answer, key_points, citations, confidence="high"):
    return _base(area, "rule_application", "open", "judge", question, gold_answer, key_points, citations, confidence)


def spot(area, question, issues, citations, confidence="high"):
    return _base(area, "issue_spotting", "list", "judge", question, " | ".join(issues), issues, citations, confidence)


def temporal(area, question, gold_answer, key_points, citations, confidence="high"):
    return _base(area, "temporal_amendment", "short", "judge", question, gold_answer, key_points, citations,
                 confidence)


def abstain(area, question, gold_answer, key_points, citations=(), reason="out_of_corpus", confidence="high"):
    return _base(area, "abstention", "abstain", "judge", question, gold_answer, key_points, list(citations),
                 confidence, abstain_reason=reason)
