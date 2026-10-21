"""Shared helpers for the v3 item files: area names, law codes, and the item constructor.

v3 is a fresh 100-question set built to the criteria of the 100-question sample the Gemma runs used
(29 Sept, 1 Oct): the same nine categories in the same numbers, the same areas, the same formats and
instructions, the same gold fields and scoring (v1 score.py, unmodified). Every item is new and is
written from knowledge of Israeli law, not checked against the statute text. `confidence` is the
author's certainty: "medium" marks an exact section number, a period or a recent amendment that a
lawyer should confirm before the item counts in a headline score."""

CON = "חוזים – חלק כללי"
REM = "תרופות, מכר, מתנה, שליחות, ערבות, ביטוח, חוזים אחידים, עשיית עושר"
LND = "מקרקעין ושכירות"
FAM = "משפחה, ירושה וכשרות משפטית"
TOR = "נזיקין, לשון הרע והתיישנות"
LAB = "דיני עבודה"
PEN = "משפט פלילי, מעצרים וראיות"
CST = "משפט חוקתי, מינהלי, בתי משפט וסדר דין אזרחי"
COM = "חברות, צרכנות, פרטיות, ייצוגיות, בוררות ואתיקה"
ISS = "זיהוי סוגיות (רב-תחומי)"
ABS = "הימנעות, הנחות שגויות ומחוץ למאגר"

# Codes the v1 gold.jsonl already uses are read from it at build time; these are the new ones.
NEW_LAWS = {
    "SIC": 'חוק דמי מחלה, התשל"ו-1976',
    "TST": "פקודת הפרוצדורה הפלילית (עדות)",
    "SRC": 'פקודת סדר הדין הפלילי (מעצר וחיפוש) [נוסח חדש], התשכ"ט-1969',
    "CNR": 'תקנות הגנת הצרכן (ביטול עסקה), התשע"א-2010',
    "SAP": 'חוק המכר (דירות), התשל"ג-1973',
    "NII": 'חוק הביטוח הלאומי [נוסח משולב], התשנ"ה-1995',
    "PAT": 'חוק זכויות החולה, התשנ"ו-1996',
}

INSTRUCTIONS = {
    "short": "ענה בקצרה לפי הדין הישראלי וציין את מקור החוק והסעיף.",
    "yesno": "ענה 'כן' או 'לא' במילה הראשונה, ולאחר מכן נמק בקצרה וציין את מקור החוק והסעיף.",
    "mcq": "בחר תשובה אחת מבין א/ב/ג/ד. כתוב את האות בלבד בשורה הראשונה, ולאחר מכן נימוק קצר.",
    "open": "נתח לפי מבנה: סוגיה, כלל (חוק וסעיף), יישום על העובדות, מסקנה.",
    "list": "זהה את כל הסוגיות המשפטיות העולות מהעובדות. לכל סוגיה ציין את החוק הרלוונטי במשפט אחד.",
    "abstain": "ענה לפי הדין הישראלי וציין מקור. אם אין בסיס לענות, אמור זאת במפורש.",
}

FORMAT = {"rule_recall": "short", "rule_conclusion": "yesno", "mcq_bar": "mcq", "rule_application": "open",
          "interpretation": "short", "citation_grounding": "short", "issue_spotting": "list",
          "abstention": "abstain", "temporal_amendment": "short"}
SCORING = {"yesno": "label_then_judge", "mcq": "exact_letter"}


def item(category, area, question, gold, key_points, cites, confidence="high", **extra):
    """One item. `cites` are "CODE:section" strings ("CON:14(א)"); an empty section cites the law as a
    whole. `extra` sets answer_label (yes/no), answer_letter + options + bar_part (multiple choice)
    and abstain_kind (abstention)."""
    return {
        "category": category, "area": area, "format": FORMAT[category], "question": question,
        "gold_answer": gold, "key_points": list(key_points), "cites": list(cites),
        "confidence": confidence, **extra,
    }


def yesno(area, question, label, gold, key_point, cites, confidence="high"):
    return item("rule_conclusion", area, question, gold, [label, key_point], cites, confidence, answer_label=label)


def mcq(area, question, options, letter, gold, key_point, cites, bar_part="substantive", confidence="high"):
    assert list(options) == ["א", "ב", "ג", "ד"] and letter in options
    return item("mcq_bar", area, question, f"{letter}. {options[letter]} — {gold}", [key_point], cites, confidence,
                options=dict(options), answer_letter=letter, bar_part=bar_part)
