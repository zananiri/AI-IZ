"""Shared helpers for the v2 item files: area names, law codes, and the item constructor.

Every new item is written from knowledge of Israeli law, not checked against the statute text:
it is built with confidence "medium" (the author's own certainty goes to author_confidence) and verified=None, and counts in scores only
after scripts/legal_data/verify_gold_v2.py has found its cited section in the corpus (and a lawyer
has looked at anything that script flags)."""

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
    "SAI": 'חוק המכר (דירות) (הבטחת השקעות של רוכשי דירות), התשל"ה-1974',
    "SAP": 'חוק המכר (דירות), התשל"ג-1973',
    "WIR": 'חוק האזנת סתר, התשל"ט-1979',
    "NII": 'חוק הביטוח הלאומי [נוסח משולב], התשנ"ה-1995',
    "MIN": 'חוק שכר מינימום, התשמ"ז-1987',
    "ANL": 'חוק חופשה שנתית, התשי"א-1951',
}

INSTRUCTIONS = {
    "short": "ענה בקצרה לפי הדין הישראלי וציין את מקור החוק והסעיף.",
    "yesno": "ענה 'כן' או 'לא' במילה הראשונה, ולאחר מכן נמק בקצרה וציין את מקור החוק והסעיף.",
    "open": "נתח לפי מבנה: סוגיה, כלל (חוק וסעיף), יישום על העובדות, מסקנה.",
    "list": "זהה את כל הסוגיות המשפטיות העולות מהעובדות. לכל סוגיה ציין את החוק הרלוונטי במשפט אחד.",
    "abstain": "ענה לפי הדין הישראלי וציין מקור. אם אין בסיס לענות, אמור זאת במפורש.",
    "compare": "הסבר בקצרה את ההבדל בין שני הכללים, וציין לכל אחד את מקור החוק והסעיף.",
    "compute": "חשב והצג את החישוב בקצרה, וציין את מקור החוק והסעיף שעליו מבוסס.",
}

FORMAT = {"rule_conclusion": "yesno", "rule_application": "open", "issue_spotting": "list",
          "comparison": "compare", "multi_hop": "short", "computation": "compute",
          "temporal_amendment": "short", "abstention": "abstain", "rule_recall": "short"}
SCORING = {"yesno": "label_then_judge"}


def item(category, area, question, gold, key_points, cites, confidence="medium", **extra):
    """One new item. `cites` are "CODE:section" strings ("CON:14(א)"); an empty section cites the
    law as a whole. `extra` sets answer_label, answer_value, as_of, hops, negated, abstain_reason."""
    fmt = extra.pop("format", FORMAT[category])
    return {
        "category": category, "area": area, "format": fmt, "question": question,
        "gold_answer": gold, "key_points": list(key_points),
        "cites": list(cites), "confidence": confidence, **extra,
    }
