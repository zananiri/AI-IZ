"""A synthetic stand-in for cases_all.parquet, in the documented column layout, for tests and
offline dry runs. Texts are made-up judgments; every edge case the pipeline handles appears:
mojibake (Latin-1 and Greek), a bogus 1920 date, a UTC-shifted VerdictDt, technical and short
החלטות, orders, decisions from 2022 on, duplicates, empty text, an English judgment."""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

JUDGES = [("ד' דורנר", "דורנר"), ("מ' נאור", "נאור"), ("א' חיות", "חיות"), ("א' ברק", "ברק"),
          ("ת' אור", "אור"), ("א' ריבלין", "ריבלין"), ("ע' ארבל", "ארבל")]
PROCEEDINGS = [('בג"ץ', "בג\"ץ", "בשבתו כבית משפט גבוה לצדק", "העותרת", "המשיבים"),
               ('ע"א', "אזרחי", "בשבתו כבית משפט לערעורים בעניינים אזרחיים", "המערער", "המשיבה"),
               ('ע"פ', "פלילי", "בשבתו כבית משפט לערעורים בעניינים פליליים", "המערער", "המשיבה"),
               ('רע"א', "אזרחי", "", "המבקשת", "המשיב")]
TOPICS = [
    "חובת ההנמקה של רשות מנהלית היא מן היסודות של המשפט המנהלי, והיא מאפשרת ביקורת שיפוטית על שיקול הדעת.",
    "עקרון המידתיות מחייב כי הפגיעה בזכות תהיה במידה שאינה עולה על הנדרש, ויש לבחון את מבחני המשנה.",
    "חובת תום הלב בקיום חוזה חלה על שני הצדדים, והפרתה מקימה לנפגע זכות לתרופות.",
    "הלכת השיתוף בנכסי בני זוג חלה על מי שנישאו לפני תחילת חוק יחסי ממון.",
    "הרשות המנהלית חייבת לשקול את כל השיקולים הרלוונטיים ורק אותם, ולתת לכל שיקול את משקלו הראוי.",
]
CITES = ['ע"א 6821/93 בנק המזרחי המאוחד בע"מ נ\' מגדל כפר שיתופי, פ"ד מט(4) 221',
         'בג"ץ 5856/03 פלוני נ\' שר הפנים', 'ע"פ 1234/99 מדינת ישראל נ\' פלוני',
         'ת"א (ת"א) 1234/98 כהן נ\' לוי', 'רע"א 2279/03', 'פ"ד נז(1) 1']


def judgment(rng: random.Random, proc, number: str, judges, n_paras: int) -> str:
    abbr, _, sitting, side_a, side_b = proc
    lines = ["בבית המשפט העליון בירושלים"]
    if sitting:
        lines.append(sitting)
    lines += ["", f"{abbr} {number}", "", f"בפני: כבוד השופטת {judges[0][0]}"]
    lines += [f"          כבוד השופט {j[0]}" for j in judges[1:]]
    lines += ["", f"{side_a}: פלונית אלמונית", "", "נ ג ד", "", f"{side_b}: 1. שר הפנים", "2. רשות האוכלוסין",
              "", f"בשם {side_a}: עו\"ד משה כהן", "", "פסק-דין", "", f"השופטת {judges[0][1]}:", "", "רקע"]
    para = 1
    for section in ("רקע", "דיון והכרעה"):
        if section != "רקע":
            lines += ["", section]
        for _ in range(n_paras):
            body = " ".join(rng.choice(TOPICS) for _ in range(rng.randint(2, 6)))
            if rng.random() < 0.5:
                body += f" ראו {rng.choice(CITES)}; וכן {rng.choice(CITES)}."
            lines += ["", f"{para}.     {body}"]
            para += 1
    lines += ["", "סוף דבר", "", f"{para}. אשר על כן, העתירה נדחית. העותרת תישא בהוצאות המשיבים בסך 10,000 ש\"ח.",
              "", "ש ו פ ט ת", "", f"השופט {judges[1][1]}:", "", "אני מסכים.", "",
              "ניתן היום, ט' בתמוז התשס\"ג (9.7.2003).", "", "העותק כפוף לשינויי עריכה וניסוח.",
              "מרכז מידע, טל' 02-6593666 ; אתר אינטרנט, www.court.gov.il"]
    return "\n".join(lines)


def english_judgment(number: str) -> str:
    return "\n".join(["IN THE SUPREME COURT OF ISRAEL", "", f"HCJ {number}", "", "Before: The Hon. Justice D. Dorner",
                      "", "Petitioner: John Doe", "", "v.", "", "Respondent: Minister of Interior", "", "JUDGMENT", "",
                      "1. The duty to state reasons is a basic principle of administrative law. " * 8,
                      "", "2. For these reasons the petition is denied."])


def make_rows(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        proc = rng.choice(PROCEEDINGS)
        year = rng.randint(1995, 2023)
        yy = f"{year % 100:02d}"
        number = f"{rng.randint(100, 9999)}/{yy}"
        judges = rng.sample(JUDGES, 3)
        kind = rng.random()
        doc_type, technical = "פסק-דין", False
        text = judgment(rng, proc, number, judges, rng.randint(2, 7))
        if kind < 0.15:
            doc_type, technical = "החלטה", True
            text = f"בבית המשפט העליון\n\n{proc[0]} {number}\n\nהחלטה\n\nהבקשה להארכת מועד מתקבלת."
        elif kind < 0.25:
            doc_type = "החלטה"  # a long, non-technical decision: kept
        elif kind < 0.30:
            doc_type = "צו ביניים"
        elif kind < 0.33:
            doc_type, text = "פסקי דין באנגלית", english_judgment(number)
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        verdict = f"{year}-{month:02d}-{day:02d}"
        row = {
            "Id": f"doc{i:06d}", "CaseId": f"case{i:06d}", "case_id": f"case{i:06d}",
            "CaseDesc": f"{proc[0]} {number}", "meta_case_nbr": f"{proc[0]} {number}",
            "document_hash": f"h{i:08x}", "text": text, "html_title": f"{proc[0]} {number} פלונית נ' שר הפנים",
            "CaseName": "פלונית נ' שר הפנים", "meta_case_nm": "פלונית נ' שר הפנים",
            "Type": doc_type, "TypeCode": 1, "meta_verdict_ty": doc_type,
            "Technical": technical, "meta_is_technical": str(technical).lower(),
            "VerdictsDt": verdict, "meta_verdict_dt": verdict, "meta_case_dt": f"{year - 1}-{month:02d}-01",
            "VerdictDt": f"{verdict}T21:00:00", "Year": year,
            "meta_judge": [j[0] for j in judges], "meta_judge_nm_last": [j[1] for j in judges],
            "meta_side_nm": ["פלונית אלמונית", "שר הפנים"], "meta_lawyer_nm": ["משה כהן"],
            "meta_court_nm": "בית המשפט העליון", "meta_mador_nm": proc[1], "meta_inyan_nm": proc[0],
            "Path": f"{yy}\\{rng.randint(100, 999)}\\{rng.randint(0, 99):03d}", "FileName": f"{yy}0{i:05d}.a01",
            "file_name": f"{yy}0{i:05d}.a01",
        }
        rows.append(row)
    # Edge cases at fixed positions.
    if n >= 12:
        rows[1]["text"] = rows[1]["text"].encode("cp1255", errors="replace").decode("latin-1")       # mojibake
        rows[2]["text"] = rows[2]["text"].encode("cp1255", errors="replace").decode("cp1253", errors="replace")
        for r in rows[1:3]:
            r.update(Type="פסק-דין", meta_verdict_ty="פסק-דין", Technical=False, meta_verdict_dt="2003-07-09",
                     VerdictsDt="2003-07-09", meta_case_dt="2002-01-01", Year=2003)
        rows[3].update(Type="פסק-דין", meta_verdict_dt="1920-07-09", VerdictsDt=None,              # bogus date
                       VerdictDt="2003-07-08T19:00", meta_case_dt="2002-05-01", Year=2003)
        rows[4].update(Type="פסק-דין", meta_verdict_dt=None, VerdictsDt=None,                      # UTC-shifted only
                       VerdictDt="2021-12-31T21:00", meta_case_dt="2021-01-01", Year=2021)
        rows[5].update(Type="פסק-דין", meta_verdict_dt="2022-03-01", VerdictsDt="2022-03-01", Year=2022)
        rows[6].update(text="")
        rows[7].update(text=rows[0]["text"], Type=rows[0]["Type"], meta_verdict_dt=rows[0]["meta_verdict_dt"],
                       VerdictsDt=rows[0]["VerdictsDt"], Technical=rows[0]["Technical"])              # duplicate
        rows[8].update(meta_verdict_dt=None, VerdictsDt=None, VerdictDt=None, Year=None, meta_case_dt=None)
    return rows


def make_dataset(path: Path, n: int = 300, seed: int = 7) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(make_rows(n, seed))
    pq.write_table(table, path, row_group_size=1000)
    return path
