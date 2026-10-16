from docslides.legal.numeric_check import unsupported_numbers

EVIDENCE = [
    'ובלבד שהתקשרות כאמור ששווייה עולה על 2.5 מיליון שקלים חדשים תיעשה באישור ועדת הפטור',
    'בסעיף 116יז10(ד), לא יאוחר מ־14 ימים; בסעיף 2(ב), במקום "25%" יבוא "18%"; התשפ"ו-2026',
]


def test_grounded_numbers_pass_by_value():
    answer = "מעל 2,500,000 ש\"ח, באישור ועדת הפטור; השיעור הוחלף מ־25% ל־18% תוך 14 ימים (סעיף 116יז10(ד), 2026)."
    assert unsupported_numbers(answer, EVIDENCE) == []


def test_invented_numbers_are_flagged():
    answer = "מעל 100,000 ש\"ח; השיעור 20%; יש למחוק לאחר 5 שנים, עד 30 ביוני."
    assert unsupported_numbers(answer, EVIDENCE) == ["100,000", "20%", "5", "30"]


def test_scaled_amounts_and_question_numbers():
    assert unsupported_numbers("3 מיליון ש\"ח", EVIDENCE) == ["3 מיליון"]
    assert unsupported_numbers("לפי סעיף 132א", EVIDENCE, question="האם סעיף 132א חל?") == []
