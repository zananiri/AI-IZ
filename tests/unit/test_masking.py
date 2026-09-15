from docslides.cleaning.masking import mask_non_translatable_spans, restore_masks


def test_masks_and_restores_numbers_and_units():
    text = "The device weighs 12.5 kg and costs $99."
    masked = mask_non_translatable_spans(text, "en", run_ner=False)

    assert "12.5" not in masked.text
    assert "kg" not in masked.text
    assert "99" not in masked.text
    assert "[[UNIT_0]]" in masked.text

    restored = restore_masks(masked.text, masked.spans)
    assert restored == text


def test_masks_placeholders_and_formulas():
    text = "Hello {{name}}, your code is `E=mc^2` and result is $x^2$."
    masked = mask_non_translatable_spans(text, "en", run_ner=False)

    assert "{{name}}" not in masked.text
    assert "`E=mc^2`" not in masked.text
    assert "$x^2$" not in masked.text

    restored = restore_masks(masked.text, masked.spans)
    assert restored == text


def test_round_trip_is_idempotent_on_plain_text():
    text = "No masks needed here at all."
    masked = mask_non_translatable_spans(text, "en", run_ner=False)
    assert masked.text == text
    assert masked.spans == []
