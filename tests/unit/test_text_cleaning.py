from docslides.cleaning.text_cleaning import (
    dehyphenate,
    detect_repeated_headers_footers,
    normalize_unicode_and_strip_control,
    normalize_whitespace,
    strip_headers_footers,
)


def test_normalize_strips_control_chars():
    text = "Hello\x00World\x07!"
    assert normalize_unicode_and_strip_control(text) == "HelloWorld!"


def test_normalize_whitespace_collapses_spaces_and_blank_lines():
    text = "Hello   world\n\n\n\nNext"
    assert normalize_whitespace(text) == "Hello world\n\nNext"


def test_dehyphenate_joins_wrapped_words_for_latin_languages():
    text = "This is a hyphen-\nated word."
    assert dehyphenate(text, "en") == "This is a hyphenated word."


def test_dehyphenate_skipped_for_non_latin_scripts():
    text = "مرحبا-\nبكم"
    assert dehyphenate(text, "ar") == text


def test_detect_and_strip_repeated_headers_footers():
    pages = [
        "Company Confidential\nBody text for page one\nmore body\nPage 1",
        "Company Confidential\nBody text for page two\nmore body\nPage 2",
        "Company Confidential\nBody text for page three\nmore body\nPage 3",
        "Company Confidential\nBody text for page four\nmore body\nPage 4",
    ]
    repeated = detect_repeated_headers_footers(pages, repetition_threshold=0.6)
    assert "Company Confidential" in repeated

    cleaned = strip_headers_footers(pages[0], repeated)
    assert "Company Confidential" not in cleaned
    assert "Body text for page one" in cleaned
