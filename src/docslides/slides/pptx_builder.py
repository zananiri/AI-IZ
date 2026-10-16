"""Build a .pptx deck from generated slide content against a configurable
.potx template, with explicit per-slide RTL handling.

Layout selection matches each slide's `layout_type` against the template's
slide-layout names (standard PowerPoint layout names are used as the
fallback mapping when a custom template doesn't define matching names), so
swapping the configured template doesn't require code changes as long as its
layouts are named conventionally.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.slide import Slide, SlideLayout
from pptx.util import Pt

from docslides.config import get_config
from docslides.llm.schemas import LayoutType, SlideContent
from docslides.logging_setup import get_logger
from docslides.slides.rtl import apply_rtl_to_slide

logger = get_logger(__name__)

# Fallback mapping onto python-pptx's built-in default template layout names,
# used whenever the configured .potx doesn't define a layout with a matching
# name for a given layout_type.
_DEFAULT_LAYOUT_NAME_BY_TYPE: dict[LayoutType, str] = {
    "title_bullets": "Title and Content",
    "two_column": "Two Content",
    "section_header": "Section Header",
    "image_caption": "Picture with Caption",
    "quote": "Title Only",
}


def _load_presentation() -> Presentation:
    template_path = Path(get_config().paths.template_potx)
    if template_path.exists():
        return Presentation(str(template_path))
    logger.warning("template_potx_not_found_using_builtin_default", path=str(template_path))
    return Presentation()  # python-pptx's built-in default template


def _find_layout(prs: Presentation, layout_type: LayoutType) -> SlideLayout:
    target_name = _DEFAULT_LAYOUT_NAME_BY_TYPE[layout_type]
    for layout in prs.slide_layouts:
        if layout.name.strip().lower() == target_name.lower():
            return layout
    logger.warning(
        "no_matching_slide_layout_found_using_first_layout",
        layout_type=layout_type,
        wanted=target_name,
    )
    return prs.slide_layouts[0]


def _set_title(slide: Slide, title: str) -> None:
    if slide.shapes.title is not None:
        slide.shapes.title.text_frame.text = title


def _set_bullets(slide: Slide, bullets: list[str]) -> None:
    body_placeholder = None
    for shape in slide.placeholders:
        if shape.placeholder_format.idx != 0 and shape.has_text_frame:
            body_placeholder = shape
            break
    if body_placeholder is None or not bullets:
        return

    text_frame = body_placeholder.text_frame
    text_frame.clear()
    for i, bullet in enumerate(bullets):
        paragraph = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0


def _set_notes(slide: Slide, notes: str) -> None:
    if notes:
        slide.notes_slide.notes_text_frame.text = notes


def build_pptx(
    slides: list[tuple[SlideContent, str]],
    output_path: str | Path,
    mirror_shape_names: set[str] | None = None,
) -> Path:
    """`slides` is a list of (content, language) pairs so mixed-direction
    decks (some slides English, some Arabic, etc.) are supported natively --
    RTL treatment is decided per slide from its own language."""
    cfg = get_config()
    prs = _load_presentation()

    for content, lang in slides:
        layout = _find_layout(prs, content.layout_type)
        slide = prs.slides.add_slide(layout)

        _set_title(slide, content.title)
        _set_bullets(slide, content.bullets)
        _set_notes(slide, content.notes)

        if cfg.languages.is_rtl(lang):
            apply_rtl_to_slide(slide, prs.slide_width, mirror_shape_names)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    logger.info("pptx_built", output=str(output_path), num_slides=len(slides))
    return output_path
