"""Codex Sentinel branding for the Gradio UI: the logo (assets/brand/codex-sentinel/)
and a theme built from its palette -- near-black navy, a cyan-to-violet edge and a
magenta core.

The logo's wordmark is drawn in light colours for a dark background, so the theme is
dark in both of Gradio's modes: every light-mode value is copied from its dark-mode
counterpart, and a visitor whose system prefers a light scheme still sees the dark UI.
"""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

BRAND_DIR = Path(__file__).resolve().parents[3] / "assets" / "brand" / "codex-sentinel"
LOCKUP_SVG = BRAND_DIR / "svg" / "section-lockup-line.svg"
FAVICON_SVG = BRAND_DIR / "svg" / "section-favicon.svg"
FAVICON_ICO = BRAND_DIR / "favicon.ico"

# The logo's palette.
INK = "#04060C"  # page background
PANEL = "#0A1122"  # the hexagon's fill: blocks and panels
PANEL_RAISED = "#111A30"
LINE = "#22304F"  # the hexagon's outline and circuit traces: borders
CYAN = "#18E4FF"
VIOLET = "#7C5CFF"
CORE = "#FF2D7A"  # the core: used sparingly, for errors and the working glow's heart
TEXT = "#EAF7FF"  # the wordmark's "SENTINEL"
TEXT_MUTED = "#8FA3C4"

_cyan = gr.themes.Color(
    c50="#E6FCFF", c100="#C2F7FF", c200="#8EF1FF", c300="#5AEBFF", c400="#2FE6FF",
    c500=CYAN, c600="#0BBFD8", c700="#0896AB", c800="#076E7E", c900="#054652", c950="#032A31",
    name="sentinel_cyan",
)
_violet = gr.themes.Color(
    c50="#F1EDFF", c100="#E0D8FF", c200="#C4B5FF", c300="#A891FF", c400="#9275FF",
    c500=VIOLET, c600="#6340F0", c700="#4F2FCC", c800="#3D24A0", c900="#2B1A72", c950="#1A1046",
    name="sentinel_violet",
)
_navy = gr.themes.Color(
    c50=TEXT, c100="#C9D6EA", c200="#A9B9D4", c300=TEXT_MUTED, c400="#6A7FA3", c500="#4A5D80",
    c600="#33456A", c700=LINE, c800=PANEL_RAISED, c900=PANEL, c950=INK,
    name="sentinel_navy",
)


def _always_dark(theme: gr.themes.Base) -> gr.themes.Base:
    for name, value in list(vars(theme).items()):
        # A dark value of None means "same as light", so there's nothing to copy.
        if name.endswith("_dark") and value is not None:
            setattr(theme, name[: -len("_dark")], value)
    return theme


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=_cyan,
        secondary_hue=_violet,
        neutral_hue=_navy,
        radius_size=gr.themes.sizes.radius_md,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill_dark=INK,
        body_text_color_dark=TEXT,
        body_text_color_subdued_dark=TEXT_MUTED,
        background_fill_primary_dark=PANEL,
        background_fill_secondary_dark=INK,
        block_background_fill_dark=PANEL,
        block_border_color_dark=LINE,
        block_label_background_fill_dark=PANEL_RAISED,
        block_label_text_color_dark=TEXT_MUTED,
        block_title_text_color_dark=TEXT,
        panel_background_fill_dark=PANEL,
        panel_border_color_dark=LINE,
        border_color_primary_dark=LINE,
        border_color_accent_dark=CYAN,
        color_accent=CYAN,
        color_accent_soft_dark="rgba(24, 228, 255, .12)",
        input_background_fill_dark=INK,
        input_border_color_dark=LINE,
        input_border_color_focus_dark=CYAN,
        input_placeholder_color_dark="#4A5D80",
        link_text_color_dark=CYAN,
        link_text_color_hover_dark=TEXT,
        link_text_color_visited_dark=VIOLET,
        button_primary_background_fill_dark=f"linear-gradient(135deg, {CYAN}, {VIOLET})",
        button_primary_background_fill_hover_dark=f"linear-gradient(135deg, #5AEBFF, #9275FF)",
        button_primary_text_color_dark=INK,
        button_primary_border_color_dark="transparent",
        button_secondary_background_fill_dark=PANEL_RAISED,
        button_secondary_background_fill_hover_dark=LINE,
        button_secondary_text_color_dark=TEXT,
        button_secondary_border_color_dark=LINE,
        checkbox_background_color_selected_dark=CYAN,
        checkbox_border_color_selected_dark=CYAN,
        slider_color_dark=CYAN,
        loader_color_dark=CYAN,
        error_background_fill_dark="rgba(255, 45, 122, .1)",
        error_border_color_dark=CORE,
        error_text_color_dark="#FF8FB5",
        code_background_fill_dark=INK,
        table_border_color_dark=LINE,
        table_even_background_fill_dark=PANEL,
        table_odd_background_fill_dark=INK,
        shadow_drop="0 1px 2px rgba(0, 0, 0, .4)",
    )
    return _always_dark(theme)


def header_html(title: str) -> str:
    """The page header: the line lockup, with `title` as its accessible name and as a
    plain-text fallback when the logo file isn't there (a checkout without assets/)."""
    try:
        svg = LOCKUP_SVG.read_text(encoding="utf-8")
    except OSError:
        return f'<div class="brand-header"><h1>{title}</h1></div>'
    svg = svg.replace('aria-label="Codex SENTINEL"', f'aria-label="{title}"', 1)
    return (
        '<div class="brand-header">'
        f'<div class="brand-logo">{svg}</div>'
        '<div class="brand-byline">By Ibrahim Zananiri</div>'
        "</div>"
    )


def favicon_path() -> str | None:
    return str(FAVICON_ICO) if FAVICON_ICO.is_file() else None


def favicon_head() -> str:
    """A <link rel=icon> for the SVG favicon, inlined so it works wherever the app is
    mounted (favicon_path only covers the .ico)."""
    try:
        data = base64.b64encode(FAVICON_SVG.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{data}">\n'
