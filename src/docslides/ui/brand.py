"""Codex Sentinel branding for the Gradio UI: the logo (assets/brand/codex-sentinel/)
and a restrained, law-office theme -- paper-white panels, deep navy for actions and a
little muted gold for marks of state.

The header uses the flat, single-ink version of the logo (section-lockup-line-mono.svg),
coloured by CSS variables so it follows the light and dark modes; the original neon
artwork in the same folder is left as it was.
"""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

BRAND_DIR = Path(__file__).resolve().parents[3] / "assets" / "brand" / "codex-sentinel"
LOCKUP_SVG = BRAND_DIR / "svg" / "section-lockup-line-mono.svg"
FAVICON_SVG = BRAND_DIR / "svg" / "section-favicon-mono.svg"
FAVICON_ICO = BRAND_DIR / "favicon-mono.ico"

# Light mode.
PAGE = "#F4F5F7"
PANEL = "#FFFFFF"
PANEL_RAISED = "#F8F9FB"
LINE = "#D9DEE5"
TEXT = "#1B2430"
TEXT_MUTED = "#5B6675"
NAVY = "#1F3A5F"  # buttons, links, the selected tab, the logo's ink
NAVY_HOVER = "#2B4C78"
GOLD = "#A8874A"  # sparingly: the selected tab's rule, the logo's accent
ERROR = "#9B2C2C"

# Dark mode: slate rather than black, the same roles.
PAGE_DARK = "#14181F"
PANEL_DARK = "#1B212A"
PANEL_RAISED_DARK = "#222A35"
LINE_DARK = "#2F3846"
TEXT_DARK = "#E6EBF2"
TEXT_MUTED_DARK = "#9AA5B4"
NAVY_DARK = "#3A5F8F"
NAVY_HOVER_DARK = "#46699A"
LINK_DARK = "#9DB8DC"
GOLD_DARK = "#C9A86A"
ERROR_DARK = "#E08A8A"

_navy = gr.themes.Color(
    c50="#EEF2F7", c100="#D6DFEA", c200="#B0C1D6", c300="#879FBE", c400="#5F7EA4",
    c500="#3A5F8F", c600=NAVY_HOVER, c700=NAVY, c800="#182E4B", c900="#112137", c950="#0A1524",
    name="sentinel_navy",
)
_gold = gr.themes.Color(
    c50="#FAF6EE", c100="#F1E8D4", c200="#E3D0A9", c300="#D4B87F", c400=GOLD_DARK,
    c500=GOLD, c600="#8C6F3B", c700="#6F582F", c800="#534223", c900="#382C18", c950="#211A0E",
    name="sentinel_gold",
)


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=_navy,
        secondary_hue=_gold,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), gr.themes.GoogleFont("IBM Plex Sans Hebrew"),
              "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill=PAGE,
        body_background_fill_dark=PAGE_DARK,
        body_text_color=TEXT,
        body_text_color_dark=TEXT_DARK,
        body_text_color_subdued=TEXT_MUTED,
        body_text_color_subdued_dark=TEXT_MUTED_DARK,
        background_fill_primary=PANEL,
        background_fill_primary_dark=PANEL_DARK,
        background_fill_secondary=PANEL_RAISED,
        background_fill_secondary_dark=PANEL_RAISED_DARK,
        block_background_fill=PANEL,
        block_background_fill_dark=PANEL_DARK,
        block_border_color=LINE,
        block_border_color_dark=LINE_DARK,
        block_label_background_fill=PANEL_RAISED,
        block_label_background_fill_dark=PANEL_RAISED_DARK,
        block_label_text_color=TEXT_MUTED,
        block_label_text_color_dark=TEXT_MUTED_DARK,
        block_title_text_color=TEXT,
        block_title_text_color_dark=TEXT_DARK,
        block_shadow="none",
        block_shadow_dark="none",
        panel_background_fill=PANEL_RAISED,
        panel_background_fill_dark=PANEL_RAISED_DARK,
        panel_border_color=LINE,
        panel_border_color_dark=LINE_DARK,
        border_color_primary=LINE,
        border_color_primary_dark=LINE_DARK,
        border_color_accent=NAVY,
        border_color_accent_dark=LINK_DARK,
        color_accent=NAVY,
        color_accent_soft="#EEF2F7",
        color_accent_soft_dark="rgba(157, 184, 220, .12)",
        input_background_fill=PANEL,
        input_background_fill_dark=PAGE_DARK,
        input_border_color=LINE,
        input_border_color_dark=LINE_DARK,
        input_border_color_focus=NAVY,
        input_border_color_focus_dark=LINK_DARK,
        input_shadow="none",
        input_shadow_dark="none",
        input_shadow_focus="0 0 0 2px rgba(31, 58, 95, .15)",
        input_shadow_focus_dark="0 0 0 2px rgba(157, 184, 220, .2)",
        input_placeholder_color="#8A94A3",
        input_placeholder_color_dark="#6B7686",
        link_text_color=NAVY,
        link_text_color_dark=LINK_DARK,
        link_text_color_hover=NAVY_HOVER,
        link_text_color_hover_dark=TEXT_DARK,
        link_text_color_visited=NAVY,
        link_text_color_visited_dark=LINK_DARK,
        button_primary_background_fill=NAVY,
        button_primary_background_fill_dark=NAVY_DARK,
        button_primary_background_fill_hover=NAVY_HOVER,
        button_primary_background_fill_hover_dark=NAVY_HOVER_DARK,
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_primary_border_color=NAVY,
        button_primary_border_color_dark=NAVY_DARK,
        button_secondary_background_fill=PANEL,
        button_secondary_background_fill_dark=PANEL_RAISED_DARK,
        button_secondary_background_fill_hover=PANEL_RAISED,
        button_secondary_background_fill_hover_dark=LINE_DARK,
        button_secondary_text_color=TEXT,
        button_secondary_text_color_dark=TEXT_DARK,
        button_secondary_border_color=LINE,
        button_secondary_border_color_dark=LINE_DARK,
        checkbox_background_color_selected=NAVY,
        checkbox_background_color_selected_dark=NAVY_DARK,
        checkbox_border_color_selected=NAVY,
        checkbox_border_color_selected_dark=NAVY_DARK,
        slider_color=NAVY,
        slider_color_dark=LINK_DARK,
        loader_color=NAVY,
        loader_color_dark=LINK_DARK,
        error_background_fill="#FBF1F1",
        error_background_fill_dark="rgba(224, 138, 138, .08)",
        error_border_color=ERROR,
        error_border_color_dark=ERROR_DARK,
        error_text_color=ERROR,
        error_text_color_dark=ERROR_DARK,
        code_background_fill=PANEL_RAISED,
        code_background_fill_dark=PAGE_DARK,
        table_border_color=LINE,
        table_border_color_dark=LINE_DARK,
        table_even_background_fill=PANEL,
        table_even_background_fill_dark=PANEL_DARK,
        table_odd_background_fill=PANEL_RAISED,
        table_odd_background_fill_dark=PANEL_RAISED_DARK,
        shadow_drop="none",
        shadow_drop_lg="0 1px 3px rgba(16, 24, 40, .08)",
    )
    return theme


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
