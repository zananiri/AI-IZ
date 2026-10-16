# Slide template

Place your organization's `.potx` PowerPoint template here and point
`paths.template_potx` in `config/config.yaml` at it (default:
`./assets/templates/default.potx`).

Requirements for `slides/pptx_builder.py`'s layout matching to work without
code changes:

- Name your slide layouts using standard PowerPoint conventions so they
  match the `layout_type` values the LLM assigns per slide:
  - `Title and Content` -> `title_bullets`
  - `Two Content` -> `two_column`
  - `Section Header` -> `section_header`
  - `Picture with Caption` -> `image_caption`
  - `Title Only` -> `quote`
- If any asymmetric layout elements (title bars, footer icons/logos anchored
  to one side) should mirror position on RTL (Arabic/Hebrew) slides, name
  those shapes and pass their names as `mirror_shape_names` to
  `build_pptx(...)`.

If no template is found at the configured path, the builder falls back to
python-pptx's built-in default template, which already uses these standard
layout names.
