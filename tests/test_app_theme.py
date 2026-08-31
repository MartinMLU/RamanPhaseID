from __future__ import annotations

import RamanPhaseID_0p99beta as app


def _captured_theme_css(monkeypatch, theme: str) -> str:
    captured: dict[str, object] = {}

    def capture_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr(app.st, "markdown", capture_markdown)
    app._apply_app_theme(theme)
    assert captured["unsafe_allow_html"] is True
    return str(captured["body"])


def test_light_app_theme_forces_widget_text_contrast(monkeypatch) -> None:
    css = _captured_theme_css(monkeypatch, "light")

    assert 'color-scheme: light' in css
    assert '[data-testid="stWidgetLabel"]' in css
    assert '[data-testid="stSidebar"] :is(' in css
    assert '[data-testid="stSliderThumbValue"]' in css
    assert '[data-testid="stSliderTickBar"]' in css
    assert '[data-baseweb="select"] *' in css
    assert '[data-testid="stNumberInput"] button' in css
    assert '[data-testid="stSidebarCollapseButton"]' in css
    assert '[data-testid="stMarkdownContainer"] code' in css
    assert 'color: #1F2933 !important' in css
    assert '-webkit-text-fill-color: #1F2933 !important' in css
    assert 'color: #44515E !important' in css


def test_dark_app_theme_keeps_the_same_component_coverage(monkeypatch) -> None:
    css = _captured_theme_css(monkeypatch, "dark")

    assert 'color-scheme: dark' in css
    assert '[data-testid="stWidgetLabel"]' in css
    assert '[data-baseweb="popover"] [role="option"]' in css
    assert '[data-testid="stFileUploaderFile"]' in css
    assert 'color: #E6EDF3 !important' in css
    assert 'color: #B7C0CA !important' in css


def test_dark_help_icon_separates_circle_from_question_mark(monkeypatch) -> None:
    dark_css = _captured_theme_css(monkeypatch, "dark")
    light_css = _captured_theme_css(monkeypatch, "light")

    circle_selector = '[data-testid="stTooltipIcon"] svg.icon circle'
    question_selector = '[data-testid="stTooltipIcon"] svg.icon path'
    assert circle_selector in dark_css
    assert question_selector in dark_css
    assert 'fill: #2A333D !important' in dark_css
    assert 'stroke: #667382 !important' in dark_css
    assert 'stroke: #F2F5F8 !important' in dark_css
    assert circle_selector not in light_css
    assert question_selector not in light_css


def test_primary_button_text_remains_white_in_both_app_themes(monkeypatch) -> None:
    for theme in ("light", "dark"):
        css = _captured_theme_css(monkeypatch, theme)
        assert 'button[data-testid="stBaseButton-primary"] *' in css
        assert 'color: #FFFFFF !important' in css
