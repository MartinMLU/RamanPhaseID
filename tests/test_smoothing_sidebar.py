from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


METHOD_LABEL = "Measurement denoising method"
MAGNIFICATION_LABEL = "Processing-difference line magnification"
METHOD_SPECIFIC_SLIDERS = {
    "Window length (points; odd)",
    "Polynomial order",
    "AI safeguard: maximum correction (× estimated noise σ)",
}


def _method_selectbox(app: AppTest):
    return next(box for box in app.sidebar.selectbox if box.label == METHOD_LABEL)


def _visible_method_sliders(app: AppTest) -> set[str]:
    return {slider.label for slider in app.sidebar.slider} & METHOD_SPECIFIC_SLIDERS


def _magnification_sliders(app: AppTest):
    return [
        slider
        for slider in app.sidebar.slider
        if slider.label == MAGNIFICATION_LABEL
    ]


def test_sidebar_renders_only_controls_for_the_selected_denoising_method() -> None:
    # Call the GUI entry point directly so pytest's own command-line arguments
    # cannot make RamanPhaseID select its intentionally empty CLI placeholder.
    app = AppTest.from_string(
        "import RamanPhaseID_0p99beta as app\napp._run_streamlit()\n",
        default_timeout=20,
    ).run()
    visible = _visible_method_sliders(app)
    assert visible == {
        "Window length (points; odd)",
        "Polynomial order",
    }, {
        "all_sliders": [slider.label for slider in app.sidebar.slider],
        "selectboxes": [(box.label, box.value) for box in app.sidebar.selectbox],
        "exceptions": [exception.value for exception in app.exception],
    }
    assert len(_magnification_sliders(app)) == 1
    assert _magnification_sliders(app)[0].value == 2.0
    window_slider = next(
        slider
        for slider in app.sidebar.slider
        if slider.label == "Window length (points; odd)"
    )
    assert window_slider.value == 5
    lambda_slider = next(
        slider for slider in app.sidebar.slider if slider.label == "λ (10^x)"
    )
    assert lambda_slider.value == 5
    calibrant_input = next(
        text_input
        for text_input in app.sidebar.text_input
        if text_input.label == "Calibrant / reference peak (optional)"
    )
    assert calibrant_input.placeholder == "e.g. silicon 520.5 cm⁻¹"
    assert all(
        expander.label != "Precompute storage and recovery"
        for expander in app.sidebar.expander
    )
    assert all(
        button.key not in {
            "inspect_cache_storage_btn",
            "quarantine_cache_candidates_btn",
        }
        for button in app.sidebar.button
    )

    _method_selectbox(app).set_value("AI-assisted · guarded DeepeR (full range)")
    app.run()
    assert _visible_method_sliders(app) == {
        "AI safeguard: maximum correction (× estimated noise σ)"
    }
    assert len(_magnification_sliders(app)) == 1
    assert _magnification_sliders(app)[0].value == 2.0

    _method_selectbox(app).set_value("None (keep measurement unchanged)")
    app.run()
    assert _visible_method_sliders(app) == set()
    assert _magnification_sliders(app) == []
    assert not list(app.exception)


def test_reload_button_is_between_sidebar_divider_and_appearance() -> None:
    source = Path("RamanPhaseID_0p99beta.py").read_text(encoding="utf-8")
    divider_position = source.index("    st.sidebar.divider()")
    reload_position = source.index('    if st.sidebar.button("Reload DB"):')
    appearance_position = source.index(
        '    with st.sidebar.expander("Appearance", expanded=False):'
    )

    assert divider_position < reload_position < appearance_position
