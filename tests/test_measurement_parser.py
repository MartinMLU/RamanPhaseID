import numpy as np
import pytest

from raman_core import parse_measurement


def assert_spectrum(text: str, expected_x: list[float], expected_y: list[float]) -> None:
    x, y = parse_measurement(text)
    assert x.dtype.kind == "f"
    assert y.dtype.kind == "f"
    np.testing.assert_allclose(x, expected_x)
    np.testing.assert_allclose(y, expected_y)


def test_uncommented_header_is_coerced_and_does_not_leak_strings() -> None:
    assert_spectrum(
        "Raman shift,Intensity\n100,1.5\n101,2.5\n",
        [100.0, 101.0],
        [1.5, 2.5],
    )


def test_three_column_csv_uses_exactly_the_first_two_physical_columns() -> None:
    assert_spectrum(
        "shift,intensity,processed\n100,10,910\n101,20,920\n",
        [100.0, 101.0],
        [10.0, 20.0],
    )


@pytest.mark.parametrize(
    ("text", "expected_y"),
    [
        ("x,y,ignored\n100,1,91\n101,2,92\n", [1.0, 2.0]),
        ("x\ty\tignored\n100\t3\t93\n101\t4\t94\n", [3.0, 4.0]),
        ("x;y;ignored\n100;5;95\n101;6;96\n", [5.0, 6.0]),
        ("x y ignored\n100 7 97\n101 8 98\n", [7.0, 8.0]),
    ],
)
def test_supported_delimiters_are_detected_deterministically(
    text: str,
    expected_y: list[float],
) -> None:
    assert_spectrum(text, [100.0, 101.0], expected_y)


def test_decimal_commas_work_when_the_field_delimiter_is_semicolon() -> None:
    assert_spectrum(
        "shift;intensity\n100,5;1,25\n101,5;2,75\n",
        [100.5, 101.5],
        [1.25, 2.75],
    )


def test_descending_axis_is_validated_and_returned_in_ascending_order() -> None:
    assert_spectrum(
        "shift\tintensity\n102\t30\n101\t20\n100\t10\n",
        [100.0, 101.0, 102.0],
        [10.0, 20.0, 30.0],
    )


def test_duplicate_shifts_are_averaged_deterministically() -> None:
    assert_spectrum(
        "100,2\n100,4\n101,8\n",
        [100.0, 101.0],
        [3.0, 8.0],
    )


def test_non_monotonic_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be monotonic"):
        parse_measurement("100,1\n102,2\n101,3\n")


def test_malformed_row_after_data_start_is_rejected_with_line_number() -> None:
    with pytest.raises(ValueError, match=r"line 3"):
        parse_measurement("shift,intensity\n100,1\nbroken,row\n101,2\n")


def test_leading_free_form_metadata_is_allowed() -> None:
    assert_spectrum(
        "Instrument model Alpha\nOperator: Example\nshift intensity\n100 1\n101 2\n",
        [100.0, 101.0],
        [1.0, 2.0],
    )


def test_two_distinct_shifts_are_required_after_duplicate_consolidation() -> None:
    with pytest.raises(ValueError, match="two distinct Raman shifts"):
        parse_measurement("100,1\n100,2\n")
