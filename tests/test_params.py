from __future__ import annotations

import pytest

from vision_lab.params import ParamError, ProcessingParams, parse_params

DETECTORS = ["fasterrcnn"]


def test_empty_form_gives_the_documented_defaults() -> None:
    assert parse_params({}, DETECTORS) == ProcessingParams(detector="fasterrcnn")


def test_valid_values_are_converted_to_their_types() -> None:
    form = {"rotation_angle": "-90", "scale_factor": "0.5", "kernel_size": "9", "flip": "both"}
    parsed = parse_params(form, DETECTORS)
    assert (parsed.rotation_angle, parsed.scale_factor, parsed.kernel_size, parsed.flip) == (
        -90, 0.5, 9, "both",
    )  # fmt: skip


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("kernel_size", "8", "odd"),
        ("kernel_size", "99", "between 3 and 31"),
        ("kernel_size", "abc", "whole number"),
        ("scale_factor", "1000", "between 0.1 and 4.0"),
        ("scale_factor", "nan", "between"),
        ("scale_factor", "inf", "between"),
        ("crop_x", "-5", "between 0 and 10000"),
        ("edge_algorithm", "../../etc/passwd", "must be one of"),
        ("detector", "yolov5su", "must be one of"),
        ("color_space", "<script>", "must be one of"),
    ],
)
def test_bad_values_are_reported_per_field(field: str, value: str, fragment: str) -> None:
    with pytest.raises(ParamError) as caught:
        parse_params({field: value}, DETECTORS)
    assert fragment in caught.value.errors[field]


def test_every_error_is_reported_at_once() -> None:
    with pytest.raises(ParamError) as caught:
        parse_params({"kernel_size": "4", "flip": "sideways", "clip_limit": "x"}, DETECTORS)
    assert set(caught.value.errors) == {"kernel_size", "flip", "clip_limit"}


def test_canny_thresholds_must_be_ordered() -> None:
    with pytest.raises(ParamError) as caught:
        parse_params({"threshold1": "300", "threshold2": "100"}, DETECTORS)
    assert "threshold1" in caught.value.errors
