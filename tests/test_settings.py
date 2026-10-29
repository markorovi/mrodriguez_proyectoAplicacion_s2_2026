import json

import pytest
from pydantic import ValidationError

from config import (
    AppSettings,
    CameraIntrinsics,
    DepthModelVariant,
    DepthSettings,
    ImageSettings,
    load_settings,
)


def test_defaults_match_minimum_resolution_requirement():
    settings = AppSettings()
    assert (settings.image.min_width, settings.image.min_height) == (640, 480)


def test_default_depth_model_is_metric_indoor_small():
    depth = DepthSettings()
    assert depth.model_id == "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    assert depth.is_metric


@pytest.mark.parametrize(
    ("variant", "suffix"),
    [(DepthModelVariant.BASE, "Base-hf"), (DepthModelVariant.LARGE, "Large-hf")],
)
def test_variant_selects_checkpoint(variant, suffix):
    assert DepthSettings(variant=variant).model_id.endswith(suffix)


def test_model_id_override_takes_precedence():
    assert DepthSettings(model_id_override="local/model").model_id == "local/model"


def test_intrinsics_scale_with_resolution():
    k = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
    scaled = k.scaled_to(1280, 960)
    assert (scaled.fx, scaled.fy, scaled.cx, scaled.cy) == (1000, 1000, 640, 480)
    assert (scaled.reference_width, scaled.reference_height) == (1280, 960)


def test_intrinsics_reject_non_positive_focal_length():
    with pytest.raises(ValidationError):
        CameraIntrinsics(fx=0)


def test_settings_are_immutable():
    with pytest.raises(ValidationError):
        AppSettings().image.min_width = 10


def test_extensions_are_normalized():
    assert ImageSettings(allowed_extensions=("PNG", ".JpG")).allowed_extensions == (".png", ".jpg")


def test_load_settings_from_json_keeps_unspecified_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"camera": {"fx": 600.0}, "depth": {"variant": "base"}}), "utf-8")

    settings = load_settings(path)

    assert settings.camera.fx == 600.0
    assert settings.camera.fy == CameraIntrinsics().fy
    assert settings.depth.variant is DepthModelVariant.BASE
    assert settings.image == ImageSettings()


def test_load_settings_without_path_returns_defaults():
    assert load_settings() == AppSettings()
