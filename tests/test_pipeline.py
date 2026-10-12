"""Fast smoke tests: parsers, image proxies, stage-1 end to end on local images,
stage-2 conversion, and stage-3 training on synthetic data. No network access needed."""

import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import extract_and_explain as stage1  # noqa: E402
import prepare_model_inputs as stage2  # noqa: E402
import train_ensemble as stage3  # noqa: E402


@pytest.mark.parametrize("raw,expected", [("1/250", 0.004), ("1/250s", 0.004), ("0.5", 0.5), ("2 sec", 2.0)])
def test_parse_exposure_time(raw, expected):
    assert stage1.parse_exposure_time(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [("f/2.8", 2.8), ("f2.8", 2.8), ("2.8", 2.8)])
def test_parse_fnumber(raw, expected):
    assert stage1.parse_fnumber(raw) == pytest.approx(expected)


def test_parse_numeric_and_missing():
    assert stage1.parse_generic_numeric("50mm") == 50.0
    assert stage1.parse_generic_numeric("ISO 100") == 100.0
    assert math.isnan(stage1.parse_exposure_time(""))
    assert math.isnan(stage1.parse_exposure_time("1/0"))


def test_image_proxies_behave_as_expected():
    grey = np.full((32, 32, 3), 128, np.uint8)
    red = np.zeros((32, 32, 3), np.uint8); red[..., 2] = 255
    checker = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    checker = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)

    sat_grey, con_grey = stage1.image_proxies(grey)
    sat_red, _ = stage1.image_proxies(red)
    _, con_checker = stage1.image_proxies(checker)

    assert sat_grey == pytest.approx(0.0, abs=1e-6)
    assert sat_red == pytest.approx(1.0, abs=1e-6)
    assert con_grey == pytest.approx(0.0, abs=1e-6)
    assert con_checker == pytest.approx(0.5, abs=1e-3)


def _fake_unsplash(tmp_path, n=60):
    """Write n random local images and a photos table in the Unsplash Lite format."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        img = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
        img = (img * rng.uniform(0.2, 1.0)).astype(np.uint8)
        path = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(path), img)
        rows.append({
            "photo_id": f"p{i}", "photo_image_url": str(path),
            "exif_exposure_time": f"1/{rng.choice([60, 125, 250, 500])}",
            "exif_aperture_value": str(rng.choice([1.8, 2.8, 4.0, 8.0])),
            "exif_iso": str(rng.choice([100, 400, 1600])),
            "exif_focal_length": f"{rng.choice([24, 50, 85])}mm",
            "exif_camera_make": rng.choice(["Canon", "SONY", None]),
            "stats_downloads": int(rng.integers(10, 5000)),
        })
    tsv = tmp_path / "photos.tsv000"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    return tsv


def test_stage1_and_stage2_end_to_end(tmp_path):
    tsv = _fake_unsplash(tmp_path)
    df = stage1.load_photos(tsv)
    df = stage1.add_proxies(df, str(tmp_path / "cache.json"))
    records = stage1.fit_and_explain(df)

    assert len(records) == 60
    r = records[0]
    assert set(r) == {"photo_id", "photo_url", "raw_features", "scores", "shap_values"}
    # SHAP values + expected value reconstruct the prediction only approximately
    # (averaged explainers), but every feature must have an attribution.
    assert set(r["shap_values"]) == set(r["raw_features"])

    exif, proxy = stage2.to_frames(json.loads(json.dumps(records)))
    assert exif["aesthetic_score"].nunique() > 1, "real target must be carried through"
    assert not exif[["shutter_speed", "aperture", "iso", "focal_length"]].isna().any().any()
    assert list(proxy.columns) == ["image_id", "saturation_proxy", "contrast_proxy"]


def test_stage3_demo_runs(tmp_path):
    exif_csv, proxy_csv = stage3.generate_demo_data(n=400)
    _, metrics = stage3.run_pipeline(exif_csv, proxy_csv, output_dir=str(tmp_path))
    for model in ("OLS", "RF", "XGBoost", "Ensemble"):
        assert set(metrics[model]) == {"MAE", "RMSE", "PLCC", "SRCC"}
    # the synthetic target is learnable, so the ensemble should correlate strongly
    assert metrics["Ensemble"]["SRCC"] > 0.5
    for name in ("ensemble_model.pkl", "splits.pkl", "predictions.csv", "metrics.json"):
        assert (tmp_path / name).exists()
