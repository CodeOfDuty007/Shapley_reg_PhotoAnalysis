"""
Stage 1: feature extraction, multicollinearity check, ensemble fit and SHAP attribution.

Reads the Unsplash Lite photos table (`photos.csv000`, tab-separated), cleans the EXIF fields, downloads each
photo once to compute two post-processing proxies (saturation and contrast), fits a
Random Forest + XGBoost voting ensemble on the full table, and writes one JSON record
per photo with its raw features, target, prediction and per-feature SHAP values.

Note: this stage fits and predicts on the same rows, so its predictions are in-sample
and are meant for attribution, not for measuring accuracy. Use `train_ensemble.py`
for held-out evaluation.

Usage:
    python pipeline/extract_and_explain.py --dataset photos.csv000 --nrows 500
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_FEATURES = ["ExposureTime_Num", "FNumber", "ISO", "FocalLength",
                 "Saturation_Proxy", "Contrast_Proxy"]


# ── 1. EXIF parsers ──────────────────────────────────────────────────────────

def parse_exposure_time(value):
    """'1/250', '1/250s', '0.004 sec' -> seconds as float."""
    if pd.isna(value) or value == "":
        return np.nan
    try:
        value = re.sub(r"[a-z]", "", str(value).strip().lower())
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return np.nan


def parse_fnumber(value):
    """'f/2.8', 'f2.8', '2.8f' -> 2.8"""
    if pd.isna(value) or value == "":
        return np.nan
    try:
        return float(str(value).strip().lower().replace("f/", "").replace("f", ""))
    except ValueError:
        return np.nan


def parse_generic_numeric(value):
    """'50mm', 'ISO 100' -> 50.0, 100.0"""
    if pd.isna(value) or value == "":
        return np.nan
    match = re.search(r"([0-9]+\.?[0-9]*)", str(value).strip())
    return float(match.group(1)) if match else np.nan


# ── 2. Image proxies ─────────────────────────────────────────────────────────

def image_proxies(img_bgr):
    """Mean HSV saturation and grayscale standard deviation, both on a 0-1 scale."""
    img = img_bgr.astype(np.float32) / 255.0
    saturation = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]))
    contrast = float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return saturation, contrast


def load_image(source, timeout=10):
    """Load a colour image from a URL or a local path (always 3-channel BGR)."""
    if os.path.exists(str(source)):
        return cv2.imread(str(source), cv2.IMREAD_COLOR)
    with urllib.request.urlopen(source, timeout=timeout) as resp:
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_image_proxies(source, retries=3):
    """Return (saturation, contrast) for one image, retrying on network errors."""
    for attempt in range(retries):
        try:
            img = load_image(source)
            if img is None:
                return None, None
            return image_proxies(img)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Network error {attempt + 1}/{retries} for {source}: {e}")
            time.sleep(2)
        except Exception as e:  # corrupt file, decode error, etc.
            print(f"Processing error for {source}: {e}")
            break
    return None, None


def calculate_vif(X_df):
    """Variance Inflation Factor per feature (VIF > 5 suggests multicollinearity)."""
    return pd.DataFrame({
        "Feature": X_df.columns,
        "VIF": [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])],
    })


# ── 3. Pipeline ──────────────────────────────────────────────────────────────

def load_photos(dataset_path, nrows=None):
    df = pd.read_csv(dataset_path, sep="\t", nrows=nrows)
    df = df.rename(columns={
        "exif_exposure_time": "ExposureTime",
        "exif_aperture_value": "FNumber",
        "exif_iso": "ISO",
        "exif_focal_length": "FocalLength",
        "exif_camera_make": "CameraMake",
    })
    df["ExposureTime_Num"] = df["ExposureTime"].apply(parse_exposure_time)
    df["FNumber"] = df["FNumber"].apply(parse_fnumber)
    df["ISO"] = df["ISO"].apply(parse_generic_numeric)
    df["FocalLength"] = df["FocalLength"].apply(parse_generic_numeric)
    # Target: log-scaled download count, used as a popularity / aesthetic proxy
    df["AestheticScore"] = np.log1p(df["stats_downloads"])
    return df


def add_proxies(df, cache_file):
    """Compute image proxies with a resumable JSON cache keyed by photo_id."""
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached images.")

    for i, row in enumerate(df.itertuples(index=False)):
        pid = str(row.photo_id)
        if pid in cache:
            continue
        print(f"Processing {i + 1}/{len(df)}: {pid} ...", end=" ", flush=True)
        sat, con = extract_image_proxies(row.photo_image_url)
        ok = sat is not None
        cache[pid] = {"Saturation_Proxy": sat if ok else None,
                      "Contrast_Proxy": con if ok else None}
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        print("ok" if ok else "skipped")

    for col in ("Saturation_Proxy", "Contrast_Proxy"):
        df[col] = df["photo_id"].map(lambda p: (cache.get(str(p)) or {}).get(col)).astype(float)
    return df


def fit_and_explain(df, chunk_size=1000):
    df = df.dropna(subset=["Saturation_Proxy", "Contrast_Proxy"]).copy()
    df["CameraMake"] = df["CameraMake"].fillna("Unknown")
    df = pd.get_dummies(df, columns=["CameraMake"], drop_first=True, dtype=float)

    features = BASE_FEATURES + [c for c in df.columns if c.startswith("CameraMake_")]
    X_raw, y = df[features], df["AestheticScore"]

    X_imp = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X_raw),
                         columns=features, index=X_raw.index)
    X = pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=features, index=X_raw.index)

    print("\n--- Variance Inflation Factor (VIF > 5 = problematic multicollinearity) ---")
    print(calculate_vif(X[BASE_FEATURES]).to_string(index=False))

    ensemble = VotingRegressor(estimators=[
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("xgb", xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1,
                                 objective="reg:squarederror")),
    ])
    ensemble.fit(X.values, y)
    predictions = ensemble.predict(X.values)

    # TreeExplainer on each member, averaged (VotingRegressor is an equal-weight mean)
    explainers = [shap.TreeExplainer(m) for m in ensemble.estimators_]
    parts = []
    for start in range(0, len(X), chunk_size):
        chunk = X.values[start:start + chunk_size]
        print(f" -> SHAP rows {start}-{start + len(chunk)} of {len(X)}")
        parts.append(np.mean([e.shap_values(chunk, check_additivity=False) for e in explainers], axis=0))
    shap_values = np.vstack(parts)

    records = []
    for idx, (orig_idx, row) in enumerate(df.iterrows()):
        records.append({
            "photo_id": str(row["photo_id"]),
            "photo_url": str(row["photo_image_url"]),
            "raw_features": {f: (None if pd.isna(X_raw.loc[orig_idx, f]) else float(X_raw.loc[orig_idx, f]))
                             for f in features},
            "scores": {"actual_proxy_score": float(y.loc[orig_idx]),
                       "predicted_score": float(predictions[idx])},
            "shap_values": {f: float(shap_values[idx][i]) for i, f in enumerate(features)},
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="photos.csv000", help="Unsplash Lite photos table (tab-separated)")
    ap.add_argument("--nrows", type=int, default=None, help="Only process the first N photos")
    ap.add_argument("--cache", default="proxy_cache.json", help="Resumable proxy cache")
    ap.add_argument("--output", default="final_results.json", help="Per-photo JSON output")
    args = ap.parse_args()

    df = load_photos(args.dataset, args.nrows)
    print(f"Loaded {len(df)} photos from {args.dataset}")
    df = add_proxies(df, args.cache)
    records = fit_and_explain(df)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
