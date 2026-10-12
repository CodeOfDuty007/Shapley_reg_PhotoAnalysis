"""
Reproduce the held-out evaluation from the shipped feature table, with no downloads.

`data/pipeline_output.csv` holds the features and targets already extracted for
24,996 Unsplash Lite photos. This script:

  1. splits it into the EXIF / proxy CSVs that `train_ensemble.py` expects,
  2. runs the full held-out training + evaluation (OLS, RF, XGBoost, ensemble),
  3. computes SHAP values for the tuned models on the held-out test set,
  4. writes metrics and figures to --outdir.

Camera-make one-hot columns are not in the shipped CSV, so this run uses the six
continuous features only (shutter speed, aperture, ISO, focal length, saturation,
contrast).

Usage:
    python pipeline/evaluate_holdout.py                 # full run
    python pipeline/evaluate_holdout.py --sample 5000   # quicker run on a subset
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_ensemble  # noqa: E402

FEATURE_LABELS = {
    "shutter_speed": "Shutter speed",
    "aperture": "Aperture (f-number)",
    "iso": "ISO",
    "focal_length": "Focal length",
    "saturation_proxy": "Saturation proxy",
    "contrast_proxy": "Contrast proxy",
}


def split_inputs(csv_path, outdir, sample=None, seed=42):
    df = pd.read_csv(csv_path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed)
    df = df.rename(columns={"photo_id": "image_id", "exposure_time": "shutter_speed",
                            "fnumber": "aperture", "actual_score": "aesthetic_score"})
    for col in ["shutter_speed", "aperture", "iso", "focal_length"]:
        df[col] = df[col].fillna(df[col].mean())
    exif = df[["image_id", "shutter_speed", "aperture", "iso", "focal_length", "aesthetic_score"]]
    proxy = df[["image_id", "saturation_proxy", "contrast_proxy"]]
    exif_p, proxy_p = os.path.join(outdir, "exif_clean.csv"), os.path.join(outdir, "proxy_clean.csv")
    exif.to_csv(exif_p, index=False)
    proxy.to_csv(proxy_p, index=False)
    return exif_p, proxy_p, len(df)


def shap_on_test(bundle, splits):
    X_test = splits["X_test"]
    w_rf, w_xgb = bundle["ensemble_weights"]
    sv_rf = shap.TreeExplainer(bundle["rf_model"]).shap_values(X_test, check_additivity=False)
    sv_xgb = shap.TreeExplainer(bundle["xgb_model"]).shap_values(X_test, check_additivity=False)
    sv = w_rf * sv_rf + w_xgb * sv_xgb  # SHAP is linear in a weighted model average
    return pd.Series(np.abs(sv).mean(axis=0), index=X_test.columns).sort_values(ascending=False), sv


def plot_metrics(metrics, path):
    models = ["OLS", "RF", "XGBoost", "Ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key in zip(axes, ["MAE", "SRCC"]):
        vals = [metrics[m][key] for m in models]
        ax.bar(models, vals, color=["#9aa5b1", "#4c78a8", "#4c78a8", "#2f5d8a"])
        ax.set_title(f"Test {key}" + (" (lower is better)" if key == "MAE" else " (higher is better)"))
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap(importance, path):
    imp = importance[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh([FEATURE_LABELS.get(k, k) for k in imp.index], imp.values, color="#2f5d8a")
    ax.set_xlabel("Mean |SHAP value| on held-out test set (log-download units)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(pred_csv, path):
    df = pd.read_csv(pred_csv)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(df["y_true"], df["y_pred_ens"], s=4, alpha=0.35, color="#2f5d8a")
    lo, hi = df["y_true"].min(), df["y_true"].max()
    ax.plot([lo, hi], [lo, hi], color="#9aa5b1", lw=1)
    ax.set_xlabel("Actual log(1 + downloads)")
    ax.set_ylabel("Ensemble prediction")
    ax.set_title("Held-out test set")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/pipeline_output.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sample", type=int, default=None, help="Use a random subset of N photos")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    work = os.path.join(args.outdir, "_work")
    os.makedirs(work, exist_ok=True)

    exif_p, proxy_p, n = split_inputs(args.data, work, args.sample)
    print(f"Prepared {n} photos for held-out evaluation")

    bundle, metrics = train_ensemble.run_pipeline(exif_p, proxy_p, output_dir=work)
    import pickle
    with open(os.path.join(work, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    importance, _ = shap_on_test(bundle, splits)
    print("\nMean |SHAP| on the test set:")
    print(importance.round(4).to_string())

    # Baseline: always predict the training mean
    y_tr, y_te = splits["y_train"], splits["y_test"]
    mean_mae = float(np.mean(np.abs(y_te - y_tr.mean())))

    summary = {
        "n_photos": n,
        "n_train": len(y_tr), "n_val": len(splits["y_val"]), "n_test": len(y_te),
        "test_metrics": metrics,
        "mean_predictor_test_MAE": mean_mae,
        "mean_abs_shap_test": importance.to_dict(),
    }
    with open(os.path.join(args.outdir, "holdout_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    plot_metrics(metrics, os.path.join(args.outdir, "test_metrics.png"))
    plot_shap(importance, os.path.join(args.outdir, "shap_importance.png"))
    plot_pred_vs_actual(os.path.join(work, "predictions.csv"), os.path.join(args.outdir, "pred_vs_actual.png"))
    print(f"\nSaved holdout_metrics.json and figures to {args.outdir}/")


if __name__ == "__main__":
    main()
