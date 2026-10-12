"""
Stage 2: turn the per-photo JSON from `extract_and_explain.py` into the two CSVs
that `train_ensemble.py` expects.

  exif_clean.csv   image_id, shutter_speed, aperture, iso, focal_length,
                   aesthetic_score, CameraMake_* one-hot columns
  proxy_clean.csv  image_id, saturation_proxy, contrast_proxy

Missing EXIF values are filled with the column mean.

Usage:
    python pipeline/prepare_model_inputs.py --input final_results.json --outdir .
"""

import argparse
import json
import os

import pandas as pd

EXIF_RENAME = {
    "ExposureTime_Num": "shutter_speed",
    "FNumber": "aperture",
    "ISO": "iso",
    "FocalLength": "focal_length",
}


def to_frames(records):
    rows = []
    for entry in records:
        row = {"image_id": entry["photo_id"]}
        row.update(entry["raw_features"])
        row["aesthetic_score"] = entry["scores"]["actual_proxy_score"]
        rows.append(row)
    df = pd.DataFrame(rows).rename(columns=EXIF_RENAME)

    for col in EXIF_RENAME.values():
        missing = int(df[col].isna().sum())
        if missing:
            df[col] = df[col].fillna(df[col].mean())
            print(f"  filled {missing} missing '{col}' values with the column mean")

    ohe = [c for c in df.columns if c.startswith("CameraMake_")]
    exif = df[["image_id", *EXIF_RENAME.values(), "aesthetic_score", *ohe]].copy()
    proxy = df[["image_id", "Saturation_Proxy", "Contrast_Proxy"]].rename(columns={
        "Saturation_Proxy": "saturation_proxy", "Contrast_Proxy": "contrast_proxy"})
    return exif, proxy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="final_results.json")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} photos from {args.input}")

    exif, proxy = to_frames(records)
    os.makedirs(args.outdir, exist_ok=True)
    exif.to_csv(os.path.join(args.outdir, "exif_clean.csv"), index=False)
    proxy.to_csv(os.path.join(args.outdir, "proxy_clean.csv"), index=False)
    print(f"exif_clean.csv  -> {exif.shape}")
    print(f"proxy_clean.csv -> {proxy.shape}")
    print("Next: python pipeline/train_ensemble.py --exif exif_clean.csv --proxy proxy_clean.csv")


if __name__ == "__main__":
    main()
