import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

def rates(y_true, y_score, thr):
    # Predict FAKE if score >= thr
    y_pred = (y_score >= thr).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn + 1e-12)  # recall for FAKE
    fpr = fp / (fp + tn + 1e-12)

    prec = tp / (tp + fp + 1e-12)
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    fnr = fn / (tp + fn + 1e-12)  # miss rate for FAKE (fake predicted REAL)

    return {
        "thr": float(thr),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_fpr", type=float, default=0.01, help="Constraint for T_high (FAKE decision)")
    ap.add_argument("--target_tpr", type=float, default=0.95, help="Constraint for T_low via FNR<=1-target_tpr (REAL decision safety)")
    args = ap.parse_args()

    df = pd.read_parquet(args.embeddings)
    dfv = df[df["split"] == "val"].copy()
    Xv = np.stack(dfv["emb"].values)
    yv = dfv["label"].astype(int).values

    clf = joblib.load(args.model)
    sv = clf.predict_proba(Xv)[:, 1]  # P(fake)

    auc = roc_auc_score(yv, sv)

    thrs = np.linspace(0, 1, 1001)
    stats = [rates(yv, sv, t) for t in thrs]

    # -------------------------
    # T_high: FAKE when score>=T_high
    # Goal: keep FPR <= target_fpr, maximize TPR under that constraint
    highs = [s for s in stats if s["fpr"] <= args.target_fpr]
    if highs:
        # maximize TPR, tie-breaker: smaller threshold (gives higher coverage)
        T_high = max(highs, key=lambda s: (s["tpr"], -s["thr"]))
    else:
        # best effort: minimize FPR, then maximize TPR
        T_high = min(stats, key=lambda s: (s["fpr"], -s["tpr"]))

    # -------------------------
    # T_low: REAL when score<=T_low
    # Safety constraint: fake-miss-rate (FNR) <= 1 - target_tpr
    target_fnr = 1.0 - args.target_tpr
    lows = [s for s in stats if s["fnr"] <= target_fnr]
    if lows:
        # choose largest threshold to maximize REAL coverage while keeping fake miss bounded
        T_low = max(lows, key=lambda s: s["thr"])
    else:
        # best effort: minimize FNR, then maximize threshold
        T_low = min(stats, key=lambda s: (s["fnr"], -s["thr"]))

    out = {
        "val_auc": float(auc),
        "target_fpr_for_T_high": args.target_fpr,
        "target_tpr_for_T_low_via_fnr": args.target_tpr,
        "T_high": T_high,
        "T_low": T_low,
        "note": "Use: score>=T_high => FAKE, score<=T_low => REAL, else SUSPICIOUS. "
                "T_high chosen by FPR constraint + max TPR. T_low chosen by FNR constraint + max coverage."
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
