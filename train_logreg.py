import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="Path to embeddings.parquet")
    ap.add_argument("--out_model", required=True, help="Output .pkl path")
    ap.add_argument("--out_metrics", required=True, help="Output metrics.json path")
    args = ap.parse_args()

    emb_path = Path(args.embeddings).expanduser().resolve()
    df = pd.read_parquet(emb_path)

    # Convert embeddings column (arrays) to matrix
    X = np.stack(df["emb"].values)
    y = df["label"].astype(int).values
    split = df["split"].astype(str).values

    X_train, y_train = X[split == "train"], y[split == "train"]
    X_val, y_val = X[split == "val"], y[split == "val"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    def eval_split(name, Xs, ys):
        if len(ys) == 0:
            return None
        prob = clf.predict_proba(Xs)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(ys, prob)),
            "acc@0.5": float(accuracy_score(ys, pred)),
            "cm@0.5": confusion_matrix(ys, pred).tolist(),
        }

    metrics = {
        "train": eval_split("train", X_train, y_train),
        "val": eval_split("val", X_val, y_val),
        "test": eval_split("test", X_test, y_test),
        "note": "Threshold=0.5 is default. You should tune threshold on val for your use-case."
    }

    Path(args.out_model).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, args.out_model)

    import json
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved model:", args.out_model)
    print("[DONE] Saved metrics:", args.out_metrics)
    print(metrics)


if __name__ == "__main__":
    main()
