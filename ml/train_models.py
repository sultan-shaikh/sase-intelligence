"""
Trains real scikit-learn models on the customer dataset and exports them as
plain JSON (tree structures + encoders) so they run WITHOUT a Python backend —
netlify/functions/lib/forest.js walks these trees inside a normal Node function.

v2: schema is now auto-inferred from the CSV itself (numeric / Yes-No binary /
categorical), rather than hardcoded column lists — so this script keeps working
as the feature set grows (currently 139 features, see feature_dictionary.py)
without needing code changes each time.

Expects explicit target columns in the CSV:
  churn_risk_flag        -> "Yes"/"No"
  upsell_target_<product> -> 0/1  (one column per SASE product, e.g. upsell_target_ztna)

Run whenever you have a new/updated CRM export:
    pip install scikit-learn pandas
    python ml/train_models.py path/to/customers.csv

Writes ml/model_churn.json and ml/model_upsell.json — copy both into
netlify/functions/models/ before deploying.
"""

import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

NON_FEATURE_COLS = {"customer_id", "company_name"}
TARGET_PREFIX = "upsell_target_"
CHURN_TARGET_COL = "churn_risk_flag"


def infer_schema(df):
    numeric, binary, categorical = [], [], []
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c != CHURN_TARGET_COL and not c.startswith(TARGET_PREFIX)
    ]
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
            continue
        vals = set(df[c].dropna().unique().tolist())
        if vals <= {"Yes", "No"}:
            binary.append(c)
        else:
            categorical.append(c)
    return numeric, binary, categorical


def build_features(df, numeric, binary, categorical):
    X_num = df[numeric].astype(float).fillna(0).values
    X_bin = (df[binary] == "Yes").astype(float).values

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[categorical].astype(str))

    feature_names = numeric + binary + list(encoder.get_feature_names_out(categorical))
    X = np.hstack([X_num, X_bin, X_cat])
    return X, feature_names, encoder


def tree_to_json(tree, feature_names):
    t = tree.tree_

    def node_to_dict(i):
        if t.feature[i] == -2:
            values = t.value[i][0]
            proba = (values / values.sum()).tolist()
            return {"leaf": True, "proba": proba}
        return {
            "leaf": False,
            "feature": feature_names[t.feature[i]],
            "threshold": float(t.threshold[i]),
            "left": node_to_dict(t.children_left[i]),
            "right": node_to_dict(t.children_right[i]),
        }

    return node_to_dict(0)


def forest_to_json(forest, feature_names, classes):
    return {
        "classes": [str(c) for c in classes],
        "trees": [tree_to_json(est, feature_names) for est in forest.estimators_],
    }


def main(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    numeric, binary, categorical = infer_schema(df)
    print(f"Auto-inferred schema: {len(numeric)} numeric, {len(binary)} binary, {len(categorical)} categorical features")

    X, feature_names, encoder = build_features(df, numeric, binary, categorical)
    print(f"Total encoded feature vector length: {len(feature_names)}")

    def clean_category(v):
        # OneHotEncoder can include a literal float NaN as a category when the
        # source column has missing values. Python's json module allows bare
        # NaN by default, but that violates the JSON spec and browsers/Node's
        # JSON.parse will reject it — replace with a safe string instead.
        if isinstance(v, float) and v != v:  # NaN check without importing math
            return "__missing__"
        return v

    encoder_categories = {
        col: [clean_category(v) for v in encoder.categories_[i].tolist()]
        for i, col in enumerate(categorical)
    }
    shared_meta = {
        "feature_names": feature_names,
        "numeric_features": numeric,
        "binary_features": binary,
        "categorical_features": categorical,
        "categorical_categories": encoder_categories,
    }

    if CHURN_TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{CHURN_TARGET_COL}' not found in CSV")
    y_churn = (df[CHURN_TARGET_COL] == "Yes").astype(int).values
    churn_clf = RandomForestClassifier(n_estimators=60, max_depth=7, random_state=42, min_samples_leaf=8)
    churn_clf.fit(X, y_churn)
    churn_json = forest_to_json(churn_clf, feature_names, churn_clf.classes_)
    churn_json["feature_importances"] = dict(
        sorted(zip(feature_names, churn_clf.feature_importances_.tolist()), key=lambda kv: -kv[1])[:15]
    )
    with open("ml/model_churn.json", "w") as f:
        json.dump({**shared_meta, "model": churn_json}, f, allow_nan=False)
    print(f"Churn model trained. Base rate: {y_churn.mean():.1%}. "
          f"Top features: {list(churn_json['feature_importances'].items())[:5]}")

    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    if not target_cols:
        raise ValueError(f"No columns starting with '{TARGET_PREFIX}' found in CSV")

    upsell_models = {}
    for col in target_cols:
        product = col[len(TARGET_PREFIX):]
        y = df[col].astype(int).values
        if y.sum() < 15:
            print(f"  Skipping {product}: too few positive examples ({y.sum()})")
            continue
        clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, min_samples_leaf=8)
        clf.fit(X, y)
        fj = forest_to_json(clf, feature_names, clf.classes_)
        fj["feature_importances"] = dict(
            sorted(zip(feature_names, clf.feature_importances_.tolist()), key=lambda kv: -kv[1])[:10]
        )
        upsell_models[product] = fj
        print(f"  {product}: base rate {y.mean():.1%}, top driver: {list(fj['feature_importances'].items())[0]}")

    with open("ml/model_upsell.json", "w") as f:
        json.dump({**shared_meta, "models": upsell_models}, f, allow_nan=False)

    print(f"\nTrained on {len(df)} customers.")
    print(f"Upsell models trained for: {list(upsell_models.keys())}")
    print("Wrote ml/model_churn.json and ml/model_upsell.json")
    print("Copy both into netlify/functions/models/ before deploying.")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "customers.csv"
    main(csv_path)
