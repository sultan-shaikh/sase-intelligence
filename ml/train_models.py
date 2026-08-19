"""
Trains real scikit-learn models on the customer dataset and exports them
as plain JSON (tree structures + encoders) so they can run WITHOUT a
Python backend — a small JS inference engine (netlify/functions/lib/forest.js)
walks the exported trees at request time inside a normal Node Netlify Function.

Run this locally whenever you have a new/updated CSV export from Salesforce:
    pip install scikit-learn pandas
    python ml/train_models.py path/to/customers.csv

It writes ml/model_churn.json and ml/model_upsell.json, which you then
copy into netlify/functions/models/ before deploying.
"""

import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["segment", "revenue_band_gbp", "avg_ticket_severity", "regional_focus", "industry_vertical"]
BINARY = ["ngfw_user", "sdwan_user", "legacy_security_user"]
NUMERIC = [
    "company_size", "months_since_contract_start", "months_until_renewal",
    "contract_value_gbp", "annual_growth_pct", "support_tickets_last_12m",
    "product_adoption_rate", "feature_adoption_rate", "concurrent_users_pct",
    "security_incidents_last_12m", "breach_risk_score", "nps_score",
]

UPSELL_LABELS = ["NGFW", "SD-WAN", "SSE", "Threat Mgmt", "Enterprise Browser", "Automated SOC", "Managed SOC"]


def build_features(df):
    X_num = df[NUMERIC].astype(float).values
    X_bin = (df[BINARY] == "Yes").astype(float).values

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[CATEGORICAL].astype(str))

    feature_names = (
        NUMERIC + BINARY + list(encoder.get_feature_names_out(CATEGORICAL))
    )
    X = np.hstack([X_num, X_bin, X_cat])
    return X, feature_names, encoder


def tree_to_json(tree, feature_names):
    """Convert an sklearn DecisionTreeClassifier into a plain nested dict."""
    t = tree.tree_

    def node_to_dict(i):
        if t.feature[i] == -2:  # leaf
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

    X, feature_names, encoder = build_features(df)

    # ── Churn risk model (real binary classifier) ──────────────────────
    y_churn = (df["churn_risk_flag"] == "Yes").astype(int).values
    churn_clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, min_samples_leaf=5)
    churn_clf.fit(X, y_churn)
    churn_json = forest_to_json(churn_clf, feature_names, churn_clf.classes_)
    churn_json["feature_importances"] = dict(
        sorted(zip(feature_names, churn_clf.feature_importances_.tolist()), key=lambda kv: -kv[1])[:10]
    )

    # ── Upsell propensity models — one RandomForest per SASE product ───
    # adjacent_service_oppty is a combined string like "SSE+Threat Mgmt";
    # we split it into independent binary targets, one model per product,
    # so each product gets its own real probability rather than one
    # multi-class guess.
    upsell_models = {}
    for label in UPSELL_LABELS:
        y = df["adjacent_service_oppty"].str.contains(label, regex=False).astype(int).values
        if y.sum() < 8:  # not enough positive examples to train reliably
            continue
        clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, min_samples_leaf=5)
        clf.fit(X, y)
        upsell_models[label] = forest_to_json(clf, feature_names, clf.classes_)
        upsell_models[label]["feature_importances"] = dict(
            sorted(zip(feature_names, clf.feature_importances_.tolist()), key=lambda kv: -kv[1])[:8]
        )

    encoder_categories = {
        col: encoder.categories_[i].tolist() for i, col in enumerate(CATEGORICAL)
    }

    out = {
        "feature_names": feature_names,
        "numeric_features": NUMERIC,
        "binary_features": BINARY,
        "categorical_features": CATEGORICAL,
        "categorical_categories": encoder_categories,
    }

    with open("ml/model_churn.json", "w") as f:
        json.dump({**out, "model": churn_json}, f)

    with open("ml/model_upsell.json", "w") as f:
        json.dump({**out, "models": upsell_models}, f)

    print(f"Trained on {len(df)} customers.")
    print(f"Churn model: {len(churn_clf.estimators_)} trees, top features:",
          list(churn_json["feature_importances"].items())[:3])
    print(f"Upsell models trained for: {list(upsell_models.keys())}")
    print("Wrote ml/model_churn.json and ml/model_upsell.json")
    print("Copy both into netlify/functions/models/ before deploying.")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "customers.csv"
    main(csv_path)
