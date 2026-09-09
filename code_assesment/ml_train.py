import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SPLITS = 5

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
raw = pd.read_csv(url)
print(raw.shape)
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]
y = raw["is_canceled"].values

# ---------------------------------------------------------------------------
# BASELINE feature set (standard published approach)
# ---------------------------------------------------------------------------
def build_baseline_features(df):
    d = df.copy()
    d["children"] = d["children"].fillna(0)
    d["country"] = d["country"].fillna(d["country"].mode()[0])

    num_feats = ["lead_time", "arrival_date_year", "arrival_date_week_number",
                 "arrival_date_day_of_month", "stays_in_weekend_nights",
                 "stays_in_week_nights", "adults", "children", "babies",
                 "is_repeated_guest", "previous_cancellations"]
    cat_feats = ["hotel", "arrival_date_month", "meal", "country",
                 "market_segment", "distribution_channel", "deposit_type"]
    return d[num_feats + cat_feats], num_feats, cat_feats


# ---------------------------------------------------------------------------
# PROPOSED feature set (domain-engineered + high-cardinality-safe encoding)
# ---------------------------------------------------------------------------
def build_proposed_features(df):
    d = df.copy()
    d["children"] = d["children"].fillna(0)
    d["country"] = d["country"].fillna("UNK")
    d["agent"] = d["agent"].fillna(0).astype(int).astype(str)
    d["agent"] = d["agent"].replace("0", "None")

    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["total_guests"] = d["adults"] + d["children"] + d["babies"]
    d["is_family"] = ((d["children"] > 0) | (d["babies"] > 0)).astype(int)
    d["prior_cancel_rate"] = d["previous_cancellations"] / (
        d["previous_cancellations"] + d["previous_bookings_not_canceled"] + 1)
    d["room_type_mismatch"] = (d["assigned_room_type"] != d["reserved_room_type"]).astype(int)
    d["log_adr"] = np.log1p(d["adr"].clip(lower=0))

    # frequency-encode high-cardinality categoricals instead of one-hot
    for col in ["country", "agent"]:
        freq = d[col].value_counts(normalize=True)
        d[col + "_freq"] = d[col].map(freq)

    num_feats = ["lead_time", "arrival_date_year", "arrival_date_week_number",
                 "arrival_date_day_of_month", "total_nights", "total_guests",
                 "is_family", "adults", "children", "babies", "is_repeated_guest",
                 "previous_cancellations", "prior_cancel_rate", "booking_changes",
                 "total_of_special_requests", "required_car_parking_spaces",
                 "days_in_waiting_list", "log_adr", "room_type_mismatch",
                 "country_freq", "agent_freq"]
    cat_feats = ["hotel", "arrival_date_month", "meal", "market_segment",
                 "distribution_channel", "deposit_type", "customer_type"]
    return d[num_feats + cat_feats], num_feats, cat_feats


X_base_df, base_num, base_cat = build_baseline_features(raw)
X_prop_df, prop_num, prop_cat = build_proposed_features(raw)

# ---------------------------------------------------------------------------
# Paired Stratified K-Fold CV -- identical folds used for BOTH models
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

metrics = {"baseline": {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []},
           "proposed": {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}}

fold_num = 0
for train_idx, test_idx in skf.split(X_base_df, y):
    fold_num += 1
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- baseline: one-hot on raw categoricals (incl. 177-level country) ----
    base_pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", base_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), base_cat),
        ])),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=14,
                                        n_jobs=1, random_state=RANDOM_STATE)),
    ])
    base_pipe.fit(X_base_df.iloc[train_idx], y_train)
    p_base = base_pipe.predict(X_base_df.iloc[test_idx])
    proba_base = base_pipe.predict_proba(X_base_df.iloc[test_idx])[:, 1]

    # ---- proposed: engineered features + frequency encoding + XGBoost ----
    prop_pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", prop_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), prop_cat),
        ])),
        ("clf", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", tree_method="hist", n_jobs=1,
                               random_state=RANDOM_STATE)),
    ])
    prop_pipe.fit(X_prop_df.iloc[train_idx], y_train)
    p_prop = prop_pipe.predict(X_prop_df.iloc[test_idx])
    proba_prop = prop_pipe.predict_proba(X_prop_df.iloc[test_idx])[:, 1]

    for name, y_pred, y_proba in [("baseline", p_base, proba_base), ("proposed", p_prop, proba_prop)]:
        metrics[name]["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics[name]["precision"].append(precision_score(y_test, y_pred))
        metrics[name]["recall"].append(recall_score(y_test, y_pred))
        metrics[name]["f1"].append(f1_score(y_test, y_pred))
        metrics[name]["roc_auc"].append(roc_auc_score(y_test, y_proba))

    print(f"fold {fold_num}/{N_SPLITS} done: "
          f"baseline_acc={metrics['baseline']['accuracy'][-1]:.4f} "
          f"proposed_acc={metrics['proposed']['accuracy'][-1]:.4f}")

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
acc_base = np.array(metrics["baseline"]["accuracy"])
acc_prop = np.array(metrics["proposed"]["accuracy"])
diff = acc_prop - acc_base

t_stat, p_two_sided = stats.ttest_rel(acc_prop, acc_base)
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

rng = np.random.default_rng(RANDOM_STATE)
boot_diffs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

summary = {"folds": N_SPLITS, "n_rows": len(raw)}
for name in ["baseline", "proposed"]:
    summary[name] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]}
                      for m, v in metrics[name].items()}

summary["ab_test"] = {
    "hypothesis": "Proposed (XGBoost + engineered features + frequency encoding) has higher mean CV accuracy than baseline (RandomForest on standard one-hot feature set)",
    "mean_accuracy_diff": float(diff.mean()),
    "paired_t_stat": float(t_stat),
    "p_value_one_sided": float(p_one_sided),
    "bootstrap_95ci_diff": [float(ci_low), float(ci_high)],
    "significant_at_0.05": bool(p_one_sided < 0.05),
}

with open("results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary["ab_test"], indent=2))
print("\nBaseline accuracy: %.4f +/- %.4f" % (acc_base.mean(), acc_base.std()))
print("Proposed accuracy: %.4f +/- %.4f" % (acc_prop.mean(), acc_prop.std()))
for m in ["precision", "recall", "f1", "roc_auc"]:
    b = np.mean(metrics["baseline"][m]); p = np.mean(metrics["proposed"][m])
    print(f"{m}: baseline={b:.4f} proposed={p:.4f} diff={p-b:+.4f}")