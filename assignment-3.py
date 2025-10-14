import os, json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42
TEST_SIZE   = 0.20
CV_FOLDS    = 5

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
FIG_DIR    = os.path.join(BASE_DIR, "figs")
TAB_DIR    = os.path.join(BASE_DIR, "tables")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
for _d in [FIG_DIR, TAB_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# Helpers
def figpath(name): return os.path.join(FIG_DIR, name)
def tabpath(name): return os.path.join(TAB_DIR, name)
def savefig(name): plt.tight_layout(); plt.savefig(figpath(name), dpi=150); plt.close()
def save_table(df: pd.DataFrame, name: str) -> str:
    path = tabpath(name); df.to_csv(path, index=False); return path


# Load & Prepare
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    d1 = pd.read_csv(os.path.join(DATA_DIR, "dataset1.csv"))
    d2 = pd.read_csv(os.path.join(DATA_DIR, "dataset2.csv"))
    return d1, d2

def basic_clean(d1: pd.DataFrame, d2: pd.DataFrame):
    for col in ["risk","reward"]:
        if col in d1.columns:
            d1[col] = pd.to_numeric(d1[col], errors="coerce").fillna(0).astype(int)
    # standardise text columns
    for df in [d1, d2]:
        if "season" in df.columns:
            df["season"] = df["season"].astype(str).str.strip().str.lower()
    return d1, d2
# FEATURE ENGINEERING (Investigation A & B)
# near_rat_window:
#   Binary flag = 1 if a rat arrived within the last 30 seconds, else 0.
#   Rationale: tests the immediate-competition hypothesis (do bats respond right away?).
#   Simple, interpretable, aligns with the brief’s emphasis on behavioural drivers.
#
# rat_recency_bucket:
#   Categorical bucket of seconds_since_rat_arrival: <=10s, 11–30s, >30s.
#   Rationale: captures a graded (monotonic) recency effect without assuming linearity.
#   Useful for both EDA (box/violin plots) and as one-hot predictors in LR.
#
# hour_bin:
#   Bins hours_after_sunset into ["very_early", "early", "mid", "late"].
#   Rationale: robustly encodes nocturnal phases; reduces noise vs using raw hour.
#   Improves interpretability and lets LR estimate different baselines per night phase.
def engineer_d1(d1: pd.DataFrame) -> pd.DataFrame:
    if "seconds_after_rat_arrival" in d1.columns:
        d1["near_rat_window"] = (pd.to_numeric(d1["seconds_after_rat_arrival"], errors="coerce") <= 30).astype(int)
        d1["rat_recency_bucket"] = pd.cut(pd.to_numeric(d1["seconds_after_rat_arrival"], errors="coerce"),
                                          bins=[-1,10,30,1e9], labels=["<=10s","11-30s",">30s"])
    if "hours_after_sunset" in d1.columns:
        d1["hour_bin"] = pd.cut(pd.to_numeric(d1["hours_after_sunset"], errors="coerce"),
                                bins=[-1,1,3,5,10], labels=["very_early","early","mid","late"])
    # y is already bat_landing_to_food in dataset1
    return d1

def engineer_d2(d2: pd.DataFrame) -> pd.DataFrame:
    if "hours_after_sunset" in d2.columns:
        d2["hour_bin"] = pd.cut(pd.to_numeric(d2["hours_after_sunset"], errors="coerce"),
                                bins=[-1,1,3,5,10], labels=["very_early","early","mid","late"])
    if {"rat_minutes","bat_landing_number"}.issubset(d2.columns):
        denom = pd.to_numeric(d2["rat_minutes"], errors="coerce").replace(0, np.nan)
        d2["landing_per_rat_min"] = (pd.to_numeric(d2["bat_landing_number"], errors="coerce") / denom).fillna(0.0)
    # derive season if missing or invalid
    if "season" not in d2.columns or d2["season"].isna().all():
       
            m = pd.to_numeric(d2["month"], errors="coerce").astype("Int64")
            d2["season"] = m.apply(lambda x: "winter" if x <= 2 else "spring")
        
    return d2


# ---------- EDA ----------
def eda_plots(d1: pd.DataFrame, d2: pd.DataFrame) -> List[str]:
    figs = []
    # Heatmaps 
    for df, name, title in [(d1,"eda_d1_corr.png","Correlation Heatmap – dataset1"),
                            (d2,"eda_d2_corr.png","Correlation Heatmap – dataset2")]:
        num = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
        if num.shape[1] >= 2:
            C = num.corr()
            plt.figure(figsize=(7,5)); 
            sns.heatmap(C, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,cbar=True, square=True, linewidths=0.3)
            
            plt.xticks(range(len(C.columns)), C.columns, rotation=90, fontsize=8)
            plt.yticks(range(len(C.columns)), C.columns, fontsize=8)
            plt.title(title); plt.tight_layout();savefig(name); figs.append(name)

    # Key relationships
    if {"rat_minutes","bat_landing_number","season"}.issubset(d2.columns):
        plt.figure(figsize=(6,4))
        for g, dfg in d2.groupby(d2["season"].astype(str)):
            plt.scatter(pd.to_numeric(dfg["rat_minutes"], errors="coerce"),
                        pd.to_numeric(dfg["bat_landing_number"], errors="coerce"),
                        label=g, alpha=0.7)
        plt.legend(title="season", fontsize=8)
        plt.xlabel("rat_minutes"); plt.ylabel("bat_landing_number")
        plt.title("Bat Landing Number vs Rat Minutes (by Season)")
        savefig("eda_scatter_ratmin_batlanding.png"); figs.append("eda_scatter_ratmin_batlanding.png")

    if {"bat_landing_to_food","rat_recency_bucket"}.issubset(d1.columns):
        cats = d1["rat_recency_bucket"].astype(str).unique().tolist()
        data = [pd.to_numeric(d1[d1["rat_recency_bucket"].astype(str)==c]["bat_landing_to_food"], errors="coerce").values
                for c in cats]
        plt.figure(figsize=(6,4)); plt.boxplot(data, tick_labels=cats)
        plt.title("Behaviour proxy vs Rat Recency (dataset1)")
        plt.xlabel("rat_recency_bucket"); plt.ylabel("bat_landing_to_food")
        savefig("eda_box_bltf_by_recency.png"); figs.append("eda_box_bltf_by_recency.png")

    return figs
# ---------------------------------------------------
# PREPROCESSING PIPELINE (scaler + one-hot encoding)
# ---------------------------------------------------
# - Numeric features are StandardScaled for stable, comparable coefficients.
# - Categorical features are one-hot encoded with handle_unknown="ignore"
#   so the model remains robust if unseen categories appear in the test split.
# - This ColumnTransformer is used inside a sklearn Pipeline to avoid leakage
#   and guarantee identical transforms at train and test time.
# ---------- Modelling ----------
def prep_features(df: pd.DataFrame, y_col: str, poly=False, interactions=False):
    y = pd.to_numeric(df[y_col], errors="coerce").astype(float)
    X = df.drop(columns=[y_col]).copy()
    prefer = ["rat_minutes","rat_arrival_number","hours_after_sunset","food_availability",
              "bat_landing_number","seconds_after_rat_arrival","near_rat_window",
              "landing_per_rat_min","hour_bin","season","rat_recency_bucket"]
    cols = [c for c in prefer if c in X.columns]
    X = X[cols]
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    from sklearn.pipeline import Pipeline as SkPipe
    steps = []
    #if poly: steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("sc", StandardScaler()))
    transformers = []
    if interactions and not poly:
        if num_cols:
            transformers.append(("num", SkPipe([("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
                                                ("sc", StandardScaler())]), num_cols))
        if cat_cols: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    else:
        if num_cols: transformers.append(("num", SkPipe(steps), num_cols))
        if cat_cols: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return X, y, pre

def residual_plots(y_true, y_pred, tag):
    res = y_true - y_pred
    plt.figure(figsize=(6,4)); plt.scatter(y_pred, res, alpha=0.7); plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.title(f"Residuals vs Predicted – {tag}")
    savefig(f"diag_residuals_{tag}.png")
    plt.figure(figsize=(6,4)); plt.hist(res, bins=30); plt.title(f"Residual Distribution – {tag}")
    savefig(f"diag_residual_hist_{tag}.png")

def feature_names_after(pre: ColumnTransformer) -> List[str]:
    names = []
    for name, trans, cols in pre.transformers_:
        if hasattr(trans, "get_feature_names_out"):
            try: fn = trans.get_feature_names_out(cols)
            except Exception: fn = cols
            names.extend(list(fn))
        else:
            names.extend(list(cols) if isinstance(cols, (list,tuple)) else [cols])
    return names

def fit_and_report(model, X, y, pre, tag: str) -> Dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    from sklearn.pipeline import Pipeline as SkPipe
    pipe = SkPipe([("pre", pre), ("mdl", model)])
    cv = cross_val_score(pipe, Xtr, ytr, cv=CV_FOLDS, scoring="neg_mean_squared_error")
    pipe.fit(Xtr, ytr); yhat = pipe.predict(Xte)
    mse = mean_squared_error(yte, yhat); rmse = mse ** 0.5; r2 = r2_score(yte, yhat)
    cv_rmse = float(((-cv)**0.5).mean())
    residual_plots(yte, yhat, tag)

    coef_csv = None
    if hasattr(pipe.named_steps["mdl"], "coef_"):
        names = feature_names_after(pipe.named_steps["pre"])
        coefs = pd.Series(pipe.named_steps["mdl"].coef_.ravel(), index=names)\
                    .sort_values(key=lambda s: s.abs(), ascending=False).reset_index()
        coefs.columns = ["feature","coefficient"]
        coef_csv = save_table(coefs, f"coef_{tag}.csv")

    return {"model": tag, "MAE": float(mean_absolute_error(yte, yhat)),
            "RMSE": float(rmse), "R2": float(r2), "CV_RMSE_mean": cv_rmse, "coef_csv": coef_csv}

def main():
    d1, d2 = load_data()
    d1, d2 = basic_clean(d1, d2)
    d1 = engineer_d1(d1); d2 = engineer_d2(d2)

    eda_figs = eda_plots(d1, d2)
    #A = run_investigation_A(d1)
    #B = run_investigation_B(d2)


if __name__ == "__main__":
    main()
