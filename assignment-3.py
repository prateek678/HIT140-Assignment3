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





def main():
    d1, d2 = load_data()
    d1, d2 = basic_clean(d1, d2)
    d1 = engineer_d1(d1); d2 = engineer_d2(d2)

    eda_figs = eda_plots(d1, d2)
    A = run_investigation_A(d1)
    B = run_investigation_B(d2)

    out_path = os.path.join(REPORT_DIR, "results_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"eda_figs": eda_figs, "A": A, "B": B}, f, indent=2)
    print("Saved:", out_path)
    print("Figures:", eda_figs)

if __name__ == "__main__":
    main()
