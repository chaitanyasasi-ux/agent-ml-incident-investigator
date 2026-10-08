"""Creates data/baseline.csv, data/current.csv and data/model_metrics.json.

Run:  python src/generate_data.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def make_healthy_customers(n, rng):
    """Generate n customers AND their churn label from the TRUE (healthy) values."""
    income = np.clip(rng.normal(60000, 15000, n), 10000, None).round(2)
    balance = np.clip(rng.normal(50000, 20000, n), 0, None).round(2)
    region = rng.choice(["NORTH", "SOUTH", "EAST", "WEST"], size=n, p=[0.30, 0.25, 0.25, 0.20])
    credit_score = np.clip(rng.normal(650, 60, n), 300, 850).round()
    tenure = np.clip(rng.normal(36, 20, n), 1, 120).round()
    age = np.clip(rng.normal(40, 12, n), 18, 80).round()
    tickets = rng.poisson(1.5, n)

    # Churn probability: low balance / low income / many tickets / short tenure -> more churn
    logit = (-1.0
             - 1.0 * (balance - 50000) / 20000
             - 0.4 * (income - 60000) / 15000
             + 0.45 * tickets
             - 0.02 * tenure
             - 0.008 * (credit_score - 650))
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "income": income, "account_balance": balance, "region": region,
        "credit_score": credit_score, "tenure_months": tenure, "age": age,
        "support_tickets_90d": tickets, config.TARGET_COLUMN: churned,
    })


def make_baseline(rng):
    df = make_healthy_customers(config.N_BASELINE_ROWS, rng)
    missing = rng.random(len(df)) < config.BASELINE_INCOME_MISSING_RATE
    df.loc[missing, "income"] = np.nan
    return df


def make_current(rng):
    df = make_healthy_customers(config.N_CURRENT_ROWS, rng)
    n = len(df)
    # Incident 1: account_balance inflated with noise (labels were made from the true balance)
    noise = 1 + rng.normal(0, config.BALANCE_NOISE_STD, n)
    df["account_balance"] = (df["account_balance"] * config.BALANCE_INFLATION * noise).round(2)
    # Incident 2: income goes missing
    df.loc[rng.random(n) < config.CURRENT_INCOME_MISSING_RATE, "income"] = np.nan
    # Incident 3: region becomes UNKNOWN
    df.loc[rng.random(n) < config.CURRENT_UNKNOWN_REGION_RATE, "region"] = config.UNKNOWN_TOKEN
    return df


def build_model():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    prep = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])


def score(model, df):
    """Calculate metrics from real predictions (nothing hard-coded)."""
    X, y = df.drop(columns=[config.TARGET_COLUMN]), df[config.TARGET_COLUMN]
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def generate_all(data_dir=config.DATA_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)
    baseline = make_baseline(np.random.default_rng(config.RANDOM_SEED))
    current = make_current(np.random.default_rng(config.RANDOM_SEED + 1))

    # Train ONE model on baseline only. Its weights never change afterwards.
    train, holdout = train_test_split(baseline, test_size=config.TEST_SIZE,
                                      random_state=config.RANDOM_SEED,
                                      stratify=baseline[config.TARGET_COLUMN])
    model = build_model()
    model.fit(train.drop(columns=[config.TARGET_COLUMN]), train[config.TARGET_COLUMN])

    metrics = {
        "model": "LogisticRegression trained once on baseline; weights unchanged",
        "baseline_metrics": score(model, holdout),   # baseline hold-out rows
        "current_metrics": score(model, current),    # same model on incident data
    }

    baseline.to_csv(data_dir / "baseline.csv", index=False)
    current.to_csv(data_dir / "current.csv", index=False)
    with open(data_dir / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    result = generate_all()
    print("Created data/baseline.csv, data/current.csv, data/model_metrics.json")
    print(json.dumps(result, indent=2))