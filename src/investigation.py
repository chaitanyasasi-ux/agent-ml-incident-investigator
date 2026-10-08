"""Deterministic investigation tools. No LLM, no randomness.

Every function takes file paths and returns a plain dictionary,
so each one can later be exposed as an MCP tool without changes.
"""
import json

import numpy as np
import pandas as pd

import config


def _load(path):
    return pd.read_csv(path)


def _feature_columns(df):
    return [c for c in df.columns if c != config.TARGET_COLUMN]


# ---------------------------------------------------------------- Tool 1
def get_dataset_stats(csv_path):
    """Summarise one dataset: rows, missing rate and basic stats per column."""
    df = _load(csv_path)
    columns = {}
    for col in df.columns:
        info = {"missing_rate": round(float(df[col].isna().mean()), 4)}
        if pd.api.types.is_numeric_dtype(df[col]):
            info.update({
                "type": "numeric",
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            })
        else:
            shares = df[col].value_counts(normalize=True).round(4)
            info.update({"type": "categorical", "value_share": shares.to_dict()})
        columns[col] = info
    return {"path": str(csv_path), "rows": len(df), "columns": columns}


# ---------------------------------------------------------------- Tool 2
def check_data_quality(baseline_path, current_path):
    """Find columns whose missing rate rose, and categorical 'UNKNOWN' spikes."""
    base, cur = _load(baseline_path), _load(current_path)
    issues = []

    for col in _feature_columns(base):
        b_missing = float(base[col].isna().mean())
        c_missing = float(cur[col].isna().mean())
        if c_missing - b_missing > config.MISSING_INCREASE_THRESHOLD:
            issues.append({"type": "missing_values", "column": col,
                           "baseline_rate": round(b_missing, 4),
                           "current_rate": round(c_missing, 4)})

        if not pd.api.types.is_numeric_dtype(base[col]):
            b_unknown = float((base[col] == config.UNKNOWN_TOKEN).mean())
            c_unknown = float((cur[col] == config.UNKNOWN_TOKEN).mean())
            if c_unknown - b_unknown > config.UNKNOWN_INCREASE_THRESHOLD:
                issues.append({"type": "unknown_category", "column": col,
                               "baseline_rate": round(b_unknown, 4),
                               "current_rate": round(c_unknown, 4)})

    return {"status": "issues_found" if issues else "ok",
            "issue_count": len(issues), "issues": issues}


# ---------------------------------------------------------------- Tool 3
def _psi(expected_share, actual_share):
    """Population Stability Index: sum((actual - expected) * ln(actual / expected))."""
    e = np.clip(expected_share, 1e-4, None)   # avoid log(0)
    a = np.clip(actual_share, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _psi_numeric(base_values, cur_values):
    # Bucket edges come from the BASELINE deciles; outer edges are open-ended.
    edges = np.unique(np.quantile(base_values, np.linspace(0, 1, config.PSI_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b_counts, _ = np.histogram(base_values, bins=edges)
    c_counts, _ = np.histogram(cur_values, bins=edges)
    return _psi(b_counts / len(base_values), c_counts / len(cur_values))


def _psi_categorical(base_values, cur_values):
    b_share = base_values.value_counts(normalize=True)
    c_share = cur_values.value_counts(normalize=True)
    categories = sorted(set(b_share.index) | set(c_share.index))
    b = np.array([b_share.get(c, 0.0) for c in categories])
    c = np.array([c_share.get(c, 0.0) for c in categories])
    return _psi(b, c)


def _drift_level(psi):
    if psi >= config.PSI_MAJOR:
        return "major"
    if psi >= config.PSI_MODERATE:
        return "moderate"
    return "stable"


def detect_drift(baseline_path, current_path):
    """Compute PSI for every feature and label it stable / moderate / major."""
    base, cur = _load(baseline_path), _load(current_path)
    features = {}
    for col in _feature_columns(base):
        b, c = base[col].dropna(), cur[col].dropna()   # missing values handled by quality check
        if pd.api.types.is_numeric_dtype(base[col]):
            psi = _psi_numeric(b.to_numpy(), c.to_numpy())
            mean_change = (c.mean() - b.mean()) / b.mean() * 100
            features[col] = {"type": "numeric", "psi": round(psi, 4),
                             "drift_level": _drift_level(psi),
                             "mean_change_pct": round(float(mean_change), 2)}
        else:
            psi = _psi_categorical(b, c)
            features[col] = {"type": "categorical", "psi": round(psi, 4),
                             "drift_level": _drift_level(psi)}

    return {
        "features": features,
        "major_drift_features": [k for k, v in features.items() if v["drift_level"] == "major"],
        "moderate_drift_features": [k for k, v in features.items() if v["drift_level"] == "moderate"],
        "stable_features": [k for k, v in features.items() if v["drift_level"] == "stable"],
    }


# ---------------------------------------------------------------- Tool 4
def compare_model_metrics(metrics_path):
    """Compare baseline_metrics vs current_metrics and flag meaningful drops."""
    with open(metrics_path) as f:
        data = json.load(f)
    baseline, current = data["baseline_metrics"], data["current_metrics"]

    comparison, degraded = {}, []
    for name in baseline:
        change = current[name] - baseline[name]
        is_degraded = bool(-change > config.METRIC_DROP_THRESHOLD)
        comparison[name] = {"baseline": baseline[name], "current": current[name],
                            "change": round(change, 4), "degraded": is_degraded}
        if is_degraded:
            degraded.append(name)

    return {"status": "degraded" if degraded else "ok",
            "degraded_metrics": degraded, "metrics": comparison}