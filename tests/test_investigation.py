import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
import generate_data
import investigation as inv


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    """Generate the datasets once into a temp folder (does not touch data/)."""
    folder = tmp_path_factory.mktemp("data")
    generate_data.generate_all(folder)
    return {"baseline": folder / "baseline.csv", "current": folder / "current.csv",
            "metrics": folder / "model_metrics.json", "folder": folder}


def test_files_are_generated(paths):
    for key in ("baseline", "current", "metrics"):
        assert paths[key].exists()
    stats = inv.get_dataset_stats(paths["baseline"])
    assert stats["rows"] == config.N_BASELINE_ROWS


def test_generation_is_reproducible(paths, tmp_path):
    generate_data.generate_all(tmp_path)
    for name in ("baseline.csv", "current.csv", "model_metrics.json"):
        assert (tmp_path / name).read_bytes() == (paths["folder"] / name).read_bytes()


def test_missing_value_degradation_detected(paths):
    result = inv.check_data_quality(paths["baseline"], paths["current"])
    issue = next(i for i in result["issues"]
                 if i["type"] == "missing_values" and i["column"] == "income")
    assert issue["baseline_rate"] < 0.04
    assert issue["current_rate"] > 0.15


def test_unknown_region_detected(paths):
    result = inv.check_data_quality(paths["baseline"], paths["current"])
    issue = next(i for i in result["issues"] if i["type"] == "unknown_category")
    assert issue["column"] == "region"
    assert issue["current_rate"] > 0.05


def test_account_balance_shift_detected(paths):
    drift = inv.detect_drift(paths["baseline"], paths["current"])
    balance = drift["features"]["account_balance"]
    assert balance["drift_level"] == "major"
    assert balance["mean_change_pct"] > 25
    assert "account_balance" in drift["major_drift_features"]


def test_stable_features_do_not_trigger_major_drift(paths):
    drift = inv.detect_drift(paths["baseline"], paths["current"])
    for col in ["credit_score", "tenure_months", "age", "support_tickets_90d"]:
        assert drift["features"][col]["drift_level"] == "stable"
        assert col not in drift["major_drift_features"]


def test_model_metrics_compared(paths):
    result = inv.compare_model_metrics(paths["metrics"])
    assert result["status"] == "degraded"
    assert len(result["degraded_metrics"]) >= 1
    recall = result["metrics"]["recall"]
    assert recall["current"] < recall["baseline"]


def test_functions_return_structured_dicts(paths):
    outputs = [
        inv.get_dataset_stats(paths["current"]),
        inv.check_data_quality(paths["baseline"], paths["current"]),
        inv.detect_drift(paths["baseline"], paths["current"]),
        inv.compare_model_metrics(paths["metrics"]),
    ]
    for out in outputs:
        assert isinstance(out, dict)
        json.dumps(out)   # must be JSON-serialisable (needed later for MCP)