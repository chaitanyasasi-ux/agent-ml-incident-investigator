"""Local MCP server that exposes the Day-1 investigation functions as tools.

Run:  python src/mcp_server.py
(It talks over stdio, so it waits silently for an MCP client. Never use print()
in this file: stdout carries the MCP protocol messages.)
"""
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

import config
import investigation


def _safe_call(func, *args):
    """Run a Day-1 function; turn any failure into a clean error dictionary."""
    try:
        return func(*args)
    except FileNotFoundError as e:
        return {"status": "error",
                "error": f"Data file not found ({e}). Run: python src/generate_data.py"}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def build_server(data_dir=config.DATA_DIR):
    """Create the MCP server. data_dir is a parameter so tests can use temp data."""
    data_dir = Path(data_dir)
    dataset_paths = {"baseline": data_dir / "baseline.csv",
                     "current": data_dir / "current.csv"}
    metrics_path = data_dir / "model_metrics.json"

    mcp = FastMCP("ml-incident-investigator")#This creates the server object with the name "ml-incident-investigator" which clients see when they connect.

    @mcp.tool()#This is a decorator that registers the below function as a tool FastMCP build 3 things from the function tool name ,description and input schema. 
    def get_dataset_stats(dataset: Literal["baseline", "current"]) -> dict[str, Any]:
        """Summarise one dataset: row count, and for every column its missing rate
        plus mean/std/min/max (numbers) or category shares (text).
        Argument 'dataset' must be 'baseline' or 'current'."""
        return _safe_call(investigation.get_dataset_stats, dataset_paths[dataset])

    @mcp.tool()
    def check_data_quality() -> dict[str, Any]:
        """Compare current vs baseline data quality. Reports columns whose
        missing-value rate rose and categorical columns with an 'UNKNOWN' spike."""
        return _safe_call(investigation.check_data_quality,
                          dataset_paths["baseline"], dataset_paths["current"])

    @mcp.tool()
    def detect_drift() -> dict[str, Any]:
        """Detect feature distribution drift between baseline and current data
        using PSI. Labels each feature stable, moderate or major."""
        return _safe_call(investigation.detect_drift,
                          dataset_paths["baseline"], dataset_paths["current"])

    @mcp.tool()
    def compare_model_metrics() -> dict[str, Any]:
        """Compare baseline_metrics with current_metrics (accuracy, precision,
        recall, f1, roc_auc) and list the metrics that degraded."""
        return _safe_call(investigation.compare_model_metrics, metrics_path)

    return mcp


if __name__ == "__main__":
    build_server().run()   # default transport: stdio