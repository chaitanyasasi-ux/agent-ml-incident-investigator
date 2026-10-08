import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generate_data
import investigation as inv
import mcp_server

EXPECTED_TOOLS = {"get_dataset_stats", "check_data_quality",
                  "detect_drift", "compare_model_metrics"}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mcp_data")
    generate_data.generate_all(folder)
    return folder


# ---- helpers: talk to the server in memory (no subprocess, no external client) ----
def list_tools(server):
    async def run():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return (await client.list_tools()).tools
    return asyncio.run(run())


def call_tool(server, name, arguments=None):
    async def run():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await client.call_tool(name, arguments or {})
    return asyncio.run(run())


def data_of(result):
    """The tool's dictionary comes back as JSON text inside the MCP result."""
    assert not result.isError
    return json.loads(result.content[0].text)


# ------------------------------------------------------------------ tests
def test_server_starts_and_responds(data_dir):
    server = mcp_server.build_server(data_dir)
    assert len(list_tools(server)) > 0      # session initialised and answered


def test_all_four_tools_registered(data_dir):
    names = {t.name for t in list_tools(mcp_server.build_server(data_dir))}
    assert names == EXPECTED_TOOLS


def test_tools_have_descriptions_and_schemas(data_dir):
    tools = {t.name: t for t in list_tools(mcp_server.build_server(data_dir))}
    for tool in tools.values():
        assert tool.description and len(tool.description) > 20
        assert tool.inputSchema["type"] == "object"
    assert "dataset" in tools["get_dataset_stats"].inputSchema["properties"]
    assert "dataset" in tools["get_dataset_stats"].inputSchema["required"]


def test_get_dataset_stats_tool(data_dir):
    server = mcp_server.build_server(data_dir)
    out = data_of(call_tool(server, "get_dataset_stats", {"dataset": "current"}))
    assert out["rows"] == 5000
    assert out["columns"]["region"]["value_share"]["UNKNOWN"] > 0.05


def test_check_data_quality_tool(data_dir):
    out = data_of(call_tool(mcp_server.build_server(data_dir), "check_data_quality"))
    assert out["status"] == "issues_found"
    assert {i["type"] for i in out["issues"]} == {"missing_values", "unknown_category"}


def test_detect_drift_tool(data_dir):
    out = data_of(call_tool(mcp_server.build_server(data_dir), "detect_drift"))
    assert "account_balance" in out["major_drift_features"]
    assert "credit_score" in out["stable_features"]


def test_compare_model_metrics_tool(data_dir):
    out = data_of(call_tool(mcp_server.build_server(data_dir), "compare_model_metrics"))
    assert out["status"] == "degraded"
    assert "recall" in out["degraded_metrics"]


def test_mcp_result_equals_day1_function_result(data_dir):
    """Proves the tool reuses Day-1 logic instead of duplicating it."""
    via_mcp = data_of(call_tool(mcp_server.build_server(data_dir), "detect_drift"))
    direct = inv.detect_drift(data_dir / "baseline.csv", data_dir / "current.csv")
    assert via_mcp == json.loads(json.dumps(direct))


def test_invalid_input_does_not_crash_server(data_dir):
    server = mcp_server.build_server(data_dir)
    bad = call_tool(server, "get_dataset_stats", {"dataset": "nonsense"})
    assert bad.isError                      # rejected cleanly by the input schema
    good = call_tool(server, "get_dataset_stats", {"dataset": "baseline"})
    assert data_of(good)["rows"] == 5000    # server still works afterwards


def test_missing_data_files_return_error_dict(tmp_path):
    empty_server = mcp_server.build_server(tmp_path)   # folder has no data files
    out = data_of(call_tool(empty_server, "check_data_quality"))
    assert out["status"] == "error"
    assert "not found" in out["error"]