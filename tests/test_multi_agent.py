import asyncio
import json
import re
import sys
from pathlib import Path

import groq
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent
import config
import generate_data
import investigation as inv          # tests may use it to build realistic expectations
import mcp_server
import multi_agent as ma
from test_agent import (GOOD_REPORT, FakeGroq, FakeSession, final_answer,    # reuse Day-3 fakes
                        groq_error, tool_calls)


# ------------------------------------------------------------------ helpers
class RecordingSession(FakeSession):
    """FakeSession that remembers which MCP tools were really called."""
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        return await super().call_tool(name, arguments)


class BrokenSession(FakeSession):
    async def list_tools(self):
        raise ConnectionError("MCP server unavailable")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("multi_data")
    generate_data.generate_all(folder)
    return folder


def run(coro):
    return asyncio.run(coro)


def finding_text(observations, issues=(), hypotheses=(), limitations=()):
    def block(title, items):
        return title + ":\n" + "\n".join(f"- {item}" for item in items)
    return "\n".join([block("OBSERVATIONS", observations), block("ISSUES", issues),
                      block("HYPOTHESES", hypotheses), block("LIMITATIONS", limitations)])


def plan(*agents):
    return final_answer(json.dumps({"agents": list(agents), "reason": "test plan"}))


def income_rate(data_dir):
    quality = inv.check_data_quality(data_dir / "baseline.csv", data_dir / "current.csv")
    return quality["issues"][0]["current_rate"]          # a real number from the evidence


def data_script(rate):
    return [tool_calls(("d1", "check_data_quality", "{}")),
            final_answer(finding_text([f"check_data_quality: income missing rate is {rate}"],
                                      issues=["income values missing"],
                                      hypotheses=["an upstream feed may have changed"]))]


def model_script():
    return [tool_calls(("m1", "compare_model_metrics", "{}")),
            final_answer(finding_text(["compare_model_metrics: recall degraded"],
                                      issues=["recall dropped"]))]


def requests_of(llm, role):
    """LLM requests made by one agent, found by its system prompt."""
    return [r for r in llm.requests if r["messages"][0]["content"].startswith(f"You are the {role}")]


def tool_names(request):
    return {t["function"]["name"] for t in request["tools"]}


# ------------------------------------------------------------------ roles and permissions
def test_data_agent_only_has_data_tools(data_dir):
    llm = FakeGroq(data_script("x"))
    run(ma.run_specialist("data_agent", "incident", FakeSession(data_dir), llm, "fake-model"))
    assert tool_names(requests_of(llm, "Data Agent")[0]) == set(config.DATA_AGENT_TOOLS)
    assert "compare_model_metrics" not in tool_names(requests_of(llm, "Data Agent")[0])


def test_model_agent_only_has_model_tools(data_dir):
    llm = FakeGroq(model_script())
    run(ma.run_specialist("model_agent", "incident", FakeSession(data_dir), llm, "fake-model"))
    assert tool_names(requests_of(llm, "Model Agent")[0]) == {"compare_model_metrics"}


def test_specialists_have_different_responsibilities():
    assert ma.SPECIALISTS["data_agent"]["prompt"] != ma.SPECIALISTS["model_agent"]["prompt"]
    assert not set(config.DATA_AGENT_TOOLS) & set(config.MODEL_AGENT_TOOLS)


def test_data_agent_cannot_call_model_tools(data_dir):
    session = RecordingSession(data_dir)
    llm = FakeGroq([tool_calls(("d0", "compare_model_metrics", "{}"))] + data_script("x"))
    finding = run(ma.run_specialist("data_agent", "incident", session, llm, "fake-model"))
    assert "compare_model_metrics" not in session.calls            # never reached MCP
    assert session.calls == ["check_data_quality"]
    rejection = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "not permitted" in rejection
    assert any("not permitted" in item for item in finding["limitations"])


def test_model_agent_cannot_call_data_tools(data_dir):
    session = RecordingSession(data_dir)
    llm = FakeGroq([tool_calls(("m0", "detect_drift", "{}"))] + model_script())
    finding = run(ma.run_specialist("model_agent", "incident", session, llm, "fake-model"))
    assert session.calls == ["compare_model_metrics"]
    assert any("detect_drift" in item for item in finding["limitations"])


# ------------------------------------------------------------------ specialists investigate
def test_specialist_returns_structured_finding(data_dir):
    rate = income_rate(data_dir)
    llm = FakeGroq(data_script(rate))
    finding = run(ma.run_specialist("data_agent", "incident", FakeSession(data_dir), llm, "fake-model"))
    assert finding["status"] == "ok" and finding["agent"] == "data_agent"
    for key in ma.FINDING_KEYS:
        assert isinstance(finding[key], list)
    assert finding["evidence_sources"] == ["check_data_quality"]
    assert finding["evidence"] and finding["unverified_numbers"] == []


def test_both_specialists_investigate_independently(data_dir):
    data_session, model_session = RecordingSession(data_dir), RecordingSession(data_dir)
    data_finding = run(ma.run_specialist("data_agent", "incident", data_session,
                                         FakeGroq(data_script("x")), "fake-model"))
    model_finding = run(ma.run_specialist("model_agent", "incident", model_session,
                                          FakeGroq(model_script()), "fake-model"))
    assert data_session.calls == ["check_data_quality"]
    assert model_session.calls == ["compare_model_metrics"]
    assert data_finding["evidence_sources"] != model_finding["evidence_sources"]


def test_specialist_without_any_tool_call_is_failed(data_dir):
    llm = FakeGroq([final_answer(finding_text(["I guess the data is bad"]))])
    finding = run(ma.run_specialist("data_agent", "incident", FakeSession(data_dir), llm, "fake-model"))
    assert finding["status"] == "failed" and finding["observations"] == []


def test_unstructured_answer_is_tolerated(data_dir):
    llm = FakeGroq([tool_calls(("d1", "check_data_quality", "{}")), final_answer("free text without any sections")])
    finding = run(ma.run_specialist("data_agent", "incident", FakeSession(data_dir), llm, "fake-model"))
    assert finding["status"] == "ok"
    assert any("expected format" in item for item in finding["limitations"])


def test_parse_finding_reads_labelled_sections_and_keeps_numbers():
    text = ("## OBSERVATIONS:\n- income rate is 0.1894\n1. second item\n**ISSUES**\n* missing values\n"
            "HYPOTHESES:\n- feed changed\nLIMITATIONS:\n- none")
    finding = ma.parse_finding(text)
    assert finding["observations"] == ["income rate is 0.1894", "second item"]
    assert finding["issues_detected"] == ["missing values"]
    assert finding["hypotheses"] == ["feed changed"] and finding["limitations"] == []


def test_last_step_tells_the_model_to_answer_without_tools(data_dir):
    llm = FakeGroq([tool_calls(("c1", "detect_drift", "{}")), final_answer("done")])
    run(agent.run_mcp_loop(FakeSession(data_dir), llm, "fake-model", "system", "user", max_iterations=2))
    assert llm.requests[1]["tool_choice"] == "none"
    assert "no tool calls left" in llm.requests[1]["messages"][-1]["content"]
    assert not any("no tool calls left" in m["content"] for m in llm.requests[0]["messages"])


# ------------------------------------------------------------------ supervisor
def test_supervisor_collects_findings_and_writes_report(data_dir):
    rate = income_rate(data_dir)
    llm = FakeGroq([plan("data_agent", "model_agent")] + data_script(rate) + model_script()
                   + [final_answer(GOOD_REPORT.format(rate=rate))])
    result = run(ma.run_supervisor("churn dropped", FakeSession(data_dir), llm, "fake-model"))

    assert result["plan"]["agents"] == ["data_agent", "model_agent"]
    assert set(result["findings"]) == {"data_agent", "model_agent"}
    assert result["failed_components"] == [] and not result["used_fallback"]
    assert result["missing_sections"] == [] and result["unverified_numbers"] == []
    assert result["report"].startswith("## Incident Summary")

    synthesis = llm.requests[-1]                      # supervisor's final request
    briefing = synthesis["messages"][1]["content"]
    assert "income missing rate" in briefing and "compare_model_metrics" in briefing
    assert "tools" not in synthesis and "tools" not in llm.requests[0]   # supervisor has no tools


def test_supervisor_can_choose_a_single_specialist(data_dir):
    session = RecordingSession(data_dir)
    llm = FakeGroq([plan("model_agent")] + model_script() + [final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(ma.run_supervisor("incident", session, llm, "fake-model"))
    assert list(result["findings"]) == ["model_agent"]
    assert session.calls == ["compare_model_metrics"]
    assert '"not_consulted":["data_agent"]' in llm.requests[-1]["messages"][1]["content"]


def test_unreadable_plan_falls_back_to_all_specialists(data_dir):
    llm = FakeGroq([final_answer("banana")] + data_script("x") + model_script()
                   + [final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(ma.run_supervisor("incident", FakeSession(data_dir), llm, "fake-model"))
    assert result["plan"]["agents"] == ["data_agent", "model_agent"]


def test_invented_numbers_in_final_report_are_flagged(data_dir):
    llm = FakeGroq([plan("data_agent")] + data_script("x")
                   + [final_answer(GOOD_REPORT.format(rate="0.7777"))])
    result = run(ma.run_supervisor("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "0.7777" in result["unverified_numbers"]


# ------------------------------------------------------------------ failures
def test_specialist_failure_is_reported_to_supervisor(data_dir):
    llm = FakeGroq([plan("data_agent", "model_agent"),
                    groq_error(groq.AuthenticationError, 401)]       # data agent fails
                   + model_script() + [final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(ma.run_supervisor("incident", FakeSession(data_dir), llm, "fake-model"))
    assert result["failed_components"] == ["data_agent"]
    assert result["findings"]["data_agent"]["status"] == "failed"
    assert result["findings"]["model_agent"]["status"] == "ok"
    assert '"failed_specialists":["data_agent"]' in llm.requests[-1]["messages"][1]["content"]
    assert result["report"]                                            # report still produced


def test_mcp_failure_marks_specialists_failed(data_dir):
    llm = FakeGroq([plan("data_agent", "model_agent"), final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(ma.run_supervisor("incident", BrokenSession(data_dir), llm, "fake-model"))
    assert result["failed_components"] == ["data_agent", "model_agent"]
    assert "ConnectionError" in result["findings"]["data_agent"]["error"]


def test_supervisor_failure_uses_fallback_report(data_dir):
    llm = FakeGroq([plan("data_agent", "model_agent")] + data_script("x") + model_script()
                   + [groq_error(groq.AuthenticationError, 401)])     # synthesis call fails
    result = run(ma.run_supervisor("incident", FakeSession(data_dir), llm, "fake-model"))
    assert result["used_fallback"] and "supervisor_synthesis" in result["failed_components"]
    assert result["missing_sections"] == []
    assert "check_data_quality" in result["report"]


def test_everything_failing_still_produces_a_report(data_dir):
    auth = lambda: groq_error(groq.AuthenticationError, 401)
    llm = FakeGroq([plan("data_agent", "model_agent"), auth(), auth(), auth()])
    result = run(ma.run_supervisor("incident", FakeSession(data_dir), llm, "fake-model"))
    assert result["failed_components"] == ["data_agent", "model_agent", "supervisor_synthesis"]
    assert "data_agent failed" in result["report"] and "model_agent failed" in result["report"]
    assert result["missing_sections"] == []


# ------------------------------------------------------------------ architecture + real MCP
def test_multi_agent_does_not_import_investigation():
    source = Path(ma.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+investigation", source, re.MULTILINE)


def test_specialists_work_with_real_mcp_server_in_memory(data_dir):
    server = mcp_server.build_server(data_dir)
    llm = FakeGroq([plan("data_agent", "model_agent")] + data_script("x") + model_script()
                   + [final_answer(GOOD_REPORT.format(rate="x"))])

    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            return await ma.run_supervisor("incident", session, llm, "fake-model")

    result = run(go())
    assert result["findings"]["data_agent"]["evidence_sources"] == ["check_data_quality"]
    assert result["findings"]["model_agent"]["evidence_sources"] == ["compare_model_metrics"]
    assert "missing_values" in result["findings"]["data_agent"]["evidence"][0]


@pytest.mark.skipif(not (__import__("os").getenv("GROQ_API_KEY") and __import__("os").getenv("RUN_GROQ_INTEGRATION")),
                    reason="optional: set GROQ_API_KEY and RUN_GROQ_INTEGRATION=1")
@pytest.mark.skipif(not config.BASELINE_PATH.exists(), reason="run python src/generate_data.py first")
def test_real_groq_multi_agent_integration():
    result = run(ma.investigate_multi(agent.DEFAULT_INCIDENT))
    assert result["findings"] and result["report"]