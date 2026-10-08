import asyncio
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent
import config
import generate_data
import investigation as inv      # tests may use it to build realistic fake evidence
import mcp_server

TOOL_NAMES = ["get_dataset_stats", "check_data_quality", "detect_drift", "compare_model_metrics"]

GOOD_REPORT = """## Incident Summary
Churn model quality dropped.
## Evidence (Facts from tools)
- FACT (check_data_quality): income missing rate is now {rate}
## Detected Issues
- Missing income values
## Likely Root Cause (Hypothesis)
- HYPOTHESIS: an upstream data feed changed.
## Recommended Next Action
Check the income data pipeline.
## Uncertainty and Limitations
The cause cannot be confirmed from these tools."""


# ------------------------------------------------------------------ fakes
class FakeGroq:
    """Plays the Groq client: returns scripted responses (or raises scripted errors)."""
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_calls(*calls):
    """calls = (id, tool_name, arguments_json_string)"""
    message = SimpleNamespace(content=None, tool_calls=[
        SimpleNamespace(id=i, function=SimpleNamespace(name=n, arguments=a)) for i, n, a in calls])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def final_answer(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=None))])


def groq_error(cls, status):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


class FakeSession:
    """Plays an MCP ClientSession; its results come from the real Day-1 functions."""
    def __init__(self, data_dir, fail=False, is_error=False):
        self.dir, self.fail, self.is_error = Path(data_dir), fail, is_error

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(
            name=n, description=f"UNIQUE description of {n}",
            inputSchema={"type": "object", "properties": {}}) for n in TOOL_NAMES])

    async def call_tool(self, name, arguments):
        if self.fail:
            raise RuntimeError("server died")
        if self.is_error:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text="boom")])
        base, cur = self.dir / "baseline.csv", self.dir / "current.csv"
        data = {"check_data_quality": lambda: inv.check_data_quality(base, cur),
                "detect_drift": lambda: inv.detect_drift(base, cur),
                "compare_model_metrics": lambda: inv.compare_model_metrics(self.dir / "model_metrics.json"),
                "get_dataset_stats": lambda: inv.get_dataset_stats(self.dir / f"{arguments['dataset']}.csv"),
                }[name]()
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(data, indent=2))])


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("agent_data")
    generate_data.generate_all(folder)
    return folder


def run(coro):
    return asyncio.run(coro)


def tool_messages(request):
    return [m for m in request["messages"] if m["role"] == "tool"]


# ------------------------------------------------------------------ Groq configuration
def test_missing_api_key_is_reported_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(agent.AgentError) as err:
        agent.make_llm_client()
    assert "GROQ_API_KEY" in str(err.value)


def test_client_is_built_from_environment_key(monkeypatch):
    seen = {}
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    monkeypatch.setattr(agent, "Groq", lambda **kwargs: seen.update(kwargs) or "client")
    assert agent.make_llm_client() == "client"
    assert seen == {"api_key": "test-key-123"}


def test_default_model_and_env_override():
    assert config.DEFAULT_GROQ_MODEL == "openai/gpt-oss-120b"


def test_agent_does_not_import_investigation():
    source = Path(agent.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+investigation", source, re.MULTILINE)


# ------------------------------------------------------------------ discovery + loop
def test_tools_are_discovered_dynamically(data_dir):
    llm = FakeGroq([final_answer(GOOD_REPORT.format(rate="x"))])
    run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    sent = llm.requests[0]["tools"]
    assert [t["function"]["name"] for t in sent] == TOOL_NAMES
    assert all(t["type"] == "function" for t in sent)
    assert all("UNIQUE description" in t["function"]["description"] for t in sent)


def test_tool_result_is_fed_back_and_report_returned(data_dir):
    quality = inv.check_data_quality(data_dir / "baseline.csv", data_dir / "current.csv")
    rate = quality["issues"][0]["current_rate"]          # a real number from the evidence
    llm = FakeGroq([tool_calls(("call_1", "check_data_quality", "{}")),
                    final_answer(GOOD_REPORT.format(rate=rate))])
    result = run(agent.run_agent("churn dropped", FakeSession(data_dir), llm, "fake-model"))

    second_request_tool_msgs = tool_messages(llm.requests[1])
    assert second_request_tool_msgs[0]["tool_call_id"] == "call_1"
    assert "missing_values" in second_request_tool_msgs[0]["content"]
    assert result["tools_called"] == ["check_data_quality"]
    assert result["iterations"] == 2
    assert result["missing_sections"] == []
    assert result["unverified_numbers"] == []


def test_agent_can_make_several_rounds_of_tool_calls(data_dir):
    llm = FakeGroq([tool_calls(("c1", "check_data_quality", "{}")),
                    tool_calls(("c2", "detect_drift", "{}"), ("c3", "compare_model_metrics", "{}")),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert result["tools_called"] == ["check_data_quality", "detect_drift", "compare_model_metrics"]
    assert result["iterations"] == 3


def test_tool_with_arguments_is_called(data_dir):
    llm = FakeGroq([tool_calls(("c1", "get_dataset_stats", '{"dataset": "current"}')),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "5000" in tool_messages(llm.requests[1])[0]["content"]


def test_last_step_forbids_tools_so_a_report_is_produced(data_dir):
    llm = FakeGroq([tool_calls(("c1", "detect_drift", "{}")),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model", max_iterations=2))
    assert llm.requests[0]["tool_choice"] == "auto"
    assert llm.requests[1]["tool_choice"] == "none"
    assert result["iterations"] == 2


def test_loop_stops_if_model_keeps_asking_for_tools(data_dir):
    llm = FakeGroq([tool_calls(("c1", "detect_drift", "{}")),
                    tool_calls(("c2", "detect_drift", "{}"))])
    with pytest.raises(agent.AgentError):
        run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model", max_iterations=2))


# ------------------------------------------------------------------ report quality checks
def test_invented_numbers_are_flagged(data_dir):
    report = GOOD_REPORT.format(rate="0.7777")           # not present in any tool result
    llm = FakeGroq([tool_calls(("c1", "check_data_quality", "{}")), final_answer(report)])
    result = run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "0.7777" in result["unverified_numbers"]


def test_missing_report_sections_are_flagged(data_dir):
    llm = FakeGroq([final_answer("Everything looks fine.")])
    result = run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "detected issues" in result["missing_sections"]


# ------------------------------------------------------------------ error handling: tools
def test_malformed_tool_arguments_do_not_crash(data_dir):
    llm = FakeGroq([tool_calls(("c1", "detect_drift", "{not json")),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "error" in tool_messages(llm.requests[1])[0]["content"]
    assert result["report"]


def test_unknown_tool_name_does_not_crash(data_dir):
    llm = FakeGroq([tool_calls(("c1", "delete_everything", "{}")),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "Unknown tool" in tool_messages(llm.requests[1])[0]["content"]


def test_mcp_tool_exception_is_returned_to_llm(data_dir):
    llm = FakeGroq([tool_calls(("c1", "detect_drift", "{}")),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(agent.run_agent("incident", FakeSession(data_dir, fail=True), llm, "fake-model"))
    assert "MCP tool call failed" in tool_messages(llm.requests[1])[0]["content"]
    assert result["report"]


def test_mcp_error_result_is_returned_to_llm(data_dir):
    llm = FakeGroq([tool_calls(("c1", "detect_drift", "{}")),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    run(agent.run_agent("incident", FakeSession(data_dir, is_error=True), llm, "fake-model"))
    assert "Tool reported an error" in tool_messages(llm.requests[1])[0]["content"]


# ------------------------------------------------------------------ error handling: Groq
def test_unexpected_llm_response_is_an_agent_error(data_dir):
    llm = FakeGroq([SimpleNamespace(choices=[])])
    with pytest.raises(agent.AgentError):
        run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))


def test_empty_final_answer_is_an_agent_error(data_dir):
    llm = FakeGroq([final_answer("   ")])
    with pytest.raises(agent.AgentError):
        run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))


def test_rate_limit_is_retried_then_succeeds(data_dir, monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    llm = FakeGroq([groq_error(groq.RateLimitError, 429), final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert result["report"] and len(llm.requests) == 2


def test_rate_limit_gives_up_with_clear_message(data_dir, monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    llm = FakeGroq([groq_error(groq.RateLimitError, 429)] * (config.GROQ_RATE_LIMIT_RETRIES + 1))
    with pytest.raises(agent.AgentError) as err:
        run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "rate limit" in str(err.value).lower()


def test_invalid_api_key_error_does_not_leak_the_key(data_dir, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key-abc")
    llm = FakeGroq([groq_error(groq.AuthenticationError, 401)])
    with pytest.raises(agent.AgentError) as err:
        run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "secret-key-abc" not in str(err.value)
    assert "API key" in str(err.value)


def test_groq_connection_failure_is_an_agent_error(data_dir):
    request = httpx.Request("POST", "https://api.groq.com")
    llm = FakeGroq([groq.APIConnectionError(request=request)])
    with pytest.raises(agent.AgentError) as err:
        run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "Groq" in str(err.value)


def test_groq_status_error_is_an_agent_error(data_dir):
    llm = FakeGroq([groq_error(groq.BadRequestError, 400)])
    with pytest.raises(agent.AgentError) as err:
        run(agent.run_agent("incident", FakeSession(data_dir), llm, "fake-model"))
    assert "400" in str(err.value)


def test_missing_mcp_server_script_is_an_agent_error(tmp_path):
    with pytest.raises(agent.AgentError) as err:
        run(agent.investigate("incident", server_script=tmp_path / "nope.py",
                              llm_client=FakeGroq([]), model="fake-model"))
    assert "not found" in str(err.value)


# ------------------------------------------------------------------ real MCP (Day-2 server)
def test_agent_works_with_real_mcp_server_in_memory(data_dir):
    server = mcp_server.build_server(data_dir)
    llm = FakeGroq([tool_calls(("c1", "detect_drift", "{}")),
                    final_answer(GOOD_REPORT.format(rate="x"))])

    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            return await agent.run_agent("incident", session, llm, "fake-model")

    result = run(go())
    assert {t["function"]["name"] for t in llm.requests[0]["tools"]} == set(TOOL_NAMES)
    assert "account_balance" in tool_messages(llm.requests[1])[0]["content"]
    assert result["tools_called"] == ["detect_drift"]


@pytest.mark.skipif(not config.BASELINE_PATH.exists(), reason="run python src/generate_data.py first")
def test_agent_connects_to_real_server_over_stdio():
    llm = FakeGroq([tool_calls(("c1", "check_data_quality", "{}")),
                    final_answer(GOOD_REPORT.format(rate="x"))])
    result = run(agent.investigate("incident", llm_client=llm, model="fake-model"))
    assert result["tools_called"] == ["check_data_quality"]
    assert "missing_values" in tool_messages(llm.requests[1])[0]["content"]


# ------------------------------------------------------------------ optional: real Groq call
@pytest.mark.skipif(not (os.getenv("GROQ_API_KEY") and os.getenv("RUN_GROQ_INTEGRATION")),
                    reason="optional: set GROQ_API_KEY and RUN_GROQ_INTEGRATION=1")
@pytest.mark.skipif(not config.BASELINE_PATH.exists(), reason="run python src/generate_data.py first")
def test_real_groq_integration():
    result = run(agent.investigate(agent.DEFAULT_INCIDENT))
    assert result["tools_called"]
    assert result["missing_sections"] == []