import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import multi_agent

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")

FINDING = {"agent": "data_agent", "status": "ok",
           "observations": ["check_data_quality: income missing"], "issues_detected": ["income values missing"],
           "hypotheses": ["an upstream feed may have changed"], "limitations": [],
           "evidence_sources": ["check_data_quality"], "evidence": ['{"status":"issues_found"}'],
           "unverified_numbers": [], "error": None}

CANNED = {"report": "## Incident Summary\nUI test report\n## Uncertainty and Limitations\nnone",
          "plan": {"agents": ["data_agent"], "reason": "test plan"},
          "findings": {"data_agent": FINDING},
          "failed_components": [], "used_fallback": False,
          "missing_sections": [], "unverified_numbers": []}


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Incident Investigator" in at.title[0].value


def test_app_runs_an_investigation_and_shows_the_report(monkeypatch):
    async def fake_investigate(incident, **kwargs):          # no real Groq or MCP call
        return CANNED

    monkeypatch.setattr(multi_agent, "investigate_multi", fake_investigate)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("UI test report" in m.value for m in at.markdown)