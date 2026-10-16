"""Day 5: Streamlit web UI for the multi-agent incident investigator.

Run locally:  streamlit run streamlit_app.py
Deploy:       Streamlit Community Cloud (main file: streamlit_app.py)
"""
import asyncio
import contextlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")       # avoids an OpenBLAS memory error on some machines
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from groq import Groq

import agent
import config
import generate_data
import multi_agent

FINDING_TITLES = {"observations": "Observations (facts from tools)",
                  "issues_detected": "Issues detected",
                  "hypotheses": "Hypotheses (inferences, not facts)",
                  "limitations": "Limitations"}

ARCHITECTURE = """
digraph {
  rankdir=LR;
  node [shape=box, style=rounded];
  Incident -> Supervisor;
  Supervisor -> "Data Agent";
  Supervisor -> "Model Agent";
  "Data Agent" -> "MCP server" [label="3 data tools"];
  "Model Agent" -> "MCP server" [label="metrics tool"];
  "MCP server" -> "investigation.py";
  Supervisor -> Report;
}
"""


# ------------------------------------------------------------ setup helpers
@st.cache_resource
def ensure_data():
    """Create the demo data the first time the app starts (also on a fresh deployment)."""
    if not (config.BASELINE_PATH.exists() and config.CURRENT_PATH.exists()
            and config.METRICS_PATH.exists()):
        generate_data.generate_all()
    return True


@st.cache_resource
def run_lock():
    """Only one investigation at a time: protects the free-tier rate limit."""
    return threading.Lock()


def configured_key():
    """Key set by the app owner (Streamlit secrets) or the environment."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


class LiveLog(io.StringIO):
    """Receives the agents' print() output and shows it live in the page."""
    def __init__(self, box):
        super().__init__()
        self.box = box

    def write(self, text):
        written = super().write(text)
        self.box.code(self.getvalue() or " ", language="text")
        return written


def run_investigation(incident, client, model, log_box):
    with contextlib.redirect_stdout(LiveLog(log_box)):
        return asyncio.run(multi_agent.investigate_multi(incident, llm_client=client, model=model))


# ------------------------------------------------------------ rendering
def show_findings(findings):
    for name, finding in findings.items():
        icon = "✅" if finding["status"] == "ok" else "❌"
        st.subheader(f"{icon} {name.replace('_', ' ').title()}")
        st.caption("MCP tools used: " + (", ".join(finding["evidence_sources"]) or "none"))
        if finding["error"]:
            st.error(finding["error"])
        for key, title in FINDING_TITLES.items():
            if finding[key]:
                st.markdown(f"**{title}**")
                for item in finding[key]:
                    st.markdown(f"- {item}")


def show_evidence(findings):
    st.caption("Raw results returned by the MCP tools. The report's numbers are checked against these.")
    for name, finding in findings.items():
        st.subheader(name.replace("_", " ").title())
        if not finding["evidence"]:
            st.write("No tool results (this component failed).")
        for number, text in enumerate(finding["evidence"], start=1):
            with st.expander(f"Tool result {number}"):
                try:
                    st.json(json.loads(text))
                except ValueError:
                    st.code(text)


def show_checks(result):
    if result["failed_components"]:
        st.error(f"Failed components: {', '.join(result['failed_components'])}. "
                 "The report explains what is missing.")
    else:
        st.success("All components ran.")
    if result["used_fallback"]:
        st.warning("The Supervisor's own LLM call failed, so this report was assembled in plain code.")
    if result["missing_sections"]:
        st.warning(f"Report is missing sections: {', '.join(result['missing_sections'])}")
    if result["unverified_numbers"]:
        st.warning("Numbers in the report that were not found in any tool result "
                   f"(the LLM may have rounded or invented them): {', '.join(result['unverified_numbers'])}")
    else:
        st.success("Every number in the report appears in the tool output.")


def render_result(result):
    findings = result["findings"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Specialists consulted", len(findings))
    col2.metric("Failed components", len(result["failed_components"]))
    col3.metric("Unverified numbers", len(result["unverified_numbers"]))
    st.caption(f"Supervisor plan: {', '.join(result['plan']['agents'])}. {result['plan']['reason']}")

    report_tab, findings_tab, evidence_tab, checks_tab = st.tabs(
        ["Report", "Specialist findings", "Evidence (tool output)", "Guardrail checks"])
    with report_tab:
        st.markdown(result["report"])
    with findings_tab:
        show_findings(findings)
    with evidence_tab:
        show_evidence(findings)
    with checks_tab:
        show_checks(result)


# ------------------------------------------------------------ the page
def handle_click(incident, typed_key, model):
    key = typed_key.strip() or configured_key()
    if not key:
        st.error("Add a free Groq API key in the sidebar (https://console.groq.com/keys).")
        return
    if not incident.strip():
        st.warning("Describe the incident first.")
        return
    wait = config.UI_COOLDOWN_SECONDS - (time.time() - st.session_state.get("last_run", 0))
    if wait > 0:
        st.warning(f"Please wait {int(wait) + 1} seconds before the next run.")
        return
    lock = run_lock()
    if not lock.acquire(blocking=False):
        st.warning("Another investigation is running. Please try again in a minute.")
        return

    try:
        with st.status("Agents are investigating (30-120 seconds on the free tier)...",
                       expanded=True) as status:
            log_box = st.empty()
            result = run_investigation(incident, Groq(api_key=key), model, log_box)
            status.update(label="Investigation complete", state="complete", expanded=False)
        st.session_state["result"] = result
    except agent.AgentError as e:
        st.error(f"Investigation failed: {agent.redact(str(e)).replace(key, '***')}")
    except Exception as e:
        st.error(f"Unexpected error ({type(e).__name__}). Please try again.")
    finally:
        lock.release()
        st.session_state["last_run"] = time.time()


def main():
    st.set_page_config(page_title="AI Incident Investigator", page_icon="🔍", layout="wide")
    ensure_data()

    st.title("🔍 AI Data/ML Incident Investigator")
    st.caption("A Supervisor agent coordinates a Data Agent and a Model Agent. They gather evidence "
               "through MCP tools, and every number in the report is checked against tool output.")
    with st.expander("How it works"):
        st.graphviz_chart(ARCHITECTURE)
        st.markdown("- Deterministic Python tools compute the facts (PSI drift, data quality, metrics).\n"
                    "- The tools are exposed over **MCP**; each agent can only use its own tools.\n"
                    "- The LLM (Groq) decides what to call and explains the findings.\n"
                    "- Code flags any number in the report that is not in the tool output.")

    with st.sidebar:
        st.header("Settings")
        typed_key = st.text_input("Groq API key", type="password",
                                  help="Optional if this app already has a key configured. "
                                       "Used only for your run and never stored.")
        model = st.text_input("Model", value=os.getenv("GROQ_MODEL") or config.DEFAULT_GROQ_MODEL)
        st.divider()
        st.subheader("About the data")
        st.markdown("Synthetic customer-churn data with three injected problems: income values "
                    "going missing, `account_balance` inflated about 1.35x, and about 8% `UNKNOWN` "
                    "regions. A model trained on the healthy baseline loses recall on the new data.")

    choice = st.selectbox("Try an example", ["(write your own)"] + config.EXAMPLE_INCIDENTS)
    incident = st.text_area("Incident", value=config.EXAMPLE_INCIDENTS[0] if choice == "(write your own)" else choice,
                            max_chars=config.UI_MAX_INCIDENT_CHARS, height=90)

    if st.button("Investigate", type="primary"):
        handle_click(incident, typed_key, model)

    if "result" in st.session_state:
        render_result(st.session_state["result"])


main()