"""Day 4: a Supervisor agent that coordinates a Data Agent and a Model Agent.

Flow:  incident -> Supervisor picks specialists -> each specialist investigates through MCP
       (own prompt, own permitted tools) -> structured findings -> Supervisor writes the report

Groq client, MCP connection, the tool loop and the number check are reused from agent.py.
Like agent.py, this file must NOT import investigation.py.

Run:  python src/multi_agent.py "Churn model recall dropped this week"
"""
import asyncio
import json
import os
import re
import sys

import agent
import config
from agent import AgentError

FINDING_KEYS = ["observations", "issues_detected", "hypotheses", "limitations"]

# ------------------------------------------------------------ prompts
NUMBER_RULE = ("Every number must be copied EXACTLY from a tool result, digit for digit "
               "(write 0.1894, never 18.9% or 19%; write 0.5383, never 0.538). Never calculate, "
               "round, convert or estimate numbers, and never add numbers of your own such as "
               "thresholds. Do not invent facts that the tools did not return.")

FINDING_FORMAT = """When you have finished investigating, write your final answer in EXACTLY this plain-text layout:
OBSERVATIONS:
- facts taken from tool results (mention the tool name)
ISSUES:
- problems the tool results clearly show
HYPOTHESES:
- your own inferences about causes (these are NOT facts)
LIMITATIONS:
- what you could not check, or are unsure about
Put one item per line starting with "- ". If a section is empty write "- none"."""

DATA_AGENT_PROMPT = f"""You are the Data Agent in an ML incident investigation team.
Your job: find DATA causes. Use your tools to check data quality (missing values, unknown
categories) and feature drift between baseline and current data. Request all the tools you
need together in ONE step, and only use get_dataset_stats if you need extra detail.
You cannot see model metrics, so do not state conclusions about model performance.
{NUMBER_RULE}
{FINDING_FORMAT}"""

MODEL_AGENT_PROMPT = f"""You are the Model Agent in an ML incident investigation team.
Your job: find MODEL-PERFORMANCE causes. Use your tool to compare baseline and current model metrics.
Say which metrics degraded and what the pattern suggests (label that as a hypothesis).
You cannot see the data, so do not state data problems as facts.
{NUMBER_RULE}
{FINDING_FORMAT}"""

PLAN_PROMPT = """You are the Supervisor of an ML incident investigation team.
Your team:
- data_agent: investigates data quality problems and data drift
- model_agent: investigates model performance metrics
Decide which specialists this incident needs. If unsure, choose both.
Reply with ONLY JSON: {"agents": ["data_agent", "model_agent"], "reason": "<one sentence>"}"""

SUPERVISOR_PROMPT = f"""You are the Supervisor of an ML incident investigation team.
Your specialists have already investigated. Their findings are given to you as JSON.
Write the final investigation report.

Rules:
1. Use ONLY the specialists' findings. You have no tools. Do not invent facts.
2. Every number must be copied EXACTLY, digit for digit, from the findings (write 0.1894, never 18.9% or 19%; write 0.5383, never 0.538). Never calculate, round, convert or add numbers of your own, including thresholds in recommendations (do not write things like 'alert above 5%').
3. FACTS come from findings (name the specialist and its evidence_sources). Your own inferences are HYPOTHESES: label them.
4. If specialists disagree, say so and state which view the evidence supports better, or that it is unresolved.
5. If a specialist failed or was not consulted, say so under Uncertainty and Limitations. Do not guess what it would have found.
6. A data problem and a model drop that occur together are only a HYPOTHESIS of cause and effect, not proof.

Write the report in Markdown using exactly these headings, in this order:
{agent.REPORT_HEADINGS}"""

# Each specialist has its own prompt AND its own permitted MCP tools (set in config.py).
SPECIALISTS = {
    "data_agent": {"prompt": DATA_AGENT_PROMPT, "tools": config.DATA_AGENT_TOOLS},
    "model_agent": {"prompt": MODEL_AGENT_PROMPT, "tools": config.MODEL_AGENT_TOOLS},
}


# ------------------------------------------------------------ findings
SECTION_TO_KEY = {"OBSERVATIONS": "observations", "ISSUES": "issues_detected",
                  "HYPOTHESES": "hypotheses", "LIMITATIONS": "limitations"}
BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")      # "- ", "* ", "1. " (keeps numbers like 0.1894)


def parse_finding(text):
    """Read the specialist's plain-text answer (OBSERVATIONS: / ISSUES: / ...)."""
    finding = {key: [] for key in FINDING_KEYS}
    current = None
    for line in text.splitlines():
        heading = line.strip().strip("#*: ").upper()         # tolerates "## OBSERVATIONS:" or "**Issues**"
        if heading in SECTION_TO_KEY:
            current = SECTION_TO_KEY[heading]
            continue
        item = BULLET.sub("", line).strip()
        if current and item and item.lower() != "none":
            finding[current].append(item)
    if not any(finding.values()):                            # no recognisable sections
        finding["observations"] = [text.strip()[:1500]]
        finding["limitations"] = ["The specialist did not use the expected format; its raw "
                                  "answer is shown as an observation."]
    return finding


def failed_finding(name, reason):
    if isinstance(reason, AgentError):
        reason = agent.redact(str(reason))
    elif not isinstance(reason, str):
        reason = f"{type(reason).__name__} while investigating"
    return {"agent": name, "status": "failed", "observations": [], "issues_detected": [],
            "hypotheses": [], "limitations": [f"{name} failed: {reason}"],
            "evidence_sources": [], "evidence": [], "unverified_numbers": [], "error": reason}


async def run_specialist(name, incident, session, llm_client, model):
    """Run one specialist. Never raises: a failure becomes a finding with status 'failed'."""
    spec = SPECIALISTS[name]
    print(f"[supervisor] asking {name} to investigate")
    try:
        loop = await agent.run_mcp_loop(
            session, llm_client, model, spec["prompt"], f"Incident: {incident}",
            allowed_tools=spec["tools"], max_iterations=config.SPECIALIST_MAX_ITERATIONS,
            label=name)
    except Exception as e:                                   # Groq, MCP or loop failure
        return failed_finding(name, e)
    if not loop["tools_called"]:
        return failed_finding(name, "it called no MCP tool, so it has no evidence.")

    parsed = parse_finding(loop["text"])
    limitations = parsed["limitations"] + [
        f"Requested tool '{tool}' is not permitted for {name}; the request was rejected."
        for tool in loop["rejected_tools"]]
    all_text = " ".join(item for key in FINDING_KEYS for item in parsed[key])
    return {"agent": name, "status": "ok", **parsed, "limitations": limitations,
            "evidence_sources": sorted(set(loop["tools_called"])),
            "evidence": loop["evidence"],            # raw tool results, used for number checking
            "unverified_numbers": agent.find_unverified_numbers(all_text, loop["evidence"] + [incident]),
            "error": None}


# ------------------------------------------------------------ supervisor
def decide_specialists(llm_client, model, incident):
    """Supervisor step 1: choose the relevant specialists. If unsure or broken, use ALL."""
    everyone = list(SPECIALISTS)
    try:
        response = agent.ask_llm(llm_client, model,
                                 [{"role": "system", "content": PLAN_PROMPT},
                                  {"role": "user", "content": f"Incident: {incident}"}])
        text = response.choices[0].message.content or ""
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        chosen = [name for name in everyone if name in data["agents"]]
        reason = str(data.get("reason", ""))[:200]
    except (AgentError, AttributeError, IndexError, TypeError, KeyError, ValueError):
        return everyone, "The plan could not be read, so all specialists were consulted."
    if not chosen:
        return everyone, "The plan named no valid specialist, so all specialists were consulted."
    return chosen, reason


def fallback_report(incident, findings, reason):
    """Plain-code report used only if the supervisor's own LLM call fails."""
    def bullets(key, empty):
        lines = [f"- ({name}) {item}" for name, f in findings.items() for item in f[key]]
        return lines or [empty]

    sources = [f"- {name}: tools used = {f['evidence_sources'] or 'none'} ({f['status']})"
               for name, f in findings.items()]
    return "\n".join([
        "## Incident Summary", incident,
        f"Note: the supervisor could not write its synthesis ({reason}). "
        "This report was assembled directly from the specialist findings.",
        "## Evidence (Facts from tools)", *bullets("observations", "- No evidence was gathered."), *sources,
        "## Detected Issues", *bullets("issues_detected", "- None reported."),
        "## Likely Root Cause (Hypothesis)", *bullets("hypotheses", "- No hypothesis available."),
        "## Recommended Next Action",
        "Re-run the investigation once the failed components work, and review the issues above.",
        "## Uncertainty and Limitations", *bullets("limitations", "- None reported."),
    ])


async def run_supervisor(incident, session, llm_client, model):
    chosen, reason = decide_specialists(llm_client, model, incident)

    findings = {}
    for name in chosen:                                      # one at a time keeps free-tier usage low
        findings[name] = await run_specialist(name, incident, session, llm_client, model)
    failed = [name for name, f in findings.items() if f["status"] == "failed"]

    briefing = json.dumps({
        "incident": incident,
        "specialist_findings": {name: {k: v for k, v in f.items() if k != "evidence"}
                                for name, f in findings.items()},
        "not_consulted": [n for n in SPECIALISTS if n not in chosen],
        "failed_specialists": failed}, separators=(",", ":"))

    used_fallback = False
    try:
        response = agent.ask_llm(llm_client, model,
                                 [{"role": "system", "content": SUPERVISOR_PROMPT},
                                  {"role": "user", "content": briefing}])
        report = (response.choices[0].message.content or "").strip()
        if not report:
            raise AgentError("the supervisor returned an empty report")
    except (AgentError, AttributeError, IndexError, TypeError) as e:
        used_fallback = True
        report = fallback_report(incident, findings, agent.redact(str(e)))

    all_evidence = [text for f in findings.values() for text in f["evidence"]] + [incident]
    return {"report": report,
            "plan": {"agents": chosen, "reason": reason},
            "findings": findings,
            "failed_components": failed + (["supervisor_synthesis"] if used_fallback else []),
            "used_fallback": used_fallback,
            "missing_sections": agent.find_missing_sections(report),
            "unverified_numbers": agent.find_unverified_numbers(report, all_evidence)}


async def investigate_multi(incident, server_script=config.SERVER_SCRIPT, llm_client=None, model=None):
    model = model or os.getenv("GROQ_MODEL") or config.DEFAULT_GROQ_MODEL
    llm_client = llm_client or agent.make_llm_client()
    return await agent.run_with_mcp(
        server_script, lambda session: run_supervisor(incident, session, llm_client, model))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    incident = " ".join(sys.argv[1:]).strip() or agent.DEFAULT_INCIDENT
    print(f"Incident: {incident}\n")
    try:
        result = asyncio.run(investigate_multi(incident))
    except AgentError as e:
        print(f"Agent error: {agent.redact(str(e))}", file=sys.stderr)
        sys.exit(1)

    print("\n" + result["report"])
    print(f"\n--- Supervisor plan: {result['plan']['agents']} ({result['plan']['reason']})")
    for name, f in result["findings"].items():
        print(f"--- {name}: {f['status']}, MCP tools used: {f['evidence_sources']}")
        if f["error"]:
            print(f"    error: {f['error']}")
        else:
            for note in f["limitations"]:
                print(f"    limitation: {note}")
        if f["unverified_numbers"]:
            print(f"    numbers not found in its tool results: {f['unverified_numbers']}")
    if result["failed_components"]:
        print(f"WARNING: failed components: {result['failed_components']}")
    if result["missing_sections"]:
        print(f"WARNING: report is missing sections: {result['missing_sections']}")
    if result["unverified_numbers"]:
        print("WARNING: numbers not found in any tool result (check them): "
              f"{result['unverified_numbers']}")


if __name__ == "__main__":
    main()