# AI Data/ML Incident Investigator

A multi-agent system that investigates machine-learning and data incidents. A **Supervisor**
agent coordinates a **Data Agent** and a **Model Agent**. They gather evidence through
**MCP (Model Context Protocol)** tools and produce an evidence-based incident report. A
Streamlit web app lets anyone try it in the browser.

**Live demo:** https://agent-ml-incident-investigator-4nwezfjabdqvdc4jpvxkbb.streamlit.app/


> Portfolio project on **synthetic data**, built with free tools only (Groq free tier,
> open-source Python libraries, free hosting). It demonstrates agent design, MCP, tool
> permissions, and guardrails against invented numbers. It is not a production system.

## What it does

You describe an incident ("Churn model recall dropped this week. What happened?"). The system:

1. The **Supervisor** decides which specialists are needed.
2. The **Data Agent** and **Model Agent** each investigate with their own tools through MCP.
3. Each returns a structured finding (observations, issues, hypotheses, limitations).
4. The Supervisor writes a six-section report: summary, evidence, issues, likely root cause,
   next action, limitations.
5. Code checks every number in the report against the raw tool output and flags any that
   are not found there.

## Architecture

```
                         Incident (text, from the browser or CLI)
                               │
                        ┌──────▼───────┐
                        │  Supervisor  │  1) plans which specialists are needed
                        └──┬────────┬──┘
                           │        │
                ┌──────────▼──┐  ┌──▼───────────┐
                │ Data Agent  │  │ Model Agent  │   own prompt, own tools, own conversation
                └──────┬──────┘  └──────┬───────┘
     get_dataset_stats │                │ compare_model_metrics
     check_data_quality│                │
     detect_drift      │                │
                ┌──────▼────────────────▼───────┐
                │   MCP server (local, stdio)   │
                └───────────────┬───────────────┘
                                │
                  src/investigation.py  (deterministic Python: pandas / numpy / PSI)

   specialist findings flow back up → Supervisor writes the final six-section report
```

**Why multiple agents?**
- **Least privilege:** each specialist can only call its own tools, enforced in code.
- **Focus:** short, role-specific prompts keep each LLM on its task.
- **Failure isolation:** if one component fails, the report still explains what is missing.
- **Reconciliation:** only the Supervisor sees both findings. It connects a data problem to a
  model drop and labels that link as a hypothesis, not proof.

All agents use the same Groq model. They differ in role, prompt, tools and context.

## The web app

The Streamlit app (`streamlit_app.py`) runs the full path in the browser and shows:
- a **live agent trace** while the agents work,
- the **final report**,
- each **specialist's findings** (status, tools used, observations, hypotheses, limitations),
- the **raw MCP tool output** the report is checked against,
- **guardrail checks** (failed components, missing sections, numbers not found in tool output).

Demo-safety choices: one investigation at a time, a cooldown between runs, a 300-character
incident limit, and API keys are never stored. Visitors can paste their own free Groq key in
the sidebar. The demo data is generated automatically on first start.

## What was built, day by day

**Day 1: Deterministic investigation layer** (no LLM)
- Reproducible customer-churn incident: income missing rate ~2% to ~20%, `account_balance`
  inflated ~1.35x with noise, ~8% `UNKNOWN` region. Four control features stay stable.
- One scikit-learn Logistic Regression trained on baseline data. `baseline_metrics` and
  `current_metrics` are computed, not hard-coded.
- Four functions: `get_dataset_stats`, `check_data_quality`, `detect_drift` (PSI),
  `compare_model_metrics`. All thresholds live in `src/config.py`.

**Day 2: MCP server**
- The four functions are exposed as MCP tools (official Python SDK) with descriptions and
  input schemas. Tools take no file paths; invalid input and missing files return clean errors.

**Day 3: Single Groq agent**
- An MCP client discovers tools dynamically. The Groq model chooses tools, receives results
  and writes the report, separating tool facts from hypotheses.
- Handles rate limits (retry), invalid keys, API and MCP failures, and malformed tool calls.

**Day 4: Multi-agent system**
- Supervisor plans, runs the Data Agent and Model Agent, and synthesizes their findings.
- Role-based permissions: Data Agent gets `get_dataset_stats`, `check_data_quality`,
  `detect_drift`; Model Agent gets `compare_model_metrics`. Unauthorized requests are rejected
  before they reach MCP.
- A specialist that calls no tool is marked failed. If a specialist or the Supervisor's own
  LLM call fails, the report still explains what failed (plain-code fallback report).

**Day 5: Web UI and deployment**
- Streamlit app with live trace, findings, evidence and guardrail tabs.
- Deployed on Streamlit Community Cloud. Optional CI runs the tests on every push.

## Example result (from a real run)

| Evidence (from MCP tools) | Baseline | Current |
|---|---|---|
| `income` missing rate | 0.0186 | 0.1894 |
| `region` UNKNOWN share | 0.0 | 0.0806 |
| `account_balance` PSI | | 0.5383 (major) |
| `region` PSI | | 0.547 (major) |
| Model recall | 0.4788 | 0.2922 |
| Model F1 | 0.5833 | 0.4201 |

The four control features (`credit_score`, `tenure_months`, `age`, `support_tickets_90d`)
are reported as stable. LLM wording varies from run to run.

## Project structure

```
agent-ml-incident-investigator/
├── README.md
├── requirements.txt
├── streamlit_app.py           # web UI
├── .gitignore
├── .github/workflows/ci.yml   # optional: runs tests on every push
├── data/                      # generated: baseline.csv, current.csv, model_metrics.json
├── src/
│   ├── config.py              # all thresholds and settings
│   ├── generate_data.py       # reproducible data + model metrics
│   ├── investigation.py       # deterministic tools (no LLM)
│   ├── mcp_server.py          # MCP server exposing the 4 tools
│   ├── agent.py               # Groq client, MCP client, generic tool loop, single agent
│   └── multi_agent.py         # Supervisor + Data Agent + Model Agent
└── tests/
    ├── test_investigation.py
    ├── test_mcp_server.py
    ├── test_agent.py
    ├── test_multi_agent.py
    └── test_streamlit_app.py
```

## Run it locally (Windows PowerShell; on macOS/Linux use `source .venv/bin/activate`)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py
```

Create a free Groq key at https://console.groq.com/keys and set it as an environment
variable. Never commit it.

```powershell
$env:GROQ_API_KEY = "your-key-here"
$env:GROQ_MODEL = "openai/gpt-oss-20b"    # optional; this is the default
```

Web app:

```powershell
streamlit run streamlit_app.py
```

If `streamlit` is not found inside a virtual environment, use `python -m streamlit run streamlit_app.py`.

Command line:

```powershell
python src/multi_agent.py "Churn model recall dropped this week. What happened?"   # multi-agent
python src/agent.py "Churn model recall dropped this week. What happened?"         # single agent
python src/mcp_server.py                                                           # MCP server only (stdio)
```

## Test

```powershell
python -m pytest
```

All tests run offline with fake Groq and MCP components, so they need no network or API key.
Two optional tests make real Groq calls and are skipped unless you set `GROQ_API_KEY` and
`RUN_GROQ_INTEGRATION=1`.

## Deploy (Streamlit Community Cloud, free)

1. Push the repository to GitHub (public).
2. On https://share.streamlit.io choose **Create app**, select the repo and branch `main`,
   and set the main file to `streamlit_app.py`.
3. Under **Advanced settings**, choose Python 3.11 and add the secret
   `GROQ_API_KEY = "your-key"`.
4. Deploy. The demo data is generated automatically on first start.

## Guardrails and design decisions

- **Deterministic evidence, LLM explanation:** all numbers come from tested Python functions
  reached through MCP, never from the LLM.
- **Number verification:** numbers in reports are compared with tool output, and any number
  not found is flagged.
- **No file paths in tool arguments:** tools accept only fixed options such as
  `baseline` / `current`, so an agent cannot be steered to arbitrary files.
- **Graceful failure:** tool errors go back to the LLM as text, specialist failures become
  findings with `status: failed`, and a fallback report is built in plain code if needed.
- **Free-tier friendly:** compact JSON tool results, limited steps, rate-limit retries.

## Known limitations

- The LLM sometimes rounds numbers, converts them to percentages, or adds its own thresholds
  (for example an alert level in the recommendations). The verifier flags these but does not
  rewrite the report automatically.
- Groq free-tier limits can cause waits, and `gpt-oss` occasionally fails a tool call
  (HTTP 400). Re-running or switching `GROQ_MODEL` usually fixes it.
- Synthetic data only. Specialists run one after another, not in parallel.
- The free hosted demo may sleep after inactivity and take a moment to wake up.
- `mcp` is pinned to 1.x (`<2`); version 2 renamed `FastMCP` and changed the testing API.
- Out of scope: RAG, memory, observability, Docker.

## Possible next steps

- Automatic "rewrite with exact numbers" pass when unverified numbers are detected.
- Run specialists in parallel; add more specialists and tools.
- Replace synthetic data with a real dataset and an MCP connector to a monitoring system.