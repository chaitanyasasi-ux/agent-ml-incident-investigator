# AI Data/ML Incident Investigator

A multi-agent system that investigates machine-learning and data incidents. A **Supervisor**
agent coordinates a **Data Agent** and a **Model Agent**. They gather evidence through
**MCP (Model Context Protocol)** tools and produce an evidence-based incident report.
Built with free tools only (Groq free tier, open-source Python libraries, local data).

> Portfolio project on **synthetic data**. It demonstrates agent design, MCP, tool
> permissions, and guardrails against invented numbers. It is not a production system.

## Architecture

```
                         Incident (text)
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

## What was built, day by day

**Day 1: Deterministic investigation layer** (no LLM)
- Reproducible customer-churn incident: income missing rate ~2% → ~20%, `account_balance`
  inflated ~1.35× with noise, ~8% `UNKNOWN` region. Four control features stay stable.
- One scikit-learn Logistic Regression trained on baseline data; `baseline_metrics` and
  `current_metrics` are computed, not hard-coded.
- Four functions: `get_dataset_stats`, `check_data_quality`, `detect_drift` (PSI),
  `compare_model_metrics`. All thresholds live in `src/config.py`.

**Day 2: MCP server**
- The four functions are exposed as MCP tools (official Python SDK) with descriptions and
  input schemas. Tools take no file paths; invalid input and missing files return clean errors.
- Tested in memory with the SDK's client/server test helper.

**Day 3: Single Groq agent**
- An MCP client discovers tools dynamically; the Groq model chooses tools, receives results
  and writes a six-section report (summary, evidence, issues, root cause, next action,
  limitations), separating tool facts from hypotheses.
- Guardrail: every number in the report is checked against tool results.
- Handles rate limits (retry), invalid key, API/MCP failures and malformed tool calls.

**Day 4: Multi-agent system**
- Supervisor plans, runs the Data Agent and Model Agent, and synthesizes their findings.
- Role-based permissions: Data Agent → `get_dataset_stats`, `check_data_quality`,
  `detect_drift`; Model Agent → `compare_model_metrics`. Unauthorized requests are rejected
  before reaching MCP.
- Structured findings (observations, issues, hypotheses, limitations) plus evidence sources
  added by code. A specialist that calls no tool is marked failed.
- If a specialist or the Supervisor's own LLM call fails, a plain-code fallback report
  explains what failed.

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
ai-incident-investigator/
├── README.md
├── requirements.txt
├── .gitignore
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
    └── test_multi_agent.py
```

## Setup (Windows PowerShell; on macOS/Linux use `source .venv/bin/activate`)

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

## Run

```powershell
python src/multi_agent.py "Churn model recall dropped this week. What happened?"   # multi-agent
python src/agent.py "Churn model recall dropped this week. What happened?"         # single agent
python src/mcp_server.py                                                           # MCP server only (stdio)
```

The footer of a multi-agent run shows the Supervisor's plan, each specialist's status and
tools used, and warnings (failed components, missing sections, unverified numbers).

## Test

```powershell
python -m pytest
```

Expected: `65 passed, 2 skipped`. The two skipped tests make real Groq calls and run only if
you set `GROQ_API_KEY` and `RUN_GROQ_INTEGRATION=1`. All other tests use fake Groq and MCP
components, so they need no network or key.

## Guardrails and design decisions

- **Deterministic evidence, LLM explanation:** all numbers come from tested Python functions
  reached through MCP, never from the LLM.
- **Number verification:** numbers in reports are compared with tool output, and any number
  not found is flagged.
- **No file paths in tool arguments:** tools accept only fixed options such as
  `baseline` / `current`.
- **Graceful failure:** tool errors go back to the LLM as text, specialist failures become
  findings with `status: failed`, and a fallback report is built in plain code if needed.
- **Free-tier friendly:** compact JSON tool results, limited steps, rate-limit retries.

## Known limitations

- The LLM sometimes rounds numbers, converts them to percentages, or adds its own thresholds
  (for example an alert level in the recommendations). The verifier flags these but does not
  rewrite the report automatically.
- Groq free-tier limits can cause waits, and `gpt-oss` occasionally fails a tool call
  (HTTP 400); re-running or switching `GROQ_MODEL` usually fixes it.
- Synthetic data only. Specialists run one after another, not in parallel.
- `mcp` is pinned to 1.x (`<2`); version 2 renamed `FastMCP` and changed the testing API.
- Out of scope: RAG, memory, observability, UI, Docker, cloud deployment.

## Possible next steps

- Automatic "rewrite with exact numbers" pass when unverified numbers are detected.
- Run specialists in parallel; add more specialists and tools.
- Replace synthetic data with a real dataset and an MCP connector to a monitoring system.