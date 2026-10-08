"""Shared agent infrastructure (Day 3) + the single investigation agent.

Flow:  incident -> Groq LLM -> tool request -> MCP client -> MCP server -> evidence
       -> back to the LLM -> ... -> final report

Day 4 reuses run_mcp_loop / run_with_mcp from this file for every specialist agent.
This file must NOT import investigation.py. All evidence has to come through MCP.

Run:  python src/agent.py "Churn model quality dropped this week"
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import groq
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config

DEFAULT_INCIDENT = ("The customer churn model's performance dropped recently. "
                    "Investigate what changed in the data and why.")

REPORT_HEADINGS = """## Incident Summary
## Evidence (Facts from tools)
## Detected Issues
## Likely Root Cause (Hypothesis)
## Recommended Next Action
## Uncertainty and Limitations"""

SYSTEM_PROMPT = f"""You are an ML/data incident investigator.
You can call tools that analyse a baseline dataset, a current dataset and model metrics.

Rules:
1. Gather evidence with the tools before drawing conclusions. Call only the tools you need and do not repeat a call.
2. Every number in your report must be copied EXACTLY from a tool result (same digits). Never calculate, round, convert to percentages, or estimate numbers.
3. Separate FACTS (taken from a tool result, name the tool) from HYPOTHESES (your own inference, label it HYPOTHESIS).
4. If a tool returns an error, report it under limitations instead of guessing.
5. Do not invent facts that are not in tool results (for example deployments, retraining or code changes).

Write the final report in Markdown using exactly these headings, in this order:
{REPORT_HEADINGS}
"""

REQUIRED_SECTIONS = ["incident summary", "evidence", "detected issues",
                     "likely root cause", "recommended next action", "uncertainty"]

NUMBER = re.compile(r"\d+(?:\.\d+)?")

LAST_STEP_NOTE = ("You have no tool calls left. Do not call any tool. "
                  "Write your final answer now, using only the evidence already gathered.")


class AgentError(Exception):
    """An expected failure. The message is safe to show the user."""


def redact(text):
    """Make sure the API key can never appear in output."""
    key = os.getenv("GROQ_API_KEY")
    return text.replace(key, "***") if key else text


# ------------------------------------------------------------ Groq side
def make_llm_client():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise AgentError("GROQ_API_KEY is not set. Create a free key at "
                         "https://console.groq.com/keys and set it as an environment variable.")
    return Groq(api_key=key)


def ask_llm(client, model, messages, tools=None, tool_choice="auto"):
    """One LLM request (tools are optional). Waits and retries on rate limits;
    other failures become AgentError."""
    tool_options = {"tools": tools, "tool_choice": tool_choice} if tools else {}
    for attempt in range(config.GROQ_RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=0,
                max_completion_tokens=config.GROQ_MAX_OUTPUT_TOKENS, **tool_options)
        except groq.RateLimitError:
            if attempt == config.GROQ_RATE_LIMIT_RETRIES:
                raise AgentError("Groq rate limit reached (free tier). Wait a minute and run again.")
            print(f"[agent] Groq rate limit hit, waiting {config.GROQ_RETRY_WAIT_SECONDS}s "
                  f"(retry {attempt + 1}/{config.GROQ_RATE_LIMIT_RETRIES})...")
            time.sleep(config.GROQ_RETRY_WAIT_SECONDS)
        except groq.AuthenticationError:
            raise AgentError("Groq rejected the API key. Check GROQ_API_KEY.")
        except groq.APIConnectionError as e:
            cause = e.__cause__
            detail = redact(f"{type(cause).__name__}: {str(cause)[:150]}") if cause else "unknown cause"
            raise AgentError(f"Could not reach the Groq API ({detail}). Check your internet connection.")
        except groq.APIStatusError as e:
            detail = redact(str(getattr(e, "message", ""))[:200])
            raise AgentError(f"Groq API error (HTTP {e.status_code}): {detail} "
                             "Try again, or set GROQ_MODEL to another tool-calling model.")


# ------------------------------------------------------------ MCP side
def mcp_tools_to_groq_format(mcp_tools):
    """Turn tools discovered from the MCP server into the format the LLM API expects."""
    return [{"type": "function",
             "function": {"name": t.name,
                          "description": t.description or "",
                          "parameters": t.inputSchema}} for t in mcp_tools]


def tidy_tool_text(text):
    """Compact JSON (saves tokens on the free tier) and cap the length."""
    try:
        text = json.dumps(json.loads(text), separators=(",", ":"))
    except (TypeError, ValueError):
        pass                                   # not JSON: keep the raw text
    if len(text) > config.MAX_TOOL_RESULT_CHARS:
        text = text[:config.MAX_TOOL_RESULT_CHARS] + "...[truncated]"
    return text


async def run_tool(session, valid_names, name, raw_arguments):
    """Run one tool through MCP. Never raises: problems are returned to the LLM as text."""
    if name not in valid_names:                # permission check: also blocks real tools this agent may not use
        return json.dumps({"error": f"Unknown tool '{name}' (not permitted for this agent). "
                                    f"Allowed: {sorted(valid_names)}"})
    try:
        arguments = json.loads(raw_arguments) if raw_arguments else {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
    except ValueError:
        return json.dumps({"error": "Tool arguments were not a valid JSON object."})

    try:
        result = await session.call_tool(name, arguments)
    except Exception as e:                     # server crashed, protocol error, ...
        return json.dumps({"error": f"MCP tool call failed ({type(e).__name__})."})

    try:
        text = result.content[0].text
    except (AttributeError, IndexError, TypeError):
        return json.dumps({"error": "Malformed tool result (no text content)."})
    if getattr(result, "isError", False):
        return json.dumps({"error": f"Tool reported an error: {str(text)[:300]}"})
    return tidy_tool_text(str(text))


# ------------------------------------------------------------ report checks
def find_missing_sections(report):
    text = report.lower()
    return [s for s in REQUIRED_SECTIONS if s not in text]


def find_unverified_numbers(report, source_texts):
    """Numbers in the report that appear in NO tool result (or the incident text)."""
    allowed = {float(n) for text in source_texts for n in NUMBER.findall(text)}
    unverified = set()
    for n in NUMBER.findall(report):
        if n.isdigit() and int(n) < 10:        # list numbering, tiny counts
            continue
        if float(n) not in allowed:
            unverified.add(n)
    return sorted(unverified)


# ------------------------------------------------------------ the generic tool loop
async def run_mcp_loop(session, llm_client, model, system_prompt, user_message,
                       allowed_tools=None, max_iterations=config.AGENT_MAX_ITERATIONS,
                       label="agent"):
    """Tool-calling loop over MCP, used by the Day-3 agent and by every Day-4 specialist.

    allowed_tools limits which MCP tools this agent can see AND call (None = all tools).
    Returns the final text plus which tools were called and what they returned.
    """
    listed = await session.list_tools()                      # dynamic tool discovery
    visible = [t for t in listed.tools if allowed_tools is None or t.name in allowed_tools]
    tools = mcp_tools_to_groq_format(visible)
    if not tools:
        raise AgentError("The MCP server exposes no tools for this agent.")
    valid_names = {t.name for t in visible}

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}]
    tools_called, evidence, rejected = [], [], []

    for step in range(1, max_iterations + 1):
        if step == max_iterations:     # last step: forbid tools and tell the model to answer now
            request_messages = messages + [{"role": "user", "content": LAST_STEP_NOTE}]
            tool_choice = "none"
        else:
            request_messages, tool_choice = messages, "auto"
        response = ask_llm(llm_client, model, request_messages, tools, tool_choice)

        try:
            message = response.choices[0].message
            content = message.content or ""
            requests = [(c.id, c.function.name, c.function.arguments or "{}")
                        for c in (message.tool_calls or [])]
        except (AttributeError, IndexError, TypeError):
            raise AgentError("Unexpected response format from Groq.")

        if not requests:                                     # no tool call = final answer
            text = content.strip()
            if not text:
                raise AgentError("Groq returned an empty answer.")
            return {"text": text, "tools_called": tools_called, "evidence": evidence,
                    "rejected_tools": rejected, "iterations": step}

        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [{"id": i, "type": "function",
                                         "function": {"name": n, "arguments": a}}
                                        for i, n, a in requests]})
        for call_id, name, raw_arguments in requests:
            output = await run_tool(session, valid_names, name, raw_arguments)
            if name in valid_names:
                print(f"[{label}] calling MCP tool: {name} {raw_arguments}")
                tools_called.append(name)
                evidence.append(output)
            else:
                print(f"[{label}] REJECTED request for tool: {name}")
                rejected.append(name)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

    raise AgentError(f"No final answer after {max_iterations} steps.")


# ------------------------------------------------------------ Day-3 single agent
async def run_agent(incident, session, llm_client, model,
                    max_iterations=config.AGENT_MAX_ITERATIONS):
    """Investigate `incident` with one agent that may use every MCP tool."""
    loop = await run_mcp_loop(session, llm_client, model, SYSTEM_PROMPT,
                              f"Incident: {incident}", max_iterations=max_iterations)
    report = loop["text"]
    return {"report": report,
            "tools_called": loop["tools_called"],
            "iterations": loop["iterations"],
            "missing_sections": find_missing_sections(report),
            "unverified_numbers": find_unverified_numbers(report, loop["evidence"] + [incident])}


# ------------------------------------------------------------ MCP connection
async def run_with_mcp(server_script, work):
    """Start the local MCP server, connect as a client, then run `await work(session)`."""
    server_script = Path(server_script)
    if not server_script.exists():
        raise AgentError(f"MCP server script not found: {server_script}")

    params = StdioServerParameters(command=sys.executable, args=[str(server_script)],
                                   env=config.MCP_SERVER_ENV)
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(),
                                       timeout=config.MCP_CONNECT_TIMEOUT_SECONDS)
                return await work(session)
    except AgentError:
        raise
    except Exception as e:                                   # server crash, timeout, protocol error
        while getattr(e, "exceptions", None):                # unwrap ExceptionGroup to the real error
            e = e.exceptions[0]
        raise AgentError(f"MCP server problem ({type(e).__name__}: {str(e)[:100]}). "
                         "The server may have failed to start; see its messages above. "
                         "Also check the data exists: python src/generate_data.py") from e


async def investigate(incident, server_script=config.SERVER_SCRIPT, llm_client=None, model=None):
    model = model or os.getenv("GROQ_MODEL") or config.DEFAULT_GROQ_MODEL
    llm_client = llm_client or make_llm_client()             # fail fast if the key is missing
    return await run_with_mcp(server_script,
                              lambda session: run_agent(incident, session, llm_client, model))


def main():
    if hasattr(sys.stdout, "reconfigure"):                   # avoid Windows console encoding crashes
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    incident = " ".join(sys.argv[1:]).strip() or DEFAULT_INCIDENT
    print(f"Incident: {incident}\n")
    try:
        result = asyncio.run(investigate(incident))
    except AgentError as e:
        print(f"Agent error: {redact(str(e))}", file=sys.stderr)
        sys.exit(1)

    print("\n" + result["report"])
    print(f"\n--- Tools called via MCP: {result['tools_called']} "
          f"(LLM steps: {result['iterations']})")
    if result["missing_sections"]:
        print(f"WARNING: report is missing sections: {result['missing_sections']}")
    if result["unverified_numbers"]:
        print("WARNING: numbers not found in any tool result (check them): "
              f"{result['unverified_numbers']}")


if __name__ == "__main__":
    main()