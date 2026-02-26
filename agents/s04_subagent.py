#!/usr/bin/env python3
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import json
import os
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("MODEL_ID", "arcee-ai/trinity-large-preview:free")

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# -- Tool implementations shared by parent and child --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


def call_openrouter(messages: list, tools: list = None, system: str = None):
    """Call OpenRouter API and return parsed response."""
    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    openai_messages.extend(messages)

    payload = {
        "model": MODEL,
        "messages": openai_messages,
        "max_tokens": 8000,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/shareAI-lab/learn-claude-code",
        },
        data=json.dumps(payload),
    )
    response.raise_for_status()
    data = response.json()

    choice = data.get("choices", [{}])[0].get("message", {})
    result = {
        "content": choice.get("content", ""),
        "tool_calls": [],
    }

    for tc in choice.get("tool_calls", []):
        if tc.get("type") == "function":
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            result["tool_calls"].append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": args,
            })

    return result


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    last_content = ""
    for _ in range(30):  # safety limit
        response = call_openrouter(sub_messages, CHILD_TOOLS, SUBAGENT_SYSTEM)
        resp_content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        sub_messages.append({"role": "assistant", "content": resp_content})
        if resp_content:
            last_content = resp_content
        if not tool_calls:
            break
        results = []
        for tc in tool_calls:
            handler = TOOL_HANDLERS.get(tc["name"])
            output = handler(**tc["arguments"]) if handler else f"Unknown tool: {tc['name']}"
            results.append({"role": "tool", "tool_call_id": tc["id"], "content": str(output)[:50000]})
        sub_messages.extend(results)
    # Only the final text returns to the parent -- child context is discarded
    return last_content or "(no summary)"


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {"name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}},
]


def agent_loop(messages: list):
    while True:
        response = call_openrouter(messages, PARENT_TOOLS, SYSTEM)
        resp_content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        messages.append({"role": "assistant", "content": resp_content})
        if not tool_calls:
            return
        results = []
        for tc in tool_calls:
            if tc["name"] == "task":
                desc = tc["arguments"].get("description", "subtask")
                print(f"> task ({desc}): {tc['arguments']['prompt'][:80]}")
                output = run_subagent(tc["arguments"]["prompt"])
            else:
                handler = TOOL_HANDLERS.get(tc["name"])
                output = handler(**tc["arguments"]) if handler else f"Unknown tool: {tc['name']}"
            print(f"  {str(output)[:200]}")
            results.append({"role": "tool", "tool_call_id": tc["id"], "content": str(output)})
        messages.extend(results)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
