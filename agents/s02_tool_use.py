#!/usr/bin/env python3
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


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
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
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


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
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


def agent_loop(messages: list):
    while True:
        response = call_openrouter(messages, TOOLS, SYSTEM)
        resp_content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        messages.append({"role": "assistant", "content": resp_content})
        if not tool_calls:
            return
        results = []
        for tc in tool_calls:
            handler = TOOL_HANDLERS.get(tc["name"])
            output = handler(**tc["arguments"]) if handler else f"Unknown tool: {tc['name']}"
            print(f"> {tc['name']}: {output[:200]}")
            results.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
        messages.extend(results)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
