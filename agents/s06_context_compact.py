#!/usr/bin/env python3
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("MODEL_ID", "arcee-ai/trinity-large-preview:free")

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(json.dumps(messages, default=str)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            # In OpenAI format, we don't have structured tool_use blocks in content
            # We'll use a simpler approach - just clear old results
    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            result["content"] = "[Previous: tool result cleared]"
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = call_openrouter(
        [{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}]
    )
    summary = response.get("content", "")
    # Replace all messages with compressed summary
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]


# -- Tool implementations --
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
    "compact":    lambda **kw: "Manual compression requested.",
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
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "parameters": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}, "required": []},
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
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        response = call_openrouter(messages, TOOLS, SYSTEM)
        resp_content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        messages.append({"role": "assistant", "content": resp_content})
        if not tool_calls:
            return
        results = []
        manual_compact = False
        for tc in tool_calls:
            if tc["name"] == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(tc["name"])
                try:
                    output = handler(**tc["arguments"]) if handler else f"Unknown tool: {tc['name']}"
                except Exception as e:
                    output = f"Error: {e}"
            print(f"> {tc['name']}: {str(output)[:200]}")
            results.append({"role": "tool", "tool_call_id": tc["id"], "content": str(output)})
        messages.extend(results)
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
