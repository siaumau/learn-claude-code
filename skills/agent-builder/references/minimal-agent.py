#!/usr/bin/env python3
"""
Minimal Agent Template - Copy and customize this.

This is the simplest possible working agent (~80 lines).
It has everything you need: 3 tools + loop.

Usage:
    1. Set OPENROUTER_API_KEY environment variable
    2. python minimal-agent.py
    3. Type commands, 'q' to quit
"""

import json
import requests
from pathlib import Path
import subprocess
import os

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("MODEL_NAME", "arcee-ai/trinity-large-preview:free")
WORKDIR = Path.cwd()

# System prompt - keep it simple
SYSTEM = f"""You are a coding agent at {WORKDIR}.

Rules:
- Use tools to complete tasks
- Prefer action over explanation
- Summarize what you did when done"""

# Minimal tool set - add more as needed
TOOLS = [
    {
        "name": "bash",
        "description": "Run shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read file contents",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
]


def call_openrouter(messages, system=None):
    """Call OpenRouter API and return parsed response."""
    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    openai_messages.extend(messages)
    
    payload = {
        "model": MODEL,
        "messages": openai_messages,
        "max_tokens": 8000,
        "tools": TOOLS,
        "tool_choice": "auto"
    }
    
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
    )
    response.raise_for_status()
    data = response.json()
    
    choice = data.get("choices", [{}])[0].get("message", {})
    content = choice.get("content", "")
    tool_calls = []
    for tc in choice.get("tool_calls", []):
        if tc.get("type") == "function":
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": args
            })
    return content, tool_calls


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return result."""
    if name == "bash":
        try:
            r = subprocess.run(
                args["command"], shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=60
            )
            return (r.stdout + r.stderr).strip() or "(empty)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout"

    if name == "read_file":
        try:
            return (WORKDIR / args["path"]).read_text()[:50000]
        except Exception as e:
            return f"Error: {e}"

    if name == "write_file":
        try:
            p = WORKDIR / args["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"Wrote {len(args['content'])} bytes to {args['path']}"
        except Exception as e:
            return f"Error: {e}"

    return f"Unknown tool: {name}"


def agent(prompt: str, history: list = None) -> str:
    """Run the agent loop."""
    if history is None:
        history = []

    history.append({"role": "user", "content": prompt})

    while True:
        content, tool_calls = call_openrouter(history, SYSTEM)

        # Build assistant message
        history.append({"role": "assistant", "content": content})

        # If no tool calls, return text
        if not tool_calls:
            return content

        # Execute tools
        results = []
        for tc in tool_calls:
            print(f"> {tc['name']}: {tc['arguments']}")
            output = execute_tool(tc["name"], tc["arguments"])
            print(f"  {output[:100]}...")
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": output
            })

        history.extend(results)


if __name__ == "__main__":
    print(f"Minimal Agent - {WORKDIR}")
    print("Type 'q' to quit.\n")

    history = []
    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query in ("q", "quit", "exit", ""):
            break
        print(agent(query, history))
        print()
