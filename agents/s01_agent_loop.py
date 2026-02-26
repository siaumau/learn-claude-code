#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import json
import os
import subprocess

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("MODEL_ID", "arcee-ai/trinity-large-preview:free")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


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
        # Note: Some OpenRouter models don't support tool_choice
        # Only include it if the model supports it
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
    if not response.ok:
        print(f"OpenRouter API error: {response.status_code}")
        print(f"Response: {response.text}")
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


# -- The core pattern: a while loop that calls tools until the model stops --
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
            if tc["name"] == "bash":
                print(f"\033[33m$ {tc['arguments']['command']}\033[0m")
                output = run_bash(tc["arguments"]["command"])
                print(output[:200])
                results.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
        messages.extend(results)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the last assistant message
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    print(content)
                break
        print()
