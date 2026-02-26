import json
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ===== 第一個 API 呼叫 =====
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "arcee-ai/trinity-large-preview:free",
        "messages": [
            {
                "role": "user",
                "content": "知道義守大學嗎?"
            }
        ],
        "reasoning": {"enabled": True}
    })
)

response_json = response.json()

# 印出完整 JSON（除錯用）
# print("First API raw response:")
# print(json.dumps(response_json, indent=2))

assistant_message = response_json['choices'][0]['message']

print("\nFirst answer:")
print(assistant_message.get("content"))


 