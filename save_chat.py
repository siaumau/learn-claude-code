#!/usr/bin/env python3
"""
save_chat.py - 保存 Qwen Code 對話記錄

使用方法：
1. 在 Qwen Code 聊天視窗中，複製對話內容
2. 執行：python save_chat.py
3. 貼上對話內容，輸入空行結束
4. 對話會保存到 logs/qwen_chat_YYYYMMDD_HHMMSS.json

或者直接保存文字檔：
    python save_chat.py --file "對話內容.txt"
"""

import json
import os
import sys
from datetime import datetime


def save_chat(messages: list, logs_dir: str = "logs"):
    """保存對話到 JSON 檔案"""
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qwen_chat_{timestamp}.json"
    filepath = os.path.join(logs_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    
    print(f"\n[對話已保存至：{filepath}]")
    return filepath


def interactive_mode():
    """互動模式：手動輸入對話內容"""
    print("=" * 50)
    print("Qwen Code 對話記錄保存工具")
    print("=" * 50)
    print("說明：")
    print("1. 從 Qwen Code 聊天視窗複製對話")
    print("2. 貼上到這裡（可分多次貼上）")
    print("3. 輸入空行（直接按 Enter）結束輸入")
    print("=" * 50)
    print("開始貼上對話內容：")
    print("-" * 50)
    
    messages = []
    lines = []
    
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    
    full_text = "\n".join(lines)
    
    if not full_text.strip():
        print("[沒有輸入內容，已取消]")
        return
    
    # 將對話保存為文字檔和 JSON
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存原始文字檔
    txt_path = os.path.join("logs", f"qwen_chat_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n[原始文字已保存至：{txt_path}]")
    
    # 嘗試解析為結構化 JSON（簡單版本）
    messages = [
        {"role": "user", "content": "從 Qwen Code 複製的對話"},
        {"role": "assistant", "content": full_text}
    ]
    
    json_path = os.path.join("logs", f"qwen_chat_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"[結構化 JSON 已保存至：{json_path}]")


def save_from_file(input_file: str):
    """從檔案讀取並保存"""
    if not os.path.exists(input_file):
        print(f"[錯誤] 檔案不存在：{input_file}")
        return
    
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存為 JSON
    messages = [
        {"role": "user", "content": "從檔案匯入的對話"},
        {"role": "assistant", "content": content}
    ]
    
    json_path = os.path.join("logs", f"qwen_chat_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"[對話已保存至：{json_path}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--file" and len(sys.argv) > 2:
            save_from_file(sys.argv[2])
        else:
            print("用法：python save_chat.py [--file <對話檔>]")
    else:
        interactive_mode()
