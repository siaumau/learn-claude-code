#!/usr/bin/env python3
"""
export_chat.py - 匯出 Qwen Code 對話記錄

這個腳本會讀取 .qwen 資料夾中的對話歷史並匯出

使用方法：
    python export_chat.py              # 匯出最近的對話
    python export_chat.py --all        # 匯出所有對話
    python export_chat.py --list       # 列出可用的對話記錄
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def find_qwen_dirs():
    """尋找 Qwen Code 的設定目錄"""
    possible_paths = [
        Path.home() / ".qwen",
        Path.cwd() / ".qwen",
    ]
    
    for p in possible_paths:
        if p.exists():
            return p
    return None


def list_conversations(qwen_dir: Path):
    """列出可用的對話記錄"""
    print("可用的對話記錄：")
    print("-" * 50)
    
    # 檢查是否有 conversation 相關的檔案
    for f in qwen_dir.glob("*.json"):
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        print(f"  {f.name} - {mtime.strftime('%Y-%m-%d %H:%M')}")
    
    # 檢查 history 資料夾
    history_dir = qwen_dir / "history"
    if history_dir.exists():
        for f in history_dir.glob("*.json"):
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"  history/{f.name} - {mtime.strftime('%Y-%m-%d %H:%M')}")


def export_conversation(qwen_dir: Path, output_dir: str = "logs"):
    """匯出對話記錄"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exported_files = []
    
    # 嘗試匯出不同的可能位置
    
    # 1. 檢查 session 檔案
    for session_file in qwen_dir.glob("session*.json"):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            output_path = Path(output_dir) / f"qwen_session_{timestamp}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            exported_files.append(str(output_path))
            print(f"[已匯出] {session_file.name} -> {output_path}")
        except Exception as e:
            print(f"[跳過] {session_file.name}: {e}")
    
    # 2. 檢查 history 資料夾
    history_dir = qwen_dir / "history"
    if history_dir.exists():
        for hist_file in history_dir.glob("*.json"):
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                output_path = Path(output_dir) / f"qwen_history_{hist_file.stem}_{timestamp}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                exported_files.append(str(output_path))
                print(f"[已匯出] history/{hist_file.name} -> {output_path}")
            except Exception as e:
                print(f"[跳過] history/{hist_file.name}: {e}")
    
    # 3. 檢查其他可能的對話檔案
    for f in qwen_dir.glob("*.json"):
        if f.name not in [os.path.basename(ef) for ef in exported_files]:
            if f.name not in ["settings.json", "config.json"]:
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    
                    # 檢查是否包含對話內容
                    if isinstance(data, list) or "messages" in data or "conversation" in data:
                        output_path = Path(output_dir) / f"qwen_{f.stem}_{timestamp}.json"
                        with open(output_path, "w", encoding="utf-8") as file:
                            json.dump(data, file, ensure_ascii=False, indent=2)
                        exported_files.append(str(output_path))
                        print(f"[已匯出] {f.name} -> {output_path}")
                except Exception as e:
                    pass  # 靜默跳過非對話檔案
    
    if not exported_files:
        print("\n[没有找到對話記錄]")
        print("\n提示：")
        print("  Qwen Code 的對話可能儲存在以下位置：")
        print(f"  - {Path.home() / '.qwen'}")
        print("  - 或專案的 .qwen/ 資料夾")
    else:
        print(f"\n[共匯出 {len(exported_files)} 個檔案到 {output_dir}/]")


def main():
    qwen_dir = find_qwen_dirs()
    
    if not qwen_dir:
        print("[錯誤] 找不到 Qwen Code 的設定目錄")
        return
    
    print(f"Qwen 目錄：{qwen_dir}")
    print("-" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_conversations(qwen_dir)
        elif sys.argv[1] == "--all":
            export_conversation(qwen_dir)
        else:
            print("用法：python export_chat.py [--all] [--list]")
            print("  --list  列出可用的對話記錄")
            print("  --all   匯出所有對話記錄")
    else:
        export_conversation(qwen_dir)


if __name__ == "__main__":
    main()
