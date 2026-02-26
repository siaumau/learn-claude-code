 使用教學

    1. 設定 API Key

    複製 .env.example 為 .env 並填入你的 OpenRouter API Key：

     1 cp .env.example .env

    編輯 .env 檔案：

     1 OPENROUTER_API_KEY=sk-or-v1-你的_key
     2 MODEL_ID=arcee-ai/trinity-large-preview:free

    > 取得 API Key：https://openrouter.ai/

    2. 安裝依賴

     1 pip install -r requirements.txt

    3. 執行 Agent

    從最簡單的開始：

     1 # Level 1: 基礎 agent (只有 bash 工具)
     2 python agents/s01_agent_loop.py
     3
     4 # Level 2: 加入檔案工具 (read/write/edit)
     5 python agents/s02_tool_use.py
     6
     7 # Level 3: 加入 Todo 功能
     8 python agents/s03_todo_write.py

    完整功能版本：

     1 # 完整功能 agent (所有工具)
     2 python agents/s_full.py
     3
     4 # 自治 agent 團隊
     5 python agents/s11_autonomous_agents.py
     6
     7 # 工作樹隔離（需要 git）
     8 python agents/s12_worktree_task_isolation.py

    4. 與 Agent 互動

    執行後會看到提示符號，例如：

     1 s01 >>

    輸入指令：

     1 s01 >> 幫我建立一個 hello.py 檔案，內容是 print("Hello World")
     2 s01 >> 執行 python hello.py
     3 s01 >> q    # 退出

    5. 特殊命令（s_full.py）

     1 /compact   # 壓縮對話上下文
     2 /tasks     # 查看所有任務
     3 /team      # 查看團隊成員
     4 /inbox     # 查看收件箱

    6. 建立你自己的 Agent

    使用 scaffold 工具：

     1 cd skills/agent-builder/scripts
     2 python init_agent.py my-agent --level 1
     3 cd ../../my-agent
     4 cp .env.example .env
     5 # 編輯 .env 填入 API key
     6 python my-agent.py

    推薦學習順序

     1 1. s01_agent_loop.py     → 了解核心循環
     2 2. s02_tool_use.py       → 加入工具
     3 3. s03_todo_write.py     → 任務規劃
     4 4. s_full.py             → 完整版本