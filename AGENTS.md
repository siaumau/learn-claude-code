# Repository Guidelines

## Project Structure & Module Organization
`agents/` contains the Python reference implementations for sessions `s01` through `s12` plus `s_full.py`.  
`docs/{en,zh,ja}/` holds the learning materials and session write-ups.  
`skills/` contains skill packs used by the agent examples.  
`web/` is the Next.js learning platform (source in `web/src`, static assets in `web/public`).  
`screenshots/` stores UI reference images used in documentation.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies for the agent scripts.
- `python agents/s01_agent_loop.py` runs the first session locally; pick other scripts under `agents/` as needed.
- `cd web && npm ci` installs web dependencies exactly as CI does.
- `cd web && npm run dev` starts the web app (pre-runs the content extractor).
- `cd web && npm run build` builds the web app for production.
- `cd web && npx tsc --noEmit` matches the CI type-check step.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and standard library-first imports. Keep docstrings and inline ASCII diagrams consistent with existing files in `agents/`.
- Web code is TypeScript in `web/src`; keep file and component names descriptive and aligned with Next.js conventions.
- No repo-wide formatter is configured. Match surrounding style and run `npx tsc --noEmit` before submitting.

## Testing Guidelines
- CI expects Python tests under `tests/` with names like `test_unit.py` and `test_v0.py` (see `.github/workflows/test.yml`). If you add tests, follow this naming pattern.
- Run individual tests with `python tests/<file>.py`.
- Web verification is via `npm run build` in `web/`.

## Commit & Pull Request Guidelines
- Recent history shows both conventional prefixes (`feat:`) and descriptive messages. Use `type: summary` when appropriate, or a concise imperative summary if not.
- PRs should include: a clear description, linked issue (if any), and screenshots for web/UI changes. Note the commands you ran.

## Security & Configuration Tips
- Set API credentials via `.env` (see `.env.example` for required keys).
- Avoid committing secrets or generated content; `.gitignore` already covers common artifacts.
