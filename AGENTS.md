# ghostgrid — Agent & Contributor Instructions

This file is the **single source of truth** for coding conventions, tooling, and rules.
All other AI agent config files (`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`) defer to this file.

---

## Project overview

- **Package**: `ghostgrid` (Python ≥ 3.10)
- **Source root**: `src/ghostgrid/`
- **CLI entry point**: `ghostgrid` (maps to `ghostgrid.cli:main`)
- **GitHub repo**: `https://github.com/olibartfast/ghostgrid`

---

## Development setup

```bash
pip install -e ".[dev]"         # install with dev deps
pytest tests/                   # run tests
ruff check src/ tests/          # lint
ruff format src/ tests/         # format
pylint src/ghostgrid/ --fail-under=8.0   # static analysis
mypy src/ || true               # type check (soft fail)
```

---

## Code quality rules

### Linting tools

| Tool | Purpose | Config |
|------|---------|--------|
| `ruff` | Fast lint + format | `[tool.ruff]` in `pyproject.toml` |
| `pylint` | Deep static analysis, min score 8.0 | `[tool.pylint]` in `pyproject.toml` |
| `mypy` | Type checking (soft fail) | `[tool.mypy]` in `pyproject.toml` |

### Never introduce duplicate code (pylint R0801)

Pylint R0801 (`duplicate-code`) fires when ≥ 6 similar lines appear across two or more modules.
**Always extract shared logic into a helper** instead of repeating it.

Common patterns that have triggered R0801 in this codebase and how to fix them:

| Pattern | Shared location |
|---------|----------------|
| Per-agent result dict `{agent_id, model, provider, latency_ms, success, error}` | Extract a `_result_to_dict(r)` helper in `ghostgrid/workflows/_utils.py` |
| `run_agent(agent, ..., image_paths, detail, max_tokens, resize, target_size)` call signature | Pass through as `**kwargs` or use a shared `_call_agent` wrapper |

### Other conventions

- Line length: 120 characters (enforced by ruff + pylint).
- Python style: ruff `select = ["E", "F", "I", "UP", "B", "C4", "SIM"]`.
- Do **not** add docstrings, comments, or type annotations to code you did not change.
- Do **not** add error handling for scenarios that cannot happen.
- Do **not** create helper abstractions for one-off operations.

---

## Repository structure

```
src/ghostgrid/          # main package
  workflows/            # sequential, parallel, moa, react, iterative, conditional
  tools/                # builtin ReAct tools + parsing helpers
  providers.py          # provider dispatch + run_agent
  backends.py           # external agent backend dispatch (claude-code, codex, opencode, pi)
  models.py             # Agent, AgentResult, Tool dataclasses
  config.py             # env/config helpers + system prompts
  cli.py                # argparse CLI
  image.py              # image encoding + resize helpers
tests/                  # pytest suite (mirrors src layout)
examples/               # runnable example scripts
docs/                   # markdown documentation
```

---

## CI

The CI pipeline (`.github/workflows/ci.yml`) runs:

1. **lint** job: `ruff` check + format, then `pylint --fail-under=8.0`
2. **test** job (needs lint): mypy, pytest with coverage across Python 3.10/3.11/3.12
3. **build** job: `hatch build`

---

## Agent tooling (Claude Code)

This repo ships a Claude Code kit under `.claude/` that enforces the rules
above *before* a push reaches CI. If you use Claude Code, the kit activates
automatically. If you use another agent (Copilot, Codex, Gemini), the
catalogue in `.claude/skills/ci-guardian/SKILL.md` is still the right
reference — point your tool at it.

### Hooks

| Hook | Event | What it does |
|---|---|---|
| `ruff_on_edit.sh` | `PostToolUse(Edit\|Write\|MultiEdit)` | Runs `ruff check --fix` + `ruff format` on every edited `.py` under `src/`, `tests/`, `examples/`. Blocks the turn if unfixable issues remain. |
| `pre_push_ci_gate.sh` | `PreToolUse(Bash)` on `git push` | Runs the exact CI lint + pylint + pytest commands locally. Blocks the push if any fail. Opt out with `GHOSTGRID_SKIP_CI_GATE=1` or `--no-verify`. |
| `block_secrets.sh` | `PreToolUse(Bash)` | Denies commands and staged commits containing a live provider API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, etc.) or a staged `.env`. |
| `inject_ci_context.sh` | `UserPromptSubmit` | Prepends a short reminder of the core CI rules to every prompt. |

### Subagents

| Subagent | When to invoke |
|---|---|
| `@ci-triage` | A CI run is red. Reads failing logs via `gh`, classifies against the catalogue, produces a minimal fix plan. Does not edit. |
| `@lint-fixer` | Ruff or pylint errors need mechanical cleanup. Runs the `--fix` loop and hand-fixes the residue using the catalogue. Hands R0801 to `@dedup-refactor`. |
| `@dedup-refactor` | Pylint `R0801` fired, or you see the same block in two modules. Extracts to the canonical location (`workflows/_utils.py`, `video.py`, etc.) per the map above. |

### Skill

`.claude/skills/ci-guardian/SKILL.md` — the playbook. One section per error
class actually seen in this repo's CI history (ruff codes, pylint `R0801`,
cv2 `no-member`, pytest, install, build). Every section includes the exact
local-reproduce command.

If you fix a CI failure whose cause isn't yet catalogued, **add a section
to the skill** in the same PR. The skill compounds in value; don't let it
decay.

### The one local gate command

```bash
ruff check src/ tests/ && \
ruff format --check src/ tests/ && \
pylint src/ghostgrid/ --fail-under=8.0 && \
pytest tests/ -q -x
```

If this is green locally, CI will be green. The `pre_push_ci_gate.sh` hook
runs exactly these commands.

**Rule: always run the full gate command before every commit and push.**
A push that skips local tests risks breaking CI for everyone. No exceptions.
