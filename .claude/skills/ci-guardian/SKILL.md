---
name: ci-guardian
description: Use whenever a ghostgrid CI failure needs fixing, whenever you see ruff or pylint output with rule codes, whenever pylint R0801 duplicate-code fires, whenever cv2 triggers pylint no-member, or whenever you're about to push and want to pre-empt the common failures. Contains the catalogued fix for every error class that has appeared in ghostgrid's CI history, plus the local-verify command for each.
---

# ci-guardian — ghostgrid CI failure playbook

This is the single operational reference for every CI failure class ghostgrid
has actually hit. Each section is a **bucket** the `ci-triage` subagent maps
onto, and each section ends with the **exact local command** that reproduces
and verifies the fix.

If a new failure mode appears that isn't catalogued, fix it, then **add a
section here**. The skill compounds in value; do not let it decay.

---

## §ruff — ruff check failures

**Symptom.** CI `lint` job exits non-zero on `ruff check src/ tests/`.

**Local reproduce.**
```bash
ruff check src/ tests/ --output-format=concise
```

**Active rule set** (from `pyproject.toml`):
`select = ["E", "F", "I", "UP", "B", "C4", "SIM"]`, `line-length = 120`.

### Codes seen in ghostgrid CI history

| Code | Meaning | Fix |
|---|---|---|
| `I001` | Unsorted / unformatted imports | `ruff check --fix` — autofixes every time |
| `F401` | Unused import | `ruff check --fix` removes it. If it's a re-export, add the name to `__all__` in the module instead |
| `F841` | Local variable assigned but never used | Either delete the assignment, or rename to `_name` if needed for unpacking: `result, _unused = fn()` |
| `C408` | Unnecessary `dict(...)` call | `ruff check --fix` rewrites to a literal `{"k": v}` |
| `E501` | Line too long (> 120) | Break at a comma / paren. Do NOT raise the limit. Ruff does not autofix long lines. |
| `UP006` / `UP007` / `UP035` | Old typing syntax (`List[x]` → `list[x]`, `Optional[x]` → `x \| None`) | `ruff check --fix` |
| `B008` | Function call in default arg | Move the call inside the function body with a `None` sentinel |
| `B904` | `raise` in `except` without `from` | Add `from err` (propagate) or `from None` (suppress) |
| `SIM102` | Nested `if` statements | Combine with `and` |
| `SIM117` | Nested `with` | Merge into a single `with a, b:` |

### Workflow

```bash
# 1. Autofix what can be autofixed
ruff check --fix src/ tests/
# 2. Format
ruff format src/ tests/
# 3. Strict check — anything here needs a hand fix from the table above
ruff check src/ tests/ --output-format=concise
```

---

## §format — ruff format --check diff

**Symptom.** CI fails on `ruff format --check src/ tests/` with a unified diff.

**Fix.** Run `ruff format src/ tests/` locally and commit the result.
There is no judgment call here. If the hook `ruff_on_edit.sh` is installed
and executable, this bucket should never fire.

**Local reproduce.**
```bash
ruff format --check src/ tests/
```

---

## §R0801 — pylint duplicate-code

**Symptom.**
```
R0801: Similar lines in 2 files
==ghostgrid.workflows.parallel:[45:61]
==ghostgrid.workflows.moa:[32:48]
```

**Threshold.** pylint fires at **≥ 6 similar lines** across ≥ 2 modules.

**Why it matters.** This is the single most common non-ruff failure in
ghostgrid's CI history. AGENTS.md calls it out explicitly. Rearranging
lines does not fix it — extraction does.

### Canonical extraction map (from AGENTS.md)

| Duplicated pattern | Goes here | Helper signature |
|---|---|---|
| `{agent_id, model, provider, latency_ms, success, error}` dict | `src/ghostgrid/workflows/_utils.py` | `_result_to_dict(r: AgentResult) -> dict` |
| Full `run_agent(agent, prompt, image_paths, detail, max_tokens, resize, target_size)` call | `src/ghostgrid/workflows/_utils.py` | `_call_agent(agent, prompt, **kwargs) -> AgentResult` |
| Provider URL / headers construction | `src/ghostgrid/providers.py` | extend the dispatch, do not duplicate |

### Fix

Always delegate this bucket to the `dedup-refactor` subagent. See its
definition at `.claude/agents/dedup-refactor.md` for the extraction
procedure.

**Local reproduce.**
```bash
pylint src/ghostgrid/ --disable=all --enable=R0801
```

### Non-fixes (do not use)

- `# pylint: disable=duplicate-code` on one of the sites — just hides it.
- Creating a new `common.py` or `helpers.py` — violates the canonical map.
- Renaming variables to "dedupe" by confusing pylint's token comparison —
  pure tech debt, will reappear on the next change.

---

## §score — pylint --fail-under=8.0

**Symptom.** pylint exits non-zero with `Your code has been rated at
7.XX/10` and no specific blocker you've already fixed.

**Fix procedure.**

1. Look at the top 3 message categories in the pylint report. Usually one
   of these dominates:

   | Category | Typical fix |
   |---|---|
   | `C0301` line-too-long | Same as ruff `E501` — break the line |
   | `R0913` too-many-arguments | Legit in CLI builders; add a specific disable with a comment OR extract a dataclass if the args really group |
   | `R0914` too-many-locals | If the function has a natural split point, extract; otherwise disable with a comment |
   | `W0613` unused-argument | For callback/protocol signatures, prefix with `_`; for mistakes, remove |
   | `C0114` / `C0115` / `C0116` missing docstring | AGENTS.md forbids speculative docstrings. Add `# pylint: disable=missing-module-docstring` to `pyproject.toml` if it's making score drop, do NOT add docstrings to untouched code. |

2. **Never** lower `--fail-under` below 8.0 to make CI pass. That is the
   architectural floor and the whole point of the gate.

3. If a specific disable is needed, put it in `[tool.pylint.messages_control]`
   in `pyproject.toml`, not scattered as inline `# pylint: disable` comments.

**Local reproduce.**
```bash
pylint src/ghostgrid/ --fail-under=8.0
```

---

## §pytest — test failures

**Symptom.** `pytest tests/` fails.

**First move.** Run the single failing test with verbose output *before*
doing anything else:

```bash
pytest tests/path/to/test_file.py::test_name -xvs
```

Never mutate code before you've read the actual assertion error. The most
common accident when chasing lint failures is "fixing" a test by deleting
the assertion.

### Common causes in ghostgrid

- **Imports reorganised** → test imports stale path. Fix the import, not
  the test.
- **Provider mocks** → the provider dispatch changed signature; update
  the mock's `return_value` shape to match the new `AgentResult`.
- **`run_react` tool registry** → a tool was renamed in
  `src/ghostgrid/tools/`; grep for the old name, update the test.

---

## §mypy — type-check failures

**Status in CI.** Currently soft-fail (`mypy src/ || true`). Do not harden
this without the maintainer's sign-off — there is a reason it's soft.

If you are touching a file anyway and want to improve types, fine, but
do not run `mypy --strict` across the whole codebase as part of a lint
fix. That is a separate project.

---

## §install — ModuleNotFoundError during test collection

**Symptom.** `ImportError: No module named 'cv2'` (or `openai`, `anthropic`,
etc.) during `pytest` collection in CI.

**Fix.** CI must install with the right extras. The correct install line in
`ci.yml` is:

```yaml
- run: pip install -e ".[dev]"
```

If you see `ModuleNotFoundError` for a test dependency, verify the `[dev]` extra is present in the install step.

---

## §build — hatch build failure

**Symptom.** The `build` job fails at `hatch build`.

**Fix procedure.**

1. Check that `VERSION` exists and is a valid PEP 440 version string.
2. Check that `pyproject.toml`'s `[project] dynamic = ["version"]` source
   points at `VERSION` (or wherever the version actually lives).
3. Check that `src/ghostgrid/__init__.py` exists and exports the public
   API declared in README.

Most `hatch build` failures are version-string drift after a merge. Run:

```bash
python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project'])"
hatch build
```

---

## Local CI gate — the one command to rule them all

Before pushing, run:

```bash
ruff check src/ tests/ && \
ruff format --check src/ tests/ && \
pylint src/ghostgrid/ --fail-under=8.0 && \
pytest tests/ -q -x && \
echo "OK — safe to push"
```

This is exactly what `.claude/hooks/pre_push_ci_gate.sh` runs on `git push`.
If that command is green, CI will be green.

---

## When to update this file

If you fix a CI failure and the cause doesn't appear in any section above,
**add a new section** at the point of fix. Include:

1. The exact error line (so future grep finds it)
2. Root cause in 1–3 sentences
3. The fix, as a command or a code pattern
4. The local-reproduce command

A playbook that isn't maintained is worse than no playbook.
