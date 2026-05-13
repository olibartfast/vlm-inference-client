#!/usr/bin/env bash
# .claude/hooks/inject_ci_context.sh
#
# UserPromptSubmit hook. Injects a short, stable reminder about the ghostgrid
# CI rules so they stay in the active context across long sessions. Without
# this, the agent tends to forget R0801 duplicate-code and re-introduce the
# same helpers in multiple workflow modules.
#
# Whatever this script prints to stdout is prepended to the user's prompt
# inside Claude Code. Keep it short — every token here is a token not
# available for the task.

cat <<'EOF'
<ghostgrid-ci-reminder>
Before editing any Python under src/ or tests/:
  - Line length: 120
  - Ruff rules active: E, F, I, UP, B, C4, SIM
  - Pylint gate: --fail-under=8.0 on src/ghostgrid/
  - **No duplicated blocks of ≥ 6 lines across modules (pylint R0801).**
    Shared result-dict construction → ghostgrid/workflows/_utils.py
    Shared run_agent call signature → pass through **kwargs
  - Do NOT add docstrings/comments/types to code you did not change.
Playbook: .claude/skills/ci-guardian/SKILL.md
</ghostgrid-ci-reminder>
EOF
