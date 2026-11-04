# AGENTS.md

## Agent Delegation Rules

1. **Subagent Delegation Policy**:
   - The main agent coordinates planning and high-level decisions.
   - The main agent must delegate all code implementation, file editing, and test running to `flash` subagents using `invoke_subagent` with `Model="flash"`.
   - Subagents must report back completed changes, test results, and status to the caller via `send_message`.
   - Do not accumulate or chain tasks onto an existing subagent via `send_message`; spawn a **new `flash` subagent** (`invoke_subagent`, `Model="flash"`) for each separate task.
   - When delegating, assign a **narrow, file-level scope**, and invoke **multiple subagents in parallel** for independent files/tasks.
   - When running tests, subagents must not run the full `pytest` suite; run **only the single test file directly related to the modified file**.
