# CLI Coding Agent

Starter repository for building and deploying a local CLI coding agent.

## What is included

- `src/cli_coding_agent/`: application source
- `tests/`: test scaffolding
- `docs/`: architecture and deployment notes
- `scripts/`: local developer helpers
- `examples/`: example configuration
- `.github/workflows/`: CI starter
- `Dockerfile`: container packaging baseline

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
agent --help
pytest
```

## Interactive usage from any repo

Run the agent from the repository you want indexed:

```bash
cd /path/to/any/repo
agent
```

It will start an interactive loop and keep the conversation in the same stored
session until you type `exit` or `stop`.

The agent can now take actions through a bounded tool loop using:
- `READ`
- `WRITE`
- `RUN`

You can also point at a repo explicitly:

```bash
agent --repo-root /path/to/any/repo
```

To continue the same historical conversation later, reuse the same session id:

```bash
agent --session-id your-session-id
```

## Docker with PostgreSQL

For a containerized app plus database setup, use:

```bash
docker compose up --build
```

If you override `AGENT_DATABASE_URL`, remember that `localhost` inside the app
container does not point to the PostgreSQL container. Use `db` for the Compose
service name or `host.docker.internal` when connecting back to a database on the
host machine.

## Suggested next steps

1. Replace the stubbed agent loop in `src/cli_coding_agent/agent.py` with your model orchestration.
2. Expand `src/cli_coding_agent/tools.py` with the tools your agent is allowed to call.
3. Wire the runtime to your configured provider. The repo now defaults to Hugging Face with `Qwen/Qwen2.5-Coder-1.5B` in the config examples, but the current runtime still needs provider-specific inference code.
4. Update `docs/deployment.md` once you choose your runtime target.

## Project layout

```text
.
├── .github/workflows/ci.yml
├── Dockerfile
├── Makefile
├── docs/
├── examples/
├── scripts/
├── src/cli_coding_agent/
└── tests/
```
