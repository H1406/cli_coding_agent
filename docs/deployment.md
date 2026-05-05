# Deployment Notes

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Container build

```bash
docker build -t cli-coding-agent .
docker run --rm cli-coding-agent --help
```

## Docker with PostgreSQL

If you run the agent in Docker and want it to connect to PostgreSQL, do not use
`localhost` in `AGENT_DATABASE_URL` unless PostgreSQL is running inside the same
container. In Docker, `localhost` points back to that container itself.

Common connection failures come from:

- using `localhost` instead of the database service name
- starting the app before PostgreSQL is ready to accept connections
- using a database name in `AGENT_DATABASE_URL` that does not match `POSTGRES_DB`

This repository includes a `compose.yaml` that runs both services together:

```bash
docker compose up --build
```

The app service uses:

```text
postgresql://postgres:postgres@db:5432/cli_coding_agent
```

If you are connecting from a container to a PostgreSQL server running on your
host machine instead, use one of these patterns instead of `localhost`:

- macOS/Windows Docker Desktop: `host.docker.internal`
- Linux: publish the port and connect through the host IP or use host networking when appropriate

Example:

```bash
docker run --rm \
  -e AGENT_DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/cli_coding_agent \
  cli-coding-agent --help
```

## Distribution options

- Package to PyPI for `pip install`
- Ship a Docker image for containerized execution
- Wrap with `pipx` for clean local CLI installs

Pick one primary distribution target early so packaging and auth flows stay simple.
