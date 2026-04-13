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

## Distribution options

- Package to PyPI for `pip install`
- Ship a Docker image for containerized execution
- Wrap with `pipx` for clean local CLI installs

Pick one primary distribution target early so packaging and auth flows stay simple.
