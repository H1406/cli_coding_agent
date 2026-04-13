# Architecture Notes

## Core modules

- `cli.py`: CLI entrypoint and argument parsing
- `agent.py`: high-level agent loop
- `config.py`: environment and runtime settings
- `prompts.py`: instruction loading
- `tools.py`: local tool contract and execution stubs

## Recommended growth path

1. Keep the CLI thin and push logic into service modules.
2. Separate model provider logic from the tool loop.
3. Add typed request and response objects before the agent grows complex.
4. Introduce persistence only after the interaction model is stable.
