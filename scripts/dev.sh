#!/usr/bin/env bash

set -euo pipefail

# python -m venv .venv
source ~/python-envs/shared-env/bin/activate
pip install -e .[dev]
python -m cli_coding_agent.cli --help
