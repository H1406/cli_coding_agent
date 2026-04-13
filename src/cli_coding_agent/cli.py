from __future__ import annotations

import argparse

from cli_coding_agent.agent import CodingAgent
from cli_coding_agent.config import AgentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI coding agent")
    parser.add_argument("prompt", nargs="?", default="", help="Prompt for the agent")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional path to the instruction file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AgentConfig.from_env(instructions_path=args.instructions)
    agent = CodingAgent(config)
    print(agent.run(args.prompt))


if __name__ == "__main__":
    main()
