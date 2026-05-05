from __future__ import annotations

import argparse

from cli_coding_agent.agent import CodingAgent
from cli_coding_agent.config import AgentConfig
from cli_coding_agent.storage import PostgresConversationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI coding agent")
    parser.add_argument("prompt", nargs="?", default="", help="Prompt for the agent")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional path to the instruction file",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session id to resume or create",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AgentConfig.from_env(
        instructions_path=args.instructions,
        session_id=args.session_id,
    )
    store = PostgresConversationStore(config.database_url)
    agent = CodingAgent(config, store)
    result = agent.run(args.prompt)
    print(result)


if __name__ == "__main__":
    main()
