from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repository root to index. Defaults to the current working directory.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = str(Path(args.repo_root).expanduser().resolve()) if args.repo_root else str(Path.cwd())

    config = AgentConfig.from_env(
        instructions_path=args.instructions,
        session_id=args.session_id,
    )
    config.repo_root = repo_root
    config.repo_id = repo_root
    store = PostgresConversationStore(config.database_url)
    agent = CodingAgent(config, store)

    if args.prompt:
        print(agent.run(args.prompt))
        return

    session_id = agent.ensure_session()
    print(f"[{config.agent_name}] interactive session started.")
    print(f"Session ID: {session_id}")
    print(f"Repo root: {config.repo_root}")
    print("Type `exit` or `stop` to end the session.")

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nStopping session.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "stop"}:
            break

        try:
            response = agent.run_turn(prompt)
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            continue
        print(response)
        print()


if __name__ == "__main__":
    main()
