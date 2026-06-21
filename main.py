"""
FALL - Fully Autonomous Learning Language model
Main entry point. Three lines to command 5 trillion parameters.
"""
import sys
import os
import argparse
import asyncio
from fall.sdk.cli import main as cli_main
from fall.training.launch import main as train_main
from fall.inference.api import start_server
from fall.agent.runtime import main as agent_main


def main():
    parser = argparse.ArgumentParser(
        description="FALL - Fully Autonomous Learning Language model (5T params)",
        epilog="For all.",
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("--config", default="fall_5t.yaml")
    train_parser.add_argument("--data-path", required=True)
    
    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start inference server")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--host", default="0.0.0.0")
    
    # Agent
    agent_parser = subparsers.add_parser("agent", help="Start autonomous agent")
    
    # CLI
    cli_parser = subparsers.add_parser("run", help="Run a task")
    cli_parser.add_argument("task", help="Task to execute")
    cli_parser.add_argument("--mode", default="autonomous")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_main()
    elif args.mode == "serve":
        start_server(args.host, args.port)
    elif args.mode == "agent":
        asyncio.run(agent_main())
    elif args.mode == "run":
        cli_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()