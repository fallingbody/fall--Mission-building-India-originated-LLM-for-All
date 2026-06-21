#!/usr/bin/env python3
"""
FALL CLI - Command Line Interface.
The three-line API call made even simpler.
"""
import argparse
import sys
import os
import json
from fall.sdk.client import FALLClient

def main():
    parser = argparse.ArgumentParser(
        description="FALL - Fully Autonomous Learning Language model",
        epilog="Three lines to command 5 trillion parameters.",
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="The task to execute (e.g., 'hack 192.168.1.50')",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FALL_API_KEY"),
        help="FALL API key (default: $FALL_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("FALL_BASE_URL", "https://api.fall.ai"),
        help="FALL API base URL",
    )
    parser.add_argument(
        "--mode",
        default="autonomous",
        choices=["autonomous", "interactive"],
        help="Execution mode",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        help="Tools to enable",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream results",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show system statistics",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Health check",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="FALL CLI 1.0.0",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    
    hack_parser = subparsers.add_parser("hack", help="Penetration test a target")
    hack_parser.add_argument("target", help="Target IP or hostname")
    
    defend_parser = subparsers.add_parser("defend", help="Harden a system")
    defend_parser.add_argument("target", nargs="?", default="localhost", help="Target to defend")
    
    code_parser = subparsers.add_parser("code", help="Write code")
    code_parser.add_argument("spec", help="Code specification")
    code_parser.add_argument("--language", "-l", default="python", help="Programming language")
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code for vulnerabilities")
    analyze_parser.add_argument("file", help="Code file to analyze")
    
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: FALL_API_KEY not set. Use --api-key or set FALL_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    
    client = FALLClient(api_key=args.api_key, base_url=args.base_url)
    
    # Handle flags
    if args.stats:
        stats = client.get_stats()
        print(json.dumps(stats, indent=2))
        return
    
    if args.health:
        healthy = client.health_check()
        print(f"FALL is {'healthy' if healthy else 'unreachable'}")
        return
    
    # Handle subcommands
    if args.command == "hack":
        result = client.hack(args.target)
    elif args.command == "defend":
        result = client.defend(args.target)
    elif args.command == "code":
        result = client.write_code(args.spec, args.language)
    elif args.command == "analyze":
        with open(args.file, 'r') as f:
            code = f.read()
        result = client.analyze_code(code)
    elif args.command == "ask":
        result = client.ask(args.question)
    elif args.task:
        result = client.execute(args.task, mode=args.mode, tools=args.tools)
    else:
        parser.print_help()
        return
    
    # Output
    if args.json:
        output = {
            "task_id": result.task_id,
            "status": result.status,
            "result": result.result,
            "duration_seconds": result.duration_seconds,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Task ID: {result.task_id}")
        print(f"Status: {result.status}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Tokens: {result.tokens_generated}")
        print()
        if isinstance(result.result, dict):
            print(result.result.get("output", json.dumps(result.result, indent=2)))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "task_id": result.task_id,
                "result": result.result,
            }, f, indent=2)


if __name__ == "__main__":
    main()