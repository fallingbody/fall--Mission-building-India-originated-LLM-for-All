#!/usr/bin/env python3
"""
Quick launcher for FALL Chat.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.app import app

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="FALL Chat Interface")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="sk-local-dev")
    args = parser.parse_args()
    
    os.environ["FALL_API_URL"] = args.api_url
    os.environ["FALL_API_KEY"] = args.api_key
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    ⚡ FALL CHAT ⚡                        ║
║                                                          ║
║  Chat Interface:  http://{args.host}:{args.port}                    ║
║  FALL API:        {args.api_url}                    ║
║                                                          ║
║  Mode: Interactive Chat                                  ║
║  Type your message and FALL will respond.                ║
║  Switch to Agent mode for autonomous tool use.           ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=args.host, port=args.port, debug=True)