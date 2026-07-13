#!/bin/bash
# start.sh — one command to launch everything
cd "$(dirname "$0")"
kill $(pgrep -f "serve.py") 2>/dev/null
nohup python3 serve.py > watcher.log 2>&1 &
echo "✅ Started (PID $!) — http://localhost:8765"
echo "   Logs: tail -f watcher.log"
