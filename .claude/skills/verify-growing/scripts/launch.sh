#!/usr/bin/env bash
# Start an isolated, mock-LLM Growing web server for verification and print its run directory.
# Usage: scripts/launch.sh [seed]        (run from the repo root)
# Prints:  RUN_DIR=<dir> PORT=<port> URL=<url>   -- keep RUN_DIR, every other script takes it as $1.
set -euo pipefail
SEED="${1:-1}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
[ -f server/web/__main__.py ] || { echo "not the Growing repo root: $ROOT" >&2; exit 2; }
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$ROOT/saves/verify/$RUN_ID"          # saves/ is git-ignored; never saves/web/current (that is the user's own game)
mkdir -p "$RUN_DIR/children" "$RUN_DIR/evidence"   # empty children dir: nothing to adopt, nothing of the user's is reachable
PORT="$(uv run python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')"
echo "$PORT" > "$RUN_DIR/port"
# GROWING_LLM=mock beats the live key in .env: no network, no cost, deterministic prose.
GROWING_LLM=mock nohup uv run python -m server.web --seed "$SEED" --lang en --port "$PORT" \
  --save "$RUN_DIR/save" --children "$RUN_DIR/children" > "$RUN_DIR/server.log" 2>&1 &
echo $! > "$RUN_DIR/launcher.pid"
# Ready = usable: the first scene has options, not just "the port answers".
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/api/scene" 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);sys.exit(0 if len(d["options"])>=2 and d["narration"] else 1)' 2>/dev/null; then
    LISTENER="$(lsof -ti "tcp:$PORT" -sTCP:LISTEN | head -1)"
    echo "$LISTENER" > "$RUN_DIR/server.pid"
    echo "RUN_DIR=$RUN_DIR PORT=$PORT URL=http://127.0.0.1:$PORT"
    exit 0
  fi
  kill -0 "$(cat "$RUN_DIR/launcher.pid")" 2>/dev/null || { echo "server exited early:" >&2; tail -20 "$RUN_DIR/server.log" >&2; exit 1; }
  sleep 1
done
echo "no usable first scene after 90s; see $RUN_DIR/server.log" >&2
"$(dirname "$0")/cleanup.sh" "$RUN_DIR" >/dev/null 2>&1 || true
exit 1
