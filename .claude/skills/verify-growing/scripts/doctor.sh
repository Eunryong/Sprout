#!/usr/bin/env bash
# Read-only: is this instance the one this run started, and is it worth driving?  Usage: scripts/doctor.sh <RUN_DIR>
# Exit 0 = healthy. Any other exit = do not drive it; relaunch.
set -uo pipefail
RUN_DIR="${1:?usage: doctor.sh <RUN_DIR>}"
PORT="$(cat "$RUN_DIR/port")"; PID="$(cat "$RUN_DIR/server.pid" 2>/dev/null || true)"
fail() { echo "DOCTOR FAIL: $*"; exit 1; }
[ -n "$PID" ] && kill -0 "$PID" 2>/dev/null || fail "server pid '$PID' is not running"
OWNER="$(lsof -ti "tcp:$PORT" -sTCP:LISTEN | head -1)"
[ "$OWNER" = "$PID" ] || fail "port $PORT is owned by '$OWNER', not our pid $PID (someone else's instance: do not drive it)"
INFO="$(curl -fsS "http://127.0.0.1:$PORT/api/info")" || fail "/api/info does not answer"
echo "$INFO" | python3 -c '
import sys, json
d = json.load(sys.stdin)
bad = []
if d["mode"] != "mock":
    bad.append("mode=%r, expected mock (a live run would spend API calls)" % d["mode"])
if d["lang"] != "en":
    bad.append("lang=%r, expected en" % d["lang"])
print("mode=%s lang=%s lived=%s ended=%s world=%s" % (d["mode"], d["lang"], d["lived"], d["ended"], d["world"]))
sys.exit("; ".join(bad) if bad else 0)' || fail "info check"
[ -f "$RUN_DIR/save/meta.json" ] || fail "no save at $RUN_DIR/save: this server is not writing where the run expects"
echo "DOCTOR OK  url=http://127.0.0.1:$PORT  pid=$PID  save=$RUN_DIR/save"
