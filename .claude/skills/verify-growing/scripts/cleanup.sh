#!/usr/bin/env bash
# Stop the server this run started and remove its scratch state. Evidence in <RUN_DIR>/evidence and server.log are kept.
# Usage: scripts/cleanup.sh <RUN_DIR>        Never kills by name: only the pids launch.sh recorded, and only if they still own our port.
set -uo pipefail
RUN_DIR="${1:?usage: cleanup.sh <RUN_DIR>}"
case "$RUN_DIR" in */saves/verify/*) ;; *) echo "refusing: $RUN_DIR is not under saves/verify/" >&2; exit 2;; esac
PORT="$(cat "$RUN_DIR/port" 2>/dev/null || true)"
for f in server.pid launcher.pid; do
  PID="$(cat "$RUN_DIR/$f" 2>/dev/null || true)"
  [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
done
for _ in 1 2 3 4 5; do
  if [ -z "$PORT" ] || ! lsof -ti "tcp:$PORT" -sTCP:LISTEN >/dev/null 2>&1; then break; fi
  sleep 1
done
if [ -n "$PORT" ] && lsof -ti "tcp:$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  SP="$(cat "$RUN_DIR/server.pid" 2>/dev/null || true)"; [ -n "$SP" ] && kill -9 "$SP" 2>/dev/null || true
fi
# scratch state goes (including the "-old-*" games a New child leaves next to the save); proof stays
rm -rf "$RUN_DIR"/save "$RUN_DIR"/save-old-* "$RUN_DIR"/children "$RUN_DIR"/children-raising
echo "cleaned $RUN_DIR (kept: evidence/, server.log)"
