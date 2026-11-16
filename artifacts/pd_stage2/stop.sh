#!/usr/bin/env bash
# 停掉 Stage 2 的所有进程
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

for f in proxy decode prefill memcache-meta; do
  p="$LOG_DIR/$f.pid"
  if [ -f "$p" ]; then
    pid=$(cat "$p")
    kill "$pid" 2>/dev/null || true
    echo "killed $f (pid $pid)"
    rm -f "$p"
  fi
done

pkill -9 -f "load_balance_proxy" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "MetaService" 2>/dev/null || true
sleep 5
echo "done"
