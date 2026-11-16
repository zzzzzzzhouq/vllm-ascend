#!/usr/bin/env bash
# 等待某个端口 /v1/models 就绪
# 用法: ./wait_ready.sh <port> [timeout_seconds]
set -euo pipefail
port="${1:?usage: wait_ready.sh <port> [timeout]}"
timeout_s="${2:-3600}"
deadline=$(( $(date +%s) + timeout_s ))
while :; do
  if curl -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "port $port ready"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timeout waiting for port $port" >&2
    exit 1
  fi
  sleep 10
done
