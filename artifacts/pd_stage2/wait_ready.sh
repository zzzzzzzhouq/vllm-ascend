#!/usr/bin/env bash
# 等待某个端口就绪。
# vLLM 引擎的 /v1/models 返回 200；layerwise proxy 不实现该端点，返回 404。
# 只要拿到任意 HTTP 响应（不是连接失败/000）就认为服务已起来。
# 用法: ./wait_ready.sh <port> [timeout_seconds]
set -euo pipefail
port="${1:?usage: wait_ready.sh <port> [timeout]}"
timeout_s="${2:-3600}"
deadline=$(( $(date +%s) + timeout_s ))
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
          "http://127.0.0.1:$port/v1/models" 2>/dev/null || true)
  if [ -n "$code" ] && [ "$code" != "000" ]; then
    echo "port $port ready (HTTP $code)"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timeout waiting for port $port" >&2
    exit 1
  fi
  sleep 10
done
