#!/usr/bin/env bash
# Layerwise PD proxy：客户端打这个端口（默认 9000）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
unset http_proxy https_proxy || true

cd "$REPO"
nohup python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host 127.0.0.1 \
  --port "$PROXY_PORT" \
  --prefiller-hosts 127.0.0.1 --prefiller-ports "$P_PORT" \
  --decoder-hosts 127.0.0.1 --decoder-ports "$D_PORT" \
  > "$LOG_DIR/proxy.log" 2>&1 &
echo $! > "$LOG_DIR/proxy.pid"
echo "proxy started (pid $(cat "$LOG_DIR/proxy.pid")), log: $LOG_DIR/proxy.log"
echo "验证: curl -s localhost:$PROXY_PORT/v1/models"
