#!/usr/bin/env bash
# 状态检查：拿到任意 HTTP 响应（200/404 都算）即视为服务已监听
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

for pair in "prefill:$P_PORT" "decode:$D_PORT" "proxy:$PROXY_PORT"; do
  name=${pair%%:*}; port=${pair##*:}
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
          "http://127.0.0.1:$port/v1/models" 2>/dev/null || true)
  printf "%-8s :%-5s " "$name" "$port"
  if [ -n "$code" ] && [ "$code" != "000" ]; then
    echo "OK (HTTP $code)"
  else
    echo "DOWN"
  fi
done

echo
echo "== 进程 =="
pgrep -af "vllm serve" | sed 's/^/  /' || true
pgrep -af "load_balance_proxy" | sed 's/^/  /' || true
pgrep -af "MetaService" | sed 's/^/  /' || true

echo
echo "== decode log 关键行 =="
grep -iE "sparse KV offload|fused_overlap|SfaRemoteD2H|initialize failed|ERROR" "$LOG_DIR/decode.log" 2>/dev/null | tail -20 || true
