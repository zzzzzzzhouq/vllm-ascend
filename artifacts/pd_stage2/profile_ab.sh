#!/usr/bin/env bash
# 在同一个 PD 场景下，对 base/opt 两套包各抓一次 Ascend PyTorch Profiler。
# 换包 = 拷贝预编译包 + 重启引擎（不需要重新编译）；每个 arm 的输出目录彼此独立。
#
#   $PROF_ROOT/<arm>/decode/     decode 引擎的 ascend_pt（目标算子在这里）
#   $PROF_ROOT/<arm>/prefill/    可选（PROFILE_PREFILL=1）
#   $PROF_ROOT/<arm>/request.json / response.json / *_server.log / meta.txt
#
# 用法：
#   ./profile_ab.sh                          # base opt
#   ./profile_ab.sh opt base                 # 自定义顺序
#   PROF_INPUT_LEN=4096 PROF_OUTPUT_LEN=128 ./profile_ab.sh
#   PROFILE_PREFILL=1 ./profile_ab.sh        # 连 P 也抓
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
unset http_proxy https_proxy || true

TOOLS=${TOOLS:-/home/z00943842/base}
[ -x "$TOOLS/swap_ops.sh" ] || { echo "missing $TOOLS/swap_ops.sh" >&2; exit 1; }

PROF_ROOT=${PROF_ROOT:-$HERE/prof}
PROF_INPUT_LEN=${PROF_INPUT_LEN:-4096}
PROF_OUTPUT_LEN=${PROF_OUTPUT_LEN:-128}
PROFILE_PREFILL=${PROFILE_PREFILL:-0}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-3600}

ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(base opt)

for arm in "${ARMS[@]}"; do
  case "$arm" in base|opt) ;; *) echo "unknown arm: $arm (base|opt)" >&2; exit 1;; esac
  DIR="$PROF_ROOT/$arm"
  mkdir -p "$DIR/decode"
  [ "$PROFILE_PREFILL" = "1" ] && mkdir -p "$DIR/prefill"

  echo "=== $arm -> $DIR ==="

  # 1) 换包（会先 kill 旧 vllm）
  "$TOOLS/swap_ops.sh" "$arm"

  # 2) 起 P / D（D 带 --profiler-config，输出到 $DIR/decode）
  if [ "$PROFILE_PREFILL" = "1" ]; then
    PROFILE_DIR_P="$DIR/prefill" "$HERE/start_prefill.sh"
  else
    "$HERE/start_prefill.sh"
  fi
  "$HERE/wait_ready.sh" "$P_PORT" "$STARTUP_TIMEOUT"

  PROFILE_DIR="$DIR/decode" "$HERE/start_decode.sh"
  "$HERE/wait_ready.sh" "$D_PORT" "$STARTUP_TIMEOUT"

  "$HERE/start_proxy.sh"
  "$HERE/wait_ready.sh" "$PROXY_PORT" 120

  # 3) 生成一条固定请求
  python3 - "$DIR/request.json" "$PROF_INPUT_LEN" "$PROF_OUTPUT_LEN" "$SERVED_NAME" <<'PY'
import json
import sys
path, n, out, model = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
json.dump(
    {
        "model": model,
        "messages": [{"role": "user", "content": "a " * n}],
        "max_tokens": out,
        "temperature": 0,
    },
    open(path, "w"),
)
PY

  # 4) 先开 profile，再发请求，最后关
  echo "[$arm] start_profile @ $D_PORT"
  curl -fsS -X POST "http://127.0.0.1:$D_PORT/start_profile" >/dev/null
  curl -sS -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @"$DIR/request.json" > "$DIR/response.json"
  sleep 2
  echo "[$arm] stop_profile @ $D_PORT"
  curl -fsS -X POST "http://127.0.0.1:$D_PORT/stop_profile" >/dev/null
  sleep 10

  # 5) 留档
  cp "$LOG_DIR/prefill.log" "$DIR/prefill_server.log" 2>/dev/null || true
  cp "$LOG_DIR/decode.log"  "$DIR/decode_server.log"  2>/dev/null || true
  {
    echo "arm=$arm time=$(date -Is)"
    echo "input_len=$PROF_INPUT_LEN output_len=$PROF_OUTPUT_LEN"
    echo "max_model_len=$MAX_MODEL_LEN tp=$TP dram=$DRAM_GB"
    echo "--- ascend_pt dirs ---"
    find "$DIR" -maxdepth 3 -type d -name "*_ascend_pt" | sort
  } > "$DIR/meta.txt"

  "$HERE/stop.sh"
  sleep "${ROUND_GAP:-30}"
done

echo "done. profiles under: $PROF_ROOT"
echo "解析: ./parse_profile.sh $PROF_ROOT"
