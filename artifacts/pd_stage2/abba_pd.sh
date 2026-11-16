#!/usr/bin/env bash
# 单机 PD 场景的算子 A/B：默认 base opt opt base base opt。
# 前提:
#   1) 已完成 Stage 2 的单轮 smoke，P/D/proxy 能起来；
#   2) aisbench 的 config.py 已把 HOST_PORT 指向 $PROXY_PORT(9000)、
#      MODEL_NAME 指向 $SERVED_NAME(glm-5.1)；
#   3) $TOOLS 下有 swap_ops.sh / env.sh（你原来那套）。
#
# 用法:
#   ./abba_pd.sh
#   ./abba_pd.sh base opt
#   INPUT_LEN=32768 OUTPUT_LEN=1024 DATA_NUM=8 CONCURRENCY=8 ./abba_pd.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

TOOLS=${TOOLS:-/home/z00943842/base}
[ -x "$TOOLS/swap_ops.sh" ] || { echo "missing $TOOLS/swap_ops.sh"; exit 1; }

# 复用原来的压测工具路径/AB 根目录（会顺带设好 REPO / AB / AISV_DIR / AISBENCH_TOOL）
source "$TOOLS/env.sh"
export RESULTS=${PD_RESULTS:-$AB/results_pd}
mkdir -p "$RESULTS"

# PD 场景的压测参数：用 PD_* 覆盖，避免被 tools/env.sh 的 32K/cc8 默认值盖掉
INPUT_LEN=${PD_INPUT_LEN:-4096}
OUTPUT_LEN=${PD_OUTPUT_LEN:-256}
DATA_NUM=${PD_DATA_NUM:-4}
CONCURRENCY=${PD_CONCURRENCY:-4}
REPEAT_RATE=${PD_REPEAT_RATE:-0}
NPU_NUM=${PD_NPU_NUM:-16}
SEED=${PD_SEED:-2}

if [ "$#" -gt 0 ]; then SEQ=("$@"); else SEQ=(base opt opt base base opt); fi

n=1
for b in "${SEQ[@]}"; do
  case "$b" in base|opt) ;; *) echo "unknown build: $b" >&2; exit 1;; esac
  tag="${b}_r${n}"; n=$((n+1))
  echo "=== $tag ==="

  "$TOOLS/swap_ops.sh" "$b"     # 会先 kill 掉 vllm
  "$HERE/start_prefill.sh"
  "$HERE/wait_ready.sh" "$P_PORT" "${STARTUP_TIMEOUT:-3600}"
  "$HERE/start_decode.sh"
  "$HERE/wait_ready.sh" "$D_PORT" "${STARTUP_TIMEOUT:-3600}"
  "$HERE/start_proxy.sh"
  "$HERE/wait_ready.sh" "$PROXY_PORT" 120

  (
    # shellcheck disable=SC1091
    source "$AISV_DIR/bin/activate"
    cd "$AISBENCH_TOOL"
    unset http_proxy https_proxy || true
    rm -f aisbench_result.csv aisbench.log
    python3 aisbench_test.py \
      --input_len "$INPUT_LEN" --output_len "$OUTPUT_LEN" \
      --data_num "$DATA_NUM" --concurrency "$CONCURRENCY" \
      --request_rate 0 --npu_num "$NPU_NUM" \
      --dataset_type prefix_cache --repeat_rate "$REPEAT_RATE" \
      --seed "$SEED"
  ) 2>&1 | tee "$RESULTS/${tag}_bench.console"

  [ -f "$AISBENCH_TOOL/aisbench_result.csv" ] && cp "$AISBENCH_TOOL/aisbench_result.csv" "$RESULTS/${tag}.csv"
  [ -f "$AISBENCH_TOOL/aisbench.log" ] && cp "$AISBENCH_TOOL/aisbench.log" "$RESULTS/${tag}.log"

  {
    echo "tag=$tag time=$(date -Is)"
    echo "input_len=$INPUT_LEN output_len=$OUTPUT_LEN data_num=$DATA_NUM concurrency=$CONCURRENCY repeat_rate=$REPEAT_RATE"
    echo "max_model_len=$MAX_MODEL_LEN dram=$DRAM_GB topk_buffer=$TP_BUFFER"
    cat "$RESULTS/${tag}.csv" 2>/dev/null || true
  } > "$RESULTS/${tag}_meta.txt"

  "$HERE/stop.sh"
  sleep "${ROUND_GAP:-60}"
done

"$AISV_DIR/bin/python" "$TOOLS/summary.py" 2>/dev/null || true
echo "results: $RESULTS"
