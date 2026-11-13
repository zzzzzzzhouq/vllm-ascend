#!/usr/bin/env bash
# 跑一轮 aisbench。两端必须用同一份本脚本 / 同一组环境变量。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

# shellcheck disable=SC1091
source "$AISV_DIR/bin/activate"
cd "$AISBENCH_TOOL"
unset http_proxy https_proxy || true

# CSV 是追加写：每轮开始前清掉，确保本轮结果独立
rm -f aisbench_result.csv aisbench.log

# PREFIX_TEST=1（默认）保留原来的"先预热共享前缀"行为；
# 干净 decode 场景（repeat_rate=0，无共享前缀）要设成 0。
if [ "${PREFIX_TEST:-1}" = "1" ]; then
  PREFIX_ARGS=(--prefix_test)
else
  PREFIX_ARGS=()
fi

python3 aisbench_test.py \
  --input_len "$INPUT_LEN" \
  --output_len "$OUTPUT_LEN" \
  --data_num "$DATA_NUM" \
  --concurrency "$CONCURRENCY" \
  --request_rate 0 \
  --npu_num "$NPU_NUM" \
  --dataset_type prefix_cache \
  --repeat_rate "$REPEAT_RATE" \
  "${PREFIX_ARGS[@]+"${PREFIX_ARGS[@]}"}" \
  --seed "$SEED"
