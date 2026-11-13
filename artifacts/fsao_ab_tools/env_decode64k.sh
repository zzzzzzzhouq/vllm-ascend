#!/usr/bin/env bash
# 干净 decode 预设：长上下文 + 单波次（data_num == concurrency），decode 期间没有新 prefill。
#
# 对应 kernel profiling 形状（tests/ut/test_fuse_sparse_attention_overlap_precisions.py）：
#   bs=4, msl=64K, topk=2048
# 注意 topk 由模型 config 的 index_topk 决定，E2E 不可调；profiling 的 hit_ratio 也没有
# 客户端旋钮，不要用 repeat_rate 去"模拟"它。
#
# 可覆盖：DECODE_BS / DECODE_IN_LEN / DECODE_OUT_LEN / DECODE_DRAM_GB
# 用法：
#   source ./env_decode64k.sh
#   PREFIX_TEST=0 ./round.sh base
# 或直接：
#   ./abba_decode64k.sh
#   DECODE_BS=8 DECODE_OUT_LEN=256 ./abba_decode64k.sh

# 被 source 时 $0 是当前 shell，必须用 BASH_SOURCE
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_ENV_DIR/env.sh"

# 场景隔离：这组结果不要和 32K/cc8 混在一个目录里，否则 summary 会把不同工况混批
export RESULTS=${DECODE_RESULTS:-$AB/results_decode64k}
mkdir -p "$RESULTS"

export DECODE_BS=${DECODE_BS:-4}
export DECODE_IN_LEN=${DECODE_IN_LEN:-65536}
export DECODE_OUT_LEN=${DECODE_OUT_LEN:-64}

# 压测形状：单波次 + 无共享前缀
export INPUT_LEN="$DECODE_IN_LEN"
export OUTPUT_LEN="$DECODE_OUT_LEN"
export DATA_NUM="$DECODE_BS"
export CONCURRENCY="$DECODE_BS"
export MAX_NUM_SEQS="$DECODE_BS"
export REPEAT_RATE=0
export NPU_NUM=${NPU_NUM:-16}
export PREFIX_TEST=0

# prompt + output 之上留 2K 余量
export MAX_MODEL_LEN=${DECODE_MAX_MODEL_LEN:-$((DECODE_IN_LEN + DECODE_OUT_LEN + 2048))}

# bs x 64K 独立 KV 约 25GB（BS=4 时）；48 够用，64 更稳。
# 若改成 BS=16，需要 ~100GB，必须同步调大。
export DRAM_GB=${DECODE_DRAM_GB:-64}

echo "[decode preset] repo=$REPO"
echo "[decode preset] ab=$AB results=$RESULTS"
echo "[decode preset] bs=$DECODE_BS in=$DECODE_IN_LEN out=$DECODE_OUT_LEN " \
     "max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS " \
     "repeat_rate=$REPEAT_RATE prefix_test=$PREFIX_TEST dram=${DRAM_GB}GB"

# 路径自检：脚本和 vllm-ascend 同级（base/）或放在 tools/ 子目录都应该通过
for _p in "$REPO/vllm_ascend" \
          "$AISV_DIR/bin/activate" \
          "$AISBENCH_TOOL/aisbench_test.py"; do
  [ -e "$_p" ] || echo "[decode preset] WARN missing: $_p" >&2
done
