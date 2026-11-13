#!/usr/bin/env bash
# 启动整网服务（GLM-5.1 + sparse_kv_offload + use_fused_overlap）。
# 由 round.sh 调用；也可以手动: ./serve.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

PKG="$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
[ -d "$PKG" ] || { echo "missing custom op package: $PKG" >&2; exit 1; }

# CANN toolkit
CANN_SET_ENV=${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}
if [ -f "$CANN_SET_ENV" ]; then
  # shellcheck disable=SC1091
  source "$CANN_SET_ENV"
fi

# 显式指向当前主树里的包，避免 set_env.bash 里写死的旧绝对路径
export ASCEND_CUSTOM_OPP_PATH="$PKG${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$PKG/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 15)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
unset HCCL_INTRA_PCIE_ENABLE HCCL_INTRA_ROCE_ENABLE || true
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
unset http_proxy https_proxy || true

ADDITIONAL=$(cat <<JSON
{
  "enable_balance_scheduling": true,
  "multistream_overlap_shared_expert": true,
  "sparse_kv_offload_config": {
    "enabled": true,
    "use_fused_overlap": true,
    "keep_device_kv_cache": true,
    "dram_size_per_dp_GB": ${DRAM_GB}
  }
}
JSON
)

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --data-parallel-size 1 \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config "$ADDITIONAL"
