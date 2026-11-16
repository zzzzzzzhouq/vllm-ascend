#!/usr/bin/env bash
# Decode engine: NPU 8-15, TP8/EP8, 开 sparse offload + fused overlap（你的算子在这里）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
unset http_proxy https_proxy || true

export ASCEND_RT_VISIBLE_DEVICES="$DECODE_NPUS"

ADDITIONAL=$(cat <<JSON
{
  "multistream_overlap_shared_expert": true,
  "recompute_scheduler_enable": true,
  "sparse_kv_offload_config": {
    "enabled": true,
    "use_fused_overlap": true,
    "dram_size_per_dp_GB": ${DRAM_GB},
    "topk_buffer_size": ${TP_BUFFER}
  }
}
JSON
)

KV_CFG='{"kv_connector":"SfaRemoteD2HConnector","kv_role":"kv_consumer","kv_port":"'"$KV_PORT_D"'","kv_connector_extra_config":{"memfabric_transfer_protocol":"sdma"}}'

nohup vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$D_PORT" --served-model-name "$SERVED_NAME" \
  --data-parallel-size 1 \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$D_MAX_BATCHED" \
  --gpu-memory-utilization "$GPU_MEM" \
  --trust-remote-code --quantization ascend \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --async-scheduling \
  --hf-overrides "$HF_OVERRIDES" \
  --additional-config "$ADDITIONAL" \
  --kv-transfer-config "$KV_CFG" \
  > "$LOG_DIR/decode.log" 2>&1 &
echo $! > "$LOG_DIR/decode.pid"
echo "decode started (pid $(cat "$LOG_DIR/decode.pid")), log: $LOG_DIR/decode.log"
echo "等待: curl -s localhost:$D_PORT/v1/models"
echo "确认算子路径: grep -i 'sparse KV offload\\|fused_overlap' $LOG_DIR/decode.log"
