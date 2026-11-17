#!/usr/bin/env bash
# Prefill engine: NPU 0-7, TP8/EP8
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
unset http_proxy https_proxy || true

export ASCEND_RT_VISIBLE_DEVICES="$PRE_FILL_NPUS"

if [ "$USE_LAYERWISE_PREFILL" = "1" ]; then
  # 文档里的组合：MultiConnector = SfaRemoteD2H(producer) + AscendStore(memcache, layerwise)
  export MMC_LOCAL_CONFIG_PATH="$MMC_DIR/mmc-local.conf"
  KV_CFG='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_port":"'"$KV_PORT_P"'","kv_connector_extra_config":{"memfabric_transfer_protocol":"sdma","connectors":[{"kv_connector":"SfaRemoteD2HConnector","kv_role":"kv_producer"},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","use_layerwise":true,"layerwise_num_shared_buffers":3,"layerwise_independent_layers":[0]}}]}}'
  EXTRA=(--enforce-eager)
else
  # 最小组件：只用 SfaRemoteD2H 把 KV 暴露给 Decode，不碰 Memcache
  KV_CFG='{"kv_connector":"SfaRemoteD2HConnector","kv_role":"kv_producer","kv_port":"'"$KV_PORT_P"'","kv_connector_extra_config":{"memfabric_transfer_protocol":"sdma"}}'
  EXTRA=()
fi

# 可选：--profiler-config（Ascend PyTorch Profiler），输出到 PROFILE_DIR_P
PROFILE_ARGS=()
if [ -n "${PROFILE_DIR_P:-}" ]; then
  mkdir -p "$PROFILE_DIR_P"
  PROFILE_ARGS=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROFILE_DIR_P\", \"torch_profiler_with_stack\": false}")
fi

nohup vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$P_PORT" --served-model-name "$SERVED_NAME" \
  --data-parallel-size 1 \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$P_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$P_MAX_BATCHED" \
  --gpu-memory-utilization "$GPU_MEM" \
  --trust-remote-code --quantization ascend \
  --no-enable-prefix-caching \
  --hf-overrides "$HF_OVERRIDES" \
  --additional-config '{"multistream_overlap_shared_expert": true}' \
  --kv-transfer-config "$KV_CFG" \
  "${EXTRA[@]+"${EXTRA[@]}"}" \
  "${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"}" \
  > "$LOG_DIR/prefill.log" 2>&1 &
echo $! > "$LOG_DIR/prefill.pid"
echo "prefill started (pid $(cat "$LOG_DIR/prefill.pid")), log: $LOG_DIR/prefill.log"
echo "等待: curl -s localhost:$P_PORT/v1/models"
