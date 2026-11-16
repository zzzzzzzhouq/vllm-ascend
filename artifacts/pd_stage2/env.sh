#!/usr/bin/env bash
# Stage 2: 单机 PD 分离 + Sparse KV offload（decode 侧跑 npu_fused_sparse_attention_overlap）
#
#   P: NPU 0-7   (TP8 / EP8)  port 8100  kv_port 20000
#   D: NPU 8-15  (TP8 / EP8)  port 8200  kv_port 20050
#   proxy: 9000 (layerwise proxy)
#
# 关键点：
#   - decode 必须用 BF16 主 KV（不要传 --kv-cache-dtype）
#   - decode 不要开 keep_device_kv_cache（那是 colocate debug 专用，PD consumer 下会断言）
#   - 只有 use_fused_overlap=true 才会走你的算子
#
# 被 source 时 $0 是当前 shell，必须用 BASH_SOURCE
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO=${REPO:-/home/z00943842/base/vllm-ascend}
export MODEL=${MODEL:-/mnt/weight/GLM-5.1-w4a8}
export SERVED_NAME=${SERVED_NAME:-glm-5.1}

export PRE_FILL_NPUS=${PRE_FILL_NPUS:-0,1,2,3,4,5,6,7}
export DECODE_NPUS=${DECODE_NPUS:-8,9,10,11,12,13,14,15}
export TP=${TP:-8}

export P_PORT=${P_PORT:-8100}
export D_PORT=${D_PORT:-8200}
export PROXY_PORT=${PROXY_PORT:-9000}
export KV_PORT_P=${KV_PORT_P:-20000}
export KV_PORT_D=${KV_PORT_D:-20050}

# 4K smoke: MAX_MODEL_LEN=8192 MAX_NUM_SEQS=8 P_MAX_NUM_SEQS=8 P_MAX_BATCHED=8192
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-67840}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}       # decode 侧并发（KV 在 host DRAM）
export P_MAX_NUM_SEQS=${P_MAX_NUM_SEQS:-1}   # prefill 侧 KV 在 device，很小
export P_MAX_BATCHED=${P_MAX_BATCHED:-4096}
export D_MAX_BATCHED=${D_MAX_BATCHED:-256}
export GPU_MEM=${GPU_MEM:-0.93}
export DRAM_GB=${DRAM_GB:-110}               # decode host 池，TP 共享、DP 独立
export TP_BUFFER=${TP_BUFFER:-2048}          # 必须 >= 模型 index_topk(2048) 且为 256 的倍数

export MMC_DIR=${MMC_DIR:-$_ENV_DIR}
export LOG_DIR=${LOG_DIR:-$MMC_DIR/logs}
export META_HOST=${META_HOST:-127.0.0.1}
export META_PORT=${META_PORT:-5000}
export CONFIG_STORE_PORT=${CONFIG_STORE_PORT:-6000}

# 0 = 只用 SfaRemoteD2H（不需要 Memcache，先试这个）
# 1 = 文档里的 MultiConnector + AscendStore/Memcache（需要先跑 start_memcache.sh）
export USE_LAYERWISE_PREFILL=${USE_LAYERWISE_PREFILL:-0}

export HF_OVERRIDES=${HF_OVERRIDES:-'{"use_index_cache": true, "index_topk_freq": 4}'}

# ---- 网卡 / HCCL ----
if [ -z "${LOCAL_IP:-}" ]; then
  LOCAL_IP=$(ip -o -4 addr show 2>/dev/null | awk '$4 !~ /^127/ {print $4; exit}' | cut -d/ -f1)
fi
if [ -z "${NIC:-}" ]; then
  NIC=$(ip -o -4 addr show 2>/dev/null | awk '$4 !~ /^127/ {print $2; exit}')
fi
export LOCAL_IP NIC
export HCCL_IF_IP=${HCCL_IF_IP:-$LOCAL_IP}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$NIC}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NIC}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$NIC}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---- 自定义算子包（与 serve.sh 一致）----
PKG="$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$PKG${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$PKG/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$LOG_DIR"

echo "[stage2] repo=$REPO model=$MODEL"
echo "[stage2] P=$PRE_FILL_NPUS:$P_PORT (kv $KV_PORT_P)  D=$DECODE_NPUS:$D_PORT (kv $KV_PORT_D)"
echo "[stage2] max_model_len=$MAX_MODEL_LEN max_num_seqs(D)=$MAX_NUM_SEQS max_num_seqs(P)=$P_MAX_NUM_SEQS"
echo "[stage2] dram=${DRAM_GB}GB topk_buffer=$TP_BUFFER layerwise_prefill=$USE_LAYERWISE_PREFILL"
echo "[stage2] proxy=$PROXY_PORT logs=$LOG_DIR  local_ip=$LOCAL_IP nic=$NIC"
