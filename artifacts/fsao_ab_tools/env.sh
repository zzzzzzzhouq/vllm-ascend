#!/usr/bin/env bash
# FSAO A/B 环境变量 —— 按机器实际路径修改后再用。
# 其他脚本会自动 source 本文件。

# 本文件被 source，$0 是当前 shell，必须用 BASH_SOURCE 定位脚本目录。
# 脚本可以放在 base/（与 vllm-ascend 同级），也可以放在 tools/ 子目录。
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# REPO：优先脚本同级/上级的 vllm-ascend，最后退到历史绝对路径
if [ -z "${REPO:-}" ]; then
  for _cand in "$_ENV_DIR/vllm-ascend" "$_ENV_DIR/../vllm-ascend"; do
    if [ -d "$_cand/vllm_ascend" ]; then REPO="$_cand"; break; fi
  done
fi
export REPO=${REPO:-/home/z00943842/base/vllm-ascend}

# AB：ops 包和 results 的根目录。优先脚本同级，其次上级，最后历史路径
if [ -z "${AB:-}" ]; then
  for _cand in "$_ENV_DIR" "$_ENV_DIR/.." "/home/z00943842/fsao_ab"; do
    if [ -d "$_cand/ops/base" ] || [ -d "$_cand/ops/opt" ]; then AB="$_cand"; break; fi
  done
fi
export AB=${AB:-$_ENV_DIR}
export OPS_BASE=$AB/ops/base
export OPS_OPT=$AB/ops/opt
# 允许预设（如 env_decode64k.sh）把不同场景的结果分目录存放，避免汇总时混批
export RESULTS=${RESULTS:-$AB/results}

# ---- 需要你填的路径 ----
export MODEL_PATH=${MODEL_PATH:-/path/to/GLM-5.1-w4a8}
export SERVED_NAME=${SERVED_NAME:-glm-5.1}
export AISV_DIR=${AISV_DIR:-$HOME/aisvenv}
export AISBENCH_TOOL=${AISBENCH_TOOL:-$HOME/aisbench_auto_tools_prefix}
export DATASET_DIR=${DATASET_DIR:-$HOME/dataset}

export HOST_IP=${HOST_IP:-127.0.0.1}
export PORT=${PORT:-8000}

# ---- 压测参数：base/opt 两端必须完全一致 ----
export INPUT_LEN=${INPUT_LEN:-32768}
export OUTPUT_LEN=${OUTPUT_LEN:-1024}
export DATA_NUM=${DATA_NUM:-32}
export CONCURRENCY=${CONCURRENCY:-8}
export NPU_NUM=${NPU_NUM:-16}
export REPEAT_RATE=${REPEAT_RATE:-0.5}
export SEED=${SEED:-2}

# ---- 服务参数 ----
export TP=${TP:-16}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
export DRAM_GB=${DRAM_GB:-48}

export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export CANN_SET_ENV=${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}
export SOC_VERSION=${SOC_VERSION:-ascend910_9382}
export MAX_JOBS=${MAX_JOBS:-8}

# ---- HCCL：默认取第一块非回环网卡，可显式覆盖 ----
if [ -z "${HCCL_IF_IP:-}" ]; then
  HCCL_IF_IP=$(ip -o -4 addr show 2>/dev/null | awk '$4 !~ /^127/ {print $4; exit}' | cut -d/ -f1)
fi
if [ -z "${HCCL_SOCKET_IFNAME:-}" ]; then
  HCCL_SOCKET_IFNAME=$(ip -o -4 addr show 2>/dev/null | awk '$4 !~ /^127/ {print $2; exit}')
fi
export HCCL_IF_IP
export HCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}

mkdir -p "$RESULTS"
