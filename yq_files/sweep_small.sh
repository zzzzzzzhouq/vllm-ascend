#!/usr/bin/env bash
# 小内存版性能冒烟：memlock / 宿主机内存受限（例如 ulimit -l=64 MiB）时用。
#
# 宿主 KV 估算（int8, 576B/token/head）：
#   bs × (msl/128) blocks × 128 token × 576 B
#   默认 msl=8192, bs≤8 → 8×64×128×576 ≈ 37 MB，池 0.06 GiB ≈ 61 MB 放得下
#
# ⚠ 与标准 campaign（sweep_cases.sh 的 hit/bs/bs0）不可比：
#   这里 msl 更小、bs 更少、A/A 标签省略，只用于"内存受限机器上先确认
#   算子能跑、曲线形状合理"。正式 A/B 必须回到标准脚本。
#
# 用法：
#   source yq_files/env.sh && export ASCEND_RT_VISIBLE_DEVICES=<卡号>
#   bash yq_files/sweep_small.sh
#   MSL=4096 BS=1,2,4 PROCS=1 bash yq_files/sweep_small.sh
set -u

cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." || exit 1

if [ -z "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
    echo "⚠ 先 source yq_files/env.sh（ASCEND_CUSTOM_OPP_PATH 为空）" >&2
    exit 1
fi

DRAM_GB="${DSA_DRAM_POOL_GB:-0.06}"
CPU_KV_BACKEND="${DSA_CPU_KV_BACKEND:-pinned}"
PROCS="${PROCS:-1}"
MSL="${MSL:-8192}"
BS="${BS:-1,2,4,8}"
RATIOS="${RATIOS:-0,0.5,1}"
DEVICES="${DEVICES:-0}"
ulimit -l unlimited 2>/dev/null || true

_busy="$(pgrep -f 'test_fuse_sparse_attention_overlap_precisions|sweep_median\.py' 2>/dev/null || true)"
if [ -n "$_busy" ]; then
    echo "⚠ 已有 bench 在跑（pid: $_busy）。一次只跑一个，先等它结束。" >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$DEVICES}"

LOGDIR="${DSA_LOGDIR:-$PWD/yq_logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/case_small_$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD 2>/dev/null || echo nogit).log"

echo "==================================================================="
echo " 小内存性能冒烟"
echo "   池大小      : ${DRAM_GB} GiB（64 MiB 限制内）"
echo "   KV 后端     : ${CPU_KV_BACKEND}（pinned = 不经 MemFabric）"
echo "   msl / bs    : ${MSL} / ${BS}"
echo "   命中率      : ${RATIOS}（无 A/A 标签）"
echo "   进程数      : ${PROCS}　卡: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "   日志        : $LOG"
echo "   开始        : $(date +%H:%M:%S)"
echo "==================================================================="

python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo -- \
    --batch-size "$BS" \
    --max-seq-len "$MSL" \
    --dram-size-gb "$DRAM_GB" \
    --cpu-kv-backend "$CPU_KV_BACKEND" \
    --topk 2048 --q-heads 16 --topk-buffer-size 4096 --plan-direct on \
    --warmup 10 --iters 50 --repeat 1 --kernel-time 200 --disable-gc \
    --hit-layout strided \
    --plan-slots off,scattered \
    --hit-ratio "$RATIOS" >"$LOG" 2>&1
rc=$?

echo "==================================================================="
echo " 退出码 $rc　$(date +%H:%M:%S)"
echo "==================================================================="
tail -n 40 "$LOG"
echo "==================================================================="
exit "$rc"
