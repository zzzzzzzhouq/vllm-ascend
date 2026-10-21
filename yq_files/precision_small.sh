#!/usr/bin/env bash
# 小内存版精度门禁：用于 memlock / 宿主机内存受限的机器（例如 ulimit -l 只有 64 MiB）。
#
# 与标准版（pytest yq_files/tests/test_fsao_precision.py）的区别：
#   1. MemFabric 池压到 DSA_DRAM_POOL_GB（默认 0.06 GiB ≈ 61 MiB，落在 64 MiB 内）
#   2. 跳过两条重尾巴用例类：baseline_batch_sizes / baseline_hit_ratios
#      —— 它们固定 msl=65536，单点宿主 KV 几百 MB，小池必炸
#   3. 其余用例（topk ≤ 1024、msl ≤ 4096）全部保留：
#      miss/hit 两条路等价性、selection cache、plan 各路径
#
# 判定：末尾必须是 "N passed"，0 failed；出现 skipped 不算通过。
#
# 用法：
#   source yq_files/env.sh
#   bash yq_files/precision_small.sh
#   DSA_DRAM_POOL_GB=0.05 bash yq_files/precision_small.sh     # 更紧
#   PREC_K="128" bash yq_files/precision_small.sh              # 自定义 -k 筛选
set -u

cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." || exit 1

if [ -z "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
    echo "⚠ 先 source yq_files/env.sh（ASCEND_CUSTOM_OPP_PATH 为空，算子会找不到）" >&2
    exit 1
fi

export DSA_DRAM_POOL_GB="${DSA_DRAM_POOL_GB:-0.06}"
# 宿主 MemFabric 池起不来（HYBM HalMemSetAccess 65534）时用 swap_memory：
# torch_npu.empty_with_swapped_memory —— 宿主内存但映射进 NPU 地址空间，
# kernel 的 MTE 能读（普通 pinned 内存不行，会 aicore exception）。
export DSA_CPU_KV_BACKEND="${DSA_CPU_KV_BACKEND:-swap_memory}"
ulimit -l unlimited 2>/dev/null || true

K="${PREC_K:-not baseline_batch_sizes and not baseline_hit_ratios}"

LOGDIR="${DSA_LOGDIR:-$PWD/yq_logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/precision_small_$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD 2>/dev/null || echo nogit).log"

echo "==================================================================="
echo " 小内存精度门禁"
echo "   池大小      : ${DSA_DRAM_POOL_GB} GiB（64 MiB 限制内）"
echo "   KV 后端     : ${DSA_CPU_KV_BACKEND}（swap_memory = 映射进 NPU 地址空间的宿主内存）"
echo "   memlock     : $(ulimit -l) KB"
echo "   用例筛选    : -k '$K'"
echo "   日志        : $LOG"
echo "   开始        : $(date +%H:%M:%S)"
echo "==================================================================="

pytest -sv yq_files/tests/test_fsao_precision.py -k "$K" >"$LOG" 2>&1
rc=$?

echo "==================================================================="
echo " 退出码 $rc　$(date +%H:%M:%S)"
grep -E "^[0-9]+ (passed|failed|skipped|error)|passed|failed" "$LOG" | tail -n 3
echo " 日志尾部："
tail -n 12 "$LOG"
echo "==================================================================="
exit "$rc"
