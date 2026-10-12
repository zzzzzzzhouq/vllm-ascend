#!/usr/bin/env bash
# 三组标准测试（hitratio / batchsize / seqlen）+ 归因/验证模式的统一入口。
# 从 poolingDSA 的 sweep_up0723.sh 移植：方法论原样保留（A/A 伴生标签、
# plan 轴同趟轮换、串行固定单卡），去掉写死的卡号与日志目录，新增 seq 组。
#
#   bash yq_files/sweep_cases.sh hit      # 命中率曲线 @bs=16，44 点/进程
#   bash yq_files/sweep_cases.sh hit4     # 命中率曲线 @bs=4， 44 点/进程（helper 主战场）
#   bash yq_files/sweep_cases.sh bs       # BS 曲线 @80% 命中，40 点/进程
#   bash yq_files/sweep_cases.sh bs0      # BS 曲线 @0%  命中，40 点/进程
#   bash yq_files/sweep_cases.sh seq      # seqlen 曲线 @80% 命中，5 档 msl ×4 点/进程
#   bash yq_files/sweep_cases.sh frag     # 连续性归因，  16 点/进程
#   bash yq_files/sweep_cases.sh ablate   # 消融归因单点， 2 点/进程
#   bash yq_files/sweep_cases.sh validate # 串行 vs 并行是否等价，20 点/进程
#
# 覆盖矩阵的环境变量：RATIOS_MAIN / RATIOS_AA（命中率点与 A/A 标签）、
# AA_LABELS=0（关误差棒）、BS_LIST（batchsize 档位）、SEQ_LENS（msl 档位）
#
# ⚠ **前台跑，别加 nohup / &。** 提示符回来 = 跑完了，末尾有横幅。怕 SSH 断用 tmux。
# ⚠ 跑的过程中别 git pull —— bash 边读边执行脚本文件，改掉会从中间错位。
# ⚠ 一次只跑一趟（脚本有占用检查）：主机内存带宽共用，两趟叠跑会偏向性污染
#   miss 重的那一端。
#
# 覆盖项（环境变量）：
#   PROCS       进程数，默认 5（串行取中位数，耗时的主项）
#   DEVICES     卡列表，默认 0。**串行只用第一张**；后面几张只有 validate 用
#   DSA_LOGDIR  日志目录，默认 <repo>/yq_logs
#   SEQ_LENS    seq 组的 msl 档位，默认 4096,8192,16384,32768,65536
#   PLAN_AXIS   砍掉 plan 轴，如 PLAN_AXIS=scattered 只跑 LRU 稳定态（省一半时间）
#   SMOKE=1     换假 bench 空跑验流程，秒回不碰 NPU。**改过脚本先 SMOKE=1**
#   PARALLEL=1  多卡并行。⚠ yq 实测已验不可用（并行臂 A/A 13.6%），只留给复验
#   ABL_RATIOS / ABL_PLANS  ablate 模式的点与布局
#   QUICK=1     validate 只留两端

MODE="${1:-}"
case "$MODE" in
    hit|hit4|bs|bs0|seq|frag|ablate|validate) ;;
    *) echo "用法: bash yq_files/sweep_cases.sh hit|hit4|bs|bs0|seq|frag|ablate|validate" >&2; exit 2 ;;
esac

DEVICES="${DEVICES:-0}"
PARALLEL="${PARALLEL:-0}"
SMOKE="${SMOKE:-0}"
BENCH_ARG=()
if [ "$SMOKE" = "1" ]; then
    BENCH_ARG=(--bench yq_files/fake_bench.py)
    echo "⚠⚠ SMOKE=1：用假 bench 空跑，数字**没有任何测量意义**，只验流程通不通。"
fi
# 串行时用哪张卡：调用方 export 了就听它的，没有就取 DEVICES 的第一张。
: "${ASCEND_RT_VISIBLE_DEVICES:=${DEVICES%%,*}}"
export ASCEND_RT_VISIBLE_DEVICES

REPO="${DSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO" || exit 1

_busy=""
[ "$SMOKE" = "1" ] || _busy="$(pgrep -f 'test_fuse_sparse_attention_overlap_precisions|sweep_median\.py' 2>/dev/null || true)"
if [ -n "$_busy" ]; then
    echo "❌ 已经有一趟在跑，别叠着跑 —— 主机内存带宽是共用的，两边的数字都会被污染。" >&2
    ps -o pid=,etime=,cmd= -p $(echo "$_busy" | tr '\n' ',' | sed 's/,$//') 2>/dev/null | sed 's/^/     /' >&2
    echo "   等它跑完，或者先停掉：pkill -f sweep_cases; pkill -f sweep_median.py; pkill -f test_fuse" >&2
    exit 1
fi

LOGDIR="${DSA_LOGDIR:-$REPO/yq_logs}"
mkdir -p "$LOGDIR"
PROCS="${PROCS:-5}"
STAMP="$(date +%Y%m%d-%H%M%S)-$(git -C "$REPO" rev-parse --short=8 HEAD 2>/dev/null || echo nochash)"
T0="$(date +%s)"

if [ "$SMOKE" != "1" ]; then
    # shellcheck disable=SC1090
    source "$REPO/yq_files/env.sh" "$REPO"
    if [ -z "$ASCEND_CUSTOM_OPP_PATH" ]; then
        echo "❌ ASCEND_CUSTOM_OPP_PATH 为空 —— 环境没起来（vendor 包编了吗？）" >&2
        exit 1
    fi
fi

# 命中率点：默认 11 点（0→100% 每 10%）；后 11 个是 A/A 伴生标签
# （round(r*2048) 与对应整点完全相同，只是名字不同、排在队尾，
#  两次读数的差 = 这趟自己的误差棒，cmp_sweeps 的显著性门槛由它算出。
#  ⚠ 标签必须在小数点后 4 位互不相同；加减号按「加 0.0001 后 round 会不会
#    跳一格」逐个挑的：0.3/0.8 两端用 -0.0001，其余用 +0.0001）。
# 想换点数用环境变量覆盖：RATIOS_MAIN=... RATIOS_AA=... AA_LABELS=0（关误差棒）
RATIOS_MAIN="${RATIOS_MAIN:-0.0000,0.1000,0.2000,0.3000,0.4000,0.5000,0.6000,0.7000,0.8000,0.9000,1.0000}"
RATIOS_AA="${RATIOS_AA:-0.0001,0.1001,0.2001,0.2999,0.4001,0.5001,0.6001,0.7001,0.7999,0.9001,0.9999}"
# BS 档位（batchsize 组）：默认 10 档。
BS_LIST="${BS_LIST:-1,2,4,8,12,13,16,32,64,128}"
AA_LABELS="${AA_LABELS:-1}"
if [ "$AA_LABELS" = "1" ]; then
    HIT_RATIO_ARG="$RATIOS_MAIN,$RATIOS_AA"
else
    HIT_RATIO_ARG="$RATIOS_MAIN"
fi

COMMON_TAIL="--topk 2048 --q-heads 16 --topk-buffer-size 4096 --plan-direct on"
COMMON_TAIL="$COMMON_TAIL --warmup 10 --iters 50 --repeat 1 --kernel-time 200 --disable-gc"
COMMON_TAIL="$COMMON_TAIL --hit-layout strided"
COMMON="--max-seq-len 65536 $COMMON_TAIL"
PLANS="--plan-slots ${PLAN_AXIS:-off,scattered}"

PAR_ARGS=()
WHERE="串行，卡 $ASCEND_RT_VISIBLE_DEVICES"
if [ "$PARALLEL" = "1" ]; then
    PAR_ARGS=(--devices "$DEVICES")
    WHERE="并行，卡 $DEVICES"
    echo "⚠⚠ PARALLEL=1：多卡并行在 yq 实测已验不可用（并行臂自己的 A/A 13.64%，" >&2
    echo "   形状被改）。这一趟的数字不能跟串行基线比。要放开先跑 validate。" >&2
fi

case "$MODE" in
  hit)
    LOG="$LOGDIR/case_hit_curve_$STAMP.log"
    echo "命中率曲线：11 点 × 2 条路 × 2 标签 = 44 点/进程 × $PROCS 进程（$WHERE）"
    echo "日志 $LOG"
    echo "开始 $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo "${PAR_ARGS[@]}" "${BENCH_ARG[@]}" -- \
        --batch-size 16 $COMMON $PLANS --hit-ratio "$HIT_RATIO_ARG" > "$LOG" 2>&1
    RC=$?
    ;;
  hit4)
    LOG="$LOGDIR/case_hit4_curve_$STAMP.log"
    echo "命中率曲线 @ bs=4：11 点 × 2 条路 × 2 标签 = 44 点/进程 × $PROCS 进程（$WHERE）"
    echo "日志 $LOG"
    echo "开始 $(date '+%H:%M:%S')"
    # 与 hit 只差 batch size。bs=16 时 24 核已用满，「用上空闲核」（helper）收益
    # 恒为零 —— bs=16 那条是零对照臂（必须平），bs≤8 这条才是战场。
    # shellcheck disable=SC2086
    python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo "${PAR_ARGS[@]}" "${BENCH_ARG[@]}" -- \
        --batch-size 4 $COMMON $PLANS --hit-ratio "$HIT_RATIO_ARG" > "$LOG" 2>&1
    RC=$?
    ;;
  bs)
    LOG="$LOGDIR/case_bs_curve_$STAMP.log"
    echo "BS 曲线：10 个 bs × 2 条路 × 2 标签 = 40 点/进程 × $PROCS 进程（$WHERE）"
    echo "日志 $LOG"
    echo "开始 $(date '+%H:%M:%S')"
    # 命中率固定 80%（标量受限那一端）。0.7999 与 0.8000 的 round(r*2048) 都是
    # 1638，是真正的 A/A。
    # shellcheck disable=SC2086
    python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo "${PAR_ARGS[@]}" "${BENCH_ARG[@]}" -- \
        --batch-size "$BS_LIST" $COMMON $PLANS --hit-ratio 0.8000,0.7999 > "$LOG" 2>&1
    RC=$?
    ;;
  bs0)
    LOG="$LOGDIR/case_bs0_curve_$STAMP.log"
    echo "BS 曲线 @ 0% 命中：10 个 bs × 2 条路 × 2 标签 = 40 点/进程 × $PROCS 进程（$WHERE）"
    echo "日志 $LOG"
    echo "开始 $(date '+%H:%M:%S')"
    # 固定 0%（搬运受限那一端，AIV MTE2 占 79%）—— 判「用上空闲核能快多少」
    # 必须看这条。round(0.0001*2048)=0 与 0.0000 同为 miss=2048，是真正的 A/A。
    # shellcheck disable=SC2086
    python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo "${PAR_ARGS[@]}" "${BENCH_ARG[@]}" -- \
        --batch-size "$BS_LIST" $COMMON $PLANS --hit-ratio 0.0000,0.0001 > "$LOG" 2>&1
    RC=$?
    ;;
  seq)
    # seqlen 组：msl 逐档跑（bench 的 --max-seq-len 是单值参数，不支持逗号列表）。
    # 命中率固定 80% + A/A 伴生标签，bs=16，两条 plan 路同趟轮换。
    # 每档 msl 一个日志 —— A/B 对比时逐档 cmp（基线/优化各跑一遍同名模式）。
    # ⚠ msl 越大 CPU KV 池越大（bench 会按 msl 自动算 dram-size），131072 起先
    #   估内存（bench 的 CASE 行会打印 num_blocks）。
    SEQ_LENS="${SEQ_LENS:-4096,8192,16384,32768,65536}"
    echo "seqlen 曲线 @80% 命中：msl 档位 $SEQ_LENS，每档 1 点 × 2 条路 × 2 标签 = 4 点/进程 × $PROCS 进程"
    echo "开始 $(date '+%H:%M:%S')"
    RC=0
    SEQ_LOGS=""
    IFS=',' read -r -a _lens <<< "$SEQ_LENS"
    for m in "${_lens[@]}"; do
        LOG="$LOGDIR/case_seq_msl${m}_$STAMP.log"
        SEQ_LOGS="$SEQ_LOGS$LOG
"
        echo "  msl=$m → $LOG"
        # shellcheck disable=SC2086
        python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo "${PAR_ARGS[@]}" "${BENCH_ARG[@]}" -- \
            --batch-size 16 --max-seq-len "$m" $COMMON_TAIL $PLANS --hit-ratio 0.8000,0.7999 > "$LOG" 2>&1
        _rc=$?
        [ $_rc -ne 0 ] && RC=$_rc
        echo "  msl=$m 完成 $(date '+%H:%M:%S')（退出码 $_rc）"
    done
    LOG="$(echo "$SEQ_LOGS" | head -1)"
    ;;
  ablate)
    LOG="$LOGDIR/case_ablate_$STAMP.log"
    # 消融归因：只测一个物理点（默认全命中 + LRU 稳定态），2 点/进程。
    # ⚠⚠ 消融构建是测量专用的，结果是错的，别拿它跑精度用例。
    ABL_RATIOS="${ABL_RATIOS:-1.0000,0.9999}"
    ABL_PLANS="${ABL_PLANS:-scattered}"
    echo "消融归因：命中率 $ABL_RATIOS × plan=$ABL_PLANS × $PROCS 进程（$WHERE）"
    echo "日志 $LOG"
    echo "开始 $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo "${PAR_ARGS[@]}" "${BENCH_ARG[@]}" -- \
        --batch-size 16 $COMMON --plan-slots "$ABL_PLANS" --hit-ratio "$ABL_RATIOS" > "$LOG" 2>&1
    RC=$?
    ;;
  frag)
    LOG="$LOGDIR/case_frag_$STAMP.log"
    # 连续性归因：--hit-ratio 是「多少名次命中」，--plan-slots 是「命中的名次
    # 落到哪个槽位」。contiguous=理想连续（快路生效），scattered=LRU 稳定态。
    # 两者的差 = 碎片化的代价。⚠ contiguous 不是能拿来用的配置，别当「优化后」。
    R="0.0000,0.5000,0.9000,1.0000,0.0001,0.5001,0.9001,0.9999"
    echo "连续性归因：4 点 × 2 布局 × 2 标签 = 16 点/进程 × $PROCS 进程（$WHERE）"
    echo "日志 $LOG"
    echo "开始 $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    python3 yq_files/sweep_median.py --processes "$PROCS" --e2e --echo "${PAR_ARGS[@]}" "${BENCH_ARG[@]}" -- \
        --batch-size 16 $COMMON --plan-slots contiguous,scattered --hit-ratio "$R" > "$LOG" 2>&1
    RC=$?
    ;;
  validate)
    # 串行 vs 并行等价性复验。0% 查并发争抢、100% 查跨芯片偏置、50% 防中间塌。
    R="0.0000,0.5000,1.0000,0.0001,0.5001,0.9999"
    [ "${QUICK:-0}" = "1" ] && R="0.0000,1.0000,0.0001,0.9999"
    A="$LOGDIR/case_validate_serial_$STAMP.log"
    B="$LOGDIR/case_validate_parallel_$STAMP.log"
    D1="${DEVICES%%,*}"
    echo "验证并行是否等价：3 点 × 2 条路 × 2 标签 = 12 点/进程"
    echo "  A 串行  $PROCS 进程，全在卡 $D1"
    echo "  B 并行  8 进程，卡 $DEVICES"
    echo "开始 $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    ASCEND_RT_VISIBLE_DEVICES="$D1" python3 yq_files/sweep_median.py --processes "$PROCS" "${BENCH_ARG[@]}" -- \
        --batch-size 16 $COMMON $PLANS --hit-ratio "$R" > "$A" 2>&1
    echo "  A 完成 $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    python3 yq_files/sweep_median.py --processes 8 --devices "$DEVICES" "${BENCH_ARG[@]}" -- \
        --batch-size 16 $COMMON $PLANS --hit-ratio "$R" > "$B" 2>&1
    echo "  B 完成 $(date '+%H:%M:%S')"
    LOG="$B"
    EXTRA_LOG="$A"
    echo
    python3 yq_files/cmp_sweeps.py "$A" "$B" --label-a 串行 --label-b 并行
    RC=$?
    echo
    if [ $RC -eq 0 ]; then
        echo "  → 可以并行跑了：PARALLEL=1 bash yq_files/sweep_cases.sh hit"
        echo "  ⚠ 此后所有对比必须用同一组卡（$DEVICES），换卡组要重验。"
    else
        echo "  → 别并行。按串行口径跑：bash yq_files/sweep_cases.sh hit"
    fi
    ;;
esac

rc=${RC:-1}
SECS=$(( $(date +%s) - T0 ))
# 点数从日志里数，不从参数推 —— 中途报错少测了点的话，推出来的是错的。
_log_args=()
if [ "$MODE" = seq ]; then
    while IFS= read -r _l; do [ -n "$_l" ] && _log_args+=("$_l"); done <<< "$SEQ_LOGS"
else
    _log_args+=("$LOG")
    [ -n "${EXTRA_LOG:-}" ] && _log_args+=("$EXTRA_LOG")
fi
PTS=$(cat "${_log_args[@]}" 2>/dev/null |
      grep -oE 'KERNEL [0-9]+ 点' | awk '{s+=$2} END{print s+0}')

echo
echo "==================================================================="
if [ "$MODE" = validate ]; then
    if [ "$rc" -eq 0 ]; then
        echo "  ✅ 验证通过：串行和并行等价　$(date '+%H:%M:%S')"
    else
        echo "  ⚠ 验证不通过：两种设定量出来的不是同一个东西　$(date '+%H:%M:%S')"
    fi
elif [ "$rc" -eq 0 ]; then
    echo "  ✅ 跑完了　$(date '+%H:%M:%S')　退出码 0"
else
    echo "  ❌ 中断了　$(date '+%H:%M:%S')　退出码 $rc　—— 下面的表不要用"
fi
awk -v s="$SECS" -v p="$PTS" -v n="$PROCS" 'BEGIN {
    printf "  用时 %d:%02d:%02d（%d 秒）　测了 %d 个点（%d 进程）", s/3600, s%3600/60, s%60, s, p, n
    if (p > 0) printf "　平均 %.1f s/点", s/p
    printf "\n"
    if (p > 0) printf "  下次估时：点数 × %.1f 秒。点数 = 标签数 × 路数(--plan-slots 给了几个) × 进程数\n", s/p
}'
echo "==================================================================="
echo
echo "===== 进度（每行的点数要一样，不一样就是有进程少测了）====="
cat "${_log_args[@]}" 2>/dev/null | grep -E '^  进程 [0-9]+/'
echo
echo "===== 告警（一条都别忽略）====="
cat "${_log_args[@]}" 2>/dev/null | grep -nE '⚠|SWEEP NOT SELF-CONSISTENT|WARNING|Traceback' | head -40
echo
echo "===== 这一趟测的是哪份代码 ====="
sed -n '1,12p' "$(echo "${_log_args[0]}" | head -1)" 2>/dev/null | grep -vE '^\s*$'
echo
echo "整表在日志文件里，贴回去的是 kernel 和端到端两张中位数表："
if [ "$MODE" = seq ]; then
    echo "  各档 msl 的日志在上面逐行列出；A/B 对比逐档："
    echo "  python3 yq_files/cmp_sweeps.py 基线/case_seq_msl<msl>_*.log 优化/case_seq_msl<msl>_*.log"
else
    echo "  sed -n '/=== kernel duration：绝对值/,\$p' $LOG"
fi
