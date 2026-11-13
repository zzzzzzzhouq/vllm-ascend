#!/usr/bin/env bash
# per_test 一键入口：环境自动就绪，不需要先 source env.sh，也不依赖目录名。
#
# 用法：
#   bash per_test/run.sh env                  # 只打印解析出的仓库/环境，跑测试前先自检
#   bash per_test/run.sh build [算子名]       # 编译 vendor 算子包 + vllm_ascend_C .so
#   bash per_test/run.sh precision            # 精度门禁（小内存版，默认；约 90 个用例）
#   bash per_test/run.sh precision-full       # 完整精度用例（需要更大的 MemFabric 池）
#   bash per_test/run.sh perf [mode]          # 性能曲线，mode 默认 hit
#                                             #   hit|hit4|bs|bs0|seq|frag|ablate|validate
#   bash per_test/run.sh perf-small           # 小内存性能冒烟（内存受限机器）
#   bash per_test/run.sh compare A.log B.log  # 两趟 sweep 逐点比较
#   bash per_test/run.sh check 本次.log [干净.log]   # 查这趟有没有被别的任务干扰
#
# 常用覆盖：
#   DEVICES=0,1,2,3 bash per_test/run.sh perf hit       # 串行仍只用第一张卡
#   PROCS=3 bash per_test/run.sh perf hit               # 进程数（取中位数，耗时主项）
#   SMOKE=1 bash per_test/run.sh perf hit               # 假 bench 空跑，秒回，不碰 NPU
#   DSA_CPU_KV_BACKEND=swap_memory bash per_test/run.sh precision
#   DSA_REPO=/别的/clone bash per_test/run.sh env       # 显式指定仓库
#
# ⚠ 性能测试跑之前先 `SMOKE=1` 空跑一遍；精度和性能都要在同一份编译产物上跑。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="${DSA_REPO:-$(cd "$HERE/.." && pwd)}"

usage() {
    cat <<'EOF'
用法：
  bash per_test/run.sh env                  # 打印解析出的仓库/环境
  bash per_test/run.sh build [算子名]       # 编译 vendor 算子包 + vllm_ascend_C .so
  bash per_test/run.sh precision            # 精度门禁（小内存版，默认）
  bash per_test/run.sh precision-full       # 完整精度用例
  bash per_test/run.sh perf [mode]          # 性能曲线，mode 默认 hit
                                            #   hit|hit4|bs|bs0|seq|frag|ablate|validate
  bash per_test/run.sh perf-small           # 小内存性能冒烟
  bash per_test/run.sh compare A.log B.log  # 两趟 sweep 逐点比较
  bash per_test/run.sh check 本次.log [干净.log]

常用覆盖：
  DEVICES=0,1,2,3 PROCS=3 bash per_test/run.sh perf hit
  SMOKE=1 bash per_test/run.sh perf hit             # 假 bench 空跑，不碰 NPU
  DSA_CPU_KV_BACKEND=swap_memory bash per_test/run.sh precision
  DSA_REPO=/别的/clone bash per_test/run.sh env
EOF
}

ensure_env() {
    # env.sh 幂等；每次都用本文件所在目录重新解析，避免旧 shell 里的 DSA_REPO 残留。
    # shellcheck disable=SC1091
    source "$HERE/env.sh" "$REPO"
}

cmd="${1:-help}"
[ "$#" -gt 0 ] && shift

case "$cmd" in
    help|-h|--help)
        usage
        ;;
    env)
        ensure_env
        echo
        echo "per_test  = $HERE"
        echo "repo      = $DSA_REPO"
        echo "SOC       = ${SOC_VERSION:-<未设置>}"
        echo "vendor op = ${ASCEND_CUSTOM_OPP_PATH:-<未设置，先跑 run.sh build>}"
        echo "python    = $(command -v python3 || echo '<未找到 python3>')"
        echo "pytest    = $(command -v pytest || echo '<未找到 pytest>')"
        echo "日志目录  = ${DSA_LOGDIR:-$HERE/logs}"
        ;;
    build)
        # build_ops.sh 自己处理 SOC_VERSION/CANN，不需要先 source env.sh。
        # 给了参数就只编该算子（等价于 OPS=<算子名> build_ops.sh）。
        if [ "$#" -gt 0 ]; then
            export OPS="$1"
            shift
        fi
        exec bash "$HERE/build_ops.sh" "$REPO" "$@"
        ;;
    precision)
        # 小内存版：池 0.06 GiB，自动 source env，日志写 $HERE/logs。
        exec bash "$HERE/precision_small.sh" "$@"
        ;;
    precision-full)
        ensure_env
        cd "$DSA_REPO" || exit 1
        # 完整版不压池，需要 ulimit -l 足够大（env.sh 会尽力抬高）。
        exec pytest -sv "$HERE/tests/test_fsao_precision.py" "$@"
        ;;
    perf)
        # 标准曲线；SMOKE=1 时交给 sweep_cases.sh 用假 bench 空跑。
        [ "$#" -gt 0 ] || set -- hit
        if [ "${SMOKE:-0}" != "1" ]; then
            ensure_env
        fi
        exec bash "$HERE/sweep_cases.sh" "$@"
        ;;
    perf-small|small)
        exec bash "$HERE/sweep_small.sh" "$@"
        ;;
    compare)
        exec python3 "$HERE/cmp_sweeps.py" "$@"
        ;;
    check)
        exec python3 "$HERE/check_contention.py" "$@"
        ;;
    *)
        echo "未知子命令: $cmd" >&2
        echo >&2
        usage >&2
        exit 2
        ;;
esac
