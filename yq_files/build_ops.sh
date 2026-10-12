#!/usr/bin/env bash
# 在本仓库内就地全量编译：算子 vendor 包 + vllm_ascend_C .so，编完即可测。
#
# 与原仓库（poolingDSA）build_ops.sh 的关键差异，改脚本前先看：
#   1) 本仓库算子编译入口是 `python3 setup.py build_aclnn`
#      （→ csrc/build_aclnn.sh → csrc/build.sh --pkg，.run 产物直接就地装进
#      vllm_ascend/_cann_ops_custom/vendors/custom_transformer）。
#      setup.py build_ext **不会**带它 —— 两步都要跑，缺一步就是
#      「.so 换了 kernel 没换」或反过来。
#   2) 本仓库 setup.py 不自动探测 SOC_VERSION，缺了 build_aclnn 会静默跳过
#      算子编译。这里兜底设 ascend910_93（A3）。
#   3) 运行时不需要手动 export ASCEND_CUSTOM_OPP_PATH：vllm_ascend.utils 的
#      bootstrap_custom_op_env 在 import 时自动补 vendor 路径
#      （source yq_files/env.sh 显式导出是给裸 torch 脚本用的）。
#
# 用法：bash yq_files/build_ops.sh 2>&1 | tee /tmp/build_ops.txt
# 开关：OPS=<算子名>    只编单算子（走 csrc/build.sh 直编，绕过 catlass 子模块
#                       和全量算子列表，快一个数量级；内网拉不动 gitcode 时必用）
#       INCREMENTAL=1     跳过删 build（测出来的数字不可信，只用于调编译错误）
#       SKIP_TOUCH=1      跳过 touch 改动过的算子目录
#       TOUCH_BASE_REF=<ref>  touch 的 git 基准（默认 main）
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
REPO="${1:-${DSA_REPO:-}}"
if [ -n "$REPO" ] && [ ! -d "$REPO" ]; then
    echo "⚠ 参数/DSA_REPO 指向的目录不存在: $REPO"
    echo "  多半是之前 source env.sh 导出的 DSA_REPO 残留（指向了旧克隆/已改名的目录）。"
    echo "  改用脚本所在仓库: $_here。如需强制指定：unset DSA_REPO 后重跑，或传显式路径。"
    REPO=""
fi
REPO="${REPO:-$_here}"
echo "REPO=$REPO"
cd "$REPO" || exit 1

if [ -z "${SOC_VERSION:-}" ]; then
    export SOC_VERSION=ascend910_9391
    echo "⚠ SOC_VERSION 未导出，兜底设为 $SOC_VERSION（A3 具体型号；cmake 只认型号，不认家族名）"
fi
echo "SOC_VERSION=$SOC_VERSION"

# csrc/build.sh 的 --soc 要家族名（ascend910_93），从具体型号推导：
case "$SOC_VERSION" in
    ascend910_93*) SOC_FAMILY=ascend910_93 ;;
    ascend910b*)   SOC_FAMILY=ascend910b ;;
    ascend310p*)   SOC_FAMILY=ascend310p ;;
    ascend950*)    SOC_FAMILY=ascend950 ;;
    *)             SOC_FAMILY="$SOC_VERSION" ;;
esac
echo "SOC_FAMILY=$SOC_FAMILY（csrc/build.sh --soc 用）"

# CANN toolkit
if [ -z "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u
    echo "sourced ascend-toolkit set_env.sh"
fi

ALL_CORES="$(nproc --all 2>/dev/null || echo 4)"
JOBS="${MAX_JOBS:-$(( ALL_CORES < 64 ? ALL_CORES : 64 ))}"
export MAX_JOBS="$JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
echo "MAX_JOBS=$MAX_JOBS (物理核 $ALL_CORES)"

# ---------------------------------------------------------------------------
# touch 改过的算子目录：csrc/ 下的 .h 没写进 CMake 依赖，只改头文件跑增量编译，
# 依赖它的 kernel 不会重编 —— 退出码 0、日志干净、跑的还是旧代码。这是本项目
# 最贵的一类失败。按 git 找出相对基准有改动的文件，把它们所在目录整个 touch。
# ⚠ touch 只保证「改动可见」，不保证「一定重编」—— 所以下面默认仍然全量删 build。
# ---------------------------------------------------------------------------
if [ "${SKIP_TOUCH:-0}" != "1" ] && [ -d "$REPO/.git" ]; then
    _touch_base="${TOUCH_BASE_REF:-main}"
    if git -C "$REPO" rev-parse --verify --quiet "$_touch_base" >/dev/null; then
        _changed=$(git -C "$REPO" diff --name-only "$_touch_base" -- csrc/ 2>/dev/null || true)
        if [ -n "$_changed" ]; then
            _dirs=$(printf '%s\n' "$_changed" | xargs -n1 dirname | sort -u)
            _n=0
            for _d in $_dirs; do
                [ -d "$REPO/$_d" ] || continue
                touch "$REPO/$_d"/* 2>/dev/null || true
                _n=$((_n + 1))
            done
            echo "已 touch $_n 个改动过的算子目录（相对 $_touch_base）："
            printf '%s\n' "$_dirs" | sed 's/^/  /'
        else
            echo "csrc/ 与 $_touch_base 一致，无需 touch"
        fi
    else
        echo "⚠ 找不到基准 $_touch_base，跳过 touch —— 只改头文件时增量编译可能不重编 kernel"
    fi
fi

# ---------------------------------------------------------------------------
# 默认全量：自己删 build，不指望调用方记得。INCREMENTAL=1 只用来调编译错误。
# ---------------------------------------------------------------------------
if [ "${INCREMENTAL:-0}" = "1" ]; then
    echo "⚠⚠ INCREMENTAL=1：跳过删 build。编出来的 kernel 很可能还是上一版 —— 别拿它测任何数字。"
else
    echo "删除旧构建产物（全量编译）..."
    rm -rf "$REPO/build" "$REPO/csrc/build" "$REPO/csrc/build_out" "$REPO/csrc/output"
fi

BUILD_LOG="${BUILD_LOG:-/tmp/build_ops_full.log}"
: > "$BUILD_LOG"

echo
if [ -n "${OPS:-}" ]; then
    # 单算子模式：OPS=fused_sparse_attention_overlap bash yq_files/build_ops.sh
    # csrc/build.sh 直编 + .run 就地安装。不经过 setup.py build_aclnn，因此
    # 不会触发 catlass 子模块拉取（内网代理 407 踩过），只编 --ops 指定的算子。
    # csrc/build.sh 的 --soc 用家族名（$SOC_FAMILY），cmake 用具体型号（$SOC_VERSION）。
    # ⚠ 若换别的芯片，核对 build_ops.sh 顶部 SOC_FAMILY 的 case 映射是否覆盖。
    echo "==================== [1/2] 算子 vendor 包（单算子 $OPS，csrc/build.sh 直编） ===================="
    (
        set -e
        cd csrc
        rm -rf -- output build_out
        bash build.sh --pkg --ops="$OPS" --soc="$SOC_FAMILY"
        shopt -s nullglob
        run_installer=(./build/cann-ops-transformer*.run)
        shopt -u nullglob
        [ ${#run_installer[@]} -eq 1 ] || { echo "ERROR: 期望 1 个 .run 安装包，实际 ${#run_installer[@]} 个"; exit 1; }
        chmod +x -- "${run_installer[0]}"
        "${run_installer[0]}" --install-path="$REPO/vllm_ascend/_cann_ops_custom"
        if [ -d "$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/scripts" ]; then
            chmod u+w "$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/scripts"
        fi
    ) 2>&1 | tee -a "$BUILD_LOG"
    rc1=${PIPESTATUS[0]}
else
    echo "==================== [1/2] 算子 vendor 包（setup.py build_aclnn） ===================="
    python3 setup.py build_aclnn 2>&1 | tee -a "$BUILD_LOG"
    rc1=${PIPESTATUS[0]}
fi

echo
echo "==================== [2/2] vllm_ascend_C .so（setup.py build_ext --inplace） ===================="
# ⚠ setup.py 的 cmake_build_ext 在 COMPILE_CUSTOM_KERNELS=1（默认）时，每次都会先
#   重跑一遍全量 build_aclnn（40+ 算子 + catlass 子模块）——单算子迭代纯属浪费。
#   .so 只含 pybind/torch_binding 注册层：改 kernel / op_host tiling 不影响它。
#   ⚠ 若改了 csrc/torch_binding.cpp（算子 schema），请 rm vllm_ascend/*.so 强制全量重编。
if ls vllm_ascend/*.so >/dev/null 2>&1; then
    echo ".so 已存在 → COMPILE_CUSTOM_KERNELS=0 跳过内部的全量 build_aclnn 重触发"
    COMPILE_CUSTOM_KERNELS=0 python3 setup.py build_ext --inplace 2>&1 | tee -a "$BUILD_LOG"
else
    echo "未发现已有 .so → 首次完整编（内含全量 build_aclnn，需要 catlass 子模块，一次性成本）"
    if [ -n "${OPS:-}" ]; then
        echo "OPS=$OPS → 内部触发的 build_aclnn 也只编该算子（VLLM_ACLNN_OPS），"
        echo "  绕开与本地 CANN 版本不兼容的其他算子（如 OP_LOGE_FOR_INVALID_* 宏族）"
        VLLM_ACLNN_OPS="$OPS" python3 setup.py build_ext --inplace 2>&1 | tee -a "$BUILD_LOG"
    else
        python3 setup.py build_ext --inplace 2>&1 | tee -a "$BUILD_LOG"
    fi
fi
rc2=${PIPESTATUS[0]}

echo
echo "==================== 结果 ===================="
echo "build_aclnn 退出码: $rc1 / build_ext 退出码: $rc2"
echo "完整编译日志: $BUILD_LOG"
rc="$rc1"
[ "$rc2" -ne 0 ] && rc="$rc2"

# 只认本仓库目录下的 .so。环境里若有另一份编好的 vllm-ascend，直接 import 会
# 拿到那一份，编译失败却报「算子有」。
SO_COUNT=$(ls -1 vllm_ascend/*.so 2>/dev/null | wc -l | tr -d ' ')
if [ "$SO_COUNT" -eq 0 ]; then
    echo "❌ 本仓库 vllm_ascend/ 下未生成任何 .so —— 编译没成功"
    echo
    echo "-- 日志里的错误行 --"
    grep -nE 'CMake Error|error:|Error running|No such file|not found|Configuring incomplete' "$BUILD_LOG" \
        | head -20 || echo "（没抓到明显错误行，看完整日志）"
    exit 1
fi
ls -lh vllm_ascend/*.so

# kernel 指纹：算子 kernel 二进制的 md5。
# ⚠ 时间戳、文件名里那串哈希、编译日志里出现算子名 —— 三样全都骗人，只有内容
#   不骗人。记下这个数，跟上次编译一比就知道 kernel 换没换。
_ko=$(ls vllm_ascend/_cann_ops_custom/vendors/*/op_impl/ai_core/tbe/kernel/*/fused_sparse_attention_overlap/*.o 2>/dev/null | head -1)
if [ -n "$_ko" ]; then
    echo "kernel 指纹: $(md5sum < "$_ko" | cut -d' ' -f1)  $(basename "$_ko")"
else
    echo "⚠ 没找到算子 kernel 的 .o —— vendor 包没装上？看 build_aclnn 段日志"
fi

python3 - <<'PYEOF'
import os, torch
import vllm_ascend
here = os.path.abspath(os.getcwd())
pkg = os.path.dirname(os.path.abspath(vllm_ascend.__file__))
print("import 到的 vllm_ascend:", pkg)
if not pkg.startswith(here):
    print(f"❌ 拿到的不是本仓库那份（cwd={here}），下面的结论不可信")
    raise SystemExit(1)
try:
    import vllm_ascend.vllm_ascend_C  # noqa
    ops = torch.ops._C_ascend
    n = "npu_fused_sparse_attention_overlap"
    print(f"{n:38s} {'有' if hasattr(ops, n) else '缺失'}")
except Exception as e:
    print(f"加载失败: {type(e).__name__}: {e}")
    raise SystemExit(1)
PYEOF

exit "$rc"
