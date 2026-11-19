#!/usr/bin/env bash
# 一次 source 就绪：CANN toolkit + 目标芯片 + memfabric offload + 本仓库自定义算子 vendor 包。
# （原仓库 env.sh 写死了容器路径；atb 段与本仓库无关未照搬，memfabric 段保留——
#   bench / 精度用例的 CPU KV 池走 offload.empty，缺了必报
#   "offload launch load library failed" 并挂死。）
#
# 用法：
#   bash per_test/run.sh env                    # 一键入口：source 本文件并打印解析结果
#   source per_test/env.sh                      # 自动定位仓库（脚本上一级）
#   source per_test/env.sh /path/to/repo        # 显式指定（兼容原版用法）
#   DSA_REPO=/path/to/repo source per_test/env.sh
#
# 也可以不用手动 source：run.sh / sweep_cases.sh / sweep_small.sh /
# precision_small.sh 都会在需要时自己 source 本文件。
# per_test 目录可以改名/整体搬走，只要还在仓库根目录下（仓库 = 本文件上一级）。
#
# 幂等：重复 source 不会把路径变量越堆越长。
#
# 四段缺一不可：
#   - ascend-toolkit     : CANN 编译/运行时
#   - SOC_VERSION        : 本仓库 setup.py **不自动探测**，缺了 build_aclnn
#                          会静默跳过整个算子编译（只编 .so，看不出异常）
#   - memfabric          : 1.3+ 的 pip wheel 自带 libmf_*.so，import 时自己
#                          ctypes 加载，不需要 set_env.sh；只有老版本/.run 安装
#                          才需要 MEMFABRIC_HYBRID_EXTEND_LIB_PATH
#   - ASCEND_CUSTOM_OPP_PATH : vendor 包路径。vllm_ascend import 时也会自动补
#                          （utils.bootstrap_custom_op_env），显式导出是为了
#                          脱离 vllm_ascend 的裸 torch 脚本也能找到算子。
#   - PYTHONPATH         : 仓库前置。bench / 精度用例 import
#                          vllm_ascend.vllm_ascend_C（op 的 torch_binding 层）
#                          和 vllm_ascend.utils；机器上另装有 pip 版 vllm-ascend
#                          时不前置就会解析到别的包（yq 教训：一直用错包，
#                          白量一轮基线）。

# 本文件所在目录（per_test 根）。下游脚本用它定位内部文件，不依赖目录名。
export PER_TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

_dsa_repo="${1:-${DSA_REPO:-}}"
if [ -n "$_dsa_repo" ] && [ ! -d "$_dsa_repo" ]; then
    echo "[env.sh] ⚠ 参数/DSA_REPO 指向的目录不存在: $_dsa_repo —— 忽略"
    echo "[env.sh]   （可能是同一 shell 里切换克隆后的残留值；需要的话 unset DSA_REPO 重 source）"
    _dsa_repo=""
fi
if [ -z "$_dsa_repo" ]; then
    _dsa_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
fi
# ⚠ DSA_REPO 会导出到当前 shell 并长期留存；同一 shell 里切换到别的克隆后，
#   必须重新 source（或 unset DSA_REPO），否则脚本会继续用旧值。
export DSA_REPO="$_dsa_repo"
echo "[env.sh] repo = $_dsa_repo"

# 1) CANN toolkit。已 source 过（ASCEND_TOOLKIT_HOME 非空）就跳过。
#    atb 等脚本引用未定义变量，source 期间临时关 -u，完了按原样恢复。
if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        _dsa_had_u=0; case "$-" in *u*) _dsa_had_u=1; set +u ;; esac
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        [ "$_dsa_had_u" = 1 ] && set -u
        echo "[env.sh] sourced ascend-toolkit set_env.sh"
    else
        echo "[env.sh] ⚠ 找不到 /usr/local/Ascend/ascend-toolkit/set_env.sh —— CANN 装了吗？"
    fi
fi

# 2) 目标芯片。⚠ 两个消费者要的值不同：
#    - setup.py 的 cmake（编 .so）要**具体芯片型号**（CANN host_config.cmake
#      只认型号，如 ascend910_9382 / ascend910_9391，不认家族名 ascend910_93）
#    - csrc/build.sh --soc / build_aclnn.sh 要**家族名**，会自动从具体型号映射
#      回去（正则 ^ascend910_93）
#    SOC_VERSION 未导出时优先用 npu-smi 识别真实型号；识别不了就留空，交给
#    setup.py 自行探测（A2/310P/950 等非 A3 机器请手动 export 具体型号）。
if [ -z "${SOC_VERSION:-}" ]; then
    _dsa_chip=""; _dsa_npu=""
    if command -v npu-smi >/dev/null 2>&1; then
        _dsa_info="$(npu-smi info -m 2>/dev/null || true)"
        _dsa_chip="$(printf '%s\n' "$_dsa_info" | awk -F': ' '/Chip Name/{print $2; exit}')"
        _dsa_npu="$(printf '%s\n' "$_dsa_info" | awk -F': ' '/NPU Name/{print $2; exit}')"
    fi
    case "${_dsa_chip,,}" in
        *910*|*950*) if [ -n "$_dsa_npu" ]; then export SOC_VERSION="${_dsa_chip,,}_${_dsa_npu,,}"; fi ;;
    esac
    if [ -n "${SOC_VERSION:-}" ]; then
        echo "[env.sh] SOC_VERSION 未导出，已从 npu-smi 识别为 $SOC_VERSION"
    else
        echo "[env.sh] ⚠ SOC_VERSION 未导出且 npu-smi 未能识别 —— 交给 setup.py 自行探测；仍失败请手动 export SOC_VERSION=<具体型号，如 ascend910_9382>"
    fi
else
    echo "[env.sh] SOC_VERSION=$SOC_VERSION（沿用外部导出）"
    case "$SOC_VERSION" in
        ascend910_93|ascend910b|ascend310p|ascend950)
            echo "[env.sh] ⚠ 这是家族名：csrc/build.sh --soc 认它，但 setup.py 的 cmake"
            echo "[env.sh]   只认具体型号（A3=ascend910_9382）。build_ops.sh 会自动纠正，"
            echo "[env.sh]   想手动改就 unset SOC_VERSION 后重 source。"
            ;;
    esac
fi

# 3) 自定义算子 vendor 包（build_ops.sh 的产物就装在仓库里）。
_dsa_vendor="$_dsa_repo/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
if [ -d "$_dsa_vendor" ]; then
    case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
        *":$_dsa_vendor:"*) ;;
        *) export ASCEND_CUSTOM_OPP_PATH="$_dsa_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
           echo "[env.sh] ASCEND_CUSTOM_OPP_PATH += $_dsa_vendor" ;;
    esac
else
    echo "[env.sh] ⚠ $_dsa_vendor 不存在 —— 算子还没编？先跑 bash $PER_TEST_DIR/run.sh build"
fi

# 4) memfabric offload。
#    1.3+ 的 pip wheel 在 memfabric_hybrid/__init__.py 里直接 ctypes.CDLL
#    libmf_*.so，不依赖 set_env.sh / MEMFABRIC_HYBRID_EXTEND_LIB_PATH。
#    老版本（或 .run 安装）才需要 source set_env.sh 补库路径。
#    以「import 得动 + OffloadConfig 有 1.3 字段」作为可用判据，避免误报。
_dsa_mf_state=""
if [ -n "${MEMFABRIC_HYBRID_EXTEND_LIB_PATH:-}" ]; then
    echo "[env.sh] memfabric EXTEND_LIB_PATH=${MEMFABRIC_HYBRID_EXTEND_LIB_PATH}"
    _dsa_mf_state="ok"
else
    _dsa_mf_state="$(python3 -c '
from memfabric_hybrid import offload
c = offload.OffloadConfig()
print("ok" if (hasattr(c, "reserve_size") and hasattr(c, "alloc_size")
               and hasattr(offload, "Scene")) else "old")
' 2>/dev/null || true)"
fi

case "$_dsa_mf_state" in
    ok)
        if [ -z "${MEMFABRIC_HYBRID_EXTEND_LIB_PATH:-}" ]; then
            _dsa_mf_ver="$(python3 -c 'import importlib.metadata as m; print(m.version("memfabric_hybrid"))' 2>/dev/null || echo unknown)"
            echo "[env.sh] memfabric offload 可用（${_dsa_mf_ver}，wheel 自带 libmf_*.so，无需 set_env.sh）"
        fi
        ;;
    old)
        echo "[env.sh] ⚠ memfabric offload 是旧版 API（OffloadConfig 没有 reserve_size/alloc_size）"
        echo "[env.sh]   per_test 需要 1.3+，请升级：pip install -U memfabric_hybrid"
        ;;
    *)
        # import 失败：回退到 set_env.sh（老版本 / .run 安装）
        _dsa_mf_setenv="/usr/local/memfabric_hybrid/set_env.sh"
        if [ ! -f "$_dsa_mf_setenv" ]; then
            _dsa_mf_root="$(python3 -c 'import memfabric_hybrid, os; print(os.path.dirname(os.path.dirname(os.path.abspath(memfabric_hybrid.__file__))))' 2>/dev/null || true)"
            if [ -n "$_dsa_mf_root" ]; then
                _dsa_mf_found="$(find "$_dsa_mf_root" -maxdepth 3 -name set_env.sh 2>/dev/null | head -1)"
                [ -n "$_dsa_mf_found" ] && _dsa_mf_setenv="$_dsa_mf_found"
            fi
        fi
        if [ -f "$_dsa_mf_setenv" ]; then
            _dsa_had_u=0; case "$-" in *u*) _dsa_had_u=1; set +u ;; esac
            source "$_dsa_mf_setenv"
            [ "$_dsa_had_u" = 1 ] && set -u
            echo "[env.sh] sourced memfabric set_env.sh（MEMFABRIC_HYBRID_EXTEND_LIB_PATH=${MEMFABRIC_HYBRID_EXTEND_LIB_PATH:-该脚本没设这个变量}）"
        else
            echo "[env.sh] ⚠ 没找到可用的 memfabric：import memfabric_hybrid 失败，也没有 set_env.sh"
            echo "[env.sh]   offload 后端会报 'offload launch load library failed'。"
            echo "[env.sh]   修法：pip install memfabric_hybrid==1.3.0（镜像里通常已有）"
        fi
        ;;
esac

# 5) PYTHONPATH 仓库前置（yq 原版第 143 行，去容器化时同样误删）。
#    按解析后的真实路径去重：同仓库多路径/符号链接只留第一次出现的写法，
#    重复 source 不堆叠。
_dsa_repo_real="$(cd "$_dsa_repo" 2>/dev/null && pwd -P)"
case ":${PYTHONPATH:-}:" in
    *":$_dsa_repo_real:"*) ;;
    *) export PYTHONPATH="$_dsa_repo_real${PYTHONPATH:+:$PYTHONPATH}"
       echo "[env.sh] PYTHONPATH 前置仓库 = $_dsa_repo_real" ;;
esac

# 6) memlock 上限。MemFabric 的 offload 池是 pinned 宿主机内存（给 NPU DMA），
#    受 ulimit -l 限制。默认 64 MiB 时 6 GiB 池必挂：
#    "HalMemSetAccess failed: 65534 / alloc local host mem failed"（2026-10-21 实测）。
#    这里尽力抬到 unlimited；抬不动就给明确提示（root 可 ulimit -l unlimited，
#    或用 DSA_DRAM_POOL_GB 调小池——精度用例 3 GiB 就够）。
if [ "$(ulimit -l)" != "unlimited" ]; then
    _dsa_lk="$(ulimit -l)"
    ulimit -l unlimited 2>/dev/null || true
    if [ "$(ulimit -l)" = "unlimited" ]; then
        echo "[env.sh] ulimit -l 由 ${_dsa_lk} KB 提升为 unlimited（MemFabric 池需要）"
    else
        echo "[env.sh] ⚠ ulimit -l 仍是 $(ulimit -l) KB —— 6 GiB 的 MemFabric 池会申请失败"
        echo "[env.sh]   处理：ulimit -l unlimited（root）或改 /etc/security/limits.conf；"
        echo "[env.sh]   或调小池：DSA_DRAM_POOL_GB=3 pytest ...（精度用例 1.2 GiB 峰值）"
    fi
else
    echo "[env.sh] ulimit -l = unlimited"
fi
