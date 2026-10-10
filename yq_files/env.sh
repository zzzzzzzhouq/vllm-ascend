#!/usr/bin/env bash
# 一次 source 就绪：CANN toolkit + 目标芯片 + 本仓库自定义算子 vendor 包。
# （原仓库 env.sh 写死了容器路径、还带 memfabric/atb 段，与本仓库无关，未照搬。）
#
# 用法：
#   source yq_files/env.sh
#   DSA_REPO=/path/to/repo source yq_files/env.sh   # 仓库不是脚本上一级时
#
# 幂等：重复 source 不会把路径变量越堆越长。
#
# 三段缺一不可：
#   - ascend-toolkit     : CANN 编译/运行时
#   - SOC_VERSION        : 本仓库 setup.py **不自动探测**，缺了 build_aclnn
#                          会静默跳过整个算子编译（只编 .so，看不出异常）
#   - ASCEND_CUSTOM_OPP_PATH : vendor 包路径。vllm_ascend import 时也会自动补
#                          （utils.bootstrap_custom_op_env），显式导出是为了
#                          脱离 vllm_ascend 的裸 torch 脚本也能找到算子。

_dsa_repo="${DSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
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

# 2) 目标芯片。A3 = ascend910_93。
case "${SOC_VERSION:-}" in
    ascend*) echo "[env.sh] SOC_VERSION=$SOC_VERSION（沿用外部导出）" ;;
    *)       export SOC_VERSION=ascend910_93
             echo "[env.sh] SOC_VERSION 未导出，默认设为 ascend910_93（A3）" ;;
esac

# 3) 自定义算子 vendor 包（build_ops.sh 的产物就装在仓库里）。
_dsa_vendor="$_dsa_repo/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
if [ -d "$_dsa_vendor" ]; then
    case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
        *":$_dsa_vendor:"*) ;;
        *) export ASCEND_CUSTOM_OPP_PATH="$_dsa_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
           echo "[env.sh] ASCEND_CUSTOM_OPP_PATH += $_dsa_vendor" ;;
    esac
else
    echo "[env.sh] ⚠ $_dsa_vendor 不存在 —— 算子还没编？先跑 bash yq_files/build_ops.sh"
fi
