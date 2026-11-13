#!/usr/bin/env bash
# 切换完整算子包。
# 用法: ./swap_ops.sh base|opt
#
# 兼容两种保存布局：
#   $AB/ops/base/vendors/custom_transformer/...                  （推荐）
#   $AB/ops/base/_cann_ops_custom/vendors/custom_transformer/... （cp 嵌套）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

resolve_pkg_root() {
  local d="$1"
  if [ -d "$d/vendors/custom_transformer" ]; then
    printf '%s\n' "$d"
  elif [ -d "$d/_cann_ops_custom/vendors/custom_transformer" ]; then
    printf '%s\n' "$d/_cann_ops_custom"
  else
    return 1
  fi
}

build="${1:?usage: swap_ops.sh base|opt}"
src="$AB/ops/$build"
[ -d "$src" ] || { echo "missing package: $src" >&2; exit 1; }

if ! src_root=$(resolve_pkg_root "$src"); then
  echo "ERROR: $src 下找不到 vendors/custom_transformer" >&2
  echo "       也不是 _cann_ops_custom/vendors/custom_transformer" >&2
  exit 1
fi
if [ "$src_root" != "$src" ]; then
  echo "WARN: $src 使用嵌套布局，实际包根为 $src_root"
fi

# 先停旧服务；op 只在进程启动时加载
pkill -9 -f "vllm serve" || true
pkill -9 -f EngineCore || true
sleep 10

rm -rf "$REPO/vllm_ascend/_cann_ops_custom"
cp -a "$src_root" "$REPO/vllm_ascend/_cann_ops_custom"

# 安装目录必须与解析出的包根完全一致
diff -rq "$REPO/vllm_ascend/_cann_ops_custom" "$src_root" >/dev/null \
  && echo "ACTIVE=$build"

# 记录本包指纹，事后可追溯
fp="$RESULTS/pkg_${build}_$(date +%H%M%S).md5"
mkdir -p "$RESULTS"
find "$REPO/vllm_ascend/_cann_ops_custom" -type f \
  -path '*fused_sparse_attention_overlap*' -print0 \
  | sort -z | xargs -0 md5sum > "$fp"
echo "fingerprint: $fp"
