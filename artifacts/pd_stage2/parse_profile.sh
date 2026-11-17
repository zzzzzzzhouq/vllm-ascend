#!/usr/bin/env bash
# 解析 Ascend PyTorch Profiler 输出（*_ascend_pt）。
# 用法:
#   ./parse_profile.sh <profile_root>            # 解析全部 rank
#   ./parse_profile.sh <profile_root> 0          # 只解析排序后的第 0 个（建议两臂取同一个）
set -euo pipefail
root="${1:?usage: parse_profile.sh <root> [rank_index]}"
idx="${2:-}"

mapfile -t dirs < <(find "$root" -maxdepth 4 -type d -name "*_ascend_pt" | sort)
if [ "${#dirs[@]}" -eq 0 ]; then
  echo "no *_ascend_pt under $root" >&2
  exit 1
fi
if [ -n "$idx" ]; then
  dirs=("${dirs[$idx]}")
fi

for d in "${dirs[@]}"; do
  echo "== analyse $d"
  python3 - "$d" <<'PY'
import sys
from torch_npu.profiler.profiler import analyse
analyse(sys.argv[1].rstrip("/") + "/")
PY
  echo "-- output --"
  ls "$d/ASCEND_PROFILER_OUTPUT" 2>/dev/null | head || true
done

echo
echo "看这些文件（每臂取同一个 rank）:"
echo "  ASCEND_PROFILER_OUTPUT/kernel_details.csv      # kernel 耗时"
echo "  ASCEND_PROFILER_OUTPUT/op_statistic.csv        # 算子占比"
echo "  ASCEND_PROFILER_OUTPUT/trace_view.json         # MindStudio Insight 打开看时间轴/空隙"
