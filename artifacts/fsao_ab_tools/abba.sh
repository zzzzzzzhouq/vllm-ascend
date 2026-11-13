#!/usr/bin/env bash
# 按给定顺序跑多轮，默认 ABBA: base opt opt base base opt
# 用法:
#   ./abba.sh                      # 默认 3+3 ABBA
#   ./abba.sh base opt             # 自定义顺序
#   ./abba.sh base base            # 只做同包噪声对照
# 轮次编号由 round.sh 自动从 1 递增，不会覆盖已有结果。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

if [ "$#" -gt 0 ]; then
  SEQ=("$@")
else
  SEQ=(base opt opt base base opt)
fi

for b in "${SEQ[@]}"; do
  case "$b" in
    base|opt) ;;
    *) echo "unknown build: $b (use base|opt)" >&2; exit 1;;
  esac
  "$HERE/round.sh" "$b"
  sleep "${ROUND_GAP:-120}"
done

# summary 需要 pandas；优先用 aisbench 的 venv
if [ -x "$AISV_DIR/bin/python" ]; then
  "$AISV_DIR/bin/python" "$HERE/summary.py"
else
  python3 "$HERE/summary.py"
fi
