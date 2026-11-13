#!/usr/bin/env bash
# 测前硬校验：两套包差异范围、fsao 产物 md5、build cache HIT/MISS。
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

for p in "$OPS_BASE" "$OPS_OPT"; do
  [ -d "$p" ] || { echo "missing $p" >&2; exit 1; }
done

if ! ROOT_BASE=$(resolve_pkg_root "$OPS_BASE"); then
  echo "ERROR: $OPS_BASE 不是算子包根目录" >&2
  exit 1
fi
if ! ROOT_OPT=$(resolve_pkg_root "$OPS_OPT"); then
  echo "ERROR: $OPS_OPT 不是算子包根目录" >&2
  exit 1
fi

if [ "$ROOT_BASE" != "$OPS_BASE" ]; then
  echo "note: base 使用嵌套布局 $ROOT_BASE"
fi
if [ "$ROOT_OPT" != "$OPS_OPT" ]; then
  echo "note: opt  使用嵌套布局 $ROOT_OPT"
fi
echo "base root: $ROOT_BASE"
echo "opt root:  $ROOT_OPT"

echo
echo "== file counts =="
find "$ROOT_BASE" -type f | wc -l
find "$ROOT_OPT" -type f | wc -l

echo
echo "== diff base vs opt =="
diff -rq "$ROOT_BASE" "$ROOT_OPT" | tee "$AB/pkgs.diff" || true
echo "diff entries: $(wc -l < "$AB/pkgs.diff")"

echo
echo "== fused_sparse_attention_overlap fingerprints (must differ) =="
for p in "$ROOT_BASE" "$ROOT_OPT"; do
  echo "== $p"
  find "$p" -type f -path '*fused_sparse_attention_overlap*' -print0 \
    | sort -z | xargs -0 md5sum
done

echo
echo "== build cache HIT/MISS (fused_sparse_attention_overlap) =="
for f in "$AB"/build_*.log; do
  [ -f "$f" ] || continue
  echo "== $f"
  grep -E "fused_sparse_attention_overlap|\[build-cache\] (HIT|MISS)" "$f" | tail -20 || true
done
