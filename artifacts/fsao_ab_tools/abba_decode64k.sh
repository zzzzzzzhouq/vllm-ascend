#!/usr/bin/env bash
# 干净 decode 场景的 ABBA：默认 base opt opt base base opt
# 与 abba.sh 的唯一区别：加载 env_decode64k.sh（64K/bs4/out64、单波次、无前缀共享）。
#
# 用法：
#   ./abba_decode64k.sh
#   ./abba_decode64k.sh base opt
#   DECODE_BS=8 DECODE_OUT_LEN=256 ./abba_decode64k.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env_decode64k.sh"

# 压测脚本不要加 --prefix_test
export PREFIX_TEST=0

exec "$HERE/abba.sh" "$@"
