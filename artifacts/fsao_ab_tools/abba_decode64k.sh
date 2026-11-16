#!/usr/bin/env bash
# 干净 decode 场景的 ABBA：默认 base opt opt base base opt
# 与 abba.sh 的区别：加载 env_decode64k.sh（默认 64K/bs4/out64、单波次、无前缀共享）。
#
# 用法：
#   ./abba_decode64k.sh
#   ./abba_decode64k.sh base opt
#   DECODE_BS=8 DECODE_OUT_LEN=256 ./abba_decode64k.sh
#   DECODE_BS=8 DECODE_OUT_LEN=256 DECODE_REPEAT_RATE=0.99 DECODE_PREFIX_TEST=1 \
#     ./abba_decode64k.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
# repeat_rate / prefix_test 由预设按 DECODE_REPEAT_RATE / DECODE_PREFIX_TEST 设置，
# 这里不要再覆盖 PREFIX_TEST。
source "$HERE/env_decode64k.sh"

exec "$HERE/abba.sh" "$@"
