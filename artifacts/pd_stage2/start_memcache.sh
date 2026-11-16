#!/usr/bin/env bash
# 仅 USE_LAYERWISE_PREFILL=1（MultiConnector + AscendStore）时需要。
# 先确认: python3 -c "import memcache_hybrid"
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

export MMC_META_CONFIG_PATH="$MMC_DIR/mmc-meta.conf"

nohup python3 -c "from memcache_hybrid import MetaService; MetaService.main()" \
  > "$LOG_DIR/memcache-meta.log" 2>&1 &
echo $! > "$LOG_DIR/memcache-meta.pid"
echo "memcache MetaService started (pid $(cat "$LOG_DIR/memcache-meta.pid")), log: $LOG_DIR/memcache-meta.log"
echo "注意: 5000(meta) / 6000(config store) 必须有进程监听；若 MetaService 不带 config store，需要单独启它。"
