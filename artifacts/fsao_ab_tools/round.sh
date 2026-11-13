#!/usr/bin/env bash
# 跑一轮完整测试: 换包 -> 启服务 -> 等 health -> 压测 -> 保存 CSV/log/元数据 -> 停服务。
# 用法: ./round.sh base|opt
# 轮次编号自动递增（看 $RESULTS/<build>_rN.* 是否已存在）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

build="${1:?usage: round.sh base|opt}"
case "$build" in base|opt) ;; *) echo "unknown build: $build" >&2; exit 1;; esac

# 自动取下一个轮次编号
n=1
while [ -e "$RESULTS/${build}_r${n}.csv" ] || [ -e "$RESULTS/${build}_r${n}_meta.txt" ]; do
  n=$((n + 1))
done
tag="${build}_r${n}"
mkdir -p "$RESULTS"

echo "=== round $tag ==="

# 1) 换包（内部会先 kill 旧服务并 sleep）
"$HERE/swap_ops.sh" "$build"

# 2) 启动服务
srv_log="$RESULTS/${tag}_server.log"
rm -f "$srv_log"
nohup "$HERE/serve.sh" >"$srv_log" 2>&1 &
SRV_PID=$!

cleanup() {
  kill "$SRV_PID" 2>/dev/null || true
  pkill -9 -f "vllm serve" || true
  pkill -9 -f EngineCore || true
  sleep 5
}
trap cleanup EXIT

# 3) 等 /health（首次从 NFS 加载 391GB 权重可能 10~30 分钟）
deadline=$(( $(date +%s) + ${HEALTH_TIMEOUT:-3600} ))
while :; do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "server exited; tail of $srv_log:" >&2
    tail -80 "$srv_log" >&2 || true
    exit 1
  fi
  if curl -fsS "http://${HOST_IP}:${PORT}/health" >/dev/null 2>&1; then
    echo "health OK"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "health timeout; tail of $srv_log:" >&2
    tail -80 "$srv_log" >&2 || true
    exit 1
  fi
  sleep 10
done

# 4) 确认 sparse offload 指标存在（可选，失败只告警）
curl -fsS "http://${HOST_IP}:${PORT}/metrics" 2>/dev/null | grep -q prefix_cache \
  || echo "WARNING: /metrics 里没有 prefix_cache，确认代理已关、PROMETHEUS_MULTIPROC_DIR 已设" >&2

# 5) 压测
"$HERE/bench.sh" 2>&1 | tee "$RESULTS/${tag}_bench.console"

# 6) 保存产物
[ -f "$AISBENCH_TOOL/aisbench_result.csv" ] && cp "$AISBENCH_TOOL/aisbench_result.csv" "$RESULTS/${tag}.csv"
[ -f "$AISBENCH_TOOL/aisbench.log" ] && cp "$AISBENCH_TOOL/aisbench.log" "$RESULTS/${tag}.log"

# 7) 元数据：提交、包指纹、参数、命中率相关日志
pkg_hash=$(find "$REPO/vllm_ascend/_cann_ops_custom" -type f \
  -path '*fused_sparse_attention_overlap*' -print0 \
  | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1)

{
  echo "tag=$tag"
  echo "time=$(date -Is)"
  echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
  echo "pkg_hash=$pkg_hash"
  echo "repeat_rate=$REPEAT_RATE"
  echo "input_len=$INPUT_LEN output_len=$OUTPUT_LEN data_num=$DATA_NUM concurrency=$CONCURRENCY npu_num=$NPU_NUM seed=$SEED"
  echo "--- hit / cache lines ---"
  grep -iE "hit|命中|cache" "$RESULTS/${tag}.log" 2>/dev/null | tail -30 || true
  echo "--- CSV ---"
  cat "$RESULTS/${tag}.csv" 2>/dev/null || true
} > "$RESULTS/${tag}_meta.txt"

echo "=== done $tag ==="
