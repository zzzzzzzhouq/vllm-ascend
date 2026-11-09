# FSAO A3 整网 A/B 完整执行手册（PR #20927）

> 目标：在 Atlas A3 单机（8 卡 × 2 chip = 16 逻辑 NPU）上，干净地对比
> **PR #20927 的 `fused_sparse_attention_overlap` 优化** 与 **它的真实基线**。
>
> 固定版本：
>
> - 基线（base）：`f138fdfb` —— PR 的 merge-base
> - 优化（opt）：`653d08d` —— PR head（`happy-math:temp2`）
> - 模型：`GLM-5.1-w4a8`
> - 场景：32K / cc8 / `repeat_rate 0.5`
> - 图模式：`FULL_DECODE_ONLY`

---

## 0. 先读这一段

### 0.1 为什么不能用当前 `main` 当基线

PR #20927 的 merge-base 是 `f138fdfb`。当前 `origin/main` 已经比它领先很多提交
（实测 2026-11-09 时领先 88 个提交、300 个文件）。直接拿 `main` 对比 PR，测到的差异是
**main 的新提交 + PR 的 3 个文件**，不能归因到 FSAO。

本手册的做法：

```text
base = f138fdfb
opt  = 653d08d
```

两者净差异应该只有：

```text
csrc/attention/fused_sparse_attention_overlap/op_host/fused_sparse_attention_overlap_tiling.cpp
csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_kernel_mla.h
csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_service_vector_mla.h
release_notes/fragments/20927.yaml
```

### 0.2 铁律

1. **base 和 opt 都必须重新编译完整算子包**，不能只换 `.so`，不能复用旧包。
2. `vllm_ascend/_cann_ops_custom` 是 gitignore 的，**`git checkout` 不会换它**。
3. 每轮必须 **停服务 → 换包 → 新进程启动**；算子只在进程启动时加载。
4. base/opt 两端必须用 **同一份 serve 参数、同一份压测命令**。
5. 不要一边 `FULL_DECODE_ONLY`、一边 `--enforce-eager`。
6. 轮次必须 **ABBA 交错**，不能先连测 3 遍 base 再连测 3 遍 opt。
7. 先确认两套包的 fsao 产物 **md5 不同**，再谈性能。
8. 不要拿 GLM-5.2 的历史数字和 GLM-5.1 的新数字直接比。

---

## 1. 机器检查

```bash
# 逻辑 NPU = 8 物理卡 × 2 chip = 16
npu-smi info -l
python3 -c "import torch, torch_npu; print('device_count:', torch.npu.device_count())"

# host 内存（sparse offload host pool 用）
free -g

# 磁盘（两套完整算子构建，建议预留 >= 100GB）
df -h /home/z00943842

# 权重：必须有 index_topk，不能有 compress_ratios，不能是 c8
MODEL_PATH=/实际路径/GLM-5.1-w4a8
du -sh "$MODEL_PATH"
grep -oE '"(index_topk|compress_ratios|index_kpool)"' "$MODEL_PATH/config.json" | sort -u
grep -o '"kv_cache_type"[^,}]*' "$MODEL_PATH/quant_model_description.json" 2>/dev/null

# vLLM / vllm-ascend 环境
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
vllm serve --help > /dev/null && echo "vllm CLI OK"
```

期望：

| 检查项 | 期望 |
|---|---|
| 逻辑 NPU | 16 |
| 权重 | 有 `index_topk`，无 `compress_ratios` |
| KV | 非 C8 |
| `vllm_ascend.__file__` | 指向 `/home/z00943842/base/vllm-ascend/vllm_ascend/` |

---

## 2. 代码与提交

```bash
cd /home/z00943842/base/vllm-ascend

git fetch origin
git fetch https://github.com/happy-math/vllm-ascend.git temp2:pr20927

git rev-parse origin/main pr20927
base=$(git merge-base origin/main pr20927)
echo "base=$base"
git rev-list --count "$base"..origin/main
git diff --stat "$base"..pr20927
```

期望（2026-11-09 实测）：

```text
origin/main = 3725f6b3...   # 可能继续前进，不重要
pr20927     = 653d08d1...   # PR head
base        = f138fdfb36... # PR 的 merge-base
git rev-list --count base..origin/main  -> 88（或更大）
git diff --stat base..pr20927           -> 只有 3 个 csrc + release note
```

如果 `pr20927` 的 head 不是 `653d08d`，说明 PR 被更新过。请以实际 head 为准，但
`base` 仍然取 `git merge-base origin/main pr20927`；本手册后续把 `653d08d` 替换成实际 head。

> 注意：`git diff origin/main...pr20927` 只显示 PR 改了什么，**不代表两个版本运行起来只差
> 这些文件**。运行差异 = `merge-base..origin/main` + PR 的改动。所以基线必须用 merge-base。

---

## 3. 编译 base 包（f138fdfb）

```bash
cd /home/z00943842/base/vllm-ascend

git status -sb
git checkout f138fdfb
git log --oneline -1
git status --porcelain -- csrc/attention/fused_sparse_attention_overlap

export SOC_VERSION=ascend910_9382
export MAX_JOBS=8
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/devlib:${LD_LIBRARY_PATH:-}"

export AB=/home/z00943842/fsao_ab
mkdir -p "$AB/ops" "$AB/results"

# 首次全量构建约 30 分钟以上，建议 tmux / nohup
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION" 2>&1 | tee "$AB/build_base.log"

cp -a vllm_ascend/_cann_ops_custom "$AB/ops/base"
```

如果 `catlass` 子模块缺失，`build_aclnn.sh` 会自动 fetch；失败时手动拷：

```bash
cp -a /home/z00943842/base/vllm-ascend/csrc/third_party/catlass csrc/third_party/
```

验收：

```bash
ls "$AB/ops/base/vendors/custom_transformer/"
find "$AB/ops/base" -type f | wc -l
```

---

## 4. 编译 opt 包（653d08d）

```bash
cd /home/z00943842/base/vllm-ascend

git checkout 653d08d
git log --oneline -1

bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION" 2>&1 | tee "$AB/build_opt.log"

cp -a vllm_ascend/_cann_ops_custom "$AB/ops/opt"
```

编译完成后 **让仓库停在 `653d08d`**，后续 Python 运行时都用这棵树。
Python 代码在 `f138fdfb` 和 `653d08d` 之间没有净差异，不需要重新 `pip install -e .`。

---

## 5. 创建测试工具

```bash
mkdir -p /home/z00943842/fsao_ab/tools
cd /home/z00943842/fsao_ab/tools
```

### 5.1 `env.sh`

```bash
cat > env.sh <<'EOF'
#!/usr/bin/env bash
# FSAO A/B 环境变量 —— 按机器实际路径修改后再用。
# 其他脚本会自动 source 本文件。

export REPO=${REPO:-/home/z00943842/base/vllm-ascend}
export AB=${AB:-/home/z00943842/fsao_ab}
export OPS_BASE=$AB/ops/base
export OPS_OPT=$AB/ops/opt
export RESULTS=$AB/results

# ---- 需要你填的路径 ----
export MODEL_PATH=${MODEL_PATH:-/path/to/GLM-5.1-w4a8}
export SERVED_NAME=${SERVED_NAME:-glm-5.1}
export AISV_DIR=${AISV_DIR:-$HOME/aisvenv}
export AISBENCH_TOOL=${AISBENCH_TOOL:-$HOME/aisbench_auto_tools_prefix}
export DATASET_DIR=${DATASET_DIR:-$HOME/dataset}

export HOST_IP=${HOST_IP:-127.0.0.1}
export PORT=${PORT:-8000}

# ---- 压测参数：base/opt 两端必须完全一致 ----
export INPUT_LEN=${INPUT_LEN:-32768}
export OUTPUT_LEN=${OUTPUT_LEN:-1024}
export DATA_NUM=${DATA_NUM:-32}
export CONCURRENCY=${CONCURRENCY:-8}
export NPU_NUM=${NPU_NUM:-16}
export REPEAT_RATE=${REPEAT_RATE:-0.5}
export SEED=${SEED:-2}

# ---- 服务参数 ----
export TP=${TP:-16}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
export DRAM_GB=${DRAM_GB:-48}

export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export SOC_VERSION=${SOC_VERSION:-ascend910_9382}
export MAX_JOBS=${MAX_JOBS:-8}

# ---- HCCL：默认取第一块非回环网卡，可显式覆盖 ----
if [ -z "${HCCL_IF_IP:-}" ]; then
  HCCL_IF_IP=$(ip -o -4 addr show 2>/dev/null | awk '$4 !~ /^127/ {print $4; exit}' | cut -d/ -f1)
fi
if [ -z "${HCCL_SOCKET_IFNAME:-}" ]; then
  HCCL_SOCKET_IFNAME=$(ip -o -4 addr show 2>/dev/null | awk '$4 !~ /^127/ {print $2; exit}')
fi
export HCCL_IF_IP
export HCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}

mkdir -p "$RESULTS"
EOF
```

### 5.2 `serve.sh`

```bash
cat > serve.sh <<'EOF'
#!/usr/bin/env bash
# 启动整网服务（GLM-5.1 + sparse_kv_offload + use_fused_overlap）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

PKG="$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
[ -d "$PKG" ] || { echo "missing custom op package: $PKG" >&2; exit 1; }

if [ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]; then
  source "$ASCEND_HOME_PATH/bin/setenv.bash"
fi

# 显式指向当前主树里的包，避免 set_env.bash 里写死的旧绝对路径
export ASCEND_CUSTOM_OPP_PATH="$PKG${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$PKG/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 15)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
unset HCCL_INTRA_PCIE_ENABLE HCCL_INTRA_ROCE_ENABLE || true
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
unset http_proxy https_proxy || true

ADDITIONAL=$(cat <<JSON
{
  "enable_balance_scheduling": true,
  "multistream_overlap_shared_expert": true,
  "sparse_kv_offload_config": {
    "enabled": true,
    "use_fused_overlap": true,
    "keep_device_kv_cache": true,
    "dram_size_per_dp_GB": ${DRAM_GB}
  }
}
JSON
)

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --data-parallel-size 1 \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config "$ADDITIONAL"
EOF
```

### 5.3 `bench.sh`

```bash
cat > bench.sh <<'EOF'
#!/usr/bin/env bash
# 跑一轮 aisbench。两端必须用同一份本脚本 / 同一组环境变量。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

source "$AISV_DIR/bin/activate"
cd "$AISBENCH_TOOL"
unset http_proxy https_proxy || true

# CSV 是追加写：每轮开始前清掉，确保本轮结果独立
rm -f aisbench_result.csv aisbench.log

python3 aisbench_test.py \
  --input_len "$INPUT_LEN" \
  --output_len "$OUTPUT_LEN" \
  --data_num "$DATA_NUM" \
  --concurrency "$CONCURRENCY" \
  --request_rate 0 \
  --npu_num "$NPU_NUM" \
  --dataset_type prefix_cache \
  --repeat_rate "$REPEAT_RATE" \
  --prefix_test \
  --seed "$SEED"
EOF
```

### 5.4 `swap_ops.sh`

```bash
cat > swap_ops.sh <<'EOF'
#!/usr/bin/env bash
# 切换完整算子包。用法: ./swap_ops.sh base|opt
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

build="${1:?usage: swap_ops.sh base|opt}"
src="$AB/ops/$build"
[ -d "$src" ] || { echo "missing package: $src" >&2; exit 1; }

pkill -9 -f "vllm serve" || true
pkill -9 -f EngineCore || true
sleep 10

rm -rf "$REPO/vllm_ascend/_cann_ops_custom"
cp -a "$src" "$REPO/vllm_ascend/_cann_ops_custom"

diff -rq "$REPO/vllm_ascend/_cann_ops_custom" "$src" >/dev/null \
  && echo "ACTIVE=$build"

fp="$RESULTS/pkg_${build}_$(date +%H%M%S).md5"
mkdir -p "$RESULTS"
find "$REPO/vllm_ascend/_cann_ops_custom" -type f \
  -path '*fused_sparse_attention_overlap*' -print0 \
  | sort -z | xargs -0 md5sum > "$fp"
echo "fingerprint: $fp"
EOF
```

### 5.5 `round.sh`

```bash
cat > round.sh <<'EOF'
#!/usr/bin/env bash
# 一轮完整测试: 换包 -> 启服务 -> 等 health -> 压测 -> 保存 -> 停服务。
# 用法: ./round.sh base|opt
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

build="${1:?usage: round.sh base|opt}"
case "$build" in base|opt) ;; *) echo "unknown build: $build" >&2; exit 1;; esac

n=1
while [ -e "$RESULTS/${build}_r${n}.csv" ] || [ -e "$RESULTS/${build}_r${n}_meta.txt" ]; do
  n=$((n + 1))
done
tag="${build}_r${n}"
mkdir -p "$RESULTS"

echo "=== round $tag ==="

"$HERE/swap_ops.sh" "$build"

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

curl -fsS "http://${HOST_IP}:${PORT}/metrics" 2>/dev/null | grep -q prefix_cache \
  || echo "WARNING: /metrics 里没有 prefix_cache，确认代理已关、PROMETHEUS_MULTIPROC_DIR 已设" >&2

"$HERE/bench.sh" 2>&1 | tee "$RESULTS/${tag}_bench.console"

[ -f "$AISBENCH_TOOL/aisbench_result.csv" ] && cp "$AISBENCH_TOOL/aisbench_result.csv" "$RESULTS/${tag}.csv"
[ -f "$AISBENCH_TOOL/aisbench.log" ] && cp "$AISBENCH_TOOL/aisbench.log" "$RESULTS/${tag}.log"

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
EOF
```

### 5.6 `abba.sh`

```bash
cat > abba.sh <<'EOF'
#!/usr/bin/env bash
# 默认 ABBA: base opt opt base base opt
# 用法:
#   ./abba.sh
#   ./abba.sh base opt
#   ./abba.sh base base
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

if [ -x "$AISV_DIR/bin/python" ]; then
  "$AISV_DIR/bin/python" "$HERE/summary.py"
else
  python3 "$HERE/summary.py"
fi
EOF
```

### 5.7 `summary.py`

```bash
cat > summary.py <<'EOF'
#!/usr/bin/env python3
"""汇总 base_r*.csv 与 opt_r*.csv：每轮值、中位数、极差、opt vs base 变化。"""
import glob
import os

import pandas as pd

AB = os.environ.get("AB", os.path.expanduser("~/fsao_ab"))
RESULTS = os.path.join(AB, "results")

METRICS = [
    "TPOT avg",
    "TPOT SLO_P90",
    "output_throughput",
    "single_output_throughput",
    "E2E_time",
    "repeat_rate",
]


def load(build):
    files = sorted(glob.glob(os.path.join(RESULTS, f"{build}_r*.csv")))
    if not files:
        return None, []
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        df["__round__"] = os.path.basename(path).replace(".csv", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True), files


def main():
    data = {}
    for build in ("base", "opt"):
        df, files = load(build)
        if df is None:
            print(f"[warn] no CSV for {build}")
        else:
            data[build] = df

    if not data:
        raise SystemExit("no results")

    rows = []
    for build, df in data.items():
        extra = [c for c in df.columns if "hit" in c.lower() or "命中" in c]
        cols = [c for c in METRICS + extra if c in df.columns]
        for _, row in df.iterrows():
            rec = {"build": build, "round": row["__round__"]}
            for c in cols:
                rec[c] = row[c]
            rows.append(rec)

    out = pd.DataFrame(rows)
    print("=== per-round ===")
    print(out.to_string(index=False))

    print("\n=== median / spread ===")
    summary = []
    for build in ("base", "opt"):
        sub = out[out["build"] == build]
        if sub.empty:
            continue
        rec = {"build": build, "rounds": len(sub)}
        for c in METRICS:
            if c not in sub.columns:
                continue
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            if vals.empty:
                continue
            med = vals.median()
            rec[f"{c} med"] = round(med, 3)
            rec[f"{c} min"] = round(vals.min(), 3)
            rec[f"{c} max"] = round(vals.max(), 3)
            rec[f"{c} spread%"] = round((vals.max() - vals.min()) / med * 100, 2) if med else None
        summary.append(rec)
    print(pd.DataFrame(summary).to_string(index=False))

    if "base" in data and "opt" in data:
        print("\n=== opt vs base (median) ===")
        for c in METRICS:
            if c not in out.columns:
                continue
            b = pd.to_numeric(out.loc[out["build"] == "base", c], errors="coerce").dropna()
            o = pd.to_numeric(out.loc[out["build"] == "opt", c], errors="coerce").dropna()
            if b.empty or o.empty or b.median() == 0:
                continue
            bm, om = b.median(), o.median()
            delta = (om - bm) / bm * 100
            noise = max(
                (b.max() - b.min()) / bm * 100,
                (o.max() - o.min()) / om * 100,
            )
            flag = "   <-- within noise" if abs(delta) <= noise else ""
            print(
                f"{c:24s} base={bm:10.3f} opt={om:10.3f} "
                f"delta={delta:+7.2f}% noise~{noise:5.2f}%{flag}"
            )

    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "rounds.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    print(f"\nsaved: {RESULTS}/rounds.csv, {RESULTS}/summary.csv")


if __name__ == "__main__":
    main()
EOF
```

### 5.8 `check_pkgs.sh`

```bash
cat > check_pkgs.sh <<'EOF'
#!/usr/bin/env bash
# 测前硬校验：两套包差异范围、fsao 产物 md5、build cache HIT/MISS。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

for p in "$OPS_BASE" "$OPS_OPT"; do
  [ -d "$p" ] || { echo "missing $p" >&2; exit 1; }
done

echo "== file counts =="
find "$OPS_BASE" -type f | wc -l
find "$OPS_OPT" -type f | wc -l

echo
echo "== diff base vs opt =="
diff -rq "$OPS_BASE" "$OPS_OPT" | tee "$AB/pkgs.diff" || true
echo "diff entries: $(wc -l < "$AB/pkgs.diff")"

echo
echo "== fused_sparse_attention_overlap fingerprints (must differ) =="
for p in "$OPS_BASE" "$OPS_OPT"; do
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
EOF
```

### 5.9 赋可执行权限

```bash
chmod +x env.sh serve.sh bench.sh swap_ops.sh round.sh abba.sh check_pkgs.sh
chmod +x summary.py
```

---

## 6. 修改 `env.sh`

```bash
cd /home/z00943842/fsao_ab/tools
vi env.sh
```

必须改：

```bash
export REPO=/home/z00943842/base/vllm-ascend
export AB=/home/z00943842/fsao_ab
export MODEL_PATH=/实际路径/GLM-5.1-w4a8
export AISV_DIR=/实际路径/aisvenv
export AISBENCH_TOOL=/实际路径/aisbench_auto_tools_prefix
export DATASET_DIR=/实际路径/dataset
```

同时确认 `$AISBENCH_TOOL/config.py` 里：

```python
MODEL_NAME = "glm-5.1"      # 与 SERVED_NAME 一致
HOST_PORT  = "8000"         # 与 PORT 一致
MODEL_PATH = "/实际路径/GLM-5.1-w4a8"
```

自检：

```bash
source env.sh
echo "$MODEL_PATH"
echo "$AISV_DIR $AISBENCH_TOOL"
ls "$AISV_DIR/bin/activate"
ls "$AISBENCH_TOOL/aisbench_test.py"
ls "$AB/ops/base/vendors/custom_transformer/"
ls "$AB/ops/opt/vendors/custom_transformer/"
```

---

## 7. 测前校验

```bash
cd /home/z00943842/fsao_ab/tools
source env.sh
./check_pkgs.sh
```

必须满足：

1. `diff entries` 只有少量 fsao 相关产物（个位数到几十）；
2. `$AB/ops/base` 和 `$AB/ops/opt` 的 `fused_sparse_attention_overlap` 产物 **md5 不同**；
3. `$AB/build_opt.log` 里该算子的 unit 不是 `[build-cache] HIT`（否则 kernel 没重编）。

再各跑一次单算子测试（两个包都要 `4 passed`）：

```bash
cd "$REPO"
# 先切到 base 包
"$AB/tools/swap_ops.sh" base
source "$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
python3 - <<'PY'
import torch, vllm_ascend
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
print("算子加载成功")
PY
VLLM_USE_MODELSCOPE=true pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py

# 再切到 opt 包，重复一次
"$AB/tools/swap_ops.sh" opt
source "$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
VLLM_USE_MODELSCOPE=true pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
```

> 这一步会启动/停止算子相关进程，不影响后续 A/B；做完后保持哪个包都可以，
> `round.sh` 会重新切。

---

## 8. 跑 A/B

### 8.1 默认 3+3 ABBA

```bash
cd /home/z00943842/fsao_ab/tools
source env.sh
./abba.sh
```

实际顺序：

```text
base_r1 -> opt_r1 -> opt_r2 -> base_r2 -> base_r3 -> opt_r3
```

每轮自动：

1. kill 旧服务；
2. 切换算子包并记录 md5；
3. 启动服务；
4. 等 `/health`（首次加载 391GB 权重可能 10~30 分钟）；
5. 检查 `/metrics` 有 `prefix_cache`；
6. 跑 aisbench；
7. 保存 CSV / log / 元数据；
8. 停服务；
9. 等 `ROUND_GAP` 秒（默认 120）。

### 8.2 只做同包噪声对照

```bash
./abba.sh base base
```

如果同包两轮的 TPOT 极差就很大，先别继续，查：

- 命中率是否一致；
- 是否有其他进程占用 NPU；
- 命令/参数是否一致；
- 机器温度/频率是否稳定。

### 8.3 自定义场景

覆盖环境变量即可，**两端仍然必须一致**：

```bash
# 50% 命中，覆盖 Opt05
REPEAT_RATE=0.5 ./abba.sh

# P0 场景（bs=32）：需要同时把 max-num-seqs 调到 32
REPEAT_RATE=0.5 CONCURRENCY=32 MAX_NUM_SEQS=32 DATA_NUM=64 \
  INPUT_LEN=16384 ./abba.sh
```

> 注意：`CONCURRENCY=32` 但 `MAX_NUM_SEQS=8` 时 decode bs 仍然是 8，测不到 P0。

---

## 9. 结果判读

每轮产物：

```text
$AB/results/
├── base_r1.csv / base_r1.log / base_r1_server.log / base_r1_meta.txt
├── opt_r1.csv  / ...
├── pkg_base_HHMMSS.md5
├── pkg_opt_HHMMSS.md5
├── rounds.csv
└── summary.csv
```

`abba.sh` 结束会自动打印汇总，也可以手动：

```bash
$AISV_DIR/bin/python summary.py
```

关注：

| 指标 | 期望方向 |
|---|---|
| `TPOT avg` / `TPOT SLO_P90` | opt 下降 |
| `output_throughput` | opt 上升 |
| `E2E_time` | opt 下降 |
| `TTFT avg` | 基本不变 |
| 实际命中率 | 两端接近 |

判读规则：

- **差值小于批间极差 → `within noise`，不能下结论**；补轮次或换更聚焦的场景；
- 两端命中率不一致 → 先查 KV 容量/换出，不要归因 kernel；
- 单轮噪声 1~2%，整网预期收益个位数百分比，3 轮是最低要求；
- 如果 opt 稳定更慢，再回到单算子 hit-ratio sweep，先确认 kernel 本身是否更快。

---

## 10. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `check_pkgs.sh` 两套 md5 相同 | kernel 没重编 / 源码没切换 | 看 `build_opt.log` 的 `[build-cache] HIT/MISS`；确认 `git checkout` 后 3 个文件内容变了 |
| `diff entries` 成百上千 | 两次构建的算子集/环境不一致 | 回到第 3、4 节重建 |
| 服务起不来，`loading safetensors ... 0%` 很久 | 391GB 从 NFS 首次加载 | 正常，等；`HEALTH_TIMEOUT` 默认 3600s |
| `Sparse KV offload offload.initialize failed` | host pool 不够 | 降 `DRAM_GB`（`DRAM_GB=32 ./abba.sh`） |
| `HCCL_BUFFSIZE_EP is too SMALL` | MoE AllToAll 缓冲不足 | `serve.sh` 里已是 512；确认生效 |
| `HCCL_INTRA_... invalid` | 只设了一个 intra 变量 | 脚本已 unset 两个 |
| `does not support packed SFA main caches` | c8 权重或 `--kv-cache-dtype` | 换非 c8 权重，不要加 `--kv-cache-dtype` |
| `/metrics` 没有 `prefix_cache` | 代理 / 指标目录 | 脚本已 unset proxy、设 `PROMETHEUS_MULTIPROC_DIR` |
| `unrecognized arguments: --num-warmups` | aisbench 版本差异 | 按工具 README FAQ 3 删掉该参数 |
| 下一轮启动时显存没释放 | 旧进程残留 | 增大 `swap_ops.sh` 的 `sleep`；`npu-smi info` 确认 |
| `bad interpreter: ...^M` | 脚本被写成 CRLF | `sed -i 's/\r$//' *.sh summary.py` |

---

## 11. 清理

```bash
# 停服务
pkill -9 -f "vllm serve" || true
pkill -9 -f EngineCore || true

# 删掉临时包（保留 results 再删）
cd /home/z00943842/fsao_ab
ls results/
# rm -rf ops
```

---

## 附：一页纸命令清单

```bash
# 1. 代码
cd /home/z00943842/base/vllm-ascend
git fetch origin
git fetch https://github.com/happy-math/vllm-ascend.git temp2:pr20927
base=$(git merge-base origin/main pr20927)
echo "$base"          # 期望 f138fdfb36...
git rev-parse pr20927 # 期望 653d08d...

# 2. base 包
git checkout "$base"
export SOC_VERSION=ascend910_9382 MAX_JOBS=8
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION" 2>&1 | tee /home/z00943842/fsao_ab/build_base.log
cp -a vllm_ascend/_cann_ops_custom /home/z00943842/fsao_ab/ops/base

# 3. opt 包
git checkout pr20927
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION" 2>&1 | tee /home/z00943842/fsao_ab/build_opt.log
cp -a vllm_ascend/_cann_ops_custom /home/z00943842/fsao_ab/ops/opt

# 4. 工具（按第 5 节创建后）
cd /home/z00943842/fsao_ab/tools
chmod +x *.sh summary.py
vi env.sh
source env.sh
./check_pkgs.sh

# 5. 跑
./abba.sh
```
