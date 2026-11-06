# FusedSparseAttentionOverlap A3 整网接入与验证手册

> 对应 PR：<https://github.com/vllm-project/vllm-ascend/pull/20927>
> head 分支：`happy-math:temp2`，base：`vllm-project:main`
> 适用环境：Atlas A3 单机（8 张物理卡 × 2 chip = **16 个逻辑 NPU**，每 chip 64GB，合计 1TB）
> 权重：`GLM-5.1-w4a8`（实测 391GB，非 c8，BF16 主 KV）
> 目标：把 PR 中优化后的 `fused_sparse_attention_overlap` 编进**完整** vllm-ascend，在整网 decode 路径上跑通、验证算子被调用，并采集性能数据

> 本文是**单套优化算子**的整网接入流程，不需要基线包、不需要 A/B。
> 如果后续需要和优化前的 kernel 做严格 A/B，见同目录的 `FSAO_E2E_AB_操作手册.md`。

---

## 硬件与软件要求

以下为已在 A3 单机验证过的环境；换硬件或换版本后需重新确认兼容性。

### 硬件要求

| 项目 | 要求 |
|---|---|
| 设备 | Atlas A3（`ascend910_9382`，SOC `ascend910_93`，arch22 路径） |
| NPU | 8 张物理卡 × 2 chip = **16 个逻辑 NPU**（`npu-smi info -l` 显示 8 个 NPU、每个 chipcount 2） |
| 单 chip HBM | 64GB；合计 **1TB** |
| 部署拓扑 | 单机，TP16 + EP16 |
| Host 内存 | ≥ 512GB（实测 2.9TB）；需容纳 391GB 权重 page cache + sparse offload host pool（`dram_size_per_dp_GB`） |
| `/dev/shm` | ≥ 1GB（建议 8~16GB；实测 500GB） |
| 磁盘 | 构建产物预留 ≥ 100GB；权重 391GB（本机在 NFS 上） |
| 网络 | HCCL 可用网卡/IP（本机 `enp162s0f0` / `10.246.63.33`） |

### 软件要求

| 项目 | 要求 |
|---|---|
| OS / Python | Linux；Python 3.12（实测 3.12.16） |
| CANN | 与 torch_npu 2.10.0.post4 匹配的 CANN toolkit + 驱动 |
| vLLM | `.github/vllm-main-verified.commit` 对应的 commit（日志版本号可能显示 `v0.30.0`，正常） |
| vllm-ascend | PR #20927 分支（`happy-math:temp2` 或本地 `dsa-a3-opt-only`） |
| torch / torch_npu | `2.10.0` / `2.10.0.post4` |
| transformers | `5.19.0` |
| triton-ascend | `3.2.2` |
| memfabric_hybrid | `1.2.0`（sparse offload 必需） |
| memcache_hybrid | `1.2.0` |
| 压测客户端 | `ais-bench-benchmark` + `pandas`（独立 venv） |
| 模型权重 | `GLM-5.1-w4a8`，391GB，非 c8，BF16 主 KV，带 `index_topk` |

> 版本以仓库的 `requirements.txt` 和 `.github/vllm-main-verified.commit` 为准。
> 本 PR 只改 arch22（A2/A3）；Ascend 950（arch35）路径未改动，不在本文验证范围。

---

## 0. 一页纸执行清单

- [ ] 1. 确认硬件：`npu-smi info -l` 显示 8 个 NPU × chipcount 2 = 16 个逻辑 NPU
- [ ] 2. 确认代码：PR #20927 的三个 csrc 文件已在当前分支上
- [ ] 3. 完整编译安装（`COMPILE_CUSTOM_KERNELS=1 pip install -e .`），生成完整算子包
- [ ] 4. 确认完整算子包里有 `fused_sparse_attention_overlap`，单算子测试 `4 passed`
- [ ] 5. 安装 aisbench 压测客户端（独立 venv）
- [ ] 6. 启动整网服务（带 `sparse_kv_offload_config` + `use_fused_overlap=true`）
- [ ] 7. 验证 `/health`、`/metrics` 和一条 decode 请求，确认算子路径生效
- [ ] 8. 跑 3 轮压测，记录 TPOT / 输出吞吐 / 命中率
- [ ] 9. （可选）跑补充场景，覆盖 P0 行划分和 Opt05

---

## 1. 原理：算子什么时候被调用

`npu_fused_sparse_attention_overlap` 只在下面这条路径上执行：

```
GLM-5.1（SFA，有 index_topk）
  → sparse_kv_offload_config.enabled=true
  → use_fused_overlap=true
  → decode 每个 step 调 npu_fused_sparse_attention_overlap
  → A3 编译产物 arch22 内核（PR 修改的 3 个文件）
```

代码位置：

- 算子调用点：`vllm_ascend/attention/sfa_kv_offload.py:1456`
- decode 分发：`vllm_ascend/attention/sfa_kv_offload.py:1594`
- 配置默认值：`vllm_ascend/ascend_config.py:1698`

**关键**：不加 `sparse_kv_offload_config` 时，服务只是普通推理，**算子一次都不会执行**，性能数据没有意义。

本机 16 个逻辑 NPU，生产形态是 PD 分离（P、D 各占一组 chip）。本文先用代码里预留的单机调试模式：

```json
"keep_device_kv_cache": true
```

此时 prefill 产生的 KV 会 D2H 写入 host pool，decode 阶段算子从 host pool 读取。绝对性能不代表生产 PD，但能真实验证算子接入和运行。

---

## 2. 变量约定

`fsao_ab` 与 `vllm-ascend` 仓库平级：

```
/home/z00943842/base/
├── vllm-ascend/
├── fsao_ab/
│   ├── env.sh
│   └── results/
├── aisvenv/
├── aisbench_auto_tools_prefix/
└── dataset/
```

```bash
WORK_ROOT=/home/z00943842/base
mkdir -p "$WORK_ROOT/fsao_ab/results"

cat > "$WORK_ROOT/fsao_ab/env.sh" <<'EOF'
export WORK_ROOT=/home/z00943842/base
export VLLM_SRC=$WORK_ROOT/vllm-ascend
export AB=$WORK_ROOT/fsao_ab
export AISV_DIR=$WORK_ROOT/aisvenv
export AISBENCH_TOOL=$WORK_ROOT/aisbench_auto_tools_prefix
export DATASET_DIR=$WORK_ROOT/dataset
export MODEL_PATH=/mnt/weight/GLM-5.1-w4a8
export SERVED_NAME=glm-5.1
export HOST_IP=127.0.0.1
export PORT=8000
EOF

source "$WORK_ROOT/fsao_ab/env.sh"
mkdir -p "$AB/results" "$DATASET_DIR"
```

每个新终端先执行：

```bash
source /home/z00943842/base/fsao_ab/env.sh
```

---

## 3. 环境盘点

```bash
source /home/z00943842/base/fsao_ab/env.sh
cd "$VLLM_SRC"

# 1) 逻辑 NPU 数（A3：8 NPU × chipcount 2 = 16）
npu-smi info -l
python3 -c "import torch, torch_npu; print('device_count:', torch.npu.device_count())"

# 2) 主机内存（sparse offload 的 host pool 用）
free -g

# 3) 权重
ls -l "$MODEL_PATH"
du -sh "$MODEL_PATH"

# 4) 权重结构：必须有 index_topk，不能有 compress_ratios / index_kpool
grep -oE '"(index_topk|compress_ratios|index_kpool)"' "$MODEL_PATH/config.json" | sort -u

# 5) 不能是 C8 KV
grep -o '"kv_cache_type"[^,}]*' "$MODEL_PATH/quant_model_description.json" 2>/dev/null

# 6) vLLM 环境
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
vllm serve --help > /dev/null && echo "vllm CLI OK"

# 7) main 分支应装 .github/vllm-main-verified.commit 对应的 vLLM commit
VLLM_UPSTREAM=${VLLM_UPSTREAM:-/vllm-workspace/vllm}
test "$(git -C "$VLLM_UPSTREAM" rev-parse HEAD 2>/dev/null)" = \
     "$(tr -d '[:space:]' < .github/vllm-main-verified.commit)" && echo "vllm commit OK"
```

| 检查项 | 期望 |
|---|---|
| 逻辑 NPU | 16（8 物理卡 × 2 chip），每 chip 64GB |
| 权重 | `GLM-5.1-w4a8`，实测约 391GB，非 c8 |
| config.json | 有 `index_topk`，无 `compress_ratios` |
| vLLM | main-verified commit（版本号可能是 `v0.30.0` 之类，正常） |
| 依赖 | `memfabric_hybrid`、`memcache_hybrid` 已安装 |

> `GLM-5.2-w8a8` 文档下限 1TB，本机刚好达到但余量极小；`GLM-5.2-w4a8` 实测 755GB，TP16 下每 chip 约 47GB，32K 可试但很紧。本文以 391GB 的 `GLM-5.1-w4a8` 为准。

---

## 4. 获取优化后的代码（PR #20927）

你的仓库已经在优化分支上（`dsa-a3-opt-only`，与 PR `happy-math:temp2` 内容一致）时，直接确认即可：

```bash
cd "$VLLM_SRC"
git status -sb
git log --oneline -5

# 三个 csrc 文件应已被修改
git log --oneline -- \
  csrc/attention/fused_sparse_attention_overlap/op_host/fused_sparse_attention_overlap_tiling.cpp \
  csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_kernel_mla.h \
  csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_service_vector_mla.h
```

如果本地没有这个分支，从 PR 拉：

```bash
cd "$VLLM_SRC"
git remote add pr20927 https://github.com/happy-math/vllm-ascend.git 2>/dev/null || true
git fetch pr20927 temp2
git checkout -b pr20927 pr20927/temp2
```

确认改动范围（本地 `dsa-a3-opt-only` 的基线是 `96e222490`）：

```bash
git diff --stat 96e222490..HEAD | tail -10
```

应只涉及 `fused_sparse_attention_overlap` 的 3 个 csrc 文件（外加 release note 等非代码文件）。
如果你是从 `pr20927/temp2` 拉的，基线 commit 可能不同，直接确认这 3 个文件被改过即可。

> PR 说明：所有 kernel 改动都限定在 arch22（A2/A3），arch35（Ascend 950）路径和算子 schema 未动；优化后的 kernel 在现有路径上自动生效，**不需要新增开关**。

---

## 5. 完整编译安装（单套）

> ⚠️ 跑模型必须安装**完整**的 vllm-ascend。模型会调用 `mla_preprocess`、`sparse_flash_attention`、`moe_gating_top_k`、`grouped_matmul_swiglu_quant`、`dispatch_ffn_combine` 等约 40 个 A3 自定义算子；只装 `--ops=fused_sparse_attention_overlap` 的单算子包会让服务起不来。

```bash
cd "$VLLM_SRC"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export SOC_VERSION=ascend910_9382
export MAX_JOBS=8
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/devlib:${LD_LIBRARY_PATH:-}"

# 全量算子构建会占用较多磁盘，先确认
df -h "$VLLM_SRC"

rm -rf -- csrc/build csrc/output csrc/build_out

# 默认就是 1：构建全部 A3 自定义算子 + vllm_ascend_C 扩展
COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e . --no-build-isolation
```

首次构建时间较长（常见 30 分钟以上），建议放在 tmux / nohup 里跑。

> **换算子 / 换分支的正确做法**：`git checkout` 之后必须 `rm -rf csrc/build` 再执行 `COMPILE_CUSTOM_KERNELS=1 pip install -e .`。
> `COMPILE_CUSTOM_KERNELS=0` 只重装 Python 包，**不会重新编译算子**；也不能只替换 `build.sh --ops=...` 的单算子包。

> 如果 `pip install -e .` 在构建全量算子阶段失败，可以拆成两步：
> 1. `bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"`（构建完整算子包）；
> 2. 按[附录 B](#附录-bpython-扩展编译备用方案)单独构建 `vllm_ascend_C` 扩展。

### 5.1 验收

```bash
cd "$VLLM_SRC"

ls vllm_ascend/_cann_ops_custom/vendors/custom_transformer/
find vllm_ascend/_cann_ops_custom -type f | wc -l

python3 - <<'PY'
import torch, vllm_ascend
from vllm_ascend.utils import enable_custom_op
print("vllm_ascend:", vllm_ascend.__file__)
enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
print("fused_sparse_attention_overlap 已加载")
PY
```

再跑单算子精度测试（见[附录 A](#附录-a单算子测试)），应为 `4 passed`。

---

## 6. 安装压测客户端

> 单独 venv，避免 aisbench 的依赖降级 numpy / transformers，把 vLLM 环境搞坏。

```bash
source /home/z00943842/base/fsao_ab/env.sh

python3 -m venv "$AISV_DIR"
source "$AISV_DIR/bin/activate"
pip install -U pip
pip install ais-bench-benchmark pandas

# 记下 WORK_PATH（目录下必须有 ais_bench/）
python3 -c "import ais_bench, os; print(os.path.dirname(os.path.dirname(ais_bench.__file__)))"
# 或者： pip show -f ais-bench-benchmark | grep Location
```

下载工具：

```bash
cd "$WORK_ROOT"
git clone https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git   # 已存在则跳过
cd "$AISBENCH_TOOL"
```

编辑 `config.py`：

```python
DATASET_PATH = "/home/z00943842/base/dataset"      # 与 env.sh 的 DATASET_DIR 一致
WORK_PATH    = "<上面打印的目录>"                   # 包含 ais_bench/ 的 site-packages
MODEL_NAME   = "glm-5.1"                          # 必须等于 --served-model-name
MODEL_PATH   = "/mnt/weight/GLM-5.1-w4a8"          # 只用来读 tokenizer
HOST_IP      = "127.0.0.1"                        # 客户端与服务同机
HOST_PORT    = "8000"
API_KEY      = ""            # 本机 vllm serve 没配 --api-key，留空
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR   = "./outputs/default"
POD_INFO     = []
URL          = ""
```

> `WORK_PATH` 是**包含 `ais_bench/` 包的目录**。工具会往里写数据集软链和 `vllm_api_chat_temp.py`。
> `API_KEY` 只有服务端加了 `--api-key` 或走带鉴权网关时才需要填。

自检：

```bash
source "$AISV_DIR/bin/activate"
python3 -c "from config import *; print(WORK_PATH, MODEL_NAME); import os; print(os.path.isdir(os.path.join(WORK_PATH,'ais_bench')))"
```

---

## 7. 启动整网服务（带 offload）

### 7.1 启动前清理

```bash
source /home/z00943842/base/fsao_ab/env.sh

pkill -9 -f "vllm serve" || true
pkill -9 -f EngineCore || true
sleep 5
npu-smi info          # 确认显存已释放
```

### 7.2 环境变量（HCCL）

```bash
export HCCL_IF_IP=10.246.63.33
export HCCL_SOCKET_IFNAME=enp162s0f0
export GLOO_SOCKET_IFNAME=enp162s0f0
export TP_SOCKET_IFNAME=enp162s0f0
export HCCL_BUFFSIZE=512          # MoE AllToAll 需要 >206MB
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

# 不要设置 HCCL_INTRA_PCIE_ENABLE / HCCL_INTRA_ROCE_ENABLE
# （要么都 unset 自动选择，要么显式 PCIE=1 + ROCE=0，不能只设一个）
unset HCCL_INTRA_PCIE_ENABLE
unset HCCL_INTRA_ROCE_ENABLE

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
```

### 7.3 启动命令

```bash
source /home/z00943842/base/fsao_ab/env.sh

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-num-seqs 8 \
  --max-model-len 40960 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "enable_balance_scheduling": true,
    "multistream_overlap_shared_expert": true,
    "sparse_kv_offload_config": {
      "enabled": true,
      "use_fused_overlap": true,
      "keep_device_kv_cache": true,
      "dram_size_per_dp_GB": 48
    }
  }' 2>&1 | tee "$AB/results/server_$(date +%m%d_%H%M).log"
```

参数约束：

| 约束 | 说明 |
|---|---|
| 不要 `--kv-cache-dtype int8/fp8` | 主 KV 会变成 C8 packed cache，sparse offload 直接拒绝 |
| 不要 `enable_dsa_cp` | sparse offload 不支持 context parallel |
| `--tensor-parallel-size 16` | 391GB ÷ 16 ≈ 24.5GB/chip；TP8 时每 chip 49GB，留给 KV 只有约 8GB |
| 权重不能带 `c8` | `w8a8c8` / `w4a8c8` 均被拒 |
| 第一次先不加 `--speculative-config` | MTP 支持，但先减少变量 |
| `HCCL_BUFFSIZE=512` | 否则 MoE AllToAll 报 `HCCL_BUFFSIZE_EP is too SMALL` |

> 首次启动会从 NFS 加载 391GB 权重，日志停在 `Loading safetensors checkpoint shards: 0%` 很久是正常的（见第 10 节）。

### 7.4 启动验收

```bash
curl -s http://127.0.0.1:8000/health; echo
curl -s http://127.0.0.1:8000/v1/models | head -c 500; echo

curl -s http://127.0.0.1:8000/metrics | grep prefix_cache | head

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0}'
```

---

## 8. 验证优化算子确实生效

`use_fused_overlap=true` 时：

- decode 分发走 `_execute_fused_overlap_offload_decode`（`sfa_kv_offload.py:1594`）；
- 该函数通过 `_require_custom_op("npu_fused_sparse_attention_overlap")` 取算子，**算子缺失会直接抛错**；
- 因此：**decode 请求正常返回且日志无异常，就说明算子被调用了**；
- `/metrics` 能查到 `vllm:prefix_cache_*`，说明 sparse offload 路径整体生效。

补充佐证（可选）：

1. 用同一条请求，分别以 `use_fused_overlap=true` / `false` 启动服务，比较 TPOT。两条路径实现不同，差异明显说明走的确实是不同路径（注意：`false` 是 Copy-SFA，不是优化前的 kernel 基线）。
2. 抓一次短 profile，确认 kernel 名出现在设备侧（可选，见 vLLM profiler 文档）。

---

## 9. 压测与记录

### 9.1 默认场景（覆盖 helper）

```bash
source /home/z00943842/base/fsao_ab/env.sh
source "$AISV_DIR/bin/activate"
cd "$AISBENCH_TOOL"
unset http_proxy https_proxy

python3 aisbench_test.py \
  --input_len 32768 \
  --output_len 1024 \
  --data_num 32 \
  --concurrency 8 \
  --request_rate 0 \
  --npu_num 16 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test \
  --seed 2
```

### 9.2 补充场景（覆盖 P0 / Opt05）

默认 cc8/90% 场景 `bs=8`，只覆盖 helper 和通用回写；PR 里的两项收益需要：

| 优化点 | 生效条件 | 场景 |
|---|---|---|
| helper | `bs ≤ 12` | 默认 cc8 |
| P0 行划分 | `bs > 24` | cc32 |
| Opt05 UB 直写 | 命中率 0~70% | repeat_rate 0.5 |

```bash
python3 aisbench_test.py \
  --input_len 16384 \
  --output_len 1024 \
  --data_num 64 \
  --concurrency 32 \
  --request_rate 0 \
  --npu_num 16 \
  --dataset_type prefix_cache \
  --repeat_rate 0.5 \
  --prefix_test \
  --seed 2
```

> 跑这个场景时把服务的 `--max-num-seqs` 调到 32，并重启。

### 9.3 保存结果

```bash
mv aisbench_result.csv "$AB/results/default_r1.csv"
cp aisbench.log          "$AB/results/default_r1.log"
```

建议每个场景跑 **3 轮**，取中位数（单轮噪声可能有 1~2%）。CSV 是追加写，每轮跑完先移走。

### 9.4 结果记录表

| 场景 | 轮次 | TPOT avg(ms) | TPOT P90(ms) | output_throughput | single_output_throughput | E2E_time(s) | 命中率 |
|---|---|---|---|---|---|---|---|
| 默认 32K/cc8/90% | 1 | | | | | | |
| | 2 | | | | | | |
| | 3 | | | | | | |
| | **中位数** | | | | | | |
| 补充 16K/cc32/50% | 1 | | | | | | |
| | 2 | | | | | | |
| | 3 | | | | | | |
| | **中位数** | | | | | | |

快速取中位数（在 `$AB/results` 下）：

```bash
python3 - <<'PY'
import pandas as pd, glob
for tag in ("default", "supplement"):
    files = glob.glob(f"{tag}_r*.csv")
    if not files:
        continue
    df = pd.concat([pd.read_csv(f) for f in files])
    print(tag)
    print(df[["TPOT avg", "TPOT SLO_P90", "output_throughput",
              "single_output_throughput", "E2E_time", "repeat_rate"]].median(numeric_only=True))
PY
```

---

## 10. 结果判读

| 指标 | 说明 |
|---|---|
| `TPOT avg` / `TPOT SLO_P90` | 核心指标，算子每个 decode step 执行 |
| `output_throughput` | 端到端吞吐 |
| `E2E_time` | 整轮耗时 |
| `TTFT avg` | 基本不变（算子在 decode） |
| 命中率 | 默认场景应 ≈90%；明显偏低说明 KV 容量不足 |

写报告时建议这样表述：

- 本文验证的是**优化后的算子在整网 decode 路径上正确接入并运行**，给出绝对性能；
- PR 的收益数据来自单算子测试（selection-update 路径 -19%~-42%、拷贝发射 -33%、命中率 0~70% 时 +10%~+12%）；
- 如果评审需要整网 A/B 对比，再另起一份优化前代码的环境做对照（见 `FSAO_E2E_AB_操作手册.md`）。

---

## 11. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'orjson'` | vLLM 依赖不全 | 用 `vllm` 入口同一个解释器装：`/usr/local/python3.12.16/bin/python3 -m pip install orjson`；缺得多就装 `-r /vllm-workspace/vllm/requirements/common.txt`（不要装 `cuda.txt`） |
| `hcclGetRootInfo ... error code is 19` / `ERR02200` | HCCL 选网卡失败或 hostname 解析不到 | 设 `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`；`/etc/hosts` 补 hostname；清残留进程 |
| `HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE is invalid` | 只设了一个 intra 变量 | 两个都 unset；或显式 `PCIE=1 + ROCE=0`，不能同时为 1 |
| `same physical device ID 0` | 自检脚本没绑定各自 NPU | 加 `torch.npu.set_device(local_rank)`，张量用 `npu:{local_rank}` |
| `HCCL_BUFFSIZE_EP is too SMALL ... NEEDED_HCCL_BUFFSIZE=...`，随后 `vector core exception` | MoE AllToAll 缓冲区不足 | `export HCCL_BUFFSIZE=512`（单位 MB），重启 |
| `Loading safetensors checkpoint shards: 0%` 很久不动 | 391GB 权重从 NFS 首次加载 | 正常，等；`cat /proc/<pid>/io` 看 `read_bytes` 是否增长；NFS 0.3~1GB/s 需要 7~20 分钟 |
| 加载完初始化 KV 时 OOM | TP8 下每 chip 权重 49GB，余量太小 | 改 `--tensor-parallel-size 16` |
| 权重上卡后日志停在编译/图捕获 | `FULL_AND_PIECEWISE` 图捕获慢/OOM | 用 `FULL_DECODE_ONLY`，或 `--enforce-eager` |
| `Sparse KV offload offload.initialize failed` | host 连续内存不足 | 降 `dram_size_per_dp_GB`（先 48），检查 `free -g` |
| `does not support packed SFA main caches (C8 or TQ4)` | 用了 c8 权重或 `--kv-cache-dtype int8/fp8` | 换非 c8 权重，删掉 `--kv-cache-dtype` |
| `Sparse KV offload currently requires TP without context parallelism` | 打开了 `enable_dsa_cp` 或 CP | 从 additional-config 删掉 `enable_dsa_cp` |
| `Sparse KV offload don't support compress now` | 权重 config 有 `compress_ratios` | 换 GLM-5.1/5.2 的标准 DSA 权重 |
| 找不到算子 / `libcust_opapi.so` 加载失败 | 只装了单算子包，或服务没重启 | 按第 5 节完整构建；重启进程 |
| `/metrics` 查不到 `prefix_cache` | 代理未关 / 指标目录未设 | `unset http_proxy https_proxy`；设置 `PROMETHEUS_MULTIPROC_DIR` |
| `unrecognized arguments: --num-warmups` | aisbench 版本差异 | 按工具 README FAQ 3 删除该参数 |

---

## 12. 模型 / 权重约束

| 权重 | 结论 |
|---|---|
| `GLM-5.1-w4a8`（实测 391GB） | ✅ 本文使用；TP16 下每 chip 约 24.5GB |
| `GLM-5.2-w4a8`（755GB） | ⚠️ TP16 下每 chip 约 47GB，32K 可试但余量小 |
| `GLM-5.2-w8a8` | ⚠️ 文档下限 1TB，本机刚好达到，余量极小 |
| 任何 `*c8` 权重 | ❌ C8 主 KV 与 sparse offload 互斥 |
| DeepSeek-V4 / V4.1-Flash | ❌ 带 `compress_ratios`（CSA2），不走 SFA/offload 路径 |

权重自检：

```bash
grep -oE '"(index_topk|compress_ratios|index_kpool)"' "$MODEL_PATH/config.json" | sort -u
# 期望只有 "index_topk"

grep -c '"compress_ratios"' "$MODEL_PATH/config.json"
# 0 才可以
```

场景建议（8 × 64GB / 16 chip）：

| 场景 | 说明 |
|---|---|
| 32K × 32 请求 | 首选，先出结论 |
| 64K × 16 请求 | 可以试 |
| 128K × 32 请求 | 不建议，每 chip KV 约 45GB + 权重容易超 64GB |

---

## 附录 A：单算子测试

```bash
source /home/z00943842/base/fsao_ab/env.sh
cd "$VLLM_SRC"
source "$VLLM_SRC/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

python3 - <<'PY'
import torch, vllm_ascend
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
print("算子加载成功")
PY

VLLM_USE_MODELSCOPE=true pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
```

期望：`4 passed`。

---

## 附录 B：Python 扩展编译（备用方案）

当 `COMPILE_CUSTOM_KERNELS=1 pip install -e .` 在构建全量算子阶段失败时：

```bash
cd "$VLLM_SRC"
rm -rf -- csrc/build csrc/output csrc/build_out
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"

# 只重装 Python 包并编译扩展，不再重建算子
COMPILE_CUSTOM_KERNELS=0 python3 -m pip install -e . --no-build-isolation

PYBIND11_CMAKE_DIR="$(python3 -m pybind11 --cmakedir)"
PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
TORCH_NPU_PATH="$(python3 -c 'import os, torch_npu; print(os.path.dirname(torch_npu.__file__))')"
EXT_BUILD_DIR="$VLLM_SRC/build/fsao_only_ext"

cmake -S "$VLLM_SRC" -B "$EXT_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DPYTHON_INCLUDE_PATH="$PYTHON_INCLUDE_PATH" \
  -DCMAKE_INSTALL_PREFIX="$VLLM_SRC/vllm_ascend" \
  -DCMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR" \
  -DTORCH_NPU_PATH="$TORCH_NPU_PATH" \
  -DSOC_VERSION="$SOC_VERSION"

cmake --build "$EXT_BUILD_DIR" --target vllm_ascend_C -j"$MAX_JOBS"
cmake --install "$EXT_BUILD_DIR"
```

---

## 附录 C：16 逻辑 NPU 的 PD 分离（可选）

本机可以切成 P TP8（chip 0-7）+ D TP8（chip 8-15）。每个实例都要加载完整权重：`GLM-5.1-w4a8` 391GB，每侧 8 × 64GB = 512GB，每 chip 约 49GB 权重，余量约 15GB，偏紧。

- Prefill（`ASCEND_RT_VISIBLE_DEVICES=0-7`，`--tensor-parallel-size 8`）：

```bash
--kv-transfer-config '{
  "kv_connector": "SfaRemoteD2HConnector",
  "kv_role": "kv_producer"
}'
```

  不设置 `sparse_kv_offload_config`。

- Decode（`ASCEND_RT_VISIBLE_DEVICES=8-15`，`--tensor-parallel-size 8`）：

```bash
--additional-config '{
  "sparse_kv_offload_config": {
    "enabled": true,
    "use_fused_overlap": true,
    "dram_size_per_dp_GB": 110
  }
}'
--kv-transfer-config '{
  "kv_connector": "SfaRemoteD2HConnector",
  "kv_role": "kv_consumer",
  "kv_port": 20050
}'
```

  注意：**不要**带 `keep_device_kv_cache`。

- 代理：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host 127.0.0.1 --port 9000 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 8200
```

- 压测客户端 `HOST_PORT` 改成代理端口 `9000`，`POD_INFO` 填 Decode 节点地址。

---

## 附录 D：清理

```bash
source /home/z00943842/base/fsao_ab/env.sh

pkill -f "vllm serve" || true

cd "$VLLM_SRC"
git status
```

---

## 附录 E：HCCL 最小自检

```python
# /tmp/hccl_test.py
import os
import torch
import torch_npu
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group("hccl", rank=int(os.environ["RANK"]),
                        world_size=int(os.environ["WORLD_SIZE"]))
t = torch.ones(1, device=f"npu:{local_rank}")
dist.all_reduce(t)
print(f"rank {os.environ['RANK']} device {local_rank} ok {t.item()}")
dist.destroy_process_group()
```

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=29500 /tmp/hccl_test.py
```

| 现象 | 含义 |
|---|---|
| `same physical device ID 0` | 脚本没绑定各自 NPU（缺 `torch.npu.set_device`） |
| `EI0015 Ranktable_Detect` + 网卡提示 | `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` 不对 |
| `HCCL_INTRA_* invalid` | intra 变量要么都 unset，要么显式互斥 |
| 8 个 rank 都 `ok` | HCCL 正常 |
