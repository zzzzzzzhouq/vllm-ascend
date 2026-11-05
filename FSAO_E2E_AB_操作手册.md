# FusedSparseAttentionOverlap 整网 E2E A/B 测试操作手册

> 适用环境：Atlas A3 单机 8 卡
> - 64GB × 8：优先 **GLM-5.2-w4a8**（非 c8），放不下再换 **GLM-5.1-w4a8**；场景用 32K / 64K（见 10.2）
> - 128GB × 8：可用 **GLM-5.2-w8a8**，场景可上 128K
> 代码分支：`dsa-a3-opt-only`（基线 `96e222490`，优化 `HEAD`）
> 目标：验证 `csrc/attention/fused_sparse_attention_overlap` 的 arch22 kernel 优化在整网 decode 上的性能收益
> 编写日期：2026-11-05

---

## 目录

- [0. 一页纸执行清单](#0-一页纸执行清单)
- [1. 背景：为什么不能直接起服务](#1-背景为什么不能直接起服务)
- [2. 变量约定（先填好）](#2-变量约定先填好)
- [3. 阶段 0：环境盘点](#3-阶段-0环境盘点)
- [4. 阶段 1：编译基线 / 优化两个算子包](#4-阶段-1编译基线--优化两个算子包)
- [5. 阶段 2：安装压测客户端](#5-阶段-2安装压测客户端)
- [6. 阶段 3：启动服务并确认算子生效](#6-阶段-3启动服务并确认算子生效)
- [7. 阶段 4：执行 A/B 测试](#7-阶段-4执行-ab-测试)
- [8. 结果判读](#8-结果判读)
- [9. 故障排查表](#9-故障排查表)
- [10. 权重与模型专属问题](#10-权重与模型专属问题glm-51--52)
- [附录 A：单算子测试复核](#附录-a单算子测试复核)
- [附录 B：Python 扩展编译（若未做过）](#附录-bpython-扩展编译若未做过)
- [附录 C：借到 16 卡时的 PD 分离方案](#附录-c借到-16-卡时的-pd-分离方案)
- [附录 D：清理与回滚](#附录-d清理与回滚)

---

## 0. 一页纸执行清单

全部在**同一个 docker 容器内**执行。

- [ ] 1. `npu-smi info`、`free -g`、确认每卡显存（65536≈64GB，131072≈128GB），据此选权重
- [ ] 2. 确认权重：64GB 卡优先 `GLM-5.2-w4a8`（兜底 `GLM-5.1-w4a8`），128GB 卡用 `GLM-5.2-w8a8`；都必须是**非 c8**，`config.json` 有 `index_topk`、无 `compress_ratios`
- [ ] 3. 编译**基线**算子包 → `~/fsao_ab/ops/base`
- [ ] 4. 编译**优化**算子包 → `~/fsao_ab/ops/opt`
- [ ] 5. 两个包分别跑仓内单算子测试，均 `4 passed`
- [ ] 6. 单独 venv 安装 `ais-bench-benchmark`，克隆压测工具、改 `config.py`
- [ ] 7. 起服务（带 `sparse_kv_offload_config`），`/health` 正常、单请求成功
- [ ] 8. 按 ABBA 顺序跑 6 轮（base 3 轮 / opt 3 轮），每轮换算子包 + 重启服务
- [ ] 9. 每轮保存 `aisbench_result.csv` 和 `aisbench.log`
- [ ] 10. 取 3 轮中位数，比较 TPOT / 输出吞吐 / 命中率

---

## 1. 背景：为什么不能直接起服务

同事给的 `aisbench_test.py` 只是**压测客户端**，默认已经有一个 vLLM 服务在跑。本算子在整个模型里的调用条件很严格：

1. 必须是稀疏注意力模型（`index_topk`）：GLM-5.1 / GLM-5.2 / DeepSeek-V3.2；
2. 启动配置必须 `sparse_kv_offload_config.enabled=true`；
3. 必须 `use_fused_overlap=true`（默认 `false`，走的是 Copy-SFA 路径，本算子不会被调用）；
4. 只在 **decode** 阶段调用，读的是 host 侧 KV pool。

代码位置：

- 算子调用点：`vllm_ascend/attention/sfa_kv_offload.py:1456`
- decode 分发逻辑：`vllm_ascend/attention/sfa_kv_offload.py:1594`
- 配置默认值：`vllm_ascend/ascend_config.py:1698`

**8 卡为什么只能跑 colocate 调试模式**：生产形态是 PD 分离（P、D 各占一组卡），TP8 + TP8 需要 16 卡。8 卡机器使用代码里预留的单机调试模式：

```json
"keep_device_kv_cache": true
```

此时 prefill 产生的 KV 会 D2H 写入 host pool，decode 阶段算子照常从 host pool 读取。**绝对性能不代表生产 PD 分离，但 A/B 的相对收益有效。**

> PR 里请把结论标注为「单机 PD-colocate 调试模式下的相对收益验证」。

---

## 2. 变量约定（先填好）

在容器内创建环境文件，后续每个新终端都先 `source` 它：

```bash
mkdir -p ~/fsao_ab/results
cat > ~/fsao_ab/env.sh <<'EOF'
# ===== 按实际情况修改这三行 =====
# 64GB 卡首选：/root/models/GLM-5.2-w4a8（确认非 c8；OOM 再换 GLM-5.1-w4a8）
# 128GB 卡：/root/models/GLM-5.2-w8a8
export MODEL_PATH=/root/models/GLM-5.2-w4a8        # 权重绝对路径
export VLLM_SRC=$HOME/vllm-ascend                  # 你的 vllm-ascend 仓库
export NPU_NUM=8                                   # 本机卡数
# ==============================

export AB=$HOME/fsao_ab
export OPS_BASE=$AB/ops/base
export OPS_OPT=$AB/ops/opt
export SERVED_NAME=glm-5.2                         # 与 --served-model-name 一致（换 5.1 时同步改）
export HOST_IP=127.0.0.1
export PORT=8000
export DATASET_DIR=/root/dataset
EOF

source ~/fsao_ab/env.sh
mkdir -p "$OPS_BASE" "$OPS_OPT" "$DATASET_DIR"
```

---

## 3. 阶段 0：环境盘点

```bash
source ~/fsao_ab/env.sh

# 1) 卡型号和显存（看 HBM-Usage 的总量：65536≈64GB，131072≈128GB）
npu-smi info

#    更明确的两种方式：
npu-smi info -t memory -i 0          # 看 Memory Size(MB) 一类字段
python3 - <<'PY'
import torch, torch_npu
for i in range(torch.npu.device_count()):
    free, total = torch.npu.mem_get_info(i)
    print(f"NPU {i}: {total/1024**3:.1f} GB  (free {free/1024**3:.1f} GB)")
PY

#    板卡型号：
npu-smi info -t board -i 0 | grep -iE "product|chip|model"

# 2) 主机内存（决定 dram_size_per_dp_GB；注意这不是卡上的 HBM）
free -g

# 3) 权重目录与大小（512GB 显存建议 w4 权重目录 ≤ 450GB）
ls -l "$MODEL_PATH"
du -sh "$MODEL_PATH"

# 4) 权重结构检查：期望 index_topk，不能有 compress_ratios / index_kpool
grep -oE '"(index_topk|compress_ratios|index_kpool)"' "$MODEL_PATH/config.json" | sort -u

#    确认不是 C8 KV（有输出且包含 C8 就是 c8 权重，不能用）
grep -o '"kv_cache_type"[^,}]*' "$MODEL_PATH/quant_model_description.json" 2>/dev/null

# 5) transformers 版本（GLM-5 系列教程要求 >= 5.2.0）
python3 -c "import transformers; print(transformers.__version__)"

# 6) 确认 vllm-ascend 是源码可编辑安装（后面要换 _cann_ops_custom）
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
```

**硬门槛**（任一不满足，先停下来处理）：

| 检查项 | 期望 | 不满足怎么办 |
|---|---|---|
| 每卡显存 | 128GB 或 64GB | **64GB 不能跑 GLM-5.2-w8a8**；用 5.2-w4a8 / 5.1-w4a8 并缩小场景，见 10.2 |
| 权重后缀 | 64GB：`GLM-5.2-w4a8` / `GLM-5.1-w4a8`；128GB：`GLM-5.2-w8a8` | 不能用任何 `*c8`（C8 主 KV 被 offload 拒绝） |
| config.json | 有 `index_topk`，无 `compress_ratios` | 该权重不支持 sparse offload |
| KV 量化 | `quant_model_description.json` 里不是 `C8` | c8 权重不能用，换非 c8 版本 |
| 权重体积 | w4 权重目录 ≤ 450GB（8 × 64GB 场景） | 超了换 GLM-5.1-w4a8，或借 16 卡 |
| 主机内存 | ≥ 200GB 空闲 | 下调 `dram_size_per_dp_GB` |
| vllm_ascend 路径 | `$VLLM_SRC/vllm_ascend/__init__.py` | 重新执行 `pip install -e .` |

记录到下表，方便写报告：

| 项目 | 值 |
|---|---|
| 卡数 / 型号 / 单卡显存 | |
| host 内存 | |
| 权重目录 | |
| transformers 版本 | |
| vllm-ascend commit | `git -C $VLLM_SRC rev-parse --short HEAD` |
| 驱动 / CANN 版本 | `npu-smi info` + `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` |

---

## 4. 阶段 1：编译基线 / 优化两个算子包

> 你的分支相对基线只改了 3 个 csrc 文件，**Python 代码完全一致**，所以 A/B 只需要两个算子包，不需要两个镜像、两套 vLLM。

### 4.1 拉基线代码（worktree，不切分支）

```bash
source ~/fsao_ab/env.sh
cd "$VLLM_SRC"
git fetch --all
git worktree add ~/vllm-ascend-base 96e222490
git -C ~/vllm-ascend-base log --oneline -1
```

### 4.2 编译环境变量（基线 / 优化两个终端都 source）

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export SOC_VERSION=ascend910_9382
export MAX_JOBS=8
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/devlib:${LD_LIBRARY_PATH:-}"
```

### 4.3 编译并安装**基线**包

```bash
cd ~/vllm-ascend-base
rm -rf -- csrc/build csrc/output csrc/build_out
cd csrc
bash build.sh --pkg --ops=fused_sparse_attention_overlap --soc=ascend910_93
cd ..

chmod +x csrc/build/cann-ops-transformer*.run
csrc/build/cann-ops-transformer*.run --install-path="$HOME/fsao_ab/ops/base"
```

### 4.4 编译并安装**优化**包

```bash
cd "$VLLM_SRC"
rm -rf -- csrc/build csrc/output csrc/build_out
cd csrc
bash build.sh --pkg --ops=fused_sparse_attention_overlap --soc=ascend910_93
cd ..

chmod +x csrc/build/cann-ops-transformer*.run
csrc/build/cann-ops-transformer*.run --install-path="$HOME/fsao_ab/ops/opt"
```

> ⚠️ **不要**在 `~/vllm-ascend-base` 里执行 `pip install -e .`。Python 包必须始终指向 `$VLLM_SRC`（HEAD），基线只取它的算子二进制。

### 4.5 验收

```bash
# 两个包目录结构应一致
ls ~/fsao_ab/ops/base/vendors/custom_transformer/
ls ~/fsao_ab/ops/opt/vendors/custom_transformer/

# 产物应有差异（证明是两个不同二进制）
diff -rq ~/fsao_ab/ops/base ~/fsao_ab/ops/opt | head -20
```

再用[附录 A](#附录-a单算子测试复核)分别在两个包下跑单算子测试，均应为 `4 passed`。

### 4.6 切换机制（重要）

`vllm_ascend/utils.py:454` 的 `bootstrap_custom_op_env()` 会**写死从包目录下的 `_cann_ops_custom` 加载算子**，并 prepend 到 `ASCEND_CUSTOM_OPP_PATH`。所以不能只改环境变量，必须在每次启动服务前替换目录：

```bash
# 切到基线
rm -rf "$VLLM_SRC/vllm_ascend/_cann_ops_custom"
cp -a "$OPS_BASE" "$VLLM_SRC/vllm_ascend/_cann_ops_custom"

# 切到优化
rm -rf "$VLLM_SRC/vllm_ascend/_cann_ops_custom"
cp -a "$OPS_OPT" "$VLLM_SRC/vllm_ascend/_cann_ops_custom"

# 确认当前是哪个
ls -l "$VLLM_SRC/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
```

切换后必须**重启服务进程**（算子在进程 import 时加载），并且建议**开新终端**，避免上一边 `set_env.bash` 留下的环境变量污染。

---

## 5. 阶段 2：安装压测客户端

> 建议单独 venv，避免 aisbench 的依赖降级 numpy / transformers，把跑 vLLM 的环境搞坏。

### 5.1 安装 aisbench

```bash
python3 -m venv ~/aisvenv
source ~/aisvenv/bin/activate
pip install -U pip
pip install ais-bench-benchmark pandas

# 记下 WORK_PATH（目录下必须有 ais_bench/）
python3 -c "import ais_bench, os; print(os.path.dirname(os.path.dirname(ais_bench.__file__)))"
# 或者： pip show -f ais-bench-benchmark | grep Location
```

### 5.2 下载压测工具并改配置

```bash
cd ~
git clone https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git
cd aisbench_auto_tools_prefix
```

编辑 `config.py`：

```python
DATASET_PATH = "/root/dataset"                     # 阶段 2 已建好
WORK_PATH    = "<5.1 打印的目录>"                   # 目录下要有 ais_bench/
MODEL_NAME   = "glm-5.2"                          # 必须等于 --served-model-name
MODEL_PATH   = "/root/models/GLM-5.2-w4a8"         # 只用来读 tokenizer（与 env.sh 一致）
HOST_IP      = "127.0.0.1"
HOST_PORT    = "8000"
API_KEY      = ""
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR   = "./outputs/default"
POD_INFO     = []
URL          = ""
```

### 5.3 客户端自检

```bash
source ~/aisvenv/bin/activate
python3 -c "from config import *; print(WORK_PATH, MODEL_NAME); import os; print(os.path.isdir(os.path.join(WORK_PATH,'ais_bench')))"
# 期望最后输出 True
```

> 如果运行时报 `ais_bench: error: unrecognized arguments: --num-warmups`，按工具 README 的 FAQ 3，在 `aisbench_test.py` 第 94 行删掉 `--num-warmups 0`。

---

## 6. 阶段 3：启动服务并确认算子生效

### 6.1 启动前检查

```bash
source ~/fsao_ab/env.sh

# 1) 没有残留服务
pkill -f "vllm serve" || true
sleep 10
npu-smi info          # 确认显存已释放

# 2) 确认当前算子包（base 或 opt）
rm -rf "$VLLM_SRC/vllm_ascend/_cann_ops_custom"
cp -a "$OPS_BASE" "$VLLM_SRC/vllm_ascend/_cann_ops_custom"   # 本轮跑基线
source "$VLLM_SRC/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

# 3) 关代理（客户端抓 /metrics 需要）
unset http_proxy https_proxy
```

### 6.2 启动命令（64GB × 8：GLM-5.2-w4a8 优先，GLM-5.1-w4a8 兜底）

建议用 tmux 开两个窗口：一个跑服务，一个跑压测。

```bash
source ~/fsao_ab/env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p $PROMETHEUS_MULTIPROC_DIR

source "$VLLM_SRC/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
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
    "enable_flashcomm1": true,
    "sparse_kv_offload_config": {
      "enabled": true,
      "use_fused_overlap": true,
      "keep_device_kv_cache": true,
      "dram_size_per_dp_GB": 48
    }
  }' 2>&1 | tee "$AB/results/server_$(date +%m%d_%H%M).log"
```

> **128GB × 8 + GLM-5.2-w8a8 时**可放宽为：
> `--max-model-len 140000 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95`，
> `dram_size_per_dp_GB` 设 80。其余不变。

**必须遵守的参数约束**（否则进不了算子路径或直接报错）：

| 约束 | 说明 |
|---|---|
| 不要 `--kv-cache-dtype int8/fp8` | 主 KV 会变成 C8 packed cache，sparse offload 直接拒绝 |
| 不要 `enable_dsa_cp` | sparse offload 不支持 context parallel，会 `NotImplementedError` |
| `--data-parallel-size 1` | 8 卡单机，DP2×TP8 是 16 卡配置 |
| 权重不能带 `c8` | `w8a8c8` / `w4a8c8` 均被拒 |
| 第一次先不加 `--speculative-config` | MTP 支持但先减少变量，跑通后再加 |

### 6.3 启动验收

```bash
# 1) 健康检查（首次加载模型可能需要几分钟）
curl -s http://$HOST_IP:$PORT/health

# 2) 模型列表
curl -s http://$HOST_IP:$PORT/v1/models

# 3) prefix cache 指标（必须能查到）
curl -s http://$HOST_IP:$PORT/metrics | grep prefix_cache | head

# 4) 发一条短请求
curl -s http://$HOST_IP:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

### 6.4 如何证明算子真的在跑

- `use_fused_overlap=true` 时，decode 分发走 `_execute_fused_overlap_offload_decode`（`sfa_kv_offload.py:1594`）；
- 该函数通过 `_require_custom_op("npu_fused_sparse_attention_overlap")` 取算子，**算子缺失会直接抛错**；
- 因此：**只要 decode 请求成功返回且没有异常，就说明算子被调用了**；
- 再配合 `/metrics` 能查到 `vllm:prefix_cache_*`，说明 sparse offload 路径整体生效。

如果启动日志出现 `Sparse KV offload offload.initialize failed`、`packed SFA main caches`、`requires TP without context parallelism`，直接查第 9、10 节。

---

## 7. 阶段 4：执行 A/B 测试

### 7.1 单轮流程（每轮都重复）

```
1) 停服务 → 确认显存释放
2) 换目标算子包（base / opt）
3) 新终端 source set_env.bash → 启动服务 → 等 /health 通过
4) 跑压测命令
5) 保存 aisbench_result.csv / aisbench.log，命名带轮次和构建
```

### 7.2 压测命令

```bash
source ~/aisvenv/bin/activate
cd ~/aisbench_auto_tools_prefix
unset http_proxy https_proxy

python3 aisbench_test.py \
  --input_len 32768 \
  --output_len 1024 \
  --data_num 32 \
  --concurrency 8 \
  --request_rate 0 \
  --npu_num 8 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test \
  --seed 2
```

**场景选择（8 × 64GB + GLM-5.2-w4a8 / GLM-5.1-w4a8）**：

| 场景 | 命令参数 | 说明 |
|---|---|---|
| 首选 | `--input_len 32768 --data_num 32` | 显存最稳，先出 A/B 结论 |
| 加长 | `--input_len 65536 --data_num 16` | KV 约为首选 1.8 倍，需确认不 OOM |
| 128K | `--input_len 131072 --data_num 32` | **8 × 64GB 放不下**（见 10.2），不要照搬同事的 16 卡命令 |

> 同事原命令的 `--input_len 131072` 是按 16 卡 / 128GB 卡环境给的，直接搬到 8 × 64GB 会 OOM 或命中率骤降。

说明：

- `--npu_num 8`：本机 8 卡（同事原命令的 16 是按他的机器算单卡吞吐）；
- `--prefix_test`：先把前缀预热一遍，保证全量测试时命中率接近 90%；
- `--repeat` 对 prefix_cache 分支**不生效**，重复靠手动多跑几轮；
- 数据集由 `seed=2` 生成在 `DATASET_PATH`，**几轮之间不要删除、不要换 seed**，保证数据完全一致。

### 7.3 保存结果

```bash
# 每轮跑完立即执行（CSV 是追加写，不搬走会混在一起）
mv aisbench_result.csv "$AB/results/base_r1.csv"   # 按实际构建/轮次改名
cp aisbench.log          "$AB/results/base_r1.log"
```

### 7.4 轮次安排（ABBA，抵消机器漂移）

| 顺序 | 构建 | 结果文件 |
|---|---|---|
| 1 | base | `base_r1.csv` |
| 2 | opt | `opt_r1.csv` |
| 3 | opt | `opt_r2.csv` |
| 4 | base | `base_r2.csv` |
| 5 | base | `base_r3.csv` |
| 6 | opt | `opt_r3.csv` |

每轮之间建议间隔 1~2 分钟，让 NPU 温度和频率稳定。

### 7.5 结果记录表

| 轮次 | 构建 | TPOT avg(ms) | TPOT P90(ms) | output_throughput | single_output_throughput | E2E_time(s) | 命中率 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 1 | base | | | | | | | |
| 2 | opt | | | | | | | |
| 3 | opt | | | | | | | |
| 4 | base | | | | | | | |
| 5 | base | | | | | | | |
| 6 | opt | | | | | | | |
| **中位数** | base | | | | | | | |
| **中位数** | opt | | | | | | | |
| **变化** | opt vs base | | | | | | | |

快速取中位数（在 `~/fsao_ab/results` 下）：

```bash
python3 - <<'PY'
import pandas as pd, glob
for tag in ("base", "opt"):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{tag}_r*.csv")])
    print(tag)
    print(df[["TPOT avg", "TPOT SLO_P90", "output_throughput",
              "single_output_throughput", "E2E_time", "repeat_rate"]].median(numeric_only=True))
PY
```

---

## 8. 结果判读

核心指标：

| 指标 | 期望方向 | 说明 |
|---|---|---|
| `TPOT avg` / `TPOT SLO_P90` | **下降** | 算子每个 decode step 都执行，是最直接的表征 |
| `output_throughput` | **上升** | 端到端吞吐 |
| `single_output_throughput` | **上升** | 单卡吞吐（除以 8） |
| `E2E_time` | 下降 | 整轮耗时 |
| `TTFT avg` | 基本不变 | 算子在 decode，prefill 影响很小 |
| 打屏命中率 | ≈ 90% | 明显偏低说明 KV 容量不足，结果不可信 |

预期量级：单算子优化在特定路径上收益 10%~40%，但整网 decode 里算子只占一部分，**E2E 的 TPOT 改善通常是个位数百分比**。因此：

- 必须 3 轮取中位数，单轮噪声可能有 1~2%；
- 如果差异接近噪声，补充一个“算子占比更高”的场景：`--repeat_rate 0.5 --concurrency 16 --data_num 64`（同时把 `--max-num-seqs` 调到 16）；
- 结论里同时给出单算子 A/B 数据（`artifacts/fsao_hit_ratio`）做交叉印证。

---

## 9. 故障排查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `Sparse KV offload offload.initialize failed` | host 连续内存不足 | `free -g` 确认；把 `dram_size_per_dp_GB` 降到 48~64；仍失败按 memfabric 前置条件文档处理 |
| `does not support packed SFA main caches (C8 or TQ4)` | 用了 c8 权重或 `--kv-cache-dtype int8/fp8` | 换非 c8 权重，删掉 `--kv-cache-dtype` |
| `Sparse KV offload currently requires TP without context parallelism` | 打开了 `enable_dsa_cp` 或 PCP/DCP | 从 `additional-config` 删掉 `enable_dsa_cp`，CP 参数保持 1 |
| `Sparse KV offload don't support compress now` | 权重 config 有 `compress_ratios` | 换 GLM-5.2 / GLM-5.1 的标准 DSA 权重 |
| `nodename nor servname provided` / 找不到算子 | `_cann_ops_custom` 没换或没重启 | 重新换目录、重启进程、开新终端 |
| `/metrics` 查不到 `prefix_cache` | 代理未关 / 多进程指标目录未设 | `unset http_proxy https_proxy`；设置 `PROMETHEUS_MULTIPROC_DIR` |
| `unrecognized arguments: --num-warmups` | aisbench 版本差异 | 按工具 README FAQ 3 删除该参数 |
| 启动 OOM | 128K + 8 并发 + 图模式内存不够 | 降 `--max-model-len`（先 65536 验证流程）、降 `--max-num-seqs`、或去掉 `--compilation-config` |
| 命中率远低于 90% | host pool 太小导致 block 被换出 | 提高 `dram_size_per_dp_GB`（host 内存允许时）或降低并发 |
| 图模式启动失败 | 该路径图捕获有问题 | 加 `--enforce-eager` 先跑通（A/B 两端保持一致即可） |

---

## 10. 权重与模型专属问题（GLM-5.1 / 5.2）

### 10.1 GLM-5.2-w8a8 可以吗？

**只在 128GB 卡上可以。** 它是官方文档列出的 **1 台 Atlas 800 A3（128GB × 8）** 部署配置；`npu-smi` 显示 `65536` 说明是 64GB 卡，权重本身就放不下，不能用。

如果能换到 128GB × 8 机器，注意：

- **w8a8 是权重量化，不影响 KV cache dtype**；不加 `--kv-cache-dtype` 主 KV 就是 BF16，满足 sparse offload；
- 不能用 `GLM-5.2-w8a8c8` / `w4a8c8`（C8 主 KV 会被拒）；
- 团队 GLM-5.2 命令默认带 `enable_dsa_cp`，打开 offload 时必须删掉。

### 10.2 本机是 64GB 卡（8 × 64GB = 512GB）怎么办

按优先级选权重：

1. **`GLM-5.2-w4a8`（首选，你已有）**：w4 权重比 w8 小，512GB 有机会放下；但官方部署表没有单独列它，属于“官方支持的模型 + 未单列的量化版本”，必须先用 32K 场景实测显存；
2. **`GLM-5.1-w4a8`（兜底）**：官方明确支持 64GB × 8 单机部署，最稳；
3. `GLM-5.2-w8a8`：需要 128GB × 8，本机放不下。

无论用哪个，都必须确认是**非 c8** 且主 KV 为 BF16（见 10.3）。

> 如果 5.2-w4a8 启动 OOM：先降 `--max-num-seqs` / `--max-model-len`，不行就换 5.1-w4a8，启动参数不用改，只改 `MODEL_PATH` 和 `--served-model-name`。

同时必须**缩小测试场景**，不能照搬同事的 128K / 16 卡命令。原因：

- 用 `keep_device_kv_cache=true` 时，device 上仍保留完整 paged KV，且 MLA 的 KV 在 TP rank 之间是**复制**的，每张卡都要放一份；
- 64GB 卡要同时装下：权重（w4a8 约 20~30GB/卡）+ KV（每卡一份）+ 激活 / 图内存（约 10GB）。

显存预算参考（按 MLA 每 token 约 70~90KB 估算；GLM-5.2 层数可能与 5.1 不同，以实测 `npu-smi info` 为准）：

| 场景 | 唯一 token 数（90% 前缀共享） | 每卡 KV | 结论 |
|---|---|---|---|
| 32K × 32 请求 | ≈ 13 万 | ≈ 10GB | ✅ 首选 |
| 64K × 16 请求 | ≈ 16 万 | ≈ 13GB | ✅ 可以试 |
| 64K × 32 请求 | ≈ 27 万 | ≈ 22GB | ⚠️ 偏紧，可能触发换出 |
| 128K × 32 请求 | ≈ 53 万 | ≈ 45GB | ❌ 放不下 |

对应的启动参数已按此校准（见 6.2）：`--max-model-len 40960`、`--max-num-batched-tokens 4096`、`--gpu-memory-utilization 0.90`、`dram_size_per_dp_GB 48`。

其他选择：

- 借 16 卡 / 128GB 卡环境，按同事的原命令跑 128K（见附录 C）；
- 不建议把 `GLM-5.2-w4a8c8` 强行降到 BF16 KV 来省显存——C8 与 sparse offload 互斥，且这是未验证组合。

### 10.3 w4a8 与 w4a8c8 的区别（重要）

官方文档的部署表没有单独列 `GLM-5.2-w4a8`，但你手里有；它和 `w4a8c8` 的唯一区别是**主 KV cache 是否量化（c8）**：

| 权重名 | 含义 | 能否用于本次 E2E |
|---|---|---|
| `GLM-5.2-w8a8` | W8A8 权重，主 KV 由 `--kv-cache-dtype` 决定（默认 BF16） | ✅ 但仅限 128GB × 8 |
| `GLM-5.2-w4a8` | W4A8 权重，主 KV 默认 BF16 | ✅ 64GB × 8 首选，需实测显存 |
| `GLM-5.2-w4a8c8` | W4A8 权重 + **C8 主 KV cache** | ❌ C8 与 sparse offload 互斥 |
| `GLM-5.1-w4a8` | W4A8 权重，主 KV 默认 BF16 | ✅ 最稳兜底（官方 64GB × 8） |

名字里的 `c8` 指的是 **KV cache 量化**，不是权重。C8 模型的 checkpoint 里带 `kv_cache_scale`，运行时需要 `--kv-cache-dtype int8`；而一旦 cache dtype 是 int8/fp8：

```python
# vllm_ascend/ascend_config.py:928
enable_sparse_sfa_c8 = cache_config.cache_dtype in ["fp8", "int8"] and use_sparse_sfa
```

sparse offload 就会直接拒绝（`ascend_config.py:1015`）：

```text
Sparse KV offload does not support packed SFA main caches (C8 or TQ4).
```

> 理论上也可以对 c8 权重强行不传 `--kv-cache-dtype`，让主 KV 退回 BF16，但这是**未验证的 off-label 组合**（checkpoint 里的 KV scale 权重会变成无用参数），不建议用在正式 A/B 结论里。

**判断方法**：直接看权重目录名，以及：

```bash
python3 - <<'PY'
import json, os
p = os.path.join(os.environ["MODEL_PATH"], "quant_model_description.json")
print("存在 kv_cache_type=C8:", "kv_cache_type" in open(p).read() and "C8" in open(p).read())
PY
```

### 10.4 权重自检

```bash
grep -oE '"(index_topk|compress_ratios|index_kpool)"' "$MODEL_PATH/config.json" | sort -u
# 期望：只有 "index_topk"
```

### 10.5 可以保留 / 必须去掉的参数

| 参数 | 结论 |
|---|---|
| `--attention_config.indexer_kv_dtype int8` | 允许（indexer KV 留在 device），但第一次建议不加，减少变量 |
| `--kv-cache-dtype int8` / `fp8` | **禁止** |
| `enable_dsa_cp` | **禁止** |
| `enable_fused_mc2` / `enable_balance_scheduling` / `multistream_overlap_shared_expert` / `enable_flashcomm1` | 与 offload 无关，按团队原样保留，A/B 两端一致即可 |
| `--speculative-config`（MTP） | 支持，但第一次先去掉；跑通后再加回来重测 |
| `--compilation-config FULL_DECODE_ONLY` | 支持；失败时退回 `--enforce-eager` |

---

## 附录 A：单算子测试复核

对每个算子包都要验证一次（先切包，再跑）：

```bash
source ~/fsao_ab/env.sh
source "$VLLM_SRC/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

cd "$VLLM_SRC"
python3 - <<'PY'
import torch, vllm_ascend
from vllm_ascend.utils import enable_custom_op
print("vllm_ascend 路径:", vllm_ascend.__file__)
enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
print("算子加载成功")
PY

VLLM_USE_MODELSCOPE=true pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
```

期望：`4 passed`。base 和 opt 两个包都要通过。

---

## 附录 B：Python 扩展编译（若未做过）

如果你已经能跑通单算子测试，本节可跳过。完整步骤见仓库里的
`FUSED_SPARSE_ATTENTION_OVERLAP_ONLY_GUIDE.md` 第 7~10 步，摘要：

```bash
cd "$VLLM_SRC"
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

## 附录 C：借到 16 卡时的 PD 分离方案

8 卡做不了生产的 PD 分离。如果能借到 16 卡机器，按官方文档部署：

- Prefill（8 卡）：
  ```bash
  --kv-transfer-config '{
    "kv_connector": "SfaRemoteD2HConnector",
    "kv_role": "kv_producer"
  }'
  ```
  不设置 `sparse_kv_offload_config`。

- Decode（8 卡）：
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

- 启动代理：
  ```bash
  python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
    --host 127.0.0.1 --port 9000 \
    --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
    --decoder-hosts 127.0.0.1 --decoder-ports 8200
  ```
- 压测客户端 `HOST_PORT` 改成代理端口 `9000`，`POD_INFO` 填 Decode 节点地址。

参考：`docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md`

---

## 附录 D：清理与回滚

```bash
# 停服务
pkill -f "vllm serve" || true

# 删除基线 worktree
cd "$VLLM_SRC"
git worktree remove ~/vllm-ascend-base --force

# 可选：删除编译产物和临时包
rm -rf "$AB/ops" "$AB/results"
```

如需恢复仓库原状（只删本手册产生的文件，不改代码）：

```bash
cd "$VLLM_SRC"
git status
```

---

## 附：本次测试的判断标准（写报告用）

1. **功能正确性**：两个算子包单算子测试均 `4 passed`；整网请求输出正常无异常。
2. **是否命中目标路径**：`use_fused_overlap=true` + decode 正常返回 + `/metrics` 有 `prefix_cache`。
3. **性能收益**：opt 相对 base，3 轮中位数 `TPOT avg` 下降、`output_throughput` 上升，且命中率两端一致（≈90%）。
4. **证据边界**：单机 colocate 调试模式的相对收益，不等同于生产 PD 分离的绝对值。
