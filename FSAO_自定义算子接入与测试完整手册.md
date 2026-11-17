# FusedSparseAttentionOverlap 自定义算子接入与测试完整手册（命令级）

> 适用：Atlas A2（`ascend910b`）、A3（`ascend910_93`）、A5/950（`ascend950`）通用
> 文中的 NPU 编号、TP、端口、内存数值均为示例，按实际机器替换
> 模型：GLM-5.1-w4a8（约 391GB，`index_topk=2048`，非 C8）
> 原则：**所有命令逐行给出，不依赖任何一键脚本**；每一步都说明"做什么、看什么、失败怎么办"。

---

## 目录

1. 术语与前提
2. 编译与安装自定义算子包
3. 把算子接入模型（触发条件与服务启动）
4. 单机 PD 分离部署（示例 P/D 各 8 逻辑 NPU，A2/A3/A5 通用）
5. 压力测试（aisbench）与 A/B
6. 指标解读
7. Profiling（整网 + 算子级）
8. 附录：端口、环境变量、排错、参考值

---

## 1. 术语与前提

| 名称 | 含义 |
|---|---|
| vendor 包 | `vllm_ascend/_cann_ops_custom/vendors/custom_transformer`，包含全部自定义算子的 kernel/tiling/aclnn 接口 |
| 目标算子 | `fused_sparse_attention_overlap`，torch 入口 `torch.ops._C_ascend.npu_fused_sparse_attention_overlap` |
| 触发条件 | `sparse_kv_offload_config.enabled=true` **且** `use_fused_overlap=true`（decode 阶段） |
| 算子级测试 | 直接调用 torch op，验证精度与 kernel 时延（不经过模型） |
| 整网测试 | 起 vLLM 服务，用压测客户端测 TTFT/TPOT/吞吐 |

版本必须对齐：CANN、torch、torch_npu、vLLM（见仓库 `.github/vllm-main-verified.commit`）、vllm-ascend、transformers。换硬件/换版本后重新确认。

SOC 名称：

| 平台 | `build.sh --soc`（家族名） | `SOC_VERSION`（具体型号，setup.py 用） |
|---|---|---|
| A2 | `ascend910b` | 以 CANN 实际型号为准（如 `ascend910b4`） |
| A3 | `ascend910_93` | `ascend910_9382` |
| A5/950 | `ascend950` | 以 CANN 实际型号为准（如 `ascend950pr` / `ascend950dt`） |

单机 TP 一般等于**逻辑 NPU 数**：A2 常见 8，A3 常见 16（8 卡 × 2 chip），A5 以 `npu-smi info -l` 为准。
PD 分离时每侧的 TP 等于该侧分到的逻辑 NPU 数。

---

## 2. 编译与安装自定义算子包

### 2.1 环境变量

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export SOC_VERSION=<具体型号>   # 例：A3 ascend910_9382；A2/A5 以 CANN 实际型号为准
export MAX_JOBS=8
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/devlib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

确认：

```bash
npu-smi info -l        # 逻辑 NPU 数（A3：8 卡 × 2 chip = 16）
echo "$SOC_VERSION"
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
```

### 2.2 编译方式

**方式 A：全量算子（跑模型必须用这个）**

```bash
cd <repo>
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"
```

**方式 B：只编目标算子（仅算子级测试）**

```bash
cd <repo>/csrc
bash build.sh --pkg --ops=fused_sparse_attention_overlap --soc=<家族名>
# 家族名：A2 ascend910b；A3 ascend910_93；A5 ascend950
```

产物：

```bash
ls -l csrc/build/*.run
```

> 注意：只编单算子得到的包不能用于整网（模型会加载其它自定义算子）。

### 2.3 安装 vendor 包

```bash
cd <repo>
chmod +x csrc/build/cann-ops-transformer*.run
csrc/build/cann-ops-transformer*.run --install-path="$PWD/vllm_ascend/_cann_ops_custom"
```

检查：

```bash
ls vllm_ascend/_cann_ops_custom/vendors/custom_transformer/
```

### 2.4 安装 Python 包与扩展

全量编译通常已经带上扩展；若只编了算子包，需要：

```bash
cd <repo>
COMPILE_CUSTOM_KERNELS=0 python3 -m pip install -e . --no-build-isolation
```

如果 `vllm_ascend_C*.so` 缺失，手动编译：

```bash
cd <repo>
PYBIND11_CMAKE_DIR="$(python3 -m pybind11 --cmakedir)"
PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
TORCH_NPU_PATH="$(python3 -c 'import os, torch_npu; print(os.path.dirname(torch_npu.__file__))')"
EXT_BUILD_DIR="$PWD/build/fsao_ext"

cmake -S "$PWD" -B "$EXT_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DPYTHON_INCLUDE_PATH="$PYTHON_INCLUDE_PATH" \
  -DCMAKE_INSTALL_PREFIX="$PWD/vllm_ascend" \
  -DCMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR" \
  -DTORCH_NPU_PATH="$TORCH_NPU_PATH" \
  -DSOC_VERSION="$SOC_VERSION"
cmake --build "$EXT_BUILD_DIR" --target vllm_ascend_C -j"$MAX_JOBS"
cmake --install "$EXT_BUILD_DIR"
```

### 2.5 加载与验证

```bash
cd <repo>
source "$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python3 - <<'PY'
import torch
import vllm_ascend
from vllm_ascend.utils import enable_custom_op
print("vllm_ascend:", vllm_ascend.__file__)
enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
print("op loaded")
PY
```

### 2.6 包溯源（A/B 必做）

```bash
cd <repo>
find vllm_ascend/_cann_ops_custom -type f -path '*fused_sparse_attention_overlap*' \
  | sort | xargs md5sum
```

两套包必须 md5 不同；和对应 commit 的源码头文件 md5 对得上。

---

## 3. 把算子接入模型

### 3.1 触发条件（代码事实）

算子只在 **decode 的 sparse KV offload 路径**被调用，需要同时满足：

1. `sparse_kv_offload_config.enabled = true`
2. `use_fused_overlap = true`
3. 主 KV 为 **BF16**（不要传 `--kv-cache-dtype int8/fp8`；C8/TQ4 打包 KV 会被拒）
4. 模型 config 有 `index_topk`，且没有 `compress_ratios` / `index_kpool`
5. 不支持 CP/PP
6. 部署形态：
   - PD decode 节点（生产）：`keep_device_kv_cache=false`（默认）
   - 单机 colocate 调试：`keep_device_kv_cache=true`（会占满 device KV，仅调试）

### 3.2 单机 colocate 调试起服务（以 A3 TP16 为例；A2/A5 换成对应逻辑 NPU 数与 TP）

```bash
# 逻辑 NPU 列表按平台替换：A2 常见 0-7，A3 0-15，A5 以 npu-smi 为准
export ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 15)
export HCCL_IF_IP=<本机业务IP>
export HCCL_SOCKET_IFNAME=<业务网卡名>
export GLOO_SOCKET_IFNAME=<业务网卡名>
export TP_SOCKET_IFNAME=<业务网卡名>
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
unset HCCL_INTRA_PCIE_ENABLE HCCL_INTRA_ROCE_ENABLE
unset http_proxy https_proxy

export PYTHONPATH=<repo>:${PYTHONPATH:-}
export ASCEND_CUSTOM_OPP_PATH=<repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=<repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}

vllm serve /mnt/weight/GLM-5.1-w4a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm-5.1 \
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --additional-config '{"enable_balance_scheduling":true,"multistream_overlap_shared_expert":true,"sparse_kv_offload_config":{"enabled":true,"use_fused_overlap":true,"keep_device_kv_cache":true,"dram_size_per_dp_GB":64}}'
```

### 3.3 验证算子真的在路径上

```bash
# 1) 服务就绪
curl -s http://127.0.0.1:8000/health

# 2) offload 初始化日志
grep -i "Sparse KV offload reserved" <server.log>

# 3) 发一条 decode
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'

# 4) 指标侧
curl -s http://127.0.0.1:8000/metrics | grep prefix_cache

# 5) 对照实验（可选）：use_fused_overlap=false 走 Copy-SFA，不是优化 kernel
```

若算子缺失，`_require_custom_op("npu_fused_sparse_attention_overlap")` 会在首个 decode 直接抛错。

### 3.4 常见启动失败

| 报错 | 原因 | 处理 |
|---|---|---|
| `does not support packed SFA main caches` | 传了 `--kv-cache-dtype int8/fp8` 或用了 C8 权重 | 去掉 kv-cache-dtype，换非 C8 权重 |
| `Sparse KV offload offload.initialize failed` | host 连续内存不足（MemFabric） | 降 `dram_size_per_dp_GB`/DP，或按 memfabric 文档准备大页 |
| `assert ... PD disaggregate ... D node` | 开了 offload 但不是 PD consumer | PD 下用 consumer；colocate 调试开 `keep_device_kv_cache=true` |
| `Sparse KV offload don't support compress now` | 权重有 `compress_ratios` | 换标准 DSA 权重 |
| `requires TP without context parallelism` | 开了 CP | 关闭 CP/PCP/DCP |

---

## 4. 单机 PD 分离部署（示例 P/D 各 8 逻辑 NPU，A2/A3/A5 通用）

### 4.1 拓扑

| 角色 | 逻辑 NPU | TP/EP | HTTP 端口 | KV 端口 |
|---|---|---|---|---|
| P（prefill） | 0–7 | TP8 / EP8 | 8100 | 20000（+7） |
| D（decode） | 8–15 | TP8 / EP8 | 8200 | 20050（+7） |
| proxy | — | — | 9000 | — |

内存预算（通用公式，A2/A3/A5 同法计算）：

- 每 NPU 权重 ≈ 权重总量 / TP。例：391GB / 8 ≈ 49GB/NPU；391GB / 16 ≈ 24.4GB/NPU。
- 单 chip HBM 以 `npu-smi info` 为准（A2/A3 常见 64GB；A5 以实际为准）。
- 剩余 HBM ≈ HBM × `gpu-memory-utilization` − 每 NPU 权重，留给 KV/激活。
- 每条 KV ≈ `(kv_lora_rank + qk_rope_head_dim) × dtype_bytes × 层数 × 序列长度`。
  对 GLM-5.1 的 MLA 粗算：64K ≈ 6.6GB/条，32K ≈ 3.3GB/条。
- 例：TP8 / 64GB / 0.95 → 60.8 − 49 ≈ 11.8GB → 64K 只能 `max-num-seqs=1`；32K 可 2；4K 不受限。

decode 侧的 KV 在 **host DRAM**：`dram_size_per_dp_GB`（示例 110）是**每个 DP rank** 的 host 池，
DP=1 时就是 110GB；TP rank 共享该池。先 `free -g` 确认 host 有足够**连续**内存，
不足时降低该值（容量随之下降），否则会报 `Sparse KV offload offload.initialize failed`。

### 4.2 连接器选择

| 方案 | Prefill | Decode | 是否要 Memcache |
|---|---|---|---|
| A（最小） | `SfaRemoteD2HConnector` producer | `SfaRemoteD2HConnector` consumer | 否 |
| B（文档完整组合） | `MultiConnector`（SfaRemoteD2H + AscendStore/memcache） | 同上 | 是 |

先跑 A；连不通再上 B。

### 4.3 起 Prefill（逐行）

```bash
# P 侧逻辑 NPU（示例 8 个；按机器改，且与 D 侧不相交）
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP=<本机业务IP>
export HCCL_SOCKET_IFNAME=<业务网卡名>
export GLOO_SOCKET_IFNAME=<业务网卡名>
export TP_SOCKET_IFNAME=<业务网卡名>
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
unset http_proxy https_proxy

export PYTHONPATH=<repo>:${PYTHONPATH:-}
export ASCEND_CUSTOM_OPP_PATH=<repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=<repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}

nohup vllm serve /mnt/weight/GLM-5.1-w4a8 \
  --host 0.0.0.0 --port 8100 --served-model-name glm-5.1 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 34816 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.93 \
  --trust-remote-code --quantization ascend \
  --no-enable-prefix-caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --additional-config '{"multistream_overlap_shared_expert":true}' \
  --kv-transfer-config '{"kv_connector":"SfaRemoteD2HConnector","kv_role":"kv_producer","kv_port":"20000","kv_connector_extra_config":{"memfabric_transfer_protocol":"sdma"}}' \
  > /tmp/prefill.log 2>&1 &
```

等待就绪（每 10 秒探一次）：

```bash
until curl -fsS http://127.0.0.1:8100/v1/models >/dev/null 2>&1; do sleep 10; done
echo "prefill ready"
```

64K 时把 `--max-model-len 67840 --max-num-seqs 1`。日志里的 `tensor_parallel_size` 必须等于该侧 NPU 数（示例 8）。

### 4.4 起 Decode（逐行）

```bash
# D 侧逻辑 NPU（示例 8 个；按机器改，且与 P 侧不相交）
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export HCCL_IF_IP=<本机业务IP>
export HCCL_SOCKET_IFNAME=<业务网卡名>
export GLOO_SOCKET_IFNAME=<业务网卡名>
export TP_SOCKET_IFNAME=<业务网卡名>
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
unset http_proxy https_proxy

export PYTHONPATH=<repo>:${PYTHONPATH:-}
export ASCEND_CUSTOM_OPP_PATH=<repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=<repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}

nohup vllm serve /mnt/weight/GLM-5.1-w4a8 \
  --host 0.0.0.0 --port 8200 --served-model-name glm-5.1 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 34816 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.93 \
  --trust-remote-code --quantization ascend \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --async-scheduling \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --additional-config '{"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"sparse_kv_offload_config":{"enabled":true,"use_fused_overlap":true,"dram_size_per_dp_GB":110}}' \
  --kv-transfer-config '{"kv_connector":"SfaRemoteD2HConnector","kv_role":"kv_consumer","kv_port":"20050","kv_connector_extra_config":{"memfabric_transfer_protocol":"sdma"}}' \
  > /tmp/decode.log 2>&1 &
```

等待并检查：

```bash
until curl -fsS http://127.0.0.1:8200/v1/models >/dev/null 2>&1; do sleep 10; done
grep -iE "sparse KV offload|topk buffers|fused_overlap|initialize failed" /tmp/decode.log | head
```

关键点：
- 主 KV 必须 BF16：**不要**传 `--kv-cache-dtype`；
- **不要**开 `keep_device_kv_cache`；
- `--async-scheduling` 若导致启动失败，先去掉；
- 64K 时 `--max-model-len 67840 --max-num-seqs 1`。

### 4.5 起 Proxy（逐行）

```bash
unset http_proxy https_proxy
cd <repo>
nohup python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host 127.0.0.1 \
  --port 9000 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8200 \
  > /tmp/proxy.log 2>&1 &
```

proxy **不实现** `/v1/models`（返回 404 是正常的），用真实请求验证：

```bash
curl -s -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

返回 `system_fingerprint` 里带 `tp8-ep` 即链路正常。

### 4.6 停止

```bash
pkill -f load_balance_proxy || true
pkill -f "vllm serve" || true
pkill -f EngineCore || true
sleep 5
ss -ltnp | grep -E '8100|8200|9000' || true
```

---

## 5. 压力测试（aisbench）与 A/B

### 5.1 安装与配置

```bash
python3 -m venv ~/aisvenv
source ~/aisvenv/bin/activate
python3 -m pip install -U pip
python3 -m pip install ais-bench-benchmark
# 压测工具（含 prefix_cache 数据集与 --prefix_test）
cd ~
git clone https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git
cd aisbench_auto_tools_prefix
```

编辑 `config.py`：

```python
MODEL_PATH = "/mnt/weight/GLM-5.1-w4a8"   # 只用于 tokenizer/数据集生成
MODEL_NAME = "glm-5.1"                    # 必须等于 --served-model-name
HOST_PORT  = "8000"                       # 单机服务；PD 时填 proxy 端口 9000
```

### 5.2 参数含义

| 参数 | 含义 |
|---|---|
| `--input_len` | 每条请求 prompt 长度（token） |
| `--output_len` | 每条请求生成 token 数（建议 256/512，别用 64） |
| `--data_num` | 总请求数 |
| `--concurrency` | 客户端并发 |
| `--request_rate` | 0 = 不额外限速（闭环并发） |
| `--npu_num` | 只用于 `single_output_throughput` 归一化 |
| `--dataset_type prefix_cache` | 前缀缓存数据集 |
| `--repeat_rate` | 请求间共享前缀比例（0=独立，0.5=一半，1.0=全共享） |
| `--prefix_test` | 先用一条请求预热前缀；`repeat_rate=0` 时不要加 |
| `--seed` | 固定数据集 |

### 5.3 三种标准场景（逐行）

**A. 纯 decode 基线（bs=1，预填一次后无干扰）**

```bash
source ~/aisvenv/bin/activate
cd ~/aisbench_auto_tools_prefix
unset http_proxy https_proxy
rm -f aisbench_result.csv aisbench.log

python3 aisbench_test.py \
  --input_len 65536 \
  --output_len 256 \
  --data_num 1 \
  --concurrency 1 \
  --request_rate 0 \
  --npu_num 16 \
  --dataset_type prefix_cache \
  --repeat_rate 0 \
  --seed 2
```

**B. 高命中多请求（prefill≈0，干净 decode）**

```bash
python3 aisbench_test.py \
  --input_len 65536 \
  --output_len 256 \
  --data_num 8 \
  --concurrency 8 \
  --request_rate 0 \
  --npu_num 16 \
  --dataset_type prefix_cache \
  --repeat_rate 0.99 \
  --prefix_test \
  --seed 2
```

**C. 目标混合场景（32K / cc8 / 32 请求 / 50% 命中）**

```bash
python3 aisbench_test.py \
  --input_len 32768 \
  --output_len 1024 \
  --data_num 32 \
  --concurrency 8 \
  --request_rate 0 \
  --npu_num 16 \
  --dataset_type prefix_cache \
  --repeat_rate 0.5 \
  --prefix_test \
  --seed 2
```

> 场景 C 的 TPOT 会被"并发 prefill"污染（见第 6 节），只用于相对 A/B；绝对 decode 能力看 A / B / PD。

### 5.4 结果保存

```bash
mkdir -p ~/fsao_ab/results
cp aisbench_result.csv ~/fsao_ab/results/<arm>_r<N>.csv
cp aisbench.log        ~/fsao_ab/results/<arm>_r<N>.log

{
  echo "arm=<base|opt> round=<N> time=$(date -Is)"
  echo "repo_head=$(git -C <repo> rev-parse HEAD)"
  echo "input_len=<...> output_len=<...> data_num=<...> concurrency=<...> repeat_rate=<...>"
  md5sum <repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/*/fused_sparse_attention_overlap/*.o
  cat aisbench_result.csv
} > ~/fsao_ab/results/<arm>_r<N>_meta.txt
```

逐请求结果在：`~/aisbench_auto_tools_prefix/outputs/default/<时间戳>/performances/vllm-api-stream-chat/`。

### 5.5 换包 A/B（逐行，不用脚本）

准备两套完整 vendor 包：

```bash
# base
cd <repo-base-tree>
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"
rm -rf ~/fsao_ab/ops/base
cp -a vllm_ascend/_cann_ops_custom ~/fsao_ab/ops/base

# opt
cd <repo-opt-tree>
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"
rm -rf ~/fsao_ab/ops/opt
cp -a vllm_ascend/_cann_ops_custom ~/fsao_ab/ops/opt
```

每轮流程（重复 6 次，顺序 base → opt → opt → base → base → opt）：

```bash
# 1) 停服务
pkill -9 -f "vllm serve" || true
pkill -9 -f EngineCore || true
sleep 10

# 2) 换包
rm -rf <repo>/vllm_ascend/_cann_ops_custom
cp -a ~/fsao_ab/ops/<base|opt> <repo>/vllm_ascend/_cann_ops_custom

# 3) 确认包内容
find <repo>/vllm_ascend/_cann_ops_custom -type f -path '*fused_sparse_attention_overlap*' | sort | xargs md5sum

# 4) 起服务（第 3/4 节命令）
# 5) 压测（第 5.3 节命令）
# 6) 保存结果（第 5.4 节）
# 7) 停服务，间隔 60-120 秒再下一轮
```

> 算子只在进程启动时加载，**每轮必须重启引擎**；不需要重新编译（换的是预编译包）。

### 5.6 汇总判读

对每个 arm 取 3 轮中位数，计算：

```text
delta = (opt_median - base_median) / base_median × 100%
noise = max(base_max-base_min, opt_max-opt_min) / median × 100%
```

- `|delta| <= noise` → **within noise**，不能下结论；
- `|delta| > noise` → 真实差异；
- 同时检查两臂 TTFT、命中率、并发是否一致。

---

## 6. 指标解读

### 6.1 每个指标是什么

| 指标 | 定义 | 说明 |
|---|---|---|
| **TTFT** | 请求发出 → 首 token | 含排队 + prefill（PD 下还含 KV 传输） |
| **TPOT** | `(E2EL − TTFT) / (输出token数 − 1)`，逐请求后聚合 | decode 每 token 平均耗时；**不是模型能力的唯一指标** |
| **ITL** | 相邻 token 到达间隔（逐 token） | 用来看双峰/毛刺；与 TPOT 接近说明是稳态 decode |
| **E2EL** | 请求总时延 | `≈ TTFT + (N−1)×TPOT` |
| **output_throughput** | 总输出 token / benchmark 时长 | 含 TTFT；瞬时值 ≈ `并发 × 1000 / TPOT_ms` |
| **single_output_throughput** | 上者 / npu_num | 每 NPU 归一化 |
| **cc / max_cc** | 实际/配置并发 | cc 明显小于 max_cc = 有请求提前结束或排队 |
| **命中率** | prefix cache hit | PD 下在 P 侧统计；影响 TTFT，不直接进入 decode TPOT |
| **qps / qpm** | 请求吞吐 | — |

### 6.2 三步自洽校验（每次必做）

```text
① E2EL ≈ TTFT + (N−1) × TPOT
② 并发 × 1000 / TPOT_ms ≈ 瞬时 output throughput
③ TPOT 的 avg / min / median / P90 是否接近
```

- ①② 对不上 → 解析错误/抓错表；
- ③ 差距大 → 仍有 prefill 重叠或调度毛刺。

### 6.3 怎么识别"被污染的 TPOT"

典型特征：

| 现象 | 含义 |
|---|---|
| **TTFT 呈等差数列**（如 27 / 52 / 78 / 104 s） | chunked prefill 把多条长 prompt **串行**预填；先开始的请求 decode 全程与后来者的 prefill 重叠 |
| **TPOT min 远小于 median**（如 73ms vs 655ms） | 只有最后开始的请求是干净的；其余都被拉长 |
| **ITL 双峰**（77ms 与 1.5s） | 一波是纯 decode 步，一波是混入 4096-token prefill 的步 |
| **avg ≈ median 但都很高** | 稳态就慢（不是偶发毛刺） |

结论：**被污染的场景只能做相对 A/B，不能报绝对 TPOT。**

### 6.4 min / median / P90 怎么用

| 统计量 | 含义 | 用途 |
|---|---|---|
| **min**（请求级） | 污染最少那条请求的平均 TPOT | 干净 decode 的**下界**（偏乐观），单轮 sanity |
| **median** | 请求 TPOT 的中位数 | A/B 主判据（需无污染场景） |
| **P90 / P99** | 尾部 | SLO 判据 |
| **avg** | 均值 | 有长尾时会被拉高，和 median 一起看 |

### 6.5 参考量级

| 场景 | TPOT | 来源 |
|---|---|---|
| Qwen3-8B，10 并发，单机 | ~26ms | 仓库性能文档 |
| 视觉模型，10 prompts | mean 44ms / P99 125ms | 仓库性能文档 |
| MiniMax-M3，128K/1K，cc58，3 节点 PD + EAGLE3 | 27.3ms | 模型文档 |
| DeepSeek-V4 系列 PD | 按 TPOT 20ms / 50ms 两档设 SLO | CI 用例命名 |
| GLM-5.1-w4a8，单机 PD + offload，4K/bs8 | ~60ms | 本次实测（A3 单机） |
| 同模型 64K/bs1 colocate | ~50ms | 本次实测（A3 单机） |

### 6.6 抓错表的坑

aisbench 每个 `*_r*.csv` 里可能有**两行**：

1. `--prefix_test` 的预热行：`TPOT/TTFT = 99999`；
2. 正式测量行。

汇总必须**剔除 99999 行**，否则中位数会被完全污染（曾出现 median=50076ms、noise=199%）。

另外，日志里可能有两张性能表（预热表 + 正式表）：`ITL` 第一张可能是 `0.1ms`（单 token），要取**第二张**。

---

## 7. Profiling

### 7.1 整网 profiling（Ascend PyTorch Profiler）

**启动时加参数**（每个引擎独立目录；PD 下 P 和 D 都要加才能分别抓）：

```bash
vllm serve ... \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/<user>/prof/decode","torch_profiler_with_stack":false}'
```

**控制**（proxy 不转发这些端点，必须直接 curl 引擎端口）：

```bash
# 开始
curl -X POST http://127.0.0.1:8200/start_profile

# 发 1–2 条请求（经 proxy）
curl -s -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/one_request.json >/dev/null

# 停止
curl -X POST http://127.0.0.1:8200/stop_profile
```

**解析**：

```bash
ls /home/<user>/prof/decode/            # 每个 TP rank 一个 *_ascend_pt
python3 - <<'PY'
from torch_npu.profiler.profiler import analyse
analyse("/home/<user>/prof/decode/<host>_<pid>_<ts>_ascend_pt/")
PY
```

**看结果**（`ASCEND_PROFILER_OUTPUT/`）：

| 文件 | 看什么 |
|---|---|
| `kernel_details.csv` | `FusedSparseAttentionOverlap` 每层耗时；memcpy/SDMA/GVA；allreduce/all2all |
| `op_statistic.csv` | 算子总耗时占比 |
| `trace_view.json` | MindStudio Insight / chrome://tracing 看时间轴与层间空隙 |
| `api_statistic.csv` | API 层开销 |

注意：

- 8 个 TP rank 各写一个 `*_ascend_pt`，两臂对比取**同一个 rank**；
- 只抓 1–2 条、128–512 token；`torch_profiler_with_stack=false`；
- profiling 有开销，绝对时间会偏高，只看占比与两臂 kernel 差异。

### 7.2 框架级 profiling（MS Service Profiler）

```bash
cd /path/to/prof_dir
export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json
export PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml   # 可选
# 配置里 enable=1、prof_dir、acl_task_time、timelimit
sed -i 's/"enable":\s*0/"enable": 1/' ./ms_service_profiler_config.json
# 起服务、发请求
msserviceprofiler parse --input-path=./ --output-path output
```

产出：`chrome_tracing.json`、`profiler.db`、`request.csv`、`kvcache.csv`、`batch.csv`。

### 7.3 算子级性能/精度（per_test，原始命令）

**环境**：

```bash
cd <repo>
source per_test/env.sh "$PWD"
```

**精度门禁**：

```bash
cd <repo>
pytest -sv per_test/tests/test_fsao_precision.py
```

小内存版：

```bash
DSA_CPU_KV_BACKEND=swap_memory pytest -sv per_test/tests/test_fsao_precision.py
```

**kernel 计时**（单进程，直接调用 torch op）：

```bash
cd <repo>
export ASCEND_RT_VISIBLE_DEVICES=0
python3 per_test/bench_fuse_sparse_attention_overlap.py \
  --topk 2048 \
  --q-heads 16,4 \
  --batch-size 8 \
  --max-seq-len 65536 \
  --hit-ratio 0.0,0.5,0.8,1.0 \
  --cpu-kv-backend offload \
  --dram-size-gb 64 \
  --membership-width full \
  --warmup 10 \
  --iters 50
```

常用附加参数：

| 参数 | 作用 |
|---|---|
| `--use-graph` | 把算子捕获进 NPUGraph，计时 `replay()` |
| `--kernel-time N` | 用 profiler 抓 N 次调用的 **kernel duration**（而非 host 端到端） |
| `--profile-iters N --profile-dir <dir>` | 抓 N 次算子的 torch profile |
| `--plan-direct on/off` | 外部 plan 的直接寻址/块表寻址 |
| `--hit-layout prefix/scattered` | 命中布局 |
| `--repeat N` / `--input-sets N` | 进程内重复/多组输入 |
| `--check-plan` | 正确性自检（全 miss / 交替 plan 等） |

**多进程取中位数**：

```bash
cd <repo>
python3 per_test/sweep_median.py \
  --processes 3 \
  --shape \
  -- --topk 2048 --batch-size 8 --max-seq-len 65536 \
     --hit-ratio 0.0,0.5,0.8,1.0 --iters 50
```

说明：`--` 之后是**传给 bench 脚本的参数**（不要重复写 `python3 bench_*.py`）；
`sweep_median.py` 自己会按 `--bench` 找到 bench 脚本。`--devices 0,1,...` 才会并行绑卡，
并行会改变判据，先做串行/并行对照再用；对比实验建议保持串行。

> 算子级 bench 用 **device tensor** 建模，不含真实 host gather；它回答"kernel 本身多快"，不能直接乘层数当 E2E TPOT。

### 7.4 平台差异（A2/A3/A5）

- 编译：`--soc` 家族名 A2=`ascend910b`、A3=`ascend910_93`、A5=`ascend950`；
- per_test 的 `env.sh` 从 `npu-smi` 识别 SOC；识别不出具体型号时手动 `export SOC_VERSION=<具体型号>`；
- A5/950 走 arch35 实现（外部 plan）；A2/A3 走各自既有 kernel 路径（本算子源码提供 arch22 与 arch35，具体由 SOC/构建选择）。若精度用例里有"内部 planner"分支，A5 上需要跳过/改；
- 回归重点（各平台共用）：首次 miss、all-hit、hit/miss 交替、split-G 奇数起点、selection block 边界、成对拷贝（两个 hit 物理连续 / 两个 miss 跨步可表示）。

---

## 8. 附录

### 8.1 端口表

| 用途 | 端口 |
|---|---|
| P HTTP | 8100 |
| D HTTP | 8200 |
| proxy | 9000 |
| P KV | 20000–20007 |
| D KV | 20050–20057 |
| Memcache（B 方案） | 5000（meta）、6000（config store） |

### 8.2 关键环境变量

| 变量 | 作用 |
|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | 逻辑 NPU 绑定 |
| `ASCEND_CUSTOM_OPP_PATH` | vendor 算子包路径 |
| `LD_LIBRARY_PATH` | op_api/lib 等 |
| `PYTHONPATH` | 仓库前置，避免解析到别的 vllm-ascend |
| `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | HCCL 通信 |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 显存分配 |
| `SOC_VERSION` | 编译目标芯片 |
| `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` | offload 库路径（缺了会 initialize failed） |

### 8.3 排错速查

| 现象 | 先查 |
|---|---|
| 服务起不来 / rank 越界 | `ASCEND_RT_VISIBLE_DEVICES` 数量 vs TP |
| `tensor_parallel_size` 与预期不符 | 旧 shell 残留 `TP` 变量 |
| 算子没被调用 | `enabled` + `use_fused_overlap` + 日志 grep |
| offload 初始化失败 | host 连续内存 / `dram_size_per_dp_GB` |
| proxy 404 | layerwise proxy 不实现 `/v1/models`，用 POST 验证 |
| TPOT 几百 ms | 先看 TTFT 是否等差数列、TPOT min vs median、ITL 是否双峰 |
| A/B 没差异 | 先确认 delta > noise、两臂配置一致、包 md5 不同且实际加载 |

### 8.4 参考命令清单（复制即用）

```bash
# 编译（全量；SOC_VERSION 按平台替换，A3 例：ascend910_9382）
cd <repo> && bash csrc/build_aclnn.sh "$PWD" "<具体型号>"

# 安装 vendor 包
csrc/build/cann-ops-transformer*.run --install-path="$PWD/vllm_ascend/_cann_ops_custom"

# 验证算子
source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
python3 -c "import torch; from vllm_ascend.utils import enable_custom_op; enable_custom_op(); print(hasattr(torch.ops._C_ascend,'npu_fused_sparse_attention_overlap'))"

# 压测（单机 8000 / PD proxy 9000）
cd ~/aisbench_auto_tools_prefix
python3 aisbench_test.py --input_len 4096 --output_len 256 --data_num 8 --concurrency 8 \
  --request_rate 0 --npu_num 16 --dataset_type prefix_cache --repeat_rate 0 --seed 2

# profiler 控制（PD decode 8200）
curl -X POST http://127.0.0.1:8200/start_profile
curl -X POST http://127.0.0.1:8200/stop_profile

# 解析
python3 -c "from torch_npu.profiler.profiler import analyse; analyse('/path/to/xxx_ascend_pt/')"
```
