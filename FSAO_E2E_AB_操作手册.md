# FusedSparseAttentionOverlap 整网 E2E A/B 测试操作手册

> 适用环境：Atlas A3 单机（8 张物理卡 × 2 chip = **16 个逻辑 NPU**，每 chip 64GB，合计 **1TB**）
> - 推荐 **GLM-5.1-w4a8**（实测 391GB，TP16 后每 chip 约 24.5GB，最稳）
> - 可试 **GLM-5.2-w4a8**（755GB，TP16 后每 chip 约 47GB，余量较小）
> - 场景用 32K 起步，64K 视情况；128K 受每 chip KV 限制仍不建议
> 代码分支：`dsa-a3-opt-only`（基线 `96e222490`，优化 `HEAD`）
> 目标：验证 `csrc/attention/fused_sparse_attention_overlap` 的 arch22 kernel 优化在整网 decode 上的性能收益
> 编写日期：2026-11-05

---

## 目录

- [0. 一页纸执行清单](#0-一页纸执行清单)
- [1. 背景：为什么不能直接起服务](#1-背景为什么不能直接起服务)
- [2. 变量约定（先填好）](#2-变量约定先填好)
- [3. 阶段 0：环境盘点](#3-阶段-0环境盘点)
- [4. 阶段 1：完整安装与两套完整算子包](#4-阶段-1完整安装与两套完整算子包)
- [5. 阶段 2：安装压测客户端](#5-阶段-2安装压测客户端)
- [6. 阶段 3：启动服务并确认算子生效](#6-阶段-3启动服务并确认算子生效)
- [7. 阶段 4：执行 A/B 测试](#7-阶段-4执行-ab-测试)
- [8. 结果判读](#8-结果判读)
- [9. 故障排查表](#9-故障排查表)
- [10. 权重与模型专属问题](#10-权重与模型专属问题glm-51--52)
- [附录 A：单算子测试复核](#附录-a单算子测试复核)
- [附录 B：Python 扩展编译（备用方案）](#附录-bpython-扩展编译备用方案)
- [附录 C：16 逻辑 NPU 的 PD 分离方案](#附录-c16-逻辑-npu-的-pd-分离方案)
- [附录 D：清理与回滚](#附录-d清理与回滚)
- [附录 E：HCCL 最小自检（可选）](#附录-ehccl-最小自检可选)

---

## 0. 一页纸执行清单

全部在**同一个 docker 容器内**执行。

- [ ] 1. `npu-smi info -l` 确认逻辑 NPU 数（A3：8 NPU × chipcount 2 = **16**），每 chip 64GB，合计 **1TB**；`free -g` 看 host 内存
- [ ] 2. 确认权重：优先 `GLM-5.1-w4a8`（最稳），或 `GLM-5.2-w4a8`（755GB，TP16 可试）；必须**非 c8**，`config.json` 有 `index_topk`、无 `compress_ratios`
- [ ] 3. 编译**基线版完整算子包**（全部 A3 算子，不是单算子）→ `/home/z00943842/base/fsao_ab/ops/base`
- [ ] 4. 完整安装并编译**优化版完整算子包** → `/home/z00943842/base/fsao_ab/ops/opt`
- [ ] 5. 两个包分别跑仓内单算子测试，均 `4 passed`
- [ ] 6. 单独 venv 安装 `ais-bench-benchmark`，克隆压测工具、改 `config.py`
- [ ] 7. 起服务（带 `sparse_kv_offload_config`），`/health` 正常、单请求成功
- [ ] 8. 按 ABBA 顺序跑 6 轮（base 3 轮 / opt 3 轮），每轮换算子包 + 重启服务；补充场景（7.3）另跑 6 轮
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

**本机 16 个逻辑 NPU（8 物理卡 × 2 chip）**：生产形态是 PD 分离（P、D 各占一组 chip），P TP8 + D TP8 正好用满 16 个逻辑 NPU，理论上可以做。但 PD 需要 memfabric、host pool 和代理进程，配置更复杂；本手册先用代码里预留的单机调试模式：

```json
"keep_device_kv_cache": true
```

此时 prefill 产生的 KV 会 D2H 写入 host pool，decode 阶段算子照常从 host pool 读取。**绝对性能不代表生产 PD 分离，但 A/B 的相对收益有效。**

> PR 里请把结论标注为「单机 PD-colocate 调试模式下的相对收益验证」。

---

## 2. 变量约定（先填好）

在容器内创建环境文件，后续每个新终端都先 `source` 它：

> `fsao_ab` 与 `vllm-ascend` 仓库**平级**（都放在 `/home/z00943842/base` 下）。

```bash
# vllm-ascend 仓库的上一级目录（按本机实际路径修改）
WORK_ROOT=/home/z00943842/base
mkdir -p "$WORK_ROOT/fsao_ab/results"

cat > "$WORK_ROOT/fsao_ab/env.sh" <<'EOF'
# ===== 按实际情况修改 =====
export WORK_ROOT=/home/z00943842/base              # vllm-ascend 的上一级目录
export VLLM_SRC=$WORK_ROOT/vllm-ascend             # 仓库根目录
export BASE_WORKTREE=$WORK_ROOT/vllm-ascend-base   # 基线 worktree
export AB=$WORK_ROOT/fsao_ab                       # 测试工作目录（与仓库平级）
export AISV_DIR=$WORK_ROOT/aisvenv                 # 压测客户端 venv
export AISBENCH_TOOL=$WORK_ROOT/aisbench_auto_tools_prefix
export DATASET_DIR=$WORK_ROOT/dataset

# 权重路径
export MODEL_PATH=/mnt/weight/GLM-5.1-w4a8
export MODEL_PATH_52=/mnt/weight/GLM-5.2-w4a8      # 可试

export NPU_NUM=16
export SERVED_NAME=glm-5.1                         # 与 --served-model-name 一致
export HOST_IP=127.0.0.1                           # 客户端与服务同机时用回环；跨机改成服务端 IP
export PORT=8000
export OPS_BASE=$AB/ops/base
export OPS_OPT=$AB/ops/opt
EOF

source "$WORK_ROOT/fsao_ab/env.sh"
mkdir -p "$OPS_BASE" "$OPS_OPT" "$DATASET_DIR"
```

---

## 3. 阶段 0：环境盘点

```bash
source /home/z00943842/base/fsao_ab/env.sh

# 1) 卡型号和显存（看 HBM-Usage 的总量：65536≈64GB，131072≈128GB）
npu-smi info

#    更明确的两种方式：
npu-smi info -t memory -i 0          # 看 Memory Size(MB) 一类字段
python3 - <<'PY'
import torch, torch_npu
n = torch.npu.device_count()
print("逻辑 NPU 数:", n)
total_all = 0
for i in range(n):
    free, total = torch.npu.mem_get_info(i)
    total_all += total
    print(f"NPU {i}: {total/1024**3:.1f} GB  (free {free/1024**3:.1f} GB)")
print(f"合计 HBM: {total_all/1024**3:.0f} GB")
PY

#    ⚠️ A3 的一张物理卡通常有 2 个 die，npu-smi 会显示两行/卡。
#    显存总量必须按“逻辑 NPU 数 × 单卡容量”算，不能用物理卡数 × 单卡容量。

#    板卡型号：
npu-smi info -t board -i 0 | grep -iE "product|chip|model"

# 2) 主机内存（决定 dram_size_per_dp_GB；注意这不是卡上的 HBM）
free -g

# 3) 权重目录与大小（16 chip 合计 1TB；755GB 目录约 47GB/chip）
ls -l "$MODEL_PATH"
du -sh "$MODEL_PATH"

# 4) 权重结构检查：期望 index_topk，不能有 compress_ratios / index_kpool
grep -oE '"(index_topk|compress_ratios|index_kpool)"' "$MODEL_PATH/config.json" | sort -u

#    确认不是 C8 KV（有输出且包含 C8 就是 c8 权重，不能用）
grep -o '"kv_cache_type"[^,}]*' "$MODEL_PATH/quant_model_description.json" 2>/dev/null

# 5) transformers 版本（requirements.txt 固定 5.19.0；GLM-5 教程要求 >= 5.2.0）
python3 -c "import transformers; print(transformers.__version__)"

# 6) 确认 vllm-ascend 是源码可编辑安装（后面要换 _cann_ops_custom）
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
```

**硬门槛**（任一不满足，先停下来处理）：

| 检查项 | 期望 | 不满足怎么办 |
|---|---|---|
| 逻辑 NPU 数 | 16（8 物理卡 × 2 chip），每 chip 64GB，合计 1TB | 用 `npu-smi info -l` 和 `torch.npu.device_count()` 确认 |
| 权重后缀 | 优先 `GLM-5.1-w4a8`；可试 `GLM-5.2-w4a8` | 不能用任何 `*c8`（C8 主 KV 被 offload 拒绝） |
| config.json | 有 `index_topk`，无 `compress_ratios` | 该权重不支持 sparse offload |
| KV 量化 | `quant_model_description.json` 里不是 `C8` | c8 权重不能用，换非 c8 版本 |
| 权重体积 | 755GB / 16 chip ≈ 47GB/chip；0.9 利用率下可用约 57.6GB/chip | 余量不足就换 GLM-5.1-w4a8（实测 391GB，约 24.5GB/chip） |
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

## 4. 阶段 1：完整安装与两套完整算子包

> ⚠️ **跑模型和跑单算子测试不一样**：单算子测试只需要一个 kernel，而模型会调用 `mla_preprocess`、`sparse_flash_attention`、`moe_gating_top_k`、`grouped_matmul_swiglu_quant`、`dispatch_ffn_combine` 等 **40 个左右**的 A3 自定义算子（完整列表见 `csrc/build_aclnn.sh` 的 `ascend910_93` 分支）。
>
> 所以**不能**用 `build.sh --ops=fused_sparse_attention_overlap` 的单算子包来跑模型——其他算子缺失，服务起不来。A/B 需要构建**两套完整包**，两者只有融合算子内核不同。

### 4.0 先检查运行环境（只做一次）

```bash
cd "$VLLM_SRC"
python3 -c "import vllm; print('vllm', vllm.__version__)"       # 版本号仅供参考
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"     # 必须指向 $VLLM_SRC
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
vllm serve --help > /dev/null && echo "vllm CLI OK"    # 提前暴露缺失依赖（如 orjson）

# main 分支应装 .github/vllm-main-verified.commit 对应的 vLLM commit
VLLM_UPSTREAM=${VLLM_UPSTREAM:-/vllm-workspace/vllm}
test "$(git -C "$VLLM_UPSTREAM" rev-parse HEAD 2>/dev/null)" = \
     "$(tr -d '[:space:]' < .github/vllm-main-verified.commit)" && echo "vllm commit OK"
```

| 检查项 | 期望 |
|---|---|
| vLLM commit | 等于 `.github/vllm-main-verified.commit`（main 分支）；版本号可能显示 `v0.30.0` 之类，正常。release 分支才用 `vllm-release-tag.commit` |
| vllm_ascend 路径 | `$VLLM_SRC/vllm_ascend/__init__.py` |
| torch / torch_npu | `2.10.0` / `2.10.0.post4`（见 `requirements.txt`） |
| 依赖 | `memfabric_hybrid`、`memcache_hybrid` 已安装 |

不满足就先补：

```bash
cd "$VLLM_SRC"
python3 -m pip install -r requirements.txt
```

### 4.1 拉基线代码（worktree，不切分支）

```bash
source /home/z00943842/base/fsao_ab/env.sh
cd "$VLLM_SRC"
git fetch --all
git worktree add "$BASE_WORKTREE" 96e222490
git -C "$BASE_WORKTREE" log --oneline -1
```

### 4.2 编译环境变量（两个 worktree 都 source）

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export SOC_VERSION=ascend910_9382
export MAX_JOBS=8
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/devlib:${LD_LIBRARY_PATH:-}"
```

### 4.3 完整安装并构建**优化版**（在 `$VLLM_SRC`，只做一次）

```bash
cd "$VLLM_SRC"
rm -rf -- csrc/build csrc/output csrc/build_out

# COMPILE_CUSTOM_KERNELS=1 是默认值：构建全部 A3 自定义算子 + vllm_ascend_C 扩展
COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e . --no-build-isolation
```

这一步内部会调用 `csrc/build_aclnn.sh`，把**完整算子包**装到
`$VLLM_SRC/vllm_ascend/_cann_ops_custom`。首次构建时间较长（常见 30 分钟以上），建议放在 tmux / nohup 里跑。

> 构建前确认磁盘空间：`df -h "$VLLM_SRC"`，全量算子构建的 `csrc/build` 可能占用几十 GB，建议预留 ≥ 100GB。
>
> 如果 `pip install -e .` 在构建全量算子时失败，可以拆成两步：
> 1. `bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"`（构建完整算子包）；
> 2. 按[附录 B](#附录-bpython-扩展编译备用方案)单独构建 `vllm_ascend_C` 扩展。

验收并保存为优化版：

```bash
cd "$VLLM_SRC"
ls vllm_ascend/_cann_ops_custom/vendors/custom_transformer/

python3 - <<'PY'
import torch, vllm_ascend
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
print("算子加载成功")
PY

rm -rf "$OPS_OPT"
cp -a "$VLLM_SRC/vllm_ascend/_cann_ops_custom" "$OPS_OPT"
```

### 4.4 构建**基线版完整算子包**（在 worktree，不重装 Python）

```bash
cd "$BASE_WORKTREE"

# worktree 不会自动带 submodule；从主树拷过来
if [ ! -d csrc/third_party/catlass/include ]; then
  cp -a "$VLLM_SRC/csrc/third_party/catlass" csrc/third_party/
fi

rm -rf -- csrc/build csrc/output csrc/build_out

# build_aclnn.sh 会先清空 vllm_ascend/_cann_ops_custom，再安装完整的 A3 算子集
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"

rm -rf "$OPS_BASE"
cp -a "$PWD/vllm_ascend/_cann_ops_custom" "$OPS_BASE"
```

> ⚠️ **不要**在 `$BASE_WORKTREE` 里执行 `pip install -e .`。Python 包和 `vllm_ascend_C` 扩展始终用 `$VLLM_SRC`（HEAD）那份，基线只取它的完整算子包。

### 4.5 验收

```bash
# 两套都应是完整的 vendor 目录，文件数量应接近（都包含全部 A3 算子）
du -sh "$OPS_BASE" "$OPS_OPT"
find "$OPS_BASE" -type f | wc -l
find "$OPS_OPT" -type f | wc -l

# 差异应只集中在 fused_sparse_attention_overlap 相关产物
diff -rq "$OPS_BASE" "$OPS_OPT" | head -20
```

如果差异铺满整个目录（几万个文件），说明两次构建的算子集不一致，回到 4.3/4.4 重做。

再用[附录 A](#附录-a单算子测试复核)分别在两套包下跑单算子测试，均应为 `4 passed`。

> 验证“完整”：`ls "$OPS_OPT/vendors/custom_transformer/op_impl"` 和在主树上 `python3 -c "import torch; ..."` 调用几个常用算子（如 `npu_mla_preprocess`）不应报 “operator not found”。

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
source /home/z00943842/base/fsao_ab/env.sh
python3 -m venv "$AISV_DIR"
source "$AISV_DIR/bin/activate"
pip install -U pip
pip install ais-bench-benchmark pandas

# 记下 WORK_PATH（目录下必须有 ais_bench/）
python3 -c "import ais_bench, os; print(os.path.dirname(os.path.dirname(ais_bench.__file__)))"
# 或者： pip show -f ais-bench-benchmark | grep Location
```

### 5.2 下载压测工具并改配置

```bash
source /home/z00943842/base/fsao_ab/env.sh
cd "$WORK_ROOT"
git clone https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git   # 已存在则跳过
cd "$AISBENCH_TOOL"
```

编辑 `config.py`：

```python
DATASET_PATH = "/home/z00943842/base/dataset"      # 与 env.sh 的 DATASET_DIR 一致
WORK_PATH    = "<5.1 打印的目录>"                   # 目录下要有 ais_bench/
MODEL_NAME   = "glm-5.1"                          # 必须等于 --served-model-name
MODEL_PATH   = "/mnt/weight/GLM-5.1-w4a8"          # 只用来读 tokenizer（与 env.sh 一致）
HOST_IP      = "127.0.0.1"   # 客户端与服务同机时用回环；跨机改成服务端 IP
HOST_PORT    = "8000"        # 与 env.sh 的 PORT 一致
API_KEY      = ""            # 本机 vllm serve 没配 --api-key，留空即可
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR   = "./outputs/default"
POD_INFO     = []
URL          = ""
```

> `API_KEY` 只有两种情况需要填：你自己启动时加了 `--api-key xxx`，或者服务前面有带鉴权的网关。
> 手册里的 `vllm serve` 没有鉴权，**留空**，不需要去申请任何 key；压测和 `/metrics` 都能正常访问。

**`WORK_PATH` 是干什么的**：它是**包含 `ais_bench/` 包的目录**。工具会往里面写三样东西：

| 路径 | 用途 |
|---|---|
| `$WORK_PATH/ais_bench/datasets/gsm8k/test.jsonl` | 生成的前缀数据集软链 |
| `$WORK_PATH/ais_bench/datasets/gsm8k/train.jsonl` | `ais_bench` 的 gsm8k 入口文件，不存在时建空文件 |
| `$WORK_PATH/ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py` | 按 `config.py` 生成的 API 配置 |

所以它必须指向**实际执行 `ais_bench` 命令的那个安装**（用 venv 就是该 venv 的 `site-packages`），且目录可写：

```bash
pip show -f ais-bench-benchmark | grep Location
# 例如：Location: /home/z00943842/base/aisvenv/lib/python3.12/site-packages
```

填错时常见报错：`dataset not found`、`vllm_api_chat_temp not found`、找不到 `train.jsonl`。

### 5.3 客户端自检

```bash
source /home/z00943842/base/fsao_ab/env.sh
source "$AISV_DIR/bin/activate"
python3 -c "from config import *; print(WORK_PATH, MODEL_NAME); import os; print(os.path.isdir(os.path.join(WORK_PATH,'ais_bench')))"
# 期望最后输出 True
```

> 如果运行时报 `ais_bench: error: unrecognized arguments: --num-warmups`，按工具 README 的 FAQ 3，在 `aisbench_test.py` 第 94 行删掉 `--num-warmups 0`。

---

## 6. 阶段 3：启动服务并确认算子生效

### 6.1 启动前检查

```bash
source /home/z00943842/base/fsao_ab/env.sh

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

### 6.2 启动命令（16 逻辑 NPU：GLM-5.1-w4a8，colocate 调试模式）

建议用 tmux 开两个窗口：一个跑服务，一个跑压测。

```bash
source /home/z00943842/base/fsao_ab/env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p $PROMETHEUS_MULTIPROC_DIR

source "$VLLM_SRC/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

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
    "enable_flashcomm1": true,
    "sparse_kv_offload_config": {
      "enabled": true,
      "use_fused_overlap": true,
      "keep_device_kv_cache": true,
      "dram_size_per_dp_GB": 48
    }
  }' 2>&1 | tee "$AB/results/server_$(date +%m%d_%H%M).log"
```

> **如果换到每 chip 128GB 的机器**（GLM-5.2-w8a8 场景）可放宽为：
> `--max-model-len 140000 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95`，
> `dram_size_per_dp_GB` 设 80。其余不变。

**必须遵守的参数约束**（否则进不了算子路径或直接报错）：

| 约束 | 说明 |
|---|---|
| 不要 `--kv-cache-dtype int8/fp8` | 主 KV 会变成 C8 packed cache，sparse offload 直接拒绝 |
| 不要 `enable_dsa_cp` | sparse offload 不支持 context parallel，会 `NotImplementedError` |
| `--data-parallel-size 1` | 本机为 16 逻辑 NPU，用 DP1 + TP16 用满；DP2×TP8 也是 16 卡组合，但本方案不用 |
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
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
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

**场景选择（16 逻辑 NPU × 64GB = 1TB + GLM-5.1-w4a8）**：

| 场景 | 命令参数 | 说明 |
|---|---|---|
| 首选 | `--input_len 32768 --data_num 32` | 显存最稳，先出 A/B 结论 |
| 加长 | `--input_len 65536 --data_num 16` | KV 约为首选 1.8 倍，需确认不 OOM |
| 128K | `--input_len 131072 --data_num 32` | 不建议：每 chip 仍要放完整 KV（约 45GB），加上权重和激活容易超 64GB |

> 同事原命令的 `--input_len 131072` 依赖的是“每 chip KV 放得下”，和卡数多少无关；在 64GB chip 上先跑 32K。

说明：

- `--npu_num 16`：本机 16 个逻辑 NPU（8 物理卡 × 2 chip）；
- `--prefix_test`：先把前缀预热一遍，保证全量测试时命中率接近 90%；
- `--repeat` 对 prefix_cache 分支**不生效**，重复靠手动多跑几轮；
- 数据集由 `seed=2` 生成在 `DATASET_PATH`，**几轮之间不要删除、不要换 seed**，保证数据完全一致。

### 7.3 优化覆盖：默认场景测不到全部改动

你改的优化是按 batch size 和命中率分条件生效的，**算子被调用 ≠ 每个优化点都被测到**：

| 优化点 | 生效条件 | 默认场景（32K / cc8 / 命中90%） |
|---|---|---|
| 空闲核 helper | `bs ≤ 12` | ✅ 覆盖（cc8 → bs=8） |
| P0 行划分并行度 | 行数 > 24（即 `bs > 24`） | ❌ 覆盖不到（bs=8） |
| Opt01 配对排序 / Opt02-04 回写 | 有 selection update | ✅ 覆盖 |
| Opt05 UB 直写 | 命中率 0~70% 时收益明显 | ⚠️ 90% 命中下覆盖较弱 |
| 窗口化两阶段 MergeKv | 所有 decode | ✅ 覆盖 |

GLM-5.1 上算子本身没问题（`kv_head_num=1`、`q_head_num>0` 即可，TP16 后 head 数正常），但**只用同事那条 cc8/90% 命令，P0 和 Opt05 的收益基本测不出来**。

建议再补一个针对性场景（单独的 A/B，启动时把 `--max-num-seqs` 调到 32）：

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

- `concurrency 32` → decode 时 `bs=32 > 24`，覆盖 P0 行划分；
- `repeat_rate 0.5` → 命中率 50%，覆盖 Opt05；
- 16K 输入是为了控制每 chip 的 KV（约 18GB），32K + 50% 命中在 64GB chip 上会很紧。

> 两个场景各自跑一套 ABBA（base/opt 各 3 轮），**同一场景内服务启动参数必须完全一致**，只换算子包。

### 7.4 保存结果

```bash
# 每轮跑完立即执行（CSV 是追加写，不搬走会混在一起）
mv aisbench_result.csv "$AB/results/base_r1.csv"   # 按实际构建/轮次改名
cp aisbench.log          "$AB/results/base_r1.log"
```

### 7.5 轮次安排（ABBA，抵消机器漂移）

| 顺序 | 构建 | 结果文件 |
|---|---|---|
| 1 | base | `base_r1.csv` |
| 2 | opt | `opt_r1.csv` |
| 3 | opt | `opt_r2.csv` |
| 4 | base | `base_r2.csv` |
| 5 | base | `base_r3.csv` |
| 6 | opt | `opt_r3.csv` |

每轮之间建议间隔 1~2 分钟，让 NPU 温度和频率稳定。

### 7.6 结果记录表

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

快速取中位数（在 `$AB/results` 下）：

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
| `single_output_throughput` | **上升** | 单 chip 吞吐（除以 16） |
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
| 找不到算子 / `libcust_opapi.so` 加载失败 | `_cann_ops_custom` 没换、只装了单算子包，或服务没重启 | 按 4.6 换完整包、重启进程、开新终端 |
| `/metrics` 查不到 `prefix_cache` | 代理未关 / 多进程指标目录未设 | `unset http_proxy https_proxy`；设置 `PROMETHEUS_MULTIPROC_DIR` |
| `unrecognized arguments: --num-warmups` | aisbench 版本差异 | 按工具 README FAQ 3 删除该参数 |
| `ModuleNotFoundError: No module named 'orjson'`（或其它 vllm 依赖） | `vllm` 命令使用的 Python 环境缺少 vLLM 依赖 | 用 `vllm` 入口同一个解释器安装：`/usr/local/python3.12.16/bin/python3 -m pip install orjson`；缺得多就装 `-r /vllm-workspace/vllm/requirements/common.txt`（**不要**装 `cuda.txt`） |
| `hcclGetRootInfo ... error code is 19 (call network api fail)` / `ERR02200 DIST call hccl api failed` | HCCL 选网卡失败、hostname 解析不到，或上次崩溃残留 | 设 `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`（用 `ip -o -4 addr show \| grep <本机IP>` 找网卡）；`/etc/hosts` 补 hostname；清理残留进程；`/dev/shm` ≥ 1GB |
| `HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE is invalid (EmptyString or 0 ... not both 1)` | 只设了 `HCCL_INTRA_ROCE_ENABLE=0`，而 `HCCL_INTRA_PCIE_ENABLE` 为空 | 两个都 `unset`（自动选择）；或显式 `HCCL_INTRA_PCIE_ENABLE=1` + `HCCL_INTRA_ROCE_ENABLE=0`。两者不能同时为 1 |
| `HCCL_BUFFSIZE_EP is too SMALL ... NEEDED_HCCL_BUFFSIZE=... actual CCL_BUFFSIZE=...`，随后 `vector core exception` / `aclnnMoeDistributeCombineV4 failed` | `HCCL_BUFFSIZE` 小于 MoE AllToAll 所需（常见需 ~206MB 以上） | 把 `HCCL_BUFFSIZE` 单位 MB 调大，例如 `export HCCL_BUFFSIZE=512`，重启服务 |
| `Loading safetensors checkpoint shards: 0%` 长时间不动 | 391GB 权重从 NFS 加载，首次没有 page cache | 通常是正常的，等；用 `cat /proc/<pid>/io` 看 `read_bytes` 是否增长，NFS 0.3~1GB/s 需要 7~20 分钟 |
| 权重已上卡但日志停在编译/图捕获 | `FULL_AND_PIECEWISE` 图捕获在 391GB 权重 + TP8 下很慢或 OOM | 加 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`，或 `--enforce-eager`；E2E 用 TP16 |
| 加载完初始化 KV 时 OOM | TP8 下每 chip 权重约 49GB，只剩约 8GB 给 KV/激活 | 改用 `-tp 16`（每 chip 约 24.5GB） |
| 启动 OOM | 权重 + KV + 图内存超过 64GB/chip | 降 `--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens`，或去掉 `--compilation-config`、换 TP16 |
| 命中率远低于 90% | host pool 太小导致 block 被换出 | 提高 `dram_size_per_dp_GB`（host 内存允许时）或降低并发 |
| 图模式启动失败 | 该路径图捕获有问题 | 加 `--enforce-eager` 先跑通（A/B 两端保持一致即可） |

---

## 10. 权重与模型专属问题（GLM-5.1 / 5.2）

### 10.1 GLM-5.2-w8a8 可以吗？

官方文档给出的最低配置是 **1TB HBM**（1 台 A3 128GB×8，或 2 台 A2 64GB×8）。你的机器是 **16 逻辑 NPU × 64GB = 1TB**，刚好达到下限，但余量取决于实际权重目录大小：

- 先 `du -sh` 实测；如果权重 ≥ 800GB，每 chip 只剩不到 15GB 给 KV/激活，32K 场景也很危险；
- 打开 offload 时同样要删掉 `enable_dsa_cp`，且不能用 c8 版本。

结论：**理论可达下限，但非常紧，不推荐作为首选**。

### 10.2 本机拓扑：16 逻辑 NPU × 64GB = 1TB

先纠正一个最容易算错的地方：

- `npu-smi info -l` 显示 **8 个 NPU**、每个 `chipcount = 2` → 实际是 **16 个逻辑 NPU**；
- 每个 chip 的 HBM 为 `65536 MB = 64 GiB` → 合计 `16 × 64 = 1024 GiB = 1TB`；
- 所以 **TP 要设 16**，`ASCEND_RT_VISIBLE_DEVICES=0..15`，不能按“8 张物理卡 × 64GB = 512GB”来算。

权重预算（每 chip；`--gpu-memory-utilization 0.90` 时可用约 57.6 GiB/chip）：

| 权重 | `du -sh` 实测 | 每 chip 权重 | 剩给 KV/激活 | 结论 |
|---|---|---|---|---|
| `GLM-5.1-w4a8` | 391GB（实测） | 约 24.5GB | 约 33GB | ✅ 最稳，首选 |
| `GLM-5.2-w4a8` | 755GB | 约 47GB | 约 10GB | ⚠️ 32K 可试，可能 OOM |
| `GLM-5.2-w8a8` | 需实测（文档下限 1TB） | 可能 > 50GB | 很小 | ⚠️ 不推荐 |

> 注意：TP16 只分摊**权重**；MLA 的 KV 在 TP rank 之间是复制的，每个 chip 仍要放一份完整 KV。
>
> 另外区分：`dram_size_per_dp_GB` 用的是**host 内存**，不是这里的 1TB HBM。

无论用哪个，都必须确认是**非 c8** 且主 KV 为 BF16（见 10.3）。

> 如果 5.2-w4a8 启动 OOM：先降 `--max-num-seqs` / `--max-model-len`，不行就换 5.1-w4a8，启动参数不用改，只改 `MODEL_PATH` 和 `--served-model-name`。

同时必须**缩小测试场景**，不能照搬同事的 128K / 16 卡命令。原因：

- 用 `keep_device_kv_cache=true` 时，device 上仍保留完整 paged KV，且 MLA 的 KV 在 TP rank 之间是**复制**的，每张卡都要放一份；
- 每个 64GB chip 要同时装下：权重（GLM-5.1-w4a8 约 24.5GB/chip；GLM-5.2-w4a8 约 47GB/chip）+ KV（每 chip 一份）+ 激活 / 图内存（约 10GB）。

显存预算参考（按 MLA 每 token 约 70~90KB 估算；GLM-5.2 层数可能与 5.1 不同，以实测 `npu-smi info` 为准）：

| 场景 | 唯一 token 数（90% 前缀共享） | 每 chip KV | 结论 |
|---|---|---|---|
| 32K × 32 请求 | ≈ 13 万 | ≈ 10GB | ✅ 首选 |
| 64K × 16 请求 | ≈ 16 万 | ≈ 13GB | ✅ 可以试 |
| 64K × 32 请求 | ≈ 27 万 | ≈ 22GB | ⚠️ 偏紧，可能触发换出 |
| 128K × 32 请求 | ≈ 53 万 | ≈ 45GB | ❌ 放不下 |

对应的启动参数已按此校准（见 6.2）：`--max-model-len 40960`、`--max-num-batched-tokens 4096`、`--gpu-memory-utilization 0.90`、`dram_size_per_dp_GB 48`。

其他选择：

- 换到每 chip 128GB 的机器（或增加节点），按同事的原命令跑 128K（见附录 C）；
- 不建议把 `GLM-5.2-w4a8c8` 强行降到 BF16 KV 来省显存——C8 与 sparse offload 互斥，且这是未验证组合。

### 10.3 w4a8 与 w4a8c8 的区别（重要）

官方文档的部署表没有单独列 `GLM-5.2-w4a8`，但你手里有；它和 `w4a8c8` 的唯一区别是**主 KV cache 是否量化（c8）**：

| 权重名 | 含义 | 能否用于本次 E2E |
|---|---|---|
| `GLM-5.2-w8a8` | W8A8 权重，主 KV 由 `--kv-cache-dtype` 决定（默认 BF16） | ⚠️ 文档下限 1TB；本机刚好达到，余量取决于实际体积，不推荐 |
| `GLM-5.2-w4a8` | W4A8 权重，主 KV 默认 BF16；**实测约 755GB** | ⚠️ TP16 下约 47GB/chip，32K 可试，余量小 |
| `GLM-5.2-w4a8c8` | W4A8 权重 + **C8 主 KV cache** | ❌ C8 与 sparse offload 互斥 |
| `GLM-5.1-w4a8` | W4A8 权重，主 KV 默认 BF16；**实测 391GB** | ✅ 最稳（官方 64GB × 8 配置） |

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
import os
p = os.path.join(os.environ["MODEL_PATH"], "quant_model_description.json")
if not os.path.exists(p):
    print("没有 quant_model_description.json（一般是非量化的 BF16 权重）")
else:
    print("存在 kv_cache_type=C8:", "C8" in open(p).read())
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

### 10.6 DeepSeek-V4 / V4.1-Flash 能用吗？——不能

即使权重体积够小（例如 V4.1-Flash 看到约 149GB），也**不能**用于本算子的 E2E：

1. **架构不同**：V4.1-Flash 用的是 CSA2（Compressed Sparse Attention 2）+ FP4 主 KV，不是 SFA 的 BF16 主 KV；它走 `vllm_ascend/attention/dsa_v41.py`，不走 `sfa_kv_offload.py`。
2. **配置直接被拒**：它的 config 带 `compress_ratios`，而 sparse KV offload 明确拒绝：

   ```python
   # vllm_ascend/ascend_config.py:1738
   if hasattr(vllm_config.model_config.hf_text_config, "compress_ratios"):
       raise ValueError("Sparse KV offload don't support compress now.")
   ```

3. **SFA backend 不会被选中**：`vllm_ascend/utils.py:206` 的 `model_uses_sfa_sparse()` 在检测到 `compress_ratios` 时返回 False。

结论：`npu_fused_sparse_attention_overlap` **根本不会被调用**，A/B 测不出任何差异。本算子目前只适用于 SFA + sparse KV offload 路径，即 **GLM-5.1 / GLM-5.2（非 c8）/ DeepSeek-V3.2**。

一条命令判断权重是否可用：

```bash
grep -c '"compress_ratios"' "$MODEL_PATH/config.json"
# 输出 0 → 可以；>=1 → 不行
```

---

## 附录 A：单算子测试复核

对每个算子包都要验证一次：**先按 4.6 切到目标包**，再执行下面的命令。

```bash
source /home/z00943842/base/fsao_ab/env.sh
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

## 附录 B：Python 扩展编译（备用方案）

当 `COMPILE_CUSTOM_KERNELS=1 pip install -e .` 在构建全量算子阶段失败，或你只需要补 `vllm_ascend_C` 扩展时，用本节。

> 前提：完整算子包已经通过 `bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"` 构建到 `vllm_ascend/_cann_ops_custom`。
> 这里 `COMPILE_CUSTOM_KERNELS=0` 是**故意**的——只重装 Python 包并编译扩展，不再重建算子。
> 如果 `_cann_ops_custom` 里没有算子，先回到 4.3。

完整步骤见仓库里的 `FUSED_SPARSE_ATTENTION_OVERLAP_ONLY_GUIDE.md` 第 7~10 步，摘要：

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

## 附录 C：16 逻辑 NPU 的 PD 分离方案

本机 16 个逻辑 NPU 正好可以切成 **P TP8（chip 0-7）+ D TP8（chip 8-15）**。前提是**每个实例都要加载完整权重**：

- `GLM-5.1-w4a8`（实测 391GB）→ 每侧 8 × 64GB = 512GB，每 chip 约 49GB 权重，余量约 15GB/chip，偏紧 ⚠️
- `GLM-5.2-w4a8`（755GB）→ 每侧 512GB 放不下 ❌（这种情况只能用 TP16 colocate）

PD 需要额外的 memfabric 配置和代理进程；如果只是做算子 A/B，优先用第 6 节的 colocate 调试模式。

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
source /home/z00943842/base/fsao_ab/env.sh

# 停服务
pkill -f "vllm serve" || true

# 删除基线 worktree
cd "$VLLM_SRC"
git worktree remove "$BASE_WORKTREE" --force

# 可选：删除编译产物和临时包
rm -rf "$AB/ops" "$AB/results"
```

如需恢复仓库原状（只删本手册产生的文件，不改代码）：

```bash
cd "$VLLM_SRC"
git status
```

---

## 附录 E：HCCL 最小自检（可选）

当 vLLM 报 HCCL 相关错误时，用这段独立脚本区分「环境问题」和「vLLM 问题」：

```python
# /tmp/hccl_test.py
import os
import torch
import torch_npu
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)                     # 关键：每个 rank 绑定自己的 NPU
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

判读：

| 现象 | 含义 |
|---|---|
| `same physical device ID 0` | 脚本/进程没绑定各自的 NPU（缺 `torch.npu.set_device(local_rank)`） |
| `EI0015 Ranktable_Detect` + 网卡提示 | `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` 配错 |
| `HCCL_INTRA_* invalid` | 两个 intra 变量要么都 unset，要么显式互斥 |
| 8 个 rank 都打印 `ok` | HCCL 正常，问题在 vLLM 启动参数或环境变量继承 |

---

## 附：本次测试的判断标准（写报告用）

1. **功能正确性**：两个算子包单算子测试均 `4 passed`；整网请求输出正常无异常。
2. **是否命中目标路径**：`use_fused_overlap=true` + decode 正常返回 + `/metrics` 有 `prefix_cache`。
3. **性能收益**：opt 相对 base，3 轮中位数 `TPOT avg` 下降、`output_throughput` 上升，且命中率两端一致（≈90%）。
4. **证据边界**：单机 colocate 调试模式的相对收益，不等同于生产 PD 分离的绝对值。
