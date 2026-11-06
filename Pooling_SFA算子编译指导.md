# Pooling SFA算子编译指导

本文提供一条固定流程：从拉取官方仓库 PR #20927 开始，到编译 `fused_sparse_attention_overlap` 算子，再到跑通仓内精度测试。

PR 地址：<https://github.com/vllm-project/vllm-ascend/pull/20927>

> 按章节顺序复制命令即可。第 1、2 节为硬件和软件要求，第 3 节在宿主机执行，其余章节均在容器内执行。

## 1. 硬件信息

| 项目 | 要求 / 取值 | 说明 |
|---|---|---|
| 硬件产品 | Atlas A3 | 已在 A3-syn-23、A3-syn-47 验证 |
| CPU 架构 | aarch64（ARM64） | 与镜像内 aarch64 wheel 对应 |
| 芯片型号 | `ascend910_9382` | 编译 `SOC_VERSION` 与运行时必须一致 |
| 达芬奇架构 | DAV_2201，代码路径 `op_kernel/arch22` | 本次优化仅对 arch22 生效 |
| 单卡可见 AI Core | 24 AIC（每 AIC 配 2 个 AIV subcore） | tiling 与 workspace 按 AIC 数（24）为单位分配 |
| 测试占用卡数 | 1 张（`ASCEND_RT_VISIBLE_DEVICES=0`） | 官方单算子精度用例单卡即可 |
| NPU 驱动 | 宿主机安装，容器挂载 `/usr/local/Ascend/driver` | `npu-smi info` 能正常显示设备即合格 |
| 共享存储 | `/mnt/share_space` 挂载进容器 | 代码与编译产物目录 |

在容器内确认硬件信息：

```bash
npu-smi info
python3 -c "import platform; print(platform.machine())"
```

`platform.machine()` 必须输出 `aarch64`；`npu-smi info` 失败时先检查宿主机驱动和容器设备挂载，不要继续编译。

## 2. 软件信息要求

| 组件 | 要求 / 取值 | 说明 |
|---|---|---|
| 容器镜像 | `quay.io/ascend/vllm-ascend:nightly-main-a3` 或 `vllm-ascend:30.0-a3` | 两种镜像均已验证 |
| 操作系统 | 镜像自带 Linux（aarch64） | 直接使用镜像内置环境 |
| Python | 3.12 | 与镜像一致 |
| CANN | 镜像内置，已验证 9.2.1 | 统一通过 `/usr/local/Ascend/ascend-toolkit` 加载，不要硬编码 `cann-9.x` 路径 |
| vllm-ascend | 官方仓库 PR #20927 head | `git fetch origin pull/20927/head` |
| vLLM | 取 PR 分支 `.github/vllm-main-verified.commit`，当前为 `a98247ab4db686ee03c66d5feb3c761e52a2f8ab` | 必须严格对齐，不能用镜像预装版本 |
| torch / torch_npu | 镜像内置 | 与 CANN 绑定，不要随意升级 |
| fla_npu | `flash_linear_attention_npu_a3-26.9.1+dev5cc6990`（aarch64 wheel） | pytest 公共 `conftest.py` 导入链需要 |
| modelscope | `<1.38` | 同上，不要设置 `VLLM_USE_MODELSCOPE` |
| pytest | `>=6,<9` | |
| setuptools-rust / ninja / pybind11 | pip 可安装的最新版本 | 编译依赖 |

一条命令打印当前实际版本：

```bash
python3 - <<'PY'
import platform
import sys

import torch
import torch_npu
import vllm

print("arch     :", platform.machine())
print("python   :", sys.version.split()[0])
print("torch    :", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm     :", vllm.__version__)
print("vllm path:", vllm.__file__)
PY

echo "CANN_HOME: $(readlink -f /usr/local/Ascend/ascend-toolkit/latest)"
```

## 3. 拉取镜像并启动容器

在宿主机执行：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:nightly-main-a3
export WORKSPACE=/mnt/share_space/turbo-ops/z00943842

mkdir -p "$WORKSPACE"
mkdir -p "$HOME/.cache"
docker pull "$IMAGE"
```

> 若使用其他已验证镜像（例如 `vllm-ascend:30.0-a3`），只需替换 `IMAGE` 变量。

启动容器并挂载 NPU 设备、驱动和代码目录：

```bash
docker run \
  --name pooling-sfa \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /mnt/share_space:/mnt/share_space \
  -v "$HOME/.cache:/root/.cache" \
  -w /mnt/share_space/turbo-ops/z00943842 \
  -it "$IMAGE" bash
```

命令执行后会进入容器。先检查 NPU：

```bash
npu-smi info
python3 --version
```

`npu-smi info` 必须能正常显示设备。若这里失败，应先检查宿主机的 NPU 驱动和容器设备挂载，不要继续编译。

## 4. 拉取 PR #20927 代码

在容器内执行。官方仓库直接支持拉取 PR 引用，不需要 fork：

```bash
cd /mnt/share_space/turbo-ops/z00943842

if [[ ! -d vllm-ascend/.git ]]; then
  git clone https://github.com/vllm-project/vllm-ascend.git vllm-ascend
fi

cd vllm-ascend
git fetch origin pull/20927/head:pr20927
git checkout pr20927
git submodule update --init --recursive
```

> 目录已存在且有未提交改动时，先 `git stash` 或换一个目录，避免覆盖。

确认当前代码和配套 vLLM 提交：

```bash
git log --oneline -3
git rev-parse HEAD
cat .github/vllm-main-verified.commit
```

预期：

- `git log` 中包含 `perf(arch22): port A3 optimizations from poolingDSA yq/opt-all0908`；
- `.github/vllm-main-verified.commit` 输出 `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`。

确认优化代码在位：

```bash
grep -n "workerCount = GetBlockNum() \* 2" \
  csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_service_vector_mla.h
grep -n "InitHelperSync" \
  csrc/attention/fused_sparse_attention_overlap/op_kernel/arch22/fused_sparse_attention_overlap_kernel_mla.h
```

两条命令都要有输出。

> PR 的 head 基于较早的 main，切过来等于把整棵树回退到该基线，所以第 6 节必须把 vLLM 切到 PR 自己指定的提交，不能沿用镜像或最新 main 的 vLLM。

## 5. 配置编译环境

`nightly-main-a3` 是滚动镜像，上游 main 当前默认使用 CANN 9.2.1 和 vLLM v0.32.0，而 PR #20927 分支的 Dockerfile 仍标记 CANN 9.1.0 和 vLLM v0.31.0。这里通过 `/usr/local/Ascend/ascend-toolkit` 通用路径加载镜像实际安装的 CANN，不要在新的 nightly 镜像中硬编码 `cann-9.1.0`；vLLM 则在第 6 节单独切换到 PR 指定的提交。

在仓库根目录执行：

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source "$ASCEND_HOME_PATH/set_env.sh"

unset LD_PRELOAD
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export SOC_VERSION=ascend910_9382
export MAX_JOBS=8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

安装编译和测试所需的 Python 工具：

```bash
python3 -m pip install \
  setuptools-rust \
  ninja \
  pybind11 \
  'modelscope<1.38' \
  'pytest>=6,<9'
```

检查 ninja 没有指向其他 Python 环境：

```bash
which python3
which ninja
ninja --version
```

`ninja --version` 必须正常打印版本号。

## 6. 安装 PR 配套的 vLLM

镜像内预装的 vLLM 不一定与 PR 分支匹配，必须切换到仓库指定的提交。这里只操作 `/vllm-workspace/vllm`，不会切换或修改当前 `vllm-ascend` 分支。

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

export VLLM_COMMIT="$(tr -d '[:space:]' < .github/vllm-main-verified.commit)"
export VLLM_SRC=/vllm-workspace/vllm

if [[ ! -d "$VLLM_SRC/.git" ]]; then
  git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi

git -C "$VLLM_SRC" fetch --depth 1 origin "$VLLM_COMMIT"
git -C "$VLLM_SRC" checkout --detach FETCH_HEAD

test "$(git -C "$VLLM_SRC" rev-parse HEAD)" = "$VLLM_COMMIT"

VLLM_TARGET_DEVICE=empty python3 -m pip install -e "$VLLM_SRC" \
  --no-build-isolation \
  --no-deps
```

检查当前 PR 需要的两个 vLLM 接口，并确认 FLA 依赖可用：

```bash
python3 - <<'PY'
import vllm
import fla_npu.ops.ascendc
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens

print("vLLM 版本:", vllm.__version__)
print("vLLM 路径:", vllm.__file__)
print("vLLM 接口和 fla_npu 检查通过")
PY
```

必须看到：

```text
vLLM 接口和 fla_npu 检查通过
```

## 7. 全量编译并安装 vllm-ascend

如果这个目录编译过其他分支或其他镜像的代码，先清掉旧产物，避免 CMake 缓存和 ninja 缓存串版本：

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

rm -rf csrc/build csrc/build_out csrc/output build
```

然后在仓库根目录执行：

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

export SOC_VERSION=ascend910_9382
export MAX_JOBS=8

COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e . \
  --no-build-isolation
```

该命令会编译 CANN 自定义算子和 Python 扩展，并以可编辑模式安装当前仓库。看到以下信息后表示安装完成：

```text
Successfully built vllm_ascend
Successfully installed ... vllm_ascend-...
```

安装过程中如果出现 `fastapi` 或 `starlette` 的 pip 依赖冲突警告，本次单算子精度测试可以先忽略；只有出现 `ERROR: Failed building...` 或命令非零退出时才表示编译安装失败。

## 8. 加载自定义算子环境

编译成功后执行：

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

source "$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
export SOC_VERSION=ascend910_9382
```

检查算子注册：

```bash
python3 - <<'PY'
import torch
import torch_npu
import vllm_ascend
from vllm_ascend.utils import enable_custom_op

print("vllm_ascend 路径:", vllm_ascend.__file__)
assert vllm_ascend.__file__.startswith(
    "/mnt/share_space/turbo-ops/z00943842/vllm-ascend/"
), "导入的不是当前仓库"
assert torch.npu.is_available(), "NPU 不可用"

enable_custom_op()
assert hasattr(
    torch.ops._C_ascend,
    "npu_fused_sparse_attention_overlap",
), "算子未注册"

print("NPU 和 fused_sparse_attention_overlap 算子检查通过")
PY
```

必须看到：

```text
NPU 和 fused_sparse_attention_overlap 算子检查通过
```

## 9. 运行仓内精度测试

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
```

测试包含 BF16、FP16 与 `topk=32、128` 的 4 个参数化用例。每个用例都会检查首次 Miss 和再次 All-Hit 的 Attention 输出及 Selection Cache 状态。

最终通过标准：

```text
4 passed
```

不需要设置 `VLLM_USE_MODELSCOPE=true`。该测试不下载模型，但 pytest 的公共 `conftest.py` 会导入 ModelScope，因此第 5 节已经安装 `modelscope<1.38`。

## 10. 重新进入容器后的复测

如果退出过容器，在宿主机执行：

```bash
docker start -ai pooling-sfa
```

进入容器后，不需要重新拉取代码或编译，执行：

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source "$ASCEND_HOME_PATH/set_env.sh"
source "$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

unset LD_PRELOAD
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
export SOC_VERSION=ascend910_9382

pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
```

## 11. 常见报错

### 11.1 缺少 `setuptools_rust`

错误包含：

```text
ModuleNotFoundError: No module named 'setuptools_rust'
```

执行后重新运行第 6 节：

```bash
python3 -m pip install setuptools-rust
```

### 11.2 缺少 `CircularBufferSpec` 或 `prepare_dcp_local_seq_lens`

这说明当前导入的 vLLM 与 PR 分支不匹配。重新执行第 6 节，并确认：

```bash
git -C /vllm-workspace/vllm rev-parse HEAD
cat /mnt/share_space/turbo-ops/z00943842/vllm-ascend/.github/vllm-main-verified.commit
```

两条命令输出的提交必须完全一致。修复后不需要重新编译算子。

### 11.3 缺少 `vllm.v1.core.kv_cache_planning`

错误包含：

```text
ModuleNotFoundError: No module named 'vllm.v1.core.kv_cache_planning'
```

这是相反的方向：vLLM 比 PR 分支新（例如切过最新 main 后没有切回来）。同样执行第 6 节把 vLLM 切回 `.github/vllm-main-verified.commit` 指定的提交即可。

### 11.4 缺少 `fla_npu`

错误包含：

```text
ModuleNotFoundError: No module named 'fla_npu'
```

安装当前分支指定的 FLA 包：

```bash
python3 -m pip install --no-deps \
  "https://github.com/flashserve/flash-linear-attention-npu/releases/download/v26.9.1-20261020-020004/flash_linear_attention_npu_a3-26.9.1+dev5cc6990-py3-none-manylinux_2_34_aarch64.whl"

python3 -c "import fla_npu.ops.ascendc; print('fla_npu 检查通过')"
```

然后重新运行第 9 节。

### 11.5 缺少 `modelscope`

```bash
python3 -m pip install 'modelscope<1.38'
```

然后重新运行第 9 节。

### 11.6 Ninja 指向旧 Python 路径

如果错误包含 `/usr/local/python3.11.10/bin/ninja`，但当前 Python 是其他版本，执行：

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend

python3 -m pip install ninja
export PATH="$(dirname "$(command -v python3)"):$PATH"
hash -r

which python3
which ninja
ninja --version

rm -rf csrc/build csrc/build_out csrc/output build
```

确认 `ninja --version` 正常后，重新执行第 7 节。

### 11.7 `libcust_opmaster_rt2.0.so` 报 undefined symbol

错误包含：

```text
Failed to load libcust_opmaster_rt2.0.so wish env ASCEND_HOME_PATH(...)
error message is .../libgraph_base.so: undefined symbol: ...
```

原因是复用了其他镜像或其他 CANN 版本留下的 CMake/ninja 缓存，编译期与运行期 CANN 不一致。按第 7 节清理 `csrc/build`、`csrc/build_out`、`csrc/output`、`build` 后重新全量编译。不要用 `--force` 跳过安装器校验来绕过。

确认编译实际使用的 CANN 路径：

```bash
cd /mnt/share_space/turbo-ops/z00943842/vllm-ascend
grep -E "ASCEND_CANN_PACKAGE_PATH|OP_BUILD_TOOL" csrc/build/CMakeCache.txt
```

该路径必须等于当前容器内 `readlink -f /usr/local/Ascend/ascend-toolkit/latest` 的结果。

### 11.8 编译在 `build_aclnn.sh` 阶段非零退出

先看日志里第一处 `[ERROR]`，再对照第 7 节确认：

- `ASCEND_HOME_PATH` 是否已通过 `/usr/local/Ascend/ascend-toolkit` 加载；
- `SOC_VERSION` 是否为 `ascend910_9382`；
- 是否清过旧编译产物。

清理后重新执行第 7 节。
