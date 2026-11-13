# FSAO A/B 工具

在 A3 单机上做 `fused_sparse_attention_overlap` 的整网 A/B。默认假设：

- 基线源码：`f138fdfb`（PR #20927 的 merge-base）
- 优化源码：`653d08d`（PR #20927 的 head）
- 模型：GLM-5.1-w4a8
- 运行模式：`FULL_DECODE_ONLY`
- 场景：32K / cc8 / `repeat_rate 0.5`

## 文件

| 文件 | 作用 |
|---|---|
| `env.sh` | 所有路径、模型、压测/服务参数（**先改这里**） |
| `serve.sh` | 启动 vLLM 服务（sparse offload + fused overlap） |
| `bench.sh` | 跑一轮 aisbench |
| `swap_ops.sh` | 切换 `base` / `opt` 完整算子包并记录指纹 |
| `round.sh` | 一轮完整流程：换包 → 启服务 → 等 health → 压测 → 存结果 → 停服务 |
| `abba.sh` | 按顺序跑多轮，默认 `base opt opt base base opt` |
| `check_pkgs.sh` | 测前校验：diff 范围、fsao md5、build-cache HIT/MISS |
| `summary.py` | 汇总每轮值、中位数、极差、opt vs base 变化 |
| `env_decode64k.sh` | 干净 decode 预设：64K / bs4 / out64、单波次、无前缀共享 |
| `abba_decode64k.sh` | 用 decode 预设跑 ABBA，结果单独存 `$AB/results_decode64k` |

## 使用

### 1. 拷到机器并改路径

脚本按**自身所在目录**解析 `REPO` / `AB`，所以可以整体放在：

- `~/base/`（与 `vllm-ascend/` 同级，推荐）；或
- 任意 `tools/` 子目录（历史布局）。

同一目录下必须放齐所有脚本（`abba.sh`、`round.sh`、`serve.sh`、`bench.sh`、`env.sh`、
`swap_ops.sh`、`check_pkgs.sh`、`summary.py`，以及新增的两个 decode 脚本）。

```bash
# 例：全部放在 ~/base 下，与 vllm-ascend 同级
chmod +x ~/base/*.sh
vi ~/base/env.sh        # 至少核对 MODEL_PATH / AISV_DIR / AISBENCH_TOOL
source ~/base/env_decode64k.sh   # 会打印解析出的 repo/ab/results 路径
```

至少改这几项：

```bash
REPO=/home/z00943842/base/vllm-ascend
AB=/home/z00943842/fsao_ab
MODEL_PATH=/实际路径/GLM-5.1-w4a8
AISV_DIR=/实际路径/aisvenv
AISBENCH_TOOL=/实际路径/aisbench_auto_tools_prefix
DATASET_DIR=/实际路径/dataset
```

同时确认 `AISBENCH_TOOL/config.py` 里 `MODEL_NAME`、`HOST_PORT` 与 `env.sh` 的
`SERVED_NAME`、`PORT` 一致。

### 2. 准备两套包

按以下提交各编一套**完整**算子包，保存到 `$AB/ops/base`、`$AB/ops/opt`：

```bash
# base
cd $REPO && git checkout f138fdfb
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"
rm -rf $AB/ops/base
cp -a vllm_ascend/_cann_ops_custom $AB/ops/base

# opt
cd $REPO && git checkout 653d08d
bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"
rm -rf $AB/ops/opt
cp -a vllm_ascend/_cann_ops_custom $AB/ops/opt
```

> 推荐布局：`$AB/ops/base` 和 `$AB/ops/opt` 本身就是 `_cann_ops_custom` 的内容，
> 正确路径是 `base/vendors/custom_transformer`。
> `cp -a` 前先 `rm -rf` 目标目录，避免多嵌套一层 `_cann_ops_custom`。
> 如果已经嵌套了，`swap_ops.sh` / `check_pkgs.sh` 也能自动识别，不必强制拍平。

编译前确认 `SOC_VERSION=ascend910_9382`，并
`source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh`。最终让 `$REPO` 停在
`653d08d`（Python 运行时用这棵树）。

### 3. 测前校验

```bash
cd /home/z00943842/fsao_ab/tools
source env.sh
./check_pkgs.sh
```

必须满足：

- `diff entries` 只有少量 fsao 相关产物；
- base 和 opt 的 `fused_sparse_attention_overlap` 产物 **md5 不同**；
- opt build 日志里该算子不是 `HIT`。

### 4. 跑 A/B

```bash
cd /home/z00943842/fsao_ab/tools
./abba.sh
```

默认顺序：

```text
base_r1 -> opt_r1 -> opt_r2 -> base_r2 -> base_r3 -> opt_r3
```

每轮之间默认间隔 120 秒，可覆盖：

```bash
ROUND_GAP=60 ./abba.sh
```

首次启动要加载 391GB 权重，health 默认等 3600 秒，可覆盖：

```bash
HEALTH_TIMEOUT=5400 ./abba.sh
```

只做同包噪声对照（两个 base）：

```bash
./abba.sh base base
```

自定义顺序：

```bash
./abba.sh base opt opt base base opt
```

轮次编号自动递增，不会覆盖已有结果。

### 5. 看结果

每轮产物在 `$AB/results/`：

```text
base_r1.csv / base_r1.log / base_r1_server.log / base_r1_meta.txt
opt_r1.csv  / ...
rounds.csv  / summary.csv
```

`abba.sh` 结束时自动打印汇总；也可以手动：

```bash
$AISV_DIR/bin/python summary.py
```

汇总会给出：

- 每轮 `TPOT avg / TPOT SLO_P90 / output_throughput / E2E_time`；
- 每个构建的中位数与极差；
- `opt vs base` 的百分比变化；
- 当变化小于批间极差时标 `within noise`。

### 6. 干净 decode 场景（64K / bs4 / out64）

默认场景（32K、cc8、32 请求、repeat_rate 0.5）里 prefill 和 decode 同批执行，
TPOT 量到的是"被 prefill 拖慢的 inter-token 时间"，会到几百 ms，不能当 decode 能力。

`env_decode64k.sh` 换成单波次形状：`data_num == concurrency`，一波请求同时 prefill、
同时 decode，后面没有第二批请求插入；`repeat_rate=0` 保证各请求上下文不共享。

```bash
# 默认 bs=4 / 64K / out=64
./abba_decode64k.sh

# 自定义
DECODE_BS=8 DECODE_OUT_LEN=256 ./abba_decode64k.sh
```

参数对应 kernel profiling（`tests/ut/test_fuse_sparse_attention_overlap_precisions.py`）：

| profiling | 本预设 |
|---|---|
| `bs` | `DECODE_BS`（= concurrency = max-num-seqs = data_num） |
| `msl=64K` | `DECODE_IN_LEN=65536`，`max-model-len` 自动 +2K |
| `topk=2048` | 模型 `index_topk`，E2E 不可调 |
| `hit_ratio` | **无客户端旋钮**，不要用 `repeat_rate` 替代 |

注意：

- 结果单独存 `$AB/results_decode64k`，不要和默认场景混在一起汇总；
- 64K × bs4 的独立 KV 约 25GB，`DRAM_GB=64` 够用；改成 bs16 需要 ~100GB；
- `out=64` 只有 63 个 TPOT 采样，跑完确认 CSV 的 `output_len` 总量约等于
  `data_num × 64`，若明显偏小说明提前 EOS，这次均值不可用；
- bs=4 覆盖的是 helper 路径；P0 收益需要 bs>24，得另开场景。

## 注意事项

- **两端只用同一份 `env.sh` / `bench.sh`**，不要手工改其中一边。
- 每轮必须换包并重启服务；算子只在进程启动时加载。
- 不要一边 `FULL_DECODE_ONLY` 一边 `--enforce-eager`。
- 不要拿 GLM-5.2 的历史数字和 GLM-5.1 的新数字直接比。
- 如果哪天手动改了 `--repeat_rate`、`concurrency`、`max-num-seqs`，两轮结果不可比。
- 如果 `aisbench_test.py` 报 `unrecognized arguments: --num-warmups`，按工具 README FAQ 3 删除该参数。
