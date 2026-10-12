# DSA 算子（fused_sparse_attention_overlap）性能测试套件

从 `vllm-ascend-poolingDSA` 仓库的 `happy-math/yq/opt-all0908` 分支移植，
给上游 vllm-ascend 仓库的 DSA 算子用。方法论、参数口径、判据与原仓库一致。

**一句话**：`tests/ut/test_fuse_sparse_attention_overlap_precisions.py` 是单进程
计时 bench，`sweep_median.py` 多进程跑它取中位数出表，`cmp_sweeps.py` 逐点比
两趟给判定。**差别小于门槛的不算改进**——门槛不是拍脑袋的百分比，是这趟测量
自己算出来的。

## 1. 文件清单

| 文件 | 干什么 | 依赖 |
| --- | --- | --- |
| `tests/ut/test_fuse_sparse_attention_overlap_precisions.py` | **计时 bench**。一个进程里扫完所有点，产出 `KERNEL` 行 | torch / torch_npu / numpy / memfabric_hybrid |
| `yq_files/sweep_median.py` | **多进程跑 bench 取中位数**，出表 | 只依赖 bench（按相对路径找 `tests/ut/`） |
| `yq_files/cmp_sweeps.py` | **逐点比两趟，给判定**（A/A 门槛自动算） | 只读日志，纯文本处理 |
| `yq_files/check_contention.py` | 查这趟有没有被别的任务干扰 | 只读日志 |
| `yq_files/fake_bench.py` | 假 bench，`--bench` 指向它就能空跑验流程，不碰 NPU | 无 |
| `yq_files/tests/test_fsao_precision.py` | **精度用例**，带 fp32 golden，两种签名都认 | pytest / memfabric_hybrid |
| `yq_files/env.sh` | 一次 source 就绪：toolkit + `SOC_VERSION`（默认 ascend910_9391）+ vendor 路径 | bash |
| `yq_files/build_ops.sh` | 全量编译：算子 vendor 包 + `vllm_ascend_C.so`，编完即可测 | setup.py / CANN |

⚠ **保持相对位置**：`sweep_median.py` 按 `__file__` 往上两级找
`tests/ut/test_fuse_sparse_attention_overlap_precisions.py`。放到别处就显式传
`--bench <路径>`。

⚠ **精度用例必须一起跑。** 性能变快但精度挂了是「错的更快」。两条路（内部
planner / 外部计划 LRU）**只测一条会漏掉真问题**——原仓库撞过
`slots=4096` 那一半挂、`slots=topk` 全过，根因是 kernel 在 capacity > topk
时把一半命中静默判成未命中。

## 2. 本仓库的一次性环境准备（A3 机器）

bench 开头会检查 `ASCEND_CUSTOM_OPP_PATH`，为空直接退出并打印修法
（确实想跳过检查设 `DSA_SKIP_ENV_CHECK=1`）。本仓库 kernel 不在
`vllm_ascend_C.so` 里，在 CANN vendor 算子包里。

**一键流程（推荐）**：

```bash
source yq_files/env.sh        # toolkit + SOC_VERSION（缺省补 ascend910_9391）+ vendor 路径
bash yq_files/build_ops.sh 2>&1 | tee /tmp/build_ops.txt
# 只编本算子（更快，且不需要 catlass 子模块——内网拉不动 gitcode 时用这个）：
OPS=fused_sparse_attention_overlap bash yq_files/build_ops.sh 2>&1 | tee /tmp/build_ops.txt
```

`build_ops.sh` 做的事 = `python3 setup.py build_aclnn`（编 vendor 算子包并
就地装进 `vllm_ascend/_cann_ops_custom/`，本仓库 `vllm_ascend` import 时会
自动补 `ASCEND_CUSTOM_OPP_PATH`）+ `python3 setup.py build_ext --inplace`
（编 pybind 模块）。两步缺一不可——只跑 build_ext 就是「.so 换了、kernel
没换」。脚本结尾会打印 `.so` 列表、算子 kernel `.o` 的 md5 指纹，并试
import 确认算子注册上了。

首次使用（或换机器/换 venv）还要补两件事，之后日常增量就不用了：

```bash
pip install -e . --no-build-isolation   # 注册包路径（build_ops.sh 不代劳，避免双重编译）
pip install memfabric_hybrid            # bench 的 CPU KV 池后端
```

⚠ **改过 csrc/ 下任何头文件必须全量重编**（脚本默认删 build 全量）。
增量编译是否生效只能看 `.o` 的 md5 指纹——时间戳、文件名里的哈希、编译
日志里出现算子名，三样全都骗人（ninja 返回 0 但 vendor 继续用旧 kernel
的情况真实发生过）。

⚠ **CANN 版本倾斜**：main 分支部分算子（`sparse_flash_mla` 等 25 个文件）
依赖比 cann-9.0.0 更新的工具链头文件（`log/log.h` 里的 `OP_LOGE_FOR_INVALID_*`
宏族），全量 build_aclnn 在这类机器上编不过。bench 只需要 fused 算子（不受
影响），用 `OPS=` 单算子模式即可绕开；必要时也可用
`VLLM_ACLNN_OPS=算子1;算子2` 手工指定 build_aclnn 的算子列表。

手动等价流程（排查脚本问题时用）：

```bash
# 1) CANN 环境（按机器实际路径）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910_93    # 本仓库 setup.py 不自动探测，必须显式导出

# 2) 编 vendor 算子包并就地安装（等价于 setup.py build_aclnn）
cd csrc
bash build.sh --pkg --soc=ascend910_93 --ops=fused_sparse_attention_overlap
bash ./build/cann-ops-transformer*.run --install-path=../vllm_ascend/_cann_ops_custom
cd ..

# 3) 本仓库 Python 侧（注册 torch.ops._C_ascend.* 的 pybind 模块）
pip install -e . --no-build-isolation

# 4) memfabric_hybrid（CPU KV 池后端；yq 环境同款）
pip install memfabric_hybrid   # 或按机器已有安装方式
```

## 3. 一条命令

```bash
python3 yq_files/sweep_median.py --processes 6 --e2e --echo -- \
    --batch-size 16 --max-seq-len 65536 --topk 2048 --q-heads 16 \
    --topk-buffer-size 4096 --plan-direct on \
    --warmup 50 --iters 100 --repeat 1 --kernel-time 200 --disable-gc \
    --hit-layout strided --plan-slots off,scattered \
    --hit-ratio 0.0000,0.0500,0.1000,0.1500,0.2000,0.2500,0.3000,0.3500,0.4000,0.4500,0.5000,0.5500,0.6000,0.6500,0.7000,0.7500,0.8000,0.8500,0.9000,0.9500,1.0000,0.0001,0.0499,0.1001,0.1501,0.2001,0.2501,0.2999,0.3501,0.4001,0.4501,0.5001,0.5499,0.6001,0.6501,0.7001,0.7999,0.8501,0.9001,0.9501,0.9999 \
    > 曲线.log 2>&1
```

84 点/进程 × 6 进程，实测约 5.3 s/点，约 45 分钟。改了脚本之后先空跑：

```bash
python3 yq_files/sweep_median.py --processes 3 \
    --bench yq_files/fake_bench.py -- --topk 2048 --hit-ratio 0.0,1.0 --plan-slots off,scattered
```

0.3 秒，不碰 NPU。**拿 NPU 时间调工具的 bug 太贵。**

### 每段参数为什么这么定

| 参数 | 理由 |
| --- | --- |
| `--hit-ratio` 前 21 个 | 0→100% 每 5% 一个点。这个算子 0% 到 100% 有 4 倍跨度，只在一个命中率上快说明不了什么 |
| `--hit-ratio` 后 21 个 | **A/A 复测标签**：`round(r×topk)` 与对应整点完全相同，只是名字不同、排在队尾。两次读数的差就是这趟自己的误差棒。标签必须在小数点后 4 位互不相同（`KERNEL` 行只印 4 位） |
| `--plan-slots off,scattered` | **两条路一趟里交错测**（plan 是最内层轴）。分两趟跑的话中间横着 2~4%，正好是要测的差值本身的量级 |
| `--topk-buffer-size 4096` | 槽位数不跟着 topk 走（生产默认 4096 而 topk 2048）。量成 = topk 会让命中显得比实际连续 |
| `--plan-direct on` | 控制区 `[5][6]` 走直接寻址，跟生产一致。关掉会退回块表寻址，那条路上 `slot >= topk` 的命中被静默判成未命中 |
| `--hit-layout strided` | 碎片化最重的一档。`prefix` 最友好，量出来偏乐观 |
| `--kernel-time 200` | 出 `KERNEL` 行（判据）。降它不省时间——耗时主项是 profiler 启停 |
| `--processes 6` | 串行跑 6 个进程取中位数，**这是耗时的主项** |
| `--disable-gc` | Python GC 收持续的税，关掉后 host API 平均耗时降 26~43% |

## 4. 判据：什么叫「变快了」

```bash
python3 yq_files/cmp_sweeps.py 基线.log 优化后.log --label-a 基线 --label-b 优化后
```

| | |
| --- | --- |
| **只看 `kernel_p50_us`** | 端到端不作判据——含约 160 μs 框架开销、还在漂 |
| **门槛 = 基线那趟 A/A 的两倍** | `cmp_sweeps.py` 自动算，不用记 |
| **⚠ 门槛只能取参照臂的 A/A** | 取两臂最大值的话，一个臂越烂判据越松 |
| **⚠ 任一臂自己 A/A > 2% 就直接不合格** | 连自己都复现不出来的数，跟谁比都没意义 |
| **两条路都要报** | `plan=off`（无 LRU，按名次判）和 `plan=scattered`（有 LRU，读外部计划）。别只报快的那条 |
| **形状也要看** | 只在一个命中率上快 ≠ 整条曲线快 |

⚠ **换任何一个条件就不可比**：卡号、进程数、topk、槽位数、bs、hit_layout、
plan_slots、--kernel-time。要跟自己的优化比，**基线也得用同样的参数在同一台
机器上重量一遍**。

## 5. 六条坑（原仓库用血泪换的，原样继承）

1. **⚠ 「流程报成功」不等于「跑的是你以为的东西」。** 换签名/换头文件的分支之间
   跳而不全量重编，驱动可能还在用旧 kernel。验证只能看 `.o` 的 md5 指纹。
2. **⚠ 验证增量编译生效，只能用活代码当探针**——被优化掉的死代码改了也不改变
   二进制。
3. **⚠ `csrc/` 是混合行尾**（主体 CRLF，selection 那几段裸 LF）。别用 Python 的
   `read_text()/write_text()` 改它，会把整文件规范化成 LF，32 行的改动变成
   1884 行重写。
4. **⚠ 一次只跑一趟。** 主机内存带宽是共用的，两趟同时跑会偏向性地抬高 miss
   重的那一端（实测 +6%），而那正是曲线的关键段。卡不冲突不代表不冲突。
5. **⚠ 多卡并行默认不可用，老实串行、固定一张卡。** 8 进程分绑 8 张卡时并行臂
   自己的 A/A 是 13.64%（串行 0.20%），且形状被改了。`--devices` 是给复验用的。
6. **⚠ 长任务别接 `| head`**——管道一关脚本吃 SIGPIPE 被杀在半路。要截断先落盘。

## 6. 出问题时

| 现象 | 查什么 |
| --- | --- |
| 数字跟上次差很多但代码没动 | `python3 yq_files/check_contention.py 本趟.log 干净的那趟.log` |
| 表里报「进程分成了几团」 | 中位数落在空档里，报的是一个没有任何进程测到的值，不能用一个数概括 |
| 表里报「只测了一次」 | 那些点没有 A/A 标签，绝对值不可用，补标签重跑 |
| 精度用例挂了 | 先看是不是只挂 `slots=4096` 那一半——比性能问题重要得多 |
| bench 直接报错 | ⚠ `full_q_actual_seq` 必须填**累加值**（旁边的 `full_kv_actual_seq` 反而不累加）。填错不报错，静默只算 1 个请求、只用 1 个核。bench 里有断言 |

## 7. 与原仓库（poolingDSA）的差别

- 环境准备换成 vllm-ascend 上游的 `setup.py build_aclnn` vendor 包流程；
  `env.sh`/`build_ops.sh` 是原仓库同名脚本的去容器化改写（写死的容器路径、
  memfabric/atb 段去掉，SOC_VERSION 兜底补 ascend910_93，编译入口换成
  build_aclnn + build_ext 两步）。
- bench 开头指向的精度用例路径改为本仓库的
  `tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py`。
- 其余（bench 主体、sweep_median、cmp_sweeps、fake_bench、check_contention、
  精度用例）与 `happy-math/yq/opt-all0908` 逐字节一致。
- 本仓库算子 schema 为 15 参（含 `selection_membership_map`、无 `timer`），
  bench 自适应签名，外部计划（LRU）路径会被测到。
