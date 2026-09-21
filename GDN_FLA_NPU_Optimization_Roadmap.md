# fla-npu GDN 算子优化空间分析与路线图

> 本文回答三个问题：fla-npu 当前版本在计算效率和硬件资源利用率上还剩哪些优化点；FlashQLA（Qwen 团队的 GPU 版 GDN kernel 库）做了哪些优化；这些优化里哪些 fla-npu 已经做了、哪些可以借鉴。最后给出按推荐执行顺序排列的优化路线表。
> 分析对象：`flash-linear-attention-npu` 仓，最新形态为 `perf/gdn-a5-ho-generalized-pr` 分支（含 H/O 块级流水）。目标硬件：昇腾 950（arch35）。
> 关联文档：`GDN_Three_Repo_Comparison.md`（三版对照）、`GDN_Implementation.md` §9（vllm-ascend 侧账本）。

---

## 第一部分：fla-npu 现状——计算效率与硬件资源利用率

### 1.1 已经做到的

fla-npu 是三个 NPU 实现里融合最深的版本，已落地的优化：

- **单核启动**：整条 prefill 流水（cumsum → KKT → Solve → Recompute → FwdH → FwdO）收进一个 `chunk_gdn_core_fwd` 核启动（vllm-ascend 是 8 次）；
- **布局转置消除**：中间量直接按消费方需要的布局产生，无 5 张量拷贝；
- **②③④ 融合**：Phase A-C 在同一核内，中间量走核内 workspace；
- **H/O 块级流水**（perf 分支）：FwdH 完成一个 (块,头) 即 IBSet 置位，FwdO 的 12 个消费核组 IBWait 消费——从全局屏障改成 16 生产者组 : 12 消费者组的块级接力；
- **cumsum ‖ KKT 跨核并行起步**：AIC 算 K·Kᵀ 的同时 AIV 算 cumsum；
- **SolveTri 双实现**：Cube 版和 Vector 版，16/32/64 尺寸 dispatch；
- **flag 代数复用**：相位间不复位 flag，按代数交接，减少同步开销；
- **fwd_prepare 算子**（演进中）：把 L2Norm + 门控 + cumsum + KKT + SolveTri + RecomputeWU 再收一层。

### 1.2 三块"地板"——逼近程度评估

| 地板 | 含义 | fla-npu 逼近程度 |
|---|---|---|
| **I/O 带宽地板** | 必读 q/k/v/g/β + 必写 o/finalState ≈ 1~1.5 KB/token·头，不可省 | **未到**——workspace 中转是否被 L2 吸收是关键未知数 |
| **跨核通信地板** | MIX_AIC_1_2 下 AIC/AIV 的 UB 私有，Cube↔Vector 每次接力必经 GM workspace | **架构性地板**——不换核型无法消除，只能减少接力次数和优化 L2 命中 |
| **小 tile Cube 效率地板** | BT=64 决定矩阵乘形状是 64×128×64 级别，Cube 深流水吃不满 | 部分逼近——TileShapes128/256 已做，BT=64 本身限制了上限 |

### 1.3 当前 bound 与剩余优化点

PR #492 自报的 mte2_ratio = 46.1%（搬运管道忙碌率近半、未打满）说明：**算子仍是搬运 bound，但既不是纯带宽极限、也还有重叠空间**。结合对 csrc 的走读，剩余优化点按维度分：

**搬运维度：**

- **workspace 的 L2 亲和**：跨核中转（A、w、u、h、vNew）是"写完立刻读"的热数据，理应命中 L2 而非真打 HBM——slab 按核组切分、地址对齐 L2 行、生产者-消费者对物理相邻；
- **vNew 免写**：vNew 是 u 的派生量，只在 FwdO 被消费——能否免物化（直接从 u 侧重建）省 256 B/token·头；
- **A 免落盘**：SolveTri 的输出 A 只被 RecomputeWU 消费——`chunk_gdn_fwd_prepare`（已在演进）正是把这条链收进一个核，A 只在 UB/L2 过一手；
- **上游流水**：A→C→D 之间仍是全局屏障（只有 D→E 流水化了）。

**计算维度：**

- **MTE1 预取深度**：PING_PONG_STAGES=2，UB 从 247KB 压到 148KB 后有余量加深到 3 级；
- **Fixpipe 附带计算**：L0C→UB 直通路径上捎带 scale/cast 类操作（950 的 Fixpipe epilogue 能力待验证）；
- **VF 链融合**：门控链（softplus→exp→乘）、掩码链（比较→选择→exp）逐条检查是否全部 VF 化（部分已做，如 OutputFusedVf、BRC_B32）。

**利用率维度：**

- **空闲 AIV 填充**：FwdO 的 Cube 拍（三个矩阵乘）期间，组内第二个 AIV 大概率闲置——偷跑下一块的掩码预生成/g 预载；
- **相位内角色失衡**：Solve 偏 Vector 时 AIC 空闲、FwdH 的 Cube 拍期间 AIV 有空窗——跨拍偷跑的调度空间；
- **生产者:消费者比例**：16:12 为常数，形状变化时未必最优（HoPipelineContext 已支持 tiling 下发，缺选择策略）；
- **串行深度**：SolveTri 前代换 15 步（倍增法可到 6 步，但属实验项）。

---

## 第二部分：FlashQLA 做了哪些优化

FlashQLA 是 Qwen 团队发布的 **NVIDIA GPU（Hopper/Blackwell）上的 GDN kernel 库**，基于 TileLang 编写，作为 FLA 官方库的 GDN 后端，宣称对 FLA Triton 基线前向 2-3×、反向 2×。它有 README 列出的三个 Key Feature，以及代码里可见的若干额外优化。

### 2.1 Key Feature 1：门控驱动的自动卡内 CP

**问题**：Delta 规则递推跨块依赖初态——多卡/多组切分序列后，每段需要前段终态当初态，产生通信和等待（vllm-ascend 的 PCP 用"两轮 all_gather + 线性修正"解决）。

**FlashQLA 的做法**：利用衰减的指数遗忘特性——扫描每头 g 的块内累计值，找出"初态影响已衰减到阈值以下"的**前缀块（warm-up chunks）**。这些块的初态是 50 还是 30 已不影响结果（低于精度容差），**直接跨 CP 组并行计算，免任何修正通信**；只有衰减不够深的边界块才走精确修正（`correct_initial_states/correct_terminal_states`）；衰减太慢的序列自动回退（`fallback_mask`）。

**效果**：衰减快的序列几乎全部块免修正、并行度拉满；衰减慢的自动降级。通信量从"全量终态 all_gather"降到"仅边界块"。

**与 PCP 的关系**：PCP 是精确修正（任意切分、一次到位）；门控 CP 是阈值近似（数据依赖的自适应并行）。阈值设在精度容差内时结果实际无损。两者可组合：warm-up 段免修正、边界段走 PCP。

### 2.2 Key Feature 2：硬件友好的代数重写

- **衰减矩阵的对角化**：块内 64×64 的衰减矩阵不逐元素算，而是**预计算 64 个 exp，用两次广播对角乘拼出 G 矩阵**（`G = diag(exp(g_i)) · 全1下三角 · diag(1/exp(g_j))`）——把逐元素特殊函数从热点路径挤出去；
- **全程 exp2**：`exp2(x × 1/ln2)` 全局替代 exp（GPU 上 exp2 是原生快指令）；
- **净写入的 FMA 形式**：检索回声的扣除写成 `u ×= −γ; u += v`——让乘加融合成单指令，少一次中间寄存器写。

### 2.3 Key Feature 3：TileLang 融合核 + warp specialization

**一个核启动 512 线程，按 threadIdx 分成六种工人**，用双缓冲 SMEM + 8 组 barrier 互锁成流水线：

| 线程组 | 角色 | 干什么 |
|---|---|---|
| 0-127 | 状态工人 | **记忆矩阵 S 常驻寄存器（fragment），跨块不下片**：逐块"衰减 + Kᵀ·净写入累加" |
| 128-255 | 输出工人 | 每块两路输出（q·S 块间读出 + 块内注意力）**在寄存器里相加后一次写出** |
| 256-383 | u/w 工人 | 每块算检索回声（k·S）和净写入（v − 回声） |
| 384-416 | 装载工人 1 | TMA 异步预取下一块的 q、k |
| 416-448 | 装载工人 2 | TMA 预取 v、β |
| 448-480 | 装载工人 3 | TMA 预取 A（kkt_solve 产出的结算矩阵） |

流水效果：**装载第 i+2 块、u/w 算第 i+1 块、状态推进和输出算第 i 块——三类硬件单元（TMA/Tensor Core/CUDA Core）同时忙碌**。

**最深的一项**：状态 `h_fragment` 从第一块到最后一块**常驻寄存器**，中间零显存往返（对比：NPU 三版的 FwdH 里状态驻 AIV 的 UB——部分等效，但 FwdO 消费 h 时 NPU 必须经 GM workspace，GPU 在 CTA 内走 SMEM 直通）。

### 2.4 三个 Key Feature 之外的优化

- **核数量 7 → 3，且边界按依赖结构选**：cumsum（小核）、kkt_solve（②+③ 合体）、fused_fwd（④+⑤+⑥ 合体）。**求逆独立成核**——15 步行依赖会卡流水，塞进大融合核不划算；fused_fwd 通过 TMA 载入现成的结算矩阵。融合边界选在"结算矩阵"这个天然结算点上；
- **分层求逆上 Tensor Core**：kkt_solve 内部，4 个对角 16×16 用 `T.unroll` 完全展开的前代换（编译期展开、无循环开销），16→32→64 两级合并用 **T.gemm（Tensor Core）** 做 Schur 补——串行前代换限制在最小范围，块间合并全部上矩阵乘；
- **反向同等融合 + 反向 CP**：`fused_bwd` 同样 warp specialization，且反向支持卡内 CP——训练链路整体 2×，不是只优化 forward；
- **TMA 异步装载 + 双缓冲 SMEM**：装载延迟被两级缓冲完全隐藏，装载完成自动 barrier 通知消费者，生产者线程零开销；
- **元数据缓存**：`tensor_cache` 装饰器缓存 cu_seqlens/chunk_offsets，避免每步重分配；
- **接口对齐**：与 FLA 最新接口同名同参，用户零改动切换（生态卡位）。

---

## 第三部分：FlashQLA 的优化 fla-npu 做了哪些、哪些可借鉴

| FlashQLA 的优化 | fla-npu 现状 | 可借鉴性 |
|---|---|---|
| 核数量收敛（7→3） | **已超越**——收成 1 个核 | 无需借鉴，fla-npu 走得更远 |
| warp specialization（六角色单核流水） | **部分等价**——MIX_AIC_1_2 分 Cube/Vector 阶段 + H/O 块级流水，但粒度是"相位接力"而非"逐块交错" | ⭐ **可借鉴（fla-npu 下一个最大单项）**：把 FwdO 的 Cube 拍折进 FwdH 的 AIC 循环逐块交错，免 h/vNew 的 workspace 往返和相位切换 |
| 状态常驻计算单元 | **部分等价**——FwdH 内状态驻 AIV 的 UB ping-pong 跨块不下片；但 FwdO 消费 h 必经 GM workspace | 架构限制（AIC/AIV UB 私有），只能靠减少交接次数逼近（同上条）；GPU 的 CTA 内 SMEM 共享无法 1:1 移植 |
| 分层求逆上 Tensor Core（16 展开 + gemm 合并） | **未做**——SolveTri 是 Vector 前代换（双实现 + 尺寸 dispatch，无 Tensor Core 合并） | ⭐ 可借鉴：gemm 化的 Schur 合并 = 优化 C 的落地形态；先 profile 确认 Solve 是否相位瓶颈 |
| 门控驱动的卡内 CP | **未做**（也无 PCP） | ⭐ **可借鉴**：warm-up 判定（扫描 gcum 找衰减充分段）在 NPU 上实现不难，且可作为 vllm-ascend PCP 的预筛组合 |
| exp2 替代 exp | ✅ 已做（use_exp2 路径，RCP_LN2 常量在 prepare 设计文档中） | — |
| 衰减矩阵对角化重写 | **未确认**——② 的 exp 逐元素写法未做对角化重排 | 可借鉴（中等收益，G 矩阵两次广播替代逐元素） |
| TMA 异步装载 + 多生产者 | NPU 对应物是 MTE 双缓冲（已有） | 机制不同，无需借鉴 |
| 反向同等融合 + 反向 CP | **未做**——反向是独立算子套件（bwd_dqkwg/dv_local/dhu/prepare_wy_repr 已 AscendC 化但未融合） | 可借鉴：正向 Phase6 的编排经验直接复用 |
| 元数据缓存 / 接口对齐 | ✅ 已有（CPU 预计算 + ctypes 壳） | — |

**总结**：FlashQLA 的三个 Key Feature 里——Feature 3（warp specialization）fla-npu 做了粗粒度版、**细粒度逐块交错是剩余最大空间**；Feature 1（门控 CP）fla-npu 完全没有、**是独有可借鉴项**；Feature 2（代数重写）大部分已覆盖。此外 FlashQLA 验证了两件我们已分析的事：求逆独立成核是合理形态（佐证 fla-npu 的 SolveTri 独立算子）、gemm 化合并是求逆的提速正解（佐证优化 C）。

---

## 第四部分：fla-npu 还值得优化吗——结论与路线表

**结论：值得，但空间结构已经变了。**

- **低垂果实已摘完**：核启动、布局转置、②③④ 融合、D→E 流水、跨核并行起步——清单上的结构性项目全部落地；
- **剩余空间是三类"精装修"**：流水深度（A→D 全链化）、局部性（workspace 的 L2 亲和）、角色填充（空闲核偷跑）——单项收益从 20-30% 降到 3-15%，**但加总估计仍有 20-40%**；
- **有一个决定性未知数**：workspace 的 L2 命中率。它决定"搬运类"剩余优化的真实上限——**这是所有后续工作之前必须先 profile 的数字**；
- **另有一个方向性机会**：FlashQLA 验证过的两条（逐块交错、门控 CP）在 NPU 上没有对应实现，属于"已被别人证明可行、等价迁移有路可循"的确定增量。

### 推荐执行顺序表

| 顺序 | 优化项 | 优化位置 | 收益原理 | 预期结果 |
|---|---|---|---|---|
| 0 | **AscendTimerV2 打点基线**（照 fused_sparse_attention_overlap 的 AscendTimerV2.hpp 模式） | RunPhase6 相位边界 + FwdH 四拍/FwdO 五拍内部 + workspace 读写两侧；动态项 iter=块号 | 把"相位耗时、flag 空泡、workspace 税、L2 命中率"四个未知数全部变成数字——后续所有项的取舍依据 | 一份按相位/按拍/按核分解的耗时报告（决定 1-7 的取舍与排序） |
| 1 | **H/O 流水门槛泛化** | `CanRunChunkPipeline`（chunk_gdn_core_fwd）：去掉 seqlen==11274、kNumHead==16 等硬编码，改为 HoPipelineContext 按 tiling 资格判定（GEN-v2 布局门已开一半） | 已验证的流水机制目前只覆盖特调形状；泛化后收益覆盖全部部署形状 | 非 11274 形状场景 10-20%（机制无新风险，纯放开） |
| 2 | **workspace 的 L2 亲和布局** | workspace 分配：按 AIC/AIV 核组切 slab、生产者-消费者对物理相邻、对齐 L2 行 | 跨核中转是"写完立刻读"的热数据——命中 L2 则免真打 HBM | 视基线命中率：搬运时间 10-30%（命中率已高则此项收益小） |
| 3 | **A→C→D 全链块级流水** | 系数生成与 FwdH 之间的交接复用 IBSet/IBWait 机制（照 H/O 流水模式上推一层）；fwd_prepare 落地后接力点只剩两个 | 隐藏相位串行延迟：块 0 的 w/u 一出，FwdH 块 0 即起步，不必等全部系数生成 | 块数多场景 10-20%；短序列改善延迟地板 |
| 4 | **FwdO 折进 FwdH 逐块交错** | FwdH 的 AIC 循环内：C2（kᵀ@v_update）之后就地折入 FwdO 的 q·h 与 attn·v_new 两拍 | 免 h/vNew 的 workspace 往返（每块·头 ~48 KB）+ 免 D→E 相位切换 | 相关流量段 -100%，端到端 10-15%；依赖 0/3 的 profile 数据支持 |
| 5 | **生产者:消费者比例自适应** | HoPipelineContext 的组数下发策略：按 T/块数/头数查表选 16:12 或其它配比 | 流水化之后两类核的负载随形状变化，固定比例必有一侧闲置 | 形状相关 5-10% |
| 6 | **空闲 AIV 填充** | FwdO 的 Cube 拍期间，组内第二个 AIV 偷跑下一块的掩码预生成/g 预载 | 白捡的并行度——AIV 在 Cube 重拍期间闲置 | 3-8% |
| 7 | **门控 warm-up CP**（多卡场景） | NPU 版 PCP 增加预筛：扫描 gcum 找衰减充分段，免修正段跳过 all_gather 与重跑 | 衰减遗忘使跨界修正可省（阈值在精度容差内） | 多卡长序列 prefill 的通信与等待显著下降；衰减慢序列自动回退 |
| 8 | **求逆倍增实验**（可选） | SolveTri：`(I+L)⁻¹ = (I−L)(I+L²)···(I+L³²)`，6 次 64³ 矩阵乘 | 串行深度 15→6；验证 Cube 倍增 vs Vector 前代换哪个快 | 实验项：③ 本步 1.5-2× 或负收益（视批量摊薄现状），以 0 的数据决定 |

### 执行注意

1. **收益叠加是非线性的**：1/3/4 都在缩短同一关键路径，做完 1 再估 3/4 的剩余空间，不要按表线性外推；
2. **精度红线**：所有涉及数据格式的项（workspace dtype、状态精度、门控 CP 的阈值）必须过 GSM8K/端到端对齐；
3. **顺序 0 不是可选项**：没有 L2 命中率和相位空泡的实测数字，2/3/4 的预期收益都只是假设——参考 `fused_sparse_attention_overlap` 的 AscendTimerV2 打点方案（AIC/AIV 分核计时、动态项按块号、NoBarrier 版防扰动），搬进 `chunk_gdn_core_fwd` 即可。