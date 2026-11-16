# Stage 2：单机 PD + Sparse KV Offload（decode 跑 FSAO 算子）

## 拓扑

```text
P (prefill)  NPU 0-7   TP8/EP8   port 8100   kv_port 20000
D (decode)   NPU 8-15  TP8/EP8   port 8200   kv_port 20050-20057
proxy        port 9000（layerwise proxy）
```

- decode 侧：`sparse_kv_offload_config.enabled=true` + `use_fused_overlap=true` → 走
  `npu_fused_sparse_attention_overlap`；主 KV 必须 **BF16**，不要传 `--kv-cache-dtype`；
  不要开 `keep_device_kv_cache`（PD consumer 下会断言）。
- 两种 prefill 连接方式：
  - **A（默认，组件最少）**：prefill 只用 `SfaRemoteD2HConnector` producer，把 KV buffer
    直接暴露给 decode。不需要 Memcache。
  - **B（文档完整组合）**：`USE_LAYERWISE_PREFILL=1`，prefill 用 `MultiConnector`
    （`SfaRemoteD2H` + `AscendStore/memcache` layerwise），需要先启 Memcache。

## 前置检查

```bash
python3 -c "import memcache_hybrid"          # 只有 B 需要
free -g                                       # decode 侧 DRAM_GB 要放得下
npu-smi info                                  # 0-15 全空闲
```

端口：8100、8200、9000、20000-20007、20050-20057（B 还要 5000、6000）。

## 单轮 smoke（建议先用 4K）

```bash
cd /home/z00943842/base/pd_stage2
chmod +x *.sh

export MAX_MODEL_LEN=8192 MAX_NUM_SEQS=8 P_MAX_NUM_SEQS=8 \
       P_MAX_BATCHED=8192 DRAM_GB=32

./stop.sh
./start_prefill.sh
./wait_ready.sh 8100 3600
./start_decode.sh
./wait_ready.sh 8200 3600
./start_proxy.sh
./wait_ready.sh 9000 120
./status.sh
```

发送一条请求（注意服务名）：

```bash
curl -s localhost:9000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

确认算子路径生效：

```bash
grep -iE "sparse KV offload|fused_overlap|SfaRemoteD2H" logs/decode.log | head
```

## 压测

工具（`~/aisbench_auto_tools_prefix/config.py`）先改成：

```python
MODEL_NAME = "glm-5.1"
HOST_PORT  = "9000"
```

然后：

```bash
source ~/aisvenv/bin/activate
cd ~/aisbench_auto_tools_prefix
python3 aisbench_test.py \
  --input_len 4096 --output_len 256 --data_num 4 --concurrency 4 \
  --request_rate 0 --npu_num 16 \
  --dataset_type prefix_cache --repeat_rate 0 --seed 2
```

## 正式场景

- **32K / 64K 才是算子的目标区间**。4K 下 `topk=2048` 已经覆盖一半上下文，稀疏退化成近似稠密，
  算子的行为/收益都不代表 32K+。
- 64K 时 prefill 侧 KV 在 device：`MAX_MODEL_LEN=67840 P_MAX_NUM_SEQS=1 P_MAX_BATCHED=4096`；
  decode 侧 KV 在 host：`MAX_NUM_SEQS=4 DRAM_GB=110`（按 host 内存调整）。

## 算子 A/B

前提：工具 `config.py` 指向 proxy(9000)，`$TOOLS` 下有原来的 `swap_ops.sh`/`env.sh`。

```bash
cd /home/z00943842/base/pd_stage2
PD_INPUT_LEN=32768 PD_OUTPUT_LEN=1024 PD_DATA_NUM=8 PD_CONCURRENCY=8 \
  ./abba_pd.sh
```

每轮流程：`swap_ops.sh` 换包 → 起 P → 起 D → 起 proxy → 压测 → 存 CSV → 停。结果在
`$AB/results_pd`。

## 常见问题

| 现象 | 处理 |
|---|---|
| `Sparse KV offload offload.initialize failed` | host 连续内存不够：降 `DRAM_GB`/DP，或按 memfabric 文档准备大页 |
| `assert ... PD disaggregate ... D node` | 开了 offload 但连接器不是 consumer，或 `keep_device_kv_cache=true` |
| `packed SFA main caches` | 传了 `--kv-cache-dtype int8/fp8`；关掉，主 KV 必须 BF16 |
| decode 起不来 | 先去掉 `--async-scheduling`；再不行降 `MAX_MODEL_LEN` / `MAX_NUM_SEQS` |
| proxy 502 / KV 等待超时 | 检查 20000/20050 端口区间、`memfabric_transfer_protocol` 两侧一致（sdma） |
| B 模式 memcache 起不来 | 确认 `memcache_hybrid` 存在；5000/6000 有监听 |
