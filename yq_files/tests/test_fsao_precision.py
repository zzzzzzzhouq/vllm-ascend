# 这条线（上游 15 参数 + LRU）的精度用例**正本**。直接改这里。
#
# 2026-08-12 换基线之前它是 tests/ut/attention/a2/test_fused_sparse_attention_
# overlap_precision.py 的副本，靠 yq_files/tests/sync_from_tests.sh 同步。
# 换基线之后那个方向反了：**新上游把 a2 那个用例删掉了**（换成
# tests/ut/test_fused_overlap_vs_lidu_perf.py，只测时间、没有任何数值校验），
# 而 yq/upstream-0723 现在是基线分支，所以这一份就是正本，同步脚本已从这条线删掉。
# 历史线（yq/overlap-base0723，14 参数版）那边的 a2 文件各自冻结，两边不再同步。
#
# 用例本身两种签名都认：开头按 `_OP_SCHEMA` 判有没有 selection_membership_map，
# 15 参数版会在位置 5 补上它，并按 16400 int16 的正确行宽分配（传窄了会写穿）。
#
# 两条路都验：
#   回退路（按名次判）—— membership 区填 −1、控制区标记不写
#   LRU 计划路        —— _write_plan 按生产的样子填控制区八格，见那个函数
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Precision coverage for ``npu_fused_sparse_attention_overlap``.

Mirrors ``test_sfa_v1_precision.py``: an fp32 dense MLA baseline, an
``assert_close`` against it, and the same signal-relative error budgets.

Two properties are checked separately because the kernel takes two different
paths through the selection cache:

* **miss path** -- the selection cache starts cold, so the kernel has to pull
  every topk block out of the CPU-side MemFabric pool before it can compute.
  This is what the fp32 baseline is compared against.
* **hit path** -- once the cache is warm, replaying the same call with every
  slot marked as a hit must reproduce the miss-path output. This catches
  selection-cache bookkeeping bugs that the miss path alone cannot see.

Note that the op mutates five of its inputs (``Tensor(a!)`` through
``Tensor(e!)``). Every call here is therefore preceded by a full reset of the
selection state -- without it the second call runs against a warm cache and
the result stops being reproducible.
"""

import math
import os
import sys
from collections import Counter

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401  (registers the NPU device)
import vllm_ascend.vllm_ascend_C  # noqa: F401  (registers the custom ops)
from memfabric_hybrid import offload

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Metrics log: survives pytest's stdout capture, path overridable via env.
_METRICS_LOG_PATH = os.environ.get(
    "FSA_OVERLAP_PRECISION_METRICS_LOG",
    "/tmp/fsa_overlap_precision_metrics.log",
)
with open(_METRICS_LOG_PATH, "w", encoding="utf-8"):
    pass


def _emit_metric(line: str) -> None:
    """Append ``line`` to the metrics log and echo it to stderr."""
    with open(_METRICS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass


fused = torch.ops._C_ascend.npu_fused_sparse_attention_overlap

# The operator ships in two shapes and they differ by a positional argument.
# This branch's baseline (mte_fuse_0723) takes 14; upstream
# szy_fuse_new_token_adapt and everything built on it insert
# selection_membership_map at position 5, making 15. Passing 14 to the
# 15-argument build would silently shift every later argument by one, so ask
# the schema rather than assume.
_OP_SCHEMA = str(fused.default._schema)
_OP_TAKES_MEMBERSHIP = "selection_membership_map" in _OP_SCHEMA

# Same budgets as test_sfa_v1_precision.py.
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2
_MAX_SIG_REL_ERR = 1e-2  # max |out-ref| / peak |ref|
_MAX_MEAN_SIG_ERR = 5e-3  # mean |out-ref| / mean |ref|
_MAX_REL_ERR = 1e-2  # max per-element rel err where |ref| >= floor
_SIG_FLOOR_FRAC = 5e-1  # floor = this fraction of peak |ref|

# MLA dims, matching the DeepSeek-style sparse MLA this op targets.
_BLOCK_SIZE = 128
_KVD = 512  # kv_lora_rank / nope width
_KRD = 64  # rope width
_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
_GIB = 1024 * 1024 * 1024
# Host pool for the whole module. The baseline shapes need max_seq_len=65536,
# which is 72 MiB of full KV + rope per case, and the two baseline tests build
# 16 cases (8 batch sizes + 8 hit ratios). 1.2 GiB if MemFabric does not hand
# anything back between cases, against a pool that used to be 2.0 GiB -- too
# close to the edge to leave alone. Raised with the arithmetic written down so
# the next person adding cases can redo it rather than guess.
_DRAM_POOL_GB = float(os.environ.get("DSA_DRAM_POOL_GB", "6"))

# CPU/host KV 后端：
#   offload（默认）= MemFabric offload.empty + 2MiB 对齐（生产同路径）
#   pinned         = 纯 pinned 宿主内存（torch.empty(pin_memory=True)），不经过
#                    MemFabric/HAL。宿主上 MemFabric 池起不来时（HYBM 的
#                    HalMemSetAccess 失败）用它，算子语义不变——它只需要钉住的
#                    宿主 KV，不需要池本身。
_CPU_KV_BACKEND = os.environ.get("DSA_CPU_KV_BACKEND", "offload").strip().lower()

# selection_membership_map row stride, mirroring the kernel constants in
# fused_sparse_attention_overlap_service_vector_mla.h:
#   MEMBERSHIP_MAX_TOKEN / MAP_INT16_COUNT = 16376
#   ALIGNMENT_INT16_COUNT = BYTE_BLOCK / sizeof(int16) = 32 / 2 = 16
#   CONTROL_OFFSET = ceil(16376 / 16) * 16           = 16384
#   CONTROL_INT16_COUNT                               = 8
#   STORAGE = ceil((16384 + 8) / 16) * 16             = 16400
#
# Init() raises the row stride to at least STORAGE whatever tensor it is
# handed, so anything narrower is written past its end. The timing UT passes
# an 8-wide placeholder and gets away with it only because it never reads its
# own output back -- here the overrun corrupted an unrelated tensor and
# surfaced as an all-NaN result from a call that had already completed.
_MEMBERSHIP_MAX_TOKEN = 16376
_MEMBERSHIP_ALIGNMENT_INT16_COUNT = 16
_MEMBERSHIP_CONTROL_INT16_COUNT = 8
_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT = (
    (_MEMBERSHIP_MAX_TOKEN + _MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1) // _MEMBERSHIP_ALIGNMENT_INT16_COUNT
) * _MEMBERSHIP_ALIGNMENT_INT16_COUNT
_MEMBERSHIP_INT16_COUNT = (
    (_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT + _MEMBERSHIP_CONTROL_INT16_COUNT + _MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1)
    // _MEMBERSHIP_ALIGNMENT_INT16_COUNT
) * _MEMBERSHIP_ALIGNMENT_INT16_COUNT


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def _empty_pinned_cpu_tensor(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    """Allocate host KV the same way KVOffloadDecodeManager does.

    Raw int8 bytes, 2MiB-aligned, then viewed as ``dtype``. Source depends on
    ``DSA_CPU_KV_BACKEND``: MemFabric ``offload.empty`` (default) or plain
    pinned host memory (no MemFabric/HAL involvement).
    """
    nbytes = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    if _CPU_KV_BACKEND == "pinned":
        raw = torch.empty([nbytes + _CPU_CACHE_ALIGNMENT], dtype=torch.int8, pin_memory=True)
    elif _CPU_KV_BACKEND == "offload":
        raw = offload.empty([nbytes + _CPU_CACHE_ALIGNMENT], dtype=torch.int8, pin_memory=True)
    else:
        raise RuntimeError(
            f"unknown DSA_CPU_KV_BACKEND={_CPU_KV_BACKEND!r} (use 'offload' or 'pinned')"
        )
    return _align_memory(raw, _CPU_CACHE_ALIGNMENT)[:nbytes].view(dtype).view(shape)


@pytest.fixture(scope="module")
def offload_pool():
    """Bring up the MemFabric CPU KV pool for the whole module."""
    if _CPU_KV_BACKEND != "offload":
        print(f"CPU_KV_BACKEND={_CPU_KV_BACKEND}: skip MemFabric pool init", flush=True)
        yield
        return
    config = offload.OffloadConfig()
    config.device_id = torch_npu.npu.current_device()
    config.size = int(_DRAM_POOL_GB * _GIB)
    for name, value in (("world_size", 1), ("rank_id", 0)):
        if hasattr(config, name):
            setattr(config, name, value)
    try:
        offload.initialize(config)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"MemFabric offload unavailable: {type(exc).__name__}: {exc}")
    try:
        yield
    finally:
        offload.uninitialize()


def _build_case(
    topk: int,
    q_heads: int,
    dtype: torch.dtype,
    bsz: int = 1,
    max_seq_len: int = 0,
    topk_buffer_size: int = 0,
) -> dict:
    """Build one decode-shaped case: q on NPU, full KV in the CPU pool.

    Layout follows the timing UT so both exercise the same kernel path:
    one query token per request, kv_heads=1, ``selection_topk_block_size=1``.

    ``bsz`` and ``max_seq_len`` default to what this file used before they were
    parameters, so the cases that predate them keep their data. Pass them to
    reach the shapes the two baseline sweeps actually run at -- bs 1..128,
    topk 2048, max_seq_len 65536. See docs/命中率曲线-mte_fuse_0723基线/ and
    docs/BS曲线-mte_fuse_0723基线/.
    """
    torch.npu.set_device(0)
    # (bsz - 1) rather than bsz, so the single-request cases generate exactly
    # the data they generated before this parameter existed.
    seed = 20260804 + topk + q_heads + (bsz - 1) * 101
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

    seq = kv_heads = 1
    stbs = 1
    max_seq_len = max_seq_len or max(topk * 4, 256)
    full_blocks = _cdiv(max_seq_len, _BLOCK_SIZE)
    # 一行 selection 缓存的槽位数。0 = 等于 topk，是这个文件原来的行为；
    # 生产是 kv_offload_decode_config.topk_buffer_size，默认 4096 而 topk 常是
    # 2048，槽位是 topk 的两倍。算子只认块表最后一维，递宽一点就按宽的走。
    if topk_buffer_size:
        if topk_buffer_size % _BLOCK_SIZE or topk_buffer_size < topk:
            raise ValueError(
                f"topk_buffer_size 必须是 {_BLOCK_SIZE} 的整数倍且 >= topk，得到 {topk_buffer_size}"
            )
        sel_blocks = topk_buffer_size // _BLOCK_SIZE
    else:
        sel_blocks = _cdiv(topk * stbs, _BLOCK_SIZE)
    scale = 1.0 / math.sqrt(_KVD)

    q_nope = torch.randn(bsz * seq, q_heads, _KVD, dtype=dtype, device="npu")
    q_rope = torch.randn(bsz * seq, q_heads, _KRD, dtype=dtype, device="npu")
    query = torch.cat([q_nope, q_rope], dim=-1).contiguous()

    # Host-side KV must hold real values: a zero-filled pool would make the
    # baseline trivially zero and hide any fetch bug.
    full_kv = _empty_pinned_cpu_tensor([full_blocks, _BLOCK_SIZE, _KVD], dtype)
    full_rope = _empty_pinned_cpu_tensor([full_blocks, _BLOCK_SIZE, _KRD], dtype)
    full_kv.copy_(torch.randn(full_kv.shape, dtype=torch.float32).to(dtype))
    full_rope.copy_(torch.randn(full_rope.shape, dtype=torch.float32).to(dtype))

    # Identity block table: physical block id == logical block id.
    full_bt = torch.arange(full_blocks, dtype=torch.int32, device="npu").unsqueeze(0).expand(bsz, -1).contiguous()
    # ⚠ These two lines look alike and are not the same formula.
    #
    # query is TND, so full_q_actual_seq is CUMULATIVE: entry b is where
    # request b's tokens end, and the last entry must equal bsz*seq. The KV
    # side is PA_BSND, not TND, so full_kv_actual_seq is NOT cumulative -- one
    # plain length per request, read straight out at index b.
    #
    # [seq]*bsz for the query side is what this file had while bsz was pinned
    # at 1, where the two formulas coincide. It is silent when they do not:
    # every difference after the first is zero, so requests 1..bsz-1 get zero
    # query tokens and never run, while the call still reports a bsz-request
    # shape. The timing UT shipped exactly that and it cost a day of baseline.
    actual_q = torch.tensor([(i + 1) * seq for i in range(bsz)], dtype=torch.int32, device="npu")
    actual_k = torch.tensor([max_seq_len] * bsz, dtype=torch.int32, device="npu")
    assert int(actual_q[-1].item()) == bsz * seq, (
        "full_q_actual_seq must be cumulative and end at the total token count: "
        f"got {int(actual_q[-1].item())}, expected {bsz * seq}"
    )

    topk_np = np.zeros((bsz * seq, kv_heads, topk), dtype=np.int32)
    for row in range(bsz * seq):
        for kh in range(kv_heads):
            topk_np[row, kh] = np.sort(np.random.choice(max_seq_len, topk, replace=False).astype(np.int32))
    topk_indices = torch.tensor(topk_np, dtype=torch.int32, device="npu").reshape(bsz, seq, kv_heads, topk).contiguous()

    total_sel = sel_blocks * bsz * seq * kv_heads
    return {
        "topk": topk,
        "q_heads": q_heads,
        "bsz": bsz,
        "dtype": dtype,
        "scale": scale,
        "stbs": stbs,
        "max_seq_len": max_seq_len,
        "query": query,
        "full_kv": full_kv,
        "full_rope": full_rope,
        "full_bt": full_bt,
        "actual_q": actual_q,
        "actual_k": actual_k,
        "topk_indices": topk_indices,
        "sel_kv": torch.zeros(total_sel, _BLOCK_SIZE, _KVD, dtype=dtype, device="npu"),
        "sel_rope": torch.zeros(total_sel, _BLOCK_SIZE, _KRD, dtype=dtype, device="npu"),
        "sel_bt": torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(bsz * seq * kv_heads, sel_blocks),
        "sel_status": torch.full((bsz, seq, kv_heads, topk + 1), -1, dtype=torch.int32, device="npu"),
        "sel_membership": (
            torch.full((total_sel, _MEMBERSHIP_INT16_COUNT), -1, dtype=torch.int16, device="npu")
            if _OP_TAKES_MEMBERSHIP
            else None
        ),
        "_sel_bt_pristine": torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(
            bsz * seq * kv_heads, sel_blocks
        ),
    }


def _reset_selection_state(case: dict) -> None:
    """Restore every mutated input to its cold-cache state.

    The op declares selection_k_rope, selection_kv_cache,
    selection_kv_block_table, selection_kv_block_status and, where the build has
    it, selection_membership_map as mutable. Skipping this makes a second call
    run against whatever the first one left behind.
    """
    case["sel_kv"].zero_()
    case["sel_rope"].zero_()
    case["sel_bt"].copy_(case["_sel_bt_pristine"])
    case["sel_status"].fill_(-1)
    if case["sel_membership"] is not None:
        case["sel_membership"].fill_(-1)


_HIT_LAYOUTS = ("prefix", "strided")


def _hit_positions(k: int, n_hit: int, layout: str) -> torch.Tensor:
    """哪些槽位标成驻留。

    - ``prefix``：前 ``n_hit`` 个。命中连成一片，计算流的配对网格上几乎每一对
      要么两个槽位都命中、要么都不命中，**混合对每行最多出现一次**（``n_hit``
      是奇数时才有）。
    - ``strided``：命中尽量均匀铺开，公式与计时 UT 的同名布局一致。命中率 50%
      时命中与未命中在相邻槽位上交替，**每一对都是混合对**。

    两种布局的命中率相同，正确性要求也完全相同，但走的搬运路不同：整对命中成段
    读缓存，混合对逐行搬，两个都未命中才走配对搬运。维护层给未命中的槽位算源行
    时要按 topk 值预测这一对有没有被配对搬运按 keyOffset 升序交换过 —— 只有
    「两个都未命中」的对才可能交换。prefix 下混合对几乎不出现，这条分支结构上
    测不到，得靠 strided。
    """
    if layout == "prefix":
        return torch.arange(n_hit)
    if layout == "strided":
        return (torch.arange(n_hit, dtype=torch.float64) * k / n_hit).long()
    raise ValueError(f"unknown hit layout: {layout}")


def _apply_hit_ratio(case: dict, hit_ratio: float, layout: str = "prefix") -> int:
    """Mark ``round(hit_ratio*topk)`` slots resident. Returns that count.

    The kernel's hit test is positional: ``status[i] == topk[i]``. Clear the
    whole tensor first -- the trailing element is a readiness side channel the
    kernel writes itself, and leaving the previous call's value there changes
    which path this call takes.

    ``layout`` picks *which* slots those are; see :func:`_hit_positions`.
    """
    topk = case["topk_indices"]
    k = topk.shape[-1]
    n_hit = max(0, min(k, int(round(hit_ratio * k))))
    case["sel_status"].fill_(-1)
    if n_hit:
        idx = _hit_positions(k, n_hit, layout).to(case["sel_status"].device)
        case["sel_status"][..., idx] = topk[..., idx]
    return n_hit


def _call_fused(case: dict) -> torch.Tensor:
    # A list rather than a call, because selection_membership_map sits in the
    # middle: position 5 where the build has it, absent otherwise, which shifts
    # everything after it. See _OP_TAKES_MEMBERSHIP.
    args = [
        case["query"],  # query
        case["sel_rope"],  # selection_k_rope
        case["sel_kv"],  # selection_kv_cache
        case["sel_bt"],  # selection_kv_block_table
        case["sel_status"],  # selection_kv_block_status
    ]
    if _OP_TAKES_MEMBERSHIP:
        args.append(case["sel_membership"])  # selection_membership_map
    args += [
        case["topk_indices"],  # selection_topk_indices
        case["full_rope"],  # full_k_rope
        case["full_kv"],  # full_kv_cache
        case["full_bt"],  # full_kv_block_table
        case["actual_k"],  # full_kv_actual_seq
        case["actual_q"],  # full_q_actual_seq
        case["scale"],  # scale_value
        1,  # sparse_block_size
        case["stbs"],  # selection_topk_block_size
    ]
    out = fused(
        *args,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )
    torch.npu.synchronize()
    return out[..., :_KVD].contiguous()


def _golden(case: dict) -> torch.Tensor:
    """Fp32 dense MLA baseline over exactly the selected tokens.

    MLA shares K and V: scores use ``cat(nope, rope)`` while the value side is
    the nope half alone. The block table is the identity here, so a token id
    indexes the flattened cache directly.
    """
    dtype = case["dtype"]
    query = case["query"]
    num_tokens, q_heads, _ = query.shape

    q_nope = query[..., :_KVD].float().cpu()
    q_rope = query[..., _KVD:].float().cpu()
    kv_flat = case["full_kv"].reshape(-1, _KVD).float()
    rope_flat = case["full_rope"].reshape(-1, _KRD).float()
    topk_flat = case["topk_indices"].reshape(num_tokens, -1).cpu()

    outputs = []
    for t in range(num_tokens):
        idx = topk_flat[t]
        idx = idx[idx >= 0].long()
        k_nope = kv_flat[idx]  # [n, 512]
        k_rope = rope_flat[idx]  # [n, 64]
        keys = torch.cat([k_nope, k_rope], dim=-1)  # [n, 576]

        q_t = torch.cat([q_nope[t], q_rope[t]], dim=-1)  # [H, 576]
        scores = (q_t @ keys.transpose(0, 1)) * case["scale"]  # [H, n]
        attn = torch.softmax(scores, dim=-1)
        outputs.append(attn @ k_nope)  # [H, 512]

    return torch.stack(outputs, dim=0).to(dtype)


def _finite_report(tensor: torch.Tensor, tag: str) -> str:
    """Summarise a kernel output's health, computed on the host.

    Deliberately copies to CPU and widens to fp32 first: ``torch.isfinite`` on
    the NPU was observed to miss NaN in bf16 tensors, which let a NaN reach a
    comparison unflagged.
    """
    host = tensor.detach().float().cpu()
    nans = int(torch.isnan(host).sum())
    infs = int(torch.isinf(host).sum())
    finite = host[torch.isfinite(host)]
    span = f"[{float(finite.min()):.4e}, {float(finite.max()):.4e}]" if finite.numel() else "none"
    return f"[{tag}] n={host.numel()} nan={nans} inf={infs} finite_range={span}"


def _assert_finite(tensor: torch.Tensor, tag: str) -> None:
    """Fail with a count rather than letting NaN leak into a later comparison."""
    host = tensor.detach().float().cpu()
    bad = int((~torch.isfinite(host)).sum())
    if bad:
        raise AssertionError(f"{_finite_report(tensor, tag)} -- {bad} non-finite outputs")


def _check_against_golden(out: torch.Tensor, ref: torch.Tensor, tag: str, dtype: torch.dtype) -> None:
    out_cpu = out.cpu()
    assert out_cpu.shape == ref.shape, f"[{tag}] output shape {tuple(out_cpu.shape)} != baseline {tuple(ref.shape)}"
    assert torch.isfinite(out_cpu).all(), f"[{tag}] kernel produced non-finite values"

    torch.testing.assert_close(
        out_cpu,
        ref,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        msg=lambda m: f"[FSA_OVERLAP:{tag}] kernel output diverges from baseline. {m}",
    )

    ref_f32 = ref.float()
    out_f32 = out_cpu.float()
    diff = (out_f32 - ref_f32).abs()
    ref_abs = ref_f32.abs()
    peak = float(ref_abs.max())
    mean_ref_abs = float(ref_abs.mean())

    max_abs_err = float(diff.max())
    mean_abs_err = float(diff.mean())
    max_sig_rel_err = max_abs_err / peak if peak > 0 else 0.0
    mean_sig_rel_err = mean_abs_err / mean_ref_abs if mean_ref_abs > 0 else 0.0

    significant = ref_abs >= peak * _SIG_FLOOR_FRAC
    max_rel_err_sig = float((diff[significant] / ref_abs[significant]).max()) if significant.any() else 0.0

    _emit_metric(
        f"[FSA_OVERLAP:{tag}] dtype={dtype} "
        f"peak={peak:.4e} "
        f"max_abs_err={max_abs_err:.4e} "
        f"max_sig_rel_err={max_sig_rel_err * 100:.4f}% "
        f"mean_sig_rel_err={mean_sig_rel_err * 100:.4f}% "
        f"max_rel_err_sig(>={int(_SIG_FLOOR_FRAC * 100)}%peak)={max_rel_err_sig * 100:.4f}%"
    )

    assert max_sig_rel_err < _MAX_SIG_REL_ERR, (
        f"[FSA_OVERLAP:{tag}] dtype={dtype} signal-relative max error "
        f"{max_sig_rel_err * 100:.4f}% exceeds 1% budget "
        f"(peak={peak:.4e}, max_abs_err={max_abs_err:.4e})"
    )
    assert mean_sig_rel_err < _MAX_MEAN_SIG_ERR, (
        f"[FSA_OVERLAP:{tag}] dtype={dtype} signal-relative mean error "
        f"{mean_sig_rel_err * 100:.4f}% exceeds 0.5% drift budget "
        f"(mean_ref_abs={mean_ref_abs:.4e}, mean_abs_err={mean_abs_err:.4e})"
    )
    assert max_rel_err_sig < _MAX_REL_ERR, (
        f"[FSA_OVERLAP:{tag}] dtype={dtype} per-element relative error on "
        f">={int(_SIG_FLOOR_FRAC * 100)}%-of-peak elements "
        f"{max_rel_err_sig * 100:.4f}% exceeds 1% budget"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("q_heads", [16, 4])
@pytest.mark.parametrize("topk", [128, 1024])
def test_overlap_miss_path_matches_dense_baseline(offload_pool, topk: int, q_heads: int, dtype: torch.dtype) -> None:
    """Cold cache: kernel fetches from the CPU pool, must match fp32 dense."""
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    case = _build_case(topk, q_heads, dtype)

    _reset_selection_state(case)
    out = _call_fused(case)
    ref = _golden(case)

    _check_against_golden(out, ref, f"miss,topk={topk},q_heads={q_heads}", dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("q_heads", [16])
@pytest.mark.parametrize("topk", [128, 1024])
def test_overlap_hit_path_reproduces_miss_path(offload_pool, topk: int, q_heads: int, dtype: torch.dtype) -> None:
    """Warm cache: replaying with every slot marked hit must not change output.

    The first call populates selection_kv_cache and rewrites
    selection_kv_block_status in place. Marking every topk slot as resident and
    calling again makes the kernel serve everything from the selection cache
    instead of the CPU pool -- a different code path that has to agree.
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    case = _build_case(topk, q_heads, dtype)

    tag = f"hit-vs-miss,topk={topk},q_heads={q_heads}"

    _reset_selection_state(case)
    miss_out = _call_fused(case).clone()
    _assert_finite(miss_out, f"{tag}/cold")

    # Mark every slot resident. The kernel's hit test is positional:
    # status[i] == topk[i]. Clear the whole tensor first -- the trailing
    # element is a readiness side channel the kernel writes itself, and leaving
    # the previous call's value there changes which path this call takes. The
    # timing UT resets the same way before applying a hit ratio.
    case["sel_status"].fill_(-1)
    case["sel_status"][..., :topk] = case["topk_indices"]
    hit_out = _call_fused(case)

    # Report both before asserting: if the warm call is what breaks, the cold
    # numbers are needed to tell "kernel wrote garbage" from "kernel wrote
    # nothing at all".
    _emit_metric(_finite_report(miss_out, f"{tag}/cold"))
    _emit_metric(_finite_report(hit_out, f"{tag}/warm"))
    _assert_finite(hit_out, f"{tag}/warm")
    torch.testing.assert_close(
        hit_out.cpu(),
        miss_out.cpu(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        msg=lambda m: f"[FSA_OVERLAP:{tag}] warm-cache output differs from cold-cache output. {m}",
    )

    diff = (hit_out.float() - miss_out.float()).abs()
    _emit_metric(
        f"[FSA_OVERLAP:{tag}] dtype={dtype} max_abs_diff={float(diff.max()):.4e} mean_abs_diff={float(diff.mean()):.4e}"
    )


# ---------------------------------------------------------------------------
# 覆盖两条基线扫过的参数组合
#
# docs/命中率曲线-mte_fuse_0723基线/  bs=16 固定，命中率 0 → 100% 共 21 点
# docs/BS曲线-mte_fuse_0723基线/      命中率 80% 固定，bs 1 → 128 共 8 点
#
# 两条都是 topk=2048 / max_seq_len=65536 / q_heads=16 / prefix，而上面那几条
# 用例跑的是 topk 128|1024、bs 写死 1、max_seq_len = topk*4。也就是说在这两条
# 用例之前，**基线跑的形状一个都没有精度覆盖**。
# ---------------------------------------------------------------------------

_BASELINE_TOPK = 2048
_BASELINE_MSL = 65536
_BASELINE_Q_HEADS = 16
_BASELINE_BATCH_SIZES = (1, 2, 4, 8, 16, 25, 32, 36, 48, 64, 128)
# 25/36/48 为 helper-extend（波次调度+尾波 helper）的边界覆盖：25=k1r1（尾波
# helper 最小触发）、36=r12（尾波 helper 满配）、48=r0（整好满波、不开 helper）；
# 64/128 兼作 r>12（不开 helper）回归点。helper-extend 之前 507015 那类崩溃
# 正是这个区间缺精度覆盖漏出去的。
# 命中率曲线那 21 个点里挑的：两端、四分位、以及下降段（90% 之后耗时掉 78%，
# 出问题最可能在那一段）。
_BASELINE_HIT_RATIOS = (0.0, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 1.0)


@pytest.mark.parametrize("bsz", _BASELINE_BATCH_SIZES)
def test_overlap_baseline_batch_sizes_match_dense(offload_pool, bsz: int) -> None:
    """BS 曲线扫的每个 batch size，都跟 fp32 dense 基准对一遍。

    这条就是 full_q_actual_seq 那个 bug 该被抓住的地方：actual_q 不是累加值时，
    请求 1..bsz-1 拿到零个 query token、根本不参与计算，它们的输出行留在
    输出缓冲原来的内容上，跟 golden 一比立刻失败。在这条用例之前 bsz 写死 1，
    两种写法恰好重合，所以什么都抓不到。
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    dtype = torch.bfloat16
    case = _build_case(
        _BASELINE_TOPK, _BASELINE_Q_HEADS, dtype, bsz=bsz, max_seq_len=_BASELINE_MSL
    )
    tag = f"baseline-bs,bs={bsz},topk={_BASELINE_TOPK},msl={_BASELINE_MSL}"

    _reset_selection_state(case)
    out = _call_fused(case)
    _assert_finite(out, tag)
    ref = _golden(case)
    _check_against_golden(out, ref, tag, dtype)


@pytest.mark.parametrize("layout", _HIT_LAYOUTS)
@pytest.mark.parametrize("hit_ratio", _BASELINE_HIT_RATIOS)
def test_overlap_baseline_hit_ratios_reproduce_miss_path(
    offload_pool, hit_ratio: float, layout: str
) -> None:
    """命中率曲线扫的那条线上，部分命中不得改变答案。

    上面那条 hit-path 用例只测两端 —— 全驻留或全不驻留。而基线扫的是中间 21 个
    点，簿记类的 bug 恰恰藏在中间：某个槽位被标成驻留但缓存里那一行是旧的，
    或者"第 i 个槽位对应哪一块缓存"差一位，在 0% 和 100% 上都看不出来，
    到 50% 才现形。

    先跑一次冷缓存：它既从 CPU 池算出结果，又把这次 topk 选中的那些行填进缓存。
    之后把其中一部分标成驻留，答案必须一样 —— 标哪一部分都一样。
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    dtype = torch.bfloat16
    bsz = 16  # 命中率曲线固定的 batch size
    case = _build_case(
        _BASELINE_TOPK, _BASELINE_Q_HEADS, dtype, bsz=bsz, max_seq_len=_BASELINE_MSL
    )
    tag = f"baseline-hit,ratio={hit_ratio:g},layout={layout},bs={bsz},topk={_BASELINE_TOPK}"

    _reset_selection_state(case)
    miss_out = _call_fused(case).clone()
    _assert_finite(miss_out, f"{tag}/cold")

    n_hit = _apply_hit_ratio(case, hit_ratio, layout)
    warm_out = _call_fused(case)
    _assert_finite(warm_out, f"{tag}/warm")

    diff = (warm_out.float() - miss_out.float()).abs()
    _emit_metric(
        f"[FSA_OVERLAP:{tag}] n_hit={n_hit}/{_BASELINE_TOPK} "
        f"max_abs_diff={float(diff.max()):.4e} mean_abs_diff={float(diff.mean()):.4e}"
    )
    torch.testing.assert_close(
        warm_out.cpu(),
        miss_out.cpu(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        msg=lambda m: (
            f"[FSA_OVERLAP:{tag}] 部分驻留的输出与冷缓存的输出不一致。{m}"
        ),
    )


# ---------------------------------------------------------------------------
# selection 缓存的内容判据
#
# 上面每一条用例检查的都是 attention_out。而 14 参数版的计算流不读 selection
# 缓存（kernel 里 4 处写、0 处读，见 docs/0723计算流不读缓冲/），于是：
#
#     只改维护层的改动，上面那些用例必然全过。
#
# 不是覆盖不够，是结构上测不到 —— 缓存写没写、写对没写对、驻留的槽位该不该
# 跳过，attention_out 一律不受影响。要动维护层就得直接对缓存的内容。
#
# 好处是这一段是纯搬运、不含任何浮点运算，所以判据可以是**逐位相等**，
# 不设容差。
#
# ⚠ 已知缺口：CopySelectionUpdateTokenFromFullCache 开头那几条 early return
# （topkValue 越界、块号为负）在这里的输入下不会触发，所以"没搬数据、status
# 却标成驻留"那条路径仍然测不到。要覆盖它得往 topk 里塞越界值，会连带改变
# attention 的期望值，另开一条用例做。
# ---------------------------------------------------------------------------

# 维护层有两条路，都要测：
#   流水线路 CopyOutSelectionUpdateFromKvMerge —— 嵌在 merge 循环里，复用计算流
#     已经搬下来的 KV
#   串行路   RunAllCoreSelectionUpdate → CopySelectionUpdateTokenFromFullCache
#     —— 接在计算之后，从 CPU 池重读
# 走哪条由 UsePipelineSelectionUpdate() 决定，**跟分支有关**：
#   基线（tag baseline-0723-v1）：sparseBlockCount <= 1024 才走流水线路，
#     所以 1024 走流水线、2048 走串行
#   优化分支（优化 02，commit 43082e16）：闸放开，两个 topk 都走流水线路
_CACHE_TOPKS = (1024, 2048)
_CACHE_HIT_RATIOS = (0.0, 0.5, 1.0)
_CACHE_BSZ = 4
# 基线上 UsePipelineSelectionUpdate() 的分界。只用来决定 status 判据严不严，
# 见 _check_selection_status。⚠ 优化分支上这个值已经不成立（两条都走流水线），
# 那边 topk=1024 的 status 缺陷被这条降级掩掉了 —— 诊断输出照打，不受影响。
_PIPELINE_TOPK_MAX = 1024


def _expected_selection_cache(
    case: dict,
    status_before: torch.Tensor,
    kv_before: torch.Tensor,
    rope_before: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """按 kernel 自己的地址映射，在 host 上算出调用之后缓存该是什么样。

    返回 ``(期望的 sel_kv, 期望的 sel_rope, 该搬的槽位数)``。

    映射抄自 CopySelectionUpdateTokenFromFullCache；流水线路
    CopyOutSelectionUpdateFromKvMerge 写的是同一组地址：

        tv       = selection_topk_indices[row][pos]
        dst 行号 = selection_kv_block_table[row][pos // 128] * 128 + pos % 128
        src 行号 = full_kv_block_table[b][tv // 128] * 128 + tv % 128

    搬不搬只看位置判据 ``status[row][pos] != tv``，再加上那个函数开头的几条
    early return。
    """
    bsz = case["bsz"]
    k = case["topk"]
    assert tuple(case["topk_indices"].shape) == (bsz, 1, 1, k), (
        f"本判据假定 seq = kv_heads = 1，实际 topk_indices 形状 "
        f"{tuple(case['topk_indices'].shape)}"
    )

    topk_flat = case["topk_indices"].reshape(bsz, k).cpu().long()
    status_flat = status_before[..., :k].reshape(bsz, k).cpu().long()
    # sparse_mode=3、每个请求一个 query token：s2IdLimit 退化成该请求的 KV 长度
    s2_limit = case["actual_k"].cpu().long().reshape(bsz, 1)

    moved = (status_flat != topk_flat) & (topk_flat >= 0) & (topk_flat < s2_limit)

    pos = torch.arange(k)
    sel_bt = case["sel_bt"].cpu().long()  # [bsz, sel_blocks]，seq = kv_heads = 1
    sel_block_num = sel_bt[:, pos // _BLOCK_SIZE]
    moved &= sel_block_num >= 0
    dst_row = sel_block_num * _BLOCK_SIZE + (pos % _BLOCK_SIZE)

    full_bt = case["full_bt"].cpu().long()  # [bsz, full_blocks]
    tv_safe = topk_flat.clamp(min=0)
    full_block_num = torch.gather(
        full_bt, 1, (tv_safe // _BLOCK_SIZE).clamp(max=full_bt.shape[1] - 1)
    )
    moved &= full_block_num >= 0
    src_row = full_block_num * _BLOCK_SIZE + (tv_safe % _BLOCK_SIZE)

    dst_moved = dst_row[moved]
    assert dst_moved.unique().numel() == dst_moved.numel(), (
        "两个槽位映射到了同一行缓存，期望值本身就不确定 —— 先查 selection_kv_block_table"
    )

    exp_kv = kv_before.reshape(-1, _KVD).clone()
    exp_rope = rope_before.reshape(-1, _KRD).clone()
    exp_kv[dst_moved] = case["full_kv"].reshape(-1, _KVD)[src_row[moved]]
    exp_rope[dst_moved] = case["full_rope"].reshape(-1, _KRD)[src_row[moved]]
    return (
        exp_kv.reshape(kv_before.shape),
        exp_rope.reshape(rope_before.shape),
        int(moved.sum()),
    )


def _rows_differ(got: torch.Tensor, want: torch.Tensor, width: int) -> int:
    """有多少行不是逐位相同。

    按位比而不是用 ``!=``：NaN 跟自己也不相等，会把"两边都是 NaN"报成差异。
    """
    a = got.reshape(-1, width).contiguous().view(torch.int16)
    b = want.reshape(-1, width).contiguous().view(torch.int16)
    return int((a != b).any(dim=1).sum())


def _check_selection_cache(
    case: dict,
    exp_kv: torch.Tensor,
    exp_rope: torch.Tensor,
    kv_before: torch.Tensor,
    n_expected: int,
    tag: str,
) -> None:
    """逐位比对缓存内容，失败时报清楚是哪一种失败。"""
    got_kv = case["sel_kv"].cpu()
    got_rope = case["sel_rope"].cpu()

    kv_bad = _rows_differ(got_kv, exp_kv, _KVD)
    rope_bad = _rows_differ(got_rope, exp_rope, _KRD)
    # 内容跟调用前不同的行数。⚠ 它**可以合法地小于** n_expected：搬进来的数据
    # 可能跟那一行原来的内容逐位相同（换一批 topk 时同一槽位撞上同一个 token）。
    # 真判据是 kv_bad / rope_bad，这个只在失败时用来分辨是哪一种失败。
    n_changed = _rows_differ(got_kv, kv_before, _KVD)

    _emit_metric(
        f"[FSA_OVERLAP:{tag}] slots={case['bsz'] * case['topk']} "
        f"expect_moved={n_expected} rows_changed={n_changed} "
        f"kv_rows_bad={kv_bad} rope_rows_bad={rope_bad}"
    )

    if kv_bad or rope_bad:
        if n_changed == 0:
            hint = "缓存跟调用前逐位相同 —— 维护层一行都没写（这个构建里被关掉了？）"
        elif n_changed > n_expected:
            hint = (
                f"内容变了 {n_changed} 行，多于该搬的 {n_expected} 行 —— "
                f"标成驻留的槽位也被搬了"
            )
        else:
            hint = f"内容变了 {n_changed} 行，该搬 {n_expected} 行；是内容或者地址错了"
        raise AssertionError(
            f"[FSA_OVERLAP:{tag}] selection 缓存内容不符：kv 有 {kv_bad} 行、"
            f"rope 有 {rope_bad} 行跟期望不一致。{hint}"
        )


def _check_selection_status(case: dict, tag: str, strict: bool = True) -> None:
    """调用之后，status 的每个槽位都该等于这一轮的 topk。

    ⚠ 2026-08-09 实测：**流水线路上这条会挂，串行路不挂**，而两条路的缓存内容
    都是对的（kv_rows_bad / rope_rows_bad 全 0）。挂的是这张簿记表，它不参与
    计算 —— 14 参数版的 attention_out 根本不读缓存。基线上挂的是 topk=1024，
    优化分支（闸放开后两条 topk 都走流水线）上 topk=2048 也挂。

    ``strict=False`` 时只记录不失败。**降级只降这一条 status 判据**，缓存内容的
    判据对两条路一律严格。诊断输出不受 strict 影响，一律打印。
    """
    bsz, k = case["bsz"], case["topk"]
    got = case["sel_status"][..., :k].cpu().reshape(bsz, k)
    want = case["topk_indices"].cpu().reshape(bsz, k)
    stale = got != want
    n_stale = int(stale.sum())
    if n_stale == 0:
        return

    rows, cols = torch.nonzero(stale, as_tuple=True)
    row_list, col_list = rows.tolist(), cols.tolist()
    per_row = torch.bincount(rows, minlength=bsz).tolist()
    _emit_metric(
        f"[FSA_OVERLAP:{tag}] status_stale={n_stale}{'' if strict else ' (KNOWN-GAP)'} "
        f"per_row={per_row}"
    )
    # 逐行打印相邻间隔的直方图 —— 间隔的众数直接给出周期，比看前 16 个位置有用。
    # 2026-08-09 已知：漏刷数 = 每行 miss 数 / 32，且 row 0 从不漏。
    #
    # 按 32 字节数据块归并（GM 一个块 = 8 个 int32，statusBase = row * (topk+1)）。
    # 待判的假设是「两个 AIV 的标量写落在同一个块上、互相整块覆盖」，它预言：
    #   ① 漏的位置成块出现 —— 一个块里属于本行的槽位要么全漏要么全不漏
    #      （full=n/n 表示这个块本行的槽位全漏）
    #   ② 块的位置贴着核间边界：轮边界 q*512、轮内 subblock 边界 q*512+256、
    #      行头 0、行尾那个单独写的 statusBase+topk
    #   ③ row 0 的 statusBase 是 8 的整数倍，上面这些边界天然对齐、不共享块，
    #      所以一个都不漏
    # 假设不成立的样子：块只漏一部分（full 不等）、或块的位置跟边界对不上。
    s2_base = 512  # tiling.cpp: sInnerSize_ 默认 512
    for r in range(bsz):
        cols_r = [c for rr, c in zip(row_list, col_list) if rr == r]
        if not cols_r:
            continue
        status_base = r * (k + 1)
        gaps = [b - a for a, b in zip(cols_r, cols_r[1:])]
        hist = sorted(Counter(gaps).items(), key=lambda kv: -kv[1])[:4]
        # 本行每个块里有几个槽位、其中漏了几个
        stale_per_blk = Counter((status_base + c) // 8 for c in cols_r)
        rowcnt_per_blk = Counter((status_base + c) // 8 for c in range(k))
        blocks = sorted(stale_per_blk)
        detail = " ".join(
            f"{b}[{(b * 8 - status_base)}]={stale_per_blk[b]}/{rowcnt_per_blk[b]}"
            for b in blocks[:10]
        )
        full = all(stale_per_blk[b] == rowcnt_per_blk[b] for b in blocks)
        edges = sorted(
            {0, k}
            | {q * s2_base for q in range(1, (k + s2_base - 1) // s2_base)}
            | {q * s2_base + s2_base // 2 for q in range((k + s2_base - 1) // s2_base)}
        )
        edge_blocks = sorted({(status_base + e) // 8 for e in edges})
        _emit_metric(
            f"[FSA_OVERLAP:{tag}]   row={r} n={len(cols_r)} base%8={status_base % 8} "
            f"gap_hist={hist} nblk={len(blocks)} whole_blocks={full} "
            f"blk[起始槽位]=漏/本行槽位: {detail}"
        )
        _emit_metric(
            f"[FSA_OVERLAP:{tag}]   row={r} 边界槽位={edges} 对应块={edge_blocks} "
            f"漏的块⊆边界块={set(blocks) <= set(edge_blocks)}"
        )
    if not strict:
        return
    sample = "; ".join(
        f"row={r} pos={c} status={int(got[r, c])} topk={int(want[r, c])}"
        for r, c in list(zip(row_list, col_list))[:6]
    )
    raise AssertionError(
        f"[FSA_OVERLAP:{tag}] 调用后 selection_kv_block_status 有 {n_stale} 个槽位"
        f"没被更新成本轮的 topk（每行 {per_row}）。前几个：{sample}"
    )


@pytest.mark.parametrize("layout", _HIT_LAYOUTS)
@pytest.mark.parametrize("hit_ratio", _CACHE_HIT_RATIOS)
@pytest.mark.parametrize("topk", _CACHE_TOPKS)
def test_selection_cache_contents_match_expected(
    offload_pool, topk: int, hit_ratio: float, layout: str
) -> None:
    """维护层写进 selection 缓存的内容，逐位对一遍。

    hit_ratio 在这里的作用跟别处不同：缓存刚被清零，把前一部分槽位谎称驻留，
    kernel 信了就不该去搬，那部分应该**保持为零**。所以这条用例同时钉住两件事
    —— miss 的槽位搬对了，驻留的槽位真的没搬。
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    case = _build_case(topk, _BASELINE_Q_HEADS, torch.bfloat16, bsz=_CACHE_BSZ)
    tag = f"sel-cache,topk={topk},ratio={hit_ratio:g},layout={layout},bs={_CACHE_BSZ}"

    _reset_selection_state(case)
    _apply_hit_ratio(case, hit_ratio, layout)

    status_before = case["sel_status"].clone()
    bt_before = case["sel_bt"].clone()
    kv_before = case["sel_kv"].cpu().clone()
    rope_before = case["sel_rope"].cpu().clone()

    _call_fused(case)

    assert torch.equal(case["sel_bt"], bt_before), (
        f"[FSA_OVERLAP:{tag}] kernel 改了 selection_kv_block_table，"
        f"期望值用的地址映射不再成立"
    )
    exp_kv, exp_rope, n_expected = _expected_selection_cache(
        case, status_before, kv_before, rope_before
    )
    _check_selection_cache(case, exp_kv, exp_rope, kv_before, n_expected, tag)
    _check_selection_status(case, tag, strict=topk > _PIPELINE_TOPK_MAX)


@pytest.mark.parametrize("layout", _HIT_LAYOUTS)
@pytest.mark.parametrize("topk", _CACHE_TOPKS)
def test_selection_cache_keeps_resident_slots(offload_pool, topk: int, layout: str) -> None:
    """标成驻留的槽位内容必须原封不动 —— 哪怕 topk 已经换了一批。

    第一轮冷缓存，缓存被这一轮的 topk 填满。第二轮换一批 topk，把前一半谎称
    驻留：kernel 信了就不搬那一半，于是前一半应该还留着**第一轮**的内容
    （对应旧 topk），后一半换成新的。

    上一条用例里驻留槽位的期望值是零，而零跟"根本没写"分不开；这一条的期望值
    是上一轮真实搬进去的数据，分得开。
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    case = _build_case(topk, _BASELINE_Q_HEADS, torch.bfloat16, bsz=_CACHE_BSZ)
    tag = f"sel-cache-resident,topk={topk},layout={layout},bs={_CACHE_BSZ}"

    # 第一轮：冷缓存，全部 miss，缓存被填满
    _reset_selection_state(case)
    _call_fused(case)

    # 换一批 topk，跟第一轮不同
    bsz, k = case["bsz"], case["topk"]
    rng = np.random.RandomState(20260809 + topk)
    fresh = np.stack(
        [
            np.sort(rng.choice(case["max_seq_len"], k, replace=False).astype(np.int32))
            for _ in range(bsz)
        ]
    )
    case["topk_indices"] = (
        torch.tensor(fresh, dtype=torch.int32, device="npu")
        .reshape(bsz, 1, 1, k)
        .contiguous()
    )

    # 一半谎称驻留 —— 缓存里那一半其实是旧 topk 的数据。
    # strided 下驻留与未驻留在相邻槽位上交替，计算流的每一对都是混合对。
    n_hit = _apply_hit_ratio(case, 0.5, layout)

    status_before = case["sel_status"].clone()
    bt_before = case["sel_bt"].clone()
    kv_before = case["sel_kv"].cpu().clone()
    rope_before = case["sel_rope"].cpu().clone()

    _call_fused(case)

    assert torch.equal(case["sel_bt"], bt_before), (
        f"[FSA_OVERLAP:{tag}] kernel 改了 selection_kv_block_table，"
        f"期望值用的地址映射不再成立"
    )
    exp_kv, exp_rope, n_expected = _expected_selection_cache(
        case, status_before, kv_before, rope_before
    )
    assert n_expected == bsz * (k - n_hit), (
        f"[FSA_OVERLAP:{tag}] 该搬的槽位数算出来是 {n_expected}，"
        f"按构造应该是 {bsz * (k - n_hit)}"
    )
    _check_selection_cache(case, exp_kv, exp_rope, kv_before, n_expected, tag)
    _check_selection_status(case, tag, strict=topk > _PIPELINE_TOPK_MAX)

# ---------------------------------------------------------------------------
# LRU 计划路（只在 15 参数版上存在）
#
# 上面每一条用例都把 selection_membership_map 填 -1、控制区标记不写，算子因此
# **退回按名次判**（status[i] == topk[i]）。也就是说在这一节之前，外部计划那条
# 路一行都没有精度覆盖 —— 而上游 mte_fuse_0723 的生产路径走的正是它。
#
# 这里手工编计划，不依赖 host 规划器。格式见文档 上游的LRU常驻集/ §4：
#   * 计划区没有自己的存储，压在 membership 区尾部，起点 16384 − topk
#   * 一格对应一个 topk 下标，值 = 槽位号加一
#   * 正号 = 命中，直接读该槽位；负号 = 未命中，会被搬进该槽位
#   * 控制区八格里，host 负责 [1] 标记、[2] topk、[3] 计划区起点、[7] 配对搬运标记
# ---------------------------------------------------------------------------

_PLAN_READY_MARKER = 0x5A45  # 控制区 [1]，kernel 认它才走计划路
_PAIRED_COPY_MARKER = 0x5A56  # 控制区 [7]
_DIRECT_LAYOUT_MARKER = 0x5A44  # 控制区 [5]，打开直接寻址；[6] 放 capacity
_PLAN_SLOT_LAYOUTS = ("contiguous", "scattered")
# 一行有几个槽位。0 = 等于 topk，是这个文件原来唯一测过的形状；4096 是生产
# 的 kv_offload_decode_config.topk_buffer_size 默认值，槽位比 topk 多一倍以上。
# 两个都要测：槽位多出来之后 scattered 的槽位号摊得更开，计划区里编的号越界与否、
# kernel 按 blockTable[slot / 128] 取到的是不是本行的块，全都换了一组取值。
_PLAN_SLOT_BUDGETS = (0, 4096)


def _budget_label(budget: int) -> str:
    return "slots=topk" if budget == 0 else f"slots={budget}"


def _plan_slots(k: int, capacity: int, layout: str) -> torch.Tensor:
    """topk 的第 i 项落在哪个槽位。

    - ``contiguous``：第 i 项 → 第 i 号槽位。槽位号连续，块内相邻，搬运能合并。
    - ``scattered``：从整个槽位空间随机取 k 个（种子定死）。跨块的比例高，
      合并不了 —— 与计时 UT 的同名布局同一个构造。
    """
    if layout == "contiguous":
        return torch.arange(k, dtype=torch.int64)
    gen = torch.Generator().manual_seed(capacity * 7919 + k)
    return torch.randperm(capacity, generator=gen)[:k]


def _write_plan(case: dict, slots: torch.Tensor, hit_ranks: torch.Tensor) -> None:
    """把一份外部计划写进 selection_membership_map。

    ``slots[i]`` 是第 i 个 topk 项的槽位号，``hit_ranks`` 是哪些**名次**算命中。
    两个轴分开：换命中集合时槽位映射保持不变，缓存里的数据才对得上。

    ⚠ 每一行写的是同一份计划。membership 的行数是 total_sel（缓存行数），
      比请求数多；本用例里所有请求的 topk 集合虽不同，但计划的「第 i 项去几号
      槽位」是同一套，所以整块铺同一行即可。
    """
    membership = case["sel_membership"]
    assert membership is not None, "14 参数版没有这块，调用方应当先 skip"
    k = int(case["topk_indices"].shape[-1])
    assert slots.numel() == k, f"槽位映射长度 {slots.numel()} 与 topk {k} 对不上"

    enc = -(slots.to(torch.int64) + 1)  # 先全标未命中
    if hit_ranks.numel():
        enc[hit_ranks] = -enc[hit_ranks]  # 命中的翻正

    ctl = _MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT
    plan_start = ctl - k
    row = torch.full((_MEMBERSHIP_INT16_COUNT,), -1, dtype=torch.int16)
    row[plan_start:ctl] = enc.to(torch.int16)
    row[ctl:ctl + _MEMBERSHIP_CONTROL_INT16_COUNT] = 0
    row[ctl + 1] = _PLAN_READY_MARKER
    row[ctl + 2] = k
    row[ctl + 3] = plan_start
    # 第 5、6 格：直接寻址（界是 capacity）。第 7 格：配对搬运。
    capacity = _plan_capacity(case)
    assert capacity <= 32767, f"capacity {capacity} 放不进 int16 的控制格"
    row[ctl + 5] = _DIRECT_LAYOUT_MARKER
    row[ctl + 6] = capacity
    row[ctl + 7] = _PAIRED_COPY_MARKER

    # ⚠⚠ 第 4 格是**物理行号**，而且 [5][6][7] 三格全都嵌在它 >= 0 这个条件里
    #   （service_vector_mla.h:3297 `if (useExternalPlanSource && externalPhysicalRow >= 0)`）。
    #   所以这一格有三种填法，两种是错的：
    #     填 0   → 每个请求都去读第 0 行的缓存，bs>1 时输出既不是 golden(A) 也不是 golden(B)
    #     填 -1  → 整个块被跳过，[5][6] 写了也白写，退回**块表**寻址。那条路上有一道
    #              过紧的界（GetSelectionSlotTokenOffset 里 `slot >= sparseBlockCount`，
    #              界是 topk），于是**槽位号 >= topk 的命中全被静默判成未命中**。
    #              capacity 等于 topk 时看不出来，capacity 4096 而 topk 2048 时一半中招
    #     逐行填自己的行号 → 对。生产就是这么发的（kv_offload_decode.cpp:166 按
    #              logical_row 逐行写 physical_row），我们的缓存布局一行一请求，
    #              所以第 b 行填 b
    #   ⚠ 前两种都不报错，只是数字不对 —— 2026-08-12 一天之内先后栽了这两种。
    rows = int(membership.shape[0])
    bsz = int(case["topk_indices"].shape[0])
    assert rows <= 32767, f"membership 行数 {rows} 放不进 int16 的控制格"
    block = row.unsqueeze(0).repeat(rows, 1)
    # membership 的行数是缓存行数（total_sel），比请求数多。多出来的行 kernel 不读，
    # 填 -1 而不是行号 —— 填行号的话那些行指向的缓存位置根本不存在。
    physical = torch.full((rows,), -1, dtype=torch.int16)
    physical[:bsz] = torch.arange(bsz, dtype=torch.int16)
    block[:, ctl + 4] = physical
    membership.copy_(block.to(membership.device))


def _plan_capacity(case: dict) -> int:
    """一行有几个槽位 = 块表一行的块数 × 块大小。

    计划里的槽位号必须落在这个范围内，否则 kernel 按 blockTable[slot / 128]
    取块会越过本行的条目、串到别的请求上 —— 上游自带的 perf 用例就是这么错的。
    """
    return int(case["sel_bt"].shape[-1]) * _BLOCK_SIZE


@pytest.mark.skipif(not _OP_TAKES_MEMBERSHIP, reason="14 参数版没有 selection_membership_map，走不到计划路")
@pytest.mark.parametrize("slot_budget", _PLAN_SLOT_BUDGETS)
@pytest.mark.parametrize("slot_layout", _PLAN_SLOT_LAYOUTS)
@pytest.mark.parametrize("topk", _CACHE_TOPKS)
def test_plan_all_miss_matches_dense_baseline(
    offload_pool, topk: int, slot_layout: str, slot_budget: int
) -> None:
    """全未命中的计划：每一项都要按计划从 CPU 池搬进指定槽位，结果对上 fp32 golden。

    这条同时验两件事：计划区的编码 kernel 认得（否则它会退回按名次判，而冷缓存
    下按名次判也是全 miss，两条路的**输出**恰好一样 —— 所以下一条用例才是
    真正区分两条路的那条）。
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    dtype = torch.bfloat16
    case = _build_case(topk, _BASELINE_Q_HEADS, dtype, topk_buffer_size=slot_budget)
    tag = f"plan-miss,topk={topk},layout={slot_layout},{_budget_label(slot_budget)}"

    _reset_selection_state(case)
    slots = _plan_slots(topk, _plan_capacity(case), slot_layout)
    _write_plan(case, slots, torch.empty(0, dtype=torch.int64))

    out = _call_fused(case)
    _emit_metric(_finite_report(out, tag))
    _assert_finite(out, tag)
    _check_against_golden(out, _golden(case), tag, dtype)


@pytest.mark.skipif(not _OP_TAKES_MEMBERSHIP, reason="14 参数版没有 selection_membership_map，走不到计划路")
@pytest.mark.parametrize("hit_ratio", (0.5, 1.0))
@pytest.mark.parametrize("hit_layout", _HIT_LAYOUTS)
@pytest.mark.parametrize("slot_budget", _PLAN_SLOT_BUDGETS)
@pytest.mark.parametrize("slot_layout", _PLAN_SLOT_LAYOUTS)
@pytest.mark.parametrize("topk", _CACHE_TOPKS)
def test_plan_hit_reproduces_plan_miss(
    offload_pool, topk: int, slot_layout: str, slot_budget: int, hit_layout: str, hit_ratio: float
) -> None:
    """喂热之后把同一份槽位映射改成命中，输出必须复现未命中那次。

    第一次调用给全未命中的计划，kernel 按计划把第 i 个 topk 项搬进 slots[i]；
    **中间不重置状态**，第二次只把计划里那些名次翻成正号，kernel 改从槽位直接
    读。两次取的是同一份数据，输出必须一致。

    这条才是真正区分两条路的用例：按名次判的回退路读的是 status[i]，而这里
    status 停在上一次调用留下的值、跟 slots 的映射对不上 —— 如果 kernel 没走
    计划路，读到的就是别的槽位，结果对不上。
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    dtype = torch.bfloat16
    case = _build_case(topk, _BASELINE_Q_HEADS, dtype, topk_buffer_size=slot_budget)
    n_hit = int(round(topk * hit_ratio))
    tag = (
        f"plan-hit,topk={topk},layout={slot_layout},{_budget_label(slot_budget)},"
        f"hits={hit_layout}@{hit_ratio}"
    )

    _reset_selection_state(case)
    slots = _plan_slots(topk, _plan_capacity(case), slot_layout)
    _write_plan(case, slots, torch.empty(0, dtype=torch.int64))
    miss_out = _call_fused(case).clone()
    _emit_metric(_finite_report(miss_out, f"{tag}/cold"))
    _assert_finite(miss_out, f"{tag}/cold")

    _write_plan(case, slots, _hit_positions(topk, n_hit, hit_layout))
    hot_out = _call_fused(case)
    _emit_metric(_finite_report(hot_out, f"{tag}/warm"))
    _assert_finite(hot_out, f"{tag}/warm")

    torch.testing.assert_close(
        hot_out.cpu(),
        miss_out.cpu(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        msg=lambda m: f"[FSA_OVERLAP:{tag}] 走计划命中之后的输出与全未命中那次不一致。{m}",
    )
    diff = (hot_out.float() - miss_out.float()).abs()
    _emit_metric(
        f"[FSA_OVERLAP:{tag}] dtype={dtype} max_abs_diff={float(diff.max()):.4e} "
        f"mean_abs_diff={float(diff.mean()):.4e}"
    )

@pytest.mark.skipif(not _OP_TAKES_MEMBERSHIP, reason="14 参数版没有 selection_membership_map，走不到计划路")
@pytest.mark.parametrize("slot_budget", _PLAN_SLOT_BUDGETS)
@pytest.mark.parametrize("slot_layout", _PLAN_SLOT_LAYOUTS)
@pytest.mark.parametrize("topk", _CACHE_TOPKS)
def test_plan_is_actually_obeyed(
    offload_pool, topk: int, slot_layout: str, slot_budget: int
) -> None:
    """证明 kernel 真按计划取数，而不是退回按名次判。

    ⚠ 上面两条用例证明不了这一点：**只要簿记自洽，两条路的输出就一样**。
    走计划路时第一次把 A[i] 搬进 slots[i]、第二次从 slots[i] 读；退回按名次判
    时第一次搬进槽位 i（status 随之更新）、第二次从槽位 i 读 —— 用的槽位完全
    不同，各自却都自洽，输出一模一样。

    这条把两者拆开：让计划指向的槽位里装着**另一组 token**。

      1. 冷缓存 + 全未命中的计划（槽位映射 S，topk 集合 A）→ 缓存里 S[i] 装 A[i]
      2. 把 topk 换成一组新的 B（与 A 几乎不相交），计划改成「全部命中、就在 S[i]」
      3. 若走计划路：读到的是 A 的 KV，输出应当等于 **golden(A)**
         若退回按名次判：status[i] = A[i] ≠ B[i]，全部判 miss、从 CPU 池取 B，
         输出等于 **golden(B)**

    两个 golden 差得很远，一比就分得开。这条挂了要先看是哪一边 —— 等于
    golden(B) 说明计划被忽略了，两个都不等于说明取数取错了槽位。
    """
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    dtype = torch.bfloat16
    case = _build_case(topk, _BASELINE_Q_HEADS, dtype, bsz=_CACHE_BSZ, topk_buffer_size=slot_budget)
    tag = f"plan-obeyed,topk={topk},layout={slot_layout},{_budget_label(slot_budget)},bs={_CACHE_BSZ}"

    # 第一轮：冷缓存 + 全未命中的计划，把 A 按计划搬进 S
    _reset_selection_state(case)
    slots = _plan_slots(topk, _plan_capacity(case), slot_layout)
    _write_plan(case, slots, torch.empty(0, dtype=torch.int64))
    _call_fused(case)
    golden_a = _golden(case)  # A 还在 case["topk_indices"] 里

    # 换成另一组 topk：B
    bsz, k = case["bsz"], case["topk"]
    rng = np.random.RandomState(20260812 + topk)
    fresh = np.stack(
        [np.sort(rng.choice(case["max_seq_len"], k, replace=False).astype(np.int32)) for _ in range(bsz)]
    )
    case["topk_indices"] = (
        torch.tensor(fresh, dtype=torch.int32, device="npu").reshape(bsz, 1, 1, k).contiguous()
    )
    golden_b = _golden(case)

    # ⚠ _golden 返回的是 CPU 张量，算子输出在 NPU 上 —— 一律搬到 CPU 再比，
    #   直接相减会报 "Expected all tensors to be on the same device"（写这条时踩过）。
    golden_a_c = golden_a.detach().float().cpu()
    golden_b_c = golden_b.detach().float().cpu()

    # 两个 golden 必须真的不同，否则这条用例什么也证明不了
    gap = float((golden_a_c - golden_b_c).abs().max())
    assert gap > 10 * DEFAULT_ATOL, (
        f"[FSA_OVERLAP:{tag}] 两组 topk 的 golden 只差 {gap:.4e}，"
        f"分不开两条路 —— 换个随机种子"
    )

    # 第二轮：计划说「全部命中，就在 S[i]」，而缓存里 S[i] 装的还是 A
    _write_plan(case, slots, torch.arange(topk, dtype=torch.int64))
    out = _call_fused(case)
    _emit_metric(_finite_report(out, tag))
    _assert_finite(out, tag)

    out_c = out.detach().float().cpu()
    d_a = float((out_c - golden_a_c).abs().max())
    d_b = float((out_c - golden_b_c).abs().max())
    _emit_metric(f"[FSA_OVERLAP:{tag}] max_abs_diff vs golden(A)={d_a:.4e} vs golden(B)={d_b:.4e}")
    assert d_a < d_b, (
        f"[FSA_OVERLAP:{tag}] 输出更像 golden(B)（差 {d_b:.4e}）而不是 golden(A)（差 {d_a:.4e}）"
        f" —— 计划没被采纳，kernel 退回了按名次判"
    )
    torch.testing.assert_close(
        out.cpu(),
        golden_a.cpu(),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
        msg=lambda m: f"[FSA_OVERLAP:{tag}] 按计划读到的不是 S[i] 里那份数据。{m}",
    )
