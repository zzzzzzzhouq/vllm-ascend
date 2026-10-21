"""Time npu_fused_sparse_attention_overlap. **This file does not check precision.**

The name says "precisions" and that is misleading -- it has been read as a
correctness test at least once, by someone who grepped it, found no comparison,
and concluded the operator had no precision coverage at all. It has:

    tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py

That one carries an fp32 dense MLA baseline and error budgets. Correctness
judgements belong there (and in yq_files/tests/test_fsao_precision.py, which
carries an fp32 golden for both op signatures). This file measures time:
kernel_p50_us, pipeline occupancy, host overhead. ``_make_reference_load`` here
is a comparison *workload* for timing, not a golden reference.
"""

import argparse
import atexit
import gc
import math
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Literal


def _require_env() -> None:
    """环境没起来就当场退出，别让它挂在那儿。

    ⚠⚠ 这个检查必须在 `import torch_npu` / `memfabric_hybrid` **之前**跑 ——
      环境没起来时挂住的正是那几个导入。2026-08-12 一条忘了 source 的命令
      挂了 37 分钟，表现极具迷惑性：sweep_median 的构建戳照打（那段只用 git、
      不碰 NPU），进程 State=S 在睡、76 个线程阻塞在 futex，卡的功耗一切正常，
      profiler 的采集目录一个都没建。从外面看就是「跑了很久没动静」。

    ASCEND_CUSTOM_OPP_PATH 是最关键也最容易漏的那个 ——
    kernel 不在 vllm_ascend_C.so 里，而在独立的 CANN vendor 算子包里，
    这个变量才让运行时找得到它。
    """
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH") or os.environ.get("DSA_SKIP_ENV_CHECK"):
        return
    repo = pathlib.Path(__file__).resolve().parent.parent.parent
    sys.stderr.write(
        "\n❌ ASCEND_CUSTOM_OPP_PATH 为空 —— 没准备算子环境，跑下去会挂在 import。\n"
        "   本仓库（vllm-ascend）的一次性准备（详见 yq_files/README.md）：\n"
        f"     1. cd {repo}/csrc && bash build.sh --pkg --soc=ascend910_93 "
        "--ops=fused_sparse_attention_overlap\n"
        "     2. bash ./build_out/cann-ops-transformer-custom_linux-*.run \\\n"
        "            --quiet --install-path=$HOME/fsao_opp\n"
        f"     3. export ASCEND_CUSTOM_OPP_PATH=$HOME/fsao_opp/vendors/custom\n"
        "   （另外还需要 CANN set_env、memfabric_hybrid、pip install -e 本仓库）\n"
        "   ⚠ 确实想跳过这道检查：设 DSA_SKIP_ENV_CHECK=1\n\n"
    )
    raise SystemExit(2)


_require_env()

import numpy as np
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401
from memfabric_hybrid import offload

from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch_npu.npu.set_device(0)
torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
fused = torch.ops._C_ascend.npu_fused_sparse_attention_overlap
empty_swapped = getattr(torch_npu, "empty_with_swapped_memory", None)

# The operator ships in two shapes and they differ by a positional argument, so
# the caller cannot assume one. This branch's baseline (mte_fuse_0723) takes 14;
# upstream szy_fuse_new_token_adapt and everything built on it insert
# selection_membership_map at position 5, making 15.
#
# Getting it wrong is not symmetric. Passing 15 to the 14-argument build raises
# at the call, which is loud. Passing 14 to the 15-argument build shifts every
# later argument by one, which is not. So ask the schema rather than assume, and
# print it (OP_SCHEMA) so every run records which operator produced its numbers.
_OP_SCHEMA = str(fused.default._schema)
_OP_TAKES_MEMBERSHIP = "selection_membership_map" in _OP_SCHEMA

CpuKvBackend = Literal["offload", "swap_memory", "pinned"]


def _scratch_dir(prefix: str) -> str:
    """A private directory for this run's profiler captures, gone when it ends.

    Nothing in here outlives the run, and keeping it around is worse than
    useless: the profiler adds a subdirectory per capture instead of replacing,
    so any directory reused across runs holds several captures and the reader
    has to guess which one belongs to the call just made. It guessed wrong for
    a while -- every topk reported the first topk's kernel time. A fresh
    directory per run removes the guess, and removing it on exit means running
    this script leaves nothing behind on a shared box.

    Pass --kernel-time-dir / --profile-dir to keep the csv files instead.
    """
    path = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


# Match KVOffloadDecodeManager CPU KV pool alignment / init.
_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
_GIB = 1024 * 1024 * 1024

# MLA dims used by this UT (same as DeepSeek-style sparse MLA).
_BLOCK_SIZE = 128
_KVD = 512  # nope / kv_lora_rank
_KRD = 64  # rope
_DTYPE_SIZE = 2  # bf16

# selection_membership_map row width, selectable via --membership-width. Only
# reached on builds whose schema declares the argument; inert on the others.
#
# The placeholder is what the container baseline passes. It is not what the
# kernel strides by: Init() in fused_sparse_attention_overlap_service_vector_mla.h
# raises the row stride to at least 16400 whatever tensor it is handed, so an
# 8-wide row overruns by 16392 int16. The overrun is invisible in this UT's
# output (it never reads its own output back) but not harmless: it took the
# device down at bs=2 and bs=3 whenever the miss count crossed 64, which is why
# 'full' exists.
# The module-level value is the SAFE one, not the placeholder. --membership-width
# defaults to 'full' and overwrites this, but only main() reads that argument,
# and other scripts (yq_files/probe_pool_aliasing.py) import this module and
# call _build_inputs directly. Those callers used to get the 8-wide row -- the
# width that reproducibly took the device down -- purely because they never went
# through argument parsing. Anything narrower is now an explicit opt-in.
_MEMBERSHIP_PLACEHOLDER_INT16_COUNT = 8
_MEMBERSHIP_STORAGE_INT16_COUNT = 16400
_MEMBERSHIP_INT16_COUNT = _MEMBERSHIP_STORAGE_INT16_COUNT

# 外部计划（LRU 那条路）在 selection_membership_map 里的位置和编码。
# 取自上游 kv_offload_decode_manager.py 与 kernel 侧常量：
#   [0, 16376)                  membership 区，按 token 序号索引
#   [16384 - topk, 16384)       计划区，**跟 membership 区尾部重叠**
#   [16384, 16392)              控制区 8 个 int16
# 计划区取值：0 = 没分到槽位，正数 v = 命中读 v-1 号槽，
#            负数 v = 未命中、数据搬进 -v-1 号槽。
_MEMBERSHIP_CONTROL_OFFSET_INT16 = 16384
_MEMBERSHIP_CONTROL_INT16 = 8
_EXTERNAL_PLAN_READY_MARKER = 0x5A45
_PAIRED_SELECTION_COPY_MARKER = 0x5A56
_DIRECT_SELECTION_LAYOUT_MARKER = 0x5A44  # 控制区 [5]，打开直接寻址；[6] 放 capacity
# 默认打开 —— 生产就是这么发的。关掉是为了跟 2026-08-11 那轮对照同口径。
_PLAN_DIRECT = {"on": True}
# 命中槽位的分布。两种，结论会不一样：
#   contiguous —— 第 i 名放 i 号槽位。这是**按名次**方案（14 参数版）的天然
#                 布局，连续命中能合成一条 DataCopy
#   scattered  —— 槽位打散。这是 **LRU 稳定态**：一个 token 的槽位是它当初
#                 落在哪儿，跟名次无关，成不了段
# off = 不写计划，kernel 回退到按名次判据（14 参数版只有这一种）
PLAN_SLOT_LAYOUTS = ("off", "contiguous", "scattered")


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def _empty_aligned_cpu_tensor(
    shape: list[int],
    dtype: torch.dtype,
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> torch.Tensor:
    """Same allocation path as KVOffloadDecodeManager / model_runner CPU KV.

    Allocate raw int8 bytes from MemFabric, align, then view as ``dtype``.
    Avoids ``offload.empty(..., bfloat16)`` which some MemFabric builds reject,
    and matches production ``int8`` pool then ``.view(dtype).view(shape)``.
    """
    num_elements = int(np.prod(shape))
    nbytes = num_elements * torch.empty((), dtype=dtype).element_size()
    extra_bytes = alignment
    print(
        "OFFLOAD_EMPTY shape={} dtype={} nbytes={:.3f} GiB (+align {:.3f} MiB)".format(
            shape, dtype, nbytes / _GIB, extra_bytes / (1024 * 1024)
        ),
        flush=True,
    )
    try:
        raw = offload.empty([nbytes + extra_bytes], dtype=torch.int8, pin_memory=True)
    except Exception as exc:
        raise RuntimeError(
            f"offload.empty failed for {(nbytes + extra_bytes) / _GIB:.3f} GiB "
            f"(shape={shape}, dtype={dtype}): {type(exc).__name__}: {exc}. "
            "Check MemFabric pool size (--dram-size-gb) and host free/locked memory; "
            "avoid a second full host copy via torch.randn."
        ) from exc
    aligned = _align_memory(raw, alignment)[:nbytes]
    return aligned.view(dtype).view(shape)


def _fill_host_kv_inplace(tensor: torch.Tensor) -> None:
    """Fill host KV without a second full-size host malloc (timing UT only)."""
    tensor.zero_()


def init_offload_pool(
    dram_size_gb: float = 1.0,
    world_size: int = 1,
    rank_id: int = 0,
) -> None:
    """Mirror kv_offload_decode_manager MemFabric OffloadConfig initialize."""
    config = offload.OffloadConfig()
    config.device_id = torch_npu.npu.current_device()
    config.size = int(dram_size_gb * _GIB)
    config.world_size = world_size
    config.rank_id = rank_id
    print(
        "OFFLOAD_INIT device_id={} size_gb={} world_size={} rank_id={}".format(
            config.device_id, dram_size_gb, world_size, rank_id
        ),
        flush=True,
    )
    try:
        offload.initialize(config)
    except Exception as exc:
        raise RuntimeError(
            f"offload.initialize failed for {dram_size_gb} GiB pool: "
            f"{type(exc).__name__}: {exc}. Host may lack free/locked memory for "
            "MemFabric pinned pool; try smaller --dram-size-gb or free host RAM."
        ) from exc


def uninit_offload_pool() -> None:
    offload.uninitialize()


def estimate_cpu_kv_bytes(max_seq_len: int, block_size: int = _BLOCK_SIZE) -> int:
    """Bytes for one-layer full K + rope CPU cache covering max_seq_len tokens."""
    num_blocks = cdiv(max_seq_len, block_size)
    k_bytes = num_blocks * block_size * _KVD * _DTYPE_SIZE
    rope_bytes = num_blocks * block_size * _KRD * _DTYPE_SIZE
    # 2MiB align padding per tensor (upper bound).
    align_pad = 2 * _CPU_CACHE_ALIGNMENT
    return k_bytes + rope_bytes + align_pad


def recommended_dram_size_gb(max_seq_len: int, block_size: int = _BLOCK_SIZE) -> float:
    need_bytes = estimate_cpu_kv_bytes(max_seq_len, block_size)
    # One host KV copy + ~1GiB MemFabric headroom; round up to 0.5 GiB.
    return math.ceil((need_bytes / _GIB + 1.0) * 2) / 2.0


def _empty_aligned_host_tensor_pinned(
    shape: list[int],
    dtype: torch.dtype,
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> torch.Tensor:
    """Pinned host memory without MemFabric.

    Same contract as ``_empty_aligned_cpu_tensor`` (raw int8 + 2MiB align +
    view) but allocated by torch_npu's host allocator. Used when the MemFabric
    offload pool is unavailable on the host (e.g. HYBM HalMemSetAccess fails):
    the fused op only needs pinned host KV, not the pool itself.
    """
    num_elements = int(np.prod(shape))
    nbytes = num_elements * torch.empty((), dtype=dtype).element_size()
    raw = torch.empty([nbytes + alignment], dtype=torch.int8, pin_memory=True)
    aligned = _align_memory(raw, alignment)[:nbytes]
    return aligned.view(dtype).view(shape)


def _alloc_cpu_kv_tensor(
    backend: CpuKvBackend,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate host-side KV used by fused op.

    - offload: MemFabric ``offload.empty`` + 2MiB align (same as KVOffloadDecodeManager)
    - swap_memory: ``torch_npu.empty_with_swapped_memory``
    - pinned: plain pinned host memory (no MemFabric); functional fallback when
      the offload pool cannot be initialized on this host

    WARNING: on some torch_npu builds, swapped tensors SIGSEGV if you touch
    ``.shape`` / ``.device`` / ``.data_ptr()`` / ``zero_()`` / ``__repr__``.
    Only pass them into the fused op; do not print or introspect them.
    """
    if backend == "offload":
        return _empty_aligned_cpu_tensor(shape, dtype=dtype)
    if backend == "pinned":
        return _empty_aligned_host_tensor_pinned(shape, dtype=dtype)
    if backend == "swap_memory":
        if empty_swapped is None:
            raise RuntimeError("cpu-kv-backend=swap_memory requires torch_npu.empty_with_swapped_memory")
        # Keep args minimal; do not print/return-repr the result here.
        print(
            "SWAP_EMPTY request_shape={} dtype={} (no tensor introspection)".format(shape, dtype),
            flush=True,
        )
        return empty_swapped(tuple(shape), dtype=dtype, device="npu")
    raise ValueError(f"unknown cpu-kv-backend: {backend}")


def _alloc_shared_host_kv(
    topk: int,
    cpu_kv_backend: CpuKvBackend,
    max_seq_len: int | None,
) -> dict:
    """Allocate one shared full host KV for a topk/msl; reused across q_heads."""
    msl = max_seq_len if max_seq_len is not None else max(topk * 4, 256)
    assert msl >= topk, f"max_seq_len ({msl}) must be >= topk ({topk})"
    fmbn = cdiv(msl, _BLOCK_SIZE)
    dt = torch.bfloat16
    kv_shape = [fmbn, _BLOCK_SIZE, _KVD]
    rope_shape = [fmbn, _BLOCK_SIZE, _KRD]
    cpu_kv_bytes = estimate_cpu_kv_bytes(msl, _BLOCK_SIZE)
    # Print planned shapes only (Python lists). Never print swapped tensor.shape.
    print(
        "CPU_KV_PLAN backend={} max_seq_len={} block_size={} num_blocks={} "
        "k_shape={} rope_shape={} est_bytes={:.3f} GiB".format(
            cpu_kv_backend,
            msl,
            _BLOCK_SIZE,
            fmbn,
            kv_shape,
            rope_shape,
            cpu_kv_bytes / _GIB,
        ),
        flush=True,
    )
    full_kv_fused = _alloc_cpu_kv_tensor(cpu_kv_backend, kv_shape, dt)
    full_rope_fused = _alloc_cpu_kv_tensor(cpu_kv_backend, rope_shape, dt)

    if cpu_kv_backend == "offload":
        # Timing only: inplace zero; do NOT torch.randn (extra host malloc).
        _fill_host_kv_inplace(full_kv_fused)
        _fill_host_kv_inplace(full_rope_fused)
        print(
            "FULL_KV_HOST backend=offload shape={} rope_shape={} dtype={} "
            "device={} ptr_align={} bytes={:.3f} GiB".format(
                tuple(full_kv_fused.shape),
                tuple(full_rope_fused.shape),
                dt,
                full_kv_fused.device,
                full_kv_fused.data_ptr() % _CPU_CACHE_ALIGNMENT == 0,
                (
                    full_kv_fused.numel() * full_kv_fused.element_size()
                    + full_rope_fused.numel() * full_rope_fused.element_size()
                )
                / _GIB,
            ),
            flush=True,
        )
    else:
        # swap_memory: skip zero_/shape/device/data_ptr — known SIGSEGV on some builds.
        # Leave allocation as-is for timing; values are uninitialized.
        print(
            "FULL_KV_HOST backend=swap_memory requested_k_shape={} "
            "requested_rope_shape={} dtype={} "
            "(skip tensor introspection to avoid SIGSEGV)".format(kv_shape, rope_shape, dt),
            flush=True,
        )

    return {
        "msl": msl,
        "num_blocks": fmbn,
        "full_kv_fused": full_kv_fused,
        "full_rope_fused": full_rope_fused,
    }


def _selection_slots_per_row(topk: int, topk_buffer_size: int) -> int:
    """一行 selection 缓存有几个槽位，对应生产的 kv_offload_decode_config.topk_buffer_size。

    ⚠ 这个数**不等于 topk**，2026-08-12 之前这个脚本一直当它等于。生产默认 4096
      而 topk 通常是 2048，槽位是 topk 的两倍：
        vllm_ascend/attention/sfa_kv_offload.py
            lru_resident_capacity = topk_buffer_size                  # 默认 4096
            cache_blocks_per_row  = lru_resident_capacity // block_size
            递给算子的是 selection_kv_block_table[:, :cache_blocks_per_row]
      算子不认 topk，只认块表最后一维（tiling.cpp 的 selectionBlockTableStride
      取的就是这个张量的形状），所以**只要块表递宽一点，kernel 自己就按 4096 走，
      不用改算子源码、也不用重编**。

      量成 2048 会让命中显得比实际连续：槽位摊在一半的地址空间上，scattered
      布局下更容易凑出成段的命中，而成段与否正是快路的触发条件。
    """
    if topk_buffer_size % _BLOCK_SIZE != 0:
        raise ValueError(
            f"--topk-buffer-size 必须是 {_BLOCK_SIZE} 的整数倍（生产同样要求），得到 {topk_buffer_size}"
        )
    if topk_buffer_size < topk:
        raise ValueError(
            f"--topk-buffer-size 不能小于 topk（生产会直接拒绝）：{topk_buffer_size} < {topk}"
        )
    return topk_buffer_size


def _build_inputs(
    topk: int,
    q_heads: int,
    host_kv: dict,
    kv_heads: int = 1,
    bsz: int = 1,
    topk_buffer_size: int = 0,
):
    assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads"
    assert kv_heads == 1, "fused op currently requires kv_heads=1"
    assert topk > 0
    assert bsz > 0

    # Decode shape: one query token per request, bsz requests in the batch.
    # Every request gets its own random topk (see topk_np below), so a larger
    # batch is not just the same row repeated.
    seq = 1
    hd, krd, kvd = _KVD, _KRD, _KVD
    sbs = fbs = _BLOCK_SIZE
    stbs = 1
    msl = host_kv["msl"]
    fmbn = host_kv["num_blocks"]
    # 0 = 老行为（槽位数 = topk），留着是为了跟 2026-08-12 之前的数据同口径。
    slots = _selection_slots_per_row(topk, topk_buffer_size or cdiv(topk * stbs, sbs) * sbs)
    smbn = slots // sbs
    dt = torch.bfloat16
    scale = 1.0 / math.sqrt(kvd)
    torch.manual_seed(910000 + topk + q_heads * 17)
    np.random.seed((910000 + topk + q_heads * 17) % (2**32 - 1))

    q = torch.randn(bsz * seq, q_heads, hd, dtype=dt, device="npu")
    q_rope = torch.randn(bsz * seq, q_heads, krd, dtype=dt, device="npu")
    q_fused = torch.cat([q, q_rope], dim=-1).contiguous()

    full_bt = torch.arange(fmbn, dtype=torch.int32, device="npu").unsqueeze(0).expand(bsz, -1).contiguous()
    # These two lines look like each other and do not mean the same thing.
    #
    # The query layout is TND, so full_q_actual_seq is CUMULATIVE: entry b is
    # where request b's tokens end, and the last entry must equal the total
    # token count query.size(0) == bsz*seq. The kernel takes the difference of
    # consecutive entries as the request's length
    # (GetBalanceActualSeqLengths in fused_sparse_attention_overlap_kernel_mla.h).
    #
    # The KV layout is PA_BSND, not TND, so full_kv_actual_seq is NOT
    # cumulative: it is one plain length per request, read straight out at
    # index b (GetActualSeqLenKV, same file).
    #
    # Writing [seq]*bsz for the query side is what this file did until
    # 2026-08-08 and it is silent: nothing rejects it. Every difference after
    # the first is seq-seq == 0, so requests 1..bsz-1 have zero query tokens,
    # totalBaseNum collapses to 1, usedCoreNum collapses to 1, and one core
    # computes one request while the timing gets reported as a bsz-request
    # call. Upstream never saw it because upstream hard-coded bsz = 1, where
    # the two formulas coincide.
    actual_q = torch.tensor([(i + 1) * seq for i in range(bsz)], dtype=torch.int32, device="npu")
    actual_k = torch.tensor([msl] * bsz, dtype=torch.int32, device="npu")
    assert int(actual_q[-1].item()) == bsz * seq, (
        "full_q_actual_seq must be cumulative and end at the total token count: "
        f"got {int(actual_q[-1].item())}, expected {bsz * seq}"
    )
    topk_np = np.zeros((bsz * seq, kv_heads, topk), dtype=np.int32)
    for row in range(bsz * seq):
        for kh in range(kv_heads):
            topk_np[row, kh] = np.sort(np.random.choice(msl, topk, replace=False).astype(np.int32))
    topk_fused = torch.tensor(topk_np, dtype=torch.int32, device="npu").reshape(bsz, seq, kv_heads, topk).contiguous()

    total_sel = smbn * bsz * seq * kv_heads
    sel_kv = torch.zeros(total_sel, sbs, kvd, dtype=dt, device="npu")
    sel_rope = torch.zeros(total_sel, sbs, krd, dtype=dt, device="npu")
    sel_bt = torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(bsz * seq * kv_heads, smbn)
    sel_status = torch.full((bsz, seq, kv_heads, topk + 1), -1, dtype=torch.int32, device="npu")
    # selection_membership_map, only on builds whose schema declares it. None
    # otherwise, and _call_fused then drops the argument entirely.
    #
    # The width is not free to choose. The kernel does not stride by the tensor
    # it is handed; from fused_sparse_attention_overlap_service_vector_mla.h:
    #   MEMBERSHIP_MAX_TOKEN / MAP_INT16_COUNT = 16376
    #   ALIGNMENT_INT16_COUNT = BYTE_BLOCK / sizeof(int16) = 32 / 2 = 16
    #   CONTROL_OFFSET = ceil(16376 / 16) * 16 = 16384
    #   STORAGE = ceil((16384 + 8) / 16) * 16  = 16400
    # Init() raises the stride to at least STORAGE whatever tensor it is given,
    # so anything narrower is written past its end once the membership path is
    # reached at all. See --membership-width.
    sel_membership = (
        torch.full((total_sel, _MEMBERSHIP_INT16_COUNT), -1, dtype=torch.int16, device="npu")
        if _OP_TAKES_MEMBERSHIP
        else None
    )

    return {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "topk": topk,
        "msl": msl,
        "num_blocks": fmbn,
        "slots": slots,
        "stbs": stbs,
        "scale": scale,
        "q_fused": q_fused,
        "full_kv_fused": host_kv["full_kv_fused"],
        "full_rope_fused": host_kv["full_rope_fused"],
        "full_bt": full_bt,
        "actual_q": actual_q,
        "actual_k": actual_k,
        "topk_fused": topk_fused,
        "sel_kv": sel_kv,
        "sel_rope": sel_rope,
        "sel_bt": sel_bt,
        "sel_status": sel_status,
        "sel_membership": sel_membership,
    }


def _is_float(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


_INPUT_CHECK_DONE = False


def _report_input_layout(inp) -> None:
    """One-off: which inputs are not contiguous.

    The profile shows two aclnnContiguous_Slice ops per call, and the host KV
    tensors come out of _align_memory as a slice of the offload pool, so a
    non-contiguous view would make PyTorch fix it up on every call -- outside
    the kernel, which is exactly where the jumping was measured to live, and
    something a plain matmul on preallocated tensors never pays.
    """
    keys = (
        "q_fused",
        "sel_rope",
        "sel_kv",
        "sel_bt",
        "sel_status",
        "sel_membership",
        "topk_fused",
        "full_rope_fused",
        "full_kv_fused",
        "full_bt",
        "actual_k",
        "actual_q",
    )
    for key in keys:
        tensor = inp.get(key)
        if not torch.is_tensor(tensor):
            continue
        try:
            print(
                "INPUT {:18s} contiguous={} storage_offset={} shape={}".format(
                    key, tensor.is_contiguous(), tensor.storage_offset(), tuple(tensor.shape)
                ),
                flush=True,
            )
        except Exception as exc:  # swapped-memory tensors reject introspection
            print(f"INPUT {key:18s} (not introspectable: {type(exc).__name__})", flush=True)


def _call_fused(inp):
    global _INPUT_CHECK_DONE
    if not _INPUT_CHECK_DONE:
        _INPUT_CHECK_DONE = True
        _report_input_layout(inp)
    # Built as a list because selection_membership_map sits in the middle: it is
    # position 5 on the builds that have it, and absent on the ones that do not,
    # which shifts everything after it. See _OP_TAKES_MEMBERSHIP.
    args = [
        inp["q_fused"],  # query
        inp["sel_rope"],  # selection_k_rope
        inp["sel_kv"],  # selection_kv_cache
        inp["sel_bt"],  # selection_kv_block_table
        inp["sel_status"],  # selection_kv_block_status
    ]
    if _OP_TAKES_MEMBERSHIP:
        args.append(inp["sel_membership"])  # selection_membership_map
    args += [
        inp["topk_fused"],  # selection_topk_indices
        inp["full_rope_fused"],  # full_k_rope
        inp["full_kv_fused"],  # full_kv_cache
        inp["full_bt"],  # full_kv_block_table
        inp["actual_k"],  # full_kv_actual_seq
        inp["actual_q"],  # full_q_actual_seq
        inp["scale"],  # scale_value
        1,  # sparse_block_size
        inp["stbs"],  # selection_topk_block_size
    ]
    return fused(
        *args,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )


_HIT_INDEX_CACHE = {}

HIT_LAYOUTS = ("prefix", "strided", "random")


def _hit_indices(k: int, n_hit: int, layout: str, device) -> torch.Tensor:
    """Which of the k slots are hits, for the non-default layouts."""
    key = (k, n_hit, layout)
    cached = _HIT_INDEX_CACHE.get(key)
    if cached is None:
        if layout == "strided":
            pos = (torch.arange(n_hit, dtype=torch.float64) * k / n_hit).long()
        elif layout == "random":
            gen = torch.Generator().manual_seed(k * 7919 + n_hit)
            pos = torch.randperm(k, generator=gen)[:n_hit].sort().values
        else:
            raise ValueError(f"unknown hit layout: {layout}")
        cached = pos.to(device=device, dtype=torch.long)
        _HIT_INDEX_CACHE[key] = cached
    return cached


def _apply_hit_ratio(inp, hit_ratio: float) -> int:
    """Set selection status so the next fused call has the target positional hit ratio.

    Kernel hit condition is positional: ``status[i] == topk[i]``. Hit slots get
    ``topk[i]``, miss slots get ``-1`` (H2D from full KV).

    ``inp["hit_layout"]`` decides *which* slots hit. The ratio alone does not
    pin down the cost: what the transfer plan sees is the number of separate
    miss runs, and that depends entirely on the layout.

    - ``prefix`` (default) -- the first ``round(hit_ratio*topk)`` slots. Misses
      then form ONE contiguous run, the friendliest case a transfer plan can
      get. Every number measured up to 2026-08-04 used this.
    - ``strided`` -- hits spread as evenly as possible, which maximises the
      number of miss runs (~topk/2 of them at ratio 0.5). Worst case.
    - ``random`` -- hits drawn without replacement under a fixed seed; closest
      to what a real selection produces. Deterministic across runs and builds.

    Returns the number of hit slots per row.
    """
    if not 0.0 <= hit_ratio <= 1.0:
        raise ValueError(f"hit_ratio must be in [0, 1], got {hit_ratio}")
    topk = inp["topk_fused"]
    status = inp["sel_status"]
    k = topk.shape[-1]
    n_hit = int(round(hit_ratio * k))
    n_hit = max(0, min(k, n_hit))

    # Build each (n_hit, layout) pattern once and copy it in from then on, so
    # the per-iter prepare costs the same whatever the layout is -- an indexed
    # scatter is slower than a slice, and that must not leak into the timings.
    layout = inp.get("hit_layout", "prefix")
    cache = inp.setdefault("_status_cache", {})
    template = cache.get((n_hit, layout))
    if template is None:
        # status layout: [B, S, H, K+1]; last elem is side-channel, keep -1.
        template = torch.full_like(status, -1)
        if n_hit > 0:
            if layout == "prefix" or n_hit == k:
                template[..., :n_hit] = topk[..., :n_hit]
            else:
                idx = _hit_indices(k, n_hit, layout, status.device)
                template[..., idx] = topk[..., idx]
        cache[(n_hit, layout)] = template
    status.copy_(template)
    # 计划跟 status 用同一份名次集合。plan_slots=off 时这里空转，14 参数版
    # 也空转 —— 那两种情况下 kernel 只看 status。
    _write_external_plan(inp, n_hit)
    return n_hit


def _write_external_plan(inp, n_hit: int) -> None:
    """把外部计划写进 selection_membership_map，让 kernel 走 LRU 那条路。

    只在两个条件都成立时干活：算子声明了 selection_membership_map
    （15 参数版），且 inp["plan_slots"] 不是 off。14 参数版上整个函数是空转。

    命中/未命中的**名次分布**跟 _apply_hit_ratio 用的是同一份 _hit_indices，
    所以 --hit-layout 对两条路含义一致；这个函数只决定命中的名次映射到**哪个
    槽位**。两个轴必须分开：名次分布两边可以对齐，槽位映射是各自设计的性质。

    ⚠ 按 (n_hit, hit_layout, slot_layout) 缓存整块模板，每步只做一次 copy_ ——
      索引散写比切片慢，那点开销不能漏进计时。
    """
    membership = inp.get("sel_membership")
    slot_layout = inp.get("plan_slots", "off")
    if membership is None:
        return
    if slot_layout not in PLAN_SLOT_LAYOUTS:
        raise ValueError(f"unknown plan slot layout: {slot_layout}")
    if slot_layout == "off":
        # ⚠⚠ 不能直接 return。--plan-slots 可以在一趟里轮换，而 kernel 只看
        #   控制区 [1] 的就绪标记：上一个点写下的计划还在图里，这个点就仍然
        #   走计划路，量出来的「无 LRU」其实是上一档 LRU。把整行抹回 -1
        #   （标记随之清掉）才是真的关掉。单值运行时这一步是幂等的。
        cleared = inp.get("_plan_cleared_template")
        if cleared is None:
            cleared = torch.full_like(membership, -1)
            inp["_plan_cleared_template"] = cleared
        membership.copy_(cleared)
        return

    topk = inp["topk_fused"]
    k = topk.shape[-1]
    hit_layout = inp.get("hit_layout", "prefix")
    # 每行的槽位数 = 块表一行的块数 × 块大小。计划里的槽位号必须落在这个范围
    # 内，否则 kernel 按 blockTable[slot / 128] 取块就越过本行的条目、串到别的
    # 请求上 —— 上游自己的 perf 用例就是这么错的（capacity 4096 / 缓存 2048）。
    capacity = inp["sel_bt"].shape[-1] * _BLOCK_SIZE
    if capacity < k:
        raise ValueError(f"槽位数 {capacity} 少于 topk {k}，计划编不出来")

    cache = inp.setdefault("_plan_cache", {})
    key = (n_hit, hit_layout, slot_layout, k, capacity)
    template = cache.get(key)
    if template is None:
        if slot_layout == "contiguous":
            slots = torch.arange(k, dtype=torch.int64)
        else:
            gen = torch.Generator().manual_seed(capacity * 7919 + k)
            slots = torch.randperm(capacity, generator=gen)[:k]
        enc = (slots + 1).to(torch.int32)
        # 先全标成未命中，再把命中的那些名次翻正 —— 跟 _apply_hit_ratio 用的是
        # 同一份名次集合
        enc = -enc
        if n_hit > 0:
            if hit_layout == "prefix" or n_hit == k:
                enc[:n_hit] = -enc[:n_hit]
            else:
                # ⚠ _hit_indices 按 (k, n_hit, layout) 缓存，**不看 device** ——
                #   _apply_hit_ratio 先用 NPU device 填过缓存，这里要 CPU 也会拿回
                #   NPU 张量，跟 CPU 的 enc 一索引就报 device 不一致。显式搬回来。
                idx = _hit_indices(k, n_hit, hit_layout, torch.device("cpu")).cpu()
                enc[idx] = -enc[idx]
        row = torch.full((_MEMBERSHIP_INT16_COUNT,), -1, dtype=torch.int16)
        plan_start = _MEMBERSHIP_CONTROL_OFFSET_INT16 - k
        row[plan_start:_MEMBERSHIP_CONTROL_OFFSET_INT16] = enc.to(torch.int16)
        ctl = _MEMBERSHIP_CONTROL_OFFSET_INT16
        row[ctl:ctl + _MEMBERSHIP_CONTROL_INT16] = 0
        row[ctl + 1] = _EXTERNAL_PLAN_READY_MARKER
        row[ctl + 2] = k
        row[ctl + 3] = plan_start
        # 第 5、6 格：直接寻址（界是 capacity）。第 7 格：配对搬运。
        if _PLAN_DIRECT["on"]:
            if capacity > 32767:
                raise ValueError(f"capacity {capacity} 放不进 int16 的控制格")
            row[ctl + 5] = _DIRECT_SELECTION_LAYOUT_MARKER
            row[ctl + 6] = capacity
        row[ctl + 7] = _PAIRED_SELECTION_COPY_MARKER

        # ⚠⚠ 第 4 格是**物理行号**，而且 [5][6][7] 三格全都嵌在它 >= 0 这个条件里
        #   （service_vector_mla.h:3297）。所以这一格有三种填法，两种是错的：
        #     填 0   → 每个请求都去读第 0 行的缓存。bs=16 时十六个请求共用一行，
        #              局部性远好于真实 —— 2026-08-11 那轮上游对照就是这么跑的
        #     填 -1  → 整个块被跳过，[5][6] 写了也白写，退回**块表**寻址。那条路上
        #              有一道过紧的界（GetSelectionSlotTokenOffset 里
        #              `slot >= sparseBlockCount`，界是 topk），于是**槽位号 >= topk
        #              的命中全被静默判成未命中**。capacity 等于 topk 时看不出来，
        #              capacity 4096 而 topk 2048 时一半中招
        #     逐行填自己的行号 → 对。生产就是这么发的（kv_offload_decode.cpp:166
        #              按 logical_row 逐行写 physical_row）
        #   ⚠ 三种都不报错，只是数字不对。2026-08-12 一天之内先后栽了前两种。
        rows = int(membership.shape[0])
        if rows > 32767:
            raise ValueError(f"membership 行数 {rows} 放不进 int16 的控制格")
        block = row.unsqueeze(0).repeat(rows, 1)
        if _PLAN_DIRECT["on"]:
            # membership 的行数是缓存行数，比请求数多。多出来的行 kernel 不读，
            # 填 -1 而不是行号 —— 填行号的话那些行指向的缓存位置根本不存在。
            physical = torch.full((rows,), -1, dtype=torch.int16)
            bsz = int(inp["topk_fused"].shape[0])
            physical[:bsz] = torch.arange(bsz, dtype=torch.int16)
            block[:, ctl + 4] = physical
        else:
            block[:, ctl + 4] = -1
        template = block.to(membership.device).contiguous()
        cache[key] = template
    membership.copy_(template)


def check_plan_all_miss_matches_fallback(inp) -> bool:
    """全 miss 的计划，输出必须等于完全不给计划时的输出。

    这条是上游那套唯一的正确性抓手 —— 它自带的 perf 用例一个校验都没有，
    「kernel 照着计划把本该搬的行跳过了」这类错完全不显形。

    立论：两种设定下每一行都得从 CPU 池搬。
      ① 不给计划（plan=off）+ status 全 -1  → 按名次判据，全 miss
      ② 计划里每个名次都标成未命中          → 按计划，全 miss
    合并缓冲里的内容因此该逐位相同，注意力输出也该逐位相同。

    ⚠ 只在 15 参数版上有意义。14 参数版没有计划这条路，直接判过。
    ⚠ 输出用 bf16 逐位比，不设容差：两边的浮点运算次序完全一样，差一位就是错。
    """
    if inp.get("sel_membership") is None:
        print("PLAN_CHECK skipped: 这个构建没有 selection_membership_map（14 参数版）", flush=True)
        return True

    saved = inp.get("plan_slots", "off")
    try:
        inp["plan_slots"] = "off"
        _apply_hit_ratio(inp, 0.0)
        base_out = _call_fused(inp).clone()
        torch.npu.synchronize()

        inp["plan_slots"] = "contiguous"
        _apply_hit_ratio(inp, 0.0)          # n_hit=0 → 计划里每个名次都是负数
        plan_out = _call_fused(inp).clone()
        torch.npu.synchronize()
    finally:
        inp["plan_slots"] = saved

    same = torch.equal(base_out, plan_out)
    if same:
        print("PLAN_CHECK ok: 全 miss 的计划与回退路径逐位相同", flush=True)
        return True
    diff = (base_out.float() - plan_out.float()).abs()
    bad = int((base_out != plan_out).sum())
    print(
        "PLAN_CHECK FAILED: 全 miss 的计划与回退路径不一致 —— "
        f"{bad} 个元素不同，max_abs_diff={float(diff.max()):.4e}。"
        "计划让 kernel 跳过了本该搬的行，或者槽位算错了。",
        flush=True,
    )
    return False


def _sanity_check_output(out, tag: str) -> None:
    """把一次预热调用的输出读回来看一眼。**只在计时区外做一次。**

    ⚠ 2026-08-11 之前这个脚本**从不读回输出** —— 一份算错的构建照样给出漂亮的
      时间。当天在 upstream 分支上就凑出过「一版 .so + 另一版 kernel」，那次侥幸
      在加载期就报错；换成参数错位而没崩的情形，我们会拿到一组看着完全正常的
      数字。精度用例查得到，但性能扫描常常先跑、也不一定有人接着跑精度。

    代价是每个配置一次 D2H（bs=16/q_heads=16 时约 260 KB），在预热之后、
    计时之前做，不进任何一条采样。
    """
    host = out.detach().float().cpu()
    n_bad = int((~torch.isfinite(host)).sum())
    peak = float(host.abs().max()) if host.numel() else 0.0
    print("OUT_SANITY {} nonfinite={} peak={:.4e}".format(tag, n_bad, peak), flush=True)
    if n_bad:
        raise RuntimeError(
            "[{}] 输出里有 {} 个非有限值 —— 构建或输入有问题，这一趟的时间没有意义".format(tag, n_bad)
        )
    # ⚠ 全零在这个脚本里是**正常的**：CPU 池由 _fill_host_kv_inplace 填零
    #   （计时不关心数值，这样省掉一次几 GB 的主机分配），softmax 出来自然是 0。
    #   所以这里只提示、不报错 —— 2026-08-12 第一版写成 raise，重测上游直接被
    #   自己拦下。代价是：**这条体检只能查出 NaN/Inf，查不出「数值算错了」**，
    #   后者归精度用例管。
    if peak == 0.0:
        print("OUT_SANITY {} 输出全零（CPU 池本来就是零，属正常）".format(tag), flush=True)


def _prime_selection_cache(inp) -> None:
    """One full-miss call so sel_kv/sel_rope hold current topk (for realistic hits)."""
    _apply_hit_ratio(inp, 0.0)
    out = _call_fused(inp)
    torch.npu.synchronize()
    _sanity_check_output(
        out, "prime topk={} bs={}".format(inp["topk_fused"].shape[-1], inp["topk_fused"].shape[0])
    )


def _make_reference_load(size: int = 4096):
    """A plain cube-only matmul, measured right next to the operator.

    Measured on its own across six processes this sits at 0.70% while the
    operator sits at 6-13%, but those were separate runs, so the machine could
    always have been different. Interleaving the two inside one process removes
    that objection: whenever the operator's timing moves and this does not,
    whatever is moving is the operator's, not the machine's.

    Cube only, no vector work, no cross-core handoff -- which is exactly the
    thing the operator has and this does not.
    """
    a = torch.randn(size, size, dtype=torch.bfloat16, device="npu")
    b = torch.randn(size, size, dtype=torch.bfloat16, device="npu")
    out = torch.empty(size, size, dtype=torch.bfloat16, device="npu")

    def run_once():
        torch.matmul(a, b, out=out)

    return run_once


def _result_flag(spread_pct: float, worst_std: float) -> str:
    """What to append to a RESULT line.

    Reproducibility is the spread across repeats, NOT the std inside one run.
    p50 already discards the slow tail a high std comes from: the all-hit point
    under graph mode has been seen at std 24% with its three repeats agreeing
    to 0.8%, and that p50 is perfectly usable. Flagging on std alone condemned
    three such points in one six-process run.
    """
    if spread_pct > 3.0:
        return "  <-- NOT REPRODUCIBLE, rerun"
    if worst_std >= 10.0:
        return "  (high in-run std; repeats still agree)"
    return ""


# Set from --profile-iters. The first _bench_loop of the run captures a profile
# and then clears this, so profiling happens once and the normal measurement
# that follows still tells us which tier this run landed in.
_PROFILE = {"dir": None, "iters": 0}
# --check-plan：只跑那条不变量就退出，不做任何计时。
_CHECK_PLAN = {"on": False}


def _summarise_profile(out_dir: str, e2e_us=None, quiet: bool = False, prefer=None):
    """Print a compact per-kernel table from a profiler output directory.

    ``prefer`` names the kernel the caller actually wants reported. Without it
    the headline is whichever kernel had the largest total duration in the
    window, which is a guess -- and it guesses wrong whenever the operator
    barely made it into the trace, handing back one of the harness's own setup
    kernels (a 2.8us Slice from applying the hit pattern) as if it were the
    operator. Passing the name turns the guess into a lookup, so the value comes
    out right even in a capture where something else happened to dominate.
    """
    import csv
    import glob
    import os

    result = {}
    files = glob.glob(os.path.join(out_dir, "*", "ASCEND_PROFILER_OUTPUT", "*.csv"))
    if not files:
        print(f"PROFILE no csv under {out_dir}; what is there:", flush=True)
        for root, _dirs, names in os.walk(out_dir):
            for name in names[:25]:
                print("   ", os.path.join(root, name), flush=True)
        return result
    # tensorboard_trace_handler adds one subdirectory per capture, so a reused
    # out_dir accumulates them. Picking sorted()[0] returned whichever capture
    # sorted first by host_pid_timestamp -- an older run's, in practice. That
    # made every topk report the first topk's kernel time, and made twelve
    # independent processes all report one leftover file. Take the newest, and
    # say so when there is more than one to choose from.
    roots = glob.glob(os.path.join(out_dir, "*", "ASCEND_PROFILER_OUTPUT"))
    if len(roots) > 1:
        print(
            f"PROFILE WARNING: {len(roots)} captures under {out_dir}, using the newest",
            flush=True,
        )
    kernels = [f for f in files if "kernel" in os.path.basename(f).lower()] or files
    path = max(kernels, key=os.path.getmtime)
    if not quiet:
        print(f"PROFILE reading {os.path.basename(path)}", flush=True)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("PROFILE csv is empty", flush=True)
        return result
    cols = list(rows[0].keys())
    if not quiet:
        print("PROFILE columns: " + " | ".join(cols), flush=True)

    def pick(*wanted):
        for w in wanted:
            for c in cols:
                if w in c.lower():
                    return c
        return None

    name_col = pick("name", "op name", "type")
    dur_col = pick("duration", "time")
    if name_col is None or dur_col is None:
        for row in rows[:5]:
            print("PROFILE row: " + " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        return result

    ratio_cols = [c for c in cols if "ratio" in c.lower() or "util" in c.lower()]
    agg = {}
    for row in rows:
        try:
            dur = float(row[dur_col])
        except (TypeError, ValueError):
            continue
        key = row[name_col]
        rec = agg.setdefault(key, {"n": 0, "total": 0.0, "ratios": {c: [] for c in ratio_cols}})
        rec["n"] += 1
        rec["total"] += dur
        for c in ratio_cols:
            try:
                rec["ratios"][c].append(float(row[c]))
            except (TypeError, ValueError):
                pass
    # Per-call spread of the kernel itself. The end-to-end p50 jumps between
    # tiers ~12% apart while the kernel is only 171us of a 320us call, so the
    # question is whether the kernel duration jumps with it. If these
    # individual durations are tight while the end-to-end number is not, the
    # jump happens outside the kernel and there is no point looking inside.
    headline = prefer if prefer in agg else (max(agg, key=lambda k: agg[k]["total"]) if agg else None)
    if prefer and headline != prefer:
        print(
            "PROFILE WARNING {} 不在这次采集里，采到的是 {}".format(prefer, " ".join(sorted(agg)) or "（空）"),
            flush=True,
        )
    if headline is not None:
        hd = sorted(float(r[dur_col]) for r in rows if r[name_col] == headline and _is_float(r[dur_col]))
        if hd:
            result["kernel"] = headline
            mid = hd[len(hd) // 2]
            result["kernel_p50_us"] = mid
            result["kernel_n"] = len(hd)
            # How many distinct kernels the op launched. p50 describes only the
            # biggest one, so a change that moves work from one kernel into
            # another would not show up here unless this says 1.
            result["kernel_count"] = len(agg)
            # range is (max - min), which is decided by the two unluckiest of
            # the n calls and grows with n -- fine for spotting that something
            # went wrong, useless for saying how tight the run was. p10-p90
            # answers that instead, so keep both.
            result["kernel_range_pct"] = 100.0 * (hd[-1] - hd[0]) / mid if mid > 0 else 0.0
            lo, hi = hd[len(hd) // 10], hd[(9 * len(hd)) // 10]
            result["kernel_p10_90_pct"] = 100.0 * (hi - lo) / mid if mid > 0 else 0.0
            result["ratios"] = {
                c: sum(agg[headline]["ratios"][c]) / len(agg[headline]["ratios"][c])
                for c in ratio_cols
                if agg[headline]["ratios"][c]
            }
            # Which kernel this number is about, and what else ran in the window.
            # A count of 2 could mean the operator is split across two device
            # kernels -- in which case p50 covers one of them -- or it could be
            # the harness's own status.copy_ between calls. Only the names say
            # which, and the table below that would have said so is behind the
            # quiet return, so carry the names out here.
            grand = sum(r["total"] for r in agg.values()) or 1.0
            result["kernel_share_pct"] = 100.0 * agg[headline]["total"] / grand
            result["kernel_others"] = [
                (k, agg[k]["n"], 100.0 * agg[k]["total"] / grand)
                for k in sorted(agg, key=lambda k: -agg[k]["total"])[1:4]
            ]
    if quiet:
        return result

    for key in sorted(agg, key=lambda k: -agg[k]["total"])[:2]:
        durs = []
        for row in rows:
            if row[name_col] != key:
                continue
            try:
                durs.append(float(row[dur_col]))
            except (TypeError, ValueError):
                pass
        if len(durs) < 4:
            continue
        durs.sort()
        n = len(durs)
        mid = durs[n // 2]
        print(
            "PROFILE dist kernel={} n={} min={:.1f} p10={:.1f} p50={:.1f} p90={:.1f} max={:.1f} range={:.1f}%".format(
                key[:40],
                n,
                durs[0],
                durs[max(0, int(n * 0.1))],
                mid,
                durs[min(n - 1, int(n * 0.9))],
                durs[-1],
                100.0 * (durs[-1] - durs[0]) / mid if mid > 0 else 0.0,
            ),
            flush=True,
        )

    # The decisive comparison, and the one an earlier attempt got wrong: pair
    # each call's kernel duration with the end-to-end time of THAT SAME call.
    # Reading a kernel duration from the profiled calls against an end-to-end
    # number from a later measurement compares two different stretches of time
    # -- which is how a 0.7% kernel spread got set against a 19% end-to-end
    # spread and mistaken for evidence.
    if e2e_us:
        target = max(agg, key=lambda k: agg[k]["total"])
        krows = [r for r in rows if r[name_col] == target]
        start_col = pick("start time")
        if start_col:
            try:
                krows.sort(key=lambda r: float(r[start_col]))
            except (TypeError, ValueError):
                pass
        kdur = []
        for row in krows:
            try:
                kdur.append(float(row[dur_col]))
            except (TypeError, ValueError):
                pass
        paired = min(len(kdur), len(e2e_us))
        if paired >= 4:
            outside = [e2e_us[i] - kdur[i] for i in range(paired)]
            for tag, vals in (
                ("e2e", e2e_us[:paired]),
                ("kernel", kdur[:paired]),
                ("outside", outside),
            ):
                ordered = sorted(vals)
                mid = ordered[len(ordered) // 2]
                print(
                    "PROFILE paired {:8s} n={} min={:.1f} p10={:.1f} p50={:.1f} "
                    "p90={:.1f} max={:.1f} range={:.1f}%".format(
                        tag,
                        paired,
                        ordered[0],
                        ordered[max(0, int(paired * 0.1))],
                        mid,
                        ordered[min(paired - 1, int(paired * 0.9))],
                        ordered[-1],
                        100.0 * (ordered[-1] - ordered[0]) / mid if mid > 0 else 0.0,
                    ),
                    flush=True,
                )

    for key, rec in sorted(agg.items(), key=lambda kv: -kv[1]["total"])[:8]:
        extra = ""
        for c in ratio_cols:
            vals = rec["ratios"][c]
            if vals:
                extra += f" {c}={sum(vals) / len(vals):.3f}"
        print(
            "PROFILE kernel={} n={} total_us={:.1f} avg_us={:.2f}{}".format(
                key[:60], rec["n"], rec["total"], rec["total"] / rec["n"], extra
            ),
            flush=True,
        )
    return result


def _capture_profile(run_once, prepare, iters: int, out_dir: str):
    """Profile the operator itself.

    Everything variable from outside has been ruled out -- platform, op
    library, host pool, tensor placement, DMA, membership width, host launch,
    core count, working set, workspace size -- and the jumping shifts the whole
    distribution rather than a tail, so p10 and min do not dodge it either.
    What is left is which stage inside the kernel runs longer in the slow tier.
    Capture repeatedly until one lands in each tier, then diff the two tables.
    """
    prof = torch_npu.profiler
    exp = prof._ExperimentalConfig(
        profiler_level=prof.ProfilerLevel.Level1,
        aic_metrics=prof.AiCMetrics.PipeUtilization,
    )
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    e2e_us = []
    with prof.profile(
        activities=[prof.ProfilerActivity.NPU],
        experimental_config=exp,
        on_trace_ready=prof.tensorboard_trace_handler(out_dir),
    ) as session:
        for _ in range(max(1, iters)):
            if prepare is not None:
                prepare()
                torch.npu.synchronize()
            # Time the very calls being profiled, so the kernel duration and
            # the end-to-end number describe the same call rather than two
            # different stretches of the run.
            start.record()
            run_once()
            end.record()
            torch.npu.synchronize()
            e2e_us.append(start.elapsed_time(end) * 1000.0)
            session.step()
    torch.npu.synchronize()
    return _summarise_profile(
        out_dir, e2e_us, quiet=_PROFILE.get("quiet", False), prefer=_PROFILE.get("prefer")
    )


def _launch_rate_check(run_once, prepare, iters: int, rounds: int) -> None:
    """Does the host overhead cost anything once calls are pipelined?

    Every end-to-end number we have serialises host and device: the operator
    rewrites five of its inputs, so each call needs a reset, and the reset needs
    a synchronize. That makes the ~160us of dispatch + GetWorkspaceSize + sync
    fully visible. Production does not work that way -- the host enqueues and
    moves on, and host time only costs anything when the device runs out of work
    to do. With host ~160us against a ~170us kernel the two are close enough
    that it could go either way, so measure it rather than argue about it.

    Three numbers from the same process, then compare against kernel_p50_us:

        serial     reset + sync around every call -- what we have been measuring
        pipelined  no reset, no per-call sync, one synchronize at the end
        kernel     from the --kernel-time capture

    pipelined ~= kernel   -> the host is hidden, that 160us is free in
                            production and not worth optimising
    pipelined ~= serial   -> the host is the bottleneck, the 160us is real, and
                            the GetWorkspaceSize part of it is our own tiling

    Dropping the reset means calls after the first see whatever state the
    operator left behind, so the work may not be identical. That is why the
    kernel duration is printed alongside: if it has not moved, the comparison
    holds. Results are garbage either way -- this measures timing, not output.
    """
    import time

    serial, pipelined = [], []
    for _ in range(max(1, rounds)):
        torch.npu.synchronize()
        began = time.perf_counter()
        for _ in range(iters):
            if prepare is not None:
                prepare()
                torch.npu.synchronize()
            run_once()
            torch.npu.synchronize()
        serial.append((time.perf_counter() - began) * 1e6 / iters)

        if prepare is not None:
            prepare()
        torch.npu.synchronize()
        began = time.perf_counter()
        for _ in range(iters):
            run_once()
        torch.npu.synchronize()
        pipelined.append((time.perf_counter() - began) * 1e6 / iters)

    s_mid = sorted(serial)[len(serial) // 2]
    p_mid = sorted(pipelined)[len(pipelined) // 2]
    print(
        "LAUNCHRATE rounds={} iters={} serial_us={:.1f} pipelined_us={:.1f} "
        "hidden_us={:.1f} ({:.0f}% of serial)  serial=[{}] pipelined=[{}]".format(
            len(serial),
            iters,
            s_mid,
            p_mid,
            s_mid - p_mid,
            100.0 * (s_mid - p_mid) / s_mid if s_mid > 0 else 0.0,
            " ".join(f"{v:.1f}" for v in serial),
            " ".join(f"{v:.1f}" for v in pipelined),
        ),
        flush=True,
    )


def _batch_timing_check(run_once, prepare, iters: int, rounds: int) -> None:
    """Time whole batches on the host clock, with no per-call events at all.

    A pair of NPU events costs 60.9us per call, 14% of the end-to-end time, so
    every e2e number we have is inflated by our own instrument -- and whatever
    jitter the instrument has is jitter we then went looking for. This measures
    what the operator costs when nothing watches each individual call: run a
    batch, divide by the count, repeat. Closer to production too, which does
    not time calls one by one.

    Compare the spread across rounds against the e2e spread from the evented
    path. If it collapses, most of what we chased was the measurement.
    """
    import time

    per_round = []
    for _ in range(max(1, rounds)):
        torch.npu.synchronize()
        began = time.perf_counter()
        for _ in range(iters):
            if prepare is not None:
                prepare()
                torch.npu.synchronize()
            run_once()
        torch.npu.synchronize()
        per_round.append((time.perf_counter() - began) * 1e6 / iters)

    ordered = sorted(per_round)
    mid = ordered[len(ordered) // 2]
    print(
        "BATCHTIME rounds={} iters={} per_call_us=[{}] p50={:.1f} min={:.1f} max={:.1f} range={:.1f}%".format(
            len(per_round),
            iters,
            " ".join(f"{v:.1f}" for v in per_round),
            mid,
            ordered[0],
            ordered[-1],
            100.0 * (ordered[-1] - ordered[0]) / mid if mid > 0 else 0.0,
        ),
        flush=True,
    )


def _event_overhead_check(run_once, prepare, iters: int) -> None:
    """Measure what the per-call NPU event timing costs us.

    With GC off the time outside the kernel settles at 150us but still spikes
    to 484us, and api_statistic puts aclrtRecordEvent at 7us average with a max
    of 197.58us. That call is not the operator's -- it is ours, inserted by the
    measurement. So time the same batch two ways, identical except for the
    event calls: if the evented pass is slower, the difference is what our
    instrument costs, and any spike that lives in it is an artefact rather than
    something the operator does.
    """
    import time

    def one_pass(with_events):
        start = torch.npu.Event(enable_timing=True) if with_events else None
        end = torch.npu.Event(enable_timing=True) if with_events else None
        per_call = []
        torch.npu.synchronize()
        began = time.perf_counter()
        for _ in range(iters):
            if prepare is not None:
                prepare()
                torch.npu.synchronize()
            if with_events:
                start.record()
            run_once()
            if with_events:
                end.record()
            torch.npu.synchronize()
            if with_events:
                per_call.append(start.elapsed_time(end) * 1000.0)
        torch.npu.synchronize()
        return (time.perf_counter() - began) * 1e6 / iters, per_call

    one_pass(False)  # warm
    plain_us, _ = one_pass(False)
    evt_us, per_call = one_pass(True)
    plain2_us, _ = one_pass(False)  # bracket it, in case the machine drifted
    print(
        "EVENTCOST iters={} plain={:.1f}us plain_again={:.1f}us evented={:.1f}us overhead={:+.1f}us ({:+.1f}%)".format(
            iters,
            plain_us,
            plain2_us,
            evt_us,
            evt_us - (plain_us + plain2_us) / 2,
            100.0 * (evt_us - (plain_us + plain2_us) / 2) / ((plain_us + plain2_us) / 2),
        ),
        flush=True,
    )
    if per_call:
        ordered = sorted(per_call)
        n = len(ordered)
        print(
            "EVENTCOST per_call_kernel_us n={} p50={:.1f} p90={:.1f} max={:.1f}".format(
                n, ordered[n // 2], ordered[min(n - 1, int(n * 0.9))], ordered[-1]
            ),
            flush=True,
        )


def _bench_loop(run_once, prepare, warmup: int, iters: int) -> dict:
    """Time ``run_once``; ``prepare`` runs before each timed iter (not included).

    Returns the distribution, not just the mean. These calls are sub-millisecond
    and repeat measurements of the same configuration have been seen to differ
    by up to 10%, so a mean on its own cannot separate a real few-percent change
    from noise -- and this benchmark is what every optimisation gets judged
    against. p50 is the number to compare; std tells you whether comparing is
    meaningful at all.
    """
    for _ in range(max(0, warmup)):
        if prepare is not None:
            prepare()
            torch.npu.synchronize()
        run_once()
    torch.npu.synchronize()

    if _PROFILE.get("launch_rate"):
        rounds = _PROFILE["launch_rate"]
        _PROFILE["launch_rate"] = 0
        _launch_rate_check(run_once, prepare, iters, rounds)

    if _PROFILE.get("batch_timing"):
        rounds = _PROFILE["batch_timing"]
        _PROFILE["batch_timing"] = 0
        _batch_timing_check(run_once, prepare, iters, rounds)

    if _PROFILE.get("event_cost"):
        n = _PROFILE["event_cost"]
        _PROFILE["event_cost"] = 0
        _event_overhead_check(run_once, prepare, n)

    if _PROFILE["dir"]:
        out_dir = _PROFILE["dir"]
        _PROFILE["dir"] = None  # capture once, then measure normally
        _PROFILE["last"] = _capture_profile(run_once, prepare, _PROFILE["iters"], out_dir)

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        if prepare is not None:
            prepare()
            torch.npu.synchronize()
        start.record()
        run_once()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    count = len(samples)
    mean = sum(samples) / count
    variance = sum((s - mean) ** 2 for s in samples) / count
    return {
        "avg_ms": mean,
        "min_ms": samples[0],
        # The tiers only ever run slow: interference adds time, it never
        # removes any. So the fastest iterations are the undisturbed ones, and
        # a low quantile should be a property of the hardware rather than of
        # what else was happening. p50 tracks the disturbance; p10 should not.
        "p10_ms": samples[min(count - 1, int(count * 0.1))],
        "p50_ms": samples[count // 2],
        "p90_ms": samples[min(count - 1, int(count * 0.9))],
        "max_ms": samples[-1],
        "std_pct": 100.0 * variance**0.5 / mean if mean > 0 else 0.0,
    }


def _bench_fused_eager(inp, warmup: int, iters: int, hit_ratio: float) -> dict:
    _prime_selection_cache(inp)
    n_hit = _apply_hit_ratio(inp, hit_ratio)
    print(
        "HIT_RATIO target={:.4f} n_hit={}/{}".format(hit_ratio, n_hit, inp["topk_fused"].shape[-1]),
        flush=True,
    )

    def prepare():
        _apply_hit_ratio(inp, hit_ratio)

    def run_once():
        _ = _call_fused(inp)

    return _bench_loop(run_once, prepare, warmup=warmup, iters=iters)


def _bench_fused_graph(inp, warmup: int, iters: int, hit_ratio: float) -> dict:
    """Capture fused op into ACL graph and time ``graph.replay()``."""
    if not hasattr(torch.npu, "NPUGraph"):
        raise RuntimeError("torch.npu.NPUGraph is unavailable on this torch_npu build")

    _prime_selection_cache(inp)
    n_hit = _apply_hit_ratio(inp, hit_ratio)
    print(
        "HIT_RATIO target={:.4f} n_hit={}/{}".format(hit_ratio, n_hit, inp["topk_fused"].shape[-1]),
        flush=True,
    )

    # Eager warmup at target hit ratio (compile / settle).
    for _ in range(max(1, warmup)):
        _apply_hit_ratio(inp, hit_ratio)
        torch.npu.synchronize()
        _ = _call_fused(inp)
    torch.npu.synchronize()

    _apply_hit_ratio(inp, hit_ratio)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    print("GRAPH_CAPTURE begin", flush=True)
    try:
        with torch.npu.graph(
            graph,
            capture_error_mode="thread_local",
            auto_dispatch_capture=True,
        ):
            out = _call_fused(inp)
    except TypeError:
        # Older torch_npu may not accept auto_dispatch_capture.
        with torch.npu.graph(graph, capture_error_mode="thread_local"):
            out = _call_fused(inp)
    torch.npu.synchronize()
    print(
        "GRAPH_CAPTURE done out_shape={}".format(tuple(out.shape)),
        flush=True,
    )

    def prepare():
        # Replay reads same addresses; rewrite status so each replay keeps target hit ratio.
        _apply_hit_ratio(inp, hit_ratio)

    def run_once():
        graph.replay()

    return _bench_loop(run_once, prepare, warmup=warmup, iters=iters)


def run_case(
    topk: int,
    q_heads: int,
    host_kv: dict,
    warmup: int,
    iters: int,
    cpu_kv_backend: CpuKvBackend = "offload",
    use_graph: bool = False,
    hit_ratios: tuple = (1.0,),
    repeat: int = 1,
    batch_size: int = 1,
    hit_layouts: tuple = ("prefix",),
    plan_slot_layouts: tuple = ("off",),
    topk_buffer_size: int = 0,
    input_sets: int = 1,
    reference_load: bool = False,
    kernel_time: int = 0,
    kernel_time_dir: str = "",
) -> list:
    # Several independent sets exist to test one thing: whether the tier a
    # process seems to draw is really a property of where its tensors landed
    # in device memory. Same process means same cores, same clocks, same pool,
    # same library -- the only difference between sets is the allocation.
    inps = [
        _build_inputs(topk, q_heads, host_kv=host_kv, bsz=batch_size, topk_buffer_size=topk_buffer_size)
        for _ in range(max(1, input_sets))
    ]
    inp = inps[0]
    reference = _make_reference_load() if reference_load else None
    mode = "graph" if use_graph else "eager"
    print(
        "CASE topk={} q_heads={} kv_heads={} bs={} max_seq_len={} num_blocks={} "
        "cpu_kv_backend={} mode={} hit_layouts={} plan_slots={} hit_ratios={} "
        "q_shape={} topk_shape={}".format(
            topk,
            q_heads,
            inp["kv_heads"],
            batch_size,
            inp["msl"],
            inp["num_blocks"],
            cpu_kv_backend,
            mode,
            ",".join(hit_layouts),
            ",".join(plan_slot_layouts),
            ",".join(f"{h:g}" for h in hit_ratios),
            tuple(inp["q_fused"].shape),
            tuple(inp["topk_fused"].shape),
        ),
        flush=True,
    )

    bench = _bench_fused_graph if use_graph else _bench_fused_eager

    # Warm the process once and throw the result away. This started as a fix
    # for a first measurement that came in at std ~36% against ~1.5% for
    # everything after it -- but that was seen exactly once and did not
    # reproduce: warmup 50 vs 600, GC on vs off, three rounds each, and all
    # twelve first measurements were normal. Kept because it is cheap insurance
    # against a bad sample, not because the first one is expected to be bad.
    #
    # Doing this once here, rather than per hit_ratio, is the whole point of
    # sweeping inside one process: every point then shares one warm process
    # instead of paying its own cold start. Measuring each hit_ratio in a
    # separate process adds 8-10% of cross-process noise to every point, which
    # buries any structure smaller than that -- the first sweep done that way
    # showed 2-9% wiggles that turned out to be entirely noise.
    inp["hit_layout"] = hit_layouts[0]
    inp["plan_slots"] = plan_slot_layouts[0]
    if _CHECK_PLAN["on"]:
        ok = check_plan_all_miss_matches_fallback(inp)
        print("PLAN_CHECK result: " + ("PASS" if ok else "FAIL"), flush=True)
        # ⚠ 这里必须退进程，不能 return —— 调用方要对返回值解包，返回 None 会
        #   在校验通过之后抛 TypeError，把一次 PASS 显示成失败。
        raise SystemExit(0 if ok else 1)
    bench(inp, warmup=warmup, iters=iters, hit_ratio=hit_ratios[0])
    print("WARMUP done, first measurement discarded", flush=True)

    outcome = []
    anchor = None

    # Interleave everything -- layout, hit ratio and input set all rotate
    # within each repeat instead of one being finished before the next starts.
    # The operator's timing jumps between discrete tiers on a timescale of tens
    # of seconds, so sweeping one layout's whole curve before starting the next
    # puts a tier change squarely between the two things being compared. The
    # first layout sweep done that way failed its own sanity check: the three
    # layouts must agree exactly at hit_ratio 0 and 1, where no layout applies
    # because every slot hits or none do, and they differed by 6%.
    # plan 也是一个轴。上游 15 参数版两条路都跑得起来（off = 按名次判据，
    # contiguous/scattered = 外部计划），而它们的差值是要写进报告的结论，
    # 所以必须在同一个进程里交错量 —— 分两趟跑，两条曲线之间会横着 2~4% 的
    # 跨进程噪声，正好是这个差值本身的量级。14 参数版上非 off 的档自动空转。
    combos = [
        (hit_layout, hit_ratio, plan_slot, set_idx)
        for hit_layout in hit_layouts
        for hit_ratio in hit_ratios
        for plan_slot in plan_slot_layouts
        for set_idx in range(len(inps))
    ]
    samples = {combo: [] for combo in combos}

    if kernel_time > 0:
        # The measurement that actually means something. End-to-end time is
        # half framework: PyTorch dispatch, three GetWorkspaceSize calls and a
        # stream sync come to ~160us against the kernel's ~168us, and all of
        # the instability lives there -- paired per call over 300 calls, the
        # kernel spans 3-4% while everything outside it spans 210-478%, with a
        # tail reaching five times its own median. None of that moves when we
        # change the kernel, so measuring it only adds noise to our own signal.
        #
        # The cross-process reproducibility of this number is currently unknown:
        # the figures we had (2.1-2.4%) were taken while _summarise_profile
        # could read a stale capture, so they describe the leftover files as
        # much as the kernel. Re-measure before quoting one.
        if not kernel_time_dir:
            kernel_time_dir = _scratch_dir("dsa_kt_")

        # Which kernel these captures are supposed to be about. Learned from the
        # first capture the operator clearly dominated, then used to reject and
        # redo captures that came back describing something else.
        kname_expected = None
        for idx, combo in enumerate(combos):
            hit_layout, hit_ratio, plan_slot, set_idx = combo
            s_inp = inps[set_idx]
            s_inp["hit_layout"] = hit_layout
            s_inp["plan_slots"] = plan_slot
            # One directory per capture. combos covers layout, hit_ratio, plan
            # and set only -- topk and q_heads are swept one level up, so idx
            # restarts at 0 for each of them and a bare "kt{idx}" collides.
            # A colliding directory does not fail; it silently reports the
            # earlier capture's numbers, which is how every topk came to report
            # the first topk's kernel time.
            # Two things used to let a capture report the wrong kernel, and both
            # are now closed. Observed once as 16 of 168 captures in a 56-point
            # sweep coming back at 2.8-8.4us against the operator's ~110, spread
            # across two of three processes and clustered at neither end of
            # either sweep -- so not disk filling up, and the underlying cause is
            # still open. What is fixed is that it no longer changes the number.
            #
            # (1) The headline used to be "whichever kernel had the largest total
            #     duration", a guess that goes wrong the moment the operator is
            #     under-represented in the trace: the harness's own setup kernels
            #     (aclnnContiguous_SliceAiCore_Slice from applying the hit
            #     pattern, lnnInplaceZero from the reset) take the headline.
            #     _PROFILE["prefer"] turns that guess into a lookup by name, so a
            #     capture holding both reports the operator either way.
            # (2) When the operator is missing from the trace outright, or the
            #     trace is short of the calls that were asked for, no lookup can
            #     help -- so redo the capture, up to three times, and say so.
            info = {}
            for attempt in range(3):
                tag = "kt{}_bs{}_tk{}_qh{}_a{}".format(idx, batch_size, topk, q_heads, attempt)
                out_dir = os.path.join(kernel_time_dir, tag)
                shutil.rmtree(out_dir, ignore_errors=True)
                _PROFILE["dir"] = out_dir
                _PROFILE["iters"] = kernel_time
                _PROFILE["quiet"] = True
                _PROFILE["last"] = None
                _PROFILE["prefer"] = kname_expected
                bench(s_inp, warmup=warmup, iters=1, hit_ratio=hit_ratio)
                info = _PROFILE.get("last") or {}
                got = str(info.get("kernel", "")).strip()
                # A trace that holds far fewer calls than were asked for is a
                # truncated one, and its p50 describes whatever happened to land
                # in the part that got written. This check needs no reference
                # name, so it also guards the very first capture of the sweep.
                seen = int(info.get("kernel_n", 0))
                short = seen < 0.9 * kernel_time
                if not short:
                    if kname_expected is not None:
                        if got == kname_expected:
                            break
                    elif got and float(info.get("kernel_share_pct", 0.0)) >= 60.0:
                        # Nothing to compare against on the first point, so adopt
                        # this name -- but only from a capture the operator clearly
                        # dominated. Adopting a truncated one would make every good
                        # point after it get retried three times and rejected.
                        kname_expected = got
                        break
                why = "只采到 {}/{} 次调用".format(seen, kernel_time) if short else "采到的是 {}".format(got or "?")
                print(
                    "PROFILE WARNING hit_ratio={:.4f} {}（{:.1f} μs，占 {:.0f}%），"
                    "要的是 {} —— 第 {} 次重采".format(
                        hit_ratio,
                        why,
                        info.get("kernel_p50_us", 0.0),
                        info.get("kernel_share_pct", 0.0),
                        kname_expected or "本算子",
                        attempt + 1,
                    ),
                    flush=True,
                )
            else:
                print(
                    "PROFILE WARNING hit_ratio={:.4f} 重采三次都没采到 {}，这个点的数字不要用".format(
                        hit_ratio, kname_expected or "本算子"
                    ),
                    flush=True,
                )
            ratios = info.get("ratios", {})
            wanted = ("aic_mac_ratio", "aic_scalar_ratio", "aiv_vec_ratio", "aiv_scalar_ratio")
            shown = " ".join(f"{k.replace('_ratio', '')}={ratios[k]:.3f}" for k in wanted if k in ratios)
            # Whatever else the capture measured that was actually busy. These four
            # were chosen when the picture was "everything is idle except scalar",
            # and they cannot answer the question they now raise: aiv_scalar drops
            # from 0.675 all-hit to 0.236 all-miss while the total time goes up, so
            # the vector cores issue 2.6x fewer scalar instructions on the point
            # that costs more. The candidate explanation is that maintenance turns
            # from scalar-bound into transfer-bound as misses pile up, which an MTE
            # ratio would show -- but the column names differ between CANN
            # versions, so pick by value rather than by guessing names.
            extra = sorted(
                ((v, k) for k, v in ratios.items() if k not in wanted and v >= 0.05),
                reverse=True,
            )[:4]
            if extra:
                shown += " " + " ".join(f"{k.replace('_ratio', '')}={v:.3f}" for v, k in extra)
            # kname/share go on every line, not just the KERNELMIX one below.
            # kernel_p50_us describes whichever kernel had the largest total
            # duration in the capture, and nothing guarantees that is the same
            # kernel from one point to the next. When it is not, the column
            # silently changes meaning -- so carry the name next to the number
            # it belongs to, and let the sweep tool compare names across points.
            kname = str(info.get("kernel", "?")).strip().replace(" ", "_")[-40:] or "?"
            print(
                "KERNEL hit_ratio={:.4f} layout={} plan={} set={} bs={} topk={} slots={} q_heads={} "
                "kernel_p50_us={:.1f} n={} kernels={} kname={} share={:.0f}% "
                "p10_90={:.1f}% range={:.1f}% {}".format(
                    hit_ratio,
                    hit_layout,
                    plan_slot,
                    set_idx,
                    batch_size,
                    topk,
                    inp["slots"],
                    q_heads,
                    info.get("kernel_p50_us", float("nan")),
                    info.get("kernel_n", 0),
                    info.get("kernel_count", 0),
                    kname,
                    info.get("kernel_share_pct", float("nan")),
                    info.get("kernel_p10_90_pct", float("nan")),
                    info.get("kernel_range_pct", float("nan")),
                    shown,
                ),
                flush=True,
            )
            # A bare "kernels=2" says nothing about whether the operator is
            # split across two device kernels -- in which case the p50 above
            # covers one of them -- or whether the second is the harness's own
            # copy between calls. Print the names once per point so the number
            # above is never ambiguous about what it measured.
            if info.get("kernel_others"):
                print(
                    "KERNELMIX head={}@{:.0f}%  others: {}".format(
                        str(info.get("kernel", ""))[-52:],
                        info.get("kernel_share_pct", 0.0),
                        "  ".join(f"{name[-52:]}(n={n},{pct:.0f}%)" for name, n, pct in info["kernel_others"]),
                    ),
                    flush=True,
                )
        _PROFILE["quiet"] = False
    ref_runs = []

    for run_idx in range(max(1, repeat)):
        if reference is not None:
            ref = _bench_loop(reference, None, warmup, iters)
            ref_runs.append(ref["p50_ms"])
            print(
                "REF run={}/{} p50_ms={:.3f} std={:.1f}%".format(
                    run_idx + 1, max(1, repeat), ref["p50_ms"], ref["std_pct"]
                ),
                flush=True,
            )
        for combo in combos:
            hit_layout, hit_ratio, plan_slot, set_idx = combo
            s_inp = inps[set_idx]
            s_inp["hit_layout"] = hit_layout
            s_inp["plan_slots"] = plan_slot
            stats = bench(s_inp, warmup=warmup, iters=iters, hit_ratio=hit_ratio)
            samples[combo].append(stats)
            print(
                "BENCH mode={} hit_ratio={:.4f} layout={} plan={} set={} bs={} topk={} "
                "q_heads={} max_seq_len={} warmup={} iters={} run={}/{} "
                "avg_ms={:.3f} p50_ms={:.3f} min_ms={:.3f} p90_ms={:.3f} "
                "std={:.1f}%".format(
                    mode,
                    hit_ratio,
                    hit_layout,
                    plan_slot,
                    set_idx,
                    batch_size,
                    topk,
                    q_heads,
                    s_inp["msl"],
                    warmup,
                    iters,
                    run_idx + 1,
                    max(1, repeat),
                    stats["avg_ms"],
                    stats["p50_ms"],
                    stats["min_ms"],
                    stats["p90_ms"],
                    stats["std_pct"],
                ),
                flush=True,
            )

    def _summarise(runs):
        p50s = sorted(s["p50_ms"] for s in runs)
        median_p50 = p50s[len(p50s) // 2]
        return median_p50, (100.0 * (p50s[-1] - p50s[0]) / median_p50 if median_p50 > 0 else 0.0)

    for combo in combos:
        hit_layout, hit_ratio, plan_slot, set_idx = combo
        runs = samples[combo]
        median_p50, spread_pct = _summarise(runs)
        worst_std = max(s["std_pct"] for s in runs)
        result = {
            "p50_ms": median_p50,
            "p10_ms": sorted(s["p10_ms"] for s in runs)[len(runs) // 2],
            "avg_ms": sum(s["avg_ms"] for s in runs) / len(runs),
            "min_ms": min(s["min_ms"] for s in runs),
            "p90_ms": max(s["p90_ms"] for s in runs),
            "std_pct": worst_std,
            "spread_pct": spread_pct,
            "usable_runs": len(runs),
        }
        print(
            "RESULT mode={} hit_ratio={:.4f} layout={} plan={} set={} bs={} topk={} slots={} "
            "q_heads={} p50_ms={:.3f} p10_ms={:.3f} min_ms={:.3f} "
            "spread={:.1f}% worst_std={:.1f}% runs={}{}".format(
                mode,
                hit_ratio,
                hit_layout,
                plan_slot,
                set_idx,
                batch_size,
                topk,
                inp["slots"],
                q_heads,
                median_p50,
                result["p10_ms"],
                result["min_ms"],
                spread_pct,
                worst_std,
                len(runs),
                _result_flag(spread_pct, worst_std),
            ),
            flush=True,
        )
        if anchor is None:
            anchor = (hit_layout, hit_ratio, median_p50, spread_pct, plan_slot)
        outcome.append((hit_layout, hit_ratio, plan_slot, result))

    if len(inps) > 1:
        for hit_layout in hit_layouts:
            for hit_ratio in hit_ratios:
                set_p50s = [_summarise(samples[(hit_layout, hit_ratio, i)])[0] for i in range(len(inps))]
                mid = sorted(set_p50s)[len(set_p50s) // 2]
                print(
                    "SETSPREAD hit_ratio={:.4f} layout={} sets={} p50s={} range={:.1f}%".format(
                        hit_ratio,
                        hit_layout,
                        len(inps),
                        " ".join(f"{v:.3f}" for v in set_p50s),
                        100.0 * (max(set_p50s) - min(set_p50s)) / mid if mid > 0 else 0.0,
                    ),
                    flush=True,
                )

    if len(hit_layouts) > 1:
        # The layouts are now measured in the same rounds, so this is the
        # comparison the sweep exists for. At hit_ratio 0 and 1 it doubles as a
        # sanity check: no layout applies there, so range must be ~0. If it is
        # not, something moved that should not have and the middle points are
        # not trustworthy either.
        for hit_ratio in hit_ratios:
            layout_p50s = [
                sorted(_summarise(samples[(lay, hit_ratio, i)])[0] for i in range(len(inps)))[len(inps) // 2]
                for lay in hit_layouts
            ]
            mid = sorted(layout_p50s)[len(layout_p50s) // 2]
            check = ""
            if hit_ratio in (0.0, 1.0):
                rng = 100.0 * (max(layout_p50s) - min(layout_p50s)) / mid if mid > 0 else 0.0
                check = "  (sanity: must be ~0)" if rng <= 2.0 else "  <-- SANITY CHECK FAILED"
            print(
                "LAYOUTSPREAD hit_ratio={:.4f} layouts={} p50s={} range={:.1f}%{}".format(
                    hit_ratio,
                    ",".join(hit_layouts),
                    " ".join(f"{v:.3f}" for v in layout_p50s),
                    100.0 * (max(layout_p50s) - min(layout_p50s)) / mid if mid > 0 else 0.0,
                    check,
                ),
                flush=True,
            )

    if ref_runs:
        # Per round: the operator (median over everything measured in that
        # round) next to the reference measured in the same round. Read the two
        # columns down -- if they move together it is the machine, if only the
        # operator moves it is the operator.
        ref_mid = sorted(ref_runs)[len(ref_runs) // 2]
        op_per_round = [sorted(samples[c][r]["p50_ms"] for c in combos)[len(combos) // 2] for r in range(len(ref_runs))]
        p10_per_round = [
            sorted(samples[c][r]["p10_ms"] for c in combos)[len(combos) // 2] for r in range(len(ref_runs))
        ]
        op_mid = sorted(op_per_round)[len(op_per_round) // 2]
        p10_mid = sorted(p10_per_round)[len(p10_per_round) // 2]
        print(
            "PAIRED rounds={} op={} ref={} op_range={:.1f}% ref_range={:.1f}%".format(
                len(ref_runs),
                " ".join(f"{v:.3f}" for v in op_per_round),
                " ".join(f"{v:.3f}" for v in ref_runs),
                100.0 * (max(op_per_round) - min(op_per_round)) / op_mid if op_mid > 0 else 0.0,
                100.0 * (max(ref_runs) - min(ref_runs)) / ref_mid if ref_mid > 0 else 0.0,
            ),
            flush=True,
        )
        print(
            "PAIRED10 rounds={} op_p10={} range={:.1f}%".format(
                len(p10_per_round),
                " ".join(f"{v:.3f}" for v in p10_per_round),
                100.0 * (max(p10_per_round) - min(p10_per_round)) / p10_mid if p10_mid > 0 else 0.0,
            ),
            flush=True,
        )

    # Sweeping inside one process makes the points comparable, but only as long
    # as the process itself does not drift while the sweep runs. Re-measure the
    # very first point now that everything else is done: the two readings are
    # minutes apart, so if they still agree, every point in between is on the
    # same footing. If they do not, the sweep is not internally consistent and
    # no amount of --repeat will fix it -- shorten the point list and re-run.
    if anchor is not None and len(outcome) > 1:
        inp["hit_layout"] = anchor[0]
        # ⚠ plan 也要还原。它随 combos 轮换，收尾时留在最后一档上，不还原就是
        #   拿另一条路的耗时去跟锚点比，漂移读数整个作废。
        inp["plan_slots"] = anchor[4]
        recheck = bench(inp, warmup=warmup, iters=iters, hit_ratio=anchor[1])
        drift_pct = 100.0 * (recheck["p50_ms"] - anchor[2]) / anchor[2] if anchor[2] > 0 else 0.0
        # Judge the drift against the anchor's own repeat spread, not a fixed
        # threshold. Some points are inherently noisier than others -- the
        # all-hit point is the fastest one and has been seen to spread 8.8%
        # across three repeats, so a 7% "drift" there says nothing. Anchoring
        # on a steady point (put the lowest hit ratio first) makes this check
        # far more sensitive than anchoring on a jumpy one.
        limit = max(3.0, 2.0 * anchor[3])
        print(
            "DRIFT anchor=(layout={} plan={} hit_ratio={:.4f} spread={:.1f}%) first_p50_ms={:.3f} "
            "last_p50_ms={:.3f} drift={:+.1f}% limit={:.1f}% points={}{}".format(
                anchor[0],
                anchor[4],
                anchor[1],
                anchor[3],
                anchor[2],
                recheck["p50_ms"],
                drift_pct,
                limit,
                len(outcome),
                "" if abs(drift_pct) <= limit else "  <-- SWEEP NOT SELF-CONSISTENT",
            ),
            flush=True,
        )
    return outcome


def parse_args():
    parser = argparse.ArgumentParser(description="Fused sparse attention overlap timing UT")
    parser.add_argument(
        "--topk",
        default="2048",
        help="Comma-separated topk list, e.g. 64,1024,2048",
    )
    parser.add_argument(
        "--q-heads",
        default="16,4",
        help="Comma-separated q head counts, e.g. 16,4",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Full KV sequence length covered by CPU cache (tokens). "
        "Default: max(topk*4, 256). For 1M long context use 1048576.",
    )
    parser.add_argument(
        "--cpu-kv-backend",
        choices=("offload", "swap_memory", "pinned"),
        default="offload",
        help="CPU/host KV init: offload=MemFabric offload.empty (default); "
        "swap_memory=torch_npu.empty_with_swapped_memory; "
        "pinned=plain pinned host memory, no MemFabric (functional fallback)",
    )
    parser.add_argument(
        "--dram-size-gb",
        type=float,
        default=None,
        help="Only for --cpu-kv-backend=offload: MemFabric OffloadConfig.size in GB "
        "(same as kv_offload_decode_config.dram_size_per_dp_GB). "
        "Default: auto from --max-seq-len (1M -> ~2GiB).",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Capture fused op into torch.npu.NPUGraph and time graph.replay() "
        "(ACL graph / aclgraph path). Default is eager launch timing.",
    )
    parser.add_argument(
        "--hit-ratio",
        default="1.0",
        help="Comma-separated reuse ratios in [0, 1], e.g. 0.0,0.5,1.0. All of "
        "them are swept inside ONE process, sharing one set of input tensors "
        "and one warm-up, so the points are comparable with each other. "
        "Measuring each ratio in its own process instead adds 8-10%% of "
        "cross-process noise to every point and buries anything smaller. "
        "Each timed iter resets status to the target ratio; which slots hit is "
        "set by --hit-layout. Default 1.0 (all hit).",
    )
    parser.add_argument(
        "--topk-buffer-size",
        type=int,
        default=0,
        metavar="SLOTS",
        help="一行 selection 缓存有几个槽位，等同生产的 "
        "kv_offload_decode_config.topk_buffer_size。必须是 128 的整数倍且 >= topk"
        "（生产同样要求）。**生产默认 4096，而 topk 通常是 2048** —— 想跟生产同"
        "口径就写 4096。默认 0 = 等于 topk，是 2026-08-12 之前的老行为，只为跟"
        "那之前的历史数据比对时才用。⚠ 算子源码不用改也不用重编：kernel 的行距"
        "取的是 selection_kv_block_table 最后一维，递多宽就按多宽走。"
        "⚠ 量成 = topk 会让命中显得比实际连续（槽位摊在一半的地址空间上，"
        "scattered 下更容易凑出成段命中，而成段与否正是快路的触发条件）。",
    )
    parser.add_argument(
        "--plan-direct",
        choices=("on", "off"),
        default="on",
        help="控制区 [5][6]：打开直接寻址（生产就是这么发的，默认 on）。"
        "off 退回块表寻址 —— 那条路上有一道过紧的界 slot >= topk，"
        "capacity > topk 时**槽位号超过 topk 的命中会被静默判成未命中**。"
        "2026-08-11 那轮上游对照量的就是 off 这条路（当时没意识到），"
        "留这个开关是为了跟那批数据同口径重现。",
    )
    parser.add_argument(
        "--check-plan",
        action="store_true",
        help="只跑一条正确性不变量就退出：全 miss 的计划，输出必须等于完全不给"
        "计划时的输出（两种设定下每一行都得从 CPU 池搬）。上游那套 perf 用例一个"
        "校验都没有，这条是最便宜的抓手。14 参数版上直接判过。",
    )
    parser.add_argument(
        "--plan-slots",
        default="off",
        help="逗号分隔，跟 --hit-ratio 一样在**同一个进程里**轮换，所以「有 LRU」"
        "和「无 LRU」两条曲线之间不横着跨进程噪声。可选值："
        + "/".join(PLAN_SLOT_LAYOUTS)
        + "。往 selection_membership_map 写外部计划，走 LRU 那条路。"
        "只在算子声明了该参数（15 参数版）时有效，14 参数版上一律空转。"
        "off（默认）= 不写，kernel 回退到按名次判据 status[i]==topk[i]。"
        "contiguous = 第 i 名放 i 号槽位，按名次方案的天然布局，连续命中能合成"
        "一条 DataCopy。scattered = 槽位打散，LRU 稳定态的样子（token 的槽位是"
        "它当初落在哪儿，跟名次无关，成不了段）。"
        "⚠ 跟 --hit-layout 是两个不同的轴：--hit-layout 决定哪些**名次**命中，"
        "本参数决定命中的名次映射到哪个**槽位**。跨版本比较时前者要对齐，"
        "后者是各自设计的性质、对齐不了。",
    )
    parser.add_argument(
        "--hit-layout",
        default="prefix",
        help="Comma-separated layouts deciding WHICH slots hit for a given "
        "--hit-ratio; like --hit-ratio, all of them are swept inside ONE "
        "process so they are comparable with each other. Choices: "
        + "/".join(HIT_LAYOUTS)
        + ". 'prefix' (default) hits the first round(ratio*topk) slots, so the "
        "misses are ONE contiguous run -- the easiest layout for the transfer "
        "plan, and what every measurement up to 2026-08-04 used. 'strided' "
        "spreads the hits evenly, maximising the number of separate miss runs "
        "(~topk/2 at ratio 0.5). 'random' draws them under a fixed seed, "
        "closest to a real selection. Same ratio, very different amounts of "
        "fragmentation: sweep the three to find out whether the cost tracks "
        "the miss count or the run count.",
    )
    parser.add_argument(
        "--batch-size",
        default="1",
        help="Decode requests in the batch, one query token each. Default 1. "
        "Each request draws its own random topk selection, so a larger batch "
        "is not the same row repeated. Comma-separated sweeps them inside ONE "
        "process, which is the only way the points are comparable to better "
        "than the 2%% that separates two process launches -- but note each "
        "batch size needs its own input tensors, so it also gets its own "
        "warm-up and its own throw-away first measurement.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Measure each configuration this many times within the same "
        "process. With >=2 the first run is discarded and the median p50 of "
        "the rest is printed as a RESULT line. The first measurement after "
        "process start is reliably an outlier (std ~36%% against ~1.5%% for "
        "every run after it) and warmup does not fix it. Use 3 for numbers "
        "you intend to compare against another build.",
    )
    parser.add_argument(
        "--launch-rate",
        type=int,
        default=0,
        metavar="ROUNDS",
        help="Ask whether the ~160us of host overhead costs anything once calls "
        "are pipelined. Times each round twice over --iters calls: once with a "
        "reset and a synchronize around every call (what every end-to-end "
        "number we have does, because the operator rewrites its inputs), and "
        "once with neither, syncing only at the end. Compare the pipelined "
        "number against kernel_p50_us: equal means the host is hidden and that "
        "160us is free in production; equal to the serial number instead means "
        "the host is the bottleneck and the GetWorkspaceSize part of it is our "
        "own tiling code. Outputs are wrong in the pipelined pass -- it drops "
        "the reset on purpose -- so use it for timing only.",
    )
    parser.add_argument(
        "--batch-timing",
        type=int,
        default=0,
        metavar="ROUNDS",
        help="Time this many batches of --iters calls on the host clock with "
        "no per-call NPU events, and print the per-call average of each round. "
        "The event pair costs 60.9us per call, so this is what the operator "
        "costs unobserved -- and comparing the spread across rounds against "
        "the evented path says how much of the jitter was the instrument.",
    )
    parser.add_argument(
        "--event-cost",
        type=int,
        default=0,
        metavar="N",
        help="Time N calls twice, with and without the per-call NPU event, and "
        "print how much the event timing itself costs. aclrtRecordEvent is our "
        "own call and shows a max of 197.58us against a 7us average, so a spike "
        "living in it would be an artefact of measuring rather than the "
        "operator's behaviour.",
    )
    parser.add_argument(
        "--disable-gc",
        action="store_true",
        help="Turn off Python's cyclic garbage collector for the measurement. "
        "The end-to-end time outside the kernel has a tail reaching 484-888us, "
        "but no single host API accounts for it -- the largest max in "
        "api_statistic is 167us. Something outside CANN's accounting is "
        "stalling, and a GC pause is the usual suspect: it is synchronous, "
        "variable in length, and invisible to the profiler.",
    )
    parser.add_argument(
        "--kernel-time",
        type=int,
        default=0,
        metavar="N",
        help="Measure kernel duration instead of end-to-end time: profile N "
        "calls per point and print a KERNEL line with the kernel's own p50 and "
        "pipe ratios. This is the number to compare. End-to-end time is about "
        "half framework overhead -- dispatch, GetWorkspaceSize, stream sync -- "
        "and all the instability lives there: paired per call, the kernel "
        "spans 3-4%% while everything outside spans 210-478%%. 100-300 is "
        "plenty.",
    )
    parser.add_argument(
        "--kernel-time-dir",
        default="",
        help="Where --kernel-time writes its profiles. Defaults to a private "
        "temp directory per run (/tmp/dsa_kt_XXXXXXXX, honouring TMPDIR) that "
        "is deleted on exit. Name one here to keep the csv files.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=0,
        help="Capture a profile of this many operator calls before the normal "
        "measurement, and print a per-kernel summary. The measurement still "
        "runs afterwards, so the RESULT line says which tier this run landed "
        "in -- capture repeatedly until you have one of each, then diff.",
    )
    parser.add_argument(
        "--profile-dir",
        default="",
        help="Where the profiler writes. Defaults to a private temp directory "
        "per run (/tmp/dsa_prof_XXXXXXXX, honouring TMPDIR) that is deleted on "
        "exit. Name one here to keep the csv files.",
    )
    parser.add_argument(
        "--input-sets",
        type=int,
        default=1,
        help="Build N independent sets of input tensors and measure all of "
        "them, alternating between sets within each repeat. Same process means "
        "same cores, same clocks, same pool, same library -- the only thing "
        "differing between sets is where the tensors landed in device memory. "
        "If the sets disagree by as much as separate processes do, the "
        "per-process tier is really per-allocation. Prints a SETSPREAD line. "
        "Costs ~40 MiB of device memory per set at topk=2048, bs=16.",
    )
    parser.add_argument(
        "--reference-load",
        action="store_true",
        help="Measure a plain cube-only matmul once per repeat, interleaved "
        "with the operator, and print a PAIRED line. Both numbers then come "
        "from the same moments, so a wander that shows up in one and not the "
        "other cannot be blamed on the machine.",
    )
    parser.add_argument(
        "--membership-width",
        choices=("full", "placeholder"),
        default="full",
        help="selection_membership_map row width, ignored on builds whose "
        "schema has no such argument. 'full' (default) is the kernel's real row "
        "stride, 16400 int16. 'placeholder' is the 8-wide tensor the container "
        "baseline passes, kept only for reproducing their runs: the kernel "
        "strides by 16400 either way, so 8 writes 16392 int16 per row out of "
        "bounds. That is not a bandwidth footnote -- it took the device down "
        "(SUSPECT REMOTE ERROR 507057) at bs=2 and bs=3 once the miss count "
        "crossed 64, reproducibly, 3 runs out of 3.",
    )
    # Backward-compatible: bare `script.py 64,1024` still works.
    parser.add_argument("legacy_topk", nargs="?", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Which operator this run measured. Printed first and unconditionally: two
    # builds of this repo disagree by one positional argument, and comparing
    # numbers across them without noticing has already cost a round of analysis.
    print("OP_SCHEMA {}".format(_OP_SCHEMA), flush=True)
    # 一眼可查的闸：15 参数版是 12 个张量参数 + membership=True，14 参数版是 11 + False。
    # 原来要另开一条 python 命令去数，现在直接印在日志里。
    print(
        "OP_TENSORS n={} membership={}".format(
            sum(1 for a in _OP_SCHEMA[_OP_SCHEMA.index("(") + 1:_OP_SCHEMA.rindex(") ->")].split(",")
                if a.strip().startswith("Tensor")),
            _OP_TAKES_MEMBERSHIP,
        ),
        flush=True,
    )

    global _MEMBERSHIP_INT16_COUNT
    _MEMBERSHIP_INT16_COUNT = (
        _MEMBERSHIP_STORAGE_INT16_COUNT if args.membership_width == "full" else _MEMBERSHIP_PLACEHOLDER_INT16_COUNT
    )
    print(
        "MEMBERSHIP_WIDTH mode={} int16_per_row={} used_by_op={}".format(
            args.membership_width,
            _MEMBERSHIP_INT16_COUNT,
            _OP_TAKES_MEMBERSHIP,
        ),
        flush=True,
    )
    if args.membership_width == "placeholder" and _OP_TAKES_MEMBERSHIP:
        # 算子的 Init() 把行宽无条件抬到 16400 个 int16，不看传进来的张量多宽 ——
        # 传 8 宽就写到张量外面去了。上游自带的 perf 用例正是这么传的，它侥幸没事
        # 只因为从不读回自己的输出；我们的精度用例读回输出，表现为一次「已经返回
        # 成功的调用」给出全 NaN。
        print(
            "⚠ MEMBERSHIP_WIDTH=placeholder 且算子会读这块 —— 它按 16400 int16 写，"
            "8 宽的张量会被写穿，污染别的显存。只在复现上游那种写法时才用。",
            flush=True,
        )

    topk_str = args.legacy_topk if args.legacy_topk else args.topk
    cases = [int(x) for x in topk_str.split(",") if x]
    # 槽位数跟命中率一样是结论的前提，所以单独打一行，别人读日志不用去推。
    # ⚠ 2026-08-12 之前这个脚本恒等于 topk，跟生产差一倍，而日志里没有任何
    #   一行能看出来 —— 整轮上游对照就是这么带着偏差跑完的。
    _PLAN_DIRECT["on"] = args.plan_direct == "on"
    for _tk in cases:
        _slots = args.topk_buffer_size or _tk
        print(
            "SELECTION_SLOTS topk={} slots_per_row={} blocks_per_row={} ratio={:.2f} "
            "addressing={} source={}".format(
                _tk,
                _slots,
                _slots // _BLOCK_SIZE,
                _slots / _tk,
                "direct" if _PLAN_DIRECT["on"] else "block_table",
                "--topk-buffer-size" if args.topk_buffer_size else "默认（= topk，与生产不符）",
            ),
            flush=True,
        )
        if not _PLAN_DIRECT["on"] and _slots > _tk:
            print(
                "⚠ addressing=block_table 且 slots > topk —— kernel 会把槽位号 >= {} 的"
                "命中静默判成未命中（GetSelectionSlotTokenOffset 的 slot >= sparseBlockCount"
                " 那道界），这一趟量的命中率不是你设的那个。".format(_tk),
                flush=True,
            )
    q_heads_list = [int(x) for x in args.q_heads.split(",") if x]
    batch_sizes = [int(x) for x in str(args.batch_size).split(",") if x]
    if not batch_sizes:
        raise ValueError("--batch-size is empty")
    cpu_kv_backend: CpuKvBackend = args.cpu_kv_backend
    max_seq_len = args.max_seq_len
    hit_ratios = tuple(float(x) for x in args.hit_ratio.split(",") if x.strip())
    if not hit_ratios:
        raise ValueError("--hit-ratio is empty")
    for _hr in hit_ratios:
        if not 0.0 <= _hr <= 1.0:
            raise ValueError(f"--hit-ratio values must be in [0, 1], got {_hr}")
    if args.launch_rate > 0:
        _PROFILE["launch_rate"] = args.launch_rate

    if args.batch_timing > 0:
        _PROFILE["batch_timing"] = args.batch_timing
    if args.event_cost > 0:
        _PROFILE["event_cost"] = args.event_cost
    if args.check_plan:
        _CHECK_PLAN["on"] = True
    if args.disable_gc:
        gc.disable()
        print("GC disabled for this run", flush=True)
    if args.profile_iters > 0:
        _PROFILE["dir"] = args.profile_dir or _scratch_dir("dsa_prof_")
        _PROFILE["iters"] = args.profile_iters
    hit_layouts = tuple(x.strip() for x in args.hit_layout.split(",") if x.strip())
    if not hit_layouts:
        raise ValueError("--hit-layout is empty")
    for _hl in hit_layouts:
        if _hl not in HIT_LAYOUTS:
            raise ValueError(f"--hit-layout must be one of {HIT_LAYOUTS}, got {_hl!r}")
    plan_slot_layouts = tuple(x.strip() for x in args.plan_slots.split(",") if x.strip())
    if not plan_slot_layouts:
        raise ValueError("--plan-slots is empty")
    for _ps in plan_slot_layouts:
        if _ps not in PLAN_SLOT_LAYOUTS:
            raise ValueError(f"--plan-slots must be one of {PLAN_SLOT_LAYOUTS}, got {_ps!r}")
    plan_msl = max_seq_len if max_seq_len is not None else max(max(cases) * 4, 256)
    need_bytes = estimate_cpu_kv_bytes(plan_msl)
    dram_size_gb = args.dram_size_gb
    if cpu_kv_backend == "offload":
        if dram_size_gb is None:
            dram_size_gb = recommended_dram_size_gb(plan_msl)
        if dram_size_gb * _GIB < need_bytes:
            raise ValueError(
                f"dram_size_gb={dram_size_gb} GiB is too small for max_seq_len={plan_msl} "
                f"(need >= {need_bytes / _GIB:.3f} GiB; try --dram-size-gb "
                f"{recommended_dram_size_gb(plan_msl)})"
            )
    elif cpu_kv_backend == "swap_memory" and empty_swapped is None:
        raise RuntimeError("cpu-kv-backend=swap_memory but torch_npu.empty_with_swapped_memory is unavailable")

    mode = "graph" if args.use_graph else "eager"
    print(
        f"{cases=} {q_heads_list=} {batch_sizes=} max_seq_len={plan_msl} "
        f"num_blocks={cdiv(plan_msl, _BLOCK_SIZE)} "
        f"warmup={args.warmup} iters={args.iters} mode={mode} "
        f"hit_ratios={','.join(f'{h:g}' for h in hit_ratios)} "
        f"hit_layouts={','.join(hit_layouts)} plan_slots={','.join(plan_slot_layouts)} "
        f"cpu_kv_backend={cpu_kv_backend} dram_size_gb={dram_size_gb} "
        f"cpu_kv_est_gib={need_bytes / _GIB:.3f}",
        flush=True,
    )

    if cpu_kv_backend == "offload":
        init_offload_pool(dram_size_gb=dram_size_gb)

    results = []
    try:
        for topk in cases:
            # One host KV per topk/msl; reuse across q_heads to avoid pool OOM.
            host_kv = _alloc_shared_host_kv(topk, cpu_kv_backend, max_seq_len)
            try:
                for q_heads in q_heads_list:
                    for batch_size in batch_sizes:
                        for layout, ratio, plan_slot, stats in run_case(
                            topk,
                            q_heads,
                            host_kv=host_kv,
                            warmup=args.warmup,
                            iters=args.iters,
                            cpu_kv_backend=cpu_kv_backend,
                            use_graph=args.use_graph,
                            hit_ratios=hit_ratios,
                            repeat=args.repeat,
                            batch_size=batch_size,
                            hit_layouts=hit_layouts,
                            plan_slot_layouts=plan_slot_layouts,
                            topk_buffer_size=args.topk_buffer_size,
                            input_sets=args.input_sets,
                            reference_load=args.reference_load,
                            kernel_time=args.kernel_time,
                            kernel_time_dir=args.kernel_time_dir,
                        ):
                            results.append((topk, q_heads, batch_size, mode, layout, plan_slot, ratio, stats))
            finally:
                del host_kv
                gc.collect()
    finally:
        if cpu_kv_backend == "offload":
            try:
                uninit_offload_pool()
            except Exception as exc:
                print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)

    print("----- TIMING SUMMARY -----", flush=True)
    print("(compare p50; std tells you whether a comparison means anything)", flush=True)
    for topk, q_heads, bsz, bench_mode, layout, plan_slot, ratio, stats in results:
        print(
            "mode={} hit_ratio={:.4f} layout={} plan={} topk={} q_heads={} bs={} "
            "p50_ms={:.3f} avg_ms={:.3f} min_ms={:.3f} p90_ms={:.3f} std={:.1f}%".format(
                bench_mode,
                ratio,
                layout,
                plan_slot,
                topk,
                q_heads,
                bsz,
                stats["p50_ms"],
                stats["avg_ms"],
                stats["min_ms"],
                stats["p90_ms"],
                stats["std_pct"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
