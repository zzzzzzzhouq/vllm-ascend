#!/usr/bin/env python3
"""Minimal MemFabric offload init probe for A5 VMM diagnosis.

This reproduces the per_test bench's offload initialization WITHOUT building or
loading the fused_sparse_attention_overlap custom op. Use it to iterate on the
A5 `HalMemMap -6` problem in seconds instead of rebuilding per_test.

It mirrors per_test/bench_fuse_sparse_attention_overlap.py:init_offload_pool():
  - MemFabric >= 1.3 fields: reserve_size / alloc_size / scene = Scene.LOCAL
  - DSA_OFFLOAD_GIANT_PAGE=1 or --giant-page: OFFLOAD_FLAG_GIANT_PAGE
    (A5 conn-based fallback; init works but FSAO kernel cannot MTE-read it)

Examples:
  python3 a5_offload_probe.py --gb 1                 # default VMM path
  python3 a5_offload_probe.py --gb 1 --alloc         # also offload.empty()
  python3 a5_offload_probe.py --gb 1 --giant-page    # conn-based fallback
  python3 a5_offload_probe.py --gb 1 --device 0 --alloc
"""

import argparse
import os
import sys
import traceback

GIB = 1024 ** 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gb", type=float, default=1.0, help="host KV pool size in GiB")
    parser.add_argument("--device", type=int, default=int(os.environ.get("DSA_PROBE_DEVICE", "0")))
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank-id", type=int, default=0)
    parser.add_argument("--giant-page", action="store_true",
                        help="set OFFLOAD_FLAG_GIANT_PAGE (A5 conn-based fallback)")
    parser.add_argument("--alloc", action="store_true",
                        help="also call offload.empty() for the full pool")
    args = parser.parse_args()

    print("=" * 72)
    print("a5_offload_probe")
    print("  python        :", sys.version.replace("\n", " "))
    print("  ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    print("  SOC_VERSION   :", os.environ.get("SOC_VERSION"))
    print("  MF_HYBM_ENABLE_4K_PAGE:", os.environ.get("MF_HYBM_ENABLE_4K_PAGE"))
    print("  DSA_OFFLOAD_GIANT_PAGE:", os.environ.get("DSA_OFFLOAD_GIANT_PAGE"))
    print("  requested pool: %.3f GiB (%d bytes)" % (args.gb, int(args.gb * GIB)))

    try:
        import torch
        import torch_npu
        print("  torch         :", torch.__version__)
        print("  torch_npu     :", torch_npu.__version__)
    except Exception:
        traceback.print_exc()
        return 10

    try:
        from memfabric_hybrid import offload
        try:
            from memfabric_hybrid import get_last_err_msg
        except Exception:
            get_last_err_msg = None
        try:
            import importlib.metadata as md
            print("  memfabric     :", md.version("memfabric_hybrid"))
        except Exception:
            print("  memfabric     : <version unknown>")
        print("  offload module:", getattr(offload, "__file__", "<unknown>"))
    except Exception:
        traceback.print_exc()
        return 11

    try:
        torch.npu.set_device(args.device)
        current = torch.npu.current_device()
        print("  npu device    : requested=%d current=%d" % (args.device, current))
    except Exception:
        traceback.print_exc()
        return 12

    cfg = offload.OffloadConfig()
    cfg.device_id = torch.npu.current_device()
    cfg.reserve_size = int(args.gb * GIB)
    cfg.alloc_size = cfg.reserve_size if args.rank_id == 0 else 0
    cfg.world_size = args.world_size
    cfg.rank_id = args.rank_id
    cfg.scene = offload.Scene.LOCAL
    if args.giant_page:
        cfg.flags = offload.OFFLOAD_FLAG_GIANT_PAGE

    print("  config        : device_id=%s reserve_size=%s alloc_size=%s "
          "world_size=%s rank_id=%s scene=LOCAL flags=%s" % (
              cfg.device_id, cfg.reserve_size, cfg.alloc_size,
              cfg.world_size, cfg.rank_id, getattr(cfg, "flags", "<none>")))

    try:
        ret = offload.initialize(cfg)
    except Exception:
        traceback.print_exc()
        print("RESULT: initialize RAISED")
        return 20

    print("  initialize ret:", ret)
    if ret != 0:
        detail = "<get_last_err_msg unavailable>"
        if get_last_err_msg is not None:
            try:
                detail = get_last_err_msg()
            except Exception as exc:
                detail = "<get_last_err_msg raised %r>" % (exc,)
        print("  last err msg  :", detail)
        print("RESULT: initialize FAILED ret=%s" % ret)
        return 21

    print("  initialize    : OK")

    if args.alloc:
        try:
            tensor = offload.empty([int(args.gb * GIB)], dtype=torch.int8, pin_memory=True)
            print("  offload.empty : OK shape=%s nbytes=%s" % (
                tuple(tensor.shape), tensor.numel()))
        except Exception:
            traceback.print_exc()
            print("RESULT: initialize OK but offload.empty FAILED")
            try:
                offload.uninitialize()
            except Exception:
                pass
            return 22

    try:
        offload.uninitialize()
        print("  uninitialize  : OK")
    except Exception:
        traceback.print_exc()

    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
