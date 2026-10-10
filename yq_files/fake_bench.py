"""假的 bench：吃同样的参数，秒回，不碰 NPU。

**用来空跑整条流程，把工具自己的 bug 在几秒内抖出来，别拿 NPU 时间去调。**
2026-08-12 花了 8 分钟跑并行验证，结果发现是 sweep_median 里少了一个字段
（`SimpleNamespace` 漏了 `returncode`）—— 那次 8 分钟里真正在测东西的只有几秒。

    SMOKE=1 bash yq_files/sweep_up0723.sh validate     # 整条流程约 5 秒

它按 `--hit-ratio` × `--plan-slots` 的组合打出格式完全一致的 KERNEL / RESULT 行，
所以 sweep_median 的表、cmp_sweeps 的逐点对照、check_contention 的档位分析
全都走得通。数值是按 miss 数编出来的，形状像真的（miss 越多越慢、
scattered 比 off 快），但**没有任何测量意义**。

⚠ 输出里带 `FAKE=1`，日志一眼能看出不是真数据。

两个旋钮，用来验工具的失败路径：
    FAKE_DEV_SPREAD=0.4   每张卡多 0.4%（默认 0.1，模拟芯片偏置）
    FAKE_FAIL=1           让并行那趟系统性 +5%，验判定能不能抓出来
"""

import os
import sys

_BASE = {0: 112.0, 512: 177.0, 1024: 264.0, 1536: 360.0, 2048: 454.0}
_OFF_FACTOR = 1.22  # plan=off 比 scattered 慢这么多，跟实测的量级一致


def _arg(name, default=""):
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _interp(miss):
    ks = sorted(_BASE)
    if miss <= ks[0]:
        return _BASE[ks[0]]
    if miss >= ks[-1]:
        return _BASE[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= miss <= b:
            t = (miss - a) / (b - a)
            return _BASE[a] + t * (_BASE[b] - _BASE[a])
    return _BASE[ks[-1]]


def main():
    topk = int(_arg("--topk", "2048").split(",")[0])
    bss = [int(x) for x in _arg("--batch-size", "1").split(",") if x]
    ratios = [float(x) for x in _arg("--hit-ratio", "1.0").split(",") if x]
    plans = [x for x in _arg("--plan-slots", "off").split(",") if x]
    q_heads = int(_arg("--q-heads", "16").split(",")[0])
    slots = int(_arg("--topk-buffer-size", "0") or topk)
    layout = _arg("--hit-layout", "prefix").split(",")[0]

    dev = int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0") or 0)
    spread = float(os.environ.get("FAKE_DEV_SPREAD", "0.1"))
    fail = os.environ.get("FAKE_FAIL", "") == "1"
    # 并行时每个进程绑不同的卡；串行时全是同一张。用卡号造一点偏置，
    # 好让 check_contention / cmp_sweeps 有东西可比。
    bias = 1.0 + dev * spread / 100.0
    if fail and os.environ.get("FAKE_PARALLEL", "") == "1":
        bias *= 1.05

    print(f"FAKE=1 这是假 bench，数值没有测量意义。topk={topk} slots={slots} dev={dev}", flush=True)
    print(f"OP_SCHEMA (fake) OP_TENSORS n=12 membership=True", flush=True)
    print(f"SELECTION_SLOTS topk={topk} slots_per_row={slots} blocks_per_row={slots // 128} "
          f"ratio={slots / topk:.2f} addressing=direct source=fake", flush=True)

    for bs in bss:
        for r in ratios:
            n_hit = int(round(r * topk))
            miss = topk - n_hit
            for plan in plans:
                us = _interp(miss) * (_OFF_FACTOR if plan == "off" else 1.0) * bias
                us *= bs / 16.0 if bs != 16 else 1.0
                print(
                    f"KERNEL hit_ratio={r:.4f} layout={layout} plan={plan} set=0 bs={bs} topk={topk} "
                    f"slots={slots} q_heads={q_heads} kernel_p50_us={us:.1f} n=200 kernels=1 "
                    f"kname=FusedSparseAttentionOverlap share=97% p10_90=1.0% range=2.0% "
                    f"aic_mac=0.020 aiv_scalar=0.500 FAKE=1",
                    flush=True,
                )
    for bs in bss:
        for r in ratios:
            n_hit = int(round(r * topk))
            miss = topk - n_hit
            for plan in plans:
                us = _interp(miss) * (_OFF_FACTOR if plan == "off" else 1.0) * bias
                print(
                    f"RESULT mode=eager hit_ratio={r:.4f} layout={layout} plan={plan} set=0 bs={bs} "
                    f"topk={topk} slots={slots} q_heads={q_heads} p50_ms={(us + 160) / 1000:.3f} "
                    f"p10_ms={(us + 150) / 1000:.3f} min_ms={(us + 140) / 1000:.3f} "
                    f"spread=1.0% worst_std=1.0% runs=1",
                    flush=True,
                )


if __name__ == "__main__":
    main()
