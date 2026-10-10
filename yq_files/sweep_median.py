"""Sweep a curve in one process, repeat across processes, report the median.

Two things are true at once and they call for different designs:

  - Points on a curve compare best inside one process. One command sweeps them
    all sharing one warm-up and one set of input tensors, so what separates two
    points is the thing being varied and not two process launches.
  - Absolute values only settle down across processes. A single run's
    kernel_p50_us spans 3.7%; the median of eight settles to 1.03%.

So: sweep the whole curve in one process, do that in N processes, and take each
point's median across them. Both numbers come out of the same runs.

    python3 yq_files/sweep_median.py --processes 8 -- \
        --batch-size 16 --max-seq-len 65536 --topk 2048 --q-heads 16 \
        --warmup 50 --iters 100 --repeat 1 --kernel-time 200 --disable-gc \
        --hit-ratio 0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0

The default output is one table: each point's median, how far the N processes
disagreed, and every process's raw value. Conclusions come from those
microseconds. --shape and --e2e add the normalised and end-to-end tables, both
off by default because neither is a criterion for whether a change made anything
faster -- see their help text.

A third table appears only when the same physical point was measured more than
once in a sweep. Ask for that by passing several hit ratios that round to the
same hit count -- 0, 0.00005 and 0.0001 all put zero of 2048 slots in the cache,
so the inputs are bit-identical and only the label differs. Placing those copies
at different positions of the sweep separates two explanations of a wide
cross-process spread that otherwise look the same:

  - a process genuinely offset for its whole life -> copies inside one process
    agree closely, processes disagree
  - a value that wanders in time -> copies inside one process disagree just as
    much as the processes do, and the position in the sweep is what matters
"""

import argparse
import datetime
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import types

# Resolve the bench relative to THIS file, not the cwd. As a bare relative path
# it only worked when the caller happened to stand in the repo root, and on
# 2026-08-08 six processes in a row died with "can't open file
# /mnt/share_space/yq/tests/ut/..." because the caller stood one level up.
_BENCH = str(
    pathlib.Path(__file__).resolve().parent.parent
    / "tests/ut/test_fuse_sparse_attention_overlap_precisions.py"
)
_REPO = pathlib.Path(__file__).resolve().parent.parent


def _build_stamp() -> str:
    """One line saying which code produced the numbers below it.

    Every sweep result gets pasted into a document eventually, and a pasted log
    that does not name its own commit cannot be matched back to a build later.
    That is not hypothetical: on 2026-08-08 a table of ten readings was copied
    into a document from the wrong run, and the only reason it was caught was a
    line-by-line recompute of the medians.

    Three things go on the line, because three different things have silently
    diverged before:
      - the commit, with a tag when HEAD carries one (which source),
      - whether csrc/ or tests/ is dirty (source that is not in any commit),
      - the newest built .so and its mtime (the binary actually loaded can be
        older than the source -- an incremental build does not rebuild a kernel
        when only a header it includes changed; see yq_files/which_build.sh).
    """

    def _git(*a):
        try:
            out = subprocess.run(
                ["git", "-C", str(_REPO), *a], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return out.stdout.strip() if out.returncode == 0 else ""

    head = _git("log", "--format=%h %s", "-1") or "（不是 git 仓库或 git 不可用）"
    tag = _git("tag", "--points-at", "HEAD").replace("\n", ",")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    dirty = [
        line[3:]
        for line in _git("status", "--porcelain").splitlines()
        if line[3:].startswith(("csrc/", "tests/"))
    ]

    # Only the install directory, and only the two files that actually get
    # loaded. Walking the whole tree also picked up csrc/build/compat/, where a
    # dangling symlink made stat() raise and killed a sweep that had just spent
    # 13 minutes on a full rebuild. Anything under build/ is an intermediate
    # anyway -- what the process imports is vllm_ascend/.
    so = ""
    stamped = []
    for c in _REPO.glob("vllm_ascend/*.so"):
        try:
            stamped.append((c.stat().st_mtime, c))
        except OSError:
            continue
    if stamped:
        mtime, newest = max(stamped)
        so = f"{newest.name} @ {datetime.datetime.fromtimestamp(mtime):%m-%d %H:%M}"

    out = [f"构建：{branch}{' [' + tag + ']' if tag else ''} {head}"]
    if dirty:
        out.append(f"  ⚠ csrc/ 或 tests/ 有未提交改动，这趟的数字对不上任何 commit：{' '.join(dirty[:6])}")
    out.append(f"  最新 .so：{so or '（没找到）'}")
    out.append("  ⚠ .so 比源码旧就是在测旧二进制。改过 csrc/ 下的头文件必须全量重编。")
    return "\n".join(out)


_KV = re.compile(r"(\w+)=([-\d.]+|\w+)")
# ⚠ plan 必须在这里。bench 能在一趟里轮换 --plan-slots（有 LRU / 无 LRU），
# 表是按 hit_ratio 索引的，plan 不进 key 的话同一个命中率的两条路会被并成一行，
# 两个不同配置的读数当成同一点的重复采样求中位数 —— 差值直接消失。
_KEY_FIELDS = ("layout", "plan", "bs", "topk", "slots", "q_heads", "set")
# Candidates for dropping from an echoed line: they are usually constant across
# a sweep, and dropping them roughly halves what has to be copied out of the
# container. Only the ones that really are constant get dropped -- an earlier
# version dropped bs unconditionally, which hid exactly the axis a batch sweep
# varies, and made "only one batch size ran" invisible in the echo.
_ECHO_CANDIDATES = ("layout", "plan", "set", "bs", "topk", "slots", "q_heads", "n")


def _echo_trim(lines):
    """Drop the key=value pairs that are identical on every line."""
    seen = {f: {m for line in lines for m in re.findall(rf"\b{f}=(\S+)", line)} for f in _ECHO_CANDIDATES}
    drop = [f for f, vals in seen.items() if len(vals) <= 1]
    if not drop:
        return lines
    pat = re.compile(r"\b(" + "|".join(drop) + r")=\S+\s*")
    return [pat.sub("", line) for line in lines]


def _parse(line):
    """KERNEL/RESULT lines are key=value throughout, so read them generically."""
    return dict(_KV.findall(line))


def _spread_pct(xs):
    mid = statistics.median(xs)
    return 100.0 * (max(xs) - min(xs)) / mid if mid else 0.0


def _collect(stdout, prefix, value_key):
    """[(point_key, hit_ratio, value)] for every matching line."""
    out = []
    for line in stdout.splitlines():
        if not line.startswith(prefix + " "):
            continue
        kv = _parse(line)
        if value_key not in kv or "hit_ratio" not in kv:
            continue
        key = tuple(kv.get(f, "") for f in _KEY_FIELDS)
        out.append((key, float(kv["hit_ratio"]), float(kv[value_key])))
    return out


def _ratio_label(ratio):
    """Enough digits to tell 0.999 from 0.9995 -- two decimals printed both as 1.00."""
    return f"{ratio:.6f}".rstrip("0").rstrip(".")


def _varying(points):
    """Which of _KEY_FIELDS actually differ across the swept points.

    The tables are indexed by hit_ratio, so anything else that was swept has to
    be spelled out in the row label or the rows become indistinguishable -- a
    batch-size sweep at a single hit ratio otherwise prints one label N times.
    """
    return [i for i in range(len(_KEY_FIELDS)) if len({k[i] for k, _ in points}) > 1]


def _key_label(key, varying):
    return " ".join(f"{_KEY_FIELDS[i]}={key[i]}" for i in varying if key[i] != "")


def _misses(key, ratio):
    """How many of the topk slots still have to be fetched."""
    try:
        topk = int(key[_KEY_FIELDS.index("topk")])
    except (ValueError, IndexError):
        return None
    return int(round((1.0 - ratio) * topk))


def _clumps(vals):
    """Split the processes into groups if they came out in separated clusters.

    Returns None when the values form one group, otherwise the list of groups.

    This exists because the median stops meaning anything when they do not.
    One sweep reported 94.9 us for a point whose twelve processes sat at
    88.0-90.6 and 99.5-101.8: no process measured anything within 4 us of the
    number in the table, and averaging across the gap invented a value.

    Cut at any gap that is both wide in absolute terms (>4% of the median, i.e.
    well past the ~1% a settled point spreads over) and wide relative to the
    typical step between neighbouring values.
    """
    v = sorted(vals)
    if len(v) < 6:
        return None
    mid = statistics.median(v)
    gaps = [(v[i + 1] - v[i], i) for i in range(len(v) - 1)]
    typical = statistics.median(g for g, _ in gaps) or 1e-9
    cuts = [i for g, i in gaps if g > 0.04 * mid and g > 4 * typical]
    if not cuts:
        return None
    groups, start = [], 0
    for i in cuts:
        groups.append(v[start : i + 1])
        start = i + 1
    groups.append(v[start:])
    # One process sitting on its own past a gap is an outlier, and the median
    # already survives those. Only call it a split when at least two of the
    # groups have company -- that is the case where the median lands in the gap.
    if sum(1 for g in groups if len(g) >= 2) < 2:
        return None
    return groups


def _shape_table(title, per_process, points, ref_ratio, order, varying):
    """Each point divided by that process's own baseline.

    ⚠⚠ NOT a criterion for "did this change make it faster". The denominator is
    that process's median over all its points, and a sweep's points mostly sit on
    one stream -- 8 of the 11 points of the hit-ratio baseline are at miss >= 176.
    Speed that stream up by 10% and numerator and denominator fall together:
    worked through on real numbers, shape at miss=2048 moves 1.024 -> 1.022 and
    the improvement is invisible. The reason shape looks so good across an A/A
    pair (0.09-0.26% against 1.3-1.8% for the median) is the same mechanism: it
    cancels the run-to-run drift, and anything that cancels drift that thoroughly
    cancels real change too.

    What it is still good for: seeing where a step sits when per-process offsets
    swamp the absolute numbers. Shape of the curve, not size of an improvement.
    """
    ratios_seen = sorted({r for _, r in points})
    ref = None if ref_ratio is None else min(ratios_seen, key=lambda r: abs(r - ref_ratio))
    shapes = []
    for d in per_process:
        for key in {k for k, _ in points}:
            own = {r: v for (k2, r), v in d.items() if k2 == key}
            if not own:
                continue
            base = own.get(ref) if ref is not None else statistics.median(own.values())
            if not base:
                continue
            shapes.append({(key, r): v / base for r, v in own.items()})
    if len(shapes) < 2:
        return

    anchor = f"hit_ratio={ref:g}" if ref is not None else "该进程所有点的中位数"
    print(f"\n=== {title}：形状（每个进程内部相对{anchor}的比值）===", flush=True)
    print(
        f"  {'hit_ratio':>9}  {'miss数':>6}  {'序':>3}  {'比值中位数':>11}  {'进程间极差':>10}   各进程的比值（不截断）"
    )
    print("  " + "-" * 104)
    worst_abs, worst_shape = 0.0, 0.0
    for key, ratio in points:
        vals = [s[(key, ratio)] for s in shapes if (key, ratio) in s]
        raw = [d[(key, ratio)] for d in per_process if (key, ratio) in d]
        # Same fix as in _table: a ratio is taken within one process, so one
        # process is enough to have something to print.
        if not vals:
            continue
        miss = _misses(key, ratio)
        # The per-process ratios, not just their spread: a tier shows up as one
        # process sitting apart on every row, a point-specific effect as one row
        # scattering while its neighbours stay put. Only the raw list tells them
        # apart, and the summary column cannot.
        label = _ratio_label(ratio)
        if varying:
            label = _key_label(key, varying) + "  " + label
        print(
            f"  {label:>9}  {'' if miss is None else miss:>6}  "
            f"{order.get((key, ratio), ''):>3}  "
            f"{statistics.median(vals):>11.4f}  {_spread_pct(vals):>9.2f}%   " + " ".join(f"{v:.3f}" for v in vals)
        )
        worst_shape = max(worst_shape, _spread_pct(vals))
        if len(raw) >= 2:
            worst_abs = max(worst_abs, _spread_pct(raw))
    print(f"\n  最差点：绝对值极差 {worst_abs:.2f}%，形状极差 {worst_shape:.2f}%")
    if worst_shape < worst_abs / 2:
        # This used to read "结论按比值下，别按微秒数下", and that advice was wrong
        # -- see the docstring. Say what the number means instead of what to do.
        print("  ⚠ 形状比绝对值稳，说明这一趟有整体位移被归一化消掉了。")
        print("     别据此改用比值下结论 —— 同一个消法也会把真改进消掉。判据一律用微秒。")


def _table(title, unit, per_process, ref_ratio, order=None, show_shape=False):
    """per_process: list of {(key, hit_ratio): value}, one dict per process.

    order: {(key, hit_ratio): 1-based position in the sweep}. Printed because
    the rows come out sorted by ratio, which hides when each point was taken --
    and when a point was taken turns out to matter as much as which point it is.
    """
    order = order or {}
    points = sorted({p for d in per_process for p in d})
    if not points:
        return

    absurd, split = [], []
    print(f"\n=== {title}：绝对值（{unit}）===", flush=True)
    print(
        f"  {'hit_ratio':>9}  {'miss数':>6}  {'序':>3}  {'中位数':>9}  "
        f"{'进程间极差':>10}   各进程的值（按进程顺序，不截断）"
    )
    print("  " + "-" * 104)
    varying = _varying(points)
    if len(per_process) == 1:
        print("  只有一个进程，「进程间极差」这一列恒为 0.00%，不要读它。", flush=True)
    for key, ratio in points:
        vals = [d[(key, ratio)] for d in per_process if (key, ratio) in d]
        # Only skip a row that nobody measured. This used to require two values
        # and so printed nothing at all under --processes 1, silently throwing
        # away the whole sweep -- which is the one case where every row is a
        # single value by construction. A one-process sweep is the cheap way to
        # find where a curve changes shape, and it is worth six times less than
        # the three-process run it replaces, so it has to be readable.
        if not vals:
            continue
        label = _ratio_label(ratio)
        if varying:
            label = _key_label(key, varying) + "  " + label
        # miss count, not hit percentage, is what the transfer plan actually
        # sees -- and the interesting structure sits where the percentages are
        # indistinguishable (0.999 and 0.9995 differ by one slot out of 2048).
        miss = _misses(key, ratio)
        # Never abbreviate the samples column. The point of the tool is to hand
        # over the numbers; a median whose samples are hidden cannot be checked,
        # and "one process ran slow throughout" only shows up in the raw list.
        med = statistics.median(vals)
        print(
            f"  {label:>9}  {'' if miss is None else miss:>6}  "
            f"{order.get((key, ratio), ''):>3}  "
            f"{med:>9.4g}  {_spread_pct(vals):>9.2f}%   " + " ".join(f"{v:g}" for v in vals)
        )
        # A value under half the point's median is not a slow process, it is a
        # reading of something else -- one sweep reported 3.1 us where the other
        # eleven processes all sat at 89-96. The median absorbed it (it moved by
        # 0.22%), which is exactly why it has to be said out loud: a silently
        # absorbed garbage sample is one that never gets investigated.
        absurd += [(label, order.get((key, ratio), "?"), v, med) for v in vals if v < med / 2]
        groups = _clumps(vals)
        if groups:
            split.append((label, order.get((key, ratio), "?"), med, groups, list(vals)))

    if show_shape:
        _shape_table(title, per_process, points, ref_ratio, order, varying)

    if split:
        print("\n  ⚠⚠ 这些点的进程分成了几团，中间是空的 —— 上表的中位数落在空档里，")
        print("     报的是一个没有任何进程测到的值。这种点不能用一个数概括：")
        for label, seq, med, groups, vals in split:
            print(f"      {label}（序 {seq}）  上表中位数 {med:.4g}  ← 别用")
            for g in groups:
                who = [f"P{i + 1}" for i, v in enumerate(vals) if v in g]
                print(f"          {statistics.median(g):>8.4g}   {len(g):>2} 个进程   {' '.join(who)}")
        print("     先查这几团跟什么对得上（进程序号奇偶、启动顺序、这一趟里还测了哪些点）。")

    if absurd:
        print(f"\n  ⚠ {len(absurd)} 个采样不到该点中位数的一半 —— 这不是「某进程慢」，是读到了别的东西：")
        for label, seq, v, med in absurd:
            print(f"      {label}（序 {seq}）：{v:g} vs 中位数 {med:.4g}")
        print("      中位数把它吸收掉了，但被吸收的坏值就是永远不会有人去查的坏值。")
        print("      先看这一趟的 profiler 采集目录是不是被别的进程占了（见 docs/测试方法/）。")


def _repeat_table(title, unit, per_process, order):
    """Points that are the same measurement wearing different labels.

    Two numbers decide between the competing stories, and they are only both
    available when a point is measured several times per process:

      组内极差 -- one process, one point, different moments of the same sweep
      组间极差 -- different processes, same point

    组内 close to 组间 means the value moves on the timescale of a sweep and
    "this process is slow" is the wrong description; 组内 much smaller means the
    process really is carrying a fixed offset. The first table cannot separate
    them because a single measurement per process is consistent with both.
    """
    groups = {}
    for d in per_process:
        for key, ratio in d:
            miss = _misses(key, ratio)
            if miss is not None:
                groups.setdefault((key, miss), set()).add(ratio)
    # A point measured once per process cannot be trusted: the first measurement
    # of a point runs high (+5.7 to +14.1us at all-miss, across all twelve
    # processes; -5.8 to +14.8 at all-hit) and every process carries the same
    # bias, so the cross-process median keeps it. Say so rather than printing a
    # table that looks as settled as any other.
    singles = [g for g, rs in sorted(groups.items()) if len(rs) == 1]
    if singles:
        print(
            "\n⚠ 这些点在每个进程里只测了一次，第一次测量系统性偏高，绝对值不可用：\n  "
            + "  ".join(f"miss={m}" for _, m in singles[:12])
            + ("  …" if len(singles) > 12 else "")
            + "\n  改法：同一个点给两三个标签（取 round(r×topk) 相同的几个 hit_ratio，"
            "输入逐位相同、\n  只是名字不同），排在一趟里的不同位置，用后面的。",
            flush=True,
        )
    groups = {g: sorted(rs) for g, rs in sorted(groups.items()) if len(rs) > 1}
    if not groups:
        return

    print(f"\n=== {title}：同一物理点在一趟里量了几次（{unit}）===", flush=True)
    varying = _varying([(g[0], 0) for g in groups])
    for (key, miss), ratios in groups.items():
        positions = [order.get((key, r)) for r in ratios]
        shown = "/".join(str(p) for p in positions if p is not None)
        tag = _key_label(key, varying)
        print(f"\n  miss数={miss}{'  ' + tag if tag else ''}   测量顺序 {shown}（共 {len(ratios)} 次）")
        within, mids = [], []
        for i, d in enumerate(per_process, 1):
            vals = [d[(key, r)] for r in ratios if (key, r) in d]
            if len(vals) < 2:
                continue
            within.append(_spread_pct(vals))
            mids.append(statistics.median(vals))
            print(f"    进程 {i:>2}: " + " ".join(f"{v:>7.4g}" for v in vals) + f"   组内 {_spread_pct(vals):>5.2f}%")
        if not within or len(mids) < 2:
            continue
        w, b = statistics.median(within), _spread_pct(mids)
        print(f"    组内极差中位数 {w:.2f}%（最大 {max(within):.2f}%）    组间极差 {b:.2f}%")
        # Half is arbitrary but has to be somewhere; what matters is that the
        # two readings get named instead of left as a pair of percentages.
        if w > b / 2:
            print("    → 同一个进程里就在游走，「某个进程慢」讲不通，看的是时间/位置。")
        else:
            print("    → 进程内部稳、进程之间散，确实是每进程一个固定档位。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument(
        "--ref-ratio",
        type=float,
        default=None,
        help="anchor the shape table on this hit_ratio instead of on each "
        "process's median over all its points. Only worth doing when that point "
        "is known to be quiet -- anchoring on a noisy one pushes its noise into "
        "every other point.",
    )
    ap.add_argument(
        "--shape",
        action="store_true",
        help="also print the normalised tables (each point over that process's "
        "own median). Off by default: it is not a criterion for whether a "
        "change made anything faster -- the denominator sits on the same stream "
        "as most of the points, so a uniform improvement cancels itself out. "
        "Turn it on only to see where a step sits when per-process offsets "
        "swamp the absolute numbers.",
    )
    ap.add_argument(
        "--e2e",
        action="store_true",
        help="also print the end-to-end tables. Off by default: end-to-end is "
        "about half framework overhead, it drifts, and taking a median does not "
        "pin it down, so it is not the criterion for anything. Worth turning on "
        "to cross-check a kernel-side effect against a different timing path.",
    )
    ap.add_argument("--bench", default=_BENCH)
    ap.add_argument(
        "--echo",
        action="store_true",
        help="also print every KERNEL/KERNELMIX line each process produced, "
        "prefixed with the process number. The tables drop the pipeline "
        "utilisation and p10_90 fields, and those are what say whether two "
        "different timings came from doing different work or from the same "
        "work running slower.",
    )
    ap.add_argument(
        "--launcher",
        action="append",
        default=[],
        metavar="CMD",
        help='prefix every process with this command, e.g. "numactl '
        '--cpunodebind=0 --membind=0". Repeat the flag to rotate several '
        "launchers: process i gets launcher (i-1) %% count. Split with shlex, "
        "no shell involved. Use one launcher to ask whether an effect survives "
        "being pinned, several to place processes deliberately -- but note that "
        "rotating an even number of launchers lines up with process parity, so "
        "an effect that follows the launch order and one that follows the "
        "launcher become indistinguishable.",
    )
    ap.add_argument(
        "--devices",
        default="",
        metavar="A,B,C",
        help="并行跑：进程 i 绑第 i 张卡（ASCEND_RT_VISIBLE_DEVICES），同时起。"
        "不给就是串行、全在当前那张卡上（默认，也是所有历史数据的口径）。"
        "⚠⚠ 并行改的不只是快慢，判据本身也变了：跨芯片偏置进了中位数（换芯片"
        "实测同方向偏约 1%%），而且几个进程同时 H2D 会互相争抢 —— 曲线 miss 重"
        "那一端正是搬运受限的，我们自己就成了噪声源。**用之前先做一次串行/并行"
        "对照，两趟的差要落在同趟 A/A 的两倍以内**，且此后所有对比必须用同一组卡。",
    )
    ap.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    bench_args = [a for a in args.bench_args if a != "--"]
    if not bench_args:
        raise SystemExit("需要在 -- 之后给出基准脚本的参数")

    # Never let the stamp kill the sweep. It is a label on the result, not part
    # of the measurement, and a bug in it once threw away a full rebuild.
    try:
        print(_build_stamp(), flush=True)
    except Exception as exc:  # noqa: BLE001 -- a label must not be fatal
        print(f"构建：（构建戳失败，不影响下面的数字）{type(exc).__name__}: {exc}", flush=True)

    kernel_runs, e2e_runs = [], []
    kernel_order, e2e_order = {}, {}
    # kname is which kernel the p50 on that line describes. The bench picks
    # whichever kernel had the largest total duration in the capture, so the
    # column can change meaning between points without anything looking wrong.
    # Collect the names and say so at the end if more than one shows up.
    knames = {}
    last_mix = None
    # --devices 打开并行：进程 i 绑第 i 张卡，同时跑。
    # ⚠⚠ 并行改变的不只是快慢，**判据本身也变了**，用之前先跑一次验证：
    #   ① 跨芯片偏置进了中位数。6 进程取中位数压的是「每进程一个固定档位」，
    #     摊到 6 颗芯片上每个样本还多带一份芯片偏置（实测换芯片四个耗时值
    #     同方向偏约 1%）。此后所有对比都必须用**同一组芯片**。
    #   ② 6 个进程同时 H2D 会互相争抢，而曲线 miss 重那一端正是搬运受限的 ——
    #     我们自己成了噪声源，且偏向性地抬高最关心的那一端。
    # 判据：同一组点，串行一趟、并行一趟，两趟的差要落在同趟 A/A 的两倍以内。
    devices = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else []
    if devices and len(devices) < args.processes:
        raise SystemExit(f"--devices 给了 {len(devices)} 张卡，少于 --processes {args.processes}")

    def _spawn(i):
        prefix = shlex.split(args.launcher[(i - 1) % len(args.launcher)]) if args.launcher else []
        env = None
        if devices:
            env = dict(os.environ)
            env["ASCEND_RT_VISIBLE_DEVICES"] = devices[i - 1]
        return prefix, subprocess.Popen(
            [*prefix, sys.executable, args.bench, *bench_args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )

    if devices:
        print(f"  并行：{args.processes} 个进程分别绑卡 {','.join(devices[:args.processes])}", flush=True)
        spawned = [_spawn(i) for i in range(1, args.processes + 1)]
        # ⚠ returncode 必须一起带上。下面有一处判它非 0 就打 stderr ——
        #   少带这个字段，第一个进程收完就 AttributeError 崩掉，而崩之前已经
        #   打印了「进程 1/8：KERNEL 20 点」，看着像跑起来了。2026-08-12 栽过。
        finished = [(p, pr.communicate(), pr.returncode) for p, pr in spawned]
    else:
        finished = None

    for i in range(1, args.processes + 1):
        if finished is not None:
            prefix, (out, err), code = finished[i - 1]
            proc = types.SimpleNamespace(stdout=out, stderr=err, returncode=code)
        else:
            prefix = shlex.split(args.launcher[(i - 1) % len(args.launcher)]) if args.launcher else []
            proc = subprocess.run([*prefix, sys.executable, args.bench, *bench_args],
                                  capture_output=True, text=True)
        k = _collect(proc.stdout, "KERNEL", "kernel_p50_us")
        e = _collect(proc.stdout, "RESULT", "p50_ms")
        for line in proc.stdout.splitlines():
            if line.startswith("KERNEL "):
                kv = _parse(line)
                if "kname" in kv:
                    knames.setdefault(kv["kname"], []).append(f"P{i}")
        # Position in the sweep, taken from the first process that produced any
        # points. Every process runs the same combo list in the same order.
        if k and not kernel_order:
            kernel_order = {(key, r): n for n, (key, r, _) in enumerate(k, 1)}
        if e and not e2e_order:
            e2e_order = {(key, r): n for n, (key, r, _) in enumerate(e, 1)}
        if k:
            kernel_runs.append({(key, r): v for key, r, v in k})
        if e:
            e2e_runs.append({(key, r): v for key, r, v in e})
        where = f"（{' '.join(prefix)}）" if prefix else ""
        print(f"  进程 {i}/{args.processes}{where}：KERNEL {len(k)} 点，RESULT {len(e)} 点", flush=True)
        if args.echo:
            kern = [ln for ln in proc.stdout.splitlines() if ln.startswith("KERNEL ")]
            trimmed = dict(zip(kern, _echo_trim(kern)))
            for line in proc.stdout.splitlines():
                if line.startswith("KERNEL "):
                    print(f"    P{i:<2} " + trimmed[line], flush=True)
                elif line.startswith("KERNELMIX ") and line != last_mix:
                    # Identical on every point of every process once the operator
                    # is a single kernel, so printing all of them buries the one
                    # case worth seeing: a point where the mix actually changes.
                    last_mix = line
                    print(f"    P{i:<2} {line}", flush=True)
        # 退出码非 0 就要吭声，哪怕已经收到一些点。基准脚本一次扫多个配置，
        # 中途某个配置崩掉的话前面的点照样进了 stdout —— 只在"一个点都没有"时
        # 才打印 stderr，等于把这种半途夭折的运行当成正常结果。
        if proc.returncode != 0:
            print(
                f"  ⚠ 进程 {i} 退出码 {proc.returncode}，只拿到 {len(k)} 个点。下面是它的 stderr 末尾：",
                flush=True,
            )
            sys.stderr.write((proc.stderr or "(空)")[-1500:] + "\n")
        if not k:
            sys.stderr.write(proc.stdout[-1200:] + proc.stderr[-800:])

    _table("kernel duration", "μs", kernel_runs, args.ref_ratio, kernel_order, args.shape)
    _repeat_table("kernel duration", "μs", kernel_runs, kernel_order)

    if len(knames) > 1:
        print("\n  ⚠⚠ 这一趟的 kernel_p50_us 不是同一个 kernel 的时间，上面整张表不可比：")
        for name, where in sorted(knames.items(), key=lambda kv: -len(kv[1])):
            seen = sorted(set(where), key=lambda p: int(p[1:]))
            print(f"      {name}   {len(where)} 行，出现在 {' '.join(seen)}")
        print("      bench 取的是「这次采集里总时长最大的那个 kernel」，名字换了这一列就换了含义。")
    elif knames:
        print(f"\n  kernel_p50_us 全程是同一个 kernel：{next(iter(knames))}")

    if args.e2e:
        _table("端到端", "ms", e2e_runs, args.ref_ratio, e2e_order, args.shape)
        print(
            "\n注意：端到端那两张表只作参考。同一批 48 次实测，端到端的进程间离散"
            "8% 打底且取中位数压不动（它在漂），判据以 kernel 为准。",
            flush=True,
        )


if __name__ == "__main__":
    main()
