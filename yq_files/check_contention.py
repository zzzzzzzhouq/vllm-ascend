"""查一趟 sweep 里有没有「某几个进程系统性偏高」。

    python3 yq_files/check_contention.py 本趟.log [干净的那趟.log]

**用来判「这一趟是不是被别的任务污染了」。** sweep_median 是串行跑 N 个进程的，
所以进程号就是时间顺序：如果中途有别的任务插进来抢主机内存带宽，被压到的是
**靠后的那几个进程**，而不是所有点均匀变慢。

判法：每一行（一个物理点）用该行的中位数把各进程的值归一化，再把每个进程在
所有行上的相对偏移取中位数 —— 得到「这个进程的档位比全场高/低百分之几」。
**「每进程一个固定档位」本来就存在**（实测组间极差 0.5~2.5%），所以光看档位散
说明不了问题；要看的是**档位随进程号单调上升**，那才是「越晚跑越慢」。

给第二个日志（一趟公认干净的）就多打一列对照：干净那趟的档位应当没有趋势。

⚠ 只看 `=== kernel duration：绝对值（μs）===` 那张表。
⚠ 点数不全的进程（中途被杀）会被跳过并点名 —— 它的档位只由前半段的点算出来，
  跟别人不可比。
"""

import argparse
import re
import statistics
import sys

TABLE_START = "=== kernel duration：绝对值"
ROW = re.compile(
    r"^\s+(?P<label>(?:\w+=\S+\s+)*?)"
    r"(?P<ratio>\d+(?:\.\d+)?)\s+(?P<miss>\d+)\s+(?P<order>\d+)\s+"
    r"(?P<median>\d+(?:\.\d+)?)\s+(?P<spread>\d+(?:\.\d+)?)%\s+"
    r"(?P<vals>[\d.\s]+)$"
)


def parse(path):
    """→ [(点名, [各进程的值])]"""
    rows, inside = [], False
    for line in open(path, encoding="utf-8", errors="replace"):
        if TABLE_START in line:
            inside = True
            continue
        if inside and line.startswith("==="):
            break
        if not inside:
            continue
        m = ROW.match(line.rstrip("\n"))
        if m:
            vals = [float(v) for v in m["vals"].split()]
            rows.append((f'{m["label"].strip()} miss={m["miss"]}'.strip(), vals))
    if not rows:
        sys.exit(f"{path} 里没找到 kernel duration 的中位数表")
    return rows


def tiers(rows):
    """每个进程相对本行中位数的偏移（%），在所有行上取中位数。"""
    width = max(len(v) for _, v in rows)
    full = [(n, v) for n, v in rows if len(v) == width]
    short = len(rows) - len(full)
    per = {i: [] for i in range(width)}
    for _, vals in full:
        med = statistics.median(vals)
        if med <= 0:
            continue
        for i, v in enumerate(vals):
            per[i].append(100.0 * (v - med) / med)
    return width, short, {i: statistics.median(x) for i, x in per.items() if x}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("clean", nargs="?")
    args = ap.parse_args()

    width, short, t = tiers(parse(args.log))
    ct = None
    if args.clean:
        cw, cs, ct = tiers(parse(args.clean))

    print(f"本趟：{args.log}")
    if args.clean:
        print(f"对照：{args.clean}（应当没有趋势）")
    if short:
        print(f"\n⚠ 有 {short} 行的进程数不足 {width} —— 有进程中途被杀，那些行没参与计算。")

    print(f"\n  进程（= 串行的时间顺序）  档位（相对每行中位数）" + ("      干净那趟" if ct else ""))
    print("  " + "-" * (46 + (14 if ct else 0)))
    for i in range(width):
        v = t.get(i)
        if v is None:
            continue
        extra = ""
        if ct and i in ct:
            extra = f"      {ct[i]:+6.2f}%"
        print(f"  P{i + 1:<24}{v:+7.2f}%{extra}")

    vals = [t[i] for i in sorted(t)]
    if len(vals) < 3:
        sys.exit("\n进程太少，判不了趋势。")
    half = len(vals) // 2
    early, late = statistics.median(vals[:half]), statistics.median(vals[half:])
    drift = late - early
    # 单调性：后半段是不是整体高于前半段，而且逐个递增的居多
    ups = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
    print(f"\n  前半段档位中位数 {early:+.2f}%　后半段 {late:+.2f}%　差 {drift:+.2f}%")
    print(f"  相邻递增的次数 {ups}/{len(vals) - 1}")

    if ct:
        cvals = [ct[i] for i in sorted(ct)]
        ch = len(cvals) // 2
        cdrift = statistics.median(cvals[ch:]) - statistics.median(cvals[:ch])
        print(f"  干净那趟的同一个量：{cdrift:+.2f}%")
        limit = max(1.0, 2.0 * abs(cdrift))
    else:
        limit = 1.5
        print(f"  （没给对照，门槛按 {limit}% 拍的 —— 有干净的那趟就传第二个参数，判据会硬得多）")

    print()
    if drift > limit and ups >= len(vals) - 2:
        print(f"❌ 越靠后的进程越慢（+{drift:.2f}% > 门槛 {limit:.2f}%，且几乎逐个递增）")
        print("   符合「中途被别的任务抢了主机带宽」的样子。这一趟的绝对值不要用。")
        sys.exit(1)
    if drift > limit:
        print(f"⚠ 后半段比前半段高 {drift:+.2f}%（门槛 {limit:.2f}%），但不是逐个递增")
        print("   可能是污染，也可能只是进程档位本来就散。再跑一趟干净的对照才能定。")
        sys.exit(2)
    print(f"✅ 没有「越晚越慢」的趋势（{drift:+.2f}%，门槛 {limit:.2f}%）—— 看不出污染。")


if __name__ == "__main__":
    main()
