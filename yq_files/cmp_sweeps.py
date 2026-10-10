"""逐点比两趟 sweep_median 的中位数表，给出判据。

    python3 yq_files/cmp_sweeps.py A.log B.log [--label-a 串行] [--label-b 并行]

用在任何「同一组点、两种设定」的对照上：串行 vs 并行、芯片 A vs 芯片 B、
优化前 vs 优化后。**几十行的表人眼校对等于没校对** —— 2026-08-08 抄一张 10 行
的表串了 5 行，肉眼完全看不出来。

判据不是拍脑袋的百分比，是**参照那一趟自己的 A/A**：同一个物理点在一趟里有
两个标签（`round(r×topk)` 相同、名字不同），它们的差就是那一趟的误差棒，
门槛取它的两倍（下限 0.5%）。超过门槛的行标 ⚠。

⚠⚠ **门槛只能来自参照臂，不能取两臂的最大值。** 2026-08-12 第一版取了最大值：
串行 A/A 0.20%、并行 A/A 13.64%，门槛于是涨到 27%，六个点全过 —— 而真相恰恰
是「并行那一臂自己重复不出来」。**一个臂越烂判据越松，方向是反的。**

所以还有一道硬闸：**任一臂自己的 A/A 超过 AA_MAX（2%）就直接判不合格**，
不看逐点差 —— 连自己都重复不出来的数，跟谁比都没有意义。

⚠ 它只比 `=== kernel duration：绝对值（μs）===` 那张表。端到端不作判据
（含约 160 μs 框架开销且在漂），要看另说。
"""

import argparse
import collections
import re
import sys

# 行长这样（前面的 key 字段可能没有，取决于这一趟扫了哪些轴）：
#   plan=off  0.05    1946    3      577.6       2.22%   571.8 580.7 ...
ROW = re.compile(
    r"^\s+(?P<label>(?:\w+=\S+\s+)*?)"
    r"(?P<ratio>\d+(?:\.\d+)?)\s+"
    r"(?P<miss>\d+)\s+(?P<order>\d+)\s+"
    r"(?P<median>\d+(?:\.\d+)?)\s+"
    r"(?P<spread>\d+(?:\.\d+)?)%\s+"
    r"(?P<vals>[\d.\s]+)$"
)
TABLE_START = "=== kernel duration：绝对值"
TABLE_END = "==="


def parse(path):
    """→ {(label, miss): [中位数, ...]}，同一个物理点的多个标签归到一起。"""
    rows = collections.defaultdict(list)
    inside = False
    for line in open(path, encoding="utf-8", errors="replace"):
        if TABLE_START in line:
            inside = True
            continue
        if inside and line.startswith(TABLE_END):
            break
        if not inside:
            continue
        m = ROW.match(line.rstrip("\n"))
        if m:
            rows[(m["label"].strip(), int(m["miss"]))].append(float(m["median"]))
    if not rows:
        sys.exit(f"{path} 里没找到 kernel duration 的中位数表 —— 这一趟是不是没跑完？")
    return rows


def aa(rows):
    """一趟自己的 A/A：同一个物理点两个标签之间的最大相对差（%）。

    → (最大差, 在哪个点, 有几个配对点)。
    ⚠ 第三个值必须带上：差恰好为 0 和「压根没有配对标签」是两回事，
      前者是好消息、后者是判不了。混成一个 None 会把好消息说成坏消息。
    """
    worst, where, pairs = 0.0, None, 0
    for key, vals in rows.items():
        if len(vals) < 2:
            continue
        pairs += 1
        d = 100.0 * (max(vals) - min(vals)) / min(vals)
        if d > worst:
            worst, where = d, key
    return worst, where, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--aa-max", type=float, default=2.0,
                    help="任一臂自己的 A/A 超过这个就直接判不合格（默认 2%%）")
    ap.add_argument("--floor", type=float, default=0.5,
                    help="门槛下限（默认 0.5%%），免得 A/A 近乎 0 时门槛紧到误报")
    args = ap.parse_args()

    A, B = parse(args.log_a), parse(args.log_b)
    aa_a, wa, na = aa(A)
    aa_b, wb, nb = aa(B)
    # ⚠ 门槛只取**参照臂**（A）的 A/A，不取两臂最大值 —— 取最大值等于让越烂的
    #   一臂把判据撑得越松。见文件抬头那段。
    limit = max(args.floor, 2.0 * aa_a)

    print(f"{args.label_a}：{args.log_a}")
    print(f"{args.label_b}：{args.log_b}")
    print(f"\nA/A（同一趟内同一物理点两个标签的最大差）：")
    def _aa_line(label, val, where, n):
        if n == 0:
            return f"  {label} 没有配对标签，测不出"
        return f"  {label} {val:.2f}%（{n} 个配对点" + (f"，最大在 {where}" if where else "，全部为 0") + "）"

    print(_aa_line(args.label_a, aa_a, wa, na))
    print(_aa_line(args.label_b, aa_b, wb, nb))
    unusable = []
    if na and aa_a > args.aa_max:
        unusable.append((args.label_a, aa_a))
    if nb and aa_b > args.aa_max:
        unusable.append((args.label_b, aa_b))

    if na == 0 and nb == 0:
        print("\n⚠ 两趟都没有配对标签，判不了 —— 门槛只能拍脑袋，别用这个结果。")
        limit = 2.0
    elif limit <= 0:
        print("\n⚠ A/A 恰好为 0（假数据才会这样）—— 门槛退到 2%，真数据下不该出现。")
        limit = 2.0
    print(f"门槛 = 参照臂（{args.label_a}）A/A 的两倍 = {limit:.2f}%（下限 {args.floor}%）")
    if unusable:
        print()
        for name, v in unusable:
            print(f"❌ {name} 这一臂自己就重复不出来：A/A {v:.2f}% > {args.aa_max}%")
        print("   连自己都复现不了的数，跟谁比都没意义 —— 逐点差只作参考，不构成判定。")
    print()

    shared = sorted(set(A) & set(B), key=lambda k: (k[0], -k[1]))
    if not shared:
        sys.exit("两趟没有共同的点，比不了 —— 检查是不是扫的点列不一样。")
    only_a, only_b = set(A) - set(B), set(B) - set(A)
    if only_a or only_b:
        print(f"⚠ 只在一边出现的点：{args.label_a} {len(only_a)} 个，{args.label_b} {len(only_b)} 个，已跳过\n")

    hdr = f"  {'点':<22}{args.label_a:>10}{args.label_b:>10}{'差':>9}   判"
    print(hdr)
    print("  " + "-" * (len(hdr) + 6))
    bad = []
    for key in shared:
        va = sum(A[key]) / len(A[key])
        vb = sum(B[key]) / len(B[key])
        d = 100.0 * (vb - va) / va if va else 0.0
        ok = abs(d) <= limit
        if not ok:
            bad.append((key, d))
        label = f"{key[0]} miss={key[1]}".strip()
        print(f"  {label:<22}{va:>10.1f}{vb:>10.1f}{d:>8.2f}%   {'ok' if ok else '⚠'}")

    print()
    if unusable:
        for name, v in unusable:
            print(f"❌ 不合格：{name} 的 A/A 是 {v:.2f}%（上限 {args.aa_max}%）")
        if bad:
            worst = max(bad, key=lambda x: abs(x[1]))
            print(f"   另有 {len(bad)}/{len(shared)} 个点超门槛，最大 {worst[1]:+.2f}%")
        sys.exit(1)
    if bad:
        worst = max(bad, key=lambda x: abs(x[1]))
        print(f"❌ {len(bad)}/{len(shared)} 个点超门槛，最大 {worst[1]:+.2f}%（{worst[0][0]} miss={worst[0][1]}）")
        print("   两种设定量出来的不是同一个东西，别混用。")
        sys.exit(1)
    print(f"✅ {len(shared)} 个点全部落在门槛内 —— 两种设定等价，可以混用。")


if __name__ == "__main__":
    main()
