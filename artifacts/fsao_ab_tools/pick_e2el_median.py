#!/usr/bin/env python3
"""取 base/opt 中 E2EL avg 最接近各自中位数的那一轮，比较四个指标。

用法:
    AB=/home/z00943842/base $AISV_DIR/bin/python pick_e2el_median.py

指标:
    TPOT avg, TPOT med, output_throughput, E2E L avg
"""
import glob
import os
import sys

import pandas as pd

AB = os.environ.get("AB", os.path.expanduser("~/fsao_ab"))
RESULTS = os.path.join(AB, "results")


def _norm(s):
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def pick_col(df, key):
    """按语义找列名，兼容不同版本的 CSV。"""
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}

    def match(pred):
        for c in cols:
            if pred(norm[c]):
                return c
        return None

    if key == "tpot_avg":
        return match(lambda s: "tpot" in s and "avg" in s and "slo" not in s and "p90" not in s)
    if key == "tpot_med":
        return match(lambda s: "tpot" in s and ("med" in s or "median" in s))
    if key == "throughput":
        return match(lambda s: "output" in s and "throughput" in s and "single" not in s)
    if key == "e2el_avg":
        return match(lambda s: ("e2el" in s or "e2e latency" in s) and "avg" in s and "med" not in s)
    return None


def pick_measured_row(df):
    """一个 CSV 可能包含 warmup/哨兵行；只保留正式测试那一行。"""
    tpot = pick_col(df, "tpot_avg")
    if tpot is not None:
        df = df[pd.to_numeric(df[tpot], errors="coerce").fillna(0) < 99999]

    thr = pick_col(df, "throughput")
    if len(df) > 1 and thr is not None:
        df = (
            df.assign(__thr=pd.to_numeric(df[thr], errors="coerce"))
            .sort_values("__thr")
            .tail(1)
        )
    return df


def load_build(build):
    files = sorted(glob.glob(os.path.join(RESULTS, f"{build}_r*.csv")))
    if not files:
        raise SystemExit(f"no CSV for {build} under {RESULTS}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        df = pick_measured_row(df)
        df["__round__"] = os.path.basename(path).replace(".csv", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    data = {b: load_build(b) for b in ("base", "opt")}

    keys = ["tpot_avg", "tpot_med", "throughput", "e2el_avg"]
    cols = {}
    for key in keys:
        c = pick_col(data["base"], key)
        if c is None:
            print(f"[error] 找不到 {key} 对应的列。CSV 列有：", file=sys.stderr)
            print(list(data["base"].columns), file=sys.stderr)
            sys.exit(1)
        cols[key] = c

    print("=== 所有有效轮次 ===")
    for build, df in data.items():
        show = df[["__round__"] + [cols[k] for k in keys]].copy()
        print(f"\n[{build}]")
        print(show.to_string(index=False))

    selected = {}
    print("\n=== 选择 E2EL avg 最接近中位数的那一轮 ===")
    for build, df in data.items():
        e = pd.to_numeric(df[cols["e2el_avg"]], errors="coerce")
        med = e.median()
        idx = (e - med).abs().idxmin()
        selected[build] = df.loc[idx]
        print(
            f"{build}: median E2EL avg = {med:.3f}  "
            f"-> 选中 {df.loc[idx, '__round__']} = {e.loc[idx]:.3f}"
        )

    base, opt = selected["base"], selected["opt"]
    print("\n=== 四个指标对比 ===")
    print(f"base 轮次: {base['__round__']}")
    print(f"opt  轮次: {opt['__round__']}\n")

    header = f"{'指标':<24s} {'base':>14s} {'opt':>14s} {'变化%':>10s}"
    print(header)
    print("-" * len(header))
    labels = {
        "tpot_avg": "TPOT avg",
        "tpot_med": "TPOT med",
        "throughput": "output_throughput",
        "e2el_avg": "E2EL avg",
    }
    for key in keys:
        bv = float(base[cols[key]])
        ov = float(opt[cols[key]])
        delta = (ov - bv) / bv * 100 if bv else float("nan")
        print(f"{labels[key]:<24s} {bv:14.3f} {ov:14.3f} {delta:+10.2f}")

    # 顺便保存一份
    out = os.path.join(RESULTS, "compare_e2el_median.csv")
    pd.DataFrame(
        [
            {
                "metric": labels[k],
                "base_round": base["__round__"],
                "opt_round": opt["__round__"],
                "base": float(base[cols[k]]),
                "opt": float(opt[cols[k]]),
                "delta_pct": (float(opt[cols[k]]) - float(base[cols[k]]))
                / float(base[cols[k]])
                * 100
                if float(base[cols[k]])
                else float("nan"),
            }
            for k in keys
        ]
    ).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
