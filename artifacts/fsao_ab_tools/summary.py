#!/usr/bin/env python3
"""汇总 $RESULTS/base_r*.csv 与 opt_r*.csv：每轮值、中位数、极差、opt vs base 变化。"""
import glob
import os

import pandas as pd

AB = os.environ.get("AB", os.path.expanduser("~/fsao_ab"))
RESULTS = os.environ.get("RESULTS") or os.path.join(AB, "results")

METRICS = [
    "TPOT avg",
    "TPOT SLO_P90",
    "output_throughput",
    "single_output_throughput",
    "E2E_time",
    "repeat_rate",
]


def load(build):
    files = sorted(glob.glob(os.path.join(RESULTS, f"{build}_r*.csv")))
    if not files:
        return None, []
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        df["__round__"] = os.path.basename(path).replace(".csv", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True), files


def main():
    data = {}
    for build in ("base", "opt"):
        df, files = load(build)
        if df is None:
            print(f"[warn] no CSV for {build}")
        else:
            data[build] = df

    if not data:
        raise SystemExit("no results")

    rows = []
    for build, df in data.items():
        # 同时收集命中率相关列（列名因版本而异）
        extra = [c for c in df.columns if "hit" in c.lower() or "命中" in c]
        cols = [c for c in METRICS + extra if c in df.columns]
        for _, row in df.iterrows():
            rec = {"build": build, "round": row["__round__"]}
            for c in cols:
                rec[c] = row[c]
            rows.append(rec)

    out = pd.DataFrame(rows)
    print("=== per-round ===")
    print(out.to_string(index=False))

    print("\n=== median / spread ===")
    summary = []
    for build in ("base", "opt"):
        sub = out[out["build"] == build]
        if sub.empty:
            continue
        rec = {"build": build, "rounds": len(sub)}
        for c in METRICS:
            if c not in sub.columns:
                continue
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            if vals.empty:
                continue
            med = vals.median()
            rec[f"{c} med"] = round(med, 3)
            rec[f"{c} min"] = round(vals.min(), 3)
            rec[f"{c} max"] = round(vals.max(), 3)
            rec[f"{c} spread%"] = round((vals.max() - vals.min()) / med * 100, 2) if med else None
        summary.append(rec)
    print(pd.DataFrame(summary).to_string(index=False))

    if "base" in data and "opt" in data:
        print("\n=== opt vs base (median) ===")
        for c in METRICS:
            if c not in out.columns:
                continue
            b = pd.to_numeric(out.loc[out["build"] == "base", c], errors="coerce").dropna()
            o = pd.to_numeric(out.loc[out["build"] == "opt", c], errors="coerce").dropna()
            if b.empty or o.empty or b.median() == 0:
                continue
            bm, om = b.median(), o.median()
            delta = (om - bm) / bm * 100
            noise = max(
                (b.max() - b.min()) / bm * 100,
                (o.max() - o.min()) / om * 100,
            )
            flag = "   <-- within noise" if abs(delta) <= noise else ""
            print(
                f"{c:24s} base={bm:10.3f} opt={om:10.3f} "
                f"delta={delta:+7.2f}% noise~{noise:5.2f}%{flag}"
            )

    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "rounds.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    print(f"\nsaved: {RESULTS}/rounds.csv, {RESULTS}/summary.csv")


if __name__ == "__main__":
    main()
