"""
输出 20 个经典因子的训练集明细表，交给人核对方向（PRD 15-2）。
5 日反转、20 日反转的 IC 应为正；其余方向由人核对。

结果打印到屏幕，并写到 <runs_dir>/check_classics/<时间>.csv 与 .json（只给人看）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from src.contracts import RUN_DATA_END
from src.agents.common import to_jsonable
from src.orchestrator import load_classics, load_settings, resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=str(ROOT / "config" / "settings.yaml"))
    args = ap.parse_args(argv)

    settings = load_settings(args.settings)
    cache_dir = resolve(settings.get("cache_dir", "cache"))
    runs_dir = resolve(settings.get("runs_dir", "runs"))
    from src.data import load_market
    from src.expr import compute
    from src.evaluate import evaluate

    market = load_market(settings["data_dir"], end=RUN_DATA_END, cache_dir=cache_dir)
    rows, details = [], {}
    for c in load_classics(ROOT / "library" / "classic_factors.yaml"):
        try:
            rep = evaluate(compute(c["expression"], market), market, "train", None)
            details[c["id"]] = rep
            rows.append({"id": c["id"], "name": c["name"], "expression": c["expression"],
                         "ic_mean": rep["ic_mean"], "icir": rep["icir"], "t_stat": rep["t_stat"],
                         "ic_pos_ratio": rep["ic_pos_ratio"], "n_days": rep["n_days"],
                         "coverage_mean": rep["coverage_mean"],
                         **{f"ic_h{h}": v["ic_mean"] for h, v in rep["horizons"].items()},
                         **{f"icir_h{h}": v["icir"] for h, v in rep["horizons"].items()},
                         "long_short": rep["long_short"],
                         "turnover_autocorr": rep["turnover_autocorr"], "error": ""})
        except Exception as e:  # noqa: BLE001
            rows.append({"id": c["id"], "name": c["name"], "expression": c["expression"],
                         "error": f"{type(e).__name__}: {e}"})
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(df.drop(columns=["expression"]).to_string(index=False))

    out_dir = runs_dir / "check_classics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"{stamp}.csv", index=False, encoding="utf-8")
    (out_dir / f"{stamp}.json").write_text(
        json.dumps(to_jsonable(details), ensure_ascii=False, indent=2), encoding="utf-8")
    for cid in ("C01", "C02"):
        hit = df[df["id"].astype(str).str.startswith(cid)]
        if not hit.empty and "ic_mean" in hit and not (hit["ic_mean"] > 0).all():
            print(f"注意：{cid} 反转因子的训练集 IC 不为正，先查 bug。", file=sys.stderr)
    print(f"已写入 {out_dir / (stamp + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
