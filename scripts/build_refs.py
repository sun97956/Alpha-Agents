"""
用表达式引擎一次性算出 20 个经典因子的训练集截面排序值，存入 cache/ref_values/<id>.parquet（PRD 第 13 节）。

  python scripts/build_refs.py            # 已存在的跳过
  python scripts/build_refs.py --force    # 全部重建（公式或原始数据变了时用）
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from src.contracts import RUN_DATA_END
from src.orchestrator import load_classics, load_settings, resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--settings", default=str(ROOT / "config" / "settings.yaml"))
    args = ap.parse_args(argv)

    settings = load_settings(args.settings)
    cache_dir = resolve(settings.get("cache_dir", "cache"))
    from src.data import load_market
    from src.expr import compute
    from src.refstore import save_ref

    market = load_market(settings["data_dir"], end=RUN_DATA_END, cache_dir=cache_dir)
    classics = load_classics(ROOT / "library" / "classic_factors.yaml")
    n_ok = 0
    for c in classics:
        target = cache_dir / "ref_values" / f"{c['id']}.parquet"
        if target.exists() and not args.force:
            print(f"跳过（已存在）：{c['id']}")
            n_ok += 1
            continue
        try:
            save_ref(c["id"], compute(c["expression"], market), market, cache_dir)
            print(f"已保存：{c['id']}  {c['expression']}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"失败：{c['id']}  {c['expression']} —— {type(e).__name__}: {e}", file=sys.stderr)
    print(f"完成 {n_ok}/{len(classics)}")
    return 0 if n_ok == len(classics) else 1


if __name__ == "__main__":
    sys.exit(main())
