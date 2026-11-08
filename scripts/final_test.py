"""
最终测试（PRD 第 11 节末）：由人在项目最后手动运行一次。
对因子库中的全部因子在测试集上计算完整明细，结果只给人看，不回流给任何环节。

这是全项目唯一允许 load_market(end=None) 的地方。必须显式加 --confirm 才会运行。

  python scripts/final_test.py --confirm
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from src.agents.common import to_jsonable
from src.orchestrator import load_settings, resolve
from src.records import read_library


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="确认要触碰测试集（只应在项目最后运行一次）")
    ap.add_argument("--settings", default=str(ROOT / "config" / "settings.yaml"))
    args = ap.parse_args(argv)
    if not args.confirm:
        print("final_test 会加载测试集数据，只应在项目最后由人运行一次。确认请加 --confirm。", file=sys.stderr)
        return 2

    settings = load_settings(args.settings)
    cache_dir = resolve(settings.get("cache_dir", "cache"))
    runs_dir = resolve(settings.get("runs_dir", "runs"))
    library = read_library(ROOT / "factor_library.jsonl")
    if not library:
        print("因子库为空，无需测试。")
        return 0

    from src.data import load_market
    from src.expr import compute
    from src.evaluate import evaluate

    market = load_market(settings["data_dir"], end=None, cache_dir=cache_dir)
    results = []
    for e in library:
        try:
            rep = evaluate(compute(e["expression"], market), market, "test", None)
            results.append({"id": e["id"], "name": e["name"], "expression": e["expression"],
                            "direction": e["direction"], "train_ic_mean": e["train_ic_mean"],
                            "test_report": rep, "error": None})
            same = (rep["ic_mean"] > 0) == (e["direction"] > 0)
            print(f"{e['id']} {e['name']}: 训练 IC {e['train_ic_mean']:.4f}  测试 IC {rep['ic_mean']:.4f}  "
                  f"ICIR {rep['icir']:.4f}  方向{'一致' if same else '相反'}")
        except Exception as ex:  # noqa: BLE001
            results.append({"id": e["id"], "name": e["name"], "expression": e["expression"],
                            "error": f"{type(ex).__name__}: {ex}"})
            print(f"{e['id']} 失败：{type(ex).__name__}: {ex}", file=sys.stderr)

    out_dir = runs_dir / ("final_test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_dir / 'results.json'}（只供人查看）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
