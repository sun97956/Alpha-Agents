"""
主运行脚本（PRD 第 11 节）。

  python scripts/run.py --rounds 3 --mock                 # 模拟 LLM 端到端
  python scripts/run.py --rounds 3 --mock --inject-error 2  # 第 2 轮人为注入错误（PRD 15-3）
  python scripts/run.py --rounds 10                       # 真实 LLM（先填好 settings.yaml 的 llm 段）

数据一律只加载到 RUN_DATA_END（2021-12-31）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from src.contracts import RUN_DATA_END
from src.orchestrator import Orchestrator, default_engine, load_classics, load_settings, resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多 Agent 因子挖掘：运行若干轮")
    ap.add_argument("--rounds", type=int, default=None, help="轮数，默认读 settings.n_rounds")
    ap.add_argument("--mock", action="store_true", help="使用 MockLLM（不调用真实模型）")
    ap.add_argument("--inject-error", type=int, default=None, metavar="ROUND",
                    help="在第 ROUND 轮人为注入一次错误，验证该轮被记为 skipped")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--settings", default=str(ROOT / "config" / "settings.yaml"))
    args = ap.parse_args(argv)

    settings = load_settings(args.settings)
    n_rounds = args.rounds if args.rounds is not None else int(settings.get("n_rounds", 10))
    cache_dir = resolve(settings.get("cache_dir", "cache"))
    runs_dir = resolve(settings.get("runs_dir", "runs"))
    thr_path = ROOT / "config" / "thresholds.yaml"

    from src.data import load_market
    from src.evaluate import load_thresholds, thresholds_sha256
    from src.agents.llm import MockLLM, OpenAICompatibleLLM

    market = load_market(settings["data_dir"], end=RUN_DATA_END, cache_dir=cache_dir)
    engine = default_engine()
    refs = engine.load_refs(cache_dir)
    llm = MockLLM() if args.mock else OpenAICompatibleLLM.from_settings(settings)
    run_id = args.run_id
    library_path = ROOT / "factor_library.jsonl"
    write_cache = cache_dir
    if args.mock:
        # 模拟运行与真实运行隔离：run_id 加 mock_ 前缀（不进入假设 Agent 的历史），
        # 因子库和新参照值写在本次运行目录里，不污染真实的因子库和 cache
        from src.orchestrator import new_run_id
        run_id = run_id or f"mock_{new_run_id()}"
        library_path = Path(runs_dir) / run_id / "factor_library.jsonl"
        write_cache = Path(runs_dir) / run_id / "cache"

    orch = Orchestrator(market=market, llm=llm, engine=engine,
                        thresholds=load_thresholds(thr_path),
                        thresholds_sha256=thresholds_sha256(thr_path),
                        classics=load_classics(ROOT / "library" / "classic_factors.yaml")
                        + (load_classics(ROOT / "library" / "wq101_factors.yaml")
                           if (ROOT / "library" / "wq101_factors.yaml").exists() else []),
                        refs=refs, runs_dir=runs_dir,
                        library_path=library_path,
                        cache_dir=write_cache, run_id=run_id,
                        inject_error_round=args.inject_error)
    print(f"run_id = {orch.run_id}，共 {n_rounds} 轮，数据截止 {market.end.date()}")
    for i in range(1, n_rounds + 1):
        rec = orch.run_round(i)
        extra = rec["library_id"] or rec["error"] or rec["cannot_build_reason"] or ""
        print(f"第 {i} 轮：训练={rec['train_conclusion']} 验证={rec['validation'] or '-'} "
              f"版本数={len(rec['versions'])} {extra}")
    print(f"运行记录：{orch.recorder.run_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
