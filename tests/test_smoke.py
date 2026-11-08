"""
线 D 端到端冒烟测试（PRD 15-3、15-5）。

单元测试只用 contracts.make_fake_market()；表达式引擎、评估、门槛、参照因子存取
都用本文件里的最小替身（不依赖线 A/B/C 的代码）。
标记 real 的测试使用各线真实实现与真实数据，留到集成阶段运行。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import contracts as C
from src.agents.llm import MockLLM
from src.orchestrator import Engine, Orchestrator, resolve
from src.records import LIBRARY_KEYS, ROUND_KEYS, read_jsonl

THRESHOLDS = {
    "train": {"abs_ic_mean_min": 0.02, "abs_icir_min": 0.06, "max_abs_ref_corr": 0.7},
    "valid": {"abs_ic_mean_min": 0.02, "abs_icir_min": 0.06, "same_sign_as_train": True},
}


# =====================================================================
# 最小替身：表达式引擎（线 B）
# =====================================================================

_TS = {"ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_skew", "ts_rank",
       "ts_delay", "ts_delta", "ts_corr"}
_ALLOWED_NAMES = set(C.FIELDS) | _TS | set(C.CS_OPS) | set(C.GROUP_OPS) | set(C.ARITH_FUNCS) | {"sector"}


def stub_validate(expr: str) -> None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"语法错误：{e.msg}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"不允许的标识符：{node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("不允许的调用形式")
            fn = node.func.id
            if fn in _TS:
                w = node.args[-1]
                if not isinstance(w, ast.Constant):
                    raise ValueError("窗口必须是常数")
                allowed = C.DELAY_WINDOWS if fn in ("ts_delay", "ts_delta") else C.WINDOWS
                if w.value not in allowed:
                    raise ValueError(f"窗口 {w.value} 不在允许范围内")
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda)):
            raise ValueError("不允许的语法")


def stub_compute(expr: str, market: C.Market) -> pd.DataFrame:
    stub_validate(expr)
    uni = market.universe

    def roll(x, d):
        return x.rolling(d, min_periods=d)

    def cs(x):
        return x.where(uni)

    def gn(x, sec):
        x = cs(x)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for s in np.unique(sec.values[~np.isnan(sec.values)]):
            m = sec == s
            xm = x.where(m)
            out = out.where(~m, xm.sub(xm.mean(axis=1), axis=0))
        return out

    ns = {k: v for k, v in market.fields.items()}
    ns.update({
        "sector": market.sector,
        "ts_mean": lambda x, d: roll(x, d).mean(), "ts_std": lambda x, d: roll(x, d).std(),
        "ts_sum": lambda x, d: roll(x, d).sum(), "ts_max": lambda x, d: roll(x, d).max(),
        "ts_min": lambda x, d: roll(x, d).min(), "ts_skew": lambda x, d: roll(x, d).skew(),
        "ts_delay": lambda x, d: x.shift(d), "ts_delta": lambda x, d: x - x.shift(d),
        "ts_rank": lambda x, d: roll(x, d).rank(pct=True),
        "ts_corr": lambda x, y, d: roll(x, d).corr(y),
        "cs_rank": lambda x: cs(x).rank(axis=1, pct=True),
        "cs_zscore": lambda x: cs(x).sub(cs(x).mean(axis=1), axis=0).div(cs(x).std(axis=1), axis=0),
        "group_neutralize": gn,
        "log": lambda x: np.log(x.where(x > 0)), "abs": lambda x: x.abs(), "sign": np.sign,
    })
    out = eval(compile(ast.parse(expr, mode="eval"), "<expr>", "eval"), {"__builtins__": {}}, ns)
    out = out.astype("float64").replace([np.inf, -np.inf], np.nan)
    return out.where(uni)


# =====================================================================
# 最小替身：评估 / 门槛 / 参照（线 A、C）
# =====================================================================

def stub_split_dates(market: C.Market, split: str, h: int) -> pd.DatetimeIndex:
    lo, hi = (pd.Timestamp(x) for x in C.SPLITS[split])
    d = market.dates
    pos = np.arange(len(d)) + 1 + h
    ok = (d >= lo) & (d <= hi) & (pos < len(d))
    ok_idx = np.where(ok)[0]
    keep = [i for i in ok_idx if d[pos[i]] <= hi]
    return d[keep]


def _row_corr(a: pd.DataFrame, b: pd.DataFrame, min_n: int = C.MIN_STOCKS_PER_DAY) -> pd.Series:
    m = a.notna() & b.notna()
    ra = a.where(m).rank(axis=1)
    rb = b.where(m).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    c = (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return c.where(m.sum(axis=1) >= min_n).dropna()


def stub_evaluate(factor, market, split, refs=None, horizon=5) -> dict:
    f = factor.where(market.universe)

    def ic(h):
        d = stub_split_dates(market, split, h)
        return _row_corr(f.loc[d], market.fwd_ret[h].loc[d])

    ic5 = ic(C.MAIN_HORIZON)
    n = len(ic5)
    mu, sd = float(ic5.mean()), float(ic5.std())
    yearly = []
    for y, s in ic5.groupby(ic5.index.year):
        yearly.append({"year": int(y), "ic_mean": float(s.mean()),
                       "icir": float(s.mean() / s.std()) if len(s) > 1 else float("nan"),
                       "ic_pos_ratio": float((s > 0).mean()), "n_days": int(len(s))})
    d5 = stub_split_dates(market, split, C.MAIN_HORIZON)
    pct = f.loc[d5].rank(axis=1, pct=True)
    q = np.ceil(pct * C.N_QUANTILES).clip(1, C.N_QUANTILES)
    fwd = market.fwd_ret[C.MAIN_HORIZON].loc[d5]
    qret = [float(fwd.where(q == k).stack().mean()) for k in range(1, C.N_QUANTILES + 1)]
    max_corr = max_id = None
    if refs:
        dt = stub_split_dates(market, "train", C.MAIN_HORIZON)
        best = None
        for rid, r in refs.items():
            cc = float(_row_corr(f.loc[dt], r.reindex(index=dt, columns=f.columns), 2).mean())
            if best is None or abs(cc) > abs(best[0]):
                best = (cc, rid)
        max_corr, max_id = best
    return {
        "split": split, "n_days": n, "coverage_mean": float(f.loc[d5].notna().sum(axis=1).mean()),
        "ic_mean": mu, "ic_std": sd, "icir": mu / sd, "t_stat": mu / sd * np.sqrt(n),
        "ic_pos_ratio": float((ic5 > 0).mean()), "yearly": yearly,
        "horizons": {},
        "hac_t_stat": mu / sd * np.sqrt(n), "ic_ci95_low": mu - 1.96 * sd / np.sqrt(n),
        "ic_ci95_high": mu + 1.96 * sd / np.sqrt(n),
        "decay": {str(h): float(ic(h).mean()) for h in C.HORIZONS},
        "quantile_ret": qret, "long_short": qret[-1] - qret[0],
        "net_long_short": qret[-1] - qret[0], "long_short_turnover": 0.0,
        "monotonicity": 0.0, "group_ic": {},
        "turnover_autocorr": float(_row_corr(f.loc[d5].iloc[1:],
                                             f.loc[d5].shift(1).iloc[1:], 2).mean()),
        "max_ref_corr": max_corr, "max_ref_id": max_id,
    }


def stub_gate_train(rep, thr) -> dict:
    t = thr["train"]
    corr = rep["max_ref_corr"]
    checks = [
        {"name": "abs_ic_mean", "value": abs(rep["ic_mean"]), "threshold": t["abs_ic_mean_min"],
         "passed": abs(rep["ic_mean"]) >= t["abs_ic_mean_min"]},
        {"name": "abs_icir", "value": abs(rep["icir"]), "threshold": t["abs_icir_min"],
         "passed": abs(rep["icir"]) >= t["abs_icir_min"]},
        {"name": "max_ref_corr", "value": abs(corr) if corr is not None else 0.0,
         "threshold": t["max_abs_ref_corr"],
         "passed": corr is None or abs(corr) < t["max_abs_ref_corr"]},
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def stub_gate_valid(vrep, trep, thr) -> dict:
    t = thr["valid"]
    checks = [
        {"name": "abs_ic_mean", "value": abs(vrep["ic_mean"]), "threshold": t["abs_ic_mean_min"],
         "passed": abs(vrep["ic_mean"]) >= t["abs_ic_mean_min"]},
        {"name": "abs_icir", "value": abs(vrep["icir"]), "threshold": t["abs_icir_min"],
         "passed": abs(vrep["icir"]) >= t["abs_icir_min"]},
        {"name": "same_sign", "value": float(np.sign(vrep["ic_mean"]) * np.sign(trep["ic_mean"])),
         "threshold": 1.0, "passed": np.sign(vrep["ic_mean"]) == np.sign(trep["ic_mean"])},
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def stub_save_ref(ref_id, factor, market, cache_dir="cache") -> None:
    p = Path(cache_dir) / "ref_values"
    p.mkdir(parents=True, exist_ok=True)
    lo, hi = C.SPLITS[C.REF_SPLIT]
    ranks = factor.where(market.universe).loc[lo:hi].rank(axis=1, pct=True).astype("float32")
    ranks.to_pickle(p / f"{ref_id}.pkl")


def stub_load_refs(cache_dir="cache") -> dict:
    p = Path(cache_dir) / "ref_values"
    return {f.stem: pd.read_pickle(f) for f in sorted(p.glob("*.pkl"))} if p.exists() else {}


def stub_engine(**over) -> Engine:
    kw = dict(validate=stub_validate, compute=stub_compute, evaluate=stub_evaluate,
              gate_train=stub_gate_train, gate_valid=stub_gate_valid,
              save_ref=stub_save_ref, load_refs=stub_load_refs)
    kw.update(over)
    return Engine(**kw)


CLASSICS = [
    {"id": "C05_lowvol", "name": "低波动", "expression": "-ts_std(ret, 20)", "logic": "低波异象"},
    {"id": "C07_size", "name": "规模", "expression": "-log(mktcap)", "logic": "小市值溢价"},
]


@pytest.fixture(scope="module")
def market():
    return C.make_fake_market()


def make_orch(tmp_path, market, llm=None, engine=None, refs=None, **kw) -> Orchestrator:
    return Orchestrator(market=market, llm=llm or MockLLM(), engine=engine or stub_engine(),
                        thresholds=THRESHOLDS, thresholds_sha256="sha-test",
                        classics=CLASSICS, refs=refs, runs_dir=tmp_path / "runs",
                        library_path=tmp_path / "factor_library.jsonl",
                        cache_dir=tmp_path / "cache", run_id=kw.pop("run_id", "r1"), **kw)


# =====================================================================
# 替身自检：埋入的反转信号能被替身评估出来
# =====================================================================

def test_stub_evaluate_sees_reversal(market):
    rep = stub_evaluate(stub_compute("-ts_sum(ret, 5)", market), market, "train")
    assert rep["ic_mean"] > 0.02 and rep["n_days"] > 100


# =====================================================================
# 冒烟：MockLLM 跑 3 轮
# =====================================================================

def _check_round_record(r: dict) -> None:
    assert tuple(r.keys()) == ROUND_KEYS
    assert isinstance(r["run_id"], str) and isinstance(r["round"], int)
    pd.Timestamp(r["timestamp"])                               # ISO 8601 可解析
    assert r["thresholds_sha256"] == "sha-test"
    assert r["train_conclusion"] in ("pass", "fail", "cannot_build", "skipped")
    assert r["validation"] in (None, "pass", "fail")
    if r["hypothesis"] is not None:
        assert set(r["hypothesis"]) == {"hypothesis", "logic", "mechanism", "horizon"}
    for v in r["versions"]:
        assert set(v) == set(C.VersionRecord.__annotations__)
        assert 0 <= v["version"] <= C.MAX_ADJUST
    assert len(r["versions"]) <= C.MAX_ADJUST + 1
    if r["train_conclusion"] == "fail":
        assert r["attribution"]["attribution"] in ("formula", "hypothesis")
        assert r["validation"] is None
    if r["train_conclusion"] == "pass":
        assert r["versions"][-1]["train_gate_passed"] is True
        assert r["validation"] in ("pass", "fail")
    if r["train_conclusion"] == "skipped":
        assert r["error"]
    else:
        assert r["error"] is None
    if r["library_id"] is not None:
        assert r["validation"] == "pass"


def test_mock_three_rounds_end_to_end(tmp_path, market):
    orch = make_orch(tmp_path, market)
    recs = orch.run(3)
    assert len(recs) == 3

    log = read_jsonl(tmp_path / "runs" / "r1" / "run_log.jsonl")
    assert [r["round"] for r in log] == [1, 2, 3]
    for r in log:
        _check_round_record(r)
        assert r["train_conclusion"] != "skipped"
        for v in r["versions"]:
            if v["report_path"]:
                saved = json.loads(Path(v["report_path"]).read_text(encoding="utf-8"))
                assert saved["report"]["split"] == "train"

    # 第 1 轮的 -ts_sum(ret,5) 在假数据上训练、验证都应通过并入库
    assert log[0]["train_conclusion"] == "pass" and log[0]["validation"] == "pass"
    lib = read_jsonl(tmp_path / "factor_library.jsonl")
    assert len(lib) >= 1
    for e in lib:
        assert tuple(e.keys()) == LIBRARY_KEYS
        assert e["validation"] == "pass" and e["direction"] in (1, -1)
        assert np.sign(e["train_ic_mean"]) == e["direction"]
        src = [r for r in log if r["run_id"] == e["run_id"] and r["round"] == e["round"]]
        assert len(src) == 1 and src[0]["library_id"] == e["id"]          # 可追溯
        assert src[0]["versions"][-1]["expression"] == e["expression"]
    assert [e["id"] for e in lib] == [f"F{i:04d}" for i in range(1, len(lib) + 1)]
    # 入库因子成为后续去重参照
    assert set(e["id"] for e in lib) <= set(stub_load_refs(tmp_path / "cache"))
    assert set(e["id"] for e in lib) <= set(orch.refs)

    # LLM 日志：每次调用都有 prompt 和回复
    llm_log = read_jsonl(tmp_path / "runs" / "r1" / "llm_log.jsonl")
    assert len(llm_log) == len(orch.llm.inner.calls)
    for e in llm_log:
        assert e["system"] and e["user"] and e["response"] and e["error"] is None
        assert e["run_id"] == "r1" and e["round"] in (1, 2, 3)


def test_injected_error_round_is_skipped_and_next_continues(tmp_path, market):
    orch = make_orch(tmp_path, market, inject_error_round=2)
    recs = orch.run(3)
    assert recs[1]["train_conclusion"] == "skipped" and "人为注入" in recs[1]["error"]
    assert recs[0]["train_conclusion"] != "skipped"
    assert recs[2]["train_conclusion"] != "skipped" and recs[2]["hypothesis"] is not None
    log = read_jsonl(tmp_path / "runs" / "r1" / "run_log.jsonl")
    assert [r["train_conclusion"] == "skipped" for r in log] == [False, True, False]
    for r in log:
        _check_round_record(r)


def test_llm_exception_is_skipped(tmp_path, market):
    mock = MockLLM()
    orch = make_orch(tmp_path, market, llm=mock)
    orch.run_round(1)
    mock.fail_on_calls = {len(mock.calls) + 1: ConnectionError("网络断了")}
    r2 = orch.run_round(2)
    r3 = orch.run_round(3)
    assert r2["train_conclusion"] == "skipped" and "ConnectionError" in r2["error"]
    assert r3["train_conclusion"] != "skipped"
    errs = [e for e in read_jsonl(tmp_path / "runs" / "r1" / "llm_log.jsonl") if e["error"]]
    assert len(errs) == 1 and errs[0]["round"] == 2


def test_compute_exception_is_skipped(tmp_path, market):
    calls = {"n": 0}

    def flaky(expr, mkt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("计算炸了")
        return stub_compute(expr, mkt)

    orch = make_orch(tmp_path, market, engine=stub_engine(compute=flaky))
    r1, r2 = orch.run_round(1), orch.run_round(2)
    assert r1["train_conclusion"] == "skipped" and "MemoryError" in r1["error"]
    assert r1["hypothesis"] is not None                  # 已完成的部分保留
    assert r2["train_conclusion"] != "skipped"


def test_isolation_no_validation_metrics_reach_agents(tmp_path, market):
    SENTINEL = 0.0987654321

    def ev(factor, mkt, split, refs=None):
        rep = stub_evaluate(factor, mkt, split, refs)
        if split != "train":
            assert refs is None                           # refs 只在训练集传入
            rep.update(ic_mean=SENTINEL, icir=SENTINEL, t_stat=SENTINEL)
        return rep

    orch = make_orch(tmp_path, market, engine=stub_engine(evaluate=ev))
    orch.run(3)
    run_dir = tmp_path / "runs" / "r1"
    text = "".join(p.read_text(encoding="utf-8") for p in run_dir.rglob("*") if p.is_file())
    text += (tmp_path / "factor_library.jsonl").read_text(encoding="utf-8") \
        if (tmp_path / "factor_library.jsonl").exists() else ""
    assert "0.0987654" not in text
    assert '"split": "valid"' not in text and '"split": "test"' not in text
    for e in read_jsonl(run_dir / "llm_log.jsonl"):
        assert "valid_report" not in e["user"] and "0.0987654" not in e["user"]


def test_hypothesis_agent_sees_history_and_library(tmp_path, market):
    mock = MockLLM()
    orch = make_orch(tmp_path, market, llm=mock)
    r1 = orch.run_round(1)
    orch.run_round(2)
    hyp_prompts = [u for s, u in mock.calls if "[TASK: hypothesis]" in s]
    assert r1["hypothesis"]["hypothesis"] not in hyp_prompts[0]
    assert r1["hypothesis"]["hypothesis"] in hyp_prompts[1]
    assert f"验证：{r1['validation']}" in hyp_prompts[1]
    if r1["library_id"]:
        assert r1["library_id"] in hyp_prompts[1]


def test_history_spans_previous_runs(tmp_path, market):
    o1 = make_orch(tmp_path, market, run_id="runA")
    r = o1.run_round(1)
    mock = MockLLM()
    o2 = make_orch(tmp_path, market, llm=mock, run_id="runB")
    o2.run_round(1)
    first_hyp_prompt = next(u for s, u in mock.calls if "[TASK: hypothesis]" in s)
    assert "runA 第 1 轮" in first_hyp_prompt and r["hypothesis"]["hypothesis"] in first_hyp_prompt


def test_rejects_market_beyond_run_data_end(tmp_path):
    late = C.make_fake_market(n_days=120, start="2021-10-01")
    assert late.end > pd.Timestamp(C.RUN_DATA_END)
    with pytest.raises(ValueError):
        make_orch(tmp_path, late)


# =====================================================================
# 真实数据（集成阶段：python -m pytest -m real -q）
# =====================================================================

@pytest.mark.real
def test_real_mock_three_rounds(tmp_path):
    from src.data import load_market
    from src.evaluate import load_thresholds, thresholds_sha256
    from src.orchestrator import (PROJECT_ROOT, default_engine, load_classics, load_settings,
                                  resolve)

    settings = load_settings()
    cache = resolve(settings.get("cache_dir", "cache"))
    mkt = load_market(resolve(settings["data_dir"]), end=C.RUN_DATA_END, cache_dir=cache)
    assert mkt.end <= pd.Timestamp(C.RUN_DATA_END)
    assert mkt.dates.max() <= pd.Timestamp(C.RUN_DATA_END)
    eng = default_engine()
    thr = PROJECT_ROOT / "config" / "thresholds.yaml"
    # 去重参照读真实 cache，新入库因子的参照值写到临时目录，避免污染真实 cache
    refs = eng.load_refs(cache)
    orch = Orchestrator(market=mkt, llm=MockLLM(), engine=eng, thresholds=load_thresholds(thr),
                        thresholds_sha256=thresholds_sha256(thr),
                        classics=load_classics(PROJECT_ROOT / "library" / "classic_factors.yaml"),
                        refs=refs, runs_dir=tmp_path / "runs",
                        library_path=tmp_path / "factor_library.jsonl",
                        cache_dir=tmp_path / "cache", run_id="real_smoke", inject_error_round=2)
    recs = orch.run(3)
    assert recs[1]["train_conclusion"] == "skipped"
    assert recs[0]["train_conclusion"] != "skipped", recs[0]["error"]
    assert recs[2]["train_conclusion"] != "skipped", recs[2]["error"]
    for r in read_jsonl(tmp_path / "runs" / "real_smoke" / "run_log.jsonl"):
        assert tuple(r.keys()) == ROUND_KEYS
