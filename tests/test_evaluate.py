"""线 C 单元测试：src/evaluate.py、src/refstore.py。只用 make_fake_market() 的假数据。"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src import evaluate as ev
from src import refstore as rs
from src.contracts import (
    HORIZONS,
    MAIN_HORIZON,
    MIN_STOCKS_PER_DAY,
    N_QUANTILES,
    REF_SPLIT,
    SPLITS,
    Market,
    make_fake_market,
)

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = ROOT / "config" / "thresholds.yaml"


# ---------------------------------------------------------------------
# 最小替身（线 B 的表达式引擎）
# ---------------------------------------------------------------------

def stub_rev5(m: Market) -> pd.DataFrame:
    """替身：-ts_sum(ret, 5)，窗口不完整为 NaN，只保留股票池。"""
    f = -m.fields["ret"].rolling(5, min_periods=5).sum()
    return f.where(m.universe)


def stub_noise(m: Market, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.normal(size=m.fields["close"].shape), index=m.dates, columns=m.permnos)
    return f.where(m.universe)


@pytest.fixture(scope="module")
def market() -> Market:
    return make_fake_market()


@pytest.fixture(scope="module")
def thresholds() -> dict:
    return ev.load_thresholds(THRESHOLDS_PATH)


@pytest.fixture(scope="module")
def thr_corr(thresholds) -> dict:
    """门槛文件已不含相关性门槛；去重逻辑仍保留，用这个夹具单独测试。"""
    return {"train": dict(thresholds["train"], max_abs_ref_corr=0.8), "valid": thresholds["valid"]}


@pytest.fixture(scope="module")
def rev5_train(market) -> dict:
    return ev.evaluate(stub_rev5(market), market, "train")


def make_report(ic_mean=0.03, icir=0.1, max_ref_corr=0.1, max_ref_id="C01") -> dict:
    return dict(split="train", n_days=100, coverage_mean=150.0, ic_mean=ic_mean, ic_std=0.2,
                icir=icir, t_stat=1.0, ic_pos_ratio=0.55, yearly=[], decay={"1": 0.0, "5": 0.0, "20": 0.0},
                quantile_ret=[0.0] * 5, long_short=0.0, turnover_autocorr=0.5,
                max_ref_corr=max_ref_corr, max_ref_id=max_ref_id)


def checks_by_name(res) -> dict:
    return {c["name"]: c for c in res["checks"]}


# ---------------------------------------------------------------------
# 门槛文件
# ---------------------------------------------------------------------

def test_load_thresholds(thresholds):
    assert thresholds["train"]["ic_mean_min"] == 0.01
    assert thresholds["train"]["icir_min"] == 0.06
    assert thresholds["train"]["max_abs_ref_corr"] == 0.8      # 只和已入库因子比
    assert thresholds["valid"] == {"ic_mean_min": 0.01, "icir_min": 0.06}


def test_thresholds_sha256(tmp_path):
    h = ev.thresholds_sha256(THRESHOLDS_PATH)
    assert h == hashlib.sha256(THRESHOLDS_PATH.read_bytes()).hexdigest()
    assert len(h) == 64
    p = tmp_path / "t.yaml"
    p.write_bytes(THRESHOLDS_PATH.read_bytes())
    assert ev.thresholds_sha256(p) == h
    p.write_bytes(THRESHOLDS_PATH.read_bytes() + b"\n# x\n")
    assert ev.thresholds_sha256(p) != h


def test_load_thresholds_bad_file(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ev.load_thresholds(p)


# ---------------------------------------------------------------------
# split_dates 替身（集成后由线 A 提供）
# ---------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "valid"])
@pytest.mark.parametrize("h", HORIZONS)
def test_split_dates_fallback_no_cross(market, split, h):
    d = ev._split_dates_fallback(market, split, h)
    start, end = (pd.Timestamp(s) for s in SPLITS[split])
    assert len(d) > 0
    assert d.min() >= start and d.max() <= end
    pos = market.dates.get_indexer(d)
    tgt = pos + 1 + h
    assert (tgt < len(market.dates)).all()
    assert (market.dates[tgt] <= end).all()
    # 再往后一天就会跨段或越界
    nxt = pos.max() + 1
    assert nxt + 1 + h >= len(market.dates) or market.dates[nxt + 1 + h] > end or market.dates[nxt] > end


def test_evaluate_uses_split_dates(market, monkeypatch):
    """IC 日期必须来自 split_dates：替换成只给 30 天的替身，n_days 不能超过 30。"""
    calls = []

    def stub(m, split, h=MAIN_HORIZON):
        calls.append(h)
        return ev._split_dates_fallback(m, split, h)[:30]

    monkeypatch.setattr(ev, "_split_dates", stub)
    rep = ev.evaluate(stub_rev5(market), market, "train")
    assert 0 < rep["n_days"] <= 30
    assert set(calls) == set(HORIZONS)       # 每个 h 单独取


# ---------------------------------------------------------------------
# evaluate：IC 方向与数值
# ---------------------------------------------------------------------

def test_rev5_ic_positive(rev5_train, market):
    assert rev5_train["ic_mean"] > 0
    assert rev5_train["t_stat"] > 2
    rv = ev.evaluate(stub_rev5(market), market, "valid")
    assert rv["ic_mean"] > 0


def test_rev5_long_short_positive(rev5_train):
    assert rev5_train["long_short"] > 0
    assert rev5_train["quantile_ret"][-1] > rev5_train["quantile_ret"][0]


def test_negative_factor_flips_sign(market, rev5_train):
    neg = ev.evaluate(-stub_rev5(market), market, "train")
    assert neg["ic_mean"] < 0
    assert neg["ic_mean"] == pytest.approx(-rev5_train["ic_mean"], abs=1e-12)
    assert neg["icir"] == pytest.approx(-rev5_train["icir"], abs=1e-12)


def test_daily_ic_matches_scipy(market):
    f = stub_rev5(market)
    dates = ev._split_dates_fallback(market, "train", MAIN_HORIZON)
    uni = ev._universe(market)
    ic = ev._daily_ic(ev._align(f, market).where(uni), market, uni, MAIN_HORIZON, dates)
    fwd = market.fwd_ret[MAIN_HORIZON]
    for t in [dates[10], dates[50], dates[-1]]:
        x = f.loc[t].where(market.universe.loc[t])
        y = fwd.loc[t]
        ok = x.notna() & y.notna()
        expected = spearmanr(x[ok], y[ok]).statistic
        assert ic.loc[t] == pytest.approx(expected, abs=1e-10)


def test_stats_formulas(rev5_train, market):
    dates = ev._split_dates_fallback(market, "train", MAIN_HORIZON)
    uni = ev._universe(market)
    ic = ev._daily_ic(ev._align(stub_rev5(market), market).where(uni), market, uni, MAIN_HORIZON, dates).dropna()
    r = rev5_train
    assert r["n_days"] == len(ic)
    assert r["ic_mean"] == pytest.approx(ic.mean())
    assert r["ic_std"] == pytest.approx(ic.std(ddof=1))
    assert r["icir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
    assert r["t_stat"] == pytest.approx(ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)))
    assert r["ic_pos_ratio"] == pytest.approx((ic > 0).mean())
    assert r["decay"]["5"] == r["ic_mean"]
    assert sum(y["n_days"] for y in r["yearly"]) == r["n_days"]


def test_skip_days_with_too_few_stocks(market, rev5_train):
    f = stub_rev5(market).copy()
    dates = ev._split_dates_fallback(market, "train", MAIN_HORIZON)
    bad = dates[20:40]
    # 这些日子只留 MIN_STOCKS_PER_DAY - 1 只有效股票
    keep = f.loc[bad].notna().cumsum(axis=1) <= (MIN_STOCKS_PER_DAY - 1)
    f.loc[bad] = f.loc[bad].where(keep)
    assert (f.loc[bad].notna().sum(axis=1) == MIN_STOCKS_PER_DAY - 1).all()
    rep = ev.evaluate(f, market, "train")
    assert rep["n_days"] == rev5_train["n_days"] - len(bad)


def test_universe_and_inf_handling(market, rev5_train):
    f = stub_rev5(market).copy()
    # 股票池外放极端值，不应影响结果
    f = f.where(market.universe, 1e9)
    rep = ev.evaluate(f, market, "train")
    assert rep["ic_mean"] == pytest.approx(rev5_train["ic_mean"], abs=1e-12)
    # inf 视为 NaN
    g = stub_rev5(market).copy()
    g.iloc[100, 10] = np.inf
    rep2 = ev.evaluate(g, market, "train")
    assert math.isfinite(rep2["ic_mean"])


def test_turnover_autocorr(market):
    rng = np.random.default_rng(7)
    static = pd.DataFrame(np.tile(rng.normal(size=len(market.permnos)), (len(market.dates), 1)),
                          index=market.dates, columns=market.permnos).where(market.universe)
    assert ev.evaluate(static, market, "train")["turnover_autocorr"] == pytest.approx(1.0)
    noise = ev.evaluate(stub_noise(market), market, "train")["turnover_autocorr"]
    assert abs(noise) < 0.05


def test_empty_factor_gives_nan_and_fails(market, thresholds):
    f = pd.DataFrame(np.nan, index=market.dates, columns=market.permnos)
    rep = ev.evaluate(f, market, "train")
    assert rep["n_days"] == 0
    assert math.isnan(rep["ic_mean"])
    assert not ev.gate_train(rep, thresholds)["passed"]


# ---------------------------------------------------------------------
# Report 字段完整、类型正确
# ---------------------------------------------------------------------

def test_report_fields_and_types(market):
    refs = {"R_noise": rs.to_ranks(stub_noise(market), market)}
    rep = ev.evaluate(stub_rev5(market), market, "train", refs=refs)
    expected = set(ev.Report.__annotations__)
    assert set(rep) == expected
    assert rep["split"] == "train"
    assert type(rep["n_days"]) is int and rep["n_days"] > 0
    for k in ["coverage_mean", "ic_mean", "ic_std", "icir", "t_stat", "ic_pos_ratio",
              "long_short", "turnover_autocorr", "max_ref_corr"]:
        assert type(rep[k]) is float, k
        assert math.isfinite(rep[k]), k
    assert 0 < rep["coverage_mean"] <= len(market.permnos)
    assert 0 <= rep["ic_pos_ratio"] <= 1
    assert isinstance(rep["yearly"], list) and len(rep["yearly"]) >= 1
    for y in rep["yearly"]:
        assert set(y) == set(ev.YearStat.__annotations__)
        assert type(y["year"]) is int and type(y["n_days"]) is int
        for k in ["ic_mean", "icir", "ic_pos_ratio"]:
            assert type(y[k]) is float
    assert set(rep["decay"]) == {str(h) for h in HORIZONS}
    assert all(type(v) is float and math.isfinite(v) for v in rep["decay"].values())
    assert len(rep["quantile_ret"]) == N_QUANTILES
    assert all(type(v) is float and math.isfinite(v) for v in rep["quantile_ret"])
    assert rep["long_short"] == pytest.approx(rep["quantile_ret"][-1] - rep["quantile_ret"][0])
    assert rep["max_ref_id"] == "R_noise"


def test_refs_none_gives_none(rev5_train):
    assert rev5_train["max_ref_corr"] is None
    assert rev5_train["max_ref_id"] is None


def test_refs_only_in_train(market):
    refs = {"R": rs.to_ranks(stub_noise(market), market)}
    with pytest.raises(ValueError):
        ev.evaluate(stub_rev5(market), market, "valid", refs=refs)


def test_report_json_serializable(rev5_train):
    import json
    json.dumps(rev5_train, ensure_ascii=False)


# ---------------------------------------------------------------------
# 去重：相关性
# ---------------------------------------------------------------------

def test_max_ref_corr_identical_and_sign(market):
    rev = stub_rev5(market)
    refs = {"C01_rev5": rs.to_ranks(rev, market), "R_noise": rs.to_ranks(stub_noise(market), market)}
    same = ev.evaluate(rev, market, "train", refs=refs)
    assert same["max_ref_id"] == "C01_rev5"
    assert same["max_ref_corr"] == pytest.approx(1.0, abs=1e-6)
    neg = ev.evaluate(-rev, market, "train", refs=refs)
    assert neg["max_ref_id"] == "C01_rev5"
    assert neg["max_ref_corr"] == pytest.approx(-1.0, abs=1e-6)     # 保留符号
    # 单调变换不改变 Spearman
    mono = ev.evaluate(rev * 3 + 1, market, "train", refs=refs)
    assert mono["max_ref_corr"] == pytest.approx(1.0, abs=1e-6)


def test_max_ref_corr_low_for_unrelated(market):
    refs = {"R_noise": rs.to_ranks(stub_noise(market, seed=1), market)}
    rep = ev.evaluate(stub_rev5(market), market, "train", refs=refs)
    assert abs(rep["max_ref_corr"]) < 0.1


def test_ranked_corr_fast_path_matches_exact(market):
    """快速路径（掩码相同直接 Pearson）与逐行重新 rank 的精确 Spearman 一致，含掩码不同和有并列值的行。"""
    a_raw = stub_rev5(market).round(3)                 # 制造并列值
    b_raw = stub_noise(market) + 0.3 * stub_rev5(market).fillna(0)
    b_raw.iloc[200:230, :40] = np.nan                  # 这些行两者掩码不同
    a = rs.to_ranks(a_raw, market).to_numpy("float64")
    b = rs.to_ranks(b_raw, market).to_numpy("float64")
    fast = ev._ranked_corr(a, b, MIN_STOCKS_PER_DAY)
    exact, _ = ev._rowwise_spearman(a, b, MIN_STOCKS_PER_DAY)
    np.testing.assert_allclose(fast, exact, atol=1e-6, equal_nan=True)
    assert np.isfinite(fast).sum() > 50


def test_empty_refs_dict(market):
    rep = ev.evaluate(stub_rev5(market), market, "train", refs={})
    assert rep["max_ref_corr"] is None and rep["max_ref_id"] is None


# ---------------------------------------------------------------------
# 训练门
# ---------------------------------------------------------------------

def test_gate_train_pass(thr_corr):
    res = ev.gate_train(make_report(), thr_corr)
    assert res["passed"] is True
    c = checks_by_name(res)
    assert set(c) == {"ic_mean", "icir", "max_ref_corr"}
    for chk in res["checks"]:
        assert set(chk) == {"name", "value", "threshold", "passed"}
        assert type(chk["value"]) is float and type(chk["threshold"]) is float
        assert type(chk["passed"]) is bool
    assert c["max_ref_corr"]["threshold"] == 0.8


def test_gate_train_negative_factor_fails(thr_corr):
    res = ev.gate_train(make_report(ic_mean=-0.03, icir=-0.1, max_ref_corr=-0.2), thr_corr)
    assert res["passed"] is False                     # 不取绝对值：负 IC 不达标
    c = checks_by_name(res)
    assert c["ic_mean"]["value"] == pytest.approx(-0.03) and c["ic_mean"]["passed"] is False
    assert c["icir"]["value"] == pytest.approx(-0.1) and c["icir"]["passed"] is False
    assert c["max_ref_corr"]["value"] == pytest.approx(0.2)   # 相关性仍取绝对值


@pytest.mark.parametrize("ic,ok", [(0.01, True), (0.0099, False), (-0.01, False), (-0.05, False), (0.0, False)])
def test_gate_train_ic_boundary(thresholds, ic, ok):
    res = ev.gate_train(make_report(ic_mean=ic), thresholds)
    assert checks_by_name(res)["ic_mean"]["passed"] is ok
    assert res["passed"] is ok


@pytest.mark.parametrize("ir,ok", [(0.06, True), (0.0599, False), (-0.06, False), (-0.2, False)])
def test_gate_train_icir_boundary(thresholds, ir, ok):
    res = ev.gate_train(make_report(icir=ir), thresholds)
    assert checks_by_name(res)["icir"]["passed"] is ok
    assert res["passed"] is ok


@pytest.mark.parametrize("rc,ok", [(0.79, True), (0.8, False), (-0.8, False), (0.95, False), (-0.79, True)])
def test_gate_train_ref_corr_strict(thr_corr, rc, ok):
    res = ev.gate_train(make_report(max_ref_corr=rc), thr_corr)
    assert checks_by_name(res)["max_ref_corr"]["passed"] is ok
    assert res["passed"] is ok


def test_gate_train_missing_ref_passes(thr_corr):
    # 没有参照因子（因子库为空）时无可重复，去重检查视为通过
    res = ev.gate_train(make_report(max_ref_corr=None, max_ref_id=None), thr_corr)
    assert res["passed"] is True
    assert checks_by_name(res)["max_ref_corr"]["passed"] is True


def test_gate_train_nan_fails(thresholds):
    res = ev.gate_train(make_report(ic_mean=float("nan"), icir=float("nan")), thresholds)
    assert res["passed"] is False


def test_gate_train_end_to_end_corr_exceeds(market, thr_corr):
    rev = stub_rev5(market)
    refs = {"C01_rev5": rs.to_ranks(rev, market)}
    rep = ev.evaluate(rev, market, "train", refs=refs)
    res = ev.gate_train(rep, thr_corr)
    assert checks_by_name(res)["max_ref_corr"]["passed"] is False
    assert res["passed"] is False


def test_gate_train_end_to_end_pass(market, thresholds):
    refs = {"R_noise": rs.to_ranks(stub_noise(market), market)}
    rep = ev.evaluate(stub_rev5(market), market, "train", refs=refs)
    res = ev.gate_train(rep, thresholds)
    assert res["passed"] is True, res
    neg = ev.gate_train(ev.evaluate(-stub_rev5(market), market, "train", refs=refs), thresholds)
    assert neg["passed"] is False                     # 方向反了就不达标


# ---------------------------------------------------------------------
# 验证门
# ---------------------------------------------------------------------

def test_gate_valid_pass(thresholds):
    res = ev.gate_valid(make_report(ic_mean=0.025, icir=0.08), make_report(), thresholds)
    assert res["passed"] is True
    assert set(checks_by_name(res)) == {"ic_mean", "icir"}


def test_gate_valid_negative_fails(thresholds):
    res = ev.gate_valid(make_report(ic_mean=-0.03, icir=-0.1), make_report(ic_mean=-0.04, icir=-0.2), thresholds)
    assert res["passed"] is False


def test_gate_valid_opposite_sign_fails(thresholds):
    res = ev.gate_valid(make_report(ic_mean=-0.05, icir=-0.2), make_report(ic_mean=0.03), thresholds)
    c = checks_by_name(res)
    assert c["ic_mean"]["passed"] is False and c["icir"]["passed"] is False
    assert res["passed"] is False


@pytest.mark.parametrize("ic,ir,ok", [(0.01, 0.06, True), (0.0099, 0.1, False), (0.03, 0.059, False)])
def test_gate_valid_boundary(thresholds, ic, ir, ok):
    res = ev.gate_valid(make_report(ic_mean=ic, icir=ir), make_report(), thresholds)
    assert res["passed"] is ok




def test_gate_valid_end_to_end(market, thresholds):
    f = stub_rev5(market)
    tr = ev.evaluate(f, market, "train")
    va = ev.evaluate(f, market, "valid")
    assert va["max_ref_corr"] is None
    assert ev.gate_valid(va, tr, thresholds)["passed"] is True
    # 验证段方向反了（因子取负）就不通过
    va_neg = ev.evaluate(-f, market, "valid")
    assert ev.gate_valid(va_neg, tr, thresholds)["passed"] is False


# ---------------------------------------------------------------------
# refstore
# ---------------------------------------------------------------------

def test_to_ranks_shape_and_values(market):
    f = stub_rev5(market)
    r = rs.to_ranks(f, market)
    start, end = (pd.Timestamp(s) for s in SPLITS[REF_SPLIT])
    assert r.index.min() >= start and r.index.max() <= end
    assert len(r.index) == ((market.dates >= start) & (market.dates <= end)).sum()
    assert r.index.name == "date" and isinstance(r.index, pd.DatetimeIndex)
    assert r.columns.dtype == np.int64 and r.columns.name == "permno"
    assert (r.dtypes == np.float32).all()
    vals = r.to_numpy()
    fin = vals[~np.isnan(vals)]
    assert fin.min() > 0 and fin.max() <= 1.0
    # 股票池外为 NaN
    out_uni = ~market.universe.loc[r.index]
    assert r.where(out_uni).isna().all().all()
    # 与逐日 pct rank 一致
    t = r.index[-1]
    expected = f.loc[t].where(market.universe.loc[t]).rank(pct=True).astype("float32")
    pd.testing.assert_series_equal(r.loc[t], expected, check_names=False)


def test_save_load_roundtrip(market, tmp_path):
    f1, f2 = stub_rev5(market), stub_noise(market)
    rs.save_ref("C01_rev5", f1, market, cache_dir=tmp_path)
    rs.save_ref("F0001", f2, market, cache_dir=tmp_path)
    assert (tmp_path / "ref_values" / "C01_rev5.parquet").is_file()
    refs = rs.load_refs(cache_dir=tmp_path)
    assert list(refs) == ["C01_rev5", "F0001"]
    pd.testing.assert_frame_equal(refs["C01_rev5"], rs.to_ranks(f1, market))
    pd.testing.assert_frame_equal(refs["F0001"], rs.to_ranks(f2, market))
    assert refs["F0001"].columns.dtype == np.int64


def test_save_overwrites(market, tmp_path):
    rs.save_ref("X", stub_noise(market, 1), market, cache_dir=tmp_path)
    rs.save_ref("X", stub_noise(market, 2), market, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(rs.load_refs(tmp_path)["X"], rs.to_ranks(stub_noise(market, 2), market))


def test_load_refs_missing_dir(tmp_path):
    assert rs.load_refs(cache_dir=tmp_path / "nope") == {}


def test_save_ref_rejects_bad_id(market, tmp_path):
    with pytest.raises(ValueError):
        rs.save_ref("../evil", stub_noise(market), market, cache_dir=tmp_path)


def test_loaded_refs_work_in_evaluate(market, tmp_path):
    rs.save_ref("C01_rev5", stub_rev5(market), market, cache_dir=tmp_path)
    rep = ev.evaluate(stub_rev5(market), market, "train", refs=rs.load_refs(tmp_path))
    assert rep["max_ref_id"] == "C01_rev5"
    assert rep["max_ref_corr"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------
# 真实数据（默认不跑，集成时 python -m pytest -m real -q）
# ---------------------------------------------------------------------

@pytest.mark.real
def test_real_rev5_ic_positive():
    import yaml
    data = pytest.importorskip("src.data")
    expr = pytest.importorskip("src.expr")
    settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    m = data.load_market(settings["data_dir"], cache_dir=ROOT / settings.get("cache_dir", "cache"))
    for e in ["-ts_sum(ret, 5)", "-ts_sum(ret, 20)"]:
        rep = ev.evaluate(expr.compute(e, m), m, "train")
        assert rep["ic_mean"] > 0, e
        assert rep["n_days"] > 3000
        assert 400 <= rep["coverage_mean"] <= 520


def test_report_horizons(market):
    rep = ev.evaluate(stub_rev5(market), market, "train")
    assert set(rep["horizons"]) == {str(h) for h in HORIZONS}
    for h, v in rep["horizons"].items():
        assert set(v) == {"ic_mean", "icir", "ic_pos_ratio", "n_days"}
        assert v["ic_mean"] == rep["decay"][h]
    assert rep["horizons"]["5"]["icir"] == rep["icir"]
