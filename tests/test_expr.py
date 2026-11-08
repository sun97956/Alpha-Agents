"""线 B 单元测试：算子（src/operators.py）、表达式（src/expr.py）、经典因子库。

只用 contracts.make_fake_market() 生成的假数据；真实数据测试标记为 real，默认不跑。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import operators as ops
from src.contracts import (
    ARITH_FUNCS, CS_OPS, DELAY_WINDOWS, FIELDS, GROUP_OPS, TS_OPS, WINDOWS, Market,
    compute_fwd_ret, make_fake_market,
)
from src.expr import ExprError, compute, validate

ROOT = Path(__file__).resolve().parents[1]
CLASSIC_PATH = ROOT / "library" / "classic_factors.yaml"
CJK = re.compile(r"[一-鿿]")


@pytest.fixture(scope="module")
def market() -> Market:
    return make_fake_market()


def _small(values, index=None, columns=None) -> pd.DataFrame:
    a = np.asarray(values, dtype="float64")
    index = index if index is not None else pd.bdate_range("2016-01-04", periods=a.shape[0], name="date")
    columns = columns if columns is not None else pd.Index(
        np.arange(1, a.shape[1] + 1, dtype=np.int64), name="permno")
    return pd.DataFrame(a, index=index, columns=columns)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_freq=False, check_names=False, rtol=1e-9, atol=1e-12)


def _rank_ic(f: pd.DataFrame, fwd: pd.DataFrame, u: pd.DataFrame) -> float:
    """测试用的最小 Rank IC 替身（线 C 的 evaluate 不可依赖）。"""
    fr = f.where(u).rank(axis=1)
    rr = fwd.where(u & f.notna()).rank(axis=1)
    fr = fr.where(rr.notna())
    ics = pd.Series([fr.iloc[i].corr(rr.iloc[i]) for i in range(len(fr))], index=fr.index)
    return float(ics.dropna().mean())


# =====================================================================
# 时序算子
# =====================================================================

class TestTimeSeries:
    x = _small([[1, 10], [2, np.nan], [3, 30], [4, 40], [5, 50], [6, 60]])

    def test_ts_mean_incomplete_window_is_nan(self):
        r = ops.ts_mean(self.x, 3)
        assert r.iloc[:2].isna().all().all()                 # 窗口不完整
        assert r.iloc[2, 0] == pytest.approx(2.0)
        assert r.iloc[5, 0] == pytest.approx(5.0)
        # 第 2 列第 1 行缺失：包含它的窗口全部为 NaN，不跳过、不填补
        assert r.iloc[2:4, 1].isna().all()
        assert r.iloc[4, 1] == pytest.approx(40.0)

    def test_ts_sum_std_max_min(self):
        assert ops.ts_sum(self.x, 3).iloc[5, 0] == pytest.approx(15.0)
        assert ops.ts_std(self.x, 3).iloc[5, 0] == pytest.approx(1.0)       # ddof=1
        assert ops.ts_max(self.x, 3).iloc[5, 1] == pytest.approx(60.0)
        assert ops.ts_min(self.x, 3).iloc[5, 1] == pytest.approx(40.0)
        for f in (ops.ts_sum, ops.ts_std, ops.ts_max, ops.ts_min):
            r = f(self.x, 3)
            assert r.iloc[:2].isna().all().all()
            assert r.iloc[2:4, 1].isna().all()

    def test_ts_skew(self):
        x = _small([[1], [2], [10], [np.nan], [1], [2], [3], [4]])
        r = ops.ts_skew(x, 3)
        assert r.iloc[:2].isna().all().all()
        expected = pd.Series([1.0, 2.0, 10.0]).skew()
        assert r.iloc[2, 0] == pytest.approx(expected)
        assert r.iloc[3:6, 0].isna().all()                   # 窗口含 NaN
        assert r.iloc[7, 0] == pytest.approx(0.0, abs=1e-9)  # 对称

    def test_ts_delay_and_delta(self):
        r = ops.ts_delay(self.x, 1)
        assert np.isnan(r.iloc[0, 0]) and r.iloc[1, 0] == 1.0
        assert np.isnan(r.iloc[2, 1])                        # 前一日缺失
        d = ops.ts_delta(self.x, 2)
        assert d.iloc[:2].isna().all().all()
        assert d.iloc[3, 0] == pytest.approx(2.0)
        assert np.isnan(d.iloc[3, 1])                        # 2 日前缺失

    def test_ts_rank(self):
        x = _small([[3, 1], [1, 1], [2, 1], [5, np.nan], [4, 2], [4, 3]])
        r = ops.ts_rank(x, 3)
        assert r.iloc[:2].isna().all().all()
        assert r.iloc[2, 0] == pytest.approx(0.5)            # [3,1,2] 中 2 居中
        assert r.iloc[3, 0] == pytest.approx(1.0)            # 最大
        assert r.iloc[4, 0] == pytest.approx(0.5)            # [2,5,4]
        assert r.iloc[5, 0] == pytest.approx(0.25)           # [5,4,4] 当日 4 与另一 4 并列取平均
        assert r.iloc[2, 1] == pytest.approx(0.5)            # 全部相等
        assert r.iloc[3:6, 1].isna().all()                   # 窗口含 NaN
        assert ((r.stack().dropna() >= 0) & (r.stack().dropna() <= 1)).all()

    def test_ts_rank_matches_pandas_and_chunks(self, market, monkeypatch):
        c = market.fields["close"]
        ref = c.rolling(20, min_periods=20).apply(
            lambda w: ((w < w[-1]).sum() + 0.5 * ((w == w[-1]).sum() - 1)) / 19, raw=True)
        monkeypatch.setattr(ops, "_TS_RANK_CHUNK_ELEMS", 5000)   # 强制多块
        _same(ops.ts_rank(c, 20), ref)

    def test_ts_corr(self):
        x = _small([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        y = _small([[2, 5], [4, 5], [6, 5], [8, np.nan], [9, 5], [13, 5]])
        r = ops.ts_corr(x, y, 3)
        assert r.iloc[:2].isna().all().all()
        assert r.iloc[2, 0] == pytest.approx(1.0)
        assert r.iloc[2, 1] != r.iloc[2, 1]                  # y 为常数 -> NaN
        assert r.iloc[3:6, 1].isna().all()                   # 窗口含 NaN 或常数
        exp = np.corrcoef([4, 5, 6], [8, 9, 13])[0, 1]
        assert r.iloc[5, 0] == pytest.approx(exp)

    def test_ts_corr_matches_numpy_on_fake(self, market):
        c, v = market.fields["close"], market.fields["volume"]
        r = ops.ts_corr(c, v, 20)
        j, t = 7, 150
        exp = np.corrcoef(c.iloc[t - 19:t + 1, j], v.iloc[t - 19:t + 1, j])[0, 1]
        assert r.iloc[t, j] == pytest.approx(exp, rel=1e-7)
        assert (r.abs().stack().dropna() <= 1).all()

    @pytest.mark.parametrize("op", ["ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min",
                                    "ts_skew", "ts_rank", "ts_delay", "ts_delta"])
    def test_no_lookahead(self, market, op):
        """改动未来的数据，不影响过去的因子值。"""
        c = market.fields["close"]
        t = 200
        c2 = c.copy()
        c2.iloc[t + 1:] = c2.iloc[t + 1:] * 3.0 + 7.0
        f = getattr(ops, op)
        _same(f(c, 20).iloc[:t + 1], f(c2, 20).iloc[:t + 1])

    def test_ts_corr_no_lookahead(self, market):
        c, v = market.fields["close"], market.fields["volume"]
        v2 = v.copy()
        v2.iloc[201:] *= 5
        _same(ops.ts_corr(c, v, 20).iloc[:201], ops.ts_corr(c, v2, 20).iloc[:201])


# =====================================================================
# 截面算子
# =====================================================================

class TestCrossSection:
    x = _small([[1, 2, 3, 100], [4, np.nan, 6, 7]])
    u = pd.DataFrame([[True, True, True, False], [True, True, True, True]],
                     index=x.index, columns=x.columns)

    def test_cs_rank_only_in_universe(self):
        r = ops.cs_rank(self.x, self.u)
        assert np.isnan(r.iloc[0, 3])                        # 不在股票池
        assert r.iloc[0, :3].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
        assert np.isnan(r.iloc[1, 1])                        # 缺失值保持缺失
        assert r.iloc[1, [0, 2, 3]].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_cs_zscore_only_in_universe(self):
        r = ops.cs_zscore(self.x, self.u)
        assert np.isnan(r.iloc[0, 3])
        assert r.iloc[0, :3].tolist() == pytest.approx([-1.0, 0.0, 1.0])   # ddof=1
        assert np.isnan(r.iloc[1, 1])
        row = r.iloc[1].dropna()
        assert row.mean() == pytest.approx(0.0, abs=1e-12)
        assert row.std(ddof=1) == pytest.approx(1.0)

    def test_cs_zscore_constant_row_is_nan(self):
        x = _small([[5, 5, 5]])
        u = pd.DataFrame(True, index=x.index, columns=x.columns)
        assert ops.cs_zscore(x, u).isna().all().all()

    def test_group_neutralize(self):
        x = _small([[1, 3, 10, 20, 100, 7], [2, 4, 6, np.nan, 8, 9]])
        g = _small([[10, 10, 20, 20, 20, np.nan], [10, 10, 10, 20, 20, 20]])
        u = pd.DataFrame([[True, True, True, True, False, True]] + [[True] * 6],
                         index=x.index, columns=x.columns)
        r = ops.group_neutralize(x, g, u)
        # 第 1 天：组 10 = {1,3} 均值 2；组 20 只有 {10,20}（100 不在股票池），均值 15
        assert r.iloc[0, :4].tolist() == pytest.approx([-1, 1, -5, 5])
        assert np.isnan(r.iloc[0, 4])                        # 不在股票池
        assert np.isnan(r.iloc[0, 5])                        # 行业缺失
        # 第 2 天：组 10 = {2,4,6} 均值 4；组 20 = {8,9}（缺失的不参与）
        assert r.iloc[1, [0, 1, 2]].tolist() == pytest.approx([-2, 0, 2])
        assert np.isnan(r.iloc[1, 3])
        assert r.iloc[1, [4, 5]].tolist() == pytest.approx([-0.5, 0.5])

    def test_group_neutralize_uses_point_in_time_sector(self, market):
        """第 5 只股票中途换行业，前后分别减去不同行业的均值。"""
        x = market.fields["close"]
        r = ops.group_neutralize(x, market.sector, market.universe)
        j = 4
        for t in (10, len(x) - 10):
            code = market.sector.iloc[t, j]
            peers = (market.sector.iloc[t] == code) & market.universe.iloc[t] & x.iloc[t].notna()
            assert r.iloc[t, j] == pytest.approx(x.iloc[t, j] - x.iloc[t][peers].mean())
        assert market.sector.iloc[10, j] != market.sector.iloc[-10, j]

    def test_cs_ops_ignore_out_of_universe_values(self, market):
        """股票池外的值改成极端值，不影响池内结果。"""
        x = market.fields["close"]
        out = ~market.universe
        x2 = x.mask(out, 1e12)
        for f in (lambda v: ops.cs_rank(v, market.universe),
                  lambda v: ops.cs_zscore(v, market.universe),
                  lambda v: ops.group_neutralize(v, market.sector, market.universe)):
            _same(f(x), f(x2))


# =====================================================================
# 算术
# =====================================================================

class TestArith:
    def test_log_nonpositive_is_nan(self):
        r = ops.log(_small([[1, 0, -1, np.nan, np.e]]))
        assert r.iloc[0, 0] == 0.0
        assert r.iloc[0, 1:4].isna().all()
        assert r.iloc[0, 4] == pytest.approx(1.0)

    def test_div_by_zero_is_nan(self):
        a = _small([[1, 0, -1, 2]])
        b = _small([[0, 0, 0, 4]])
        r = ops.safe_div(a, b)
        assert r.iloc[0, :3].isna().all() and r.iloc[0, 3] == 0.5
        assert ops.safe_div(a, 0).isna().all().all()
        assert not np.isinf(ops.safe_div(a, b).to_numpy()).any()

    def test_abs_sign(self):
        x = _small([[-2, 0, 3, np.nan]])
        assert ops.abs_(x).iloc[0, :3].tolist() == [2, 0, 3]
        assert ops.sign(x).iloc[0, :3].tolist() == [-1, 0, 1]
        assert np.isnan(ops.abs_(x).iloc[0, 3]) and np.isnan(ops.sign(x).iloc[0, 3])


# =====================================================================
# validate
# =====================================================================

VALID = [
    "close",
    "-ts_sum(ret, 5)",
    "ts_delay(close, 1)",
    "ts_delta(volume, 1)",
    "ts_rank(close, 250)",
    "ts_corr(close, volume, 60)",
    "cs_rank(ts_mean(ret, 10))",
    "cs_zscore(-log(mktcap))",
    "group_neutralize(ret, sector)",
    "sign(ts_delta(close, 5)) * abs(ret)",
    "close * volume / (mktcap * 1000)",
    "(high - low) / close + 0.5",
    "ts_skew(ret, 120) * -1",
    "1 / close",
    "close * (2 * 3)",
]


@pytest.mark.parametrize("expr", VALID)
def test_validate_accepts(expr):
    validate(expr)


def test_validate_accepts_every_op_and_window():
    for op, n in TS_OPS.items():
        windows = DELAY_WINDOWS if op in ("ts_delay", "ts_delta") else WINDOWS
        for d in windows:
            validate(f"{op}({', '.join(['close', 'volume'][:n])}, {d})")
    for op in list(CS_OPS) + list(ARITH_FUNCS):
        validate(f"{op}(close)")
    for op in GROUP_OPS:
        validate(f"{op}(close, sector)")
    for f in FIELDS:
        validate(f)


INVALID = [
    # 非白名单字段 / 标识符
    ("vwap", "字段"),
    ("ts_mean(returns, 5)", "字段"),
    ("industry + close", "group_neutralize"),
    ("sector", "group_neutralize"),
    ("sector * close", "group_neutralize"),
    ("ts_mean", "算子"),
    # 非白名单算子
    ("ts_foo(close, 5)", "算子"),
    ("exp(close)", "算子"),
    ("pow(close, 2)", "算子"),
    ("__import__('os')", "算子"),
    ("np.log(close)", "调用"),
    ("close.shift(1)", "调用"),
    # 窗口
    ("ts_mean(close, 7)", "窗口"),
    ("ts_mean(close, 1)", "ts_delay"),
    ("ts_sum(ret, 1)", "ts_delay"),
    ("ts_rank(close, 1)", "ts_delay"),
    ("ts_corr(close, volume, 1)", "ts_delay"),
    ("ts_delay(close, 4)", "窗口"),
    ("ts_delta(close, 0)", "窗口"),
    ("ts_delay(close, -1)", "窗口"),
    ("ts_mean(close, 5.0)", "窗口"),
    ("ts_mean(close, True)", "窗口"),
    ("ts_mean(close, close)", "窗口"),
    ("ts_mean(close, 2 + 3)", "窗口"),
    ("ts_mean(close, 500)", "窗口"),
    # 参数个数 / 形式
    ("ts_mean(close)", "参数"),
    ("ts_corr(close, 20)", "参数"),
    ("cs_rank(close, 5)", "参数"),
    ("log(close, 2)", "参数"),
    ("group_neutralize(close)", "参数"),
    ("group_neutralize(close, country)", "sector"),
    ("group_neutralize(close, close)", "sector"),
    ("ts_mean(x=close, d=5)", "关键字"),
    ("ts_mean(5, 5)", "字段"),
    ("log(10)", "字段"),
    # 非法语法
    ("", "空"),
    ("   ", "空"),
    ("close +", "语法"),
    ("ts_mean(close, 5", "语法"),
    ("close ** 2", "运算符"),
    ("close % 2", "运算符"),
    ("close // 2", "运算符"),
    ("+close", "取负"),
    ("not close", "取负"),
    ("close == 1", "比较"),
    ("close if ret else open", "条件"),
    ("close[0]", "下标"),
    ("lambda: close", "lambda"),
    ("[close]", "列表"),
    ("'close'", "常量"),
    ("close; ret", "语法"),
    ("a = close", "语法"),
    ("1 + 2", "常数"),
    ("1e999 * close", "有限"),
    ("−ts_sum(ret, 5)", "全角"),
    ("close × volume", "全角"),
]


@pytest.mark.parametrize("expr,keyword", INVALID)
def test_validate_rejects_with_chinese_message(expr, keyword):
    with pytest.raises(ExprError) as ei:
        validate(expr)
    msg = str(ei.value)
    assert CJK.search(msg), f"错误信息应为中文：{msg}"
    assert keyword in msg, f"错误信息应提到“{keyword}”：{msg}"


def test_validate_rejects_non_string():
    with pytest.raises(ExprError):
        validate(None)  # type: ignore[arg-type]


def test_validate_rejects_too_deep():
    with pytest.raises(ExprError):
        validate("abs(" * 100 + "close" + ")" * 100)


def test_expr_error_is_value_error():
    assert issubclass(ExprError, ValueError)


# =====================================================================
# compute
# =====================================================================

def test_compute_shape_dtype_and_universe(market):
    f = compute("-ts_sum(ret, 5)", market)
    assert f.shape == market.fields["close"].shape
    assert f.index.equals(market.dates) and f.columns.equals(market.permnos)
    assert f.columns.dtype == np.int64
    assert (f.dtypes == "float64").all()
    assert f.where(~market.universe).isna().all().all()     # 股票池外全部 NaN
    assert not np.isinf(f.to_numpy()).any()


def test_compute_rejects_invalid(market):
    with pytest.raises(ExprError):
        compute("ts_mean(close, 7)", market)


def test_compute_matches_operators(market):
    m = market
    u = m.universe
    c, v, r = m.fields["close"], m.fields["volume"], m.fields["ret"]
    _same(compute("-ts_sum(ret, 5)", m), (-r.rolling(5, min_periods=5).sum()).where(u))
    _same(compute("ts_corr(close, volume, 20)", m), ops.ts_corr(c, v, 20).where(u))
    _same(compute("close / ts_delay(close, 1) - 1", m), r.where(u))
    _same(compute("cs_rank(close)", m), ops.cs_rank(c, u))


def test_compute_does_not_mutate_market(market):
    before = {k: v.copy() for k, v in market.fields.items()}
    compute("cs_zscore(group_neutralize(-ts_sum(ret, 20), sector)) / 0", market)
    for k in before:
        _same(market.fields[k], before[k])


def test_compute_div_zero_and_log_nonpositive(market):
    assert compute("close / 0", market).isna().all().all()
    f = compute("log(ret)", market)                          # 负收益的 log 为 NaN
    neg = (market.fields["ret"] <= 0) & market.universe
    assert f[neg].isna().all().all()
    assert not np.isinf(compute("1 / (ret - ret)", market).to_numpy()).any()


def test_compute_ts_uses_history_outside_universe():
    """时序算子在完整历史上算：股票暂时不在池内的日子，数据仍参与之后的窗口。"""
    m = make_fake_market(n_days=120, n_stocks=120, seed=1)
    u = m.universe.copy()
    j = 10
    u.iloc[50:70, j] = False                                 # 暂时调出
    m2 = Market(fields=m.fields, universe=u, sector=m.sector, fwd_ret=m.fwd_ret, end=m.end)
    f = compute("ts_mean(close, 20)", m2)
    assert f.iloc[50:70, j].isna().all()                     # 池外为 NaN
    expected = m.fields["close"].iloc[51:71, j].mean()       # 第 70 天的窗口全部来自池外日期
    assert f.iloc[70, j] == pytest.approx(expected)


def test_compute_late_listing_and_delisting(market):
    f = compute("ts_mean(close, 20)", market)
    n = len(market.dates)
    first = n // 3                                           # 第 1 只晚上市
    assert f.iloc[: first + 19, 0].isna().all()              # 上市后前 19 天窗口不完整
    assert f.iloc[first + 19:, 0].notna().all()
    assert f.iloc[2 * n // 3:, 1].isna().all()               # 第 2 只退市后无值
    g = compute("ts_mean(open, 5)", market)                  # open 第 10 行第 3 只缺失
    assert g.iloc[10:15, 2].isna().all() and g.iloc[15, 2] == g.iloc[15, 2]


def test_reversal_signal_on_fake(market):
    """假数据埋了短期反转：-ts_sum(ret, 5) 的 Rank IC 应为正。"""
    f = compute("-ts_sum(ret, 5)", market)
    fwd = compute_fwd_ret(market.fields["close"], 1)
    assert _rank_ic(f, fwd, market.universe) > 0.02


# =====================================================================
# 经典因子库
# =====================================================================

PRD_CLASSICS = [
    "-ts_sum(ret, 5)",
    "-ts_sum(ret, 20)",
    "ts_delay(close, 20) / ts_delay(close, 250) - 1",
    "ts_delay(close, 20) / ts_delay(close, 120) - 1",
    "-ts_std(ret, 20)",
    "-ts_std(ret, 250)",
    "-log(mktcap)",
    "-ts_mean(close * volume / (mktcap * 1000), 20)",
    "ts_mean(close * volume / (mktcap * 1000), 5) / ts_mean(close * volume / (mktcap * 1000), 60)",
    "ts_mean(volume, 5) / ts_mean(volume, 60)",
    "ts_mean(abs(ret) / (close * volume), 20)",
    "-ts_max(ret, 20)",
    "close / ts_max(high, 250)",
    "-ts_corr(close, volume, 20)",
    "-ts_mean((high - low) / close, 20)",
    "ts_mean(open / ts_delay(close, 1) - 1, 20)",
    "ts_mean(close / open - 1, 20)",
    "group_neutralize(-ts_sum(ret, 20), sector)",
    "group_neutralize(ts_delay(close, 20) / ts_delay(close, 250) - 1, sector)",
    "-ts_skew(ret, 60)",
]


@pytest.fixture(scope="module")
def classics() -> list[dict]:
    with open(CLASSIC_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def test_classics_structure(classics):
    assert isinstance(classics, list) and len(classics) == 20
    ids = [c["id"] for c in classics]
    assert len(set(ids)) == 20
    for i, c in enumerate(classics, 1):
        assert set(c) == {"id", "name", "expression", "logic"}
        assert all(isinstance(c[k], str) and c[k].strip() for k in c)
        assert c["id"].startswith(f"C{i:02d}_")


def test_classics_match_prd(classics):
    assert [_norm(c["expression"]) for c in classics] == [_norm(e) for e in PRD_CLASSICS]


def test_classics_all_validate(classics):
    for c in classics:
        validate(c["expression"])


def test_classics_compute_on_fake(classics):
    m = make_fake_market(n_days=320)                         # 够 250 日窗口留出几十天
    for c in classics:
        f = compute(c["expression"], m)
        assert f.shape == m.fields["close"].shape, c["id"]
        assert not np.isinf(f.to_numpy()).any(), c["id"]
        assert f.where(~m.universe).isna().all().all(), c["id"]
        assert f.iloc[-1].notna().sum() >= 100, c["id"]


def test_classic_reversal_positive_ic(classics):
    m = make_fake_market()
    rev5 = next(c for c in classics if c["id"] == "C01_rev5")
    f = compute(rev5["expression"], m)
    assert _rank_ic(f, m.fwd_ret[5], m.universe) > 0


# =====================================================================
# 真实数据（默认不跑；集成时 python -m pytest -m real -q）
# =====================================================================

@pytest.mark.real
def test_real_classics_compute(classics):
    data_mod = pytest.importorskip("src.data")
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    m = data_mod.load_market(settings["data_dir"], cache_dir=ROOT / settings.get("cache_dir", "cache"))
    assert m.end <= pd.Timestamp("2021-12-31")
    for c in classics:
        f = compute(c["expression"], m)
        assert f.shape == m.fields["close"].shape, c["id"]
        assert (f.dtypes == "float64").all(), c["id"]
        assert not np.isinf(f.to_numpy()).any(), c["id"]
        assert f.where(~m.universe).isna().all().all(), c["id"]
        cov = f.notna().sum(axis=1)
        assert cov.iloc[-250:].mean() > 400, c["id"]       # 股票池约 500


# ---------------------------------------------------------------------
# 扩充算子
# ---------------------------------------------------------------------

def test_new_operators_validate():
    for e in ["where(close > ts_mean(close, 20), ret, -ret)",
              "where(volume >= ts_mean(volume, 20), 1, -1) * ret",
              "decay_linear(ret, 10)", "ts_argmax(close, 10) - ts_argmin(close, 10)",
              "ts_cov(cs_rank(close), cs_rank(volume), 5)", "power(ret, 2)",
              "group_neutralize(ret, industry)", "group_neutralize(ret, subindustry)",
              "ts_delta(close, 2)", "ts_mean(close, 3)"]:
        validate(e)
    for bad in ["power(ret, close)", "where(1, 2, 3)", "1 < 2", "close < open < high", "where(close, ret)"]:
        with pytest.raises(ExprError):
            validate(bad)


def test_decay_linear_and_argmax_values():
    idx = pd.date_range("2020-01-01", periods=5)
    x = pd.DataFrame({1: [1.0, 3.0, 2.0, 5.0, 4.0]}, index=idx)
    d = ops.decay_linear(x, 3)
    assert np.isnan(d.iloc[1, 0])
    assert d.iloc[2, 0] == pytest.approx((1 * 1 + 3 * 2 + 2 * 3) / 6)
    am = ops.ts_argmax(x, 3)
    assert am.iloc[2, 0] == 1 and am.iloc[3, 0] == 0 and am.iloc[4, 0] == 1
    an = ops.ts_argmin(x, 3)
    assert an.iloc[2, 0] == 2 and an.iloc[4, 0] == 2


def test_where_compare_power_cov(market):
    m = market
    c = m.fields["close"]
    w = compute("where(close > ts_mean(close, 20), 1, -1)", m)
    u = m.universe
    exp = np.where(c > c.rolling(20, min_periods=20).mean(), 1.0, -1.0)
    ok = u & c.rolling(20, min_periods=20).mean().notna()
    assert np.allclose(w.where(ok).to_numpy()[ok.to_numpy()], exp[ok.to_numpy()])
    p = ops.power(pd.DataFrame({1: [-2.0, 3.0]}), 2)
    assert list(p[1]) == [-4.0, 9.0]
    cv = ops.ts_cov(c, m.fields["volume"], 5)
    ref = c.rolling(5, min_periods=5).cov(m.fields["volume"])
    assert np.allclose(cv.to_numpy(), ref.to_numpy(), equal_nan=True)


def test_group_neutralize_industry(market):
    g = compute("group_neutralize(ret, industry)", market)
    row = g.iloc[100]
    ind = market.industry.iloc[100]
    for code in ind[row.notna()].unique():
        assert abs(row[ind == code].mean()) < 1e-9
