"""
线 A 数据层测试。
- 单元测试：只用 contracts.make_fake_market() 生成的假数据。先把假 Market 还原成"原始文件"
  （panel.parquet + 成分表 + 证券信息表 + ticker 映射），再用 load_market 读回来比对。
- 真实数据测试：@pytest.mark.real，默认不跑（python -m pytest -m real -q）。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src.contracts import (
    FIELDS, HORIZONS, MAIN_HORIZON, MIN_STOCKS_PER_DAY, RUN_DATA_END, SPLITS,
    Market, compute_fwd_ret, make_fake_market,
)

N_DAYS, N_STOCKS = 300, 150


# =====================================================================
# 假原始文件
# =====================================================================

def _ymd(ts) -> int:
    return int(pd.Timestamp(ts).strftime("%Y%m%d"))


def write_fake_raw(fake: Market, data_dir: Path, panel_in_parent: bool = False) -> Path:
    """把假 Market 写成与 PRD 6.1 同名的原始文件。返回 data_dir。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    dates, permnos = fake.dates, fake.permnos
    n = len(dates)

    # panel：长表，只写"有记录"的行（close 非空）；open 的个别缺失保留为 NaN
    exists = fake.fields["close"].notna().to_numpy()
    ii, jj = np.nonzero(exists)
    panel = pd.DataFrame({"date": dates.values[ii], "permno": permnos.values[jj]})
    for f in ("open", "high", "low", "close", "volume", "mktcap"):
        panel[f] = fake.fields[f].to_numpy()[ii, jj]
    panel["ticker"] = "s" + panel["permno"].astype(str)
    panel = panel[["date", "permno", "ticker", "open", "high", "low", "close", "volume", "mktcap"]]
    panel = panel.sample(frac=1.0, random_state=0)        # 乱序，检验不依赖原始顺序
    panel_dir = data_dir.parent if panel_in_parent else data_dir
    panel.to_parquet(panel_dir / D.PANEL_FILE, index=False)

    # 成分表：全部从第一天起；第 4 只（索引 3）在 n//2 前一天退出；外加一只 panel 里没有的股票
    rows = [{"permno": int(p), "start": _ymd("1995-01-01"), "ending": ""} for p in permnos]
    rows[3]["ending"] = _ymd(dates[n // 2 - 1])
    rows.append({"permno": 99999, "start": _ymd("1995-01-01"), "ending": ""})
    pd.DataFrame(rows).to_csv(data_dir / D.MEMBERSHIP_FILE, index=False, encoding="utf-8")

    # 证券信息表：每只一行；第 5 只（索引 4）在 n//2 换行业
    sec0 = fake.sector.iloc[0]
    srows = []
    for j, p in enumerate(permnos):
        if j == 4:
            srows.append({"permno": int(p), "gsector": sec0.iloc[j], "indfrom": _ymd("1990-01-01"),
                          "indthru": _ymd(dates[n // 2 - 1])})
            srows.append({"permno": int(p), "gsector": fake.sector.iloc[-1, j],
                          "indfrom": _ymd(dates[n // 2]), "indthru": ""})
        else:
            srows.append({"permno": int(p), "gsector": sec0.iloc[j], "indfrom": _ymd("1990-01-01"),
                          "indthru": ""})
    pd.DataFrame(srows).to_csv(data_dir / D.SECURITY_FILE, index=False, encoding="utf-8")

    pd.DataFrame({"permno": [int(p) for p in permnos],
                  "ticker": [f"T{p}" for p in permnos]}).to_csv(
        data_dir / D.TICKER_FILE, index=False, encoding="utf-8")
    return data_dir


@pytest.fixture(scope="module")
def fake() -> Market:
    return make_fake_market(N_DAYS, N_STOCKS)


@pytest.fixture()
def raw_dir(fake, tmp_path) -> Path:
    return write_fake_raw(fake, tmp_path / "data")


@pytest.fixture()
def loaded(raw_dir, tmp_path) -> Market:
    return D.load_market(raw_dir, cache_dir=tmp_path / "cache")          # 默认 end=RUN_DATA_END


# =====================================================================
# load_market：形状、类型、内容
# =====================================================================

def test_shape_and_dtypes(loaded):
    m = loaded
    assert isinstance(m.dates, pd.DatetimeIndex) and m.dates.name == "date"
    assert m.dates.is_monotonic_increasing and m.dates.is_unique
    assert m.permnos.dtype == np.int64 and m.permnos.name == "permno"
    assert m.permnos.is_monotonic_increasing
    assert set(m.fields) == set(FIELDS)
    for f, df in m.fields.items():
        assert df.index.equals(m.dates) and df.columns.equals(m.permnos), f
        assert (df.dtypes == np.float64).all(), f
    assert (m.universe.dtypes == bool).all()
    assert m.universe.index.equals(m.dates) and m.universe.columns.equals(m.permnos)
    assert (m.sector.dtypes == np.float64).all()
    assert m.sector.index.equals(m.dates) and m.sector.columns.equals(m.permnos)
    assert set(m.fwd_ret) == set(HORIZONS)
    for h, df in m.fwd_ret.items():
        assert df.index.equals(m.dates) and df.columns.equals(m.permnos)
    assert isinstance(m.end, pd.Timestamp) and m.end == m.dates[-1]


def test_roundtrip_matches_fake(loaded, fake):
    m = loaded
    assert m.dates.equals(fake.dates)
    assert m.permnos.equals(fake.permnos)          # 成分表里多出的 99999 不进列
    for f in FIELDS:
        pd.testing.assert_frame_equal(m.fields[f], fake.fields[f], check_freq=False)
    pd.testing.assert_frame_equal(m.universe, fake.universe, check_freq=False)
    pd.testing.assert_frame_equal(m.sector, fake.sector, check_freq=False)
    for h in HORIZONS:
        pd.testing.assert_frame_equal(m.fwd_ret[h], fake.fwd_ret[h], check_freq=False)


def test_no_fill_of_missing(loaded, fake):
    m = loaded
    assert np.isnan(m.fields["open"].iloc[10, 2])                 # 个别缺失不填补
    assert m.fields["close"].iloc[: N_DAYS // 3, 0].isna().all()   # 晚上市
    assert m.fields["close"].iloc[2 * N_DAYS // 3:, 1].isna().all()  # 提前退市
    assert np.isnan(m.fields["ret"].iloc[N_DAYS // 3, 0])          # 上市首日 ret 为 NaN
    assert not np.isinf(m.fields["ret"].to_numpy()).any()


def test_universe_rules(loaded):
    m = loaded
    # 股票池只在有 close 的位置为 True
    assert not (m.universe & m.fields["close"].isna()).to_numpy().any()
    # 第 4 只在 n//2 被调出
    assert m.universe.iloc[: N_DAYS // 2, 3].all()
    assert not m.universe.iloc[N_DAYS // 2:, 3].any()
    # 晚上市、提前退市
    assert not m.universe.iloc[: N_DAYS // 3, 0].any()
    assert not m.universe.iloc[2 * N_DAYS // 3:, 1].any()
    assert (m.universe.sum(axis=1) >= MIN_STOCKS_PER_DAY).all()


def test_sector_point_in_time(loaded, fake):
    m = loaded
    col = m.sector.iloc[:, 4]
    assert col.iloc[: N_DAYS // 2].nunique() == 1
    assert col.iloc[N_DAYS // 2:].nunique() == 1
    assert col.iloc[0] != col.iloc[-1]
    assert col.iloc[0] == fake.sector.iloc[0, 4] and col.iloc[-1] == fake.sector.iloc[-1, 4]


def test_ret_and_labels_use_contracts(loaded):
    m = loaded
    c = m.fields["close"]
    for h in HORIZONS:
        pd.testing.assert_frame_equal(m.fwd_ret[h], compute_fwd_ret(c, h), check_freq=False)
    # 手算一个点：fwd[t] = close[t+1+h] / close[t+1] - 1
    t, j, h = 50, 7, MAIN_HORIZON
    expect = c.iloc[t + 1 + h, j] / c.iloc[t + 1, j] - 1
    assert m.fwd_ret[h].iloc[t, j] == pytest.approx(expect)
    expect_ret = c.iloc[t, j] / c.iloc[t - 1, j] - 1
    assert m.fields["ret"].iloc[t, j] == pytest.approx(expect_ret)


# =====================================================================
# end 截断
# =====================================================================

def test_end_truncation_no_future(raw_dir, tmp_path, fake):
    end = "2016-12-31"
    m = D.load_market(raw_dir, end=end, cache_dir=tmp_path / "cache")
    assert m.dates.max() <= pd.Timestamp(end)
    assert m.end <= pd.Timestamp(end)
    assert m.dates.equals(fake.dates[fake.dates <= end])
    # 截止日附近的标签必然是 NaN（不能用到截止日之后的价格）
    for h in HORIZONS:
        assert m.fwd_ret[h].iloc[-(1 + h):].isna().all().all()
    # 截止日之前的标签与全量数据一致
    full = fake.fwd_ret[MAIN_HORIZON].loc[m.dates[: -(1 + MAIN_HORIZON)]]
    pd.testing.assert_frame_equal(m.fwd_ret[MAIN_HORIZON].iloc[: -(1 + MAIN_HORIZON)], full,
                                  check_freq=False)


def test_end_none_loads_everything(raw_dir, tmp_path, fake):
    """end=None（仅 final_test 使用）读全部数据；这里只在假数据上验证该选项。"""
    m = D.load_market(raw_dir, end=None, cache_dir=tmp_path / "cache")
    assert m.dates.equals(fake.dates) and m.end == fake.dates[-1]


def test_default_end_is_run_data_end():
    import inspect
    sig = inspect.signature(D.load_market)
    assert sig.parameters["end"].default == RUN_DATA_END
    assert str(sig.parameters["cache_dir"].default) == "cache"


# =====================================================================
# 缓存
# =====================================================================

def test_second_load_uses_cache(raw_dir, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    m1 = D.load_market(raw_dir, cache_dir=cache)
    assert len([p for p in cache.iterdir() if p.name.startswith("market_")]) == 1

    def boom(*a, **k):
        raise AssertionError("第二次加载不应重新构建")
    monkeypatch.setattr(D, "_build_market", boom)
    m2 = D.load_market(raw_dir, cache_dir=cache)
    for f in FIELDS:
        pd.testing.assert_frame_equal(m1.fields[f], m2.fields[f])
    pd.testing.assert_frame_equal(m1.universe, m2.universe)
    pd.testing.assert_frame_equal(m1.sector, m2.sector)
    for h in HORIZONS:
        pd.testing.assert_frame_equal(m1.fwd_ret[h], m2.fwd_ret[h])
    assert m1.end == m2.end
    assert m2.permnos.dtype == np.int64 and m2.dates.name == "date"


def test_cache_keyed_by_end_and_source(raw_dir, tmp_path, fake):
    cache = tmp_path / "cache"
    D.load_market(raw_dir, cache_dir=cache)
    D.load_market(raw_dir, end="2016-12-31", cache_dir=cache)
    assert len([p for p in cache.iterdir() if p.name.startswith("market_")]) == 2
    # 原始文件变化 => 重建：把第 4 只的调出日改掉
    mem = pd.read_csv(raw_dir / D.MEMBERSHIP_FILE, encoding="utf-8", dtype=str, keep_default_na=False)
    mem.loc[3, "ending"] = str(_ymd(fake.dates[N_DAYS // 4]))
    time.sleep(0.01)
    mem.to_csv(raw_dir / D.MEMBERSHIP_FILE, index=False, encoding="utf-8")
    m = D.load_market(raw_dir, cache_dir=cache)
    assert not m.universe.iloc[N_DAYS // 4 + 1:, 3].any()
    assert len([p for p in cache.iterdir() if p.name.startswith("market_")]) == 3


def test_cache_does_not_touch_data_dir(raw_dir, tmp_path):
    before = sorted(p.name for p in raw_dir.iterdir())
    D.load_market(raw_dir, cache_dir=tmp_path / "cache")
    assert sorted(p.name for p in raw_dir.iterdir()) == before


# =====================================================================
# split_dates
# =====================================================================

def _pos(m: Market):
    return pd.Series(np.arange(len(m.dates)), index=m.dates)


@pytest.mark.parametrize("split", ["train", "valid"])
@pytest.mark.parametrize("h", HORIZONS)
def test_split_dates_do_not_cross(fake, split, h):
    d = D.split_dates(fake, split, h)
    s0, s1 = (pd.Timestamp(x) for x in SPLITS[split])
    assert isinstance(d, pd.DatetimeIndex) and d.name == "date"
    assert len(d) > 0 and d.is_monotonic_increasing
    pos = _pos(fake)
    for t in d:
        i = pos[t]
        assert s0 <= t <= s1
        assert i + 1 + h < len(fake.dates)
        assert s0 <= fake.dates[i + 1 + h] <= s1


@pytest.mark.parametrize("h", HORIZONS)
def test_split_dates_exact_boundaries(fake, h):
    dates = fake.dates
    train_all = dates[dates <= "2016-12-31"]
    valid_all = dates[dates >= "2017-01-01"]
    tr = D.split_dates(fake, "train", h)
    va = D.split_dates(fake, "valid", h)
    # 训练段：去掉最后 1+h 个交易日
    assert tr.equals(pd.DatetimeIndex(train_all[: len(train_all) - (1 + h)], name="date"))
    # 验证段：数据在段内结束前截止，去掉最后 1+h 个交易日
    assert va.equals(pd.DatetimeIndex(valid_all[: len(valid_all) - (1 + h)], name="date"))
    # 各 h 单独取：h 越大日期越少
    assert len(tr) == len(train_all) - 1 - h


def test_split_dates_test_split_empty_on_fake(fake):
    assert len(D.split_dates(fake, "test", MAIN_HORIZON)) == 0


def test_split_dates_labels_all_available(fake):
    """split_dates 给出的日期上，股票池内的标签基本都有值（除退市股票）。"""
    for h in HORIZONS:
        d = D.split_dates(fake, "train", h)
        lab = fake.fwd_ret[h].loc[d].where(fake.universe.loc[d])
        n_valid = lab.notna().sum(axis=1)
        assert (n_valid >= MIN_STOCKS_PER_DAY).all()


def test_split_dates_bad_args(fake):
    with pytest.raises(ValueError):
        D.split_dates(fake, "dev", 5)          # type: ignore[arg-type]
    with pytest.raises(ValueError):
        D.split_dates(fake, "train", -1)


def test_split_default_horizon(fake):
    assert D.split_dates(fake, "train").equals(D.split_dates(fake, "train", MAIN_HORIZON))


def test_label_alignment_reversal_signal(loaded):
    """用最小替身算 -ts_sum(ret,5) 的日 Rank IC：标签对齐正确时应显著为正。"""
    m = loaded
    factor = -m.fields["ret"].rolling(5, min_periods=5).sum()
    d = D.split_dates(m, "train", MAIN_HORIZON)
    f = factor.loc[d].where(m.universe.loc[d])
    y = m.fwd_ret[MAIN_HORIZON].loc[d]
    ok = f.notna() & y.notna()
    fr = f.where(ok).rank(axis=1)
    yr = y.where(ok).rank(axis=1)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    yr = yr.sub(yr.mean(axis=1), axis=0)
    ic = (fr * yr).sum(axis=1) / np.sqrt((fr ** 2).sum(axis=1) * (yr ** 2).sum(axis=1))
    ic = ic[ok.sum(axis=1) >= MIN_STOCKS_PER_DAY]
    assert ic.mean() > 0
    assert ic.mean() / ic.std() * np.sqrt(len(ic)) > 2


# =====================================================================
# 行业按时点取：边界规则
# =====================================================================

def _sec_df(rows):
    df = pd.DataFrame(rows, columns=["permno", "gsector", "indfrom", "indthru"])
    df["indfrom"] = pd.to_datetime(df["indfrom"])
    df["indthru"] = pd.to_datetime(df["indthru"])
    return df


def test_sector_rules_overlap_before_gap_after():
    dates = pd.bdate_range("2020-01-01", "2020-03-31", name="date")
    permnos = pd.Index([1, 2, 3], dtype="int64", name="permno")
    sec = _sec_df([
        # permno 1：两段重叠，重叠处取 indfrom 更晚的一行
        (1, 10.0, "2020-01-15", "2020-03-01"),
        (1, 20.0, "2020-02-01", None),
        # permno 2：早于最早 indfrom 用最早一行；两段之间有空档 -> NaN；最后一段结束后 -> NaN
        (2, 30.0, "2020-02-03", "2020-02-14"),
        (2, 40.0, "2020-03-02", "2020-03-13"),
    ])
    out = D._build_sector(sec, dates, permnos)
    s1 = out[1]
    assert (s1.loc[:"2020-01-31"] == 10.0).all()          # 早于最早 indfrom 也用 10
    assert (s1.loc["2020-02-01":] == 20.0).all()          # 重叠段取 indfrom 晚的
    s2 = out[2]
    assert (s2.loc[:"2020-02-14"] == 30.0).all()
    assert s2.loc["2020-02-17":"2020-02-28"].isna().all()  # 空档
    assert (s2.loc["2020-03-02":"2020-03-13"] == 40.0).all()
    assert s2.loc["2020-03-16":].isna().all()             # 已结束
    assert out[3].isna().all()                            # 证券表中没有的股票
    assert (out.dtypes == np.float64).all()


def test_universe_reentry_and_inclusive_bounds():
    dates = pd.bdate_range("2020-01-01", periods=20, name="date")
    permnos = pd.Index([1, 2], dtype="int64", name="permno")
    member = pd.DataFrame({
        "permno": [1, 1, 2],
        "start": pd.to_datetime([dates[2], dates[10], dates[0]]),
        "end": pd.to_datetime([dates[5], None, dates[19]]),
    })
    has_close = pd.DataFrame(True, index=dates, columns=permnos)
    has_close.iloc[15, 1] = False                          # 当天无 close -> 不在池内
    u = D._build_universe(member, dates, permnos, has_close)
    exp1 = np.zeros(20, bool); exp1[2:6] = True; exp1[10:] = True
    assert (u[1].to_numpy() == exp1).all()
    assert u[2].sum() == 19 and not u[2].iloc[15]


# =====================================================================
# 解析、路径、错误处理
# =====================================================================

def test_parse_dates_formats():
    s = pd.Series([20000103, "2000-01-04", "20000105", None, "", "20001231.0"])
    out = D._parse_dates(s)
    assert list(out.iloc[[0, 1, 2, 5]]) == [pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-04"),
                                            pd.Timestamp("2000-01-05"), pd.Timestamp("2000-12-31")]
    assert out.iloc[3:5].isna().all()
    with pytest.raises(ValueError):
        D._parse_dates(pd.Series(["not-a-date"]))


def test_panel_in_parent_dir_fallback(fake, tmp_path):
    data_dir = write_fake_raw(fake, tmp_path / "root" / "data", panel_in_parent=True)
    m = D.load_market(data_dir, cache_dir=tmp_path / "cache")
    pd.testing.assert_frame_equal(m.fields["close"], fake.fields["close"], check_freq=False)


def test_missing_panel_raises(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        D.load_market(tmp_path / "data", cache_dir=tmp_path / "cache")


def test_duplicate_panel_rows_raise(raw_dir, tmp_path):
    p = raw_dir / D.PANEL_FILE
    df = pd.read_parquet(p)
    pd.concat([df, df.iloc[:1]]).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="重复"):
        D.load_market(raw_dir, cache_dir=tmp_path / "cache")


def test_missing_column_message(raw_dir, tmp_path):
    p = raw_dir / D.SECURITY_FILE
    df = pd.read_csv(p, encoding="utf-8").rename(columns={"gsector": "sector_code"})
    df.to_csv(p, index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="gsector"):
        D.load_market(raw_dir, cache_dir=tmp_path / "cache")


# =====================================================================
# ticker_map / load_benchmarks
# =====================================================================

def test_ticker_map_basic(raw_dir, fake):
    tm = D.ticker_map(raw_dir)
    assert len(tm) == len(fake.permnos)
    assert all(isinstance(k, int) and isinstance(v, str) for k, v in tm.items())
    assert tm[int(fake.permnos[0])] == f"T{fake.permnos[0]}"


def test_ticker_map_latest_by_date(tmp_path):
    pd.DataFrame({"permno": [1, 1, 2], "ticker": ["NEW", "OLD", "BB"],
                  "date": ["20200101", "20100101", "20150101"]}).to_csv(
        tmp_path / D.TICKER_FILE, index=False, encoding="utf-8")
    assert D.ticker_map(tmp_path) == {1: "NEW", 2: "BB"}


def test_load_benchmarks_sorted(tmp_path):
    pd.DataFrame({"date": [20200103, 20200101, 20200102, 20230101],
                  "close": [3.0, 1.0, 2.0, 9.0]}).to_csv(
        tmp_path / D.SP500TR_FILE, index=False, encoding="utf-8")
    pd.DataFrame({"date": [20200101, 20200102], "close": [1.0, 2.0]}).to_csv(
        tmp_path / D.SPY_FILE, index=False, encoding="utf-8")
    b = D.load_benchmarks(tmp_path)
    tr = b["sp500_tr"]
    assert tr["date"].is_monotonic_increasing
    assert tr["close"].tolist() == [1.0, 2.0, 3.0]        # 2023 被 RUN_DATA_END 截断
    assert len(D.load_benchmarks(tmp_path, end=None)["sp500_tr"]) == 4


# =====================================================================
# 真实数据（默认不跑：python -m pytest -m real -q）
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]


def _settings() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def _real_dirs():
    s = _settings()
    cache = Path(s.get("cache_dir", "cache"))
    if not cache.is_absolute():
        cache = ROOT / cache
    return Path(s["data_dir"]), cache


@pytest.fixture(scope="module")
def real_market():
    data_dir, cache = _real_dirs()
    return D.load_market(data_dir, cache_dir=cache)       # 默认 end=RUN_DATA_END，不用 end=None


@pytest.mark.real
def test_real_panel_file_size():
    """panel 总行数 / 股票数（PRD 6.1）。只读 parquet 元数据和 permno 一列，不加载截止日之后的行情。"""
    import pyarrow.parquet as pq
    data_dir, _ = _real_dirs()
    path = D._resolve_panel_path(data_dir)
    assert pq.ParquetFile(path).metadata.num_rows == 5_260_971
    assert pd.read_parquet(path, columns=["permno"])["permno"].nunique() == 1_195


@pytest.mark.real
def test_real_rows_stocks_dates(real_market):
    data_dir, _ = _real_dirs()
    end = pd.Timestamp(RUN_DATA_END)
    panel = pd.read_parquet(D._resolve_panel_path(data_dir), columns=["date", "permno", "close"],
                            filters=[("date", "<=", end)])
    m = real_market
    assert m.dates[0] == pd.Timestamp("2000-01-03")
    assert m.dates[-1] <= end and m.end <= end
    assert m.dates.equals(pd.DatetimeIndex(np.sort(panel["date"].unique()), name="date").astype(m.dates.dtype))
    assert len(m.permnos) == panel["permno"].nunique()
    # 宽表里的记录数 = 截止日前 panel 的 close 非空数
    assert int(m.fields["close"].notna().to_numpy().sum()) == int(panel["close"].notna().sum())
    assert m.fwd_ret[MAIN_HORIZON].iloc[-(1 + MAIN_HORIZON):].isna().all().all()


@pytest.mark.real
def test_real_universe_about_500(real_market):
    n = real_market.universe.sum(axis=1)
    assert 490 <= n.median() <= 510
    assert n.min() >= 450 and n.max() <= 550


@pytest.mark.real
def test_real_sector_changers(real_market):
    data_dir, _ = _real_dirs()
    sec = D._read_security(data_dir / D.SECURITY_FILE)
    n_distinct = sec.dropna(subset=["gsector"]).groupby("permno")["gsector"].nunique()
    changers = n_distinct[n_distinct >= 2].index
    assert len(changers) == 185                            # PRD 6.2
    m = real_market
    s = m.sector
    # 换行业发生在已加载区间内（第二个不同行业的 indfrom <= 最后一天）且有行情的股票，必须取到 >=2 个行业
    expect = []
    for p in changers:
        if p not in s.columns:
            continue
        g = sec[sec["permno"] == p].sort_values("indfrom")
        first = g["gsector"].dropna().iloc[0]
        later = g[(g["gsector"] != first) & g["gsector"].notna()]
        # 换行业必须发生在已加载区间内部：早于首个交易日（如 1999 年）的换行业在数据里看不到
        if len(later) and m.dates[0] < later["indfrom"].min() <= m.dates[-1]:
            expect.append(p)
    got = s[expect].nunique(axis=0, dropna=True)
    assert (got >= 2).all(), got[got < 2].index.tolist()
    vals = s.to_numpy()
    assert set(np.unique(vals[~np.isnan(vals)])) <= {10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.}


@pytest.mark.real
def test_real_second_load_uses_cache(real_market, monkeypatch):
    data_dir, cache = _real_dirs()

    def boom(*a, **k):
        raise AssertionError("第二次加载不应重新构建")
    monkeypatch.setattr(D, "_build_market", boom)
    m = D.load_market(data_dir, cache_dir=cache)
    assert m.dates.equals(real_market.dates)
    pd.testing.assert_frame_equal(m.universe, real_market.universe)
