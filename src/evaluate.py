"""
评估程序（PRD 第 9、10 节，线 C）。

- evaluate：在某一段（train / valid / test）上算回测明细 Report
- gate_train / gate_valid：按冻结门槛文件逐项判定
- load_thresholds / thresholds_sha256：读门槛文件、算哈希

口径：
- 日 Rank IC：第 t 日股票池内，因子值与 fwd_ret_h 的 Spearman 相关；有效股票数 < MIN_STOCKS_PER_DAY 的日子跳过。
- IC 日期一律来自 split_dates(market, split, h)，每个 h 单独取（标签不跨段）。
- 向量化：逐行 rank 后按行算 Pearson，不做逐日 Python 循环。
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.contracts import (
    HORIZONS,
    MAIN_HORIZON,
    MIN_STOCKS_PER_DAY,
    N_QUANTILES,
    SPLITS,
    GateCheck,
    GateResult,
    Market,
    Report,
    Split,
    YearStat,
)

DEFAULT_THRESHOLDS_PATH = "config/thresholds.yaml"


# =====================================================================
# split_dates：归线 A（src/data.py）。没有该模块时用严格按 contracts 写的本地替身。
# =====================================================================

def _split_dates_fallback(market: Market, split: Split, h: int = MAIN_HORIZON) -> pd.DatetimeIndex:
    """t 在 SPLITS[split] 内，且 t+1+h 个交易日后仍在同一段内并存在于 market。"""
    start, end = (pd.Timestamp(s) for s in SPLITS[split])
    dates = market.dates
    n = len(dates)
    pos = np.arange(n)
    tgt = pos + 1 + h
    exists = tgt < n
    tgt_dates = dates[np.minimum(tgt, n - 1)]
    in_seg = (dates >= start) & (dates <= end)
    ok = in_seg & exists & (tgt_dates >= start) & (tgt_dates <= end)
    return pd.DatetimeIndex(dates[ok], name="date")


try:  # 集成后使用线 A 的实现
    from src.data import split_dates as _split_dates  # type: ignore
except ModuleNotFoundError as _e:  # pragma: no cover - 取决于是否已合并线 A
    if _e.name not in ("src.data", "src"):
        raise
    _split_dates = _split_dates_fallback


# =====================================================================
# 门槛文件
# =====================================================================

def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS_PATH) -> dict:
    with open(Path(path), "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "train" not in data or "valid" not in data:
        raise ValueError(f"门槛文件格式不正确：{path}")
    return data


def thresholds_sha256(path: str | Path = DEFAULT_THRESHOLDS_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =====================================================================
# 向量化工具
# =====================================================================

def _align(df: pd.DataFrame, market: Market) -> pd.DataFrame:
    out = df.reindex(index=market.dates, columns=market.permnos).astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _universe(market: Market) -> pd.DataFrame:
    return market.universe.reindex(index=market.dates, columns=market.permnos,
                                   fill_value=False).astype(bool)


def _row_rank(a: np.ndarray) -> np.ndarray:
    """逐行平均秩，NaN 保持 NaN。"""
    return pd.DataFrame(a).rank(axis=1, method="average").to_numpy(dtype="float64")


def _rowwise_spearman(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    逐行 Spearman：只用两者都非 NaN 的位置，在该公共集合内各自 rank，再按行算 Pearson。
    返回 (corr, n)；有效数 < min_n 或方差为 0 的行 corr 为 NaN。
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = mask.sum(axis=1)
    xr = _row_rank(np.where(mask, x, np.nan))
    yr = _row_rank(np.where(mask, y, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_n = np.where(n > 0, n, 1)
        xm = np.nansum(xr, axis=1) / safe_n
        ym = np.nansum(yr, axis=1) / safe_n
        xd = np.where(mask, xr - xm[:, None], 0.0)
        yd = np.where(mask, yr - ym[:, None], 0.0)
        num = (xd * yd).sum(axis=1)
        den = np.sqrt((xd * xd).sum(axis=1) * (yd * yd).sum(axis=1))
        corr = num / den
    corr[(n < min_n) | ~(den > 0)] = np.nan
    return corr, n


def _pearson_rows(xr: np.ndarray, yr: np.ndarray, mask: np.ndarray, min_n: int) -> np.ndarray:
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_n = np.where(n > 0, n, 1)
        xm = np.where(mask, xr, 0.0).sum(axis=1) / safe_n
        ym = np.where(mask, yr, 0.0).sum(axis=1) / safe_n
        xd = np.where(mask, xr - xm[:, None], 0.0)
        yd = np.where(mask, yr - ym[:, None], 0.0)
        num = (xd * yd).sum(axis=1)
        den = np.sqrt((xd * xd).sum(axis=1) * (yd * yd).sum(axis=1))
        corr = num / den
    corr[(n < min_n) | ~(den > 0)] = np.nan
    return corr


def _ranked_corr(a: np.ndarray, b: np.ndarray, min_n: int) -> np.ndarray:
    """
    a、b 为已在各自有效集合内排好序的值（to_ranks 结果）。
    某行两者的有效集合相同 -> 百分位秩与公共集合内的秩成线性关系，直接按行 Pearson 即为 Spearman；
    集合不同的行，在公共集合内重新 rank 后再算（结果与 _rowwise_spearman 完全一致，只是更快）。
    """
    va, vb = ~np.isnan(a), ~np.isnan(b)
    mask = va & vb
    corr = _pearson_rows(a, b, mask, min_n)
    diff = np.nonzero((va != vb).any(axis=1))[0]
    if len(diff):
        corr[diff], _ = _rowwise_spearman(a[diff], b[diff], min_n)
    return corr


def _f(v) -> float:
    """转 Python float；NaN 保留为 float('nan')。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _ic_stats(ic: pd.Series) -> tuple[int, float, float, float, float, float]:
    """返回 (n_days, mean, std, icir, t_stat, pos_ratio)；std 用 ddof=1。"""
    ic = ic.dropna()
    n = int(len(ic))
    if n == 0:
        nan = float("nan")
        return 0, nan, nan, nan, nan, nan
    mean = _f(ic.mean())
    std = _f(ic.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if (n > 1 and std > 0) else float("nan")
    t_stat = icir * math.sqrt(n) if not math.isnan(icir) else float("nan")
    pos = _f((ic > 0).mean())
    return n, mean, std, _f(icir), _f(t_stat), pos


def _daily_ic(f: pd.DataFrame, market: Market, uni: pd.DataFrame, h: int,
              dates: pd.DatetimeIndex) -> pd.Series:
    fwd = _align(market.fwd_ret[h], market)
    mask = uni.loc[dates].to_numpy()
    x = np.where(mask, f.loc[dates].to_numpy(), np.nan)
    y = np.where(mask, fwd.loc[dates].to_numpy(), np.nan)
    corr, _ = _rowwise_spearman(x, y, MIN_STOCKS_PER_DAY)
    return pd.Series(corr, index=dates, dtype="float64")


# =====================================================================
# evaluate
# =====================================================================

def evaluate(factor: pd.DataFrame, market: Market, split: Split,
             refs: dict[str, pd.DataFrame] | None = None) -> Report:
    """在 split 段上计算回测明细（PRD 第 9 节）。refs 只允许在 split="train" 时传入。"""
    if split not in SPLITS:
        raise ValueError(f"未知的数据段：{split!r}")
    if refs is not None and split != "train":
        raise ValueError("refs（参照因子）只允许在 split='train' 时传入")

    uni = _universe(market)
    f = _align(factor, market).where(uni)

    # ---- 主口径 h = MAIN_HORIZON ----
    dates = _split_dates(market, split, MAIN_HORIZON)
    dates = pd.DatetimeIndex(dates).intersection(market.dates)
    ic = _daily_ic(f, market, uni, MAIN_HORIZON, dates)
    n_days, ic_mean, ic_std, icir, t_stat, pos = _ic_stats(ic)

    # 覆盖：主口径 IC 日期上，每日股票池内有因子值的股票数的均值
    cov = f.loc[dates].notna().sum(axis=1)
    coverage_mean = _f(cov.mean()) if len(cov) else float("nan")

    # ---- 分年表 ----
    yearly: list[YearStat] = []
    ic_valid = ic.dropna()
    for year, s in ic_valid.groupby(ic_valid.index.year):
        yn, ym, _, yir, _, ypos = _ic_stats(s)
        yearly.append(YearStat(year=int(year), ic_mean=ym, icir=yir, ic_pos_ratio=ypos, n_days=yn))

    # ---- IC 衰减：各 h 用各自的 split_dates ----
    decay: dict[str, float] = {}
    for h in HORIZONS:
        if h == MAIN_HORIZON:
            decay[str(h)] = ic_mean
            continue
        d_h = pd.DatetimeIndex(_split_dates(market, split, h)).intersection(market.dates)
        decay[str(h)] = _ic_stats(_daily_ic(f, market, uni, h, d_h))[1]

    # ---- 分组收益（只用主口径有效 IC 日）----
    quantile_ret = _quantile_returns(f, market, uni, pd.DatetimeIndex(ic_valid.index))
    long_short = _f(quantile_ret[-1] - quantile_ret[0])

    # ---- 换手：相邻两个交易日（都在主口径 IC 日期内）因子截面 Rank 相关的均值 ----
    turnover = _turnover_autocorr(f, market, dates)

    # ---- 去重 ----
    max_ref_corr: Optional[float] = None
    max_ref_id: Optional[str] = None
    if refs is not None:
        max_ref_corr, max_ref_id = _max_ref_corr(f, market, refs)

    return Report(
        split=str(split),
        n_days=n_days,
        coverage_mean=coverage_mean,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        t_stat=t_stat,
        ic_pos_ratio=pos,
        yearly=yearly,
        decay=decay,
        quantile_ret=quantile_ret,
        long_short=long_short,
        turnover_autocorr=turnover,
        max_ref_corr=max_ref_corr,
        max_ref_id=max_ref_id,
    )


def _quantile_returns(f: pd.DataFrame, market: Market, uni: pd.DataFrame,
                      days: pd.DatetimeIndex) -> list[float]:
    """每日在因子与 fwd_ret_5 都有效的股票内按因子值分 N_QUANTILES 组，组内平均收益再对日求均值。"""
    nan = float("nan")
    if len(days) == 0:
        return [nan] * N_QUANTILES
    fwd = _align(market.fwd_ret[MAIN_HORIZON], market)
    mask = uni.loc[days].to_numpy()
    x = np.where(mask, f.loc[days].to_numpy(), np.nan)
    y = np.where(mask, fwd.loc[days].to_numpy(), np.nan)
    both = ~np.isnan(x) & ~np.isnan(y)
    x = np.where(both, x, np.nan)
    n = both.sum(axis=1)
    rk = pd.DataFrame(x).rank(axis=1, method="first").to_numpy()  # 1..n
    with np.errstate(invalid="ignore"):
        grp = np.floor((rk - 1.0) * N_QUANTILES / np.where(n > 0, n, 1)[:, None])
    out: list[float] = []
    for g in range(N_QUANTILES):
        sel = both & (grp == g)
        cnt = sel.sum(axis=1)
        s = np.where(sel, y, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            daily = np.where(cnt > 0, s / np.where(cnt > 0, cnt, 1), np.nan)
        out.append(_f(np.nanmean(daily)) if np.any(~np.isnan(daily)) else nan)
    return out


def _turnover_autocorr(f: pd.DataFrame, market: Market, dates: pd.DatetimeIndex) -> float:
    if len(dates) < 2:
        return float("nan")
    pos = market.dates.get_indexer(dates)
    pairs = np.nonzero(np.diff(pos) == 1)[0]          # dates[i] 与 dates[i+1] 为相邻交易日
    if len(pairs) == 0:
        return float("nan")
    arr = f.loc[dates].to_numpy()
    prev = arr[pairs]
    curr = arr[pairs + 1]
    corr, _ = _rowwise_spearman(prev, curr, MIN_STOCKS_PER_DAY)
    return _f(np.nanmean(corr)) if np.any(~np.isnan(corr)) else float("nan")


def _max_ref_corr(f: pd.DataFrame, market: Market,
                  refs: dict[str, pd.DataFrame]) -> tuple[Optional[float], Optional[str]]:
    """与每个参照因子的日截面 Spearman 相关均值（训练段），取绝对值最大者，保留符号。"""
    from src.refstore import to_ranks  # 同线模块

    new = to_ranks(f, market).astype("float64")
    best_val: Optional[float] = None
    best_id: Optional[str] = None
    for ref_id in sorted(refs):
        ref = refs[ref_id]
        idx = new.index.intersection(ref.index)
        cols = new.columns.intersection(ref.columns)
        if len(idx) == 0 or len(cols) == 0:
            continue
        a = new.loc[idx, cols].to_numpy(dtype="float64")
        b = ref.loc[idx, cols].to_numpy(dtype="float64")
        corr = _ranked_corr(a, b, MIN_STOCKS_PER_DAY)
        if not np.any(~np.isnan(corr)):
            continue
        m = _f(np.nanmean(corr))
        if best_val is None or abs(m) > abs(best_val):
            best_val, best_id = m, str(ref_id)
    return best_val, best_id


# =====================================================================
# 门槛判定
# =====================================================================

def _check(name: str, value: float, threshold: float, passed: bool) -> GateCheck:
    return GateCheck(name=name, value=_f(value), threshold=_f(threshold), passed=bool(passed))


def _ge(value: float, threshold: float) -> bool:
    return (not math.isnan(value)) and value >= threshold


def gate_train(report: Report, thresholds: dict,
               expected_direction: Optional[int] = None) -> GateResult:
    t = thresholds["train"]
    ic = _f(report["ic_mean"])
    ir = _f(report["icir"])
    checks = [
        _check("abs_ic_mean", abs(ic), t["abs_ic_mean_min"], _ge(abs(ic), _f(t["abs_ic_mean_min"]))),
        _check("abs_icir", abs(ir), t["abs_icir_min"], _ge(abs(ir), _f(t["abs_icir_min"]))),
    ]
    if t.get("direction_must_match_hypothesis", False) and expected_direction is not None:
        # IC 的符号必须等于假设声明的方向：value = sign(IC) * 预期方向，需为 +1
        prod = float("nan") if math.isnan(ic) else float(np.sign(ic) * np.sign(expected_direction))
        checks.append(_check("direction", prod, 1.0, (not math.isnan(prod)) and prod > 0))
    ref = report.get("max_ref_corr")
    lim = _f(t["max_abs_ref_corr"])
    if ref is None:
        # 没有参照集就无法证明不重复：按最保守的理解判为不通过（见 issues/C.md）
        checks.append(_check("max_ref_corr", float("nan"), lim, False))
    else:
        r = abs(_f(ref))
        checks.append(_check("max_ref_corr", r, lim, (not math.isnan(r)) and r < lim))
    return GateResult(passed=all(c["passed"] for c in checks), checks=checks)


def gate_valid(valid_report: Report, train_report: Report, thresholds: dict) -> GateResult:
    t = thresholds["valid"]
    ic = _f(valid_report["ic_mean"])
    ir = _f(valid_report["icir"])
    checks = [
        _check("abs_ic_mean", abs(ic), t["abs_ic_mean_min"], _ge(abs(ic), _f(t["abs_ic_mean_min"]))),
        _check("abs_icir", abs(ir), t["abs_icir_min"], _ge(abs(ir), _f(t["abs_icir_min"]))),
    ]
    if t.get("same_sign_as_train", True):
        tr = _f(train_report["ic_mean"])
        if math.isnan(ic) or math.isnan(tr):
            prod = float("nan")
        else:
            prod = float(np.sign(ic) * np.sign(tr))
        checks.append(_check("same_sign", prod, 1.0, (not math.isnan(prod)) and prod > 0))
    return GateResult(passed=all(c["passed"] for c in checks), checks=checks)
