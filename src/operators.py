"""
算子实现（PRD 第 7 节）。全部为宽表向量化运算：
  index = 交易日（升序），columns = permno。

约定：
  - 时序算子（ts_*）逐只股票沿时间往回看，在完整历史上计算，不受股票池限制；
    窗口内有任何缺失则结果为 NaN（rolling(d, min_periods=d)）。
  - 截面算子（cs_*、group_neutralize）逐日计算，先 x.where(universe) 再算，
    只在当日股票池内取值。
  - 除以 0、log 非正数、任何 inf 一律转成 NaN；不做任何填补。
  - 只取过去：shift(k) 只用 k >= 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ts_rank 分块时每块最多处理的元素数（行数 × 股票数 × 窗口），控制内存
_TS_RANK_CHUNK_ELEMS = 20_000_000


def clean(x: pd.DataFrame) -> pd.DataFrame:
    """inf -> NaN，并统一为 float64。"""
    out = x.astype("float64")
    return out.mask(np.isinf(out.to_numpy()))


def _check_window(d: int) -> None:
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError(f"窗口必须是正整数，收到 {d!r}")


def _rolling(x: pd.DataFrame, d: int):
    _check_window(d)
    return x.rolling(window=d, min_periods=d)


# ---------------------------------------------------------------------
# 时序算子
# ---------------------------------------------------------------------

def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return clean(_rolling(x, d).mean())


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """样本标准差（ddof=1）。"""
    return clean(_rolling(x, d).std())


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return clean(_rolling(x, d).sum())


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return clean(_rolling(x, d).max())


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return clean(_rolling(x, d).min())


def ts_skew(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """样本偏度（pandas 口径，偏差修正）。"""
    return clean(_rolling(x, d).skew())


def ts_delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """d 个交易日前的值（按全局交易日历移位，与 contracts.compute_ret 一致）。"""
    _check_window(d)
    return clean(x.shift(d))


def ts_delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    _check_window(d)
    return clean(x - x.shift(d))


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    当日值在过去 d 日（含当日）中的分位，取值 0–1：
      (窗口内严格小于当日值的个数 + 0.5 × (与当日值相等的个数 − 1)) / (d − 1)
    最小为 0、最大为 1，并列取平均。窗口内有缺失则为 NaN。d = 1 时恒为 NaN。
    """
    _check_window(d)
    a = x.to_numpy(dtype="float64")
    T, N = a.shape
    out = np.full((T, N), np.nan)
    if d < 2 or T < d:
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    win = sliding_window_view(a, window_shape=d, axis=0)   # (T-d+1, N, d)，视图不复制
    n_win = win.shape[0]
    step = max(1, _TS_RANK_CHUNK_ELEMS // max(1, N * d))
    for s in range(0, n_win, step):
        w = win[s:s + step]                                 # (k, N, d)
        cur = w[..., -1:]                                   # 当日值
        less = (w < cur).sum(axis=-1)
        equal = (w == cur).sum(axis=-1)
        r = (less + 0.5 * (equal - 1)) / (d - 1)
        r[np.isnan(w).any(axis=-1)] = np.nan
        out[s + d - 1: s + d - 1 + w.shape[0]] = r
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """过去 d 日 x 与 y 的 Pearson 相关。两者在窗口内必须都完整；方差为 0 时为 NaN。"""
    _check_window(d)
    x, y = x.align(y, join="outer")
    both = x.notna() & y.notna()
    xm, ym = x.where(both), y.where(both)
    mx = xm.rolling(d, min_periods=d).mean()
    my = ym.rolling(d, min_periods=d).mean()
    # 相关系数用 pandas 的稳定实现；方差只用来识别"窗口内为常数"的情形（结果记 NaN）
    corr = xm.rolling(d, min_periods=d).corr(ym)
    vx = (xm * xm).rolling(d, min_periods=d).mean() - mx * mx
    vy = (ym * ym).rolling(d, min_periods=d).mean() - my * my
    degenerate = (vx <= 1e-12 * (mx.abs() + 1.0) ** 2) | (vy <= 1e-12 * (my.abs() + 1.0) ** 2)
    corr = corr.mask(degenerate)
    return clean(corr.clip(lower=-1.0, upper=1.0))


# ---------------------------------------------------------------------
# 截面算子（只在当日股票池内）
# ---------------------------------------------------------------------

def _in_universe(x: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    u = universe.reindex(index=x.index, columns=x.columns, fill_value=False).astype(bool)
    return x.where(u)


def cs_rank(x: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """当日股票池内的截面百分位（pandas rank(pct=True)，并列取平均，取值 (0, 1]）。"""
    xu = _in_universe(x, universe)
    return clean(xu.rank(axis=1, pct=True, method="average"))


def cs_zscore(x: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """当日股票池内标准化：(x − 截面均值) / 截面标准差（ddof=1）；标准差为 0 时为 NaN。"""
    xu = _in_universe(x, universe)
    mu = xu.mean(axis=1)
    sd = xu.std(axis=1, ddof=1)
    sd = sd.where(sd > 0)
    return clean(xu.sub(mu, axis=0).div(sd, axis=0))


def group_neutralize(x: pd.DataFrame, group: pd.DataFrame,
                     universe: pd.DataFrame) -> pd.DataFrame:
    """减去当日股票池内同组（行业）均值；行业缺失的股票结果为 NaN。"""
    xu = _in_universe(x, universe)
    g = group.reindex(index=x.index, columns=x.columns)
    xu = xu.where(g.notna())
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    gv = g.to_numpy()
    codes = np.unique(gv[~np.isnan(gv)])
    for c in codes:
        m = g == c
        xm = xu.where(m)
        mu = xm.mean(axis=1)
        out = out.mask(m, xm.sub(mu, axis=0))
    return clean(out)


# ---------------------------------------------------------------------
# 算术
# ---------------------------------------------------------------------

def safe_div(a, b):
    """除法；除以 0 与 inf 结果为 NaN。a、b 可以是宽表或数字。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(b, pd.DataFrame):
            b = b.mask(b == 0)
            r = a / b
        else:
            if b == 0:
                if isinstance(a, pd.DataFrame):
                    return pd.DataFrame(np.nan, index=a.index, columns=a.columns)
                return float("nan")
            r = a / b
    return clean(r) if isinstance(r, pd.DataFrame) else _clean_scalar(r)


def log(x: pd.DataFrame) -> pd.DataFrame:
    """自然对数；输入 ≤ 0 时为 NaN。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return clean(np.log(x.where(x > 0)))


def abs_(x: pd.DataFrame) -> pd.DataFrame:
    return clean(x.abs())


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return clean(np.sign(x))


def _clean_scalar(v: float) -> float:
    v = float(v)
    return float("nan") if np.isinf(v) else v


# ---------------------------------------------------------------------
# 扩充算子（参考 WorldQuant 101 的写法）
# ---------------------------------------------------------------------

def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """过去 d 日 x 与 y 的样本协方差（ddof=1）。两者在窗口内必须都完整。"""
    _check_window(d)
    x, y = x.align(y, join="outer")
    both = x.notna() & y.notna()
    xm, ym = x.where(both), y.where(both)
    return clean(xm.rolling(d, min_periods=d).cov(ym))


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """线性衰减加权均值：当日权重 d，前一日 d-1，…，最早一日 1，权重归一化。窗口内有缺失则为 NaN。"""
    _check_window(d)
    a = x.to_numpy(dtype="float64")
    T, N = a.shape
    out = np.full((T, N), np.nan)
    if T >= d:
        w = np.arange(1, d + 1, dtype="float64")
        w /= w.sum()
        win = sliding_window_view(a, window_shape=d, axis=0)   # (T-d+1, N, d)，最后一维从旧到新
        out[d - 1:] = win @ w                                   # 有 NaN 时结果自然为 NaN
    return clean(pd.DataFrame(out, index=x.index, columns=x.columns))


def _ts_argext(x: pd.DataFrame, d: int, use_max: bool) -> pd.DataFrame:
    _check_window(d)
    a = x.to_numpy(dtype="float64")
    T, N = a.shape
    out = np.full((T, N), np.nan)
    if T >= d:
        win = sliding_window_view(a, window_shape=d, axis=0)   # (T-d+1, N, d)
        has_nan = np.isnan(win).any(axis=-1)
        filled = np.where(np.isnan(win), -np.inf if use_max else np.inf, win)
        # 并列时取最近的一天：把窗口倒过来再找第一个极值
        rev = filled[..., ::-1]
        idx = (rev.argmax(axis=-1) if use_max else rev.argmin(axis=-1)).astype("float64")
        idx[has_nan] = np.nan
        out[d - 1:] = idx
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """过去 d 日（含当日）最大值出现在几天前：0 = 当日，d-1 = 窗口最早一天。"""
    return _ts_argext(x, d, use_max=True)


def ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """过去 d 日（含当日）最小值出现在几天前：0 = 当日，d-1 = 窗口最早一天。"""
    return _ts_argext(x, d, use_max=False)


def power(x: pd.DataFrame, a: float) -> pd.DataFrame:
    """带符号的幂：sign(x) * |x|^a（与 101 的 SignedPower 一致，负数也有定义）。"""
    with np.errstate(over="ignore", invalid="ignore"):
        return clean(np.sign(x) * np.abs(x) ** float(a))


def compare(a, b, op: str):
    """比较运算，结果为 1.0 / 0.0；任一方缺失则为 NaN。"""
    fn = {"<": np.less, ">": np.greater, "<=": np.less_equal, ">=": np.greater_equal}[op]
    if not isinstance(a, pd.DataFrame) and not isinstance(b, pd.DataFrame):
        return float(fn(a, b))
    ref = a if isinstance(a, pd.DataFrame) else b
    av = a.to_numpy(dtype="float64") if isinstance(a, pd.DataFrame) else float(a)
    bv = b.to_numpy(dtype="float64") if isinstance(b, pd.DataFrame) else float(b)
    with np.errstate(invalid="ignore"):
        r = fn(av, bv).astype("float64")
    r = np.where(np.isnan(av) | np.isnan(bv), np.nan, r)
    return pd.DataFrame(r, index=ref.index, columns=ref.columns)


def where(cond, a, b):
    """cond > 0 取 a，否则取 b；cond 缺失则为 NaN。a、b 可以是宽表或数字。"""
    ref = next(v for v in (cond, a, b) if isinstance(v, pd.DataFrame))
    def arr(v):
        if isinstance(v, pd.DataFrame):
            return v.reindex(index=ref.index, columns=ref.columns).to_numpy(dtype="float64")
        return np.full(ref.shape, float(v))
    c, av, bv = arr(cond), arr(a), arr(b)
    with np.errstate(invalid="ignore"):
        r = np.where(c > 0, av, bv)
    r = np.where(np.isnan(c), np.nan, r)
    return clean(pd.DataFrame(r, index=ref.index, columns=ref.columns))
