"""
参照因子值存储（PRD 第 13 节，线 C）。

存储的是 REF_SPLIT（训练集）区间内、逐日在股票池内做的截面百分位排序值（float32），
供 evaluate 计算与新因子的日截面 Spearman 相关。

文件：<cache_dir>/ref_values/<ref_id>.parquet
  parquet 要求列名为字符串，所以落盘时 permno 转成 str，读回时恢复为 int64。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import REF_SPLIT, SPLITS, Market

REF_SUBDIR = "ref_values"
_REF_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _ref_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / REF_SUBDIR


def _check_ref_id(ref_id: str) -> None:
    if not isinstance(ref_id, str) or not _REF_ID_RE.match(ref_id):
        raise ValueError(f"非法的参照因子 ID：{ref_id!r}（只允许字母、数字、下划线、连字符）")


def to_ranks(factor: pd.DataFrame, market: Market) -> pd.DataFrame:
    """截取 REF_SPLIT 区间，逐日在股票池内做截面百分位排序，float32。"""
    start, end = (pd.Timestamp(s) for s in SPLITS[REF_SPLIT])
    f = factor.reindex(index=market.dates, columns=market.permnos).astype("float64")
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.where(market.universe.reindex(index=f.index, columns=f.columns, fill_value=False).astype(bool))
    f = f.loc[(f.index >= start) & (f.index <= end)]
    ranks = f.rank(axis=1, pct=True, method="average").astype("float32")
    idx = pd.DatetimeIndex(ranks.index, name="date")
    idx.freq = None                         # 与读回的 parquet 一致，不携带频率
    ranks.index = idx
    ranks.columns = pd.Index(ranks.columns.astype("int64"), name="permno")
    return ranks


def save_ref(ref_id: str, factor: pd.DataFrame, market: Market, cache_dir: str | Path = "cache") -> None:
    """写入 <cache_dir>/ref_values/<ref_id>.parquet（存 to_ranks 的结果）。"""
    _check_ref_id(ref_id)
    ranks = to_ranks(factor, market)
    out = ranks.copy()
    out.columns = [str(c) for c in out.columns]
    d = _ref_dir(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / f"{ref_id}.parquet")


def load_refs(cache_dir: str | Path = "cache") -> dict[str, pd.DataFrame]:
    """读取 <cache_dir>/ref_values/ 下全部参照因子，按 ID 排序；目录不存在返回空字典。"""
    d = _ref_dir(cache_dir)
    if not d.is_dir():
        return {}
    refs: dict[str, pd.DataFrame] = {}
    for p in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(p)
        df.columns = pd.Index(df.columns.astype("int64"), name="permno")
        df.index = pd.DatetimeIndex(df.index, name="date")
        refs[p.stem] = df.astype("float32")
    return refs
