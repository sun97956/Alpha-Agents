"""
线 A：数据层（PRD 第 6 节）。

对外接口（签名见 src/contracts.py 第 4 部分）：
  load_market(data_dir, end=RUN_DATA_END, cache_dir="cache") -> Market
  split_dates(market, split, h=MAIN_HORIZON) -> pd.DatetimeIndex
  ticker_map(data_dir) -> dict[int, str]
附加（本线自用/展示用，不在 contracts 中）：
  load_benchmarks(data_dir) -> dict[str, pd.DataFrame]

规则要点：
  - 原始数据只读；派生结果只写 cache_dir。
  - 缺失值一律不填补。
  - ret / fwd_ret 只调用 contracts.compute_ret / compute_fwd_ret。
  - 股票池 = 成分区间内 且 panel 当天有 close（PRD 6.3）。
  - 行业按时点取（PRD 6.2）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.contracts import (
    FIELDS,
    HORIZONS,
    MAIN_HORIZON,
    RUN_DATA_END,
    SPLITS,
    Market,
    Split,
    compute_fwd_ret,
    compute_ret,
)

# ---------------------------------------------------------------------
# 原始文件名（PRD 6.1）
# ---------------------------------------------------------------------
PANEL_FILE = "panel.parquet"
MEMBERSHIP_FILE = "MASTER_UNIVERSE_sp500_membership_extended_asOf20260930_v2.csv"
SECURITY_FILE = "security_master_with_provisional_classifications.csv"
TICKER_FILE = "permno_to_ticker_mapping.csv"
SPY_FILE = "SPY_OHLCVolume_upTo20260930.csv"
SP500TR_FILE = "SP500_TR_upTo20260930.csv"

PRICE_FIELDS: tuple[str, ...] = tuple(f for f in FIELDS if f != "ret")  # panel 里直接有的字段

CACHE_VERSION = "A2"   # 改了生成逻辑就改这里，旧缓存自动失效

# 当前 pandas 版本的默认日期精度（pandas 2.x 为 ns，3.x 为 us），与 make_fake_market 保持一致
_DT_DTYPE = pd.bdate_range("2000-01-03", periods=1).dtype

# 列名候选（真实 CSV 的表头在实现时未读取，按常见 CRSP/Compustat 命名做容错，见 issues/A.md）
_PERMNO_COLS = ("permno", "PERMNO", "lpermno", "LPERMNO")
_MEMBER_START_COLS = ("start", "start_date", "startdate", "mbrstartdt", "from", "from_date",
                      "date_in", "entry_date", "in_date", "begdate", "begin", "start_dt")
_MEMBER_END_COLS = ("ending", "end", "end_date", "enddate", "mbrenddt", "thru", "to", "to_date",
                    "date_out", "exit_date", "out_date", "enddt", "end_dt")
_SECTOR_COLS = ("gsector",)
_INDFROM_COLS = ("indfrom",)
_INDTHRU_COLS = ("indthru",)
_TICKER_COLS = ("ticker", "tic", "symbol", "tsymbol")
_TICKER_DATE_COLS = ("date", "namedt", "start", "start_date", "from", "datadate", "nameendt",
                     "end", "end_date")


# =====================================================================
# 小工具
# =====================================================================

def _pick_col(df: pd.DataFrame, candidates: Iterable[str], what: str, path: Path) -> str:
    """大小写不敏感地找列；找不到就报清楚的错，不猜。"""
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(
        f"{path.name}: 找不到「{what}」列，候选 {list(candidates)}，实际列 {list(df.columns)}"
    )


def _parse_dates(s: pd.Series) -> pd.Series:
    """解析日期列：支持 YYYYMMDD 整数/字符串、ISO 字符串；空值 -> NaT。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).astype(_DT_DTYPE)
    raw = s.astype("string").str.strip()
    raw = raw.mask(raw.isin(["", "nan", "NaN", "NaT", "None", "<NA>"]))
    raw = raw.str.replace(r"\.0$", "", regex=True)          # 20000103.0 -> 20000103
    out = pd.Series(pd.NaT, index=s.index, dtype=_DT_DTYPE)
    is8 = raw.str.fullmatch(r"\d{8}").fillna(False).astype(bool)
    if is8.any():
        out[is8] = pd.to_datetime(raw[is8], format="%Y%m%d", errors="coerce")
    rest = ~is8 & raw.notna()
    if rest.any():
        out[rest] = pd.to_datetime(raw[rest], errors="coerce")
    bad = raw.notna() & out.isna()
    if bad.any():
        raise ValueError(f"无法解析的日期值，例如 {raw[bad].iloc[:3].tolist()}")
    return out


def _to_permno(s: pd.Series) -> pd.Series:
    # 映射表里的 permno 写成 "s10078"，先去掉开头的 s/S 再转整数
    s = s.astype(str).str.strip().str.replace(r"^[sS]", "", regex=True)
    return pd.to_numeric(s, errors="raise").astype("int64")


def _end_ts(end: Optional[str]) -> Optional[pd.Timestamp]:
    return None if end is None else pd.Timestamp(end).normalize()


def _resolve_panel_path(data_dir: Path) -> Path:
    """PRD 6.1 说 panel 在 data/ 下；build_panel.py 写在 data/ 的上一级。两处都找（见 issues/A.md）。"""
    p = data_dir / PANEL_FILE
    if p.exists():
        return p
    alt = data_dir.parent / PANEL_FILE
    if alt.exists():
        return alt
    raise FileNotFoundError(f"找不到 {PANEL_FILE}：已查 {p} 与 {alt}")


def _file_sig(p: Path) -> dict:
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


# =====================================================================
# 读原始文件
# =====================================================================

def _read_panel(path: Path, end: Optional[pd.Timestamp]) -> pd.DataFrame:
    cols = ["date", "permno", *PRICE_FIELDS]
    filters = [("date", "<=", end)] if end is not None else None
    try:
        df = pd.read_parquet(path, columns=cols, filters=filters)
    except Exception:
        # date 列若不是时间类型，过滤器可能报错；退回全读后再截断
        df = pd.read_parquet(path, columns=cols)
    df["date"] = _parse_dates(df["date"])
    if end is not None:
        df = df[df["date"] <= end]                     # 双保险：截止日之后的数据不进内存表
    df["permno"] = _to_permno(df["permno"])
    for f in PRICE_FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce").astype("float64")
    dup = df.duplicated(["date", "permno"])
    if dup.any():
        ex = df.loc[dup, ["date", "permno"]].head(3).to_dict("records")
        raise ValueError(f"panel 有 {int(dup.sum())} 条重复 (date, permno)，例如 {ex}")
    return df


def _read_membership(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8")
    pc = _pick_col(raw, _PERMNO_COLS, "permno", path)
    sc = _pick_col(raw, _MEMBER_START_COLS, "进入日期", path)
    ec = _pick_col(raw, _MEMBER_END_COLS, "退出日期", path)
    out = pd.DataFrame({
        "permno": _to_permno(raw[pc]),
        "start": _parse_dates(raw[sc]),
        "end": _parse_dates(raw[ec]),            # 空 = 至今
    })
    if out["start"].isna().any():
        raise ValueError(f"{path.name}: 有 {int(out['start'].isna().sum())} 行进入日期为空")
    return out


def _read_security(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8")
    pc = _pick_col(raw, _PERMNO_COLS, "permno", path)
    gc = _pick_col(raw, _SECTOR_COLS, "gsector", path)
    fc = _pick_col(raw, _INDFROM_COLS, "indfrom", path)
    tc = _pick_col(raw, _INDTHRU_COLS, "indthru", path)
    out = pd.DataFrame({
        "permno": _to_permno(raw[pc]),
        "gsector": pd.to_numeric(raw[gc], errors="coerce").astype("float64"),
        "gind": pd.to_numeric(raw["gind"], errors="coerce").astype("float64") if "gind" in raw else np.nan,
        "gsubind": pd.to_numeric(raw["gsubind"], errors="coerce").astype("float64") if "gsubind" in raw else np.nan,
        "indfrom": _parse_dates(raw[fc]),
        "indthru": _parse_dates(raw[tc]),        # 空 = 至今
    })
    # indfrom 为空的行无法按时点定位，丢弃（见 issues/A.md）
    return out[out["indfrom"].notna()].reset_index(drop=True)


# =====================================================================
# 构造宽表
# =====================================================================

def _build_universe(member: pd.DataFrame, dates: pd.DatetimeIndex, permnos: pd.Index,
                    has_close: pd.DataFrame) -> pd.DataFrame:
    """PRD 6.3：成分区间 [start, end]（闭区间，end 空=至今）内，且当天 panel 有 close。"""
    arr = np.zeros((len(dates), len(permnos)), dtype=bool)
    col_pos = pd.Series(np.arange(len(permnos)), index=permnos)
    dv = dates.values
    for row in member.itertuples(index=False):
        j = col_pos.get(row.permno)
        if j is None:
            continue                            # 成分表里有、panel 里没有：当天无 close，本就不进池
        i0 = np.searchsorted(dv, np.datetime64(row.start), side="left")
        i1 = len(dv) if pd.isna(row.end) else np.searchsorted(dv, np.datetime64(row.end), side="right")
        if i1 > i0:
            arr[i0:i1, j] = True
    uni = pd.DataFrame(arr, index=dates, columns=permnos)
    return uni & has_close.to_numpy()


def _build_sector(sec: pd.DataFrame, dates: pd.DatetimeIndex, permnos: pd.Index,
                  col_name: str = "gsector") -> pd.DataFrame:
    """
    PRD 6.2 按时点取 gsector：
      - t 落在 [indfrom, indthru]（indthru 空=至今）的行中取 indfrom 最晚的一行；
      - t 早于该股最早 indfrom：用最早一行；
      - 其它情况（两段分类之间的空档、最后一段 indthru 之后）：NaN（见 issues/A.md）。
    """
    arr = np.full((len(dates), len(permnos)), np.nan, dtype="float64")
    col_pos = pd.Series(np.arange(len(permnos)), index=permnos)
    dv = dates.values
    sec = sec.sort_values(["permno", "indfrom"], kind="mergesort")
    for permno, g in sec.groupby("permno", sort=False):
        j = col_pos.get(permno)
        if j is None:
            continue
        col = np.full(len(dv), np.nan)
        covered = np.zeros(len(dv), dtype=bool)
        # 按 indfrom 升序逐行覆盖 => 重叠处 indfrom 最晚的一行胜出（同 indfrom 时文件中靠后的胜出）
        for row in g.itertuples(index=False):
            i0 = np.searchsorted(dv, np.datetime64(row.indfrom), side="left")
            i1 = len(dv) if pd.isna(row.indthru) else np.searchsorted(dv, np.datetime64(row.indthru), side="right")
            if i1 > i0:
                col[i0:i1] = getattr(row, col_name)
                covered[i0:i1] = True
        # 早于最早 indfrom：用最早一行
        first_from = g["indfrom"].iloc[0]
        earliest = g[g["indfrom"] == first_from].iloc[-1]
        k = np.searchsorted(dv, np.datetime64(first_from), side="left")
        col[:k] = earliest[col_name]
        arr[:, j] = col
    return pd.DataFrame(arr, index=dates, columns=permnos)


def _build_market(data_dir: Path, end: Optional[pd.Timestamp]) -> Market:
    panel = _read_panel(_resolve_panel_path(data_dir), end)
    if panel.empty:
        raise ValueError("panel 在截止日之前没有任何数据")

    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()), name="date").astype(_DT_DTYPE)
    permnos = pd.Index(np.sort(panel["permno"].unique()).astype("int64"), name="permno")

    fields: dict[str, pd.DataFrame] = {}
    for f in PRICE_FIELDS:
        w = panel.pivot(index="date", columns="permno", values=f)
        w = w.reindex(index=dates, columns=permnos).astype("float64")
        w.index.name, w.columns.name = "date", "permno"
        fields[f] = w
    fields["ret"] = compute_ret(fields["close"])
    del panel

    member = _read_membership(data_dir / MEMBERSHIP_FILE)
    universe = _build_universe(member, dates, permnos, fields["close"].notna())
    sec = _read_security(data_dir / SECURITY_FILE)
    sector = _build_sector(sec, dates, permnos, "gsector")
    industry = _build_sector(sec, dates, permnos, "gind")
    subindustry = _build_sector(sec, dates, permnos, "gsubind")
    fwd_ret = {h: compute_fwd_ret(fields["close"], h) for h in HORIZONS}
    return Market(fields=fields, universe=universe, sector=sector, fwd_ret=fwd_ret, end=dates[-1],
                  industry=industry, subindustry=subindustry)


# =====================================================================
# 缓存
# =====================================================================

def _cache_key(data_dir: Path, end: Optional[pd.Timestamp]) -> str:
    sigs = [_file_sig(_resolve_panel_path(data_dir)),
            _file_sig(data_dir / MEMBERSHIP_FILE),
            _file_sig(data_dir / SECURITY_FILE)]
    payload = {"v": CACHE_VERSION, "end": None if end is None else end.strftime("%Y-%m-%d"),
               "files": sigs}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    tag = "all" if end is None else end.strftime("%Y%m%d")
    return f"market_{tag}_{h}"


def _frames_to_save(m: Market) -> dict[str, pd.DataFrame]:
    out = {f"field_{k}": v for k, v in m.fields.items()}
    out["universe"] = m.universe
    out["sector"] = m.sector
    if m.industry is not None:
        out["industry"] = m.industry
    if m.subindustry is not None:
        out["subindustry"] = m.subindustry
    return out                                  # fwd_ret 不存，读缓存时用 compute_fwd_ret 重算


def _write_cache(m: Market, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp_{target.name}_{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        for name, df in _frames_to_save(m).items():
            d = df.copy()
            d.columns = d.columns.astype(str)   # parquet 列名必须是字符串
            d.to_parquet(tmp / f"{name}.parquet")
        meta = {"version": CACHE_VERSION, "end": m.end.strftime("%Y-%m-%d"),
                "n_dates": len(m.dates), "n_permnos": len(m.permnos)}
        (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(tmp, target)
        except OSError:
            if not (target / "meta.json").exists():   # 别的进程没写成，才算真失败
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)   # 只清理本函数自己建的临时目录


def _read_cache(target: Path) -> Market:
    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))

    def rd(name: str) -> pd.DataFrame:
        d = pd.read_parquet(target / f"{name}.parquet")
        d.columns = pd.Index(d.columns.astype("int64"), name="permno")
        d.index = pd.DatetimeIndex(d.index, name="date").astype(_DT_DTYPE)
        return d

    fields = {f: rd(f"field_{f}").astype("float64") for f in FIELDS}
    universe = rd("universe").astype(bool)
    sector = rd("sector").astype("float64")
    industry = rd("industry").astype("float64") if (target / "industry.parquet").exists() else None
    subindustry = rd("subindustry").astype("float64") if (target / "subindustry.parquet").exists() else None
    fwd_ret = {h: compute_fwd_ret(fields["close"], h) for h in HORIZONS}
    return Market(fields=fields, universe=universe, sector=sector, fwd_ret=fwd_ret,
                  end=pd.Timestamp(meta["end"]), industry=industry, subindustry=subindustry)


# =====================================================================
# 对外接口
# =====================================================================

def load_market(data_dir: str | Path, end: str | None = RUN_DATA_END,
                cache_dir: str | Path = "cache") -> Market:
    """
    读原始数据生成 Market。end=None 表示全部数据（只有 scripts/final_test.py 可以这样调用）。
    第一次生成后缓存到 cache_dir，之后直接读缓存（原始文件大小/修改时间或 end 变了会重建）。
    """
    data_dir = Path(data_dir)
    end_ts = _end_ts(end)
    target = Path(cache_dir) / _cache_key(data_dir, end_ts)
    if (target / "meta.json").exists():
        return _read_cache(target)
    m = _build_market(data_dir, end_ts)
    _write_cache(m, target)
    return m


def split_dates(market: Market, split: Split, h: int = MAIN_HORIZON) -> pd.DatetimeIndex:
    """
    该段内可用于计算 IC 的日期：t 在 SPLITS[split]（闭区间）内，
    且 t 之后第 1+h 个交易日仍存在于 market 且仍在同一段内（标签不跨段）。
    """
    if split not in SPLITS:
        raise ValueError(f"未知的 split：{split!r}，只能是 {list(SPLITS)}")
    if int(h) != h or h < 0:
        raise ValueError(f"h 必须是非负整数，收到 {h!r}")
    h = int(h)
    s0, s1 = (pd.Timestamp(x) for x in SPLITS[split])
    dates = market.dates
    n = len(dates)
    idx = np.arange(n)
    label_pos = idx + 1 + h
    ok = label_pos < n
    label_dates = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    label_dates[ok] = dates.values[label_pos[ok]].astype("datetime64[ns]")
    in_split = (dates >= s0) & (dates <= s1)
    label_in = ok & (label_dates >= np.datetime64(s0)) & (label_dates <= np.datetime64(s1))
    return pd.DatetimeIndex(dates[in_split & label_in], name="date")


def ticker_map(data_dir: str | Path) -> dict[int, str]:
    """
    permno -> 真实 ticker，只用于展示。
    一个 permno 有多行时：有日期列则取日期最晚的一行，否则取文件中最后一行（见 issues/A.md）。
    """
    path = Path(data_dir) / TICKER_FILE
    raw = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    pc = _pick_col(raw, _PERMNO_COLS, "permno", path)
    tc = _pick_col(raw, _TICKER_COLS, "ticker", path)
    df = pd.DataFrame({"permno": _to_permno(raw[pc]), "ticker": raw[tc].astype(str).str.strip()})
    df = df[df["ticker"] != ""]
    try:
        dc = _pick_col(raw, _TICKER_DATE_COLS, "日期", path)
        df["_d"] = _parse_dates(raw.loc[df.index, dc])
        df["_o"] = np.arange(len(df))
        df = df.sort_values(["_d", "_o"], na_position="first", kind="mergesort")
    except ValueError:
        pass
    last = df.groupby("permno", sort=True)["ticker"].last()
    return {int(k): str(v) for k, v in last.items()}


def load_benchmarks(data_dir: str | Path, end: str | None = RUN_DATA_END) -> dict[str, pd.DataFrame]:
    """
    读 SPY 与 SP500_TR（PRD 6.1：MVP 只读入，不参与计算）。
    按日期升序排序（SP500_TR 原文件乱序，PRD 6.2-3），并截断到 end。
    """
    data_dir = Path(data_dir)
    end_ts = _end_ts(end)
    out: dict[str, pd.DataFrame] = {}
    for key, fname in (("spy", SPY_FILE), ("sp500_tr", SP500TR_FILE)):
        path = data_dir / fname
        df = pd.read_csv(path, encoding="utf-8")
        dc = _pick_col(df, ("date", "caldt", "datadate", "Date"), "日期", path)
        df[dc] = _parse_dates(df[dc])
        df = df.sort_values(dc, kind="mergesort").reset_index(drop=True)
        if end_ts is not None:
            df = df[df[dc] <= end_ts].reset_index(drop=True)
        out[key] = df
    return out
