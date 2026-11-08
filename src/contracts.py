"""
冻结接口（contracts）。所有工作线 import 这里的定义，任何开发 Agent 不得修改本文件。
发现接口有问题：写进 issues/<你的线>.md，由人决定是否修改。

内容：
  1. 方法论常量（字段、算子、窗口、收益期限、三段切分）
  2. 标签与日收益的唯一参考实现（compute_ret / compute_fwd_ret）
  3. 数据结构 Market
  4. 各模块必须实现的函数签名（只写签名与说明，实现放在各自模块）
  5. 回测明细、门槛判定、Agent 输出、运行记录、因子库条目的 JSON 结构
  6. 假数据生成 make_fake_market()，供所有单元测试使用

业务规则见 PRD.md，本文件只定义"形状"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, TypedDict

import numpy as np
import pandas as pd

# =====================================================================
# 1. 方法论常量（PRD 第 6、7、9 节）
# =====================================================================

FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "mktcap", "ret")

# 算子名 -> 数据参数个数（窗口参数另计）
TS_OPS: dict[str, int] = {
    "ts_mean": 1, "ts_std": 1, "ts_sum": 1, "ts_max": 1, "ts_min": 1, "ts_skew": 1,
    "ts_rank": 1, "ts_delay": 1, "ts_delta": 1, "ts_corr": 2,
    "ts_cov": 2, "decay_linear": 1, "ts_argmax": 1, "ts_argmin": 1,
}
CS_OPS: dict[str, int] = {"cs_rank": 1, "cs_zscore": 1}
GROUP_OPS: dict[str, int] = {"group_neutralize": 1}      # 第二个参数为分组标识符（见 GROUP_KEYS）
ARITH_FUNCS: dict[str, int] = {"log": 1, "abs": 1, "sign": 1}
POWER_OPS: dict[str, int] = {"power": 1}                # power(x, a)：a 为数字常量；sign(x)*|x|^a
COND_OPS: dict[str, int] = {"where": 3}                 # where(cond, a, b)：cond>0 取 a，否则取 b
COMPARE_OPS: tuple[str, ...] = ("<", ">", "<=", ">=")   # 比较结果为 1.0 / 0.0
GROUP_KEYS: tuple[str, ...] = ("sector", "industry", "subindustry")   # GICS 一级 / 三级 / 四级

WINDOWS: tuple[int, ...] = (2, 3, 5, 10, 20, 60, 120, 250)
DELAY_WINDOWS: tuple[int, ...] = (1,) + WINDOWS            # 仅 ts_delay / ts_delta 可用 1

MAIN_HORIZON: int = 5                                      # 门槛只看 h=5
HORIZONS: tuple[int, ...] = (1, 5, 20)                     # 明细里的 IC 衰减
MIN_STOCKS_PER_DAY: int = 100                              # 当日有效股票少于此数则跳过该日
N_QUANTILES: int = 5

Split = Literal["train", "valid", "test"]
SPLITS: dict[str, tuple[str, str]] = {                     # 闭区间，日历日
    "train": ("2000-01-01", "2016-12-31"),
    "valid": ("2017-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-09-30"),
}
RUN_DATA_END: str = "2021-12-31"                           # 运行期间加载数据的截止日
REF_SPLIT: Split = "train"                                 # 参照因子值只存训练集区间

MAX_ADJUST: int = 3                                        # 每因子最多调整 3 次 => v0..v3
MAX_EXPR_RETRY: int = 2                                    # 公式校验失败后的重写次数，不计入调整
HISTORY_ROUNDS: int = 30                                   # 假设 Agent 可见的历史轮数


# =====================================================================
# 2. 标签与日收益的唯一参考实现（任何模块都调用这里，不得自写）
# =====================================================================

def compute_ret(close: pd.DataFrame) -> pd.DataFrame:
    """日收益 ret[t] = close[t] / close[t-1] - 1。宽表、按全局交易日历移位；前一日缺失则为 NaN。"""
    return (close / close.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)


def compute_fwd_ret(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """未来收益 fwd[t] = close[t+1+h] / close[t+1] - 1（t 日收盘后算因子，t+1 才能交易）。"""
    return (close.shift(-(1 + h)) / close.shift(-1) - 1.0).replace([np.inf, -np.inf], np.nan)


# =====================================================================
# 3. 数据结构
# =====================================================================

@dataclass
class Market:
    """
    所有宽表形状一致：
      index   = pd.DatetimeIndex(name="date")，升序，全部交易日（截至 end）
      columns = pd.Index(dtype=int64, name="permno")，升序
    """
    fields: dict[str, pd.DataFrame]    # FIELDS 中每个字段，float64；缺失为 NaN，不填补
    universe: pd.DataFrame             # bool；PRD 6.3 的每日股票池
    sector: pd.DataFrame               # float64；当日 GICS gsector（按时点取，PRD 6.2），无则 NaN
    fwd_ret: dict[int, pd.DataFrame]   # key ∈ HORIZONS，由 compute_fwd_ret 生成
    end: pd.Timestamp                  # 数据截止日
    industry: Optional[pd.DataFrame] = None      # float64；当日 GICS gind（按时点取）
    subindustry: Optional[pd.DataFrame] = None   # float64；当日 GICS gsubind（按时点取）

    def group(self, key: str) -> pd.DataFrame:
        g = getattr(self, key, None)
        if g is None:
            raise ValueError(f"Market 没有分组 {key}")
        return g

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    @property
    def permnos(self) -> pd.Index:
        return self.fields["close"].columns


# =====================================================================
# 4. 各模块必须实现的函数签名（实现写在对应模块，签名必须一致）
# =====================================================================
#
# ---- 线 A：src/data.py ----
# def load_market(data_dir: str | Path, end: str | None = RUN_DATA_END,
#                 cache_dir: str | Path = "cache") -> Market
#     读原始数据生成 Market。end=None 表示全部数据（只有 scripts/final_test.py 可以这样调用）。
#     第一次生成后缓存到 cache_dir，之后直接读缓存。
#
# def split_dates(market: Market, split: Split, h: int = MAIN_HORIZON) -> pd.DatetimeIndex
#     返回该段内可用于计算 IC 的日期：t 在 SPLITS[split] 内，且 t+1+h 个交易日后仍在同一段内并存在于 market。
#
# def ticker_map(data_dir: str | Path) -> dict[int, str]
#     permno -> 真实 ticker，只用于展示。
#
# ---- 线 B：src/expr.py（算子实现放在 src/operators.py）----
# class ExprError(ValueError)        错误信息用中文写清楚原因，会直接返回给因子 Agent
# def validate(expr: str) -> None    不合规则抛 ExprError
# def compute(expr: str, market: Market) -> pd.DataFrame
#     返回与 market 同形状的 float64 宽表；只保留 universe 为 True 的位置，其余 NaN；inf -> NaN。
#
# ---- 线 C：src/evaluate.py ----
# def load_thresholds(path: str | Path = "config/thresholds.yaml") -> dict
# def thresholds_sha256(path: str | Path = "config/thresholds.yaml") -> str
# def evaluate(factor: pd.DataFrame, market: Market, split: Split,
#              refs: dict[str, pd.DataFrame] | None = None) -> Report
#     refs 为 None 时 max_ref_corr / max_ref_id 为 None。refs 只在 split="train" 时传入。
# def gate_train(report: Report, thresholds: dict) -> GateResult
# def gate_valid(valid_report: Report, train_report: Report, thresholds: dict) -> GateResult
#
# ---- 线 C：src/refstore.py（参照因子值，PRD 第 13 节）----
# def to_ranks(factor: pd.DataFrame, market: Market) -> pd.DataFrame
#     截取 REF_SPLIT 区间，逐日在股票池内做截面百分位排序，float32。
# def save_ref(ref_id: str, factor: pd.DataFrame, market: Market, cache_dir="cache") -> None
#     写入 cache/ref_values/<ref_id>.parquet（存 to_ranks 的结果）
# def load_refs(cache_dir="cache") -> dict[str, pd.DataFrame]
#
# ---- 线 D：src/agents/、src/records.py、src/orchestrator.py、scripts/ ----
# 见下方 LLM 协议与第 5 部分的 JSON 结构。


class LLM(Protocol):
    """换模型 = 新写一个实现此协议的类；系统其他部分不变。"""

    def complete(self, system: str, user: str) -> str:
        """返回模型的原始文本输出。调用方负责解析 JSON。"""
        ...


# =====================================================================
# 5. JSON 结构
# =====================================================================

class YearStat(TypedDict):
    year: int
    ic_mean: float
    icir: float
    ic_pos_ratio: float
    n_days: int


class Report(TypedDict):
    """回测明细（PRD 第 9 节）。只有 split="train" 的明细会给到因子 Agent。"""
    split: str
    n_days: int                      # 有效 IC 天数
    coverage_mean: float             # 平均每日有因子值的股票数
    ic_mean: float                   # 日 Rank IC 均值（h=MAIN_HORIZON）
    ic_std: float
    icir: float                      # ic_mean / ic_std，日频不年化
    t_stat: float                    # ic_mean / ic_std * sqrt(n_days)
    ic_pos_ratio: float
    yearly: list[YearStat]
    decay: dict[str, float]          # {"1": .., "5": .., "20": ..}，各 h 用各自的 split_dates
    quantile_ret: list[float]        # N_QUANTILES 个，第 1 个 = 因子值最低组，平均 fwd_ret_5
    long_short: float                # 最高组 - 最低组
    turnover_autocorr: float         # 相邻两日因子截面 Rank 相关的均值
    max_ref_corr: Optional[float]    # 与参照因子日截面 Spearman 相关均值，取绝对值最大者（保留符号）
    max_ref_id: Optional[str]


class GateCheck(TypedDict):
    name: str                        # 如 "ic_mean"、"icir"、"max_ref_corr"
    value: float
    threshold: float
    passed: bool


class GateResult(TypedDict):
    passed: bool
    checks: list[GateCheck]


# ---- Agent 输出 ----

class Hypothesis(TypedDict):
    hypothesis: str
    logic: str


class FactorProposal(TypedDict):
    status: Literal["ok", "cannot_build"]
    name: str
    expression: str                  # status="cannot_build" 时为 ""
    explanation: str
    reason: str                      # status="cannot_build" 时必填
    change_note: str                 # 调整时必填；v0 为 ""


class Attribution(TypedDict):
    attribution: Literal["formula", "hypothesis"]
    note: str


# ---- 记录 ----

class VersionRecord(TypedDict):
    version: int                     # 0..MAX_ADJUST
    expression: str
    explanation: str
    change_note: str
    ic_mean: Optional[float]
    icir: Optional[float]
    train_gate_passed: Optional[bool]
    report_path: Optional[str]       # runs/<run_id>/reports/round_XXX_vY.json
    error: Optional[str]


class RoundRecord(TypedDict):
    """run_log.jsonl 中的一行（PRD 12.1）。"""
    run_id: str
    round: int
    timestamp: str                   # ISO 8601
    thresholds_sha256: str
    hypothesis: Optional[Hypothesis]
    factor_name: Optional[str]
    versions: list[VersionRecord]
    train_conclusion: Literal["pass", "fail", "cannot_build", "skipped"]
    cannot_build_reason: Optional[str]
    attribution: Optional[Attribution]
    validation: Optional[Literal["pass", "fail"]]   # 只记结论，不记指标
    library_id: Optional[str]
    error: Optional[str]


class LibraryEntry(TypedDict):
    """factor_library.jsonl 中的一行（PRD 12.2）。"""
    id: str                          # 如 "F0001"
    name: str
    expression: str
    direction: Literal[1, -1]        # 训练集 IC 的符号
    hypothesis: str
    logic: str
    explanation: str
    train_ic_mean: float
    train_icir: float
    validation: Literal["pass"]
    run_id: str
    round: int
    created_at: str


class ClassicFactor(TypedDict):
    """library/classic_factors.yaml 中的一项（PRD 第 13 节）。"""
    id: str                          # 如 "C01_rev5"
    name: str
    expression: str
    logic: str


# =====================================================================
# 6. 假数据（所有单元测试用它，不依赖真实数据）
# =====================================================================

SECTOR_CODES: tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)


def make_fake_market(n_days: int = 300, n_stocks: int = 150, seed: int = 0,
                     start: str = "2016-06-01") -> Market:
    """
    生成与真实 Market 形状、类型完全一致的小数据集。
    特点（方便写测试）：
      - 默认日期跨越 train/valid 分界（2016-12-31），可测 split_dates 和标签不跨段；
      - 有晚上市、提前退市、个别缺失值、股票池进出、一只股票中途换行业；
      - 股票数默认 150，保证每日有效股票数 ≥ MIN_STOCKS_PER_DAY；
      - 埋了一个短期反转信号：下一日收益与过去 5 日收益负相关，
        所以 -ts_sum(ret, 5) 的 Rank IC 应显著为正。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, name="date")
    permnos = pd.Index(np.arange(10001, 10001 + n_stocks, dtype=np.int64), name="permno")

    # 收益：噪声 + 短期反转
    noise = rng.normal(0.0, 0.02, size=(n_days, n_stocks))
    rets = np.zeros_like(noise)
    for t in range(n_days):
        if t >= 5:
            past = rets[t - 5:t].sum(axis=0)
            z = (past - past.mean()) / (past.std() + 1e-12)
            rets[t] = noise[t] - 0.004 * z
        else:
            rets[t] = noise[t]
    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * (1 + rng.normal(0, 0.005, size=close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, size=close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, size=close.shape)))
    volume = np.exp(rng.normal(14, 0.5, size=close.shape))
    shares = rng.uniform(1e8, 5e9, size=n_stocks)
    mktcap = close * shares / 1000.0                       # 千美元，与 CRSP 口径一致

    def wide(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(a, index=dates, columns=permnos, dtype="float64")

    raw = {"open": wide(open_), "high": wide(high), "low": wide(low),
           "close": wide(close), "volume": wide(volume), "mktcap": wide(mktcap)}

    # 晚上市 / 提前退市：之前 / 之后整行无记录
    exists = np.ones((n_days, n_stocks), dtype=bool)
    exists[: n_days // 3, 0] = False                       # 第 1 只晚上市
    exists[2 * n_days // 3:, 1] = False                    # 第 2 只提前退市
    for k in raw:
        raw[k] = raw[k].where(exists)
    raw["open"].iloc[10, 2] = np.nan                       # 个别缺失值，不填补

    fields = dict(raw)
    fields["ret"] = compute_ret(fields["close"])

    # 股票池：有记录 且 在成分区间内（第 3 只中途被调出）
    member = np.ones((n_days, n_stocks), dtype=bool)
    member[n_days // 2:, 3] = False
    universe = pd.DataFrame(member & exists, index=dates, columns=permnos)

    # 行业：按股票随机分配，第 5 只中途换行业（模拟按时点取）
    sec = rng.choice(SECTOR_CODES, size=n_stocks).astype("float64")
    sector = pd.DataFrame(np.tile(sec, (n_days, 1)), index=dates, columns=permnos)
    sector.iloc[n_days // 2:, 4] = 45.0 if sec[4] != 45.0 else 20.0
    # 行业 / 子行业：在一级行业下再细分，编码方式与 GICS 一致（gind = gsector*10000 + ...）
    sub_ind = rng.integers(1, 4, size=n_stocks)
    sub_sub = rng.integers(1, 3, size=n_stocks)
    industry = sector * 10000 + np.tile(sub_ind * 1010, (n_days, 1))
    subindustry = industry * 100 + np.tile(sub_sub * 10, (n_days, 1))

    fwd_ret = {h: compute_fwd_ret(fields["close"], h) for h in HORIZONS}
    return Market(fields=fields, universe=universe, sector=sector,
                  fwd_ret=fwd_ret, end=dates[-1], industry=industry, subindustry=subindustry)
