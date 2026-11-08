"""
Agent 公共部分：提示词加载、上下文渲染、JSON 解析与重试。

信息可见性（PRD 12.3）由这里的渲染函数把关：
  - 因子库只渲染名称、公式、方向、逻辑、训练集指标；
  - 历史摘要只渲染假设、训练结论、训练集 IC/ICIR、归因、验证"通过/不通过"。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src import contracts as C

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
JSON_RETRY: int = 1                     # JSON 解析失败后重试次数


class AgentOutputError(RuntimeError):
    """Agent 输出无法解析或不符合约定（已重试仍失败）。"""


class ExprRetryExhausted(AgentOutputError):
    """公式校验失败，重写 MAX_EXPR_RETRY 次后仍不合规。"""


@dataclass
class AgentContext:
    """两个 Agent 共享的只读上下文。history 只给假设 Agent 用。"""
    classics: list[dict] = field(default_factory=list)      # ClassicFactor
    library: list[dict] = field(default_factory=list)       # LibraryEntry
    history: list[dict] = field(default_factory=list)       # records.history_summary 的输出


# ---------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------

def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def render_data_description() -> str:
    tr, va = C.SPLITS["train"], C.SPLITS["valid"]
    return "\n".join([
        "## 数据说明",
        "- 股票范围：S&P 500 历史成分股，日频；每天的股票池为当天在成分区间内且有收盘价的股票。",
        "- 字段（均为复权口径，缺失不填补）：",
        "  - open / high / low / close：复权开、高、低、收盘价",
        "  - volume：复权成交量（股）",
        "  - mktcap：市值，单位**千美元**（换手率 = close*volume/(mktcap*1000)）",
        "  - ret：日收益 = close(t)/close(t-1) - 1",
        "- 分组标识：sector / industry / subindustry（GICS 一级行业 / 行业 / 子行业，按时点取），只能作为 group_neutralize 的第二个参数。",
        f"- 评价口径：t 日收盘后算因子，标签为未来收益 close(t+1+{C.MAIN_HORIZON})/close(t+1) - 1；"
        f"每日在股票池内计算因子与标签的 Spearman 相关（Rank IC），当日有效股票少于 {C.MIN_STOCKS_PER_DAY} 只则跳过。",
        f"- 你能看到的回测明细只来自训练集（{tr[0]} 至 {tr[1]}）。",
        _render_thresholds(),
    ])


THRESHOLDS_FILE = Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"


def _render_thresholds() -> str:
    """门槛一律从冻结的 thresholds.yaml 读取，避免提示词里写死的数字和程序判定不一致。"""
    try:
        import yaml
        t = yaml.safe_load(THRESHOLDS_FILE.read_text(encoding="utf-8"))["train"]
        return (f"- 训练门（程序判定，不取绝对值）：IC 均值 ≥ {t['ic_mean_min']}，ICIR ≥ {t['icir_min']}（日频不年化），"
                "因子值越大必须代表预期未来收益越高。")
    except Exception:  # noqa: BLE001 —— 读不到就不写，程序判定仍以门槛文件为准
        return "- 训练门：以程序返回的门槛判定为准；因子值越大必须代表预期未来收益越高。"


def render_operators() -> str:
    win = ", ".join(str(w) for w in C.WINDOWS)
    return "\n".join([
        "## 算子清单",
        "时序算子（逐只股票沿时间往回看；窗口内数据不完整时结果为缺失）：",
        "- ts_mean(x, d) / ts_std(x, d) / ts_sum(x, d) / ts_max(x, d) / ts_min(x, d) / ts_skew(x, d)",
        "- ts_rank(x, d)：当日值在过去 d 日中的分位（0-1）",
        "- ts_delay(x, d)：d 日前的值；ts_delta(x, d)：x - ts_delay(x, d)",
        "- ts_corr(x, y, d)：过去 d 日 x 与 y 的相关系数；ts_cov(x, y, d)：协方差",
        "- decay_linear(x, d)：线性衰减加权均值（当日权重最大）",
        "- ts_argmax(x, d) / ts_argmin(x, d)：过去 d 日最大 / 最小值出现在几天前（0 = 当日）",
        "截面算子（逐日，只在当日股票池内计算）：",
        "- cs_rank(x)：截面分位（0-1）；cs_zscore(x)：截面标准化",
        "- group_neutralize(x, g)：减去当日同组均值；g 只能是 sector（一级行业）、industry（行业）、subindustry（子行业）",
        "条件与比较：",
        "- 比较 a < b、a > b、a <= b、a >= b：成立为 1、不成立为 0（每个式子只能写一个比较符）",
        "- where(cond, a, b)：cond 成立取 a，否则取 b；多个条件请嵌套 where",
        "算术：+ - * /、取负、log(x)、abs(x)、sign(x)、power(x, a)（sign(x)*|x|^a，a 为数字常量）；数字常量。"
        "log 的输入 ≤ 0、除以 0 结果为缺失。",
        f"窗口 d 只能取 {{{win}}}；ts_delay 和 ts_delta 额外允许取 1。",
    ])


def render_classics(classics: list[dict]) -> str:
    lines = ["## 经典因子库（经过检验、有经济含义的有效因子，是构建新因子的思想来源和积木）",
             "可以借鉴它们的经济机制和公式结构、在其基础上改进或组合。"]
    if not classics:
        lines.append("（空）")
    for c in classics:
        lines.append(f"- {c.get('id', '')} {c.get('name', '')}：`{c.get('expression', '')}` —— {c.get('logic', '')}")
    return "\n".join(lines)


def render_library(library: list[dict]) -> str:
    lines = ["## 当前因子库（已通过验证入库的因子）"]
    if not library:
        lines.append("（空）")
    for e in library:
        lines.append(
            f"- {e.get('id', '')} {e.get('name', '')}：`{e.get('expression', '')}`，方向 {e.get('direction', '')}，"
            f"训练集 IC {_fmt(e.get('train_ic_mean'))}、ICIR {_fmt(e.get('train_icir'))}；"
            f"假设：{e.get('hypothesis', '')}；逻辑：{e.get('logic', '')}")
    return "\n".join(lines)


def render_history(history: list[dict]) -> str:
    lines = [f"## 历史轮次摘要（最近 {C.HISTORY_ROUNDS} 轮，旧→新）"]
    if not history:
        lines.append("（暂无）")
    for h in history:
        attr = h.get("attribution") or {}
        attr_s = f"{attr.get('attribution')}（{attr.get('note', '')}）" if attr else "-"
        lines.append(
            f"- [{h.get('run_id', '')} 第 {h.get('round')} 轮] 假设：{h.get('hypothesis') or '-'}；"
            f"逻辑：{h.get('logic') or '-'}；训练结论：{h.get('train_conclusion')}；"
            f"训练集 IC {_fmt(h.get('train_ic_mean'))}、ICIR {_fmt(h.get('train_icir'))}；"
            f"归因：{attr_s}；验证：{h.get('validation') or '-'}")
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "-" if math.isnan(xf) else f"{xf:.4f}"


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """numpy 标量 → Python 标量；NaN/inf → None；递归处理 dict/list。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def dumps(obj: Any, indent: Optional[int] = None) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, indent=indent)


def extract_json(text: str) -> dict:
    """从模型原始输出里取出第一个 JSON 对象（容忍 ```json 代码块和前后多余文字）。"""
    if text is None:
        raise ValueError("输出为空")
    s = str(text).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("输出中找不到合法的 JSON 对象")


def _tag(llm: Any, **tag: Any) -> None:
    if hasattr(llm, "set_tag"):
        llm.set_tag(**tag)


def ask_json(llm: Any, system: str, user: str,
             check: Callable[[dict], dict], tag: Optional[dict] = None) -> dict:
    """
    调 LLM 并解析 JSON。解析或结构检查失败时，附上错误说明重试 JSON_RETRY 次；仍失败抛 AgentOutputError。
    LLM 调用本身抛出的异常不重试，原样上抛（由调度程序把本轮记为 skipped）。
    """
    tag = dict(tag or {})
    prompt = user
    last_err = ""
    for attempt in range(JSON_RETRY + 1):
        _tag(llm, **tag, attempt=attempt)
        raw = llm.complete(system, prompt)
        try:
            return check(extract_json(raw))
        except (ValueError, TypeError, KeyError) as e:
            last_err = str(e)
            prompt = (user + "\n\n## 注意\n上一次的输出无法按约定解析："
                      f"{last_err}。请严格只输出一个 JSON 对象，键名与格式说明完全一致，不要输出其他文字。")
    raise AgentOutputError(f"Agent 输出解析失败（已重试 {JSON_RETRY} 次）：{last_err}")


def require_str(obj: dict, key: str, nonempty: bool = True) -> str:
    if key not in obj:
        raise KeyError(f"缺少键 {key}")
    v = obj[key]
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise TypeError(f"键 {key} 应为字符串")
    v = v.strip()
    if nonempty and not v:
        raise ValueError(f"键 {key} 不能为空")
    return v
