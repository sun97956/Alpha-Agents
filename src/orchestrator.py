"""
调度程序（PRD 第 11 节）。普通程序：按固定顺序调用各环节、传递数据、控制循环、失败时跳过本轮。

其他线的函数（表达式校验/计算、评估、门槛判定、参照因子存取）通过 Engine 注入；
默认实现 default_engine() 在调用时才导入 src.expr / src.evaluate / src.refstore，
所以本模块不依赖其他线的代码即可导入和测试。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import yaml

from src.contracts import (MAX_ADJUST, MAX_EXPR_RETRY, RUN_DATA_END, HISTORY_ROUNDS,
                           LibraryEntry, Market, RoundRecord, VersionRecord)
from src.agents import factor_agent, hypothesis_agent
from src.agents.common import AgentContext, ExprRetryExhausted
from src.agents.llm import LoggedLLM
from src import records

PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------

def load_settings(path: PathLike = PROJECT_ROOT / "config" / "settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_classics(path: PathLike = PROJECT_ROOT / "library" / "classic_factors.yaml") -> list[dict]:
    """读 library/classic_factors.yaml（线 B 维护）。兼容顶层是列表或 {factors: [...]}。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if isinstance(data, dict):
        data = data.get("factors") or data.get("classic_factors") or []
    out = []
    for item in data:
        out.append({"id": str(item["id"]), "name": str(item.get("name", "")),
                    "expression": str(item["expression"]), "logic": str(item.get("logic", ""))})
    return out



def resolve(path: PathLike, root: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


# ---------------------------------------------------------------------
# 依赖注入
# ---------------------------------------------------------------------

@dataclass
class Engine:
    validate: Callable[[str], None]
    compute: Callable[[str, Market], pd.DataFrame]
    evaluate: Callable[..., dict]          # (factor, market, split, refs=None) -> Report
    gate_train: Callable[[dict, dict], dict]
    gate_valid: Callable[[dict, dict, dict], dict]
    save_ref: Callable[..., None]          # (ref_id, factor, market, cache_dir) -> None
    load_refs: Callable[..., dict]         # (cache_dir) -> dict[str, DataFrame]


def default_engine() -> Engine:
    from src import evaluate as ev, expr, refstore
    return Engine(validate=expr.validate, compute=expr.compute, evaluate=ev.evaluate,
                  gate_train=ev.gate_train, gate_valid=ev.gate_valid,
                  save_ref=refstore.save_ref, load_refs=refstore.load_refs)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InjectedError(RuntimeError):
    """人为注入的错误（PRD 15-3 验收用）。"""


# ---------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------

class Orchestrator:
    def __init__(self, market: Market, llm: Any, engine: Engine, thresholds: dict,
                 thresholds_sha256: str, classics: list[dict],
                 refs: Optional[dict] = None, runs_dir: PathLike = "runs",
                 library_path: PathLike = "factor_library.jsonl",
                 cache_dir: PathLike = "cache", run_id: Optional[str] = None,
                 inject_error_round: Optional[int] = None):
        # 隔离：运行期间加载的数据不得晚于 RUN_DATA_END
        if pd.Timestamp(market.end) > pd.Timestamp(RUN_DATA_END) or \
                pd.Timestamp(market.dates.max()) > pd.Timestamp(RUN_DATA_END):
            raise ValueError(f"运行期间的数据不能晚于 {RUN_DATA_END}（当前截止 {market.end}）")
        self.market = market
        self.engine = engine
        self.thresholds = thresholds
        self.thresholds_sha256 = thresholds_sha256
        self.classics = list(classics)
        self.refs = dict(refs or {})
        self.runs_dir = Path(runs_dir)
        self.library_path = Path(library_path)
        self.cache_dir = cache_dir
        self.run_id = run_id or new_run_id()
        self.inject_error_round = inject_error_round
        self.recorder = records.RunRecorder(self.runs_dir, self.run_id)
        self.llm = LoggedLLM(llm, self.recorder.llm_log, run_id=self.run_id)

    # ---- 上下文 ----
    def _context(self) -> AgentContext:
        hist = records.history_summary(records.all_round_records(self.runs_dir), HISTORY_ROUNDS)
        return AgentContext(classics=self.classics,
                            library=records.read_library(self.library_path),
                            history=hist)

    # ---- 一轮 ----
    def run_round(self, round_no: int) -> dict:
        rec: dict = RoundRecord(
            run_id=self.run_id, round=round_no, timestamp=_now(),
            thresholds_sha256=self.thresholds_sha256, hypothesis=None, factor_name=None,
            versions=[], train_conclusion="skipped", cannot_build_reason=None,
            attribution=None, validation=None, library_id=None, error=None)
        tag = {"round": round_no}
        try:
            if self.inject_error_round is not None and round_no == self.inject_error_round:
                raise InjectedError("人为注入的错误（验收用）")
            self._run_round_body(rec, tag)
        except Exception as e:  # noqa: BLE001 —— PRD 11：任一步异常，本轮记为跳过
            rec["train_conclusion"] = "skipped"
            rec["library_id"] = None
            rec["error"] = f"{type(e).__name__}: {e}"
        self.recorder.append_round(rec)
        return rec

    def _run_round_body(self, rec: dict, tag: dict) -> None:
        eng, mkt = self.engine, self.market
        ctx = self._context()

        # 2. 假设
        hyp = hypothesis_agent.propose(self.llm, ctx, tag=tag)
        rec["hypothesis"] = hyp

        # 3. 构建 v0
        prop = factor_agent.build(self.llm, ctx, hyp, eng.validate, tag={**tag, "version": 0})
        if prop["status"] == "cannot_build":
            rec["train_conclusion"] = "cannot_build"
            rec["cannot_build_reason"] = prop["reason"]
            rec["factor_name"] = prop["name"] or None
            return
        rec["factor_name"] = prop["name"]

        # 4. 内层循环（训练集）
        agent_versions: list[dict] = []          # 给因子 Agent 看的：只有训练集明细
        passed = False
        version = 0
        factor = train_report = None
        while True:
            factor = eng.compute(prop["expression"], mkt)
            train_report = eng.evaluate(factor, mkt, "train", self.refs)
            gate = eng.gate_train(train_report, self.thresholds)
            path = self.recorder.save_report(rec["round"], version, train_report, gate)
            rec["versions"].append(VersionRecord(
                version=version, expression=prop["expression"], explanation=prop["explanation"],
                change_note=prop["change_note"], ic_mean=_f(train_report.get("ic_mean")),
                icir=_f(train_report.get("icir")), train_gate_passed=bool(gate["passed"]),
                report_path=path, error=None))
            agent_versions.append({"version": version, "expression": prop["expression"],
                                   "explanation": prop["explanation"],
                                   "change_note": prop["change_note"],
                                   "report": train_report, "gate": gate})
            if gate["passed"]:
                passed = True
                break
            if version >= MAX_ADJUST:
                break
            version += 1
            try:
                new = factor_agent.adjust(self.llm, ctx, hyp, agent_versions, eng.validate,
                                          tag={**tag, "version": version})
            except Exception as e:  # noqa: BLE001
                # 调整阶段失败（公式重写用完、LLM 空回复或超时等）：记一个出错的版本，
                # 结束内层循环，按未达标处理；已算出的版本保留，整轮不作废
                rec["versions"].append(VersionRecord(
                    version=version, expression="", explanation="", change_note="",
                    ic_mean=None, icir=None, train_gate_passed=None, report_path=None,
                    error=f"{type(e).__name__}: {e}"))
                break
            if new["status"] == "cannot_build":
                # 调整阶段 Agent 放弃：结束内层循环，按未达标处理，原因保留
                rec["cannot_build_reason"] = new["reason"]
                break
            prop = new

        if not passed:
            rec["train_conclusion"] = "fail"
            rec["attribution"] = factor_agent.attribute(self.llm, hyp, agent_versions, tag=tag)
            return
        rec["train_conclusion"] = "pass"
        rec["factor_name"] = prop["name"] or rec["factor_name"]

        # 5. 外层验证：只取结论，指标不落盘、不进记录、不给 Agent
        valid_report = eng.evaluate(factor, mkt, "valid", None)
        vgate = eng.gate_valid(valid_report, train_report, self.thresholds)
        valid_passed = bool(vgate["passed"])
        del valid_report, vgate
        rec["validation"] = "pass" if valid_passed else "fail"
        if not valid_passed:
            return

        # 6. 入库：先存参照值，再写因子库，最后更新去重参照集
        lib = records.read_library(self.library_path)
        lib_id = records.next_library_id(lib)
        ic = float(train_report["ic_mean"])
        entry = LibraryEntry(
            id=lib_id, name=prop["name"], expression=prop["expression"],
            direction=1 if ic > 0 else -1, hypothesis=hyp["hypothesis"], logic=hyp["logic"],
            explanation=prop["explanation"], train_ic_mean=ic,
            train_icir=float(train_report["icir"]), validation="pass",
            run_id=self.run_id, round=rec["round"], created_at=_now())
        eng.save_ref(lib_id, factor, mkt, self.cache_dir)
        records.append_library(self.library_path, entry)
        rec["library_id"] = lib_id
        try:
            self.refs.update(eng.load_refs(self.cache_dir))
        except Exception:  # noqa: BLE001 —— 重新加载失败不影响本轮结论，下次运行会读到
            pass

    # ---- 全部轮次 ----
    def run(self, n_rounds: int) -> list[dict]:
        return [self.run_round(i) for i in range(1, int(n_rounds) + 1)]


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v
