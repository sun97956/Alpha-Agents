"""
运行记录与因子库（PRD 第 12 节）。

- runs/<run_id>/run_log.jsonl：每轮一行，只追加不修改。
- runs/<run_id>/reports/round_XXX_vY.json：每版公式的训练集完整明细（含训练门判定）。
  验证集明细不落盘、不进运行记录，只记"pass"/"fail"。
- factor_library.jsonl：跨运行累积，只追加。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from src.contracts import HISTORY_ROUNDS, LibraryEntry, RoundRecord
from src.agents.common import to_jsonable

PathLike = Union[str, Path]

ROUND_KEYS: tuple[str, ...] = tuple(RoundRecord.__annotations__.keys())
LIBRARY_KEYS: tuple[str, ...] = tuple(LibraryEntry.__annotations__.keys())


# ---------------------------------------------------------------------
# 通用 jsonl
# ---------------------------------------------------------------------

def append_jsonl(path: PathLike, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")


def read_jsonl(path: PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------
# 运行记录
# ---------------------------------------------------------------------

class RunRecorder:
    def __init__(self, runs_dir: PathLike, run_id: str):
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id
        self.run_dir = self.runs_dir / run_id
        self.reports_dir = self.run_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.run_log = self.run_dir / "run_log.jsonl"
        self.llm_log = self.run_dir / "llm_log.jsonl"

    def save_report(self, round_no: int, version: int, report: dict,
                    gate: Optional[dict] = None) -> str:
        """保存训练集明细，返回写进 VersionRecord.report_path 的路径（posix 风格）。"""
        if report.get("split") not in (None, "train"):
            raise ValueError("只允许保存训练集明细")
        path = self.reports_dir / f"round_{round_no:03d}_v{version}.json"
        payload = {"run_id": self.run_id, "round": round_no, "version": version,
                   "report": report, "train_gate": gate}
        path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path.as_posix()

    def append_round(self, record: dict) -> None:
        missing = [k for k in ROUND_KEYS if k not in record]
        if missing:
            raise ValueError(f"运行记录缺少字段：{missing}")
        append_jsonl(self.run_log, {k: record[k] for k in ROUND_KEYS})

    def rounds(self) -> list[dict]:
        return read_jsonl(self.run_log)


def all_round_records(runs_dir: PathLike) -> list[dict]:
    """读取 runs_dir 下所有运行的 run_log.jsonl，按 (timestamp, run_id, round) 排序。"""
    recs: list[dict] = []
    root = Path(runs_dir)
    if root.exists():
        for p in sorted(root.glob("*/run_log.jsonl")):
            if p.parent.name.startswith("mock_"):     # 模拟运行不算历史
                continue
            recs.extend(read_jsonl(p))
    recs.sort(key=lambda r: (str(r.get("timestamp", "")), str(r.get("run_id", "")), int(r.get("round", 0))))
    return recs


def final_train_version(rec: dict) -> Optional[dict]:
    """本轮最后一个有训练集指标的版本（达标时就是达标的那一版）。"""
    for v in reversed(rec.get("versions") or []):
        if v.get("ic_mean") is not None:
            return v
    return None


def history_summary(records: Iterable[dict], n: int = HISTORY_ROUNDS) -> list[dict]:
    """
    假设 Agent 可见的历史摘要。只保留假设、机制、期限、训练结论、
    最后一版训练 IC/ICIR 和验证结论。
    """
    recs = list(records)[-n:] if n > 0 else []
    out = []
    for r in recs:
        hyp = r.get("hypothesis") or {}
        v = final_train_version(r)
        val = r.get("validation")
        out.append({
            "run_id": r.get("run_id"),
            "round": r.get("round"),
            "hypothesis": hyp.get("hypothesis"),
            "mechanism": hyp.get("mechanism"),
            "horizon": hyp.get("horizon"),
            "train_conclusion": r.get("train_conclusion"),
            "train_ic_mean": v.get("ic_mean") if v else None,
            "train_icir": v.get("icir") if v else None,
            "validation": val if val in ("pass", "fail") else None,
        })
    return out


def mechanism_table(records: Iterable[dict], runs_dir: PathLike | None = None) -> list[dict]:
    """
    按机制类别汇总全部历史轮次，只使用记录和训练报告中的数值。
    """
    from collections import Counter, OrderedDict
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in records:
        hyp = r.get("hypothesis") or {}
        mech = hyp.get("mechanism")
        if not mech or r.get("train_conclusion") == "skipped":
            continue
        groups.setdefault(mech, []).append(r)

    def fmt(counter: Counter) -> str:
        return "，".join(f"{k}:{v}" for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))) or "-"

    def report_for(v: dict) -> dict:
        path = Path(str(v.get("report_path", "")))
        if not path.exists() and runs_dir is not None:
            path = Path(runs_dir) / path
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("report", {})
        except (OSError, ValueError, TypeError):
            return {}

    rows = []
    for mech, rs in groups.items():
        initial = [v.get("ic_mean") for r in rs for v in (r.get("versions") or [])[:1]
                   if v.get("ic_mean") is not None]
        best_ics = [v.get("ic_mean") for r in rs for v in (r.get("versions") or [])
                    if v.get("ic_mean") is not None]
        horizons = []
        for r in rs:
            v = final_train_version(r)
            report = report_for(v) if v else {}
            hs = report.get("horizons") or {}
            valid_h = [(float(data["ic_mean"]), h) for h, data in hs.items()
                       if data.get("ic_mean") is not None]
            if valid_h:
                horizons.append(max(valid_h)[1] + "日")
        rows.append({
            "mechanism": mech,
            "tries": len(rs),
            "train_pass": sum(r.get("train_conclusion") == "pass" for r in rs),
            "valid_pass": sum(r.get("validation") == "pass" for r in rs),
            "declared": fmt(Counter(f"{(r.get('hypothesis') or {}).get('horizon')}日" for r in rs)),
            "best_horizon": fmt(Counter(horizons)),
            "initial_ic_median": float(np.median(initial)) if initial else None,
            "best_ic": max(best_ics) if best_ics else None,
        })
    rows.sort(key=lambda x: -x["tries"])
    return rows


# ---------------------------------------------------------------------
# 因子库
# ---------------------------------------------------------------------

_ID_RE = re.compile(r"^F(\d+)$")


def read_library(path: PathLike) -> list[dict]:
    return read_jsonl(path)


def next_library_id(entries: Iterable[dict]) -> str:
    mx = 0
    for e in entries:
        m = _ID_RE.match(str(e.get("id", "")))
        if m:
            mx = max(mx, int(m.group(1)))
    return f"F{mx + 1:04d}"


def append_library(path: PathLike, entry: dict) -> None:
    missing = [k for k in LIBRARY_KEYS if k not in entry]
    if missing:
        raise ValueError(f"因子库条目缺少字段：{missing}")
    if entry["validation"] != "pass":
        raise ValueError("只有验证通过的因子可以入库")
    if entry["direction"] not in (1, -1):
        raise ValueError("direction 只能是 1 或 -1")
    append_jsonl(path, {k: entry[k] for k in LIBRARY_KEYS})
