"""假设 Agent（PRD 8.1）：提出一条可检验的假设 + 经济逻辑。"""
from __future__ import annotations

from typing import Any, Optional

from src.contracts import HOLDING_PERIODS, MECHANISMS, Hypothesis
from src.agents.common import (AgentContext, ask_json, load_prompt, render_experience,
                               render_history, render_hypothesis_data_description,
                               render_library_directory, require_str)


def build_user_prompt(ctx: AgentContext) -> str:
    """输入：环境说明、机制经验表、已试假设清单、当前因子库目录。"""
    return "\n\n".join([
        render_hypothesis_data_description(),
        render_experience(ctx.experience),
        render_history(ctx.history),
        render_library_directory(ctx.library),
        "请提出一条新的假设。",
    ])


def _check(obj: dict) -> Hypothesis:
    mech = require_str(obj, "mechanism")
    if mech not in MECHANISMS:
        raise ValueError(f"mechanism 只能是：{'、'.join(MECHANISMS)}")
    try:
        h = int(float(str(obj.get("horizon")).strip()))
    except (TypeError, ValueError):
        h = -1
    if h not in HOLDING_PERIODS:
        raise ValueError(f"horizon 只能是 {HOLDING_PERIODS} 之一")
    return Hypothesis(hypothesis=require_str(obj, "hypothesis"), logic=require_str(obj, "logic"),
                      mechanism=mech, horizon=h)


def propose(llm: Any, ctx: AgentContext, tag: Optional[dict] = None) -> Hypothesis:
    return ask_json(llm, load_prompt("hypothesis"), build_user_prompt(ctx), _check,
                    tag={"agent": "hypothesis", "call": "propose", **(tag or {})})
