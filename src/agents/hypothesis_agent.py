"""假设 Agent（PRD 8.1）：提出一条可检验的假设 + 经济逻辑。"""
from __future__ import annotations

from typing import Any, Optional

from src.contracts import Hypothesis
from src.agents.common import (AgentContext, ask_json, load_prompt, render_classics,
                               render_data_description, render_history, render_library,
                               render_operators, require_str)


def build_user_prompt(ctx: AgentContext) -> str:
    """输入：数据说明、算子清单、经典因子库、当前因子库、历史轮次摘要。"""
    return "\n\n".join([
        render_data_description(),
        render_operators(),
        render_classics(ctx.classics),
        render_library(ctx.library),
        render_history(ctx.history),
        "请提出一条新的假设。",
    ])


def _check(obj: dict) -> Hypothesis:
    return Hypothesis(hypothesis=require_str(obj, "hypothesis"), logic=require_str(obj, "logic"))


def propose(llm: Any, ctx: AgentContext, tag: Optional[dict] = None) -> Hypothesis:
    return ask_json(llm, load_prompt("hypothesis"), build_user_prompt(ctx), _check,
                    tag={"agent": "hypothesis", "call": "propose", **(tag or {})})
