"""
因子 Agent（PRD 8.2）：构建、调整、归因三种调用。

- 不接触原始行情，只看数据说明和训练集明细（调用方保证只传入 split="train" 的明细）。
- 公式没通过 validate 时，把错误信息返回给 Agent 重写，最多 MAX_EXPR_RETRY 次（不计入调整次数）；
  仍不合规抛 ExprRetryExhausted。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from src.contracts import (HORIZONS, MAX_EXPR_RETRY, REVIEW_CONCLUSIONS, Attribution, FactorProposal,
                           Hypothesis)
from src.agents.common import (AgentContext, ExprRetryExhausted, ask_json, dumps, load_prompt,
                               render_classics, render_data_description, render_library,
                               render_operators, require_str)

Validator = Callable[[str], None]


# ---------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------

def _hyp_block(h: Hypothesis) -> str:
    return (f"## 本轮假设\n- 假设：{h['hypothesis']}\n- 经济逻辑：{h['logic']}"
            f"\n- 机制类别：{h.get('mechanism', '-')}\n- 声明的持有期：{h.get('horizon', '-')} 日"
            "（训练门、验证门都按这个持有期判定）")


def render_versions(versions: list[dict]) -> str:
    """versions: [{version, expression, explanation, change_note, report, gate}]，report 必须是训练集明细。"""
    lines = ["## 本轮各版公式及训练集明细"]
    for v in versions:
        lines.append(f"### v{v['version']}")
        lines.append(f"- 公式：`{v.get('expression', '')}`")
        lines.append(f"- 说明：{v.get('explanation', '')}")
        if v.get("change_note"):
            lines.append(f"- 改动说明：{v['change_note']}")
        if v.get("error"):
            lines.append(f"- 错误：{v['error']}")
        rep, gate = v.get("report"), v.get("gate")
        if v is versions[-1]:
            # 只有最新版本给完整明细，避免提示词随版本数膨胀、模型过度思考
            if rep is not None:
                lines.append("- 训练集回测明细：")
                lines.append("```json\n" + dumps(rep, indent=1) + "\n```")
            if gate is not None:
                lines.append("- 训练门判定：")
                lines.append("```json\n" + dumps(gate, indent=1) + "\n```")
        elif rep is not None:
            failed = [ck["name"] for ck in (gate or {}).get("checks", []) if not ck.get("passed")]
            lines.append(f"- 训练集摘要：IC 均值={rep.get('ic_mean')}，ICIR={rep.get('icir')}，"
                         f"最大参照相关={rep.get('max_ref_corr')}（{rep.get('max_ref_id')}），"
                         f"未达标项={failed or '无'}")
    return "\n".join(lines)


def _base_blocks(ctx: AgentContext, hyp: Hypothesis) -> list[str]:
    return [_hyp_block(hyp), render_data_description(), render_operators(),
            render_classics(ctx.classics), render_library(ctx.library)]


def build_user_prompt(ctx: AgentContext, hyp: Hypothesis) -> str:
    return "\n\n".join(_base_blocks(ctx, hyp) + ["请把本轮假设写成一个因子公式。"])


def adjust_user_prompt(ctx: AgentContext, hyp: Hypothesis, versions: list[dict]) -> str:
    return "\n\n".join(_base_blocks(ctx, hyp) + [
        render_versions(versions),
        f"请在 v{versions[-1]['version']} 的基础上调整公式，产出新版本。"])


def attribution_user_prompt(hyp: Hypothesis, versions: list[dict]) -> str:
    return "\n\n".join([_hyp_block(hyp), render_versions(versions),
                        "请对本轮研究做复盘。"])


# ---------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------

def _make_check(require_change_note: bool) -> Callable[[dict], FactorProposal]:
    def check(obj: dict) -> FactorProposal:
        status = require_str(obj, "status")
        if status not in ("ok", "cannot_build"):
            raise ValueError('status 只能是 "ok" 或 "cannot_build"')
        name = require_str(obj, "name", nonempty=(status == "ok"))
        explanation = require_str(obj, "explanation", nonempty=False) if "explanation" in obj else ""
        reason = require_str(obj, "reason", nonempty=False) if "reason" in obj else ""
        change_note = require_str(obj, "change_note", nonempty=False) if "change_note" in obj else ""
        if status == "ok":
            expression = require_str(obj, "expression")
            if require_change_note and not change_note:
                raise ValueError("调整时 change_note 必填")
            if not require_change_note:
                change_note = ""
            return FactorProposal(status="ok", name=name, expression=expression,
                                  explanation=explanation, reason="", change_note=change_note)
        if not reason:
            raise ValueError("status 为 cannot_build 时 reason 必填")
        return FactorProposal(status="cannot_build", name=name, expression="",
                              explanation=explanation, reason=reason,
                              change_note=change_note if require_change_note else "")
    return check


def _check_attr(obj: dict) -> Attribution:
    a = require_str(obj, "attribution")
    if a not in ("formula", "hypothesis", "none"):
        raise ValueError('attribution 只能是 "formula"、"hypothesis" 或 "none"')
    c = require_str(obj, "conclusion")
    if c not in REVIEW_CONCLUSIONS:
        raise ValueError(f"conclusion 只能是：{'、'.join(REVIEW_CONCLUSIONS)}")
    try:
        bh = int(float(str(obj.get("best_horizon")).strip()))
    except (TypeError, ValueError):
        bh = -1
    if bh not in HORIZONS:
        raise ValueError(f"best_horizon 只能是 {HORIZONS} 之一")
    return Attribution(attribution=a, conclusion=c, best_horizon=bh,
                       lesson=require_str(obj, "lesson"), note=require_str(obj, "note"))


# ---------------------------------------------------------------------
# 带公式校验的调用
# ---------------------------------------------------------------------

def _with_validation(llm: Any, system: str, user: str, validate: Validator,
                     require_change_note: bool, tag: dict) -> FactorProposal:
    check = _make_check(require_change_note)
    prop = ask_json(llm, system, user, check, tag={**tag, "rewrite": 0})
    for k in range(MAX_EXPR_RETRY + 1):
        if prop["status"] != "ok":
            return prop
        try:
            validate(prop["expression"])
            return prop
        except ValueError as e:          # ExprError 是 ValueError 的子类
            err = str(e)
            if k == MAX_EXPR_RETRY:
                raise ExprRetryExhausted(
                    f"公式校验失败，已重写 {MAX_EXPR_RETRY} 次仍不合规：{prop['expression']} —— {err}") from e
            retry_user = (user + f"\n\n## 公式校验未通过（第 {k + 1} 次重写）\n"
                          f"你给出的公式：`{prop['expression']}`\n程序返回的错误：{err}\n"
                          "请修正公式后按同样的 JSON 格式重新输出。")
            prop = ask_json(llm, system, retry_user, check, tag={**tag, "rewrite": k + 1})
    raise AssertionError("unreachable")


def build(llm: Any, ctx: AgentContext, hyp: Hypothesis, validate: Validator,
          tag: Optional[dict] = None) -> FactorProposal:
    """构建 v0。"""
    return _with_validation(llm, load_prompt("factor_build"), build_user_prompt(ctx, hyp),
                            validate, require_change_note=False,
                            tag={"agent": "factor", "call": "build", **(tag or {})})


def adjust(llm: Any, ctx: AgentContext, hyp: Hypothesis, versions: list[dict],
           validate: Validator, tag: Optional[dict] = None) -> FactorProposal:
    """调整：输入含本轮各版公式及其训练集明细。"""
    return _with_validation(llm, load_prompt("factor_adjust"), adjust_user_prompt(ctx, hyp, versions),
                            validate, require_change_note=True,
                            tag={"agent": "factor", "call": "adjust", **(tag or {})})


def attribute(llm: Any, hyp: Hypothesis, versions: list[dict],
              tag: Optional[dict] = None) -> Attribution:
    """复盘：本轮训练集结束后（验证之前）调用一次，无论是否达标。"""
    return ask_json(llm, load_prompt("factor_attribution"), attribution_user_prompt(hyp, versions),
                    _check_attr, tag={"agent": "factor", "call": "attribute", **(tag or {})})
