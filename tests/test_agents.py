"""
线 D 单元测试：LLM 接入、JSON 解析与重试、公式重写、因子 Agent 三种调用、
运行记录 / 因子库、调度程序各分支。只用 make_fake_market() 和本文件里的最小替身。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import contracts as C
from src import records
from src.agents import factor_agent, hypothesis_agent
from src.agents.common import (AgentContext, AgentOutputError, ExprRetryExhausted, ask_json,
                               extract_json, load_prompt, to_jsonable)
from src.agents.llm import (TASK_ADJUST, TASK_ATTRIBUTION, TASK_BUILD, TASK_HYPOTHESIS,
                            LoggedLLM, MockLLM, OpenAICompatibleLLM, detect_task)
from src.orchestrator import Engine, Orchestrator, load_classics

HYP = {"hypothesis": "测试假设", "logic": "测试逻辑", "mechanism": "反转", "horizon": 5}
CTX = AgentContext(classics=[{"id": "C01_rev5", "name": "5 日反转",
                              "expression": "-ts_sum(ret, 5)", "logic": "短期过度反应后回调"}])


def J(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False)


def ok_prop(expr="-ts_sum(ret, 10)", note="", name="f"):
    return J(status="ok", name=name, expression=expr, explanation="说明", reason="", change_note=note)


def accept_all(expr: str) -> None:
    return None


def reject_bad(expr: str) -> None:
    if "bad" in expr:
        raise ValueError(f"非法字段：bad（公式 {expr}）")


# =====================================================================
# JSON 解析
# =====================================================================

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '好的，结果如下：\n{"a": 1}\n以上。',
    '前缀 {不是json} 然后 {"a": 1}',
])
def test_extract_json_tolerant(raw):
    assert extract_json(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ["", "没有 JSON", "[1, 2]", '{"a": '])
def test_extract_json_rejects(raw):
    with pytest.raises(ValueError):
        extract_json(raw)


def test_to_jsonable_nan_and_numpy():
    out = to_jsonable({"a": np.float64("nan"), "b": [np.int64(3), float("inf")], "c": np.float32(1.5)})
    assert out == {"a": None, "b": [3, None], "c": 1.5}
    json.dumps(out, allow_nan=False)


# =====================================================================
# LLM 实现
# =====================================================================

class FakeCompletions:
    def __init__(self, content="{}"):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def fake_client(content="{}"):
    comp = FakeCompletions(content)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp)), comp


SETTINGS = {"llm": {"provider": "openai_compatible", "base_url": "https://example.invalid/v1",
                    "model": "m-1", "temperature": 0.3, "max_tokens": 123,
                    "api_key_env": "TEST_LLM_KEY"}}


def test_openai_llm_plain_chat_call():
    client, comp = fake_client('{"x": 1}')
    llm = OpenAICompatibleLLM.from_settings(SETTINGS, client=client)
    assert llm.complete("SYS", "USR") == '{"x": 1}'
    kw = comp.kwargs
    assert kw["model"] == "m-1" and kw["temperature"] == 0.3 and kw["max_tokens"] == 123
    assert kw["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    # 不依赖服务商特有功能：没有 JSON mode / 工具调用
    assert set(kw) == {"model", "messages", "temperature", "max_tokens"}


def test_openai_llm_empty_content_raises():
    client, _ = fake_client(None)
    llm = OpenAICompatibleLLM.from_settings(SETTINGS, client=client)
    with pytest.raises(RuntimeError):
        llm.complete("s", "u")


def test_openai_llm_reads_key_from_env(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None, timeout=None):
            captured.update(base_url=base_url, api_key=api_key)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("TEST_LLM_KEY", "sk-test")
    OpenAICompatibleLLM.from_settings(SETTINGS)
    assert captured == {"base_url": "https://example.invalid/v1", "api_key": "sk-test"}


def test_openai_llm_config_errors(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_LLM_KEY"):
        OpenAICompatibleLLM.from_settings(SETTINGS)
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleLLM.from_settings({"llm": {**SETTINGS["llm"], "model": ""}})
    monkeypatch.setenv("TEST_LLM_KEY", "k")
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleLLM.from_settings({"llm": {**SETTINGS["llm"], "base_url": ""}})
    with pytest.raises(ValueError, match="provider"):
        OpenAICompatibleLLM.from_settings({"llm": {**SETTINGS["llm"], "provider": "xyz"}})


def test_settings_file_has_llm_keys():
    import yaml
    s = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "settings.yaml")
                       .read_text(encoding="utf-8"))
    assert {"base_url", "model", "api_key_env", "temperature", "max_tokens"} <= set(s["llm"])


def test_prompts_have_task_markers():
    for name, task in [("hypothesis", TASK_HYPOTHESIS), ("factor_build", TASK_BUILD),
                       ("factor_adjust", TASK_ADJUST), ("factor_attribution", TASK_ATTRIBUTION)]:
        assert detect_task(load_prompt(name)) == task


def test_mock_llm_default_outputs_parse():
    m = MockLLM()
    h = hypothesis_agent.propose(m, CTX)
    assert h["hypothesis"] and h["logic"]
    p = factor_agent.build(m, CTX, h, accept_all)
    assert p["status"] == "ok" and p["expression"] and p["change_note"] == ""
    a = factor_agent.adjust(m, CTX, h, [{"version": 0, "expression": p["expression"],
                                         "explanation": "", "change_note": "", "report": {}, "gate": {}}],
                            accept_all)
    assert a["change_note"]
    at = factor_agent.attribute(m, h, [])
    assert at["attribution"] in ("formula", "hypothesis")


def test_mock_llm_script_and_fault_injection():
    m = MockLLM(script=["a", RuntimeError("boom"), lambda s, u: s + u],
                fail_on_calls={5: TimeoutError("t")})
    assert m.complete("s", "u") == "a"
    with pytest.raises(RuntimeError):
        m.complete("s", "u")
    assert m.complete("x", "y") == "xy"
    assert "hypothesis" in m.complete(load_prompt("hypothesis"), "")       # 脚本用完走默认
    with pytest.raises(TimeoutError):
        m.complete("s", "u")
    assert len(m.calls) == 5


def test_logged_llm_writes_prompt_reply_and_errors(tmp_path):
    log = tmp_path / "runs" / "r" / "llm_log.jsonl"
    llm = LoggedLLM(MockLLM(script=["回复一", ValueError("坏了")]), log, run_id="r")
    llm.set_tag(round=1, agent="hypothesis")
    assert llm.complete("系统", "用户") == "回复一"
    llm.set_tag(round=2)
    with pytest.raises(ValueError):
        llm.complete("系统2", "用户2")
    rows = records.read_jsonl(log)
    assert rows[0]["system"] == "系统" and rows[0]["user"] == "用户" and rows[0]["response"] == "回复一"
    assert rows[0]["round"] == 1 and rows[0]["agent"] == "hypothesis" and rows[0]["run_id"] == "r"
    assert rows[1]["error"].startswith("ValueError") and rows[1]["response"] is None
    assert "agent" not in rows[1]                        # 标注被覆盖，不残留
    assert "用户" in log.read_text(encoding="utf-8")       # utf-8，不转义中文


# =====================================================================
# JSON 重试 1 次
# =====================================================================

def test_hypothesis_parse_retry_once_then_ok():
    m = MockLLM(script=["这不是 JSON", J(hypothesis="h", logic="l", mechanism="动量", horizon=20)])
    assert hypothesis_agent.propose(m, CTX) == {"hypothesis": "h", "logic": "l", "mechanism": "动量", "horizon": 20}
    assert len(m.calls) == 2
    assert "无法按约定解析" in m.calls[1][1]


def test_hypothesis_parse_fails_twice_raises():
    m = MockLLM(script=["坏", J(hypothesis="h")])           # 第二次缺 logic
    with pytest.raises(AgentOutputError):
        hypothesis_agent.propose(m, CTX)
    assert len(m.calls) == 2


def test_llm_exception_not_retried():
    m = MockLLM(script=[ConnectionError("x")])
    with pytest.raises(ConnectionError):
        hypothesis_agent.propose(m, CTX)
    assert len(m.calls) == 1


def test_build_schema_errors_count_as_parse_failures():
    # cannot_build 没写 reason / status 非法 → 解析失败
    m = MockLLM(script=[J(status="cannot_build", name="", expression="", explanation="", reason=""),
                        J(status="maybe", name="f", expression="x", explanation="")])
    with pytest.raises(AgentOutputError):
        factor_agent.build(m, CTX, HYP, accept_all)


def test_build_cannot_build():
    m = MockLLM(script=[J(status="cannot_build", name="", expression="", explanation="",
                          reason="需要财报数据")])
    p = factor_agent.build(m, CTX, HYP, accept_all)
    assert p["status"] == "cannot_build" and p["reason"] == "需要财报数据" and p["expression"] == ""


def test_adjust_requires_change_note():
    m = MockLLM(script=[ok_prop(note=""), ok_prop(note="")])
    with pytest.raises(AgentOutputError):
        factor_agent.adjust(m, CTX, HYP, [{"version": 0, "expression": "x"}], accept_all)


def test_attribution_value_checked():
    m = MockLLM(script=[J(attribution="data", conclusion="不支持", best_horizon=20, lesson="经验", note="n"), J(attribution="hypothesis", conclusion="不支持", best_horizon=20, lesson="经验", note="n")])
    assert factor_agent.attribute(m, HYP, [])["attribution"] == "hypothesis"
    assert len(m.calls) == 2


# =====================================================================
# 公式校验失败重写，最多 MAX_EXPR_RETRY 次
# =====================================================================

def test_expr_rewrite_then_ok():
    m = MockLLM(script=[ok_prop("bad + 1"), ok_prop("-ts_sum(ret, 5)")])
    p = factor_agent.build(m, CTX, HYP, reject_bad)
    assert p["expression"] == "-ts_sum(ret, 5)"
    assert "非法字段：bad" in m.calls[1][1] and "bad + 1" in m.calls[1][1]


def test_expr_rewrite_exhausted():
    m = MockLLM(script=[ok_prop("bad")] * (C.MAX_EXPR_RETRY + 5))
    with pytest.raises(ExprRetryExhausted):
        factor_agent.build(m, CTX, HYP, reject_bad)
    assert len(m.calls) == 1 + C.MAX_EXPR_RETRY


def test_expr_rewrite_json_retry_is_separate():
    # 重写那次输出坏 JSON → JSON 重试；不消耗公式重写次数
    m = MockLLM(script=[ok_prop("bad"), "坏", ok_prop("bad"), ok_prop("ok_expr")])
    assert factor_agent.build(m, CTX, HYP, reject_bad)["expression"] == "ok_expr"


# =====================================================================
# 提示词内容（可见性）
# =====================================================================

def test_factor_prompts_contain_required_inputs():
    lib = [{"id": "F0001", "name": "lib_f", "expression": "ts_mean(ret, 5)", "direction": 1,
            "hypothesis": "h0", "logic": "l0", "train_ic_mean": 0.03, "train_icir": 0.1}]
    ctx = AgentContext(classics=CTX.classics, library=lib)
    u = factor_agent.build_user_prompt(ctx, HYP)
    for s in ("测试假设", "测试逻辑", "mktcap", "千美元", "ts_corr", "group_neutralize",
              "C01_rev5", "F0001", "250"):
        assert s in u
    rep = {"split": "train", "ic_mean": 0.0123, "icir": float("nan")}
    u2 = factor_agent.adjust_user_prompt(ctx, HYP, [{"version": 0, "expression": "e0",
                                                     "explanation": "x", "change_note": "",
                                                     "report": rep, "gate": {"passed": False}}])
    assert "0.0123" in u2 and "e0" in u2 and "NaN" not in u2


def test_hypothesis_prompt_history_has_no_errors_or_metrics_beyond_spec():
    recs = [{"run_id": "r", "round": 1, "timestamp": "t", "hypothesis": HYP,
             "versions": [{"ic_mean": 0.031, "icir": 0.07}], "train_conclusion": "pass",
             "attribution": None, "validation": "fail", "error": None},
            {"run_id": "r", "round": 2, "timestamp": "t2", "hypothesis": None, "versions": [],
             "train_conclusion": "skipped", "attribution": None, "validation": None,
             "error": "SECRET_ERROR_TEXT"}]
    ctx = AgentContext(history=records.history_summary(recs))
    u = hypothesis_agent.build_user_prompt(ctx)
    assert "测试假设" in u and "0.0310" in u and "验证：fail" in u and "skipped" in u
    assert "SECRET_ERROR_TEXT" not in u


# =====================================================================
# 运行记录与因子库
# =====================================================================

def test_history_summary_last_30_and_fields():
    recs = [{"run_id": "r", "round": i, "hypothesis": {"hypothesis": f"h{i}", "logic": "l"},
             "versions": [{"ic_mean": 0.01, "icir": 0.02}, {"ic_mean": None, "icir": None}],
             "train_conclusion": "fail", "attribution": {"attribution": "formula", "note": "n"},
             "validation": None, "error": None} for i in range(1, 41)]
    out = records.history_summary(recs)
    assert len(out) == C.HISTORY_ROUNDS and out[0]["round"] == 11 and out[-1]["round"] == 40
    assert set(out[0]) == {"run_id", "round", "hypothesis", "logic", "mechanism", "horizon", "train_conclusion",
                           "train_ic_mean", "train_icir", "attribution", "validation"}
    assert out[0]["train_ic_mean"] == 0.01


def test_library_ids_and_validation(tmp_path):
    p = tmp_path / "lib.jsonl"
    assert records.next_library_id(records.read_library(p)) == "F0001"
    entry = {k: None for k in records.LIBRARY_KEYS}
    entry.update(id="F0001", name="n", expression="e", direction=-1, hypothesis="h", logic="l",
                 explanation="x", train_ic_mean=-0.03, train_icir=-0.1, validation="pass",
                 run_id="r", round=1, created_at="t")
    records.append_library(p, entry)
    assert records.next_library_id(records.read_library(p)) == "F0002"
    with pytest.raises(ValueError):
        records.append_library(p, {**entry, "validation": "fail"})
    with pytest.raises(ValueError):
        records.append_library(p, {**entry, "direction": 0})
    with pytest.raises(ValueError):
        records.append_library(p, {"id": "F9"})
    assert len(records.read_library(p)) == 1


def test_recorder_only_saves_train_reports(tmp_path):
    rec = records.RunRecorder(tmp_path / "runs", "r")
    path = rec.save_report(3, 1, {"split": "train", "ic_mean": float("nan")}, {"passed": False})
    assert path.endswith("runs/r/reports/round_003_v1.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["report"]["ic_mean"] is None
    with pytest.raises(ValueError):
        rec.save_report(3, 1, {"split": "valid"})
    with pytest.raises(ValueError):
        rec.append_round({"run_id": "r"})


def test_load_classics_formats(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- {id: C01_rev5, name: 反转, expression: '-ts_sum(ret, 5)', logic: l}\n",
                 encoding="utf-8")
    assert load_classics(p)[0]["id"] == "C01_rev5"
    p.write_text("factors:\n  - {id: C02, name: n, expression: e, logic: l}\n", encoding="utf-8")
    assert load_classics(p)[0]["expression"] == "e"


# =====================================================================
# 调度程序各分支（可控的假引擎）
# =====================================================================

THR = {"train": {}, "valid": {}}


class FakeEngine:
    """train_ic: {expr: ic}，ic 绝对值 ≥ 0.02 算训练达标；valid_pass: 验证是否通过。"""

    def __init__(self, train_ic: dict, valid_pass: bool = True, bad_exprs=()):
        self.train_ic, self.valid_pass, self.bad = train_ic, valid_pass, set(bad_exprs)
        self.evals, self.saved, self.refs_loaded = [], [], 0

    def validate(self, expr):
        if expr in self.bad:
            raise ValueError(f"不合规：{expr}")

    def compute(self, expr, market):
        f = market.fields["close"].copy()
        f.attrs["expr"] = expr
        return f

    def evaluate(self, factor, market, split, refs=None, horizon=5):
        self.evals.append((factor.attrs["expr"], split, refs))
        ic = self.train_ic.get(factor.attrs["expr"], 0.0) if split == "train" else 0.555
        return {"split": split, "ic_mean": ic, "icir": ic * 3, "max_ref_corr": None,
                "max_ref_id": None}

    def gate_train(self, rep, thr):
        return {"passed": abs(rep["ic_mean"]) >= 0.02, "checks": []}

    def gate_valid(self, vrep, trep, thr):
        return {"passed": self.valid_pass, "checks": []}

    def save_ref(self, ref_id, factor, market, cache_dir="cache"):
        self.saved.append(ref_id)

    def load_refs(self, cache_dir="cache"):
        self.refs_loaded += 1
        return {s: pd.DataFrame() for s in self.saved}

    def engine(self) -> Engine:
        return Engine(self.validate, self.compute, self.evaluate, self.gate_train,
                      self.gate_valid, self.save_ref, self.load_refs)


@pytest.fixture(scope="module")
def market():
    return C.make_fake_market(n_days=60, n_stocks=110)


def orch(tmp_path, market, llm, fe, **kw):
    return Orchestrator(market=market, llm=llm, engine=fe.engine(), thresholds=THR,
                        thresholds_sha256="h", classics=CTX.classics,
                        refs={"C01_rev5": pd.DataFrame()}, runs_dir=tmp_path / "runs",
                        library_path=tmp_path / "lib.jsonl", cache_dir=tmp_path / "cache",
                        run_id="t", **kw)


def H():
    return J(hypothesis="假设甲", logic="逻辑甲", mechanism="反转", horizon=5)


def test_round_pass_and_library(tmp_path, market):
    fe = FakeEngine({"e0": -0.05})
    m = MockLLM(script=[H(), ok_prop("e0", name="neg_f")])
    r = orch(tmp_path, market, m, fe).run_round(1)
    assert r["train_conclusion"] == "pass" and r["validation"] == "pass" and r["library_id"] == "F0001"
    lib = records.read_library(tmp_path / "lib.jsonl")
    assert lib[0]["direction"] == -1 and lib[0]["name"] == "neg_f" and lib[0]["round"] == 1
    assert fe.saved == ["F0001"] and fe.refs_loaded == 1
    # refs 只在训练集传入；验证只验一次
    use_refs = "max_abs_ref_corr" in THR["train"]
    assert [(s, refs is not None) for _, s, refs in fe.evals] == [("train", use_refs), ("valid", False)]


def test_round_fail_all_versions_then_attribution(tmp_path, market):
    fe = FakeEngine({})
    script = [H(), ok_prop("e0")] + [ok_prop(f"e{i}", note=f"改{i}") for i in (1, 2, 3)] + \
             [J(attribution="hypothesis", conclusion="不支持", best_horizon=20, lesson="经验", note="都不行")]
    m = MockLLM(script=script)
    r = orch(tmp_path, market, m, fe).run_round(1)
    assert r["train_conclusion"] == "fail" and r["validation"] is None
    assert [v["version"] for v in r["versions"]] == [0, 1, 2, 3]
    assert [v["expression"] for v in r["versions"]] == ["e0", "e1", "e2", "e3"]
    assert r["versions"][2]["change_note"] == "改2" and r["versions"][0]["change_note"] == ""
    assert all(v["train_gate_passed"] is False for v in r["versions"])
    assert r["attribution"] == {"attribution": "hypothesis", "conclusion": "不支持", "best_horizon": 20,
                                "lesson": "经验", "note": "都不行"}
    assert all(s == "train" for _, s, _ in fe.evals)              # 未达标不做验证
    assert not m.script                                          # 恰好用完：3 次调整 + 1 次归因
    # 调整的输入含之前各版的训练集明细
    adjust_prompts = [u for s, u in m.calls if "[TASK: factor_adjust]" in s]
    assert len(adjust_prompts) == 3 and "e0" in adjust_prompts[2] and "e2" in adjust_prompts[2]


def test_round_pass_after_adjust(tmp_path, market):
    fe = FakeEngine({"e1": 0.03})
    m = MockLLM(script=[H(), ok_prop("e0"), ok_prop("e1", note="加窗口")])
    r = orch(tmp_path, market, m, fe).run_round(1)
    assert r["train_conclusion"] == "pass" and len(r["versions"]) == 2
    assert r["versions"][1]["train_gate_passed"] is True
    assert records.read_library(tmp_path / "lib.jsonl")[0]["expression"] == "e1"


def test_round_cannot_build(tmp_path, market):
    fe = FakeEngine({})
    m = MockLLM(script=[H(), J(status="cannot_build", name="", expression="", explanation="",
                               reason="缺基本面字段")])
    r = orch(tmp_path, market, m, fe).run_round(1)
    assert r["train_conclusion"] == "cannot_build" and r["cannot_build_reason"] == "缺基本面字段"
    assert r["versions"] == [] and r["attribution"] is None and fe.evals == []


def test_round_valid_fail_not_in_library(tmp_path, market):
    fe = FakeEngine({"e0": 0.05}, valid_pass=False)
    m = MockLLM(script=[H(), ok_prop("e0")])
    r = orch(tmp_path, market, m, fe).run_round(1)
    assert r["train_conclusion"] == "pass" and r["validation"] == "fail" and r["library_id"] is None
    assert records.read_library(tmp_path / "lib.jsonl") == [] and fe.saved == []
    assert "0.555" not in (tmp_path / "runs" / "t" / "run_log.jsonl").read_text(encoding="utf-8")


def test_round_v0_expr_retry_exhausted_is_skipped(tmp_path, market):
    fe = FakeEngine({}, bad_exprs={"bad"})
    m = MockLLM(script=[H()] + [ok_prop("bad")] * (1 + C.MAX_EXPR_RETRY) + [H(), ok_prop("ok")])
    o = orch(tmp_path, market, m, fe)
    r1 = o.run_round(1)
    assert r1["train_conclusion"] == "skipped" and "ExprRetryExhausted" in r1["error"]
    r2 = o.run_round(2)
    assert r2["train_conclusion"] != "skipped"


def test_round_adjust_expr_retry_exhausted_ends_inner_loop(tmp_path, market):
    fe = FakeEngine({}, bad_exprs={"bad"})
    script = [H(), ok_prop("e0")] + [ok_prop("bad", note="n")] * (1 + C.MAX_EXPR_RETRY) + \
             [J(attribution="formula", note="写不出来")]
    r = orch(tmp_path, market, MockLLM(script=script), fe).run_round(1)
    assert r["train_conclusion"] == "fail" and len(r["versions"]) == 2
    assert r["versions"][1]["error"] and r["versions"][1]["ic_mean"] is None
    assert r["attribution"]["attribution"] == "formula"


def test_round_adjust_cannot_build_ends_inner_loop(tmp_path, market):
    fe = FakeEngine({})
    script = [H(), ok_prop("e0"),
              J(status="cannot_build", name="", expression="", explanation="", reason="没法再改",
                change_note="放弃"),
              J(attribution="hypothesis", note="n")]
    r = orch(tmp_path, market, MockLLM(script=script), fe).run_round(1)
    assert r["train_conclusion"] == "fail" and len(r["versions"]) == 1
    assert r["cannot_build_reason"] == "没法再改" and r["attribution"] is not None


def test_round_parse_failure_is_skipped(tmp_path, market):
    fe = FakeEngine({})
    r = orch(tmp_path, market, MockLLM(script=["坏", "还是坏"]), fe).run_round(1)
    assert r["train_conclusion"] == "skipped" and "AgentOutputError" in r["error"]
    assert r["hypothesis"] is None


def test_run_log_append_only_one_line_per_round(tmp_path, market):
    fe = FakeEngine({"-ts_sum(ret, 5)": 0.04})
    o = orch(tmp_path, market, MockLLM(), fe, inject_error_round=1)
    o.run(2)
    lines = (tmp_path / "runs" / "t" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(x)["round"] for x in lines] == [1, 2]
    assert all(tuple(json.loads(x)) == records.ROUND_KEYS for x in lines)
