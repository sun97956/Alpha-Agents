"""
LLM 接入层（线 D，PRD 第 14 节）。

- OpenAICompatibleLLM：用 openai SDK 接入任何兼容 OpenAI 接口的服务。
  只用最基础的 chat.completions 文本输出，不用 JSON mode、工具调用等服务商特有功能；
  JSON 由提示词约定、调用方自行解析。
- MockLLM：离线模拟模型，按系统提示词里的任务标记返回预设 JSON；可脚本化、可注入异常。
- LoggedLLM：包装任意 LLM，把每次调用的 prompt 和回复写进 runs/<run_id>/llm_log.jsonl。

三者都满足 contracts.LLM 协议：complete(system, user) -> str。
"""
from __future__ import annotations

import json
import os
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

# 系统提示词里的任务标记（prompts/*.md 第一行），MockLLM 靠它识别该回什么
TASK_MARKER_RE = re.compile(r"\[TASK:\s*([a-z_]+)\s*\]")
TASK_HYPOTHESIS = "hypothesis"
TASK_BUILD = "factor_build"
TASK_REWRITE = "factor_rewrite"
TASK_ADJUST = "factor_adjust"
TASK_ATTRIBUTION = "factor_attribution"


def detect_task(system: str) -> Optional[str]:
    m = TASK_MARKER_RE.search(system or "")
    return m.group(1) if m else None


# =====================================================================
# 真实模型
# =====================================================================

def _response_meta(resp: Any, latency_s: float) -> dict:
    """提取用于 trace 的元数据：思考内容、结束原因、token 用量、耗时。"""
    meta: dict[str, Any] = {"latency_s": round(latency_s, 2)}
    choices = getattr(resp, "choices", None) or []
    if choices:
        msg = getattr(choices[0], "message", None)
        meta["finish_reason"] = getattr(choices[0], "finish_reason", None)
        meta["reasoning_content"] = getattr(msg, "reasoning_content", None) if msg else None
    usage = getattr(resp, "usage", None)
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        meta["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
        }
    return meta


class OpenAICompatibleLLM:
    """通过 openai SDK 调用兼容 OpenAI 接口的服务。"""

    def __init__(self, base_url: str, model: str, api_key_env: str = "LLM_API_KEY",
                 temperature: float = 0.7, max_tokens: int = 2000,
                 api_key: Optional[str] = None, timeout: float = 300.0,
                 client: Any = None, extra_body: Optional[dict] = None):
        if not model:
            raise ValueError("settings.yaml 里 llm.model 为空，请先填写模型名")
        self.base_url = base_url or None
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # 服务商特有参数（如关闭思考模式），原样透传；默认不传
        self.extra_body = dict(extra_body) if extra_body else None
        if client is not None:              # 测试时注入假 client
            self._client = client
            return
        key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        if not key:
            raise ValueError(f"环境变量 {api_key_env} 未设置，无法读取 API Key")
        if not base_url:
            raise ValueError("settings.yaml 里 llm.base_url 为空，请先填写服务地址")
        from openai import OpenAI          # 延迟导入，Mock 模式不需要
        self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)

    @classmethod
    def from_settings(cls, settings: dict, client: Any = None) -> "OpenAICompatibleLLM":
        llm_cfg = dict(settings.get("llm") or {})
        provider = llm_cfg.get("provider", "openai_compatible")
        if provider != "openai_compatible":
            raise ValueError(f"不支持的 llm.provider：{provider}（目前只支持 openai_compatible）")
        return cls(base_url=llm_cfg.get("base_url", ""),
                   model=llm_cfg.get("model", ""),
                   api_key_env=llm_cfg.get("api_key_env", "LLM_API_KEY"),
                   temperature=llm_cfg.get("temperature", 0.7),
                   max_tokens=llm_cfg.get("max_tokens", 2000),
                   extra_body=llm_cfg.get("extra_body") or None,
                   client=client)

    def complete(self, system: str, user: str) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        self.last_meta = None
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(**kwargs)
        self.last_meta = _response_meta(resp, time.monotonic() - t0)
        choices = getattr(resp, "choices", None)
        if not choices:
            raise RuntimeError("LLM 返回为空（没有 choices）")
        content = choices[0].message.content
        if content is None or not str(content).strip():
            # 常见原因：思考型模型把 max_tokens 全用在思考上，正文为空（finish_reason=length）
            finish = getattr(choices[0], "finish_reason", None)
            usage = getattr(resp, "usage", None)
            details = getattr(usage, "completion_tokens_details", None) if usage else None
            reasoning = getattr(details, "reasoning_tokens", None) if details else None
            completion = getattr(usage, "completion_tokens", None) if usage else None
            hint = "；可能是思考用完了 max_tokens，请调大 settings.yaml 的 llm.max_tokens" if finish == "length" else ""
            raise RuntimeError(f"LLM 返回内容为空（finish_reason={finish}，completion_tokens={completion}，"
                               f"reasoning_tokens={reasoning}，max_tokens={self.max_tokens}）{hint}")
        return str(content)


# =====================================================================
# 模拟模型
# =====================================================================

# 预设假设：(假设, 经济逻辑, 因子名, 公式)。公式全部只用 PRD 第 7 节的字段/算子/窗口。
MOCK_IDEAS: list[tuple[str, str, str, str]] = [
    ("过去一周涨幅大的股票下周倾向回调",
     "短期过度反应：流动性冲击和情绪推动的价格偏离会在随后几天修正",
     "mock_rev5", "-ts_sum(ret, 5)"),
    ("成交量相对放大时的短期反转更强",
     "放量伴随的价格变动更多来自噪音交易，回调更明显",
     "mock_rev_vol", "-ts_sum(ret, 5) * ts_mean(volume, 5) / ts_mean(volume, 60)"),
    ("近期价格接近区间高点的股票后续表现更好",
     "锚定效应：投资者不愿在高点附近追买，导致信息被缓慢吸收",
     "mock_high_dist", "close / ts_max(high, 60)"),
    ("剔除行业影响后的短期输家会反弹",
     "行业轮动掩盖了个股层面的过度反应，中性化后反转更干净",
     "mock_ind_rev", "group_neutralize(-ts_sum(ret, 10), sector)"),
]
MOCK_ADJUST_EXPRS: list[str] = [
    "-ts_sum(ret, 10)",
    "cs_rank(-ts_sum(ret, 5))",
    "-ts_mean(ret, 20)",
]

ScriptItem = Union[str, Exception, Callable[[str, str], str]]


class MockLLM:
    """
    离线模拟模型。
      script     ：按顺序消费的回复（字符串 / 异常实例 / callable(system, user)）；用完后走默认行为。
      fail_on_calls：{第 n 次调用(从 1 起): 异常}，到该次调用时抛出，用于人为注入错误。
    默认行为：根据系统提示词里的 [TASK: ...] 标记返回合法 JSON。
    """

    def __init__(self, script: Optional[list[ScriptItem]] = None,
                 fail_on_calls: Optional[dict[int, Exception]] = None):
        self.script: list[ScriptItem] = list(script or [])
        self.fail_on_calls = dict(fail_on_calls or {})
        self.calls: list[tuple[str, str]] = []
        self._n_hyp = 0
        self._n_adjust = 0

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        n = len(self.calls)
        if n in self.fail_on_calls:
            raise self.fail_on_calls[n]
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            if callable(item):
                return item(system, user)
            return item
        return self._default(system, user)

    def _default(self, system: str, user: str) -> str:
        task = detect_task(system)
        if task == TASK_HYPOTHESIS:
            hyp, logic, _, _ = MOCK_IDEAS[self._n_hyp % len(MOCK_IDEAS)]
            self._n_hyp += 1
            return json.dumps({"hypothesis": hyp, "logic": logic}, ensure_ascii=False)
        if task in (TASK_BUILD, TASK_REWRITE):
            m = re.search(r"## 本轮假设\s*\n- 假设：(.*)", user)
            cur = m.group(1).strip() if m else ""
            idea = next((i for i in MOCK_IDEAS if i[0] == cur), MOCK_IDEAS[0])
            return "```json\n" + json.dumps({
                "status": "ok", "name": idea[2], "expression": idea[3],
                "explanation": f"模拟构建：{idea[3]}", "reason": "", "change_note": "",
            }, ensure_ascii=False) + "\n```"
        if task == TASK_ADJUST:
            expr = MOCK_ADJUST_EXPRS[self._n_adjust % len(MOCK_ADJUST_EXPRS)]
            self._n_adjust += 1
            return json.dumps({
                "status": "ok", "name": "mock_adjusted", "expression": expr,
                "explanation": f"模拟调整：{expr}", "reason": "",
                "change_note": "模拟调整：换一个窗口/形式",
            }, ensure_ascii=False)
        if task == TASK_ATTRIBUTION:
            return json.dumps({"attribution": "formula", "note": "模拟归因：公式形式可能不合适"},
                              ensure_ascii=False)
        raise RuntimeError(f"MockLLM 无法识别任务标记：{task!r}")


# =====================================================================
# 日志包装
# =====================================================================

class LoggedLLM:
    """包装一个 LLM，把每次调用（含失败）追加写入 log_path（jsonl，utf-8）。"""

    def __init__(self, inner: Any, log_path: Union[str, Path], run_id: str = ""):
        self.inner = inner
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.tag: dict[str, Any] = {}

    def set_tag(self, **tag: Any) -> None:
        """调用方标注本次调用属于哪一轮、哪个 Agent、哪类调用（覆盖上一次的标注）。"""
        self.tag = dict(tag)

    def complete(self, system: str, user: str) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id, **self.tag,
            "system": system, "user": user, "response": None, "error": None,
        }
        if hasattr(self.inner, "last_meta"):
            self.inner.last_meta = None
        try:
            out = self.inner.complete(system, user)
            entry["response"] = out
            return out
        except Exception as e:  # noqa: BLE001 —— 记录后原样抛出
            entry["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            meta = getattr(self.inner, "last_meta", None)
            if meta:
                entry.update(meta)   # 思考内容、用量、耗时
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
