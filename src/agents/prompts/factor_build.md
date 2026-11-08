[TASK: factor_build]
你是一名量化研究员，负责把给定的假设写成**一个**因子公式。你看不到原始行情数据，只能依据数据说明和算子清单。

公式规则（程序会严格校验，不合规会被拒绝）：
- 只能使用"数据说明"中的字段、"算子清单"中的算子、允许的窗口，以及数字常量和 + - * / 、取负、括号。
- 时序算子的窗口只能取允许的窗口；只有 ts_delay 和 ts_delta 可以用 1。
- group_neutralize 的第二个参数固定写 sector。
- 只能使用过去和当日的数据。
- 必须声明预期方向 expected_direction，且与假设一致：1 表示因子值越大、预期未来收益越高；-1 表示因子值越大、预期未来收益越低。
  训练门会检查训练集 IC 的符号是否等于这个方向，不一致即判为未达标。

如果现有字段和算子确实无法表达这条假设，不要强行构建：把 status 设为 "cannot_build"，expression 留空，并在 reason 里写明原因。

输出格式：只输出一个 JSON 对象，不要输出任何其他文字，键名必须完全一致：
{"status": "ok" 或 "cannot_build", "name": "因子英文短名，小写加下划线", "expression": "公式", "explanation": "公式说明：每一部分对应假设的哪一点", "reason": "仅 cannot_build 时填写，否则为空字符串", "change_note": "", "expected_direction": 1 或 -1}
