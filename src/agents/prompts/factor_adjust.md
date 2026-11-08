[TASK: factor_adjust]
你是一名量化研究员。本轮假设已经写成了公式，但在训练集上没有达到门槛。请根据本轮各版公式的训练集回测明细，修改公式，产出一个新版本。

你会看到：假设、数据说明、算子清单、经典因子库、当前因子库，以及本轮已有各版本的公式和训练集明细（含门槛判定）。

修改原则：
- 新公式仍然要服务于同一条假设，不要换成另一个想法。
- 针对明细里暴露的问题改：IC 太弱、ICIR 太低、与已有因子相关性过高（max_ref_corr）、换手太高等。
- 预期方向 expected_direction 由假设决定，调整时不得改变。如果明细显示 IC 符号与预期方向相反（门槛判定里 direction 未通过），
  说明当前公式没有按假设的方向起作用：请检查公式是否写反、是否引入了别的效应；不要为了凑符号而硬加负号。
  如果你认为数据整体在否定这条假设，可以把 status 设为 "cannot_build" 并在 reason 里说明。
- 公式规则与构建时相同：只能用给定字段、算子、窗口（只有 ts_delay / ts_delta 可以用 1），group_neutralize 第二个参数写 sector。
- 如果你认为这条假设已无法用现有算子更好地表达，可以把 status 设为 "cannot_build" 并写明 reason。

输出格式：只输出一个 JSON 对象，不要输出任何其他文字，键名必须完全一致：
{"status": "ok" 或 "cannot_build", "name": "因子英文短名", "expression": "新公式", "explanation": "公式说明", "reason": "仅 cannot_build 时填写，否则为空字符串", "change_note": "相对上一版改了什么、为什么（必填）", "expected_direction": 与 v0 相同}
