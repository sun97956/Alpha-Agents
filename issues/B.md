# 线 B 问题记录

以下是 PRD / AGENTS / contracts 没写清楚的地方，以及我按最保守理解做出的选择。都不阻塞，需要人确认。

## 1. 口径未定义，已自行选定

1. **ts_rank 的"分位（0–1）"**：实现为 `(窗口内严格小于当日值的个数 + 0.5 × (相等个数 − 1)) / (d − 1)`，窗口含当日；最小 0、最大 1，并列取平均；窗口内有缺失则 NaN。
2. **cs_rank 的"0–1"**：用 pandas `rank(axis=1, pct=True, method="average")`，取值范围是 (0, 1]，不是严格的 [0, 1]。线 C 的 `to_ranks` 如果用别的口径，两者会有细微差别（不影响 Spearman 相关）。
3. **标准差自由度**：`ts_std`、`cs_zscore` 都用 ddof=1（pandas 默认）。`ts_skew` 用 pandas rolling skew（偏差修正）。
4. **ts_corr**：窗口内 x、y 必须都完整；任一方在窗口内为常数时记 NaN；结果截断在 [−1, 1]。
5. **ts_delay / ts_delta**：按全局交易日历移位（与 `contracts.compute_ret` 一致），所以某股在 t−d 日恰好缺失时结果为 NaN，不会往前找最近一个有效值。
6. **group_neutralize**：当日行业为 NaN 的股票结果为 NaN；组均值只用当日股票池内且 x 非缺失的股票。

## 2. 校验从严的地方（可能多拒绝了一些公式）

7. **全角 / Unicode 运算符直接拒绝，不自动转换**。PRD 第 13 节的公式里用的是 `−`（U+2212）和 `×`，LLM 很可能照抄。现在会报错并提示改成 ASCII（`-` `*` `/`），但每次会消耗一次重写机会。建议线 D 在提示词里明确"只能用 ASCII 运算符"；如果希望引擎自动转换，请人决定后改。
8. **常数相关**：整个公式不含字段（如 `1 + 2`）拒绝；算子的数据参数只是常数（如 `ts_mean(5, 5)`、`log(10)`）拒绝。常数之间的算术（如 `close * (2 * 3)`）允许。
9. **一元正号 `+x` 拒绝**：PRD 只列了"取负"。
10. **窗口必须直接写整数字面量**：`5.0`、`True`、`2 + 3`、`-1` 都拒绝。
11. 额外加了两个防护：公式长度上限 2000 字符、嵌套深度上限 60 层。PRD 没有规定，如不需要可去掉。

## 3. 环境与集成

12. **测试环境**：我在 Python 3.13 / pandas 3.0.5 / numpy 2.5 上跑的单元测试，而 AGENTS 写的是 Python 3.11。`requirements.txt` 没锁版本，集成时在 3.11 上请再跑一次。
13. **导入路径**：代码和测试用 `from src import ...`，需要像 AGENTS 写的那样在项目根目录用 `python -m pytest` 运行（裸 `pytest` 不会把根目录加进 sys.path）。如需支持裸 `pytest`，需要在根目录加 `conftest.py` 或在 `pytest.ini` 加 `pythonpath = .`，这两处都不属于线 B。
14. **real 测试**依赖线 A 的 `src.data.load_market`，用 `pytest.importorskip("src.data")`，线 A 合并前会自动跳过。
