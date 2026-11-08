# 线 C 待确认问题

以下是规范没说清楚的地方，均已按最保守的理解实现，等人确认。

1. **`split_dates` 的依赖**：`evaluate` 必须用 `split_dates`（归线 A），但完成标准又要求"不依赖其他线"。
   现做法：`src/evaluate.py` 优先 `from src.data import split_dates`；模块不存在时用本地替身 `_split_dates_fallback`（严格按 contracts 注释实现）。集成后自动切换为线 A 的实现，替身不再生效。

2. **训练门在 `max_ref_corr is None` 时的判定**：contracts 允许 `refs=None`，此时 `max_ref_corr=None`，但 PRD 没说训练门怎么判。
   现做法：判为不通过（check 的 value 为 NaN，passed=False），理由是没有参照集就无法证明不重复。传入空字典 `refs={}` 同样会得到 None，也判不通过。
   **对线 D 的影响**：调度程序在训练段必须传入 refs（至少包含 20 个经典因子），否则任何因子都过不了训练门。如果希望"无参照 = 通过"，需要人来决定。

3. **向非训练段传 refs**：contracts 写的是"refs 只在 split='train' 时传入"。现做法：`split != "train"` 时传了 refs 就抛 `ValueError`。

4. **口径细节（PRD 没写死，按以下实现）**：
   - `ic_std`、分年 ICIR 用样本标准差（ddof=1）；有效天数 < 2 时为 NaN。
   - `coverage_mean`：在主口径（h=5）的 `split_dates` 日期上，每日股票池内有因子值的股票数，再对日期求均值（含被跳过的日子）。
   - `quantile_ret`：只用主口径的有效 IC 日（有效股票数 ≥ 100），且只在因子值和 fwd_ret_5 都有值的股票里分组；每组按 rank 均分（`floor((rank-1)*5/n)`，并列值按出现顺序拆开），先算每组当日的平均收益，再对日期求均值。
   - `turnover_autocorr`：用相邻两个交易日 (t−1, t)，两天都要落在主口径的 `split_dates` 内；只用两天都有因子值的股票做 Spearman；有效股票数 < 100 的配对跳过。
   - `max_ref_corr`：在训练段逐日、只在两个因子都有值的股票里算 Spearman（按公共集合重新排序），有效股票数 < 100 的日子跳过，再对日期求均值；对所有参照因子取 |均值| 最大的那个，保留符号。所有日子都无效的参照因子不参与比较。
   - 指标算不出来时（如 0 个有效 IC 日）字段值为 `float('nan')`，门槛判为不通过。注意：`json.dumps` 会把 NaN 写成 `NaN`，这不是标准 JSON，线 D 写报告时需要知道这一点。

5. **refstore 文件格式**：parquet 要求列名是字符串，所以 permno 落盘时转成 str，`load_refs` 读回时恢复成 int64；读回的 index 不带 freq。
