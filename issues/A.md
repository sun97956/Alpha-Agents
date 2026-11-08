# 线 A（数据层）问题记录

实现时没有读取真实数据，下列各项均按最保守的理解处理。集成阶段跑 `python -m pytest -m real -q` 时请重点核对 1、2、6。

## 1. panel.parquet 的位置不一致
- PRD 6.1 说 panel 在 `data/` 下；`build_panel.py` 的输出路径是 `D:\Alpha-Agents\panel.parquet`，也就是 `data_dir` 的上一级。
- 处理：先找 `data_dir/panel.parquet`，找不到再找 `data_dir.parent/panel.parquet`，两处都没有就报错。
- 建议：人确认后统一放到一处。

## 2. 原始 CSV 的列名未知
- 没读真实文件，所以列名按 CRSP/Compustat 的常见命名做了大小写不敏感的候选匹配（见 `src/data.py` 顶部的 `_*_COLS`）：
  - 成分表：permno；进入日期候选 `start/start_date/mbrstartdt/from/...`；退出日期候选 `ending/end/end_date/mbrenddt/thru/...`
  - 证券信息表：`permno, gsector, indfrom, indthru`
  - ticker 映射：permno；ticker 候选 `ticker/tic/symbol/tsymbol`
- 匹配不到时直接报错，并列出实际列名，不会猜。真实表头如果不在候选里，只需在候选元组里加一项。
- 日期格式支持 YYYYMMDD（整数、字符串、`20000103.0`）和 ISO 字符串，空值视为"至今"。

## 3. ret 的口径：PRD 与 contracts 不完全一致
- PRD 6.4："ret 在该股自己的序列上计算"（如果某股中间缺了一天，可以理解为用它上一条记录的 close）。
- contracts.compute_ret："按全局交易日历移位；前一日缺失则为 NaN"。
- 处理：按 AGENTS 禁止第 6 条，一律调用 `contracts.compute_ret`。所以个股中途停牌后的第一天 ret 为 NaN。

## 4. 行业按时点取：PRD 没写的情形
- 两段分类之间有空档、或最后一段的 indthru 已过：取 **NaN**，不向前或向后借用。
- 同一 permno 有多行 indfrom 相同且区间重叠：取文件中靠后的一行。
- indfrom 为空的行直接丢弃，因为无法定位时点。
- gsector 不是数字时记为 NaN。

## 5. 成分区间
- [进入日期, 退出日期] 按闭区间处理，退出日期为空表示至今。同一 permno 可以有多段（再次入选）。
- 成分表里有、但 panel 里没有的 permno 直接忽略（当天没有 close，本来也进不了股票池）。

## 6. 完成标准与"禁止 end=None"冲突
- A 的完成标准要求"行数 5,260,971、股票数 1,195、日期 2000-01-03 至截止日"。前两个数字是全量 panel 的，而 AGENTS 禁止第 4 条不允许在 final_test 以外用 `end=None` 加载。另外 185 只换行业的股票中，有些换行业时间可能晚于 2021-12-31。
- 处理（真实数据测试，均未运行）：
  - `load_market` 只用默认 `end=RUN_DATA_END` 加载；
  - 5,260,971 用 parquet 元数据的行数核对；1,195 只读 panel 的 permno 一列核对，不加载截止日之后的行情。人如果认为这一列也不该读，可以删掉这一断言；
  - 宽表的记录数与"截止日前 panel 的 close 非空行数"核对；
  - 185：先从证券信息表统计"有 ≥2 个不同 gsector 的 permno"是否等于 185，再只对换行业发生在已加载区间内的股票，检查宽表里确实取到了 ≥2 个行业。
- "每日股票池约 500"：测试写成中位数在 490–510、每日在 450–550。这个范围是我设定的，需要人确认。

## 7. Market.end 的取值
- contracts 只说"数据截止日"。这里取**实际加载到的最后一个交易日**（不晚于传入的 end），与 make_fake_market 中 `end=dates[-1]` 的做法一致。

## 8. 缓存
- 每个 (end, panel/成分表/证券信息表的大小与修改时间, 缓存版本号) 组合各对应一个目录：`cache/market_<end>_<hash>/`。字段、universe、sector 各存一个 parquet，fwd_ret 不存，读缓存时用 `compute_fwd_ret` 重算。
- 旧缓存不会自动删除（不删文件）。如需清理，由人手动删 `cache/`。
- 写缓存时先写临时目录、再整体改名。失败时只清理本次自己建的临时目录，不碰其他文件。

## 9. 本线额外提供的函数（不在 contracts 里）
- `load_benchmarks(data_dir, end=RUN_DATA_END)`：读 SPY 和 SP500_TR，按日期排序（PRD 6.2-3）并截断到 end。MVP 不参与计算，只是把"只读入"落实下来。
- `ticker_map`：一个 permno 有多行时，有日期列就取日期最晚的一行，否则取文件中最后一行。

## 10. 环境（需要线 D 处理的共享文件）
- `pytest.ini` 没设 `pythonpath = .`，所以直接运行 `pytest -q` 会报 `No module named 'src'`，只有 `python -m pytest -q` 能跑通（AGENTS 里写的就是这个命令）。建议 D 在 pytest.ini 里加 `pythonpath = .`。
- `requirements.txt` 没有锁版本。pandas 2.x 与 3.x 的默认日期精度不同（ns 和 us）。本线跟随当前 pandas 的默认精度（与 make_fake_market 一致），已在 pandas 2.3 和 3.0 两个版本下各跑一遍单元测试，都通过。建议 D 锁定一个大版本。
