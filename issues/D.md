# 线 D 待确认问题

以下是规范里没写清楚或可能有冲突的地方。每条都已按最保守的理解实现，等人来确认。

## 流程细节（PRD 11 / 8.2 没写到的分支）

1. **v0 公式重写次数用完**：构建阶段校验失败、重写 `MAX_EXPR_RETRY` 次后仍不合规时，本轮记为 `skipped`，error 里写明原因（ExprRetryExhausted）。没有记成 `cannot_build`，因为那是 Agent 主动放弃时用的。
2. **调整阶段公式重写次数用完**：追加一个出错的 VersionRecord（`error` 有值，指标为 None），结束内层循环，按"未达标"处理并调用归因。这个版本占用 1 次调整次数。
3. **调整阶段 Agent 返回 `cannot_build`**：不产生新版本，结束内层循环，按"未达标"处理并调用归因；原因写进 `cannot_build_reason`（此时 `train_conclusion` 是 `fail`）。
4. **验证通过后入库时出错**（如 save_ref 失败）：本轮记为 `skipped`，`validation` 字段保留已经得出的结论，`library_id` 为 None，不写因子库。顺序是先 save_ref、再写因子库，保证不会出现"在库里但没有参照值"的因子。

## 信息可见性

5. **历史摘要的范围**：假设 Agent 看到的历史是 `runs/*/run_log.jsonl` 里**所有运行**中最近的 30 轮（按时间排序），不只是本次运行。PRD 12.3 只写了"最近 30 轮"。
6. 历史摘要里会出现 `skipped` 轮次，但只显示训练结论，不显示错误信息、公式和任何验证指标。
7. **验证集明细不落盘**：PRD 12.1 只要求记录验证结论。验证集的 Report 和 GateResult 用完即丢，只在 run_log 里记 `"pass"/"fail"`。如果以后需要人工查看验证明细，要另外约定存放位置，并保证 Agent 读不到。

## 与其他线的接口

8. **`evaluate(..., "train", refs)` 的 refs 可能是空 dict**（例如还没跑 build_refs）：调度程序在训练集上总是传 dict，不会传 None。需要线 C 确认空 dict 时 `max_ref_corr` / `max_ref_id` 为 None，并且 `gate_train` 遇到 None 时不报错。另外，参照集为空时 run.py 会打印警告。
9. **classic_factors.yaml 的格式**（线 B）：contracts 只定义了单个条目 ClassicFactor。`load_classics` 两种写法都接受：顶层直接是列表，或者 `{factors: [...]}`。
10. **load_refs 的返回值**：调度程序只负责传递，不读取内容。新因子入库后会把 `load_refs(cache_dir)` 的结果**合并**进当前参照集，不会整体替换。
11. **thresholds_sha256 / load_thresholds** 用的是线 C 的实现，run.py 传入的是 `config/thresholds.yaml` 的绝对路径。

## 路径与产物（规范没写，我自己定的）

12. `factor_library.jsonl` 固定放在项目根目录（PRD 6.6），不在 settings 里配置。
13. `check_classics.py` 的输出写到 `<runs_dir>/check_classics/<时间>.csv|.json`；`final_test.py` 的输出写到 `<runs_dir>/final_test_<时间>/results.json`。这两个目录下都没有 run_log.jsonl，所以不会进入 Agent 能看到的历史。
14. `final_test.py` 必须加 `--confirm` 才会加载全量数据。全项目只有这里用 `end=None`。
15. `build_refs.py` 默认跳过已经存在的 `cache/ref_values/<id>.parquet`，`--force` 全部重建。它不会自动检测公式或数据有没有变化（PRD 13 的"变了才重建"要靠人判断）。

## 归属表之外新增的文件（都在 D 线拥有的目录里）

16. 新增了 `src/agents/__init__.py`、`src/agents/common.py`（提示词渲染、JSON 解析与重试）、`scripts/_bootstrap.py`（让 `python scripts/xxx.py` 能 import `src`）。归属表里 `src/agents/` 和 `scripts/` 整个目录都属于 D，但这三个文件名没有列出来。

## 测试环境

17. 单元测试是在 Python 3.13 上跑的（云端没有 3.11）。代码没有用到 3.11 之后才有的语法。
18. `@pytest.mark.real` 的测试（`tests/test_smoke.py::test_real_mock_three_rounds`）已经写好，还没跑，留到集成时运行。
