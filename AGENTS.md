# AGENTS.md

给所有写代码的 Agent 看的施工规范。**业务规则以 `PRD.md` 为准，接口以 `src/contracts.py` 为准**，本文件不重复。三者冲突时，停下来写进 `issues/<你的线>.md`，不要自行决定。

## 项目一句话

S&P 500 单因子挖掘流水线：假设 Agent 提假设 → 因子 Agent 写公式 → 程序在训练集上评估并调优 → 验证集只回"通过/不通过" → 通过的入库。详见 PRD。

## 环境与命令

- Python 3.11，运行在 Windows（路径一律用 `pathlib`，不要写死 `\`）
- 依赖：`pandas numpy pyarrow pyyaml scipy pytest openai`（记在 `requirements.txt`）
- 安装：`python -m pip install -r requirements.txt`
- 全部单元测试：`python -m pytest -q`
- 单个模块：`python -m pytest tests/test_data.py -q`
- 用真实数据的测试标记为 `@pytest.mark.real`，默认不跑；手动跑：`python -m pytest -m real -q`
- 真实数据目录在 `config/settings.yaml` 的 `data_dir`（绝对路径，因为 worktree 里没有 data/）

## 目录与归属

只能修改**自己线**的文件。共享文件任何人都不能改。

| 线 | 拥有的文件 | 对应 PRD |
|---|---|---|
| A 数据层 | `src/data.py`、`tests/test_data.py` | 6 |
| B 表达式 + 经典因子 | `src/operators.py`、`src/expr.py`、`library/classic_factors.yaml`、`tests/test_expr.py` | 7、13 |
| C 评估 | `src/evaluate.py`、`src/refstore.py`、`tests/test_evaluate.py` | 9、10、13 |
| D Agent + 调度 + 集成 | `src/agents/`（`llm.py`、`hypothesis_agent.py`、`factor_agent.py`、`prompts/`）、`src/records.py`、`src/orchestrator.py`、`scripts/`（`run.py`、`build_refs.py`、`check_classics.py`、`final_test.py`）、`tests/test_agents.py`、`tests/test_smoke.py` | 8、11、12、15 |
| 共享（冻结） | `PRD.md`、`AGENTS.md`、`src/contracts.py`、`config/thresholds.yaml` | — |
| 共享（初始化时建，之后只有 D 可改） | `config/settings.yaml`、`requirements.txt`、`pytest.ini`、`.gitignore`、`src/__init__.py` | — |

运行产物（不进 git）：`data/`（原始数据，只读）、`cache/`、`runs/`、`factor_library.jsonl`。

## 绝对禁止

1. 写入、移动、删除 `data/` 下任何文件。
2. 修改 `config/thresholds.yaml`、`src/contracts.py`、`PRD.md`、`AGENTS.md`。
3. 填补缺失值（ffill、fillna 等）。缺失就是 NaN。
4. 在 `scripts/final_test.py` 以外的任何地方用 `end=None` 或晚于 `RUN_DATA_END` 加载数据。
5. 让 Agent 的输入里出现验证集指标、验证失败原因、任何测试集信息。
6. 自己写收益或标签：一律调用 `contracts.compute_ret` / `compute_fwd_ret`。
7. 把 `data/`、`cache/`、`runs/`、`.env`、API Key 提交进 git。

## 易错点（写代码前读一遍）

- **前视偏差**：`shift(k)` 中 k>0 取过去，k<0 取未来。因子只能用 k≥0。
- **标签不跨段**：IC 日期一律来自 `split_dates(market, split, h)`，每个 h 单独取。
- **时序 vs 截面**：时序算子在完整历史上算，不受股票池限制；截面算子（`cs_*`、`group_neutralize`）先 `x.where(universe)` 再算。
- **窗口不完整 = NaN**：`rolling(d, min_periods=d)`。
- **inf**：除以 0、log 非正数都转成 NaN。
- **permno** 一律 `int64`；日期一律 `pd.Timestamp`。
- **mktcap 单位是千美元**；panel 的 `ticker` 列是 `s+permno`，不是真代码。
- **行业按时点取**（PRD 6.2），不要用最新分类。
- **Windows 编码**：所有文件读写显式 `encoding="utf-8"`；`json.dumps(..., ensure_ascii=False)`。
- **性能**：宽表约 6,700 天 × 1,200 只。避免逐日逐股 Python 循环；`ts_rank` 用 `numpy.lib.stride_tricks.sliding_window_view`，Rank IC 用"逐行 rank 后按行算 Pearson"的向量化写法。
- 测试一律用 `contracts.make_fake_market()`，它埋了短期反转信号：`-ts_sum(ret, 5)` 的 Rank IC 应显著为正。

## 各线完成标准

每条线：所有函数签名与 `contracts.py` 第 4 部分一致，自己的测试全部通过，不依赖其他线的代码（需要别人的函数时，在测试里写最小替身）。

- **A**：fake 数据上 `split_dates` 不跨段、标签对齐正确；真实数据（`-m real`）上行数 5,260,971、股票数 1,195、日期 2000-01-03 至截止日、每日股票池约 500、185 只换过行业的股票在不同时期取到不同行业；第二次加载走缓存。
- **B**：`validate` 拒绝非白名单字段/算子/窗口、非法语法、`ts_delay`/`ts_delta` 以外用窗口 1，错误信息为中文；每个算子有单测（含 NaN、窗口不完整、截面只在股票池内）；`classic_factors.yaml` 20 条全部通过 `validate`，内容与 PRD 第 13 节一致。
- **C**：在 fake 数据上 `-ts_sum(ret,5)` 的 `ic_mean > 0`；`Report` 每个字段都有值且类型正确；两道门的每项检查有单测（含负向因子、符号相反、相关性超限）；`save_ref` → `load_refs` 往返一致。
- **D**：`src/agents/llm.py` 实现 `OpenAICompatibleLLM`（openai SDK，读 settings 的 base_url / model / api_key_env）和 `MockLLM`，都满足 `contracts.LLM`；不依赖任何服务商特有的功能（如 JSON mode、工具调用），JSON 由提示词约定并自行解析；Agent 输出的 JSON 解析失败会重试 1 次，再失败抛错；公式校验失败最多重写 `MAX_EXPR_RETRY` 次；`MockLLM` 下 `test_smoke.py` 跑 3 轮通过；人为注入异常时该轮记为 `skipped`，下一轮继续；每次 LLM 调用的 prompt 和回复写进 `runs/<run_id>/llm_log.jsonl`。

## git 规则

- 每条线在自己的 worktree 和分支上工作：分支名 `line-A` / `line-B` / `line-C` / `line-D`。
- 小步提交，提交信息以线名开头：`[A] 实现 split_dates`。
- 不要合并、不要 rebase 别人的分支，不要 push --force。合并只在集成阶段由主目录进行。
- 需要改共享文件或接口：不要改，写进 `issues/<线名>.md` 并在最终回复里说明。

## 初始化（只做一次，在主目录执行）

1. `git init`，建 `.gitignore`（含 `data/ cache/ runs/ factor_library.jsonl .env __pycache__/ .pytest_cache/`）。
2. 建 `requirements.txt`、`src/__init__.py`、`pytest.ini`（注册 `real` 标记，`addopts = -m "not real"`）。
3. 建 `config/thresholds.yaml`：

   ```yaml
   train:
     abs_ic_mean_min: 0.01
     abs_icir_min: 0.06
     max_abs_ref_corr: 0.7    # 严格小于
   valid:
     abs_ic_mean_min: 0.01
     abs_icir_min: 0.06
     same_sign_as_train: true
   ```

4. 建 `config/settings.yaml`：

   ```yaml
   data_dir: D:/Alpha-Agents/data
   cache_dir: cache
   runs_dir: runs
   n_rounds: 10
   llm:
     provider: openai_compatible   # 用 openai SDK 接入任何兼容 OpenAI 接口的服务
     base_url: ""                  # 由人填写，如服务商给的 https://.../v1
     model: ""                     # 由人填写
     temperature: 0.7
     max_tokens: 2000
     api_key_env: LLM_API_KEY      # 环境变量名，Key 本身不写进文件
   ```

   方法论常量（字段、算子、窗口、收益期限、切分日期、调整次数）在 `contracts.py`，不放 settings。
5. 首次提交，然后为四条线建 worktree：`git worktree add ../Alpha-Agents-A -b line-A`（B、C、D 同理）。

## 集成（四条线都完成后，在主目录执行）

1. 依次合并 `line-A`、`line-B`、`line-C`、`line-D`，每合并一次跑 `python -m pytest -q`。
2. `python -m pytest -m real -q`（真实数据测试）。
3. `python scripts/build_refs.py`：用表达式引擎算 20 个经典因子，存入 `cache/ref_values/`。
4. `python scripts/check_classics.py`：输出 20 个经典因子的训练集明细表，交给人核对方向（PRD 15-2）。
5. `python scripts/run.py --rounds 3 --mock`，然后由人填好模型后 `python scripts/run.py --rounds 10`。
6. 对照 PRD 第 15 节逐条验收。
