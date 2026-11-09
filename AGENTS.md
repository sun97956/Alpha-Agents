# AGENTS.md

给所有修改本项目代码的 Agent 看的规范。**业务规则以 `PRD.md` 为准，接口以 `src/contracts.py` 为准**，本文件不重复。三者冲突时停下来问人，不要自行决定。

## 项目一句话

S&P 500 单因子挖掘流水线：假设 Agent 提假设 → 因子 Agent 写公式 → 程序在训练集上评估并调优 → 验证集只回"通过/不通过" → 通过的入库。详见 PRD。

## 现状

- 并行开发（A/B/C/D 四条线）已结束并合并到 master，现在是维护和实验阶段。
- 假设要声明机制类别和持有期（1/5/20 日）；门槛按声明的持有期判定：IC ≥ 0.015、ICIR ≥ 0.08（训练、验证、各持有期统一，不取绝对值），与已入库因子相关性 < 0.8。
- 每轮训练结束后复盘，程序汇总机制经验表给假设 Agent。回测明细包含 1、5、10、20、60 日各期限表现。
- Agent 的参考因子：`library/classic_factors.yaml`（20 个）+ `library/wq101_factors.yaml`（36 个），给 Agent 公式和逻辑，不给表现数据。

## 环境与命令

- Python 3.11，运行在 Windows（路径一律用 `pathlib`，不要写死 `\`）；虚拟环境在 `.venv`
- 依赖见 `requirements.txt`（pandas numpy pyarrow pyyaml scipy pytest openai）
- 全部单元测试：`.\.venv\Scripts\python -m pytest -q`
- 真实数据测试：`.\.venv\Scripts\python -m pytest -m real -q`
- 运行：`.\.venv\Scripts\python scripts\run.py --rounds 10`（需环境变量 `LLM_API_KEY`）
- 模拟运行：`scripts\run.py --rounds 3 --mock`（记录在 `runs/mock_*`，不进入历史、不写真实因子库）
- 经典因子核对（只给人看）：`scripts\check_classics.py`
- 最终测试（项目最后由人运行一次）：`scripts\final_test.py --confirm`

## 目录

| 位置 | 内容 |
|---|---|
| `src/data.py` | 数据层：宽表、股票池、按时点取的行业（sector / industry / subindustry）、未来收益、三段切分 |
| `src/operators.py`、`src/expr.py` | 算子与表达式引擎（校验 + 计算） |
| `src/evaluate.py`、`src/refstore.py` | 回测明细与门槛判定；参照因子值存取（去重启用时才用） |
| `src/agents/` | LLM 封装、两个 Agent、提示词（`prompts/`） |
| `src/orchestrator.py`、`src/records.py` | 调度主循环、运行记录、因子库读写、历史摘要 |
| `scripts/` | run / check_classics / build_refs / final_test |
| `library/` | 参考因子库（只读） |
| `config/` | `settings.yaml`（可改）、`thresholds.yaml`（冻结） |
| `docs/` | WQ101 论文与筛选清单 |

运行产物（不进 git）：`data/`（原始数据，只读）、`cache/`、`runs/`、`factor_library.jsonl`。

## 绝对禁止

1. 写入、移动、删除 `data/` 下任何文件。
2. 未经人同意修改 `config/thresholds.yaml`、`src/contracts.py`、`PRD.md`、`AGENTS.md`。
3. 填补缺失值（ffill、fillna 等）。缺失就是 NaN。
4. 在 `scripts/final_test.py` 以外的任何地方用 `end=None` 或晚于 `RUN_DATA_END` 加载数据。
5. 让 Agent 的输入里出现验证集指标、验证失败原因、任何测试集信息、参考因子的表现数据。
6. 自己写收益或标签：一律调用 `contracts.compute_ret` / `compute_fwd_ret`。
7. 在提示词里写死门槛数字：门槛一律从 `thresholds.yaml` 读取后渲染。
8. 把 `data/`、`cache/`、`runs/`、`.env`、API Key 提交进 git。

## 易错点

- **前视偏差**：`shift(k)` 中 k>0 取过去，k<0 取未来。因子只能用 k≥0。
- **标签不跨段**：IC 日期一律来自 `split_dates(market, split, h)`，每个 h 单独取。
- **时序 vs 截面**：时序算子在完整历史上算，不受股票池限制；截面算子（`cs_*`、`group_neutralize`）先 `x.where(universe)` 再算。
- **窗口不完整 = NaN**；**inf** 一律转 NaN。
- **permno** 一律 `int64`；日期一律 `pd.Timestamp`。
- **mktcap 单位是千美元**；panel 的 `ticker` 列是 `s+permno`，不是真代码。
- **行业按时点取**（PRD 6.2），不要用最新分类。
- **Windows 编码**：所有文件读写显式 `encoding="utf-8"`；`json.dumps(..., ensure_ascii=False)`。
- **改数据生成逻辑**要同时改 `src/data.py` 里的 `CACHE_VERSION`，旧缓存才会失效。
- **写文件到用户电脑后要回读核对**：曾发生过"写入的是改动前的版本"的问题。
- 单元测试一律用 `contracts.make_fake_market()`，它埋了短期反转信号：`-ts_sum(ret, 5)` 的 Rank IC 应显著为正。

## 改代码的流程

1. 改之前读 PRD 对应章节和 `contracts.py`。
2. 改完跑 `python -m pytest -q`，全部通过；动到数据层时再跑 `-m real`。
3. 涉及规则（门槛、可见性、算子、参考库）的改动，同步更新 PRD。
4. 小步提交，提交信息写清改了什么、为什么。
