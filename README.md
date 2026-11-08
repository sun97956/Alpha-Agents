# Alpha-Agents

面向美股 S&P 500 的多智能体单因子挖掘系统。以**假设驱动**为核心：Agent 提出有经济逻辑的假设并写成因子公式，程序负责全部计算和门槛判定，通过训练集、验证集两道门的因子进入因子库。

## 工作流程

```
假设 Agent ──► 因子 Agent ──► 训练集评估 ──► 未达标则调整公式（有次数上限）
  (机制+持有期)    (写公式)      (2000–2016)          │
                                   │ 达标             ▼
                                   ▼             复盘（每轮都做）
                              验证集评估 ──► 通过则入库
                              (2017–2021，只回"通过/不通过")
```

- **假设 Agent**：提出一条假设，写明经济逻辑，并声明机制类别和预期持有期（1 / 5 / 20 日）。
- **因子 Agent**：把假设写成公式；根据训练集明细调整公式；每轮结束后做复盘，写下结论和一句话经验。
- **程序**：校验公式、计算因子值和全部指标、执行门槛判定、记录运行过程、汇总机制经验表。

数字一律由程序计算，门槛一律由程序判定；Agent 只负责提出假设和写公式。

## 主要设计

| 方面 | 做法 |
|---|---|
| 数据切分 | 训练 2000–2016、验证 2017–2021、测试 2022–2026.9（运行期间不加载测试集） |
| 标签 | t 日收盘后算因子，未来收益 = close(t+1+h) / close(t+1) − 1，h 为声明的持有期 |
| 评价 | 每日截面 Rank IC；明细含 1/5/10/20/60 日各期限表现、分年表、分组收益、换手 |
| 门槛 | 统一写在 `config/thresholds.yaml`，训练门和验证门同一套，按声明的持有期判定，不取绝对值；与已入库因子相关性 < 0.8 |
| 信息隔离 | 验证集只回传通过/不通过；测试集完全隔离；参考因子的表现数据不给 Agent |
| 参考因子 | 20 个经典因子 + 36 个从 WorldQuant 101 筛选转写的因子，作为思想来源和积木 |
| 经验积累 | 每轮复盘 + 程序按机制类别汇总的经验表，供假设 Agent 选题参考 |
| 可追溯 | 每轮记录假设、每版公式及成绩、复盘；每次模型调用的提示词、回复、思考过程、用量都有日志 |

完整规则见 [`PRD.md`](PRD.md)。修改代码的规范见 [`AGENTS.md`](AGENTS.md)。

## 目录

```
src/
  contracts.py        冻结的接口与方法论常量
  data.py             数据层：宽表、股票池、按时点取的行业、未来收益、三段切分
  operators.py        算子实现
  expr.py             因子表达式的校验与计算
  evaluate.py         回测明细与门槛判定
  refstore.py         去重参照值的存取
  records.py          运行记录、因子库、历史摘要、机制经验表
  orchestrator.py     调度主循环
  agents/             LLM 封装、两个 Agent、提示词（prompts/）
scripts/
  run.py              运行若干轮
  check_classics.py   核对经典因子在训练集上的表现（只给人看）
  build_refs.py       生成去重参照值
  final_test.py       项目最后在测试集上评估因子库（人工运行一次）
library/              经典因子库、WQ101 参考库（只读）
config/               settings.yaml（可改）、thresholds.yaml（冻结）
docs/                 WQ101 筛选清单
tests/                单元测试（假数据）与真实数据测试
```

## 数据

数据源自 CRSP，**不包含在仓库中**。需要放在 `config/settings.yaml` 的 `data_dir` 指定的目录下：

- `panel.parquet`：日频行情长表（date、permno、open、high、low、close、volume、mktcap）
- `MASTER_UNIVERSE_sp500_membership_extended_asOf20260930_v2.csv`：S&P 500 成分区间
- `security_master_with_provisional_classifications.csv`：GICS 行业分类及起止日期
- `permno_to_ticker_mapping.csv`：permno 与 ticker 映射

## 使用

环境：Python 3.11，Windows。

```powershell
# 安装
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# 测试
.\.venv\Scripts\python -m pytest -q            # 单元测试（假数据）
.\.venv\Scripts\python -m pytest -m real -q    # 真实数据测试

# 运行（模型通过 OpenAI 兼容接口接入，在 settings.yaml 填 base_url 和 model）
$env:LLM_API_KEY = "你的key"
.\.venv\Scripts\python scripts\run.py --rounds 10

# 不调用真实模型的模拟运行
.\.venv\Scripts\python scripts\run.py --rounds 3 --mock
```

运行产物：

- `runs/<run_id>/run_log.jsonl`：每轮的完整记录
- `runs/<run_id>/reports/`：每版公式的训练集明细
- `runs/<run_id>/llm_log.jsonl`：每次模型调用的完整 trace
- `factor_library.jsonl`：通过验证的因子

## 说明

- WorldQuant 101 的公式来自 Kakushadze, *101 Formulaic Alphas*（arXiv:1601.00991），论文注明公式为 WorldQuant 专有、经授权发表。本仓库中的转写仅供研究使用；经济解读除 #101 外均为推测。
- 项目处于 MVP 阶段，只做单因子挖掘，不涉及组合构建、多因子合成和交易成本。
