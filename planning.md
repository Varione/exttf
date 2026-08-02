# ETF / 场外基金量化研究项目执行规划

## 2026-07-31 当前执行计划：基线封版、历史真实性与新鲜 OOS

> 本节是当前唯一有效的执行计划，优先级高于文件下方所有历史审核记录。下方旧计划仅用于追溯，不再作为完成状态、测试数量、策略结论或下一步任务的事实来源。

### 0. 当前状态与审核基线

截至 2026-08-02，项目已经完成 ETF 数据、场外基金映射、产品规则、执行日历、FIFO 申赎、连续账户、候选策略、压力测试、实验产物验证、收益归因和前向研究工具链等主要基础设施建设。冻结基线（2026-08-01）与 8 月 2 日数据修订均已记录。当前问题已经从“功能是否存在”转变为“历史证据覆盖是否补齐、合法 T0 是否启动、前向影子观察是否开始”。

当前统一状态为：

```text
BASELINE_FREEZE_COMPLETED
POST_FREEZE_DATA_REFRESH_RECORDED
CURRENT_WORKTREE_DIRTY
CURRENT_FULL_REGRESSION_STALE
HISTORICAL_TRUTH_GATE_FAILED
T0_REGISTRATION_REQUIRES_CORRECTION
FORWARD_SHADOW_NOT_STARTED
NO_PAPER_TRADE_CANDIDATE
```

已确认的事实：

- `reports/latest_research_status.json` 为唯一状态源，已覆盖全部研究并指向冻结重跑后的 run_id（`walkforward_20260801_121725` 等）；2026-08-02 全量回归记录 `reports/test_regression/regression_20260802_013337.json`（601 passed，40 个测试文件），覆盖当前全部测试；
- 工程基线已在干净 commit `3bd2837` 上完成冻结重跑（WF/C1/C2/C3/M20/归因），输入哈希与封版前完全一致，产物 `baseline_freeze_manifest.json` 已生成；
- 2026-08-02 完成数据增量刷新并写入 `config/data_revision_registry.json`（otf_expanded 新增 26,999 行、otf_mapped 新增 712 行、执行日历扩展至 2026-07-31）；7 月 28 日至 7 月 31 日属于事后回填，标记为 `POST_FREEZE_BACKFILLED_VALIDATION_WINDOW`，不得计入新鲜 OOS；
- T0 定义修正（2026-08-02 审核）：T0 必须是严格晚于冻结完成时间（2026-08-01）且数据按前向流程采集的执行交易日；7 月 28 日早于冻结时间，不能作为 T0；当前 `config/otf_t0_registry.json` 为 `PENDING_CALENDAR_EXTENSION`，待日历扩展到冻结时间之后且首个前向日有数据时自动登记；
- 在完成本计划 P0 和 P1 前，禁止新增 C4/C5、重新搜索动量周期、修改 Gate 门槛或继续扩展 ML/RL。

---

### 1. 本阶段总目标

本阶段不以提高 CAGR 为目标，而以形成一条可冻结、可复算、可审计、可进入全新前向样本的研究链为目标。

完成定义：

```text
当前代码全量测试通过
+ C3 产物与当前 validator 一致
+ 唯一状态源覆盖全部最新研究线
+ 干净 Git commit 上完成冻结基线运行
+ 核心产品历史规则与时序证据达到最低门槛
+ 冻结参数后启动从未用于开发的新鲜 OOS
```

执行顺序严格为：

```text
P0 工程与产物封版
→ P1 历史真实性补齐
→ P2 冻结策略与新鲜 OOS
→ P3 满足证据后再决定是否开展新研究
```

任何前序 Gate 未通过，不得跨阶段推进。

---

## 2. P0：工程与研究基线封版

### P0-1 当前代码全量回归

#### 执行内容

1. 固定使用项目解释器：

```text
D:\miniconda\envs\agents\python.exe
```

2. 在项目根目录运行：

```text
D:\miniconda\envs\agents\python.exe -m pytest -q -W error::FutureWarning
```

3. 测试必须覆盖当前全部 40 个测试文件，重点包括：
   - C3 信号与 runner；
   - artifact validator；
   - execution calendar；
   - 节假日份额调整与跨基金节假日收益归集；
   - metadata consistency；
   - strategy attribution；
   - 连续账户、FIFO、费用、换手和未来数据隔离；
   - NAV 可用时点模型、前向数据 revision 与 T0 时间语义、增量 NAV 刷新保护（2026-08-02 新增）。
4. 将原始测试输出、执行时间、Python 版本、Git commit、dirty 状态和测试文件数量写入独立机器可读产物。
5. 禁止继续手工把某次历史 `passed` 数写入 README、planning 或 conclusion。

#### 验收标准

- 失败数和错误数均为 0；
- 不产生 FutureWarning；
- 测试记录对应当前 commit 和当前工作树状态；
- 测试产物可以被 `latest_research_status.json` 自动引用；
- 若当前测试未通过，停止后续所有正式实验，只允许修复测试和代码一致性问题。

### P0-2 C3 重新定版

#### 执行内容

1. 使用当前 `src/otf_rotation/artifact_validation.py` 对最新 C3 运行重新 finalize；
2. 不修改 C3 参数、Gate 门槛、数据区间或基准定义；
3. 重新生成：
   - `c3_status.json`；
   - `gate_result.json`；
   - `manifest.json`；
   - `finalize_audit.json`；
   - `c3_conclusion.md`；
   - 完整 artifact inventory；
4. C3 只要求 `market_states.csv` 非空；`state_scores.csv`、`asset_budgets.csv`、`sleeve_weights.csv`、`fund_weights.csv` 和 `risk_contributions.csv` 必须按当前 schema 标记为 `NOT_APPLICABLE`；
5. 顶层状态、策略目录状态和 manifest inventory 必须使用同一次最终验证结果。

#### 验收标准

- `artifact_validation.passed=true`；
- `artifact_validation_final.passed=true`；
- 不再出现 `required_nonempty_for_C3_LOW_TURNOVER_MOMENTUM` 的旧错误；
- 根目录与策略子目录中的 Gate、状态、SHA256 和结论一致；
- C3 最终结论仍按原冻结门槛判定，不因修复产物而改变策略研究结论。

### P0-3 建立当前唯一事实源

#### 执行内容

重新生成 `reports/latest_research_status.json`，至少覆盖：

- 当前全量测试结果；
- 当前 Git commit 和 dirty 状态；
- 数据库、规则、映射、执行日历和配置 SHA256；
- B2-LT、C1、C2、C3、M20 的最新有效 run_id；
- 执行日历修复状态；
- 最新归因 run_id 与勾稽状态；
- 每条策略的研究标签、Gate、历史真实性、样本标签和是否允许进入前向观察；
- 当前阻塞项；
- 被新运行替代的旧 run_id。

`conclusion.md`、README 的当前状态摘要和后续报告必须从该状态文件及其引用产物生成，禁止各自保存不同数字。

#### 验收标准

- `latest_research_status.json` 的时间不早于当前最终运行；
- 测试数量、策略指标、Gate、run_id、输入哈希在所有当前文档中一致；
- 输入文件发生变化时，旧状态自动变为 `STALE_INPUTS`；
- 不再将 401、416、426 等历史测试结果混用为当前测试结论。

### P0-4 工程目录与依赖收口

#### 执行内容

1. 将根目录一次性脚本分类迁移到：

```text
tools/
scripts/diagnostics/
scripts/migrations/
docs/archive/
```

2. 清理或归档：
   - 临时修复脚本；
   - 重复比较脚本；
   - 旧测试生成脚本；
   - 0 字节数据库占位文件；
   - 已被最终运行替代的临时状态文件。
3. 同步 `README.md`、`pyproject.toml` 和实际 import：
   - 基础回测依赖；
   - 数据抓取依赖；
   - 研究绘图依赖；
   - ML 可选依赖；
   - RL 可选依赖；
   - dev/test 依赖。
4. 更新 `.gitignore`，确保大体积数据、报告、模型、缓存和临时日志不会误入版本库，同时保留必要配置、测试和小型审计摘要。
5. 正式冻结前必须得到干净工作树；禁止在 `git_dirty=true` 的运行上声明最终研究基线。

#### 验收标准

- 根目录不再堆放一次性修补脚本；
- `README.md` 的数据规模、测试状态和运行入口与当前代码一致；
- 新环境可按 `pyproject.toml` 安装并执行最小 smoke test；
- 正式基线 commit 具有明确标签或记录；
- 基线运行开始前 `git status` 为 clean。

### P0-5 干净 commit 上冻结重跑

#### 执行内容

在 P0-1 至 P0-4 全部通过后，使用同一干净 commit 和同一数据快照重新运行：

1. B2-LT 静态低换手基准；
2. C1 核心—卫星动量对照；
3. C2 低换手核心—卫星历史对照；
4. C3 低换手动量对照；
5. M20 映射基金动量观察线；
6. 最新归因；
7. 双倍费用和延迟压力测试；
8. artifact validation 和重复运行一致性检查。

不得在本次重跑中修改任何策略参数。

#### 验收标准

- 所有正式运行 `git_dirty=false`；
- 同配置重复运行的核心日收益、订单、费用和指标一致；
- 每个运行均有完整 manifest、配置快照、输入哈希和 artifact inventory；
- `latest_research_status.json` 只引用本次干净基线运行；
- 输出一份 `baseline_freeze_manifest.json`，记录 commit、数据截止日、策略配置哈希和正式基线 run_id。

---

## 3. P1：历史真实性与时间点证据

### P1-1 核心产品历史规则版本化

优先处理冻结策略实际使用或可能使用的产品，不追求一次性覆盖全部 448 只研究基金。

每条规则必须逐版本记录：

```text
fund_code
rule_version_id
effective_from
effective_to
channel
source_type
source_url
verified_at
subscription_confirmation_days
redemption_confirmation_days
redemption_arrival_days
subscription_fee_tiers
redemption_fee_tiers
minimum_holding_days
subscription_limit
suspension_or_termination_event
```

要求：

- 当前页面只能作为当前快照，不能自动延伸为历史事实；
- 无法证明历史有效期的规则只允许用于保守场景；
- 每笔正式历史订单必须能够追溯到提交日命中的规则版本；
- 规则真实性 Gate 与“当前快照保守执行 Gate”继续分开。

验收：冻结策略实际成交产品的订单级历史规则覆盖率达到 100%，否则仍保持 `HISTORICAL_RULE_STATUS=NOT_ESTABLISHED`。

### P1-2 NAV 发布时间与跨市场可用时点

当前只有 NAV 所属日期，缺少真实发布时间。下一步必须建立可用时点模型：

- 国内基金默认披露时滞的证据和保守规则；
- 港股、美股、QDII 的海外收盘、净值所属日、披露日和境内可见日；
- 周末、境内外节假日错位；
- 货币基金万份收益的公布与计入时点；
- 数据修订或补录的处理方式。

要求所有信号使用 `available_at <= signal_cutoff` 的信息。没有发布时间证据时必须应用预注册保守滞后，并标记：

```text
NAV_PUBLICATION_TIMESTAMP_NOT_AVAILABLE
CONSERVATIVE_AVAILABILITY_LAG_APPLIED
```

验收：修改信号日之后公布的数据，不得改变历史信号、产品选择或协方差估计。

### P1-3 生命周期与生存偏差

建立独立生命周期表，覆盖：

- 成立；
- 暂停申购；
- 恢复申购；
- 限购；
- 清盘；
- 合并；
- 转型；
- 终止；
- 份额类别新增或取消。

要求：

- 历史候选集只能包含当时存在且可研究的产品；
- 清盘、合并和终止产品不能从历史研究池中静默消失；
- 生命周期缺失的产品必须显式标记 `PIT_PARTIAL`；
- 生命周期事件应驱动强制退出或禁止新增，而不是仅依赖当前产品目录。

### P1-4 映射证据与独立数据核验

1. 对冻结策略用到的全部产品逐只确认：
   - `fund_family_id`；
   - `share_class`；
   - `underlying_id`；
   - 跟踪标的或业绩基准；
   - 资产袖套；
   - 生效区间；
   - 官方来源和复核记录。
2. 同一基金家族和同一暴露不允许重复占用独立资产名额；
3. 对冻结策略用到的 NAV、复权收益和关键费用，建立独立来源样本核验；
4. 全库覆盖率可以逐步提高，但冻结策略的实际持仓和候选产品必须先达到完整证据覆盖。

### P1-5 历史真实性 Gate

只有同时满足以下条件，才能把研究状态从 `PIT_PARTIAL` 升级：

- 实际持仓和候选产品生命周期证据完整；
- 实际订单规则版本在提交日有效；
- 信号使用的信息具有可证明的可用时点或预注册保守滞后；
- 资产映射有官方或可核验来源；
- 独立数据样本核验通过；
- 不存在使用当前快照回填全部历史的情况。

未通过时，所有结果只能称为：

```text
RETROSPECTIVE_RESEARCH_UNDER_PARTIAL_PIT
CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO
```

---

## 4. P2：冻结策略与真正新鲜 OOS

### P2-1 冻结研究对象

| 策略 | 后续角色 | 决策 |
|---|---|---|
| B2-LT | 低换手静态控制组 | 保留并冻结 |
| C1 | 高收益、高换手动态对照 | 保留为诊断对照，不作为候选 |
| C2 | 低换手改造失败样本 | 归档，不再调参 |
| C3 | 低换手动量动态对照 | 完成定版后冻结观察，不作为候选 |
| M20 | 映射基金趋势观察线 | 标准化产物后冻结观察，不作为已验证策略 |
| B1/B3/S1 | 历史基础设施回归基准 | 保留测试，不继续策略开发 |
| ML/RL | 延后研究线 | P0、P1、P2 未完成前暂停 |

冻结内容必须包括：

- 策略代码 commit；
- 数据截止日；
- 产品池；
- 参数；
- Gate 门槛；
- 交易规则场景；
- 执行日历；
- 配置 SHA256；
- 参数冻结 ID。

### P2-2 建立新鲜样本起点 T0

`T0` 不在 planning 中预填固定日期，由 P0-5 干净基线运行成功后自动确定：

```text
T0 = 干净基线冻结后的第一个可用执行交易日
```

规则：

1. T0 之后的数据不得用于回改 T0 之前的参数；
2. T0 后任何参数、产品池、映射或 Gate 修改都生成新策略版本，旧版本继续独立记录；
3. 数据供应商补录和历史修订必须生成 revision 记录，不得静默重算并覆盖原前向结果；
4. 前向观察使用实际当时可见数据和当时规则；
5. 允许更新净值和规则事实，不允许基于前向表现调参后继续宣称同一冻结版本。

### P2-3 前向影子观察要求

前向阶段只做研究观察，不下真实订单。每次决策保存：

- 当时可见数据快照哈希；
- 信号观察时间；
- 目标权重；
- 模拟提交、确认和到账日期；
- 命中规则版本；
- 预估及实际可核验费用；
- 拒单、限购和延期；
- 目标与实际模拟持仓偏差；
- 日度净值和归因。

最低评价样本：

- 至少 252 个新鲜交易日；
- 季度策略至少 4 个独立决策时点；
- 月度策略至少 12 个独立决策时点；
- 期间不得修改冻结参数；
- 必须经历至少一个明显风险阶段，否则只报告观察结果，不作升级结论。

### P2-4 前向升级 Gate

策略只有同时满足以下条件，才允许成为 `PAPER_TRADE_CANDIDATE`：

1. P0 工程 Gate 全部通过；
2. P1 历史真实性 Gate 通过；
3. 满足最低新鲜样本长度；
4. 前向会计、费用、订单和规则勾稽全部通过；
5. 未发生冻结参数漂移；
6. 绝对风险、成本、换手和滚动窗口 Gate 通过；
7. 动态策略相对 B2-LT 达到预注册增量门槛；
8. 双倍费用和执行延迟压力测试仍通过；
9. 所有产物来自干净 commit，且可重复验证。

在此之前，禁止使用“可实盘”“已验证 Alpha”或“正式纸面交易候选”等表述。

---

## 5. P3：满足证据后的后续研究

P3 只有在 P0、P1 完成并取得首个前向检查点后才允许启动。每次只能注册一个明确假设，禁止同时搜索多个参数维度。

候选研究方向按优先级排序：

1. **收益来源与换手来源审计**：先解释 C1 相对 B2-LT 的收益来自哪些资产、阶段和调仓事件，不直接继续调参；
2. **M20 暴露去重与稳定性**：检查黄金、红利、行业主题和重复指数暴露，使用预注册去重规则而非回看收益挑选；
3. **执行成本和产品替代研究**：在同一暴露下比较不同份额、渠道、持有期和费用结构；
4. **真正 Walk-forward 参数冻结**：只有需要训练或估计参数的策略才建立训练窗—冻结—独立 OOS 流程；
5. **Regime/ML/RL 增量**：仅当简单冻结基准通过历史真实性和前向 Gate 后，才评估复杂模型的独立增量。

任何新研究必须预先写明：

- 单一研究假设；
- 固定训练区间；
- 参数搜索空间；
- 验证区间；
- 完全独立 OOS；
- 主要指标；
- 失败停止规则；
- 多重比较控制方式。

---

## 6. 建议里程碑

| 里程碑 | 建议完成日期 | 主要产物 |
|---|---|---|
| M0 当前测试与 C3 定版 | 2026-08-02 | 当前 pytest 记录、C3 最终一致产物 |
| M1 唯一状态源与文档同步 | 2026-08-05 | 新版 latest status、conclusion、README 摘要 |
| M2 工程清理与干净基线运行 | 2026-08-09 | clean commit、baseline freeze manifest、冻结重跑 |
| M3 核心产品历史证据第一轮 | 2026-08-23 | 历史规则、时序、生命周期和映射缺口报告 |
| M4 新鲜 OOS 启动 | 以 M2 与最低 P1 Gate 通过后的首个交易日为准 | T0 registry、冻结策略影子账户 |
| M5 第一次前向阶段审核 | T0 后 63 个交易日 | 数据完整性、规则命中、执行与会计审核，不作策略升级 |
| M6 年度前向审核 | T0 后至少 252 个交易日 | 新鲜 OOS Gate 与候选决策 |

日期为执行目标，不得为了满足日期而跳过 Gate。

---

## 7. 当前交付清单

### P0 工程封版

- [x] 当前代码全量 pytest 通过并保存原始结果；
- [x] C3 使用当前 validator 重新 finalize；
- [x] C3 顶层与子目录 artifact Gate 一致；
- [x] `latest_research_status.json` 覆盖 2026-07-30 后全部研究；
- [x] `conclusion.md` 和 README 当前摘要由唯一状态源生成；
- [x] 根目录一次性脚本完成分类；
- [x] `pyproject.toml` 与实际依赖一致；
- [x] 删除或归档 0 字节和无效占位产物；
- [x] 得到 clean Git commit；
- [x] 在 clean commit 上完成冻结重跑；
- [x] 生成 `baseline_freeze_manifest.json`。

### P1 历史真实性

拆分为两组：审计机制建立（已完成）与历史证据覆盖（未完成）。

审计机制建立：

- [x] 历史规则版本表机制建立（`config/otf_rule_versions.csv` 与覆盖报告 `reports/historical_truth/order_rule_version_coverage.json`；当前为 CURRENT_SNAPSHOT_ONLY，实际覆盖率 0/879，诚实标记 NOT_ESTABLISHED）；
- [x] 订单级规则版本追溯字段建立（`order_audit_frame.rule_version_id`；冻结 879 笔订单覆盖率为 0，报告存在但未满足验收）；
- [x] NAV 可用时点或保守滞后模型建立（`nav_availability.py` DOMESTIC_T1_QDII_T2，C1/C2/C3/D1/B2LT 全部接线，16 个测试含未来数据不变性验收）；
- [x] QDII 和跨市场时序专项通过（021778/050025 确认延迟 2 天与模型一致；验收测试覆盖全部信号）；
- [x] 生命周期事件表机制建立（`config/otf_lifecycle_events.csv`；13 只成立事件有 catalog 证据，公告类事件无历史数据源显式标记 PIT_PARTIAL）；
- [x] 实际持仓和候选产品映射证据完整（13/13 HIGH+APPROVED，无同家族同暴露重复）；
- [x] 独立来源 NAV/收益/费用样本核验通过（otf_mapped 24200 行 + otf_defensive 10186 行 100% 一致，10/13 产品覆盖）；
- [x] 历史真实性 Gate 机器可读输出建立（`reports/historical_truth/historical_truth_gate.json`；未通过，研究状态保持 PIT_PARTIAL / RETROSPECTIVE_RESEARCH_UNDER_PARTIAL_PIT）。

历史证据覆盖（未完成，Gate 通过 3/6）：

- [ ] 历史规则版本在提交日有效（覆盖率 0/879，0%）；
- [ ] 生命周期公告类证据完整（13 只核心产品全部 PIT_PARTIAL，无清盘/暂停/恢复/合并等公告历史）；
- [ ] 未用当前快照回填历史（当前规则仍是当前快照，非历史版本）。

### P2 新鲜 OOS

- [x] B2-LT、C1、C3、M20 冻结版本登记（`config/otf_frozen_strategy_versions.csv`，生成脚本 `scripts/migrations/build_frozen_strategy_versions.py`，9 项冻结内容完整）；
- [x] C2 和旧失败研究归档并停止调参（C2、s1_experiment、s1_s2_experiment 写入 `ARCHIVED.json`，决策 `ARCHIVED_PARAMETER_SEARCH_FORBIDDEN`）；
- [x] T0 自动登记机制（`config/otf_t0_registry.json` + `scripts/migrations/build_t0_registry.py`；T0 必须严格晚于冻结完成时间 2026-08-01 且首个前向日有按前向流程采集的数据；2026-07-28 至 07-31 为 POST_FREEZE_BACKFILLED_VALIDATION_WINDOW 不计入新鲜 OOS；当前 PENDING_CALENDAR_EXTENSION，日历扩展且数据采集后重跑脚本自动登记）；
- [x] 前向数据 revision 机制建立（`config/data_revision_registry.json` + `scripts/migrations/record_data_revision.py`，7 个数据文件哈希跟踪，2026-08-02 刷新已记录 1 条，当前 CLEAN）；
- [x] 增量 NAV 刷新保护（`scripts/migrations/refresh_nav_incremental.py`；落盘 refresh_report.json 与 provider_revision_report.csv；发现核心 NAV 修订时阻止追加该基金并暂停前向观察，退出码 2）；
- [x] 前向影子账户与订单审计建立（`config/forward_shadow_schema.json` + `scripts/forward/validate_shadow_records.py`，决策/订单/日度净值勾稽链；`reports/forward_shadow/decisions/` 为空，前向记录为 0）；
- [x] 前向升级 Gate 框架建立（`scripts/forward/forward_upgrade_gate.py`，9 条件全部接线，当前诚实输出 NOT_ELIGIBLE_FOR_PAPER_TRADE）；
- [ ] 合法 T0 登记（等待 2026-08-03 之后的日历扩展与数据前向采集）；
- [ ] 63 日阶段审核完成；
- [ ] 252 日正式前向审核完成。

---

## 8. 停止规则

1. 当前全量测试失败：停止正式实验；
2. artifact validation 不一致：对应运行作废，不允许引用指标；
3. `git_dirty=true`：运行仅作诊断，不得成为冻结基线；
4. 历史规则或信息可用时点未建立：不得称为历史真实验证；
5. 新鲜 OOS 期间修改参数：原版本继续记录，新版本重新开始独立 T0；
6. C2/C3 未达到相对 B2-LT 增量：不得继续通过小幅调参反复搜索；
7. M20 稳定性依赖单一周期或单一阶段：不得直接升级；
8. ML/RL 未证明相对简单基准的独立增量：停止复杂化；
9. 所有策略 Gate 失败：保留 B2-LT 作为研究控制组，不强行选出“主策略”；
10. 只有完整通过工程、历史真实性和新鲜 OOS Gate 的策略，才允许进入下一层纸面交易候选评估。

---

## 9. 立即执行顺序

1. 运行当前全量 pytest；
2. 只修复当前测试和一致性问题；
3. 使用当前 validator 重新 finalize C3；
4. 重新生成最新唯一状态源；
5. 同步 conclusion 和 README 当前摘要；
6. 清理根目录临时脚本与无效产物；
7. 完成依赖声明和最小安装验证；
8. 创建干净基线 commit；
9. 在干净 commit 上冻结重跑 B2-LT、C1、C2、C3、M20 和归因；
10. 生成 baseline freeze manifest；
11. 补齐冻结策略核心产品历史规则、发布时间、生命周期和映射证据；
12. 通过最低历史真实性 Gate 后登记 T0，启动新鲜前向影子观察。

本阶段唯一成功标准不是找到更高收益策略，而是建立一个当前代码可验证、历史事实不被夸大、策略版本不被回看修改、未来样本真正独立的研究基线。

---

## 历史审核记录（仅供追溯，不再作为当前计划）

## 2026-07-29 Phase 5 第五次审核后执行计划：审计闭环与可信 OOS 定版

### 0. 当前审核状态

本轮确认上一轮四个核心结构问题已经得到实质修复：

- B1/B2/B3/S1 均已使用单一连续账户贯穿 2021—2026 OOS；
- S1 已复用正式 `StateRotationSignal`；
- 调仓带与波动率缩放已统一在 sleeve 键空间处理，再映射到基金代码；
- 完整 Gate 已覆盖收益、回撤、成本、换手、滚动两年表现、数据、规则、映射与产物；
- 全量测试实测为 `401 passed`，无 FutureWarning。

当前项目状态调整为：

```text
CONTINUOUS_ACCOUNT_CORE_ACCEPTED
ARTIFACT_GATE_NOT_STRICT_ENOUGH
OOS_METHODOLOGY_LABEL_INACCURATE
HISTORICAL_RULE_GATE_BLOCKED
NO_PAPER_TRADE_CANDIDATE_YET
```

本阶段禁止继续搜索策略参数或开发新策略。必须先完成以下审计闭环，再重新判定 B2、B3 和 S1。

---

### 1. P0-A：修复报告、状态和日期唯一事实源

#### 执行内容

1. 修复 `src/run_b1_b2_b3_walkforward.py` 中已经损坏的中文 UTF-8 字符串。
2. `conclusion.md` 的实际起止日必须从 `daily_account.csv` 计算，不得直接使用配置请求日期。
3. 同时记录两个字段：
   - `requested_oos_period`：用户请求的回测区间；
   - `actual_oos_period`：数据实际覆盖且完成估值的区间。
4. `reports/latest_research_status.json` 只能由本次运行自动生成，禁止后处理脚本手工追加互相矛盾的状态。
5. 测试状态必须来自同一次测试命令；全量测试通过时不得继续保留 `TEST_ENVIRONMENT_NOT_GREEN`。
6. 报告中的策略指标、规则计数、映射计数、输入哈希和 Gate 结果必须全部从运行产物读取，禁止人工复制。

#### 验收标准

- UTF-8 读取报告后不存在 `�`、`鐮`、`杩` 等典型乱码；
- `actual_oos_period.end` 等于日度账户最后一个有效交易日；
- conclusion、manifest、latest status 三处的 run_id、哈希、日期和 Gate 状态完全一致；
- 新增自动化测试覆盖乱码检测、日期一致性和状态一致性。

---

### 2. P0-B：将产物完整性 Gate 从“存在检查”升级为“内容检查”

#### 执行内容

1. 为每类产物定义 schema：必需列、日期列、允许为空、适用策略。
2. 将产物分成两组：
   - 核心必需产物：manifest、配置、哈希、daily account、returns、orders、fees、turnover、target/actual weights、metrics、gate；
   - 策略条件产物：market state、state score、sleeve weights、risk contribution、product selection audit。
3. 静态策略不适用的条件产物必须写出带标准表头的空表，并在 manifest 中标记 `NOT_APPLICABLE`；不能写成无法解析的 5 字节文件。
4. S1 的 market state、state score、sleeve/fund weight 必须非空；B3 的 risk contribution 必须非空。
5. 完整性检查至少验证：文件存在、可解析、schema 正确、关键表非空、日期范围一致、run_id 一致。
6. manifest 增加每个产物的 SHA256、行数、列名、状态和生成时间。

#### 验收标准

- 删除表头、写入空壳文件、修改列名、截断日期范围均会导致 Gate 失败；
- 当前 B2 等静态策略的不适用表能正常读取，并明确标记为 `NOT_APPLICABLE`；
- `artifact_completeness=true` 仅在所有适用产物通过内容检查后成立。

---

### 3. P0-C：核验换手率、成交金额和费用勾稽

#### 执行内容

1. 当前 turnover 使用 `requested_amount`，需要改为实际确认成交口径：
   - 申购成交额为实际确认本金；
   - 赎回成交额为 `shares_confirmed × confirmed_nav`；
   - 未确认、拒绝、取消及 OOS 末尾仍 pending 的订单不得计入已成交换手率。
2. 同时输出三种口径，避免混淆：
   - submitted turnover；
   - confirmed bilateral turnover；
   - settled cash turnover。
3. Gate 只使用 confirmed bilateral turnover，并明确双边换手定义是买入额加卖出额除以期初或当日权益。
4. 对账以下关系：
   - orders 中的费用合计 = fees.csv 费用合计；
   - daily_account 中累计费用 = metrics 中总费用；
   - gross NAV 与 net NAV 的差异可由逐日费用拖累解释；
   - 年度换手率之和可由成交订单逐笔重建。
5. 检查最后一个 OOS 日仍未完成的订单、应收款和冻结资金，并在报告中单独披露。

#### 验收标准

- 人工构造含 pending、拒单、跨期确认和赎回到账延迟的订单案例，confirmed turnover 只统计实际确认订单；
- 费用与订单勾稽误差不超过 `1e-8 × 初始权益`；
- B1/B2/B3/S1 的年度换手 Gate 使用新口径重新计算，不沿用旧结果。

---

### 4. P0-D：补齐连续账户和正式 S1 的行为回归测试

#### 必须新增的端到端测试

1. 跨年测试：12 月末持仓、FIFO lot、pending order、receivable cash 在 1 月首日继承。
2. 连续账户与年度重启敏感性结果必须不同，并分别标记正确的 account mode。
3. 连续账户首尾权益与逐日收益复合结果一致。
4. Walk-forward runner 的 S1 权重与直接调用正式 `StateRotationSignal` 完全一致。
5. sleeve 权重中不得出现基金代码，基金权重中不得出现 `cash_mgt` 或 sleeve 名称。
6. 任一完整 Gate 子项失败时，最终状态不得变成 `PAPER_TRADE_CANDIDATE`。
7. 产物损坏时 artifact Gate 必须失败。
8. 规则时间字段缺失时 historical rule Gate 必须失败。

#### 测试要求

- 不再以源码字符串搜索作为主要验收；
- 使用小型临时数据库完成快速端到端运行；
- 保留现有 401 项测试，并新增上述行为测试；
- 全量测试必须在 `agents` 环境下通过 `-W error::FutureWarning`。

---

### 5. P1-A：纠正研究方法命名

当前 `FOLDS.train` 没有参与拟合、选参或逐折参数冻结，因此当前实验不能宣称为标准 Walk-forward。

#### 本阶段采用的命名

```text
FROZEN_PARAMETER_CONTINUOUS_OOS
```

具体要求：

- B1/B2：固定配置连续 OOS 基准；
- B3：固定算法参数、仅使用信号日前历史窗口的连续 OOS；
- S1：固定状态配置、状态递推连续 OOS；
- 年度 fold 仅称为 `calendar_year_reporting_slices`；
- 年度重启实验继续称为 `ANNUAL_RESTART_SENSITIVITY_ONLY`。

如果后续需要真正 Walk-forward，必须在每个 fold 保存训练数据范围、拟合结果、冻结参数哈希和独立 OOS 输出；该工作不在本轮范围内。

---

### 6. P1-B：历史规则数据分层，不得用当前快照冒充历史事实

#### 执行内容

1. 保留当前规则场景，但统一命名为：

```text
CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO
```

2. 对实际被 B1/B2/B3/S1 使用的基金优先补齐：
   - source_url；
   - effective_from / effective_to；
   - verified_at；
   - channel；
   - 申购确认、赎回确认、到账天数；
   - 分层申购费和持有期赎回费。
3. 无法证明历史有效期的规则继续允许用于保守情景回测，但 `historical_rules_gate=false`。
4. 只有来源可追溯且覆盖对应成交日期的规则，才允许计入历史真实执行 Gate。
5. 报告必须同时区分：
   - 策略统计 Gate；
   - 当前快照保守执行 Gate；
   - 历史规则真实性 Gate。

#### 验收标准

- 缺少有效期的产品不会被误标为历史验证通过；
- 规则覆盖率按实际成交订单而不是全基金库计算；
- 每一笔订单都能追溯到命中的规则版本和来源。

---

### 7. P2：重新运行与候选决策

完成 P0-A 至 P1-B 后，使用冻结配置重新运行 B1/B2/B3/S1，不得调整策略参数。

#### 运行输出

1. 四条连续账户 OOS 曲线；
2. 年度重启敏感性对照；
3. 修正后的 confirmed turnover 与费用勾稽；
4. 每个策略完整 Gate；
5. B2 对 B3、S1 的相对比较；
6. 唯一 latest research status；
7. 自动生成的 UTF-8 conclusion。

#### 决策规则

- B1 继续只作为风险较高的静态参考；
- B2 只有在统计 Gate、confirmed turnover Gate、数据 Gate、映射 Gate、产物 Gate 全部通过后，才可成为“保守规则情景纸面交易候选”；
- 在 historical rule Gate 通过前，不得称为“历史真实执行验证通过”；
- B3 或 S1 只有相对 B2 满足 `CAGR +0.75pct` 或 `Sharpe +0.15`，且绝对 Gate 全部通过，才允许升级；
- S1 再次低于 B2 时冻结该路线，不再继续参数搜索；
- 任一关键 Gate 失败，状态保持 `OOS_GATE_FAILED`。

---

### 8. 最终交付清单

- [ ] 修复后的报告和状态生成器；
- [ ] schema 驱动的 artifact validator；
- [ ] confirmed turnover 与费用对账模块；
- [ ] 连续账户/S1/Gate 端到端测试；
- [ ] 方法学命名修订；
- [ ] 实际使用基金的规则覆盖报告；
- [ ] 冻结配置后的全量重跑产物；
- [ ] 新版 `conclusion.md`；
- [ ] 全量测试结果及运行耗时；
- [ ] Git diff 审核清单，禁止混入策略参数优化。

本轮完成定义：报告可读、事实一致、产物可验证、成交与费用可勾稽、连续账户有行为测试保护，并在不调参的条件下重新得到可信 OOS 结论。

## 2026-07-29 Phase 5 第四次审核：年度重启敏感性实验结论与下一轮修复

### 0. 第四次审核结论

当前状态更新为：

```text
ANNUAL_RESTART_SENSITIVITY_ONLY
WALK_FORWARD_NOT_CONTINUOUS_ACCOUNT
S1_OOS_VERSION_MISMATCH
GATE_INCOMPLETE_IMPLEMENTATION
RESEARCH_CONCLUSION_INVALID_V4
```

上一轮（第三次审核后）取得以下有效进展：

- 测试绿线恢复并扩展至401项；
- P0-D规则时间/渠道字段结构已建立；
- P0-E映射表fund_family_id/source_url已填充；
- P1-A vol_scale_weights/prev_weights/reset已接入run_s1_experiment.py；
- P1-B费用金额列修复，Walk-forward费用从0变为非零；
- P1-C B3风险平价asset_cap迭代投影修复；
- P2连续Walk-forward引擎实现拼接OOS日收益，4策略×6fold全部运行。

但本轮审核确认以下结构性问题使OOS结果不能作为策略决策依据：

### P0级问题

1. **账户不连续**：每个fold新建OTFBacktestEngine并从100万元现金重新开始，没有继承上一fold的持仓、FIFO批次、应收款和待处理订单。拼接日收益只能估算“年度重启策略收益”，不能模拟跨年持有期、跨年赎回费档位和真实费用金额。48,000—69,000元总费用是六个独立账户费用相加，不是连续账户实际费用。

2. **OOS中的S1不是接线版S1**：Walk-forward脚本重新定义简化版FoldStateRotationSignal，只使用市场状态和固定产品映射，没有调用vol_scale_weights()、prev_weights或MarketStateEngine.reset()。OOS的3.18% CAGR不是报告所描述完整S1。

3. **Gate实现不完整**：代码中绝对Gate只检查Sharpe>=0.50、MDD>-15%、NetCAGR>0。没有检查成本拖累<=1.25%、年度双边换手率、两年滚动正收益比例、最差两年CAGR、数据/规则/映射Gate、产物完整性。gate_result.json中B2和S1的gate_passed不是完整Gate结果。

4. **唯一事实源未形成**：latest_research_status.json中input_hashes为空、rule_counts_by_status和mapping_counts_by_status为空但状态写为完成。conclusion.md声称56条规则（3+31+10+21=65），实际CSV为55条（3+31+20+1）。文档数字不是从运行输入动态生成。

### P1级问题

5. **规则时间字段只有结构没有历史数据**：55条effective_from全部为空、verified_at全部为空。渠道页面只能证明当前费率，不能证明2018-2026历史费率。应命名为“当前渠道规则回填场景”或“保守费率场景”。

6. **实验产物不完整**：Walk-forward目录只有拼接日表、fold汇总、summary和gate结果。缺少manifest、输入哈希、配置快照、每fold订单、拒单、目标/实际权重、状态评分、产品批次、换手率、风险贡献、费用勾稽、参数冻结记录。

7. **调仓带键类型不一致**：map_to_fixed_products()把上一期基金代码权重传给期待袖套名称权重的select_weights()。上一期键为160706，新权重键为CSI300，调仓带无法匹配。应分别保存prev_sleeve_weights和prev_fund_weights。

8. **波动率缩放层级错误**：完整S1把基金代码权重传给资产类别波动率缩放器并指定现金键cash_mgt。如果触发缩放可能产生名为cash_mgt的非法基金代码。波动率缩放应在袖套/资产类别层完成再映射到基金。

### 本轮可保留结论

- 在“年度独立启动、日收益拼接”口径下，B2和S1风险指标相近；
- S1暂未显示相对B2明显收益或Sharpe增值；
- B2值得保留为静态研究基准。

### 本轮不能确认结论

- S1正式OOS失败；
- 市场状态路线应永久停止；
- B2已通过完整Gate；
- B2可以进入正式纸面交易。

### 下一轮修复顺序

1. 将当前Walk-forward标记为ANNUAL_RESTART_SENSITIVITY_ONLY，所有结论中明确命名口径。
2. Walk-forward直接复用正式StateRotationSignal，删除脚本内FoldStateRotationSignal简化版本。
3. 修复AssetBudgetEngine调仓带：分别保存prev_sleeve_weights和prev_fund_weights，键空间分离。
4. vol_scale_weights()在袖套/资产类别层完成缩放，再调用map_to_fixed_products()映射到基金。
5. 实现跨fold连续账户：继承持仓批次、现金、应收款和待处理订单；同时保留年度重启版作为敏感性对照。
6. Walk-forward每个fold输出完整产物：manifest、输入哈希、配置快照、每日订单、拒单、目标/实际权重、状态评分、产品批次、换手率、风险贡献、费用勾稽、参数冻结记录。
7. 从拼接日收益重新计算两年滚动正收益比例、最差两年CAGR、年度双边换手率和年化成本拖累，执行完整Gate。
8. 填充规则有效期：无法核实历史规则时明确使用保守费用场景并分开命名。
9. 修复latest_research_status.json：输入哈希非空、规则计数从CSV动态读取、映射计数从映射表动态统计。
10. conclusion.md所有数字从latest_research_status.json和对应运行产物自动生成，禁止手动填写。
11. 完成后重新运行B1/B2/B3/S1连续账户Walk-forward，再决定是否停止S1或推进B2纸面交易。

本轮首要目标：得到一条费用正确、账户连续、输入冻结、产物完整、Gate完整的OOS净值曲线。

## 2026-07-29 Phase 5 第三次审核与连续 OOS 重建计划

### 0. 第三次审核结论

当前状态更新为：

```text
IMPLEMENTATION_INCONSISTENT
WALK_FORWARD_INVALID
RESEARCH_CONCLUSION_INVALID_V3
```

本轮确有以下有效进展：

- S1 与 S2 已拆分为固定产品映射和动态产品选择两个入口；
- ProductRuleBook 已增加基础规则等级过滤；
- ProductSelector 已增加选择日前真实 NAV 观察数、最近 NAV 和最大缺口检查；
- 已建立资产暴露映射表、月末调度器、实验产物模块和滚动风险平价模块的初版；
- B2 已在最新实验名称中更正为 `B2_Static_EW_4Asset`。

但最新 `conclusion.md` 仍不能用于策略决策，原因如下：

1. 最新专项回归实际为 `3 failed, 109 passed`。三个失败均来自 `tests/test_state_allocation.py` 仍调用已经删除的 `get_fund_weights()`；当前不能声明测试全部通过；
2. `experiment_artifacts.py` 虽已创建，但没有被 S1/S2/Walk-forward 运行器调用。最新报告目录仍只有汇总 CSV 和简短 manifest，P0-5 未完成；
3. `vol_scale_weights()`、`prev_weights` 和状态引擎 `reset()` 虽已实现，但没有在正式实验入口中调用，P1-2 不能判定完成；
4. ProductRuleBook 只检查规则状态，没有实现计划要求的 `effective_from/effective_to/channel/verified_at`。当前规则 CSV 也没有这些字段；
5. 当前规则统计已经变为 3 条官方、31 条渠道、21 条假设，与 `conclusion.md` 中“3/17/35”不一致，说明报告不是从运行输入动态生成；
6. 暴露映射表的 `fund_family_id` 全部为空，`mapping_source` 统一写为 `manual_review_v1`，缺少可核验来源；所有记录直接标记 HIGH/APPROVED，尚不足以证明完成独立审核；
7. 最新 Walk-forward 脚本标题包含 B3，但实际上只运行 B1/B2；没有运行 B3 和 S1，却在结论中判定“所有策略均未通过”；
8. Walk-forward 每个 fold 的 `total_fees` 均为0，因为脚本读取不存在的 `fee_amount` 列；费用已影响净值，但费用金额审计失败；
9. 所谓“Concatenated OOS”实际上只是对各 fold 的 CAGR 和 Sharpe 做简单算术平均，没有保存或拼接日收益；2026 半年 fold 与完整年度等权平均，统计口径错误；
10. 每个 OOS fold 都从100万元现金重新启动，重复产生建仓过程，且没有形成跨年度连续账户；
11. Walk-forward 的训练窗口对 B1/B2 没有任何作用，也没有冻结或训练 B3/S1 参数；当前只是年度分段回测，不是完整 Walk-forward；
12. Gate 使用“年度正收益 fold 比例”冒充“两年滚动正收益比例”，也没有计算最差两年 CAGR、年度换手和成本拖累；
13. `conclusion.md` 同时写“P2-P5未开始”和“P2/P3/P4已完成”，文档内部相互矛盾；
14. 当前 S1/S2 全样本结果仍未保存订单拒绝和实际仓位，无法确认规则等级阻断后目标预算是否真实成交。

因此：

- 当前 B1/B2/B3/S1/S2 的全样本指标仅作系统诊断；
- 当前 Walk-forward 全部作废；
- “所有策略均未通过正式验证”尚未被有效实验支持；
- 下一轮禁止优化收益参数、修改 Gate 或选择候选主策略。

### 1. P0-A：恢复测试绿线并清理失效接口

#### 任务

1. 更新 `tests/test_state_allocation.py`，分别测试：
   - `map_to_fixed_products()`；
   - `map_to_dynamic_products()`；
2. 删除或迁移所有仍调用 `get_fund_weights()` 的生产代码、测试和文档；
3. 增加 S1 不调用 ProductSelector、S2 不回退固定产品的端到端测试；
4. 增加正式规则状态阻断测试；
5. 增加运行器接线测试，确认目标波动率、prev_weights、reset和产物导出函数确实被调用；
6. 新增 `tests/test_schedule.py`、`tests/test_risk_parity.py`、`tests/test_experiment_artifacts.py` 和 `tests/test_walkforward_oos.py`；
7. 完整测试必须使用固定 `agents` 环境和 `-W error::FutureWarning`；
8. 报告测试结果必须从JUnit/pytest输出自动生成，禁止人工填写。

#### 验收

- 专项测试0失败；
- 完整回归0失败；
- 旧接口引用数为0；
- 新增模块均有行为测试而非仅存在性测试。

### 2. P0-B：建立唯一事实源与运行状态

新增机器可读 `reports/latest_research_status.json`，由实验编排器在成功完成后原子写入，包含：

```text
run_id
status
completed_phases
failed_phases
test_result
input_hashes
rule_counts_by_status
mapping_counts_by_status
strategies_executed
oos_period
gate_result
blocking_issues
```

`conclusion.md`、`planning_status_check.md` 和所有摘要必须从该文件及对应运行产物生成，不允许单独维护数字。

状态枚举统一为：

- `BUILD_FAILED`；
- `TEST_FAILED`；
- `DATA_GATE_FAILED`；
- `EXPERIMENT_INVALID`；
- `OOS_NOT_RUN`；
- `OOS_GATE_FAILED`；
- `PAPER_TRADE_CANDIDATE`。

验收：规则数量、策略列表、测试数和Gate在所有文档中完全一致；输入改变后旧结论自动标记 `STALE_INPUTS`。

### 3. P0-C：真正接入可复现实验产物

将 `ExperimentArtifacts` 接入所有正式运行器，而不是仅保留工具模块。

每个策略独立输出：

```text
manifest.json
input_hashes.json
config_snapshot/
environment.txt
daily_account.csv
daily_returns.csv
market_states.csv
state_scores.csv
asset_budgets.csv
sleeve_weights.csv
fund_target_weights.csv
fund_actual_weights.csv
product_selection_audit.csv
orders.csv
order_rejections.csv
position_lots.csv
fees.csv
turnover.csv
risk_contributions.csv
metrics.json
gate_result.json
run.log
```

#### 强制勾稽

- `sum(orders.fee_paid) == sum(fees.fee_amount)`；
- 每日费用与Gross/Net差可解释；
- 目标权重、实际权重和现金权重每日加总为100%；
- 被规则阻断的预算能追踪到现金、防守回落或未投资状态；
- 汇总指标可仅使用落盘日级文件重算；
- manifest记录Git dirty状态，脏工作树实验不得标记为最终冻结运行。

验收：最新运行目录不再只有summary和manifest；删除任一核心产物会使Gate失败。

### 4. P0-D：补齐产品规则的时间和渠道语义

扩展 `FundTradingRule` 和CSV字段：

```text
effective_from
effective_to
channel
source_type
source_url
verified_at
verification_notes
```

#### 规则

1. `is_rule_allowed(fund_code, submit_date, channel)` 必须检查状态、有效期和渠道；
2. 当前销售页面只能证明当前规则，不能自动回填2018年以来全部历史；
3. 无历史有效期的渠道规则在历史回测中标记 `CURRENT_SNAPSHOT_ONLY`；
4. 正式历史回测允许使用的规则必须明确为：
   - 历史官方规则；或
   - 保守费用上界场景；
5. 保守费用场景与“真实标准费率”分开命名；
6. 0申购费必须区分C类销售服务费、渠道折扣和真实免申购费；
7. 赎回费、最低持有期和限购事件必须与规则有效期一致。

验收：正式订单100%匹配提交日和渠道；无法证明历史规则时报告明确为保守假设，不再声称真实产品费率。

### 5. P0-E：重新审核资产暴露映射

当前 `manual_review_v1/HIGH/APPROVED` 不能作为充分证据。执行：

1. 填充稳定 `fund_family_id`；
2. 每条映射增加 `source_url`、`reviewer`、`reviewed_at` 和证据摘要；
3. `effective_from` 不得统一写2016年，必须与基金/份额实际成立和跟踪关系一致；
4. ETF联接、普通指数增强、主动债券、转债和货币基金分别设置映射规则；
5. B1/B2/B3/S1固定产品优先逐只核验；
6. 动态S2候选后续再扩展，不阻塞静态基准和S1；
7. 所有空 `fund_family_id`、无来源HIGH映射和日期矛盾均阻断正式实验。

验收：正式成交产品映射证据覆盖100%；同家族份额不会重复持有。

### 6. P1-A：将风险控制真正接入运行器

#### 状态策略运行状态

StateRotationSignal必须维护并在每次新实验显式重置：

- `prev_state`；
- `prev_sleeve_weights`；
- `prev_fund_weights`；
- 波动率估计历史；
- 状态确认计数。

#### 接线顺序

```text
原始状态预算
→ 袖套选择
→ prev_weights调仓带
→ 组合波动率估计
→ 9%目标波动率向下缩放
→ 现金残余
→ 固定/动态产品映射
→ 最小订单过滤
```

要求：

1. `vol_scale_weights()`必须由正式S1/S2入口调用；
2. 组合波动率使用信号日前历史，不得用未来实际组合收益；
3. 目标波动率缩放前后权重都落盘；
4. `prev_weights`必须传入 `select_weights()`；
5. fast switch只允许降低预定义风险等级；
6. 每次策略运行前调用 `MarketStateEngine.reset()`；
7. 输出调仓带、波动率缩放、状态切换和产品替换各自的换手贡献。

验收：关闭/开启每一组件产生预期且可解释的消融差异；仅定义函数但无调用视为失败。

### 7. P1-B：修复费用金额和账户审计

OTFBacktestEngine每日输出显式：

- `subscription_fee_amount`；
- `redemption_fee_amount`；
- `total_fee_amount`；
- `transaction_cost_return`。

所有运行器直接读取金额列，不通过模糊列名搜索，也不重复反推。

Walk-forward和全样本统一使用同一 `compute_metrics()`。禁止各脚本复制不同指标函数。

验收：每个fold费用不再错误为0；订单费用、每日费用和汇总费用精确一致。

### 8. P1-C：修复B3风险平价实现与审计

1. 将B3加入正式连续OOS运行；
2. 训练窗口真实用于协方差估计；
3. 修复“先clip再归一化可能重新超过asset_cap”的约束问题；
4. 使用受约束优化或迭代投影确保最终权重满足上限；
5. 保存每个信号日的协方差窗口、优化状态、回退原因和风险贡献；
6. 不得吞掉所有异常后静默回退，异常必须进入审计；
7. 增加奇异协方差、负相关、缺失数据和资产上限测试；
8. 明确货币基金收益模式，不将缺失/常数NAV错误用于协方差。

验收：最终权重非负、和为100%、单资产不超上限；风险贡献误差在预注册容忍度内。

### 9. P2：重建连续Walk-forward引擎

#### 原则

Walk-forward必须生成一条连续OOS账户曲线，而不是平均fold指标。

#### 正确流程

1. 每个fold训练结束时冻结下一OOS期需要的参数；
2. OOS日收益按日期顺序拼接；
3. 参数可在年度边界更新，但账户资金、持仓批次、应收款和待确认订单连续继承；
4. 若技术上暂不能继承完整账户状态，则先实现“日收益拼接版”和“每fold独立启动敏感性版”，两者分开报告；
5. 2026不完整年度按实际天数进入连续曲线，不与完整年度等权平均；
6. 静态B1/B2不需要训练，但仍在相同连续OOS区间运行；
7. B3使用训练窗/滚动历史冻结下一期算法参数；
8. S1使用训练窗冻结状态阈值、预算和固定产品；
9. S2仍被Gate阻断，不进入本阶段。

#### 必须运行的策略

- B1_Static_60_20_20；
- B2_Static_EW_4Asset；
- B3_Rolling_Risk_Parity；
- S1_State_Rotation_Fixed。

#### 必须输出

- 每fold训练配置；
- 每fold OOS日收益；
- 拼接OOS日收益和净值；
- 连续账户状态；
- 每fold及拼接费用；
- OOS目标/实际权重；
- 参数冻结哈希。

验收：标题、实际执行策略和结果表策略完全一致；不得再出现“脚本名有B3但结果无B3”。

### 10. P3：按正确统计量执行Gate

Gate只使用拼接OOS日收益计算：

- 拼接OOS Net/Gross CAGR；
- 拼接OOS Sharpe；
- 拼接OOS MDD；
- 年化成本拖累；
- 年度双边换手；
- 真正的滚动两年正收益比例；
- 最差滚动两年CAGR。

禁止：

- 平均各fold CAGR作为总体CAGR；
- 平均各fold Sharpe作为总体Sharpe；
- 年度正收益fold比例替代两年滚动比例；
- 只跑B1/B2后宣称所有策略失败；
- 因Gate严格而在看过结果后修改门槛。

#### 比较Gate

S1必须相对最佳合格静态基准满足至少一项：

- 拼接OOS净CAGR高0.75个百分点；或
- 拼接OOS净Sharpe高0.15。

并满足绝对风险、成本、换手和滚动窗口门槛。B1/B2/B3分别先通过数据、执行和绝对风险Gate，才能作为“最佳合理静态基准”。

### 11. P4：结论生成与停止规则

`conclusion.md` 必须由同一run_id自动生成，结构固定：

1. 输入哈希和代码版本；
2. 测试状态；
3. 数据/规则/映射Gate；
4. 实际执行策略列表；
5. 拼接OOS结果；
6. 成本和拒单审计；
7. Gate逐项结果；
8. 有效结论；
9. 阻塞项。

停止规则：

- 测试失败：不运行实验；
- 规则或映射Gate失败：不运行正式历史回测；
- 产物不完整：实验标记无效；
- S1 OOS失败：停止状态策略和S2/S3；
- 所有策略失败：不强行选择“回撤最好者”为主策略，转入数据改进或纸面基准观察；
- 只有通过全部Gate的策略才能进入前向纸面交易。

### 12. 下一轮严格执行顺序

1. 将当前Walk-forward和结论标记 `INVALID_WALK_FORWARD_IMPLEMENTATION`；
2. 修复3个现有测试失败；
3. 增加schedule、risk parity、artifacts和OOS专项测试；
4. 建立唯一事实源 `latest_research_status.json`；
5. 将实验产物模块接入正式运行器；
6. 扩展规则有效期和渠道字段；
7. 审核固定产品暴露映射证据和family_id；
8. 将prev_weights、目标波动率和reset接入S1；
9. 统一费用金额输出和指标计算；
10. 修复B3资产上限和异常审计；
11. 完整专项测试；
12. 完整回归测试；
13. 数据、规则、映射和产物Gate预检；
14. 重建连续Walk-forward；
15. 依次运行B1、B2、B3、S1；
16. 使用拼接OOS日收益执行Gate；
17. 自动生成新的 `conclusion.md`；
18. 若无策略通过，停止策略优化并提交失败审计；若有策略通过，再制定纸面交易计划。

本阶段的唯一成功标准是得到一条费用正确、账户连续、输入冻结、产物完整的OOS净值曲线，而不是提高任何策略的回测收益。

## 2026-07-29 Phase 5 二次审核结论与下一轮任务

### 0. 二次审核判定

本轮执行状态判定为：

```text
PARTIALLY_IMPLEMENTED
RESEARCH_CONCLUSION_INVALID_V2
```

已经确认完成或取得进展的部分：

- 正式实验入口已使用基本的“月末信号日 → 下一数据库交易日提交”映射；
- 回测输出已能区分 Gross/Net CAGR，并报告非零费用；
- ProductSelector 已要求传入日期，并增加成立日、终止日和最早 NAV 检查；
- 状态引擎增加了连续确认和最短持续期的基础实现；
- 市场状态、资产预算、产品选择和场外执行专项测试共 106 项通过。

但以下问题使 `conclusion.md` 的 `STATE_ROTATION_KEEP_FIXED_PRODUCTS` 结论仍然无效：

1. S1“固定产品”与 S2“动态产品”没有真正隔离。`ExposureSelector.get_fund_weights()` 在两种模式下都优先调用 ProductSelector；S1 只在动态选择为空时才回退固定产品。因此 S1/S2 几乎相同是实现结构的必然结果；
2. 55 条规则中仅 3 条 `OFFICIAL_VERIFIED`、17 条 `DISTRIBUTOR_VERIFIED`，其余 35 条仍为假设。严格模式只检查规则存在，不检查规则等级，假设规则仍能成交；
3. ProductSelector 没有按选择日统计真实 NAV 观察数，只用“选择日减最早 NAV 日”的日历天数代替，长期缺失数据也可能通过；
4. 资产映射仍主要依赖名称正则和基金类型回退，没有 `mapping_source`、`mapping_confidence` 和经审核的精确跟踪标的；
5. P2 可复现协议没有完成。最新实验目录只有汇总 CSV 和简短 manifest，缺少每日净值、状态、权重、订单、拒单、候选快照、输入哈希和配置快照；
6. B2 仍被错误命名为风险平价，实际只是四资产各 25% 静态等权；B3 真正滚动风险平价未实现；
7. Walk-forward 被错误暂停。S2 无增值只意味着不进入动态产品/S3，不能免除 S1 和静态基准的 OOS 验证；
8. Gate 被错误放宽为“允许落后最佳基准 0.75 个百分点”。原计划要求相对最佳合理静态基准产生正增值：净 CAGR 高至少 0.75 个百分点，或净 Sharpe 高至少 0.15；
9. manifest 中记录的 `USER_OVERRIDE_TO_S2` 没有对应的用户明确授权，不得作为跨越 Gate 的依据；
10. 状态配置中的 9% 目标波动率尚未应用，`prev_weights` 未接入资产预算调仓带，快速切换也没有限制为只降风险。

因此必须撤销以下结论，直到本计划完成：

- “S1 已完成正式验证”；
- “S2 与 S1 无差异，因此动态选择无价值”；
- “保留状态配置 + 固定产品作为候选主线”；
- “P0-A/B/C/D 和 P2/P3 已全部完成”。

### 1. P0-1：严格隔离 S1 与 S2

#### 正确定义

```text
S1 = MarketState + AssetBudget + ExposureSelector + FixedVerifiedProductMap
S2 = MarketState + AssetBudget + ExposureSelector + PointInTimeProductSelector
```

S1 不得创建、调用或依赖 ProductSelector。S2 不得在候选为空时静默回退到 S1 固定产品；候选为空时应将相应预算转入预先定义的合格现金/短债产品，并记录 `NO_ELIGIBLE_PRODUCT`。

#### 实现任务

1. 将 `ExposureSelector.get_fund_weights()` 拆成两个明确接口：
   - `map_to_fixed_products()`；
   - `map_to_dynamic_products()`；
2. S1 直接读取版本化 `fixed_product_map.csv/yaml`；
3. S2 必须显式注入 ProductSelector；
4. 固定产品缺失、无 NAV 或规则不合格时拒绝启动 S1，禁止偷偷换产品；
5. 动态产品无候选时输出预算缺口和防守回落路径；
6. 保存 S1/S2 每个信号日的袖套权重和基金权重差异表。

#### 必须新增的测试

- 给 ProductSelector 注入一个调用即抛错的假对象，S1 必须仍可运行；
- S1 所有日期的产品代码必须严格等于固定映射允许集合；
- S2 候选改变时目标产品随时间变化；
- S2 无候选时不得回退固定产品；
- 在存在两个合格动态候选的日期，S1 与 S2 目标权重必须产生可解释差异。

#### 验收

- S1 对 ProductSelector 的调用次数为 0；
- S2 对固定产品回退的调用次数为 0；
- S1/S2 实验隔离报告通过后，才允许重新比较绩效。

### 2. P0-2：按规则等级阻断交易

#### 正式研究允许等级

| 场景 | 允许规则等级 |
|---|---|
| 正式标准费率回测 | OFFICIAL_VERIFIED、DISTRIBUTOR_VERIFIED |
| 保守压力测试 | 上述等级 + CONSERVATIVE_ASSUMPTION |
| 数据探索 | 可包含 PRODUCT_TYPE_ASSUMPTION，但禁止输出可实盘结论 |

#### 实现任务

1. ProductRuleBook 增加 `allowed_rule_statuses`；
2. `strict_product_rules=True` 必须同时检查规则存在、规则等级、有效日期和渠道；
3. ProductSelector 只返回当前实验允许等级的产品；
4. 规则簿不得把所有CSV规则统一标为 `verified`；
5. `PRODUCT_TYPE_ASSUMPTION`、空来源、缺失费率和0费率假设不得进入正式回测；
6. 规则增加并强制使用：`effective_from`、`effective_to`、`channel`、`source_url`、`verified_at`；
7. 报告分开统计规则总覆盖、外部核验覆盖和正式可交易覆盖；
8. 为每笔拒单记录 `RULE_STATUS_NOT_ALLOWED` 或 `RULE_NOT_EFFECTIVE_ON_DATE`。

#### 数据补齐目标

优先为正式配置所需的核心袖套补规则，不追求55条表面数量：

- 沪深300、黄金、国债、货币/超短债至少各2只外部核验产品；
- 若保留中证500、创业板、红利、恒生、标普500、纳指，每个袖套至少2只；
- 同业存单、QDII和带最短持有期产品必须有专项规则；
- 无法达到核验覆盖的袖套从正式实验移除，不以假设规则填充。

#### 验收

- 正式实验中假设规则成交数为0；
- 所有成交产品的规则在提交日有效；
- 规则覆盖报告不得再把55/55等同于真实覆盖100%。

### 3. P0-3：完成真正的时间点产品资格

#### 实现任务

1. 在选择日执行数据库聚合：`COUNT(nav_date <= t)`，不得用日历天数替代观察数；
2. 检查选择日前最近 NAV 的新鲜度，超过容忍天数则排除；
3. 检查历史窗口中的最大缺口和连续缺口；
4. 成立日缺失时使用首个可信 NAV 日作为保守代理并标记；
5. 终止/清盘信息缺失继续标记 `PIT_PARTIAL`；
6. 历史申购状态不可得时不得用当前快照回填过去；
7. 费用、限购、规模和渠道状态必须按有效日期读取；
8. 每个信号日保存完整 `SelectionAudit`，而不是只在内存返回；
9. 产品评分分解为费率、NAV质量、跟踪质量、规则质量和切换惩罚，禁止仅按申购费排序；
10. 同一基金家族/指数/份额去重使用稳定 `fund_family_id` 和 `underlying_id`，不能只靠删名称后缀。

#### 测试

- 只有1条早期NAV、随后长期缺失的基金不得通过120日历史门槛；
- 修改选择日之后的NAV、费率和状态不影响历史选择；
- 最近NAV过旧、历史缺口超限、生命周期不完整分别产生明确排除原因；
- A/C/I/F、现汇/现钞等份额不会重复占用资产预算。

### 4. P0-4：重建正式资产暴露映射

建立版本化 `config/otf_exposure_mapping.csv`，至少包含：

```text
fund_code
fund_family_id
share_class
underlying_id
underlying_name
benchmark
asset_class
asset_sleeve
mapping_source
mapping_confidence
effective_from
effective_to
review_status
```

正式实验只允许 `mapping_confidence=HIGH` 且 `review_status=APPROVED` 的产品。名称正则和基金类型回退只生成待审核候选，不能直接成交。

必须重新检查：长债与国债期限、转债与信用债、灵活配置与信用债、港股与恒生、黄金股票与黄金ETF联接、ETF本体与ETF联接份额。

验收：正式成交产品100%有精确暴露映射；不存在 TYPE_TO_SLEEVE 自动映射产品进入正式订单。

### 5. P0-5：补齐可复现实验产物

每个策略、每次运行必须输出独立目录，至少包含：

```text
manifest.json
config_snapshot/
input_hashes.json
environment.txt
daily_nav.csv
daily_returns.csv
market_states.csv
state_scores.csv
asset_budgets.csv
sleeve_weights.csv
fund_weights.csv
product_selection_audit.csv
orders.csv
order_rejections.csv
position_lots.csv
fees.csv
turnover.csv
metrics.json
gate_result.json
run.log
```

manifest 必须记录 Git commit、工作树是否脏、数据库SHA256、规则表SHA256、映射表SHA256、配置SHA256、Python及依赖版本、运行起止时间和随机种子。

验收：任何汇总指标都能从保存的日级与订单级产物重算；审计脚本中硬编码绩效、订单数和Gate结果的数量为0。

### 6. P1-1：完成执行日历与跨境时序

当前“下一数据库日期”只完成国内基础近似，不能宣称完整P0-B。下一步：

1. 使用中国基金销售工作日生成月末信号和提交日；
2. 删除回测起始日额外生成的非月末信号；
3. 维护国内、港股、美国市场日历；
4. QDII按海外收盘、净值归属日、披露日和确认规则建模；
5. 产品无当日精确NAV时不能静默跳过，应延期或拒单；
6. `simulate_monthly()` 统一使用正式月末调度器，删除 `freq="21D"`；
7. 输出计划信号、实际提交、确认、到账及延期原因。

验收：每个自然月最多一个常规信号；不存在起始日伪月末信号；国内和QDII时序测试全部通过。

### 7. P1-2：完成状态风险控制接线

1. `fast_switch_threshold` 只允许从较高风险状态切换到较低风险状态；
2. `prev_weights` 传入资产预算调仓带；
3. 9%目标波动率应用于组合风险缩放，只降风险、不加杠杆；
4. 区分目标权重变化和实际持仓漂移；
5. 输出状态切换、调仓带触发、风险缩放和产品替换各自的换手贡献；
6. 最小交易比例与资产调仓带只保留单一职责，防止重复过滤；
7. 状态引擎提供显式 `reset()`，每次独立实验前清空内部历史。

验收：配置中所有参数均有行为测试；关闭任一组件时能通过消融实验观察到预期差异。

### 8. P1-3：修正基准定义

立即更名：

```text
B2_Risk_Parity → B2_Static_Equal_Weight_4Asset
```

新增真正的 B3：

- 使用截至信号日的滚动波动率/协方差；
- 采用风险预算或等风险贡献；
- 权重非负、总和不超过100%；
- 不加杠杆；
- 设置资产上限和协方差稳定化；
- 训练窗口不足时回退到预注册静态权重；
- 保存每期边际风险贡献。

验收：B3各资产风险贡献接近目标且不读取未来数据；B2不得再出现在“风险平价”名称下。

### 9. P2：重新执行实验，不允许跨Gate

#### 实验顺序

1. B1、B2、B3完成会计、规则、映射和时序验收；
2. 运行真正固定产品S1；
3. 对S1执行冻结Walk-forward；
4. 只有S1通过OOS Gate，才允许运行真正动态产品S2；
5. S2只有在相对S1产生费用后净增值时才进入S3；
6. 未经用户明确消息授权，程序和报告禁止写入 `USER_OVERRIDE`。

#### 不得放宽的Gate

S1相对最佳合理静态基准必须满足至少一项：

- OOS净CAGR高至少0.75个百分点；或
- OOS净Sharpe高至少0.15。

并同时满足：净Sharpe不低于0.50、最大回撤不劣于-15%、成本拖累不高于1.25%、年度双边换手不高于100%、两年滚动正收益比例不低于75%、最差两年CAGR不低于-3%。

若S1不通过：停止状态参数优化和S2开发，将B1/B2/B3中通过OOS者作为候选；不得把“风险控制看起来不错”替代正式Gate。

### 10. P3：强制 Walk-forward

Walk-forward 是最终策略决策的必要条件，不因S2暂停而取消：

```text
2018—2020 → 2021 OOS
2018—2021 → 2022 OOS
2018—2022 → 2023 OOS
2018—2023 → 2024 OOS
2018—2024 → 2025 OOS
2018—2025 → 2026 OOS（截至可用日期）
```

状态阈值、预算、固定产品、动态评分和风险参数必须在每个训练窗结束时冻结。每个OOS年度单独保存产物，最终只汇总从未参与当期调参的OOS收益。

验收：报告明确区分训练、验证、OOS和全样本描述性结果；`conclusion.md` 的策略保留/停止决策只能引用拼接OOS结果。

### 11. 下一轮严格执行顺序

1. 将当前 S1/S2 报告标记为 `INVALID_EXPERIMENT_SEPARATION`，保留但不删除；
2. 修复S1/S2隔离并增加防回归测试；
3. 为ProductRuleBook增加规则等级和有效期Gate；
4. 生成外部核验规则缺口表，补齐核心正式产品；
5. 修复时间点NAV观察数、新鲜度和缺口检查；
6. 建立并审核精确资产暴露映射表；
7. 完成日级、订单级、选择级产物与哈希manifest；
8. 修复月末/跨境执行日历；
9. 接入目标波动率、上一期权重和只降风险快速切换；
10. 将B2正确更名并实现B3；
11. 执行专项测试、完整回归和端到端手工会计算例；
12. 重跑B1/B2/B3；
13. 重跑真正固定产品S1；
14. 强制执行S1 Walk-forward并按原Gate判定；
15. 只有Gate通过后才创建S2新实验；
16. 根据拼接OOS结果重新编写 `conclusion.md` 和下一阶段计划。

本轮的首要目标不是提高CAGR，而是使“固定产品、动态产品、规则质量、时间点资格和OOS结果”五件事真正可区分、可追溯、可复算。

## 2026-07-28 Phase 5 审核后纠偏与重跑计划

### 0. 当前研究状态

`conclusion.md` 中的 Phase 5 结果暂时冻结，并统一标记为：

```text
INVALID_FOR_STRATEGY_DECISION
```

当前结果只能说明“现有 S1 实现没有跑赢静态四资产等权组合”，不能用于证明市场状态配置路线无效，也不能将 B2 升级为主策略。正式决策前必须修复以下四个 P0 问题：

1. ProductSelector 使用全样本 NAV 长度排序，`date` 参数未参与资格过滤；29 只入选产品中有 17 只在 2018 年尚未成立；
2. T 日收盘产生的 ETF/NAV 信号被用于 T 日提交场外订单，没有落实下一可申购日执行；
3. 自动默认规则被误计为真实规则覆盖，动态选择的多数产品没有进入执行引擎的产品规则簿；
4. 每日成本收益率被再次除以初始资金，导致 B1/B2/S1 成本指标错误显示为 0。

在 P0 全部通过前，禁止继续调整状态阈值、资产预算、TopN、目标波动率或产品排名参数。

### 1. P0-A：修复时间点产品选择

#### 实现要求

ProductSelector 必须在每个选择日动态构建候选集合：

```text
EligibleProducts(t)
= 已成立
+ 当时尚未终止/清盘
+ 截至 t 已满足最短历史长度
+ 截至 t 有可用 NAV
+ 当时规则状态允许申购
+ 资产暴露映射已审核
+ 执行规则达到允许等级
```

具体修改：

1. `select(sleeve, top_n, date)` 必须使用 `date`，禁止忽略；
2. NAV 历史长度只能统计到选择日，不能使用数据库末日总记录数；
3. 产品评分只能使用选择日当时可见的费率、申购状态、规模、跟踪质量和数据质量；
4. 成立日期晚于选择日、终止日期早于选择日或历史不足的产品必须排除；
5. 当前快照无法还原历史状态时，标记 `PIT_PARTIAL`，不得假设历史期间始终开放；
6. 输出每次选择的完整审计表：候选、入选、排除原因、规则来源、首个 NAV 日、当时历史长度和评分分解；
7. 同一基金家族、同一指数和 A/C/I/F 等份额先合并比较，默认只选择一个适合预计持有期的份额。

#### 新增测试

- 2021 年成立的基金不得在 2018 年入选；
- 修改未来 NAV 不得改变历史选择；
- 相同日期、相同数据必须得到相同结果；
- 历史不足、终止、暂停申购和规则不完整产品必须被排除；
- `date=None` 在正式研究模式下必须报错，不允许退回全样本选择；
- A/C 多份额不会被同时作为两个独立资产持有。

#### 验收标准

- 任一回测成交产品在提交日前已经成立并有足够历史数据；
- 未来数据扰动不改变历史产品选择；
- 每笔订单都能追溯到当日候选快照；
- 时间点状态不足时明确降级为 `PIT_PARTIAL`，而不是声称 `PIT_COMPLETE`。

### 2. P0-B：修复信号、提交和确认时序

#### 标准时序

国内场外基金首版统一采用：

```text
T 日收盘后获得信号
→ T+1 下一有效申购日提交订单
→ 按产品规则确定确认日
→ 使用确认日 NAV 获得份额
```

QDII、港股和跨境基金单独建模：

- 中国和境外市场交易日历；
- 申购截止时间；
- 海外市场收盘时点；
- NAV 所属日期和披露日期；
- T+1/T+2 或更长确认延迟；
- 境内外节假日错位。

#### 实现要求

1. Signal Engine 输出 `signal_observed_at`；
2. 调度器将其映射到 `order_submit_date`；
3. 回测引擎使用 `signal_dates` 保留信号日与提交日的审计关系；
4. 禁止将收盘信号日期直接作为订单提交日期；
5. 月末信号使用当月最后一个有效观察日，订单在下一可申购日提交；
6. 缺失 NAV、节假日和暂停申购不能静默跳过，必须记录拒单或延期原因。

#### 新增测试

- T 日价格变化只能影响 T+1 及之后的订单；
- 修改 T+1 之后的数据不得影响 T 日信号；
- 周末、春节、国庆和境外独立休市场景；
- 国内、港股、美国 QDII 使用不同确认路径；
- 信号日、提交日、确认日和到账日严格单调。

#### 验收标准

- 所有订单满足 `signal_date < submit_date <= confirmation_date <= settlement_date`；
- 不存在同日收盘信号同日提交；
- 跨境产品不存在使用尚未收盘海外数据的情况；
- 每笔延期和拒单都有机器可读原因。

### 3. P0-C：统一产品规则与严格执行

#### 规则分级

产品规则统一分为：

| 等级 | 含义 | 是否允许正式回测成交 |
|---|---|---:|
| OFFICIAL_VERIFIED | 基金公司、招募说明书或交易所核验 | 是 |
| DISTRIBUTOR_VERIFIED | 实际销售渠道页面核验 | 是，但单独标记渠道 |
| CONSERVATIVE_ASSUMPTION | 保守推定 | 仅压力测试 |
| PRODUCT_TYPE_ASSUMPTION | 按产品类型生成 | 否 |
| MISSING | 缺失 | 否 |

#### 实现要求

1. ProductSelector 和 OTFBacktestEngine 必须使用同一个版本化 ProductRuleBook；
2. 正式实验强制 `strict_product_rules=True`；
3. 数据库自动规则不能被计入“真实规则覆盖率”；
4. 缺少费用数据时不得自动填 0；
5. 不得默认所有历史日期均开放申购赎回；
6. 最低持有期、分段赎回费、确认/到账、限购和暂停申购必须分别建模；
7. 规则表增加 `source_url`、`source_type`、`verified_at`、`effective_from`、`effective_to`、`channel` 和 `rule_status`；
8. 动态产品选择只能返回执行引擎可接受的规则等级；
9. 标准产品费率、销售渠道折扣和压力费率分别报告，不得混合。

#### 首轮补齐范围

不立即核验全部 997 只产品。先按核心资产袖套建立 40—60 只高质量候选池，每个袖套至少：

- 2 只正式可交易候选；
- 1 只备用候选；
- 至少 1 只产品规则来自官方来源或实际渠道；
- QDII、同业存单和带最短持有期产品单独审核。

#### 验收标准

- 正式实验成交产品真实规则覆盖率 100%；
- 假设规则成交数为 0；
- ProductSelector 入选产品与执行规则簿完全一致；
- 缺失规则必须拒单，不能退回统一费率；
- 报告分别披露官方、渠道、假设和缺失覆盖率。

### 4. P0-D：修复成本与绩效会计

#### 输出口径

每日结果至少包含：

- `gross_return`；
- `net_return`；
- `fee_amount`；
- `cost_return`；
- `gross_equity`；
- `net_equity`；
- 申购费、赎回费和其他费用分项；
- 当日及累计换手率。

最终报告必须计算：

```text
Gross CAGR
Net CAGR
Gross Total Return
Net Total Return
Annualized Cost Drag
Cumulative Fee Amount
Cumulative Cost Ratio
Annual Turnover
```

成本拖累以 Gross/Net 净值终值差及其年化差计算，禁止将每日成本比例简单相加后再次除以账户资金。

#### 新增测试

- 单笔申购、单笔赎回和多批次 FIFO 的费用手工算例；
- 零费率时 Gross 与 Net 完全一致；
- 非零费率时 Net 不得高于 Gross；
- 两倍费率场景成本拖累不得低于标准费率；
- 每日费用金额之和等于订单审计费用之和；
- 成本指标不能因初始资金按比例放大而发生数量级变化。

#### 验收标准

- B1/B2/S1 的费用指标不再错误显示为 0；
- 日报、订单审计和汇总报告三者费用完全勾稽；
- Gross/Net 终值差与累计费用及资金机会成本可解释；
- 年化成本拖累可以进入 P4 Gate 判断。

### 5. P1-A：落实状态滞回、调仓带和风险控制

当前配置中存在但尚未真正接入的参数必须完成实现：

1. `confirm_months=2`：普通状态切换需要连续两次观察确认；
2. `min_state_duration_days=30`：非紧急情况不得在最短持续期内切换；
3. `fast_switch_threshold`：只允许用于快速降风险；
4. 3% 调仓带必须比较新目标和上一期目标/实际权重；
5. OTFBacktestEngine 设置合理 `minimum_trade_ratio`，过滤没有经济意义的小额订单；
6. 9%目标波动率只能降低风险，不允许使用杠杆放大；
7. 市场状态、资产预算、风险缩放和最终产品权重分别保存，不得只保存最终订单；
8. 状态不变时，除越过调仓带外，不进行微小再平衡；
9. 产品替换只在季度检查，且必须覆盖赎回费后仍有明确净收益。

#### 验收标准

- 配置项均有行为测试，禁止出现“配置存在但代码未读取”；
- 输出年度状态切换次数、状态持续天数和切换原因；
- 输出由状态变化、权重漂移、产品替换分别产生的换手；
- 不能再仅凭订单数量推断状态模型频繁切换。

### 6. P1-B：修复调仓日历

废除固定 `pd.date_range(..., freq="21D")` 作为月频近似。统一采用：

```text
每月最后一个有效信号观察日
→ 下一可申购日提交
```

要求：

- 使用真实交易日历生成信号日；
- 国内、港股、美股和QDII日历分别维护；
- 节假日信号顺延或取消必须有明确规则；
- 回测引擎的最小调仓间隔不能与目标日期生成器重复阻断；
- 输出计划调仓、实际提交、延期和跳过统计。

验收：不存在因日期不在 NAV 轴而静默丢失的目标；每个自然月至多一次主动加风险调仓，紧急降风险除外。

### 7. P1-C：重建资产暴露映射

名称正则只能用于生成待审核候选，不能直接决定正式资产袖套。映射优先级为：

```text
基金合同/招募说明书跟踪标的
> 官方业绩比较基准
> ETF联接关系
> 权威数据源分类
> 名称正则候选
```

必须重点修复：

- 普通长债基金不能自动等同于3—5年国债；
- 灵活配置混合基金不能自动等同于高等级信用债；
- 转债基金不能进入高等级信用债袖套；
- 名称包含“港股”不能自动等同于恒生指数；
- 黄金股票、黄金主题与黄金 ETF 联接必须区分；
- A/C/I/F、现汇/现钞等份额不得被视作不同资产暴露。

验收：所有正式候选均有 `underlying_id`、`benchmark`、`asset_sleeve`、`mapping_source` 和 `mapping_confidence`；低置信度产品禁止成交。

### 8. P2：建立可复现的重跑协议

#### 基准重新定义

| 编号 | 正确名称 | 说明 |
|---|---|---|
| B0 | Legacy_Fixed14 | 历史固定14只候选，仅作旧基线 |
| B1 | Static_60_20_20 | 60%沪深300、20%黄金、20%国债 |
| B2 | Static_Equal_Weight_4Asset | 股票、黄金、债券、货币各25%；不得再称风险平价 |
| B3 | Rolling_Risk_Parity | 仅使用历史窗口协方差/波动率计算风险贡献 |
| S1 | State_Allocation_Fixed_Product | 状态配置，固定经核验产品 |
| S2 | State_Allocation_Dynamic_Product | S1通过后才启用时间点产品选择 |
| S3 | Full_Execution | S2接入完整产品事件和压力场景 |

#### 每次实验必须保存

- 数据库和输入文件 SHA256；
- Git commit、Python环境和依赖锁；
- 完整配置快照；
- 每日市场状态及评分；
- 资产类别预算、袖套权重和基金权重；
- 每日 Gross/Net NAV；
- 全部订单、拒单、延期、确认和到账记录；
- 产品选择候选快照；
- 费用、换手、仓位和现金占用；
- 运行日志和随机种子；
- 指标汇总与验收结果。

禁止在审计脚本中硬编码历史迭代指标或订单数量。所有报告字段必须从本次运行产物自动计算。

### 9. P3：重新实验与停止条件

#### 第一步：会计和执行对照

先运行 B1、B2：

- 验证信号/提交/确认时序；
- 验证费用和换手；
- 验证静态权重回归路径；
- 验证节假日和赎回到账后的资金占用。

若静态基准仍无法通过会计勾稽，禁止运行任何状态策略。

#### 第二步：隔离状态配置贡献

运行 S1，固定产品，不启用动态 ProductSelector。这样只比较市场状态和资产预算是否相对 B1/B2/B3 增值。

只有 S1 在冻结参数的样本外结果中优于至少一个合理静态基准，才进入 S2。若 S1 失败，保留基础设施并停止状态配置路线，不再进行第七次同区间调参。

#### 第三步：隔离产品选择贡献

运行 S2，与 S1 使用完全相同的资产预算，仅替换为时间点动态产品选择。比较：

- 净收益增量；
- 费用节省；
- 产品替换换手；
- 因限购、暂停和规则缺失造成的跟踪偏差。

若 S2 不优于 S1，则保留固定低费产品，停止动态切换。

#### 第四步：完整执行和压力测试

只有 S2 通过后运行 S3，包括：

- 标准费率；
- 实际渠道折扣；
- 无折扣；
- 非零费率双倍；
- 确认延迟增加2/5日；
- 限购、暂停申购、巨额赎回延期；
- QDII节假日错位。

### 10. P4：Walk-forward 与最终 Gate

所有状态阈值、预算和产品评分只能在训练窗内确定。建议采用扩展窗口：

```text
2018—2020 → 2021 OOS
2018—2021 → 2022 OOS
2018—2022 → 2023 OOS
2018—2023 → 2024 OOS
2018—2024 → 2025 OOS
2018—2025 → 2026 OOS（截至可用日期）
```

每个OOS年度完成后冻结，不得根据后续结果回改。正式 Gate：

- 标准费率净 CAGR 不低于最佳合理静态基准 0.75个百分点，或净 Sharpe 至少高0.15；
- 净 Sharpe不低于0.50；
- 最大回撤不劣于-15%；
- 两年滚动正收益比例不低于75%；
- 最差两年 CAGR不低于-3%；
- 年化成本拖累不高于1.25%；
- 年度双边换手率不高于100%；
- 不依赖一折渠道或单一近期窗口通过；
- 所有成交产品规则覆盖和时间点资格均为100%。

最终决策：

- 修复后的 S1 在冻结 OOS 中仍稳定落后：正式停止市场状态路线；
- S1 有效但 S2 无增值：采用状态配置加固定低费产品；
- S2/S3 均通过：进入至少一个季度前向纸面交易；
- B2/B3 表现更稳健：它们只能先成为候选基线，完成相同 OOS 和真实执行验证后才能讨论主策略。

### 11. 本轮执行顺序

严格按以下顺序执行：

1. 给现有 Phase 5 产物增加 `INVALID_FOR_STRATEGY_DECISION` 状态，不删除历史文件；
2. 修复 ProductSelector 时间点资格，并增加未来数据注入测试；
3. 修复信号日到下一申购日的映射；
4. 统一规则簿并开启严格规则阻断；
5. 修复成本会计与 Gross/Net 报告；
6. 落实状态确认、最短持续期、调仓带和目标波动率；
7. 修复月末调仓日历；
8. 重建正式候选产品的资产暴露映射；
9. 清理一次性修复脚本，将可复用逻辑并入正式模块；
10. 运行完整测试和端到端手工算例；
11. 重跑 B1/B2/B3，并完成会计勾稽；
12. 重跑固定产品 S1；
13. 根据 S1 Gate 决定是否允许 S2/S3；
14. 最后运行冻结 Walk-forward，并重写 `conclusion.md`。

完成本计划前，`conclusion.md` 中“放弃 S1”和“采用 B2 为主策略”的建议均保持撤销状态。

## 2026-07-28 市场状态综合配置与全量产品选择升级计划

### 0. 为什么当前全历史只成交 14 只基金

这不是全量场外数据抓取失败，也不是订单引擎只能处理 14 只基金，而是当前候选策略的设计结果：

1. `mapped_otf_strategy.py` 的 `_select_core_sleeves()` 只定义了 9 个进攻型资产袖套，并在每个袖套中固定选择一只代表基金；
2. `all_otf_allocation.py` 又固定加入 5 只防御型产品，覆盖货币、超短债、国债、政金债和高等级信用债；
3. 因而整个历史回测最多只会在这 14 只代表产品之间成交。448 只统一研究库是候选数据池，并不等于当前策略的实际可选池；
4. 当前没有“同一指数/资产暴露下动态选择具体产品”的 Product Selector，也没有依据市场状态动态调整股票、海外、黄金、利率债、信用债和现金预算的 State Allocator；
5. 产品执行规则采用严格阻断模式。未完成申购费、赎回费、确认延迟、到账延迟、最低持有期、限购和暂停申购规则的产品，即使有 NAV，也不能进入真实交易模拟。这项约束必须保留。

结论：需要扩大的是“具备完整规则、可被动态选择的产品覆盖”，而不是机械追求更多持仓。最终组合可只持有 5—10 个互补资产袖套，但每个袖套应有多个合格的主选和备选产品。

### 1. 本阶段目标和禁止事项

目标是将当前固定 14 只产品的策略升级为：

```text
可观测市场数据
→ 市场状态识别
→ 资产类别风险预算
→ 暴露/指数选择
→ 场外具体产品选择
→ 真实申赎与资金占用模拟
→ 冻结样本外验证
```

必须同时满足以下原则：

- 只交易场外基金；场内 ETF/指数可以作为更及时的状态和资产信号，但不能被记为实盘成交标的；
- 资产配置和产品选择分层。同一指数下不同 A/C 份额不允许被误认为两个独立资产；
- 不以“成交基金数量”作为收益优化指标，不为了扩大数量而重复持有高度相关产品；
- 第一版状态模型必须透明、可解释、低参数，禁止先上黑箱分类器；
- 宏观数据只有在具备真实发布日期和修订版本时才能使用，否则只使用价格、NAV 和当时可见的横截面信息；
- 当前多防御策略在标准产品费率下净 CAGR 仅约 1.44%，成本后有效性闸门失败。它只能作为待比较候选，不再作为可实盘主策略，也停止围绕原参数做微调。

### 2. 目标系统分层

#### 2.1 Market State Engine：市场状态识别

先采用连续状态得分，再映射为离散状态。所有特征必须在信号日收盘后才可见，并由下一可申购日执行。

建议特征组：

| 特征组 | 初版指标 | 作用 |
|---|---|---|
| 趋势 | 国内宽基、海外权益、黄金、长久期国债的 60/120/200 日趋势 | 判断主要资产方向 |
| 风险 | 20/60 日实现波动、下行波动、滚动回撤 | 判断风险扩张或收缩 |
| 广度 | 各权益袖套高于长期均线且中期动量为正的比例 | 避免单一指数代表整个市场 |
| 相关性 | 股票、债券、黄金之间的滚动相关性 | 识别分散失效 |
| 信用/流动性代理 | 信用债相对国债、短债相对长债、货币基金收益趋势 | 区分流动性和信用压力 |
| 通胀代理 | 黄金相对名义债券、价值/红利相对成长的强弱 | 识别实物资产占优环境 |

初版输出五类状态：`RISK_ON`、`NEUTRAL`、`INFLATION_REAL_ASSET`、`DEFLATION_RATE_DOWN`、`STRESS`。状态切换采用滞回机制：连续两次观察确认，或状态得分超过明确边际才切换；默认月度观察，避免周频噪声导致高换手。

#### 2.2 Asset Budget Engine：按状态分配资产预算

以下为首轮预注册预算区间，不允许在看到完整样本外结果后任意修改：

| 市场状态 | 国内权益 | 海外权益 | 黄金/实物 | 国债/政金债/信用债 | 货币/超短债 |
|---|---:|---:|---:|---:|---:|
| RISK_ON | 45%—65% | 10%—20% | 0%—10% | 10%—25% | 10%—20% |
| NEUTRAL | 25%—40% | 5%—15% | 5%—15% | 30%—45% | 15%—25% |
| INFLATION_REAL_ASSET | 15%—30% | 5%—10% | 15%—30% | 15%—30% | 20%—35% |
| DEFLATION_RATE_DOWN | 10%—25% | 0%—10% | 5%—15% | 35%—55% | 20%—35% |
| STRESS | 0%—15% | 0%—5% | 5%—20% | 15%—35% | 50%—75% |

预算引擎还应执行：

- 组合目标波动率初始区间 8%—10%，只允许降风险，不使用杠杆放大；
- 单一一级资产类别、单一指数暴露和单只产品均设上限；
- 行业/主题基金首版总预算不超过 10%，且不得成为状态判断的主要来源；
- 同一指数的多个份额合并计算风险暴露；
- 预算必须精确归一化，未分配部分进入货币/超短债，不形成隐含借款。

#### 2.3 Exposure Selector：选择互补资产暴露

首版应覆盖而不是穷举以下袖套：

- 国内权益：沪深300、中证500、中证1000、创业板/科创、红利/低波；
- 海外权益：恒生/恒生科技、标普500、纳斯达克100；
- 实物资产：黄金；其他商品只有在场外产品和规则足够可靠后再加入；
- 利率债：1—3 年、3—5 年、中长久期国债/政金债；
- 信用：高等级信用债；
- 现金管理：货币、同业存单、超短债。

每个状态先决定一级资产预算，再在一级资产内部按趋势、风险调整动量和相关性选择互补暴露。禁止直接把全量基金放在同一个动量榜单中比较，因为债券、货币和股票的收益/波动尺度不同。

#### 2.4 Product Selector：从全量场外候选中选择实际产品

对每个资产暴露建立“主选 + 备选”产品集合。评分只使用当时可得信息：

```text
ProductScore
= 低持有期综合费率
+ 跟踪质量
+ NAV 数据完整性与及时性
+ 规模/申购状态可靠性
+ 执行规则完备度
- 频繁切换惩罚
```

具体要求：

1. 先按基金家族、跟踪指数和份额类别去重；
2. 产品在当日已成立、未清盘、可申购，并满足最短历史长度；
3. 费用按预计持有期计算，不能只比较管理费或只假设一折申购；
4. 只有规则覆盖完整的产品可成交。先为每个核心袖套补齐 3 个左右候选，形成约 40—60 只高质量可执行池，再逐步扩展到 448 只；
5. 对规模、限购、暂停申购等只有当前快照的数据必须标记 `PIT_PARTIAL`，不得伪装为历史完整数据；
6. 产品切换设置改进阈值和最短持有期。只有新产品综合得分显著更高时才换，避免为几 bp 费率产生赎回成本；
7. A/C 份额根据预计持有期和真实渠道费率择优，同一基金家族同一暴露默认只持有一个份额。

### 3. 调仓与真实交易协议

- 市场状态：月度评估；资产预算：月度检查；常规产品优选：季度检查；
- 允许在回撤、波动或状态得分跨越紧急阈值时触发月内降风险，但禁止月内主动加风险；
- 使用 2%—5% 的资产预算容忍带、小额订单过滤和年度换手预算；
- 默认最短组合持有期 30 天；对存在 7/30/90/180 天分档赎回费的基金按产品规则执行；
- 保留现有 FIFO 批次、申购冻结资金、赎回应收、确认日 NAV、到账延迟、限购、暂停申购和规则缺失阻断逻辑；
- 新增渠道费率档案、历史限购/暂停事件、巨额赎回延期、订单失败/撤销、境内外节假日错位；
- 标准产品费率是主报告口径，一折渠道只能作为情景分析，不能成为策略通过的唯一依据。

### 4. 数据补齐优先级

#### P0：形成可交易的核心全资产池

1. 对 448 只研究候选完成家族、指数、A/C 份额去重审计；
2. 为每个核心资产袖套列出至少 3 只产品候选；
3. 补齐候选产品的成立/清盘、NAV、费用、确认/到账、最低持有、限购和暂停规则；
4. 建立规则字段来源、抓取日期、有效起止日期和人工核验状态；
5. 将货币基金的万份收益、七日年化和净值型基金 NAV 保持为不同数据模式；
6. 对高波动 NAV 日、份额折算、分红和异常缺口建立事件账本并阻断未经解释的数据。

P0 验收：核心袖套均有至少 2 个可交易备选；任何实际成交产品规则覆盖率 100%；重复家族/份额暴露为 0；数据缺口闸门和时间点状态通过。

#### P1：实现透明市场状态模型

新增建议模块：

- `src/otf_rotation/market_state.py`
- `config/market_state.yaml`
- `tests/test_market_state.py`

测试必须覆盖信号滞后、边界日期、缺失资产、状态滞回、阈值穿越、未来值注入检测和不同交易日历。

P1 验收：逐日状态可复算；每次切换均能解释到输入特征；改变未来数据不影响历史状态；状态年切换次数和持续期符合低频设计。

#### P2：实现资产预算和暴露选择

新增建议模块：

- `src/otf_rotation/asset_budget.py`
- `src/otf_rotation/exposure_selector.py`
- `config/state_allocation.yaml`
- `tests/test_state_allocation.py`

P2 验收：所有状态下权重非负且总和为 100%；上限、现金残余、去重和无杠杆约束全部通过；没有资产可用时安全回落到货币/超短债。

#### P3：实现动态产品选择和规则扩充

新增建议模块：

- `src/otf_rotation/product_selector.py`
- `config/channel_fee_profiles.csv`
- `config/product_selection.yaml`
- `tests/test_product_selector.py`

P3 验收：同一资产暴露可在多个合格产品中按当时数据选择；A/C 份额不会重复持有；产品替换符合阈值和持有期；任何规则缺失都会明确拒单而不是套用默认值。

#### P4：统一回测、压力测试与 Walk-forward

预注册实验组：

| 编号 | 方案 | 目的 |
|---|---|---|
| B0 | 当前固定 14 只多防御策略 | 保留历史基线 |
| B1 | 静态股/债/金配置 | 判断状态模型是否真正增值 |
| B2 | 静态风险平价 | 比较单纯风险预算效果 |
| S1 | 状态配置 + 固定代表产品 | 隔离资产配置贡献 |
| S2 | 状态配置 + 动态产品选择 | 测量产品优选净贡献 |
| S3 | S2 + 完整真实交易事件 | 最终可执行口径 |

同时执行：标准费率、一折申购渠道、无折扣、非零费率双倍、确认延迟增加 2/5 日、限购和暂停申购冲击。每个状态特征做消融测试，确认结果不是由单一特征或单一时期驱动。

Walk-forward 使用扩展窗口或滚动窗口，状态阈值和产品评分只允许用训练窗确定；每个样本外年份完成后冻结结果，不允许回看后重调。基准至少包括沪深300、货币/短债、静态股债金、静态风险平价及 B0。

P4 主验收门槛：

- 标准费率净 CAGR 不低于静态基准 0.75 个百分点，或净 Sharpe 至少高 0.15；
- 净 Sharpe 不低于 0.50，最大回撤不劣于 -15%；
- 两年滚动正收益窗口比例不低于 75%，最差两年 CAGR 不低于 -3%；
- 年化交易成本拖累不高于 1.25%，年度双边换手率不高于 100%；
- 不依赖单一最近牛市区间，不依赖一折渠道才能通过；
- 所有成交订单均能追溯信号日、状态、资产预算、产品评分、费用规则、确认 NAV 和资金到账过程。

若 S1 不优于 B1/B2，停止产品层优化并判定市场状态模型无增值；若 S1 有效而 S2 无效，保留固定低费产品，放弃动态产品切换；只有 S3 通过全部门槛才进入纸面交易。

#### P5：冻结策略与前向纸面交易

1. 冻结状态特征、阈值、资产预算、产品评分、费用档案和版本哈希；
2. 每日记录当时可见数据和拟提交订单，不回填修改历史信号；
3. 至少运行一个完整季度，最好覆盖一次状态切换；
4. 每周核对基金平台可申购状态、估算确认 NAV、实际公布 NAV、到账日和费用差异；
5. 任一核心规则变化触发版本升级和重新验收，而不是静默覆盖。

P5 验收：纸面订单与模拟状态机一致；无时间穿越；资金占用和费用差异可解释；策略仍满足风险预算后，才讨论小资金实盘。

### 5. 下一轮实际执行顺序

严格按以下顺序推进，不并行做收益参数搜索：

1. 生成 14 只成交来源审计表，列出每只基金所属袖套、首次/末次成交、成交次数和规则来源；
2. 对 448 只候选完成家族和暴露去重，输出“资产袖套—候选产品—规则完整度”矩阵；
3. 优先补齐约 40—60 只核心候选的真实执行规则，达到每个袖套至少 2 个可交易备选；
4. 实现并单测 Market State Engine，只审查状态质量，不同时优化收益；
5. 实现 Asset Budget Engine 和 Exposure Selector，先使用固定代表产品完成 B1/B2/S1；
6. 只有 S1 证明资产配置有样本外增值后，才实现动态 Product Selector 并运行 S2；
7. 将 S2 接入完整执行事件，运行 S3、压力测试和 Walk-forward；
8. 输出统一审计报告，按门槛决定“进入纸面交易 / 退回状态模型 / 停止该路线”。

本计划的核心判断是：14 只本身不是过少，真正的问题是它们由代码固定指定，缺少市场状态驱动的资产预算和同暴露产品竞争机制。下一阶段应先提高可执行候选覆盖和配置逻辑的可信度，再观察最终自然形成的持仓数。

## 2026-07-28 产品级真实交易模拟升级

- 订单引擎已加入 FIFO 申购批次、自然日持有期限、分段赎回费、金额分段/按笔申购费、基金级确认与到账延迟、最低持有期、暂停申赎、限额和规则缺失阻断。
- 当前策略实际成交 14 只基金，产品规则覆盖率 100%；3 只基金公司官方核验、10 只销售机构核验、1 只货币产品类型假设。
- 旧固定费率模型的多防守策略 CAGR 为 4.31%；标准产品费率下仅 1.44%，年化成本拖累 2.84%，Sharpe 0.220。
- 比例申购费一折、赎回费不打折的渠道情景 CAGR 为 3.32%；标准费率双倍压力下 CAGR 为 -1.29%。
- 当前策略成本后有效性 Gate 已失败。停止参数微调，下一主线改为低换手配置、低费份额优选和销售渠道费率建模。
- 完整回归结果：`267 passed in 747.95s`，`FutureWarning` 零容忍。

## 2026-07-28 全量场外基金池扩展

- 全量目录已覆盖 27,345 个份额、15,145 个基金家族，并按货币、被动权益、被动固收、债券防守和其他类型分类；状态为 `PIT_PARTIAL`。
- 新增货币基金专用数据模式：以每日万份收益复利，七日年化仅作为校验字段，不作为日收益。
- 统一研究库已扩展为 448 只基金、604,140 条记录，其中包含 438 只 ETF 联接基金和 10 只货币/债券类直接场外基金。
- 新增防守袖套覆盖货币、超短债、国债、政金债和高等级信用债。同业存单已入库，但最低持有七天规则完成前禁止交易。
- 新候选多防守配置取得 4.31% CAGR、-12.92% 最大回撤、0.551 Sharpe；双倍非零费率下 CAGR 降至 3.77%。
- 多防守方案的两年滚动正收益比例为 71.4%，低于单短债方案的 85.7%，因此暂不升级为主策略。
- 完整回归测试为 `256 passed in 744.26s`，`FutureWarning` 零容忍。
- 当前下一优先级：产品级规则验证 > 全量目录家族去重与增量 NAV > 时间点产品选择 > 前向纸面交易。

详细审计见 `all_otf_allocation_report.md`。

## 2026-07-28 最新执行状态

当前主线已经由“小型场外代表池”升级为“全量场内 ETF 信号发现 → ETF 联接基金映射 → 仅场外 NAV 执行”。场内 ETF 不作为实盘交易标的。

- 已发现 1,245 个 ETF 联接份额、545 个基金家庭；442 个 HIGH、53 个 MEDIUM、50 个 LOW 映射。
- HIGH 映射中已取得 438 只基金、568,421 条 NAV，覆盖 382 个唯一 ETF 信号；4 只新基金因无完整 NAV 未纳入。
- 首版低频策略采用 60/120/252 日风险调整动量、200 日趋势过滤、Top10、资产类上限、逆波动率和现金残余。
- 发现并修复“赎回到账后未完成剩余申购”的两阶段执行缺口；修复前平均仓位仅 67.8%，修复后为 89.3%。
- 加入0.5%最小调仓带后，2018-01-01 至 2026-07-17 基线结果为净 CAGR 9.86%、最大回撤 -21.08%、Sharpe 0.742、Calmar 0.468。
- 10/20/40 日调仓、Top5/Top10 和有无趋势过滤的预设实验已完成。Top10 和趋势过滤得到支持，但 20 日显著优于 10/40 日，参数周期稳定性尚未通过。
- 场内信号日被订单覆盖成提交日的问题已修复并增加回归测试。
- 分段审计显示基线 CAGR 为 9.67% / 4.28% / 25.13%，近期窗口贡献过大；重复黄金、红利和主题暴露明显。
- 新开发的月末稳健排名与去重主动策略均未通过，分别只有 4.67% 和 5.02% CAGR。
- 已构建439只基金的统一研究库，加入 `006663` 超短债C；核心短债防守实现4.17% CAGR、-12.85%最大回撤。
- 两年滚动回顾性闸门中，基线与核心短债防守通过；短债防守正收益窗口比例85.7%，最差窗口CAGR为-1.55%。状态仍是 `RETROSPECTIVE_NOT_PRISTINE_OOS`。

当前优先级：冻结策略并积累真正未见样本 > FIFO持有批次与持有期赎回费 > 产品级执行规则与映射独立验证 > 动态代表基金选择 > 纸面交易。详细结果见 `mapped_otf_strategy_report.md`。

> 项目状态单一事实来源（Single Source of Truth）  
> 工作目录：`D:\etf`  
> 审计日期：2026-07-28  
> 固定运行环境：`D:\miniconda\envs\agents`（Python 3.12.13）

## 1. 当前阶段与结论

项目处于“研究基础设施收口、正式策略研究之前”的阶段。数据读取、价格处理、因子、信号、ETF 回测、场外基金 NAV 执行、报告和测试链路已经存在，但目前只能支持研究结论，不能宣称稳定 Alpha，也不能直接用于实盘。

本轮优先审计数据和回测正确性，已经修复多项会显著扭曲结果的问题：

1. 场外基金样本中 8 个代码名称或资产分类错误；
2. 场外累计净值字段读取错误；
3. 场外回测遗漏市场收益、冻结资金被漏计、申购费重复扣除；
4. 已退出目标组合的基金不会赎回；
5. 赎回按到账日而非确认日 NAV 计价；
6. 缺失 NAV 被按零估值；
7. ETF 回测扣费后仍按未扣费净值漂移，形成隐含融资；
8. `Transaction_Cost_Drag%` 使用每日费用率简单相加，容易被误解；
9. 仅存在首末行情推断的生命周期表就错误标记为 `PIT_COMPLETE`。

修复后，系统的会计和时序严谨性明显提高，但仍有两个关键数据门槛未完成：完整历史退市/清盘基金池，以及更高覆盖率的官方独立价格/NAV 验证。

## 2. 当前权威数据基线

### 2.1 ETF 数据

| 项目 | 当前状态 |
|---|---:|
| Canonical DB | `data/processed/etf.sqlite` |
| ETF 数量 | 1,549 |
| ETF 日频记录 | 1,386,449 |
| 日期范围 | 2005-02-23 至 2026-07-17 |
| 正式价格模式 | `total_return_proxy` |
| 正式因子文件 | `data/processed/factors_all_repaired.csv` |
| 因子规模 | 1,386,449 行、1,549 标的、131 列，约 3.38 GB |
| 因子 manifest | `completed=true`，行数、标的数、日期与 schema 已记录 |
| 重复主键 | 0 |
| PIT 状态 | `PIT_PARTIAL` |
| 独立参考样本 | 20 只、31,175 行，其中官方/交易所样本 3 只、9,748 行 |
| 独立参考覆盖 | 约 2.25%，不足以代表全库 |

`etf_lifecycle` 当前主要由当前目录和首末行情推断。它能避免基金成立前入选，但无法证明历史已退市、合并或清盘 ETF 全部进入过当时资产池。因此，存在该表不再等于 `PIT_COMPLETE`；只有来源独立且明确覆盖历史失效产品时才能升级。

### 2.2 场外基金数据

| 项目 | 当前状态 |
|---|---:|
| Canonical DB | `data/processed/otf.sqlite` |
| 基金数量 | 18 |
| NAV 记录 | 56,672 |
| 日期范围 | 2004-03-22 至 2026-07-27 |
| 资产类别 | 9 类，每类 2 只代表产品 |
| 重复主键 | 0 |
| 关键字段空值 | 0 |
| 数据质量闸门 | 9 PASS、9 WARN、0 FAIL，整体通过 |
| 总回报模式 | `published_daily_growth_total_return_reinvested` |
| 数据来源独立 | `false`（AkShare / 东方财富公开数据） |

覆盖类别为国内宽基、红利、行业、黄金、债券、债券短期避险、恒生、纳斯达克 100、标普 500。8 份误分类历史文件已可恢复地移至：

`data/raw/otf_nav/archive_misclassified_20260728/`

9 个 WARN 主要来自来源公布的单日增长率超过 10% 或累计净值个别缺口。它们不会被静默修正；正式研究前应逐项核对高波动日是否为份额折算、分红或数据异常。

## 3. 当前回测协议

### 3.1 ETF 模式

- 信号数据日为 T；默认 `signal_to_return_lag=2`；
- T 日收盘信息产生信号，T+1 收盘执行，首个持有收益为 T+2/T+1；
- 单边费用默认 3 bp，滑点默认 2 bp；
- 调仓使用严格自融资模型：手续费从同一账户资金预留，满仓时不允许隐含借款；
- 每日持仓权重按扣费后的真实净值漂移；
- 毛收益、净收益、累计费用率、毛净终值差和毛净年化差分开报告；
- 不允许负权重和总目标权重超过 100%。

### 3.2 场外 NAV 模式

- 信号、订单、确认、结算分为独立状态；
- 申购资金在确认前计入冻结资金，赎回款在到账前计入应收资金，两者均属于账户权益；
- 申购费只扣一次；申购份额按确认 NAV 计算；
- 赎回金额按确认 NAV 固定，到账延迟只影响可用现金；
- 缺少当日 NAV 时允许使用最近公布 NAV 估值，但禁止用缺失日执行订单；
- 依据来源公布的日增长率构建总回报份额调整，信号也使用总回报序列；
- 未列入新目标的持仓会触发赎回；已有待处理订单时不重复发起新一轮调仓；
- 当前统一使用保守 T+1 规则。QDII、港股和国内基金的真实差异仍需产品级规则表。

## 4. 本轮验证结果

使用固定 `agents` 环境执行：

```text
pytest -q -W error::FutureWarning
230 passed in 770.02s
```

新增或强化的关键回归测试覆盖：

- ETF 自融资调仓和扣费后权重漂移；
- 场外 NAV 市场收益进入账户净值；
- 冻结资金和应收资金权益守恒；
- 申购、赎回、确认、结算和费用；
- 目标组合删除基金时自动赎回；
- 总回报日增长重构一致性；
- 净收益等于毛收益减交易费用；
- PIT 状态不得由推断生命周期表错误升级。

修复 PIT 判定后，统一数据闸门结果为：

```text
gate_passed = true
pit_status = PIT_PARTIAL
pit_validation_scope = current_catalog_with_observed_price_lifecycle
historical_lifecycle_independently_verified = false
```

## 5. 当前场外基线结果（仅用于系统诊断）

区间为 2018-01-01 至 2026-07-17，当前统一费用和 T+1 假设：

| 策略 | 净 CAGR | 最大回撤 | Calmar | 毛 CAGR | 年化成本拖累 | 毛净终值差 |
|---|---:|---:|---:|---:|---:|---:|
| OTF_EW | 8.15% | -13.66% | 0.597 | 8.24% | 0.09 pct | 1.35 pct |
| OTF_Momentum | 6.44% | -18.91% | 0.340 | 9.79% | 3.35 pct | 48.70 pct |

解释：

- 等权只是小规模基金池基线，不代表可执行产品组合；
- 当前动量为 20 日、Top5、每 5 日尝试调仓的占位策略，成本后明显劣于等权；
- 动量累计费用率为 25.54%，毛净终值差为 48.70 个百分点，二者含义不同；
- 当前结果说明策略换手和费用假设不匹配，不能通过调参数掩盖，应先改成低频、低换手和产品级费用模型。

## 6. 尚未解决的问题

### P0：数据与回测可信度（当前必须优先）

1. **完整 PIT 生命周期**  
   获取包含历史退市、清盘、合并产品的独立目录，建立 `historical_including_inactive` 生命周期。完成前 ETF 正式报告必须保留 `PIT_PARTIAL`。

2. **场外产品级执行规则**  
   为 18 只基金补齐申购费折扣、持有期分段赎回费、确认天数、到账天数、QDII 时差、暂停申购和限额。当前统一费率只能用于敏感性研究。

3. **异常增长日逐项审计**  
   对 9 个 WARN 的高增长事件建立事件账本，核对官方公告、分红和份额折算。错误数据必须阻断对应产品区间，不能削峰或直接删除。

4. **官方独立参考覆盖**  
   优先覆盖 18 只场外基金和 ETF 公司行动样本。`source_independent` 只能由来源和覆盖闸门动态计算。

5. **缺失行情和退出处理**  
   ETF 当前在缺失报价日按零收益、下一有效报价承接累计变化。这对正常停牌合理，但终止上市不能无限按旧价持有，需要生命周期与强制退出规则配合。

### P1：工程收口

1. 清理 `src/run_unified_experiment.py` 中已经不可达的旧主流程副本；
2. 将 `quick_*`、`run_phase*`、`debug_*` 分类到正式 CLI、实验工具或归档；
3. 校验后归档 `factors_all.csv` 和 `factors_all_repaired.csv.backup_incomplete`，不直接删除；
4. 冻结依赖版本并在 README 中固定 `agents` 环境运行方式；
5. 在干净克隆中执行安装、数据闸门和测试；
6. 逐项确认当前 Git 大规模删除与扁平化迁移，禁止 `git add -A`。

### P2：策略研究（P0 完成后）

1. 场外主线改为月频或双周频的 60/120/252 日多周期动量；
2. 加入 120/200 日趋势过滤和债券/短债避险；
3. 用缓冲区、最小权重变化和最短持有期控制换手；
4. 按资产类别先选资产，再在同类基金中按费用、跟踪误差和数据质量选产品；
5. 与等权、股债基准、单资产持有比较；
6. 做滚动 walk-forward、参数邻域稳定性和费用压力测试；
7. 只有合并 OOS 净收益、回撤、稳定性和成本后优势同时成立，策略才保留。

## 7. 下一轮明确执行顺序

```text
1. 完成 9 个场外异常增长事件组的官方事件审计
2. 建立 18 只基金的产品级费用与确认规则表
3. 增加强制退出、暂停申购、限额和持有期赎回费测试
4. 获取历史失效 ETF/基金目录并升级 PIT 数据模型
5. 清理统一实验入口的不可达代码并冻结环境依赖
6. 在修复后的正式协议上重新生成 ETF 与场外基线
7. 开始低频场外轮动策略的 walk-forward 研究
```

## 8. Gate 状态

| Gate | 状态 | 说明 |
|---|---|---|
| A 工程可复现 | 部分通过 | 固定环境 230 测试通过；Git 迁移和依赖锁定未收口 |
| B 数据可可信 | 部分通过 | 无重复和关键空值；PIT_PARTIAL、独立覆盖不足、9 WARN 待审计 |
| C 回测可可信 | 部分通过 | 核心时序与会计已修复；产品级场外规则和终止退出尚缺 |
| D 策略有效 | 未通过 | ETF 主动策略未稳定胜基准；场外动量成本后劣于等权 |
| E ML/RL 增量 | 暂停 | A-D 未通过前不继续扩展 |

只有 A、B、C 全部通过后才进行正式策略优选；只有 D 在滚动 OOS 和成本压力下通过后，才进入至少一个季度的纸面交易。
