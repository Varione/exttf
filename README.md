# ETF / 场外基金量化研究平台

本项目用于日频 ETF 信号研究与场外基金 NAV 回测，当前重点是数据质量、时序、账户会计和可复现性，不是实盘交易系统。

## 固定运行环境

Windows 下统一使用：

```powershell
D:\miniconda\envs\agents\python.exe
```

当前验证环境为 Python 3.12.13。不要使用系统 Python 运行正式测试或实验。

## 正式入口

| 模块 | 文件 |
|---|---|
| ETF 数据读取 | `src/data_loader.py` |
| 因子定义与构建 | `src/factor_definitions.py`、`src/factor_engine.py` |
| ETF 策略与回测 | `src/strategy_library.py`、`src/backtest_engine.py` |
| 场外数据抓取 | `src/fetch_otf_funds.py` |
| 场外 NAV 回测 | `src/otf_backtest_engine.py` |
| 统一实验 | `src/run_unified_experiment.py` |
| 可复现性审计 | `src/reproducibility_audit.py` |
| 正式配置 | `config/unified_experiment.json` |

`quick_*`、`run_phase*` 和 `debug_*` 目前只属于研究辅助脚本，不得绕过统一入口生成“正式结果”。

## 当前权威数据

```text
data/processed/etf.sqlite
  1,549 只 ETF
  1,386,449 条日频记录

data/processed/otf.sqlite
  18 只场外基金
  56,672 条 NAV 记录

data/processed/factors_all_repaired.csv
  1,386,449 行、131 列
```

ETF 生命周期状态目前是 `PIT_PARTIAL`。场外数据来源为 AkShare / 东方财富公开数据，尚未达到全量官方独立验证。

## 验证命令

```powershell
& 'D:\miniconda\envs\agents\python.exe' -m pytest -q -W error::FutureWarning
```

2026-07-28 审计基线：

```text
230 passed
```

## 运行统一实验

复用已经完成并通过 manifest 校验的正式因子文件：

```powershell
& 'D:\miniconda\envs\agents\python.exe' src/run_unified_experiment.py --skip-factor-rebuild
```

不加 `--skip-factor-rebuild` 会重新构建约 3.38 GB 的因子文件，耗时和磁盘写入都较大。

## 当前限制

- 缺少覆盖历史退市、清盘和合并产品的独立 PIT 目录；
- 官方价格/NAV 独立验证覆盖率仍低；
- 场外基金暂用统一费用和确认规则，尚未完成产品级申赎规则；
- 当前主动策略尚未在滚动 OOS 和成本后稳定超过基准；
- 不应根据现有结果直接实盘。

完整状态、已修复问题和下一步顺序见 [planning.md](planning.md)。
