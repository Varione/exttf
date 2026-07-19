# 场外 ETF 联接基金跨资产周频轮动

这是第一版研究 baseline：在指数/场内 ETF 价格层计算信号，在场外 ETF 联接基金层执行。策略采用 60 日风险调整动量、120 日趋势过滤、周频调仓、Top 3 等权；没有合格风险资产时持有防御资产。

## 数据源决定

默认数据源：AKShare。它不要求 Token，适合第一版研究和小规模日频抓取：

- `fund_etf_hist_em`：场内 ETF 历史行情，用于信号层；网络受限时自动回退到 `fund_etf_hist_sina`；
- `fund_open_fund_info_em`：开放式基金单位净值历史，用于执行层；
- `fund_open_fund_daily_em` / `fund_info_index_em`：基金列表和基本信息辅助表。

增强数据源：Tushare Pro。需要注册 Token 和至少相应接口积分，作为可选校验源，不作为默认依赖。正式实盘前，应把关键净值和申赎规则与基金公司公告/招募说明书交叉核验。

## 快速开始

```powershell
python -m pip install -e ".[data,dev]"
python -m pytest
```

如果本机 pip 镜像找不到 AKShare，改用公开 PyPI：

```powershell
python -m pip install --index-url https://pypi.org/simple -e ".[data,dev]"
```

先用内置模拟数据验证回测：

```powershell
python -m otf_rotation.cli demo --output-dir reports
```

联网抓取场内 ETF 信号价格：

```powershell
python -m otf_rotation.cli fetch-etf --config config/universe.csv --start-date 20180101 --end-date 20260718
```

抓取当前可发现的全部 ETF 历史日线（每只 ETF 一个 CSV，支持断点续传）：

```powershell
python -m otf_rotation.cli fetch-all-etf --start-date 20000101 --end-date 20500101 --workers 4
```

输出目录包含 `etf_catalog.csv`、`history/<代码>.csv`、`etf_fetch_status.csv` 和失败清单 `etf_failed.csv`。这里的“全部”指 AKShare 当前 ETF 行情清单中的产品及其可返回的历史日线 OHLCV/成交额，不包含场外联接基金 NAV。

运行回测：

```powershell
python -m otf_rotation.cli backtest --prices data/raw/signal_prices.csv --output-dir reports
```

## 重要的执行假设

策略在周末收盘数据上生成信号，新权重从下一个可用交易日开始生效；不会用当日收盘信号获得当日成交。`backtest` 的 `--cost-bps` 是每次组合换手的简化成本，默认 10 bps。QDII 的真实净值确认、申购截止时间、赎回到账和限购状态尚未自动推断，后续应加入产品级规则表。

## 数据文件格式

`data/raw/signal_prices.csv` 为宽表：第一列 `date`，其余列为 `asset`，数值为场内 ETF 收盘价。`config/universe.csv` 中的 `signal_symbol` 是场内 ETF 代码；`implementation_fund` 预留给场外联接基金代码，不参与当前信号回测，避免未经核验地把基金代码写死。
