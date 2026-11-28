# Outcome HIP-4 纸面导出脚本

配套 `outcome_hip4_engine.py` 使用：实时纸面模拟（`paper`）、离线回测导出（`export-backtest`），以及导出结果复盘（`outcome_review.py`）。只做模拟撮合和事后分析，不下真单。

引擎源码已在本目录 `outcome_hip4_engine.py`。`pip install -r requirements.txt` 同时覆盖引擎、导出脚本和复盘脚本，其中包括未固定版本的 `matplotlib`。

## 依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 记账 / 结算 / 复盘单测（不需要联网；复盘单测需要上面的依赖）
python -m unittest test_paper_export.py test_outcome_review.py

# 查看命令说明（启动时会 import 同目录的引擎）
python outcome_paper_export.py -h

# 实时纸面模拟，默认连 mainnet 行情，仍不下真单。默认标的 BTC、周期 15m。
python outcome_paper_export.py paper --minutes 120 --out-dir paper_logs

# 从 SQLite 历史回放导出成交、拒单、校准表
python outcome_paper_export.py export-backtest --db hip4_replay.sqlite --out-dir bt_exports
```

`--settlement-fee` 默认 `0.0007`，只是可配参数，不是已核实的官方费率。这个工具只做纸面模拟，不下真单。

`discover`、`record` 和 `paper` 的默认 `--period` 仍是 `15m`。HIP-4 重复盘的周期在描述的 `period:` 里（例如 `1d`）。`template:binaryPrice` 没有这个字段；描述里的 `seconds` 是结算 TWAP 的秒数，不是合约周期。这类盘的周期用 Outcome lookup 返回的 `metadata.period`。查不到时不会拿 `seconds` 来顶替，15 分钟筛选就不会把这张盘算进去。

默认仍直接写到 `--out-dir`。可选 `--out-root` 会写到该目录下的 run_id 子目录，`--subdir-by-run-id` 会在 `--out-dir` 下再按 run_id 分子目录。不传这两个参数时，`outcome_review.py` 原来的路径继续可用。

纸面目录里复盘仍用 `paper_orders.csv`、`paper_settlements.csv`、`paper_open_positions.csv` 和 `paper_summary.json`（顶层有 `initial_equity`）。回测目录里复盘仍用 `bt_trades.csv`、`bt_rejected.csv`、`bt_calibration.csv` 和 `bt_summary.json`。同一次运行还会多写诊断文件 `decisions.csv`、`market_outcomes.csv` 和 `run_metadata.json`。

## 复盘

`outcome_review.py` 只读取已经导出的纸面或回测 CSV/JSON，不连接交易所，也不下单。

```bash
python outcome_review.py --help

# 纸面结算。期初净值优先用同目录 paper_summary.json 的 initial_equity
python outcome_review.py --settlements paper_logs/paper_settlements.csv --out-dir review_reports

# 回测成交。同目录里若有 bt_summary.json、bt_calibration.csv、bt_rejected.csv 会自动带上
python outcome_review.py --settlements bt_exports/bt_trades.csv --out-dir review_reports
```

期初净值按这个顺序取：`--summary`，否则结算文件同目录的 `paper_summary.json` / `bt_summary.json` 里的 `initial_equity`，否则 `--initial-equity`。这些都没有时，才用首行 `equity_after - pnl_usdc` 反推，并打印警告。重叠持仓时，其它未结算单的成本已经从现金里扣除，反推值会偏低，收益率和回撤都不对。

校准和订单默认不跨目录。脚本不会同时默认一份 `paper_logs` 结算和另一份 `bt_exports` 校准。看 `paper_settlements.csv` 时，只自动读同目录的 `paper_orders.csv`，不会把旁边的 `bt_calibration.csv` 混进来。看 `bt_trades.csv` 时，同目录有 `bt_calibration.csv` 或 `bt_rejected.csv` 才会读取。其它路径用 `--calibration` 和 `--orders` 显式指定。

Equity curve and MaxDD use settlement-time equity_after only, in file order (not sorted by exec_utc). This can understate drawdown versus paper_summary.json / bt_summary.json, which also count cash reserved at fill time.

净值曲线和脚本算出的 MaxDD 只用结算时点的 `equity_after`，并且保持文件原来的顺序。`paper_settlements.csv` 是按结算追加的。`bt_trades.csv` 里的 `equity_after` 是按处理顺序累加的（市场逐个处理），不能按 `exec_utc` 重排，否则这条运行净值会被打乱。成交当下已经预扣的现金不会出现在这条曲线上，所以回撤可能小于 `paper_summary.json` / `bt_summary.json`。

结算表没有 `eff_settle_fee` 列时只警告，不用 `0.0007` 填充。这个数只是本仓库里的可配估计，不是已核实的官方结算费；引擎写出该列时还会乘上该市场的 `fee_scale`。列存在时就用列里的值。缺少 `pnl_usdc`、`equity_after` 或 `tau_sec` 的行会被丢掉，并打印丢掉多少行、分别缺了哪一列。

## 这次整理改了什么

- 成交时预扣成本，结算时只加赔付，避免重叠仓位按同一笔净值反复下单
- 到期后若 60 秒 TWAP 暂时算不出，留仓下次再试，不再丢单
- 跟踪延迟成交任务，退出前等完再写汇总
- 未到期或待重试持仓写入 `paper_open_positions.csv`
- CSV 缺字段补空，使用 `utf-8-sig`，写入后立刻 flush
- 时间戳改为带 `Z` 的 ISO UTC
- 回测增加 `bt_summary.json`；拒单表字段对齐

同一市场的开仓次数跟着引擎配置 `max_trades_per_market`（默认 1）。达到上限后这次运行不再为该市场开新仓。

## 未验证

- 未与真实 `outcome_hip4_engine.py` 联调，因此 `paper` 与 `export-backtest` 主流程尚未跑通
- 未对接真实 WebSocket 行情或历史 SQLite
- 结算费、盘口字段和币对解析约定需与引擎及交易所实际数据核对
- `outcome_review.py` 用仓库里的小夹具 CSV 做过单测和命令行干跑，没有拿真实纸面或回测导出文件跑过
