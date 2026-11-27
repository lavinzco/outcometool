# Outcome HIP-4 纸面导出脚本

配套 `outcome_hip4_engine.py` 使用：实时纸面模拟（`paper`）和离线回测导出（`export-backtest`）。只做模拟撮合，不下真单。

引擎源码已在本目录 `outcome_hip4_engine.py`；`pip install -r requirements.txt` 同时覆盖引擎与导出脚本。

## 依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 记账 / 结算 / CSV 单测（不需要引擎）
python -m unittest test_paper_export.py

# 查看命令说明（不需要引擎）
python outcome_paper_export.py -h

# 实时纸面模拟，默认连 mainnet 行情，仍不下真单
python outcome_paper_export.py paper --minutes 120 --out-dir paper_logs

# 从 SQLite 历史回放导出成交、拒单、校准表
python outcome_paper_export.py export-backtest --db hip4_replay.sqlite --out-dir bt_exports
```

`--settlement-fee` 默认 `0.0007`，只是可配参数，不是已核实的官方费率。

## 这次整理改了什么

- 成交时预扣成本，结算时只加赔付，避免重叠仓位按同一笔净值反复下单
- 到期后若 60 秒 TWAP 暂时算不出，留仓下次再试，不再丢单
- 跟踪延迟成交任务，退出前等完再写汇总
- 未到期或待重试持仓写入 `paper_open_positions.csv`
- CSV 缺字段补空，使用 `utf-8-sig`，写入后立刻 flush
- 时间戳改为带 `Z` 的 ISO UTC
- 回测增加 `bt_summary.json`；拒单表字段对齐

未加“每个市场只能开一笔”的限制。如果引擎在每次盘口更新时都给出信号，仍可能连打多笔。

## 未验证

- 未与真实 `outcome_hip4_engine.py` 联调，因此 `paper` 与 `export-backtest` 主流程尚未跑通
- 未对接真实 WebSocket 行情或历史 SQLite
- 结算费、盘口字段和币对解析约定需与引擎及交易所实际数据核对
