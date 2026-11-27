import argparse
import asyncio
import csv
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import websockets

from outcome_hip4_engine import (
    ENDPOINTS,
    HIP4AsianBinaryStrategy,
    OutcomeMarket,
    RollingMarkVol,
    StrategyConfig,
    compute_strict_twap,
    fetch_l2_book_rest,
    fetch_outcome_markets,
    parse_book_levels,
    simulate_clob_buy_fill,
)


# ============================================================
# 1. 统一 CSV / JSON 导出器（实时追加 + 防崩溃丢数据）
# ============================================================

SIGNAL_FIELDS = [
    "mode", "outcome_id", "underlying", "period", "side",
    "signal_ts_utc", "exec_ts_utc", "latency_ms", "status",
    "tau_sec", "strike", "mark_px", "acc_twap", "ann_vol", "z_score",
    "fair_prob", "signal_top_ask", "exec_vwap", "slippage",
    "expected_edge", "target_usdc", "filled_shares", "cost_usdc",
]

TRADE_FIELDS = [
    "mode", "outcome_id", "underlying", "period", "side",
    "signal_ts_utc", "exec_ts_utc", "latency_ms",
    "tau_sec", "strike", "mark_px_at_signal", "acc_twap_at_signal",
    "ann_vol", "z_score", "fair_prob", "signal_top_ask",
    "exec_vwap", "slippage", "expected_edge", "realized_edge",
    "shares", "cost_usdc", "eff_settle_fee",
    "settled_twap", "outcome_up", "won", "pnl", "equity_after",
]


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class CsvAppendLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        self._fp = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames)
        if need_header:
            self._writer.writeheader()
            self._fp.flush()

    def write(self, row: dict):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(clean)
        self._fp.flush()

    def close(self):
        self._fp.close()


def write_summary(
    out_path: str,
    mode: str,
    initial_equity: float,
    final_equity: float,
    equity_curve: List[float],
    trades: List[dict],
    signal_stats: Dict[str, int],
    brier_t: Optional[float] = None,
    brier_norm: Optional[float] = None,
    latency_ms: int = 300,
):
    eq_arr = np.array(equity_curve, dtype=float)
    run_max = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((run_max - eq_arr) / np.maximum(run_max, 1e-9))) if len(eq_arr) else 0.0

    total_signals = sum(signal_stats.values())
    filled_signals = signal_stats.get("FILLED", 0)

    summary = {
        "mode": mode,
        "generated_at_utc": utc_iso(time.time()),
        "latency_ms": latency_ms,
        "initial_equity": initial_equity,
        "final_equity": round(final_equity, 4),
        "return_pct": round((final_equity / initial_equity - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "total_signals": total_signals,
        "fill_rate_pct": round((filled_signals / total_signals) * 100.0, 2) if total_signals else 0.0,
        "signal_breakdown": signal_stats,
        "settled_trades": len(trades),
        "win_rate_pct": round(float(np.mean([t["won"] for t in trades])) * 100.0, 2) if trades else 0.0,
        "avg_slippage": round(float(np.mean([t["slippage"] for t in trades])), 6) if trades else 0.0,
        "avg_realized_edge": round(float(np.mean([t["realized_edge"] for t in trades])), 5) if trades else 0.0,
        "brier_student_t": round(brier_t, 6) if brier_t is not None else None,
        "brier_normal_baseline": round(brier_norm, 6) if brier_norm is not None else None,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ============================================================
# 2. 实时模拟下单引擎（Live Paper Trading）
# ============================================================

async def run_paper_trading(
    network: str,
    underlying: str,
    period: str,
    out_dir: str,
    cfg: StrategyConfig,
    latency_ms: int = 300目前的版本**还不直接支持**：`record` 子命令只负责录制行情不运行策略，`backtest` 子命令虽然做了模拟撮合，但成交明细只打印在终端，没有落盘导出，也没有记录“被滑点/深度拦截的信号”。

为了不改动你现有的 `outcome_hip4_engine.py`，下面提供一个配套扩展脚本 **`outcome_paper_export.py`**（放在同一目录下直接导入原引擎），补齐两项核心能力：

| 新增能力 | 运行模式 | 自动导出的分析文件 | 核心复盘用途 |
| :--- | :--- | :--- | :--- |
| **实时模拟盘（Paper Trading）** | `paper` | `paper_orders.csv`<br>`paper_settlements.csv`<br>`paper_summary.json` | 接入实时 WebSocket，触发信号后按设定延迟等待真实的 $t + \Delta t$ 盘口做逐档吃单，实时追加写入，到期自动按 60s TWAP 结算 |
| **离线回测深度导出** | `export-backtest` | `bt_trades.csv`<br>`bt_rejected.csv`<br>`bt_calibration.csv` | 把 SQLite 历史回放结果拆成成交明细、滑点拒单审计表、概率校准原始表，方便直接用 Excel/Python 画净值曲线和可靠性曲线 |

---

### 扩展脚本：`outcome_paper_export.py`

将其保存在与 `outcome_hip4_engine.py` 相同的目录即可直接运行（无需额外安装新依赖）：

```python
import argparse
import asyncio
import csv
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import websockets

from outcome_hip4_engine import (
    ENDPOINTS,
    HIP4AsianBinaryStrategy,
    OutcomeMarket,
    RollingMarkVol,
    StrategyConfig,
    compute_strict_twap,
    fetch_l2_book_rest,
    fetch_outcome_markets,
    parse_book_levels,
    simulate_clob_buy_fill,
)


# ============================================================
# 1. 通用 CSV 实时追加器（防中途退出丢记录）
# ============================================================

class LiveCSVLogger:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        if write_header:
            with open(self.path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ============================================================
# 2. 实时模拟下单引擎（Live Paper Trading + 自动导出）
# ============================================================

ORDER_FIELDS = [
    "order_id", "signal_time_utc", "exec_time_utc", "latency_actual_ms",
    "outcome_id", "underlying", "period", "side", "status", "reject_reason",
    "strike", "mark_px_signal", "acc_twap_signal", "ann_vol", "tau_sec", "z_score",
    "fair_prob", "expected_edge", "top_ask_signal", "target_usdc",
    "filled_shares", "fill_vwap", "cost_usdc", "slippage", "realized_edge",
]

SETTLE_FIELDS = [
    "order_id", "settle_time_utc", "outcome_id", "underlying", "side",
    "strike", "settled_twap", "outcome_up", "won",
    "filled_shares", "fill_vwap", "cost_usdc", "eff_settle_fee",
    "payout_usdc", "pnl_usdc", "equity_after", "fair_prob", "tau_sec", "slippage",
]


async def run_paper_trading(
    network: str,
    underlying: str,
    period: str,
    cfg: StrategyConfig,
    latency_ms: int,
    max_slippage: float,
    out_dir: str,
    duration_min: float,
    poll_meta_sec: float = 20.0,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    order_logger = LiveCSVLogger(out_path / "paper_orders.csv", ORDER_FIELDS)
    settle_logger = LiveCSVLogger(out_path / "paper_settlements.csv", SETTLE_FIELDS)

    ws_url = ENDPOINTS[network]["ws"]
    underlying = underlying.upper()
    stop_ts = time.time() + duration_min * 60.0

    strat = HIP4AsianBinaryStrategy(cfg)
    vol_est = RollingMarkVol(cfg.ewma_lambda, cfg.min_ann_vol)

    equity = cfg.initial_equity
    peak_equity = equity
    max_dd = 0.0
    order_seq = 0

    tracked_markets: Dict[str, OutcomeMarket] = {}
    settled_ids: set[str] = set()
    subscribed_coins: set[str] = set()
    latest_books: Dict[str, Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float]] = {}
    mark_history: List[Tuple[float, float]] = []
    latest_mark_px: Optional[float] = None
    ann_vol = cfg.min_ann_vol

    # 未到期模拟持仓列表
    open_positions: List[dict] = []
    settled_records: List[dict] = []

    async def execute_delayed_order(
        oid_num: int,
        market: OutcomeMarket,
        sig: dict,
        mark_at_sig: float,
        twap_at_sig: Optional[float],
        vol_at_sig: float,
    ):
        """等待模拟网络延迟后，取真实最新盘口做 L2 逐档吃单"""
        await asyncio.sleep(latency_ms / 1000.0)
        exec_ts = time.time()
        book = latest_books.get(market.outcome_id)

        base_row = {
            "order_id": f"PAPER-{oid_num:05d}",
            "signal_time_utc": utc_iso(sig["signal_ts"]),
            "exec_time_utc": utc_iso(exec_ts),
            "latency_actual_ms": round((exec_ts - sig["signal_ts"]) * 1000.0, 1),
            "outcome_id": market.outcome_id,
            "underlying": market.underlying,
            "period": market.period,
            "side": sig["side"],
            "strike": market.strike,
            "mark_px_signal": round(mark_at_sig, 4),
            "acc_twap_signal": round(twap_at_sig, 4) if twap_at_sig is not None else "",
            "ann_vol": round(vol_at_sig, 4),
            "tau_sec": round(sig["tau_sec"], 1),
            "z_score": round(sig["z_score"], 4),
            "fair_prob": round(sig["fair_prob"], 5),
            "expected_edge": round(sig["expected_edge"], 5),
            "top_ask_signal": round(sig["top_ask"], 5),
            "target_usdc": sig["target_usdc"],
        }

        if not book:
            order_logger.append({**base_row, "status": "REJECTED", "reject_reason": "NO_BOOK"})
            print(f"[Paper] {base_row['order_id']} #{market.outcome_id} 拒单: 无盘口数据")
            return

        yes_bids, yes_asks, _ = book
        fill = simulate_clob_buy_fill(
            side=sig["side"],
            target_usdc=sig["target_usdc"],
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            min_notional=cfg.min_order_notional,
        )
        if fill is None:
            order_logger.append({**base_row, "status": "REJECTED", "reject_reason": "NO_LIQUIDITY"})
            print(f"[Paper] {base_row['order_id']} #{market.outcome_id} 拒单: 深度不足或低于最小名义")
            return

        shares, vwap, cost_usdc = fill
        slippage = vwap - sig["top_ask"]
        max_limit_px = min(cfg.max_entry_price, sig["top_ask"] + max_slippage)

        if vwap > max_limit_px:
            order_logger.append({
                **base_row,
                "status": "REJECTED",
                "reject_reason": "SLIPPAGE_EXCEEDED",
                "filled_shares": 0,
                "fill_vwap": round(vwap, 5),
                "cost_usdc": 0.0,
                "slippage": round(slippage, 5),
            })
            print(
                f"[Paper] {base_row['order_id']} #{market.outcome_id} {sig['side']} "
                f"IOC撤单: 预期 {sig['top_ask']:.4f} -> 延迟后 {vwap:.4f} (滑点 {slippage:+.4f})"
            )
            return

        realized_edge = sig["fair_prob"] * (1.0 - sig["eff_settle_fee"]) - vwap
        filled_row = {
            **base_row,
            "status": "FILLED",
            "reject_reason": "",
            "filled_shares": shares,
            "fill_vwap": round(vwap, 5),
            "cost_usdc": round(cost_usdc, 4),
            "slippage": round(slippage, 5),
            "realized_edge": round(realized_edge, 5),
        }
        order_logger.append(filled_row)

        pos = {
            **filled_row,
            "eff_settle_fee": sig["eff_settle_fee"],
            "expiry_ts": market.expiry_ts,
        }
        open_positions.append(pos)
        print(
            f"[Paper] 模拟成交 {filled_row['order_id']} | #{market.outcome_id} {sig['side']} "
            f"{shares}股 @ {vwap:.4f} (花费 {cost_usdc:.2f} USDC | Fair={sig['fair_prob']:.3f} | "
            f"滑点={slippage:+.4f} | 剩 {int(sig['tau_sec'])}s)"
        )

    def check_settlements():
        nonlocal equity, peak_equity, max_dd
        now = time.time()
        # 清理 40 分钟前的 markPx 缓存避免内存增长
        cutoff = now - 2400.0
        while mark_history and mark_history[0][0] < cutoff:
            mark_history.pop(0)

        remaining_pos = []
        for pos in open_positions:
            if now < pos["expiry_ts"] + 2.0:
                remaining_pos.append(pos)
                continue

            twap = compute_strict_twap(mark_history, pos["expiry_ts"] - 60.0, pos["expiry_ts"])
            if twap is None:
                print(f"[Settle] 警告: #{pos['outcome_id']} 缺少 60s markPx，跳过结算")
                continue

            outcome_up = 1 if twap > pos["strike"] else 0
            won = (pos["side"] == "UP" and outcome_up == 1) or (pos["side"] == "DOWN" and outcome_up == 0)
            payout = pos["filled_shares"] * (1.0 - pos["eff_settle_fee"]) if won else 0.0
            pnl = payout - pos["cost_usdc"]
            equity += pnl
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_dd = max(max_dd, dd)

            s_row = {
                "order_id": pos["order_id"],
                "settle_time_utc": utc_iso(now),
                "outcome_id": pos["outcome_id"],
                "underlying": pos["underlying"],
                "side": pos["side"],
                "strike": pos["strike"],
                "settled_twap": round(twap, 4),
                "outcome_up": outcome_up,
                "won": int(won),
                "filled_shares": pos["filled_shares"],
                "fill_vwap": pos["fill_vwap"],
                "cost_usdc": pos["cost_usdc"],
                "eff_settle_fee": pos["eff_settle_fee"],
                "payout_usdc": round(payout, 4),
                "pnl_usdc": round(pnl, 4),
                "equity_after": round(equity, 4),
                "fair_prob": pos["fair_prob"],
                "tau_sec": pos["tau_sec"],
                "slippage": pos["slippage"],
            }
            settle_logger.append(s_row)
            settled_records.append(s_row)
            settled_ids.add(pos["outcome_id"])
            print(
                f"[Settle] {pos['order_id']} #{pos['outcome_id']} | TWAP={twap:.2f} vs Strike={pos['strike']:.2f} "
                f"-> {'WIN' if won else 'LOSS'} | PnL={pnl:+.2f} USDC | 净值={equity:.2f}"
            )

        open_positions[:] = remaining_pos

    print(
        f"[Paper] 启动实时模拟盘 ({network}) | {underlying} {period} | "
        f"初始净值={equity} USDC | 模拟延迟={latency_ms}ms | 输出目录={out_path.resolve()}"
    )

    try:
        while time.time() < stop_ts:
            try:
                async with websockets.connect(ws_url, ping_interval=None, max_size=4 * 1024 * 1024) as ws:
                    subscribed_coins.clear()
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "activeAssetCtx", "coin": underlying},
                    }))
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "outcomeMetaUpdates"},
                    }))

                    async def sync_markets():
                        markets = await asyncio.to_thread(
                            fetch_outcome_markets, network, period, underlying, True
                        )
                        for m in markets:
                            tracked_markets[m.outcome_id] = m
                            if m.yes_coin not in subscribed_coins:
                                await ws.send(json.dumps({
                                    "method": "subscribe",
                                    "subscription": {"type": "l2Book", "coin": m.yes_coin},
                                }))
                                subscribed_coins.add(m.yes_coin)
                                snap = await asyncio.to_thread(fetch_l2_book_rest, network, m.yes_coin)
                                if snap and "levels" in snap:
                                    b, a = parse_book_levels(snap)
                                    latest_books[m.outcome_id] = (b, a, time.time())
                                print(
                                    f"[Paper] 监控盘口 {m.yes_coin} | Strike={m.strike} | "
                                    f"Exp={datetime.fromtimestamp(m.expiry_ts, tz=timezone.utc).strftime('%H:%M UTC')}"
                                )

                    await sync_markets()
                    last_meta_sync = time.time()
                    last_ping = time.time()

                    while time.time() < stop_ts:
                        now = time.time()
                        if now - last_ping >= 25.0:
                            await ws.send(json.dumps({"method": "ping"}))
                            last_ping = now

                        if now - last_meta_sync >= poll_meta_sec:
                            await sync_markets()
                            check_settlements()
                            last_meta_sync = now

                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                        except asyncio.TimeoutError:
                            check_settlements()
                            continue

                        msg = json.loads(raw_msg)
                        channel = msg.get("channel")
                        data = msg.get("data")

                        if channel == "activeAssetCtx" and isinstance(data, dict):
                            ctx = data.get("ctx", {})
                            mpx_str = ctx.get("markPx")
                            if mpx_str is not None and str(data.get("coin", underlying)).upper() == underlying:
                                t_now = time.time()
                                latest_mark_px = float(mpx_str)
                                mark_history.append((t_now, latest_mark_px))
                                ann_vol = vol_est.update(t_now, latest_mark_px)

                        elif channel == "l2Book" and isinstance(data, dict):
                            coin_str = str(data.get("coin", ""))
                            if coin_str.startswith("#") and coin_str.endswith("0"):
                                oid = coin_str[1:-1]
                                b, a = parse_book_levels(data)
                                t_now = time.time()
                                latest_books[oid] = (b, a, t_now)

                                market = tracked_markets.get(oid)
                                if market and latest_mark_px is not None and oid not in settled_ids:
                                    twap_start = market.expiry_ts - 60.0
                                    acc_twap = (
                                        compute_strict_twap(mark_history, twap_start, t_now)
                                        if t_now > twap_start
                                        else None
                                    )
                                    sig = strat.evaluate(
                                        market, t_now, latest_mark_px, ann_vol, b, a, acc_twap, equity
                                    )
                                    if sig is not None:
                                        order_seq += 1
                                        asyncio.create_task(
                                            execute_delayed_order(
                                                order_seq, market, sig, latest_mark_px, acc_twap, ann_vol
                                            )
                                        )

                        elif channel == "outcomeMetaUpdates":
                            await sync_markets()

            except Exception as exc:
                print(f"[Paper] WS 连接中断 ({exc})，3 秒后重连...")
                await asyncio.sleep(3.0)
    except KeyboardInterrupt:
        print("\n[Paper] 收到手动中断，正在结算已到期场次并导出汇总...")

    check_settlements()
    summary = {
        "finished_at_utc": utc_iso(time.time()),
        "network": network,
        "underlying": underlying,
        "period": period,
        "latency_ms": latency_ms,
        "initial_equity": cfg.initial_equity,
        "final_equity": round(equity, 4),
        "return_pct": round((equity / cfg.initial_equity - 1.0) * 100.0, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 3),
        "settled_trades": len(settled_records),
        "win_rate_pct": round(
            (sum(r["won"] for r in settled_records) / len(settled_records) * 100.0), 2
        ) if settled_records else 0.0,
        "open_unsettled_positions": len(open_positions),
        "halted_by_drawdown": strat.halted,
    }
    with open(out_path / "paper_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Paper] 汇总已导出至 {out_path / 'paper_summary.json'}:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ============================================================
# 3. 离线回测增强导出（含拒单审计表 + 概率校准表）
# ============================================================

def export_replay_backtest(
    db_path: str,
    cfg: StrategyConfig,
    latency_ms: int,
    max_slippage: float,
    out_dir: str,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    m_rows = conn.execute(
        """
        SELECT outcome_id, name, underlying, strike, expiry_ts, period,
               fee_scale, yes_coin, raw_description, settled_twap, outcome_up
        FROM markets
        WHERE outcome_up IS NOT NULL
        ORDER BY expiry_ts ASC
        """
    ).fetchall()

    if not m_rows:
        print("[ExportBT] SQLite 中暂无已结算场次。")
        conn.close()
        return

    strat = HIP4AsianBinaryStrategy(cfg)
    vol_trackers: Dict[str, RollingMarkVol] = {}
    equity = cfg.initial_equity
    latency_sec = latency_ms / 1000.0

    trades_out: List[dict] = []
    rejected_out: List[dict] = []
    calib_out: List[dict] = []

    for row in m_rows:
        m = OutcomeMarket(
            outcome_id=row[0], name=row[1], underlying=row[2], strike=row[3],
            expiry_ts=row[4], period=row[5], fee_scale=row[6], yes_coin=row[7],
            no_coin=f"#{row[0]}1",
            yes_asset_id=100_000_000 + int(row[0]) * 10,
            no_asset_id=100_000_000 + int(row[0]) * 10 + 1,
            raw_description=row[8],
        )
        settled_twap, outcome_up = row[9], int(row[10])
        vol_est = vol_trackers.setdefault(
            m.underlying, RollingMarkVol(cfg.ewma_lambda, cfg.min_ann_vol)
        )

        mark_rows = conn.execute(
            "SELECT ts, mark_px FROM mark_ticks WHERE underlying = ? AND ts >= ? AND ts <= ? ORDER BY ts ASC",
            (m.underlying, m.expiry_ts - 1800.0, m.expiry_ts),
        ).fetchall()
        book_rows = conn.execute(
            "SELECT ts, bids_json, asks_json FROM book_snapshots WHERE outcome_id = ? AND ts <= ? ORDER BY ts ASC",
            (m.outcome_id, m.expiry_ts),
        ).fetchall()
        if not mark_rows or not book_rows:
            continue

        book_ts_arr = np.array([b[0] for b in book_rows], dtype=float)
        twap_start = m.expiry_ts - 60.0
        mark_idx = 0
        latest_mark_px: Optional[float] = None
        ann_vol = cfg.min_ann_vol
        twap_buffer: List[Tuple[float, float]] = []
        executed_orders: List[dict] = []

        for b_idx, (bts, bids_s, asks_s) in enumerate(book_rows):
            while mark_idx < len(mark_rows) and mark_rows[mark_idx][0] <= bts:
                mts, mpx = mark_rows[mark_idx]
                latest_mark_px = mpx
                ann_vol = vol_est.update(mts, mpx)
                if mts >= twap_start - 10.0:
                    twap_buffer.append((mts, mpx))
                mark_idx += 1

            if latest_mark_px is None:
                continue

            tau_sec = m.expiry_ts - bts
            acc_twap = compute_strict_twap(twap_buffer, twap_start, bts) if bts > twap_start else None

            if b_idx % 5 == 0 and 10.0 <= tau_sec <= 860.0:
                p_t, p_n, z_val = strat.fair_prob_up_twap(latest_mark_px, m.strike, tau_sec, ann_vol, acc_twap)
                calib_out.append({
                    "outcome_id": m.outcome_id,
                    "ts_utc": utc_iso(bts),
                    "tau_sec": round(tau_sec, 1),
                    "mark_px": round(latest_mark_px, 4),
                    "strike": m.strike,
                    "ann_vol": round(ann_vol, 4),
                    "z_score": round(z_val, 4),
                    "p_up_student_t": round(p_t, 5),
                    "p_up_normal": round(p_n, 5),
                    "outcome_up": outcome_up,
                })

            sig = strat.evaluate(
                m, bts, latest_mark_px, ann_vol, json.loads(bids_s), json.loads(asks_s), acc_twap, equity
            )
            if sig is None:
                continue

            exec_idx = int(np.searchsorted(book_ts_arr, bts + latency_sec, side="left"))
            exec_idx = min(exec_idx, len(book_rows) - 1)
            exec_ts, exec_bids_s, exec_asks_s = book_rows[exec_idx]

            fill = simulate_clob_buy_fill(
                side=sig["side"],
                target_usdc=sig["target_usdc"],
                yes_bids=json.loads(exec_bids_s),
                yes_asks=json.loads(exec_asks_s),
                min_notional=cfg.min_order_notional,
            )
            if fill is None:
                rejected_out.append({
                    "outcome_id": m.outcome_id,
                    "signal_utc": utc_iso(bts),
                    "side": sig["side"],
                    "reason": "NO_LIQUIDITY",
                    "top_ask": sig["top_ask"],
                    "fair_prob": round(sig["fair_prob"], 5),
                    "tau_sec": round(tau_sec, 1),
                    "outcome_up": outcome_up,
                })
                continue

            shares, vwap, cost_usdc = fill
            slippage = vwap - sig["top_ask"]
            if vwap > min(cfg.max_entry_price, sig["top_ask"] + max_slippage):
                rejected_out.append({
                    "outcome_id": m.outcome_id,
                    "signal_utc": utc_iso(bts),
                    "side": sig["side"],
                    "reason": "SLIPPAGE_EXCEEDED",
                    "top_ask": sig["top_ask"],
                    "delayed_vwap": round(vwap, 5),
                    "slippage": round(slippage, 5),
                    "fair_prob": round(sig["fair_prob"], 5),
                    "tau_sec": round(tau_sec, 1),
                    "outcome_up": outcome_up,
                })
                continue

            executed_orders.append({
                **sig,
                "exec_ts": exec_ts,
                "mark_px": latest_mark_px,
                "ann_vol": ann_vol,
                "shares": shares,
                "vwap": vwap,
                "cost_usdc": cost_usdc,
                "slippage": slippage,
            })

        for od in executed_orders:
            won = (od["side"] == "UP" and outcome_up == 1) or (od["side"] == "DOWN" and outcome_up == 0)
            payout = od["shares"] * (1.0 - od["eff_settle_fee"]) if won else 0.0
            pnl = payout - od["cost_usdc"]
            equity += pnl
            trades_out.append({
                "outcome_id": m.outcome_id,
                "underlying": m.underlying,
                "signal_utc": utc_iso(od["signal_ts"]),
                "exec_utc": utc_iso(od["exec_ts"]),
                "side": od["side"],
                "strike": m.strike,
                "mark_px_signal": round(od["mark_px"], 4),
                "settled_twap": round(settled_twap, 4),
                "ann_vol": round(od["ann_vol"], 4),
                "tau_sec": round(od["tau_sec"], 1),
                "z_score": round(od["z_score"], 4),
                "fair_prob": round(od["fair_prob"], 5),
                "top_ask_signal": round(od["top_ask"], 5),
                "fill_vwap": round(od["vwap"], 5),
                "slippage": round(od["slippage"], 5),
                "shares": od["shares"],
                "cost_usdc": round(od["cost_usdc"], 4),
                "eff_settle_fee": od["eff_settle_fee"],
                "won": int(won),
                "pnl_usdc": round(pnl, 4),
                "equity_after": round(equity, 4),
            })

    conn.close()

    def write_csv(filename: str, rows: List[dict]):
        fp = out_path / filename
        if not rows:
            print(f"[ExportBT] {filename}: 无数据，跳过写入")
            return
        with open(fp, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[ExportBT] 已导出 {len(rows)} 条记录 -> {fp.resolve()}")

    write_csv("bt_trades.csv", trades_out)
    write_csv("bt_rejected.csv", rejected_out)
    write_csv("bt_calibration.csv", calib_out)


# ============================================================
# 4. CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Outcome HIP-4 实时模拟下单与回测导出工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_paper = sub.add_parser("paper", help="连接实时盘口运行纸面交易并自动导出 CSV/JSON")
    p_paper.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    p_paper.add_argument("--underlying", default="BTC")
    p_paper.add_argument("--period", default="15m")
    p_paper.add_argument("--equity", type=float, default=1000.0)
    p_paper.add_argument("--latency-ms", type=int, default=300)
    p_paper.add_argument("--max-slippage", type=float, default=0.008)
    p_paper.add_argument("--settlement-fee", type=float, default=0.0007)
    p_paper.add_argument("--out-dir", default="paper_logs")
    p_paper.add_argument("--minutes", type=float, default=120.0)

    p_exp = sub.add_parser("export-backtest", help="跑 SQLite 历史回放并导出成交/拒单/校准 CSV")
    p_exp.add_argument("--db", default="hip4_replay.sqlite")
    p_exp.add_argument("--equity", type=float, default=1000.0)
    p_exp.add_argument("--latency-ms", type=int, default=300)
    p_exp.add_argument("--max-slippage", type=float, default=0.008)
    p_exp.add_argument("--settlement-fee", type=float, default=0.0007)
    p_exp.add_argument("--out-dir", default="bt_exports")

    args = parser.parse_args()
    cfg = StrategyConfig(initial_equity=args.equity, settlement_fee_rate=args.settlement_fee)

    if args.cmd == "paper":
        asyncio.run(
            run_paper_trading(
                network=args.network,
                underlying=args.underlying,
                period=args.period,
                cfg=cfg,
                latency_ms=args.latency_ms,
                max_slippage=args.max_slippage,
                out_dir=args.out_dir,
                duration_min=args.minutes,
            )
        )
    elif args.cmd == "export-backtest":
        export_replay_backtest(
            db_path=args.db,
            cfg=cfg,
            latency_ms=args.latency_ms,
            max_slippage=args.max_slippage,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()