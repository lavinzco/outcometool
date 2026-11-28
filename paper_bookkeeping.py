"""Paper-trading bookkeeping helpers. No dependency on outcome_hip4_engine."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SETTLE_GRACE_SEC = 2.0


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def apply_fill_equity(equity: float, cost_usdc: float) -> float:
    return float(equity) - float(cost_usdc)


def apply_payout_equity(equity: float, payout: float) -> float:
    return float(equity) + float(payout)


def realized_pnl(payout: float, cost_usdc: float) -> float:
    return float(payout) - float(cost_usdc)


# 正式纸面风控回撤口径的单一定义（供运行元数据写入）
DRAWDOWN_DEFINITION = {
    "equity_basis": "prepaid_cash",
    "formula": "(peak_equity - equity) / peak_equity",
    "peak_basis": "run_peak_equity",
    "calendar_day_reset": False,
    "halt_policy": "sticky_until_process_end",
    "limit_field": "daily_drawdown_limit",
    "limit_name_note": "字段名含 daily，但实现没有日历日重置",
}


def calc_drawdown(equity: float, peak_equity: float) -> float:
    """
    正式纸面风控回撤的唯一计算函数：
    - 正式净值：预扣后现金（成交扣 cost_usdc，结算只加 payout）。
    - 峰值：本跑 peak_equity，只升不降，不按日历日重置。
    - 回撤：peak_equity > 0 时为 (peak_equity - equity) / peak_equity，否则为 0.0。
    - 触发：回撤 >= daily_drawdown_limit（默认 0.08，闭区间）；
      字段名含 daily 但不按日重置，halted 触发后在本进程内保持粘性不自动恢复。
    """
    peak = float(peak_equity)
    if peak <= 0.0:
        return 0.0
    return (peak - float(equity)) / peak


def is_drawdown_halted(equity: float, peak_equity: float, limit: float) -> Tuple[bool, float]:
    """基于唯一回撤函数判断是否达到 halt 门槛，返回 (是否触发, 当前回撤小数)。"""
    peak = float(peak_equity)
    dd = calc_drawdown(equity, peak)
    return (peak > 0.0 and dd >= float(limit)), dd


def update_peak_dd(equity: float, peak_equity: float, max_dd: float) -> Tuple[float, float]:
    """更新本跑峰值与历史最大回撤（复用 calc_drawdown，不另写除法公式）。"""
    peak = max(float(peak_equity), float(equity))
    dd = calc_drawdown(equity, peak)
    return peak, max(float(max_dd), dd)


def compute_settlement_shadow_equity(
    initial_equity: float,
    settlement_equities: Sequence[float],
) -> dict:
    """
    仅结算时点口径的影子权益（对齐 outcome_review.py 蓝线「实际净值」，不对齐绿色零滑点线）：
    从 initial_equity 出发，按结算记录文件顺序遍历各笔 equity_after，不计入成交预扣谷底。
    仅用于写入摘要/元数据对照，不得传给正式风控 halt 或仓位计算。
    """
    init_eq = float(initial_equity)
    final_eq = init_eq
    peak_eq = init_eq
    max_dd = 0.0
    for eq in settlement_equities:
        final_eq = float(eq)
        peak_eq, max_dd = update_peak_dd(final_eq, peak_eq, max_dd)
    return {
        "shadow_equity_basis": "settlement_equity_after",
        "shadow_final_equity": round(final_eq, 4),
        "shadow_peak_equity": round(peak_eq, 4),
        "shadow_max_drawdown_pct": round(max_dd * 100.0, 3),
    }


def outcome_up_from_twap(twap: float, strike: float) -> int:
    return 1 if float(twap) > float(strike) else 0


def position_won(side: str, outcome_up: int) -> bool:
    return (side == "UP" and outcome_up == 1) or (side == "DOWN" and outcome_up == 0)


def settlement_payout(shares: float, eff_settle_fee: float, won: bool) -> float:
    return float(shares) * (1.0 - float(eff_settle_fee)) if won else 0.0


def classify_positions(
    positions: Iterable[dict],
    now: float,
    get_twap: Callable[[dict], Optional[float]],
    grace_sec: float = SETTLE_GRACE_SEC,
) -> Tuple[List[dict], List[Tuple[dict, float]]]:
    """Split positions into (still_open_or_retry, settleable).

    Expired positions with a missing TWAP stay in the retry list and are not dropped.
    """
    keep: List[dict] = []
    settleable: List[Tuple[dict, float]] = []
    for pos in positions:
        if now < float(pos["expiry_ts"]) + grace_sec:
            keep.append(pos)
            continue
        twap = get_twap(pos)
        if twap is None:
            keep.append(pos)
            continue
        settleable.append((pos, float(twap)))
    return keep, settleable


def build_settlement_row(pos: dict, twap: float, now: float, equity_after: float) -> dict:
    outcome_up = outcome_up_from_twap(twap, pos["strike"])
    won = position_won(pos["side"], outcome_up)
    payout = settlement_payout(pos["filled_shares"], pos["eff_settle_fee"], won)
    pnl = realized_pnl(payout, pos["cost_usdc"])
    return {
        "order_id": pos["order_id"],
        "settle_time_utc": utc_iso(now),
        "outcome_id": pos["outcome_id"],
        "underlying": pos["underlying"],
        "side": pos["side"],
        "strike": pos["strike"],
        "settled_twap": round(float(twap), 4),
        "outcome_up": outcome_up,
        "won": int(won),
        "filled_shares": pos["filled_shares"],
        "fill_vwap": pos["fill_vwap"],
        "cost_usdc": pos["cost_usdc"],
        "eff_settle_fee": pos["eff_settle_fee"],
        "payout_usdc": round(payout, 4),
        "pnl_usdc": round(pnl, 4),
        "equity_after": round(float(equity_after), 4),
        "fair_prob": pos.get("fair_prob", ""),
        "tau_sec": pos.get("tau_sec", ""),
        "slippage": pos.get("slippage", ""),
    }


class LiveCSVLogger:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not self.path.exists()) or self.path.stat().st_size == 0
        if write_header:
            with open(self.path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.fieldnames, extrasaction="ignore", restval=""
                )
                writer.writeheader()
                f.flush()

    def append(self, row: Dict):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.fieldnames, extrasaction="ignore", restval=""
            )
            writer.writerow(clean)
            f.flush()
