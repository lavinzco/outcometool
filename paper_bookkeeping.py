"""Paper-trading bookkeeping helpers. No dependency on outcome_hip4_engine."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

SETTLE_GRACE_SEC = 2.0


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def apply_fill_equity(equity: float, cost_usdc: float) -> float:
    return float(equity) - float(cost_usdc)


def apply_payout_equity(equity: float, payout: float) -> float:
    return float(equity) + float(payout)


def realized_pnl(payout: float, cost_usdc: float) -> float:
    return float(payout) - float(cost_usdc)


def update_peak_dd(equity: float, peak_equity: float, max_dd: float) -> Tuple[float, float]:
    peak = max(float(peak_equity), float(equity))
    dd = (peak - float(equity)) / peak if peak > 0 else 0.0
    return peak, max(float(max_dd), dd)


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
