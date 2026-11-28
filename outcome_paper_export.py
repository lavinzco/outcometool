import argparse
import asyncio
import csv
import hashlib
import json
import math
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import outcome_hip4_engine as engine_mod
from market_data import BookEvent, LiveMarketFeed, MarkEvent, MarketTapeWriter
from outcome_hip4_engine import (
    ENDPOINTS,
    REJECT_COOLDOWN_SEC,
    DetailedFill,
    EvaluationDecision,
    HIP4AsianBinaryStrategy,
    OutcomeMarket,
    RollingMarkVol,
    SideQuote,
    StrategyConfig,
    compute_strict_twap,
    decompose_slippage,
    extract_side_quote,
    fetch_l2_book_rest,
    fetch_outcome_markets,
    parse_book_levels,
    simulate_clob_buy_fill_detailed,
)
from paper_bookkeeping import (
    DRAWDOWN_DEFINITION,
    apply_fill_equity,
    apply_payout_equity,
    compute_settlement_shadow_equity,
    update_peak_dd,
)

# 兼容既有引用名：与引擎策略类为同一实现，不再维护第二套 evaluate
InstrumentedHIP4Strategy = HIP4AsianBinaryStrategy

SETTLEMENT_FEE_NOTE = (
    "--settlement-fee 默认 0.0007 仅为可配模拟/回测参数，并非已核实的 Outcome 官方统一结算费率；"
    "实际结算扣费以对应市场 spec 为准。"
)


# ============================================================
# 1. 账务核算层：净值与回撤改写唯一委托给 paper_bookkeeping.py
# ============================================================

class EquityBookkeeper:
    """
    有状态账本包装：
    - 成交即调用 paper_bookkeeping.apply_fill_equity 扣减 cost_usdc 并刷新 peak/max_dd；
    - 结算只调用 paper_bookkeeping.apply_payout_equity 增加 payout_usdc 并刷新 peak/max_dd。
    """
    def __init__(self, initial_equity: float):
        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.max_dd = 0.0
        self.source = "paper_bookkeeping"

    def apply_fill(self, cost_usdc: float) -> Tuple[float, float, float]:
        self.equity = float(apply_fill_equity(self.equity, cost_usdc))
        self.peak_equity, self.max_dd = update_peak_dd(self.equity, self.peak_equity, self.max_dd)
        return self.equity, self.peak_equity, self.max_dd

    def apply_payout(self, payout_usdc: float) -> Tuple[float, float, float]:
        self.equity = float(apply_payout_equity(self.equity, payout_usdc))
        self.peak_equity, self.max_dd = update_peak_dd(self.equity, self.peak_equity, self.max_dd)
        return self.equity, self.peak_equity, self.max_dd


# ============================================================
# 2. 元数据、版本指纹与基础工具
# ============================================================

def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def file_sha256_short(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:
        return "unknown"


def get_code_version_info() -> dict:
    script_path = Path(__file__).resolve()
    engine_path = Path(engine_mod.__file__).resolve()
    git_commit = "unknown"
    git_dirty = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=script_path.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=script_path.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_dirty = bool(status_out)
    except Exception:
        pass

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runner_file": script_path.name,
        "runner_sha256": file_sha256_short(script_path),
        "engine_file": engine_path.name,
        "engine_sha256": file_sha256_short(engine_path),
    }


def make_run_id(mode: str, underlying: str, period: str) -> str:
    ts_part = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    tag = "paper" if mode == "paper" else "bt"
    return f"{ts_part}_{tag}_{underlying.lower()}_{period}_{short_uid}"


def resolve_output_dir(out_dir: str, out_root: str, run_id: str, subdir_by_run_id: bool) -> Path:
    """
    兼容路径解析：
    - 若显式传入 --out-root，则写到 out_root/run_id/
    - 若开启 --subdir-by-run-id，则写到 out_dir/run_id/
    - 默认直接写到 --out-dir（paper_logs / bt_exports），确保 outcome_review.py 路径不断
    """
    if out_root:
        target = Path(out_root) / run_id
    elif subdir_by_run_id:
        target = Path(out_dir) / run_id
    else:
        target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def calc_config_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def tau_bucket_name(tau_sec: float) -> str:
    if tau_sec <= 60.0:
        return "0-60s"
    if tau_sec <= 180.0:
        return "60-180s"
    if tau_sec <= 300.0:
        return "180-300s"
    if tau_sec <= 600.0:
        return "300-600s"
    return "600s+"


def fmt_opt(val: Optional[float], digits: int = 5):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return ""
    return round(val, digits)


def calc_bps(px: Optional[float], strike: float) -> Optional[float]:
    if px is None or strike <= 0:
        return None
    return (px - strike) / strike * 10000.0


class LiveCSVLogger:
    """逐行实时写入并立即 flush，防崩溃丢数据；默认新建覆盖避免旧表头冲突"""
    def __init__(self, path: Path, fieldnames: List[str], append: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        need_header = (not append) or (not self.path.exists()) or (self.path.stat().st_size == 0)
        self._fp = open(self.path, mode, newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames, extrasaction="ignore")
        if need_header:
            self._writer.writeheader()
            self._fp.flush()

    def append(self, row: dict):
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fp.flush()

    def close(self):
        self._fp.close()


def write_csv_snapshot(path: Path, fieldnames: List[str], rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ============================================================
# 3. 决策采样节流器
# ============================================================

class DecisionSampler:
    """按状态变化、时间段切换、接近门槛、定期心跳四类触发器节流落盘未开仓日志"""
    def __init__(
        self,
        sample_interval_sec: float = 30.0,
        near_edge_margin: float = 0.015,
        near_throttle_sec: float = 5.0,
    ):
        self.sample_interval_sec = sample_interval_sec
        self.near_edge_margin = near_edge_margin
        self.near_throttle_sec = near_throttle_sec
        self._last_periodic_ts: Dict[str, float] = {}
        self._last_near_ts: Dict[str, float] = {}
        self._last_reason: Dict[str, str] = {}
        self._last_bucket: Dict[str, str] = {}

    def check_trigger(self, dec: EvaluationDecision) -> Optional[str]:
        oid = dec.outcome_id
        ts = dec.ts
        bucket = tau_bucket_name(dec.tau_sec)

        if dec.should_trade:
            self._last_reason[oid] = dec.filter_reason
            self._last_bucket[oid] = bucket
            return "SIGNAL"

        if self._last_reason.get(oid) != dec.filter_reason:
            self._last_reason[oid] = dec.filter_reason
            self._last_bucket[oid] = bucket
            self._last_periodic_ts.setdefault(oid, ts)
            return "REASON_CHANGE"

        if self._last_bucket.get(oid) != bucket:
            self._last_bucket[oid] = bucket
            self._last_periodic_ts[oid] = ts
            return "TAU_BUCKET"

        if (
            dec.best_edge is not None
            and dec.best_edge >= dec.active_min_edge - self.near_edge_margin
            and (ts - self._last_near_ts.get(oid, -1e9)) >= self.near_throttle_sec
        ):
            self._last_near_ts[oid] = ts
            return "NEAR_THRESHOLD"

        if (ts - self._last_periodic_ts.get(oid, -1e9)) >= self.sample_interval_sec:
            self._last_periodic_ts[oid] = ts
            return "PERIODIC"

        return None


# ============================================================
# 4. CSV 表头（兼容 outcome_review.py + 补齐全部诊断字段）
# ============================================================

DECISION_FIELDS = [
    "run_id", "ts_utc", "outcome_id", "underlying", "period",
    "sample_trigger", "filter_reason", "tau_bucket", "tau_sec",
    "strike", "mark_px", "acc_twap", "ann_vol", "z_score",
    "p_up", "p_up_normal", "up_bid", "up_ask", "down_bid", "down_ask",
    "edge_up", "edge_down", "best_side", "best_edge",
    "active_min_edge", "gamma_penalty_applied",
    "book_age_ms", "book_stale",
    "equity_at_signal", "market_trades_used_before",
    "kelly_full", "kelly_scaled", "target_risk_pct", "kelly_capped", "target_usdc",
]

PAPER_ORDER_FIELDS = [
    "run_id", "order_id", "signal_time_utc", "exec_time_utc",
    "latency_target_ms", "latency_actual_ms",
    "outcome_id", "underlying", "period", "side", "status", "reject_reason",
    "tau_bucket", "tau_sec", "strike", "ann_vol", "z_score", "fair_prob",
    "active_min_edge", "gamma_penalty_applied", "expected_edge", "eff_settle_fee",
    # 仓位快照与成交后即时净值
    "equity_at_signal", "equity_after_fill", "market_trades_used_before",
    "kelly_full", "kelly_scaled", "target_risk_pct", "kelly_capped",
    "target_usdc", "actual_risk_pct",
    # 信号时盘口 & 标的（保留旧字段别名兼容）
    "mark_px_signal", "sig_mark_px", "acc_twap_signal", "sig_acc_twap",
    "top_ask_signal", "sig_side_bid", "sig_side_ask", "sig_side_mid", "sig_side_spread", "sig_book_age_ms",
    # 成交时盘口 & 标的 & 深度
    "exec_mark_px", "exec_side_bid", "exec_side_ask", "exec_side_mid", "exec_side_spread",
    "exec_book_age_ms", "exec_book_stale",
    "exec_top_ask_usdc", "exec_total_avail_usdc", "exec_depth_covers_target",
    "exec_levels_consumed", "exec_worst_fill_px", "exec_fill_ratio",
    # 成交结果与滑点拆解
    "filled_shares", "fill_vwap", "cost_usdc",
    "slippage", "slip_vs_sig_mid", "slip_vs_exec_mid", "slip_vs_sig_ask",
    "slip_book_walk", "slip_latency_drift", "realized_edge",
]

SETTLE_FIELDS = [
    "run_id", "order_id", "settle_time_utc", "signal_utc", "exec_utc",
    "outcome_id", "underlying", "period", "side",
    "tau_bucket", "tau_sec", "z_score", "ann_vol",
    "fair_prob", "active_min_edge", "gamma_penalty_applied",
    # 结算距离与边界标的价格
    "strike", "mark_px_signal", "mark_px_entry", "mark_px_expiry",
    "entry_dist_bps", "expiry_dist_bps",
    "settled_twap", "twap_minus_strike", "twap_dist_bps", "signed_settle_margin_bps",
    "outcome_up", "won",
    # 仓位快照与账务
    "equity_at_signal", "equity_after_fill", "market_trades_used_before",
    "kelly_full", "kelly_scaled", "target_risk_pct", "kelly_capped", "actual_risk_pct",
    "shares", "filled_shares", "top_ask_signal", "fill_vwap", "cost_usdc",
    "eff_settle_fee", "payout_usdc", "pnl_usdc", "equity_after",
    # 信号与成交盘口及三段滑点
    "sig_side_bid", "sig_side_ask", "sig_side_mid",
    "exec_side_bid", "exec_side_ask", "exec_side_mid",
    "exec_book_age_ms", "exec_book_stale", "exec_levels_consumed", "exec_depth_covers_target",
    "slippage", "slip_vs_sig_mid", "slip_vs_exec_mid", "slip_vs_sig_ask",
    "slip_book_walk", "slip_latency_drift",
]

OPEN_POS_FIELDS = [
    "run_id", "snapshot_time_utc", "open_reason",
    "order_id", "outcome_id", "underlying", "period", "side",
    "expiry_utc", "strike", "signal_time_utc", "exec_time_utc",
    "tau_sec_at_signal", "fair_prob", "active_min_edge",
    "filled_shares", "fill_vwap", "cost_usdc", "eff_settle_fee",
    "equity_at_signal", "equity_after_fill", "actual_risk_pct",
    "mark_px_signal", "mark_px_entry", "last_mark_px",
    "slip_vs_sig_mid", "slip_vs_sig_ask", "slip_book_walk",
]

BT_REJECTED_FIELDS = [
    "run_id", "order_id", "outcome_id", "underlying", "period",
    "signal_utc", "exec_utc", "side", "reason", "reject_reason",
    "tau_bucket", "tau_sec", "strike", "fair_prob", "active_min_edge",
    "top_ask", "delayed_vwap", "slippage", "slip_vs_sig_mid", "slip_book_walk", "slip_latency_drift",
    "sig_mark_px", "exec_mark_px",
    "sig_side_bid", "sig_side_ask", "sig_side_mid",
    "exec_side_bid", "exec_side_ask", "exec_side_mid",
    "exec_book_age_ms", "exec_book_stale", "exec_total_avail_usdc",
    "equity_at_signal", "kelly_full", "target_risk_pct", "target_usdc",
    "outcome_up",
]

BT_CALIBRATION_FIELDS = [
    "run_id", "outcome_id", "underlying", "ts_utc", "tau_sec",
    "mark_px", "strike", "ann_vol", "z_score",
    "p_up_student_t", "p_up_normal", "outcome_up",
]

OUTCOME_FIELDS = [
    "run_id", "outcome_id", "underlying", "period", "expiry_utc",
    "strike", "settled_twap", "twap_minus_strike", "twap_dist_bps",
    "mark_px_expiry", "outcome_up", "had_filled_trade",
]


def decision_to_row(run_id: str, market: OutcomeMarket, dec: EvaluationDecision, trigger: str) -> dict:
    return {
        "run_id": run_id,
        "ts_utc": utc_iso(dec.ts),
        "outcome_id": market.outcome_id,
        "underlying": market.underlying,
        "period": market.period,
        "sample_trigger": trigger,
        "filter_reason": dec.filter_reason,
        "tau_bucket": tau_bucket_name(dec.tau_sec),
        "tau_sec": round(dec.tau_sec, 1),
        "strike": market.strike,
        "mark_px": round(dec.mark_px, 4),
        "acc_twap": fmt_opt(dec.acc_twap, 4),
        "ann_vol": round(dec.ann_vol, 4),
        "z_score": round(dec.z_score, 4),
        "p_up": round(dec.p_up, 5),
        "p_up_normal": round(dec.p_up_normal, 5),
        "up_bid": fmt_opt(dec.up_bid, 5),
        "up_ask": fmt_opt(dec.up_ask, 5),
        "down_bid": fmt_opt(dec.down_bid, 5),
        "down_ask": fmt_opt(dec.down_ask, 5),
        "edge_up": fmt_opt(dec.edge_up, 5),
        "edge_down": fmt_opt(dec.edge_down, 5),
        "best_side": dec.best_side,
        "best_edge": fmt_opt(dec.best_edge, 5),
        "active_min_edge": round(dec.active_min_edge, 5),
        "gamma_penalty_applied": round(dec.gamma_penalty_applied, 5),
        "book_age_ms": round(dec.book_age_ms, 1),
        "book_stale": int(dec.book_stale),
        "equity_at_signal": round(dec.equity_at_signal, 4),
        "market_trades_used_before": dec.market_trades_used_before,
        "kelly_full": fmt_opt(dec.kelly_full, 5),
        "kelly_scaled": fmt_opt(dec.kelly_scaled, 5),
        "target_risk_pct": fmt_opt(dec.target_risk_pct, 5),
        "kelly_capped": int(dec.kelly_capped) if dec.kelly_capped is not None else "",
        "target_usdc": fmt_opt(dec.target_usdc, 2),
    }


def build_order_row(
    run_id: str,
    order_id: str,
    market: OutcomeMarket,
    dec: EvaluationDecision,
    sig_q: SideQuote,
    exec_ts: float,
    latency_target_ms: int,
    exec_q: Optional[SideQuote],
    exec_mark_px: Optional[float],
    exec_book_age_ms: Optional[float],
    exec_book_stale: Optional[bool],
    fill: Optional[DetailedFill],
    status: str,
    reject_reason: str,
    equity_after_fill: Optional[float],
) -> dict:
    slip = (
        decompose_slippage(fill.vwap, sig_q, exec_q)
        if (fill is not None and exec_q is not None)
        else {}
    )
    actual_risk = (
        (fill.cost_usdc / dec.equity_at_signal)
        if (fill is not None and status == "FILLED" and dec.equity_at_signal > 0)
        else 0.0
    )
    realized_edge = (
        (dec.fair_prob * (1.0 - dec.eff_settle_fee) - fill.vwap)
        if (fill is not None and dec.fair_prob is not None and status == "FILLED")
        else None
    )

    return {
        "run_id": run_id,
        "order_id": order_id,
        "signal_time_utc": utc_iso(dec.ts),
        "exec_time_utc": utc_iso(exec_ts),
        "latency_target_ms": latency_target_ms,
        "latency_actual_ms": round((exec_ts - dec.ts) * 1000.0, 1),
        "outcome_id": market.outcome_id,
        "underlying": market.underlying,
        "period": market.period,
        "side": dec.side,
        "status": status,
        "reject_reason": reject_reason,
        "tau_bucket": tau_bucket_name(dec.tau_sec),
        "tau_sec": round(dec.tau_sec, 1),
        "strike": market.strike,
        "ann_vol": round(dec.ann_vol, 4),
        "z_score": round(dec.z_score, 4),
        "fair_prob": fmt_opt(dec.fair_prob, 5),
        "active_min_edge": round(dec.active_min_edge, 5),
        "gamma_penalty_applied": round(dec.gamma_penalty_applied, 5),
        "expected_edge": fmt_opt(dec.expected_edge, 5),
        "eff_settle_fee": round(dec.eff_settle_fee, 6),
        "equity_at_signal": round(dec.equity_at_signal, 4),
        "equity_after_fill": fmt_opt(equity_after_fill, 4),
        "market_trades_used_before": dec.market_trades_used_before,
        "kelly_full": fmt_opt(dec.kelly_full, 5),
        "kelly_scaled": fmt_opt(dec.kelly_scaled, 5),
        "target_risk_pct": fmt_opt(dec.target_risk_pct, 5),
        "kelly_capped": int(dec.kelly_capped) if dec.kelly_capped is not None else "",
        "target_usdc": fmt_opt(dec.target_usdc, 2),
        "actual_risk_pct": round(actual_risk, 5),
        "mark_px_signal": round(dec.mark_px, 4),
        "sig_mark_px": round(dec.mark_px, 4),
        "acc_twap_signal": fmt_opt(dec.acc_twap, 4),
        "sig_acc_twap": fmt_opt(dec.acc_twap, 4),
        "top_ask_signal": fmt_opt(sig_q.ask, 5),
        "sig_side_bid": fmt_opt(sig_q.bid, 5),
        "sig_side_ask": fmt_opt(sig_q.ask, 5),
        "sig_side_mid": fmt_opt(sig_q.mid, 5),
        "sig_side_spread": fmt_opt(sig_q.spread, 5),
        "sig_book_age_ms": round(dec.book_age_ms, 1),
        "exec_mark_px": fmt_opt(exec_mark_px, 4),
        "exec_side_bid": fmt_opt(exec_q.bid if exec_q else None, 5),
        "exec_side_ask": fmt_opt(exec_q.ask if exec_q else None, 5),
        "exec_side_mid": fmt_opt(exec_q.mid if exec_q else None, 5),
        "exec_side_spread": fmt_opt(exec_q.spread if exec_q else None, 5),
        "exec_book_age_ms": fmt_opt(exec_book_age_ms, 1),
        "exec_book_stale": int(exec_book_stale) if exec_book_stale is not None else "",
        "exec_top_ask_usdc": fmt_opt(fill.top_ask_usdc if fill else None, 2),
        "exec_total_avail_usdc": fmt_opt(fill.total_avail_usdc if fill else None, 2),
        "exec_depth_covers_target": int(fill.depth_covers_target) if fill else "",
        "exec_levels_consumed": fill.levels_consumed if fill else "",
        "exec_worst_fill_px": fmt_opt(fill.worst_fill_px if fill else None, 5),
        "exec_fill_ratio": fmt_opt(fill.fill_ratio if fill else None, 4),
        "filled_shares": fill.filled_shares if (fill and status == "FILLED") else 0,
        "fill_vwap": fmt_opt(fill.vwap if fill else None, 5),
        "cost_usdc": round(fill.cost_usdc, 4) if (fill and status == "FILLED") else 0.0,
        "slippage": fmt_opt(slip.get("slip_vs_sig_ask"), 5),
        "slip_vs_sig_mid": fmt_opt(slip.get("slip_vs_sig_mid"), 5),
        "slip_vs_exec_mid": fmt_opt(slip.get("slip_vs_exec_mid"), 5),
        "slip_vs_sig_ask": fmt_opt(slip.get("slip_vs_sig_ask"), 5),
        "slip_book_walk": fmt_opt(slip.get("slip_book_walk"), 5),
        "slip_latency_drift": fmt_opt(slip.get("slip_latency_drift"), 5),
        "realized_edge": fmt_opt(realized_edge, 5),
    }


def build_settlement_row(
    run_id: str,
    order_row: dict,
    settle_ts: float,
    settled_twap: float,
    mark_px_expiry: Optional[float],
    outcome_up: int,
    equity_after: float,
    payout: float,
    pnl: float,
    won: bool,
) -> dict:
    strike = float(order_row["strike"])
    twap_minus_strike = settled_twap - strike
    twap_dist_bps = calc_bps(settled_twap, strike)
    signed_margin_bps = (
        twap_dist_bps
        if order_row["side"] == "UP"
        else (-twap_dist_bps if twap_dist_bps is not None else None)
    )
    entry_px = order_row["exec_mark_px"] if order_row["exec_mark_px"] != "" else None
    entry_dist_bps = calc_bps(float(entry_px), strike) if entry_px is not None else None
    expiry_dist_bps = calc_bps(mark_px_expiry, strike)

    return {
        "run_id": run_id,
        "order_id": order_row["order_id"],
        "settle_time_utc": utc_iso(settle_ts),
        "signal_utc": order_row["signal_time_utc"],
        "exec_utc": order_row["exec_time_utc"],
        "outcome_id": order_row["outcome_id"],
        "underlying": order_row["underlying"],
        "period": order_row["period"],
        "side": order_row["side"],
        "tau_bucket": order_row["tau_bucket"],
        "tau_sec": order_row["tau_sec"],
        "z_score": order_row["z_score"],
        "ann_vol": order_row["ann_vol"],
        "fair_prob": order_row["fair_prob"],
        "active_min_edge": order_row["active_min_edge"],
        "gamma_penalty_applied": order_row["gamma_penalty_applied"],
        "strike": strike,
        "mark_px_signal": order_row["sig_mark_px"],
        "mark_px_entry": order_row["exec_mark_px"],
        "mark_px_expiry": fmt_opt(mark_px_expiry, 4),
        "entry_dist_bps": fmt_opt(entry_dist_bps, 2),
        "expiry_dist_bps": fmt_opt(expiry_dist_bps, 2),
        "settled_twap": round(settled_twap, 4),
        "twap_minus_strike": round(twap_minus_strike, 4),
        "twap_dist_bps": fmt_opt(twap_dist_bps, 2),
        "signed_settle_margin_bps": fmt_opt(signed_margin_bps, 2),
        "outcome_up": outcome_up,
        "won": int(won),
        "equity_at_signal": order_row["equity_at_signal"],
        "equity_after_fill": order_row["equity_after_fill"],
        "market_trades_used_before": order_row["market_trades_used_before"],
        "kelly_full": order_row["kelly_full"],
        "kelly_scaled": order_row["kelly_scaled"],
        "target_risk_pct": order_row["target_risk_pct"],
        "kelly_capped": order_row["kelly_capped"],
        "actual_risk_pct": order_row["actual_risk_pct"],
        "shares": order_row["filled_shares"],
        "filled_shares": order_row["filled_shares"],
        "top_ask_signal": order_row["sig_side_ask"],
        "fill_vwap": order_row["fill_vwap"],
        "cost_usdc": order_row["cost_usdc"],
        "eff_settle_fee": order_row["eff_settle_fee"],
        "payout_usdc": round(payout, 4),
        "pnl_usdc": round(pnl, 4),
        "equity_after": round(equity_after, 4),
        "sig_side_bid": order_row["sig_side_bid"],
        "sig_side_ask": order_row["sig_side_ask"],
        "sig_side_mid": order_row["sig_side_mid"],
        "exec_side_bid": order_row["exec_side_bid"],
        "exec_side_ask": order_row["exec_side_ask"],
        "exec_side_mid": order_row["exec_side_mid"],
        "exec_book_age_ms": order_row["exec_book_age_ms"],
        "exec_book_stale": order_row["exec_book_stale"],
        "exec_levels_consumed": order_row["exec_levels_consumed"],
        "exec_depth_covers_target": order_row["exec_depth_covers_target"],
        "slippage": order_row["slippage"],
        "slip_vs_sig_mid": order_row["slip_vs_sig_mid"],
        "slip_vs_exec_mid": order_row["slip_vs_exec_mid"],
        "slip_vs_sig_ask": order_row["slip_vs_sig_ask"],
        "slip_book_walk": order_row["slip_book_walk"],
        "slip_latency_drift": order_row["slip_latency_drift"],
    }


def write_summary_and_metadata(
    out_dir: Path,
    summary_filename: str,
    run_id: str,
    mode: str,
    network: str,
    underlying: str,
    period: str,
    start_ts: float,
    end_ts: float,
    cfg: StrategyConfig,
    exec_params: dict,
    bookkeeper: EquityBookkeeper,
    extra_metrics: dict,
    db_path: str = "",
) -> dict:
    cfg_dict = asdict(cfg)
    code_ver = get_code_version_info()
    cfg_hash = calc_config_hash({"strategy_config": cfg_dict, "exec_params": exec_params})

    # 顶层保留 initial_equity 等 outcome_review.py 依赖字段 + 完整运行元数据
    payload = {
        "run_id": run_id,
        "mode": mode,
        "network": network,
        "underlying": underlying,
        "period": period,
        "db_path": db_path,
        "start_time_utc": utc_iso(start_ts),
        "end_time_utc": utc_iso(end_ts),
        "finished_at_utc": utc_iso(end_ts),
        "duration_sec": round(end_ts - start_ts, 2),
        "initial_equity": cfg.initial_equity,
        "final_equity": round(bookkeeper.equity, 4),
        "peak_equity": round(bookkeeper.peak_equity, 4),
        "return_pct": round((bookkeeper.equity / cfg.initial_equity - 1.0) * 100.0, 3) if cfg.initial_equity > 0 else 0.0,
        "max_drawdown_pct": round(bookkeeper.max_dd * 100.0, 3),
        "bookkeeping_source": bookkeeper.source,
        "drawdown_definition": dict(DRAWDOWN_DEFINITION),
        "latency_ms": exec_params.get("latency_ms"),
        "max_slippage": exec_params.get("max_slippage"),
        "stale_book_ms": exec_params.get("stale_book_ms"),
        "settlement_fee_rate": cfg.settlement_fee_rate,
        "settlement_fee_note": SETTLEMENT_FEE_NOTE,
        **extra_metrics,
        "code_version": code_ver,
        "config_hash": cfg_hash,
        "exec_params": exec_params,
        "strategy_config": cfg_dict,
    }
    payload.setdefault("live_orders", False)

    with open(out_dir / summary_filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


# ============================================================
# 5. 实时模拟盘（经由共享采集层同步写 SQLite 磁带 + 优雅退出）
# ============================================================

async def run_paper_trading(
    network: str,
    underlying: str,
    period: str,
    cfg: StrategyConfig,
    latency_ms: int,
    max_slippage: float,
    stale_book_ms: float,
    decision_sample_sec: float,
    near_edge_margin: float,
    out_dir_arg: str,
    out_root_arg: str,
    subdir_by_run_id: bool,
    duration_min: float,
    poll_meta_sec: float = 20.0,
    db_path: str = "hip4_replay.sqlite",
):
    underlying = underlying.upper()
    start_ts = time.time()
    stop_ts = start_ts + duration_min * 60.0
    run_id = make_run_id("paper", underlying, period)
    out_dir = resolve_output_dir(out_dir_arg, out_root_arg, run_id, subdir_by_run_id)

    # 同时保留 outcome_review.py 所需文件 + 新增审计文件 + 同 schema SQLite 磁带
    order_logger = LiveCSVLogger(out_dir / "paper_orders.csv", PAPER_ORDER_FIELDS)
    settle_logger = LiveCSVLogger(out_dir / "paper_settlements.csv", SETTLE_FIELDS)
    decision_logger = LiveCSVLogger(out_dir / "decisions.csv", DECISION_FIELDS)
    outcome_logger = LiveCSVLogger(out_dir / "market_outcomes.csv", OUTCOME_FIELDS)
    tape_writer = MarketTapeWriter(db_path)

    bookkeeper = EquityBookkeeper(cfg.initial_equity)
    strat = HIP4AsianBinaryStrategy(cfg, stale_book_ms=stale_book_ms)
    sampler = DecisionSampler(
        sample_interval_sec=decision_sample_sec,
        near_edge_margin=near_edge_margin,
    )
    vol_est = RollingMarkVol(cfg.ewma_lambda, cfg.min_ann_vol)

    ws_url = ENDPOINTS[network]["ws"]
    order_seq = 0
    stop_requested = False

    tracked_markets: Dict[str, OutcomeMarket] = {}
    settled_market_ids: set[str] = set()
    traded_market_ids: set[str] = set()
    latest_books: Dict[str, Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float]] = {}
    mark_history: List[Tuple[float, float]] = []
    latest_mark_px: Optional[float] = None
    ann_vol = cfg.min_ann_vol

    open_positions: List[dict] = []
    settled_records: List[dict] = []
    decision_counts: Dict[str, int] = {}
    order_status_counts: Dict[str, int] = {}
    reject_cooldown_until: Dict[str, float] = {}
    pending_order_tasks: set[asyncio.Task] = set()

    # 注册优雅 SIGINT 处理器：第一次 Ctrl+C 设标志位，避免在途延迟下单任务被直接强杀
    prev_sigint_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            print("\n[Paper] 收到退出信号，停止接新单，等待在途延迟单与结算收尾（再按一次 Ctrl+C 强制退出）...")
        else:
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    async def execute_delayed_order(oid_num: int, market: OutcomeMarket, dec: EvaluationDecision, sig_q: SideQuote):
        order_id = f"PAPER-{oid_num:05d}"
        try:
            await asyncio.sleep(latency_ms / 1000.0)
            exec_ts = time.time()
            book = latest_books.get(market.outcome_id)
            exec_mark = latest_mark_px

            if not book:
                row = build_order_row(
                    run_id, order_id, market, dec, sig_q, exec_ts, latency_ms,
                    None, exec_mark, None, None, None, "REJECTED", "NO_BOOK", None,
                )
                order_logger.append(row)
                order_status_counts["REJECTED:NO_BOOK"] = order_status_counts.get("REJECTED:NO_BOOK", 0) + 1
                reject_cooldown_until[market.outcome_id] = exec_ts + REJECT_COOLDOWN_SEC
                return

            yes_bids, yes_asks, exec_book_ts = book
            exec_q = extract_side_quote(dec.side, yes_bids, yes_asks)
            exec_book_age_ms = max(0.0, (exec_ts - exec_book_ts) * 1000.0)
            exec_book_stale = exec_book_age_ms > stale_book_ms

            if exec_book_stale:
                row = build_order_row(
                    run_id, order_id, market, dec, sig_q, exec_ts, latency_ms,
                    exec_q, exec_mark, exec_book_age_ms, exec_book_stale, None, "REJECTED", "BOOK_STALE", None,
                )
                order_logger.append(row)
                order_status_counts["REJECTED:BOOK_STALE"] = order_status_counts.get("REJECTED:BOOK_STALE", 0) + 1
                reject_cooldown_until[market.outcome_id] = exec_ts + REJECT_COOLDOWN_SEC
                return

            fill = simulate_clob_buy_fill_detailed(
                side=dec.side,
                target_usdc=dec.target_usdc,
                yes_bids=yes_bids,
                yes_asks=yes_asks,
                min_notional=cfg.min_order_notional,
            )
            if fill is None:
                row = build_order_row(
                    run_id, order_id, market, dec, sig_q, exec_ts, latency_ms,
                    exec_q, exec_mark, exec_book_age_ms, exec_book_stale, None, "REJECTED", "NO_LIQUIDITY", None,
                )
                order_logger.append(row)
                order_status_counts["REJECTED:NO_LIQUIDITY"] = order_status_counts.get("REJECTED:NO_LIQUIDITY", 0) + 1
                reject_cooldown_until[market.outcome_id] = exec_ts + REJECT_COOLDOWN_SEC
                return

            if fill.cost_usdc > bookkeeper.equity:
                row = build_order_row(
                    run_id, order_id, market, dec, sig_q, exec_ts, latency_ms,
                    exec_q, exec_mark, exec_book_age_ms, exec_book_stale, fill, "REJECTED", "INSUFFICIENT_EQUITY", None,
                )
                order_logger.append(row)
                order_status_counts["REJECTED:INSUFFICIENT_EQUITY"] = (
                    order_status_counts.get("REJECTED:INSUFFICIENT_EQUITY", 0) + 1
                )
                reject_cooldown_until[market.outcome_id] = exec_ts + REJECT_COOLDOWN_SEC
                return

            max_limit_px = min(cfg.max_entry_price, dec.top_ask + max_slippage)
            if fill.vwap > max_limit_px:
                row = build_order_row(
                    run_id, order_id, market, dec, sig_q, exec_ts, latency_ms,
                    exec_q, exec_mark, exec_book_age_ms, exec_book_stale, fill, "REJECTED", "SLIPPAGE_EXCEEDED", None,
                )
                order_logger.append(row)
                order_status_counts["REJECTED:SLIPPAGE_EXCEEDED"] = (
                    order_status_counts.get("REJECTED:SLIPPAGE_EXCEEDED", 0) + 1
                )
                reject_cooldown_until[market.outcome_id] = exec_ts + REJECT_COOLDOWN_SEC
                print(
                    f"[Paper] {order_id} #{market.outcome_id} {dec.side} IOC拒单 | "
                    f"SigAsk={dec.top_ask:.4f} -> VWAP={fill.vwap:.4f} (Walk={row['slip_book_walk']}, Drift={row['slip_latency_drift']})"
                )
                return

            # 成交立即扣减本金，刷新峰值与最大回撤，且只有 FILLED 才占用单场额度
            eq_after_fill, peak_after_fill, _ = bookkeeper.apply_fill(fill.cost_usdc)
            strat.sync_drawdown_state(eq_after_fill, peak_after_fill, ts=exec_ts)

            row = build_order_row(
                run_id, order_id, market, dec, sig_q, exec_ts, latency_ms,
                exec_q, exec_mark, exec_book_age_ms, exec_book_stale, fill, "FILLED", "", eq_after_fill,
            )
            order_logger.append(row)
            order_status_counts["FILLED"] = order_status_counts.get("FILLED", 0) + 1
            strat.record_fill(market.outcome_id)
            traded_market_ids.add(market.outcome_id)

            open_positions.append({
                "order_row": row,
                "expiry_ts": market.expiry_ts,
                "eff_settle_fee": dec.eff_settle_fee,
            })
            print(
                f"[Paper] 成交 {order_id} | #{market.outcome_id} {dec.side} {fill.filled_shares}股@{fill.vwap:.4f} "
                f"(扣本金 ${fill.cost_usdc:.2f} -> 净值 {eq_after_fill:.2f} | "
                f"vsMid={row['slip_vs_sig_mid']}, Walk={row['slip_book_walk']})"
            )
        finally:
            strat.in_flight_markets.discard(market.outcome_id)

    def check_settlements():
        now = time.time()
        cutoff = now - 2400.0
        while mark_history and mark_history[0][0] < cutoff:
            mark_history.pop(0)

        for oid, m in list(tracked_markets.items()):
            if oid in settled_market_ids or now < m.expiry_ts + 2.0:
                continue
            twap = compute_strict_twap(mark_history, m.expiry_ts - 60.0, m.expiry_ts)
            if twap is None:
                continue

            pre_exp = [px for t, px in mark_history if t < m.expiry_ts]
            mark_exp = pre_exp[-1] if pre_exp else None
            outcome_up = 1 if twap > m.strike else 0
            settled_market_ids.add(oid)

            outcome_logger.append({
                "run_id": run_id,
                "outcome_id": oid,
                "underlying": m.underlying,
                "period": m.period,
                "expiry_utc": utc_iso(m.expiry_ts),
                "strike": m.strike,
                "settled_twap": round(twap, 4),
                "twap_minus_strike": round(twap - m.strike, 4),
                "twap_dist_bps": fmt_opt(calc_bps(twap, m.strike), 2),
                "mark_px_expiry": fmt_opt(mark_exp, 4),
                "outcome_up": outcome_up,
                "had_filled_trade": int(oid in traded_market_ids),
            })

            remaining = []
            for pos in open_positions:
                orow = pos["order_row"]
                if orow["outcome_id"] != oid:
                    remaining.append(pos)
                    continue
                won = (orow["side"] == "UP" and outcome_up == 1) or (orow["side"] == "DOWN" and outcome_up == 0)
                payout = orow["filled_shares"] * (1.0 - pos["eff_settle_fee"]) if won else 0.0
                pnl = payout - float(orow["cost_usdc"])

                # 结算只加 payout_usdc，不再重复扣减 cost_usdc
                eq_after_payout, peak_after_payout, _ = bookkeeper.apply_payout(payout)
                strat.sync_drawdown_state(eq_after_payout, peak_after_payout, ts=now)

                srow = build_settlement_row(
                    run_id, orow, now, twap, mark_exp, outcome_up, eq_after_payout, payout, pnl, won
                )
                settle_logger.append(srow)
                settled_records.append(srow)
                print(
                    f"[Settle] {orow['order_id']} #{oid} | TWAP={twap:.2f} vs Strike={m.strike:.2f} "
                    f"(Margin={srow['signed_settle_margin_bps']}bps) -> {'WIN' if won else 'LOSS'} | "
                    f"Payout={payout:.2f} | PnL={pnl:+.2f} | Eq={eq_after_payout:.2f}"
                )
            open_positions[:] = remaining

    def dump_open_positions(snap_ts: float) -> List[dict]:
        rows = []
        for pos in open_positions:
            orow = pos["order_row"]
            exp_ts = float(pos["expiry_ts"])
            reason = "NOT_EXPIRED" if snap_ts < exp_ts + 2.0 else "TWAP_UNAVAILABLE"
            rows.append({
                "run_id": run_id,
                "snapshot_time_utc": utc_iso(snap_ts),
                "open_reason": reason,
                "order_id": orow["order_id"],
                "outcome_id": orow["outcome_id"],
                "underlying": orow["underlying"],
                "period": orow["period"],
                "side": orow["side"],
                "expiry_utc": utc_iso(exp_ts),
                "strike": orow["strike"],
                "signal_time_utc": orow["signal_time_utc"],
                "exec_time_utc": orow["exec_time_utc"],
                "tau_sec_at_signal": orow["tau_sec"],
                "fair_prob": orow["fair_prob"],
                "active_min_edge": orow["active_min_edge"],
                "filled_shares": orow["filled_shares"],
                "fill_vwap": orow["fill_vwap"],
                "cost_usdc": orow["cost_usdc"],
                "eff_settle_fee": orow["eff_settle_fee"],
                "equity_at_signal": orow["equity_at_signal"],
                "equity_after_fill": orow["equity_after_fill"],
                "actual_risk_pct": orow["actual_risk_pct"],
                "mark_px_signal": orow["sig_mark_px"],
                "mark_px_entry": orow["exec_mark_px"],
                "last_mark_px": fmt_opt(latest_mark_px, 4),
                "slip_vs_sig_mid": orow["slip_vs_sig_mid"],
                "slip_vs_sig_ask": orow["slip_vs_sig_ask"],
                "slip_book_walk": orow["slip_book_walk"],
            })
        write_csv_snapshot(out_dir / "paper_open_positions.csv", OPEN_POS_FIELDS, rows)
        return rows

    def handle_markets(markets: List[OutcomeMarket], _newly_subscribed: List[OutcomeMarket]):
        for m in markets:
            tracked_markets[m.outcome_id] = m

    def handle_mark(ev: MarkEvent):
        nonlocal latest_mark_px, ann_vol
        latest_mark_px = ev.mark_px
        mark_history.append((ev.recv_ts, ev.mark_px))
        ann_vol = vol_est.update(ev.recv_ts, ev.mark_px)

    def handle_book(ev: BookEvent):
        nonlocal order_seq
        oid = ev.outcome_id
        # 与现码一致：REST 补快照仅刷新内存盘口，不直接触发下单评估
        if ev.source == "rest":
            latest_books[oid] = (ev.bids, ev.asks, ev.recv_ts)
            return

        t_now = ev.recv_ts
        # 保留既有 paper 内存盘口时间约定：有交易所时间时取 min(exchange_ts, recv_ts)
        book_ts = min(ev.exchange_ts, t_now) if ev.exchange_ts is not None else t_now
        latest_books[oid] = (ev.bids, ev.asks, book_ts)

        market = tracked_markets.get(oid)
        if not market or latest_mark_px is None or oid in settled_market_ids or stop_requested:
            return

        twap_start = market.expiry_ts - 60.0
        acc_twap = (
            compute_strict_twap(mark_history, twap_start, t_now)
            if t_now > twap_start
            else None
        )
        dec = strat.evaluate_with_audit(
            market, t_now, latest_mark_px, ann_vol, ev.bids, ev.asks, book_ts,
            acc_twap, bookkeeper.equity, bookkeeper.peak_equity,
        )
        if dec.should_trade and t_now < reject_cooldown_until.get(oid, 0.0):
            dec.should_trade = False
            dec.filter_reason = "REJECT_COOLDOWN"

        trig = sampler.check_trigger(dec)
        if trig is not None:
            decision_logger.append(decision_to_row(run_id, market, dec, trig))
            decision_counts[dec.filter_reason] = decision_counts.get(dec.filter_reason, 0) + 1

        if dec.should_trade:
            order_seq += 1
            strat.in_flight_markets.add(oid)
            sig_q = extract_side_quote(dec.side, ev.bids, ev.asks)
            task = asyncio.create_task(execute_delayed_order(order_seq, market, dec, sig_q))
            pending_order_tasks.add(task)
            task.add_done_callback(pending_order_tasks.discard)

    def handle_tick(_now: float):
        check_settlements()

    def handle_disconnect(exc: Exception):
        print(f"[Paper] WS 异常 ({exc})，3 秒后重连...")

    feed = LiveMarketFeed(
        network=network,
        underlying=underlying,
        period=period,
        ws_url=ws_url,
        poll_meta_sec=poll_meta_sec,
        ping_interval_sec=25.0,
        recv_timeout_sec=1.0,
        reconnect_delay_sec=3.0,
        tape_writer=tape_writer,
        auto_settle_tape=True,
        fetch_markets_fn=fetch_outcome_markets,
        fetch_l2_book_fn=fetch_l2_book_rest,
        parse_book_fn=parse_book_levels,
        compute_twap_fn=compute_strict_twap,
    )

    print(
        f"[Paper] RunID={run_id} | {network} {underlying} {period} | "
        f"Bookkeeping={bookkeeper.source} | DB={db_path} | Output={out_dir.resolve()}"
    )

    try:
        await feed.run(
            stop_ts=stop_ts,
            should_stop=lambda: stop_requested,
            on_markets=handle_markets,
            on_mark=handle_mark,
            on_book=handle_book,
            on_tick=handle_tick,
            on_disconnect=handle_disconnect,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_requested = True
    finally:
        try:
            signal.signal(signal.SIGINT, prev_sigint_handler)
        except Exception:
            pass

        # 1) 先等待所有在途延迟下单任务写完订单
        if pending_order_tasks:
            drain_timeout = max(5.0, latency_ms / 1000.0 + 3.0)
            print(f"[Paper] 正在等待 {len(pending_order_tasks)} 笔在途延迟单完成写入...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*list(pending_order_tasks), return_exceptions=True),
                    timeout=drain_timeout,
                )
            except asyncio.TimeoutError:
                print(f"[Paper] 警告: 等待在途延迟单超时 ({drain_timeout:.1f}s)，继续执行结算收尾。")

        # 2) 再做一次到期结算，并同步回写 SQLite 磁带结算结果
        check_settlements()
        tape_writer.settle_expired_markets(compute_strict_twap)
        tape_writer.close()

        # 3) 将仍未到期或 TWAP 尚未出来的持仓写入 paper_open_positions.csv
        end_ts = time.time()
        open_rows = dump_open_positions(end_ts)
        open_cost_usdc = round(sum(float(r["cost_usdc"]) for r in open_rows), 4)

        shadow_metrics = compute_settlement_shadow_equity(
            cfg.initial_equity,
            [float(r["equity_after"]) for r in settled_records],
        )
        extra_metrics = {
            "settled_markets_total": len(settled_market_ids),
            "settled_trades": len(settled_records),
            "win_rate_pct": round(
                sum(r["won"] for r in settled_records) / len(settled_records) * 100.0, 2
            ) if settled_records else 0.0,
            "open_unsettled_positions": len(open_rows),
            "open_positions_cost_usdc": open_cost_usdc,
            "order_status_breakdown": order_status_counts,
            "decision_sample_breakdown": decision_counts,
            "halted_by_drawdown": strat.halted,
            **strat.halt_snapshot(),
            **shadow_metrics,
        }
        exec_params = {
            "latency_ms": latency_ms,
            "max_slippage": max_slippage,
            "stale_book_ms": stale_book_ms,
            "decision_sample_sec": decision_sample_sec,
            "near_edge_margin": near_edge_margin,
            "duration_min_target": duration_min,
            "db_path": db_path,
        }
        resolved_db = db_path if db_path == ":memory:" else str(Path(db_path).resolve())
        write_summary_and_metadata(
            out_dir, "paper_summary.json", run_id, "paper", network, underlying, period,
            start_ts, end_ts, cfg, exec_params, bookkeeper, extra_metrics,
            db_path=resolved_db,
        )
        order_logger.close()
        settle_logger.close()
        decision_logger.close()
        outcome_logger.close()
        print(
            f"[Paper] 退出收尾完成 | 目录: {out_dir.resolve()} | "
            f"已结算 {len(settled_records)} 笔 | 未平仓 {len(open_rows)} 笔 (已写入 paper_open_positions.csv)"
        )


# ============================================================
# 6. 唯一离线回放回测路径（export-backtest 与 engine backtest 共用）
# ============================================================

def validate_replay_db(db_path: str) -> Tuple[Optional[sqlite3.Connection], List[tuple]]:
    path = Path(db_path)
    if not path.exists():
        print(f"[ExportBT] 错误: 数据库文件不存在 -> {path.resolve()}")
        return None, []

    try:
        conn = sqlite3.connect(str(path))
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    except sqlite3.Error as exc:
        print(f"[ExportBT] 错误: 无法读取 SQLite 数据库 ({exc})")
        return None, []

    required_tables = {"markets", "mark_ticks", "book_snapshots"}
    missing = sorted(required_tables - tables)
    if missing:
        print(
            f"[ExportBT] 错误: 数据库 {path.name} 缺少必需表 {missing}（当前仅包含 {sorted(tables)}）。"
            f"请先使用 outcome_hip4_engine.py record 或 outcome_paper_export.py paper --db 录制数据。"
        )
        conn.close()
        return None, []

    total_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
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
        print(
            f"[ExportBT] 错误: 数据库 {path.name} 中共有 {total_markets} 个场次，但已结算场次（outcome_up 非空）为 0，无法回测。"
            f"请确认已通过 record 或 paper --db 录完至少一个完整到期窗口。"
        )
        conn.close()
        return None, []

    return conn, m_rows


def run_unified_replay_backtest(
    db_path: str,
    cfg: StrategyConfig,
    latency_ms: int,
    max_slippage: float,
    stale_book_ms: float,
    decision_sample_sec: float,
    near_edge_margin: float,
    out_dir_arg: str,
    out_root_arg: str,
    subdir_by_run_id: bool,
    network: str = "mainnet",
) -> Tuple[bool, Optional[dict]]:
    start_ts = time.time()
    conn, m_rows = validate_replay_db(db_path)
    if conn is None:
        return False, None

    underlyings = sorted({r[2] for r in m_rows})
    periods = sorted({r[5] for r in m_rows})
    u_tag = underlyings[0] if len(underlyings) == 1 else "MULTI"
    p_tag = periods[0] if len(periods) == 1 else "MULTI"

    run_id = make_run_id("backtest", u_tag, p_tag)
    out_dir = resolve_output_dir(out_dir_arg, out_root_arg, run_id, subdir_by_run_id)

    # 兼容 outcome_review.py 的四份文件 + 新增审计文件
    trades_logger = LiveCSVLogger(out_dir / "bt_trades.csv", SETTLE_FIELDS)
    rejected_logger = LiveCSVLogger(out_dir / "bt_rejected.csv", BT_REJECTED_FIELDS)
    calib_logger = LiveCSVLogger(out_dir / "bt_calibration.csv", BT_CALIBRATION_FIELDS)
    decision_logger = LiveCSVLogger(out_dir / "decisions.csv", DECISION_FIELDS)
    outcome_logger = LiveCSVLogger(out_dir / "market_outcomes.csv", OUTCOME_FIELDS)

    bookkeeper = EquityBookkeeper(cfg.initial_equity)
    strat = HIP4AsianBinaryStrategy(cfg, stale_book_ms=stale_book_ms)
    sampler = DecisionSampler(
        sample_interval_sec=decision_sample_sec,
        near_edge_margin=near_edge_margin,
    )
    vol_trackers: Dict[str, RollingMarkVol] = {}

    latency_sec = latency_ms / 1000.0
    order_seq = 0

    settled_records: List[dict] = []
    decision_counts: Dict[str, int] = {}
    order_status_counts: Dict[str, int] = {}
    calib_count = 0

    try:
        for row in m_rows:
            m = OutcomeMarket(
                outcome_id=row[0], name=row[1], underlying=row[2], strike=row[3],
                expiry_ts=row[4], period=row[5], fee_scale=row[6], yes_coin=row[7],
                no_coin=f"#{row[0]}1",
                yes_asset_id=100_000_000 + int(row[0]) * 10,
                no_asset_id=100_000_000 + int(row[0]) * 10 + 1,
                raw_description=row[8],
            )
            settled_twap, outcome_up = float(row[9]), int(row[10])
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

            mark_ts_arr = np.array([x[0] for x in mark_rows], dtype=float)
            book_ts_arr = np.array([b[0] for b in book_rows], dtype=float)
            pre_exp_marks = [px for t, px in mark_rows if t < m.expiry_ts]
            mark_px_expiry = pre_exp_marks[-1] if pre_exp_marks else None

            twap_start = m.expiry_ts - 60.0
            mark_idx = 0
            latest_mark_px: Optional[float] = None
            ann_vol = cfg.min_ann_vol
            twap_buffer: List[Tuple[float, float]] = []
            filled_orders_this_market: List[Tuple[dict, float]] = []
            reject_cooldown_ts = -1e9
            in_flight_until_ts = -1e9

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

                # 兼容 bt_calibration.csv 采样
                if b_idx % 5 == 0 and 10.0 <= tau_sec <= 860.0:
                    p_t, p_n, z_val = strat.fair_prob_up_twap(latest_mark_px, m.strike, tau_sec, ann_vol, acc_twap)
                    calib_logger.append({
                        "run_id": run_id,
                        "outcome_id": m.outcome_id,
                        "underlying": m.underlying,
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
                    calib_count += 1

                yes_bids = json.loads(bids_s)
                yes_asks = json.loads(asks_s)

                if bts < in_flight_until_ts:
                    strat.in_flight_markets.add(m.outcome_id)
                else:
                    strat.in_flight_markets.discard(m.outcome_id)

                dec = strat.evaluate_with_audit(
                    m, bts, latest_mark_px, ann_vol, yes_bids, yes_asks, bts,
                    acc_twap, bookkeeper.equity, bookkeeper.peak_equity,
                )
                if dec.should_trade and bts < reject_cooldown_ts:
                    dec.should_trade = False
                    dec.filter_reason = "REJECT_COOLDOWN"

                trig = sampler.check_trigger(dec)
                if trig is not None:
                    decision_logger.append(decision_to_row(run_id, m, dec, trig))
                    decision_counts[dec.filter_reason] = decision_counts.get(dec.filter_reason, 0) + 1

                if not dec.should_trade:
                    continue

                order_seq += 1
                order_id = f"BT-{order_seq:05d}"
                sig_q = extract_side_quote(dec.side, yes_bids, yes_asks)
                exec_target_ts = bts + latency_sec
                in_flight_until_ts = exec_target_ts

                exec_book_idx = max(0, int(np.searchsorted(book_ts_arr, exec_target_ts, side="right")) - 1)
                exec_book_ts, exec_bids_s, exec_asks_s = book_rows[exec_book_idx]
                exec_bids = json.loads(exec_bids_s)
                exec_asks = json.loads(exec_asks_s)
                exec_q = extract_side_quote(dec.side, exec_bids, exec_asks)

                exec_mark_idx = max(0, int(np.searchsorted(mark_ts_arr, exec_target_ts, side="right")) - 1)
                exec_mark_px = mark_rows[exec_mark_idx][1]
                exec_book_age_ms = max(0.0, (exec_target_ts - exec_book_ts) * 1000.0)
                exec_book_stale = exec_book_age_ms > stale_book_ms

                def log_bt_reject(reason: str, fill_obj: Optional[DetailedFill] = None):
                    slip = (
                        decompose_slippage(fill_obj.vwap, sig_q, exec_q)
                        if fill_obj is not None
                        else {}
                    )
                    rejected_logger.append({
                        "run_id": run_id,
                        "order_id": order_id,
                        "outcome_id": m.outcome_id,
                        "underlying": m.underlying,
                        "period": m.period,
                        "signal_utc": utc_iso(bts),
                        "exec_utc": utc_iso(exec_target_ts),
                        "side": dec.side,
                        "reason": reason,
                        "reject_reason": reason,
                        "tau_bucket": tau_bucket_name(dec.tau_sec),
                        "tau_sec": round(dec.tau_sec, 1),
                        "strike": m.strike,
                        "fair_prob": fmt_opt(dec.fair_prob, 5),
                        "active_min_edge": round(dec.active_min_edge, 5),
                        "top_ask": fmt_opt(sig_q.ask, 5),
                        "delayed_vwap": fmt_opt(fill_obj.vwap if fill_obj else None, 5),
                        "slippage": fmt_opt(slip.get("slip_vs_sig_ask"), 5),
                        "slip_vs_sig_mid": fmt_opt(slip.get("slip_vs_sig_mid"), 5),
                        "slip_book_walk": fmt_opt(slip.get("slip_book_walk"), 5),
                        "slip_latency_drift": fmt_opt(slip.get("slip_latency_drift"), 5),
                        "sig_mark_px": round(dec.mark_px, 4),
                        "exec_mark_px": round(exec_mark_px, 4),
                        "sig_side_bid": fmt_opt(sig_q.bid, 5),
                        "sig_side_ask": fmt_opt(sig_q.ask, 5),
                        "sig_side_mid": fmt_opt(sig_q.mid, 5),
                        "exec_side_bid": fmt_opt(exec_q.bid, 5),
                        "exec_side_ask": fmt_opt(exec_q.ask, 5),
                        "exec_side_mid": fmt_opt(exec_q.mid, 5),
                        "exec_book_age_ms": round(exec_book_age_ms, 1),
                        "exec_book_stale": int(exec_book_stale),
                        "exec_total_avail_usdc": fmt_opt(fill_obj.total_avail_usdc if fill_obj else None, 2),
                        "equity_at_signal": round(dec.equity_at_signal, 4),
                        "kelly_full": fmt_opt(dec.kelly_full, 5),
                        "target_risk_pct": fmt_opt(dec.target_risk_pct, 5),
                        "target_usdc": fmt_opt(dec.target_usdc, 2),
                        "outcome_up": outcome_up,
                    })
                    key = f"REJECTED:{reason}"
                    order_status_counts[key] = order_status_counts.get(key, 0) + 1

                if exec_book_stale:
                    log_bt_reject("BOOK_STALE")
                    reject_cooldown_ts = exec_target_ts + REJECT_COOLDOWN_SEC
                    continue

                fill = simulate_clob_buy_fill_detailed(
                    side=dec.side,
                    target_usdc=dec.target_usdc,
                    yes_bids=exec_bids,
                    yes_asks=exec_asks,
                    min_notional=cfg.min_order_notional,
                )
                if fill is None:
                    log_bt_reject("NO_LIQUIDITY")
                    reject_cooldown_ts = exec_target_ts + REJECT_COOLDOWN_SEC
                    continue

                if fill.cost_usdc > bookkeeper.equity:
                    log_bt_reject("INSUFFICIENT_EQUITY", fill)
                    reject_cooldown_ts = exec_target_ts + REJECT_COOLDOWN_SEC
                    continue

                if fill.vwap > min(cfg.max_entry_price, dec.top_ask + max_slippage):
                    log_bt_reject("SLIPPAGE_EXCEEDED", fill)
                    reject_cooldown_ts = exec_target_ts + REJECT_COOLDOWN_SEC
                    continue

                # 成交立即扣减本金、刷新峰值与最大回撤，且只有 FILLED 才占用单场额度
                eq_after_fill, peak_after_fill, _ = bookkeeper.apply_fill(fill.cost_usdc)
                strat.sync_drawdown_state(eq_after_fill, peak_after_fill, ts=exec_target_ts)

                orow = build_order_row(
                    run_id, order_id, m, dec, sig_q, exec_target_ts, latency_ms,
                    exec_q, exec_mark_px, exec_book_age_ms, exec_book_stale, fill, "FILLED", "", eq_after_fill,
                )
                order_status_counts["FILLED"] = order_status_counts.get("FILLED", 0) + 1
                strat.record_fill(m.outcome_id)
                filled_orders_this_market.append((orow, dec.eff_settle_fee))

            strat.in_flight_markets.discard(m.outcome_id)

            outcome_logger.append({
                "run_id": run_id,
                "outcome_id": m.outcome_id,
                "underlying": m.underlying,
                "period": m.period,
                "expiry_utc": utc_iso(m.expiry_ts),
                "strike": m.strike,
                "settled_twap": round(settled_twap, 4),
                "twap_minus_strike": round(settled_twap - m.strike, 4),
                "twap_dist_bps": fmt_opt(calc_bps(settled_twap, m.strike), 2),
                "mark_px_expiry": fmt_opt(mark_px_expiry, 4),
                "outcome_up": outcome_up,
                "had_filled_trade": int(bool(filled_orders_this_market)),
            })

            for orow, eff_fee in filled_orders_this_market:
                won = (orow["side"] == "UP" and outcome_up == 1) or (orow["side"] == "DOWN" and outcome_up == 0)
                payout = orow["filled_shares"] * (1.0 - eff_fee) if won else 0.0
                pnl = payout - float(orow["cost_usdc"])

                # 结算只加 payout_usdc，不再重复扣除 cost_usdc
                eq_after_payout, peak_after_payout, _ = bookkeeper.apply_payout(payout)
                strat.sync_drawdown_state(eq_after_payout, peak_after_payout, ts=m.expiry_ts)

                srow = build_settlement_row(
                    run_id, orow, m.expiry_ts, settled_twap, mark_px_expiry, outcome_up,
                    eq_after_payout, payout, pnl, won,
                )
                trades_logger.append(srow)
                settled_records.append(srow)
    finally:
        conn.close()
        trades_logger.close()
        rejected_logger.close()
        calib_logger.close()
        decision_logger.close()
        outcome_logger.close()

    end_ts = time.time()
    shadow_metrics = compute_settlement_shadow_equity(
        cfg.initial_equity,
        [float(r["equity_after"]) for r in settled_records],
    )
    extra_metrics = {
        "markets_replayed": len(m_rows),
        "settled_trades": len(settled_records),
        "calibration_samples": calib_count,
        "win_rate_pct": round(
            sum(r["won"] for r in settled_records) / len(settled_records) * 100.0, 2
        ) if settled_records else 0.0,
        "order_status_breakdown": order_status_counts,
        "decision_sample_breakdown": decision_counts,
        "halted_by_drawdown": strat.halted,
        **strat.halt_snapshot(),
        **shadow_metrics,
    }
    exec_params = {
        "latency_ms": latency_ms,
        "max_slippage": max_slippage,
        "stale_book_ms": stale_book_ms,
        "decision_sample_sec": decision_sample_sec,
        "near_edge_margin": near_edge_margin,
    }
    summary = write_summary_and_metadata(
        out_dir, "bt_summary.json", run_id, "export-backtest", network, u_tag, p_tag,
        start_ts, end_ts, cfg, exec_params, bookkeeper, extra_metrics,
        db_path=str(Path(db_path).resolve()),
    )

    print(
        f"\n[ExportBT] 回放完成 | RunID={run_id} | Bookkeeping={bookkeeper.source} | "
        f"输出目录: {out_dir.resolve()}"
    )
    print(json.dumps({
        "initial_equity": summary["initial_equity"],
        "final_equity": summary["final_equity"],
        "return_pct": summary["return_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "settled_trades": summary["settled_trades"],
        "win_rate_pct": summary["win_rate_pct"],
        "order_status_breakdown": summary["order_status_breakdown"],
    }, ensure_ascii=False, indent=2))
    return True, summary


def export_replay_backtest(
    db_path: str,
    cfg: StrategyConfig,
    latency_ms: int,
    max_slippage: float,
    stale_book_ms: float,
    decision_sample_sec: float,
    near_edge_margin: float,
    out_dir_arg: str,
    out_root_arg: str,
    subdir_by_run_id: bool,
    network: str = "mainnet",
) -> bool:
    ok, _ = run_unified_replay_backtest(
        db_path=db_path,
        cfg=cfg,
        latency_ms=latency_ms,
        max_slippage=max_slippage,
        stale_book_ms=stale_book_ms,
        decision_sample_sec=decision_sample_sec,
        near_edge_margin=near_edge_margin,
        out_dir_arg=out_dir_arg,
        out_root_arg=out_root_arg,
        subdir_by_run_id=subdir_by_run_id,
        network=network,
    )
    return ok


# ============================================================
# 7. CLI 入口（保留 --out-dir 默认路径，并兼容 --out-root）
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Outcome HIP-4 模拟盘与回测全链路导出引擎")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common_args(p, default_out_dir: str):
        p.add_argument("--equity", type=float, default=1000.0)
        p.add_argument("--latency-ms", type=int, default=300)
        p.add_argument("--max-slippage", type=float, default=0.008)
        p.add_argument("--stale-book-ms", type=float, default=5000.0, help="盘口超过该毫秒未更新标为过期")
        p.add_argument(
            "--settlement-fee",
            type=float,
            default=0.0007,
            help="可配模拟结算费率参数（默认 0.0007 仅为假设值，非官方核实统一费率）",
        )
        p.add_argument("--decision-sample-sec", type=float, default=30.0, help="未开仓日志周期采样间隔(秒)")
        p.add_argument("--near-edge-margin", type=float, default=0.015, help="距开仓门槛此范围内触发落盘")
        p.add_argument("--out-dir", default=default_out_dir, help=f"输出目录（默认 {default_out_dir}，兼容 outcome_review.py）")
        p.add_argument("--out-root", default="", help="可选：按 run_id 在该目录下建子目录（优先于 --out-dir）")
        p.add_argument("--subdir-by-run-id", action="store_true", help="可选：在 --out-dir 下再按 run_id 建子目录")

    p_paper = sub.add_parser("paper", help="实时模拟盘并导出兼容复盘与全链路诊断日志")
    p_paper.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    p_paper.add_argument("--underlying", default="BTC")
    p_paper.add_argument("--period", default="15m")
    p_paper.add_argument(
        "--db",
        default="hip4_replay.sqlite",
        help="同步录制行情磁带的 SQLite 路径（默认 hip4_replay.sqlite）；不要与 record 同时并发写同一文件",
    )
    p_paper.add_argument("--minutes", type=float, default=120.0)
    add_common_args(p_paper, default_out_dir="paper_logs")

    p_exp = sub.add_parser("export-backtest", help="回放 SQLite 并导出兼容复盘与全链路诊断日志")
    p_exp.add_argument("--db", default="hip4_replay.sqlite")
    p_exp.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    add_common_args(p_exp, default_out_dir="bt_exports")

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
                stale_book_ms=args.stale_book_ms,
                decision_sample_sec=args.decision_sample_sec,
                near_edge_margin=args.near_edge_margin,
                out_dir_arg=args.out_dir,
                out_root_arg=args.out_root,
                subdir_by_run_id=args.subdir_by_run_id,
                duration_min=args.minutes,
                db_path=args.db,
            )
        )
    elif args.cmd == "export-backtest":
        ok = export_replay_backtest(
            db_path=args.db,
            cfg=cfg,
            latency_ms=args.latency_ms,
            max_slippage=args.max_slippage,
            stale_book_ms=args.stale_book_ms,
            decision_sample_sec=args.decision_sample_sec,
            near_edge_margin=args.near_edge_margin,
            out_dir_arg=args.out_dir,
            out_root_arg=args.out_root,
            subdir_by_run_id=args.subdir_by_run_id,
            network=args.network,
        )
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
