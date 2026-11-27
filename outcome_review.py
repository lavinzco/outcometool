#!/usr/bin/env python3
"""Review paper-trading and backtest exports. Does not place live orders.

Equity curve and MaxDD use settlement-time equity_after only, in file order
(not sorted by exec_utc). paper_settlements.csv is appended as markets settle.
bt_trades.csv stores equity_after as a running total in processing order
(markets handled one after another); sorting by exec_utc would scramble it.
That curve can understate drawdown versus paper_summary.json / bt_summary.json,
which also count cash reserved at fill time.

Initial equity comes from --summary, else a sibling paper_summary.json or
bt_summary.json, else --initial-equity. Inferring first equity_after minus
first pnl_usdc is a last resort: with overlapping positions that value is too
low because cash for other open fills is already reserved.

A missing eff_settle_fee column is not filled with 0.0007. That number is only
a configurable estimate in this repo, not a verified official settlement fee.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import pandas as pd


EQUITY_CURVE_NOTE = (
    "[Note] Equity curve and MaxDD use settlement-time equity_after only, in file order "
    "(not sorted by exec_utc). This can understate drawdown versus paper_summary.json / "
    "bt_summary.json, which also count cash reserved at fill time.\n"
    "[说明] 净值曲线和最大回撤只用结算时的 equity_after，并保持文件顺序，不会按 exec_utc 重排。"
    "成交时已经预扣的现金不在这条曲线里，所以回撤可能小于汇总 JSON。"
)

ZERO_SLIP_NOTE = (
    "[Note] Zero-slippage PnL adds slippage * shares back onto each trade's pnl_usdc. "
    "The plotted zero-slippage equity is initial equity plus that cumulative PnL in file order. "
    "When positions overlap, settlement equity_after is not initial equity plus cumulative PnL, "
    "because cash for other open fills is still reserved."
)

MISSING_FEE_WARNING = (
    "[Warn] Settlements file has no eff_settle_fee column; not filling 0.0007. "
    "That number is only a configurable estimate in this repo "
    "(--settlement-fee / StrategyConfig.settlement_fee_rate), not a verified official settlement fee. "
    "This repo also scales that estimate by each market's fee_scale before writing eff_settle_fee, "
    "so a flat fill would be wrong. realized_edge and signal_edge are left missing."
)

BLANK_FEE_TEMPLATE = (
    "[Warn] {n} settlement row(s) have a blank eff_settle_fee; not filling 0.0007."
)

DROPPED_TEMPLATE = (
    "[Warn] Dropped {dropped} of {total} row(s) missing pnl_usdc ({n_pnl}), "
    "equity_after ({n_eq}), or tau_sec ({n_tau}). "
    "A row can miss more than one field, so the per-field counts can exceed the dropped-row count."
)

COLUMN_MISSING_TEMPLATE = (
    "[Warn] Column {col} is missing entirely; every row is dropped for this field."
)

INFER_EQUITY_WARNING = (
    "[Warn] Inferring initial equity as first equity_after - first pnl_usdc = {value:.4f}. "
    "This is too low when other positions are open, because cash for those fills is already reserved. "
    "Return and drawdown would be wrong. Pass --initial-equity or place paper_summary.json / "
    "bt_summary.json next to the CSV."
)

CALIBRATION_OFF = (
    "[Note] Calibration is off (no --calibration path, and no bt_calibration.csv next to this settlements file)."
)

CALIBRATION_PAPER_MIX = (
    "[Note] Calibration is off. Found bt_calibration.csv next to paper settlements but will not mix them. "
    "Pass --calibration to use that file."
)

ORDERS_OFF = (
    "[Note] Order audit is off (no --orders path, and no matching paper_orders.csv or "
    "bt_rejected.csv next to this settlements file)."
)

FINAL_EQ_TOL = 0.05
MAX_DD_TOL_PP = 0.05
EQUITY_PATH_GAP_TOL = 0.05

TAU_BINS = [-0.01, 60.0, 180.0, 300.0, 600.0, float("inf")]
TAU_LABELS_EN = [
    "0-60s (TWAP)",
    "60-180s (Gamma)",
    "180-300s (Late)",
    "300-600s (Mid)",
    "600s+ (Early)",
]
TAU_LABELS_ZH = [
    "0-60s (TWAP锁定)",
    "60-180s (Gamma区)",
    "180-300s (中后段)",
    "300-600s (中段)",
    "600s+ (开盘前段)",
]

REQUIRED_SETTLEMENT_FIELDS = ("pnl_usdc", "equity_after", "tau_sec")
REQUIRED_CALIBRATION_FIELDS = ("p_up_student_t", "p_up_normal", "outcome_up")
NUMERIC_SETTLEMENT_FIELDS = (
    "won",
    "filled_shares",
    "fill_vwap",
    "cost_usdc",
    "eff_settle_fee",
    "pnl_usdc",
    "equity_after",
    "fair_prob",
    "tau_sec",
    "slippage",
)

HELP_EPILOG = f"""
{EQUITY_CURVE_NOTE}

Initial equity: --summary, else a sibling paper_summary.json or bt_summary.json,
else --initial-equity. Inferring first equity_after - first pnl_usdc is a last
resort and prints a warning. It is too low when other positions are still open
because their cost is already reserved.

A missing eff_settle_fee column is not filled with 0.0007. That number is only a
configurable estimate in this repo, not a verified official settlement fee.
The column is used when it is present.

Calibration is off unless --calibration is set, or the settlements file is not a
paper export and bt_calibration.csv sits in the same directory. Order audit
follows the same idea for paper_orders.csv / bt_rejected.csv. This script does
not default those inputs to two different run directories.

Rows missing pnl_usdc, equity_after, or tau_sec are dropped and the counts are
printed. Kept rows stay in file order and are not sorted by exec_utc or
settle_time_utc.
"""


@dataclass
class SettlementLoad:
    df: pd.DataFrame
    rows_in: int
    rows_kept: int
    rows_dropped: int
    missing_counts: Dict[str, int]
    fee_column_present: bool
    messages: List[str]


@dataclass
class CalibrationLoad:
    df: pd.DataFrame
    rows_in: int
    rows_kept: int
    rows_dropped: int
    missing_counts: Dict[str, int]
    messages: List[str]


@dataclass
class EquityResolution:
    initial_equity: Optional[float]
    source: str
    summary_path: Optional[Path]
    summary: Optional[dict]
    notes: List[str]


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def setup_matplotlib(show: bool) -> bool:
    """Headless runs use the Agg backend. Pick a CJK font when one is installed."""
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in cjk_candidates if name in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen is not None


def export_family(filename: str) -> str:
    name = filename.lower()
    if "paper_settlement" in name or name.startswith("paper_"):
        return "paper"
    if "bt_trade" in name or name.startswith("bt_"):
        return "bt"
    return "unknown"


def assign_tau_bucket(tau_sec: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(tau_sec, errors="coerce"),
        bins=TAU_BINS,
        labels=TAU_LABELS_EN,
        include_lowest=True,
        right=True,
    )


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _sum_complete(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if values.size == 0:
        return 0.0
    if not np.isfinite(values).all():
        return float("nan")
    return float(values.sum())


def sum_complete(series: pd.Series) -> float:
    return _sum_complete(series)


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number


def fmt_num(value: object, digits: int = 2, signed: bool = False) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if not math.isfinite(number):
        return "n/a"
    if signed:
        return f"{number:+.{digits}f}"
    return f"{number:.{digits}f}"


def infer_initial_equity(df: pd.DataFrame) -> float:
    """First kept row only. Wrong when another position's cost is still reserved."""
    return float(df.iloc[0]["equity_after"] - df.iloc[0]["pnl_usdc"])


def summary_float(summary: dict, key: str) -> Optional[float]:
    if key not in summary or summary[key] is None or summary[key] == "":
        return None
    try:
        value = float(summary[key])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def read_summary(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"[Warn] Could not read summary {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"[Warn] Summary {path} is not a JSON object."
    return data, None


def locate_summary(
    settlements_path: Path,
    summary_arg: Optional[str],
) -> Tuple[Optional[Path], List[str]]:
    notes: List[str] = []
    if summary_arg:
        explicit = Path(summary_arg)
        if explicit.is_file():
            return explicit, notes
        notes.append(f"[Warn] --summary not found: {explicit}")

    directory = settlements_path.parent
    paper = directory / "paper_summary.json"
    bt = directory / "bt_summary.json"
    paper_ok = paper.is_file()
    bt_ok = bt.is_file()
    family = export_family(settlements_path.name)

    if family == "paper" and paper_ok:
        if bt_ok:
            notes.append(
                "[Note] Using paper_summary.json and ignoring sibling bt_summary.json "
                "because the settlements file is a paper export."
            )
        return paper, notes
    if family == "bt" and bt_ok:
        if paper_ok:
            notes.append(
                "[Note] Using bt_summary.json and ignoring sibling paper_summary.json "
                "because the settlements file is a backtest export."
            )
        return bt, notes
    if paper_ok and not bt_ok:
        if family == "bt":
            notes.append("[Note] No bt_summary.json next to the backtest trades; using paper_summary.json.")
        return paper, notes
    if bt_ok and not paper_ok:
        if family == "paper":
            notes.append("[Note] No paper_summary.json next to the paper settlements; using bt_summary.json.")
        return bt, notes
    if paper_ok and bt_ok:
        notes.append(
            "[Warn] Both paper_summary.json and bt_summary.json are next to the settlements file, "
            "and the CSV name does not say which run it is. Pass --summary. "
            "Not guessing an initial equity from them."
        )
    return None, notes


def resolve_initial_equity(
    df: pd.DataFrame,
    settlements_path: Path,
    summary_arg: Optional[str],
    override: Optional[float],
) -> EquityResolution:
    notes: List[str] = []
    path, locate_notes = locate_summary(settlements_path, summary_arg)
    notes.extend(locate_notes)

    summary: Optional[dict] = None
    loaded_path: Optional[Path] = None
    initial: Optional[float] = None
    source = "unavailable"

    if path is not None:
        summary, err = read_summary(path)
        if err:
            notes.append(err)
            summary = None
        else:
            loaded_path = path
            parsed = summary_float(summary, "initial_equity")
            if parsed is None:
                notes.append(f"[Warn] {path.name} has no numeric initial_equity.")
            else:
                initial = parsed
                source = "summary"
                notes.append(f"[Note] Initial equity {initial:.4f} from {path.name} (initial_equity).")
                if override is not None and abs(float(override) - initial) > 0.01:
                    notes.append(
                        f"[Warn] --initial-equity={float(override):.4f} was not used because "
                        f"{path.name} initial_equity={initial:.4f} takes priority."
                    )

    if source != "summary":
        if override is not None:
            initial = float(override)
            source = "flag"
            notes.append(f"[Note] Initial equity {initial:.4f} from --initial-equity.")
        elif len(df) > 0 and {"equity_after", "pnl_usdc"} <= set(df.columns):
            initial = infer_initial_equity(df)
            source = "inferred"
            notes.append(INFER_EQUITY_WARNING.format(value=initial))
        else:
            notes.append(
                "[Warn] No initial equity available. Pass --initial-equity or add a summary JSON."
            )

    return EquityResolution(
        initial_equity=initial,
        source=source,
        summary_path=loaded_path,
        summary=summary,
        notes=notes,
    )


def equity_path_note(gap: float) -> Optional[str]:
    if gap > EQUITY_PATH_GAP_TOL:
        return (
            f"[Note] equity_after differs from initial_equity + cumulative pnl_usdc by up to {gap:.4f} USDC "
            "in file order. Overlapping positions reserve cash, and a chosen initial equity can also disagree "
            "with this series, so equity_after - pnl_usdc on the first row is not the starting equity."
        )
    return None


def cross_check_summary(summary: dict, final_eq: float, max_dd: float) -> List[str]:
    notes: List[str] = []
    ref_final = summary_float(summary, "final_equity")
    if "final_equity" in summary and ref_final is None:
        notes.append("[Warn] summary final_equity is not a number; skipped that cross-check.")
    elif ref_final is not None and abs(ref_final - final_eq) > FINAL_EQ_TOL:
        notes.append(
            f"[Note] Summary final_equity={ref_final:.4f} differs from last settlement equity_after={final_eq:.4f}. "
            "The summary can include cash still reserved for open fills; this script only sees "
            "settlement-time equity_after."
        )

    ref_dd = summary_float(summary, "max_drawdown_pct")
    if "max_drawdown_pct" in summary and ref_dd is None:
        notes.append("[Warn] summary max_drawdown_pct is not a number; skipped that cross-check.")
    elif ref_dd is not None:
        script_pp = max_dd * 100.0
        if abs(ref_dd - script_pp) > MAX_DD_TOL_PP:
            notes.append(
                f"[Note] Summary max_drawdown_pct={ref_dd:.4f} differs from settlement-time MaxDD={script_pp:.4f}. "
                "Settlement-time equity_after can understate drawdown versus the summary, which also counts "
                "cash reserved at fill time."
            )
    return notes


def load_settlements(path: Path) -> SettlementLoad:
    """Load paper_settlements.csv or bt_trades.csv without reordering rows.

    bt_trades.csv uses shares and exec_utc. Those are renamed so the rest of
    the review can share one frame. equity_after stays in file order.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    fee_column_present = "eff_settle_fee" in raw.columns
    if raw.empty:
        messages = ["[Warn] Settlements file has no data rows."]
        if not fee_column_present:
            messages.append(MISSING_FEE_WARNING)
        return SettlementLoad(
            df=raw,
            rows_in=0,
            rows_kept=0,
            rows_dropped=0,
            missing_counts={col: 0 for col in REQUIRED_SETTLEMENT_FIELDS},
            fee_column_present=fee_column_present,
            messages=messages,
        )

    df = raw.copy()
    rename: Dict[str, str] = {}
    if "shares" in df.columns and "filled_shares" not in df.columns:
        rename["shares"] = "filled_shares"
    if "exec_utc" in df.columns and "settle_time_utc" not in df.columns:
        rename["exec_utc"] = "settle_time_utc"
    if rename:
        df = df.rename(columns=rename)

    for col in NUMERIC_SETTLEMENT_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    messages: List[str] = []
    missing_counts: Dict[str, int] = {}
    drop_mask = pd.Series(False, index=df.index)
    for col in REQUIRED_SETTLEMENT_FIELDS:
        if col not in df.columns:
            missing = pd.Series(True, index=df.index)
            messages.append(COLUMN_MISSING_TEMPLATE.format(col=col))
        else:
            missing = df[col].isna()
        missing_counts[col] = int(missing.sum())
        drop_mask = drop_mask | missing

    rows_in = int(len(df))
    rows_dropped = int(drop_mask.sum())
    if rows_dropped:
        messages.append(
            DROPPED_TEMPLATE.format(
                dropped=rows_dropped,
                total=rows_in,
                n_pnl=missing_counts["pnl_usdc"],
                n_eq=missing_counts["equity_after"],
                n_tau=missing_counts["tau_sec"],
            )
        )
    df = df.loc[~drop_mask].reset_index(drop=True)

    if not fee_column_present:
        messages.append(MISSING_FEE_WARNING)
        if len(df):
            df["eff_settle_fee"] = np.nan
    elif len(df):
        blank_fee = int(df["eff_settle_fee"].isna().sum())
        if blank_fee:
            messages.append(BLANK_FEE_TEMPLATE.format(n=blank_fee))

    if len(df) == 0:
        return SettlementLoad(
            df=df,
            rows_in=rows_in,
            rows_kept=0,
            rows_dropped=rows_dropped,
            missing_counts=missing_counts,
            fee_column_present=fee_column_present,
            messages=messages,
        )

    if "slippage" not in df.columns:
        messages.append(
            "[Warn] slippage column is missing; not assuming 0. "
            "Slippage dollar cost and the zero-slippage counterfactual are left missing."
        )
        df["slippage"] = np.nan
    if "won" not in df.columns:
        messages.append("[Warn] won column is missing; win rate is left missing.")
        df["won"] = np.nan
    if "fair_prob" not in df.columns:
        messages.append("[Warn] fair_prob column is missing; predicted win rate and edge are left missing.")
        df["fair_prob"] = np.nan
    if "cost_usdc" not in df.columns:
        messages.append("[Warn] cost_usdc column is missing; ROI is left missing.")
        df["cost_usdc"] = np.nan
    if "fill_vwap" not in df.columns:
        df["fill_vwap"] = np.nan

    if "filled_shares" not in df.columns:
        vwap = df["fill_vwap"].to_numpy(dtype=float)
        cost = df["cost_usdc"].to_numpy(dtype=float)
        shares = np.full(len(df), np.nan)
        ok = np.isfinite(vwap) & (np.abs(vwap) > 1e-6) & np.isfinite(cost)
        shares[ok] = cost[ok] / vwap[ok]
        df["filled_shares"] = shares
        if int(np.isfinite(shares).sum()) > 0:
            messages.append(
                "[Warn] filled_shares/shares column missing; derived shares as cost_usdc / fill_vwap "
                "where vwap is non-zero."
            )
        else:
            messages.append(
                "[Warn] filled_shares/shares column missing and cannot be derived; "
                "slippage dollar cost is left missing."
            )

    df["slip_cost_usdc"] = df["slippage"] * df["filled_shares"]
    df["zero_slip_pnl"] = df["pnl_usdc"] + df["slip_cost_usdc"]
    net_payout_per_share = 1.0 - df["eff_settle_fee"]
    df["realized_edge"] = df["fair_prob"] * net_payout_per_share - df["fill_vwap"]
    df["signal_edge"] = df["fair_prob"] * net_payout_per_share - (df["fill_vwap"] - df["slippage"])
    df["tau_bucket"] = assign_tau_bucket(df["tau_sec"])
    outside = int(df["tau_bucket"].isna().sum())
    if outside:
        messages.append(
            f"[Warn] {outside} kept row(s) have tau_sec outside the review buckets "
            "and are left out of the tau table."
        )

    return SettlementLoad(
        df=df,
        rows_in=rows_in,
        rows_kept=int(len(df)),
        rows_dropped=rows_dropped,
        missing_counts=missing_counts,
        fee_column_present=fee_column_present,
        messages=messages,
    )


def load_calibration(path: Path) -> CalibrationLoad:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    messages: List[str] = []
    if raw.empty:
        messages.append("[Warn] Calibration file has no data rows.")
        return CalibrationLoad(raw, 0, 0, 0, {col: 0 for col in REQUIRED_CALIBRATION_FIELDS}, messages)

    df = raw.copy()
    missing_counts: Dict[str, int] = {}
    drop_mask = pd.Series(False, index=df.index)
    for col in REQUIRED_CALIBRATION_FIELDS:
        if col not in df.columns:
            missing = pd.Series(True, index=df.index)
            messages.append(COLUMN_MISSING_TEMPLATE.format(col=col))
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            missing = df[col].isna()
        missing_counts[col] = int(missing.sum())
        drop_mask = drop_mask | missing

    if "tau_sec" in df.columns:
        df["tau_sec"] = pd.to_numeric(df["tau_sec"], errors="coerce")
    else:
        messages.append("[Note] Calibration file has no tau_sec column; the TWAP-zone split is skipped.")

    if "outcome_id" not in df.columns:
        messages.append("[Note] Calibration file has no outcome_id column; market count is unavailable.")

    rows_in = int(len(df))
    rows_dropped = int(drop_mask.sum())
    if rows_dropped:
        messages.append(
            "[Warn] Dropped {dropped} of {total} calibration row(s) missing {fields}.".format(
                dropped=rows_dropped,
                total=rows_in,
                fields=", ".join(
                    f"{col} ({missing_counts[col]})" for col in REQUIRED_CALIBRATION_FIELDS
                ),
            )
        )
    df = df.loc[~drop_mask].reset_index(drop=True)
    return CalibrationLoad(df, rows_in, int(len(df)), rows_dropped, missing_counts, messages)


def equity_curve(equity_after: np.ndarray, initial_equity: float) -> Tuple[np.ndarray, np.ndarray, float]:
    eq_series = np.concatenate([[float(initial_equity)], np.asarray(equity_after, dtype=float)])
    peak = np.maximum.accumulate(eq_series)
    dd = (peak - eq_series) / np.maximum(peak, 1e-9)
    max_dd = float(np.max(dd)) if len(dd) else 0.0
    return eq_series, dd, max_dd


def tau_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("tau_bucket", observed=False, dropna=True)
        .agg(
            trades=("pnl_usdc", "count"),
            win_rate=("won", "mean"),
            pred_win=("fair_prob", "mean"),
            cost_usdc=("cost_usdc", _sum_complete),
            pnl_usdc=("pnl_usdc", _sum_complete),
            avg_slip=("slippage", "mean"),
            slip_usdc=("slip_cost_usdc", _sum_complete),
            zero_slip_pnl=("zero_slip_pnl", _sum_complete),
            avg_real_edge=("realized_edge", "mean"),
        )
    )
    grouped.index = grouped.index.astype(str)
    grouped = grouped.reindex(TAU_LABELS_EN)
    grouped.index.name = "tau_bucket"
    grouped = grouped.reset_index()
    grouped["trades"] = grouped["trades"].fillna(0).astype(int)
    zero = grouped["trades"] == 0
    for col in ("cost_usdc", "pnl_usdc", "slip_usdc", "zero_slip_pnl"):
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
        grouped.loc[zero, col] = grouped.loc[zero, col].fillna(0.0)
    grouped.loc[zero, "win_rate"] = 0.0
    grouped.loc[zero, "pred_win"] = 0.0

    cost = grouped["cost_usdc"].to_numpy(dtype=float)
    pnl = grouped["pnl_usdc"].to_numpy(dtype=float)
    roi = np.full(len(grouped), np.nan)
    ok = np.isfinite(cost) & (cost > 0) & np.isfinite(pnl)
    roi[ok] = pnl[ok] / cost[ok] * 100.0
    roi[zero.to_numpy()] = 0.0
    grouped["roi_pct"] = roi
    grouped["win_rate_pct"] = grouped["win_rate"] * 100.0
    grouped["pred_win_pct"] = grouped["pred_win"] * 100.0
    grouped["win_gap_pct"] = grouped["win_rate_pct"] - grouped["pred_win_pct"]
    return grouped


def side_breakdown(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "side" not in df.columns or not df["side"].notna().any():
        return None
    table = (
        df.groupby("side", dropna=True, sort=False)
        .agg(
            trades=("pnl_usdc", "count"),
            win_rate_pct=("won", lambda s: float(pd.to_numeric(s, errors="coerce").mean()) * 100.0),
            pnl_usdc=("pnl_usdc", "sum"),
            avg_slip=("slippage", "mean"),
            slip_usdc=("slip_cost_usdc", _sum_complete),
        )
    )
    return table


def analyze_settlements(df: pd.DataFrame, initial_equity: float) -> Tuple[dict, pd.DataFrame]:
    equity_after = df["equity_after"].to_numpy(dtype=float)
    pnl = df["pnl_usdc"].to_numpy(dtype=float)
    eq_series, dd_series, max_dd = equity_curve(equity_after, initial_equity)
    final_eq = float(equity_after[-1])
    total_pnl = float(pnl.sum())
    total_cost = sum_complete(df["cost_usdc"])
    total_slip = sum_complete(df["slip_cost_usdc"])
    zero_slip_total = sum_complete(df["zero_slip_pnl"])

    cum_pnl_equity = float(initial_equity) + np.cumsum(pnl)
    equity_path_gap = float(np.nanmax(np.abs(cum_pnl_equity - equity_after)))

    wins = float(df.loc[df["pnl_usdc"] > 0, "pnl_usdc"].sum())
    losses = float((-df.loc[df["pnl_usdc"] < 0, "pnl_usdc"]).sum())
    if losses > 0:
        profit_factor = wins / losses
    elif wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    pnl_std = float(df["pnl_usdc"].std(ddof=1)) if len(df) > 1 else float("nan")
    if np.isfinite(pnl_std) and pnl_std > 1e-9:
        trade_sharpe = float(df["pnl_usdc"].mean()) / pnl_std * math.sqrt(len(df))
    else:
        trade_sharpe = float("nan")

    if initial_equity > 0 and np.isfinite(final_eq):
        return_pct = (final_eq / float(initial_equity) - 1.0) * 100.0
    else:
        return_pct = float("nan")

    zero_slip_values = df["zero_slip_pnl"].to_numpy(dtype=float)
    zero_slip_eq = np.concatenate(
        [[float(initial_equity)], float(initial_equity) + np.cumsum(zero_slip_values)]
    )
    grouped = tau_segment_table(df)
    summary = {
        "init_eq": float(initial_equity),
        "final_eq": final_eq,
        "return_pct": float(return_pct),
        "max_dd": max_dd,
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "total_slip_usdc": total_slip,
        "zero_slip_total_pnl": zero_slip_total,
        "profit_factor": float(profit_factor),
        "trade_sharpe": float(trade_sharpe),
        "equity_path_gap": equity_path_gap,
        "n_trades": int(len(df)),
        "win_rate": float(pd.to_numeric(df["won"], errors="coerce").mean()),
        "pred_win": float(pd.to_numeric(df["fair_prob"], errors="coerce").mean()),
        "eq_series": eq_series,
        "dd_series": dd_series,
        "zero_slip_eq": zero_slip_eq,
    }
    return summary, grouped


def calc_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    n = len(probs)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        upper = edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= edges[i]) & (probs <= upper)
        else:
            mask = (probs >= edges[i]) & (probs < upper)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(outcomes[mask])) - float(np.mean(probs[mask])))
    return float(ece)


def calc_log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    if len(probs) == 0:
        return float("nan")
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(outcomes * np.log(p) + (1.0 - outcomes) * np.log(1.0 - p)))


def compute_calibration_bins(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
            bracket = "]"
        else:
            mask = (probs >= lo) & (probs < hi)
            bracket = ")"
        count = int(np.sum(mask))
        if count > 0:
            pred_mean = float(np.mean(probs[mask]))
            actual_rate = float(np.mean(outcomes[mask]))
            gap = actual_rate - pred_mean
        else:
            pred_mean = float("nan")
            actual_rate = float("nan")
            gap = float("nan")
        rows.append({
            "bin": f"[{lo:.1f}, {hi:.1f}{bracket}",
            "bin_mid": (lo + hi) / 2.0,
            "count": count,
            "pred_mean": pred_mean,
            "actual_rate": actual_rate,
            "gap": gap,
        })
    return pd.DataFrame(rows)


def analyze_calibration(cal_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    p_t = cal_df["p_up_student_t"].to_numpy(dtype=float)
    p_n = cal_df["p_up_normal"].to_numpy(dtype=float)
    y = cal_df["outcome_up"].to_numpy(dtype=float)
    tail_mask = (p_t >= 0.85) | (p_t <= 0.15)
    tail_n = int(np.sum(tail_mask))
    metrics = {
        "n": int(len(cal_df)),
        "n_outcomes": int(cal_df["outcome_id"].nunique()) if "outcome_id" in cal_df.columns else None,
        "brier_t": float(np.mean((p_t - y) ** 2)),
        "brier_n": float(np.mean((p_n - y) ** 2)),
        "ece_t": calc_ece(p_t, y),
        "ece_n": calc_ece(p_n, y),
        "log_loss_t": calc_log_loss(p_t, y),
        "log_loss_n": calc_log_loss(p_n, y),
        "tail_n": tail_n,
        "tail_brier_t": float(np.mean((p_t[tail_mask] - y[tail_mask]) ** 2)) if tail_n else float("nan"),
        "tail_brier_n": float(np.mean((p_n[tail_mask] - y[tail_mask]) ** 2)) if tail_n else float("nan"),
    }
    return compute_calibration_bins(p_t, y), compute_calibration_bins(p_n, y), metrics


def resolve_calibration(
    settlements_path: Path,
    calibration_arg: Optional[str],
    skip: bool,
) -> Tuple[Optional[Path], List[str]]:
    if skip:
        return None, ["[Note] Calibration skipped (--no-calibration)."]
    if calibration_arg:
        path = Path(calibration_arg)
        if path.is_file():
            return path, [f"[Note] Using calibration file: {path}"]
        return None, [f"[Warn] Calibration file not found: {path}"]

    sibling = settlements_path.parent / "bt_calibration.csv"
    family = export_family(settlements_path.name)
    if family == "paper":
        if sibling.is_file():
            return None, [CALIBRATION_PAPER_MIX]
        return None, [CALIBRATION_OFF]
    if sibling.is_file():
        return sibling, [f"[Note] Using sibling calibration file: {sibling}"]
    return None, [CALIBRATION_OFF]


def resolve_orders(
    settlements_path: Path,
    orders_arg: Optional[str],
    skip: bool,
) -> Tuple[List[Path], List[str]]:
    if skip:
        return [], ["[Note] Order audit skipped (--no-orders)."]
    if orders_arg:
        path = Path(orders_arg)
        if path.is_file():
            return [path], [f"[Note] Using orders file: {path}"]
        return [], [f"[Warn] Orders file not found: {path}"]

    directory = settlements_path.parent
    family = export_family(settlements_path.name)
    paper = directory / "paper_orders.csv"
    rejected = directory / "bt_rejected.csv"
    if family == "paper":
        notes: List[str] = []
        if rejected.is_file():
            notes.append(
                "[Note] Not using bt_rejected.csv next to paper settlements. Pass --orders to read it."
            )
        if paper.is_file():
            notes.append(f"[Note] Using sibling orders file: {paper}")
            return [paper], notes
        notes.append(ORDERS_OFF)
        return [], notes
    if family == "bt":
        notes = []
        if paper.is_file():
            notes.append(
                "[Note] Not using paper_orders.csv next to backtest trades. Pass --orders to read it."
            )
        if rejected.is_file():
            notes.append(f"[Note] Using sibling orders file: {rejected}")
            return [rejected], notes
        notes.append(ORDERS_OFF)
        return [], notes

    found = [path for path in (paper, rejected) if path.is_file()]
    if len(found) == 1:
        return found, [f"[Note] Using sibling orders file: {found[0]}"]
    if len(found) > 1:
        return [], [
            "[Warn] Both paper_orders.csv and bt_rejected.csv are present. Pass --orders. Not guessing."
        ]
    return [], [ORDERS_OFF]


def audit_orders(path: Path) -> dict:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    result = {
        "path": path,
        "kind": "unknown",
        "rows": int(len(frame)),
        "filled": None,
        "reasons": {},
    }
    if frame.empty:
        return result
    if "status" in frame.columns:
        status = frame["status"].fillna("").astype(str).str.strip()
        filled = int((status == "FILLED").sum())
        rejected = frame.loc[status != "FILLED"]
        reason_col = "reject_reason" if "reject_reason" in frame.columns else None
        result.update({"kind": "orders", "filled": filled, "reasons": _reason_counts(rejected, reason_col)})
        return result
    if "reason" in frame.columns:
        result.update({"kind": "rejected", "reasons": _reason_counts(frame, "reason")})
        return result
    return result


def _reason_counts(frame: pd.DataFrame, column: Optional[str]) -> Dict[str, int]:
    if column is None or column not in frame.columns or frame.empty:
        return {}
    reasons = frame[column].fillna("(blank)").astype(str).str.strip()
    reasons = reasons.replace("", "(blank)")
    return {str(key): int(value) for key, value in reasons.value_counts().items()}


def format_order_audit(result: dict) -> str:
    path = result["path"]
    lines = [f"\n================ 3. 信号成交与拒单审计 ================", f"File: {path}"]
    if result["rows"] == 0:
        lines.append("No rows.")
        return "\n".join(lines)
    if result["kind"] == "orders":
        filled = int(result["filled"] or 0)
        total = int(result["rows"])
        rate = (filled / total * 100.0) if total else 0.0
        lines.append(f"Signals: {total} | Filled: {filled} ({rate:.1f}%)")
    elif result["kind"] == "rejected":
        lines.append(f"Rejected rows: {result['rows']}")
    else:
        lines.append(f"Rows: {result['rows']} (no status or reason column)")
    if result["reasons"]:
        lines.append("Reasons:")
        for reason, count in result["reasons"].items():
            lines.append(f"  {reason}: {count}")
    return "\n".join(lines)


def _table_float(value: object) -> str:
    number = _as_float(value)
    if number is None or not math.isfinite(number):
        return "n/a"
    if abs(number) < 1:
        return f"{number:.3f}"
    return f"{number:.2f}"


def print_settlement_report(df: pd.DataFrame, summary: dict, grouped: pd.DataFrame) -> None:
    init_eq = summary["init_eq"]
    final_eq = summary["final_eq"]
    total_pnl = summary["total_pnl"]
    total_cost = summary["total_cost"]
    total_slip = summary["total_slip_usdc"]
    zero_slip_total = summary["zero_slip_total_pnl"]
    win_rate = summary["win_rate"]
    pred_win = summary["pred_win"]

    print("\n================ 1. 模拟盘/回测绩效与滑点损耗报告 ================")
    print(
        f"Equity {init_eq:.2f} -> {final_eq:.2f} USDC | "
        f"MaxDD {summary['max_dd'] * 100.0:.2f}% | "
        f"PnL {total_pnl:+.2f} | "
        f"slip {fmt_num(total_slip, signed=True)} | "
        f"zero-slip PnL {fmt_num(zero_slip_total, signed=True)}"
    )
    print(f"总成交笔数             : {summary['n_trades']}")
    print(
        "胜率 / 平均预测胜率    : "
        f"{fmt_num(win_rate * 100.0 if np.isfinite(win_rate) else float('nan'))}%  /  "
        f"{fmt_num(pred_win * 100.0 if np.isfinite(pred_win) else float('nan'))}%"
    )
    print(
        f"期初 -> 期末净值       : {init_eq:.2f} -> {final_eq:.2f} USDC "
        f"({fmt_num(summary['return_pct'], signed=True)}%)"
    )
    print(f"最大回撤 (MaxDD)       : {summary['max_dd'] * 100.0:.2f}%")
    print(
        f"盈亏比 (Profit Factor) : {fmt_num(summary['profit_factor'])} | "
        f"样本交易夏普: {fmt_num(summary['trade_sharpe'])}"
    )
    if np.isfinite(total_cost) and total_cost > 0:
        roi = total_pnl / total_cost * 100.0
        cost_txt = f"{total_cost:.2f}"
        roi_txt = fmt_num(roi, signed=True)
    else:
        cost_txt = "n/a"
        roi_txt = "n/a"
    print(f"累计投入本金           : {cost_txt} USDC | 本金回报率 (ROI): {roi_txt}%")
    print("------------------------------------------------------------------")
    slip_values = pd.to_numeric(df["slippage"], errors="coerce")
    if slip_values.notna().any():
        print(
            f"平均单股价格滑点       : {slip_values.mean():+.5f} USDC/股 "
            f"(中位数: {slip_values.median():+.5f})"
        )
    else:
        print("平均单股价格滑点       : n/a")
    if np.isfinite(total_cost) and total_cost > 0 and np.isfinite(total_slip):
        slip_pct = total_slip / total_cost * 100.0
        slip_share = f" (占累计本金 {slip_pct:.2f}%)"
    else:
        slip_share = ""
    print(f"累计滑点总损耗         : {fmt_num(total_slip, signed=True)} USDC{slip_share}")
    print(
        f"实际净盈亏 vs 零滑点   : {fmt_num(total_pnl, signed=True)} USDC  vs  "
        f"{fmt_num(zero_slip_total, signed=True)} USDC"
    )
    if np.isfinite(zero_slip_total) and zero_slip_total > 0 and np.isfinite(total_slip):
        print(f"滑点吞噬毛利比例       : {total_slip / zero_slip_total * 100.0:.1f}%")

    print("\n--- 按剩余时间 (tau_sec) 分段表现 ---")
    disp = grouped[[
        "tau_bucket", "trades", "win_rate_pct", "pred_win_pct", "win_gap_pct",
        "pnl_usdc", "roi_pct", "avg_slip", "slip_usdc", "zero_slip_pnl",
    ]].copy()
    disp.columns = [
        "时间段", "笔数", "实测胜率%", "预测胜率%", "胜率偏差%",
        "净盈亏$", "ROI%", "均滑点", "滑点损耗$", "零滑点盈亏$",
    ]
    print(disp.to_string(index=False, na_rep="n/a", float_format=_table_float))

    sides = side_breakdown(df)
    if sides is not None:
        print("\n--- 多空方向 (UP / DOWN) 拆解 ---")
        print(sides.to_string(na_rep="n/a", float_format=_table_float))


def print_calibration_report(cal_df: pd.DataFrame, bins_t: pd.DataFrame, metrics: dict) -> None:
    print("\n================ 2. 定价模型概率校准检验 ================")
    if metrics["n_outcomes"] is None:
        coverage = "outcome_id 缺失"
    else:
        coverage = f"覆盖 {metrics['n_outcomes']} 个场次"
    print(f"校准样本数             : {metrics['n']} ({coverage})")
    print(
        f"Student-t  | Brier={metrics['brier_t']:.5f} | "
        f"LogLoss={metrics['log_loss_t']:.5f} | ECE={metrics['ece_t']:.5f}"
    )
    print(
        f"Normal     | Brier={metrics['brier_n']:.5f} | "
        f"LogLoss={metrics['log_loss_n']:.5f} | ECE={metrics['ece_n']:.5f}"
    )
    if metrics["tail_n"]:
        print(
            f"极端尾部段 (n={metrics['tail_n']}) | "
            f"Student-t Brier={metrics['tail_brier_t']:.5f} vs Normal Brier={metrics['tail_brier_n']:.5f}"
        )
    print("\n--- Student-t 分桶校准明细 ---")
    print(
        bins_t[["bin", "count", "pred_mean", "actual_rate", "gap"]].to_string(
            index=False, na_rep="n/a", float_format=lambda value: f"{value:.4f}"
        )
    )


def plot_equity_and_tau(
    summary: dict,
    grouped: pd.DataFrame,
    out_file: Path,
    has_cjk: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=140)
    try:
        x_idx = np.arange(len(summary["eq_series"]))
        ax1 = axes[0, 0]
        ax1.plot(
            x_idx,
            summary["eq_series"],
            color="#1a73e8",
            lw=2.2,
            label="实际净值 (含滑点)" if has_cjk else "Actual Equity",
        )
        ax1.plot(
            x_idx,
            summary["zero_slip_eq"],
            color="#34a853",
            lw=1.6,
            ls="--",
            alpha=0.85,
            label="零滑点反事实净值" if has_cjk else "Zero-Slippage Equity",
        )
        ax1.axhline(summary["init_eq"], color="#888888", lw=0.9, ls=":")
        ax1.fill_between(
            x_idx,
            summary["eq_series"],
            summary["zero_slip_eq"],
            color="#ea4335",
            alpha=0.12,
            label="累计滑点拖累" if has_cjk else "Slippage Drag Gap",
        )
        ax1.set_title(
            "净值曲线与滑点拖累对比" if has_cjk else "Equity Curve vs Zero-Slippage Counterfactual",
            fontweight="bold",
        )
        ax1.set_xlabel("成交序号" if has_cjk else "Trade #")
        ax1.set_ylabel("净值 (USDC)" if has_cjk else "Equity (USDC)")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best", fontsize=9)

        ax2 = axes[0, 1]
        dd_pct = -summary["dd_series"] * 100.0
        ax2.fill_between(x_idx, dd_pct, 0, color="#ea4335", alpha=0.30)
        ax2.plot(x_idx, dd_pct, color="#c5221f", lw=1.6)
        ax2.set_title("净值回撤曲线 (%)" if has_cjk else "Underwater Drawdown (%)", fontweight="bold")
        ax2.set_xlabel("成交序号" if has_cjk else "Trade #")
        ax2.set_ylabel("回撤 (%)" if has_cjk else "Drawdown (%)")
        ax2.grid(alpha=0.25)

        labels = TAU_LABELS_ZH if has_cjk else TAU_LABELS_EN
        if len(labels) != len(grouped):
            labels = [str(value) for value in grouped["tau_bucket"]]
        pos = np.arange(len(grouped))
        width = 0.36
        pnl_vals = grouped["pnl_usdc"].to_numpy(dtype=float)
        slip_vals = grouped["slip_usdc"].to_numpy(dtype=float)
        pnl_plot = np.where(np.isfinite(pnl_vals), pnl_vals, 0.0)
        slip_plot = np.where(np.isfinite(slip_vals), slip_vals, 0.0)

        ax3 = axes[1, 0]
        bars = ax3.bar(
            pos - width / 2,
            pnl_plot,
            width,
            color=["#34a853" if value >= 0 else "#ea4335" for value in pnl_plot],
            label="净盈亏 (USDC)" if has_cjk else "Net PnL ($)",
        )
        ax3.bar(
            pos + width / 2,
            -slip_plot,
            width,
            color="#fbbc04",
            alpha=0.85,
            label="滑点损耗 (-USDC)" if has_cjk else "Slippage Drag (-$)",
        )
        ax3.axhline(0, color="#444444", lw=0.9)
        ax3.set_xticks(pos)
        ax3.set_xticklabels(labels, rotation=16, ha="right", fontsize=9)
        ax3.set_title(
            "按剩余时间 (tau_sec) 分段盈亏与滑点损耗" if has_cjk else "PnL & Slippage Drag by Remaining Time",
            fontweight="bold",
        )
        ax3.set_ylabel("USDC")
        ax3.grid(axis="y", alpha=0.25)
        ax3.legend(loc="best", fontsize=9)
        for bar, count, height in zip(bars, grouped["trades"], pnl_plot):
            if not np.isfinite(height):
                continue
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"n={int(count)}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

        ax4 = axes[1, 1]
        win_vals = grouped["win_rate_pct"].to_numpy(dtype=float)
        pred_vals = grouped["pred_win_pct"].to_numpy(dtype=float)
        ax4.bar(
            pos - width / 2,
            np.where(np.isfinite(win_vals), win_vals, 0.0),
            width,
            color="#1a73e8",
            alpha=0.85,
            label="实际胜率 (%)" if has_cjk else "Realized Win Rate (%)",
        )
        ax4.bar(
            pos + width / 2,
            np.where(np.isfinite(pred_vals), pred_vals, 0.0),
            width,
            color="#9aa0a6",
            alpha=0.80,
            label="模型预测胜率 (%)" if has_cjk else "Predicted Fair Prob (%)",
        )
        ax4.set_xticks(pos)
        ax4.set_xticklabels(labels, rotation=16, ha="right", fontsize=9)
        ax4.set_ylim(0, 105)
        ax4.set_title(
            "各时间段实际胜率 vs 理论定价胜率" if has_cjk else "Realized Win Rate vs Predicted Prob by Segment",
            fontweight="bold",
        )
        ax4.set_ylabel("胜率 (%)" if has_cjk else "Probability (%)")
        ax4.grid(axis="y", alpha=0.25)
        ax4.legend(loc="best", fontsize=9)

        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
    finally:
        plt.close(fig)
    print(f"[Plot] 已保存净值与分段复盘图 -> {out_file.resolve()}")


def plot_calibration(
    cal_df: pd.DataFrame,
    bins_t: pd.DataFrame,
    bins_n: pd.DataFrame,
    metrics: dict,
    out_file: Path,
    has_cjk: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 7.2), dpi=140)
    try:
        gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 3, 1.8], hspace=0.28, wspace=0.24)
        ax_cal = fig.add_subplot(gs[0:2, 0])
        ax_cal.plot(
            [0, 1],
            [0, 1],
            ls="--",
            color="#5f6368",
            lw=1.2,
            label="完美校准线 (y = x)" if has_cjk else "Perfect Calibration (y=x)",
        )
        valid_t = bins_t.dropna(subset=["pred_mean", "actual_rate"])
        valid_n = bins_n.dropna(subset=["pred_mean", "actual_rate"])
        if not valid_t.empty:
            ax_cal.plot(
                valid_t["pred_mean"],
                valid_t["actual_rate"],
                marker="o",
                lw=2.2,
                color="#1a73e8",
                label=f"Student-t (Brier={metrics['brier_t']:.4f}, ECE={metrics['ece_t']:.4f})",
            )
        if not valid_n.empty:
            ax_cal.plot(
                valid_n["pred_mean"],
                valid_n["actual_rate"],
                marker="s",
                lw=1.6,
                ls=":",
                color="#ea4335",
                alpha=0.85,
                label=f"Normal (Brier={metrics['brier_n']:.4f}, ECE={metrics['ece_n']:.4f})",
            )
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.set_title(
            "总体概率校准曲线 (Reliability Diagram)" if has_cjk else "Overall Reliability Diagram",
            fontweight="bold",
        )
        ax_cal.set_ylabel("实际 UP 结算频率" if has_cjk else "Empirical UP Frequency")
        ax_cal.grid(alpha=0.25)
        ax_cal.legend(loc="upper left", fontsize=9)

        ax_hist = fig.add_subplot(gs[2, 0], sharex=ax_cal)
        ax_hist.bar(bins_t["bin_mid"], bins_t["count"], width=0.085, color="#1a73e8", alpha=0.65)
        ax_hist.set_xlabel("模型预测 UP 概率" if has_cjk else "Predicted UP Probability")
        ax_hist.set_ylabel("样本数" if has_cjk else "Count")
        ax_hist.grid(axis="y", alpha=0.25)

        ax_reg = fig.add_subplot(gs[0:2, 1])
        ax_reg.plot([0, 1], [0, 1], ls="--", color="#5f6368", lw=1.2)
        if "tau_sec" in cal_df.columns:
            twap_sub = cal_df[cal_df["tau_sec"] <= 60.0]
            pre_sub = cal_df[cal_df["tau_sec"] > 60.0]
            if len(twap_sub) >= 10:
                twap_bins = compute_calibration_bins(
                    twap_sub["p_up_student_t"].to_numpy(dtype=float),
                    twap_sub["outcome_up"].to_numpy(dtype=float),
                    n_bins=8,
                ).dropna(subset=["pred_mean", "actual_rate"])
                brier = float(np.mean((twap_sub["p_up_student_t"] - twap_sub["outcome_up"]) ** 2))
                label = (
                    f"TWAP锁定区 tau<=60s (n={len(twap_sub)}, Brier={brier:.4f})"
                    if has_cjk
                    else f"TWAP Zone <=60s (n={len(twap_sub)}, Brier={brier:.4f})"
                )
                ax_reg.plot(twap_bins["pred_mean"], twap_bins["actual_rate"], marker="^", lw=2.0, color="#34a853", label=label)
            if len(pre_sub) >= 10:
                pre_bins = compute_calibration_bins(
                    pre_sub["p_up_student_t"].to_numpy(dtype=float),
                    pre_sub["outcome_up"].to_numpy(dtype=float),
                    n_bins=10,
                ).dropna(subset=["pred_mean", "actual_rate"])
                brier = float(np.mean((pre_sub["p_up_student_t"] - pre_sub["outcome_up"]) ** 2))
                label = (
                    f"常规交易区 tau>60s (n={len(pre_sub)}, Brier={brier:.4f})"
                    if has_cjk
                    else f"Pre-TWAP >60s (n={len(pre_sub)}, Brier={brier:.4f})"
                )
                ax_reg.plot(pre_bins["pred_mean"], pre_bins["actual_rate"], marker="o", lw=2.0, color="#9334e6", label=label)
        ax_reg.set_xlim(0, 1)
        ax_reg.set_ylim(0, 1)
        ax_reg.set_title(
            "TWAP 锁定区 vs 常规区校准对比" if has_cjk else "Calibration: TWAP Zone vs Pre-TWAP",
            fontweight="bold",
        )
        ax_reg.set_ylabel("实际 UP 结算频率" if has_cjk else "Empirical UP Frequency")
        ax_reg.grid(alpha=0.25)
        if ax_reg.get_legend_handles_labels()[0]:
            ax_reg.legend(loc="upper left", fontsize=9)

        ax_gap = fig.add_subplot(gs[2, 1], sharex=ax_reg)
        populated = bins_t["count"] > 0
        gap_colors = ["#34a853" if gap >= 0 else "#ea4335" for gap in bins_t.loc[populated, "gap"].fillna(0.0)]
        ax_gap.bar(
            bins_t.loc[populated, "bin_mid"],
            bins_t.loc[populated, "gap"],
            width=0.085,
            color=gap_colors,
            alpha=0.75,
        )
        ax_gap.axhline(0, color="#444444", lw=0.9)
        ax_gap.set_xlabel("模型预测 UP 概率" if has_cjk else "Predicted UP Probability")
        ax_gap.set_ylabel("偏差 (实-预)" if has_cjk else "Gap (Obs-Pred)")
        ax_gap.grid(axis="y", alpha=0.25)
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"[Plot] 已保存概率校准图 -> {out_file.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "复盘纸面结算或回测导出（不下单）。 "
            "Review paper/backtest exports only; does not place live orders."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    parser.add_argument(
        "--settlements",
        required=True,
        help="paper_settlements.csv or bt_trades.csv. No default path, so a paper run is not mixed with another directory.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="paper_summary.json or bt_summary.json. Default: a sibling summary next to --settlements.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=None,
        help="Used only when no summary initial_equity is available. Do not rely on inferring equity_after - pnl_usdc.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Calibration CSV. Off by default. A sibling bt_calibration.csv is used only for non-paper settlements when that file exists.",
    )
    parser.add_argument(
        "--orders",
        default=None,
        help="paper_orders.csv or bt_rejected.csv. Off unless passed or a same-family sibling file exists.",
    )
    parser.add_argument("--no-calibration", action="store_true", help="Do not read a calibration file.")
    parser.add_argument("--no-orders", action="store_true", help="Do not read an orders or reject file.")
    parser.add_argument("--out-dir", default="review_reports", help="Directory for the chart and tau table.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Ask matplotlib to show figures. Headless runs stay on the Agg backend and still save PNGs.",
    )
    return parser


def _emit(lines: Sequence[str]) -> None:
    for line in lines:
        if line:
            print(line)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    has_cjk = setup_matplotlib(args.show)
    settlements_path = Path(args.settlements)
    if not settlements_path.is_file():
        print(f"[Error] Settlements file not found: {settlements_path}")
        return 1

    try:
        loaded = load_settlements(settlements_path)
    except pd.errors.EmptyDataError:
        print(f"[Error] Settlements file is empty: {settlements_path}")
        return 1
    except pd.errors.ParserError as exc:
        print(f"[Error] Could not parse {settlements_path}: {exc}")
        return 1

    _emit(loaded.messages)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = False

    if loaded.df.empty or loaded.rows_kept == 0:
        print("[Warn] No settlement rows left to analyze.")
    else:
        resolution = resolve_initial_equity(
            loaded.df,
            settlements_path,
            summary_arg=args.summary,
            override=args.initial_equity,
        )
        _emit(resolution.notes)
        if resolution.initial_equity is None:
            print("[Error] Cannot build an equity curve without an initial equity.")
            return 1
        print(EQUITY_CURVE_NOTE)
        print(ZERO_SLIP_NOTE)
        summary, grouped = analyze_settlements(loaded.df, resolution.initial_equity)
        print_settlement_report(loaded.df, summary, grouped)
        if resolution.summary:
            _emit(cross_check_summary(resolution.summary, summary["final_eq"], summary["max_dd"]))
        path_note = equity_path_note(summary["equity_path_gap"])
        if path_note:
            print(path_note)
        grouped.to_csv(out_dir / "tau_segment_stats.csv", index=False, encoding="utf-8-sig")
        plot_equity_and_tau(summary, grouped, out_dir / "equity_and_tau_report.png", has_cjk)
        produced = True

    cal_path, cal_notes = resolve_calibration(
        settlements_path,
        calibration_arg=args.calibration,
        skip=args.no_calibration,
    )
    _emit(cal_notes)
    if cal_path is not None:
        try:
            cal_loaded = load_calibration(cal_path)
        except pd.errors.EmptyDataError:
            print(f"[Warn] Calibration file is empty: {cal_path}")
            cal_loaded = None
        except pd.errors.ParserError as exc:
            print(f"[Warn] Could not parse calibration file {cal_path}: {exc}")
            cal_loaded = None
        if cal_loaded is not None:
            _emit(cal_loaded.messages)
            if cal_loaded.rows_kept == 0:
                print("[Warn] No calibration rows left to analyze.")
            else:
                bins_t, bins_n, metrics = analyze_calibration(cal_loaded.df)
                print_calibration_report(cal_loaded.df, bins_t, metrics)
                bins_t.to_csv(out_dir / "calibration_bins_student_t.csv", index=False, encoding="utf-8-sig")
                bins_n.to_csv(out_dir / "calibration_bins_normal.csv", index=False, encoding="utf-8-sig")
                plot_calibration(
                    cal_loaded.df,
                    bins_t,
                    bins_n,
                    metrics,
                    out_dir / "calibration_report.png",
                    has_cjk,
                )
                produced = True

    order_paths, order_notes = resolve_orders(
        settlements_path,
        orders_arg=args.orders,
        skip=args.no_orders,
    )
    _emit(order_notes)
    for order_path in order_paths:
        try:
            print(format_order_audit(audit_orders(order_path)))
        except pd.errors.EmptyDataError:
            print(f"[Warn] Orders file is empty: {order_path}")
        except pd.errors.ParserError as exc:
            print(f"[Warn] Could not parse orders file {order_path}: {exc}")

    if args.show and produced:
        backend = matplotlib.get_backend().lower()
        if backend == "agg":
            print("[Note] --show was set, but the matplotlib backend is Agg (headless). Figures were saved to disk.")
        else:
            import matplotlib.pyplot as plt
            plt.show()
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
