"""Unit tests for outcome_review.py. Fixtures are small paper/backtest exports."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

import outcome_review as review
from outcome_paper_export import SETTLE_FIELDS


ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests" / "fixtures"
PAPER_CSV = FIXTURES / "paper" / "paper_settlements.csv"
PAPER_ORDERS = FIXTURES / "paper" / "paper_orders.csv"
BT_CSV = FIXTURES / "bt" / "bt_trades.csv"
BT_CAL = FIXTURES / "bt" / "bt_calibration.csv"
BT_REJECTED = FIXTURES / "bt" / "bt_rejected.csv"
OVERLAP_CSV = FIXTURES / "overlap" / "paper_settlements.csv"
DROPPED_CSV = FIXTURES / "dropped" / "settlements.csv"
NO_FEE_CSV = FIXTURES / "no_fee" / "settlements.csv"
BLANK_FEE_CSV = FIXTURES / "blank_fee" / "settlements.csv"

BT_TRADE_FIELDS = [
    "outcome_id", "underlying", "signal_utc", "exec_utc", "side", "strike",
    "mark_px_signal", "settled_twap", "ann_vol", "tau_sec", "z_score", "fair_prob",
    "top_ask_signal", "fill_vwap", "slippage", "shares", "cost_usdc", "eff_settle_fee",
    "won", "pnl_usdc", "equity_after",
]
CAL_FIELDS = [
    "outcome_id", "ts_utc", "tau_sec", "mark_px", "strike", "ann_vol", "z_score",
    "p_up_student_t", "p_up_normal", "outcome_up",
]


def _header(path: Path) -> list:
    return path.read_text(encoding="utf-8").splitlines()[0].split(",")


def _run(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = review.main(args)
    return code, buf.getvalue()


class ParserAndDocsTests(unittest.TestCase):
    def test_parser_does_not_default_mixed_directories(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            review.build_parser().parse_args([])
        self.assertEqual(ctx.exception.code, 2)

        args = review.build_parser().parse_args(["--settlements", "x.csv"])
        self.assertIsNone(args.calibration)
        self.assertIsNone(args.orders)
        self.assertIsNone(args.summary)
        self.assertIsNone(args.initial_equity)
        self.assertFalse(args.no_calibration)
        self.assertFalse(args.no_orders)

        source = (ROOT / "outcome_review.py").read_text(encoding="utf-8")
        self.assertNotIn("sort_values", source)
        self.assertNotIn("paper_logs", source)
        self.assertNotIn("bt_exports", source)
        self.assertNotIn("fillna(0.0007)", source)
        self.assertNotIn('["eff_settle_fee"] = 0.0007', source)

        help_text = review.build_parser().format_help()
        self.assertIn("settlement-time equity_after", help_text)
        self.assertIn("not sorted by exec_utc", help_text)
        self.assertIn("0.0007", help_text)
        self.assertNotIn("paper_logs", help_text)
        self.assertNotIn("bt_exports", help_text)

    def test_requirements_and_readme(self):
        lines = [
            line.strip()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn("matplotlib", lines)
        self.assertIn("pandas", lines)
        self.assertFalse(any(line.startswith("matplotlib") and any(tok in line for tok in ("=", ">", "<")) for line in lines))

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("outcome_review.py", readme)
        self.assertIn("pip install -r requirements.txt", readme)
        self.assertIn("matplotlib", readme)
        self.assertIn("not sorted by exec_utc", readme)
        self.assertIn("settlement-time equity_after", readme)
        self.assertIn("0.0007", readme)
        self.assertIn("fee_scale", readme)


class TauAndScoreTests(unittest.TestCase):
    def test_tau_bucket_edges(self):
        values = [0, 60, 61, 180, 181, 300, 301, 600, 601]
        expected = [
            review.TAU_LABELS_EN[0],
            review.TAU_LABELS_EN[0],
            review.TAU_LABELS_EN[1],
            review.TAU_LABELS_EN[1],
            review.TAU_LABELS_EN[2],
            review.TAU_LABELS_EN[2],
            review.TAU_LABELS_EN[3],
            review.TAU_LABELS_EN[3],
            review.TAU_LABELS_EN[4],
        ]
        got = [str(item) for item in review.assign_tau_bucket(pd.Series(values))]
        self.assertEqual(got, expected)

    def test_ece_and_log_loss(self):
        probs = np.array([0.1, 0.9])
        outcomes = np.array([0.0, 1.0])
        self.assertAlmostEqual(review.calc_ece(probs, outcomes, n_bins=10), 0.1, places=6)
        expected_ll = -0.5 * (math.log(0.9) + math.log(0.8))
        cal_probs = np.array([0.9, 0.8, 0.2, 0.1])
        cal_y = np.array([1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(review.calc_log_loss(cal_probs, cal_y), expected_ll, places=6)

    def test_matplotlib_backend_is_agg(self):
        found = review.setup_matplotlib(False)
        self.assertIsInstance(found, bool)
        self.assertEqual(review.matplotlib.get_backend().lower(), "agg")


class PaperSettlementTests(unittest.TestCase):
    def test_paper_columns_edges_and_summary_equity(self):
        self.assertEqual(_header(PAPER_CSV), list(SETTLE_FIELDS))
        loaded = review.load_settlements(PAPER_CSV)
        self.assertEqual(loaded.rows_dropped, 0)
        self.assertTrue(loaded.fee_column_present)
        self.assertNotIn(review.MISSING_FEE_WARNING, loaded.messages)
        self.assertIn("filled_shares", loaded.df.columns)
        self.assertNotIn("shares", loaded.df.columns)
        self.assertIn("settle_time_utc", loaded.df.columns)
        self.assertNotIn("exec_utc", loaded.df.columns)
        self.assertAlmostEqual(loaded.df.iloc[0]["eff_settle_fee"], 0.001)
        self.assertAlmostEqual(loaded.df.iloc[0]["realized_edge"], 0.1994, places=6)
        self.assertAlmostEqual(loaded.df.iloc[0]["signal_edge"], 0.2094, places=6)
        self.assertAlmostEqual(loaded.df.iloc[0]["slip_cost_usdc"], 0.1, places=6)
        self.assertAlmostEqual(loaded.df.iloc[1]["slip_cost_usdc"], 0.4, places=6)

        resolution = review.resolve_initial_equity(loaded.df, PAPER_CSV, None, None)
        self.assertEqual(resolution.source, "summary")
        self.assertAlmostEqual(resolution.initial_equity, 1000.0)
        summary, grouped = review.analyze_settlements(loaded.df, resolution.initial_equity)
        self.assertAlmostEqual(summary["final_eq"], 995.99, places=4)
        self.assertAlmostEqual(summary["total_pnl"], -4.01, places=4)
        self.assertAlmostEqual(summary["total_slip_usdc"], 0.5, places=6)
        self.assertAlmostEqual(summary["zero_slip_total_pnl"], -3.51, places=4)
        self.assertAlmostEqual(summary["profit_factor"], 0.599, places=3)
        self.assertAlmostEqual(summary["max_dd"] * 100.0, 0.994045, places=3)
        np.testing.assert_allclose(summary["eq_series"], [1000.0, 1005.99, 995.99], rtol=0, atol=1e-6)
        np.testing.assert_allclose(summary["zero_slip_eq"], [1000.0, 1006.09, 996.49], rtol=0, atol=1e-6)
        self.assertLessEqual(summary["equity_path_gap"], review.EQUITY_PATH_GAP_TOL)
        self.assertIsNone(review.equity_path_note(summary["equity_path_gap"]))
        self.assertEqual(grouped["tau_bucket"].tolist(), review.TAU_LABELS_EN)
        first = grouped.iloc[0]
        second = grouped.iloc[1]
        self.assertEqual(int(first["trades"]), 1)
        self.assertAlmostEqual(first["pnl_usdc"], 5.99, places=4)
        self.assertEqual(int(second["trades"]), 1)
        self.assertAlmostEqual(second["pnl_usdc"], -10.0, places=4)
        self.assertEqual(review.cross_check_summary(resolution.summary, summary["final_eq"], summary["max_dd"]), [])

        sides = review.side_breakdown(loaded.df)
        self.assertEqual(int(sides.loc["UP", "trades"]), 1)
        self.assertEqual(int(sides.loc["DOWN", "trades"]), 1)

    def test_orders_audit_counts_fills_and_reasons(self):
        result = review.audit_orders(PAPER_ORDERS)
        self.assertEqual(result["kind"], "orders")
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["filled"], 2)
        self.assertEqual(result["reasons"]["SLIPPAGE_EXCEEDED"], 1)
        self.assertEqual(result["reasons"]["NO_BOOK"], 1)


class BacktestTradeTests(unittest.TestCase):
    def test_bt_aliases_keep_file_order(self):
        self.assertEqual(_header(BT_CSV), BT_TRADE_FIELDS)
        raw = BT_CSV.read_text(encoding="utf-8").splitlines()[1:]
        self.assertGreater(raw[0].split(",")[3], raw[1].split(",")[3])

        loaded = review.load_settlements(BT_CSV)
        self.assertNotIn("shares", loaded.df.columns)
        self.assertNotIn("exec_utc", loaded.df.columns)
        self.assertIn("filled_shares", loaded.df.columns)
        self.assertIn("settle_time_utc", loaded.df.columns)
        self.assertEqual([float(v) for v in loaded.df["filled_shares"]], [10.0, 20.0])
        equities = [float(v) for v in loaded.df["equity_after"]]
        self.assertAlmostEqual(equities[0], 1005.99, places=4)
        self.assertAlmostEqual(equities[1], 995.99, places=4)
        times = [str(v) for v in loaded.df["settle_time_utc"]]
        self.assertGreater(times[0], times[1])

        resolution = review.resolve_initial_equity(loaded.df, BT_CSV, None, None)
        self.assertEqual(resolution.source, "summary")
        self.assertEqual(resolution.summary_path.name, "bt_summary.json")
        self.assertAlmostEqual(resolution.initial_equity, 1000.0)
        summary, _grouped = review.analyze_settlements(loaded.df, resolution.initial_equity)
        self.assertAlmostEqual(summary["final_eq"], 995.99, places=4)
        np.testing.assert_allclose(summary["eq_series"], [1000.0, 1005.99, 995.99], rtol=0, atol=1e-6)

    def test_calibration_columns_and_scores(self):
        self.assertEqual(_header(BT_CAL), CAL_FIELDS)
        loaded = review.load_calibration(BT_CAL)
        self.assertEqual(loaded.rows_kept, 4)
        self.assertEqual(loaded.rows_dropped, 0)
        for column in ("p_up_student_t", "p_up_normal", "outcome_up", "outcome_id"):
            self.assertIn(column, loaded.df.columns)
        _bins_t, _bins_n, metrics = review.analyze_calibration(loaded.df)
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["n_outcomes"], 2)
        self.assertAlmostEqual(metrics["brier_t"], 0.025, places=6)
        self.assertAlmostEqual(metrics["brier_n"], 0.125, places=6)
        self.assertLess(metrics["brier_t"], metrics["brier_n"])
        self.assertEqual(metrics["tail_n"], 2)
        self.assertAlmostEqual(metrics["tail_brier_t"], 0.01, places=6)
        self.assertAlmostEqual(metrics["tail_brier_n"], 0.125, places=6)
        expected_ll = -0.5 * (math.log(0.9) + math.log(0.8))
        self.assertAlmostEqual(metrics["log_loss_t"], expected_ll, places=6)

        rejected = review.audit_orders(BT_REJECTED)
        self.assertEqual(rejected["kind"], "rejected")
        self.assertEqual(rejected["rows"], 3)
        self.assertIsNone(rejected["filled"])
        self.assertEqual(rejected["reasons"]["SLIPPAGE_EXCEEDED"], 2)
        self.assertEqual(rejected["reasons"]["NO_LIQUIDITY"], 1)

    def test_blank_calibration_row_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bt_calibration.csv"
            text = BT_CAL.read_text(encoding="utf-8") + "3,2024-06-01T00:05:00.000Z,10,1,1,0.5,0,,0.2,1\n"
            path.write_text(text, encoding="utf-8")
            loaded = review.load_calibration(path)
        self.assertEqual(loaded.rows_in, 5)
        self.assertEqual(loaded.rows_dropped, 1)
        self.assertEqual(loaded.missing_counts["p_up_student_t"], 1)
        self.assertEqual(loaded.rows_kept, 4)
        self.assertTrue(any("Dropped 1 of 5" in message for message in loaded.messages))


class OverlapEquityTests(unittest.TestCase):
    def test_summary_beats_wrong_inferred_equity(self):
        loaded = review.load_settlements(OVERLAP_CSV)
        inferred = review.infer_initial_equity(loaded.df)
        self.assertAlmostEqual(inferred, 900.0, places=4)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "paper_settlements.csv"
            dest.write_bytes(OVERLAP_CSV.read_bytes())
            inferred_resolution = review.resolve_initial_equity(loaded.df, dest, None, None)
        self.assertEqual(inferred_resolution.source, "inferred")
        self.assertAlmostEqual(inferred_resolution.initial_equity, 900.0, places=4)
        self.assertTrue(any("too low" in note for note in inferred_resolution.notes))
        self.assertTrue(any("900.0000" in note for note in inferred_resolution.notes))

        resolution = review.resolve_initial_equity(loaded.df, OVERLAP_CSV, None, 5000.0)
        self.assertEqual(resolution.source, "summary")
        self.assertAlmostEqual(resolution.initial_equity, 1000.0, places=4)
        self.assertTrue(any("not used" in note and "5000.0000" in note for note in resolution.notes))
        self.assertFalse(any("too low" in note for note in resolution.notes))

        summary, _grouped = review.analyze_settlements(loaded.df, resolution.initial_equity)
        self.assertAlmostEqual(summary["final_eq"], 959.0, places=4)
        self.assertAlmostEqual(summary["max_dd"] * 100.0, 4.1, places=2)
        self.assertAlmostEqual(summary["total_pnl"], -41.0, places=4)
        self.assertAlmostEqual(summary["total_slip_usdc"], 1.3, places=6)
        self.assertAlmostEqual(summary["zero_slip_total_pnl"], -39.7, places=4)
        self.assertGreater(summary["equity_path_gap"], 50.0)
        self.assertIn("not the starting equity", review.equity_path_note(summary["equity_path_gap"]))
        notes = review.cross_check_summary(resolution.summary, summary["final_eq"], summary["max_dd"])
        self.assertTrue(any("max_drawdown_pct=14.0000" in note and "differs" in note for note in notes))
        self.assertFalse(any("final_equity=" in note and "differs" in note for note in notes))

    def test_cross_check_notes_open_final_equity(self):
        notes = review.cross_check_summary(
            {"final_equity": 859.0, "max_drawdown_pct": 14.0},
            final_eq=959.0,
            max_dd=0.041,
        )
        self.assertTrue(any("final_equity=859.0000" in note for note in notes))
        self.assertTrue(any("settlement-time MaxDD" in note for note in notes))
        quiet = review.cross_check_summary(
            {"final_equity": 959.01, "max_drawdown_pct": 4.12},
            final_eq=959.0,
            max_dd=0.041,
        )
        self.assertEqual(quiet, [])


class FeeAndDropTests(unittest.TestCase):
    def test_missing_fee_is_not_filled(self):
        loaded = review.load_settlements(NO_FEE_CSV)
        self.assertFalse(loaded.fee_column_present)
        self.assertEqual(loaded.rows_dropped, 0)
        self.assertEqual(loaded.rows_kept, 1)
        self.assertIn(review.MISSING_FEE_WARNING, loaded.messages)
        self.assertTrue(loaded.df["eff_settle_fee"].isna().all())
        self.assertTrue(loaded.df["realized_edge"].isna().all())
        self.assertTrue(loaded.df["signal_edge"].isna().all())
        forbidden = 0.6 * (1.0 - 0.0007) - 0.4
        self.assertFalse(np.isclose(loaded.df["realized_edge"].fillna(999.0), forbidden).any())
        self.assertFalse((loaded.df["eff_settle_fee"] == 0.0007).fillna(False).any())

    def test_blank_fee_cells_are_not_filled(self):
        loaded = review.load_settlements(BLANK_FEE_CSV)
        self.assertTrue(loaded.fee_column_present)
        self.assertEqual(loaded.rows_dropped, 0)
        self.assertNotIn(review.MISSING_FEE_WARNING, loaded.messages)
        self.assertIn(review.BLANK_FEE_TEMPLATE.format(n=1), loaded.messages)
        self.assertAlmostEqual(loaded.df.iloc[0]["eff_settle_fee"], 0.01)
        self.assertTrue(math.isnan(loaded.df.iloc[1]["eff_settle_fee"]))
        self.assertAlmostEqual(loaded.df.iloc[0]["realized_edge"], 0.194, places=6)
        self.assertTrue(math.isnan(loaded.df.iloc[1]["realized_edge"]))
        self.assertFalse((loaded.df["eff_settle_fee"] == 0.0007).fillna(False).any())

    def test_present_fee_of_0007_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settlements.csv"
            path.write_text(
                "pnl_usdc,equity_after,tau_sec,won,fair_prob,filled_shares,fill_vwap,slippage,eff_settle_fee\n"
                "1,1001,30,1,0.6,10,0.4,0.01,0.0007\n",
                encoding="utf-8",
            )
            loaded = review.load_settlements(path)
        self.assertAlmostEqual(loaded.df.iloc[0]["eff_settle_fee"], 0.0007)
        self.assertAlmostEqual(loaded.df.iloc[0]["realized_edge"], 0.6 * (1.0 - 0.0007) - 0.4, places=6)
        self.assertNotIn(review.MISSING_FEE_WARNING, loaded.messages)

    def test_dropped_rows_are_counted(self):
        loaded = review.load_settlements(DROPPED_CSV)
        self.assertEqual(loaded.rows_in, 6)
        self.assertEqual(loaded.rows_dropped, 4)
        self.assertEqual(loaded.rows_kept, 2)
        self.assertEqual(loaded.missing_counts["pnl_usdc"], 1)
        self.assertEqual(loaded.missing_counts["equity_after"], 2)
        self.assertEqual(loaded.missing_counts["tau_sec"], 2)
        self.assertEqual(list(loaded.df["order_id"]), ["KEEP1", "KEEP2"])
        self.assertIn(
            review.DROPPED_TEMPLATE.format(dropped=4, total=6, n_pnl=1, n_eq=2, n_tau=2),
            loaded.messages,
        )
        self.assertTrue(loaded.fee_column_present)
        self.assertNotIn(review.MISSING_FEE_WARNING, loaded.messages)

    def test_missing_required_column_is_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settlements.csv"
            path.write_text("equity_after,tau_sec\n1001,30\n1002,40\n", encoding="utf-8")
            loaded = review.load_settlements(path)
        self.assertEqual(loaded.rows_kept, 0)
        self.assertEqual(loaded.rows_dropped, 2)
        self.assertIn(review.COLUMN_MISSING_TEMPLATE.format(col="pnl_usdc"), loaded.messages)
        self.assertTrue(any("Dropped 2 of 2" in message for message in loaded.messages))

    def test_missing_slippage_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settlements.csv"
            path.write_text(
                "pnl_usdc,equity_after,tau_sec,filled_shares,fill_vwap,eff_settle_fee\n"
                "1,1001,30,10,0.4,0.001\n",
                encoding="utf-8",
            )
            loaded = review.load_settlements(path)
        self.assertTrue(loaded.df["slippage"].isna().all())
        self.assertTrue(loaded.df["slip_cost_usdc"].isna().all())
        self.assertTrue(loaded.df["zero_slip_pnl"].isna().all())
        self.assertTrue(any("not assuming 0" in message for message in loaded.messages))

    def test_utf8_sig_bom_does_not_rename_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_settlements.csv"
            text = "order_id,pnl_usdc,equity_after,tau_sec,eff_settle_fee\nA,1,1001,10,0.001\n"
            path.write_bytes(text.encode("utf-8-sig"))
            loaded = review.load_settlements(path)
        self.assertIn("order_id", loaded.df.columns)
        self.assertNotIn("\ufefforder_id", "".join(loaded.df.columns))
        self.assertEqual(loaded.rows_kept, 1)


class PathResolutionTests(unittest.TestCase):
    def test_family_specific_summary_and_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_csv = root / "paper_settlements.csv"
            bt_csv = root / "bt_trades.csv"
            unknown = root / "settlements.csv"
            paper_csv.write_text("pnl_usdc,equity_after,tau_sec\n1,101,10\n", encoding="utf-8")
            bt_csv.write_text("pnl_usdc,equity_after,tau_sec\n1,101,10\n", encoding="utf-8")
            unknown.write_text("pnl_usdc,equity_after,tau_sec\n1,101,10\n", encoding="utf-8")
            (root / "paper_summary.json").write_text(json.dumps({"initial_equity": 1000}), encoding="utf-8")
            (root / "bt_summary.json").write_text(json.dumps({"initial_equity": 777}), encoding="utf-8")
            (root / "paper_orders.csv").write_text("status\nFILLED\n", encoding="utf-8")
            (root / "bt_rejected.csv").write_text("reason\nNO_LIQUIDITY\n", encoding="utf-8")
            (root / "bt_calibration.csv").write_text("p_up_student_t,p_up_normal,outcome_up\n0.5,0.5,1\n", encoding="utf-8")
            frame = pd.DataFrame({"pnl_usdc": [1.0], "equity_after": [101.0], "tau_sec": [10.0]})

            paper_res = review.resolve_initial_equity(frame, paper_csv, None, None)
            self.assertAlmostEqual(paper_res.initial_equity, 1000.0)
            self.assertEqual(paper_res.summary_path.name, "paper_summary.json")
            bt_res = review.resolve_initial_equity(frame, bt_csv, None, None)
            self.assertAlmostEqual(bt_res.initial_equity, 777.0)
            self.assertEqual(bt_res.summary_path.name, "bt_summary.json")
            unknown_res = review.resolve_initial_equity(frame, unknown, None, None)
            self.assertEqual(unknown_res.source, "inferred")
            self.assertTrue(any("Pass --summary" in note for note in unknown_res.notes))

            paper_cal, paper_cal_notes = review.resolve_calibration(paper_csv, None, False)
            self.assertIsNone(paper_cal)
            self.assertIn(review.CALIBRATION_PAPER_MIX, paper_cal_notes)
            paper_orders, _paper_order_notes = review.resolve_orders(paper_csv, None, False)
            self.assertEqual([path.name for path in paper_orders], ["paper_orders.csv"])

            bt_cal, bt_cal_notes = review.resolve_calibration(bt_csv, None, False)
            self.assertEqual(bt_cal.name, "bt_calibration.csv")
            self.assertTrue(any("sibling calibration" in note for note in bt_cal_notes))
            bt_orders, _bt_order_notes = review.resolve_orders(bt_csv, None, False)
            self.assertEqual([path.name for path in bt_orders], ["bt_rejected.csv"])

            explicit = root / "elsewhere.csv"
            explicit.write_text("p_up_student_t,p_up_normal,outcome_up\n0.2,0.3,0\n", encoding="utf-8")
            chosen, _notes = review.resolve_calibration(paper_csv, str(explicit), False)
            self.assertEqual(chosen, explicit)
            skipped, skip_notes = review.resolve_calibration(bt_csv, str(explicit), True)
            self.assertIsNone(skipped)
            self.assertTrue(any("--no-calibration" in note for note in skip_notes))

    def test_explicit_summary_and_flag_fallback(self):
        frame = pd.DataFrame({"pnl_usdc": [5.99], "equity_after": [1005.99], "tau_sec": [30.0]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "paper_settlements.csv"
            csv_path.write_text("pnl_usdc,equity_after,tau_sec\n5.99,1005.99,30\n", encoding="utf-8")
            (root / "paper_summary.json").write_text(json.dumps({"initial_equity": 1000}), encoding="utf-8")
            other = root / "other.json"
            other.write_text(json.dumps({"initial_equity": 4242}), encoding="utf-8")
            explicit = review.resolve_initial_equity(frame, csv_path, str(other), None)
            self.assertAlmostEqual(explicit.initial_equity, 4242.0)
            missing = review.resolve_initial_equity(frame, csv_path, str(root / "missing.json"), None)
            self.assertAlmostEqual(missing.initial_equity, 1000.0)
            self.assertTrue(any("not found" in note for note in missing.notes))

        resolution = review.resolve_initial_equity(frame, NO_FEE_CSV, None, 2500.0)
        self.assertEqual(resolution.source, "flag")
        self.assertAlmostEqual(resolution.initial_equity, 2500.0)
        bare = review.resolve_initial_equity(frame, NO_FEE_CSV, None, None)
        self.assertEqual(bare.source, "inferred")
        self.assertAlmostEqual(bare.initial_equity, 1000.0, places=4)
        self.assertTrue(any("too low" in note for note in bare.notes))

        none_cal, notes = review.resolve_calibration(NO_FEE_CSV, None, False)
        self.assertIsNone(none_cal)
        self.assertIn(review.CALIBRATION_OFF, notes)


class CliTests(unittest.TestCase):
    def test_overlap_cli_uses_summary_and_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = _run(["--settlements", str(OVERLAP_CSV), "--out-dir", tmp])
            png = Path(tmp) / "equity_and_tau_report.png"
            self.assertEqual(code, 0)
            self.assertTrue(png.is_file())
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("Initial equity 1000.0000 from paper_summary.json", text)
        self.assertIn("Equity 1000.00 -> 959.00", text)
        self.assertIn("settlement-time equity_after", text)
        self.assertIn("not sorted by exec_utc", text)
        self.assertIn("max_drawdown_pct=14.0000", text)
        self.assertIn("not the starting equity", text)
        self.assertNotIn("900.0000", text)
        self.assertNotIn("too low", text)

    def test_bt_cli_uses_sibling_calibration_without_sorting(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = _run(["--settlements", str(BT_CSV), "--out-dir", tmp])
            self.assertEqual(code, 0)
            self.assertEqual((Path(tmp) / "calibration_report.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue((Path(tmp) / "calibration_bins_student_t.csv").is_file())
        self.assertIn("bt_summary.json", text)
        self.assertIn("Using sibling calibration file", text)
        self.assertIn("Brier=", text)
        self.assertIn("Student-t", text)
        self.assertIn("Rejected rows: 3", text)
        self.assertIn("NO_LIQUIDITY", text)
        self.assertIn("Equity 1000.00 -> 995.99", text)

    def test_paper_cli_does_not_mix_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = _run(["--settlements", str(PAPER_CSV), "--out-dir", tmp])
            self.assertEqual(code, 0)
            self.assertFalse((Path(tmp) / "calibration_report.png").exists())
        self.assertIn(review.CALIBRATION_OFF, text)
        self.assertNotIn("will not mix", text)
        self.assertIn("Filled: 2", text)
        self.assertIn("SLIPPAGE_EXCEEDED", text)
        self.assertNotIn("too low", text)

    def test_dropped_and_missing_fee_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = _run([
                "--settlements", str(DROPPED_CSV),
                "--initial-equity", "5000",
                "--out-dir", tmp,
            ])
        self.assertEqual(code, 0)
        self.assertIn("Dropped 4 of 6", text)
        self.assertIn("pnl_usdc (1)", text)
        self.assertIn("equity_after (2)", text)
        self.assertIn("tau_sec (2)", text)
        self.assertIn("Equity 5000.00 -> 1005.00", text)
        self.assertNotIn("too low", text)

        with tempfile.TemporaryDirectory() as tmp:
            code, text = _run(["--settlements", str(NO_FEE_CSV), "--out-dir", tmp])
        self.assertEqual(code, 0)
        self.assertIn("not filling 0.0007", text)
        self.assertIn("too low", text)
        self.assertIn("1000.0000", text)

        with tempfile.TemporaryDirectory() as tmp:
            code, text = _run([
                "--settlements", str(NO_FEE_CSV),
                "--initial-equity", "2500",
                "--out-dir", tmp,
            ])
        self.assertEqual(code, 0)
        self.assertIn("not filling 0.0007", text)
        self.assertIn("2500.0000 from --initial-equity", text)
        self.assertIn("Equity 2500.00 -> 1005.99", text)
        self.assertNotIn("too low", text)

    def test_missing_settlements_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = _run(["--settlements", str(Path(tmp) / "missing.csv")])
        self.assertEqual(code, 1)
        self.assertIn("not found", text)


if __name__ == "__main__":
    unittest.main()
