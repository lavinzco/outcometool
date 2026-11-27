import tempfile
import unittest
from pathlib import Path

from paper_bookkeeping import (
    LiveCSVLogger,
    apply_fill_equity,
    apply_payout_equity,
    build_settlement_row,
    classify_positions,
    realized_pnl,
    update_peak_dd,
    utc_iso,
)


class UtcIsoTests(unittest.TestCase):
    def test_format_has_t_and_z(self):
        text = utc_iso(0.0)
        self.assertTrue(text.startswith("1970-01-01T"))
        self.assertTrue(text.endswith("Z"))
        self.assertNotIn(" ", text)


class EquityTests(unittest.TestCase):
    def test_fill_reserves_cost_then_payout_credits_only(self):
        equity = 1000.0
        cost = 40.0
        payout = 59.958  # won, 60 shares * (1 - 0.0007)
        equity = apply_fill_equity(equity, cost)
        self.assertAlmostEqual(equity, 960.0)
        equity = apply_payout_equity(equity, payout)
        self.assertAlmostEqual(equity, 1019.958)
        self.assertAlmostEqual(realized_pnl(payout, cost), payout - cost)

    def test_loss_payout_is_zero(self):
        equity = apply_fill_equity(1000.0, 25.0)
        equity = apply_payout_equity(equity, 0.0)
        self.assertAlmostEqual(equity, 975.0)

    def test_drawdown_after_fill(self):
        peak, max_dd = update_peak_dd(960.0, 1000.0, 0.0)
        self.assertEqual(peak, 1000.0)
        self.assertAlmostEqual(max_dd, 0.04)


class ClassifyPositionsTests(unittest.TestCase):
    def test_not_expired_stays_open(self):
        pos = {"expiry_ts": 100.0, "order_id": "A"}
        keep, settleable = classify_positions([pos], now=101.0, get_twap=lambda _: 1.0)
        self.assertEqual(keep, [pos])
        self.assertEqual(settleable, [])

    def test_missing_twap_is_retried_not_dropped(self):
        pos = {"expiry_ts": 100.0, "order_id": "B"}
        keep, settleable = classify_positions([pos], now=110.0, get_twap=lambda _: None)
        self.assertEqual(keep, [pos])
        self.assertEqual(settleable, [])

    def test_expired_with_twap_is_settleable(self):
        pos = {"expiry_ts": 100.0, "order_id": "C"}
        keep, settleable = classify_positions([pos], now=110.0, get_twap=lambda _: 91.25)
        self.assertEqual(keep, [])
        self.assertEqual(len(settleable), 1)
        self.assertEqual(settleable[0][0], pos)
        self.assertEqual(settleable[0][1], 91.25)


class SettlementRowTests(unittest.TestCase):
    def _pos(self, side="UP"):
        return {
            "order_id": "PAPER-00001",
            "outcome_id": "1",
            "underlying": "BTC",
            "side": side,
            "strike": 90.0,
            "filled_shares": 10,
            "fill_vwap": 0.4,
            "cost_usdc": 4.0,
            "eff_settle_fee": 0.0007,
            "fair_prob": 0.6,
            "tau_sec": 30.0,
            "slippage": 0.001,
        }

    def test_up_win_pays_net_of_fee(self):
        row = build_settlement_row(self._pos("UP"), twap=91.0, now=0.0, equity_after=1005.993)
        self.assertEqual(row["outcome_up"], 1)
        self.assertEqual(row["won"], 1)
        self.assertAlmostEqual(row["payout_usdc"], 9.993)
        self.assertAlmostEqual(row["pnl_usdc"], 5.993)

    def test_up_loss_zero_payout(self):
        row = build_settlement_row(self._pos("UP"), twap=89.0, now=0.0, equity_after=996.0)
        self.assertEqual(row["outcome_up"], 0)
        self.assertEqual(row["won"], 0)
        self.assertEqual(row["payout_usdc"], 0.0)
        self.assertAlmostEqual(row["pnl_usdc"], -4.0)


class CsvLoggerTests(unittest.TestCase):
    def test_missing_fields_written_as_empty(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.csv"
            logger = LiveCSVLogger(path, ["a", "b", "c"])
            logger.append({"a": "1", "extra": "ignored"})
            text = path.read_text(encoding="utf-8-sig")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            self.assertEqual(lines[0], "a,b,c")
            self.assertEqual(lines[1], "1,,")


if __name__ == "__main__":
    unittest.main()
