import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import outcome_hip4_engine as engine
import outcome_paper_export as paper_export
from market_data import BookEvent, MarkEvent, MarketTapeWriter
from outcome_hip4_engine import (
    HIP4AsianBinaryStrategy,
    OutcomeMarket,
    StrategyConfig,
    compute_strict_twap,
    simulate_clob_buy_fill,
    simulate_clob_buy_fill_detailed,
)


def make_market(oid: str = "900", strike: float = 78000.0, expiry_ts: float = 1_800_000_900.0) -> OutcomeMarket:
    oid_int = int(oid)
    return OutcomeMarket(
        outcome_id=oid,
        name=f"BTC-15m-{oid}",
        underlying="BTC",
        strike=strike,
        expiry_ts=expiry_ts,
        period="15m",
        fee_scale=1.0,
        yes_coin=f"#{oid}0",
        no_coin=f"#{oid}1",
        yes_asset_id=100_000_000 + oid_int * 10,
        no_asset_id=100_000_000 + oid_int * 10 + 1,
        raw_description=f"class:priceBinary|underlying:BTC|expiry:20260503-0600|targetPrice:{strike}|period:15m",
    )


class TestSingleEvaluationAndFill(unittest.TestCase):
    def test_single_strategy_class_and_evaluate_never_consumes_quota(self):
        """验收 1/2：paper 与 engine 共享同一策略类；evaluate / evaluate_with_audit 不递增 market_trades。"""
        self.assertIs(paper_export.InstrumentedHIP4Strategy, HIP4AsianBinaryStrategy)

        cfg = StrategyConfig(initial_equity=1000.0)
        strat = HIP4AsianBinaryStrategy(cfg)
        expiry = 1_800_000_900.0
        m = make_market("901", strike=78000.0, expiry_ts=expiry)

        sig = strat.evaluate(
            market=m,
            ts=expiry - 200.0,
            mark_px=78800.0,
            ann_vol=0.40,
            yes_bids=[(0.50, 100.0)],
            yes_asks=[(0.52, 100.0)],
            acc_twap=None,
            equity=1000.0,
        )
        self.assertIsNotNone(sig)
        self.assertEqual(sig["side"], "UP")
        self.assertEqual(strat.market_trades.get("901", 0), 0)

        dec = strat.evaluate_with_audit(
            market=m,
            ts=expiry - 200.0,
            mark_px=78800.0,
            ann_vol=0.40,
            yes_bids=[(0.50, 100.0)],
            yes_asks=[(0.52, 100.0)],
            book_ts=expiry - 200.0,
            acc_twap=None,
            equity=1000.0,
            peak_equity=1000.0,
        )
        self.assertTrue(dec.should_trade)
        self.assertEqual(strat.market_trades.get("901", 0), 0)

        # 只有 record_fill 后才占额度
        strat.record_fill("901")
        self.assertEqual(strat.market_trades["901"], 1)
        dec2 = strat.evaluate_with_audit(
            market=m,
            ts=expiry - 199.0,
            mark_px=78800.0,
            ann_vol=0.40,
            yes_bids=[(0.50, 100.0)],
            yes_asks=[(0.52, 100.0)],
            book_ts=expiry - 199.0,
            acc_twap=None,
            equity=970.0,
            peak_equity=1000.0,
        )
        self.assertFalse(dec2.should_trade)
        self.assertEqual(dec2.filter_reason, "MAX_TRADES_REACHED")

    def test_missing_side_returns_none_not_one(self):
        """4.5 差异收敛：缺单边盘口时该侧 ask 与 edge 为 None，不构造 1.0。"""
        strat = HIP4AsianBinaryStrategy(StrategyConfig())
        expiry = 1_800_000_900.0
        m = make_market("902", strike=78000.0, expiry_ts=expiry)
        dec = strat.evaluate_with_audit(
            market=m,
            ts=expiry - 200.0,
            mark_px=78800.0,
            ann_vol=0.40,
            yes_bids=[(0.50, 100.0)],
            yes_asks=[],
            book_ts=expiry - 200.0,
            acc_twap=None,
            equity=1000.0,
            peak_equity=1000.0,
        )
        self.assertIsNone(dec.up_ask)
        self.assertIsNone(dec.edge_up)
        self.assertAlmostEqual(dec.down_ask, 0.50)

    def test_fill_wrapper_matches_detailed_fill_for_both_sides(self):
        """验收 1：simulate_clob_buy_fill 与 simulate_clob_buy_fill_detailed 结果完全一致。"""
        bids = [(0.55, 10.0), (0.50, 40.0)]
        asks = [(0.57, 10.0), (0.60, 40.0)]

        for side in ("UP", "DOWN"):
            det = simulate_clob_buy_fill_detailed(side, 20.0, bids, asks, min_notional=1.0)
            thin = simulate_clob_buy_fill(side, 20.0, bids, asks, min_notional=1.0)
            self.assertIsNotNone(det)
            self.assertEqual(thin, (det.filled_shares, det.vwap, det.cost_usdc))


class TestUnifiedLedgerAndBookkeeper(unittest.TestCase):
    def test_bookkeeper_delegates_strictly_to_paper_bookkeeping(self):
        """验收 3：EquityBookkeeper 必须调用 paper_bookkeeping 函数，赢加兑付、输不双扣成本。"""
        bk = paper_export.EquityBookkeeper(1000.0)
        self.assertEqual(bk.source, "paper_bookkeeping")

        with (
            patch("outcome_paper_export.apply_fill_equity", wraps=paper_export.apply_fill_equity) as m_fill,
            patch("outcome_paper_export.apply_payout_equity", wraps=paper_export.apply_payout_equity) as m_pay,
            patch("outcome_paper_export.update_peak_dd", wraps=paper_export.update_peak_dd) as m_dd,
        ):
            eq1, peak1, dd1 = bk.apply_fill(30.0)
            self.assertAlmostEqual(eq1, 970.0)
            self.assertAlmostEqual(peak1, 1000.0)
            self.assertAlmostEqual(dd1, 0.03)

            # 输单：payout=0，净值停在 970，不再减 30
            eq2, peak2, dd2 = bk.apply_payout(0.0)
            self.assertAlmostEqual(eq2, 970.0)
            self.assertAlmostEqual(peak2, 1000.0)
            self.assertAlmostEqual(dd2, 0.03)

            # 赢单：成交扣 20 -> 950，兑付 35 -> 985
            bk.apply_fill(20.0)
            self.assertAlmostEqual(bk.equity, 950.0)
            eq3, _, _ = bk.apply_payout(35.0)
            self.assertAlmostEqual(eq3, 985.0)

            self.assertEqual(m_fill.call_count, 2)
            self.assertEqual(m_pay.call_count, 2)
            self.assertEqual(m_dd.call_count, 4)


class TestReplayQuotaEquityAndParity(unittest.TestCase):
    def test_rejections_do_not_consume_quota_and_resignal_after_cooldown(self):
        """验收 2：NO_LIQUIDITY 与 SLIPPAGE_EXCEEDED 拒单不占额度，冷却后可重发并在 FILLED 后占额。"""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "reject_then_fill.sqlite")
            out_dir = Path(td) / "bt_out"
            expiry = 1_800_000_900.0
            m = make_market("910", strike=78000.0, expiry_ts=expiry)

            writer = MarketTapeWriter(db_path, commit_batch=1)
            writer.upsert_market(m)
            # 持续强上涨，确保每个时刻模型都给出 UP 信号
            writer.write_mark(MarkEvent("BTC", 78900.0, recv_ts=expiry - 300.0))
            writer.write_mark(MarkEvent("BTC", 78950.0, recv_ts=expiry - 40.0))
            writer.write_mark(MarkEvent("BTC", 78960.0, recv_ts=expiry - 10.0))

            # t0 = expiry - 250s: 信号触发，但延迟 1s 后盘口深度 < $1 名义 -> NO_LIQUIDITY
            t0 = expiry - 250.0
            writer.write_book(BookEvent("910", "#9100", [(0.50, 100.0)], [(0.52, 100.0)], t0, t0, "ws", t0))
            writer.write_book(BookEvent("910", "#9100", [(0.50, 100.0)], [(0.52, 1.0)], t0 + 1.0, t0 + 1.0, "ws", t0 + 1.0))

            # t0 + 2.0s: 仍在 3s 冷却期内（冷却到 t0 + 4.0s） -> REJECT_COOLDOWN，不新增拒单
            writer.write_book(BookEvent("910", "#9100", [(0.50, 100.0)], [(0.52, 100.0)], t0 + 2.0, t0 + 2.0, "ws", t0 + 2.0))

            # t1 = t0 + 5.0s: 冷却结束后再次触发信号，但延迟 1s 后卖一跳升至 0.60 -> SLIPPAGE_EXCEEDED
            t1 = t0 + 5.0
            writer.write_book(BookEvent("910", "#9100", [(0.50, 100.0)], [(0.52, 100.0)], t1, t1, "ws", t1))
            writer.write_book(BookEvent("910", "#9100", [(0.58, 100.0)], [(0.60, 100.0)], t1 + 1.0, t1 + 1.0, "ws", t1 + 1.0))

            # t2 = t1 + 5.0s: 再次冷却结束，延迟 1s 后盘口稳定可成交 -> FILLED
            t2 = t1 + 5.0
            writer.write_book(BookEvent("910", "#9100", [(0.50, 100.0)], [(0.52, 100.0)], t2, t2, "ws", t2))
            writer.write_book(BookEvent("910", "#9100", [(0.50, 100.0)], [(0.52, 100.0)], t2 + 1.0, t2 + 1.0, "ws", t2 + 1.0))

            # t3 = t2 + 5.0s: 已有 1 笔 FILLED，达到 max_trades_per_market=1 -> MAX_TRADES_REACHED
            t3 = t2 + 5.0
            writer.write_book(BookEvent("910", "#9100", [(0.50, 100.0)], [(0.52, 100.0)], t3, t3, "ws", t3))

            writer.settle_expired_markets(compute_strict_twap, now_ts=expiry + 5.0)
            writer.close()

            ok, summary = paper_export.run_unified_replay_backtest(
                db_path=db_path,
                cfg=StrategyConfig(initial_equity=1000.0, max_trades_per_market=1),
                latency_ms=1000,
                max_slippage=0.008,
                stale_book_ms=5000.0,
                decision_sample_sec=30.0,
                near_edge_margin=0.015,
                out_dir_arg=str(out_dir),
                out_root_arg="",
                subdir_by_run_id=False,
            )
            self.assertTrue(ok)
            self.assertEqual(summary["order_status_breakdown"].get("REJECTED:NO_LIQUIDITY"), 1)
            self.assertEqual(summary["order_status_breakdown"].get("REJECTED:SLIPPAGE_EXCEEDED"), 1)
            self.assertEqual(summary["order_status_breakdown"].get("FILLED"), 1)
            self.assertEqual(summary["settled_trades"], 1)

            with open(out_dir / "decisions.csv", encoding="utf-8-sig") as f:
                reasons = [r["filter_reason"] for r in csv.DictReader(f)]
            self.assertIn("REJECT_COOLDOWN", reasons)
            self.assertIn("MAX_TRADES_REACHED", reasons)

    def test_post_fill_equity_used_for_next_order_and_loss_does_not_double_deduct(self):
        """验收 3：第二笔信号必须使用扣完第一笔本金后的净值算 target_usdc；输单不重复扣本金。"""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "two_fills_loss.sqlite")
            out_dir = Path(td) / "bt_out"
            expiry = 1_800_000_900.0
            m = make_market("920", strike=78000.0, expiry_ts=expiry)

            writer = MarketTapeWriter(db_path, commit_batch=1)
            writer.upsert_market(m)
            # 早期高于 strike 触发买 UP，尾盘跌破 strike 结算 DOWN（买 UP 输单）
            writer.write_mark(MarkEvent("BTC", 78900.0, recv_ts=expiry - 260.0))
            writer.write_mark(MarkEvent("BTC", 77000.0, recv_ts=expiry - 50.0))
            writer.write_mark(MarkEvent("BTC", 77000.0, recv_ts=expiry - 10.0))

            t1 = expiry - 240.0
            t2 = expiry - 220.0
            writer.write_book(BookEvent("920", "#9200", [(0.50, 200.0)], [(0.52, 200.0)], t1, t1, "ws", t1))
            writer.write_book(BookEvent("920", "#9200", [(0.50, 200.0)], [(0.52, 200.0)], t2, t2, "ws", t2))

            writer.settle_expired_markets(compute_strict_twap, now_ts=expiry + 5.0)
            writer.close()

            cfg = StrategyConfig(initial_equity=1000.0, max_trades_per_market=2)
            ok, summary = paper_export.run_unified_replay_backtest(
                db_path=db_path,
                cfg=cfg,
                latency_ms=0,
                max_slippage=0.008,
                stale_book_ms=5000.0,
                decision_sample_sec=30.0,
                near_edge_margin=0.015,
                out_dir_arg=str(out_dir),
                out_root_arg="",
                subdir_by_run_id=False,
            )
            self.assertTrue(ok)
            self.assertEqual(summary["settled_trades"], 2)

            with open(out_dir / "bt_trades.csv", encoding="utf-8-sig") as f:
                trades = list(csv.DictReader(f))
            self.assertEqual(len(trades), 2)

            cost1 = float(trades[0]["cost_usdc"])
            cost2 = float(trades[1]["cost_usdc"])
            eq_after_fill_1 = float(trades[0]["equity_after_fill"])
            eq_at_sig_2 = float(trades[1]["equity_at_signal"])
            eq_after_fill_2 = float(trades[1]["equity_after_fill"])

            # 第一笔成交立扣 cost1；第二笔下单时看到的净值就是 1000 - cost1
            self.assertAlmostEqual(eq_after_fill_1, 1000.0 - cost1, places=4)
            self.assertAlmostEqual(eq_at_sig_2, eq_after_fill_1, places=4)
            # 第二笔名义金额严格小于第一笔（因按扣减后的净值乘同一比例）
            self.assertLess(cost2, cost1)
            self.assertAlmostEqual(eq_after_fill_2, 1000.0 - cost1 - cost2, places=4)

            # 两笔均输单（payout=0），期末净值精确等于 1000 - cost1 - cost2，绝无双扣
            self.assertEqual(int(trades[0]["won"]), 0)
            self.assertEqual(int(trades[1]["won"]), 0)
            self.assertAlmostEqual(summary["final_equity"], 1000.0 - cost1 - cost2, places=4)

    def test_engine_backtest_matches_export_backtest_identically(self):
        """验收 4：outcome_hip4_engine.run_replay_backtest 与 export_replay_backtest 结果完全一致。"""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "parity.sqlite")
            dir_a = Path(td) / "export_bt"
            dir_b = Path(td) / "engine_bt"
            expiry = 1_800_000_900.0
            m = make_market("930", strike=78000.0, expiry_ts=expiry)

            writer = MarketTapeWriter(db_path, commit_batch=1)
            writer.upsert_market(m)
            writer.write_mark(MarkEvent("BTC", 78700.0, recv_ts=expiry - 240.0))
            writer.write_mark(MarkEvent("BTC", 78800.0, recv_ts=expiry - 40.0))
            writer.write_mark(MarkEvent("BTC", 78820.0, recv_ts=expiry - 10.0))
            t0 = expiry - 200.0
            writer.write_book(BookEvent("930", "#9300", [(0.50, 100.0)], [(0.52, 100.0)], t0, t0, "ws", t0))
            writer.write_book(BookEvent("930", "#9300", [(0.50, 100.0)], [(0.52, 100.0)], t0 + 0.3, t0 + 0.3, "ws", t0 + 0.3))
            writer.settle_expired_markets(compute_strict_twap, now_ts=expiry + 5.0)
            writer.close()

            cfg = StrategyConfig(initial_equity=1000.0, settlement_fee_rate=0.0007)
            ok_a = paper_export.export_replay_backtest(
                db_path=db_path,
                cfg=cfg,
                latency_ms=300,
                max_slippage=0.008,
                stale_book_ms=5000.0,
                decision_sample_sec=30.0,
                near_edge_margin=0.015,
                out_dir_arg=str(dir_a),
                out_root_arg="",
                subdir_by_run_id=False,
            )
            self.assertTrue(ok_a)

            sum_b = engine.run_replay_backtest(
                db_path=db_path,
                cfg=cfg,
                latency_ms=300,
                max_slippage=0.008,
                stale_book_ms=5000.0,
                decision_sample_sec=30.0,
                near_edge_margin=0.015,
                out_dir=str(dir_b),
            )
            self.assertIsNotNone(sum_b)

            sum_a = json.loads((dir_a / "bt_summary.json").read_text(encoding="utf-8"))
            for key in (
                "initial_equity",
                "final_equity",
                "return_pct",
                "max_drawdown_pct",
                "settled_trades",
                "win_rate_pct",
                "order_status_breakdown",
            ):
                self.assertEqual(sum_a[key], sum_b[key], f"mismatch on {key}")


if __name__ == "__main__":
    unittest.main()
