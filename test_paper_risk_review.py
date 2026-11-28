import contextlib
import csv
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import outcome_paper_export as paper_export
import paper_bookkeeping as pb
from market_data import BookEvent, MarkEvent, MarketTapeWriter
from outcome_hip4_engine import (
    HIP4AsianBinaryStrategy,
    OutcomeMarket,
    StrategyConfig,
    compute_strict_twap,
)


def make_market(oid: str, strike: float, expiry_ts: float) -> OutcomeMarket:
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


class TestFormalDrawdownDefinition(unittest.TestCase):
    def test_single_calculation_and_reference_case_boundaries(self):
        """验收 1：唯一回撤公式验算回顾跑数字与闭区间 >= 边界，update_peak_dd 必须调用 calc_drawdown。"""
        peak = 1056.7017
        limit = 0.08

        # 回顾跑 PAPER-00010 成交预扣后：回撤 8.371%，触发 halt
        triggered, dd = pb.is_drawdown_halted(968.2418, peak, limit)
        self.assertTrue(triggered)
        self.assertAlmostEqual(dd * 100.0, 8.371, places=3)

        # 净值 972.1656 恰好高于 peak * 0.92 (972.165564)，未越门槛
        triggered_just_above, dd_above = pb.is_drawdown_halted(972.1656, peak, limit)
        self.assertFalse(triggered_just_above)
        self.assertLess(dd_above, limit)

        # 精确等于门槛（闭区间 >=）：必须触发 halt
        triggered_exact, dd_exact = pb.is_drawdown_halted(920.0, 1000.0, limit)
        self.assertTrue(triggered_exact)
        self.assertAlmostEqual(dd_exact, 0.08, places=12)

        # 仅结算恢复时点：回撤 4.872%，低于 8%
        triggered_recovered, dd_rec = pb.is_drawdown_halted(1005.2159, peak, limit)
        self.assertFalse(triggered_recovered)
        self.assertAlmostEqual(dd_rec * 100.0, 4.872, places=3)

        # update_peak_dd 必须复用 calc_drawdown
        with patch("paper_bookkeeping.calc_drawdown", wraps=pb.calc_drawdown) as m_calc:
            new_peak, new_dd = pb.update_peak_dd(968.2418, peak, 0.0)
            self.assertEqual(m_calc.call_count, 1)
            self.assertAlmostEqual(new_peak, peak, places=4)
            self.assertAlmostEqual(new_dd * 100.0, 8.371, places=3)


class TestHaltLoggingAndStickiness(unittest.TestCase):
    def test_first_flip_prints_single_stdout_line_and_stays_sticky_after_recovery(self):
        """验收 2/3：首次翻转只打一行 [Risk] HALT_DRAWDOWN；净值拉回后 halted 保持粘性且不重复打印。"""
        cfg = StrategyConfig(initial_equity=1000.0)
        self.assertEqual(cfg.daily_drawdown_limit, 0.08)
        strat = HIP4AsianBinaryStrategy(cfg)

        # 2026-11-28 21:15:50.042 Asia/Hong_Kong (UTC+8) -> 2026-11-28T13:15:50.042Z
        hk_tz = timezone(timedelta(hours=8))
        flip_ts = datetime(2026, 11, 28, 21, 15, 50, 42000, tzinfo=hk_tz).timestamp()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # 1) 未越过门槛：不翻转、不打印
            flipped0 = strat.sync_drawdown_state(972.1656, 1056.7017, ts=flip_ts - 5.0)
            self.assertFalse(flipped0)
            self.assertFalse(strat.halted)
            self.assertEqual(buf.getvalue(), "")

            # 2) 首次越过门槛：翻转并打印恰好一行
            flipped1 = strat.sync_drawdown_state(968.2418, 1056.7017, ts=flip_ts)
            self.assertTrue(flipped1)
            self.assertTrue(strat.halted)

            # 3) 同一实例重复同步（含盈利结算将净值拉回到 1005.2159，回撤 4.872% < 8%）
            flipped2 = strat.sync_drawdown_state(968.2418, 1056.7017, ts=flip_ts + 1.0)
            flipped3 = strat.sync_drawdown_state(1005.2159, 1056.7017, ts=flip_ts + 861.0)
            self.assertFalse(flipped2)
            self.assertFalse(flipped3)
            self.assertTrue(strat.halted)  # 粘性不恢复

            # 4) 随后多次 evaluate_with_audit 拒绝，不得再次打印
            expiry = flip_ts + 1200.0
            m = make_market("950", strike=78000.0, expiry_ts=expiry)
            for offset in (12.0, 25.0):
                dec = strat.evaluate_with_audit(
                    market=m,
                    ts=flip_ts + offset,
                    mark_px=78900.0,
                    ann_vol=0.40,
                    yes_bids=[(0.50, 100.0)],
                    yes_asks=[(0.52, 100.0)],
                    book_ts=flip_ts + offset,
                    acc_twap=None,
                    equity=1005.2159,
                    peak_equity=1056.7017,
                )
                self.assertFalse(dec.should_trade)
                self.assertEqual(dec.filter_reason, "HALTED_DRAWDOWN")

        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertTrue(line.startswith("[Risk] HALT_DRAWDOWN "))
        self.assertIn("ts=2026-11-28T13:15:50.042Z", line)
        self.assertIn("peak_equity=1056.7017", line)
        self.assertIn("equity=968.2418", line)
        self.assertIn("drawdown_pct=8.371", line)
        self.assertIn("limit=0.08", line)

        snap = strat.halt_snapshot()
        self.assertEqual(snap["halted_at_utc"], "2026-11-28T13:15:50.042Z")
        self.assertEqual(snap["halted_peak_equity"], 1056.7017)
        self.assertEqual(snap["halted_equity"], 968.2418)
        self.assertEqual(snap["halted_drawdown_pct"], 8.371)


class TestShadowEquityAndMetadataReplay(unittest.TestCase):
    def test_shadow_equity_diverges_from_official_dd_without_affecting_halt(self):
        """验收 1/2/3/4：正式预扣现金触发 halt 且粘性阻断下一场，即使仅结算影子回撤为 0%。"""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "shadow_vs_official.sqlite")
            out_dir = Path(td) / "bt_out"
            exp1 = 1_800_000_900.0
            exp2 = exp1 + 900.0
            m1 = make_market("961", strike=78000.0, expiry_ts=exp1)
            m2 = make_market("962", strike=79000.0, expiry_ts=exp2)

            writer = MarketTapeWriter(db_path, commit_batch=1)
            writer.upsert_market(m1)
            writer.upsert_market(m2)

            # 第 1 场：单笔买 UP 预扣约 $89.96（相对 $1000 回撤 8.996% >= 8% 触发 halt），但到期结算大赢
            writer.write_mark(MarkEvent("BTC", 78900.0, recv_ts=exp1 - 240.0))
            writer.write_mark(MarkEvent("BTC", 78950.0, recv_ts=exp1 - 40.0))
            writer.write_mark(MarkEvent("BTC", 78960.0, recv_ts=exp1 - 10.0))
            t1 = exp1 - 200.0
            writer.write_book(BookEvent("961", "#9610", [(0.50, 500.0)], [(0.52, 500.0)], t1, t1, "ws", t1))

            # 第 2 场：尽管第 1 场结算后净值创新高且影子回撤为 0%，halt 粘性仍必须阻断第 2 场开仓
            writer.write_mark(MarkEvent("BTC", 79900.0, recv_ts=exp2 - 240.0))
            writer.write_mark(MarkEvent("BTC", 79950.0, recv_ts=exp2 - 40.0))
            writer.write_mark(MarkEvent("BTC", 79960.0, recv_ts=exp2 - 10.0))
            t2 = exp2 - 200.0
            writer.write_book(BookEvent("962", "#9620", [(0.50, 500.0)], [(0.52, 500.0)], t2, t2, "ws", t2))

            writer.settle_expired_markets(compute_strict_twap, now_ts=exp2 + 5.0)
            writer.close()

            # 仅在测试夹具里放大单场风险比例至 9%，以单笔构造预扣越过 8% 但结算盈利的背离场景
            cfg = StrategyConfig(
                initial_equity=1000.0,
                kelly_fraction=1.0,
                max_market_risk_pct=0.09,
                daily_drawdown_limit=0.08,
            )

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
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
            risk_lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("[Risk] HALT_DRAWDOWN")]
            self.assertEqual(len(risk_lines), 1)

            # 第 1 场成交盈利，第 2 场被粘性 halt 拦截
            self.assertEqual(summary["settled_trades"], 1)
            self.assertTrue(summary["halted_by_drawdown"])
            self.assertGreater(summary["final_equity"], 1000.0)
            self.assertGreaterEqual(summary["max_drawdown_pct"], 8.0)

            # 影子权益为仅结算口径：净值从 1000 直接跳到 >1080，影子最大回撤为 0%，不影响正式 halt
            self.assertEqual(summary["shadow_equity_basis"], "settlement_equity_after")
            self.assertEqual(summary["shadow_final_equity"], summary["final_equity"])
            self.assertEqual(summary["shadow_max_drawdown_pct"], 0.0)

            # 元数据口径与首次 halt 快照与 stdout 行一致
            for fname in ("bt_summary.json", "run_metadata.json"):
                meta = json.loads((out_dir / fname).read_text(encoding="utf-8"))
                self.assertEqual(meta["drawdown_definition"]["equity_basis"], "prepaid_cash")
                self.assertFalse(meta["drawdown_definition"]["calendar_day_reset"])
                self.assertEqual(meta["drawdown_definition"]["halt_policy"], "sticky_until_process_end")
                self.assertEqual(meta["strategy_config"]["daily_drawdown_limit"], 0.08)
                self.assertIsNotNone(meta["halted_at_utc"])
                self.assertIn(f"ts={meta['halted_at_utc']}", risk_lines[0])
                self.assertIn(f"equity={meta['halted_equity']:.4f}", risk_lines[0])
                self.assertIn(f"drawdown_pct={meta['halted_drawdown_pct']:.3f}", risk_lines[0])

            with open(out_dir / "decisions.csv", encoding="utf-8-sig") as f:
                m2_reasons = [r["filter_reason"] for r in csv.DictReader(f) if r["outcome_id"] == "962"]
            self.assertIn("HALTED_DRAWDOWN", m2_reasons)

    def test_unhalted_run_leaves_halt_snapshot_null_not_zero(self):
        """未触发 halt 时快照字段为 None（JSON null），不以 0 冒充。"""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "no_halt.sqlite")
            out_dir = Path(td) / "bt_out"
            exp1 = 1_800_000_900.0
            m1 = make_market("971", strike=78000.0, expiry_ts=exp1)

            writer = MarketTapeWriter(db_path, commit_batch=1)
            writer.upsert_market(m1)
            writer.write_mark(MarkEvent("BTC", 78900.0, recv_ts=exp1 - 240.0))
            writer.write_mark(MarkEvent("BTC", 78950.0, recv_ts=exp1 - 40.0))
            writer.write_mark(MarkEvent("BTC", 78960.0, recv_ts=exp1 - 10.0))
            t1 = exp1 - 200.0
            writer.write_book(BookEvent("971", "#9710", [(0.50, 200.0)], [(0.52, 200.0)], t1, t1, "ws", t1))
            writer.settle_expired_markets(compute_strict_twap, now_ts=exp1 + 5.0)
            writer.close()

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ok, summary = paper_export.run_unified_replay_backtest(
                    db_path=db_path,
                    cfg=StrategyConfig(initial_equity=1000.0),
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
            self.assertFalse(summary["halted_by_drawdown"])
            self.assertIsNone(summary["halted_at_utc"])
            self.assertIsNone(summary["halted_peak_equity"])
            self.assertIsNone(summary["halted_equity"])
            self.assertIsNone(summary["halted_drawdown_pct"])
            self.assertNotIn("[Risk] HALT_DRAWDOWN", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
