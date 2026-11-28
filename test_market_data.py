import asyncio
import csv
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import market_data
import outcome_hip4_engine as engine
import outcome_paper_export as paper_export
from market_data import BookEvent, LiveMarketFeed, MarkEvent, MarketTapeWriter
from outcome_hip4_engine import OutcomeMarket, StrategyConfig, compute_strict_twap, parse_book_levels


class FakeWS:
    def __init__(self, script, sent_log):
        self._script = list(script)
        self.sent_log = sent_log

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, payload):
        self.sent_log.append(json.loads(payload))

    async def recv(self):
        if not self._script:
            await asyncio.sleep(0.05)
            return ""
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if item == "__TIMEOUT__":
            await asyncio.sleep(0.05)
            return ""
        return json.dumps(item)


def make_market(oid="516", strike=78000.0, expiry_ts=1_800_000_900.0) -> OutcomeMarket:
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


class TestConvergenceConstraints(unittest.TestCase):
    def test_single_live_ws_loop_only_in_market_data(self):
        """验收 1：仅 market_data.py 存在 websockets.connect。"""
        root = Path(__file__).resolve().parent
        self.assertIn("websockets.connect", (root / "market_data.py").read_text(encoding="utf-8"))
        self.assertNotIn("websockets.connect", (root / "outcome_hip4_engine.py").read_text(encoding="utf-8"))
        self.assertNotIn("websockets.connect", (root / "outcome_paper_export.py").read_text(encoding="utf-8"))

    def test_cli_help_record_and_paper_db_option(self):
        """验收 2：record --help 与 paper --help 可用，且 paper 暴露 --db 默认 hip4_replay.sqlite。"""
        root = Path(__file__).resolve().parent
        rec = subprocess.run(
            [sys.executable, str(root / "outcome_hip4_engine.py"), "record", "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--db", rec.stdout)

        pap = subprocess.run(
            [sys.executable, str(root / "outcome_paper_export.py"), "paper", "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--db", pap.stdout)
        self.assertIn("hip4_replay.sqlite", pap.stdout)


class TestSharedFeedAndTape(unittest.TestCase):
    def test_subscribe_reconnect_rest_backfill_and_top10_truncation(self):
        """验收 1/3：只订 Yes 侧、断线重连补 REST 快照、盘口截断前 10 档、保持 ts 约定。"""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "tape.sqlite")
            writer = MarketTapeWriter(db_path, commit_batch=1)
            m = make_market("516", 78000.0, 1_800_000_900.0)

            deep_bids = [{"px": f"{0.50 - i * 0.01:.2f}", "sz": "10"} for i in range(15)]
            deep_asks = [{"px": f"{0.51 + i * 0.01:.2f}", "sz": "10"} for i in range(15)]
            rest_calls = []
            sent_msgs = []
            scripts = [
                [
                    {"channel": "activeAssetCtx", "data": {"coin": "BTC", "ctx": {"markPx": "78120.5"}}},
                    {"channel": "l2Book", "data": {"coin": "#5160", "time": 1_800_000_010_000, "levels": [deep_bids, deep_asks]}},
                    ConnectionError("drop"),
                ],
                [
                    {"channel": "l2Book", "data": {"coin": "#5160", "time": 1_800_000_020_000, "levels": [deep_bids, deep_asks]}},
                ],
            ]
            conn_count = 0

            def fake_ws_connect(url, **kwargs):
                nonlocal conn_count
                s = scripts[min(conn_count, len(scripts) - 1)]
                conn_count += 1
                return FakeWS(s, sent_msgs)

            now_box = [1_800_000_000.0]

            def fake_clock():
                now_box[0] += 0.1
                return now_box[0]

            stop_flag = [False]
            books_seen = []

            def on_book(ev: BookEvent):
                books_seen.append(ev)
                if ev.source == "ws" and ev.exchange_ts == 1_800_000_020.0:
                    stop_flag[0] = True

            feed = LiveMarketFeed(
                network="mainnet",
                underlying="BTC",
                period="15m",
                ws_url="wss://fake.local/ws",
                poll_meta_sec=999.0,
                ping_interval_sec=999.0,
                recv_timeout_sec=0.02,
                reconnect_delay_sec=0.0,
                tape_writer=writer,
                auto_settle_tape=False,
                fetch_markets_fn=lambda *a: [m],
                fetch_l2_book_fn=lambda net, coin: (
                    rest_calls.append((net, coin))
                    or {"coin": coin, "time": 1_800_000_005_000, "levels": [deep_bids, deep_asks]}
                ),
                parse_book_fn=parse_book_levels,
                compute_twap_fn=compute_strict_twap,
                ws_connect=fake_ws_connect,
                clock_fn=fake_clock,
            )

            asyncio.run(feed.run(stop_ts=1_800_000_500.0, should_stop=lambda: stop_flag[0], on_book=on_book))
            writer.close()

            subs = [x["subscription"] for x in sent_msgs if x.get("method") == "subscribe"]
            self.assertEqual(conn_count, 2)
            self.assertEqual(sum(1 for s in subs if s == {"type": "activeAssetCtx", "coin": "BTC"}), 2)
            self.assertEqual(sum(1 for s in subs if s == {"type": "outcomeMetaUpdates"}), 2)
            self.assertEqual(sum(1 for s in subs if s == {"type": "l2Book", "coin": "#5160"}), 2)
            self.assertFalse(any(s.get("coin") == "#5161" for s in subs))
            self.assertEqual(rest_calls, [("mainnet", "#5160"), ("mainnet", "#5160")])
            self.assertEqual(len(books_seen), 4)

            conn = sqlite3.connect(db_path)
            b_rows = conn.execute(
                "SELECT ts, bids_json, asks_json, exchange_ts, recv_ts, source FROM book_snapshots ORDER BY id ASC"
            ).fetchall()
            self.assertEqual(len(b_rows), 4)
            self.assertEqual(b_rows[0][5], "rest")
            self.assertAlmostEqual(b_rows[0][0], b_rows[0][4], places=6)
            self.assertEqual(len(json.loads(b_rows[0][1])), 10)
            self.assertEqual(len(json.loads(b_rows[0][2])), 10)
            self.assertEqual(b_rows[1][5], "ws")
            self.assertEqual(b_rows[1][0], 1_800_000_010.0)
            conn.close()

    def test_legacy_db_upgrade_and_validate_replay_and_export_backtest(self):
        """验收 3：旧库可平滑补列，共享层写入的 SQLite 能通过 validate_replay_db 与 export_replay_backtest。"""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "legacy.sqlite")
            raw = sqlite3.connect(db_path)
            raw.executescript(
                """
                CREATE TABLE markets (
                    outcome_id TEXT PRIMARY KEY, name TEXT, underlying TEXT, strike REAL,
                    expiry_ts REAL, period TEXT, fee_scale REAL, yes_coin TEXT,
                    raw_description TEXT, settled_twap REAL, outcome_up INTEGER
                );
                CREATE TABLE book_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                    outcome_id TEXT NOT NULL, bids_json TEXT NOT NULL, asks_json TEXT NOT NULL
                );
                CREATE TABLE mark_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                    underlying TEXT NOT NULL, mark_px REAL NOT NULL
                );
                """
            )
            raw.commit()
            raw.close()

            expiry = 1_800_000_900.0
            m = make_market("700", 78000.0, expiry)
            writer = MarketTapeWriter(db_path, commit_batch=1)
            writer.upsert_market(m)
            writer.write_mark(MarkEvent("BTC", 78400.0, recv_ts=expiry - 200.0))
            writer.write_mark(MarkEvent("BTC", 78500.0, recv_ts=expiry - 40.0))
            writer.write_mark(MarkEvent("BTC", 78600.0, recv_ts=expiry - 10.0))
            writer.write_book(
                BookEvent(
                    outcome_id="700", yes_coin="#7000",
                    bids=[(0.54, 100.0)], asks=[(0.56, 100.0)],
                    recv_ts=expiry - 180.0, exchange_ts=expiry - 180.0, source="ws", tape_ts=expiry - 180.0,
                )
            )
            settled = writer.settle_expired_markets(compute_strict_twap, now_ts=expiry + 5.0)
            writer.close()

            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0][4], 1)

            conn, m_rows = paper_export.validate_replay_db(db_path)
            self.assertIsNotNone(conn)
            self.assertEqual(len(m_rows), 1)
            conn.close()

            bt_dir = str(Path(td) / "bt_out")
            ok = paper_export.export_replay_backtest(
                db_path=db_path,
                cfg=StrategyConfig(initial_equity=1000.0),
                latency_ms=0,
                max_slippage=0.01,
                stale_book_ms=5000.0,
                decision_sample_sec=30.0,
                near_edge_margin=0.015,
                out_dir_arg=bt_dir,
                out_root_arg="",
                subdir_by_run_id=False,
            )
            self.assertTrue(ok)
            self.assertTrue((Path(bt_dir) / "bt_summary.json").exists())
            self.assertTrue((Path(bt_dir) / "bt_trades.csv").exists())

    def test_paper_run_writes_both_csv_json_and_sqlite_tape(self):
        """验收 4：夹具驱动 run_paper_trading，同时产出原有 CSV/JSON 与非空 SQLite 磁带。"""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "paper_logs"
            db_path = Path(td) / "paper_tape.sqlite"
            now = time.time()
            m = make_market("888", strike=78000.0, expiry_ts=now + 180.0)

            book_levels = [
                [{"px": "0.53", "sz": "100"}],
                [{"px": "0.55", "sz": "100"}],
            ]
            ws_messages = [
                {"channel": "activeAssetCtx", "data": {"coin": "BTC", "ctx": {"markPx": "78600.0"}}},
                {"channel": "l2Book", "data": {"coin": "#8880", "time": int(now * 1000), "levels": book_levels}},
            ]
            sent_msgs = []

            def fake_ws_connect(url, **kwargs):
                return FakeWS(ws_messages, sent_msgs)

            with (
                patch.object(market_data.websockets, "connect", side_effect=fake_ws_connect),
                patch.object(paper_export, "fetch_outcome_markets", return_value=[m]),
                patch.object(
                    paper_export,
                    "fetch_l2_book_rest",
                    return_value={"coin": "#8880", "time": int(now * 1000), "levels": book_levels},
                ),
            ):
                asyncio.run(
                    paper_export.run_paper_trading(
                        network="mainnet",
                        underlying="BTC",
                        period="15m",
                        cfg=StrategyConfig(initial_equity=1000.0),
                        latency_ms=20,
                        max_slippage=0.01,
                        stale_book_ms=5000.0,
                        decision_sample_sec=30.0,
                        near_edge_margin=0.015,
                        out_dir_arg=str(out_dir),
                        out_root_arg="",
                        subdir_by_run_id=False,
                        duration_min=0.008,  # ~0.5 秒足够处理夹具并排空 20ms 延迟单
                        poll_meta_sec=60.0,
                        db_path=str(db_path),
                    )
                )

            # 1) 验证全部既有纸面 CSV/JSON 文件存在且有效
            for fname in (
                "paper_orders.csv",
                "paper_settlements.csv",
                "paper_open_positions.csv",
                "paper_summary.json",
                "decisions.csv",
                "market_outcomes.csv",
                "run_metadata.json",
            ):
                self.assertTrue((out_dir / fname).exists(), f"missing {fname}")

            summary = json.loads((out_dir / "paper_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["initial_equity"], 1000.0)
            self.assertEqual(summary["order_status_breakdown"].get("FILLED"), 1)

            with open(out_dir / "paper_open_positions.csv", encoding="utf-8-sig") as f:
                open_rows = list(csv.DictReader(f))
            self.assertEqual(len(open_rows), 1)
            self.assertEqual(open_rows[0]["open_reason"], "NOT_EXPIRED")

            # 2) 验证同步写入的 SQLite 磁带非空（市场、mark、盘口均落库）
            conn = sqlite3.connect(str(db_path))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0], 1)
            self.assertGreaterEqual(conn.execute("SELECT COUNT(*) FROM mark_ticks").fetchone()[0], 1)
            self.assertGreaterEqual(conn.execute("SELECT COUNT(*) FROM book_snapshots").fetchone()[0], 2)
            conn.close()


if __name__ == "__main__":
    unittest.main()
