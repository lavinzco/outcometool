import inspect
import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import requests

import outcome_hip4_engine as eng


FIXTURE = Path(__file__).resolve().parent / "tests" / "fixtures" / "markets" / "outcome_period_cases.json"
# 2026-11-28 10:20 UTC: the 10:30 BTC window is still open, the 10:15 window has expired.
NOW_TS = datetime(2026, 11, 28, 10, 20, tzinfo=timezone.utc).timestamp()


def load_cases():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["outcomes"]


def outcomes_and_periods(cases=None):
    cases = load_cases() if cases is None else cases
    outcomes = [case["outcome"] for case in cases]
    periods = {}
    for case in cases:
        raw = case.get("metadata_period")
        if isinstance(raw, str) and raw.strip():
            periods[str(case["outcome"]["outcome"])] = raw.strip()
    return outcomes, periods


class ContractPeriodTests(unittest.TestCase):
    def test_default_period_remains_15m(self):
        self.assertEqual(inspect.signature(eng.fetch_outcome_markets).parameters["period"].default, "15m")
        engine = Path("outcome_hip4_engine.py").read_text(encoding="utf-8")
        paper = Path("outcome_paper_export.py").read_text(encoding="utf-8")
        self.assertIn('add_argument("--period", default="15m"', engine)
        self.assertEqual(engine.count('add_argument("--period", default="15m"'), 2)
        self.assertIn('add_argument("--period", default="15m")', paper)

    def test_seconds_key_is_not_period(self):
        desc = "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:59402|time:20261128-1030"
        parsed = eng.parse_outcome_description(desc)
        self.assertEqual(parsed["seconds"], "60")
        self.assertNotIn("period", parsed)
        self.assertEqual(eng.contract_period(parsed, "50678", None), "")
        self.assertEqual(eng.contract_period(parsed, "50678", {"50678": "15m"}), "15m")

    def test_explicit_period_beats_metadata_and_seconds(self):
        desc = "class:priceBinary|underlying:BTC|expiry:20261129-0600|targetPrice:59702|period:1d|seconds:60"
        parsed = eng.parse_outcome_description(desc)
        self.assertEqual(eng.contract_period(parsed, "50457", {"50457": "24h"}), "1d")

    def test_btc_15m_keeps_seconds_60_market(self):
        outcomes, periods = outcomes_and_periods()
        found = eng.select_outcome_markets(
            outcomes,
            period="15m",
            underlying="BTC",
            only_active=True,
            now_ts=NOW_TS,
            period_by_outcome_id=periods,
        )
        self.assertEqual([m.outcome_id for m in found], ["50678"])
        market = found[0]
        self.assertEqual(market.period, "15m")
        self.assertEqual(market.underlying, "BTC")
        self.assertEqual(market.strike, 59402.0)
        self.assertEqual(market.yes_coin, "#506780")
        self.assertEqual(market.yes_asset_id, 100_506_780)
        self.assertEqual(market.fee_scale, 1.0)
        self.assertIn("seconds:60", market.raw_description)
        self.assertNotEqual(market.period, "60s")

    def test_previous_window_is_15m_but_inactive(self):
        outcomes, periods = outcomes_and_periods()
        active = eng.select_outcome_markets(
            outcomes,
            period="15m",
            underlying="BTC",
            only_active=True,
            now_ts=NOW_TS,
            period_by_outcome_id=periods,
        )
        self.assertNotIn("50666", [m.outcome_id for m in active])
        including_expired = eng.select_outcome_markets(
            outcomes,
            period="15m",
            underlying="BTC",
            only_active=False,
            now_ts=NOW_TS,
            period_by_outcome_id=periods,
        )
        self.assertEqual([m.outcome_id for m in including_expired], ["50666", "50678"])
        self.assertTrue(all(m.period == "15m" for m in including_expired))

    def test_other_shapes_are_not_pulled_into_15m(self):
        outcomes, periods = outcomes_and_periods()
        found = eng.select_outcome_markets(
            outcomes,
            period="15m",
            underlying="BTC",
            only_active=False,
            now_ts=NOW_TS,
            period_by_outcome_id=periods,
        )
        ids = {m.outcome_id for m in found}
        self.assertEqual(ids, {"50666", "50678"})
        # Daily uses description period:1d even though Outcome metadata says 24h.
        # Fox candle text, skew seconds:90, and priceTouch seconds:3 stay out.
        self.assertTrue(ids.isdisjoint({"50457", "50378", "50681", "50658", "50566", "31270", "49984"}))

    def test_seconds_60_without_metadata_period_is_dropped(self):
        outcomes, periods = outcomes_and_periods()
        periods.pop("50678")
        periods.pop("50666")
        found = eng.select_outcome_markets(
            outcomes,
            period="15m",
            underlying="BTC",
            only_active=False,
            now_ts=NOW_TS,
            period_by_outcome_id=periods,
        )
        self.assertEqual(found, [])

    def test_eth_15m_is_separate_from_btc(self):
        outcomes, periods = outcomes_and_periods()
        found = eng.select_outcome_markets(
            outcomes,
            period="15m",
            underlying="ETH",
            only_active=True,
            now_ts=NOW_TS,
            period_by_outcome_id=periods,
        )
        self.assertEqual([m.outcome_id for m in found], ["50676"])
        self.assertEqual(found[0].period, "15m")
        self.assertEqual(found[0].strike, 1651.2)

    def test_legacy_15m_description_still_matches_without_lookup(self):
        legacy = {
            "outcome": 42,
            "name": "Recurring",
            "description": "class:priceBinary|underlying:BTC|expiry:20261128-1030|targetPrice:59000|period:15m",
            "sideSpecs": [{"name": "Yes"}, {"name": "No"}],
            "quoteToken": "USDC",
        }
        found = eng.select_outcome_markets(
            [legacy],
            period="15m",
            underlying="BTC",
            only_active=True,
            now_ts=NOW_TS,
            period_by_outcome_id=None,
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].period, "15m")
        self.assertEqual(found[0].strike, 59000.0)

    def test_price_bucket_is_excluded(self):
        bucket = {
            "outcome": 7,
            "name": "Recurring",
            "description": "class:priceBucket|underlying:BTC|expiry:20261128-1030|priceThresholds:1,2|period:15m",
        }
        found = eng.select_outcome_markets(
            [bucket],
            period="15m",
            underlying="BTC",
            only_active=False,
            now_ts=NOW_TS,
        )
        self.assertEqual(found, [])


class LookupParsingTests(unittest.TestCase):
    def test_lookup_reads_metadata_period_not_seconds(self):
        body = {
            "success": True,
            "data": {
                "markets": {
                    "50678": {
                        "outcome": {
                            "description": "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:59402|time:20261128-1030"
                        },
                        "metadata": {"period": "15m", "category": "out-btc-15m"},
                    },
                    "50681": {
                        "outcome": {
                            "description": "perp:BTC|priceDescription:the Hyperliquid BTC perp 5-minute candle close reference|seconds:1|threshold:59300|time:20261128-1030"
                        },
                        "metadata": {"asset": "btc"},
                    },
                }
            },
        }

        class Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return body

        with mock.patch.object(eng.requests, "post", return_value=Resp()) as post:
            found = eng.lookup_outcome_periods([50678, 50681])
        self.assertEqual(found, {"50678": "15m", "50681": None})
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent, {"outcome_ids": [50678, 50681]})
        self.assertEqual(post.call_args.args[0], eng.OUTCOME_MARKET_LOOKUP_URL)


class FetchWiringTests(unittest.TestCase):
    def setUp(self):
        eng._outcome_period_cache.clear()

    def tearDown(self):
        eng._outcome_period_cache.clear()

    def _fake_post(self, meta_payload, lookup_payload):
        def fake_post(url, json=None, timeout=None, headers=None):
            if isinstance(json, dict) and json.get("type") == "outcomeMeta":
                return _Resp(meta_payload)
            if url == eng.OUTCOME_MARKET_LOOKUP_URL:
                return _Resp(lookup_payload)
            raise AssertionError(url)

        return fake_post

    def test_fetch_keeps_live_shaped_15m_market(self):
        outcomes, periods = outcomes_and_periods()
        lookup_markets = {}
        for case in load_cases():
            oid = str(case["outcome"]["outcome"])
            lookup_markets[oid] = {
                "outcome": {"description": case["outcome"]["description"]},
                "metadata": {
                    "period": case.get("metadata_period"),
                    "category": case.get("metadata_category"),
                },
            }
        meta = {"outcomes": outcomes, "feeScale": 1.0}
        lookup = {"success": True, "data": {"markets": lookup_markets}}
        with mock.patch.object(eng.requests, "post", side_effect=self._fake_post(meta, lookup)):
            found = eng.fetch_outcome_markets(
                "mainnet", "15m", "BTC", True, now_ts=NOW_TS
            )
        self.assertEqual([m.outcome_id for m in found], ["50678"])
        self.assertEqual(found[0].period, "15m")
        self.assertIn("50678", eng._outcome_period_cache)
        self.assertEqual(eng._outcome_period_cache["50678"], "15m")

    def test_lookup_failure_does_not_guess_seconds(self):
        legacy = {
            "outcome": 42,
            "name": "Recurring",
            "description": "class:priceBinary|underlying:BTC|expiry:20261128-1045|targetPrice:59000|period:15m",
            "sideSpecs": [{"name": "Yes"}, {"name": "No"}],
            "quoteToken": "USDC",
        }
        seconds_only = {
            "outcome": 50678,
            "name": "template:binaryPrice",
            "description": "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:59402|time:20261128-1030",
            "sideSpecs": [{"name": "template:Yes"}, {"name": "template:No"}],
            "quoteToken": "USDC",
            "venue": "out",
            "deployerFeeScale": "1.0",
        }
        meta = {"outcomes": [legacy, seconds_only], "feeScale": 1.0}

        def fake_post(url, json=None, timeout=None, headers=None):
            if isinstance(json, dict) and json.get("type") == "outcomeMeta":
                return _Resp(meta)
            raise requests.ConnectionError("lookup down")

        buf = io.StringIO()
        with mock.patch.object(eng.requests, "post", side_effect=fake_post):
            with redirect_stdout(buf):
                found = eng.fetch_outcome_markets("mainnet", "15m", "BTC", True, now_ts=NOW_TS)
        self.assertEqual([m.outcome_id for m in found], ["42"])
        self.assertEqual(found[0].period, "15m")
        text = buf.getvalue()
        self.assertIn("metadata.period", text)
        self.assertNotIn("60s", text)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
