import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.stats import norm, t as student_t

from market_data import LiveMarketFeed, MarketTapeWriter, init_db


# ============================================================
# 1. 端点配置与数据结构
# ============================================================

ENDPOINTS = {
    "mainnet": {
        "rest": "https://api.hyperliquid.xyz/info",
        "ws": "wss://api.hyperliquid.xyz/ws",
    },
    "testnet": {
        "rest": "https://api.hyperliquid-testnet.xyz/info",
        "ws": "wss://api.hyperliquid-testnet.xyz/ws",
    },
}

# 拒单冷却时间（秒）：纸面与回放共用，不占用单场交易额度
REJECT_COOLDOWN_SEC = 3.0


@dataclass
class OutcomeMarket:
    outcome_id: str
    name: str
    underlying: str
    strike: float               # targetPrice
    expiry_ts: float            # UTC 秒级时间戳
    period: str                 # e.g. "15m", "1d"
    fee_scale: float            # deployerFeeScale 或全局 feeScale
    yes_coin: str               # f"#{outcome_id}0"
    no_coin: str                # f"#{outcome_id}1"
    yes_asset_id: int           # 100000000 + int(id)*10 + 0
    no_asset_id: int            # 100000000 + int(id)*10 + 1
    raw_description: str


@dataclass
class StrategyConfig:
    initial_equity: float = 1000.0

    # 定价模型参数（Student-t 厚尾 + 双周期波动率）
    df_t: float = 5.0
    ewma_lambda: float = 0.92
    min_ann_vol: float = 0.25

    # 交易窗口与安全边际
    min_tau_sec: float = 8.0        # 最后 8 秒防链路抖动禁开仓（最后 60s 内允许基于 TWAP 定价交易）
    max_tau_sec: float = 870.0      # 15m 盘开盘前 30 秒观察
    edge_early: float = 0.032       # 剩余大于 300s 时的最小净优势
    edge_mid: float = 0.020         # 剩余 60s~300s 时的最小净优势
    edge_twap_zone: float = 0.015   # 最后 60s TWAP 锁定区最小净优势
    gamma_zone_tau: float = 150.0   # 临近到期且贴近行权价的 Gamma 雷区
    gamma_zone_z: float = 0.30
    gamma_penalty: float = 0.025
    min_entry_price: float = 0.10
    max_entry_price: float = 0.94

    # HIP-4 费率模型（开仓 0 费率，仅结算或平仓扣费）
    base_taker_fee: float = 0.0007  # Tier 0 现货基准 0.070%
    settlement_fee_rate: float = 0.0007  # 结算扣费估算（实际以市场 spec 为准，受 fee_scale 缩放）
    min_order_notional: float = 1.0 # 最小名义约 1 USDC

    # 资金管理与风控
    kelly_fraction: float = 0.20
    max_market_risk_pct: float = 0.03
    max_trades_per_market: int = 1
    daily_drawdown_limit: float = 0.08


# ============================================================
# 2. HyperCore HIP-4 元数据与订单簿解析器
# ============================================================

# Outcome 前端使用的市场查询。HIP-4 outcomeMeta 本身没有 template:binaryPrice 的周期字段。
# 已核对：POST 这个地址，body {"outcome_ids":[...]}，返回 metadata.period / metadata.category。
OUTCOME_MARKET_LOOKUP_URL = "https://o1.outcome.xyz/api/v1/markets/lookup"

# outcome id -> metadata.period。None 表示查询成功但没有周期，避免每个轮询都再打一次。
_outcome_period_cache: Dict[str, Optional[str]] = {}


def parse_outcome_description(desc: str) -> Dict[str, str]:
    """解析用 | 分隔的 key:value。只按第一个冒号切开，值里可以再含冒号。

    旧的协议盘：class:priceBinary|underlying:BTC|expiry:YYYYMMDD-HHMM|targetPrice:N|period:15m
    部署盘：perp:BTC|priceDescription:...|seconds:60|threshold:N|time:YYYYMMDD-HHMM
    seconds 会留在字典里，但它不是 period。
    """
    parsed: Dict[str, str] = {}
    if not desc:
        return parsed
    for part in desc.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def parse_utc_expiry(expiry_str: str) -> Optional[float]:
    try:
        dt = datetime.strptime(expiry_str, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _price_binary_terms(item: dict, desc_meta: Dict[str, str]) -> Optional[Tuple[str, str, str]]:
    """返回 (标的, 行权价文本, 到期文本)。不是二元价格盘则返回 None。

    priceTouch / scalarPrice / priceBucket 不在这里面。
    """
    if desc_meta.get("class") == "priceBinary":
        return (
            desc_meta.get("underlying", ""),
            desc_meta.get("targetPrice", ""),
            desc_meta.get("expiry", ""),
        )
    if item.get("name") == "template:binaryPrice":
        return (
            desc_meta.get("perp", ""),
            desc_meta.get("threshold", ""),
            desc_meta.get("time", ""),
        )
    return None


def contract_period(desc_meta: Dict[str, str], outcome_id: str, period_by_outcome_id: Optional[Dict[str, str]]) -> str:
    """合约周期。不读取 seconds。

    描述里的 period: 是 HIP-4 重复盘的周期（如 15m、1d），有它就用它。
    template:binaryPrice 没有这个字段。周期只来自 Outcome lookup 的 metadata.period
    （实盘 BTC 盘是 15m）。seconds 在 binaryPrice 模板里是结算 TWAP 的秒数。
    """
    explicit = (desc_meta.get("period") or "").strip()
    if explicit:
        return explicit
    if not period_by_outcome_id:
        return ""
    raw = period_by_outcome_id.get(str(outcome_id))
    if isinstance(raw, str):
        return raw.strip()
    return ""


def select_outcome_markets(
    outcomes: List[dict],
    *,
    period: str = "15m",
    underlying: Optional[str] = None,
    only_active: bool = True,
    now_ts: Optional[float] = None,
    global_fee_scale: float = 1.0,
    period_by_outcome_id: Optional[Dict[str, str]] = None,
) -> List[OutcomeMarket]:
    """从 outcomeMeta 的 outcomes 里选出二元价格盘。period 为空表示不按周期过滤。"""
    if now_ts is None:
        now_ts = time.time()
    want_period = (period or "").strip().lower()
    want_underlying = underlying.upper() if underlying else ""
    results: List[OutcomeMarket] = []

    for item in outcomes:
        if not isinstance(item, dict):
            continue
        desc = item.get("description", "") or ""
        desc_meta = parse_outcome_description(desc)
        terms = _price_binary_terms(item, desc_meta)
        if terms is None:
            continue
        coin_underlying, strike_text, expiry_text = terms
        coin_underlying = (coin_underlying or "").upper()
        if want_underlying and coin_underlying != want_underlying:
            continue

        expiry_ts = parse_utc_expiry(expiry_text or "")
        if expiry_ts is None:
            continue
        if only_active and expiry_ts <= now_ts:
            continue

        try:
            oid_str = str(item["outcome"])
            oid_int = int(oid_str)
            strike = float(strike_text)
        except (KeyError, ValueError, TypeError):
            continue

        resolved_period = contract_period(desc_meta, oid_str, period_by_outcome_id)
        if want_period and resolved_period.lower() != want_period:
            continue

        deployer_scale = item.get("deployerFeeScale")
        fee_scale = float(deployer_scale) if deployer_scale is not None else global_fee_scale
        results.append(
            OutcomeMarket(
                outcome_id=oid_str,
                name=str(item.get("name", f"{coin_underlying}-{resolved_period}-{oid_str}")),
                underlying=coin_underlying,
                strike=strike,
                expiry_ts=expiry_ts,
                period=resolved_period,
                fee_scale=fee_scale,
                yes_coin=f"#{oid_str}0",
                no_coin=f"#{oid_str}1",
                yes_asset_id=100_000_000 + oid_int * 10 + 0,
                no_asset_id=100_000_000 + oid_int * 10 + 1,
                raw_description=desc,
            )
        )

    results.sort(key=lambda m: (m.expiry_ts, m.outcome_id))
    return results


def _template_binary_ids_for_lookup(outcomes: List[dict], underlying: Optional[str]) -> List[int]:
    """没有 period: 的 template:binaryPrice，才需要向 Outcome 查 metadata.period。"""
    want = underlying.upper() if underlying else ""
    ids: List[int] = []
    for item in outcomes:
        if not isinstance(item, dict) or item.get("name") != "template:binaryPrice":
            continue
        desc_meta = parse_outcome_description(item.get("description", "") or "")
        if (desc_meta.get("period") or "").strip():
            continue
        if want and (desc_meta.get("perp") or "").upper() != want:
            continue
        try:
            ids.append(int(item["outcome"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def lookup_outcome_periods(outcome_ids: List[int], timeout: float = 12.0) -> Dict[str, Optional[str]]:
    """向 Outcome lookup 取 metadata.period。

    返回的值是周期字符串，或 None（这条盘在响应里，但没有 period）。
    响应里没有的 id 不会出现在结果里。不读取描述中的 seconds。
    """
    found: Dict[str, Optional[str]] = {}
    if not outcome_ids:
        return found
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "outcometool",
    }
    for start in range(0, len(outcome_ids), 40):
        chunk = [int(x) for x in outcome_ids[start:start + 40]]
        resp = requests.post(
            OUTCOME_MARKET_LOOKUP_URL,
            json={"outcome_ids": chunk},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict) or payload.get("success") is False:
            raise ValueError("Outcome lookup 没有返回成功结果")
        markets = (payload.get("data") or {}).get("markets") or {}
        if not isinstance(markets, dict):
            raise ValueError("Outcome lookup 的 markets 不是对象")
        for key, row in markets.items():
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                found[str(key)] = None
                continue
            raw = metadata.get("period")
            found[str(key)] = raw.strip() if isinstance(raw, str) and raw.strip() else None
    return found


def fetch_outcome_markets(
    network: str = "mainnet",
    period: str = "15m",
    underlying: Optional[str] = None,
    only_active: bool = True,
    now_ts: Optional[float] = None,
) -> List[OutcomeMarket]:
    """拉取 outcomeMeta，并按标的和合约周期筛选二元价格盘。

    默认周期仍是 15m。template:binaryPrice 的 seconds 不参与周期判断。
    """
    url = ENDPOINTS[network]["rest"]
    resp = requests.post(url, json={"type": "outcomeMeta"}, timeout=12)
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, dict):
        outcomes = payload.get("outcomes", []) or []
        global_fee_scale = float(payload.get("feeScale", 1.0) or 1.0)
    elif isinstance(payload, list):
        outcomes = payload
        global_fee_scale = 1.0
    else:
        return []
    if not isinstance(outcomes, list):
        return []

    if now_ts is None:
        now_ts = time.time()

    lookup_ids = _template_binary_ids_for_lookup(outcomes, underlying)
    lookup_error = ""
    missing = [oid for oid in lookup_ids if str(oid) not in _outcome_period_cache]
    if missing:
        try:
            fresh = lookup_outcome_periods(missing)
        except (requests.RequestException, ValueError, TypeError) as exc:
            lookup_error = str(exc)
            fresh = {}
        for oid, value in fresh.items():
            _outcome_period_cache[str(oid)] = value

    period_by_outcome_id: Dict[str, str] = {}
    for oid in lookup_ids:
        cached = _outcome_period_cache.get(str(oid))
        if isinstance(cached, str) and cached:
            period_by_outcome_id[str(oid)] = cached

    results = select_outcome_markets(
        outcomes,
        period=period,
        underlying=underlying,
        only_active=only_active,
        now_ts=now_ts,
        global_fee_scale=global_fee_scale,
        period_by_outcome_id=period_by_outcome_id,
    )
    if lookup_error:
        print(f"[Markets] Outcome metadata.period 查询失败，未标注周期的盘口没有入选：{lookup_error}")
    if (period or "").strip() and not results:
        seen = select_outcome_markets(
            outcomes,
            period="",
            underlying=underlying,
            only_active=only_active,
            now_ts=now_ts,
            global_fee_scale=global_fee_scale,
            period_by_outcome_id=period_by_outcome_id,
        )
        counts: Dict[str, int] = {}
        for market in seen:
            label = market.period or "(无 period)"
            counts[label] = counts.get(label, 0) + 1
        count_text = ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "无"
        print(
            f"[Markets] 没有匹配 underlying={underlying or '*'} period={period} 的活跃二元价格盘。"
            f" 已识别周期: {count_text}。"
            " template:binaryPrice 的 seconds 是结算 TWAP 秒数，不用作合约周期；"
            " 没有 period: 时只采用 Outcome metadata.period。"
        )
    return results


def fetch_l2_book_rest(network: str, coin: str) -> Optional[dict]:
    url = ENDPOINTS[network]["rest"]
    resp = requests.post(url, json={"type": "l2Book", "coin": coin}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def parse_book_levels(book_data: dict) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """返回 side 0 (Yes/UP) 的 bids[(px, sz), ...] 与 asks[(px, sz), ...]"""
    levels = book_data.get("levels", [[], []])
    bids_raw = levels[0] if len(levels) > 0 else []
    asks_raw = levels[1] if len(levels) > 1 else []
    bids = [(float(x["px"]), float(x["sz"])) for x in bids_raw if float(x["sz"]) > 0]
    asks = [(float(x["px"]), float(x["sz"])) for x in asks_raw if float(x["sz"]) > 0]
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    return bids, asks


# ============================================================
# 2.5 唯一逐档撮合与盘口报价工具（纸面与回放共用）
# ============================================================

@dataclass
class SideQuote:
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    bid_sz: float
    ask_sz: float


@dataclass
class DetailedFill:
    filled_shares: int
    vwap: float
    cost_usdc: float
    levels_consumed: int
    worst_fill_px: float
    top_ask_px: float
    top_ask_usdc: float
    total_avail_usdc: float
    depth_covers_target: bool
    fill_ratio: float


def extract_side_quote(
    side: str,
    yes_bids: List[Tuple[float, float]],
    yes_asks: List[Tuple[float, float]],
) -> SideQuote:
    """将 side 0 (Yes/UP) 合并订单簿映射为目标方向的标准买卖一与中间价。缺单边时对应价为 None。"""
    if side == "UP":
        bid = yes_bids[0][0] if yes_bids else None
        bid_sz = yes_bids[0][1] if yes_bids else 0.0
        ask = yes_asks[0][0] if yes_asks else None
        ask_sz = yes_asks[0][1] if yes_asks else 0.0
    else:
        ask = round(1.0 - yes_bids[0][0], 5) if yes_bids else None
        ask_sz = yes_bids[0][1] if yes_bids else 0.0
        bid = round(1.0 - yes_asks[0][0], 5) if yes_asks else None
        bid_sz = yes_asks[0][1] if yes_asks else 0.0

    mid = round((bid + ask) / 2.0, 5) if (bid is not None and ask is not None) else None
    spread = round(ask - bid, 5) if (bid is not None and ask is not None) else None
    return SideQuote(bid=bid, ask=ask, mid=mid, spread=spread, bid_sz=bid_sz, ask_sz=ask_sz)


def simulate_clob_buy_fill_detailed(
    side: str,
    target_usdc: float,
    yes_bids: List[Tuple[float, float]],
    yes_asks: List[Tuple[float, float]],
    min_notional: float = 1.0,
) -> Optional[DetailedFill]:
    """
    唯一逐档吃单实现（整股约束 + 最小名义约束）：
    - 买 UP (Yes): 吃 side 0 asks，成交价 = px
    - 买 DOWN (No): 镜像吃 side 0 bids，成交价 = round(1 - px, 5)
    """
    if side == "UP":
        ladder = [(px, sz) for px, sz in yes_asks if 0.0 < px < 1.0 and sz > 0]
    else:
        ladder = [(round(1.0 - px, 5), sz) for px, sz in yes_bids if 0.0 < px < 1.0 and sz > 0]

    if not ladder:
        return None

    total_avail_usdc = sum(px * math.floor(sz) for px, sz in ladder)
    top_ask_px = ladder[0][0]
    top_ask_usdc = top_ask_px * math.floor(ladder[0][1])

    remaining_usdc = target_usdc
    total_shares = 0
    total_cost = 0.0
    levels_consumed = 0
    worst_fill_px = top_ask_px

    for px, avail_sz in ladder:
        if px <= 0.0 or remaining_usdc < px:
            break
        max_afford = int(math.floor(remaining_usdc / px))
        take_shares = min(int(math.floor(avail_sz)), max_afford)
        if take_shares <= 0:
            continue
        cost = take_shares * px
        total_shares += take_shares
        total_cost += cost
        remaining_usdc -= cost
        levels_consumed += 1
        worst_fill_px = px
        if remaining_usdc < px:
            break

    if total_shares <= 0 or total_cost < min_notional:
        return None

    vwap = total_cost / total_shares
    return DetailedFill(
        filled_shares=total_shares,
        vwap=vwap,
        cost_usdc=total_cost,
        levels_consumed=levels_consumed,
        worst_fill_px=worst_fill_px,
        top_ask_px=top_ask_px,
        top_ask_usdc=top_ask_usdc,
        total_avail_usdc=total_avail_usdc,
        depth_covers_target=(total_avail_usdc >= target_usdc),
        fill_ratio=(total_cost / target_usdc) if target_usdc > 0 else 0.0,
    )


def simulate_clob_buy_fill(
    side: str,
    target_usdc: float,
    yes_bids: List[Tuple[float, float]],
    yes_asks: List[Tuple[float, float]],
    min_notional: float = 1.0,
) -> Optional[Tuple[int, float, float]]:
    """simulate_clob_buy_fill_detailed 的薄封装，返回 (filled_shares, vwap_price, total_cost_usdc)。"""
    fill = simulate_clob_buy_fill_detailed(
        side=side,
        target_usdc=target_usdc,
        yes_bids=yes_bids,
        yes_asks=yes_asks,
        min_notional=min_notional,
    )
    if fill is None:
        return None
    return fill.filled_shares, fill.vwap, fill.cost_usdc


def decompose_slippage(fill_vwap: float, sig_q: SideQuote, exec_q: SideQuote) -> dict:
    return {
        "slip_vs_sig_mid": (fill_vwap - sig_q.mid) if sig_q.mid is not None else None,
        "slip_vs_exec_mid": (fill_vwap - exec_q.mid) if exec_q.mid is not None else None,
        "slip_vs_sig_ask": (fill_vwap - sig_q.ask) if sig_q.ask is not None else None,
        "slip_book_walk": (fill_vwap - exec_q.ask) if exec_q.ask is not None else None,
        "slip_latency_drift": (
            (exec_q.ask - sig_q.ask) if (exec_q.ask is not None and sig_q.ask is not None) else None
        ),
    }


# ============================================================
# 3. SQLite 影子录制层（经由共享采集层 market_data 驱动）
# ============================================================

def compute_strict_twap(
    ticks: List[Tuple[float, float]],
    window_start: float,
    window_end: float,
) -> Optional[float]:
    """
    严格按 HIP-4 结算规则计算 [window_start, window_end) 内的时间加权均价（TWAP）。
    恰好落在 window_end 的价格权重为 0。
    """
    if not ticks or window_end <= window_start:
        return None

    ticks_sorted = sorted(ticks, key=lambda x: x[0])
    # 过滤掉 >= window_end 的点（权重为 0）
    valid = [(t, p) for t, p in ticks_sorted if t < window_end]
    if not valid:
        return None

    # 若窗口开始前已有最近一笔 markPx，将其作为 window_start 处的初始状态
    before = [x for x in valid if x[0] <= window_start]
    inside = [x for x in valid if x[0] > window_start]

    timeline: List[Tuple[float, float]] = []
    if before:
        timeline.append((window_start, before[-1][1]))
    timeline.extend(inside)

    if not timeline:
        return None

    weighted_sum = 0.0
    total_dur = 0.0
    for i, (t_curr, px) in enumerate(timeline):
        t_next = timeline[i + 1][0] if i + 1 < len(timeline) else window_end
        dt = max(0.0, min(t_next, window_end) - max(t_curr, window_start))
        if dt > 0:
            weighted_sum += px * dt
            total_dur += dt

    if total_dur <= 0:
        return None
    return weighted_sum / total_dur


async def record_live_data(
    network: str,
    underlying: str,
    period: str,
    db_path: str,
    duration_min: float,
    poll_meta_sec: float = 20.0,
):
    """
    异步录制器（通过共享采集层 market_data.LiveMarketFeed 驱动）：
    1) 自动发现活跃的二元价格盘并写入 SQLite
    2) 订阅 HyperCore 永续 activeAssetCtx (markPx) 与目标盘口 Yes 侧 l2Book (#outcomeId0)
    3) 断线自动重连，重连后立即 REST 补拉 l2Book 快照防止丢状态
    4) 到期后自动按 [expiry-60s, expiry) 严格计算 markPx TWAP 写入结算结果
    """
    stop_ts = time.time() + duration_min * 60.0
    underlying = underlying.upper()
    writer = MarketTapeWriter(db_path)

    def handle_markets(_markets, newly_subscribed):
        for m in newly_subscribed:
            print(
                f"[Recorder] 已订阅 {m.yes_coin} | Strike={m.strike} | "
                f"Expiry={datetime.fromtimestamp(m.expiry_ts, tz=timezone.utc).strftime('%H:%M UTC')} | "
                f"feeScale={m.fee_scale}"
            )

    def handle_settled(settled_rows):
        for oid, u_sym, strike, twap, outcome_up in settled_rows:
            print(
                f"[Settle] Market #{oid} ({u_sym}) | Strike={strike:.2f} | "
                f"60s TWAP={twap:.4f} -> {'UP (Yes)' if outcome_up == 1 else 'DOWN (No)'}"
            )

    def handle_disconnect(exc: Exception):
        print(f"[Recorder] WS 断开或异常 ({exc})，3 秒后重连并补拉盘口快照...")

    feed = LiveMarketFeed(
        network=network,
        underlying=underlying,
        period=period,
        ws_url=ENDPOINTS[network]["ws"],
        poll_meta_sec=poll_meta_sec,
        ping_interval_sec=25.0,
        recv_timeout_sec=5.0,
        reconnect_delay_sec=3.0,
        tape_writer=writer,
        auto_settle_tape=True,
        fetch_markets_fn=fetch_outcome_markets,
        fetch_l2_book_fn=fetch_l2_book_rest,
        parse_book_fn=parse_book_levels,
        compute_twap_fn=compute_strict_twap,
    )

    print(f"[Recorder] 启动录制 ({network}) | underlying={underlying} | period={period} | DB={db_path}")
    try:
        await feed.run(
            stop_ts=stop_ts,
            on_markets=handle_markets,
            on_settled=handle_settled,
            on_disconnect=handle_disconnect,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        writer.close()
        print("[Recorder] 录制结束。")


# ============================================================
# 4. 策略核心：唯一 60 秒 TWAP 亚式厚尾二元定价与信号评价器
# ============================================================

class RollingMarkVol:
    """基于 HyperCore markPx 秒级重采样的双周期波动率估计器"""
    def __init__(self, lam: float = 0.92, min_ann_vol: float = 0.25):
        self.lam = lam
        self.min_ann_vol = min_ann_vol
        self.last_px: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.fast_var_per_sec = (min_ann_vol ** 2) / (365.0 * 86400.0)
        self.sec_vars: List[float] = []

    def update(self, ts: float, mark_px: float) -> float:
        if self.last_px is None:
            self.last_px = mark_px
            self.last_ts = ts
            return self.min_ann_vol

        dt = ts - self.last_ts
        if dt >= 1.0:
            ret = math.log(mark_px / self.last_px)
            inst_var_sec = (ret ** 2) / dt
            self.fast_var_per_sec = self.lam * self.fast_var_per_sec + (1.0 - self.lam) * inst_var_sec
            self.sec_vars.append(inst_var_sec)
            if len(self.sec_vars) > 1800:  # 保留约 30 分钟背景方差
                self.sec_vars.pop(0)
            self.last_px = mark_px
            self.last_ts = ts

        ann_fast = math.sqrt(self.fast_var_per_sec * 365.0 * 86400.0)
        ann_slow = (
            math.sqrt(float(np.mean(self.sec_vars)) * 365.0 * 86400.0)
            if len(self.sec_vars) >= 60
            else ann_fast
        )
        return max(0.65 * ann_fast + 0.35 * ann_slow, ann_slow * 0.9, self.min_ann_vol)


@dataclass
class EvaluationDecision:
    should_trade: bool
    filter_reason: str
    outcome_id: str
    ts: float
    tau_sec: float
    strike: float
    mark_px: float
    acc_twap: Optional[float]
    ann_vol: float
    p_up: float
    p_up_normal: float
    z_score: float
    up_bid: Optional[float]
    up_ask: Optional[float]
    down_bid: Optional[float]
    down_ask: Optional[float]
    edge_up: Optional[float]
    edge_down: Optional[float]
    best_side: str
    best_edge: Optional[float]
    active_min_edge: float
    gamma_penalty_applied: float
    eff_settle_fee: float
    book_age_ms: float
    book_stale: bool
    equity_at_signal: float
    market_trades_used_before: int
    kelly_full: Optional[float]
    kelly_scaled: Optional[float]
    target_risk_pct: Optional[float]
    kelly_capped: Optional[bool]
    target_usdc: Optional[float]
    side: Optional[str] = None
    fair_prob: Optional[float] = None
    top_ask: Optional[float] = None
    expected_edge: Optional[float] = None


class HIP4AsianBinaryStrategy:
    """
    针对 Outcome.xyz 60 秒 TWAP 结算机制定制的唯一定价与信号引擎：
    - tau > 60s : 未来 60s 均值方差有效时间 tau_eff = tau - 40s
    - tau <= 60s: 结合已锁定 TWAP 计算剩余有效行权价 K_eff，剩余方差时间 tau_eff = tau / 3
    - evaluate_with_audit / evaluate 只做评估，绝不在信号时增加 market_trades；
      只有成交 (FILLED) 后才调用 record_fill(outcome_id) 占用单场额度。
    """
    def __init__(self, cfg: StrategyConfig, stale_book_ms: float = 5000.0):
        self.cfg = cfg
        self.stale_book_ms = stale_book_ms
        self.peak_equity = cfg.initial_equity
        self.halted = False
        self.market_trades: Dict[str, int] = {}
        self.in_flight_markets: set[str] = set()

    def sync_drawdown_state(self, equity: float, peak_equity: float):
        self.peak_equity = max(self.peak_equity, peak_equity, equity)
        if self.peak_equity > 0 and (self.peak_equity - equity) / self.peak_equity >= self.cfg.daily_drawdown_limit:
            self.halted = True

    def record_fill(self, outcome_id: str) -> int:
        """仅在订单状态确认 FILLED 后调用，累加该场次已用额度。"""
        used = self.market_trades.get(outcome_id, 0) + 1
        self.market_trades[outcome_id] = used
        return used

    def fair_prob_up_twap(
        self,
        mark_px: float,
        strike: float,
        tau_sec: float,
        ann_vol: float,
        acc_twap: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """返回 (p_up_student_t, p_up_normal_baseline, z_score)"""
        if tau_sec <= 0.0:
            ref = acc_twap if acc_twap is not None else mark_px
            p = 1.0 if ref > strike else 0.0
            return p, p, 0.0

        # 情况 A：尚未进入最后 60 秒 TWAP 窗口
        if tau_sec > 60.0 or acc_twap is None:
            tau_eff_sec = max(1.0, tau_sec - 40.0)  # (tau - 60) + 60/3
            eff_strike = strike
        # 情况 B：已进入 [expiry-60s, expiry) 锁定窗口
        else:
            w_rem = tau_sec / 60.0
            w_done = 1.0 - w_rem
            eff_strike = (strike - w_done * acc_twap) / max(w_rem, 1e-6)
            if eff_strike <= 0.0:
                return 0.9999, 0.9999, 10.0
            tau_eff_sec = max(0.5, tau_sec / 3.0)

        tau_yr = tau_eff_sec / (365.0 * 86400.0)
        sigma_sqrt_t = ann_vol * math.sqrt(tau_yr)
        if sigma_sqrt_t < 1e-9:
            p = 1.0 if mark_px > eff_strike else 0.0
            return p, p, 0.0

        z = (math.log(mark_px / eff_strike) - 0.5 * (ann_vol ** 2) * tau_yr) / sigma_sqrt_t
        scale_adj = math.sqrt((self.cfg.df_t - 2.0) / self.cfg.df_t)
        p_t = float(student_t.cdf(z / scale_adj, df=self.cfg.df_t))
        p_norm = float(norm.cdf(z))
        return p_t, p_norm, z

    def evaluate_with_audit(
        self,
        market: OutcomeMarket,
        ts: float,
        mark_px: float,
        ann_vol: float,
        yes_bids: List[Tuple[float, float]],
        yes_asks: List[Tuple[float, float]],
        book_ts: float,
        acc_twap: Optional[float],
        equity: float,
        peak_equity: float,
    ) -> EvaluationDecision:
        """唯一信号评价实现（纸面与回放共用）。只读 market_trades，不修改额度。"""
        self.sync_drawdown_state(equity, peak_equity)

        tau_sec = market.expiry_ts - ts
        p_up, p_up_norm, z = self.fair_prob_up_twap(mark_px, market.strike, tau_sec, ann_vol, acc_twap)
        p_down = 1.0 - p_up

        if tau_sec > 300.0:
            base_edge = self.cfg.edge_early
        elif tau_sec > 60.0:
            base_edge = self.cfg.edge_mid
        else:
            base_edge = self.cfg.edge_twap_zone

        gamma_pen = (
            self.cfg.gamma_penalty
            if (tau_sec <= self.cfg.gamma_zone_tau and abs(z) < self.cfg.gamma_zone_z)
            else 0.0
        )
        active_min_edge = base_edge + gamma_pen
        eff_settle_fee = self.cfg.settlement_fee_rate * market.fee_scale
        net_payout = 1.0 - eff_settle_fee

        up_q = extract_side_quote("UP", yes_bids, yes_asks)
        down_q = extract_side_quote("DOWN", yes_bids, yes_asks)
        edge_up = (p_up * net_payout - up_q.ask) if up_q.ask is not None else None
        edge_down = (p_down * net_payout - down_q.ask) if down_q.ask is not None else None

        best_side = ""
        best_edge: Optional[float] = None
        if edge_up is not None and (edge_down is None or edge_up >= edge_down):
            best_side, best_edge = "UP", edge_up
        elif edge_down is not None:
            best_side, best_edge = "DOWN", edge_down

        book_age_ms = max(0.0, (ts - book_ts) * 1000.0)
        book_stale = book_age_ms > self.stale_book_ms
        used_before = self.market_trades.get(market.outcome_id, 0)

        def make_reject(
            reason: str,
            k_full: Optional[float] = None,
            k_scaled: Optional[float] = None,
            t_risk: Optional[float] = None,
            k_cap: Optional[bool] = None,
            t_usdc: Optional[float] = None,
        ) -> EvaluationDecision:
            return EvaluationDecision(
                should_trade=False,
                filter_reason=reason,
                outcome_id=market.outcome_id,
                ts=ts,
                tau_sec=tau_sec,
                strike=market.strike,
                mark_px=mark_px,
                acc_twap=acc_twap,
                ann_vol=ann_vol,
                p_up=p_up,
                p_up_normal=p_up_norm,
                z_score=z,
                up_bid=up_q.bid,
                up_ask=up_q.ask,
                down_bid=down_q.bid,
                down_ask=down_q.ask,
                edge_up=edge_up,
                edge_down=edge_down,
                best_side=best_side,
                best_edge=best_edge,
                active_min_edge=active_min_edge,
                gamma_penalty_applied=gamma_pen,
                eff_settle_fee=eff_settle_fee,
                book_age_ms=book_age_ms,
                book_stale=book_stale,
                equity_at_signal=equity,
                market_trades_used_before=used_before,
                kelly_full=k_full,
                kelly_scaled=k_scaled,
                target_risk_pct=t_risk,
                kelly_capped=k_cap,
                target_usdc=t_usdc,
            )

        if self.halted:
            return make_reject("HALTED_DRAWDOWN")
        if tau_sec < self.cfg.min_tau_sec:
            return make_reject("TAU_TOO_SMALL")
        if tau_sec > self.cfg.max_tau_sec:
            return make_reject("TAU_TOO_LARGE")
        if used_before >= self.cfg.max_trades_per_market:
            return make_reject("MAX_TRADES_REACHED")
        if market.outcome_id in self.in_flight_markets:
            return make_reject("ORDER_IN_FLIGHT")
        if not yes_bids and not yes_asks:
            return make_reject("EMPTY_BOOK")

        edge_pass = []
        if edge_up is not None and edge_up > active_min_edge:
            edge_pass.append(("UP", edge_up, up_q.ask, p_up))
        if edge_down is not None and edge_down > active_min_edge:
            edge_pass.append(("DOWN", edge_down, down_q.ask, p_down))

        if not edge_pass:
            return make_reject("EDGE_BELOW_THRESHOLD")

        edge_pass.sort(key=lambda x: x[1], reverse=True)
        band_pass = [
            c for c in edge_pass
            if c[2] is not None and (self.cfg.min_entry_price <= c[2] <= self.cfg.max_entry_price)
        ]
        if not band_pass:
            return make_reject("PRICE_OUT_OF_BAND")

        side, edge, top_ask, fair_p = band_pass[0]

        kelly_full = edge / max(1e-6, (net_payout - top_ask))
        kelly_scaled = self.cfg.kelly_fraction * kelly_full
        kelly_capped = kelly_scaled > self.cfg.max_market_risk_pct
        target_risk_pct = min(self.cfg.max_market_risk_pct, kelly_scaled)
        target_usdc = round(max(0.0, equity) * target_risk_pct, 2)

        if target_usdc < self.cfg.min_order_notional:
            return make_reject(
                "STAKE_TOO_SMALL",
                k_full=kelly_full,
                k_scaled=kelly_scaled,
                t_risk=target_risk_pct,
                k_cap=kelly_capped,
                t_usdc=target_usdc,
            )

        return EvaluationDecision(
            should_trade=True,
            filter_reason="SIGNAL_TRIGGERED",
            outcome_id=market.outcome_id,
            ts=ts,
            tau_sec=tau_sec,
            strike=market.strike,
            mark_px=mark_px,
            acc_twap=acc_twap,
            ann_vol=ann_vol,
            p_up=p_up,
            p_up_normal=p_up_norm,
            z_score=z,
            up_bid=up_q.bid,
            up_ask=up_q.ask,
            down_bid=down_q.bid,
            down_ask=down_q.ask,
            edge_up=edge_up,
            edge_down=edge_down,
            best_side=side,
            best_edge=edge,
            active_min_edge=active_min_edge,
            gamma_penalty_applied=gamma_pen,
            eff_settle_fee=eff_settle_fee,
            book_age_ms=book_age_ms,
            book_stale=book_stale,
            equity_at_signal=equity,
            market_trades_used_before=used_before,
            kelly_full=kelly_full,
            kelly_scaled=kelly_scaled,
            target_risk_pct=target_risk_pct,
            kelly_capped=kelly_capped,
            target_usdc=target_usdc,
            side=side,
            fair_prob=fair_p,
            top_ask=top_ask,
            expected_edge=edge,
        )

    def evaluate(
        self,
        market: OutcomeMarket,
        ts: float,
        mark_px: float,
        ann_vol: float,
        yes_bids: List[Tuple[float, float]],
        yes_asks: List[Tuple[float, float]],
        acc_twap: Optional[float],
        equity: float,
        book_ts: Optional[float] = None,
        peak_equity: Optional[float] = None,
    ) -> Optional[dict]:
        """evaluate_with_audit 的薄封装。只返回信号字典，不修改 market_trades。"""
        dec = self.evaluate_with_audit(
            market=market,
            ts=ts,
            mark_px=mark_px,
            ann_vol=ann_vol,
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            book_ts=ts if book_ts is None else book_ts,
            acc_twap=acc_twap,
            equity=equity,
            peak_equity=self.peak_equity if peak_equity is None else peak_equity,
        )
        if not dec.should_trade:
            return None
        return {
            "outcome_id": market.outcome_id,
            "signal_ts": ts,
            "side": dec.side,
            "fair_prob": dec.fair_prob,
            "top_ask": dec.top_ask,
            "expected_edge": dec.expected_edge,
            "target_usdc": dec.target_usdc,
            "tau_sec": dec.tau_sec,
            "z_score": dec.z_score,
            "eff_settle_fee": dec.eff_settle_fee,
        }


# ============================================================
# 5. 统一回放入口（转调 export-backtest 同一路径，不再维护独立回测语义）
# ============================================================

def run_replay_backtest(
    db_path: str,
    cfg: StrategyConfig,
    latency_ms: int = 300,
    *,
    max_slippage: float = 0.008,
    stale_book_ms: float = 5000.0,
    decision_sample_sec: float = 30.0,
    near_edge_margin: float = 0.015,
    out_dir: str = "bt_exports",
    out_root: str = "",
    subdir_by_run_id: bool = False,
    network: str = "mainnet",
) -> Optional[dict]:
    """与 outcome_paper_export.py export-backtest 共用同一条回放、撮合与账本路径。"""
    from outcome_paper_export import run_unified_replay_backtest

    ok, summary = run_unified_replay_backtest(
        db_path=db_path,
        cfg=cfg,
        latency_ms=latency_ms,
        max_slippage=max_slippage,
        stale_book_ms=stale_book_ms,
        decision_sample_sec=decision_sample_sec,
        near_edge_margin=near_edge_margin,
        out_dir_arg=out_dir,
        out_root_arg=out_root,
        subdir_by_run_id=subdir_by_run_id,
        network=network,
    )
    return summary if ok else None


# ============================================================
# 6. CLI 入口
# ============================================================

def cmd_discover(args):
    markets = fetch_outcome_markets(
        network=args.network,
        period=args.period if args.period != "all" else "",
        underlying=args.underlying or None,
        only_active=not args.include_expired,
    )
    print(f"[{args.network}] 匹配 class:priceBinary (period={args.period}) 共 {len(markets)} 个市场：")
    for m in markets[:15]:
        exp_str = datetime.fromtimestamp(m.expiry_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        book = fetch_l2_book_rest(args.network, m.yes_coin)
        bids, asks = parse_book_levels(book or {})
        best_bid = bids[0] if bids else (None, 0)
        best_ask = asks[0] if asks else (None, 0)
        print(
            f"  #{m.outcome_id} ({m.yes_coin}) | {m.underlying} {m.period} | "
            f"Strike={m.strike} | Exp={exp_str} | feeScale={m.fee_scale} | "
            f"UP Bid={best_bid[0]}({best_bid[1]}) / Ask={best_ask[0]}({best_ask[1]})"
        )


def main():
    parser = argparse.ArgumentParser(description="Outcome.xyz HIP-4 15m 盘口录制与延迟回放回测引擎")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_disc = sub.add_parser("discover", help="探活拉取 outcomeMeta 与实时 l2Book")
    p_disc.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    p_disc.add_argument("--period", default="15m", help="筛选周期，如 15m / 1d / all")
    p_disc.add_argument("--underlying", default="", help="筛选标的，如 BTC / ETH")
    p_disc.add_argument("--include-expired", action="store_true")

    p_rec = sub.add_parser("record", help="实时录制 l2Book 与 HyperCore markPx 到 SQLite")
    p_rec.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    p_rec.add_argument("--underlying", default="BTC")
    p_rec.add_argument("--period", default="15m")
    p_rec.add_argument(
        "--db",
        default="hip4_replay.sqlite",
        help="行情磁带 SQLite 路径（默认 hip4_replay.sqlite）；不要与 paper 同时并发写同一文件",
    )
    p_rec.add_argument("--minutes", type=float, default=65.0, help="连续录制时长（分钟）")

    p_bt = sub.add_parser("backtest", help="转调与 export-backtest 相同的统一回放路径并导出摘要/CSV")
    p_bt.add_argument("--db", default="hip4_replay.sqlite")
    p_bt.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet")
    p_bt.add_argument("--latency-ms", type=int, default=300)
    p_bt.add_argument("--max-slippage", type=float, default=0.008)
    p_bt.add_argument("--stale-book-ms", type=float, default=5000.0)
    p_bt.add_argument("--equity", type=float, default=1000.0)
    p_bt.add_argument("--settlement-fee", type=float, default=0.0007)
    p_bt.add_argument("--decision-sample-sec", type=float, default=30.0)
    p_bt.add_argument("--near-edge-margin", type=float, default=0.015)
    p_bt.add_argument("--out-dir", default="bt_exports")
    p_bt.add_argument("--out-root", default="")
    p_bt.add_argument("--subdir-by-run-id", action="store_true")

    args = parser.parse_args()
    if args.cmd == "discover":
        cmd_discover(args)
    elif args.cmd == "record":
        asyncio.run(record_live_data(args.network, args.underlying, args.period, args.db, args.minutes))
    elif args.cmd == "backtest":
        cfg = StrategyConfig(initial_equity=args.equity, settlement_fee_rate=args.settlement_fee)
        summary = run_replay_backtest(
            args.db,
            cfg,
            latency_ms=args.latency_ms,
            max_slippage=args.max_slippage,
            stale_book_ms=args.stale_book_ms,
            decision_sample_sec=args.decision_sample_sec,
            near_edge_margin=args.near_edge_margin,
            out_dir=args.out_dir,
            out_root=args.out_root,
            subdir_by_run_id=args.subdir_by_run_id,
            network=args.network,
        )
        if summary is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
