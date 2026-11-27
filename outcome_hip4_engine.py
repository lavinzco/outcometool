import argparse
import asyncio
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import websockets
from scipy.stats import norm, t as student_t


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

def parse_outcome_description(desc: str) -> Dict[str, str]:
    """解析 class:priceBinary|underlying:BTC|expiry:20260503-0600|targetPrice:78213|period:15m"""
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


def fetch_outcome_markets(
    network: str = "mainnet",
    period: str = "15m",
    underlying: Optional[str] = None,
    only_active: bool = True,
) -> List[OutcomeMarket]:
    """调用 POST /info {"type":"outcomeMeta"} 并精准筛选 class:priceBinary 盘口"""
    url = ENDPOINTS[network]["rest"]
    resp = requests.post(url, json={"type": "outcomeMeta"}, timeout=12)
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, dict):
        outcomes = payload.get("outcomes", [])
        global_fee_scale = float(payload.get("feeScale", 1.0) or 1.0)
    elif isinstance(payload, list):
        outcomes = payload
        global_fee_scale = 1.0
    else:
        return []

    now_ts = time.time()
    results: List[OutcomeMarket] = []

    for item in outcomes:
        desc = item.get("description", "") or ""
        meta = parse_outcome_description(desc)

        # 严格过滤：只保留 priceBinary，绝不混入 priceBucket
        if meta.get("class") != "priceBinary":
            continue
        if period and meta.get("period") != period:
            continue

        coin_underlying = meta.get("underlying", "").upper()
        if underlying and coin_underlying != underlying.upper():
            continue

        expiry_ts = parse_utc_expiry(meta.get("expiry", ""))
        if expiry_ts is None:
            continue
        if only_active and expiry_ts <= now_ts:
            continue

        try:
            strike = float(meta["targetPrice"])
            oid_str = str(item["outcome"])
            oid_int = int(oid_str)
        except (KeyError, ValueError, TypeError):
            continue

        deployer_scale = item.get("deployerFeeScale")
        fee_scale = float(deployer_scale) if deployer_scale is not None else global_fee_scale

        results.append(
            OutcomeMarket(
                outcome_id=oid_str,
                name=str(item.get("name", f"{coin_underlying}-{period}-{oid_str}")),
                underlying=coin_underlying,
                strike=strike,
                expiry_ts=expiry_ts,
                period=meta.get("period", ""),
                fee_scale=fee_scale,
                yes_coin=f"#{oid_str}0",
                no_coin=f"#{oid_str}1",
                yes_asset_id=100_000_000 + oid_int * 10 + 0,
                no_asset_id=100_000_000 + oid_int * 10 + 1,
                raw_description=desc,
            )
        )

    results.sort(key=lambda m: m.expiry_ts)
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


def simulate_clob_buy_fill(
    side: str,
    target_usdc: float,
    yes_bids: List[Tuple[float, float]],
    yes_asks: List[Tuple[float, float]],
    min_notional: float = 1.0,
) -> Optional[Tuple[int, float, float]]:
    """
    利用 HIP-4 合并镜像订单簿，按真实深度逐档吃单（整股约束 + 最小名义约束）。
    - 买 UP (Yes): 吃 side 0 asks，成交价 = px
    - 买 DOWN (No): 镜像吃 side 0 bids，成交价 = round(1 - px, 5)
    返回: (filled_shares, vwap_price, total_cost_usdc)
    """
    if side == "UP":
        ladder = [(px, sz) for px, sz in yes_asks if 0.0 < px < 1.0]
    else:
        # yes_bids 已按 px 从高到低排列，因此 1 - px 天然按从低到高（最优卖一）排列
        ladder = [(round(1.0 - px, 5), sz) for px, sz in yes_bids if 0.0 < px < 1.0]

    if not ladder:
        return None

    remaining_usdc = target_usdc
    total_shares = 0
    total_cost = 0.0

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
        if remaining_usdc < px:
            break

    if total_shares <= 0 or total_cost < min_notional:
        return None

    vwap = total_cost / total_shares
    return total_shares, vwap, total_cost


# ============================================================
# 3. SQLite 影子录制层（L2 盘口 + HyperCore markPx + 自动 TWAP 结算）
# ============================================================

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS markets (
            outcome_id TEXT PRIMARY KEY,
            name TEXT,
            underlying TEXT,
            strike REAL,
            expiry_ts REAL,
            period TEXT,
            fee_scale REAL,
            yes_coin TEXT,
            raw_description TEXT,
            settled_twap REAL,
            outcome_up INTEGER
        );
        CREATE TABLE IF NOT EXISTS book_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            outcome_id TEXT NOT NULL,
            bids_json TEXT NOT NULL,
            asks_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_book_oid_ts ON book_snapshots(outcome_id, ts);

        CREATE TABLE IF NOT EXISTS mark_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            underlying TEXT NOT NULL,
            mark_px REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_mark_u_ts ON mark_ticks(underlying, ts);
        """
    )
    conn.commit()
    return conn


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
    异步录制器：
    1) 自动发现活跃的 class:priceBinary 市场并写入 SQLite
    2) 订阅 HyperCore 永续 activeAssetCtx (markPx) 与目标盘口 l2Book (#outcomeId0)
    3) 断线自动重连，重连后立即 REST 补拉 l2Book 快照防止丢状态
    4) 到期后自动按 [expiry-60s, expiry) 严格计算 markPx TWAP 写入结算结果
    """
    conn = init_db(db_path)
    ws_url = ENDPOINTS[network]["ws"]
    stop_ts = time.time() + duration_min * 60.0
    underlying = underlying.upper()

    tracked_markets: Dict[str, OutcomeMarket] = {}
    subscribed_coins: set[str] = set()

    def upsert_market(m: OutcomeMarket):
        conn.execute(
            """
            INSERT OR IGNORE INTO markets
            (outcome_id, name, underlying, strike, expiry_ts, period, fee_scale, yes_coin, raw_description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.outcome_id, m.name, m.underlying, m.strike, m.expiry_ts,
                m.period, m.fee_scale, m.yes_coin, m.raw_description,
            ),
        )
        conn.commit()

    def settle_expired_markets():
        now = time.time()
        rows = conn.execute(
            "SELECT outcome_id, underlying, strike, expiry_ts FROM markets WHERE outcome_up IS NULL AND expiry_ts <= ?",
            (now - 2.0,),
        ).fetchall()
        for oid, u_sym, strike, expiry_ts in rows:
            t_rows = conn.execute(
                "SELECT ts, mark_px FROM mark_ticks WHERE underlying = ? AND ts >= ? AND ts <= ? ORDER BY ts ASC",
                (u_sym, expiry_ts - 120.0, expiry_ts + 2.0),
            ).fetchall()
            twap = compute_strict_twap(t_rows, expiry_ts - 60.0, expiry_ts)
            if twap is not None:
                # 严格高于 strike 结算 UP (1)，等于或低于结算 DOWN (0)
                outcome_up = 1 if twap > strike else 0
                conn.execute(
                    "UPDATE markets SET settled_twap = ?, outcome_up = ? WHERE outcome_id = ?",
                    (twap, outcome_up, oid),
                )
                conn.commit()
                print(
                    f"[Settle] Market #{oid} ({u_sym}) | Strike={strike:.2f} | "
                    f"60s TWAP={twap:.4f} -> {'UP (Yes)' if outcome_up == 1 else 'DOWN (No)'}"
                )

    print(f"[Recorder] 启动录制 ({network}) | underlying={underlying} | period={period} | DB={db_path}")

    while time.time() < stop_ts:
        try:
            async with websockets.connect(ws_url, ping_interval=None, max_size=4 * 1024 * 1024) as ws:
                subscribed_coins.clear()
                # 订阅 HyperCore 永续实时 markPx 与 outcomeMetaUpdates
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "activeAssetCtx", "coin": underlying},
                }))
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "outcomeMetaUpdates"},
                }))

                async def sync_markets_and_subscribe():
                    markets = await asyncio.to_thread(
                        fetch_outcome_markets, network, period, underlying, True
                    )
                    for m in markets:
                        tracked_markets[m.outcome_id] = m
                        upsert_market(m)
                        if m.yes_coin not in subscribed_coins:
                            await ws.send(json.dumps({
                                "method": "subscribe",
                                "subscription": {"type": "l2Book", "coin": m.yes_coin},
                            }))
                            subscribed_coins.add(m.yes_coin)
                            # 重连/首次订阅立即 REST 补拉一次完整快照，避免丢盘口状态
                            snap = await asyncio.to_thread(fetch_l2_book_rest, network, m.yes_coin)
                            if snap and "levels" in snap:
                                bids, asks = parse_book_levels(snap)
                                conn.execute(
                                    "INSERT INTO book_snapshots (ts, outcome_id, bids_json, asks_json) VALUES (?, ?, ?, ?)",
                                    (time.time(), m.outcome_id, json.dumps(bids[:10]), json.dumps(asks[:10])),
                                )
                                conn.commit()
                            print(
                                f"[Recorder] 已订阅 {m.yes_coin} | Strike={m.strike} | "
                                f"Expiry={datetime.fromtimestamp(m.expiry_ts, tz=timezone.utc).strftime('%H:%M UTC')} | "
                                f"feeScale={m.fee_scale}"
                            )

                await sync_markets_and_subscribe()
                last_meta_sync = time.time()
                last_ping = time.time()
                pending_writes = 0

                while time.time() < stop_ts:
                    now = time.time()
                    if now - last_ping >= 25.0:
                        await ws.send(json.dumps({"method": "ping"}))
                        last_ping = now

                    if now - last_meta_sync >= poll_meta_sec:
                        await sync_markets_and_subscribe()
                        settle_expired_markets()
                        last_meta_sync = now

                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue

                    msg = json.loads(raw_msg)
                    channel = msg.get("channel")
                    data = msg.get("data")

                    if channel == "activeAssetCtx" and isinstance(data, dict):
                        ctx = data.get("ctx", {})
                        mark_px_str = ctx.get("markPx")
                        coin_name = str(data.get("coin", underlying)).upper()
                        if mark_px_str is not None and coin_name == underlying:
                            conn.execute(
                                "INSERT INTO mark_ticks (ts, underlying, mark_px) VALUES (?, ?, ?)",
                                (time.time(), underlying, float(mark_px_str)),
                            )
                            pending_writes += 1

                    elif channel == "l2Book" and isinstance(data, dict):
                        coin_str = str(data.get("coin", ""))
                        if coin_str.startswith("#") and coin_str.endswith("0"):
                            oid = coin_str[1:-1]
                            bids, asks = parse_book_levels(data)
                            ts_val = (float(data["time"]) / 1000.0) if "time" in data else time.time()
                            conn.execute(
                                "INSERT INTO book_snapshots (ts, outcome_id, bids_json, asks_json) VALUES (?, ?, ?, ?)",
                                (ts_val, oid, json.dumps(bids[:10]), json.dumps(asks[:10])),
                            )
                            pending_writes += 1

                    elif channel == "outcomeMetaUpdates":
                        await sync_markets_and_subscribe()

                    if pending_writes >= 20:
                        conn.commit()
                        pending_writes = 0

                if pending_writes > 0:
                    conn.commit()
        except Exception as exc:
            print(f"[Recorder] WS 断开或异常 ({exc})，3 秒后重连并补拉盘口快照...")
            conn.commit()
            await asyncio.sleep(3.0)

    settle_expired_markets()
    conn.close()
    print("[Recorder] 录制结束。")


# ============================================================
# 4. 策略核心：支持 60 秒 TWAP 亚式修正的厚尾二元定价器
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


class HIP4AsianBinaryStrategy:
    """
    针对 Outcome.xyz 60 秒 TWAP 结算机制定制的定价与信号引擎：
    - tau > 60s : 未来 60s 均值方差有效时间 tau_eff = tau - 40s
    - tau <= 60s: 结合已锁定 TWAP 计算剩余有效行权价 K_eff，剩余方差时间 tau_eff = tau / 3
    """
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.peak_equity = cfg.initial_equity
        self.halted = False
        self.market_trades: Dict[str, int] = {}

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
    ) -> Optional[dict]:
        self.peak_equity = max(self.peak_equity, equity)
        if (self.peak_equity - equity) / self.peak_equity >= self.cfg.daily_drawdown_limit:
            self.halted = True
        if self.halted:
            return None

        tau_sec = market.expiry_ts - ts
        if not (self.cfg.min_tau_sec <= tau_sec <= self.cfg.max_tau_sec):
            return None
        if self.market_trades.get(market.outcome_id, 0) >= self.cfg.max_trades_per_market:
            return None

        if not yes_bids and not yes_asks:
            return None

        up_ask = yes_asks[0][0] if yes_asks else 1.0
        down_ask = round(1.0 - yes_bids[0][0], 5) if yes_bids else 1.0

        p_up, _, z = self.fair_prob_up_twap(mark_px, market.strike, tau_sec, ann_vol, acc_twap)
        p_down = 1.0 - p_up

        # 分段净优势门槛
        if tau_sec > 300.0:
            min_edge = self.cfg.edge_early
        elif tau_sec > 60.0:
            min_edge = self.cfg.edge_mid
        else:
            min_edge = self.cfg.edge_twap_zone

        # Gamma 雷区惩罚
        if tau_sec <= self.cfg.gamma_zone_tau and abs(z) < self.cfg.gamma_zone_z:
            min_edge += self.cfg.gamma_penalty

        # HIP-4 开仓 0 费率，但持有至结算扣除结算费（按 fee_scale 缩放）
        eff_settle_fee = self.cfg.settlement_fee_rate * market.fee_scale
        # 净期望收益：胜率 * (1 - 结算费) - 入场成本
        edge_up = p_up * (1.0 - eff_settle_fee) - up_ask
        edge_down = p_down * (1.0 - eff_settle_fee) - down_ask

        side, edge, top_ask, fair_p = None, 0.0, 0.0, 0.0
        if (
            edge_up > min_edge
            and edge_up >= edge_down
            and self.cfg.min_entry_price <= up_ask <= self.cfg.max_entry_price
        ):
            side, edge, top_ask, fair_p = "UP", edge_up, up_ask, p_up
        elif (
            edge_down > min_edge
            and self.cfg.min_entry_price <= down_ask <= self.cfg.max_entry_price
        ):
            side, edge, top_ask, fair_p = "DOWN", edge_down, down_ask, p_down

        if side is None:
            return None

        # 分数凯利仓位（按净赔率计算）
        net_payout = 1.0 - eff_settle_fee
        kelly_f = edge / max(1e-6, (net_payout - top_ask))
        risk_pct = min(self.cfg.max_market_risk_pct, self.cfg.kelly_fraction * kelly_f)
        target_usdc = round(equity * risk_pct, 2)
        if target_usdc < self.cfg.min_order_notional:
            return None

        self.market_trades[market.outcome_id] = self.market_trades.get(market.outcome_id, 0) + 1
        return {
            "outcome_id": market.outcome_id,
            "signal_ts": ts,
            "side": side,
            "fair_prob": fair_p,
            "top_ask": top_ask,
            "expected_edge": edge,
            "target_usdc": target_usdc,
            "tau_sec": tau_sec,
            "z_score": z,
            "eff_settle_fee": eff_settle_fee,
        }


# ============================================================
# 5. 延迟回放回测器（真实录制 L2 + markPx 重放）
# ============================================================

def run_replay_backtest(db_path: str, cfg: StrategyConfig, latency_ms: int = 300):
    conn = sqlite3.connect(db_path)
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
        print("[Backtest] 数据库中没有已结算的完整场次，请先运行 record 录制至少 1 个完整窗口。")
        conn.close()
        return

    strat = HIP4AsianBinaryStrategy(cfg)
    vol_trackers: Dict[str, RollingMarkVol] = {}
    equity = cfg.initial_equity
    equity_curve = [equity]
    trade_logs = []
    calib_t, calib_norm, calib_y = [], [], []

    latency_sec = latency_ms / 1000.0

    for row in m_rows:
        m = OutcomeMarket(
            outcome_id=row[0], name=row[1], underlying=row[2], strike=row[3],
            expiry_ts=row[4], period=row[5], fee_scale=row[6], yes_coin=row[7],
            no_coin=f"#{row[0]}1",
            yes_asset_id=100_000_000 + int(row[0]) * 10,
            no_asset_id=100_000_000 + int(row[0]) * 10 + 1,
            raw_description=row[8],
        )
        settled_twap, outcome_up = row[9], int(row[10])

        vol_est = vol_trackers.setdefault(
            m.underlying, RollingMarkVol(cfg.ewma_lambda, cfg.min_ann_vol)
        )

        # 读取该场次前后相关的 markPx 与 L2 快照
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

        book_ts_arr = np.array([b[0] for b in book_rows], dtype=float)
        twap_start = m.expiry_ts - 60.0
        mark_idx = 0
        latest_mark_px: Optional[float] = None
        ann_vol = cfg.min_ann_vol
        twap_buffer: List[Tuple[float, float]] = []
        executed_orders: List[dict] = []

        for b_idx, (bts, bids_s, asks_s) in enumerate(book_rows):
            # 推进 markPx 至当前盘口时间
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
            acc_twap = (
                compute_strict_twap(twap_buffer, twap_start, bts)
                if bts > twap_start
                else None
            )

            # 记录校准指标（每 5 秒采样一次避免高频盘口刷屏扭曲权重）
            if b_idx % 5 == 0 and 10.0 <= tau_sec <= 860.0:
                p_t, p_n, _ = strat.fair_prob_up_twap(
                    latest_mark_px, m.strike, tau_sec, ann_vol, acc_twap
                )
                calib_t.append(p_t)
                calib_norm.append(p_n)
                calib_y.append(outcome_up)

            yes_bids = json.loads(bids_s)
            yes_asks = json.loads(asks_s)
            sig = strat.evaluate(m, bts, latest_mark_px, ann_vol, yes_bids, yes_asks, acc_twap, equity)
            if sig is None:
                continue

            # 注入真实执行延迟：取 t + latency_sec 时刻的盘口做逐档深度撮合
            exec_target_ts = bts + latency_sec
            exec_idx = int(np.searchsorted(book_ts_arr, exec_target_ts, side="left"))
            if exec_idx >= len(book_rows):
                exec_idx = len(book_rows) - 1

            exec_ts, exec_bids_s, exec_asks_s = book_rows[exec_idx]
            fill = simulate_clob_buy_fill(
                side=sig["side"],
                target_usdc=sig["target_usdc"],
                yes_bids=json.loads(exec_bids_s),
                yes_asks=json.loads(exec_asks_s),
                min_notional=cfg.min_order_notional,
            )
            if fill is None:
                continue

            shares, vwap, cost_usdc = fill
            # 延迟后如果滑点把优势全吃掉则放弃（模拟 IOC 限价保护）
            max_limit_px = min(cfg.max_entry_price, sig["top_ask"] + 0.008)
            if vwap > max_limit_px:
                continue

            executed_orders.append({
                **sig,
                "exec_ts": exec_ts,
                "shares": shares,
                "vwap": vwap,
                "cost_usdc": cost_usdc,
                "slippage": vwap - sig["top_ask"],
            })

        # 到期结算（开仓费=0，获胜兑付按 1 - eff_settle_fee 结算）
        for od in executed_orders:
            won = (od["side"] == "UP" and outcome_up == 1) or (od["side"] == "DOWN" and outcome_up == 0)
            payout = od["shares"] * (1.0 - od["eff_settle_fee"]) if won else 0.0
            pnl = payout - od["cost_usdc"]
            equity += pnl
            equity_curve.append(equity)
            trade_logs.append({
                "oid": m.outcome_id,
                "side": od["side"],
                "shares": od["shares"],
                "vwap": round(od["vwap"], 4),
                "slip": round(od["slippage"], 4),
                "fair": round(od["fair_prob"], 3),
                "tau_s": int(od["tau_sec"]),
                "twap": round(settled_twap, 2),
                "strike": round(m.strike, 2),
                "won": won,
                "pnl": round(pnl, 2),
                "equity": round(equity, 2),
            })

    conn.close()

    eq_arr = np.array(equity_curve)
    max_dd = float(np.max((np.maximum.accumulate(eq_arr) - eq_arr) / np.maximum.accumulate(eq_arr)))

    print("\n================ Outcome HIP-4 真实盘口回放报告 ================")
    print(f"已结算回测场次数     : {len(m_rows)} | 模拟链路延迟: {latency_ms} ms")
    if calib_t:
        brier_t = float(np.mean((np.array(calib_t) - np.array(calib_y)) ** 2))
        brier_n = float(np.mean((np.array(calib_norm) - np.array(calib_y)) ** 2))
        print(f"TWAP 厚尾 Brier      : {brier_t:.5f} (正态基准: {brier_n:.5f}，越低越好)")
    print(f"实际成交笔数         : {len(trade_logs)}")
    if trade_logs:
        tdf = pd.DataFrame(trade_logs)
        ret_pct = (equity / cfg.initial_equity - 1.0) * 100.0
        print(f"胜率                 : {tdf['won'].mean() * 100:.2f}%")
        print(f"平均延迟滑点         : {tdf['slip'].mean():.5f}")
        print(f"期末净值 / 收益率    : {equity:.2f} USDC ({ret_pct:+.2f}%)")
        print(f"最大回撤 (MaxDD)     : {max_dd * 100:.2f}%")
        print("\n交易回放明细：")
        print(tdf.to_string(index=False))
    else:
        print("当前录制样本与延迟条件下无触发交易（可放宽 edge 阈值或延长录制周期）。")


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
    p_rec.add_argument("--db", default="hip4_replay.sqlite")
    p_rec.add_argument("--minutes", type=float, default=65.0, help="连续录制时长（分钟）")

    p_bt = sub.add_parser("backtest", help="基于录制的真实盘口跑延迟滑点回测")
    p_bt.add_argument("--db", default="hip4_replay.sqlite")
    p_bt.add_argument("--latency-ms", type=int, default=300)
    p_bt.add_argument("--equity", type=float, default=1000.0)
    p_bt.add_argument("--settlement-fee", type=float, default=0.0007)

    args = parser.parse_args()
    if args.cmd == "discover":
        cmd_discover(args)
    elif args.cmd == "record":
        asyncio.run(record_live_data(args.network, args.underlying, args.period, args.db, args.minutes))
    elif args.cmd == "backtest":
        cfg = StrategyConfig(initial_equity=args.equity, settlement_fee_rate=args.settlement_fee)
        run_replay_backtest(args.db, cfg, latency_ms=args.latency_ms)


if __name__ == "__main__":
    main()