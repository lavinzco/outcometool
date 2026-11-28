import asyncio
import inspect
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import websockets


# ============================================================
# 1. 共享事件数据结构
# ============================================================

@dataclass
class MarkEvent:
    underlying: str
    mark_px: float
    recv_ts: float
    exchange_ts: Optional[float] = None


@dataclass
class BookEvent:
    outcome_id: str
    yes_coin: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    recv_ts: float
    exchange_ts: Optional[float] = None
    source: str = "ws"          # "ws" | "rest"
    tape_ts: float = 0.0        # 按既有 SQLite 约定写入 book_snapshots.ts


def _normalize_ts_seconds(raw_val: Any) -> Optional[float]:
    if raw_val is None:
        return None
    try:
        val = float(raw_val)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    return val / 1000.0 if val > 1e12 else val


def _extract_mark_exchange_ts(data: dict, ctx: dict) -> Optional[float]:
    for container in (data, ctx):
        for key in ("time", "timestamp", "ts"):
            if key in container and container[key] is not None:
                parsed = _normalize_ts_seconds(container[key])
                if parsed is not None:
                    return parsed
    return None


def _ensure_column(conn: sqlite3.Connection, table: str, col_name: str, col_type: str):
    existing = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if col_name not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


# ============================================================
# 2. 唯一 SQLite 建表与磁带写入层（完全兼容既有回放 schema）
# ============================================================

def init_db(db_path: str) -> sqlite3.Connection:
    """
    初始化或打开 HIP-4 行情磁带 SQLite（WAL 模式）。
    保留原有 markets / book_snapshots / mark_ticks 列与索引，
    并对可空扩展列 (exchange_ts / recv_ts / source) 做向后兼容补列。
    """
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
            asks_json TEXT NOT NULL,
            exchange_ts REAL,
            recv_ts REAL,
            source TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_book_oid_ts ON book_snapshots(outcome_id, ts);

        CREATE TABLE IF NOT EXISTS mark_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            underlying TEXT NOT NULL,
            mark_px REAL NOT NULL,
            exchange_ts REAL,
            recv_ts REAL
        );
        CREATE INDEX IF NOT EXISTS idx_mark_u_ts ON mark_ticks(underlying, ts);
        """
    )

    _ensure_column(conn, "book_snapshots", "exchange_ts", "REAL")
    _ensure_column(conn, "book_snapshots", "recv_ts", "REAL")
    _ensure_column(conn, "book_snapshots", "source", "TEXT")
    _ensure_column(conn, "mark_ticks", "exchange_ts", "REAL")
    _ensure_column(conn, "mark_ticks", "recv_ts", "REAL")
    conn.commit()
    return conn


class MarketTapeWriter:
    """将市场元数据、Yes 侧盘口前 10 档与 HyperCore markPx 写入 SQLite，并按 60s TWAP 回写结算。"""

    def __init__(
        self,
        db_path: str,
        commit_batch: int = 20,
        max_book_levels: int = 10,
    ):
        self.db_path = db_path
        self.commit_batch = max(1, int(commit_batch))
        self.max_book_levels = max(1, int(max_book_levels))
        self.conn = init_db(db_path)
        self._pending_writes = 0

    def upsert_market(self, market: Any):
        self.conn.execute(
            """
            INSERT OR IGNORE INTO markets
            (outcome_id, name, underlying, strike, expiry_ts, period, fee_scale, yes_coin, raw_description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(market.outcome_id),
                str(market.name),
                str(market.underlying),
                float(market.strike),
                float(market.expiry_ts),
                str(market.period),
                float(market.fee_scale),
                str(market.yes_coin),
                str(market.raw_description),
            ),
        )
        self.conn.commit()

    def write_mark(self, event: MarkEvent):
        # 既有约定：mark_ticks.ts 填本机接收时间 recv_ts
        self.conn.execute(
            """
            INSERT INTO mark_ticks (ts, underlying, mark_px, exchange_ts, recv_ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                float(event.recv_ts),
                str(event.underlying).upper(),
                float(event.mark_px),
                event.exchange_ts,
                float(event.recv_ts),
            ),
        )
        self._pending_writes += 1
        if self._pending_writes >= self.commit_batch:
            self.flush()

    def write_book(self, event: BookEvent, immediate_commit: bool = False):
        # 既有约定：仅写前 10 档；WS 带 data["time"] 用 exchange_ts，REST 快照用 recv_ts
        ts_val = event.tape_ts if event.tape_ts > 0 else (
            event.exchange_ts
            if (event.source == "ws" and event.exchange_ts is not None)
            else event.recv_ts
        )
        bids_top = event.bids[: self.max_book_levels]
        asks_top = event.asks[: self.max_book_levels]
        self.conn.execute(
            """
            INSERT INTO book_snapshots (ts, outcome_id, bids_json, asks_json, exchange_ts, recv_ts, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(ts_val),
                str(event.outcome_id),
                json.dumps(bids_top),
                json.dumps(asks_top),
                event.exchange_ts,
                float(event.recv_ts),
                str(event.source),
            ),
        )
        self._pending_writes += 1
        if immediate_commit or self._pending_writes >= self.commit_batch:
            self.flush()

    def settle_expired_markets(
        self,
        compute_twap_fn: Callable[[List[Tuple[float, float]], float, float], Optional[float]],
        now_ts: Optional[float] = None,
        grace_sec: float = 2.0,
    ) -> List[Tuple[str, str, float, float, int]]:
        """
        按既有 [expiry-60s, expiry) 自算 TWAP 规则回写已到期场次：
        严格高于 strike 记 1 (UP)，等于或低于记 0 (DOWN)。
        """
        self.flush()
        now = time.time() if now_ts is None else now_ts
        rows = self.conn.execute(
            "SELECT outcome_id, underlying, strike, expiry_ts FROM markets "
            "WHERE outcome_up IS NULL AND expiry_ts <= ?",
            (now - grace_sec,),
        ).fetchall()

        settled: List[Tuple[str, str, float, float, int]] = []
        for oid, u_sym, strike, expiry_ts in rows:
            t_rows = self.conn.execute(
                "SELECT ts, mark_px FROM mark_ticks "
                "WHERE underlying = ? AND ts >= ? AND ts <= ? ORDER BY ts ASC",
                (u_sym, expiry_ts - 120.0, expiry_ts + 2.0),
            ).fetchall()
            twap = compute_twap_fn(t_rows, expiry_ts - 60.0, expiry_ts)
            if twap is None:
                continue
            outcome_up = 1 if twap > strike else 0
            self.conn.execute(
                "UPDATE markets SET settled_twap = ?, outcome_up = ? WHERE outcome_id = ?",
                (float(twap), int(outcome_up), str(oid)),
            )
            settled.append((str(oid), str(u_sym), float(strike), float(twap), int(outcome_up)))

        if settled:
            self.conn.commit()
        return settled

    def flush(self):
        if self._pending_writes > 0:
            self.conn.commit()
            self._pending_writes = 0

    def close(self):
        self.flush()
        self.conn.close()


# ============================================================
# 3. 共享实时采集循环（record 与 paper 唯一 WebSocket 入口）
# ============================================================

async def _maybe_call(fn: Optional[Callable[..., Any]], *args: Any):
    if fn is None:
        return
    res = fn(*args)
    if inspect.isawaitable(res):
        await res


def _default_engine_helpers():
    import outcome_hip4_engine as eng

    return (
        eng.ENDPOINTS,
        eng.fetch_outcome_markets,
        eng.fetch_l2_book_rest,
        eng.parse_book_levels,
        eng.compute_strict_twap,
    )


class LiveMarketFeed:
    """
    HyperCore HIP-4 共享实时行情采集层：
    - 订阅 activeAssetCtx (标的永续 markPx)、outcomeMetaUpdates、活跃市场 Yes 侧 l2Book (#<id>0)
    - 每约 25s 发送 WS ping
    - 定期同步活跃市场，对新订阅与断线重连后的市场立即 REST 补拉 l2Book 快照
    - 同时写入可选的 MarketTapeWriter (SQLite) 并向调用方派发行情与心跳回调
    """

    def __init__(
        self,
        network: str,
        underlying: str,
        period: str,
        *,
        ws_url: Optional[str] = None,
        poll_meta_sec: float = 20.0,
        ping_interval_sec: float = 25.0,
        recv_timeout_sec: float = 5.0,
        reconnect_delay_sec: float = 3.0,
        tape_writer: Optional[MarketTapeWriter] = None,
        auto_settle_tape: bool = True,
        fetch_markets_fn: Optional[Callable[..., Sequence[Any]]] = None,
        fetch_l2_book_fn: Optional[Callable[[str, str], Optional[dict]]] = None,
        parse_book_fn: Optional[Callable[[dict], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]] = None,
        compute_twap_fn: Optional[Callable[[List[Tuple[float, float]], float, float], Optional[float]]] = None,
        ws_connect: Optional[Callable[..., Any]] = None,
        clock_fn: Callable[[], float] = time.time,
    ):
        defaults = None
        if any(x is None for x in (ws_url, fetch_markets_fn, fetch_l2_book_fn, parse_book_fn, compute_twap_fn)):
            defaults = _default_engine_helpers()

        self.network = network
        self.underlying = underlying.upper()
        self.period = period
        self.ws_url = ws_url or defaults[0][network]["ws"]
        self.poll_meta_sec = poll_meta_sec
        self.ping_interval_sec = ping_interval_sec
        self.recv_timeout_sec = recv_timeout_sec
        self.reconnect_delay_sec = reconnect_delay_sec
        self.tape_writer = tape_writer
        self.auto_settle_tape = auto_settle_tape

        self.fetch_markets_fn = fetch_markets_fn or defaults[1]
        self.fetch_l2_book_fn = fetch_l2_book_fn or defaults[2]
        self.parse_book_fn = parse_book_fn or defaults[3]
        self.compute_twap_fn = compute_twap_fn or defaults[4]
        self.ws_connect = ws_connect or websockets.connect
        self.clock_fn = clock_fn

        self.tracked_markets: Dict[str, Any] = {}

    async def run(
        self,
        *,
        stop_ts: float,
        should_stop: Optional[Callable[[], bool]] = None,
        on_markets: Optional[Callable[[List[Any], List[Any]], Any]] = None,
        on_mark: Optional[Callable[[MarkEvent], Any]] = None,
        on_book: Optional[Callable[[BookEvent], Any]] = None,
        on_tick: Optional[Callable[[float], Any]] = None,
        on_settled: Optional[Callable[[List[Tuple[str, str, float, float, int]]], Any]] = None,
        on_disconnect: Optional[Callable[[Exception], Any]] = None,
    ):
        stop_check = should_stop or (lambda: False)

        def is_stopping() -> bool:
            return bool(stop_check()) or (self.clock_fn() >= stop_ts)

        async def _maybe_settle():
            if self.tape_writer is not None and self.auto_settle_tape:
                settled = self.tape_writer.settle_expired_markets(
                    self.compute_twap_fn, now_ts=self.clock_fn()
                )
                if settled:
                    await _maybe_call(on_settled, settled)

        try:
            while not is_stopping():
                try:
                    async with self.ws_connect(
                        self.ws_url, ping_interval=None, max_size=4 * 1024 * 1024
                    ) as ws:
                        subscribed_coins: set[str] = set()

                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "activeAssetCtx", "coin": self.underlying},
                        }))
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "outcomeMetaUpdates"},
                        }))

                        async def sync_markets_and_subscribe():
                            markets = list(
                                await asyncio.to_thread(
                                    self.fetch_markets_fn,
                                    self.network,
                                    self.period,
                                    self.underlying,
                                    True,
                                )
                            )
                            newly_subscribed: List[Any] = []
                            for m in markets:
                                self.tracked_markets[str(m.outcome_id)] = m
                                if self.tape_writer is not None:
                                    self.tape_writer.upsert_market(m)

                                yes_coin = str(m.yes_coin)
                                if yes_coin not in subscribed_coins:
                                    await ws.send(json.dumps({
                                        "method": "subscribe",
                                        "subscription": {"type": "l2Book", "coin": yes_coin},
                                    }))
                                    subscribed_coins.add(yes_coin)
                                    newly_subscribed.append(m)

                            # 先通知调用方更新市场集合，再补拉 REST 快照，确保调用方先认识市场再收盘口
                            await _maybe_call(on_markets, markets, newly_subscribed)

                            for m in newly_subscribed:
                                yes_coin = str(m.yes_coin)
                                snap = await asyncio.to_thread(
                                    self.fetch_l2_book_fn, self.network, yes_coin
                                )
                                if snap and "levels" in snap:
                                    bids, asks = self.parse_book_fn(snap)
                                    recv_ts = self.clock_fn()
                                    exch_ts = _normalize_ts_seconds(snap.get("time"))
                                    ev = BookEvent(
                                        outcome_id=str(m.outcome_id),
                                        yes_coin=yes_coin,
                                        bids=bids,
                                        asks=asks,
                                        recv_ts=recv_ts,
                                        exchange_ts=exch_ts,
                                        source="rest",
                                        tape_ts=recv_ts,  # REST 快照 ts 保持本机时间约定
                                    )
                                    if self.tape_writer is not None:
                                        self.tape_writer.write_book(ev, immediate_commit=True)
                                    await _maybe_call(on_book, ev)

                        await sync_markets_and_subscribe()
                        last_meta_sync = self.clock_fn()
                        last_ping = self.clock_fn()

                        while not is_stopping():
                            now = self.clock_fn()
                            if now - last_ping >= self.ping_interval_sec:
                                await ws.send(json.dumps({"method": "ping"}))
                                last_ping = now

                            if now - last_meta_sync >= self.poll_meta_sec:
                                await sync_markets_and_subscribe()
                                await _maybe_settle()
                                await _maybe_call(on_tick, self.clock_fn())
                                last_meta_sync = self.clock_fn()

                            try:
                                raw_msg = await asyncio.wait_for(
                                    ws.recv(), timeout=self.recv_timeout_sec
                                )
                            except asyncio.TimeoutError:
                                await _maybe_call(on_tick, self.clock_fn())
                                continue

                            if not raw_msg:
                                continue
                            msg = json.loads(raw_msg)
                            channel = msg.get("channel")
                            data = msg.get("data")

                            if channel == "activeAssetCtx" and isinstance(data, dict):
                                ctx = data.get("ctx", {}) or {}
                                mpx_str = ctx.get("markPx")
                                coin_name = str(data.get("coin", self.underlying)).upper()
                                if mpx_str is not None and coin_name == self.underlying:
                                    recv_ts = self.clock_fn()
                                    ev = MarkEvent(
                                        underlying=self.underlying,
                                        mark_px=float(mpx_str),
                                        recv_ts=recv_ts,
                                        exchange_ts=_extract_mark_exchange_ts(data, ctx),
                                    )
                                    if self.tape_writer is not None:
                                        self.tape_writer.write_mark(ev)
                                    await _maybe_call(on_mark, ev)

                            elif channel == "l2Book" and isinstance(data, dict):
                                coin_str = str(data.get("coin", ""))
                                if coin_str.startswith("#") and coin_str.endswith("0"):
                                    oid = coin_str[1:-1]
                                    bids, asks = self.parse_book_fn(data)
                                    recv_ts = self.clock_fn()
                                    exch_ts = _normalize_ts_seconds(data.get("time"))
                                    tape_ts = exch_ts if exch_ts is not None else recv_ts
                                    ev = BookEvent(
                                        outcome_id=oid,
                                        yes_coin=coin_str,
                                        bids=bids,
                                        asks=asks,
                                        recv_ts=recv_ts,
                                        exchange_ts=exch_ts,
                                        source="ws",
                                        tape_ts=tape_ts,
                                    )
                                    if self.tape_writer is not None:
                                        self.tape_writer.write_book(ev)
                                    await _maybe_call(on_book, ev)

                            elif channel == "outcomeMetaUpdates":
                                await sync_markets_and_subscribe()

                        if self.tape_writer is not None:
                            self.tape_writer.flush()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    if self.tape_writer is not None:
                        self.tape_writer.flush()
                    if is_stopping():
                        break
                    await _maybe_call(on_disconnect, exc)
                    if self.reconnect_delay_sec > 0:
                        await asyncio.sleep(self.reconnect_delay_sec)
        finally:
            if self.tape_writer is not None:
                self.tape_writer.flush()
                await _maybe_settle()
