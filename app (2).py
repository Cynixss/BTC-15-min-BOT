"""
app.py: Live Bitcoin dashboard for Kalshi's 15-minute BTC Up/Down market, with a TradingView live
chart (1m / 3m / 5m / 15m / 1h), a 7-test LOCK call, and win/loss records.

Read-only and keyless: it uses only public endpoints and never places orders.
  * Candles + live tick : Coinbase Exchange (primary), Binance.US / Binance (fallbacks)
  * Contracts           : Kalshi public Trade API v2  GET /markets?series_ticker=KXBTC15M

Kalshi resolves each window on CF Benchmarks' BRTI: YES if the 60-second average
before the close is >= the 60-second average before the window opened. The
"target" line is that opening reference (Kalshi's strike), editable by hand.

LOCK: a call that passes seven tests (DATA, TIME, STRENGTH, LOCK, REVERSAL, EDGE, CHOP). The first
lock in each market is the "original lock call"; it is logged with a +25% buy/sell plan and graded
against Kalshi's settled result.

WIN / LOSS: a call is a WIN when it matches Kalshi's own settled result for that market
(result "yes" = UP won, "no" = DOWN won). Ties settle YES (UP), per the rule above.

Deploy on Streamlit Community Cloud with this file plus requirements.txt.
Run locally:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:
    import streamlit.components.v1 as components
except Exception:  # if unavailable, the TradingView chart is skipped and everything else still works
    components = None

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_SECONDS = 900  # the dashboard works on 15-minute candles (Kalshi's market length)
OHLCV = ["open", "high", "low", "close", "volume"]
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = "KXBTC15M"
HISTORY_HOURS = 76  # ~300 fifteen-minute candles, the most the exchange APIs return

# TradingView chart: label -> TradingView interval code ("60" = 1 hour)
TV_INTERVALS = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "1h": "60"}
TV_SYMBOL = "COINBASE:BTCUSD"
LIVE_CHART_HEIGHT = 560

# Lock call
LOCK_BOOK_PATH = os.path.join(tempfile.gettempdir(), "btc_kalshi_lock_book.json")
LOCK_COLOR = "#7b1fa2"
DRIFT_SECS = 30  # REVERSAL test looks at how the price cushion changed over this long
EROSION_SIGMA = 0.35  # ...and fails if it shrank by more than this many 1-sigma moves
CHOP_MAX_CROSSES = 1  # CHOP test: max EMA crossovers in the last 10 closed candles
CHOP_RANGE_MULT = 2.5  # CHOP test: last candle range vs the median of the previous 20
LOCK_TESTS = ["DATA", "TIME", "STRENGTH", "LOCK", "REVERSAL", "EDGE", "CHOP"]
HEADERS = {"User-Agent": "btc-15m-dashboard/1.0 (streamlit)"}
HTTP_TIMEOUT = 10

WICK_LONG = 0.5  # a wick >= 50% of the candle's high-low range counts as "long"
FLAT_SPREAD_PCT = 0.02  # |EMA9-EMA21| below 0.02% of price = EMAs essentially flat
GREEN, RED, AMBER, BLUE = "#26a69a", "#ef5350", "#f5a623", "#4a90e2"

TIMEZONES = ["UTC", "America/Chicago", "America/New_York", "America/Los_Angeles",
             "Europe/London", "Asia/Tokyo"]


@dataclass(frozen=True)
class Spec:
    """One Kalshi market length (15 minutes) and the series that trades it."""
    minutes: int
    series: str

    @property
    def seconds(self) -> int:
        return self.minutes * 60

    @property
    def delta(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.minutes)

    @property
    def freq(self) -> str:
        return f"{self.minutes}min"

    @property
    def sigma_scale(self) -> float:
        """Volatility floors below were tuned for 15m; random-walk scaling for longer windows."""
        return math.sqrt(self.minutes / 15)


# --------------------------------------------------------------------------- #
# Data: candles (public exchange APIs)
# --------------------------------------------------------------------------- #
def _get(url: str, params: dict | None = None, timeout: float = HTTP_TIMEOUT):
    resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["time", *OHLCV]].astype({c: float for c in OHLCV})
    return df.drop_duplicates("time").sort_values("time").set_index("time")


def _from_coinbase() -> pd.DataFrame:
    # rows: [time, low, high, open, close, volume], newest first, max 300
    rows = _get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                {"granularity": BASE_SECONDS})
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return _normalise(df)


def _from_binance(base: str, symbol: str) -> pd.DataFrame:
    rows = _get(f"{base}/api/v3/klines", {"symbol": symbol, "interval": "15m", "limit": 300})
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return _normalise(df)


# Order matters. api.binance.com blocks US IPs (HTTP 451), and Streamlit Cloud
# runs in the US, so Coinbase goes first.
CANDLE_SOURCES = [
    ("Coinbase", _from_coinbase),
    ("Binance.US", lambda: _from_binance("https://api.binance.us", "BTCUSD")),
    ("Binance", lambda: _from_binance("https://api.binance.com", "BTCUSDT")),
]


@st.cache_data(ttl=10, show_spinner=False)
def fetch_candles() -> tuple[pd.DataFrame, str]:
    """Latest ~300 15m candles (includes the still-forming one). Cached and shared
    between all visitors and both tabs so a busy public app doesn't hammer the exchange."""
    errors = []
    for name, fn in CANDLE_SOURCES:
        try:
            df = fn()
            if len(df) >= 40:
                return df, name
            errors.append(f"{name}: only {len(df)} candles")
        except Exception as exc:  # network / HTTP / parse: try the next source
            errors.append(f"{name}: {exc}")
    raise RuntimeError(" | ".join(errors))


@st.cache_data(ttl=1, show_spinner=False)
def fetch_ticker() -> tuple[float, str]:
    """Latest BTC trade price. Cached 1s and shared across visitors."""
    errors = []
    sources = [
        ("Coinbase", "https://api.exchange.coinbase.com/products/BTC-USD/ticker", None),
        ("Binance.US", "https://api.binance.us/api/v3/ticker/price", {"symbol": "BTCUSD"}),
        ("Binance", "https://api.binance.com/api/v3/ticker/price", {"symbol": "BTCUSDT"}),
    ]
    for name, url, params in sources:
        try:
            return float(_get(url, params, timeout=3)["price"]), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(" | ".join(errors))


# --------------------------------------------------------------------------- #
# Data: Kalshi (public endpoint, no auth)
# --------------------------------------------------------------------------- #
def _cents(m: dict, key: str):
    """Price in cents; tolerates `key` (cents) or `key_dollars` (string dollars)."""
    if m.get(key) is not None:
        return int(round(float(m[key])))
    if m.get(f"{key}_dollars") is not None:
        return int(round(float(m[f"{key}_dollars"]) * 100))
    return None


@st.cache_data(ttl=2, show_spinner=False)
def fetch_kalshi_markets(series: str) -> pd.DataFrame:
    """Open markets of one series, soonest-closing first."""
    raw, cursor = [], None
    for _ in range(5):  # pagination safety cap
        params = {"series_ticker": series, "status": "open", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        data = _get(f"{KALSHI_BASE}/markets", params)
        raw.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break

    rows = []
    for m in raw:
        strike = m.get("floor_strike", m.get("cap_strike"))
        rows.append({
            "ticker": m.get("ticker"),
            "title": m.get("title", ""),
            "strike": float(strike) if strike is not None else np.nan,
            "close_time": pd.to_datetime(m.get("close_time"), utc=True, errors="coerce"),
            "yes_bid": _cents(m, "yes_bid"), "yes_ask": _cents(m, "yes_ask"),
            "no_bid": _cents(m, "no_bid"), "no_ask": _cents(m, "no_ask"),
            "volume": m.get("volume"),
        })
    cols = ["ticker", "title", "strike", "close_time", "yes_bid", "yes_ask",
            "no_bid", "no_ask", "volume"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("close_time").reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_kalshi_settled(series: str, hours: int = HISTORY_HOURS) -> pd.DataFrame:
    """Recently settled markets of one series with Kalshi's own result.
    `result` is "yes" (UP won) or "no" (DOWN won); unsettled/voided markets are skipped."""
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(hours=hours)

    def pull(params: dict) -> list[dict]:
        out, cursor = [], None
        for _ in range(5):
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            data = _get(f"{KALSHI_BASE}/markets", p)
            out.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return out

    try:
        raw = pull({"series_ticker": series, "min_close_ts": int(cutoff.timestamp()), "limit": 1000})
    except requests.HTTPError:  # some API versions reject the timestamp filter; fall back
        raw = pull({"series_ticker": series, "status": "settled", "limit": 1000})

    rows = []
    for m in raw:
        res = str(m.get("result") or "").lower()
        if res not in ("yes", "no"):
            continue
        rows.append({"ticker": m.get("ticker"),
                     "close_time": pd.to_datetime(m.get("close_time"), utc=True, errors="coerce"),
                     "result": res})
    df = pd.DataFrame(rows, columns=["ticker", "close_time", "result"]).dropna(subset=["close_time"])
    df = df[df["close_time"] >= cutoff]
    return df.sort_values("close_time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Indicators (pure pandas, so deployment needs no TA library)
# --------------------------------------------------------------------------- #
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)  # loss == 0 -> inf -> RSI 100


def compute_indicators(raw: pd.DataFrame, lookback: int, delta: pd.Timedelta) -> pd.DataFrame:
    df = raw.copy()
    close, vol = df["close"], df["volume"]

    # Trend: 9/21 EMA, spread, crossovers (+1 bullish, -1 bearish)
    df["ema_9"], df["ema_21"] = _ema(close, 9), _ema(close, 21)
    df["ema_spread"] = df["ema_9"] - df["ema_21"]
    df["ema_spread_pct"] = df["ema_spread"] / df["ema_21"] * 100
    sign = np.sign(df["ema_spread"])
    prev = sign.shift(1)
    df["ema_cross"] = np.select([(sign > 0) & (prev < 0), (sign < 0) & (prev > 0)], [1, -1], 0)

    # Momentum
    df["rsi_14"] = _rsi(close, 14)

    # Price-Volume Trend: cumulative volume * percent price change
    df["pvt"] = (vol * close.pct_change()).fillna(0).cumsum()
    df["pvt_signal"] = _ema(df["pvt"], 9)
    df["pvt_hist"] = df["pvt"] - df["pvt_signal"]

    # Buyer/seller mismatch: price and PVT moving in opposite directions over `lookback` bars
    df["price_chg"] = close.diff(lookback)
    df["pvt_chg"] = df["pvt"].diff(lookback)
    df["pvt_mismatch"] = np.select(
        [(df["price_chg"] < 0) & (df["pvt_chg"] > 0), (df["price_chg"] > 0) & (df["pvt_chg"] < 0)],
        [1, -1], 0)  # +1 bullish (buyers absorbing), -1 bearish (sellers distributing)

    # Candle geometry (USD distances + share of the high-low range)
    top, bottom = df[["open", "close"]].max(axis=1), df[["open", "close"]].min(axis=1)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = top - bottom
    df["upper_wick"] = df["high"] - top
    df["lower_wick"] = bottom - df["low"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick_ratio"] = (df["upper_wick"] / rng).fillna(0.0)
    df["lower_wick_ratio"] = (df["lower_wick"] / rng).fillna(0.0)
    df["body_ratio"] = (df["body"] / rng).fillna(0.0)

    df["rel_volume"] = vol / vol.rolling(20, min_periods=5).mean()
    df["is_closed"] = (df.index + delta) <= pd.Timestamp.now(tz="UTC")
    return df


def signal_series(df: pd.DataFrame, rsi_ob: float, rsi_os: float) -> pd.Series:
    """The headline trend rule evaluated on every candle (same rule as analyze())."""
    up = (df["ema_9"] > df["ema_21"]) & (df["rsi_14"] < rsi_ob)
    dn = (df["ema_9"] < df["ema_21"]) & (df["rsi_14"] > rsi_os)
    return pd.Series(np.select([up, dn], ["UP", "DOWN"], "NONE"), index=df.index)


# --------------------------------------------------------------------------- #
# Signal analysis (reads the last CLOSED candle so it never repaints)
# --------------------------------------------------------------------------- #
@dataclass
class Analysis:
    signal: str  # "UP" | "DOWN" | "NONE"
    headline: str
    candle_time: pd.Timestamp
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    mismatch: int = 0


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _c(v) -> str:
    """Cents for display; handles missing values and float-ified int columns."""
    return f"{int(v)}" if pd.notna(v) else "—"


def analyze(df: pd.DataFrame, rsi_ob: float, rsi_os: float, lookback: int) -> Analysis | None:
    closed = df[df["is_closed"]].dropna(subset=["ema_21", "rsi_14", "pvt_signal"])
    if len(closed) < 2:
        return None
    r, p = closed.iloc[-1], closed.iloc[-2]
    fast, slow, rsi = r["ema_9"], r["ema_21"], r["rsi_14"]

    # ---- headline rule (same logic as the trading bot) ----
    if fast > slow and rsi < rsi_ob:
        sig, head = "UP", f"UP trend: EMA9 above EMA21 and RSI {rsi:.1f} is below {rsi_ob:g}"
    elif fast < slow and rsi > rsi_os:
        sig, head = "DOWN", f"DOWN trend: EMA9 below EMA21 and RSI {rsi:.1f} is above {rsi_os:g}"
    elif fast > slow:
        sig, head = "NONE", f"No signal: EMAs bullish but RSI {rsi:.1f} is overbought (>= {rsi_ob:g})"
    elif fast < slow:
        sig, head = "NONE", f"No signal: EMAs bearish but RSI {rsi:.1f} is oversold (<= {rsi_os:g})"
    else:
        sig, head = "NONE", "No signal: EMA9 equals EMA21"

    f: list[str] = []

    # ---- trend ----
    side = "above" if fast > slow else "below"
    widening = abs(r["ema_spread"]) > abs(p["ema_spread"])
    f.append(f"**Trend:** EMA9 {_usd(fast)} is {side} EMA21 {_usd(slow)} "
             f"(gap {_usd(abs(r['ema_spread']))}, {abs(r['ema_spread_pct']):.3f}%), "
             f"{'widening' if widening else 'narrowing'} vs the prior candle.")
    crosses = closed.index[(closed["ema_cross"] != 0).to_numpy()]
    if len(crosses):
        last_cross = crosses[-1]
        ago = len(closed) - 1 - closed.index.get_loc(last_cross)
        kind = "Bullish" if closed.loc[last_cross, "ema_cross"] > 0 else "Bearish"
        f.append(f"**Last crossover:** {kind} cross {ago} candle{'s' if ago != 1 else ''} ago "
                 f"({last_cross:%H:%M} UTC).")
    else:
        f.append("**Last crossover:** none in the loaded window.")

    # ---- RSI ----
    zone = "overbought" if rsi >= rsi_ob else "oversold" if rsi <= rsi_os else "neutral zone"
    f.append(f"**RSI(14):** {rsi:.1f} ({zone}), {'up' if rsi > p['rsi_14'] else 'down'} "
             f"from {p['rsi_14']:.1f} on the prior candle.")

    # ---- PVT / buyer-seller mismatch ----
    pvt_side = "above" if r["pvt_hist"] > 0 else "below"
    f.append(f"**PVT:** {pvt_side} its 9-bar signal line, so volume-weighted momentum is "
             f"{'rising' if r['pvt_hist'] > 0 else 'falling'}.")
    mm, pc = int(r["pvt_mismatch"]), r["price_chg"]
    if mm == 1:
        f.append(f"**Buyer/seller mismatch (BULLISH):** price fell {_usd(abs(pc))} over the last "
                 f"{lookback} candles but PVT rose. Volume has been heavier on the buy side.")
    elif mm == -1:
        f.append(f"**Buyer/seller mismatch (BEARISH):** price rose {_usd(abs(pc))} over the last "
                 f"{lookback} candles but PVT fell. Volume has been heavier on the sell side.")
    elif pc > 0:
        f.append(f"**Buyer/seller check:** price up {_usd(pc)} and PVT rising over {lookback} "
                 "candles, so volume confirms buyers. No mismatch.")
    elif pc < 0:
        f.append(f"**Buyer/seller check:** price down {_usd(abs(pc))} and PVT falling over "
                 f"{lookback} candles, so volume confirms sellers. No mismatch.")
    else:
        f.append("**Buyer/seller check:** no net price change over the lookback. No mismatch.")

    # ---- last candle pressure (wicks) ----
    color = "green" if r["close"] >= r["open"] else "red"
    uw, lw = r["upper_wick_ratio"], r["lower_wick_ratio"]
    if uw >= WICK_LONG and uw >= lw:
        wick_txt = "long upper wick: sellers pushed price back down from the high."
    elif lw >= WICK_LONG:
        wick_txt = "long lower wick: buyers pushed price back up from the low."
    else:
        wick_txt = "no dominant wick."
    f.append(f"**Last closed candle ({closed.index[-1]:%H:%M} UTC):** {color}, body {_usd(r['body'])}, "
             f"upper wick {_usd(r['upper_wick'])} ({uw:.0%} of range), "
             f"lower wick {_usd(r['lower_wick'])} ({lw:.0%}); {wick_txt}")
    if pd.notna(r["rel_volume"]):
        f.append(f"**Volume:** {r['rel_volume']:.2f}x the 20-candle average.")

    # ---- cautions: where the evidence disagrees with the headline ----
    c: list[str] = []
    if sig == "UP":
        if mm == -1:
            c.append("Bearish PVT mismatch contradicts the UP signal.")
        if r["pvt_hist"] < 0:
            c.append("PVT is below its signal line while EMAs say UP.")
        if uw >= WICK_LONG and uw >= lw:
            c.append("Long upper wick on the last candle shows selling pressure at highs.")
    elif sig == "DOWN":
        if mm == 1:
            c.append("Bullish PVT mismatch contradicts the DOWN signal.")
        if r["pvt_hist"] > 0:
            c.append("PVT is above its signal line while EMAs say DOWN.")
        if lw >= WICK_LONG and lw > uw:
            c.append("Long lower wick on the last candle shows buying pressure at lows.")
    if abs(r["ema_spread_pct"]) < FLAT_SPREAD_PCT:
        c.append(f"EMAs are nearly flat ({abs(r['ema_spread_pct']):.3f}% apart). "
                 "Crossover signals are unreliable in choppy conditions.")

    return Analysis(sig, head, closed.index[-1], f, c, mm)


# --------------------------------------------------------------------------- #
# Live snapshot (candles + live tick + indicators + Kalshi markets for one interval)
# --------------------------------------------------------------------------- #
def overlay_live_price(raw: pd.DataFrame, price: float | None, now: pd.Timestamp,
                       spec: Spec) -> pd.DataFrame:
    """Fold the live tick into the forming candle (or start one if the feed lags)."""
    if price is None:
        return raw
    raw = raw.copy()
    window_start = now.floor(spec.freq)
    if raw.index[-1] >= window_start:  # forming candle already published
        i = raw.index[-1]
        raw.loc[i, "close"] = price
        raw.loc[i, "high"] = max(raw.at[i, "high"], price)
        raw.loc[i, "low"] = min(raw.at[i, "low"], price)
        return raw
    new = pd.DataFrame([[price, price, price, price, 0.0]], index=[window_start], columns=raw.columns)
    out = pd.concat([raw, new])
    out.index.name = raw.index.name
    return out


def realized_sigma(df: pd.DataFrame, spec: Spec) -> float:
    """Std-dev of this interval's log returns over the last ~65 closed candles (as a fraction)."""
    closes = df.loc[df["is_closed"], "close"].tail(65)
    rets = np.log(closes).diff().dropna()
    sigma = float(rets.std()) if len(rets) >= 10 else 0.002 * spec.sigma_scale
    return max(sigma, 0.0007 * spec.sigma_scale)


def get_snapshot(spec: Spec, lookback: int, offset: float, rsi_ob: float, rsi_os: float) -> dict:
    """Candles + live tick + indicators + Kalshi markets for one interval. The network calls
    inside are cached and shared between visitors and tabs; the pandas work is cheap."""
    now = pd.Timestamp.now(tz="UTC")
    raw, candle_src = fetch_candles()  # raises if every exchange fails; caller handles it
    tick, tick_src = None, None
    try:
        tick, tick_src = fetch_ticker()
    except Exception:
        pass
    raw = overlay_live_price(raw, tick, now, spec)
    raw = raw.copy()
    raw[["open", "high", "low", "close"]] += offset  # put the feed on Kalshi's (BRTI) price scale
    df = compute_indicators(raw, lookback, spec.delta)

    markets, kalshi_err = None, None
    try:
        markets = fetch_kalshi_markets(spec.series)
    except Exception as exc:
        kalshi_err = f"{type(exc).__name__}: {exc}"

    return dict(
        df=df, candle_src=candle_src, tick_src=tick_src, spot=float(df["close"].iloc[-1]),
        sigma=realized_sigma(df, spec), analysis=analyze(df, rsi_ob, rsi_os, lookback),
        markets=markets, kalshi_err=kalshi_err,
    )


# --------------------------------------------------------------------------- #
# Probability model + decision engine
# --------------------------------------------------------------------------- #
def model_prob_up(spot: float, target: float, secs_left: float, sigma: float,
                  interval_seconds: int) -> tuple[float, float]:
    """P(final settlement >= target) under a random walk, plus the 1-sigma move in USD.
    `sigma` is the std-dev of one full interval's return.

    Kalshi settles on the 60-second average of BRTI before close, so less than the full
    remaining time is 'uncertain': variance-time = secs_left - 40 (>= 60s left), and
    secs_left^3 / 10800 inside the last minute (Brownian average of a 60s window).
    """
    eff = secs_left - 40 if secs_left >= 60 else secs_left ** 3 / 10800
    eff = max(eff, 1.0)
    sd = max(sigma * spot * math.sqrt(eff / interval_seconds), 0.5)
    z = (spot - target) / sd
    return 0.5 * (1 + math.erf(z / math.sqrt(2))), sd


@dataclass
class Rules:
    min_p: float = 0.60  # model chance needed to BUY a side
    min_edge: float = 3.0  # model chance minus ask, in cents
    max_price: int = 85  # never chase an ask above this (cents)
    min_secs: int = 45  # no new entries with less time left
    sell_below: float = 0.40  # SELL when your side's model chance falls below this
    sell_trend_below: float = 0.60  # ...or when the trend flipped AND chance is below this


@dataclass
class Decision:
    action: str  # BUY UP | BUY DOWN | WAIT | HOLD | SELL
    title: str
    subtitle: str
    reasons: list[str] = field(default_factory=list)
    p_up: float | None = None
    sd: float | None = None


def _side_label(side: str) -> str:
    return "UP (YES)" if side == "yes" else "DOWN (NO)"


def decide(a: Analysis | None, spot: float | None, target: float | None, secs_left: float | None,
           sigma: float, q: dict, pos: dict | None, r: Rules, interval_seconds: int) -> Decision:
    """Turn trend analysis + price-vs-target + Kalshi quotes + your position into ONE action."""
    if a is None or spot is None or target is None or secs_left is None:
        why = ("No open Kalshi market found" if secs_left is None else
               "No target price. Enter one under 'My position & target line'" if target is None
               else "Waiting for data")
        return Decision("WAIT", "WAIT", why, [why + "."])
    if secs_left <= 0:
        return Decision("WAIT", "WAIT", "Market is closing", ["The current market has closed; a new one lists shortly."])

    p_up, sd = model_prob_up(spot, target, secs_left, sigma, interval_seconds)
    p_dn = 1 - p_up
    dist = spot - target
    ctx = (f"BTC {_usd(spot)} is {_usd(abs(dist))} {'above' if dist >= 0 else 'below'} the target "
           f"{_usd(target)} with {int(secs_left)}s left. Model chance of finishing UP: {p_up:.0%} "
           f"(typical remaining move ±{_usd(sd)}).")
    sig, mm = a.signal, a.mismatch

    # ----------------------------- you hold a position -----------------------------
    if pos:
        is_up = pos["side"] == "yes"
        label, held_p = _side_label(pos["side"]), (p_up if is_up else p_dn)
        bid = q.get("yes_bid" if is_up else "no_bid")
        entry = pos.get("entry")
        cushion = dist if is_up else -dist  # > 0 means price is on your winning side
        opp = "DOWN" if is_up else "UP"
        reasons = [ctx]
        if entry is not None and bid is not None:
            reasons.append(f"You entered at {entry}¢; selling now would fetch {bid}¢ "
                           f"({bid - entry:+d}¢ per contract).")
        if held_p < r.sell_below:
            reasons.insert(0, f"Price is {_usd(abs(cushion))} on the wrong side of the target; the model "
                              f"gives {label} only {held_p:.0%}.")
            return Decision("SELL", "SELL NOW", f"Reversal against your {label} position ({held_p:.0%} to win)",
                            reasons, p_up, sd)
        if sig == opp and held_p < r.sell_trend_below:
            reasons.insert(0, f"Trend flipped to {opp} and your {label} is only {held_p:.0%} likely.")
            return Decision("SELL", "SELL NOW", f"Trend reversed against your {label} position",
                            reasons, p_up, sd)
        if cushion >= 0:
            reasons.insert(0, f"Price is {_usd(cushion)} on the winning side of the target ({held_p:.0%}).")
        else:
            reasons.insert(0, f"Price is {_usd(abs(cushion))} on the wrong side, but still inside the "
                              f"model's tolerance ({held_p:.0%} to win).")
        if sig == opp:
            reasons.append(f"Trend shows {opp}, but price is far enough on your side. The contract pays on "
                           "the final price versus the target, not on the trend.")
        if bid is not None and bid >= 95:
            reasons.append(f"Bid is already {bid}¢. Only {100 - bid}¢ upside remains, and selling locks it in.")
        return Decision("HOLD", "HOLD", f"{label} still looks good ({held_p:.0%} to win)", reasons, p_up, sd)

    # ----------------------------- you are flat -----------------------------
    reasons = [ctx]
    if secs_left < r.min_secs:
        reasons.append(f"Only {int(secs_left)}s left; new entries are blocked in the last {r.min_secs}s.")
        return Decision("WAIT", "WAIT", "Too late to enter this market", reasons, p_up, sd)
    if sig == "NONE":
        reasons.append(a.headline)
        return Decision("WAIT", "WAIT", "No clear trend signal", reasons, p_up, sd)

    up = sig == "UP"
    side = "yes" if up else "no"
    ps = p_up if up else p_dn
    ask = q.get("yes_ask" if up else "no_ask")
    fails = []
    if ps < r.min_p:
        fails.append(f"Price isn't far enough {'above' if up else 'below'} the target yet "
                     f"(model {ps:.0%}, need {r.min_p:.0%}).")
        other = p_dn if up else p_up
        if other >= r.min_p:
            fails.append(f"Price currently favors {'DOWN' if up else 'UP'} ({other:.0%}) while the trend says "
                         f"{sig}. They disagree.")
    if (up and mm == -1) or (not up and mm == 1):
        fails.append(f"A {'bearish' if up else 'bullish'} PVT buyer/seller mismatch contradicts the {sig} trend.")
    if ask is None:
        fails.append(f"No {side.upper()} ask is quoted.")
    else:
        if ask > r.max_price:
            fails.append(f"The {side.upper()} ask is {ask}¢, above your {r.max_price}¢ limit.")
        edge = ps * 100 - ask
        if edge < r.min_edge:
            fails.append(f"Edge is {edge:+.0f}¢ (model {ps:.0%} vs ask {ask}¢); need {r.min_edge:+.0f}¢.")
    if fails:
        reasons = [f"Trend says {sig} ({a.headline.split(':', 1)[-1].strip()})."] + fails + reasons
        return Decision("WAIT", "WAIT", f"Trend {sig}, but conditions aren't met", reasons, p_up, sd)

    edge = ps * 100 - ask
    reasons = [f"Trend {sig} and price agree: model {ps:.0%} vs ask {ask}¢ (edge {edge:+.0f}¢).",
               f"Buying costs {ask}¢ and pays 100¢ if right; break-even win rate is {ask}%.", *reasons]
    return Decision("BUY UP" if up else "BUY DOWN", f"BUY {_side_label(side)}",
                    f"Ask {ask}¢ · model {ps:.0%} · edge {edge:+.0f}¢", reasons, p_up, sd)


def debounce(slot: str, action: str, ticker: str | None, has_pos: bool) -> tuple[bool, int, int]:
    """BUY needs 3 consecutive 1-second reads and SELL needs 2, so a single noisy tick can't
    flash a trade call. WAIT/HOLD show immediately. `slot` keeps each interval's counter separate.
    Returns (confirmed, count, needed)."""
    d = st.session_state.setdefault("deb", {}).setdefault(slot, {})
    key = (ticker, has_pos)
    if d.get("key") != key:
        d.update(key=key, raw=None, n=0)
    if action == d.get("raw"):
        d["n"] += 1
    else:
        d["raw"], d["n"] = action, 1
    need = {"BUY UP": 3, "BUY DOWN": 3, "SELL": 2}.get(action, 1)
    return d["n"] >= need, d["n"], need


# --------------------------------------------------------------------------- #
# LOCK call: seven tests must ALL pass (DATA, TIME, STRENGTH, LOCK, REVERSAL, EDGE, CHOP)
#
# A lock is stricter than a BUY call: it only fires late in a market, when price is decisively
# on one side of the target, has stayed there, and nothing in the data hints at a reversal.
# The FIRST lock in each market is the "original lock call". It is logged once, with a +25%
# (adjustable) buy/sell plan, and graded against Kalshi's settled result.
# It is a rule-based filter, not a guarantee: the record shows how it really performs.
# --------------------------------------------------------------------------- #
@dataclass
class LockRules:
    min_p: float = 0.80  # STRENGTH: model chance the side needs
    hold_secs: int = 10  # LOCK: how long STRENGTH + trend must hold without a break
    window_secs: int = 300  # TIME: locks are only called in the last N seconds of the market
    profit: float = 0.25  # target profit for the buy/sell plan


@dataclass
class LockResult:
    side: str | None  # "yes" (UP) | "no" (DOWN) | None
    tests: list  # [(name, ok, detail)] in LOCK_TESTS order
    passed: bool
    entry: int | None  # ask of the side right now (what a lock would buy at)
    bid: int | None
    p_side: float | None
    sd: float | None


def record_tick(k: int, ts: float, price: float) -> None:
    """Keep ~2 minutes of live prices per interval for the REVERSAL drift check."""
    t = st.session_state.setdefault(f"ticks_{k}", [])
    t.append((ts, price))
    while t and t[0][0] < ts - 120:
        t.pop(0)


def hold_seconds(k: int, ticker: str | None, side: str | None, ok: bool, now: float, gap: float) -> float:
    """Seconds the condition has held without a break (a gap longer than `gap` resets it)."""
    ss, key = st.session_state, f"lockhold_{k}"
    h = ss.get(key)
    if (not ok) or h is None or h["ticker"] != ticker or h["side"] != side or now - h["last"] > gap:
        h = {"ticker": ticker, "side": side, "since": None, "last": now}
    if ok and h["since"] is None:
        h["since"] = now
    h["last"] = now
    ss[key] = h
    return (now - h["since"]) if (ok and h["since"] is not None) else 0.0


def evaluate_lock(spec: Spec, ctx: dict, cfg: dict, now_ts: float) -> LockResult:
    ss, k = st.session_state, spec.minutes
    snap, cur, q = ctx["snap"], ctx["cur"], ctx["quotes"]
    df, a = snap["df"], snap["analysis"]
    spot, target, secs_left = ctx["spot"], ctx["target"], ctx["secs_left"]
    L, R = cfg["lock"], cfg["rules"]
    closed = df[df["is_closed"]]

    live = target is not None and secs_left is not None and secs_left > 0
    side = p_side = sd = cushion = ask = bid = None
    up = True
    if live:
        p_up, sd = model_prob_up(spot, target, secs_left, snap["sigma"], spec.seconds)
        up = p_up >= 0.5
        side, p_side = ("yes", p_up) if up else ("no", 1 - p_up)
        cushion = (spot - target) if up else (target - spot)
        ask, bid = q.get("yes_ask" if up else "no_ask"), q.get("yes_bid" if up else "no_bid")
    label = _side_label(side) if side else "n/a"
    opp = "DOWN" if up else "UP"
    tests: list = []

    # 1) DATA: every feed is live and the target is trustworthy
    bad = []
    if snap["tick_src"] is None:
        bad.append("live price feed is down")
    if len(closed) == 0 or ctx["now"] - closed.index[-1] > 2 * spec.delta + pd.Timedelta(seconds=120):
        bad.append("candle feed is stale")
    if a is None:
        bad.append("not enough closed candles for the indicators")
    if snap["kalshi_err"]:
        bad.append("Kalshi data error")
    elif cur is None:
        bad.append("no open Kalshi market")
    elif side and (ask is None or bid is None):
        bad.append("no bid/ask quote for the side")
    if ctx["tsrc"] not in ("Kalshi strike", "Manual"):
        bad.append(f"target is '{ctx['tsrc']}' (need Kalshi's strike or a manual target)")
    tests.append(("DATA", not bad, "prices, candles, Kalshi quotes and target all present" if not bad
                  else "; ".join(bad)))

    # 2) TIME: inside the lock window, with enough time left to act
    win = L.window_secs * spec.minutes / 15
    if secs_left is None:
        ok_t, d = False, "no open market"
    elif secs_left < R.min_secs:
        ok_t, d = False, f"only {int(secs_left)}s left (need at least {R.min_secs}s)"
    elif secs_left > win:
        ok_t, d = False, f"{int(secs_left)}s left; locks are only called in the last {int(win)}s"
    else:
        ok_t, d = True, f"{int(secs_left)}s left (window {R.min_secs}s to {int(win)}s)"
    tests.append(("TIME", ok_t, d))

    # 3) STRENGTH: price decisively on one side of the target
    if live:
        ok_s = p_side >= L.min_p
        d = (f"{label} model {p_side:.0%} (need {L.min_p:.0%}); BTC is {_usd(abs(cushion))} on the "
             f"{'winning' if cushion >= 0 else 'losing'} side ({cushion / sd:+.2f}σ)")
    else:
        ok_s, d = False, "no target or market to measure against"
    tests.append(("STRENGTH", ok_s, d))

    # 4) LOCK: STRENGTH and trend have held without a break
    trend_ok = a is not None and a.signal != opp
    gap = max(5.0, 3.0 * float(cfg.get("fast_secs", 1)) + 2.0)
    held = hold_seconds(k, cur["ticker"] if cur is not None else None, side, ok_s and trend_ok, now_ts, gap)
    if a is not None and not trend_ok:
        ok_l, d = False, f"trend says {opp}, against {label}"
    elif not ok_s:
        ok_l, d = False, "waiting for STRENGTH to pass and hold"
    else:
        ok_l, d = held >= L.hold_secs, f"held {held:.0f}s of the {L.hold_secs}s needed (trend agrees or is neutral)"
    tests.append(("LOCK", ok_l, d))

    # 5) REVERSAL: no sign the price is turning back
    bad = []
    if a is not None and live:
        if (up and a.mismatch == -1) or (not up and a.mismatch == 1):
            bad.append("PVT buyer/seller mismatch against the side")
        r_ = closed.iloc[-1]
        uw, lw = r_["upper_wick_ratio"], r_["lower_wick_ratio"]
        if up and uw >= WICK_LONG and uw >= lw:
            bad.append("long upper wick on the last closed candle (sellers)")
        if (not up) and lw >= WICK_LONG and lw > uw:
            bad.append("long lower wick on the last closed candle (buyers)")
        ticks = ss.get(f"ticks_{k}", [])
        older = [p for ts, p in ticks if ts <= now_ts - DRIFT_SECS]
        if not older:
            span = (now_ts - ticks[0][0]) if ticks else 0
            bad.append(f"still building {DRIFT_SECS}s of price history ({span:.0f}s so far)")
        else:
            then = (older[-1] - target) if up else (target - older[-1])
            erosion = (then - cushion) / sd
            if erosion > EROSION_SIGMA:
                bad.append(f"cushion shrank {erosion:.2f}σ in {DRIFT_SECS}s")
    elif not live:
        bad.append("no target or market to check")
    else:
        bad.append("no indicator data")
    tests.append(("REVERSAL", not bad, "no PVT mismatch, no opposing wick, cushion not eroding" if not bad
                  else "; ".join(bad)))

    # 6) EDGE: the price on offer is worth paying
    if live and ask is not None:
        edge = p_side * 100 - ask
        ok_e = edge >= R.min_edge and ask <= R.max_price
        d = (f"ask {ask}¢ vs model {p_side:.0%} = edge {edge:+.0f}¢ "
             f"(need {R.min_edge:+.0f}¢ or more, ask at most {R.max_price}¢)")
    else:
        ok_e, d = False, "no ask quoted"
    tests.append(("EDGE", ok_e, d))

    # 7) CHOP: a clean trend, not a sideways whipsaw
    ind = closed.dropna(subset=["ema_21"])
    if len(ind) < 22:
        tests.append(("CHOP", False, "not enough closed candles to judge"))
    else:
        bad = []
        spread = abs(ind["ema_spread_pct"].iloc[-1])
        crosses = int((ind["ema_cross"].tail(10) != 0).sum())
        rngs = ind["range"].tail(21)
        med = float(rngs.iloc[:-1].median())
        if spread < FLAT_SPREAD_PCT:
            bad.append(f"EMAs nearly flat ({spread:.3f}% apart)")
        if crosses > CHOP_MAX_CROSSES:
            bad.append(f"{crosses} EMA crossovers in the last 10 candles")
        if med > 0 and rngs.iloc[-1] > CHOP_RANGE_MULT * med:
            bad.append(f"last candle range {rngs.iloc[-1] / med:.1f}x normal")
        tests.append(("CHOP", not bad, f"EMAs {spread:.3f}% apart, {crosses} crossovers in 10 candles, "
                      "range normal" if not bad else "; ".join(bad)))

    return LockResult(side, tests, all(t[1] for t in tests), ask, bid, p_side, sd)


def lock_plan(entry: int, profit: float) -> dict:
    """Buy/sell levels for a contract that pays 100c if right. `profit` is a fraction (0.25 = +25%)."""
    max_buy = math.floor(100 / (1 + profit) + 1e-9)  # highest entry that still earns `profit` at settlement
    sell = math.ceil(entry * (1 + profit) - 1e-9)  # bid that is `profit` above the entry
    return dict(entry=entry, max_buy=max_buy, sell=sell, early_ok=sell <= 99,
                settle_ok=entry <= max_buy, settle_pct=(100 - entry) / entry * 100)


class LockBook:
    """Log of original lock calls, shared by every visitor. Thread-safe, and mirrored to a JSON
    file so it survives script reruns (a full app restart or redeploy can still wipe it)."""

    def __init__(self, path: str):
        self.path, self._mu, self.rows = path, threading.Lock(), []
        try:
            with open(path) as fh:
                self.rows = list(json.load(fh))
        except Exception:
            self.rows = []

    def _save(self) -> None:
        try:
            with open(self.path, "w") as fh:
                json.dump(self.rows[-2000:], fh)
        except Exception:
            pass

    def get(self, ticker: str | None) -> dict | None:
        with self._mu:
            return next((dict(r) for r in self.rows if r["ticker"] == ticker), None)

    def add(self, row: dict) -> bool:
        with self._mu:
            if any(r["ticker"] == row["ticker"] for r in self.rows):
                return False  # only the ORIGINAL lock call per market is kept
            self.rows.append(row)
            self._save()
            return True

    def update_peak(self, ticker: str, bid: int | None) -> None:
        if bid is None:
            return
        with self._mu:
            for r in self.rows:
                if r["ticker"] == ticker and r["result"] is None:
                    if r.get("peak_bid") is None or bid > r["peak_bid"]:
                        r["peak_bid"] = bid
                        r["tp_hit"] = bool(r["tp_hit"] or bid >= r["sell"])
                        self._save()

    def settle(self, results: dict) -> None:
        with self._mu:
            changed = False
            for r in self.rows:
                if r["result"] is None and r["ticker"] in results:
                    r["result"] = results[r["ticker"]]
                    r["win"] = r["side"] == r["result"]
                    r["pnl"] = (100 - r["entry"]) if r["win"] else -r["entry"]
                    changed = True
            if changed:
                self._save()

    def all(self) -> list[dict]:
        with self._mu:
            return [dict(r) for r in self.rows]


@st.cache_resource
def get_lock_book() -> LockBook:
    return LockBook(LOCK_BOOK_PATH)


def lock_settle(settled: pd.DataFrame | None) -> None:
    """Grade pending lock calls whose market Kalshi has settled."""
    if settled is None or settled.empty:
        return
    get_lock_book().settle(dict(zip(settled["ticker"], settled["result"])))


def lock_track(spec: Spec, ctx: dict, cfg: dict, lk: LockResult) -> dict | None:
    """Log the ORIGINAL lock call for the current market (first time all 7 tests pass) and keep
    its peak bid up to date. Returns that market's lock row, if any."""
    cur = ctx["cur"]
    if cur is None:
        return None
    book, ticker = get_lock_book(), cur["ticker"]
    if lk.passed and lk.entry is not None:
        plan = lock_plan(lk.entry, cfg["lock"].profit)
        book.add(dict(
            ticker=ticker, minutes=spec.minutes, side=lk.side, entry=lk.entry, bid=lk.bid,
            p_side=round(float(lk.p_side), 4), spot=round(float(ctx["spot"]), 2),
            target=round(float(ctx["target"]), 2), secs_left=int(ctx["secs_left"]),
            logged_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
            close_time=cur["close_time"].strftime("%Y-%m-%d %H:%M:%S"),
            profit=cfg["lock"].profit, max_buy=plan["max_buy"], sell=plan["sell"],
            peak_bid=lk.bid, tp_hit=bool(lk.bid is not None and lk.bid >= plan["sell"]),
            result=None, win=None, pnl=None))
    row = book.get(ticker)
    if row and row["result"] is None:
        book.update_peak(ticker, ctx["quotes"].get("yes_bid" if row["side"] == "yes" else "no_bid"))
        row = book.get(ticker)
    return row


def render_lock_panel(lk: LockResult, row: dict | None, ctx: dict, cfg: dict, spec: Spec) -> None:
    """The 7-test checklist, plus the buy/sell plan for the ORIGINAL lock call in this market."""
    n_ok = sum(1 for t in lk.tests if t[1])
    with st.expander(f"🔒 Lock check: {n_ok}/7 tests pass" + ("  ·  LOCKED" if lk.passed else ""),
                     expanded=True):
        for name, ok, detail in lk.tests:
            st.markdown(f"{'✅' if ok else '❌'} **{name}** — {detail}")
        if row is None:
            st.caption("When all 7 pass, the lock call appears here with its buy/sell plan and is saved to "
                       "the record below.")
            return
        profit = row.get("profit", cfg["lock"].profit)
        pl = lock_plan(row["entry"], profit)
        pct = int(round(profit * 100))
        label = _side_label(row["side"])
        st.markdown(f"**Original lock call:** {label} at {row['logged_at'][11:]} UTC, "
                    f"{row['secs_left']}s before the close, model {row['p_side']:.0%}.")
        lines = [f"**BUY** at the ask: **{row['entry']}¢** (about {row['entry']}% implied). "
                 f"Don't pay more than **{pl['max_buy']}¢** if you want +{pct}% at settlement."]
        if pl["early_ok"]:
            lines.append(f"**SELL for +{pct}%:** place the sell when the {label.split()[0]} bid reaches "
                         f"**{pl['sell']}¢** (entry × {1 + profit:.2f}).")
        else:
            lines.append(f"**SELL for +{pct}%:** not possible by selling early (needs {pl['sell']}¢, above "
                         f"the 99¢ cap). Holding to settlement pays at most {100 - row['entry']}¢ "
                         f"(+{pl['settle_pct']:.0f}%).")
        lines.append(f"**If the bid never reaches the target:** hold to settlement. It pays 100¢ if right "
                     f"(+{pl['settle_pct']:.0f}%) and 0¢ if wrong (−100%).")
        closes = pd.Timestamp(row["close_time"], tz="UTC")
        left = (closes - ctx["now"]).total_seconds()
        hold = (f"**HOLD** until the bid hits {pl['sell']}¢ or the market closes at "
                f"{closes.tz_convert(cfg['tz']):%H:%M} ({cfg['tz']}), at most {max(int(left), 0) // 60}:"
                f"{max(int(left), 0) % 60:02d} from now ({row['secs_left']}s from the lock). "
                f"Exit early if the dashboard calls SELL (your side drops below "
                f"{int(cfg['rules'].sell_below * 100)}%).") if row["result"] is None else \
            f"Market settled **{row['result'].upper()}**: this lock was a **{'WIN' if row['win'] else 'LOSS'}** " \
            f"({row['pnl']:+d}¢ per contract if held to settlement)."
        lines.append(hold)
        for ln in lines:
            st.markdown(f"- {ln}")
        mine = "yes_bid" if row["side"] == "yes" else "no_bid"
        st.caption(f"Bid now {_c(ctx['quotes'].get(mine))}¢ · peak bid since the lock {_c(row.get('peak_bid'))}¢ · "
                   f"+{pct}% target {'reached' if row['tp_hit'] else 'not reached yet'} · before fees. "
                   "Hold time is a deadline, not a forecast: nothing says how long the bid takes to get there.")


def render_lock_record(spec: Spec, cfg: dict) -> None:
    """Win/loss record of ORIGINAL lock calls, graded by Kalshi's settled result."""
    st.markdown("**🔒 Original lock call record** (graded by Kalshi's settled result; shared by everyone "
                "using this app)")
    rows = [r for r in get_lock_book().all() if r["minutes"] == spec.minutes]
    done = [r for r in rows if r["win"] is not None]
    wins = sum(1 for r in done if r["win"])
    _metric_row(wins, len(done) - wins)
    pending = sum(1 for r in rows if r["result"] is None)
    hit = sum(1 for r in done if r["tp_hit"])
    pnl_hold = sum(r["pnl"] for r in done)
    pnl_tp = sum((r["sell"] - r["entry"]) if r["tp_hit"] else r["pnl"] for r in done)
    pct = int(round(cfg["lock"].profit * 100))
    if rows:
        st.caption(f"{len(rows)} lock calls · {pending} awaiting settlement · +{pct}% target reached on "
                   f"{hit} of {len(done)} settled · P&L per 1-contract: hold to settlement {pnl_hold:+d}¢, "
                   f"take profit at target {pnl_tp:+d}¢ (before fees)")
        with st.expander(f"Lock call log ({len(rows)})"):
            t = pd.DataFrame(rows)
            show = pd.DataFrame({
                f"Locked ({cfg['tz']})": pd.to_datetime(t["logged_at"], utc=True).dt.tz_convert(cfg["tz"]).dt.strftime("%m-%d %H:%M:%S"),
                "Side": t["side"].map({"yes": "UP", "no": "DOWN"}), "Buy ¢": t["entry"], "Model": t["p_side"].map("{:.0%}".format),
                "Secs left": t["secs_left"], "Sell target ¢": t["sell"], "Peak bid ¢": t["peak_bid"],
                f"+{pct}% hit": t["tp_hit"].map({True: "yes", False: "no"}),
                "Kalshi": t["result"].fillna("pending").str.upper(),
                "Result": t["win"].map({True: "WIN", False: "LOSS"}).fillna("—"), "P&L ¢": t["pnl"],
            }).iloc[::-1]
            st.dataframe(show, hide_index=True, width="stretch")
            st.download_button("Download lock record (CSV)", t.to_csv(index=False).encode(),
                               file_name=f"btc_{spec.minutes}m_lock_record.csv", mime="text/csv",
                               key=f"dl_lock_{spec.minutes}")
        st.caption("This log lives on the app server while the app runs. Streamlit Cloud can wipe it on a "
                   "restart or redeploy, so download the CSV to keep a copy. Locks are only logged while "
                   "the page is open somewhere.")
    else:
        st.caption("No lock calls yet. A lock is rare by design; each one is logged here once, graded when "
                   "Kalshi settles the market.")


# --------------------------------------------------------------------------- #
# Win / loss records
#
# A WIN is a call that matches how KALSHI settled the market: result "yes" = UP won,
# "no" = DOWN won. Two records:
#   1) Trend-signal record: replays the headline UP/DOWN rule over the recent windows
#      Kalshi has already settled (works immediately, no waiting).
#   2) Live record: dashboard BUY calls and positions you mark, graded as markets settle.
#      It lives in your browser session, so download the CSV to keep it.
# --------------------------------------------------------------------------- #
def fmt_ratio(wins: int, losses: int) -> str:
    if wins + losses == 0:
        return "—"
    return "∞" if losses == 0 else f"{wins / losses:.2f}"


def fmt_rate(wins: int, losses: int) -> str:
    return "—" if wins + losses == 0 else f"{wins / (wins + losses):.1%}"


def signal_record(df: pd.DataFrame, settled: pd.DataFrame | None, spec: Spec,
                  rsi_ob: float, rsi_os: float) -> dict:
    """Grade the trend signal against Kalshi's settled results.

    The signal for the window that opens at T is the one from the candle that closed at T
    (opened T - interval), so nothing is looked up from the future."""
    sig = signal_series(df, rsi_ob, rsi_os)
    if settled is not None and not settled.empty:
        source = "Kalshi settled results"
        out = pd.DataFrame({
            "window_start": settled["close_time"] - spec.delta,
            "outcome": settled["result"].map({"yes": "UP", "no": "DOWN"}),
            "ticker": settled["ticker"],
        })
    else:
        source = "Exchange candles (estimate: close >= open)"
        c = df[df["is_closed"]]
        out = pd.DataFrame({"window_start": c.index,
                            "outcome": np.where(c["close"] >= c["open"], "UP", "DOWN"),
                            "ticker": ""})
    out = out.dropna(subset=["outcome"]).reset_index(drop=True)
    out["signal"] = sig.reindex(out["window_start"] - spec.delta).to_numpy()
    out = out[out["signal"].notna()]  # windows older than the loaded candles can't be graded
    called = out[out["signal"].isin(["UP", "DOWN"])].copy()
    called["win"] = called["signal"] == called["outcome"]
    wins = int(called["win"].sum())
    return dict(rows=called, wins=wins, losses=len(called) - wins, n_windows=len(out),
                n_none=len(out) - len(called), source=source,
                up_rate=float((out["outcome"] == "UP").mean()) if len(out) else float("nan"))


def ledger_log(kind: str, minutes: int, ticker: str, side: str, entry, replace: bool = False) -> None:
    """Record a BOT call or a ME position once per market."""
    rows = st.session_state.setdefault("ledger", [])
    for r in rows:
        if r["kind"] == kind and r["ticker"] == ticker:
            if replace and r["result"] is None:
                r["side"], r["entry"] = side, entry
            return
    rows.append(dict(kind=kind, minutes=minutes, ticker=ticker, side=side, entry=entry,
                     logged_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
                     result=None, win=None, pnl=None))


def ledger_mark_sold(kind: str, ticker: str) -> None:
    """You exited early, so Kalshi's settlement isn't your result; keep it out of the W/L count."""
    for r in st.session_state.get("ledger", []):
        if r["kind"] == kind and r["ticker"] == ticker and r["result"] is None:
            r["result"] = "SOLD"


def ledger_settle(settled: pd.DataFrame | None) -> None:
    """Grade every pending row whose market Kalshi has settled."""
    if settled is None or settled.empty:
        return
    res = dict(zip(settled["ticker"], settled["result"]))
    for r in st.session_state.get("ledger", []):
        if r["result"] is None and r["ticker"] in res:
            r["result"] = res[r["ticker"]]
            r["win"] = r["side"] == r["result"]
            e = r["entry"]
            r["pnl"] = None if e is None else ((100 - e) if r["win"] else -e)


def ledger_stats(kind: str, minutes: int) -> dict:
    mine = [r for r in st.session_state.get("ledger", [])
            if r["kind"] == kind and r["minutes"] == minutes]
    done = [r for r in mine if r["win"] is not None]
    wins = sum(1 for r in done if r["win"])
    return dict(rows=mine, wins=wins, losses=len(done) - wins,
                pending=sum(1 for r in mine if r["result"] is None),
                sold=sum(1 for r in mine if r["result"] == "SOLD"),
                pnl=sum(r["pnl"] for r in done if r["pnl"] is not None))


# --------------------------------------------------------------------------- #
# Target resolution (Kalshi strike, fallback estimate, or manual override)
# --------------------------------------------------------------------------- #
def resolve_target(cur: pd.Series | None, df: pd.DataFrame,
                   spec: Spec) -> tuple[float | None, str, str | None]:
    ss, k = st.session_state, spec.minutes
    ticker = cur["ticker"] if cur is not None else None
    note = None
    if ss.get(f"use_manual_target_{k}") and float(ss.get(f"manual_target_{k}") or 0) > 0:
        if ss.get(f"manual_target_ticker_{k}") == ticker:
            return float(ss[f"manual_target_{k}"]), "Manual", None
        note = "Your manual target was for an earlier market and is ignored. Re-enter it for this one."
    if cur is not None and pd.notna(cur["strike"]):
        return float(cur["strike"]), "Kalshi strike", note
    if cur is not None:
        start = cur["close_time"] - spec.delta
        if start in df.index:
            return float(df.at[start, "open"]), f"Estimated ({spec.minutes}m candle open)", note
    return None, "Unavailable", note


def _int_or_none(v):
    return int(v) if pd.notna(v) else None


def get_context(cfg: dict, spec: Spec) -> dict | None:
    """Everything a UI fragment needs, or None (after showing an error)."""
    try:
        snap = get_snapshot(spec, cfg["lookback"], cfg["offset"], cfg["rsi_ob"], cfg["rsi_os"])
    except Exception as exc:
        st.error(f"Could not load candle data from any exchange. {exc}")
        return None
    now = pd.Timestamp.now(tz="UTC")
    cur = None
    if snap["markets"] is not None:
        live = snap["markets"][snap["markets"]["close_time"] > now]
        if not live.empty:
            cur = live.iloc[0]
    target, tsrc, tnote = resolve_target(cur, snap["df"], spec)
    quotes = {k: _int_or_none(cur[k]) for k in ("yes_bid", "yes_ask", "no_bid", "no_ask")} if cur is not None else {}
    secs_left = (cur["close_time"] - now).total_seconds() if cur is not None else None
    return dict(snap=snap, now=now, cur=cur, target=target, tsrc=tsrc, tnote=tnote,
                quotes=quotes, secs_left=secs_left, spot=snap["spot"])


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
def build_chart(d: pd.DataFrame, rsi_ob: float, rsi_os: float, tz: str,
                target: float | None, tsrc: str) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.50, 0.17, 0.17, 0.16])
    x = d.index

    # Row 1: price + EMAs + crossover markers + TARGET line
    fig.add_trace(go.Candlestick(
        x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="BTC-USD",
        increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema_9"], name="EMA 9",
                             line=dict(color=AMBER, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema_21"], name="EMA 21",
                             line=dict(color=BLUE, width=1.6)), row=1, col=1)
    if target is not None:
        fig.add_trace(go.Scatter(
            x=[x[0], x[-1]], y=[target, target], mode="lines", name=f"Target ({tsrc})",
            line=dict(color="#ffffff", width=2, dash="dash"),
            hovertemplate=f"Target ${target:,.2f}<extra></extra>"), row=1, col=1)
    bull, bear = d[d["ema_cross"] == 1], d[d["ema_cross"] == -1]
    fig.add_trace(go.Scatter(x=bull.index, y=bull["low"] * 0.9993, mode="markers",
                             name="Bullish cross", marker=dict(symbol="triangle-up", size=11, color=GREEN)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=bear.index, y=bear["high"] * 1.0007, mode="markers",
                             name="Bearish cross", marker=dict(symbol="triangle-down", size=11, color=RED)),
                  row=1, col=1)

    # Row 2: RSI
    fig.add_trace(go.Scatter(x=x, y=d["rsi_14"], name="RSI 14",
                             line=dict(color="#ab47bc", width=1.6)), row=2, col=1)
    fig.add_hrect(y0=rsi_os, y1=rsi_ob, fillcolor="gray", opacity=0.08, line_width=0, row=2, col=1)
    fig.add_hline(y=rsi_ob, line_dash="dash", line_color=RED, line_width=1, row=2, col=1)
    fig.add_hline(y=rsi_os, line_dash="dash", line_color=GREEN, line_width=1, row=2, col=1)

    # Row 3: PVT + signal line + mismatch markers
    fig.add_trace(go.Scatter(x=x, y=d["pvt"], name="PVT",
                             line=dict(color="#26c6da", width=1.6)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["pvt_signal"], name="PVT signal (EMA 9)",
                             line=dict(color=AMBER, width=1.2, dash="dot")), row=3, col=1)
    mb, ms = d[d["pvt_mismatch"] == 1], d[d["pvt_mismatch"] == -1]
    fig.add_trace(go.Scatter(x=mb.index, y=mb["pvt"], mode="markers", name="Bullish mismatch",
                             marker=dict(symbol="diamond", size=8, color=GREEN)), row=3, col=1)
    fig.add_trace(go.Scatter(x=ms.index, y=ms["pvt"], mode="markers", name="Bearish mismatch",
                             marker=dict(symbol="diamond", size=8, color=RED)), row=3, col=1)

    # Row 4: wick distances in USD (upper = sell rejection above zero, lower = buy defense below)
    fig.add_trace(go.Bar(x=x, y=d["upper_wick"], name="Upper wick $", marker_color=RED,
                         hovertemplate="$%{y:,.2f}"), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=-d["lower_wick"], name="Lower wick $", marker_color=GREEN,
                         customdata=d["lower_wick"], hovertemplate="$%{customdata:,.2f}"), row=4, col=1)

    fig.update_layout(
        height=860, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        barmode="relative", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="PVT", row=3, col=1)
    fig.update_yaxes(title_text="Wick $", row=4, col=1)
    fig.update_xaxes(title_text=f"Time ({tz})", row=4, col=1)
    return fig


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
ACTION_COLORS = {"BUY UP": "#1b9e5a", "BUY DOWN": "#1e6fd9", "WAIT": "#6b7280",
                 "HOLD": "#d98e04", "SELL": "#d32f2f"}


def render_banner(title: str, subtitle: str, color: str, footer: str = "") -> None:
    foot = (f'<div style="font-size:.9rem;margin-top:8px;opacity:.85">{html.escape(footer)}</div>'
            if footer else "")
    st.markdown(
        f'<div style="padding:18px 24px;border-radius:14px;background:{color};color:#fff;">'
        f'<div style="font-size:2.4rem;font-weight:800;line-height:1.1">{html.escape(title)}</div>'
        f'<div style="font-size:1.1rem;margin-top:6px">{html.escape(subtitle)}</div>{foot}</div>',
        unsafe_allow_html=True)


def render_trend_breakdown(a: Analysis | None) -> None:
    st.subheader("Trend & pressure breakdown")
    if a is None:
        st.info("Not enough closed candles yet.")
        return
    st.caption(f"{a.headline} · last closed candle {a.candle_time:%H:%M} UTC")
    for line in a.findings:
        st.markdown(f"- {line}")
    if a.cautions:
        st.warning("**Caution flags**\n\n" + "\n".join(f"- {c}" for c in a.cautions))


def render_kalshi(ctx: dict, tz: str, spec: Spec) -> None:
    st.subheader(f"Kalshi {spec.minutes}-minute BTC contracts")
    snap, cur = ctx["snap"], ctx["cur"]
    if snap["kalshi_err"]:
        st.info(f"Kalshi market data is unavailable right now ({snap['kalshi_err'][:80]}). "
                "Charts and trend analysis still work, but trade calls need Kalshi prices.")
        return
    if cur is None:
        st.info(f"No open {spec.series} market found. A new one usually lists at the next "
                f"{spec.minutes}-minute mark. If this never changes, check the series ticker in the sidebar "
                "(Kalshi may not list this market length).")
        return
    mins, secs = divmod(max(int(ctx["secs_left"]), 0), 60)
    st.markdown(f"`{cur['ticker']}` · closes in **{mins}:{secs:02d}**  \n{cur['title']}")
    q = ctx["quotes"]
    c1, c2 = st.columns(2)
    c1.metric("UP / YES", f"{_c(q['yes_ask'])}¢ ask", f"bid {_c(q['yes_bid'])}¢", delta_color="off")
    c2.metric("DOWN / NO", f"{_c(q['no_ask'])}¢ ask", f"bid {_c(q['no_bid'])}¢", delta_color="off")

    live = snap["markets"][snap["markets"]["close_time"] > ctx["now"]]
    show = pd.DataFrame({
        "Ticker": live["ticker"],
        "Strike": live["strike"].map(lambda v: _usd(v) if pd.notna(v) else "—"),
        f"Closes ({tz})": live["close_time"].dt.tz_convert(tz).dt.strftime("%H:%M"),
        "YES bid/ask": live.apply(lambda r: f"{_c(r['yes_bid'])}/{_c(r['yes_ask'])}¢", axis=1),
        "NO bid/ask": live.apply(lambda r: f"{_c(r['no_bid'])}/{_c(r['no_ask'])}¢", axis=1),
        "Volume": live["volume"],
    })
    with st.expander(f"All open markets ({len(live)})"):
        st.dataframe(show, hide_index=True, width="stretch")


def live_top(cfg: dict, spec: Spec) -> None:
    """Action banner + live metrics. Re-runs every second."""
    k = spec.minutes
    ctx = get_context(cfg, spec)
    if ctx is None:
        return
    ss, cur, snap = st.session_state, ctx["cur"], ctx["snap"]
    ticker = cur["ticker"] if cur is not None else None
    ss[f"cur_ticker_{k}"], ss[f"quotes_{k}"] = ticker, ctx["quotes"]  # read by the position buttons

    # A position only lives until its own market closes
    pos_key = f"position_{k}"
    pos, expired = ss.get(pos_key), False
    if pos and pos.get("ticker") != ticker:
        ss[pos_key], pos, expired = None, None, True

    dec = decide(snap["analysis"], ctx["spot"], ctx["target"], ctx["secs_left"], snap["sigma"],
                 ctx["quotes"], pos, cfg["rules"], spec.seconds)
    confirmed, n, need = debounce(f"ui{k}", dec.action, ticker, pos is not None)
    now_ts = time.time()
    record_tick(k, now_ts, ctx["spot"])
    lk = evaluate_lock(spec, ctx, cfg, now_ts)
    lock_row = lock_track(spec, ctx, cfg, lk)  # logs the ORIGINAL lock call, tracks its peak bid

    if lk.passed and pos is None:
        title = f"🔒 LOCK {_side_label(lk.side)}"
        subtitle = (f"All 7 tests pass · ask {lk.entry}¢ · model {lk.p_side:.0%} · "
                    f"{int(ctx['secs_left'])}s left")
        color = LOCK_COLOR
    elif confirmed:
        title, subtitle, color = dec.title, dec.subtitle, ACTION_COLORS[dec.action]
        if dec.action in ("BUY UP", "BUY DOWN") and ticker:  # log it so it can be graded at settlement
            side = "yes" if dec.action == "BUY UP" else "no"
            ledger_log("BOT", k, ticker, side, ctx["quotes"].get("yes_ask" if side == "yes" else "no_ask"))
    else:  # signal is forming but hasn't held for enough consecutive seconds
        fallback = "HOLD" if pos else "WAIT"
        title = fallback
        subtitle = f"{dec.title} signal forming, confirming ({n}/{need}s)"
        color = ACTION_COLORS[fallback]
    pos_txt = ("You hold " + _side_label(pos["side"]) +
               (f" @ {pos['entry']}¢" if pos.get("entry") is not None else "")) if pos else "No position set"
    render_banner(title, subtitle, color, footer=f"{pos_txt} · updated {ctx['now']:%H:%M:%S} UTC")
    if expired:
        st.caption("Your previous position's market has closed, so it was cleared.")

    render_lock_panel(lk, lock_row, ctx, cfg, spec)

    with st.expander("Why this call", expanded=False):
        for line in dec.reasons:
            st.markdown(f"- {line}")

    # ---- metrics ----
    m1, m2, m3, m4, m5 = st.columns(5)
    spot, target = ctx["spot"], ctx["target"]
    m1.metric("BTC live", _usd(spot))
    m2.metric(f"Target ({ctx['tsrc']})", _usd(target) if target is not None else "n/a")
    if target is not None and dec.sd:
        m3.metric("Spot − target", f"{spot - target:+,.2f}", f"{(spot - target) / dec.sd:+.2f}σ", delta_color="off")
    else:
        m3.metric("Spot − target", "n/a")
    m4.metric("Model P(UP)", f"{dec.p_up:.0%}" if dec.p_up is not None else "n/a",
              f"YES ask {_c(ctx['quotes'].get('yes_ask'))}¢" if ctx["quotes"] else None, delta_color="off")
    if ctx["secs_left"] is not None:
        mins, secs = divmod(max(int(ctx["secs_left"]), 0), 60)
        m5.metric("Market closes in", f"{mins}:{secs:02d}")
    else:
        m5.metric("Market closes in", "n/a")

    if ctx["tnote"]:
        st.warning(ctx["tnote"])
    if ctx["tsrc"].startswith("Estimated"):
        st.caption(f"Kalshi's strike wasn't in the API response, so the target is estimated from the "
                   f"{k}-minute candle open. Check it against Kalshi and use the manual override if it's off.")
    if snap["tick_src"] is None:
        st.warning("Live price feed unavailable; using the last candle close instead.")
    if ctx["now"] - snap["df"].index[-1] > pd.Timedelta(minutes=20 + k):
        st.warning("The newest candle is more than 20 minutes old. The data feed may be lagging.")


def tradingview_html(interval: str, tz: str, height: int) -> str:
    """TradingView's own embeddable Advanced Chart. It streams live prices by itself inside its
    iframe, so Streamlit never needs to redraw it."""
    conf = {"autosize": True, "symbol": TV_SYMBOL, "interval": interval, "timezone": "Etc/UTC" if tz == "UTC" else tz,
            "theme": "dark",
            "style": "1", "locale": "en", "allow_symbol_change": True, "hide_side_toolbar": False,
            "calendar": False, "support_host": "https://www.tradingview.com"}
    return (f'<style>html,body{{margin:0;background:#131722}}</style>'
            f'<div class="tradingview-widget-container" style="height:{height}px;width:100%">'
            f'<div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>'
            f'<script type="text/javascript" '
            f'src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>'
            f'{json.dumps(conf)}</script></div>')


def render_live_chart(cfg: dict) -> None:
    """Runs once per full page load, NOT inside a refresh fragment, so the 1-second refreshes never
    touch it. Picking a timeframe (or changing the display timezone) reloads it."""
    tf = st.radio("Chart timeframe", list(TV_INTERVALS), index=2, horizontal=True, key="tv_tf",
                  label_visibility="collapsed")
    if components is None:
        st.info("This Streamlit version can't embed the TradingView chart. Open BTCUSD on tradingview.com.")
        return
    components.html(tradingview_html(TV_INTERVALS[tf], cfg["tz"], LIVE_CHART_HEIGHT), height=LIVE_CHART_HEIGHT)
    st.caption(f"TradingView live chart ({TV_SYMBOL}, {tf}). It updates itself; the rest of the page "
               "refreshes around it without redrawing it. The Kalshi target is shown in the numbers above.")


def chart_panel(cfg: dict, spec: Spec) -> None:
    """15-minute indicator chart (EMAs, RSI, PVT, wicks, Kalshi target). Refreshes only when you press the button."""
    ctx = get_context(cfg, spec)
    if ctx is None:
        return
    df = ctx["snap"]["df"]
    view = df.tail(cfg["n_candles"]).copy()
    view.index = view.index.tz_convert(cfg["tz"]).tz_localize(None)  # wall-clock times in chosen tz
    st.button("🔄 Refresh this chart", key=f"refresh_ind_{spec.minutes}")
    st.plotly_chart(build_chart(view, cfg["rsi_ob"], cfg["rsi_os"], cfg["tz"], ctx["target"], ctx["tsrc"]),
                    width="stretch", key=f"chart_{spec.minutes}")
    st.caption(f"{spec.minutes}-minute candles: {ctx['snap']['candle_src']} · live tick: "
               f"{ctx['snap']['tick_src'] or 'n/a'} · the rightmost candle is still forming; "
               "trend signals use the last closed candle.")


def live_side(cfg: dict, spec: Spec) -> None:
    ctx = get_context(cfg, spec)
    if ctx is None:
        return
    render_trend_breakdown(ctx["snap"]["analysis"])
    st.divider()
    render_kalshi(ctx, cfg["tz"], spec)


def _metric_row(wins: int, losses: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wins", wins)
    c2.metric("Losses", losses)
    c3.metric("W/L ratio", fmt_ratio(wins, losses))
    c4.metric("Win rate", fmt_rate(wins, losses))


def record_panel(cfg: dict, spec: Spec) -> None:
    """Win/loss records graded against Kalshi's settled results."""
    k = spec.minutes
    ctx = get_context(cfg, spec)
    if ctx is None:
        return
    settled, serr = None, None
    try:
        settled = fetch_kalshi_settled(spec.series, HISTORY_HOURS)
    except Exception as exc:
        serr = f"{type(exc).__name__}: {exc}"
    ledger_settle(settled)
    lock_settle(settled)

    st.subheader(f"Win / loss record · {k}-minute")
    st.caption("A win means your call matched Kalshi's settled result: UP wins if the 60-second BRTI average "
               f"before the close is at least the one before the {k}-minute window opened (ties go to UP), "
               "otherwise DOWN wins.")
    if serr:
        st.info(f"Kalshi's settled results are unavailable right now ({serr[:80]}).")

    # ---- 0) original lock calls (the strictest call, so it comes first) ----
    render_lock_record(spec, cfg)
    st.divider()

    # ---- 1) trend signal replayed over windows Kalshi already settled ----
    rec = signal_record(ctx["snap"]["df"], settled, spec, cfg["rsi_ob"], cfg["rsi_os"])
    st.markdown(f"**Trend signal record** (last ~{HISTORY_HOURS} hours, source: {rec['source']})")
    _metric_row(rec["wins"], rec["losses"])
    rows = rec["rows"]
    ups, dns = rows[rows["signal"] == "UP"], rows[rows["signal"] == "DOWN"]
    st.caption(
        f"UP calls {int(ups['win'].sum())}–{len(ups) - int(ups['win'].sum())} · "
        f"DOWN calls {int(dns['win'].sum())}–{len(dns) - int(dns['win'].sum())} · "
        f"{rec['n_none']} of {rec['n_windows']} windows had no signal · "
        + (f"for context, {rec['up_rate']:.0%} of those windows settled UP." if rec["n_windows"] else
           "no graded windows yet."))
    if rec["source"].startswith("Exchange") and not serr:
        st.warning(f"Kalshi returned no settled {spec.series} markets, so this record is an estimate from "
                   "exchange candles (close >= open). It is not Kalshi-graded. Check the series ticker "
                   "in the sidebar.")
    if len(rows):
        with st.expander("Graded windows (newest first)"):
            t = pd.DataFrame({
                f"Window opened ({cfg['tz']})": rows["window_start"].dt.tz_convert(cfg["tz"]).dt.strftime("%m-%d %H:%M"),
                "Signal": rows["signal"], "Kalshi outcome": rows["outcome"],
                "Result": np.where(rows["win"], "WIN", "LOSS"),
            }).iloc[::-1]
            st.dataframe(t, hide_index=True, width="stretch")

    # ---- 2) live record: dashboard BUY calls and your own marked positions ----
    st.markdown("**Live record** (graded as Kalshi settles each market; kept for this browser session)")
    for kind, title in (("BOT", "Dashboard BUY calls"), ("ME", "Positions I marked")):
        s = ledger_stats(kind, k)
        st.markdown(f"{title}")
        _metric_row(s["wins"], s["losses"])
        st.caption(f"{s['pending']} awaiting settlement · {s['sold']} exited early (not counted) · "
                   f"P&L {s['pnl']:+d}¢ per 1-contract bets, before fees")
    mine = [r for r in st.session_state.get("ledger", []) if r["minutes"] == k]
    if mine:
        table = pd.DataFrame(mine)
        with st.expander(f"Live record log ({len(mine)})"):
            st.dataframe(table.iloc[::-1], hide_index=True, width="stretch")
            st.download_button("Download live record (CSV)", table.to_csv(index=False).encode(),
                               file_name=f"btc_{k}m_live_record.csv", mime="text/csv", key=f"dl_ledger_{k}")
    else:
        st.caption("Nothing logged yet. A confirmed BUY banner or the 'I bought' buttons add a row.")


def render_data_table(cfg: dict, spec: Spec) -> None:
    """Static-ish extras (refreshes with the chart)."""
    ctx = get_context(cfg, spec)
    if ctx is None:
        return
    df = ctx["snap"]["df"]
    closed = df[df["is_closed"]]
    with st.expander("Recent candles, wick distances and indicator values"):
        cols = ["open", "high", "low", "close", "volume", "body", "upper_wick", "lower_wick",
                "upper_wick_ratio", "lower_wick_ratio", "ema_9", "ema_21", "rsi_14",
                "pvt", "pvt_mismatch"]
        table = closed[cols].tail(30).iloc[::-1].round(4)
        table.index = table.index.tz_convert(cfg["tz"]).strftime("%m-%d %H:%M")
        st.dataframe(table, width="stretch")
        st.download_button("Download all indicator data (CSV)",
                           df.drop(columns=["is_closed"]).to_csv().encode(),
                           file_name=f"btc_{spec.minutes}m_indicators.csv", mime="text/csv",
                           key=f"dl_csv_{spec.minutes}")


def render_footer() -> None:
    with st.expander("How the calls are made"):
        st.markdown(
            "**What Kalshi pays on:** each Up/Down market resolves YES if the 60-second average of "
            "CF Benchmarks' BRTI before the close is at least the 60-second average before the window "
            "opened. That opening reference is the **target**. It pays on price versus target at the end, "
            "not on trend.\n\n"
            "**LOCK:** shown only when seven tests all pass. DATA: every feed and the Kalshi target are "
            "present. TIME: inside the lock window and not too late. STRENGTH: the model gives your side "
            "the chance you set. LOCK: STRENGTH and the trend have held for the set time. REVERSAL: no "
            "opposing PVT mismatch or wick, and the price cushion isn't shrinking. EDGE: the model beats "
            "the ask by your minimum edge, under your price cap. CHOP: the EMAs aren't flat, crossing, or "
            "whipsawing. The first lock in a market is the *original lock call*: it is logged with a "
            "buy/sell plan and graded when Kalshi settles. A lock is a filter built on a crude model, "
            "not a guarantee; the record shows how it performs.\n\n"
            "**Win / loss:** a call wins when it matches Kalshi's own settled result for that market. "
            "The *trend signal record* replays the UP/DOWN rule over recently settled windows. The "
            "*live record* grades dashboard BUY calls and positions you mark as markets settle.\n\n"
            "**Model chance:** a random-walk estimate of finishing above the target, using recent "
            "volatility for that interval, time left, and the 60-second averaging. It is crude. It ignores "
            "news, order flow, and the small gap between Coinbase and BRTI (use the price offset to "
            "correct that).\n\n"
            "**BUY UP / BUY DOWN:** the EMA/RSI trend and the price-vs-target model must agree, the model "
            "chance must beat your threshold, the ask must be below your limit with enough edge over it, "
            "no opposing PVT mismatch, and enough time left. Must hold for 3 consecutive seconds.\n\n"
            "**HOLD / SELL (needs you to mark your position):** SELL if the model chance for your side "
            "drops below the sell threshold, or the trend flipped and your chance is under 60%. "
            "Otherwise HOLD. A flipped trend alone does not trigger a sell if price is still safely on "
            "your side of the target.\n\n"
            "**WAIT:** anything else. The reasons are listed under *Why this call*.")
    st.caption("Educational tool, not financial advice. Read-only: this app uses public data and cannot place "
               "or see any trades. Calls are simple rules with no proven edge, and prediction-market "
               "contracts can lose their full price.")


# --------------------------------------------------------------------------- #
# Position / target callbacks + per-interval dashboard + entry point
# --------------------------------------------------------------------------- #
def _stamp_manual_target(k: int) -> None:
    st.session_state[f"manual_target_ticker_{k}"] = st.session_state.get(f"cur_ticker_{k}")


def _set_position(side: str, k: int) -> None:
    ss = st.session_state
    q = ss.get(f"quotes_{k}") or {}
    override = int(ss.get(f"entry_override_{k}") or 0)
    entry = override if override > 0 else q.get("yes_ask" if side == "yes" else "no_ask")
    ticker = ss.get(f"cur_ticker_{k}")
    ss[f"position_{k}"] = {"side": side, "ticker": ticker, "entry": entry}
    if ticker:
        ledger_log("ME", k, ticker, side, entry, replace=True)


def _clear_position(k: int) -> None:
    ss = st.session_state
    pos = ss.get(f"position_{k}")
    ss[f"position_{k}"] = None
    if pos and pos.get("ticker"):
        ledger_mark_sold("ME", pos["ticker"])


def _frag(fn, run_every, name: str):
    """A fragment with its own unique name so the 15m and 30m copies never share state."""
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper.__name__ = wrapper.__qualname__ = name
    return st.fragment(run_every=run_every)(wrapper)


def render_dashboard(spec: Spec, cfg: dict, fast, slow) -> None:
    """The full set of features for one market length."""
    k = spec.minutes
    with st.expander("My position & target line"):
        st.caption("The app can't see your Kalshi account. Tell it what you bought to get HOLD/SELL calls "
                   f"for the {k}-minute market.")
        b1, b2, b3 = st.columns(3)
        b1.button("✅ I bought UP (YES)", on_click=_set_position, args=("yes", k), key=f"buy_up_{k}")
        b2.button("✅ I bought DOWN (NO)", on_click=_set_position, args=("no", k), key=f"buy_dn_{k}")
        b3.button("⬜ I'm flat / I sold", on_click=_clear_position, args=(k,), key=f"flat_{k}")
        st.number_input("Entry price override (¢, 0 = current ask)", 0, 99, 0, key=f"entry_override_{k}")
        st.caption("Target line: default is Kalshi's strike. Override if it doesn't match what Kalshi shows. "
                   "Applies to the current market only.")
        st.checkbox("Override target manually", key=f"use_manual_target_{k}",
                    on_change=_stamp_manual_target, args=(k,))
        st.number_input("Manual target ($)", 0.0, 1_000_000.0, 0.0, step=1.0, format="%.2f",
                        key=f"manual_target_{k}", on_change=_stamp_manual_target, args=(k,))

    top_frag = _frag(live_top, fast, f"top_{k}")
    chart_frag = _frag(chart_panel, None, f"chart_{k}")  # no timer: refreshes only on its button
    side_frag = _frag(live_side, fast, f"side_{k}")
    record_frag = _frag(record_panel, slow, f"record_{k}")
    table_frag = _frag(render_data_table, slow, f"table_{k}")

    top_frag(cfg, spec)
    chart_col, side_col = st.columns([3, 2], gap="medium")
    with chart_col:
        render_live_chart(cfg)
        with st.expander("15-minute indicator chart (EMA 9/21, RSI, PVT, wicks, Kalshi target line)"):
            chart_frag(cfg, spec)
    with side_col:
        side_frag(cfg, spec)
    st.divider()
    record_frag(cfg, spec)
    table_frag(cfg, spec)


def main() -> None:
    st.set_page_config(page_title="BTC 15m Live Action", page_icon="₿", layout="wide")
    st.title("₿ Bitcoin 15-Minute Live Action Dashboard")
    st.caption("Live BTC price + Kalshi's 15-minute Up/Down contract · public data, no login · "
               "tells you LOCK / BUY / WAIT / HOLD / SELL and keeps win/loss records")

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh market data"):
            for fn in (fetch_candles, fetch_kalshi_markets, fetch_kalshi_settled, fetch_ticker):
                fn.clear()
        auto = st.toggle("Auto-refresh", value=True)
        live_secs = st.slider("Live refresh (seconds)", 1, 10, 1, disabled=not auto)
        chart_secs = st.slider("Chart refresh (seconds)", 2, 30, 5, disabled=not auto)

        st.divider()
        offset = st.number_input("Price feed offset ($)", -500.0, 500.0, 0.0, step=1.0, format="%.2f",
                                 help="Added to every Coinbase price. Set it to (Kalshi's live BTC price − "
                                      "this app's live price) so both are on the same scale.")

        with st.expander("Kalshi series"):
            st.caption("Kalshi's 15-minute Bitcoin Up/Down series is KXBTC15M.")
            series = st.text_input("Series ticker", KALSHI_SERIES).strip().upper() or KALSHI_SERIES

        with st.expander("Decision rules"):
            min_p = st.slider("Min model chance to BUY (%)", 50, 90, 60)
            min_edge = st.slider("Min edge over ask (¢)", 0, 15, 3)
            max_price = st.slider("Max price to pay (¢)", 50, 99, 85)
            min_secs = st.slider("No new entries in last (s)", 10, 180, 45)
            sell_below = st.slider("SELL if your chance drops below (%)", 10, 50, 40)

        with st.expander("Lock rules"):
            lock_p = st.slider("STRENGTH: min model chance (%)", 60, 99, 80)
            lock_hold = st.slider("LOCK: must hold for (s)", 3, 60, 10)
            lock_window = st.slider("TIME: lock only in the last (s)", 60, 600, 300, step=30)
            lock_profit = st.slider("Profit target for the buy/sell plan (%)", 5, 60, 25)

        with st.expander("Chart & indicators"):
            n_candles = st.slider("Candles shown", 60, 300, 120, step=10)
            tz = st.selectbox("Display timezone", TIMEZONES, index=0)
            rsi_ob = st.number_input("RSI overbought", 50.0, 95.0, 70.0, step=1.0)
            rsi_os = st.number_input("RSI oversold", 5.0, 50.0, 30.0, step=1.0)
            lookback = st.slider("PVT mismatch lookback (candles)", 3, 15, 5)

    cfg = dict(n_candles=n_candles, tz=tz, rsi_ob=rsi_ob, rsi_os=rsi_os, lookback=lookback,
               offset=offset, fast_secs=live_secs if auto else 1,
               lock=LockRules(min_p=lock_p / 100, hold_secs=lock_hold, window_secs=lock_window,
                              profit=lock_profit / 100),
               rules=Rules(min_p=min_p / 100, min_edge=float(min_edge), max_price=max_price,
                           min_secs=min_secs, sell_below=sell_below / 100))

    fast = live_secs if auto else None
    slow = chart_secs if auto else None
    render_dashboard(Spec(15, series), cfg, fast, slow)
    render_footer()


if __name__ == "__main__":
    main()
