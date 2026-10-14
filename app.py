"""
app.py: Live Bitcoin 15-minute action dashboard for Kalshi KXBTC15M markets.

Read-only and keyless: it uses only public endpoints and never places orders.
  * Candles + live tick : Coinbase Exchange (primary), Binance.US / Binance (fallbacks)
  * Contracts           : Kalshi public Trade API v2  GET /markets?series_ticker=KXBTC15M

Kalshi resolves each window on CF Benchmarks' BRTI: YES if the 60-second average
before the close is >= the 60-second average before the window opened. The
"target" line is that opening reference (Kalshi's strike), editable by hand.

Deploy on Streamlit Community Cloud with this file plus requirements.txt.
Run locally:  streamlit run app.py
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
INTERVAL = pd.Timedelta(minutes=15)
INTERVAL_SECONDS = 900
OHLCV = ["open", "high", "low", "close", "volume"]
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = "KXBTC15M"
HEADERS = {"User-Agent": "btc-15m-dashboard/1.0 (streamlit)"}
HTTP_TIMEOUT = 10

WICK_LONG = 0.5  # a wick >= 50% of the candle's high-low range counts as "long"
FLAT_SPREAD_PCT = 0.02  # |EMA9-EMA21| below 0.02% of price = EMAs essentially flat
GREEN, RED, AMBER, BLUE = "#26a69a", "#ef5350", "#f5a623", "#4a90e2"

TIMEZONES = ["UTC", "America/Chicago", "America/New_York", "America/Los_Angeles",
             "Europe/London", "Asia/Tokyo"]


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
                {"granularity": INTERVAL_SECONDS})
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
    between all visitors so a busy public app doesn't hammer the exchange."""
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
def fetch_kalshi_markets() -> pd.DataFrame:
    """Open KXBTC15M markets, soonest-closing first."""
    raw, cursor = [], None
    for _ in range(5):  # pagination safety cap
        params = {"series_ticker": KALSHI_SERIES, "status": "open", "limit": 100}
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


def compute_indicators(raw: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
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
    df["is_closed"] = (df.index + INTERVAL) <= pd.Timestamp.now(tz="UTC")
    return df


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
# Live snapshot (one shared, 1-second-cached bundle of everything the UI needs)
# --------------------------------------------------------------------------- #
def overlay_live_price(raw: pd.DataFrame, price: float | None, now: pd.Timestamp) -> pd.DataFrame:
    """Fold the live tick into the forming candle (or start one if the feed lags)."""
    if price is None:
        return raw
    raw = raw.copy()
    window_start = now.floor("15min")
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


def realized_sigma(df: pd.DataFrame) -> float:
    """Std-dev of 15-minute log returns over the last ~16 hours (as a fraction)."""
    closes = df.loc[df["is_closed"], "close"].tail(65)
    rets = np.log(closes).diff().dropna()
    sigma = float(rets.std()) if len(rets) >= 10 else 0.002
    return max(sigma, 0.0007)


def get_snapshot(lookback: int, offset: float, rsi_ob: float, rsi_os: float) -> dict:
    """Candles + live tick + indicators + Kalshi markets, computed at most once per second
    no matter how many fragments or visitors ask for it."""
    now = pd.Timestamp.now(tz="UTC")
    raw, candle_src = fetch_candles()  # raises if every exchange fails; caller handles it
    tick, tick_src = None, None
    try:
        tick, tick_src = fetch_ticker()
    except Exception:
        pass
    raw = overlay_live_price(raw, tick, now)
    raw = raw.copy()
    raw[["open", "high", "low", "close"]] += offset  # put the feed on Kalshi's (BRTI) price scale
    df = compute_indicators(raw, lookback)

    markets, kalshi_err = None, None
    try:
        markets = fetch_kalshi_markets()
    except Exception as exc:
        kalshi_err = f"{type(exc).__name__}: {exc}"

    return dict(
        df=df, candle_src=candle_src, tick_src=tick_src, spot=float(df["close"].iloc[-1]),
        sigma15=realized_sigma(df), analysis=analyze(df, rsi_ob, rsi_os, lookback),
        markets=markets, kalshi_err=kalshi_err,
    )


# --------------------------------------------------------------------------- #
# Probability model + decision engine
# --------------------------------------------------------------------------- #
def model_prob_up(spot: float, target: float, secs_left: float, sigma15: float) -> tuple[float, float]:
    """P(final settlement >= target) under a random walk, plus the 1-sigma move in USD.

    Kalshi settles on the 60-second average of BRTI before close, so less than the full
    remaining time is 'uncertain': variance-time = secs_left - 40 (>= 60s left), and
    secs_left^3 / 10800 inside the last minute (Brownian average of a 60s window).
    """
    eff = secs_left - 40 if secs_left >= 60 else secs_left ** 3 / 10800
    eff = max(eff, 1.0)
    sd = max(sigma15 * spot * math.sqrt(eff / INTERVAL_SECONDS), 0.5)
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
           sigma15: float, q: dict, pos: dict | None, r: Rules) -> Decision:
    """Turn trend analysis + price-vs-target + Kalshi quotes + your position into ONE action."""
    if a is None or spot is None or target is None or secs_left is None:
        why = ("No open Kalshi market found" if secs_left is None else
               "No target price. Enter one in the sidebar" if target is None else "Waiting for data")
        return Decision("WAIT", "WAIT", why, [why + "."])
    if secs_left <= 0:
        return Decision("WAIT", "WAIT", "Market is closing", ["The current market has closed; a new one lists shortly."])

    p_up, sd = model_prob_up(spot, target, secs_left, sigma15)
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


def debounce(action: str, ticker: str | None, has_pos: bool) -> tuple[bool, int, int]:
    """BUY needs 3 consecutive 1-second reads and SELL needs 2, so a single noisy tick can't
    flash a trade call. WAIT/HOLD show immediately. Returns (confirmed, count, needed)."""
    d = st.session_state.setdefault("deb", {})
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
# Target resolution (Kalshi strike, fallback estimate, or manual override)
# --------------------------------------------------------------------------- #
def resolve_target(cur: pd.Series | None, df: pd.DataFrame) -> tuple[float | None, str, str | None]:
    ss = st.session_state
    ticker = cur["ticker"] if cur is not None else None
    note = None
    if ss.get("use_manual_target") and float(ss.get("manual_target") or 0) > 0:
        if ss.get("manual_target_ticker") == ticker:
            return float(ss["manual_target"]), "Manual", None
        note = "Your manual target was for an earlier market and is ignored. Re-enter it for this one."
    if cur is not None and pd.notna(cur["strike"]):
        return float(cur["strike"]), "Kalshi strike", note
    if cur is not None:
        start = cur["close_time"] - INTERVAL
        if start in df.index:
            return float(df.at[start, "open"]), "Estimated (15m candle open)", note
    return None, "Unavailable", note


def _int_or_none(v):
    return int(v) if pd.notna(v) else None


def get_context(cfg: dict) -> dict | None:
    """Everything a UI fragment needs, or None (after showing an error)."""
    try:
        snap = get_snapshot(cfg["lookback"], cfg["offset"], cfg["rsi_ob"], cfg["rsi_os"])
    except Exception as exc:
        st.error(f"Could not load candle data from any exchange. {exc}")
        return None
    now = pd.Timestamp.now(tz="UTC")
    cur = None
    if snap["markets"] is not None:
        live = snap["markets"][snap["markets"]["close_time"] > now]
        if not live.empty:
            cur = live.iloc[0]
    target, tsrc, tnote = resolve_target(cur, snap["df"])
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


def render_kalshi(ctx: dict, tz: str) -> None:
    st.subheader("Kalshi 15-minute BTC contracts")
    snap, cur = ctx["snap"], ctx["cur"]
    if snap["kalshi_err"]:
        st.info(f"Kalshi market data is unavailable right now ({snap['kalshi_err'][:80]}). "
                "Charts and trend analysis still work, but trade calls need Kalshi prices.")
        return
    if cur is None:
        st.info("No open KXBTC15M market found. A new one usually lists at the next quarter-hour.")
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


def live_top(cfg: dict) -> None:
    """Action banner + live metrics. Re-runs every second."""
    ctx = get_context(cfg)
    if ctx is None:
        return
    ss, cur, snap = st.session_state, ctx["cur"], ctx["snap"]
    ticker = cur["ticker"] if cur is not None else None
    ss["cur_ticker"], ss["quotes"] = ticker, ctx["quotes"]  # read by the sidebar buttons

    # A position only lives until its own market closes
    pos, expired = ss.get("position"), False
    if pos and pos.get("ticker") != ticker:
        ss["position"], pos, expired = None, None, True

    dec = decide(snap["analysis"], ctx["spot"], ctx["target"], ctx["secs_left"], snap["sigma15"],
                 ctx["quotes"], pos, cfg["rules"])
    confirmed, n, need = debounce(dec.action, ticker, pos is not None)

    if confirmed:
        title, subtitle, color = dec.title, dec.subtitle, ACTION_COLORS[dec.action]
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

    with st.expander("Why this call", expanded=True):
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
        st.caption("Kalshi's strike wasn't in the API response, so the target is estimated from the 15-minute "
                   "candle open. Check it against Kalshi and use the manual override if it's off.")
    if snap["tick_src"] is None:
        st.warning("Live price feed unavailable; using the last candle close instead.")
    if ctx["now"] - snap["df"].index[-1] > pd.Timedelta(minutes=20):
        st.warning("The newest candle is more than 20 minutes old. The data feed may be lagging.")


def chart_panel(cfg: dict) -> None:
    """Interactive chart. Re-runs every few seconds (a full redraw each second is too heavy)."""
    ctx = get_context(cfg)
    if ctx is None:
        return
    df = ctx["snap"]["df"]
    view = df.tail(cfg["n_candles"]).copy()
    view.index = view.index.tz_convert(cfg["tz"]).tz_localize(None)  # wall-clock times in chosen tz
    st.plotly_chart(build_chart(view, cfg["rsi_ob"], cfg["rsi_os"], cfg["tz"], ctx["target"], ctx["tsrc"]),
                    width="stretch")
    st.caption(f"Candles: {ctx['snap']['candle_src']} · live tick: {ctx['snap']['tick_src'] or 'n/a'} · "
               "the rightmost candle is still forming; trend signals use the last closed candle.")


def live_side(cfg: dict) -> None:
    ctx = get_context(cfg)
    if ctx is None:
        return
    render_trend_breakdown(ctx["snap"]["analysis"])
    st.divider()
    render_kalshi(ctx, cfg["tz"])


def render_data_table(cfg: dict) -> None:
    """Static-ish extras (refreshes with the chart)."""
    ctx = get_context(cfg)
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
                           file_name="btc_15m_indicators.csv", mime="text/csv", key="dl_csv")


def render_footer() -> None:
    with st.expander("How the calls are made"):
        st.markdown(
            "**What Kalshi pays on:** each 15-minute market resolves YES if the 60-second average of "
            "CF Benchmarks' BRTI before the close is at least the 60-second average before the window "
            "opened. That opening reference is the **target**. It pays on price versus target at the end, "
            "not on trend.\n\n"
            "**Model chance:** a random-walk estimate of finishing above the target, using recent 15-minute "
            "volatility, time left, and the 60-second averaging. It is crude. It ignores news, order flow, "
            "and the small gap between Coinbase and BRTI (use the price offset to correct that).\n\n"
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
# Sidebar callbacks + entry point
# --------------------------------------------------------------------------- #
def _stamp_manual_target() -> None:
    st.session_state["manual_target_ticker"] = st.session_state.get("cur_ticker")


def _set_position(side: str) -> None:
    ss = st.session_state
    q = ss.get("quotes") or {}
    override = int(ss.get("entry_override") or 0)
    entry = override if override > 0 else q.get("yes_ask" if side == "yes" else "no_ask")
    ss["position"] = {"side": side, "ticker": ss.get("cur_ticker"), "entry": entry}


def _clear_position() -> None:
    st.session_state["position"] = None


def main() -> None:
    st.set_page_config(page_title="BTC 15m Live Action", page_icon="₿", layout="wide")
    st.title("₿ Bitcoin 15-Minute Live Action Dashboard")
    st.caption("Live BTC price + Kalshi KXBTC15M contracts · public data, no login · tells you BUY / WAIT / HOLD / SELL")

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh market data"):
            for fn in (fetch_candles, fetch_kalshi_markets, fetch_ticker, get_snapshot):
                fn.clear()
        auto = st.toggle("Auto-refresh", value=True)
        live_secs = st.slider("Live refresh (seconds)", 1, 10, 1, disabled=not auto)
        chart_secs = st.slider("Chart refresh (seconds)", 2, 30, 5, disabled=not auto)

        st.divider()
        st.subheader("My position")
        st.caption("The app can't see your Kalshi account. Tell it what you bought to get HOLD/SELL calls.")
        st.button("✅ I bought UP (YES)", on_click=_set_position, args=("yes",))
        st.button("✅ I bought DOWN (NO)", on_click=_set_position, args=("no",))
        st.button("⬜ I'm flat / I sold", on_click=_clear_position)
        st.number_input("Entry price override (¢, 0 = current ask)", 0, 99, 0, key="entry_override")

        st.divider()
        st.subheader("Target line")
        st.caption("Default is Kalshi's strike. Override if it doesn't match what Kalshi shows. "
                   "Applies to the current market only.")
        st.checkbox("Override target manually", key="use_manual_target", on_change=_stamp_manual_target)
        st.number_input("Manual target ($)", 0.0, 1_000_000.0, 0.0, step=1.0, format="%.2f",
                        key="manual_target", on_change=_stamp_manual_target)
        offset = st.number_input("Price feed offset ($)", -500.0, 500.0, 0.0, step=1.0, format="%.2f",
                                 help="Added to every Coinbase price. Set it to (Kalshi's live BTC price − "
                                      "this app's live price) so both are on the same scale.")

        with st.expander("Decision rules"):
            min_p = st.slider("Min model chance to BUY (%)", 50, 90, 60)
            min_edge = st.slider("Min edge over ask (¢)", 0, 15, 3)
            max_price = st.slider("Max price to pay (¢)", 50, 99, 85)
            min_secs = st.slider("No new entries in last (s)", 10, 180, 45)
            sell_below = st.slider("SELL if your chance drops below (%)", 10, 50, 40)

        with st.expander("Chart & indicators"):
            n_candles = st.slider("Candles shown", 60, 300, 120, step=10)
            tz = st.selectbox("Display timezone", TIMEZONES, index=0)
            rsi_ob = st.number_input("RSI overbought", 50.0, 95.0, 70.0, step=1.0)
            rsi_os = st.number_input("RSI oversold", 5.0, 50.0, 30.0, step=1.0)
            lookback = st.slider("PVT mismatch lookback (candles)", 3, 15, 5)

    cfg = dict(n_candles=n_candles, tz=tz, rsi_ob=rsi_ob, rsi_os=rsi_os, lookback=lookback,
               offset=offset,
               rules=Rules(min_p=min_p / 100, min_edge=float(min_edge), max_price=max_price,
                           min_secs=min_secs, sell_below=sell_below / 100))

    fast = live_secs if auto else None
    slow = chart_secs if auto else None
    top_frag = st.fragment(run_every=fast)(live_top)
    chart_frag = st.fragment(run_every=slow)(chart_panel)
    side_frag = st.fragment(run_every=fast)(live_side)
    table_frag = st.fragment(run_every=slow)(render_data_table)

    top_frag(cfg)
    chart_col, side_col = st.columns([3, 2], gap="medium")
    with chart_col:
        chart_frag(cfg)
    with side_col:
        side_frag(cfg)
    table_frag(cfg)
    render_footer()


if __name__ == "__main__":
    main()
