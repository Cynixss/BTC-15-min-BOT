"""
app.py: Live Bitcoin 15- and 30-minute action dashboard for Kalshi BTC Up/Down markets.

Read-only and keyless: it uses only public endpoints and never places orders.
  * Candles + live tick : Coinbase Exchange (primary), Binance.US / Binance (fallbacks)
  * Contracts           : Kalshi public Trade API v2  GET /markets?series_ticker=KXBTC15M (and 30M)

Kalshi resolves each window on CF Benchmarks' BRTI: YES if the 60-second average
before the close is >= the 60-second average before the window opened. The
"target" line is that opening reference (Kalshi's strike), editable by hand.

WIN / LOSS: a call is a WIN when it matches Kalshi's own settled result for that market
(result "yes" = UP won, "no" = DOWN won). Ties settle YES (UP), per the rule above.

Deploy on Streamlit Community Cloud with this file plus requirements.txt.
Run locally:  streamlit run app.py
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_SECONDS = 900  # exchange candles are fetched at 15m and rolled up for longer windows
OHLCV = ["open", "high", "low", "close", "volume"]
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_DEFAULTS = {15: "KXBTC15M", 30: "KXBTC30M"}  # 30M ticker is a best guess; editable in the sidebar
HISTORY_HOURS = 76  # ~300 fifteen-minute candles, the most the exchange APIs return
HEADERS = {"User-Agent": "btc-15m-dashboard/1.0 (streamlit)"}
HTTP_TIMEOUT = 10

WICK_LONG = 0.5  # a wick >= 50% of the candle's high-low range counts as "long"
FLAT_SPREAD_PCT = 0.02  # |EMA9-EMA21| below 0.02% of price = EMAs essentially flat
GREEN, RED, AMBER, BLUE = "#26a69a", "#ef5350", "#f5a623", "#4a90e2"

TIMEZONES = ["UTC", "America/Chicago", "America/New_York", "America/Los_Angeles",
             "Europe/London", "Asia/Tokyo"]


@dataclass(frozen=True)
class Spec:
    """One Kalshi market length (15 or 30 minutes) and the series that trades it."""
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


def resample_candles(df15: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Roll 15-minute candles up to `minutes` (Coinbase has no 30m granularity).
    Buckets align to :00/:30 UTC. The oldest bucket is dropped if incomplete; the newest
    (still forming) bucket is kept."""
    if minutes == BASE_SECONDS // 60:
        return df15
    per = minutes // (BASE_SECONDS // 60)
    g = df15.resample(f"{minutes}min", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    n = g["close"].count()
    out, n = out[n > 0], n[n > 0]
    if len(out) and n.iloc[0] < per:
        out = out.iloc[1:]
    out.index.name = df15.index.name
    return out


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
    raw15, candle_src = fetch_candles()  # raises if every exchange fails; caller handles it
    raw = resample_candles(raw15, spec.minutes)
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

    if confirmed:
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
        st.caption(f"Kalshi's strike wasn't in the API response, so the target is estimated from the "
                   f"{k}-minute candle open. Check it against Kalshi and use the manual override if it's off.")
    if snap["tick_src"] is None:
        st.warning("Live price feed unavailable; using the last candle close instead.")
    if ctx["now"] - snap["df"].index[-1] > pd.Timedelta(minutes=20 + k):
        st.warning("The newest candle is more than 20 minutes old. The data feed may be lagging.")


def chart_panel(cfg: dict, spec: Spec) -> None:
    """Interactive chart. Re-runs every few seconds (a full redraw each second is too heavy)."""
    ctx = get_context(cfg, spec)
    if ctx is None:
        return
    df = ctx["snap"]["df"]
    view = df.tail(cfg["n_candles"]).copy()
    view.index = view.index.tz_convert(cfg["tz"]).tz_localize(None)  # wall-clock times in chosen tz
    st.plotly_chart(build_chart(view, cfg["rsi_ob"], cfg["rsi_os"], cfg["tz"], ctx["target"], ctx["tsrc"]),
                    width="stretch", key=f"chart_{spec.minutes}")
    st.caption(f"{spec.minutes}-minute candles: {ctx['snap']['candle_src']} · live tick: "
               f"{ctx['snap']['tick_src'] or 'n/a'} · "
               "the rightmost candle is still forming; trend signals use the last closed candle.")


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

    st.subheader(f"Win / loss record · {k}-minute")
    st.caption("A win means your call matched Kalshi's settled result: UP wins if the 60-second BRTI average "
               f"before the close is at least the one before the {k}-minute window opened (ties go to UP), "
               "otherwise DOWN wins.")
    if serr:
        st.info(f"Kalshi's settled results are unavailable right now ({serr[:80]}).")

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
        st.warning(f"Kalshi returned no settled {spec.series} markets, so this record is an est... (12 KB left)
app.py
62 KB
