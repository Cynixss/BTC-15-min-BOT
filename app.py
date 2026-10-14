"""
app.py: Live Bitcoin dashboard for Kalshi's 15-minute BTC Up/Down market: the 15-minute indicator
chart (EMAs, RSI, PVT, wicks, Kalshi target),
a buyer/seller pressure indicator, a scalp watch (brief flips), a 7-test LOCK call, and win/loss records.

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

import hmac
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

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_SECONDS = 900  # the dashboard works on 15-minute candles (Kalshi's market length)
OHLCV = ["open", "high", "low", "close", "volume"]
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = "KXBTC15M"
HISTORY_HOURS = 76  # ~300 fifteen-minute candles, the most the exchange APIs return

POPUP_SECS = 10  # the full-screen BUY NOW / SELL NOW pop-up closes by itself after this long

# Scalp watch: chart label -> (chart minutes, exchange candle size in seconds it is built from)
SCALP_FRAMES = {"1m": (1, 60), "3m": (3, 60), "5m": (5, 300), "15m": (15, 900), "1h": (60, 3600)}
BINANCE_INTERVAL = {60: "1m", 300: "5m", 900: "15m", 3600: "1h"}

THEME_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--bg:#070b14;--panel:#0e1526;--line:rgba(120,160,255,.14);--cyan:#22d3ee;--violet:#a78bfa;--muted:#8b9ab3}
html,body,[data-testid='stAppViewContainer'],.stApp{font-family:'Inter',system-ui,sans-serif !important;}
.stApp{background:radial-gradient(1200px 600px at 12% -10%,rgba(34,211,238,.08),transparent 60%),
 radial-gradient(900px 500px at 100% 0%,rgba(167,139,250,.08),transparent 55%),#070b14 !important;}
[data-testid='stHeader']{background:transparent !important;}
.block-container{padding-top:2.2rem !important;max-width:1500px;}
h1{font-weight:800 !important;letter-spacing:-.02em;background:linear-gradient(90deg,#e6edf7,#22d3ee 55%,#a78bfa);
 -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
h2,h3{font-weight:700 !important;letter-spacing:-.01em;color:#e6edf7 !important;}
h3::before{content:'';display:inline-block;width:6px;height:1em;margin-right:10px;vertical-align:-.12em;border-radius:3px;
 background:linear-gradient(180deg,#22d3ee,#a78bfa);}
[data-testid='stCaptionContainer'],.stCaption{color:var(--muted) !important;}
[data-testid='stSidebar']{background:linear-gradient(180deg,#0b1222,#080d18) !important;border-right:1px solid var(--line);}
/* every coloured call box: glass sheen + hairline border + soft glow */
div[style*='border-radius'][style*='color:#fff']{
 background-image:linear-gradient(135deg,rgba(255,255,255,.16),rgba(0,0,0,.28)) !important;
 border:1px solid rgba(255,255,255,.18) !important;border-radius:16px !important;
 box-shadow:0 10px 34px -14px rgba(0,0,0,.8),0 0 0 1px rgba(255,255,255,.03) inset,0 0 26px -10px rgba(34,211,238,.35) !important;
 text-shadow:0 1px 10px rgba(0,0,0,.35);}
div[style*='border:2px solid #d98e04']{border:1px solid rgba(251,191,36,.45) !important;border-radius:16px !important;
 background:rgba(251,191,36,.07) !important;color:#fde7a8;}
/* metrics = glass cards with mono numbers */
[data-testid='stMetric']{background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
 border:1px solid var(--line);border-radius:14px;padding:12px 16px;box-shadow:0 8px 24px -16px #000;}
[data-testid='stMetricValue']{font-family:'JetBrains Mono',monospace !important;font-weight:700 !important;}
[data-testid='stMetricLabel']{color:var(--muted) !important;text-transform:uppercase;letter-spacing:.08em;font-size:.72rem !important;}
.stButton>button,.stDownloadButton>button{background:rgba(34,211,238,.08) !important;color:#d9f7ff !important;
 border:1px solid rgba(34,211,238,.4) !important;border-radius:12px !important;font-weight:600;
 transition:box-shadow .15s,background .15s !important;}
.stButton>button:hover{background:rgba(34,211,238,.18) !important;box-shadow:0 0 18px -2px rgba(34,211,238,.55);}
[data-testid='stDataFrame'],[data-testid='stTable'],[data-testid='stExpander'],[data-testid='stPlotlyChart']{
 border:1px solid var(--line);border-radius:14px;overflow:hidden;background:rgba(14,21,38,.6);}
[data-testid='stPlotlyChart']{padding:6px;}
hr{border-color:var(--line) !important;}
code,.stCode,pre{font-family:'JetBrains Mono',monospace !important;}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:#1c2a47;border-radius:8px}
::-webkit-scrollbar-track{background:transparent}
"""

# Lock call
LOCK_BOOK_PATH = os.path.join(tempfile.gettempdir(), "btc_kalshi_lock_book.json")
LOCK_COLOR = "#7c3aed"
DRIFT_SECS = 30  # REVERSAL test looks at how the price cushion changed over this long
EROSION_SIGMA = 0.35  # ...and fails if it shrank by more than this many 1-sigma moves
CHOP_MAX_CROSSES = 1  # CHOP test: max EMA crossovers in the last 10 closed candles
CHOP_RANGE_MULT = 2.5  # CHOP test: last candle range vs the median of the previous 20
PRESSURE_COLORS = {"HIGH BUYERS / LOW SELLERS": "#0f9d73", "HIGH SELLERS / LOW BUYERS": "#e11d48",
                   "BALANCED": "#475569", "UNAVAILABLE": "#475569"}
LOCK_TESTS = ["DATA", "TIME", "STRENGTH", "LOCK", "REVERSAL", "EDGE", "CHOP"]
HEADERS = {"User-Agent": "btc-15m-dashboard/1.0 (streamlit)"}
HTTP_TIMEOUT = 10

WICK_LONG = 0.5  # a wick >= 50% of the candle's high-low range counts as "long"
FLAT_SPREAD_PCT = 0.02  # |EMA9-EMA21| below 0.02% of price = EMAs essentially flat
GREEN, RED, AMBER, BLUE = "#2de2a6", "#ff5d7a", "#fbbf24", "#38bdf8"

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


def _from_coinbase(gran: int = BASE_SECONDS) -> pd.DataFrame:
    # rows: [time, low, high, open, close, volume], newest first, max 300
    rows = _get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                {"granularity": gran})
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return _normalise(df)


def _from_binance(base: str, symbol: str, interval: str = "15m") -> pd.DataFrame:
    rows = _get(f"{base}/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": 300})
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


@st.cache_data(ttl=2, show_spinner=False)
def fetch_flow() -> dict:
    """Coinbase's aggregated order book and the latest trades. Each half is optional: if one
    fails the other still works. Cached 2s and shared across visitors."""
    out: dict = {"book": None, "trades": None, "errors": []}
    try:
        out["book"] = _get("https://api.exchange.coinbase.com/products/BTC-USD/book", {"level": 2}, timeout=4)
    except Exception as exc:
        out["errors"].append(f"order book: {exc}")
    try:
        out["trades"] = _get("https://api.exchange.coinbase.com/products/BTC-USD/trades", {"limit": 1000}, timeout=4)
    except Exception as exc:
        out["errors"].append(f"trades: {exc}")
    return out


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
        strike = m.get("floor_strike", m.get("cap_strike"))
        rows.append({"ticker": m.get("ticker"),
                     "close_time": pd.to_datetime(m.get("close_time"), utc=True, errors="coerce"),
                     "result": res, "strike": float(strike) if strike is not None else np.nan})
    df = pd.DataFrame(rows, columns=["ticker", "close_time", "result", "strike"]).dropna(subset=["close_time"])
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


UP_ICON, DOWN_ICON, WHALE_ICON = "📈", "📉", "🐋"


def _side_label(side: str) -> str:
    return f"{UP_ICON} UP (YES)" if side == "yes" else f"{DOWN_ICON} DOWN (NO)"


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
            reasons.append(f"You entered at {entry}%; selling now would fetch {bid}% "
                           f"({bid - entry:+d} pts per contract).")
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
            reasons.append(f"Bid is already {bid}%. Only {100 - bid} pts upside remains, and selling locks it in.")
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
            fails.append(f"The {side.upper()} ask is {ask}%, above your {r.max_price}% limit.")
        edge = ps * 100 - ask
        if edge < r.min_edge:
            fails.append(f"Edge is {edge:+.0f} pts (model {ps:.0%} vs ask {ask}%); need {r.min_edge:+.0f} pts.")
    if fails:
        reasons = [f"Trend says {sig} ({a.headline.split(':', 1)[-1].strip()})."] + fails + reasons
        return Decision("WAIT", "WAIT", f"Trend {sig}, but conditions aren't met", reasons, p_up, sd)

    edge = ps * 100 - ask
    reasons = [f"Trend {sig} and price agree: model {ps:.0%} vs ask {ask}% (edge {edge:+.0f} pts).",
               f"Buying costs {ask}% and pays 100% if right; break-even win rate is {ask}%.", *reasons]
    return Decision("BUY UP" if up else "BUY DOWN", f"BUY {_side_label(side)}",
                    f"Ask {ask}% · model {ps:.0%} · edge {edge:+.0f} pts", reasons, p_up, sd)


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
# Buyer / seller pressure (Coinbase order book + recent trades)
#
# margin = (buy - sell) / (buy + sell), from -100% (all sellers) to +100% (all buyers).
#   Order book : resting buy size vs resting sell size, compared out to the SAME distance from the
#                mid price on both sides (so a thicker-looking side can't win just by reaching further).
#   Trades     : volume of buyers who lifted offers vs sellers who hit bids over the last N seconds.
#                Coinbase reports the MAKER's side, so a trade marked "buy" is a seller hitting a bid.
# It reads one exchange (Coinbase), not all of BRTI, and book size can be pulled or faked in seconds.
# Treat it as context, not a signal on its own.
# --------------------------------------------------------------------------- #
def book_pressure(book: dict | None) -> dict | None:
    try:
        bids = [(float(r[0]), float(r[1])) for r in book["bids"]]
        asks = [(float(r[0]), float(r[1])) for r in book["asks"]]
        mid = (max(p for p, _ in bids) + min(p for p, _ in asks)) / 2
        depth = min(mid - min(p for p, _ in bids), max(p for p, _ in asks) - mid)
        if depth <= 0:
            return None
        bv = sum(sz for p, sz in bids if p >= mid - depth)
        av = sum(sz for p, sz in asks if p <= mid + depth)
        if bv + av <= 0:
            return None
        return dict(buy=bv, sell=av, imb=(bv - av) / (bv + av), depth=depth)
    except Exception:
        return None


def flow_pressure(trades: list | None, now: pd.Timestamp, window: int) -> dict | None:
    try:
        t = pd.DataFrame(trades)
        t["time"] = pd.to_datetime(t["time"], utc=True)
        t["size"] = t["size"].astype(float)
        t = t[t["time"] >= now - pd.Timedelta(seconds=window)]
        if t.empty:
            return None
        taker_buy = float(t.loc[t["side"] == "sell", "size"].sum())  # maker sold -> a buyer lifted the offer
        taker_sell = float(t.loc[t["side"] == "buy", "size"].sum())  # maker bought -> a seller hit the bid
        if taker_buy + taker_sell <= 0:
            return None
        return dict(buy=taker_buy, sell=taker_sell, imb=(taker_buy - taker_sell) / (taker_buy + taker_sell),
                    span=(t["time"].max() - t["time"].min()).total_seconds(), n=len(t))
    except Exception:
        return None


def classify_pressure(overall: float | None, thr: float) -> str:
    if overall is None:
        return "UNAVAILABLE"
    if overall >= thr:
        return "HIGH BUYERS / LOW SELLERS"
    if overall <= -thr:
        return "HIGH SELLERS / LOW BUYERS"
    return "BALANCED"


def get_pressure(cfg: dict) -> dict:
    raw = fetch_flow()
    book = book_pressure(raw["book"])
    flow = flow_pressure(raw["trades"], pd.Timestamp.now(tz="UTC"), cfg["flow_window"])
    parts = [x["imb"] for x in (book, flow) if x]
    overall = float(np.mean(parts)) if parts else None
    return dict(book=book, flow=flow, overall=overall, errors=raw["errors"],
                label=classify_pressure(overall, cfg["pressure_thr"]),
                disagree=bool(book and flow and book["imb"] * flow["imb"] < 0
                              and min(abs(book["imb"]), abs(flow["imb"])) >= cfg["pressure_thr"] / 2))


def render_pressure(cfg: dict) -> None:
    pr = get_pressure(cfg)
    st.subheader("Buyer / seller pressure")
    ov = pr["overall"]
    sub = "no order book or trade data right now" if ov is None else (
        f"margin {ov:+.0%} · buyers {(1 + ov) / 2:.0%} vs sellers {(1 - ov) / 2:.0%}")
    st.markdown(f'<div style="padding:12px 18px;border-radius:12px;background:{PRESSURE_COLORS[pr["label"]]};color:#fff;">'
                f'<div style="font-size:1.4rem;font-weight:800;line-height:1.1">{html.escape(pr["label"])}</div>'
                f'<div style="font-size:.95rem;margin-top:4px">{html.escape(sub)}</div></div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    b, f = pr["book"], pr["flow"]
    c1.metric("Order book (buyers vs sellers)", f"{(1 + b['imb']) / 2:.0%} / {(1 - b['imb']) / 2:.0%}" if b else "n/a",
              f"{b['buy']:.1f} vs {b['sell']:.1f} BTC within ±${b['depth']:,.0f}" if b else None, delta_color="off")
    c2.metric(f"Trades, last {cfg['flow_window']}s (buyers vs sellers)",
              f"{(1 + f['imb']) / 2:.0%} / {(1 - f['imb']) / 2:.0%}" if f else "n/a",
              f"{f['buy']:.1f} vs {f['sell']:.1f} BTC over {f['n']} trades" if f else None, delta_color="off")
    if pr["disagree"]:
        st.warning("The order book and recent trades point in opposite directions, so treat this reading with caution.")
    if pr["errors"]:
        st.caption("Partial data: " + "; ".join(e[:70] for e in pr["errors"]))
    st.caption(f"Margin is (buy − sell) ÷ (buy + sell); ±{cfg['pressure_thr']:.0%} or more counts as high. "
               "Coinbase only, and book size can change in seconds, so this is context, not a signal on its own.")


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
        d = (f"ask {ask}% vs model {p_side:.0%} = edge {edge:+.0f} pts "
             f"(need {R.min_edge:+.0f} pts or more, ask at most {R.max_price}%)")
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


def lock_track(spec: Spec, ctx: dict, cfg: dict, lk: LockResult, flow: float | None = None) -> dict | None:
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
            flow=None if flow is None else round(float(flow), 3),
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
        lines = [f"**BUY** at the ask: **{row['entry']}%** (the market's implied odds). "
                 f"Don't pay more than **{pl['max_buy']}%** if you want +{pct}% at settlement."]
        if pl["early_ok"]:
            lines.append(f"**SELL for +{pct}%:** place the sell when the {label.split()[0]} bid reaches "
                         f"**{pl['sell']}%** (entry × {1 + profit:.2f}).")
        else:
            lines.append(f"**SELL for +{pct}%:** not possible by selling early (needs {pl['sell']}%, above "
                         f"the 99% cap). Holding to settlement pays at most {100 - row['entry']} pts "
                         f"(+{pl['settle_pct']:.0f}%).")
        lines.append(f"**If the bid never reaches the target:** hold to settlement. It pays 100% if right "
                     f"(+{pl['settle_pct']:.0f}%) and 0% if wrong (−100%).")
        closes = pd.Timestamp(row["close_time"], tz="UTC")
        left = (closes - ctx["now"]).total_seconds()
        hold = (f"**HOLD** until the bid hits {pl['sell']}% or the market closes at "
                f"{closes.tz_convert(cfg['tz']):%H:%M} ({cfg['tz']}), at most {max(int(left), 0) // 60}:"
                f"{max(int(left), 0) % 60:02d} from now ({row['secs_left']}s from the lock). "
                f"Exit early if the dashboard calls SELL (your side drops below "
                f"{int(cfg['rules'].sell_below * 100)}%).") if row["result"] is None else \
            f"Market settled **{row['result'].upper()}**: this lock was a **{'WIN' if row['win'] else 'LOSS'}** " \
            f"({row['pnl']:+d} pts per contract if held to settlement)."
        lines.append(hold)
        for ln in lines:
            st.markdown(f"- {ln}")
        mine = "yes_bid" if row["side"] == "yes" else "no_bid"
        st.caption(f"Bid now {_c(ctx['quotes'].get(mine))}% · peak bid since the lock {_c(row.get('peak_bid'))}% · "
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
                   f"{hit} of {len(done)} settled · P&L per 1-contract: hold to settlement {pnl_hold:+d} pts, "
                   f"take profit at target {pnl_tp:+d} pts (before fees)")
        with st.expander(f"Lock call log ({len(rows)})"):
            t = pd.DataFrame(rows)
            show = pd.DataFrame({
                f"Locked ({cfg['tz']})": pd.to_datetime(t["logged_at"], utc=True).dt.tz_convert(cfg["tz"]).dt.strftime("%m-%d %H:%M:%S"),
                "Side": t["side"].map({"yes": "UP", "no": "DOWN"}), "Buy %": t["entry"], "Model": t["p_side"].map("{:.0%}".format),
                "Secs left": t["secs_left"],
                "Pressure at lock": t["flow"].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.0%}") if "flow" in t else "n/a",
                "Sell target %": t["sell"], "Peak bid %": t["peak_bid"],
                f"+{pct}% hit": t["tp_hit"].map({True: "yes", False: "no"}),
                "Kalshi": t["result"].fillna("pending").str.upper(),
                "Result": t["win"].map({True: "WIN", False: "LOSS"}).fillna("—"), "P&L pts": t["pnl"],
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
# Scalp watch: when the charts say price may flip briefly
#
# Looks at five charts (1m, 3m, 5m, 15m, 1h). On each, price is "stretched" when it is far from
# its 9-EMA, and a brief flip is flagged when at least two of these three agree:
#   RSI at an extreme (overbought -> flip DOWN, oversold -> flip UP),
#   price stretched from the EMA9 by the set number of ATRs (the 14-candle average range),
#   a long wick rejecting the move on the current or last candle.
# When enough charts flag the same direction it reads SCALP DOWN or SCALP UP. If you hold a
# position (or the original lock) and the flip runs against it while it is in profit, it says
# TAKE PROFIT. These are mean-reversion heuristics: a flip may not come, or may be too small to
# trade after the spread. Nothing here is validated yet.
# --------------------------------------------------------------------------- #
SCALP_COLOR = "#ea580c"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_candles_tf(gran: int) -> tuple[pd.DataFrame, str]:
    """Latest ~300 candles at `gran` seconds (60, 300, 900 or 3600). Cached and shared."""
    errors = []
    sources = [
        ("Coinbase", lambda: _from_coinbase(gran)),
        ("Binance.US", lambda: _from_binance("https://api.binance.us", "BTCUSD", BINANCE_INTERVAL[gran])),
        ("Binance", lambda: _from_binance("https://api.binance.com", "BTCUSDT", BINANCE_INTERVAL[gran])),
    ]
    for name, fn in sources:
        try:
            df = fn()
            if len(df) >= 40:
                return df, name
            errors.append(f"{name}: only {len(df)} candles")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(" | ".join(errors))


def resample_candles(df: pd.DataFrame, minutes: int, base_seconds: int) -> pd.DataFrame:
    """Roll candles of `base_seconds` up to `minutes` (3m from 1m). Buckets align to the clock in UTC;
    the oldest bucket is dropped if incomplete, the newest (still forming) is kept."""
    per = minutes * 60 // base_seconds
    if per <= 1:
        return df
    g = df.resample(f"{minutes}min", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    n = g["close"].count()
    out, n = out[n > 0], n[n > 0]
    if len(out) and n.iloc[0] < per:
        out = out.iloc[1:]
    out.index.name = df.index.name
    return out


def scalp_frames(ctx: dict, cfg: dict) -> dict:
    """Indicator frames for every scalp chart. A chart whose data can't be loaded is left out."""
    now = ctx["now"]
    tick = ctx["spot"] - cfg["offset"]  # the app's price is the raw feed plus the offset
    out = {}
    for label, (minutes, base_gran) in SCALP_FRAMES.items():
        if label == "15m":
            out[label] = ctx["snap"]["df"]
            continue
        try:
            base, _ = fetch_candles_tf(base_gran)
            raw = resample_candles(base, minutes, base_gran)
            raw = overlay_live_price(raw, tick, now, Spec(minutes, ""))
            raw = raw.copy()
            raw[["open", "high", "low", "close"]] += cfg["offset"]
            out[label] = compute_indicators(raw, cfg["lookback"], pd.Timedelta(minutes=minutes))
        except Exception:
            continue
    return out


def flip_read(df: pd.DataFrame | None, label: str, cfg: dict) -> dict:
    """Read one chart for a possible brief flip. `flip` is "DOWN", "UP" or None."""
    out = dict(label=label, rsi=None, stretch=None, wick="—", why=[], strength=0, flip=None, verdict="n/a")
    if df is None:
        return out
    closed = df[df["is_closed"]]
    if len(closed) < 15:
        return out
    atr = float(closed["range"].tail(14).mean())
    last = df.iloc[-1]
    if not atr > 0 or pd.isna(last["rsi_14"]) or pd.isna(last["ema_9"]):
        return out
    rsi, stretch = float(last["rsi_14"]), float((last["close"] - last["ema_9"]) / atr)
    cand = last if last["range"] >= 0.5 * atr else closed.iloc[-1]  # judge a candle that has real size
    uw, lw = float(cand["upper_wick_ratio"]), float(cand["lower_wick_ratio"])
    down = [rsi >= cfg["rsi_ob"], stretch >= cfg["scalp_stretch"], uw >= WICK_LONG and uw > lw]
    up = [rsi <= cfg["rsi_os"], stretch <= -cfg["scalp_stretch"], lw >= WICK_LONG and lw > uw]
    nd, nu = sum(down), sum(up)
    out.update(rsi=rsi, stretch=stretch,
               wick=f"upper {uw:.0%}" if uw >= lw and uw >= 0.2 else f"lower {lw:.0%}" if lw >= 0.2 else "—")
    if nd == nu:
        out["verdict"] = "—"
        return out
    side, n, flags = ("DOWN", nd, down) if nd > nu else ("UP", nu, up)
    names = ({0: f"RSI {rsi:.0f} overbought", 1: f"{stretch:+.1f} ATR above EMA9", 2: f"upper wick {uw:.0%}"} if side == "DOWN"
             else {0: f"RSI {rsi:.0f} oversold", 1: f"{stretch:+.1f} ATR below EMA9", 2: f"lower wick {lw:.0%}"})
    out.update(strength=n, why=[names[i] for i, f in enumerate(flags) if f])
    if n >= 2:
        out.update(flip=side, verdict=f"FLIP {side}")
    else:
        out["verdict"] = f"watch {side.lower()}"
    return out


def get_scalp(ctx: dict, cfg: dict) -> dict:
    """Read every chart and combine them. Computed at most once per second per visitor."""
    ss, sec = st.session_state, int(time.time())
    memo = ss.get("scalp_memo")
    if memo and memo[0] == sec:
        return memo[1]
    frames = scalp_frames(ctx, cfg)
    reads = [flip_read(frames.get(label), label, cfg) for label in SCALP_FRAMES]
    n_dn = sum(1 for r in reads if r["flip"] == "DOWN")
    n_up = sum(1 for r in reads if r["flip"] == "UP")
    direction = ("DOWN" if n_dn > n_up else "UP" if n_up > n_dn else None)
    n_agree = max(n_dn, n_up)
    if n_agree < cfg["scalp_min"]:
        direction = None
    overall = None
    try:
        overall = get_pressure(cfg)["overall"]
    except Exception:
        pass
    half = cfg["pressure_thr"] / 2
    confirm = bool(direction and overall is not None and
                   ((direction == "DOWN" and overall <= -half) or (direction == "UP" and overall >= half)))
    res = dict(reads=reads, direction=direction, n_agree=n_agree, n_total=len(SCALP_FRAMES),
               confirm=confirm, pressure=overall)
    ss["scalp_memo"] = (sec, res)
    return res


def held_for(pos: dict | None, lock_row: dict | None, ticker: str | None) -> dict | None:
    """The position the calls refer to: the one you marked, else the original lock in this market."""
    if pos is not None:
        return pos
    if lock_row and lock_row["result"] is None:
        return {"side": lock_row["side"], "ticker": ticker, "entry": lock_row["entry"]}
    return None


def scalp_advice(sm: dict, ctx: dict, cfg: dict, pos: dict | None, lock_row: dict | None) -> dict | None:
    """One line for the strip under the main banner, or None when there is nothing to say."""
    d = sm["direction"]
    if not d or ctx["cur"] is None or not ctx["secs_left"] or ctx["secs_left"] <= 0:
        return None
    q = ctx["quotes"]
    held = held_for(pos, lock_row, ctx["cur"]["ticker"])
    flip_word = "dip" if d == "DOWN" else "bounce"
    base = f"{sm['n_agree']} of {sm['n_total']} charts flag a brief {flip_word}" + \
           (" and the order flow agrees" if sm["confirm"] else "")
    if held is not None:
        against = (held["side"] == "yes") == (d == "DOWN")  # a dip hurts an UP holder, a bounce hurts DOWN
        if not against:
            return None
        label = _side_label(held["side"])
        bid, entry = q.get("yes_bid" if held["side"] == "yes" else "no_bid"), held.get("entry")
        if bid is None:
            return dict(title="SCALP: FLIP AGAINST YOUR SIDE", sub=f"{base}, against your {label}. No bid quoted.",
                        color=SCALP_COLOR)
        if entry is not None and bid > entry:
            gain = bid - entry
            return dict(title="SCALP: TAKE PROFIT",
                        sub=f"{base}, against your {label}. Bid {bid}% is {gain:+d} pts ({gain / entry:+.0%}) over your "
                            f"{entry}% entry. Selling now locks that in.", color=SCALP_COLOR)
        gap = "" if entry is None else f" vs your {entry}% entry ({bid - entry:+d} pts)"
        return dict(title="SCALP: FLIP AGAINST YOUR SIDE",
                    sub=f"{base}, against your {label}. Bid {bid}%{gap}: no profit to take yet.", color=SCALP_COLOR)
    side = "no" if d == "DOWN" else "yes"  # to profit from a brief dip, hold DOWN (NO); from a bounce, UP (YES)
    ask = q.get("no_ask" if side == "no" else "yes_ask")
    title = f"SCALP {d}"
    if ask is None:
        return dict(title=title, sub=f"{base}. No {side.upper()} ask is quoted.", color=SCALP_COLOR)
    tp, stop = ask + cfg["scalp_target"], max(ask - cfg["scalp_stop"], 1)
    if tp > 99:
        return dict(title=title, sub=f"{base}, but the {_side_label(side)} ask is {ask}%, leaving no room for a "
                                     f"{cfg['scalp_target']}-point profit.", color=SCALP_COLOR)
    return dict(title=title,
                sub=f"{base}. To scalp it: buy {_side_label(side)} at {ask}%, sell near {tp}%, get out if the bid "
                    f"falls to {stop}% or after about {cfg['scalp_hold']}s, whichever comes first. A flip may not come.",
                color=SCALP_COLOR)


def render_scalp_strip(adv: dict | None) -> None:
    if adv is None:
        return
    st.markdown(f'<div style="padding:12px 18px;border-radius:12px;background:{adv["color"]};color:#fff;margin-top:8px;">'
                f'<div style="font-size:1.5rem;font-weight:800;line-height:1.1">{html.escape(adv["title"])}</div>'
                f'<div style="font-size:.95rem;margin-top:4px">{html.escape(adv["sub"])}</div></div>',
                unsafe_allow_html=True)


def render_scalp_panel(sm: dict, cfg: dict) -> None:
    st.subheader("Scalp watch (brief flips)")
    d = sm["direction"]
    if d:
        st.markdown(f"**SCALP {d}:** {sm['n_agree']} of {sm['n_total']} charts flag a brief "
                    f"{'dip' if d == 'DOWN' else 'bounce'}."
                    + (" Buyer/seller pressure agrees." if sm["confirm"] else ""))
    else:
        st.caption(f"No scalp setup: fewer than {cfg['scalp_min']} charts flag a flip in the same direction.")
    rows = []
    for r in sm["reads"]:
        rows.append({"Chart": r["label"], "RSI": "n/a" if r["rsi"] is None else f"{r['rsi']:.0f}",
                     "Stretch (ATR vs EMA9)": "n/a" if r["stretch"] is None else f"{r['stretch']:+.1f}",
                     "Wick": r["wick"], "Signal": r["verdict"], "Why": ", ".join(r["why"]) or "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"A chart flags a flip when 2 of 3 agree: RSI past {cfg['rsi_ob']:g}/{cfg['rsi_os']:g}, price "
               f"{cfg['scalp_stretch']:g}+ ATR from its EMA9, a long rejecting wick. Rule-of-thumb mean "
               "reversion, not a forecast; it can fire in a strong trend that keeps going.")


# --------------------------------------------------------------------------- #
# Markers: F (flip up/down), J (price spike / jump) and a whale indicator
#   F↑ / F↓  the 15-minute chart says price may flip up / down for a moment (same read as the scalp watch).
#   J↑ / J↓  a spike: price moved much more in the last few seconds than the last two minutes of ticks suggest.
#   WHALE    single trades of at least `whale_btc` BTC on Coinbase in the trade window (taker side).
# These are context, not signals on their own; whales on one exchange are not all of BRTI.
# --------------------------------------------------------------------------- #
WHALE_BTC = 5.0
SPIKE_SECS, SPIKE_Z, SPIKE_USD = 10, 3.0, 15.0
WHALE_COLORS = {"WHALE BUYING": "#1b9e5a", "WHALE SELLING": "#d32f2f", "WHALES BOTH WAYS": "#d98e04", "NO WHALES": "#6b7280"}


def whale_read(cfg: dict) -> dict:
    thr, window = float(cfg.get("whale_btc", WHALE_BTC)), int(cfg.get("flow_window", 180))
    out = dict(label="NO WHALES", thr=thr, window=window, n_buy=0, n_sell=0, btc_buy=0.0, btc_sell=0.0,
               largest=None, ok=False)
    try:
        raw = fetch_flow()
        t = pd.DataFrame(raw["trades"])
        t["time"] = pd.to_datetime(t["time"], utc=True)
        t["size"] = t["size"].astype(float)
    except Exception:
        return out
    now = pd.Timestamp.now(tz="UTC")
    t = t[t["time"] >= now - pd.Timedelta(seconds=window)]
    out["ok"] = True
    big = t[t["size"] >= thr]
    if big.empty:
        return out
    buys, sells = big[big["side"] == "sell"], big[big["side"] == "buy"]  # Coinbase gives the MAKER's side
    out.update(n_buy=len(buys), n_sell=len(sells), btc_buy=float(buys["size"].sum()), btc_sell=float(sells["size"].sum()))
    top = big.loc[big["size"].idxmax()]
    out["largest"] = dict(size=float(top["size"]), side="BUY" if top["side"] == "sell" else "SELL",
                          age=max(0.0, (now - top["time"]).total_seconds()), usd=float(top["size"]) * float(top.get("price", 0) or 0))
    net, tot = out["btc_buy"] - out["btc_sell"], out["btc_buy"] + out["btc_sell"]
    out["label"] = ("WHALES BOTH WAYS" if abs(net) < 0.25 * tot else "WHALE BUYING" if net > 0 else "WHALE SELLING")
    return out


def spike_read(k: int, cfg: dict) -> dict:
    """Spike = move over the last SPIKE_SECS that is both large in dollars and many times the recent tick noise."""
    secs, zmin, usd_min = int(cfg.get("spike_secs", SPIKE_SECS)), float(cfg.get("spike_z", SPIKE_Z)), float(cfg.get("spike_usd", SPIKE_USD))
    out = dict(spike=None, delta=0.0, z=0.0, secs=secs)
    ticks = list(st.session_state.get(f"ticks_{k}", []))
    if len(ticks) < 15:
        return out
    now_t, now_p = ticks[-1]
    ref = [p for ts, p in ticks if ts <= now_t - secs]
    if not ref:
        return out
    delta = now_p - ref[-1]
    older = [(ts, p) for ts, p in ticks if ts <= now_t - secs]
    steps = [(p2 - p1) / math.sqrt(max(t2 - t1, 0.5)) for (t1, p1), (t2, p2) in zip(older, older[1:])]
    if len(steps) < 8:
        return out
    sig = float(np.std(steps))
    sig = max(sig, 0.5)  # dollars per sqrt(second); floors out a dead-flat tape
    z = delta / (sig * math.sqrt(secs))
    out.update(delta=float(delta), z=float(z))
    if abs(z) >= zmin and abs(delta) >= usd_min:
        out["spike"] = "UP" if delta > 0 else "DOWN"
    return out


def marker_chips(sread: dict | None, spike: dict | None, whale: dict | None) -> list[dict]:
    """The active markers as chips: text, color, detail."""
    chips = []
    if sread and sread.get("flip"):
        up = sread["flip"] == "UP"
        chips.append(dict(text=f"F{UP_ICON if up else DOWN_ICON}", color="#1b9e5a" if up else "#d32f2f",
                          detail=f"15m flip {sread['flip']}: " + ", ".join(sread["why"])))
    if spike and spike.get("spike"):
        up = spike["spike"] == "UP"
        chips.append(dict(text=f"J{UP_ICON if up else DOWN_ICON}", color="#1b9e5a" if up else "#d32f2f",
                          detail=f"spike {spike['spike']}: {spike['delta']:+,.0f} USD in {spike['secs']}s ({spike['z']:+.1f}σ)"))
    if whale and whale.get("label") != "NO WHALES":
        chips.append(dict(text=WHALE_ICON, color=WHALE_COLORS[whale["label"]],
                          detail=f"{whale['label'].lower()}: {whale['n_buy']} buys {whale['btc_buy']:.1f} BTC, "
                                 f"{whale['n_sell']} sells {whale['btc_sell']:.1f} BTC"))
    return chips


def get_markers(ctx: dict, cfg: dict, k: int = 15) -> dict:
    sread = flip_read(ctx["snap"]["df"], f"{k}m", cfg)
    return dict(sread=sread, spike=spike_read(k, cfg), whale=whale_read(cfg))


def render_markers(mk: dict) -> None:
    """F (flip), J (spike) and whale chips, always visible (grey when quiet). When a flip AND a spike are both
    active they are drawn as one linked pair, F ➜ J, so you read them together."""
    sread, sp, wh = mk.get("sread") or {}, mk.get("spike") or {}, mk.get("whale") or {}
    GREEN, RED, GREY, AMBER = "#1b9e5a", "#d32f2f", "#6b7280", "#d98e04"
    col = lambda d: GREEN if d == "UP" else RED
    arrow = lambda d: UP_ICON if d == "UP" else DOWN_ICON

    def pill(text, color, detail="", faded=False, joined=""):
        style = (f"display:inline-block;padding:6px 14px;background:{color};color:#fff;font-weight:800;font-size:1.25rem;"
                 f"{'opacity:.55;' if faded else ''}{joined}")
        return (f'<span title="{html.escape(detail)}" style="{style}">{html.escape(text)}'
                + (f' <span style="font-weight:500;font-size:.85rem">{html.escape(detail)}</span>' if detail else "") + "</span>")

    f_dir = sread.get("flip")
    f_weak = None
    if not f_dir and str(sread.get("verdict", "")).startswith("watch"):
        f_weak = "UP" if "up" in sread["verdict"] else "DOWN"
    j_dir = sp.get("spike")
    f_why = ", ".join(sread.get("why") or []) if sread else ""
    f_txt = (f"F{arrow(f_dir)}" if f_dir else f"F{arrow(f_weak)}?" if f_weak else "F")
    f_detail = (f"15m flip {f_dir}: {f_why}" if f_dir else f"15m leaning {f_weak.lower()}: {f_why}" if f_weak
                else "no 15m flip")
    j_txt = f"J{arrow(j_dir)}" if j_dir else "J"
    j_detail = (f"spike {j_dir}: {sp['delta']:+,.0f} USD in {sp['secs']}s ({sp['z']:+.1f}σ)" if j_dir else "no spike")

    parts = []
    if (f_dir or f_weak) and j_dir:  # linked pair
        fd = f_dir or f_weak
        rel = "flip and spike agree" if fd == j_dir else "spike is against the flip"
        parts.append('<span style="display:inline-flex;align-items:stretch;margin:0 8px 6px 0;border-radius:999px;overflow:hidden">'
                     + pill(f_txt, col(fd), f_detail, faded=not f_dir)
                     + '<span style="background:#111;color:#fff;padding:6px 8px;font-weight:900;font-size:1.25rem">➜</span>'
                     + pill(j_txt, col(j_dir), j_detail) + "</span>"
                     + f'<span style="margin:0 8px 6px 0;font-weight:700;color:{AMBER if fd != j_dir else col(fd)}">{rel}</span>')
    else:
        rnd = "border-radius:999px;margin:0 8px 6px 0;"
        parts.append(pill(f_txt, col(f_dir or f_weak) if (f_dir or f_weak) else GREY, f_detail, faded=not f_dir, joined=rnd))
        parts.append(pill(j_txt, col(j_dir) if j_dir else GREY, j_detail, faded=not j_dir, joined=rnd))
    wl = wh.get("label", "NO WHALES")
    parts.append(pill("🐋", WHALE_COLORS.get(wl, GREY),
                      (f"{wl.lower()}: {wh['n_buy']} buys {wh['btc_buy']:.1f} BTC, {wh['n_sell']} sells {wh['btc_sell']:.1f} BTC"
                       if wl != "NO WHALES" else "no whales"), faded=wl == "NO WHALES",
                      joined="border-radius:999px;margin:0 8px 6px 0;"))
    st.markdown(f'<div style="margin-top:8px">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_whale_panel(mk: dict, cfg: dict) -> None:
    w, sp = mk["whale"], mk["spike"]
    st.subheader("Whales & spikes")
    sub = (f"{w['n_buy']} whale buys ({w['btc_buy']:.1f} BTC) vs {w['n_sell']} whale sells ({w['btc_sell']:.1f} BTC) "
           f"in the last {w['window']}s" if w["ok"] else "no trade data right now")
    st.markdown(f'<div style="padding:12px 18px;border-radius:12px;background:{WHALE_COLORS[w["label"]]};color:#fff;">'
                f'<div style="font-size:1.4rem;font-weight:800;line-height:1.1">🐋 {html.escape(w["label"])}</div>'
                f'<div style="font-size:.95rem;margin-top:4px">{html.escape(sub)}</div></div>', unsafe_allow_html=True)
    if w["largest"]:
        L = w["largest"]
        st.caption(f"Largest: {L['size']:.2f} BTC {L['side']} (about ${L['usd']:,.0f}) {L['age']:.0f}s ago. "
                   f"A whale trade is {w['thr']:g} BTC or more in one trade.")
    else:
        st.caption(f"No single trade of {w['thr']:g} BTC or more in the last {w['window']}s (Coinbase only).")
    st.caption(f"Spike (J): {sp['delta']:+,.0f} USD over the last {sp['secs']}s ({sp['z']:+.1f}σ vs the last 2 minutes)."
               + (f" **SPIKE {sp['spike']}**" if sp["spike"] else ""))


# --------------------------------------------------------------------------- #
# Reversals: every time BTC crosses the target line
#   A cross is logged when price moves from one side of the target to the other and clears the line by
#   `cross_band` dollars (so a price sitting on the line doesn't log dozens of crosses a second).
#   Live crosses come from the 1-second price while the page (or the bot) is running. Older windows are
#   rebuilt from 1-minute candles against Kalshi's settled strike, so the pattern view has history from day one.
# Cross risk: the chance BTC crosses the line again before this window closes. It blends
#   1) a random-walk "first touch" chance from the distance to the line, volatility and time left,
#   2) how fast price is closing in on the line (last 30s, damped by half),
#   3) buyer/seller pressure pushing toward or away from the line,
#   4) how often past crosses happened at this minute of the window (only once there are 8+ windows of history).
#   It is a rough estimate, not a forecast.
# --------------------------------------------------------------------------- #
REVERSAL_BOOK_PATH = os.path.join(tempfile.gettempdir(), "btc_kalshi_reversal_book.json")
WHALE_LOG_PATH = os.path.join(tempfile.gettempdir(), "btc_kalshi_whale_log.json")
CROSS_BAND = 3.0  # USD past the line before a cross counts
REV_HISTORY_HOURS = 48
CROSS_RISK_HIGH = 0.60
CROSS_COLORS = {"HIGH": "#e11d48", "MEDIUM": "#c2800a", "LOW": "#0f9d73", "n/a": "#475569"}
WHALE_LOG_MIN_BTC = 1.0  # the smallest whale size the slider allows; the chart filters by your setting
WHALE_LOG_HOURS = 24


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _logit(p: float) -> float:
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))


class ReversalBook:
    """Live target-line crosses, shared by every visitor (and one per bot), mirrored to JSON."""

    def __init__(self, path: str):
        self.path, self._mu = path, threading.Lock()
        self.rows, self.state, self.cover = [], {}, {}
        try:
            with open(path) as fh:
                data = json.load(fh)
            self.rows, self.cover = list(data.get("rows", [])), dict(data.get("cover", {}))
        except Exception:
            pass

    def _save(self) -> None:
        try:
            with open(self.path, "w") as fh:
                json.dump(dict(rows=self.rows[-5000:], cover=dict(list(self.cover.items())[-800:])), fh)
        except Exception:
            pass

    def observe(self, ticker: str, window_start: pd.Timestamp, spot: float, target: float, band: float,
                now: pd.Timestamp, minutes: int) -> dict | None:
        """Feed one live price. Returns the new cross row if this price completed a cross."""
        d = spot - target
        raw = 1 if d >= 0 else -1
        side = 1 if d >= band else -1 if d <= -band else 0
        with self._mu:
            wkey = window_start.isoformat()
            if wkey not in self.cover:
                self.cover[wkey] = now.isoformat()  # live coverage of this window starts here
            s = self.state.get(ticker)
            if s is None:
                self.state = {ticker: dict(side=side, raw=raw, raw_ts=now)}  # drop old windows' state
                if side == 0:
                    self.state[ticker]["side"] = 0
                return None
            if raw != s["raw"]:
                s["raw"], s["raw_ts"] = raw, now  # when price actually stepped over the line
            if side == 0 or side == s["side"]:
                return None
            prev, s["side"] = s["side"], side
            if prev == 0:
                return None  # first time it clears the band this window: a starting side, not a cross
            at = s["raw_ts"]
            secs = max(0.0, (at - window_start).total_seconds())
            row = dict(ts=at.isoformat(), ticker=ticker, window_start=window_start.isoformat(),
                       minute=int(min(secs // 60, minutes - 1)), sec=round(secs, 1),
                       direction="UP" if side > 0 else "DOWN", price=round(float(spot), 2),
                       target=round(float(target), 2), source="live")
            self.rows.append(row)
            self._save()
            return row

    def all(self) -> list[dict]:
        with self._mu:
            return [dict(r) for r in self.rows]

    def coverage(self) -> dict:
        with self._mu:
            return dict(self.cover)


@st.cache_resource
def get_reversal_book() -> ReversalBook:
    return ReversalBook(REVERSAL_BOOK_PATH)


def reversal_track(spec: Spec, ctx: dict, cfg: dict) -> dict | None:
    """Log a cross of the target line if the live price just made one. Skipped for a manual target, because the
    log is shared and another visitor's line would mix in."""
    cur, target, spot = ctx["cur"], ctx["target"], ctx["spot"]
    if cur is None or target is None or spot is None or not ctx["secs_left"] or ctx["secs_left"] <= 0:
        return None
    if ctx["tsrc"] == "Manual":
        return None
    start = cur["close_time"] - spec.delta
    return get_reversal_book().observe(cur["ticker"], start, float(spot), float(target),
                                       float(cfg.get("cross_band", CROSS_BAND)), ctx["now"], spec.minutes)


def _from_coinbase_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = _get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                {"granularity": 60, "start": start.isoformat(), "end": end.isoformat()})
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def _from_binance_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = _get("https://api.binance.us/api/v3/klines",
                {"symbol": "BTCUSD", "interval": "1m", "startTime": int(start.timestamp() * 1000),
                 "endTime": int(end.timestamp() * 1000), "limit": 1000})
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_1m_history(hours: int) -> pd.DataFrame:
    """1-minute candles for the last `hours` hours (paged; Coinbase first, Binance.US as fallback)."""
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(hours=hours)
    for step, fn in ((pd.Timedelta(minutes=300), _from_coinbase_range), (pd.Timedelta(minutes=1000), _from_binance_range)):
        try:
            parts, t = [], start
            while t < end:
                t2 = min(t + step, end)
                parts.append(fn(t, t2))
                t = t2
            df = _normalise(pd.concat(parts, ignore_index=True))
            if len(df) >= 60:
                return df
        except Exception:
            continue
    raise RuntimeError("no 1-minute history from Coinbase or Binance.US")


def _candle_crosses(m1: pd.DataFrame, start: pd.Timestamp, target: float, band: float, minutes: int,
                    ticker: str | None) -> list[dict]:
    """Crosses inside one window, read from 1-minute closes (same band rule as the live log)."""
    w = m1[(m1.index >= start) & (m1.index < start + pd.Timedelta(minutes=minutes))]
    out, side = [], 0
    for t, close in w["close"].items():
        d = close - target
        s = 1 if d >= band else -1 if d <= -band else 0
        if s == 0 or s == side:
            continue
        if side != 0:
            m = int((t - start).total_seconds() // 60)
            at = t + pd.Timedelta(minutes=1)  # the candle's close confirmed it
            out.append(dict(ts=at.isoformat(), ticker=ticker, window_start=start.isoformat(), minute=m,
                            sec=float(m * 60 + 59), direction="UP" if s > 0 else "DOWN",
                            price=round(float(close), 2), target=round(float(target), 2), source="1m candles"))
        side = s
    return out


@st.cache_data(ttl=300, show_spinner=False)
def _history_crosses(series: str, hours: int, band: float, offset: float, minutes: int) -> tuple[list[dict], int]:
    """Crosses rebuilt from 1-minute candles for every finished window in the last `hours` hours.
    Returns (rows, number of windows checked)."""
    m1 = fetch_1m_history(hours).copy()
    m1[["open", "high", "low", "close"]] += offset
    strikes = {}
    try:
        settled = fetch_kalshi_settled(series, hours)
        for _, r in settled.dropna(subset=["strike"]).iterrows():
            strikes[r["close_time"] - pd.Timedelta(minutes=minutes)] = (float(r["strike"]), r["ticker"])
    except Exception:
        pass
    try:
        c15, _ = fetch_candles()
    except Exception:
        c15 = None
    now = pd.Timestamp.now(tz="UTC")
    first = m1.index[0].ceil(f"{minutes}min")
    rows, n = [], 0
    for start in pd.date_range(first, now - pd.Timedelta(minutes=minutes), freq=f"{minutes}min"):
        if start in strikes:
            target, ticker = strikes[start]
        elif c15 is not None and start in c15.index:
            target, ticker = float(c15.at[start, "open"]) + offset, None  # estimate: the 15m candle open
        else:
            continue
        n += 1
        rows += _candle_crosses(m1, start, target, band, minutes, ticker)
    return rows, n


def reversal_history(spec: Spec, cfg: dict) -> dict:
    """Live crosses plus candle-rebuilt ones (candle rows are dropped where the live log already covered the
    window). Returns dict(rows DataFrame, windows, live_windows, err)."""
    band, hours = float(cfg.get("cross_band", CROSS_BAND)), int(cfg.get("rev_hours", REV_HISTORY_HOURS))
    book = get_reversal_book()
    live, cover_by_start = book.all(), book.coverage()  # window start -> when the live log began watching it
    hist, n_hist, err = [], 0, None
    try:
        hist, n_hist = _history_crosses(spec.series, hours, round(band, 2), round(float(cfg.get("offset", 0.0)), 2), spec.minutes)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    keep = []
    for r in hist:  # a candle cross is kept only for the part of the window the live log didn't watch
        seen = cover_by_start.get(r["window_start"])
        if seen is None or pd.Timestamp(r["ts"]) < pd.Timestamp(seen):
            keep.append(r)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    allrows = [r for r in live + keep if pd.Timestamp(r["ts"]) >= cutoff]
    df = pd.DataFrame(allrows, columns=["ts", "ticker", "window_start", "minute", "sec", "direction", "price",
                                        "target", "source"])
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True, format="ISO8601")
        df = df.drop_duplicates(subset=["ts", "direction"]).sort_values("ts").reset_index(drop=True)
        df["gap_s"] = df["ts"].diff().dt.total_seconds()
    else:
        df["gap_s"] = pd.Series(dtype=float)
    return dict(rows=df, windows=max(n_hist, len(cover_by_start)), err=err)


def cross_risk(spec: Spec, ctx: dict, cfg: dict, pr: dict | None, hist: dict | None) -> dict | None:
    """Chance BTC crosses the target line (and clears the band) before this window closes."""
    spot, target, secs_left = ctx["spot"], ctx["target"], ctx["secs_left"]
    if spot is None or target is None or secs_left is None or secs_left <= 0 or ctx["cur"] is None:
        return None
    band = float(cfg.get("cross_band", CROSS_BAND))
    d = spot - target
    dist = abs(d) + band
    sig = max(ctx["snap"]["sigma"] * spot / math.sqrt(spec.seconds), 0.05)  # $ per sqrt(second)
    T = max(float(secs_left), 1.0)
    # closing speed: how fast the gap to the line shrank over the last 30s ($/s, + = toward the line)
    ticks = list(st.session_state.get(f"ticks_{spec.minutes}", []))
    speed = 0.0
    if ticks:
        now_t, now_p = ticks[-1]
        old = [(t, p) for t, p in ticks if t <= now_t - 30] or ticks[:1]
        t0, p0 = old[-1]
        if now_t - t0 >= 5:
            speed = (abs(p0 - target) - abs(now_p - target)) / (now_t - t0)
    mu = 0.5 * speed
    s = sig * math.sqrt(T)
    expo = min(2 * mu * dist / sig ** 2, 50.0)
    p_walk = _norm_cdf((mu * T - dist) / s) + math.exp(expo) * _norm_cdf((-dist - mu * T) / s)
    p_walk = min(max(p_walk, 0.0), 1.0)
    # pressure toward the line: sellers when above, buyers when below
    ov = pr["overall"] if pr else None
    toward = 0.0 if ov is None else (-ov if d >= 0 else ov)
    # timing: crosses at this minute (and the next two) vs the average minute
    minute = int(min(max((spec.seconds - secs_left) // 60, 0), spec.minutes - 1))
    lift, n_win = None, 0
    if hist is not None and len(hist["rows"]) and hist["windows"] >= 8:
        n_win = hist["windows"]
        counts = hist["rows"]["minute"].value_counts().reindex(range(spec.minutes), fill_value=0)
        avg = counts.mean()
        here = counts.iloc[minute:min(minute + 3, spec.minutes)].mean()
        lift = float((here + 0.5) / (avg + 0.5))
    shift = 1.2 * toward + (max(-0.7, min(0.7, 0.5 * math.log(lift))) if lift else 0.0)
    p = 1 / (1 + math.exp(-(_logit(p_walk) + shift)))
    high = float(cfg.get("cross_high", CROSS_RISK_HIGH))
    level = "HIGH" if p >= high else "MEDIUM" if p >= high * 0.6 else "LOW"
    side = "above" if d >= 0 else "below"
    why = [f"BTC is {_usd(abs(d))} {side} the line with {int(secs_left)}s left (walk model {p_walk:.0%})",
           (f"closing in at {speed:+.2f} $/s" if speed > 0.05 else f"moving away at {-speed:.2f} $/s" if speed < -0.05
            else "not moving toward or away from the line"),
           ("pressure n/a" if ov is None else
            f"pressure {'toward' if toward > 0 else 'away from'} the line ({toward:+.0%})" if abs(toward) >= 0.05
            else "pressure balanced")]
    if lift:
        why.append(f"minute {minute} of the window: crosses run {lift:.1f}× the average minute ({n_win} windows)")
    else:
        why.append("not enough window history yet for the timing pattern")
    return dict(p=p, p_walk=p_walk, level=level, minute=minute, speed=speed, toward=toward, lift=lift,
                side=side, dist=abs(d), why=why, high=high)


def render_cross_risk(cr: dict | None) -> None:
    if cr is None:
        return
    col = CROSS_COLORS[cr["level"]]
    st.markdown(f'<div style="padding:10px 18px;border-radius:12px;background:{col};color:#fff;margin-top:8px">'
                f'<div style="font-size:1.25rem;font-weight:800">↔️ Cross risk {cr["p"]:.0%} · {cr["level"]}</div>'
                f'<div style="font-size:.92rem;margin-top:3px">{html.escape(" · ".join(cr["why"]))}</div></div>',
                unsafe_allow_html=True)


def build_cross_heatmap(rows: pd.DataFrame, tz: str, minutes: int) -> go.Figure:
    hrs = rows["ts"].dt.tz_convert(tz).dt.hour
    grid = pd.crosstab(hrs, rows["minute"]).reindex(index=range(24), columns=range(minutes), fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[str(m) for m in grid.columns], y=[f"{h:02d}:00" for h in grid.index],
        colorscale=[[0, "rgba(14,21,38,1)"], [0.35, "#1e3a8a"], [0.7, "#22d3ee"], [1, "#fbbf24"]],
        hovertemplate="hour %{y} · minute %{x}<br>%{z} crosses<extra></extra>", colorbar=dict(title="crosses")))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,12,22,0.55)",
                      font=dict(family="Inter, sans-serif", color="#c7d2e5"),
                      xaxis_title="Minute inside the 15-minute window", yaxis_title=f"Hour of day ({tz})")
    fig.update_yaxes(autorange="reversed")
    return fig


def _fmt_gap(s) -> str:
    if pd.isna(s):
        return "—"
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {sec:02d}s"


def reversal_panel(cfg: dict, spec: Spec) -> None:
    """Reversal record, pattern heatmap and summary. Refreshes on the records timer."""
    st.subheader("Reversals · target-line crosses")
    try:
        hist = reversal_history(spec, cfg)
    except Exception as exc:
        st.caption(f"Reversal record unavailable right now ({type(exc).__name__}).")
        return
    st.session_state[f"rev_hist_{spec.minutes}"] = hist  # the live cross-risk strip reads this
    rows, tz = hist["rows"], cfg["tz"]
    st.caption(f"A cross is BTC moving from one side of the target to the other and clearing it by "
               f"${cfg.get('cross_band', CROSS_BAND):g}. Live crosses come from the 1-second price while this page "
               f"or the bot runs; older windows are rebuilt from 1-minute candles against Kalshi's settled strike "
               f"(last {cfg.get('rev_hours', REV_HISTORY_HOURS)}h).")
    if hist["err"]:
        st.info(f"Couldn't rebuild older windows from 1-minute candles ({hist['err'][:90]}). Showing live crosses only.")
    if not len(rows):
        st.caption("No crosses logged yet.")
        return
    n_win = max(hist["windows"], 1)
    per_win = rows.groupby("window_start").size()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crosses", len(rows))
    c2.metric("Windows with a cross", f"{len(per_win) / n_win:.0%}", f"{len(per_win)} of {n_win}", delta_color="off")
    c3.metric("Avg crosses / window", f"{len(rows) / n_win:.2f}")
    c4.metric("Median time between", _fmt_gap(rows["gap_s"].median()))
    counts = rows["minute"].value_counts().reindex(range(spec.minutes), fill_value=0)
    top = counts.sort_values(ascending=False).head(3)
    hrs = rows["ts"].dt.tz_convert(tz).dt.hour.value_counts().head(3)
    st.caption("Busiest minutes of the window: " + ", ".join(f"minute {m} ({n})" for m, n in top.items())
               + " · busiest hours: " + ", ".join(f"{h:02d}:00 ({n})" for h, n in hrs.items())
               + f" · {int((rows['direction'] == 'UP').sum())} crosses up, {int((rows['direction'] == 'DOWN').sum())} down")
    st.plotly_chart(build_cross_heatmap(rows, tz, spec.minutes), width="stretch", key=f"cross_heat_{spec.minutes}")
    st.caption("Each cell counts crosses at that minute of the window (0 = the first minute) during that hour of the "
               "day. A cell that stays bright across days is a repeating timing. With only a couple of days of data, "
               "treat a single bright cell as chance.")
    t = rows.iloc[::-1]
    ws = t["window_start"].dt.tz_convert(tz)
    show = pd.DataFrame({
        f"Time ({tz})": t["ts"].dt.tz_convert(tz).dt.strftime("%m-%d %H:%M:%S"),
        "Window": ws.dt.strftime("%H:%M") + "–" + (ws + spec.delta).dt.strftime("%H:%M"),
        "Minute in window": t["minute"].astype(int),
        "Direction": t["direction"].map({"UP": "below → above ↑", "DOWN": "above → below ↓"}),
        "Price": t["price"].map(_usd), "Target": t["target"].map(_usd),
        "Since previous cross": t["gap_s"].map(_fmt_gap), "Source": t["source"],
    })
    with st.expander(f"Reversal record ({len(rows)} crosses, newest first)", expanded=False):
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button("Download reversal record (CSV)", rows.to_csv(index=False).encode(),
                           file_name=f"btc_{spec.minutes}m_reversals.csv", mime="text/csv", key=f"dl_rev_{spec.minutes}")


# --------------------------------------------------------------------------- #
# Whale timeline: the trades behind the whale indicator, kept over time so they can be charted.
# Uses the same trades, the same size rule and the same taker-side reading as whale_read(); that function is
# unchanged. Trades are logged at 1 BTC and up, and the chart shows only those at or above your whale setting.
# --------------------------------------------------------------------------- #
class WhaleLog:
    def __init__(self, path: str):
        self.path, self._mu, self.rows, self.ids, self.saved = path, threading.Lock(), [], set(), 0.0
        try:
            with open(path) as fh:
                self.rows = list(json.load(fh))
            self.ids = {r["id"] for r in self.rows}
        except Exception:
            pass

    def add(self, new: list[dict]) -> None:
        with self._mu:
            fresh = [r for r in new if r["id"] not in self.ids]
            if not fresh:
                return
            self.rows += fresh
            self.ids.update(r["id"] for r in fresh)
            cut = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=WHALE_LOG_HOURS)).isoformat()
            if len(self.rows) > 30000 or (self.rows and self.rows[0]["time"] < cut):
                self.rows = [r for r in self.rows if r["time"] >= cut][-30000:]
                self.ids = {r["id"] for r in self.rows}
            if time.time() - self.saved >= 20:  # write the file at most every 20s
                self.saved = time.time()
                try:
                    with open(self.path, "w") as fh:
                        json.dump(self.rows, fh)
                except Exception:
                    pass

    def all(self) -> list[dict]:
        with self._mu:
            return [dict(r) for r in self.rows]


@st.cache_resource
def get_whale_log() -> WhaleLog:
    return WhaleLog(WHALE_LOG_PATH)


def whale_log_update() -> None:
    """Add any new large trades from the (cached) Coinbase trade feed to the shared whale log."""
    try:
        raw = fetch_flow()
        t = pd.DataFrame(raw["trades"])
        t["size"] = t["size"].astype(float)
    except Exception:
        return
    t = t[t["size"] >= WHALE_LOG_MIN_BTC]
    if t.empty:
        return
    times = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    rows = [dict(id=str(r.get("trade_id", f"{r['time']}-{r['size']}")), time=ts.isoformat(), size=float(r["size"]),
                 side="BUY" if r["side"] == "sell" else "SELL",  # Coinbase gives the MAKER's side, as in whale_read
                 price=float(r.get("price") or 0)) for (_, r), ts in zip(t.iterrows(), times)]
    get_whale_log().add(rows)


def whale_timeline(cfg: dict, minutes: int, bucket: str = "1min") -> dict:
    """Whale trades (>= your whale size) over the last `minutes`: the trades and per-bucket buy/sell totals."""
    thr = float(cfg.get("whale_btc", WHALE_BTC))
    now = pd.Timestamp.now(tz="UTC")
    t = pd.DataFrame(get_whale_log().all(), columns=["id", "time", "size", "side", "price"])
    if len(t):
        t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        t = t[(t["size"] >= thr) & (t["time"] >= now - pd.Timedelta(minutes=minutes))].sort_values("time")
    idx = pd.date_range((now - pd.Timedelta(minutes=minutes)).floor(bucket), now.floor(bucket), freq=bucket)
    if len(t):
        g = t.set_index("time").groupby([pd.Grouper(freq=bucket), "side"])["size"].sum().unstack(fill_value=0.0)
    else:
        g = pd.DataFrame()
    b = pd.DataFrame(index=idx)
    b["buy"] = g["BUY"].reindex(idx, fill_value=0.0) if "BUY" in g else 0.0
    b["sell"] = g["SELL"].reindex(idx, fill_value=0.0) if "SELL" in g else 0.0
    b["net"] = b["buy"] - b["sell"]
    b["cum_net"] = b["net"].cumsum()
    first = get_whale_log().all()[:1]
    since = pd.Timestamp(first[0]["time"]) if first else None
    return dict(trades=t, buckets=b, thr=thr, since=since, minutes=minutes)


def build_whale_chart(wt: dict, tz: str, price: pd.DataFrame | None) -> go.Figure:
    b, t = wt["buckets"].copy(), wt["trades"].copy()
    conv = lambda ix: ix.tz_convert(tz).tz_localize(None)
    b.index = conv(b.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.45],
                        specs=[[{}], [{"secondary_y": True}]])
    if price is not None and len(price):
        p = price.copy()
        p.index = conv(p.index)
        fig.add_trace(go.Scatter(x=p.index, y=p["close"], name="BTC (1m)", line=dict(color="#94a3b8", width=1.4)), row=1, col=1)
    for side, col, sym in (("BUY", GREEN, "triangle-up"), ("SELL", RED, "triangle-down")):
        s = t[t["side"] == side] if len(t) else t
        if len(s):
            fig.add_trace(go.Scatter(
                x=conv(pd.DatetimeIndex(s["time"])), y=s["price"], mode="markers", name=f"Whale {side.lower()}",
                marker=dict(symbol=sym, color=col, size=(8 + 3 * s["size"] ** 0.5).clip(upper=34), line=dict(width=0)),
                customdata=s["size"], hovertemplate=f"{side} %{{customdata:.2f}} BTC @ $%{{y:,.0f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=b.index, y=b["buy"], name="Whale buys (BTC)", marker_color=GREEN), row=2, col=1)
    fig.add_trace(go.Bar(x=b.index, y=-b["sell"], name="Whale sells (BTC)", marker_color=RED,
                         customdata=b["sell"], hovertemplate="%{customdata:.2f} BTC"), row=2, col=1)
    fig.add_trace(go.Scatter(x=b.index, y=b["cum_net"], name="Running net (BTC)", line=dict(color=AMBER, width=2)),
                  row=2, col=1, secondary_y=True)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), barmode="relative", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,12,22,0.55)",
                      font=dict(family="Inter, sans-serif", color="#c7d2e5"))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="BTC per minute", row=2, col=1)
    fig.update_yaxes(title_text="Running net", row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text=f"Time ({tz})", row=2, col=1)
    return fig


def whale_chart_panel(cfg: dict, spec: Spec) -> None:
    st.subheader("🐋 Whale buy / sell timeline")
    whale_log_update()
    mins = int(cfg.get("whale_minutes", 120))
    wt = whale_timeline(cfg, mins)
    price = None
    try:
        p, _ = fetch_candles_tf(60)
        price = p[p.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=mins)].copy()
        price["close"] += cfg["offset"]
    except Exception:
        pass
    b = wt["buckets"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Whale buys, last {mins}m", f"{b['buy'].sum():.1f} BTC", f"{int((wt['trades']['side'] == 'BUY').sum()) if len(wt['trades']) else 0} trades", delta_color="off")
    c2.metric(f"Whale sells, last {mins}m", f"{b['sell'].sum():.1f} BTC", f"{int((wt['trades']['side'] == 'SELL').sum()) if len(wt['trades']) else 0} trades", delta_color="off")
    c3.metric("Net", f"{b['net'].sum():+.1f} BTC")
    st.plotly_chart(build_whale_chart(wt, cfg["tz"], price), width="stretch", key=f"whale_chart_{spec.minutes}")
    since = wt["since"]
    st.caption(f"Each triangle is one trade of {wt['thr']:g} BTC or more on Coinbase (green = a buyer took the offer, "
               "red = a seller hit the bid), sized by BTC. Bars are whale BTC per minute; the gold line is the running "
               "net. Same trades and rules as the 🐋 indicator. Trades are only collected while this page or the bot "
               "runs" + (f"; the log starts {since.tz_convert(cfg['tz']):%m-%d %H:%M} ({cfg['tz']})." if since is not None else "."))


# --------------------------------------------------------------------------- #
# Suggestion: BUY or SIT OUT (or HOLD / SELL once you are in), always with a warning that it is a suggestion
# --------------------------------------------------------------------------- #
SUGGEST_WARNING = ("⚠️ These are suggestions from a simple model, not financial advice. Prices can move against you "
                   "and you can lose the whole cost of a contract. Decide for yourself.")


def suggestion(dec: "Decision", confirmed: bool, lk: "LockResult", pos: dict | None, secs_left: float | None,
               pr: dict | None = None, mk: dict | None = None, thr: float = 0.20) -> dict:
    """One plain answer: BUY (a side), SIT OUT, HOLD or SELL, with why and any cautions."""
    side, why, verdict, cautions = None, "", "SIT OUT", []
    if pos:
        if dec.action == "SELL":
            verdict, side, why = "SELL", pos["side"], dec.subtitle
        else:
            verdict, side, why = "HOLD", pos["side"], dec.subtitle
    elif secs_left is None or secs_left <= 0:
        why = "No open market right now."
    elif lk.passed:
        verdict, side, why = "BUY", lk.side, f"All 7 lock tests pass (model {lk.p_side:.0%})."
    elif confirmed and dec.action in ("BUY UP", "BUY DOWN"):
        verdict, side, why = "BUY", ("yes" if dec.action == "BUY UP" else "no"), dec.subtitle
    else:
        why = dec.subtitle if dec.action == "WAIT" else f"{dec.title} signal is still confirming."
    if side is not None and verdict in ("BUY", "HOLD"):
        up = side == "yes"
        label = "UP" if up else "DOWN"
        if pr and pr.get("overall") is not None:
            against = -pr["overall"] if up else pr["overall"]
            if against >= thr:
                cautions.append(f"{'sellers' if up else 'buyers'} dominate the order flow")
        sp = (mk or {}).get("spike") or {}
        if sp.get("spike") and (sp["spike"] == "DOWN") == up:
            cautions.append(f"a spike {sp['spike']} just happened")
        fl = ((mk or {}).get("sread") or {}).get("flip")
        if fl and (fl == "DOWN") == up:
            cautions.append(f"the 15m chart flags a flip {fl}")
        wh = ((mk or {}).get("whale") or {}).get("label", "NO WHALES")
        if (wh == "WHALE SELLING" and up) or (wh == "WHALE BUYING" and not up):
            cautions.append(f"{wh.lower()} on Coinbase")
        if verdict == "BUY" and not lk.passed and len(cautions) >= 2:
            verdict, why = "SIT OUT", f"{label} looked good, but " + " and ".join(cautions) + "."
            side, cautions = None, []
    if verdict == "BUY":
        title, color = f"BUY {_side_label(side)}", ACTION_COLORS["BUY UP" if side == "yes" else "BUY DOWN"]
    elif verdict == "SIT OUT":
        title, color = "SIT OUT", "#6b7280"
    elif verdict == "HOLD":
        title, color = f"HOLD {_side_label(side)}", ACTION_COLORS["HOLD"]
    else:
        title, color = "SELL", ACTION_COLORS["SELL"]
    return dict(verdict=verdict, side=side, title=title, color=color, why=why, cautions=cautions, warning=SUGGEST_WARNING)


def render_suggestion(sg: dict) -> None:
    extra = f" Watch out: {'; '.join(sg['cautions'])}." if sg["cautions"] else ""
    st.markdown(
        f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:10px;align-items:stretch">'
        f'<div style="flex:1 1 280px;padding:12px 18px;border-radius:12px;background:{sg["color"]};color:#fff">'
        f'<div style="font-size:.75rem;letter-spacing:.08em;font-weight:700;opacity:.9">SUGGESTION</div>'
        f'<div style="font-size:1.6rem;font-weight:800;line-height:1.1">{html.escape(sg["title"])}</div>'
        f'<div style="font-size:.95rem;margin-top:4px">{html.escape(sg["why"] + extra)}</div></div>'
        f'<div style="flex:1 1 280px;padding:12px 18px;border-radius:12px;border:2px solid #d98e04;background:rgba(217,142,4,.12);'
        f'font-size:.9rem;display:flex;align-items:center">{html.escape(sg["warning"])}</div></div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Full-screen pop-up: BUY NOW on a confirmed lock, SELL NOW on a reversal or flip risk
#
# SELL NOW is judged for the position you marked, or, if you marked none, for the ORIGINAL lock
# call in this market (it assumes you bought that lock). It fires when either:
#   1) the SELL rules fire (your side's model chance fell below the sell threshold, or the trend
#      flipped against you while your chance is under 60%), or
#   2) FLIP RISK: your side's chance fell below the flip-risk level AND buyer/seller pressure is
#      against your side by at least twice the "high pressure" margin.
# BUY NOW shows while a lock is active and you hold nothing. SELL NOW wins if both apply.
# --------------------------------------------------------------------------- #
def popup_call(spec: Spec, ctx: dict, cfg: dict, dec: "Decision", lk: LockResult, pos: dict | None,
               lock_row: dict | None, pr: dict | None) -> dict | None:
    cur = ctx["cur"]
    ticker = cur["ticker"] if cur is not None else None
    held = held_for(pos, lock_row, ticker)
    sell = None
    live = ctx["secs_left"] is not None and ctx["secs_left"] > 0
    if held is not None and live and ctx["target"] is not None:
        snap = ctx["snap"]
        d = dec if pos is not None else decide(snap["analysis"], ctx["spot"], ctx["target"], ctx["secs_left"],
                                               snap["sigma"], ctx["quotes"], held, cfg["rules"], spec.seconds)
        label = _side_label(held["side"])
        if d.action == "SELL":
            sell = (f"{label} is reversing", d.subtitle)
        elif d.p_up is not None and pr and pr["overall"] is not None:
            held_p = d.p_up if held["side"] == "yes" else 1 - d.p_up
            against = -pr["overall"] if held["side"] == "yes" else pr["overall"]
            if held_p < cfg["flip_p"] and against >= 2 * cfg["pressure_thr"]:
                dominant = "Sellers" if held["side"] == "yes" else "Buyers"
                sell = ("Flip risk", f"{dominant} dominate (margin {pr['overall']:+.0%}) and {label} is only "
                                     f"{held_p:.0%} likely")
    raw = "SELL" if sell else ("LOCK" if (lk.passed and pos is None) else "NONE")
    confirmed, _, _ = debounce(f"pop{spec.minutes}", raw, ticker, held is not None)
    if not confirmed or raw == "NONE":
        return None
    if raw == "SELL":
        who = "your position" if pos is not None else "the original lock call (if you bought it)"
        return dict(kind="SELL", title="SELL NOW", color=ACTION_COLORS["SELL"], key=(ticker, "SELL", held["side"]),
                    sub=f"{sell[0]}: {sell[1]}. This is for {who}.")
    plan = lock_plan(lk.entry, cfg["lock"].profit)
    return dict(kind="BUY", title="BUY NOW", color=ACTION_COLORS["BUY UP" if lk.side == "yes" else "BUY DOWN"],
                key=(ticker, "BUY", lk.side),
                sub=f"🔒 LOCK {_side_label(lk.side)} · all 7 tests pass · buy at {lk.entry}% (not above {plan['max_buy']}%) · "
                    f"model {lk.p_side:.0%} · {int(ctx['secs_left'])}s left")


def _dismiss_popup(key) -> None:
    st.session_state["popup_dismissed_for"] = key


def render_popup(pop: dict | None) -> None:
    """A full-viewport overlay (position: fixed), so it covers the page wherever you are scrolled.
    The dismiss button is lifted above it with CSS; dismissing hides this call until it changes."""
    ss = st.session_state
    if pop is None:
        ss.pop("popup_shown", None)  # the call is over, so the next one gets a fresh 10 seconds
        return
    if ss.get("popup_dismissed_for") == pop["key"]:
        return
    shown = ss.get("popup_shown")
    if not shown or shown[0] != pop["key"]:
        shown = ss["popup_shown"] = (pop["key"], time.time())
    left = POPUP_SECS - (time.time() - shown[1])
    if left <= 0:
        return  # it has been up for POPUP_SECS: the page is back; the banner and strips still show the call
    pop = dict(pop, sub=pop["sub"] + f"  ·  closes in {int(math.ceil(left))}s")
    st.markdown(
        '<style>.st-key-popup_dismiss_btn{position:fixed;left:50%;bottom:9%;transform:translateX(-50%);'
        'z-index:1000002;}</style>'
        f'<div style="position:fixed;inset:0;z-index:1000000;background:{pop["color"]};color:#fff;display:flex;'
        'flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:24px;">'
        f'<div style="font-size:min(15vw,10rem);font-weight:900;line-height:1">{html.escape(pop["title"])}</div>'
        f'<div style="font-size:min(4.5vw,1.8rem);margin-top:20px;max-width:900px">{html.escape(pop["sub"])}</div>'
        '</div>', unsafe_allow_html=True)
    st.button("✕ Close and go back to the page", key="popup_dismiss_btn", on_click=_dismiss_popup, args=(pop["key"],))


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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,12,22,0.55)",
        font=dict(family="Inter, sans-serif", color="#c7d2e5"),
        hoverlabel=dict(bgcolor="#0e1526", bordercolor="#22d3ee", font=dict(color="#e6edf7")),
    )
    fig.update_xaxes(gridcolor="rgba(120,160,255,.08)", zerolinecolor="rgba(120,160,255,.15)")
    fig.update_yaxes(gridcolor="rgba(120,160,255,.08)", zerolinecolor="rgba(120,160,255,.15)")
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="PVT", row=3, col=1)
    fig.update_yaxes(title_text="Wick $", row=4, col=1)
    fig.update_xaxes(title_text=f"Time ({tz})", row=4, col=1)
    return fig


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
ACTION_COLORS = {"BUY UP": "#0f9d73", "BUY DOWN": "#2563eb", "WAIT": "#475569",
                 "HOLD": "#c2800a", "SELL": "#e11d48"}


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
    c1.metric("UP / YES", f"{_c(q['yes_ask'])}% ask", f"bid {_c(q['yes_bid'])}%", delta_color="off")
    c2.metric("DOWN / NO", f"{_c(q['no_ask'])}% ask", f"bid {_c(q['no_bid'])}%", delta_color="off")

    live = snap["markets"][snap["markets"]["close_time"] > ctx["now"]]
    show = pd.DataFrame({
        "Ticker": live["ticker"],
        "Strike": live["strike"].map(lambda v: _usd(v) if pd.notna(v) else "—"),
        f"Closes ({tz})": live["close_time"].dt.tz_convert(tz).dt.strftime("%H:%M"),
        "YES bid/ask": live.apply(lambda r: f"{_c(r['yes_bid'])}/{_c(r['yes_ask'])}%", axis=1),
        "NO bid/ask": live.apply(lambda r: f"{_c(r['no_bid'])}/{_c(r['no_ask'])}%", axis=1),
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
    try:
        reversal_track(spec, ctx, cfg)  # logs a target-line cross if the price just made one
        whale_log_update()  # keeps the whale timeline filled
    except Exception:
        pass
    lk = evaluate_lock(spec, ctx, cfg, now_ts)
    pr = None
    try:
        pr = get_pressure(cfg)
    except Exception:
        pass
    lock_row = lock_track(spec, ctx, cfg, lk, pr["overall"] if (pr and lk.passed) else None)  # logs the ORIGINAL lock call

    if lk.passed and pos is None:
        title = f"🔒 LOCK {_side_label(lk.side)}"
        subtitle = (f"All 7 tests pass · ask {lk.entry}% · model {lk.p_side:.0%} · "
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
               (f" @ {pos['entry']}%" if pos.get("entry") is not None else "")) if pos else "No position set"
    render_banner(title, subtitle, color, footer=f"{pos_txt} · updated {ctx['now']:%H:%M:%S} UTC")
    if expired:
        st.caption("Your previous position's market has closed, so it was cleared.")

    try:
        mk_ = get_markers(ctx, cfg, k)
        render_suggestion(suggestion(dec, confirmed, lk, pos, ctx["secs_left"], pr, mk_, cfg["pressure_thr"]))
    except Exception:
        pass  # the suggestion box must never take the main banner down
    if cfg["scalp_on"]:
        try:
            render_scalp_strip(scalp_advice(get_scalp(ctx, cfg), ctx, cfg, pos, lock_row))
        except Exception:
            pass  # the scalp watch must never take the main banner down

    try:
        render_markers(get_markers(ctx, cfg, k))
    except Exception:
        pass  # markers must never take the main banner down
    try:
        render_cross_risk(cross_risk(spec, ctx, cfg, pr, ss.get(f"rev_hist_{k}")))  # history from the reversal panel
    except Exception:
        pass  # cross risk must never take the main banner down

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
              f"YES ask {_c(ctx['quotes'].get('yes_ask'))}%" if ctx["quotes"] else None, delta_color="off")
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

    if cfg["popup"]:
        render_popup(popup_call(spec, ctx, cfg, dec, lk, pos, lock_row, pr))


def chart_panel(cfg: dict, spec: Spec) -> None:
    """The 15-minute indicator chart (EMAs, RSI, PVT, wicks, Kalshi target). It has its own timer,
    separate from the 1-second live numbers, plus a refresh button."""
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
    render_pressure(cfg)
    try:
        st.divider()
        render_whale_panel(get_markers(ctx, cfg, spec.minutes), cfg)
    except Exception as exc:
        st.caption(f"Whale/spike panel unavailable right now ({type(exc).__name__}).")
    if cfg["scalp_on"]:
        st.divider()
        try:
            render_scalp_panel(get_scalp(ctx, cfg), cfg)
        except Exception as exc:
            st.caption(f"Scalp watch unavailable right now ({type(exc).__name__}).")
    st.divider()
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
                   f"P&L {s['pnl']:+d} pts per 1-contract bets, before fees")
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
            "**Buyer / seller pressure:** compares resting buy and sell size in Coinbase's order book (out "
            "to the same distance on both sides) and the volume of buyers lifting offers versus sellers "
            "hitting bids in recent trades. A margin past the threshold reads HIGH BUYERS / LOW SELLERS or "
            "HIGH SELLERS / LOW BUYERS. It is one exchange and easy to move, so it is context only; it does "
            "not change the BUY or LOCK calls. Each lock call is logged with the reading at that moment.\n\n"
            "**Scalp watch:** reads the 1m, 3m, 5m, 15m and 1h charts for a brief flip. A chart flags one "
            "when two of three agree: RSI at an extreme, price stretched from its 9-EMA by the set number "
            "of ATRs, and a long wick rejecting the move. When enough charts agree it reads SCALP DOWN "
            "(a brief dip) or SCALP UP (a brief bounce). If you hold a position, or the original lock, "
            "and the flip runs against it while the bid is above your entry, it says TAKE PROFIT. These "
            "are mean-reversion rules of thumb: a flip may not come, or may be too small to trade after "
            "the spread. The scalp watch has no win/loss record yet.\n\n"
            "**Pop-up:** a full-screen BUY NOW appears while a lock is active and you hold nothing. A "
            "full-screen SELL NOW appears when the SELL rules fire, or on flip risk (your side's chance "
            "under the flip-risk level while buyer/seller pressure runs against it by twice the high-pressure "
            "margin). SELL NOW is judged for the position you marked, or for the original lock call if you "
            "marked none. Use *Dismiss* to hide a call until it changes. Pop-ups need the page open.\n\n"
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
            "**WAIT:** anything else. The reasons are listed under *Why this call*.\n\n"
            "**Reversals:** every time BTC crosses the target line and clears it by the set dollar band, the time, "
            "window, minute inside the window, direction and price are logged. The heatmap counts crosses by minute "
            "of the window and hour of the day, so a repeating timing shows up as a cell that stays bright.\n\n"
            "**Cross risk:** the chance BTC crosses the line again before this window closes. It starts from a "
            "random-walk chance of reaching the line (distance, volatility, time left), then leans on how fast price "
            "is closing in, whether buyer/seller pressure pushes toward the line, and how often past crosses happened "
            "at this minute of the window. It is a rough estimate.\n\n"
            "**Whale chart:** the large Coinbase trades behind the 🐋 indicator, kept over time: each trade at its "
            "price, whale BTC per minute and the running net. The whale indicator itself is unchanged.")
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
        st.number_input("Entry odds override (%, 0 = current ask)", 0, 99, 0, key=f"entry_override_{k}")
        st.caption("Target line: default is Kalshi's strike. Override if it doesn't match what Kalshi shows. "
                   "Applies to the current market only.")
        st.checkbox("Override target manually", key=f"use_manual_target_{k}",
                    on_change=_stamp_manual_target, args=(k,))
        st.number_input("Manual target ($)", 0.0, 1_000_000.0, 0.0, step=1.0, format="%.2f",
                        key=f"manual_target_{k}", on_change=_stamp_manual_target, args=(k,))

    top_frag = _frag(live_top, fast, f"top_{k}")
    chart_frag = _frag(chart_panel, None, f"chart_{k}")  # never auto-refreshes; only its own button redraws it
    side_frag = _frag(live_side, fast, f"side_{k}")
    record_frag = _frag(record_panel, slow, f"record_{k}")
    table_frag = _frag(render_data_table, slow, f"table_{k}")
    rev_frag = _frag(reversal_panel, slow, f"reversals_{k}")
    whale_frag = _frag(whale_chart_panel, slow, f"whales_{k}")

    top_frag(cfg, spec)
    chart_col, side_col = st.columns([3, 2], gap="medium")
    with chart_col:
        chart_frag(cfg, spec)
    with side_col:
        side_frag(cfg, spec)
    st.divider()
    rev_col, whale_col = st.columns(2, gap="medium")
    with rev_col:
        rev_frag(cfg, spec)
    with whale_col:
        whale_frag(cfg, spec)
    st.divider()
    record_frag(cfg, spec)
    table_frag(cfg, spec)


def check_password() -> bool:
    """Password gate. The password lives in Streamlit's Secrets as `password = "..."`. If no password is set,
    the app stays open (so a missing secret can't lock you out)."""
    try:
        expected = st.secrets.get("password")
    except Exception:  # no secrets file at all
        expected = None
    if not expected:
        return True
    if st.session_state.get("pw_ok"):
        return True

    def _try() -> None:
        typed = st.session_state.pop("pw_typed", "")
        st.session_state["pw_ok"] = hmac.compare_digest(str(typed).encode(), str(expected).encode())
        st.session_state["pw_bad"] = not st.session_state["pw_ok"]

    st.markdown("<style>" + THEME_CSS + "</style>", unsafe_allow_html=True)
    st.title("₿ Bitcoin 15-Minute Live Action Dashboard")
    st.text_input("Password", type="password", key="pw_typed", on_change=_try)
    if st.session_state.get("pw_bad"):
        st.error("Wrong password.")
    return False


def main() -> None:
    st.set_page_config(page_title="BTC 15m Live Action", page_icon="₿", layout="wide")
    if not check_password():
        st.stop()
    # Streamlit dims whatever is being redrawn on each refresh. Keep everything fully opaque and hide the
    # "running" indicator so the live panels update in place with no fade or flicker.
    st.markdown(
        "<style>"
        "[data-stale='true'],[data-stale='true'] *{opacity:1 !important;transition:none !important;}"
        "[data-testid='stStatusWidget']{display:none !important;}"
        "[data-testid='stElementContainer'],[data-testid='stAppViewContainer'] *{transition:none !important;animation:none !important;}"
        "</style>", unsafe_allow_html=True)
    st.markdown("<style>" + THEME_CSS + "</style>", unsafe_allow_html=True)
    st.title("₿ Bitcoin 15-Minute Live Action Dashboard")
    st.caption("Live BTC price + Kalshi's 15-minute Up/Down contract · public data, no login · "
               "tells you LOCK / BUY / WAIT / HOLD / SELL and keeps win/loss records")

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh market data"):
            for fn in (fetch_candles, fetch_kalshi_markets, fetch_kalshi_settled, fetch_ticker, fetch_1m_history,
                       _history_crosses):
                fn.clear()
        auto = st.toggle("Auto-refresh", value=True)
        live_secs = st.slider("Live refresh (seconds)", 1, 10, 1, disabled=not auto)
        chart_secs = st.slider("Records & tables refresh (seconds)", 5, 60, 15, disabled=not auto,
                               help="Win/loss records and tables only. The indicator chart "
                                    "never auto-refreshes; use the button above the indicator chart.")

        st.divider()
        offset = st.number_input("Price feed offset ($)", -500.0, 500.0, 0.0, step=1.0, format="%.2f",
                                 help="Added to every Coinbase price. Set it to (Kalshi's live BTC price − "
                                      "this app's live price) so both are on the same scale.")

        with st.expander("Kalshi series"):
            st.caption("Kalshi's 15-minute Bitcoin Up/Down series is KXBTC15M.")
            series = st.text_input("Series ticker", KALSHI_SERIES).strip().upper() or KALSHI_SERIES

        with st.expander("Decision rules"):
            min_p = st.slider("Min model chance to BUY (%)", 50, 90, 60)
            min_edge = st.slider("Min edge over ask (points)", 0, 15, 3)
            max_price = st.slider("Max odds to pay (%)", 50, 99, 85)
            min_secs = st.slider("No new entries in last (s)", 10, 180, 45)
            sell_below = st.slider("SELL if your chance drops below (%)", 10, 50, 40)

        with st.expander("Buyer / seller pressure"):
            pressure_thr = st.slider("High pressure at margin of (%)", 10, 60, 20)
            flow_window = st.slider("Trade window (seconds)", 60, 600, 180, step=30)

        with st.expander("Scalp watch"):
            scalp_on = st.toggle("Show scalp watch", value=True)
            scalp_min = st.slider("Charts that must agree (of 5)", 1, 5, 2)
            scalp_stretch = st.slider("Stretch from EMA9 (ATR)", 0.5, 4.0, 1.5, step=0.1)
            scalp_target = st.slider("Scalp profit target (points)", 2, 30, 5)
            scalp_stop = st.slider("Scalp stop (points below entry)", 2, 30, 5)
            scalp_hold = st.slider("Scalp time limit (seconds)", 30, 600, 120, step=30)

        with st.expander("Pop-up alerts"):
            whale_btc = st.slider("Whale = single trade of at least (BTC)", 1.0, 50.0, 5.0, step=0.5)
            spike_z = st.slider("Spike (J) sensitivity: σ moves", 2.0, 6.0, 3.0, step=0.5)
            spike_usd = st.slider("Spike (J) needs at least (USD in 10s)", 5, 200, 15, step=5)
            popup_on = st.toggle("Full-screen BUY NOW / SELL NOW pop-up", value=True)
            flip_p = st.slider("Flip risk: your side's chance below (%)", 55, 90, 75)

        with st.expander("Reversals & cross risk"):
            cross_band = st.slider("A cross must clear the line by ($)", 0.0, 25.0, 3.0, step=0.5,
                                   help="Stops a price sitting on the target from logging a cross every second.")
            rev_hours = st.slider("History for the pattern view (hours)", 12, 72, 48, step=6)
            cross_high = st.slider("Cross risk counts as HIGH at (%)", 40, 90, 60)

        with st.expander("Whale chart"):
            whale_minutes = st.slider("Whale chart shows the last (minutes)", 30, 720, 120, step=30)

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
               pressure_thr=pressure_thr / 100, flow_window=flow_window,
               popup=popup_on, flip_p=flip_p / 100,
               scalp_on=scalp_on, scalp_min=scalp_min, scalp_stretch=scalp_stretch,
               scalp_target=scalp_target, scalp_stop=scalp_stop, scalp_hold=scalp_hold,
               whale_btc=whale_btc, spike_z=spike_z, spike_usd=spike_usd,
               cross_band=cross_band, rev_hours=rev_hours, cross_high=cross_high / 100, whale_minutes=whale_minutes,
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
