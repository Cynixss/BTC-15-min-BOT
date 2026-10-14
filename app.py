"""
app.py: Bitcoin 15-minute signal dashboard for Kalshi KXBTC15M markets.

Read-only and keyless: it uses only public endpoints and never places orders.
  * Candles : Coinbase Exchange (primary), Binance.US / Binance (fallbacks)
  * Contracts: Kalshi public Trade API v2  GET /markets?series_ticker=KXBTC15M

Deploy on Streamlit Community Cloud with this file plus requirements.txt.
Run locally:  streamlit run app.py
"""

from __future__ import annotations

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
def _get(url: str, params: dict | None = None):
    resp = requests.get(url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
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


@st.cache_data(ttl=15, show_spinner=False)
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


@st.cache_data(ttl=10, show_spinner=False)
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
# Chart
# --------------------------------------------------------------------------- #
def build_chart(d: pd.DataFrame, rsi_ob: float, rsi_os: float, tz: str) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.50, 0.17, 0.17, 0.16])
    x = d.index

    # Row 1: price + EMAs + crossover markers
    fig.add_trace(go.Candlestick(
        x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="BTC-USD",
        increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema_9"], name="EMA 9",
                             line=dict(color=AMBER, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema_21"], name="EMA 21",
                             line=dict(color=BLUE, width=1.6)), row=1, col=1)
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
def render_signal(a: Analysis | None) -> None:
    st.subheader("Signal alert")
    if a is None:
        st.info("Not enough closed candles yet to compute a signal.")
        return
    if a.signal == "UP":
        st.success(f"▲ {a.headline}\n\nRule favors the **YES** (price up) side.")
    elif a.signal == "DOWN":
        st.error(f"▼ {a.headline}\n\nRule favors the **NO** (price down) side.")
    else:
        st.warning(f"◆ {a.headline}")
    st.caption(f"Based on the last closed candle ({a.candle_time:%Y-%m-%d %H:%M} UTC)")
    for line in a.findings:
        st.markdown(f"- {line}")
    if a.cautions:
        st.warning("**Caution flags**\n\n" + "\n".join(f"- {c}" for c in a.cautions))
    elif a.signal != "NONE":
        st.caption("No contradicting evidence from PVT or wicks.")


def render_kalshi(markets: pd.DataFrame | None, err: Exception | None, a: Analysis | None,
                  spot: float, tz: str) -> None:
    st.subheader("Kalshi 15-minute BTC contracts")
    if err is not None:
        st.info(f"Kalshi market data is unavailable right now ({type(err).__name__}). "
                "The chart and signals above still work.")
        return
    now = pd.Timestamp.now(tz="UTC")
    live = markets[markets["close_time"] > now].copy() if markets is not None else pd.DataFrame()
    if live.empty:
        st.info("No open KXBTC15M markets found. A new one usually lists at the next quarter-hour.")
        return

    cur = live.iloc[0]
    left = cur["close_time"] - now
    mins, secs = divmod(int(left.total_seconds()), 60)
    st.markdown(f"**Current market:** `{cur['ticker']}`  \n{cur['title']}  \nCloses in **{mins}:{secs:02d}**")

    c1, c2 = st.columns(2)
    c1.metric("YES ask", f"{_c(cur['yes_ask'])}¢")
    c2.metric("NO ask", f"{_c(cur['no_ask'])}¢")
    if pd.notna(cur["strike"]):
        diff = spot - cur["strike"]
        st.caption(f"Strike {_usd(cur['strike'])} · BTC is {_usd(abs(diff))} "
                   f"{'above' if diff >= 0 else 'below'} strike")

    if a is not None and a.signal != "NONE":
        side, ask = ("YES", cur["yes_ask"]) if a.signal == "UP" else ("NO", cur["no_ask"])
        if pd.notna(ask):
            st.info(f"**Signal vs. price:** the rule favors **{side}**, asking **{int(ask)}¢**. "
                    f"Buying wins {100 - int(ask)}¢ or loses {int(ask)}¢ per contract, so you need a "
                    f"win rate above **{int(ask)}%** just to break even (before fees).")

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


def render_dashboard(cfg: dict) -> None:
    now = pd.Timestamp.now(tz="UTC")
    try:
        raw, source = fetch_candles()
    except Exception as exc:
        st.error(f"Could not load candle data from any exchange. {exc}")
        return

    df = compute_indicators(raw, cfg["lookback"])
    closed = df[df["is_closed"]]
    if len(closed) < 2:
        st.error("Not enough closed candles returned by the exchange.")
        return

    analysis = analyze(df, cfg["rsi_ob"], cfg["rsi_os"], cfg["lookback"])
    try:
        markets, kalshi_err = fetch_kalshi_markets(), None
    except Exception as exc:
        markets, kalshi_err = None, exc

    last, last_closed = df.iloc[-1], closed.iloc[-1]
    spot = float(last["close"])
    st.caption(f"Updated {now:%H:%M:%S} UTC · candles: {source} · "
               f"{'auto-refresh on' if cfg['auto'] else 'auto-refresh off'}")
    if now - df.index[-1] > pd.Timedelta(minutes=20):
        st.warning("The newest candle is more than 20 minutes old. The data feed may be lagging.")

    # ---- metric strip ----
    next_close = now.floor("15min") + INTERVAL
    mins, secs = divmod(int((next_close - now).total_seconds()), 60)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC-USD (live)", _usd(spot), f"{spot - df['close'].iloc[-2]:+,.2f} vs prev candle")
    m2.metric("EMA9 − EMA21", _usd(last_closed["ema_spread"]), f"{last_closed['ema_spread_pct']:+.3f}%")
    m3.metric("RSI (14)", f"{last_closed['rsi_14']:.1f}")
    mm = int(last_closed["pvt_mismatch"])
    m4.metric("PVT mismatch", {1: "Bullish", -1: "Bearish", 0: "None"}[mm])
    m5.metric("Candle closes in", f"{mins}:{secs:02d}")

    # ---- chart + signal/Kalshi panels ----
    view = df.tail(cfg["n_candles"]).copy()
    view.index = view.index.tz_convert(cfg["tz"]).tz_localize(None)  # wall-clock times in chosen tz
    chart_col, side_col = st.columns([3, 2], gap="medium")
    with chart_col:
        st.plotly_chart(build_chart(view, cfg["rsi_ob"], cfg["rsi_os"], cfg["tz"]), width="stretch")
        if not bool(last["is_closed"]):
            st.caption("The rightmost candle is still forming; signals use the last closed candle.")
    with side_col:
        render_signal(analysis)
        st.divider()
        render_kalshi(markets, kalshi_err, analysis, spot, cfg["tz"])

    # ---- data table + export ----
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
    with st.expander("How the signals are calculated"):
        st.markdown(
            "- **UP:** EMA9 > EMA21 and RSI < overbought limit. **DOWN:** EMA9 < EMA21 and RSI > oversold limit. "
            "Otherwise no signal. Computed on the last *closed* candle only.\n"
            "- **PVT** (Price-Volume Trend) = running total of `volume × % price change`; "
            "its signal line is a 9-bar EMA.\n"
            "- **Buyer/seller mismatch:** over the lookback window, price and PVT move in opposite "
            "directions. Price down + PVT up = bullish; price up + PVT down = bearish.\n"
            "- **Wicks:** upper = high − max(open, close); lower = min(open, close) − low. "
            "A wick covering at least 50% of the candle's range is flagged as long.")
    st.caption("Educational tool, not financial advice. Read-only: this app uses public data "
               "and cannot place trades. Signals are simple indicator rules with no proven edge; "
               "prediction-market contracts can lose their full price.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="BTC 15m Signal Dashboard", page_icon="₿", layout="wide")
    st.title("₿ Bitcoin 15-Minute Signal Dashboard")
    st.caption("Live BTC candles + Kalshi KXBTC15M contracts · public data, no login")

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh market data"):
            fetch_candles.clear()
            fetch_kalshi_markets.clear()
        auto = st.toggle("Auto-refresh", value=True)
        interval = st.slider("Refresh every (seconds)", 15, 120, 30, step=15, disabled=not auto)
        st.divider()
        n_candles = st.slider("Candles shown", 60, 300, 120, step=10)
        tz = st.selectbox("Display timezone", TIMEZONES, index=0)
        st.divider()
        rsi_ob = st.number_input("RSI overbought", 50.0, 95.0, 70.0, step=1.0)
        rsi_os = st.number_input("RSI oversold", 5.0, 50.0, 30.0, step=1.0)
        lookback = st.slider("PVT mismatch lookback (candles)", 3, 15, 5)

    cfg = dict(auto=auto, n_candles=n_candles, tz=tz, rsi_ob=rsi_ob, rsi_os=rsi_os, lookback=lookback)

    @st.fragment(run_every=interval if auto else None)
    def dashboard() -> None:
        render_dashboard(cfg)

    dashboard()
    render_footer()


if __name__ == "__main__":
    main()
