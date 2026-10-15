import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from google import genai
from google.genai import types

# ==========================================
# 1. MOBILE APP PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Trade Assistant",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

if "last_alert_key" not in st.session_state:
    st.session_state.last_alert_key = None
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}


# ==========================================
# 2. PUSH NOTIFICATION ENGINE (NTFY & TELEGRAM)
# ==========================================
def send_phone_notification(title, message, ntfy_topic="", tg_token="", tg_chat_id="") -> bool:
    """Send push alerts. Returns True if at least one channel succeeded."""
    sent = False

    # ntfy.sh -- JSON publish, so emojis in the title are safe
    # (HTTP headers must be latin-1, which broke the original header-based call)
    if ntfy_topic.strip():
        try:
            r = requests.post(
                "https://ntfy.sh",
                json={
                    "topic": ntfy_topic.strip(),
                    "title": title,
                    "message": message,
                    "priority": 4,
                    "tags": ["chart_with_upwards_trend", "warning"],
                },
                timeout=8,
            )
            r.raise_for_status()
            sent = True
        except Exception as e:
            st.warning(f"ntfy push failed: {e}")

    # Telegram (plain text: Markdown mode breaks on characters like _ or *)
    if tg_token.strip() and tg_chat_id.strip():
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{tg_token.strip()}/sendMessage",
                json={"chat_id": tg_chat_id.strip(), "text": f"{title}\n\n{message}"},
                timeout=8,
            )
            r.raise_for_status()
            sent = True
        except Exception as e:
            st.warning(f"Telegram push failed: {e}")

    return sent


# ==========================================
# 3. LIVE MARKET & INDICATOR ENGINE
# ==========================================
@st.cache_data(ttl=30, show_spinner=False)
def fetch_and_compute_indicators(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No live data found for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close", "High", "Low"]).copy()

    # Trend EMAs
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # VWAP -- resets every trading day.
    # Indices like ^NSEI report 0 volume, which made the old VWAP = 0 (always "bullish").
    # In that case fall back to a running average of typical price.
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0) if "Volume" in df.columns else pd.Series(0, index=df.index)
    day = df.index.date
    if vol.sum() > 0:
        pv = (typical * vol).groupby(day).cumsum()
        cv = vol.groupby(day).cumsum().replace(0, np.nan)
        df["VWAP"] = (pv / cv).fillna(typical)
    else:
        df["VWAP"] = typical.groupby(day).transform(lambda s: s.expanding().mean())

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    df = df.dropna()
    if len(df) < 4:
        raise ValueError("Not enough candles yet to compute indicators.")
    return df


def evaluate_signal(latest: pd.Series, prev: pd.Series) -> dict:
    score = 0
    reasons = []

    if latest["EMA_9"] > latest["EMA_21"]:
        score += 1
        reasons.append("EMA 9 > EMA 21 (Bullish Trend)")
    else:
        score -= 1
        reasons.append("EMA 9 < EMA 21 (Bearish Trend)")

    if latest["Close"] > latest["EMA_50"]:
        score += 1
        reasons.append("Price > EMA 50")
    else:
        score -= 1
        reasons.append("Price < EMA 50")

    if latest["Close"] > latest["VWAP"]:
        score += 1
        reasons.append("Above VWAP (Buyer Control)")
    else:
        score -= 1
        reasons.append("Below VWAP (Seller Control)")

    if latest["MACD"] > latest["MACD_Signal"] and latest["MACD_Hist"] > prev["MACD_Hist"]:
        score += 1
        reasons.append("MACD Bullish Momentum")
    elif latest["MACD"] < latest["MACD_Signal"] and latest["MACD_Hist"] < prev["MACD_Hist"]:
        score -= 1
        reasons.append("MACD Bearish Momentum")

    rsi = float(latest["RSI"])
    if 55 <= rsi <= 70:
        score += 1
        reasons.append(f"RSI Strong Bullish ({rsi:.1f})")
    elif 30 <= rsi <= 45:
        score -= 1
        reasons.append(f"RSI Strong Bearish ({rsi:.1f})")

    close = float(latest["Close"])
    atr = float(latest["ATR"])

    stop_loss = target = None
    if score >= 3:
        direction = "UP (BUY / CALL)"
        stop_loss = close - 1.5 * atr
        target = close + 3.0 * atr
    elif score <= -3:
        direction = "DOWN (SELL / PUT)"
        stop_loss = close + 1.5 * atr
        target = close - 3.0 * atr
    else:
        direction = "WAIT / NO TRADE (CHOPPY)"

    return {
        "score": score,
        "direction": direction,
        "close": close,
        "atr": atr,
        "stop_loss": stop_loss,
        "target": target,
        "reasons": reasons,
    }


def ask_gemini_analyst(symbol, interval, latest, sig, api_key) -> str:
    if not api_key:
        return "Add your Gemini API Key in the settings drawer to enable AI commentary."

    levels = (
        f"Entry: {sig['close']:.2f} | Stop-Loss: {sig['stop_loss']:.2f} | Target: {sig['target']:.2f}"
        if sig["stop_loss"] is not None
        else "No trade levels (signal is neutral)."
    )
    prompt = f"""
    Asset: {symbol} ({interval})
    Price: {sig['close']:.2f} | Signal: {sig['direction']} (Score: {sig['score']}/5)
    {levels}
    RSI: {latest['RSI']:.1f} | VWAP: {latest['VWAP']:.2f} | EMA9: {latest['EMA_9']:.2f} | EMA21: {latest['EMA_21']:.2f}
    Triggers: {', '.join(sig['reasons'])}

    Write a 4-bullet mobile trade alert explaining:
    1. Market Bias (Up, Down, or Sideways)
    2. Why this signal triggered
    3. Exact invalidation / Stop-Loss rule
    4. Key risk warning
    """
    try:
        client = genai.Client(api_key=api_key)
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return res.text or "Gemini returned an empty response."
    except Exception as e:
        return f"AI commentary unavailable: {e}"


# ==========================================
# 4. MOBILE USER INTERFACE
# ==========================================
st.title("📱 AI Live Trading Assistant")

with st.expander("⚙️ Market & Push Notification Settings", expanded=False):
    symbol = st.text_input("Ticker Symbol (e.g., ^NSEI, RELIANCE.NS, BTC-USD, AAPL)", value="^NSEI").strip()
    interval = st.selectbox("Candle Timeframe", ["1m", "5m", "15m", "1h"], index=1)
    ntfy_topic = st.text_input(
        "ntfy.sh Topic Name (For Instant Phone Alerts)",
        value="",
        help="Install the free 'ntfy' app and subscribe to this topic. Topics are public, "
             "so use a long random name that nobody can guess.",
    )
    gemini_key = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    tg_token = st.text_input("Telegram Bot Token (Optional)", type="password")
    tg_chat_id = st.text_input("Telegram Chat ID (Optional)")
    closed_only = st.checkbox(
        "Use only closed candles for signals",
        value=True,
        help="Avoids signals that flip while the current candle is still forming.",
    )
    auto_scan = st.checkbox("Auto-Scan Live Market (Every 60 Seconds)", value=True)

if st.button("🔄 Scan Live Market Now", use_container_width=True):
    fetch_and_compute_indicators.clear()


# Fragment: refreshes itself every 60s without blocking the page
# (replaces time.sleep(60) + st.rerun(), which froze the UI). Needs Streamlit >= 1.37.
@st.fragment(run_every=60 if auto_scan else None)
def scanner():
    try:
        df = fetch_and_compute_indicators(symbol, "5d", interval)
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return

    if closed_only:
        latest, prev = df.iloc[-2], df.iloc[-3]
    else:
        latest, prev = df.iloc[-1], df.iloc[-2]

    candle_time = str(latest.name)
    live_price = float(df["Close"].iloc[-1])
    sig = evaluate_signal(latest, prev)

    # Signal card
    if sig["score"] >= 3:
        st.success(f"🟢 **TAKE TRADE: {sig['direction']}**  (Confluence Score: +{sig['score']}/5)")
    elif sig["score"] <= -3:
        st.error(f"🔴 **TAKE TRADE: {sig['direction']}**  (Confluence Score: {sig['score']}/5)")
    else:
        st.warning(f"🟡 **{sig['direction']}**  (Confluence Score: {sig['score']}/5)")

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Price", f"{live_price:.2f}")
    col2.metric("Stop-Loss (1.5x ATR)", f"{sig['stop_loss']:.2f}" if sig["stop_loss"] is not None else "—")
    col3.metric("Target (1:2 R:R)", f"{sig['target']:.2f}" if sig["target"] is not None else "—")

    st.caption(
        f"Signal candle: {candle_time} | RSI: {latest['RSI']:.1f} | VWAP: {latest['VWAP']:.2f}"
        + ("" if df["Volume"].sum() > 0 else " (no volume data: VWAP = avg price)")
    )
    st.write("**Active Signal Factors:**")
    for r in sig["reasons"]:
        st.write(f"- {r}")

    st.line_chart(df[["Close", "EMA_9", "EMA_21", "VWAP"]].tail(60))

    # Push notification, once per symbol/timeframe/candle
    alert_key = f"{symbol}|{interval}|{candle_time}"
    if abs(sig["score"]) >= 3 and st.session_state.last_alert_key != alert_key:
        alert_title = f"🚨 {symbol} ({interval}): {sig['direction']}"
        alert_body = (
            f"Signal: {sig['direction']} (Score: {sig['score']}/5)\n"
            f"Entry Price: {sig['close']:.2f}\n"
            f"Stop-Loss: {sig['stop_loss']:.2f}\n"
            f"Target: {sig['target']:.2f}\n"
            f"Reasons: {', '.join(sig['reasons'])}"
        )
        if send_phone_notification(alert_title, alert_body, ntfy_topic, tg_token, tg_chat_id):
            st.session_state.last_alert_key = alert_key
            st.toast("📲 Push notification sent to your phone!")

    # AI briefing: one Gemini call per candle, not one per refresh
    st.subheader("🤖 AI Trading Assistant Plan")
    if alert_key not in st.session_state.ai_cache:
        st.session_state.ai_cache = {
            alert_key: ask_gemini_analyst(symbol, interval, latest, sig, gemini_key)
        }
    st.info(st.session_state.ai_cache[alert_key])


scanner()
