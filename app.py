"""
Quant Analyzer Pro - Single File Edition
Single-page Streamlit app: analisi tecnica + fondamentale con risoluzione ISIN/Ticker.

Tutto in un singolo file per semplicita' di deploy su Streamlit Cloud.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


# =====================================================================
# CONFIG STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Quant Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, label {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 14px !important;
}
.stApp { background-color: #000000; color: #e6edf3; }

h1 { font-size: 28px !important; color: #ffffff !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-size: 20px !important; color: #ffffff !important; font-weight: 600 !important; }
h3 { font-size: 16px !important; color: #e6edf3 !important; font-weight: 600 !important; }

.metric-card {
    background: #0a0a0a;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 8px;
}
.metric-label { font-size: 12px; color: #7aa8c8; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 22px; color: #ffffff; font-weight: 600; margin-top: 4px; }
.metric-status-ok { color: #22c55e; }
.metric-status-warn { color: #f59e0b; }
.metric-status-bad { color: #ef4444; }
.metric-status-neutral { color: #38bdf8; }

.stButton > button {
    background-color: #0099ff;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}
.stButton > button:hover { background-color: #0077cc; }

div[data-testid="stTextInput"] input {
    background-color: #0a0a0a;
    color: #ffffff;
    border: 1px solid #1f2937;
    font-size: 14px !important;
}

.divider { border-top: 1px solid #1f2937; margin: 24px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# RESOLVER ISIN / TICKER
# =====================================================================
ISIN_REGEX = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,10}(\.[A-Z]{1,4})?$")


@dataclass
class ResolvedSymbol:
    input_raw: str
    input_type: str
    ticker: str
    isin: Optional[str]
    long_name: Optional[str]
    error: Optional[str] = None


def classify_input(raw: str) -> str:
    s = raw.strip().upper()
    if ISIN_REGEX.match(s): return "ISIN"
    if TICKER_REGEX.match(s): return "TICKER"
    return "INVALID"


def resolve_isin_to_ticker(isin: str) -> Optional[str]:
    try:
        search = yf.Search(isin, max_results=5)
        quotes = search.quotes or []
        if not quotes:
            return None
        for q in quotes:
            if q.get("quoteType", "").upper() in ("EQUITY", "ETF"):
                sym = q.get("symbol")
                if sym:
                    return sym
        return quotes[0].get("symbol")
    except Exception:
        return None


def probe_ticker(ticker: str) -> tuple[bool, Optional[str], Optional[str]]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        has_price = (info.get("regularMarketPrice") is not None
                     or info.get("currentPrice") is not None
                     or info.get("symbol"))
        if not info or not has_price:
            try:
                fi = t.fast_info
                if fi is None or fi.get("last_price") is None:
                    return False, None, None
            except Exception:
                return False, None, None
        return True, info.get("longName") or info.get("shortName"), info.get("isin")
    except Exception:
        return False, None, None


def resolve_symbol(raw_input: str) -> ResolvedSymbol:
    if not raw_input or not raw_input.strip():
        return ResolvedSymbol(raw_input, "INVALID", "", None, None, "Input vuoto.")

    raw = raw_input.strip().upper()
    kind = classify_input(raw)

    if kind == "INVALID":
        return ResolvedSymbol(raw, "INVALID", "", None, None,
                              f"Formato non riconosciuto: '{raw}'. Attesi ISIN (12 caratteri) o Ticker (es. AAPL, ENI.MI).")

    if kind == "ISIN":
        ticker = resolve_isin_to_ticker(raw)
        if ticker is None:
            return ResolvedSymbol(raw, "ISIN", "", raw, None,
                                  f"ISIN '{raw}' non risolvibile a ticker tramite Yahoo Finance.")
    else:
        ticker = raw

    exists, long_name, isin_found = probe_ticker(ticker)
    if not exists:
        return ResolvedSymbol(raw, kind, ticker, raw if kind == "ISIN" else None, None,
                              f"Ticker '{ticker}' non trovato su Yahoo Finance.")

    return ResolvedSymbol(raw, kind, ticker,
                          isin_found or (raw if kind == "ISIN" else None),
                          long_name, None)


# =====================================================================
# INDICATORI TECNICI - Wilder puro
# =====================================================================
@dataclass
class TechnicalIndicators:
    rsi_value: Optional[float]
    rsi_status: str
    atr_value: Optional[float]
    atr_pct: Optional[float]
    atr_pct_status: str
    close_price: Optional[float]
    history_df: Optional[pd.DataFrame]
    error: Optional[str] = None


def wilder_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def compute_indicators(ticker: str) -> TechnicalIndicators:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            raise ValueError(f"Nessun dato storico per {ticker}.")
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        df = df.dropna(subset=["close"])
        if len(df) < 20:
            raise ValueError(f"Serie troppo corta ({len(df)} barre).")
    except Exception as e:
        return TechnicalIndicators(None, "N/A", None, None, "N/A", None, None,
                                   f"Errore dati storici: {e}")

    try:
        rsi = wilder_rsi(df["close"], 14)
        atr = wilder_atr(df["high"], df["low"], df["close"], 14)

        rsi_last = float(rsi.iloc[-1]) if not rsi.empty else None
        atr_last = float(atr.iloc[-1]) if not atr.empty else None
        close_last = float(df["close"].iloc[-1])

        if rsi_last is not None and np.isnan(rsi_last): rsi_last = None
        if atr_last is not None and np.isnan(atr_last): atr_last = None

        atr_pct = (atr_last / close_last * 100.0) if (atr_last and close_last > 0) else None

        def cls_rsi(v):
            if v >= 70: return "Overbought"
            if v <= 30: return "Oversold"
            return "Neutral"

        def cls_atr(v):
            if v >= 5.0: return "Alta"
            if v >= 2.0: return "Media"
            return "Bassa"

        return TechnicalIndicators(
            rsi_value=round(rsi_last, 2) if rsi_last is not None else None,
            rsi_status=cls_rsi(rsi_last) if rsi_last is not None else "N/A",
            atr_value=round(atr_last, 4) if atr_last is not None else None,
            atr_pct=round(atr_pct, 2) if atr_pct is not None else None,
            atr_pct_status=cls_atr(atr_pct) if atr_pct is not None else "N/A",
            close_price=round(close_last, 4),
            history_df=df,
            error=None,
        )
    except Exception as e:
        return TechnicalIndicators(None, "N/A", None, None, "N/A", None, df,
                                   f"Errore calcolo indicatori: {e}")


# =====================================================================
# ANALYST DATA
# =====================================================================
@dataclass
class AnalystData:
    recent_actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    target_low: Optional[float] = None
    target_high: Optional[float] = None
    current_price: Optional[float] = None
    num_analysts: Optional[int] = None
    recommendation_key: Optional[str] = None
    error: Optional[str] = None


def _safe_get(d, *keys, default=None):
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v is not None:
            return v
    return default


def fetch_analyst_data(ticker: str) -> AnalystData:
    out = AnalystData()
    try:
        t = yf.Ticker(ticker)
    except Exception as e:
        out.error = f"Impossibile istanziare Ticker: {e}"
        return out

    # Consensus
    try:
        apt = t.analyst_price_targets
        if isinstance(apt, dict) and apt:
            out.target_mean = _safe_get(apt, "mean")
            out.target_median = _safe_get(apt, "median")
            out.target_low = _safe_get(apt, "low")
            out.target_high = _safe_get(apt, "high")
            out.current_price = _safe_get(apt, "current")
    except Exception:
        pass

    # Fallback su info
    try:
        info = t.info or {}
        out.target_mean = out.target_mean or info.get("targetMeanPrice")
        out.target_median = out.target_median or info.get("targetMedianPrice")
        out.target_low = out.target_low or info.get("targetLowPrice")
        out.target_high = out.target_high or info.get("targetHighPrice")
        out.current_price = out.current_price or info.get("currentPrice") or info.get("regularMarketPrice")
        out.num_analysts = info.get("numberOfAnalystOpinions")
        out.recommendation_key = info.get("recommendationKey")
    except Exception:
        pass

    # Ultimi 7 record
    try:
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            df = ud.copy()
            if df.index.name and "grade" in str(df.index.name).lower():
                df = df.reset_index().rename(columns={df.index.name: "Date"})
            elif "GradeDate" in df.columns:
                df = df.rename(columns={"GradeDate": "Date"})
            else:
                df = df.reset_index().rename(columns={df.columns[0]: "Date"})

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date", ascending=False).head(7)

            out.recent_actions = pd.DataFrame({
                "Agenzia": df.get("Firm", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Azione": df.get("Action", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Rating": df.get("ToGrade", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Da": df.get("FromGrade", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Data": df["Date"].dt.strftime("%Y-%m-%d").fillna("N/A"),
                "Target": ["N/A"] * len(df),
            }).reset_index(drop=True)
    except Exception as e:
        if not out.error:
            out.error = f"Upgrades/downgrades non disponibili: {e}"

    return out


def upside_pct(target, current):
    if target is None or current is None or current == 0:
        return None
    try:
        return round((target - current) / current * 100.0, 2)
    except Exception:
        return None


# =====================================================================
# PIOTROSKI F-SCORE
# =====================================================================
@dataclass
class PiotroskiResult:
    score: Optional[int]
    details: dict = field(default_factory=dict)
    raw_values: dict = field(default_factory=dict)
    interpretation: str = "N/A"
    error: Optional[str] = None


_ALIASES = {
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income From Continuing Operation Net Minority Interest"],
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "gross_profit": ["Gross Profit"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "shares_outstanding": ["Share Issued", "Ordinary Shares Number", "Common Stock Equity"],
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
            "Total Cash From Operating Activities"],
}


def _get_row(df, keys):
    if df is None or df.empty:
        return None
    for k in keys:
        if k in df.index:
            return df.loc[k]
    lower_idx = {str(i).lower(): i for i in df.index}
    for k in keys:
        if k.lower() in lower_idx:
            return df.loc[lower_idx[k.lower()]]
    return None


def _val(series, col_idx):
    if series is None:
        return None
    try:
        v = series.iloc[col_idx]
        if pd.isna(v):
            return None
        return float(v)
    except (IndexError, ValueError, TypeError):
        return None


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _interpret(score):
    if score >= 8: return "Forte"
    if score >= 5: return "Medio"
    if score >= 3: return "Discreto"
    return "Debole"


def compute_piotroski(ticker: str) -> PiotroskiResult:
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet
        is_ = t.financials
        cf = t.cashflow
    except Exception as e:
        return PiotroskiResult(None, error=f"Errore fetch bilanci: {e}")

    if bs is None or bs.empty or is_ is None or is_.empty or cf is None or cf.empty:
        return PiotroskiResult(None, error="Bilanci/Income/Cash Flow non disponibili da yfinance.")

    if bs.shape[1] < 2 or is_.shape[1] < 2 or cf.shape[1] < 2:
        return PiotroskiResult(None,
            error=f"Necessari >=2 esercizi annuali. Disponibili: BS={bs.shape[1]}, IS={is_.shape[1]}, CF={cf.shape[1]}.")

    ni = _get_row(is_, _ALIASES["net_income"])
    rev = _get_row(is_, _ALIASES["revenue"])
    gp = _get_row(is_, _ALIASES["gross_profit"])
    ta_row = _get_row(bs, _ALIASES["total_assets"])
    ca_row = _get_row(bs, _ALIASES["current_assets"])
    cl_row = _get_row(bs, _ALIASES["current_liabilities"])
    ltd_row = _get_row(bs, _ALIASES["long_term_debt"])
    shares_row = _get_row(bs, _ALIASES["shares_outstanding"])
    cfo_row = _get_row(cf, _ALIASES["cfo"])

    NI_t = _val(ni, 0); NI_tm1 = _val(ni, 1)
    REV_t = _val(rev, 0); REV_tm1 = _val(rev, 1)
    GP_t = _val(gp, 0); GP_tm1 = _val(gp, 1)
    TA_t = _val(ta_row, 0); TA_tm1 = _val(ta_row, 1); TA_tm2 = _val(ta_row, 2)
    CA_t = _val(ca_row, 0); CA_tm1 = _val(ca_row, 1)
    CL_t = _val(cl_row, 0); CL_tm1 = _val(cl_row, 1)
    LTD_t = _val(ltd_row, 0); LTD_tm1 = _val(ltd_row, 1)
    SHARES_t = _val(shares_row, 0); SHARES_tm1 = _val(shares_row, 1)
    CFO_t = _val(cfo_row, 0)

    # Ricostruzione GP se mancante
    if GP_t is None or GP_tm1 is None:
        cogs = _get_row(is_, _ALIASES["cogs"])
        cogs_t = _val(cogs, 0); cogs_tm1 = _val(cogs, 1)
        if GP_t is None and REV_t is not None and cogs_t is not None:
            GP_t = REV_t - cogs_t
        if GP_tm1 is None and REV_tm1 is not None and cogs_tm1 is not None:
            GP_tm1 = REV_tm1 - cogs_tm1

    details = {}
    raw = {}

    roa_t = _safe_div(NI_t, TA_tm1)
    roa_tm1 = _safe_div(NI_tm1, TA_tm2) if TA_tm2 is not None else _safe_div(NI_tm1, TA_tm1)
    details["1_ROA_positivo"] = 1 if (roa_t and roa_t > 0) else 0
    raw["ROA_t"] = roa_t

    details["2_CFO_positivo"] = 1 if (CFO_t and CFO_t > 0) else 0
    raw["CFO_t"] = CFO_t

    details["3_DeltaROA_positivo"] = 1 if (roa_t is not None and roa_tm1 is not None and roa_t > roa_tm1) else 0
    raw["ROA_tm1"] = roa_tm1

    details["4_Accruals_CFO_gt_NI"] = 1 if (CFO_t is not None and NI_t is not None and CFO_t > NI_t) else 0

    lev_t = _safe_div(LTD_t, TA_t)
    lev_tm1 = _safe_div(LTD_tm1, TA_tm1)
    if lev_t is not None and lev_tm1 is not None:
        details["5_DeltaLeverage_negativo"] = 1 if lev_t < lev_tm1 else 0
    else:
        details["5_DeltaLeverage_negativo"] = 1 if (LTD_t is None and LTD_tm1 is None) else 0
    raw["Leverage_t"] = lev_t; raw["Leverage_tm1"] = lev_tm1

    cr_t = _safe_div(CA_t, CL_t)
    cr_tm1 = _safe_div(CA_tm1, CL_tm1)
    details["6_DeltaCurrentRatio_positivo"] = 1 if (cr_t is not None and cr_tm1 is not None and cr_t > cr_tm1) else 0
    raw["CurrentRatio_t"] = cr_t; raw["CurrentRatio_tm1"] = cr_tm1

    if SHARES_t is not None and SHARES_tm1 is not None:
        details["7_No_emissione_azioni"] = 1 if SHARES_t <= SHARES_tm1 * 1.005 else 0
    else:
        details["7_No_emissione_azioni"] = 0
    raw["Shares_t"] = SHARES_t; raw["Shares_tm1"] = SHARES_tm1

    gm_t = _safe_div(GP_t, REV_t)
    gm_tm1 = _safe_div(GP_tm1, REV_tm1)
    details["8_DeltaGrossMargin_positivo"] = 1 if (gm_t is not None and gm_tm1 is not None and gm_t > gm_tm1) else 0
    raw["GrossMargin_t"] = gm_t; raw["GrossMargin_tm1"] = gm_tm1

    at_t = _safe_div(REV_t, TA_tm1)
    at_tm1 = _safe_div(REV_tm1, TA_tm2) if TA_tm2 is not None else _safe_div(REV_tm1, TA_tm1)
    details["9_DeltaAssetTurnover_positivo"] = 1 if (at_t is not None and at_tm1 is not None and at_t > at_tm1) else 0
    raw["AssetTurnover_t"] = at_t; raw["AssetTurnover_tm1"] = at_tm1

    score = int(sum(v for v in details.values() if v in (0, 1)))
    return PiotroskiResult(score, details, raw, _interpret(score), None)


# =====================================================================
# TRADINGVIEW
# =====================================================================
_EXCHANGE_MAP = {
    ".MI": "MIL", ".DE": "XETR", ".PA": "EURONEXT", ".AS": "EURONEXT",
    ".BR": "EURONEXT", ".L": "LSE", ".MC": "BME", ".SW": "SIX",
    ".VI": "WBAG", ".HE": "OMXHEX", ".ST": "OMXSTO", ".CO": "OMXCOP",
    ".OL": "OSL", ".TO": "TSX", ".V": "TSXV", ".HK": "HKEX",
    ".T": "TSE", ".AX": "ASX",
}


def map_to_tradingview(yf_ticker: str) -> str:
    t = yf_ticker.strip().upper()
    if "." in t:
        sym, _, suf = t.partition(".")
        exch = _EXCHANGE_MAP.get(f".{suf}", "")
        return f"{exch}:{sym}" if exch else t
    return f"NASDAQ:{t}"


def tradingview_html(tv_symbol: str, height: int = 520) -> str:
    return f"""
<div class="tradingview-widget-container" style="height:{height}px;width:100%;">
  <div id="tv_chart_container" style="height:{height}px;width:100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Europe/Rome",
        "theme": "dark",
        "style": "1",
        "locale": "it",
        "toolbar_bg": "#000000",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
        "container_id": "tv_chart_container"
    }});
  </script>
</div>
"""


# =====================================================================
# SESSION STATE
# =====================================================================
if "saved" not in st.session_state:
    st.session_state["saved"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "input_buffer" not in st.session_state:
    st.session_state["input_buffer"] = ""


# =====================================================================
# PIPELINE
# =====================================================================
@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline(raw_input: str) -> dict:
    resolved = resolve_symbol(raw_input)
    out = {"resolved": resolved, "indicators": None, "analyst": None, "piotroski": None,
           "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if resolved.error or not resolved.ticker:
        return out
    out["indicators"] = compute_indicators(resolved.ticker)
    out["analyst"] = fetch_analyst_data(resolved.ticker)
    out["piotroski"] = compute_piotroski(resolved.ticker)
    return out


# =====================================================================
# UI HELPERS
# =====================================================================
def render_metric(label, value, status_class="metric-status-neutral", subtitle=""):
    sub = f'<div style="font-size:11px;color:#7aa8c8;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {status_class}">{value}</div>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def cls_rsi(s): return {"Overbought": "metric-status-bad", "Oversold": "metric-status-warn",
                        "Neutral": "metric-status-neutral"}.get(s, "metric-status-neutral")
def cls_atr(s): return {"Alta": "metric-status-bad", "Media": "metric-status-warn",
                        "Bassa": "metric-status-ok"}.get(s, "metric-status-neutral")
def cls_pio(s):
    if s is None: return "metric-status-neutral"
    if s >= 8: return "metric-status-ok"
    if s >= 5: return "metric-status-neutral"
    if s >= 3: return "metric-status-warn"
    return "metric-status-bad"


def validate_input(raw):
    if not raw or not raw.strip():
        return False, "Inserire un ticker o un ISIN."
    s = raw.strip().upper()
    if ISIN_REGEX.match(s) or TICKER_REGEX.match(s):
        return True, ""
    return False, f"Formato non valido: '{s}'. Atteso ISIN o Ticker (es. AAPL, ENI.MI)."


# =====================================================================
# HEADER
# =====================================================================
st.markdown("# 📊 Quant Analyzer Pro")
st.markdown(
    "<div style='color:#7aa8c8;font-size:13px;margin-bottom:16px;'>"
    "Analisi tecnica + fondamentale con risoluzione automatica ISIN→Ticker · TradingView · yfinance"
    "</div>",
    unsafe_allow_html=True,
)

# Input bar
col_input, col_b1, col_b2, col_b3 = st.columns([4, 1.2, 1.2, 1])
with col_input:
    user_input = st.text_input(
        "Ticker o ISIN",
        value=st.session_state["input_buffer"],
        placeholder="Es. AAPL, ENI.MI, IT0003132476, US0378331005",
        label_visibility="collapsed",
        key="ti_input",
    )
with col_b1:
    generate = st.button("🔍 Genera Risultato", use_container_width=True)
with col_b2:
    save = st.button("💾 Salva Risultato", use_container_width=True)
with col_b3:
    reset = st.button("♻️ Reset", use_container_width=True)


# =====================================================================
# AZIONI
# =====================================================================
if reset:
    st.cache_data.clear()
    for k in ["saved", "last_result", "input_buffer"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

if generate:
    valid, msg = validate_input(user_input)
    if not valid:
        st.error(f"❌ {msg}")
    else:
        st.session_state["input_buffer"] = user_input.strip().upper()
        with st.spinner("Esecuzione pipeline analitica…"):
            try:
                st.session_state["last_result"] = run_pipeline(user_input.strip().upper())
            except Exception as e:
                st.error(f"❌ Errore pipeline: {type(e).__name__}: {e}")
                st.session_state["last_result"] = None


# =====================================================================
# RENDER RESULT
# =====================================================================
result = st.session_state.get("last_result")

if result is not None:
    resolved = result["resolved"]

    if resolved.error:
        st.error(f"❌ Risoluzione fallita: {resolved.error}")
    else:
        name = resolved.long_name or resolved.ticker
        st.markdown(
            f"### {name}  "
            f"<span style='color:#7aa8c8;font-size:14px;'>"
            f"Ticker: <code>{resolved.ticker}</code> · "
            f"ISIN: <code>{resolved.isin or 'N/A'}</code> · "
            f"Input: {resolved.input_type}</span>",
            unsafe_allow_html=True,
        )

        ind = result["indicators"]
        an = result["analyst"]
        pio = result["piotroski"]

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_metric("Prezzo (Close)", f"{ind.close_price}" if ind and ind.close_price else "N/A")
        with c2:
            if ind and ind.rsi_value is not None:
                render_metric("RSI(14)", f"{ind.rsi_value}", cls_rsi(ind.rsi_status), ind.rsi_status)
            else:
                render_metric("RSI(14)", "N/A")
        with c3:
            if ind and ind.atr_pct is not None:
                render_metric("ATR%(14)", f"{ind.atr_pct}%", cls_atr(ind.atr_pct_status),
                              f"Volatilità {ind.atr_pct_status}")
            else:
                render_metric("ATR%(14)", "N/A")
        with c4:
            if pio and pio.score is not None:
                render_metric("Piotroski F-Score", f"{pio.score}/9", cls_pio(pio.score), pio.interpretation)
            else:
                render_metric("Piotroski F-Score", "N/A", "metric-status-neutral",
                              (pio.error or "")[:40] if pio else "")
        with c5:
            if an and an.target_mean is not None and an.current_price is not None:
                up = upside_pct(an.target_mean, an.current_price)
                clz = "metric-status-ok" if (up and up > 0) else "metric-status-bad"
                render_metric("Target Mean", f"{round(an.target_mean,2)}", clz,
                              f"Upside: {up}%" if up is not None else "")
            else:
                render_metric("Target Mean", "N/A")

        # TradingView
        st.markdown("## 📈 Grafico Real-time (TradingView)")
        tv_sym = map_to_tradingview(resolved.ticker)
        st.caption(f"Simbolo TradingView: `{tv_sym}`")
        components.html(tradingview_html(tv_sym, 540), height=560)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Rating & Target", "📒 Piotroski F-Score", "📊 Storico Indicatori"])

        with tab1:
            if an is None:
                st.warning("⚠️ Dati analisti non disponibili.")
            else:
                cA, cB, cC, cD = st.columns(4)
                cA.markdown(f"**Target Low**  \n{round(an.target_low,2) if an.target_low else 'N/A'}")
                cB.markdown(f"**Target Mean**  \n{round(an.target_mean,2) if an.target_mean else 'N/A'}")
                cC.markdown(f"**Target High**  \n{round(an.target_high,2) if an.target_high else 'N/A'}")
                cD.markdown(f"**N. Analisti**  \n{an.num_analysts or 'N/A'}")
                if an.recommendation_key:
                    st.markdown(f"**Consensus:** `{an.recommendation_key.upper()}`")
                st.markdown("#### Ultime 7 azioni analisti")
                if an.recent_actions is not None and not an.recent_actions.empty:
                    st.dataframe(an.recent_actions, use_container_width=True, hide_index=True)
                else:
                    st.info("Storico upgrades/downgrades non disponibile.")

        with tab2:
            if pio is None or pio.score is None:
                st.warning(f"⚠️ Piotroski non calcolabile. {pio.error if pio else ''}")
            else:
                st.markdown(f"### F-Score: **{pio.score} / 9** — {pio.interpretation}")
                labels = {
                    "1_ROA_positivo": "1. ROA > 0 (Redditività)",
                    "2_CFO_positivo": "2. CFO > 0 (Redditività)",
                    "3_DeltaROA_positivo": "3. ΔROA > 0 (Redditività)",
                    "4_Accruals_CFO_gt_NI": "4. CFO > Net Income (Qualità utili)",
                    "5_DeltaLeverage_negativo": "5. ΔLeverage < 0 (Leva)",
                    "6_DeltaCurrentRatio_positivo": "6. ΔCurrent Ratio > 0 (Liquidità)",
                    "7_No_emissione_azioni": "7. No emissione azioni (Equity)",
                    "8_DeltaGrossMargin_positivo": "8. ΔGross Margin > 0 (Efficienza)",
                    "9_DeltaAssetTurnover_positivo": "9. ΔAsset Turnover > 0 (Efficienza)",
                }
                rows = [{"Criterio": lab, "Pass": "✅" if pio.details.get(k, 0) == 1 else "❌",
                         "Score": pio.details.get(k, 0)} for k, lab in labels.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                with st.expander("Valori grezzi utilizzati nel calcolo"):
                    raw_df = pd.DataFrame([
                        {"Metrica": k, "Valore": (round(v, 6) if isinstance(v, float) else v)}
                        for k, v in pio.raw_values.items() if v is not None
                    ])
                    st.dataframe(raw_df, use_container_width=True, hide_index=True)

        with tab3:
            if ind and ind.history_df is not None:
                df = ind.history_df.tail(60)[["open", "high", "low", "close", "volume"]].copy()
                df.index = df.index.strftime("%Y-%m-%d")
                st.dataframe(df.iloc[::-1], use_container_width=True)
            else:
                st.warning(f"⚠️ Storico non disponibile. {ind.error if ind else ''}")


# =====================================================================
# SAVE
# =====================================================================
if save:
    if st.session_state.get("last_result") is None:
        st.warning("⚠️ Nessun risultato da salvare. Esegui prima 'Genera Risultato'.")
    else:
        res = st.session_state["last_result"]
        resolved = res["resolved"]
        if resolved.error:
            st.warning("⚠️ Non posso salvare un risultato con errore.")
        else:
            ind = res["indicators"]; an = res["analyst"]; pio = res["piotroski"]
            snap = {
                "Ticker": resolved.ticker,
                "Nome": (resolved.long_name or resolved.ticker)[:30],
                "ISIN": resolved.isin or "N/A",
                "Prezzo": ind.close_price if ind else None,
                "RSI(14)": ind.rsi_value if ind else None,
                "RSI Status": ind.rsi_status if ind else "N/A",
                "ATR%": ind.atr_pct if ind else None,
                "Vol.": ind.atr_pct_status if ind else "N/A",
                "F-Score": pio.score if pio else None,
                "Pio. Interpr.": pio.interpretation if pio else "N/A",
                "Target Mean": round(an.target_mean, 2) if (an and an.target_mean) else None,
                "Upside%": upside_pct(an.target_mean, an.current_price) if an else None,
                "N. Analisti": an.num_analysts if an else None,
                "Consensus": (an.recommendation_key or "N/A") if an else "N/A",
                "Snapshot": res["timestamp"],
            }
            saved = [s for s in st.session_state["saved"] if s["Ticker"] != snap["Ticker"]]
            saved.append(snap)
            st.session_state["saved"] = saved[-4:]
            st.success(f"✅ Salvato {snap['Ticker']} ({len(st.session_state['saved'])}/4).")


# =====================================================================
# CONFRONTO
# =====================================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("## 🔬 Tabella di Confronto")

saved = st.session_state.get("saved", [])
if not saved:
    st.info("Nessun ticker salvato. Usa **💾 Salva Risultato** dopo aver generato un'analisi (max 4 strumenti).")
else:
    df = pd.DataFrame(saved).set_index("Ticker").T
    st.dataframe(df, use_container_width=True)
    col_a, _ = st.columns([1, 5])
    with col_a:
        if st.button("🗑️ Svuota confronto", use_container_width=True):
            st.session_state["saved"] = []
            st.rerun()

st.markdown(
    "<div style='color:#475569;font-size:11px;margin-top:32px;text-align:center;'>"
    "Quant Analyzer Pro · Dati: yfinance · Grafici: TradingView · "
    "I valori sono a scopo informativo, non costituiscono raccomandazione di investimento."
    "</div>",
    unsafe_allow_html=True,
)
