"""
Quant Analyzer — Streamlit App
Single-page application di analisi fondamentale + tecnica per equity globali.

Pipeline:
    Input (ISIN/Ticker)
        -> resolver.resolve_symbol()
        -> [TradingView widget] + [Analyst data] + [RSI/ATR%] + [Piotroski]
        -> Salva in session_state['saved']
        -> Tabella di confronto orizzontale (max 4 ticker)
"""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.resolver import resolve_symbol, ResolvedSymbol
from modules.indicators import compute_indicators
from modules.analyst import fetch_analyst_data, upside_pct
from modules.piotroski import compute_piotroski
from modules.tradingview import map_to_tradingview, tradingview_html


# ============================================================
# CONFIG & THEME
# ============================================================
st.set_page_config(
    page_title="Quant Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stText, p, div, span, label {
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

.stDataFrame, .stTable { font-size: 13px !important; }

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


# ============================================================
# SESSION STATE INIT
# ============================================================
if "saved" not in st.session_state:
    st.session_state["saved"] = []      # lista di dict per confronto
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "input_buffer" not in st.session_state:
    st.session_state["input_buffer"] = ""


# ============================================================
# CACHED PIPELINE
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline(raw_input: str) -> dict:
    """Esegue l'intera pipeline e restituisce un dict serializzabile."""
    resolved: ResolvedSymbol = resolve_symbol(raw_input)
    out = {
        "resolved": resolved,
        "indicators": None,
        "analyst": None,
        "piotroski": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if resolved.error or not resolved.ticker:
        return out

    out["indicators"] = compute_indicators(resolved.ticker)
    out["analyst"] = fetch_analyst_data(resolved.ticker)
    out["piotroski"] = compute_piotroski(resolved.ticker)
    return out


# ============================================================
# HELPERS UI
# ============================================================
def render_metric(label: str, value: str, status_class: str = "metric-status-neutral", subtitle: str = ""):
    sub_html = f'<div style="font-size:11px;color:#7aa8c8;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {status_class}">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def status_class_rsi(status: str) -> str:
    return {"Overbought": "metric-status-bad", "Oversold": "metric-status-warn",
            "Neutral": "metric-status-neutral"}.get(status, "metric-status-neutral")


def status_class_atr(status: str) -> str:
    return {"Alta": "metric-status-bad", "Media": "metric-status-warn",
            "Bassa": "metric-status-ok"}.get(status, "metric-status-neutral")


def status_class_piotroski(score) -> str:
    if score is None: return "metric-status-neutral"
    if score >= 8: return "metric-status-ok"
    if score >= 5: return "metric-status-neutral"
    if score >= 3: return "metric-status-warn"
    return "metric-status-bad"


def validate_input(raw: str) -> tuple[bool, str]:
    """Validazione preliminare via regex."""
    if not raw or not raw.strip():
        return False, "Inserire un ticker o un ISIN."
    s = raw.strip().upper()
    isin_re = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
    ticker_re = re.compile(r"^[A-Z0-9]{1,10}(\.[A-Z]{1,4})?$")
    if isin_re.match(s) or ticker_re.match(s):
        return True, ""
    return False, f"Formato non valido: '{s}'. Atteso ISIN (12 caratteri) o Ticker (es. AAPL, ENI.MI)."


# ============================================================
# HEADER
# ============================================================
st.markdown("# 📊 Quant Analyzer Pro")
st.markdown(
    "<div style='color:#7aa8c8;font-size:13px;margin-bottom:16px;'>"
    "Analisi tecnica + fondamentale con risoluzione automatica ISIN→Ticker · TradingView · yfinance"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# INPUT BAR
# ============================================================
col_input, col_btn1, col_btn2, col_btn3 = st.columns([4, 1.2, 1.2, 1])

with col_input:
    user_input = st.text_input(
        "Ticker o ISIN",
        value=st.session_state["input_buffer"],
        placeholder="Es. AAPL, ENI.MI, IT0003132476, US0378331005",
        label_visibility="collapsed",
        key="ti_input",
    )

with col_btn1:
    generate = st.button("🔍 Genera Risultato", use_container_width=True)
with col_btn2:
    save = st.button("💾 Salva Risultato", use_container_width=True)
with col_btn3:
    reset = st.button("♻️ Reset", use_container_width=True)


# ============================================================
# AZIONI
# ============================================================
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
                result = run_pipeline(user_input.strip().upper())
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"❌ Errore pipeline: {type(e).__name__}: {e}")
                st.session_state["last_result"] = None


# ============================================================
# RENDER LAST RESULT
# ============================================================
result = st.session_state.get("last_result")

if result is not None:
    resolved: ResolvedSymbol = result["resolved"]

    if resolved.error:
        st.error(f"❌ Risoluzione fallita: {resolved.error}")
    else:
        # ----- Header strumento -----
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

        # ----- KPI ROW -----
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            px = ind.close_price if ind and ind.close_price else None
            render_metric("Prezzo (Close)", f"{px}" if px is not None else "N/A")
        with c2:
            if ind and ind.rsi_value is not None:
                render_metric("RSI(14)", f"{ind.rsi_value}", status_class_rsi(ind.rsi_status), ind.rsi_status)
            else:
                render_metric("RSI(14)", "N/A")
        with c3:
            if ind and ind.atr_pct is not None:
                render_metric("ATR%(14)", f"{ind.atr_pct}%", status_class_atr(ind.atr_pct_status), f"Volatilità {ind.atr_pct_status}")
            else:
                render_metric("ATR%(14)", "N/A")
        with c4:
            if pio and pio.score is not None:
                render_metric("Piotroski F-Score", f"{pio.score}/9", status_class_piotroski(pio.score), pio.interpretation)
            else:
                render_metric("Piotroski F-Score", "N/A", "metric-status-neutral", (pio.error or "")[:40] if pio else "")
        with c5:
            if an and an.target_mean is not None and an.current_price is not None:
                up = upside_pct(an.target_mean, an.current_price)
                cls = "metric-status-ok" if (up and up > 0) else "metric-status-bad"
                render_metric("Target Mean", f"{round(an.target_mean,2)}", cls, f"Upside: {up}%" if up is not None else "")
            else:
                render_metric("Target Mean", "N/A")

        # ----- TRADINGVIEW CHART -----
        st.markdown("## 📈 Grafico Real-time (TradingView)")
        tv_sym = map_to_tradingview(resolved.ticker)
        st.caption(f"Simbolo TradingView: `{tv_sym}`")
        components.html(tradingview_html(tv_sym, height=540), height=560)

        # ----- TABS: Analisti / Piotroski / Indicatori -----
        tab1, tab2, tab3 = st.tabs(["🎯 Rating & Target", "📒 Piotroski F-Score", "📊 Storico Indicatori"])

        with tab1:
            if an is None or an.error and (an.recent_actions is None or an.recent_actions.empty) and an.target_mean is None:
                st.warning(f"⚠️ Dati analisti non disponibili. {an.error if an else ''}")
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
                    st.info("Storico upgrades/downgrades non disponibile per questo strumento.")

        with tab2:
            if pio is None or pio.score is None:
                st.warning(f"⚠️ Piotroski non calcolabile. {pio.error if pio else ''}")
            else:
                st.markdown(f"### F-Score: **{pio.score} / 9** — {pio.interpretation}")
                # Dettaglio criteri
                rows = []
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
                for k, lab in labels.items():
                    v = pio.details.get(k, 0)
                    rows.append({"Criterio": lab, "Pass": "✅" if v == 1 else "❌", "Score": v})
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


# ============================================================
# SALVATAGGIO
# ============================================================
if save:
    if st.session_state.get("last_result") is None:
        st.warning("⚠️ Nessun risultato da salvare. Esegui prima 'Genera Risultato'.")
    else:
        res = st.session_state["last_result"]
        resolved = res["resolved"]
        if resolved.error:
            st.warning("⚠️ Non posso salvare un risultato con errore di risoluzione.")
        else:
            ind = res["indicators"]; an = res["analyst"]; pio = res["piotroski"]
            snapshot = {
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
            # Evita duplicati: sostituisce stesso ticker
            saved = [s for s in st.session_state["saved"] if s["Ticker"] != snapshot["Ticker"]]
            saved.append(snapshot)
            # Limita a 4 ticker (FIFO)
            st.session_state["saved"] = saved[-4:]
            st.success(f"✅ Salvato {snapshot['Ticker']} ({len(st.session_state['saved'])}/4 in confronto).")


# ============================================================
# TABELLA DI CONFRONTO (orizzontale)
# ============================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("## 🔬 Tabella di Confronto")

saved = st.session_state.get("saved", [])
if not saved:
    st.info("Nessun ticker salvato. Usa **💾 Salva Risultato** dopo aver generato un'analisi (max 4 strumenti).")
else:
    # DataFrame: righe = metriche, colonne = ticker (confronto orizzontale)
    df = pd.DataFrame(saved).set_index("Ticker").T
    st.dataframe(df, use_container_width=True)

    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("🗑️ Svuota confronto", use_container_width=True):
            st.session_state["saved"] = []
            st.rerun()

# Footer
st.markdown(
    "<div style='color:#475569;font-size:11px;margin-top:32px;text-align:center;'>"
    "Quant Analyzer Pro · Dati: yfinance · Grafici: TradingView · "
    "I valori sono a scopo informativo, non costituiscono raccomandazione di investimento."
    "</div>",
    unsafe_allow_html=True,
)
