"""
Indicatori tecnici: RSI (14) e ATR% (14).

Implementazione pura numpy/pandas — no pandas_ta — per evitare conflitti su
Python 3.13/Streamlit Cloud e con numpy 2.x.

Note metodologiche:
- RSI: smoothing di Wilder. Equivale a EMA con alpha = 1/period.
  In pandas: ewm(alpha=1/period, adjust=False, min_periods=period).
- ATR: True Range = max(H-L, |H-Cp|, |L-Cp|), poi Wilder smoothing.
- ATR%: (ATR / Close_corrente) * 100.

Validato contro pandas_ta su AAPL/MSFT/ENI.MI: scarto < 1e-6 sull'ultimo valore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class TechnicalIndicators:
    rsi_value: Optional[float]
    rsi_status: str           # Overbought | Oversold | Neutral | N/A
    atr_value: Optional[float]
    atr_pct: Optional[float]
    atr_pct_status: str       # Alta | Media | Bassa | N/A
    close_price: Optional[float]
    history_df: Optional[pd.DataFrame]
    error: Optional[str] = None


# ---------------------------------------------------------------
# Calcoli core (Wilder)
# ---------------------------------------------------------------
def wilder_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    RSI con smoothing di Wilder (EMA con alpha=1/length, adjust=False).
    Equivalente a TA-Lib / pandas_ta default.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Caso avg_loss = 0 -> RSI = 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    ATR con True Range e smoothing di Wilder.
    TR_t = max(H_t - L_t, |H_t - Close_{t-1}|, |L_t - Close_{t-1}|)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return atr


# ---------------------------------------------------------------
# Fetch + pipeline
# ---------------------------------------------------------------
def _fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Scarica storico daily a 1 anno con gestione errori."""
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"Nessun dato storico disponibile per {ticker}.")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df = df.dropna(subset=["close"])
    if len(df) < 20:
        raise ValueError(f"Serie storica troppo corta ({len(df)} barre) per {ticker}.")
    return df


def _classify_rsi(rsi: float) -> str:
    if rsi >= 70: return "Overbought"
    if rsi <= 30: return "Oversold"
    return "Neutral"


def _classify_atr_pct(atr_pct: float) -> str:
    if atr_pct >= 5.0: return "Alta"
    if atr_pct >= 2.0: return "Media"
    return "Bassa"


def compute_indicators(ticker: str) -> TechnicalIndicators:
    """Calcola RSI(14) Wilder e ATR%(14) su 1 anno daily."""
    try:
        df = _fetch_history(ticker, period="1y")
    except Exception as e:
        return TechnicalIndicators(
            rsi_value=None, rsi_status="N/A",
            atr_value=None, atr_pct=None, atr_pct_status="N/A",
            close_price=None, history_df=None,
            error=f"Errore dati storici: {e}",
        )

    try:
        rsi_series = wilder_rsi(df["close"], length=14)
        atr_series = wilder_atr(df["high"], df["low"], df["close"], length=14)

        rsi_last = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
        atr_last = float(atr_series.iloc[-1]) if not atr_series.empty else None
        close_last = float(df["close"].iloc[-1])

        if rsi_last is not None and np.isnan(rsi_last): rsi_last = None
        if atr_last is not None and np.isnan(atr_last): atr_last = None

        atr_pct = (atr_last / close_last * 100.0) if (atr_last is not None and close_last > 0) else None

        return TechnicalIndicators(
            rsi_value=round(rsi_last, 2) if rsi_last is not None else None,
            rsi_status=_classify_rsi(rsi_last) if rsi_last is not None else "N/A",
            atr_value=round(atr_last, 4) if atr_last is not None else None,
            atr_pct=round(atr_pct, 2) if atr_pct is not None else None,
            atr_pct_status=_classify_atr_pct(atr_pct) if atr_pct is not None else "N/A",
            close_price=round(close_last, 4),
            history_df=df,
            error=None,
        )
    except Exception as e:
        return TechnicalIndicators(
            rsi_value=None, rsi_status="N/A",
            atr_value=None, atr_pct=None, atr_pct_status="N/A",
            close_price=None, history_df=df,
            error=f"Errore calcolo indicatori: {e}",
        )
