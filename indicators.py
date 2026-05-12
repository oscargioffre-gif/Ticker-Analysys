"""
Indicatori tecnici: RSI (14) e ATR% (14).

Note metodologiche:
- RSI: smoothing di Wilder (EMA con alpha=1/period), standard di mercato.
- ATR: True Range = max(H-L, |H-Cp|, |L-Cp|), poi Wilder smoothing.
- ATR%: (ATR / Close_corrente) * 100, volatilità normalizzata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
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


def _fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Scarica storico daily a 1 anno con gestione errori."""
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"Nessun dato storico disponibile per {ticker}.")
    # Standardizzazione colonne
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df = df.dropna(subset=["close"])
    if len(df) < 20:
        raise ValueError(f"Serie storica troppo corta ({len(df)} barre) per {ticker}.")
    return df


def _classify_rsi(rsi: float) -> str:
    if rsi >= 70:
        return "Overbought"
    if rsi <= 30:
        return "Oversold"
    return "Neutral"


def _classify_atr_pct(atr_pct: float) -> str:
    """Classificazione basata su soglie tipiche su equity."""
    if atr_pct >= 5.0:
        return "Alta"
    if atr_pct >= 2.0:
        return "Media"
    return "Bassa"


def compute_indicators(ticker: str) -> TechnicalIndicators:
    """
    Calcola RSI(14) Wilder e ATR%(14) su 1 anno daily.
    """
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
        # RSI con smoothing di Wilder (pandas_ta default = Wilder via talib_mode)
        rsi_series = ta.rsi(df["close"], length=14)
        rsi_last = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

        # ATR con smoothing di Wilder
        atr_series = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14)
        atr_last = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else None

        close_last = float(df["close"].iloc[-1])
        atr_pct = (atr_last / close_last * 100.0) if (atr_last is not None and close_last > 0) else None

        return TechnicalIndicators(
            rsi_value=round(rsi_last, 2) if rsi_last is not None and not np.isnan(rsi_last) else None,
            rsi_status=_classify_rsi(rsi_last) if rsi_last is not None and not np.isnan(rsi_last) else "N/A",
            atr_value=round(atr_last, 4) if atr_last is not None and not np.isnan(atr_last) else None,
            atr_pct=round(atr_pct, 2) if atr_pct is not None and not np.isnan(atr_pct) else None,
            atr_pct_status=_classify_atr_pct(atr_pct) if atr_pct is not None and not np.isnan(atr_pct) else "N/A",
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
