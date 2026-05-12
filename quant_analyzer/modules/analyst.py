"""
Rating & Target Price.

Strategia di estrazione:
1. Primaria: yfinance `upgrades_downgrades` per ottenere storico azioni analisti
   con (Firm, Action, ToGrade, FromGrade, Date). Filtrata agli ultimi 7 record.
2. Secondaria/aggregata: yfinance `analyst_price_targets` (dict) per il
   consensus corrente (low/mean/median/high/current).
3. yfinance non espone TARGET PRICE per singolo analista in modo strutturato
   in tutte le versioni: gestiamo con placeholder "N/A" quando assente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class AnalystData:
    # Tabella ultimi 7 record analisti (Agenzia, Azione, Rating, Data, Target)
    recent_actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Consensus aggregato
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    target_low: Optional[float] = None
    target_high: Optional[float] = None
    current_price: Optional[float] = None
    num_analysts: Optional[int] = None
    recommendation_key: Optional[str] = None   # buy/hold/sell/...
    error: Optional[str] = None


def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v is not None:
            return v
    return default


def fetch_analyst_data(ticker: str) -> AnalystData:
    """Recupera target price e azioni analisti."""
    out = AnalystData()
    try:
        t = yf.Ticker(ticker)
    except Exception as e:
        out.error = f"Impossibile istanziare Ticker: {e}"
        return out

    # ---- Consensus aggregato ----
    try:
        apt = t.analyst_price_targets  # dict o None
        if isinstance(apt, dict) and apt:
            out.target_mean = _safe_get(apt, "mean")
            out.target_median = _safe_get(apt, "median")
            out.target_low = _safe_get(apt, "low")
            out.target_high = _safe_get(apt, "high")
            out.current_price = _safe_get(apt, "current")
    except Exception:
        # Fallback su info
        try:
            info = t.info or {}
            out.target_mean = info.get("targetMeanPrice")
            out.target_median = info.get("targetMedianPrice")
            out.target_low = info.get("targetLowPrice")
            out.target_high = info.get("targetHighPrice")
            out.current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            out.num_analysts = info.get("numberOfAnalystOpinions")
            out.recommendation_key = info.get("recommendationKey")
        except Exception as e:
            out.error = f"Analyst targets non disponibili: {e}"

    # info per num_analysts e recommendationKey (sempre tentato)
    if out.num_analysts is None or out.recommendation_key is None:
        try:
            info = t.info or {}
            out.num_analysts = out.num_analysts or info.get("numberOfAnalystOpinions")
            out.recommendation_key = out.recommendation_key or info.get("recommendationKey")
            if out.current_price is None:
                out.current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        except Exception:
            pass

    # ---- Ultimi 7 record analisti ----
    try:
        ud = t.upgrades_downgrades  # DataFrame indicizzato per GradeDate
        if ud is not None and not ud.empty:
            df = ud.copy()
            # Indice -> colonna Data
            if df.index.name and df.index.name.lower().startswith("grade"):
                df = df.reset_index().rename(columns={df.index.name: "Date"})
            elif "GradeDate" in df.columns:
                df = df.rename(columns={"GradeDate": "Date"})
            else:
                df = df.reset_index().rename(columns={df.columns[0]: "Date"})

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date", ascending=False).head(7)

            # Costruzione tabella richiesta
            recent = pd.DataFrame({
                "Agenzia": df.get("Firm", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Azione": df.get("Action", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Rating": df.get("ToGrade", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Da": df.get("FromGrade", pd.Series(["N/A"] * len(df))).fillna("N/A"),
                "Data": df["Date"].dt.strftime("%Y-%m-%d").fillna("N/A"),
                # Target per-analista non esposto da yfinance -> placeholder
                "Target": ["N/A"] * len(df),
            })
            out.recent_actions = recent.reset_index(drop=True)
    except Exception as e:
        # Non blocca: tabella resta vuota
        if not out.error:
            out.error = f"Upgrades/downgrades non disponibili: {e}"

    return out


def upside_pct(target: Optional[float], current: Optional[float]) -> Optional[float]:
    if target is None or current is None or current == 0:
        return None
    try:
        return round((target - current) / current * 100.0, 2)
    except Exception:
        return None
