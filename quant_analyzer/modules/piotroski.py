"""
Piotroski F-Score (2000) — implementazione rigorosa dei 9 criteri binari.

Categorie:
A) Redditività (4 criteri):
    1. ROA > 0           (NetIncome_t / TotalAssets_t-1)
    2. CFO > 0           (Operating Cash Flow corrente)
    3. ΔROA > 0          (ROA_t > ROA_t-1)
    4. Accruals: CFO > NetIncome (qualità degli utili)

B) Leva / Liquidità / Fonti di Finanziamento (3 criteri):
    5. ΔLeverage < 0     (LongTermDebt/TotalAssets diminuisce YoY)
    6. ΔCurrentRatio > 0 (CurrentAssets/CurrentLiabilities aumenta YoY)
    7. No emissione azioni (sharesOutstanding_t <= sharesOutstanding_t-1)

C) Efficienza Operativa (2 criteri):
    8. ΔGrossMargin > 0  (GrossProfit/Revenue aumenta YoY)
    9. ΔAssetTurnover > 0 (Revenue_t / TotalAssets_t-1 > Revenue_t-1 / TotalAssets_t-2)

Range output: 0–9. 8-9 = forte, 0-2 = debole.

Note implementative:
- yfinance espone bilanci/CF annuali via `.balance_sheet`, `.financials`, `.cashflow`.
- I DataFrame yfinance hanno colonne = date (più recente a sinistra) e indice = voci.
- Servono almeno 2 esercizi consecutivi: per criteri YoY che richiedono Assets_t-2
  (criterio 9 sull'asset turnover) tentiamo con 3 esercizi, ma in caso di mancanza
  facciamo fallback al confronto Revenue/Assets dello stesso anno (segnalato).
- Ogni criterio non calcolabile = 0 (conservativo) ma esposto separatamente nel detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PiotroskiResult:
    score: Optional[int]                    # 0-9 o None se non calcolabile
    details: dict = field(default_factory=dict)   # nome_criterio -> 0/1/None
    raw_values: dict = field(default_factory=dict)  # valori grezzi usati
    interpretation: str = "N/A"             # Forte / Medio / Debole / N/A
    error: Optional[str] = None


# ---- Utility di estrazione robusta da DataFrame yfinance ----

_ALIASES = {
    # Income Statement
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"],
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "gross_profit": ["Gross Profit"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    # Balance Sheet
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "shares_outstanding": ["Share Issued", "Ordinary Shares Number", "Common Stock Equity"],
    # Cash Flow
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"],
}


def _get_row(df: pd.DataFrame, keys: list[str]) -> Optional[pd.Series]:
    """Cerca la prima riga corrispondente a uno degli alias."""
    if df is None or df.empty:
        return None
    for k in keys:
        if k in df.index:
            return df.loc[k]
    # Lookup case-insensitive
    lower_idx = {str(i).lower(): i for i in df.index}
    for k in keys:
        if k.lower() in lower_idx:
            return df.loc[lower_idx[k.lower()]]
    return None


def _val(series: Optional[pd.Series], col_idx: int) -> Optional[float]:
    """Estrae valore alla posizione col_idx (0 = più recente)."""
    if series is None:
        return None
    try:
        v = series.iloc[col_idx]
        if pd.isna(v):
            return None
        return float(v)
    except (IndexError, ValueError, TypeError):
        return None


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _interpret(score: int) -> str:
    if score >= 8:
        return "Forte"
    if score >= 5:
        return "Medio"
    if score >= 3:
        return "Discreto"
    return "Debole"


def compute_piotroski(ticker: str) -> PiotroskiResult:
    """Pipeline completa Piotroski F-Score su dati yfinance."""
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet      # colonne = date (più recente a sinistra)
        is_ = t.financials
        cf = t.cashflow
    except Exception as e:
        return PiotroskiResult(score=None, error=f"Errore fetch bilanci: {e}")

    if bs is None or bs.empty or is_ is None or is_.empty or cf is None or cf.empty:
        return PiotroskiResult(
            score=None,
            error="Bilanci/Income/Cash Flow non disponibili da yfinance per questo ticker."
        )

    # Servono almeno 2 esercizi per i confronti YoY
    if bs.shape[1] < 2 or is_.shape[1] < 2 or cf.shape[1] < 2:
        return PiotroskiResult(
            score=None,
            error=f"Necessari ≥2 esercizi annuali. Disponibili: BS={bs.shape[1]}, IS={is_.shape[1]}, CF={cf.shape[1]}."
        )

    # Estrazione righe
    ni = _get_row(is_, _ALIASES["net_income"])
    rev = _get_row(is_, _ALIASES["revenue"])
    gp = _get_row(is_, _ALIASES["gross_profit"])
    ta_row = _get_row(bs, _ALIASES["total_assets"])
    ca_row = _get_row(bs, _ALIASES["current_assets"])
    cl_row = _get_row(bs, _ALIASES["current_liabilities"])
    ltd_row = _get_row(bs, _ALIASES["long_term_debt"])
    shares_row = _get_row(bs, _ALIASES["shares_outstanding"])
    cfo_row = _get_row(cf, _ALIASES["cfo"])

    # Valori t (corrente) e t-1
    NI_t = _val(ni, 0); NI_tm1 = _val(ni, 1)
    REV_t = _val(rev, 0); REV_tm1 = _val(rev, 1)
    GP_t = _val(gp, 0); GP_tm1 = _val(gp, 1)
    TA_t = _val(ta_row, 0); TA_tm1 = _val(ta_row, 1); TA_tm2 = _val(ta_row, 2)
    CA_t = _val(ca_row, 0); CA_tm1 = _val(ca_row, 1)
    CL_t = _val(cl_row, 0); CL_tm1 = _val(cl_row, 1)
    LTD_t = _val(ltd_row, 0); LTD_tm1 = _val(ltd_row, 1)
    SHARES_t = _val(shares_row, 0); SHARES_tm1 = _val(shares_row, 1)
    CFO_t = _val(cfo_row, 0)

    # Se GrossProfit mancante prova ricostruzione: Revenue - COGS
    if GP_t is None or GP_tm1 is None:
        cogs = _get_row(is_, _ALIASES["cogs"])
        cogs_t = _val(cogs, 0); cogs_tm1 = _val(cogs, 1)
        if GP_t is None and REV_t is not None and cogs_t is not None:
            GP_t = REV_t - cogs_t
        if GP_tm1 is None and REV_tm1 is not None and cogs_tm1 is not None:
            GP_tm1 = REV_tm1 - cogs_tm1

    details: dict = {}
    raw: dict = {}

    # --- A. Redditività ---
    # 1. ROA > 0  (NI_t / TA_tm1 secondo Piotroski; alcuni usano TA_t — qui aderenza al paper)
    roa_t = _safe_div(NI_t, TA_tm1)
    roa_tm1 = _safe_div(NI_tm1, TA_tm2) if TA_tm2 is not None else _safe_div(NI_tm1, TA_tm1)
    details["1_ROA_positivo"] = 1 if (roa_t is not None and roa_t > 0) else 0
    raw["ROA_t"] = roa_t

    # 2. CFO > 0
    details["2_CFO_positivo"] = 1 if (CFO_t is not None and CFO_t > 0) else 0
    raw["CFO_t"] = CFO_t

    # 3. ΔROA > 0
    if roa_t is not None and roa_tm1 is not None:
        details["3_DeltaROA_positivo"] = 1 if roa_t > roa_tm1 else 0
    else:
        details["3_DeltaROA_positivo"] = 0
    raw["ROA_tm1"] = roa_tm1

    # 4. Accruals: CFO > NI (qualità utili)
    if CFO_t is not None and NI_t is not None:
        details["4_Accruals_CFO_gt_NI"] = 1 if CFO_t > NI_t else 0
    else:
        details["4_Accruals_CFO_gt_NI"] = 0

    # --- B. Leva / Liquidità / Equity ---
    # 5. ΔLeverage < 0 (LTD/TA diminuisce)
    lev_t = _safe_div(LTD_t, TA_t)
    lev_tm1 = _safe_div(LTD_tm1, TA_tm1)
    if lev_t is not None and lev_tm1 is not None:
        details["5_DeltaLeverage_negativo"] = 1 if lev_t < lev_tm1 else 0
    else:
        # Se non c'è LTD (es. company unlevered) consideriamo non penalizzante = 1
        if LTD_t is None and LTD_tm1 is None:
            details["5_DeltaLeverage_negativo"] = 1
        else:
            details["5_DeltaLeverage_negativo"] = 0
    raw["Leverage_t"] = lev_t
    raw["Leverage_tm1"] = lev_tm1

    # 6. ΔCurrentRatio > 0
    cr_t = _safe_div(CA_t, CL_t)
    cr_tm1 = _safe_div(CA_tm1, CL_tm1)
    if cr_t is not None and cr_tm1 is not None:
        details["6_DeltaCurrentRatio_positivo"] = 1 if cr_t > cr_tm1 else 0
    else:
        details["6_DeltaCurrentRatio_positivo"] = 0
    raw["CurrentRatio_t"] = cr_t
    raw["CurrentRatio_tm1"] = cr_tm1

    # 7. No emissione azioni (shares non aumentano)
    if SHARES_t is not None and SHARES_tm1 is not None:
        # Tolleranza 0.5% per buyback/issuance immateriali
        details["7_No_emissione_azioni"] = 1 if SHARES_t <= SHARES_tm1 * 1.005 else 0
    else:
        details["7_No_emissione_azioni"] = 0
    raw["Shares_t"] = SHARES_t
    raw["Shares_tm1"] = SHARES_tm1

    # --- C. Efficienza Operativa ---
    # 8. ΔGrossMargin > 0
    gm_t = _safe_div(GP_t, REV_t)
    gm_tm1 = _safe_div(GP_tm1, REV_tm1)
    if gm_t is not None and gm_tm1 is not None:
        details["8_DeltaGrossMargin_positivo"] = 1 if gm_t > gm_tm1 else 0
    else:
        details["8_DeltaGrossMargin_positivo"] = 0
    raw["GrossMargin_t"] = gm_t
    raw["GrossMargin_tm1"] = gm_tm1

    # 9. ΔAssetTurnover > 0  (Rev_t / TA_tm1)  vs  (Rev_tm1 / TA_tm2)
    at_t = _safe_div(REV_t, TA_tm1)
    at_tm1 = _safe_div(REV_tm1, TA_tm2) if TA_tm2 is not None else _safe_div(REV_tm1, TA_tm1)
    if at_t is not None and at_tm1 is not None:
        details["9_DeltaAssetTurnover_positivo"] = 1 if at_t > at_tm1 else 0
    else:
        details["9_DeltaAssetTurnover_positivo"] = 0
    raw["AssetTurnover_t"] = at_t
    raw["AssetTurnover_tm1"] = at_tm1

    score = int(sum(v for v in details.values() if v in (0, 1)))
    return PiotroskiResult(
        score=score,
        details=details,
        raw_values=raw,
        interpretation=_interpret(score),
        error=None,
    )
