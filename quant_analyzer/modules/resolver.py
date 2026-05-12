"""
Modulo di risoluzione e validazione Ticker/ISIN.

Strategia:
1. Pattern matching via regex per distinguere ISIN (12 char alfanumerici) da Ticker.
2. Se ISIN -> risoluzione a Ticker simbolico via yf.Search() prima di procedere.
3. Validazione finale tramite probe su yfinance Ticker.info.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import yfinance as yf

# ISIN: 2 lettere (country) + 9 alfanumerici + 1 check digit
ISIN_REGEX = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# Ticker: 1-10 caratteri alfanumerici con possibili suffissi exchange (.MI, .DE, .PA, ecc.)
TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,10}(\.[A-Z]{1,4})?$")


@dataclass
class ResolvedSymbol:
    """Output strutturato della risoluzione."""
    input_raw: str
    input_type: str          # "ISIN" | "TICKER"
    ticker: str              # Ticker simbolico finale per le API
    isin: Optional[str]      # ISIN se disponibile
    long_name: Optional[str] # Nome esteso strumento
    error: Optional[str] = None


def classify_input(raw: str) -> str:
    """Classifica l'input come ISIN o TICKER tramite regex."""
    s = raw.strip().upper()
    if ISIN_REGEX.match(s):
        return "ISIN"
    if TICKER_REGEX.match(s):
        return "TICKER"
    return "INVALID"


def resolve_isin_to_ticker(isin: str) -> Optional[str]:
    """
    Risolve un ISIN al ticker simbolico Yahoo Finance.
    Usa yf.Search che è l'endpoint ufficiale di lookup.
    """
    try:
        search = yf.Search(isin, max_results=5)
        quotes = search.quotes or []
        if not quotes:
            return None
        # Preferenza al primo risultato di tipo EQUITY
        for q in quotes:
            if q.get("quoteType", "").upper() in ("EQUITY", "ETF"):
                sym = q.get("symbol")
                if sym:
                    return sym
        # Fallback: primo simbolo restituito
        return quotes[0].get("symbol")
    except Exception:
        return None


def probe_ticker(ticker: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Verifica che il ticker esista realmente su Yahoo Finance.
    Ritorna (esiste, long_name, isin_da_yfinance).
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        # yfinance ritorna {} o info parziale se ticker invalido
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None and not info.get("symbol"):
            # Tentativo secondario via fast_info
            try:
                fi = t.fast_info
                if fi is None or fi.get("last_price") is None:
                    return False, None, None
            except Exception:
                return False, None, None

        long_name = info.get("longName") or info.get("shortName")
        isin = info.get("isin")
        return True, long_name, isin
    except Exception:
        return False, None, None


def resolve_symbol(raw_input: str) -> ResolvedSymbol:
    """
    Pipeline completa: classifica -> risolve -> valida.
    """
    if not raw_input or not raw_input.strip():
        return ResolvedSymbol(
            input_raw=raw_input, input_type="INVALID",
            ticker="", isin=None, long_name=None,
            error="Input vuoto."
        )

    raw = raw_input.strip().upper()
    kind = classify_input(raw)

    if kind == "INVALID":
        return ResolvedSymbol(
            input_raw=raw, input_type="INVALID",
            ticker="", isin=None, long_name=None,
            error=f"Formato non riconosciuto: '{raw}'. Attesi ISIN (12 caratteri) o Ticker (es. AAPL, ENI.MI)."
        )

    # Risoluzione ISIN -> Ticker
    if kind == "ISIN":
        ticker = resolve_isin_to_ticker(raw)
        if ticker is None:
            return ResolvedSymbol(
                input_raw=raw, input_type="ISIN",
                ticker="", isin=raw, long_name=None,
                error=f"ISIN '{raw}' non risolvibile a ticker tramite Yahoo Finance."
            )
    else:
        ticker = raw

    # Validazione tramite probe
    exists, long_name, isin_found = probe_ticker(ticker)
    if not exists:
        return ResolvedSymbol(
            input_raw=raw, input_type=kind,
            ticker=ticker, isin=raw if kind == "ISIN" else None,
            long_name=None,
            error=f"Ticker '{ticker}' non trovato su Yahoo Finance."
        )

    return ResolvedSymbol(
        input_raw=raw,
        input_type=kind,
        ticker=ticker,
        isin=isin_found or (raw if kind == "ISIN" else None),
        long_name=long_name,
        error=None,
    )
