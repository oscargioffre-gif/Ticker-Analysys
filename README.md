# Quant Analyzer Pro

Analisi tecnica + fondamentale per equity globali, con risoluzione automatica ISIN→Ticker.

## Funzionalità

- **Input universale**: Ticker (es. `AAPL`, `ENI.MI`) o ISIN (es. `IT0003132476`) — validazione regex e risoluzione automatica via `yf.Search`.
- **Grafico TradingView** embedded (dark theme, RSI + SMA pre-caricati).
- **Rating & Target Price**: ultime 7 azioni analisti + consensus aggregato (low/mean/median/high) via yfinance.
- **RSI (14, Wilder)** su 1 anno daily, soglie 30/70.
- **ATR % (14, Wilder)**: `(ATR / Close) × 100`, classificato Alta / Media / Bassa.
- **Piotroski F-Score (9 criteri rigorosi)**: ROA, CFO, ΔROA, Accruals, ΔLeverage, ΔCurrent Ratio, no-issuance, ΔGross Margin, ΔAsset Turnover.
- **Tabella di confronto orizzontale** fino a 4 strumenti, persistente in `st.session_state`.

## Struttura

```
quant_analyzer/
├── app.py                    # Streamlit single-page app
├── requirements.txt
├── .streamlit/config.toml    # Dark theme
└── modules/
    ├── resolver.py           # Validazione + risoluzione ISIN/Ticker
    ├── indicators.py         # RSI(14) + ATR%(14) via pandas_ta
    ├── analyst.py            # Target price e upgrades/downgrades
    ├── piotroski.py          # F-Score (9 criteri)
    └── tradingview.py        # Mapping ticker → exchange TradingView + embed
```

## Run locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy su Streamlit Cloud

1. Push del repo su GitHub.
2. New app su [streamlit.io/cloud](https://streamlit.io/cloud), main file `app.py`.
3. Python ≥ 3.10, < 3.13 raccomandato (pandas_ta talora non installa su 3.13/3.14).

## Note metodologiche

| Metrica | Calcolo | Soglie |
|---|---|---|
| RSI(14) | Wilder smoothing su close giornalieri (1y) | >70 overbought, <30 oversold |
| ATR%(14) | TR = max(H-L, \|H-Cp\|, \|L-Cp\|), Wilder, /Close × 100 | Alta ≥5%, Media 2-5%, Bassa <2% |
| Piotroski | Somma binaria 9 criteri su 2 esercizi annuali | 8-9 Forte, 5-7 Medio, 3-4 Discreto, 0-2 Debole |

## Gestione errori

Tutte le chiamate API sono in `try/except` con messaggi tecnici precisi:
- ISIN non risolvibile → `"ISIN 'XXXX' non risolvibile a ticker tramite Yahoo Finance."`
- Ticker non esistente → `"Ticker 'XXX' non trovato su Yahoo Finance."`
- Bilanci insufficienti → `"Necessari ≥2 esercizi annuali. Disponibili: BS=1, IS=2, CF=1."`

I valori sono a scopo informativo e non costituiscono raccomandazione di investimento.
