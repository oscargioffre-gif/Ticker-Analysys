"""
Widget TradingView Advanced Chart in stile Investing.com.
Mappatura ticker yfinance -> simbolo TradingView (exchange:symbol).
"""
from __future__ import annotations


# Mappatura suffisso yfinance -> exchange TradingView
_EXCHANGE_MAP = {
    ".MI": "MIL",       # Borsa Italiana
    ".DE": "XETR",      # XETRA
    ".PA": "EURONEXT",  # Parigi
    ".AS": "EURONEXT",  # Amsterdam
    ".BR": "EURONEXT",  # Bruxelles
    ".L":  "LSE",       # Londra
    ".MC": "BME",       # Madrid
    ".SW": "SIX",       # Svizzera
    ".VI": "WBAG",      # Vienna
    ".HE": "OMXHEX",
    ".ST": "OMXSTO",
    ".CO": "OMXCOP",
    ".OL": "OSL",
    ".TO": "TSX",
    ".V":  "TSXV",
    ".HK": "HKEX",
    ".T":  "TSE",
    ".AX": "ASX",
}


def map_to_tradingview(yf_ticker: str) -> str:
    """
    Converte ticker yfinance al formato TradingView 'EXCHANGE:SYMBOL'.
    Default: NASDAQ per ticker US senza suffisso.
    """
    t = yf_ticker.strip().upper()
    if "." in t:
        sym, _, suf = t.partition(".")
        exch = _EXCHANGE_MAP.get(f".{suf}", "")
        if exch:
            return f"{exch}:{sym}"
        return t  # fallback grezzo
    # Ticker US: TradingView accetta anche solo il simbolo
    return f"NASDAQ:{t}"


def tradingview_html(tv_symbol: str, height: int = 520) -> str:
    """Genera embed HTML del widget Advanced Chart TradingView, dark theme."""
    container_id = "tv_chart_container"
    return f"""
<div class="tradingview-widget-container" style="height:{height}px;width:100%;">
  <div id="{container_id}" style="height:{height}px;width:100%;"></div>
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
        "container_id": "{container_id}"
    }});
  </script>
</div>
"""
