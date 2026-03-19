"""
universe.py — NIFTY 50 constituents for the Indian stock market.

Each ticker uses the '.NS' suffix for Yahoo Finance (NSE).
"""

NIFTY_50 = {
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy"},
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Banking"},
    "INFY.NS": {"name": "Infosys", "sector": "IT"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Banking"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "Banking"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel", "sector": "Telecom"},
    "ITC.NS": {"name": "ITC Limited", "sector": "FMCG"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Banking"},
    "LT.NS": {"name": "Larsen & Toubro", "sector": "Infrastructure"},
    "AXISBANK.NS": {"name": "Axis Bank", "sector": "Banking"},
    "WIPRO.NS": {"name": "Wipro", "sector": "IT"},
    "HCLTECH.NS": {"name": "HCL Technologies", "sector": "IT"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Auto"},
    "SUNPHARMA.NS": {"name": "Sun Pharma", "sector": "Pharma"},
    "TATAMOTORS.NS": {"name": "Tata Motors", "sector": "Auto"},
    "NTPC.NS": {"name": "NTPC Limited", "sector": "Power"},
    "TITAN.NS": {"name": "Titan Company", "sector": "Consumer"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance", "sector": "Finance"},
    "POWERGRID.NS": {"name": "Power Grid Corp", "sector": "Power"},
    "ONGC.NS": {"name": "ONGC", "sector": "Energy"},
    "ULTRACEMCO.NS": {"name": "UltraTech Cement", "sector": "Cement"},
    "ADANIENT.NS": {"name": "Adani Enterprises", "sector": "Conglomerate"},
    "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Metals"},
    "BAJAJFINSV.NS": {"name": "Bajaj Finserv", "sector": "Finance"},
    "TECHM.NS": {"name": "Tech Mahindra", "sector": "IT"},
    "ASIANPAINT.NS": {"name": "Asian Paints", "sector": "Consumer"},
    "JSWSTEEL.NS": {"name": "JSW Steel", "sector": "Metals"},
    "M&M.NS": {"name": "Mahindra & Mahindra", "sector": "Auto"},
    "NESTLEIND.NS": {"name": "Nestle India", "sector": "FMCG"},
    "COALINDIA.NS": {"name": "Coal India", "sector": "Mining"},
    "ADANIPORTS.NS": {"name": "Adani Ports", "sector": "Infrastructure"},
    "BPCL.NS": {"name": "BPCL", "sector": "Energy"},
    "GRASIM.NS": {"name": "Grasim Industries", "sector": "Cement"},
    "CIPLA.NS": {"name": "Cipla", "sector": "Pharma"},
    "DRREDDY.NS": {"name": "Dr. Reddy's Labs", "sector": "Pharma"},
    "DIVISLAB.NS": {"name": "Divi's Laboratories", "sector": "Pharma"},
    "EICHERMOT.NS": {"name": "Eicher Motors", "sector": "Auto"},
    "HEROMOTOCO.NS": {"name": "Hero MotoCorp", "sector": "Auto"},
    "INDUSINDBK.NS": {"name": "IndusInd Bank", "sector": "Banking"},
    "BRITANNIA.NS": {"name": "Britannia Industries", "sector": "FMCG"},
    "APOLLOHOSP.NS": {"name": "Apollo Hospitals", "sector": "Healthcare"},
    "HINDALCO.NS": {"name": "Hindalco", "sector": "Metals"},
    "SBILIFE.NS": {"name": "SBI Life Insurance", "sector": "Insurance"},
    "HDFCLIFE.NS": {"name": "HDFC Life Insurance", "sector": "Insurance"},
    "TATACONSUM.NS": {"name": "Tata Consumer Products", "sector": "FMCG"},
    "BAJAJ-AUTO.NS": {"name": "Bajaj Auto", "sector": "Auto"},
    "UPL.NS": {"name": "UPL", "sector": "Chemicals"},
    
    # Custom Additions
    "KARURVYSYA.NS": {"name": "Karur Vysya Bank", "sector": "Banking"},
    "KAYNES.NS": {"name": "Kaynes Technology", "sector": "Electronics"},
    "GOLDBEES.NS": {"name": "Nippon India ETF Gold BeES", "sector": "ETF"},
    "NIFTYBEES.NS": {"name": "Nippon India ETF Nifty 50 BeES", "sector": "ETF"},
    "CPSEETF.NS": {"name": "CPSE ETF", "sector": "ETF"},
    "RVNL.NS": {"name": "Rail Vikas Nigam", "sector": "Infrastructure"},
    "WAAREERTL.NS": {"name": "Waaree Renewable Technologies", "sector": "Renewables"},
    "LLOYDSENT.NS": {"name": "Lloyds Enterprises", "sector": "Trading"},
    "ADANIPOWER.NS": {"name": "Adani Power", "sector": "Power"},
    "KPRMILL.NS": {"name": "KPR Mill", "sector": "Textiles"},
}

def get_universe() -> dict:
    """Return the full stock universe."""
    return NIFTY_50


def get_symbols() -> list:
    """Return list of ticker symbols."""
    return list(NIFTY_50.keys())


def get_symbol_name(symbol: str) -> str:
    """Get human-readable name for a symbol."""
    return NIFTY_50.get(symbol, {}).get("name", symbol)


def get_symbol_sector(symbol: str) -> str:
    """Get sector for a symbol."""
    return NIFTY_50.get(symbol, {}).get("sector", "Unknown")
