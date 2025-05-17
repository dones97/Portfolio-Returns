import streamlit as st
import yfinance as yf
import pandas as pd

# Simulate a scrip code coming from Excel or CSV (change this as needed for your test)
raw_scrip_code = 543306.0  # Could also be "543306", " 543306 ", etc.

# Robust ticker construction
def clean_scrip_code(scrip_code):
    if pd.isnull(scrip_code):
        return None
    s = str(scrip_code).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.isdigit():
        return f"{s.zfill(6)}.BO"
    return None

ticker = clean_scrip_code(raw_scrip_code)
st.write(f"Ticker being queried: '{ticker}' (type: {type(ticker)})")

if ticker:
    data = yf.Ticker(ticker).history(period="1mo")
    st.write(f"Data for {ticker}:", data)
else:
    st.write("Ticker could not be constructed from the scrip code.")
