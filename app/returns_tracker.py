import streamlit as st
import yfinance as yf
import pandas as pd

def clean_bse_ticker(scrip_code):
    if pd.isnull(scrip_code):
        return None
    s = str(scrip_code).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if s.isdigit():
        return f"{s.zfill(6)}.BO"
    return None

user_input = st.text_input("Enter scrip code (try: 543306, 543306.0, '543306')", "543306.0")
ticker = clean_bse_ticker(user_input)
st.write(f"Yahoo ticker generated: {repr(ticker)}")
if ticker:
    data = yf.Ticker(ticker).history(period="1mo")
    st.write(data)
else:
    st.write("Could not build Yahoo ticker from input.")
