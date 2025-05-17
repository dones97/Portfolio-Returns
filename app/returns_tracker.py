import yfinance as yf
import streamlit as st

ticker = "DODLA.BO"
data = yf.Ticker(ticker).history(period="1mo")
st.write(f"Fetched data for {ticker}:", data)
