import yfinance as yf
import streamlit as st

ticker = "543306.BO"
data = yf.Ticker(ticker).history(period="1mo")
st.write(f"Fetched data for {ticker}:", data)
