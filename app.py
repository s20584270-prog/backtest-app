import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

st.set_page_config(page_title="AI Backtesting Platform", layout="wide")
st.title("📈 AI Backtesting Platform")
st.write("कोई भी स्टॉक चुनें, strategy चुनें, और बार-बार backtest करें।")

st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="RELIANCE.NS",
    help="Indian: RELIANCE.NS, TCS.NS | US: AAPL, MSFT | Crypto: BTC-USD")

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

cash = st.sidebar.number_input("Starting Cash", value=100000, step=10000)
commission = st.sidebar.number_input("Commission (%)", value=0.2, step=0.1) / 100

strategy_name = st.sidebar.selectbox("Strategy",
    ["SMA Crossover", "RSI", "MACD", "Buy & Hold"])

if strategy_name == "SMA Crossover":
    fast = st.sidebar.slider("Fast SMA", 5, 100, 50)
    slow = st.sidebar.slider("Slow SMA", 20, 300, 200)
elif strategy_name == "RSI":
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_low = st.sidebar.slider("Oversold (Buy below)", 10, 40, 30)
    rsi_high = st.sidebar.slider("Overbought (Sell above)", 60, 90, 70)

class SmaCross(Strategy):
    fast = 50
    slow = 200
    def init(self):
        f, s = self.fast, self.slow
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(f).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(s).mean(), self.data.Close)
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

class RsiStrategy(Strategy):
    period = 14
    low = 30
    high = 70
    def init(self):
        p = self.period
        def rsi(x):
            delta = pd.Series(x).diff()
            gain = delta.clip(lower=0).rolling(p).mean()
            loss = -delta.clip(upper=0).rolling(p).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        self.rsi = self.I(rsi, self.data.Close)
    def next(self):
        if self.rsi[-1] < self.low and not self.position:
            self.buy()
        elif self.rsi[-1] > self.high and self.position:
            self.sell()

class MacdStrategy(Strategy):
    def init(self):
        def ema(x, span):
            return pd.Series(x).ewm(span=span, adjust=False).mean()
        self.macd = self.I(lambda x: ema(x,12) - ema(x,26), self.data.Close)
        self.signal = self.I(lambda x: (ema(x,12)-ema(x,26)).ewm(span=9, adjust=False).mean(), self.data.Close)
    def next(self):
        if crossover(self.macd, self.signal):
            self.buy()
        elif crossover(self.signal, self.macd):
            self.sell()

class BuyHold(Strategy):
    def init(self): pass
    def next(self):
        if not self.position:
            self.buy()

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Data डाउनलोड हो रहा है..."):
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    if data.empty:
        st.error("डेटा नहीं मिला। Ticker या Date ठीक करें।")
    else:
        st.success(f"{len(data)} दिन का डेटा मिला।")

        if strategy_name == "SMA Crossover":
            SmaCross.fast, SmaCross.slow = fast, slow
            strat = SmaCross
        elif strategy_name == "RSI":
            RsiStrategy.period, RsiStrategy.low, RsiStrategy.high = rsi_period, rsi_low, rsi_high
            strat = RsiStrategy
        elif strategy_name == "MACD":
            strat = MacdStrategy
        else:
            strat = BuyHold

        bt = Backtest(data, strat, cash=cash, commission=commission, exclusive_orders=True)
        stats = bt.run()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Return", f"{stats['Return [%]']:.2f}%")
        col2.metric("Buy & Hold", f"{stats['Buy & Hold Return [%]']:.2f}%")
        col3.metric("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%")
        col4.metric("Trades", int(stats['# Trades']))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Sharpe", f"{stats.get('Sharpe Ratio', 0):.2f}")
        col6.metric("Sortino", f"{stats.get('Sortino Ratio', 0):.2f}")
        col7.metric("Win Rate", f"{stats.get('Win Rate [%]', 0):.2f}%")
        col8.metric("Final Equity", f"₹{stats['Equity Final [$]']:.0f}")

        st.subheader("📊 Equity Curve")
        st.line_chart(stats['_equity_curve']['Equity'])

        st.subheader("📋 Trade Log")
        st.dataframe(stats['_trades'])

        csv = stats['_trades'].to_csv().encode('utf-8')
        st.download_button("⬇️ Trades CSV डाउनलोड करें", csv, "trades.csv", "text/csv")
