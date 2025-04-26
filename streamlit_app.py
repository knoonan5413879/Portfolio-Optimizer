import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import date

# Set Streamlit Page Config
st.set_page_config(page_title="📈 Portfolio Optimizer", layout="wide")

# Sidebar - User Inputs
with st.sidebar:
    st.title("Settings ⚙️")
    tickers_input = st.text_input("Enter Tickers (comma-separated)", "VTI, SCHD, BND, VXUS")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    start_date = st.date_input("Start Date", pd.to_datetime("2017-01-01"))
    n_portfolios = st.slider("Number of Portfolios to Simulate", 1000, 20000, 10000)

# Use today's date automatically for real-time data
end_date = date.today()

# Function to Download and Clean Data
@st.cache_data
def download_and_clean_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    adj_close = data.xs('Adj Close', level=1, axis=1)
    adj_close = adj_close.dropna(how='all', axis=1).ffill()
    return adj_close

# Portfolio performance
def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Minimize volatility function
def minimize_volatility(weights, expected_returns, cov_matrix):
    return portfolio_performance(weights, expected_returns, cov_matrix)[1]

# Main Logic
try:
    adj_close = download_and_clean_prices(tickers, start_date, end_date)
    returns = adj_close.pct_change().dropna()
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = portfolio_return / portfolio_volatility  # Sharpe Ratio

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]

    # Optimization
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    initial_guess = np.array([1/len(tickers)] * len(tickers))

    optimal_result = minimize(
        minimize_volatility,
        initial_guess,
        args=(expected_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = optimal_result.x
    optimal_return, optimal_risk = portfolio_performance(optimal_weights, expected_returns, cov_matrix)

    # Streamlit Tabs
    tab1, tab2, tab3 = st.tabs(["📥 Inputs", "📈 Plot", "📊 Results"])

    with tab1:
        st.header("User Inputs")
        st.write(f"**Tickers:** {', '.join(tickers)}")
        st.write(f"**Start Date:** {start_date}")
        st.write(f"**End Date (Today):** {end_date}")
        st.write(f"**Portfolios Simulated:** {n_portfolios}")

    with tab2:
        st.header("Efficient Frontier")
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
        ax.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], c='red', marker='*', s=200)
        ax.set_xlabel('Portfolio Volatility (Risk)')
        ax.set_ylabel('Portfolio Return')
        ax.set_title('Efficient Frontier')
        fig.colorbar(scatter, label='Sharpe Ratio')
        st.pyplot(fig)

    with tab3:
        st.header("Optimal Portfolio Details")
        st.subheader("🔹 Optimal Weights")
        for ticker, weight in zip(tickers, optimal_weights):
            st.write(f"**{ticker}:** {weight:.2%}")

        st.subheader("🔹 Performance Metrics")
        col1, col2 = st.columns(2)
        col1.metric(label="Expected Annual Return", value=f"{optimal_return:.2%}")
        col2.metric(label="Expected Annual Risk (Volatility)", value=f"{optimal_risk:.2%}")

except Exception as e:
    st.error(f"An error occurred: {e}")