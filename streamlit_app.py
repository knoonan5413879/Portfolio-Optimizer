import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import date
import plotly.express as px

## Page config
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
    Optimize your investment portfolio using real-time financial data.  
    Simulate thousands of portfolio combinations and find the one with the maximum Sharpe Ratio.
    """)


# Sidebar - User Inputs
with st.sidebar:
    st.title("Portfolio Optimizer App")
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

#Loading Data Icon
with st.spinner("📡 Fetching data... please wait..."):
    adj_close = download_and_clean_prices(tickers, start_date, end_date)

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
        

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Volatility': results[1, :],
            'Return': results[0, :],
            'Sharpe Ratio': results[2, :]
        })

        # Plot Efficient Frontier using Plotly
        fig = px.scatter(
            plot_df,
            x='Volatility',
            y='Return',
            color='Sharpe Ratio',
            color_continuous_scale='viridis',
            title="Efficient Frontier (Interactive)",
            labels={'Volatility': 'Portfolio Volatility (Risk)', 'Return': 'Portfolio Return'},
            width=900,
            height=600,
        )
        # Customize hover info
        fig.update_traces(
            hovertemplate=
            '<b>Return:</b> %{y:.2%}<br>' +
            '<b>Volatility:</b> %{x:.2%}<br>' +
            '<b>Sharpe Ratio:</b> %{marker.color:.2f}<br>' +
            '<extra></extra>'
        )

        # Highlight the optimal portfolio point
        fig.add_scatter(
            x=[results[1, max_sharpe_idx]],
            y=[results[0, max_sharpe_idx]],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Optimal Portfolio'
        )

        st.plotly_chart(fig)

       

    with tab3:
        st.header("Optimal Portfolio Details")
        st.subheader("🔹 Optimal Weights")
        for ticker, weight in zip(tickers, optimal_weights):
            st.write(f"**{ticker}:** {weight:.2%}")
        
        # Convert optimal weights to a DataFrame for download
        weights_df = pd.DataFrame({
            'Ticker': tickers,
            'Optimal Weight': optimal_weights
        })

        # Display DataFrame
        st.dataframe(weights_df.style.format({"Optimal Weight": "{:.2%}"}))

        # Create download button
        csv = weights_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📌 Download Portfolio as CSV",
            data=csv,
            file_name='optimal_portfolio.csv',
            mime='text/csv',
        )


        st.subheader("🔹 Performance Metrics")
        col1, col2 = st.columns(2)
        col1.metric(label="Expected Annual Return", value=f"{optimal_return:.2%}")
        col2.metric(label="Expected Annual Risk (Volatility)", value=f"{optimal_risk:.2%}")

except Exception as e:
    st.error(f"An error occurred: {e}")