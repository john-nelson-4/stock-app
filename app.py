# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Portfolio Settings")

ticker_text = st.sidebar.text_input(
    "Enter 3 to 10 stock tickers (comma separated)",
    value="AAPL,MSFT,JPM"
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=date.today() - timedelta(days=365 * 5)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)

rf_annual = st.sidebar.number_input(
    "Annual Risk-Free Rate",
    min_value=0.0,
    max_value=0.20,
    value=0.02,
    step=0.005
)

tickers = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
benchmark = "^GSPC"

# -- Validation ------------------------------------------
if len(tickers) < 3 or len(tickers) > 10:
    st.error("Please enter between 3 and 10 stock tickers.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < 365 * 2:
    st.error("Please choose at least a 2-year date range.")
    st.stop()

# -- Data download ----------------------------------------
#It now downloads all user-entered stocks, not just one. Also-uses adjusted close prices.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(tickers: list[str], start_date: date, end_date: date) -> tuple[pd.DataFrame, list[str]]:
    """
    Download adjusted close-equivalent prices for selected tickers plus S&P 500 benchmark
    in one batch request.

    With auto_adjust=True, yfinance adjusts prices for splits/dividends and stores them in 'Close'.
    """
    all_tickers = tickers + ["^GSPC"]
    failed = []

    try:
        raw = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False
        )
    except Exception as e:
        st.error(f"Download failed: {e}")
        return pd.DataFrame(), all_tickers

    if raw.empty:
        return pd.DataFrame(), all_tickers

    # Multi-ticker download
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            return pd.DataFrame(), all_tickers
    else:
        # Single-ticker fallback
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            prices.columns = [all_tickers[0]]
        else:
            return pd.DataFrame(), all_tickers

    # Identify failures
    for t in all_tickers:
        if t not in prices.columns:
            failed.append(t)
        elif prices[t].dropna().empty:
            failed.append(t)

    good_cols = [c for c in prices.columns if c not in failed]
    prices = prices[good_cols]

    return prices, failed

# -- Main logic -------------------------------------------
prices, failed = load_data(tickers, start_date, end_date)

if failed:
    st.warning(f"These tickers failed or returned no usable data: {', '.join(failed)}")

if prices.empty:
    st.error("No usable data downloaded. Please check your tickers.")
    st.stop()

# Keep only valid stock tickers that actually downloaded
valid_tickers = [t for t in tickers if t in prices.columns]
benchmark = "^GSPC"

if len(valid_tickers) < 3:
    st.error("Fewer than 3 valid stock tickers remain after download. Please try again.")
    st.stop()

# Handle missing data: drop any stock ticker with more than 5% missing values
missing_pct = prices[valid_tickers].isna().mean()
drop_cols = missing_pct[missing_pct > 0.05].index.tolist()

if drop_cols:
    st.warning(f"Dropping tickers with more than 5% missing data: {', '.join(drop_cols)}")
    valid_tickers = [t for t in valid_tickers if t not in drop_cols]

if len(valid_tickers) < 3:
    st.error("Fewer than 3 valid stock tickers remain after handling missing data.")
    st.stop()

# Keep valid stocks plus benchmark, then align dates
keep_cols = valid_tickers + ([benchmark] if benchmark in prices.columns else [])
prices = prices[keep_cols].dropna()

if prices.empty:
    st.error("No overlapping data remains after alignment.")
    st.stop()

# Compute returns
returns = prices.pct_change(fill_method=None).dropna()
stock_returns = returns[valid_tickers]
benchmark_returns = returns[benchmark]

# Daily risk-free rate
rf_daily = rf_annual / 252

# -- App sections -----------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Risk Analysis",
    "Correlation",
    "Portfolio Optimization",
    "Sensitivity & Custom Portfolio",
    "About"
])

with tab1:
    st.header("Return Computation and Exploratory Analysis")

    # Summary statistics
    summary_stats = pd.DataFrame({
        "Annualized Mean Return": returns[valid_tickers + [benchmark]].mean() * 252,
        "Annualized Volatility": returns[valid_tickers + [benchmark]].std() * np.sqrt(252),
        "Skewness": returns[valid_tickers + [benchmark]].skew(),
        "Kurtosis": returns[valid_tickers + [benchmark]].kurt(),
        "Min Daily Return": returns[valid_tickers + [benchmark]].min(),
        "Max Daily Return": returns[valid_tickers + [benchmark]].max()
    })

    st.subheader("Summary Statistics")
    st.dataframe(summary_stats.style.format({
        "Annualized Mean Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Skewness": "{:.3f}",
        "Kurtosis": "{:.3f}",
        "Min Daily Return": "{:.2%}",
        "Max Daily Return": "{:.2%}"
    }))

    st.divider()

    # Wealth index
    st.subheader("Growth of $10,000")

    wealth_choices = st.multiselect(
        "Select series to display",
        options=valid_tickers + [benchmark],
        default=valid_tickers + [benchmark]
    )

    wealth_index = (1 + returns[valid_tickers + [benchmark]]).cumprod() * 10000

    if wealth_choices:
        fig = go.Figure()
        for col in wealth_choices:
            fig.add_trace(go.Scatter(
                x=wealth_index.index,
                y=wealth_index[col],
                mode="lines",
                name=col
            ))

        fig.update_layout(
            title="Cumulative Wealth Index",
            xaxis_title="Date",
            yaxis_title="Value of $10,000 Investment",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Distribution section
    st.subheader("Distribution Analysis")

    selected_stock = st.selectbox("Choose a stock", valid_tickers)
    dist_view = st.radio(
        "Select view",
        ["Histogram + Normal Curve", "Q-Q Plot"],
        horizontal=True
    )

    selected_returns = stock_returns[selected_stock].dropna()

    if dist_view == "Histogram + Normal Curve":
        hist_fig = go.Figure()

        hist_fig.add_trace(go.Histogram(
            x=selected_returns,
            histnorm="probability density",
            name="Daily Returns",
            opacity=0.7
        ))

        x_vals = np.linspace(selected_returns.min(), selected_returns.max(), 200)
        mu = selected_returns.mean()
        sigma = selected_returns.std()
        y_vals = stats.norm.pdf(x_vals, mu, sigma)

        hist_fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name="Normal Curve"
        ))

        hist_fig.update_layout(
            title=f"{selected_stock} Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    else:
        qq = stats.probplot(selected_returns, dist="norm")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(qq[0][0], qq[0][1], s=12)
        ax.plot(qq[0][0], qq[1][1] + qq[1][0] * qq[0][0], color="red")
        ax.set_title(f"{selected_stock} Q-Q Plot")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        st.pyplot(fig)

with tab2:
    st.header("Risk Analysis")

    # Annualized metrics
    ann_return = returns[valid_tickers].mean() * 252
    ann_vol = returns[valid_tickers].std() * np.sqrt(252)
    sharpe = (ann_return - rf_annual) / ann_vol

    # Combine into table
    risk_df = pd.DataFrame({
        "Annual Return": ann_return,
        "Volatility": ann_vol,
        "Sharpe Ratio": sharpe
    })

    st.subheader("Risk Metrics")
    st.dataframe(risk_df.style.format({
        "Annual Return": "{:.2%}",
        "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}"
    }))

    st.divider()

    # --- Bar chart: Volatility ---
    st.subheader("Volatility Comparison")

    fig_vol = go.Figure([
        go.Bar(x=ann_vol.index, y=ann_vol.values)
    ])

    fig_vol.update_layout(
        yaxis_title="Volatility",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    # --- Bar chart: Sharpe Ratio ---
    st.subheader("Sharpe Ratio Comparison")

    fig_sharpe = go.Figure([
        go.Bar(x=sharpe.index, y=sharpe.values)
    ])

    fig_sharpe.update_layout(
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig_sharpe, use_container_width=True)

with tab3:
    st.header("Correlation Analysis")

    corr = returns[valid_tickers].corr()

    st.subheader("Correlation Heatmap")

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        text=corr.round(2),
        texttemplate="%{text}",
        showscale=True
    ))

    fig.update_layout(
        title="Correlation Matrix",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Portfolio Optimization")

    # Mean returns & covariance
    mu = returns[valid_tickers].mean() * 252
    cov = returns[valid_tickers].cov() * 252

    ones = np.ones(len(valid_tickers))

    # --- GMV Portfolio ---
    inv_cov = np.linalg.inv(cov)
    w_gmv = inv_cov @ ones / (ones.T @ inv_cov @ ones)

    ret_gmv = w_gmv @ mu
    vol_gmv = np.sqrt(w_gmv @ cov @ w_gmv)

    # --- Tangency Portfolio ---
    excess_mu = mu - rf_annual
    w_tan = inv_cov @ excess_mu / (ones.T @ inv_cov @ excess_mu)

    ret_tan = w_tan @ mu
    vol_tan = np.sqrt(w_tan @ cov @ w_tan)
    sharpe_tan = (ret_tan - rf_annual) / vol_tan

    # Display weights
    st.subheader("Optimal Portfolio Weights")

    gmv_df = pd.DataFrame({
        "Weight": w_gmv
    }, index=valid_tickers)

    tan_df = pd.DataFrame({
        "Weight": w_tan
    }, index=valid_tickers)

    col1, col2 = st.columns(2)

    col1.write("GMV Portfolio")
    col1.dataframe(gmv_df.style.format("{:.2%}"))

    col2.write("Tangency Portfolio")
    col2.dataframe(tan_df.style.format("{:.2%}"))

    st.divider()

    # --- Efficient Frontier ---
    st.subheader("Efficient Frontier")

    target_returns = np.linspace(mu.min(), mu.max(), 50)
    frontier_vol = []

    for r in target_returns:
        w = inv_cov @ (mu - r)
        w = w / np.sum(w)
        vol = np.sqrt(w @ cov @ w)
        frontier_vol.append(vol)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_vol,
        y=target_returns,
        mode="lines",
        name="Efficient Frontier"
    ))

    # Plot GMV
    fig.add_trace(go.Scatter(
        x=[vol_gmv],
        y=[ret_gmv],
        mode="markers",
        name="GMV",
        marker=dict(size=10)
    ))

    # Plot Tangency
    fig.add_trace(go.Scatter(
        x=[vol_tan],
        y=[ret_tan],
        mode="markers",
        name="Tangency",
        marker=dict(size=10)
    ))

    fig.update_layout(
        xaxis_title="Volatility",
        yaxis_title="Return",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Sensitivity & Custom Portfolio")

    st.subheader("Custom Portfolio Builder")

    weights_input = []

    cols = st.columns(len(valid_tickers))

    for i, ticker in enumerate(valid_tickers):
        w = cols[i].number_input(
            f"{ticker} weight",
            min_value=0.0,
            max_value=1.0,
            value=1/len(valid_tickers),
            step=0.01
        )
        weights_input.append(w)

    weights = np.array(weights_input)

    if weights.sum() == 0:
        st.error("Weights cannot all be zero.")
        st.stop()

    weights = weights / weights.sum()  # normalize

    port_return = weights @ (returns[valid_tickers].mean() * 252)
    port_vol = np.sqrt(weights @ (returns[valid_tickers].cov() * 252) @ weights)
    port_sharpe = (port_return - rf_annual) / port_vol

    st.divider()

    st.subheader("Custom Portfolio Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Expected Return", f"{port_return:.2%}")
    col2.metric("Volatility", f"{port_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{port_sharpe:.2f}")

    st.divider()

    st.subheader("Compare to Tangency Portfolio")

    st.write(f"Tangency Return: {ret_tan:.2%}")
    st.write(f"Tangency Volatility: {vol_tan:.2%}")
    st.write(f"Tangency Sharpe: {sharpe_tan:.2f}")
with tab6:
    st.header("About This Project")

    st.write("""
    This dashboard was built to analyze stock performance, portfolio risk, and portfolio optimization
    using historical market data. It allows users to compare multiple stocks, study return patterns,
    evaluate risk, and explore how portfolio construction affects performance.
    """)

    st.subheader("Main Features")
    st.markdown("""
    - Summary statistics for each selected stock
    - Growth of a $10,000 investment over time
    - Distribution analysis with histogram and Q-Q plot
    - Risk metrics including volatility and Sharpe ratio
    - Correlation heatmap
    - Portfolio optimization with GMV and Tangency portfolios
    - Efficient Frontier visualization
    - Custom portfolio builder for user-selected weights
    """)

    st.subheader("Data Source")
    st.write("""
    Market data is pulled from Yahoo Finance using the `yfinance` Python package.
    Prices are adjusted and returns are calculated from the selected date range.
    """)

    st.subheader("Tools Used")
    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - SciPy
    - Matplotlib
    """)

    st.subheader("Why This Matters")
    st.write("""
    This project shows how investors can use data and optimization tools to better understand
    the relationship between risk and return. Instead of just looking at individual stocks,
    the dashboard helps show how combining assets into portfolios can improve decision-making.
    """) 