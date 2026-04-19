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
from scipy.optimize import minimize

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

lookback_options = {
    "1 Year": 1,
    "2 Years": 2,
    "3 Years": 3,
    "5 Years": 5,
    "Full Sample": None
}

available_lookbacks = []
years_available = (end_date - start_date).days / 365.25

for label, yrs in lookback_options.items():
    if yrs is None or years_available >= yrs:
        available_lookbacks.append(label)

lookback_choice = st.sidebar.selectbox(
    "Optimization Lookback Window",
    available_lookbacks,
    index=len(available_lookbacks) - 1
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

    Download adjusted close prices for selected tickers plus the S&P 500 benchmark
    in one batch request using yfinance.

    """
    all_tickers = tickers + ["^GSPC"]
    failed = []

    try:
        raw = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False
        )
    except Exception as e:
        st.error(f"Download failed: {e}")
        return pd.DataFrame(), all_tickers

    if raw.empty:
        return pd.DataFrame(), all_tickers

    # Multi-ticker download
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        else:
            return pd.DataFrame(), all_tickers
    else:
        # Single-ticker fallback
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
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

if benchmark not in prices.columns:
    st.error("S&P 500 benchmark (^GSPC) failed to download. Please try again.")
    st.stop()

# Compute returns
returns = prices.pct_change(fill_method=None).dropna()
stock_returns = returns[valid_tickers]
benchmark_returns = returns[benchmark]

# Daily risk-free rate
rf_daily = rf_annual / 252

def downside_deviation(series, rf_daily):
    downside = series[series < rf_daily] - rf_daily
    if len(downside) == 0:
        return np.nan
    return downside.std() * np.sqrt(252)

def sortino_ratio(series, rf_daily, rf_annual):
    dd = downside_deviation(series, rf_daily)
    if dd == 0 or pd.isna(dd):
        return np.nan
    ann_ret = series.mean() * 252
    return (ann_ret - rf_annual) / dd

def wealth_index(series, initial=10000):
    return (1 + series).cumprod() * initial

def drawdown_series(series):
    wealth = wealth_index(series, 1.0)
    running_peak = wealth.cummax()
    dd = wealth / running_peak - 1
    return dd

def max_drawdown(series):
    return drawdown_series(series).min()

def portfolio_return(weights, mu):
    return float(weights @ mu)

def portfolio_vol(weights, cov):
    return float(np.sqrt(weights @ cov @ weights))

def negative_sharpe(weights, mu, cov, rf_annual):
    port_ret = portfolio_return(weights, mu)
    port_vol_val = portfolio_vol(weights, cov)
    return -((port_ret - rf_annual) / port_vol_val)

def compute_portfolio_returns(weights, returns_df):
    return returns_df @ weights

def portfolio_metrics(weights, returns_df, rf_daily, rf_annual):
    port_rets = compute_portfolio_returns(weights, returns_df)
    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_annual) / ann_vol
    sortino = sortino_ratio(port_rets, rf_daily, rf_annual)
    mdd = max_drawdown(port_rets)
    wealth = wealth_index(port_rets)
    return {
        "returns": port_rets,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "wealth": wealth
    }

def prc(weights, cov):
    port_var = weights @ cov @ weights
    return (weights * (cov @ weights)) / port_var

def optimize_gmv(mu, cov):
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    result = minimize(
        lambda w: portfolio_vol(w, cov),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result

def optimize_tangency(mu, cov, rf_annual):
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    result = minimize(
        lambda w: negative_sharpe(w, mu, cov, rf_annual),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result

def optimize_target_return(mu, cov, target_return):
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_return(w, mu) - target_return}
    ]
    result = minimize(
        lambda w: portfolio_vol(w, cov),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result

if lookback_choice == "Full Sample":
    opt_stock_returns = stock_returns.copy()
else:
    lookback_years = lookback_options[lookback_choice]
    lookback_start = stock_returns.index.max() - pd.DateOffset(years=lookback_years)
    opt_stock_returns = stock_returns.loc[lookback_start:]

opt_mu = opt_stock_returns.mean() * 252
opt_cov = opt_stock_returns.cov() * 252

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

    wealth_df = (1 + returns[valid_tickers + [benchmark]]).cumprod() * 10000

    if wealth_choices:
        fig = go.Figure()
        for col in wealth_choices:
            fig.add_trace(go.Scatter(
                x=wealth_df.index,
                y=wealth_df[col],
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(hist_fig, width="stretch")

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

    rolling_window = st.selectbox(
        "Rolling Volatility Window",
        [30, 60, 90, 120],
        index=1
    )

    rolling_vol = returns[valid_tickers + [benchmark]].rolling(rolling_window).std() * np.sqrt(252)

    st.subheader("Rolling Annualized Volatility")
    fig_roll = go.Figure()
    for col in rolling_vol.columns:
        fig_roll.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[col],
            mode="lines",
            name=col
        ))
    fig_roll.update_layout(
        title=f"Rolling Annualized Volatility ({rolling_window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_roll, width="stretch")

    st.divider()

    dd_stock = st.selectbox("Select stock for drawdown analysis", valid_tickers, key="dd_stock")
    dd = drawdown_series(stock_returns[dd_stock])

    col1, col2 = st.columns([1, 3])
    col1.metric("Max Drawdown", f"{dd.min():.2%}")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index,
        y=dd,
        mode="lines",
        name=dd_stock
    ))
    fig_dd.update_layout(
        title=f"Drawdown Chart: {dd_stock}",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        height=400
    )
    col2.plotly_chart(fig_dd, width="stretch")

    st.divider()

    sharpe_table = {}
    sortino_table = {}

    for col in returns[valid_tickers + [benchmark]].columns:
        series = returns[col]
        ann_ret = series.mean() * 252
        ann_vol = series.std() * np.sqrt(252)
        sharpe_table[col] = (ann_ret - rf_annual) / ann_vol
        sortino_table[col] = sortino_ratio(series, rf_daily, rf_annual)

    risk_metrics = pd.DataFrame({
        "Sharpe Ratio": pd.Series(sharpe_table),
        "Sortino Ratio": pd.Series(sortino_table)
    })

    st.subheader("Risk-Adjusted Metrics")
    st.dataframe(risk_metrics.style.format("{:.3f}"))

with tab3:
    st.header("Correlation and Covariance Analysis")

    corr = returns[valid_tickers].corr()

    st.subheader("Correlation Heatmap")
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    fig_corr.update_layout(
        title="Correlation Matrix of Daily Returns",
        xaxis_title="Stock",
        yaxis_title="Stock",
        height=500
    )
    st.plotly_chart(fig_corr, width="stretch")

    st.divider()

    col1, col2, col3 = st.columns(3)
    rc_stock1 = col1.selectbox("Stock 1", valid_tickers, key="rc1")
    rc_stock2 = col2.selectbox("Stock 2", valid_tickers, index=1 if len(valid_tickers) > 1 else 0, key="rc2")
    rc_window = col3.selectbox("Rolling Correlation Window", [30, 60, 90, 120], index=1)

    rolling_corr = stock_returns[rc_stock1].rolling(rc_window).corr(stock_returns[rc_stock2])

    st.subheader("Rolling Correlation")
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr,
        mode="lines",
        name=f"{rc_stock1} vs {rc_stock2}"
    ))
    fig_rc.update_layout(
        title=f"Rolling Correlation: {rc_stock1} vs {rc_stock2} ({rc_window} Days)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_rc, width="stretch")

    with st.expander("Show Covariance Matrix"):
        cov_matrix = returns[valid_tickers].cov()
        st.dataframe(cov_matrix.style.format("{:.6f}"))

with tab4:
    st.header("Portfolio Construction and Optimization")

    st.caption("This section uses scipy.optimize.minimize with no-short-selling constraints (weights between 0 and 1, sum to 1).")

    n_assets = len(valid_tickers)
    ew_weights = np.ones(n_assets) / n_assets

    gmv_result = optimize_gmv(opt_mu.values, opt_cov.values)
    tan_result = optimize_tangency(opt_mu.values, opt_cov.values, rf_annual)

    if not gmv_result.success:
        st.error("GMV optimization failed.")
        st.stop()

    if not tan_result.success:
        st.error("Tangency optimization failed.")
        st.stop()

    w_gmv = gmv_result.x
    w_tan = tan_result.x

    ew_metrics = portfolio_metrics(ew_weights, stock_returns, rf_daily, rf_annual)
    gmv_metrics = portfolio_metrics(w_gmv, stock_returns, rf_daily, rf_annual)
    tan_metrics = portfolio_metrics(w_tan, stock_returns, rf_daily, rf_annual)

    st.subheader("Equal-Weight Portfolio Metrics")
    ew_table = pd.DataFrame({
        "Annualized Return": [ew_metrics["ann_return"]],
        "Annualized Volatility": [ew_metrics["ann_vol"]],
        "Sharpe Ratio": [ew_metrics["sharpe"]],
        "Sortino Ratio": [ew_metrics["sortino"]],
        "Max Drawdown": [ew_metrics["max_drawdown"]],
    }, index=["Equal Weight"])
    st.dataframe(ew_table.style.format({
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Sortino Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}"
    }))

    st.divider()

    weights_df = pd.DataFrame({
        "Equal Weight": ew_weights,
        "GMV": w_gmv,
        "Tangency": w_tan
    }, index=valid_tickers)

    st.subheader("Portfolio Weights")
    st.dataframe(weights_df.style.format("{:.2%}"))

    fig_w = go.Figure()
    for col in weights_df.columns:
        fig_w.add_trace(go.Bar(
            x=weights_df.index,
            y=weights_df[col],
            name=col
        ))
    fig_w.update_layout(
        barmode="group",
        title="Portfolio Weights",
        xaxis_title="Stock",
        yaxis_title="Weight",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_w, width="stretch")

    st.divider()

    opt_summary = pd.DataFrame({
        "Annualized Return": [gmv_metrics["ann_return"], tan_metrics["ann_return"]],
        "Annualized Volatility": [gmv_metrics["ann_vol"], tan_metrics["ann_vol"]],
        "Sharpe Ratio": [gmv_metrics["sharpe"], tan_metrics["sharpe"]],
        "Sortino Ratio": [gmv_metrics["sortino"], tan_metrics["sortino"]],
        "Max Drawdown": [gmv_metrics["max_drawdown"], tan_metrics["max_drawdown"]],
    }, index=["GMV", "Tangency"])
    st.subheader("Optimized Portfolio Metrics")
    st.dataframe(opt_summary.style.format({
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Sortino Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}"
    }))

    st.divider()

    prc_gmv = prc(w_gmv, opt_cov.values)
    prc_tan = prc(w_tan, opt_cov.values)

    st.subheader("Percentage Risk Contribution (PRC)")
    st.write("PRC shows how much of total portfolio risk each stock contributes. A stock can have a smaller weight than another stock but still contribute more risk if it is more volatile or more correlated.")

    prc_df = pd.DataFrame({
        "GMV PRC": prc_gmv,
        "Tangency PRC": prc_tan
    }, index=valid_tickers)
    st.dataframe(prc_df.style.format("{:.2%}"))

    fig_prc = go.Figure()
    fig_prc.add_trace(go.Bar(x=valid_tickers, y=prc_gmv, name="GMV PRC"))
    fig_prc.add_trace(go.Bar(x=valid_tickers, y=prc_tan, name="Tangency PRC"))
    fig_prc.update_layout(
        barmode="group",
        title="Percentage Risk Contribution",
        xaxis_title="Stock",
        yaxis_title="PRC",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_prc, width="stretch")

    st.divider()

    frontier_targets = np.linspace(opt_mu.min(), opt_mu.max(), 40)
    frontier_rets = []
    frontier_vols = []

    for target in frontier_targets:
        res = optimize_target_return(opt_mu.values, opt_cov.values, target)
        if res.success:
            frontier_rets.append(portfolio_return(res.x, opt_mu.values))
            frontier_vols.append(portfolio_vol(res.x, opt_cov.values))

    custom_weights = np.array([
        st.session_state.get(f"custom_weight_{ticker}", 1 / n_assets)
        for ticker in valid_tickers
    ])
    custom_weights = custom_weights / custom_weights.sum()
    custom_metrics = portfolio_metrics(custom_weights, stock_returns, rf_daily, rf_annual)

    benchmark_ann_ret = benchmark_returns.mean() * 252
    benchmark_ann_vol = benchmark_returns.std() * np.sqrt(252)

    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_rets,
        mode="lines",
        name="Efficient Frontier"
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[ew_metrics["ann_vol"]],
        y=[ew_metrics["ann_return"]],
        mode="markers+text",
        text=["Equal Weight"],
        textposition="top center",
        name="Equal Weight"
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[gmv_metrics["ann_vol"]],
        y=[gmv_metrics["ann_return"]],
        mode="markers+text",
        text=["GMV"],
        textposition="top center",
        name="GMV"
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[tan_metrics["ann_vol"]],
        y=[tan_metrics["ann_return"]],
        mode="markers+text",
        text=["Tangency"],
        textposition="top center",
        name="Tangency"
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[custom_metrics["ann_vol"]],
        y=[custom_metrics["ann_return"]],
        mode="markers+text",
        text=["Custom"],
        textposition="top center",
        name="Custom Portfolio"
    ))

    fig_frontier.add_trace(go.Scatter(
        x=[benchmark_ann_vol],
        y=[benchmark_ann_ret],
        mode="markers+text",
        text=["S&P 500"],
        textposition="top center",
        name="S&P 500"
    ))

    stock_ann_ret = stock_returns.mean() * 252
    stock_ann_vol = stock_returns.std() * np.sqrt(252)
    fig_frontier.add_trace(go.Scatter(
        x=stock_ann_vol.values,
        y=stock_ann_ret.values,
        mode="markers+text",
        text=stock_ann_ret.index.tolist(),
        textposition="top center",
        name="Individual Stocks"
    ))

    cal_x = np.linspace(0, max(frontier_vols + [tan_metrics["ann_vol"]]) * 1.1, 50)
    cal_y = rf_annual + ((tan_metrics["ann_return"] - rf_annual) / tan_metrics["ann_vol"]) * cal_x
    fig_frontier.add_trace(go.Scatter(
        x=cal_x,
        y=cal_y,
        mode="lines",
        name="Capital Allocation Line"
    ))

    fig_frontier.update_layout(
        title="Efficient Frontier and Capital Allocation Line",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
        height=550
    )
    st.subheader("Efficient Frontier")
    st.write("The efficient frontier shows the highest return available for each risk level. The CAL starts at the risk-free rate and touches the frontier at the tangency portfolio.")
    st.plotly_chart(fig_frontier, width="stretch")

    st.divider()

    wealth_compare = pd.DataFrame({
        "Equal Weight": ew_metrics["wealth"],
        "GMV": gmv_metrics["wealth"],
        "Tangency": tan_metrics["wealth"],
        "Custom": custom_metrics["wealth"],
        "S&P 500": wealth_index(benchmark_returns)
    })

    st.subheader("Portfolio Wealth Comparison")
    fig_wc = go.Figure()
    for col in wealth_compare.columns:
        fig_wc.add_trace(go.Scatter(x=wealth_compare.index, y=wealth_compare[col], mode="lines", name=col))
    fig_wc.update_layout(
        title="Cumulative Wealth: Portfolios vs S&P 500",
        xaxis_title="Date",
        yaxis_title="Wealth Index ($10,000 Start)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_wc, width="stretch")

    st.subheader("Portfolio Summary Comparison")
    summary_compare = pd.DataFrame({
        "Annualized Return": [
            ew_metrics["ann_return"],
            gmv_metrics["ann_return"],
            tan_metrics["ann_return"],
            custom_metrics["ann_return"],
            benchmark_ann_ret
        ],
        "Annualized Volatility": [
            ew_metrics["ann_vol"],
            gmv_metrics["ann_vol"],
            tan_metrics["ann_vol"],
            custom_metrics["ann_vol"],
            benchmark_ann_vol
        ],
        "Sharpe Ratio": [
            ew_metrics["sharpe"],
            gmv_metrics["sharpe"],
            tan_metrics["sharpe"],
            custom_metrics["sharpe"],
            (benchmark_ann_ret - rf_annual) / benchmark_ann_vol
        ],
        "Sortino Ratio": [
            ew_metrics["sortino"],
            gmv_metrics["sortino"],
            tan_metrics["sortino"],
            custom_metrics["sortino"],
            sortino_ratio(benchmark_returns, rf_daily, rf_annual)
        ],
        "Max Drawdown": [
            ew_metrics["max_drawdown"],
            gmv_metrics["max_drawdown"],
            tan_metrics["max_drawdown"],
            custom_metrics["max_drawdown"],
            max_drawdown(benchmark_returns)
        ]
    }, index=["Equal Weight", "GMV", "Tangency", "Custom", "S&P 500"])

    st.dataframe(summary_compare.style.format({
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Sortino Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}"
    }))

with tab5:
    st.header("Estimation Window Sensitivity and Custom Portfolio")

    st.subheader("Custom Portfolio Builder")
    st.write("Move the sliders, then the weights are normalized so the portfolio always sums to 100%.")

    raw_weights = []
    for ticker in valid_tickers:
        val = st.slider(
            f"{ticker} weight",
            min_value=0.0,
            max_value=1.0,
            value=float(1 / len(valid_tickers)),
            step=0.01,
            key=f"custom_weight_{ticker}"
        )
        raw_weights.append(val)

    raw_weights = np.array(raw_weights)

    if raw_weights.sum() == 0:
        st.error("Weights cannot all be zero.")
        st.stop()

    custom_weights = raw_weights / raw_weights.sum()

    normalized_df = pd.DataFrame({
        "Normalized Weight": custom_weights
    }, index=valid_tickers)

    st.write("Normalized Weights")
    st.dataframe(normalized_df.style.format("{:.2%}"))

    custom_metrics = portfolio_metrics(custom_weights, stock_returns, rf_daily, rf_annual)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Return", f"{custom_metrics['ann_return']:.2%}")
    col2.metric("Volatility", f"{custom_metrics['ann_vol']:.2%}")
    col3.metric("Sharpe", f"{custom_metrics['sharpe']:.3f}")
    col4.metric("Sortino", f"{custom_metrics['sortino']:.3f}")
    col5.metric("Max Drawdown", f"{custom_metrics['max_drawdown']:.2%}")

    st.divider()

    st.subheader("Estimation Window Sensitivity")
    st.write("Optimization results are sensitive to their inputs. Small changes in the historical window used to estimate expected returns and covariances can produce noticeably different portfolio weights and Sharpe ratios.")

    windows = []
    if years_available >= 1:
        windows.append(("1 Year", 1))
    if years_available >= 2:
        windows.append(("2 Years", 2))    
    if years_available >= 3:
        windows.append(("3 Years", 3))
    if years_available >= 5:
        windows.append(("5 Years", 5))
    windows.append(("Full Sample", None))

    sensitivity_rows = []

    for label, yrs in windows:
        if yrs is None:
            temp_returns = stock_returns.copy()
        else:
            temp_start = stock_returns.index.max() - pd.DateOffset(years=yrs)
            temp_returns = stock_returns.loc[temp_start:]

        temp_mu = temp_returns.mean() * 252
        temp_cov = temp_returns.cov() * 252

        gmv_res = optimize_gmv(temp_mu.values, temp_cov.values)
        tan_res = optimize_tangency(temp_mu.values, temp_cov.values, rf_annual)

        if gmv_res.success:
            gmv_w = gmv_res.x
            sensitivity_rows.append({
                "Window": label,
                "Portfolio": "GMV",
                "Return": portfolio_return(gmv_w, temp_mu.values),
                "Volatility": portfolio_vol(gmv_w, temp_cov.values),
                **{ticker: gmv_w[i] for i, ticker in enumerate(valid_tickers)}
            })

        if tan_res.success:
            tan_w = tan_res.x
            tan_ret = portfolio_return(tan_w, temp_mu.values)
            tan_vol = portfolio_vol(tan_w, temp_cov.values)
            sensitivity_rows.append({
                "Window": label,
                "Portfolio": "Tangency",
                "Return": tan_ret,
                "Volatility": tan_vol,
                "Sharpe": (tan_ret - rf_annual) / tan_vol,
                **{ticker: tan_w[i] for i, ticker in enumerate(valid_tickers)}
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    format_dict = {}
    for col in sensitivity_df.columns:
        if col not in ["Window", "Portfolio", "Sharpe"]:
            format_dict[col] = "{:.2%}"
    if "Sharpe" in sensitivity_df.columns:
        format_dict["Sharpe"] = "{:.3f}"

    st.dataframe(sensitivity_df.style.format(format_dict))

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
    st.subheader("Methodology / Assumptions")
    st.markdown("""
    - Uses simple daily returns, not log returns
    - Annualized return = mean daily return × 252
    - Annualized volatility = daily standard deviation × sqrt(252)
    - Sortino uses downside deviation only
    - Risk-free rate input is annualized and converted to a daily rate by dividing by 252
    - Prices are based on adjusted close-equivalent data from Yahoo Finance
    - Portfolio optimization uses no-short-selling constraints (weights between 0 and 1, sum to 1)
    """)
    