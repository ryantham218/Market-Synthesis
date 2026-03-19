import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# Takes a metric name and its value, returns a colour-coded
# rating based on thresholds defined for each metric
def rate_metric(metric_name, value):

    # If yfinance couldn't find the data, return grey
    if value == "N/A":
        return "⚪ N/A"

    # PE Ratio — measures how expensive the stock is relative to earnings
    # Lower is cheaper, higher means market expects strong future growth
    if metric_name == "pe":
        if value < 15:
            return "🟢 Good"
        elif value < 25:
            return "🟡 Fair"
        else:
            return "🔴 Expensive"

    # Dividend Yield — percentage of stock price paid out as dividends annually
    # Higher means more income returned to shareholders
    if metric_name == "dividend_yield":
        if value > 4:
            return "🟢 Good"
        elif value > 2:
            return "🟡 Fair"
        else:
            return "🔴 Low"

    # Debt to Equity — how much debt the company carries vs shareholder equity
    # Lower is safer, higher means more financial risk
    if metric_name == "debt_to_equity":
        if value < 50:
            return "🟢 Good"
        elif value < 100:
            return "🟡 Fair"
        else:
            return "🔴 High"

    # Revenue Growth — year on year percentage change in total revenue
    # Positive means growing, negative means shrinking
    if metric_name == "revenue_growth":
        if value > 10:
            return "🟢 Strong"
        elif value > 0:
            return "🟡 Moderate"
        else:
            return "🔴 Declining"

    # Profit Margin — percentage of revenue that becomes actual profit
    # Higher means the company keeps more of what it earns
    if metric_name == "profit_margin":
        if value > 20:
            return "🟢 Strong"
        elif value > 10:
            return "🟡 Moderate"
        else:
            return "🔴 Weak"

    # Return on Equity — how efficiently the company generates profit
    # from shareholder money. Higher is better
    if metric_name == "roe":
        if value > 15:
            return "🟢 Strong"
        elif value > 8:
            return "🟡 Moderate"
        else:
            return "🔴 Weak"

    # Fallback — if metric name doesn't match anything above
    return "⚪ N/A"


# Converts raw numbers like 89000000000 into readable
# format like $89.00B for display purposes
def format_market_cap(value):
    if value == "N/A":
        return "N/A"
    elif value >= 1_000_000_000:
        # Divide by 1 billion and label with B
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        # Divide by 1 million and label with M
        return f"${value / 1_000_000:.2f}M"
    else:
        # Small enough to display as is with comma formatting
        return f"${value:,}"


# Pulls and cleans all data for a given ticker
# Returns a dictionary of processed values, or None if ticker is invalid
def fetch_stock(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # If both longName and currentPrice are None the ticker is invalid
    if info.get("longName") is None and info.get("currentPrice") is None:
        return None

    # .get() retrieves each value by its key
    # Second argument is the fallback if the key doesn't exist
    name = info.get("longName", "N/A")              # Full company name
    price = info.get("currentPrice", "N/A")         # Current stock price
    # Trailing PE ratio — None if missing
    raw_pe = info.get("trailingPE")
    # Round to 2 decimal places
    pe = round(float(raw_pe), 2) if raw_pe is not None else "N/A"
    # Raw dividend yield (needs fixing)
    raw_yield = info.get("dividendYield", "N/A")
    debt_to_equity = info.get("debtToEquity", "N/A")  # Debt to equity ratio
    raw_revenue_growth = info.get("revenueGrowth", "N/A")  # Raw revenue growth
    raw_profit_margin = info.get("profitMargins", "N/A")   # Raw profit margin
    raw_roe = info.get("returnOnEquity", "N/A")     # Raw return on equity
    # Total market capitalisation
    market_cap = info.get("marketCap", "N/A")
    week52_low = info.get("fiftyTwoWeekLow", "N/A")    # Lowest price in past 52 weeks
    week52_high = info.get("fiftyTwoWeekHigh", "N/A")  # Highest price in past 52 weeks
    raw_recommend = info.get("recommendationKey", "N/A")   # Overall recommendation e.g. "buy", "hold"
    analyst_count = info.get("numberOfAnalystOpinions", "N/A")  # Number of analysts covering the stock

    # yfinance sometimes returns 0.0553 (decimal) sometimes 5.53 (percentage)
    # We normalise to decimal first, then convert to percentage for display
    if raw_yield != "N/A":
        dividend_yield = raw_yield if raw_yield < 1 else raw_yield / 100
        dividend_yield_pct = round(dividend_yield * 100, 2)
    else:
        dividend_yield_pct = "N/A"
        dividend_yield = "N/A"

    # yfinance returns as decimal e.g. 0.12 meaning 12%
    # Multiply by 100 and round to 2 decimal places for display
    if raw_revenue_growth != "N/A":
        revenue_growth = round(raw_revenue_growth * 100, 2)
    else:
        revenue_growth = "N/A"

    # Same as revenue growth — returned as decimal, convert to percentage
    if raw_profit_margin != "N/A":
        profit_margin = round(raw_profit_margin * 100, 2)
    else:
        profit_margin = "N/A"

    # Same as above — returned as decimal, convert to percentage
    if raw_roe != "N/A":
        roe = round(raw_roe * 100, 2)
    else:
        roe = "N/A"

    # Build summary sentences based on each metric's rating
    sentences = []

    # Valuation sentence — based on PE ratio
    if pe != "N/A":
        if pe < 15:
            sentences.append(f"{name} appears attractively valued with a low PE ratio of {pe}.")
        elif pe < 25:
            sentences.append(f"{name} is fairly valued with a PE ratio of {pe}.")
        else:
            sentences.append(f"{name} trades at a premium valuation with a PE ratio of {pe}, suggesting the market expects strong future growth.")

    # Dividend sentence — based on dividend yield
    if dividend_yield_pct != "N/A":
        if dividend_yield_pct > 4:
            sentences.append(f"It offers a strong dividend yield of {dividend_yield_pct}%, making it attractive for income investors.")
        elif dividend_yield_pct > 2:
            sentences.append(f"It pays a modest dividend yield of {dividend_yield_pct}%.")
        else:
            sentences.append(f"The dividend yield of {dividend_yield_pct}% is relatively low.")

    # Profitability sentence — based on profit margin
    if profit_margin != "N/A":
        if profit_margin > 20:
            sentences.append(f"Profitability is strong with a healthy profit margin of {profit_margin}%.")
        elif profit_margin > 10:
            sentences.append(f"Profit margins are moderate at {profit_margin}%.")
        else:
            sentences.append(f"Profit margins are thin at {profit_margin}%, which warrants attention.")

    # Efficiency sentence — based on ROE
    if roe != "N/A":
        if roe > 15:
            sentences.append(f"The company generates returns efficiently with a return on equity of {roe}%.")
        elif roe > 8:
            sentences.append(f"Return on equity is moderate at {roe}%.")
        else:
            sentences.append(f"Return on equity is weak at {roe}%, suggesting the company may not be deploying capital efficiently.")

    # Growth sentence — based on revenue growth
    if revenue_growth != "N/A":
        if revenue_growth > 10:
            sentences.append(f"Revenue is growing strongly at {revenue_growth}% year on year.")
        elif revenue_growth > 0:
            sentences.append(f"Revenue growth is modest at {revenue_growth}% year on year.")
        else:
            sentences.append(f"Revenue has declined {abs(revenue_growth)}% year on year, which is worth monitoring.")

    # Debt sentence — based on debt to equity
    if debt_to_equity != "N/A":
        if debt_to_equity < 50:
            sentences.append(f"The balance sheet is healthy with a low debt to equity ratio of {debt_to_equity}.")
        elif debt_to_equity < 100:
            sentences.append(f"Debt levels are manageable with a debt to equity ratio of {debt_to_equity}.")
        else:
            sentences.append(f"The company carries significant debt with a debt to equity ratio of {debt_to_equity}.")

    # Return all processed data as a dictionary so display_stock can use it
    return {
        "stock": stock,
        "name": name,
        "price": price,
        "pe": pe,
        "dividend_yield_pct": dividend_yield_pct,
        "dividend_yield": dividend_yield,
        "debt_to_equity": debt_to_equity,
        "revenue_growth": revenue_growth,
        "profit_margin": profit_margin,
        "roe": roe,
        "market_cap": market_cap,
        "week52_low": week52_low,
        "week52_high": week52_high,
        "raw_recommend": raw_recommend,
        "analyst_count": analyst_count,
        "summary": " ".join(sentences)
    }

# Renders all metrics, chart and summary for one stock
# Called once for single mode, twice side by side for comparison mode
def display_stock(data):

    # Displays company name, price and market cap at the top
    # Truncate long company names to prevent layout breaking in comparison mode
    st.markdown(f"""
        <div style="font-size: 1.5em; font-weight: 700; 
        overflow: hidden; white-space: nowrap; 
        text-overflow: ellipsis; max-width: 100%;">
            {data["name"]}
        </div>
    """, unsafe_allow_html=True)

    # Split into two columns so price and market cap sit side by side
    col_price, col_cap = st.columns(2)
    with col_price:
        st.metric(label="Current Price", value=f"${data['price']}")
    with col_cap:
        st.metric(label="Market Cap", value=format_market_cap(data["market_cap"]))

    # 52 Week High and Low
    # Shows where the current price sits between the yearly low and high
    week52_low = data["week52_low"]
    week52_high = data["week52_high"]
    current_price = data["price"]

    if week52_low != "N/A" and week52_high != "N/A" and current_price != "N/A":

        # Calculate how far current price is between low and high as a percentage
        # e.g. if low is 40, high is 60, price is 50 — position is 50%
        position = (current_price - week52_low) / (week52_high - week52_low)

# Build the HTML separately to avoid quote conflicts
        bar_html = (
            '<div style="display: flex; align-items: center; gap: 16px; margin: 16px 0;">'
            '<div style="min-width: 80px;">'
            '<div style="font-size: 0.8em; opacity: 0.6;">52W Low</div>'
            f'<div style="font-size: 1.3em; font-weight: 600;">${week52_low}</div>'
            '</div>'
            '<div style="flex: 1;">'
            '<div style="text-align: center; font-size: 0.85em; opacity: 0.7; margin-bottom: 6px;">Current Price Position</div>'
            '<div style="background-color: rgba(128,128,128,0.2); border-radius: 999px; height: 12px; width: 100%;">'
            f'<div style="background-color: #0068c9; width: {position * 100:.1f}%; height: 12px; border-radius: 999px;"></div>'
            '</div>'
            f'<div style="text-align: center; font-size: 0.8em; opacity: 0.6; margin-top: 4px;">{position * 100:.1f}% of 52W range</div>'
            '</div>'
            '<div style="min-width: 80px; text-align: right;">'
            '<div style="font-size: 0.8em; opacity: 0.6;">52W High</div>'
            f'<div style="font-size: 1.3em; font-weight: 600;">${week52_high}</div>'
            '</div>'
            '</div>'
        )
        st.markdown(bar_html, unsafe_allow_html=True)
        
    # Analyst Recommendations
    # Shows the overall analyst consensus and number of analysts covering the stock
    raw_recommend = data["raw_recommend"]
    analyst_count = data["analyst_count"]

    if raw_recommend != "N/A":

        # Map yfinance recommendation keys to readable labels and colours
        recommend_map = {
            "strongBuy":  ("Strong Buy",  "#22c55e"),   # Green
            "buy":        ("Buy",         "#86efac"),   # Light green
            "hold":       ("Hold",        "#facc15"),   # Yellow
            "sell":       ("Sell",        "#f97316"),   # Orange
            "strongSell": ("Strong Sell", "#ef4444"),   # Red
        }

        # Get the label and colour, fallback to grey if key not recognised
        label, colour = recommend_map.get(raw_recommend, (raw_recommend.title(), "#9ca3af"))

        analyst_text = f"Based on {analyst_count} analysts" if analyst_count != "N/A" else ""

        recommend_html = (
            '<div style="display: flex; align-items: center; gap: 12px; margin: 16px 0;">'
            '<div style="font-size: 0.85em; opacity: 0.6;">Analyst Consensus</div>'
            f'<div style="background-color: {colour}; color: #000; font-weight: 700; '
            f'padding: 4px 14px; border-radius: 999px; font-size: 0.95em;">{label}</div>'
            f'<div style="font-size: 0.8em; opacity: 0.5;">{analyst_text}</div>'
            '</div>'
        )
        st.markdown(recommend_html, unsafe_allow_html=True)
    st.divider()

    st.subheader("Valuation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="PE Ratio", value=data["pe"])
        st.write(rate_metric("pe", data["pe"]))

    with col2:
        st.metric(label="Dividend Yield", value=f"{data['dividend_yield_pct']}%")
        st.write(rate_metric("dividend_yield", data["dividend_yield_pct"]))

    with col3:
        st.metric(label="Debt to Equity", value=data["debt_to_equity"])
        st.write(rate_metric("debt_to_equity", data["debt_to_equity"]))

    st.divider()

    st.subheader("Performance")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(label="Revenue Growth", value=f"{data['revenue_growth']}%")
        st.write(rate_metric("revenue_growth", data["revenue_growth"]))

    with col5:
        st.metric(label="Profit Margin", value=f"{data['profit_margin']}%")
        st.write(rate_metric("profit_margin", data["profit_margin"]))

    with col6:
        st.metric(label="Return on Equity", value=f"{data['roe']}%")
        st.write(rate_metric("roe", data["roe"]))

    st.divider()

    # Price History Chart
    # Pulls 1 year of daily closing prices and plots as a line chart
    st.subheader("Price History (1 Year)")

    # .history() fetches historical price data for a given period
    # period="1y" means last 1 year, interval="1d" means daily data points
    history = data["stock"].history(period="1y", interval="1d")

    # history is a dataframe — a table of data with rows and columns
    # We only need the Close column — the closing price each day
    if not history.empty:

        # Create a plotly figure object
        fig = go.Figure()

        # Add a line trace using the date as x axis and closing price as y axis
        # history.index contains the dates, history["Close"] contains the prices
        fig.add_trace(go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",           # Draw as a line not dots
            name="Close Price",
            line=dict(color="#0068c9", width=2)  # Blue line
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (SGD)",
            hovermode="x unified",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),

            # Transparent backgrounds so it adapts to light and dark mode
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",

            hoverlabel=dict(
                bgcolor="rgba(30,30,30,0.8)",
                bordercolor="rgba(128,128,128,0.2)",
                font=dict(color="#ffffff")
            ),

            # X axis
            xaxis=dict(
                tickformat="%b %Y",
                tickangle=-45,
                nticks=12,
                showgrid=False,
                linecolor="rgba(128,128,128,0.3)",
                tickfont=dict(color="rgba(128,128,128,0.8)"),
                # Colour of the vertical hover line
                spikecolor="rgba(128,128,128,0.4)",
                spikethickness=1,
                spikedash="dot",
                spikesnap="cursor",
            ),

            # Y axis
            yaxis=dict(
                tickprefix="$",
                tickformat=",.2f",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                linecolor="rgba(128,128,128,0.3)",
                tickfont=dict(color="rgba(128,128,128,0.8)"),
                showspikes=False,
            ),

            shapes=[dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="rgba(128,128,128,0.3)", width=1)
            )]
        )

        # Render the chart in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)

    else:
        # If no historical data is available show a simple message
        st.write("No price history available for this ticker.")

    st.divider()

    # Auto-generates a plain English analysis based on the ratings
    st.subheader("Summary")

    # Join all sentences into one paragraph and display
    st.write(data["summary"])

    # Final disclaimer on its own line
    st.caption("⚠️ This analysis is auto-generated and should not be taken as financial advice.")


# Making the page
st.set_page_config(page_title="Market Synthesis", page_icon="📈", layout="wide")

st.title("📈 Market Synthesis")
st.subheader("SGX Stock Screener")

# Toggle between single stock and comparison mode
mode = st.radio("Mode", ["Single Stock", "Compare Two Stocks"], horizontal=True)

if mode == "Single Stock":

    st.write("Enter a Singapore stock ticker to analyse its key metrics.")
    ticker = st.text_input("Stock Ticker (e.g. D05.SI for DBS)", "").upper()

    # Auto-append .SI if user forgot it
    if ticker and not ticker.endswith(".SI"):
        ticker = ticker + ".SI"

    # Everything below only runs if the user has typed something
    # Prevents the app from crashing on an empty ticker
    if ticker:

        # Fetch data from Yahoo Finance
        # st.spinner shows a loading animation while this runs
        with st.spinner("Fetching data..."):
            data = fetch_stock(ticker)

        # If both longName and currentPrice are None the ticker is invalid
        if data is None:
            st.error("❌ Ticker not found. Please check and try again.")
            st.stop()  # Stops the rest of the code from running

        display_stock(data)

else:

    # Two input boxes side by side for comparison mode
    st.write("Enter two Singapore stock tickers to compare them side by side.")
    col_a, col_divider, col_b = st.columns([10, 1, 10])
    with col_a:
        ticker_a = st.text_input("First Stock Ticker (e.g. D05.SI for DBS)", "").upper()
        # Auto-append .SI if user forgot it
        if ticker_a and not ticker_a.endswith(".SI"):
            ticker_a = ticker_a + ".SI"
    with col_b:
        ticker_b = st.text_input("Second Stock Ticker (e.g. O39.SI for OCBC)", "").upper()
        # Auto-append .SI if user forgot it
        if ticker_b and not ticker_b.endswith(".SI"):
            ticker_b = ticker_b + ".SI"

    # Everything below only runs if both tickers have been entered
    if ticker_a and ticker_b:

        # Fetch data from Yahoo Finance
        # st.spinner shows a loading animation while this runs
        with st.spinner("Fetching data..."):
            data_a = fetch_stock(ticker_a)
            data_b = fetch_stock(ticker_b)

        # If both longName and currentPrice are None the ticker is invalid
        if data_a is None:
            st.error(f"❌ {ticker_a} not found. Please check and try again.")
            st.stop()  # Stops the rest of the code from running
        if data_b is None:
            st.error(f"❌ {ticker_b} not found. Please check and try again.")
            st.stop()  # Stops the rest of the code from running

        # Display both stocks side by side in two columns
        col_a, col_divider, col_b = st.columns([10, 1, 10])
        with col_a:
            display_stock(data_a)
        with col_divider:
            st.markdown(
                '<div style="border-left: 1px solid rgba(128,128,128,0.3); height: 100%; min-height: 800px; margin: auto;"></div>',
                unsafe_allow_html=True
            )
        with col_b:
            display_stock(data_b)