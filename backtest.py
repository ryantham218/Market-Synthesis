import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# DATA FETCHER
# Downloads historical daily price data for a given ticker
# Returns a clean dataframe with just the closing prices
# ============================================================


def fetch_data(ticker, period):
    stock = yf.Ticker(ticker)

    # Download historical data for the given period
    # period can be "1y", "2y", "5y" etc
    df = stock.history(period=period)

    # Check if we got any data back
    if df.empty:
        return None

    # Keep only the Close column — that's all strategies need
    df = df[["Close"]].copy()

    # Reset index so dates become a regular column instead of the index
    df.reset_index(inplace=True)

    # Rename columns for clarity
    df.columns = ["Date", "Close"]

    return df

# ============================================================
# BUY AND HOLD STRATEGY
# Buys on day one and holds until the last day
# Used as the benchmark to compare all other strategies against
# ============================================================


def strategy_buy_and_hold(df, capital):

    # Buy price is the first day's closing price
    buy_price = df["Close"].iloc[0]

    # Sell price is the last day's closing price
    sell_price = df["Close"].iloc[-1]

    # Calculate how many shares we can buy with our capital
    shares = capital / buy_price

    # Final value is shares multiplied by the sell price
    final_value = shares * sell_price

    # Total return as a percentage
    total_return = ((final_value - capital) / capital) * 100

    # Build a portfolio value line for the chart
    # Shows how the value of our holding changed every day
    df["BuyHold"] = (df["Close"] / buy_price) * capital

    return {
        "strategy": "Buy and Hold",
        "total_return": round(total_return, 2),
        "final_value": round(final_value, 2),
        "trades": 1,                    # Only one trade — the initial buy
        "win_rate": 100 if total_return > 0 else 0,
        "portfolio": df["BuyHold"].tolist()
    }

# ============================================================
# MOVING AVERAGE CROSSOVER STRATEGY
# Uses a fast and slow moving average
# Buy when fast crosses above slow — Golden Cross
# Sell when fast crosses below slow — Death Cross
# ============================================================


def strategy_ma_crossover(df, capital, fast=50, slow=200):

    # Calculate fast (50 day) and slow (200 day) moving averages
    # .rolling(window) computes the average of the last N closing prices each day
    df["MA_fast"] = df["Close"].rolling(window=fast).mean()
    df["MA_slow"] = df["Close"].rolling(window=slow).mean()

    # Track portfolio state
    cash = capital         # Start with full capital in cash
    shares = 0             # No shares held initially
    in_market = False      # Are we currently holding shares
    trades = []            # List of all trades made

    # Loop through every day of price data
    for i in range(1, len(df)):

        # Skip days where moving averages aren't calculated yet
        # Rolling(200) needs 200 days of data before it produces a value
        if pd.isna(df["MA_fast"].iloc[i]) or pd.isna(df["MA_slow"].iloc[i]):
            continue

        price = df["Close"].iloc[i]
        fast = df["MA_fast"].iloc[i]
        slow = df["MA_slow"].iloc[i]
        prev_fast = df["MA_fast"].iloc[i - 1]
        prev_slow = df["MA_slow"].iloc[i - 1]

        # Golden Cross — fast crosses above slow — buy signal
        if prev_fast <= prev_slow and fast > slow and not in_market:
            shares = cash / price       # Buy as many shares as we can
            cash = 0                    # All cash is now invested
            in_market = True
            trades.append({"type": "buy", "price": price,
                          "date": df["Date"].iloc[i]})

        # Death Cross — fast crosses below slow — sell signal
        elif prev_fast >= prev_slow and fast < slow and in_market:
            cash = shares * price       # Sell all shares
            shares = 0
            in_market = False
            trades.append({"type": "sell", "price": price,
                          "date": df["Date"].iloc[i]})

    # If we're still holding at the end, sell at last price
    if in_market:
        cash = shares * df["Close"].iloc[-1]

    # Calculate performance metrics
    total_return = ((cash - capital) / capital) * 100

    # Calculate win rate — percentage of sell trades that were profitable
    wins = 0
    buy_price = None
    for trade in trades:
        if trade["type"] == "buy":
            buy_price = trade["price"]
        elif trade["type"] == "sell" and buy_price:
            if trade["price"] > buy_price:
                wins += 1

    sell_trades = len([t for t in trades if t["type"] == "sell"])
    win_rate = (wins / sell_trades * 100) if sell_trades > 0 else 0

    # Build portfolio value line for the chart
    # Each day portfolio value is cash + value of shares held
    portfolio = []
    cash_track = capital
    shares_track = 0
    in_market_track = False

    for i in range(len(df)):
        if i > 0:
            if pd.isna(df["MA_fast"].iloc[i]) or pd.isna(df["MA_slow"].iloc[i]):
                portfolio.append(cash_track)
                continue
            price = df["Close"].iloc[i]
            fast = df["MA_fast"].iloc[i]
            slow = df["MA_slow"].iloc[i]
            prev_fast = df["MA_fast"].iloc[i - 1]
            prev_slow = df["MA_slow"].iloc[i - 1]
            if prev_fast <= prev_slow and fast > slow and not in_market_track:
                shares_track = cash_track / price
                cash_track = 0
                in_market_track = True
            elif prev_fast >= prev_slow and fast < slow and in_market_track:
                cash_track = shares_track * price
                shares_track = 0
                in_market_track = False
            portfolio.append(cash_track + shares_track * price)
        else:
            portfolio.append(capital)

    return {
        "strategy": "MA Crossover",
        "total_return": round(total_return, 2),
        "final_value": round(cash, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "portfolio": portfolio
    }

# ============================================================
# RSI STRATEGY
# RSI measures how fast a stock has moved recently on a 0-100 scale
# Below 30 — oversold — buy signal — price likely to recover
# Above 70 — overbought — sell signal — price likely to fall
# ============================================================


def strategy_rsi(df, capital, oversold=30, overbought=70):

    # Calculate RSI manually using pandas
    # Step 1: Calculate daily price changes
    delta = df["Close"].diff()

    # Step 2: Separate gains and losses
    # Gains are positive changes, losses are negative changes (made positive)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Step 3: Calculate average gains and losses over 14 days
    # ewm = exponential weighted mean — recent days count more than older days
    avg_gain = gains.ewm(com=13, adjust=False).mean()
    avg_loss = losses.ewm(com=13, adjust=False).mean()

    # Step 4: Calculate RS — ratio of average gain to average loss
    rs = avg_gain / avg_loss

    # Step 5: Convert RS to RSI on a 0-100 scale
    df["RSI"] = 100 - (100 / (1 + rs))

    # Track portfolio state
    cash = capital
    shares = 0
    in_market = False
    trades = []

    # Loop through every day of price data
    for i in range(1, len(df)):

        # Skip days where RSI isn't calculated yet
        if pd.isna(df["RSI"].iloc[i]):
            continue

        price = df["Close"].iloc[i]
        rsi = df["RSI"].iloc[i]

        # RSI below 30 — oversold — buy signal
        if rsi < oversold and not in_market:
            shares = cash / price       # Buy as many shares as we can
            cash = 0
            in_market = True
            trades.append({"type": "buy", "price": price,
                          "date": df["Date"].iloc[i]})

        # RSI above 70 — overbought — sell signal
        elif rsi > overbought and in_market:
            cash = shares * price       # Sell all shares
            shares = 0
            in_market = False
            trades.append({"type": "sell", "price": price,
                          "date": df["Date"].iloc[i]})

    # If still holding at the end sell at last price
    if in_market:
        cash = shares * df["Close"].iloc[-1]

    # Calculate total return
    total_return = ((cash - capital) / capital) * 100

    # Calculate win rate
    wins = 0
    buy_price = None
    for trade in trades:
        if trade["type"] == "buy":
            buy_price = trade["price"]
        elif trade["type"] == "sell" and buy_price:
            if trade["price"] > buy_price:
                wins += 1

    sell_trades = len([t for t in trades if t["type"] == "sell"])
    win_rate = (wins / sell_trades * 100) if sell_trades > 0 else 0

    # Build portfolio value line for the chart
    portfolio = []
    cash_track = capital
    shares_track = 0
    in_market_track = False

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        if i > 0 and not pd.isna(df["RSI"].iloc[i]):
            rsi = df["RSI"].iloc[i]
            if rsi < oversold and not in_market_track:
                shares_track = cash_track / price
                cash_track = 0
                in_market_track = True
            elif rsi > overbought and in_market_track:
                cash_track = shares_track * price
                shares_track = 0
                in_market_track = False
        portfolio.append(cash_track + shares_track * price)

    return {
        "strategy": "RSI",
        "total_return": round(total_return, 2),
        "final_value": round(cash, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "portfolio": portfolio
    }

# ============================================================
# MACD STRATEGY
# Tracks momentum by comparing two moving averages
# Buy when MACD line crosses above signal line — momentum turning positive
# Sell when MACD line crosses below signal line — momentum turning negative
# ============================================================


def strategy_macd(df, capital, fast=12, slow=26, signal=9):

    # Step 1: Calculate 12 and 26 day exponential moving averages
    # EMA gives more weight to recent prices than a simple moving average
    ema12 = df["Close"].ewm(span=fast, adjust=False).mean()
    ema26 = df["Close"].ewm(span=slow, adjust=False).mean()

    # Step 2: MACD line is the difference between the two EMAs
    # Positive means short term momentum is stronger than long term
    df["MACD"] = ema12 - ema26

    # Step 3: Signal line is a 9 day EMA of the MACD line
    # Acts as a smoother slower version of the MACD line
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    # Track portfolio state
    cash = capital
    shares = 0
    in_market = False
    trades = []

    # Loop through every day of price data
    for i in range(1, len(df)):

        # Skip days where MACD or Signal aren't calculated yet
        if pd.isna(df["MACD"].iloc[i]) or pd.isna(df["Signal"].iloc[i]):
            continue

        price = df["Close"].iloc[i]
        macd = df["MACD"].iloc[i]
        signal = df["Signal"].iloc[i]
        prev_macd = df["MACD"].iloc[i - 1]
        prev_signal = df["Signal"].iloc[i - 1]

        # MACD crosses above signal line — momentum turning positive — buy
        if prev_macd <= prev_signal and macd > signal and not in_market:
            shares = cash / price       # Buy as many shares as we can
            cash = 0
            in_market = True
            trades.append({"type": "buy", "price": price,
                          "date": df["Date"].iloc[i]})

        # MACD crosses below signal line — momentum turning negative — sell
        elif prev_macd >= prev_signal and macd < signal and in_market:
            cash = shares * price       # Sell all shares
            shares = 0
            in_market = False
            trades.append({"type": "sell", "price": price,
                          "date": df["Date"].iloc[i]})

    # If still holding at the end sell at last price
    if in_market:
        cash = shares * df["Close"].iloc[-1]

    # Calculate total return
    total_return = ((cash - capital) / capital) * 100

    # Calculate win rate
    wins = 0
    buy_price = None
    for trade in trades:
        if trade["type"] == "buy":
            buy_price = trade["price"]
        elif trade["type"] == "sell" and buy_price:
            if trade["price"] > buy_price:
                wins += 1

    sell_trades = len([t for t in trades if t["type"] == "sell"])
    win_rate = (wins / sell_trades * 100) if sell_trades > 0 else 0

    # Build portfolio value line for the chart
    portfolio = []
    cash_track = capital
    shares_track = 0
    in_market_track = False

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        if i > 0 and not pd.isna(df["MACD"].iloc[i]) and not pd.isna(df["Signal"].iloc[i]):
            macd = df["MACD"].iloc[i]
            signal = df["Signal"].iloc[i]
            prev_macd = df["MACD"].iloc[i - 1]
            prev_signal = df["Signal"].iloc[i - 1]
            if prev_macd <= prev_signal and macd > signal and not in_market_track:
                shares_track = cash_track / price
                cash_track = 0
                in_market_track = True
            elif prev_macd >= prev_signal and macd < signal and in_market_track:
                cash_track = shares_track * price
                shares_track = 0
                in_market_track = False
        portfolio.append(cash_track + shares_track * price)

    return {
        "strategy": "MACD",
        "total_return": round(total_return, 2),
        "final_value": round(cash, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "portfolio": portfolio
    }

# ============================================================
# BOX STRATEGY
# Identifies when a stock breaks out of a trading range
# Buy when price breaks above the recent high — upward breakout
# Sell when price falls below the recent low — downward breakout
# Based on the Darvas Box method
# ============================================================


def strategy_box(df, capital, box_window=20):

    # Track portfolio state
    cash = capital
    shares = 0
    in_market = False
    trades = []

    # Loop through every day starting after the first box window
    for i in range(box_window, len(df)):

        price = df["Close"].iloc[i]

        # Define the box using the previous N days
        # Ceiling is the highest price in the window
        # Floor is the lowest price in the window
        window = df["Close"].iloc[i - box_window:i]
        ceiling = window.max()
        floor = window.min()

        # Price breaks above ceiling — upward breakout — buy signal
        if price > ceiling and not in_market:
            shares = cash / price       # Buy as many shares as we can
            cash = 0
            in_market = True
            trades.append({"type": "buy", "price": price,
                          "date": df["Date"].iloc[i]})

        # Price falls below floor while holding — trend reversed — sell signal
        elif price < floor and in_market:
            cash = shares * price       # Sell all shares
            shares = 0
            in_market = False
            trades.append({"type": "sell", "price": price,
                          "date": df["Date"].iloc[i]})

    # If still holding at the end sell at last price
    if in_market:
        cash = shares * df["Close"].iloc[-1]

    # Calculate total return
    total_return = ((cash - capital) / capital) * 100

    # Calculate win rate
    wins = 0
    buy_price = None
    for trade in trades:
        if trade["type"] == "buy":
            buy_price = trade["price"]
        elif trade["type"] == "sell" and buy_price:
            if trade["price"] > buy_price:
                wins += 1

    sell_trades = len([t for t in trades if t["type"] == "sell"])
    win_rate = (wins / sell_trades * 100) if sell_trades > 0 else 0

    # Build portfolio value line for the chart
    portfolio = []
    cash_track = capital
    shares_track = 0
    in_market_track = False

    for i in range(len(df)):
        if i >= box_window:
            price = df["Close"].iloc[i]
            window = df["Close"].iloc[i - box_window:i]
            ceiling = window.max()
            floor = window.min()
            if price > ceiling and not in_market_track:
                shares_track = cash_track / price
                cash_track = 0
                in_market_track = True
            elif price < floor and in_market_track:
                cash_track = shares_track * price
                shares_track = 0
                in_market_track = False
            portfolio.append(cash_track + shares_track * price)
        else:
            # Not enough data yet to define a box
            portfolio.append(capital)

    return {
        "strategy": "Box Strategy",
        "total_return": round(total_return, 2),
        "final_value": round(cash, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "portfolio": portfolio
    }


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Strategy Backtester",
                   page_icon="📊", layout="wide")

st.title("📊 Strategy Backtester")
st.write("Test different trading strategies against historical stock data.")


# ============================================================
# USER INPUTS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Market selector — determines whether to append .SI or not
    market = st.selectbox("Market", ["US", "SGX"])

with col2:
    # Ticker input
    ticker = st.text_input("Stock Ticker (e.g. AAPL or D05)", "").upper()

with col3:
    # Time period selector
    period = st.selectbox("Time Period", ["1y", "2y", "3y", "5y"], index=2)

with col4:
    # Starting capital
    capital = st.number_input("Starting Capital ($)",
                              min_value=1000, value=10000, step=1000)

# ============================================================
# STRATEGY PARAMETERS
# Collapsible panel letting users adjust each strategy's settings
# ============================================================
with st.expander("⚙️ Strategy Parameters"):
    st.caption(
        "Adjust the parameters for each strategy. Changes apply when you enter a ticker.")

    param_col1, param_col2 = st.columns(2)

    with param_col1:
        st.markdown("**MA Crossover**")
        ma_fast = st.slider(
            "Fast MA Window (days)",
            min_value=10, max_value=100, value=50, step=5,
            help="Number of days for the fast moving average. Default is 50."
        )
        ma_slow = st.slider(
            "Slow MA Window (days)",
            min_value=50, max_value=300, value=200, step=10,
            help="Number of days for the slow moving average. Default is 200."
        )

        st.markdown("**RSI**")
        rsi_oversold = st.slider(
            "Oversold Threshold",
            min_value=10, max_value=40, value=30, step=5,
            help="RSI below this value triggers a buy. Default is 30."
        )
        rsi_overbought = st.slider(
            "Overbought Threshold",
            min_value=60, max_value=90, value=70, step=5,
            help="RSI above this value triggers a sell. Default is 70."
        )

    with param_col2:
        st.markdown("**MACD**")
        macd_fast = st.slider(
            "Fast EMA (days)",
            min_value=5, max_value=20, value=12, step=1,
            help="Fast EMA period for MACD calculation. Default is 12."
        )
        macd_slow = st.slider(
            "Slow EMA (days)",
            min_value=20, max_value=50, value=26, step=1,
            help="Slow EMA period for MACD calculation. Default is 26."
        )
        macd_signal = st.slider(
            "Signal Line (days)",
            min_value=5, max_value=20, value=9, step=1,
            help="Signal line period for MACD calculation. Default is 9."
        )

        st.markdown("**Box Strategy**")
        box_window = st.slider(
            "Box Window (days)",
            min_value=5, max_value=60, value=20, step=5,
            help="Number of days to look back when defining the box. Default is 20."
        )

# ============================================================
# MAIN LOGIC
# Only runs when ticker is entered
# ============================================================
if ticker:

    # Auto append .SI if user selected SGX and forgot to add it
    # Otherwise leave the ticker as is for US stocks
    if market == "SGX" and not ticker.endswith(".SI"):
        ticker = ticker + ".SI"

    with st.spinner("Fetching historical data..."):
        df = fetch_data(ticker, period)

    # If no data came back the ticker is invalid
    if df is None:
        st.error("❌ Ticker not found or no historical data available.")
        st.stop()

    st.success(f"✅ Loaded {len(df)} days of data for {ticker}")
    st.divider()

    # Run all strategies first so we can compare them
    bh = strategy_buy_and_hold(df.copy(), capital)
    ma = strategy_ma_crossover(df.copy(), capital, ma_fast, ma_slow)
    rsi_result = strategy_rsi(df.copy(), capital, rsi_oversold, rsi_overbought)
    macd_result = strategy_macd(
        df.copy(), capital, macd_fast, macd_slow, macd_signal)
    box_result = strategy_box(df.copy(), capital, box_window)

    # ============================================================
    # COMPARISON TABLE
    # Shows all strategies side by side at the top
    # ============================================================
    st.subheader("Strategy Comparison")

    # Build a dataframe for the comparison table
    comparison = pd.DataFrame([
        {
            "Strategy": bh["strategy"],
            "Total Return (%)": f"{bh['total_return']:.2f}%",
            "Final Value ($)": f"${bh['final_value']:,.2f}",
            "Trades": bh["trades"],
            "Win Rate (%)": f"{bh['win_rate']:.2f}%"
        },
        {
            "Strategy": ma["strategy"],
            "Total Return (%)": f"{ma['total_return']:.2f}%",
            "Final Value ($)": f"${ma['final_value']:,.2f}",
            "Trades": ma["trades"],
            "Win Rate (%)": f"{ma['win_rate']:.2f}%"
        },
        {
            "Strategy": rsi_result["strategy"],
            "Total Return (%)": f"{rsi_result['total_return']:.2f}%",
            "Final Value ($)": f"${rsi_result['final_value']:,.2f}",
            "Trades": rsi_result["trades"],
            "Win Rate (%)": f"{rsi_result['win_rate']:.2f}%"
        },
        {
            "Strategy": macd_result["strategy"],
            "Total Return (%)": f"{macd_result['total_return']:.2f}%",
            "Final Value ($)": f"${macd_result['final_value']:,.2f}",
            "Trades": macd_result["trades"],
            "Win Rate (%)": f"{macd_result['win_rate']:.2f}%"
        },
        {
            "Strategy": box_result["strategy"],
            "Total Return (%)": f"{box_result['total_return']:.2f}%",
            "Final Value ($)": f"${box_result['final_value']:,.2f}",
            "Trades": box_result["trades"],
            "Win Rate (%)": f"{box_result['win_rate']:.2f}%"
        },
    ])

    # Find the winning strategy based on total return
    # Extract numeric returns for comparison — strip % sign first
    returns = [bh["total_return"], ma["total_return"], rsi_result["total_return"],
               macd_result["total_return"], box_result["total_return"]]
    strategies = [bh["strategy"], ma["strategy"], rsi_result["strategy"],
                  macd_result["strategy"], box_result["strategy"]]
    best_return = max(returns)
    winner = strategies[returns.index(best_return)]

    # Highlight the winning row in green
    def highlight_winner(row):
        if row["Strategy"] == winner:
            return ["background-color: rgba(34,197,94,0.2); font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comparison.style.apply(highlight_winner, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # Show winner banner
    winner_return = comparison.loc[comparison["Strategy"]
                                   == winner, "Total Return (%)"].values[0]
    st.success(
        f"🏆 Best strategy for {ticker} over {period}: **{winner}** with a return of {best_return:.2f}%")

    # ============================================================
    # WINNER SUMMARY
    # Auto-generates a plain English explanation of why the
    # winning strategy outperformed the others
    # ============================================================
    winner_data = {
        bh["strategy"]: bh,
        ma["strategy"]: ma,
        rsi_result["strategy"]: rsi_result,
        macd_result["strategy"]: macd_result,
        box_result["strategy"]: box_result,
    }[winner]

    loser_returns = [r for r, s in zip(returns, strategies) if s != winner]
    avg_other_return = sum(loser_returns) / len(loser_returns)

    sentences = []

    # Opening sentence — what won and by how much
    sentences.append(
        f"Over the {period} period, {winner} was the strongest performing strategy for {ticker}, "
        f"returning {best_return:.2f}% compared to an average of {avg_other_return:.2f}% across the other strategies."
    )

    # Strategy specific explanation of why it won
    if winner == "Buy and Hold":
        sentences.append(
            f"This suggests {ticker} trended strongly upward over the period with few major reversals, "
            f"making active trading counterproductive as strategies kept moving in and out of the market and missing gains."
        )
    elif winner == "MA Crossover":
        sentences.append(
            f"The MA Crossover strategy succeeded by catching the major trends in {ticker} early "
            f"and staying invested through sustained moves, while avoiding the largest drawdowns by exiting on Death Crosses."
        )
    elif winner == "RSI":
        sentences.append(
            f"The RSI strategy thrived because {ticker} experienced meaningful oversold periods that were followed by strong recoveries, "
            f"allowing the strategy to buy at depressed prices and sell into strength."
        )
    elif winner == "MACD":
        sentences.append(
            f"The MACD strategy succeeded by identifying momentum shifts in {ticker} early, "
            f"capturing strong directional moves while avoiding prolonged periods of weak or negative momentum."
        )
    elif winner == "Box Strategy":
        sentences.append(
            f"The Box Strategy worked well because {ticker} exhibited clear breakout patterns, "
            f"consolidating in ranges before making strong directional moves that the strategy captured effectively."
        )

    # Trade activity sentence
    if winner_data["trades"] <= 5:
        sentences.append(
            f"Notably the strategy made only {winner_data['trades']} trade(s) over the period, "
            f"suggesting patience and selectivity were key to its outperformance."
        )
    elif winner_data["trades"] > 20:
        sentences.append(
            f"The strategy was active, making {winner_data['trades']} trades over the period. "
            f"In real trading this would incur transaction costs which could reduce the actual return."
        )
    else:
        sentences.append(
            f"The strategy made {winner_data['trades']} trades over the period, "
            f"balancing activity with selectivity."
        )

    # Win rate sentence
    if winner_data["win_rate"] >= 60:
        sentences.append(
            f"A win rate of {winner_data['win_rate']:.2f}% means the majority of its trades were profitable, "
            f"indicating the strategy's signals were well timed for this stock."
        )
    elif winner_data["win_rate"] > 0:
        sentences.append(
            f"Despite a win rate of {winner_data['win_rate']:.2f}%, the strategy still outperformed "
            f"because its winning trades gained more than its losing trades gave back."
        )

    # Disclaimer
    sentences.append(
        "Note that past performance does not guarantee future results, "
        "and these simulations do not account for transaction costs, taxes, or slippage."
    )

    st.info(" ".join(sentences))

    st.divider()

    # ============================================================
    # COMBINED CHART
    # All strategies on one chart for easy visual comparison
    # ============================================================
    st.subheader("All Strategies vs Buy and Hold")

    fig_all = go.Figure()

    fig_all.add_trace(go.Scatter(
        x=df["Date"], y=bh["portfolio"],
        mode="lines", name="Buy and Hold",
        line=dict(color="#0068c9", width=2, dash="dot")
    ))
    fig_all.add_trace(go.Scatter(
        x=df["Date"], y=ma["portfolio"],
        mode="lines", name="MA Crossover",
        line=dict(color="#22c55e", width=2)
    ))
    fig_all.add_trace(go.Scatter(
        x=df["Date"], y=rsi_result["portfolio"],
        mode="lines", name="RSI",
        line=dict(color="#f59e0b", width=2)
    ))
    fig_all.add_trace(go.Scatter(
        x=df["Date"], y=macd_result["portfolio"],
        mode="lines", name="MACD",
        line=dict(color="#a855f7", width=2)
    ))
    fig_all.add_trace(go.Scatter(
        x=df["Date"], y=box_result["portfolio"],
        mode="lines", name="Box Strategy",
        line=dict(color="#ef4444", width=2)
    ))

    fig_all.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=-45,
            nticks=12,
            showgrid=False,
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
    )

    st.plotly_chart(fig_all, use_container_width=True)

    st.divider()

    # Display results
    st.subheader("Buy and Hold Benchmark")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Total Return", value=f"{bh['total_return']}%")
    with col_b:
        st.metric(label="Final Value", value=f"${bh['final_value']:,.2f}")
    with col_c:
        st.metric(label="Starting Capital", value=f"${capital:,.2f}")

    # Plot the portfolio value over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=bh["portfolio"],
        mode="lines",
        name="Buy and Hold",
        line=dict(color="#0068c9", width=2)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=-45,
            nticks=12,
            showgrid=False,
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Run MA crossover strategy
    ma = strategy_ma_crossover(df.copy(), capital)

    st.divider()
    st.subheader("Moving Average Crossover")
    col_d, col_e, col_f, col_g = st.columns(4)
    with col_d:
        st.metric(label="Total Return", value=f"{ma['total_return']}%")
    with col_e:
        st.metric(label="Final Value", value=f"${ma['final_value']:,.2f}")
    with col_f:
        st.metric(label="Total Trades", value=ma["trades"])
    with col_g:
        st.metric(label="Win Rate", value=f"{ma['win_rate']}%")

# Plot MA crossover portfolio value over time
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=df["Date"],
        y=ma["portfolio"],
        mode="lines",
        name="MA Crossover",
        line=dict(color="#22c55e", width=2)  # Green line
    ))
    fig_ma.add_trace(go.Scatter(
        x=df["Date"],
        y=bh["portfolio"],
        mode="lines",
        name="Buy and Hold",
        # Blue dotted line for comparison
        line=dict(color="#0068c9", width=2, dash="dot")
    ))
    fig_ma.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=-45,
            nticks=12,
            showgrid=False,
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
    )
    st.plotly_chart(fig_ma, use_container_width=True)

# Run RSI strategy
    rsi_result = strategy_rsi(df.copy(), capital)

    st.divider()
    st.subheader("RSI Strategy")
    col_h, col_i, col_j, col_k = st.columns(4)
    with col_h:
        st.metric(label="Total Return", value=f"{rsi_result['total_return']}%")
    with col_i:
        st.metric(label="Final Value",
                  value=f"${rsi_result['final_value']:,.2f}")
    with col_j:
        st.metric(label="Total Trades", value=rsi_result["trades"])
    with col_k:
        st.metric(label="Win Rate", value=f"{rsi_result['win_rate']}%")

    # Plot RSI portfolio value over time
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df["Date"],
        y=rsi_result["portfolio"],
        mode="lines",
        name="RSI",
        line=dict(color="#f59e0b", width=2)  # Amber line
    ))
    fig_rsi.add_trace(go.Scatter(
        x=df["Date"],
        y=bh["portfolio"],
        mode="lines",
        name="Buy and Hold",
        # Blue dotted benchmark
        line=dict(color="#0068c9", width=2, dash="dot")
    ))
    fig_rsi.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=-45,
            nticks=12,
            showgrid=False,
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# Run MACD strategy
    macd_result = strategy_macd(df.copy(), capital)

    st.divider()
    st.subheader("MACD Strategy")
    col_l, col_m, col_n, col_o = st.columns(4)
    with col_l:
        st.metric(label="Total Return",
                  value=f"{macd_result['total_return']}%")
    with col_m:
        st.metric(label="Final Value",
                  value=f"${macd_result['final_value']:,.2f}")
    with col_n:
        st.metric(label="Total Trades", value=macd_result["trades"])
    with col_o:
        st.metric(label="Win Rate", value=f"{macd_result['win_rate']}%")

    # Plot MACD portfolio value over time
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(
        x=df["Date"],
        y=macd_result["portfolio"],
        mode="lines",
        name="MACD",
        line=dict(color="#a855f7", width=2)  # Purple line
    ))
    fig_macd.add_trace(go.Scatter(
        x=df["Date"],
        y=bh["portfolio"],
        mode="lines",
        name="Buy and Hold",
        # Blue dotted benchmark
        line=dict(color="#0068c9", width=2, dash="dot")
    ))
    fig_macd.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=-45,
            nticks=12,
            showgrid=False,
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
    )
    st.plotly_chart(fig_macd, use_container_width=True)

# Run Box strategy
    box_result = strategy_box(df.copy(), capital)

    st.divider()
    st.subheader("Box Strategy (Darvas)")
    col_p, col_q, col_r, col_s = st.columns(4)
    with col_p:
        st.metric(label="Total Return", value=f"{box_result['total_return']}%")
    with col_q:
        st.metric(label="Final Value",
                  value=f"${box_result['final_value']:,.2f}")
    with col_r:
        st.metric(label="Total Trades", value=box_result["trades"])
    with col_s:
        st.metric(label="Win Rate", value=f"{box_result['win_rate']}%")

    # Plot Box strategy portfolio value over time
    fig_box = go.Figure()
    fig_box.add_trace(go.Scatter(
        x=df["Date"],
        y=box_result["portfolio"],
        mode="lines",
        name="Box Strategy",
        line=dict(color="#ef4444", width=2)  # Red line
    ))
    fig_box.add_trace(go.Scatter(
        x=df["Date"],
        y=bh["portfolio"],
        mode="lines",
        name="Buy and Hold",
        # Blue dotted benchmark
        line=dict(color="#0068c9", width=2, dash="dot")
    ))
    fig_box.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %Y",
            tickangle=-45,
            nticks=12,
            showgrid=False,
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            linecolor="rgba(128,128,128,0.3)",
            tickfont=dict(color="rgba(128,128,128,0.8)"),
        ),
    )
    st.plotly_chart(fig_box, use_container_width=True)
