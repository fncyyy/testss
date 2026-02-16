import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import sys
import os

# Add parent directory to path to allow importing from 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data_loader import fetch_data, resample_data
from app.backtester import Backtester
from app.strategies import VWAPTrendStrategy, AITextStrategy, ICTStrategy

# Set page config
st.set_page_config(page_title="Futures Trading Bot", layout="wide")

st.title("AI-Powered Futures Trading Bot")

# Sidebar Controls
st.sidebar.header("Configuration")

# 1. Data Selection
ticker = st.sidebar.selectbox("Ticker", ["NQ=F", "ES=F", "MNQ=F", "MES=F"])
today = datetime.date.today()
# Default range: last 7 days (yfinance 1m data limit is usually 7 days, 5m is 60 days)
start_date = st.sidebar.date_input("Start Date", today - datetime.timedelta(days=7))
end_date = st.sidebar.date_input("End Date", today)

base_timeframe = st.sidebar.selectbox("Base Timeframe", ["5m", "1m", "15m", "1h"], index=0)

# 2. Strategy Selection
strategy_type = st.sidebar.selectbox("Strategy", ["VWAP Trend (Default)", "AI Text Strategy", "ICT Strategy (Hardcoded)"])

strategy_text = ""
api_key = ""

if strategy_type == "AI Text Strategy":
    st.sidebar.subheader("AI Configuration")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

    # Model Selection
    model_options = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3-pro-preview"]
    model_name = st.sidebar.selectbox("Model Name", model_options, index=0, help="Select the Gemini model to use.")

    # Optimization
    st.sidebar.markdown("---")
    st.sidebar.subheader("Performance Optimization")
    skip_period = st.sidebar.number_input("Process Every N Candles", min_value=1, value=1, help="Skip N-1 candles to speed up AI backtesting. E.g., 5 means process every 5th candle.")
    max_candles = st.sidebar.number_input("Max Recent Candles to Test", min_value=10, max_value=1000, value=100, step=10, help="Limit backtest to the most recent N candles to save time/API costs.")

    default_prompt = """Futures/indexes refined
treat 1hr/4hr as our bias
mark out 1hr 4hr session highs and lows
5min confirmation confluence ( bos, ifvg, 79% extension)
IF #1 happens before mkt open wait for 5min liq sweep
5min continuation conf ( eq fvg if 2b  happens)
1min conf conf ( bos ifvg 79% ext)
Enter
Target draws of liq in our direction"""

    strategy_text = st.sidebar.text_area("Paste Strategy Here", value=default_prompt, height=200)

# 3. Risk Management
st.sidebar.subheader("Risk Parameters")
initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0)
commission = st.sidebar.number_input("Commission per Trade", value=2.0)

# 4. Visualization Settings
st.sidebar.subheader("Visualization")
active_scan = st.sidebar.checkbox("Active Chart Scan", value=True, help="Update the chart in real-time during backtest.")
scan_window = st.sidebar.slider("Scan Window (Candles)", 50, 500, 100, help="Number of recent candles to show during active scan.")
update_frequency = st.sidebar.slider("Update Frequency (Steps)", 1, 50, 1, help="Update chart every N steps.")

# Main Execution Button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Fetching Data..."):
        # Fetch Data
        df = fetch_data(ticker, str(start_date), str(end_date), interval=base_timeframe)

        if df.empty:
            st.error("No data found for the selected range/ticker.")
        else:
            # Optimization: Slice data if user requested Max Candles limit
            # This is critical because loading 5000 candles means 5000 AI calls unless limited.
            if strategy_type == "AI Text Strategy":
                # Ensure we have enough data for lookback context (e.g. 1h resampling)
                # But strictly limit the MAIN dataframe for iteration.
                # Actually, data_loader fetches based on date range.
                # If we truncate df, we might lose HTF calculation ability on early candles.
                # Better: Keep full history for indicators, but slice the dataframe passed to backtester?
                # Or just slice the range we iterate over.

                # Backtester iterates over `data`.
                # Let's slice `df` to keep only the last `max_candles`.
                # BUT we need history for indicators.
                # Backtester assumes `data` is the period to trade.
                # So we can calculate indicators on FULL `df`, then slice it.

                # However, resample_data uses `df`.
                # Let's keep `df` full for resampling first.
                pass

            st.success(f"Data Loaded: {len(df)} candles. (Will process last {max_candles} if AI strategy)")

            # Prepare Higher Timeframes
            # We strictly need 1h (and maybe 4h) for the strategy
            # Resample from base timeframe
            ht_data = {}
            if base_timeframe in ['1m', '5m', '15m']:
                df_1h = resample_data(df, '1h')
                ht_data['1h'] = df_1h
                df_4h = resample_data(df, '4h')
                ht_data['4h'] = df_4h
            elif base_timeframe == '1h':
                df_4h = resample_data(df, '4h')
                ht_data['4h'] = df_4h
                # ht_data['1h'] is basically df itself, but we handle it in strategy
                ht_data['1h'] = df.copy() # Strategy uses ht_data['1h'] for bias

            # Initialize Strategy
            strategy = None
            if strategy_type == "VWAP Trend (Default)":
                strategy = VWAPTrendStrategy()
            elif strategy_type == "ICT Strategy (Hardcoded)":
                strategy = ICTStrategy()
            else:
                if not api_key:
                    st.warning("Please provide a Gemini API Key for AI Strategy.")
                else:
                    strategy = AITextStrategy(strategy_text=strategy_text, api_key=api_key, model_name=model_name)

            if strategy:
                # Initialize Backtester
                # Determine multiplier
                mult = 1.0
                if "NQ" in ticker or "MNQ" in ticker: mult = 20.0 # NQ is $20 per point. MNQ is $2?
                if "MNQ" in ticker: mult = 2.0
                if "ES" in ticker: mult = 50.0
                if "MES" in ticker: mult = 5.0

                bt = Backtester(initial_capital=initial_capital, commission=commission, multiplier=mult)

                # Slicing Dataframe for AI Efficiency
                # We need to preserve original DF for indicators if they are calculated inside strategy (AI strategy doesn't calculate technicals inside, only reads raw price).
                # But for safety, let's slice `df` passed to run() to only include the last N candles.

                backtest_df = df
                if strategy_type == "AI Text Strategy":
                    if len(df) > max_candles:
                        st.warning(f"Optimization: Limiting backtest to last {max_candles} candles to save time.")
                        backtest_df = df.iloc[-max_candles:]

                # Pass skip_period to strategy if it accepts it?
                # Or handle skipping inside backtester loop?
                # Easiest: Handle in Backtester. But Backtester iterates every candle.
                # If we skip in Backtester, we might miss SL/TP checks.
                # SL/TP checks must happen EVERY candle.
                # Strategy.next() calls happen every candle.
                # We should modify Strategy to skip internally or modify Backtester to skip strategy calls?
                # If we modify Backtester to have `strategy_interval`, we check SL/TP every candle, but call strategy every N.

                # Let's add `strategy_interval` to `bt.run`.

                # Run Backtest
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.empty()
                chart_placeholder = st.empty()

                # Store visualization confluences
                confluence_history = []

                def progress_callback(current_step, total_steps, current_time, signal):
                    # Check if signal has reason/action
                    last_reason = "Hold/No Signal"
                    if signal and isinstance(signal, dict):
                        last_reason = signal.get('reason', "Hold/No Signal")

                        # Store confluences if present
                        if 'confluences' in signal and isinstance(signal['confluences'], list):
                            for c in signal['confluences']:
                                c['time'] = current_time # Tag with current time
                                confluence_history.append(c)

                    is_signal = last_reason and last_reason != "Hold/No Signal"

                    # Update Progress Bar
                    if current_step % 5 == 0 or is_signal:
                        progress = int((current_step / total_steps) * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {current_time} ({current_step}/{total_steps})")
                        if is_signal:
                             log_container.info(f"Signal at {current_time}: {last_reason}")

                    # Active Chart Scan
                    if active_scan and (current_step % update_frequency == 0 or is_signal):
                        # Slice data for the window
                        # current_step corresponds to index i. Data up to i is backtest_df.iloc[:i+1]
                        end_idx = current_step + 1
                        start_idx = max(0, end_idx - scan_window)
                        scan_df = backtest_df.iloc[start_idx:end_idx]

                        if not scan_df.empty:
                            fig_scan = go.Figure()

                            # Candles
                            fig_scan.add_trace(go.Candlestick(
                                x=scan_df.index,
                                open=scan_df['open'],
                                high=scan_df['high'],
                                low=scan_df['low'],
                                close=scan_df['close'],
                                name='Price'
                            ))

                            # Confluences
                            window_start_time = scan_df.index[0]
                            window_end_time = scan_df.index[-1]

                            for c in confluence_history:
                                # Show confluences created within the window OR persistently if they are 'levels'
                                # For simplicity, let's show all that were created within the visible time window.
                                # Or better: show recently created ones?
                                # Let's show those created within the window range.
                                if c.get('time') and c['time'] >= window_start_time and c['time'] <= window_end_time:
                                    color = 'blue'
                                    if 'bullish' in str(c.get('label', '')).lower(): color = 'green'
                                    if 'bearish' in str(c.get('label', '')).lower(): color = 'red'

                                    if c.get('type') == 'line' and 'price' in c:
                                        fig_scan.add_hline(y=c['price'], line_dash="dash", line_color=color, annotation_text=c.get('label', 'Level'))

                                    elif c.get('type') == 'zone' and 'top' in c and 'bottom' in c:
                                        fig_scan.add_shape(type="rect",
                                            x0=c['time'], x1=window_end_time, # Extend to right?
                                            y0=c['bottom'], y1=c['top'],
                                            line=dict(color=color, width=1),
                                            fillcolor=color, opacity=0.2
                                        )
                                        # Annotation?
                                        # fig_scan.add_annotation(x=c['time'], y=c['top'], text=c.get('label', 'Zone'), showarrow=False)

                            # Trades (Only those within the scan window)
                            if bt.trades:
                                # Buys
                                buys = [t for t in bt.trades if t.side == 'long' and t.entry_time >= window_start_time and t.entry_time <= window_end_time]
                                if buys:
                                    fig_scan.add_trace(go.Scatter(
                                        x=[t.entry_time for t in buys],
                                        y=[t.entry_price for t in buys],
                                        mode='markers',
                                        marker=dict(symbol='triangle-up', size=12, color='green'),
                                        name='Buy'
                                    ))

                                # Sells
                                sells = [t for t in bt.trades if t.side == 'short' and t.entry_time >= window_start_time and t.entry_time <= window_end_time]
                                if sells:
                                    fig_scan.add_trace(go.Scatter(
                                        x=[t.entry_time for t in sells],
                                        y=[t.entry_price for t in sells],
                                        mode='markers',
                                        marker=dict(symbol='triangle-down', size=12, color='red'),
                                        name='Sell'
                                    ))

                                # Exits
                                exits = [t for t in bt.trades if t.status == 'closed' and t.exit_time >= window_start_time and t.exit_time <= window_end_time]
                                if exits:
                                    fig_scan.add_trace(go.Scatter(
                                        x=[t.exit_time for t in exits],
                                        y=[t.exit_price for t in exits],
                                        mode='markers',
                                        marker=dict(symbol='x', size=10, color='black'),
                                        name='Exit'
                                    ))

                            # Layout
                            fig_scan.update_layout(
                                title=f"Live Scan - {current_time}",
                                height=500,
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )

                            chart_placeholder.plotly_chart(fig_scan, use_container_width=True)

                status_text.text("Running Backtest...")

                strategy_interval = 1
                if strategy_type == "AI Text Strategy":
                    strategy_interval = int(skip_period)

                # Pass callback to backtester
                bt.run(backtest_df, strategy, ht_data, progress_callback=progress_callback, strategy_interval=strategy_interval)

                progress_bar.progress(100)
                status_text.text("Backtest Complete.")

                # Display Results
                metrics = bt.get_metrics()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total PnL", f"${metrics.get('total_pnl', 0):,.2f}")
                col2.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
                col3.metric("Trades", metrics.get('total_trades', 0))
                col4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")

                # Debugging Information for AI Strategy
                if strategy_type == "AI Text Strategy":
                    with st.expander("AI Strategy Debug Logs"):
                        if hasattr(strategy, 'logs') and strategy.logs:
                            st.write("Execution Logs:")
                            for log in strategy.logs[-20:]: # Show last 20 logs
                                st.text(log)

                        if hasattr(strategy, 'last_error') and strategy.last_error:
                            st.error(f"Last Error: {strategy.last_error}")

                        if hasattr(strategy, 'last_response') and strategy.last_response:
                            st.subheader("Last Raw Response from AI:")
                            st.code(strategy.last_response, language="json")

                            # Parse reason if available
                            try:
                                import json
                                resp = json.loads(strategy.last_response)
                                if 'reason' in resp:
                                    st.info(f"AI Reasoning: {resp['reason']}")
                            except:
                                pass

                st.subheader("Equity Curve")
                if bt.equity_curve:
                    equity_df = pd.DataFrame(bt.equity_curve)
                    st.line_chart(equity_df.set_index('time')['equity'])

                st.subheader("Trade List")
                if bt.trades:
                    trades_df = pd.DataFrame([vars(t) for t in bt.trades])
                    st.dataframe(trades_df)

                    # Candlestick Chart with Trades
                    st.subheader("Chart")

                    # Downsample for charting if too large
                    plot_df = df
                    if len(df) > 2000:
                        st.warning("Data too large for full chart, showing last 2000 candles.")
                        plot_df = df.iloc[-2000:]

                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                                        row_width=[0.2, 0.7])

                    # Candlestick
                    fig.add_trace(go.Candlestick(x=plot_df.index,
                                    open=plot_df['open'],
                                    high=plot_df['high'],
                                    low=plot_df['low'],
                                    close=plot_df['close'], name='Price'), row=1, col=1)

                    # Buy Markers
                    buys = [t for t in bt.trades if t.side == 'long']
                    if buys:
                        buy_times = [t.entry_time for t in buys if t.entry_time in plot_df.index]
                        buy_prices = [t.entry_price for t in buys if t.entry_time in plot_df.index]
                        fig.add_trace(go.Scatter(x=buy_times, y=buy_prices, mode='markers',
                                                marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy'), row=1, col=1)

                    # Sell Markers (Short Entries)
                    sells = [t for t in bt.trades if t.side == 'short']
                    if sells:
                        sell_times = [t.entry_time for t in sells if t.entry_time in plot_df.index]
                        sell_prices = [t.entry_price for t in sells if t.entry_time in plot_df.index]
                        fig.add_trace(go.Scatter(x=sell_times, y=sell_prices, mode='markers',
                                                marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell'), row=1, col=1)

                    # Exits
                    exits = [t for t in bt.trades if t.status == 'closed']
                    if exits:
                        exit_times = [t.exit_time for t in exits if t.exit_time in plot_df.index]
                        exit_prices = [t.exit_price for t in exits if t.exit_time in plot_df.index]
                        fig.add_trace(go.Scatter(x=exit_times, y=exit_prices, mode='markers',
                                                marker=dict(symbol='x', size=8, color='black'), name='Exit'), row=1, col=1)

                    # VWAP if available
                    if 'VWAP' in plot_df.columns:
                        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['VWAP'], mode='lines',
                                                line=dict(color='orange', width=1), name='VWAP'), row=1, col=1)

                    # Volume
                    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name='Volume'), row=2, col=1)

                    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trades executed.")
