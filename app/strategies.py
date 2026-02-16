from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
import traceback
import pytz
from datetime import time as dt_time

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Warning: google-generativeai not installed. AI Strategy will fail.")

class Strategy(ABC):
    def __init__(self):
        self.logs = [] # To store debug logs

    @abstractmethod
    def initialize(self, data: pd.DataFrame, higher_timeframe_data: Dict[str, pd.DataFrame]):
        """
        Initialize strategy with historical data.
        """
        pass

    @abstractmethod
    def next(self, current_time: pd.Timestamp, current_row: pd.Series, portfolio: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Called on every candle.
        Returns a signal dict or None.
        """
        pass

    def log(self, message: str):
        self.logs.append(message)

    def check_trading_hours(self, current_time: pd.Timestamp) -> bool:
        """
        Checks if current time is within trading hours (09:30 - 16:30 NY Time).
        """
        ny_tz = pytz.timezone('America/New_York')
        if current_time.tzinfo is None:
            # Assume UTC if naive, convert to NY
            now_utc = current_time.replace(tzinfo=pytz.utc)
            now_ny = now_utc.astimezone(ny_tz)
        else:
            now_ny = current_time.astimezone(ny_tz)

        # Trading Hours: 09:30 - 16:30
        start_time = dt_time(9, 30)
        end_time = dt_time(16, 30)

        # Check if weekday (0-4)
        if now_ny.weekday() >= 5:
            return False

        return start_time <= now_ny.time() <= end_time

class VWAPTrendStrategy(Strategy):
    """
    Adaptive Top-Down Market Structure & VWAP Strategy.
    """
    def __init__(self, short_window=20, long_window=50, vwap_window=20):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.vwap_window = vwap_window
        self.data = None
        self.ht_data = {}

    def initialize(self, data: pd.DataFrame, higher_timeframe_data: Dict[str, pd.DataFrame]):
        self.data = data
        self.ht_data = higher_timeframe_data or {}

        # Calculate indicators
        self._calculate_indicators(self.data)
        for tf, df in self.ht_data.items():
            self._calculate_indicators(df, prefix=f"{tf}_")

    def _calculate_indicators(self, df: pd.DataFrame, prefix=""):
        # Simple Moving Averages
        df[f'{prefix}SMA_{self.short_window}'] = df['close'].rolling(window=self.short_window).mean()
        df[f'{prefix}SMA_{self.long_window}'] = df['close'].rolling(window=self.long_window).mean()

        # VWAP (Rolling Window)
        tp = (df['high'] + df['low'] + df['close']) / 3
        rolling_pv = (tp * df['volume']).rolling(self.vwap_window).sum()
        rolling_vol = df['volume'].rolling(self.vwap_window).sum()

        # Avoid division by zero
        rolling_vol = rolling_vol.replace(0, np.nan)

        df[f'{prefix}VWAP'] = rolling_pv / rolling_vol

    def next(self, current_time: pd.Timestamp, current_row: pd.Series, portfolio: Dict[str, float]) -> Optional[Dict[str, Any]]:
        # 1. Determine Bias from Higher Timeframe (e.g., 1h)
        bias = "neutral"
        target_tf = '1h'

        if target_tf in self.ht_data:
            df_ht = self.ht_data[target_tf]
            # STRICT Look-ahead Prevention:
            # Assuming index is candle START time (yfinance default).
            # A 1h candle at 09:00 closes at 10:00.
            # We can only use it if current_time >= 10:00.
            # So index <= current_time - 1h.

            # TODO: Detect timeframe duration dynamically?
            # For now, hardcode 1h offset for '1h' timeframe.
            if target_tf == '1h':
                offset = pd.Timedelta(hours=1)
            elif target_tf == '4h':
                offset = pd.Timedelta(hours=4)
            else:
                offset = pd.Timedelta(minutes=0)

            cutoff_time = current_time - offset

            try:
                # Find latest candle starting before or at cutoff
                valid_indices = df_ht.index[df_ht.index <= cutoff_time]
                if not valid_indices.empty:
                    last_idx = valid_indices[-1]
                    ht_row = df_ht.loc[last_idx]

                    sma_short = ht_row.get(f'{target_tf}_SMA_{self.short_window}')
                    sma_long = ht_row.get(f'{target_tf}_SMA_{self.long_window}')

                    if pd.notna(sma_short) and pd.notna(sma_long):
                        if sma_short > sma_long:
                            bias = "bullish"
                        elif sma_short < sma_long:
                            bias = "bearish"
            except Exception:
                pass

        # 2. Check Signals
        vwap = current_row.get('VWAP')
        price = current_row['close']

        if pd.isna(vwap):
            return None

        signal = None

        # Entry Logic
        if bias == "bullish":
            if portfolio['position'] <= 0:
                if price > vwap:
                    sl_price = price * 0.995
                    tp_price = price * 1.01
                    signal = {'action': 'buy', 'quantity': 1, 'sl': sl_price, 'tp': tp_price}

        elif bias == "bearish":
            if portfolio['position'] >= 0:
                if price < vwap:
                    sl_price = price * 1.005
                    tp_price = price * 0.99
                    signal = {'action': 'sell', 'quantity': 1, 'sl': sl_price, 'tp': tp_price}

        # Exit Logic
        if portfolio['position'] > 0 and bias == "bearish":
             signal = {'action': 'close'}
        if portfolio['position'] < 0 and bias == "bullish":
             signal = {'action': 'close'}

        return signal

class AITextStrategy(Strategy):
    def __init__(self, strategy_text: str, api_key: str, model_name: str = "gemini-1.5-pro"):
        super().__init__()
        self.strategy_text = strategy_text
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.data = None
        self.ht_data = {}
        self.last_error = None
        self.last_response = None

        if HAS_GEMINI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                self.log(f"Error configuring Gemini API: {e}")
                self.model = None
        else:
            self.log("Gemini API Key missing or library not installed.")

    def initialize(self, data: pd.DataFrame, higher_timeframe_data: Dict[str, pd.DataFrame]):
        self.data = data
        self.ht_data = higher_timeframe_data or {}

    def next(self, current_time: pd.Timestamp, current_row: pd.Series, portfolio: Dict[str, float]) -> Optional[Dict[str, Any]]:
        if not self.model:
            return None

        try:
            loc = self.data.index.get_loc(current_time)
            if isinstance(loc, slice): # Duplicate index?
                loc = loc.stop - 1

            # Increased context for pattern recognition (BOS, Swing Points)
            # 5m: Last 30 candles (~2.5 hours)
            start_idx = max(0, loc - 30)
            recent_data = self.data.iloc[start_idx:loc+1]
            recent_str = recent_data[['open', 'high', 'low', 'close', 'volume']].to_string()

            # Calculate Session/Day High/Low for context
            day_high = recent_data['high'].max()
            day_low = recent_data['low'].min()

            # Timezone Context (NY Time)
            ny_tz = pytz.timezone('America/New_York')
            if current_time.tzinfo is None:
                # Assume UTC if naive (yfinance usually returns UTC for intraday if requested, or naive)
                # Actually yfinance returns naive localized to exchange usually, or UTC.
                # Let's assume it's UTC for futures.
                now_utc = current_time.replace(tzinfo=pytz.utc)
                now_ny = now_utc.astimezone(ny_tz)
            else:
                now_ny = current_time.astimezone(ny_tz)

            time_str = now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')

            # Determine Market Status
            is_rth = self.check_trading_hours(current_time)
            market_status = "OPEN (RTH)" if is_rth else "CLOSED/EXTENDED HOURS"

            htf_str = ""
            if '1h' in self.ht_data:
                df_1h = self.ht_data['1h']
                # Get last 5 COMPLETE candles (~5 hours)
                cutoff_1h = current_time - pd.Timedelta(hours=1)
                valid_indices = df_1h.index[df_1h.index <= cutoff_1h]
                if not valid_indices.empty:
                    last_idx = valid_indices[-1]
                    try:
                        h_loc = df_1h.index.get_loc(last_idx)
                        if isinstance(h_loc, slice): h_loc = h_loc.stop - 1
                        start_h = max(0, h_loc - 4)
                        htf_data = df_1h.iloc[start_h:h_loc+1]
                        htf_str = htf_data[['open', 'high', 'low', 'close', 'volume']].to_string()
                    except KeyError:
                        pass

            prompt = f"""
            You are an automated trading bot. Analyze the market data below based STRICTLY on the following strategy logic.

            IMPORTANT:
            1. You assume the chart has 1:1 real liquidity and movements with real markets.
            2. Do not spend minutes wondering. Process this single candle and context immediately. Keep thoughts concise.
            3. Strictly follow the strategy logic below. Use ICT concepts as defined.
            4. Do not check for SMT divergence unless you can see multiple ticker data (currently only one is provided).

            STRATEGY LOGIC:
            {self.strategy_text}

            MARKET DATA:
            Current Time: {time_str}
            Market Status: {market_status} (Trading is ONLY allowed if OPEN)

            Session Context (Last ~2.5h):
            High: {day_high}
            Low: {day_low}

            Recent 5m Data (Last 30 candles):
            {recent_str}

            Recent 1h Data (Last 5 closed candles):
            {htf_str}

            Current Position: {portfolio['position']} (Positive=Long, Negative=Short, 0=Flat)

            INSTRUCTIONS:
            - Decide the action: 'buy', 'sell', 'close', or 'hold'.
            - Return ONLY valid JSON. No Markdown.
            - Format: {{"action": "...", "quantity": 1, "sl": float, "tp": float, "reason": "short explanation"}}
            - The "reason" field is mandatory. Explain why you are taking the action or holding based on the strategy.
            - If 'hold', return {{"action": "hold", "reason": "..."}}.
            - If the strategy conditions are not met, return {{"action": "hold", "reason": "Conditions not met..."}}.
            """

            response = self.model.generate_content(prompt)
            text = response.text.strip()
            self.last_response = text # Store raw response for debugging

            # Clean markdown
            if text.startswith("```"):
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)

            # Try parsing JSON
            try:
                signal = json.loads(text)
                if isinstance(signal, dict) and 'action' in signal:
                    if signal['action'] in ['buy', 'sell', 'close']:
                        self.log(f"Signal: {signal}")
                        return signal
                return None
            except json.JSONDecodeError as e:
                self.log(f"JSON Parse Error: {e}. Raw Text: {text}")
                return None

        except Exception as e:
            self.last_error = str(e)
            self.log(f"AI Execution Error: {e}")
            # print(traceback.format_exc()) # Print to console too
            return None

class ICTStrategy(Strategy):
    """
    Hardcoded implementation of the Futures/Indexes Refined strategy.

    Bias: 4H Market Structure or SMA.
    Levels: Previous 4H High/Low.
    Entry: 5m BOS + FVG.
    """
    def __init__(self):
        super().__init__()
        self.data = None
        self.ht_data = {}

    def initialize(self, data: pd.DataFrame, higher_timeframe_data: Dict[str, pd.DataFrame]):
        self.data = data
        self.ht_data = higher_timeframe_data or {}

        # Calculate 4H Bias Indicator (Simple SMA for now as proxy for structure)
        if '4h' in self.ht_data:
            df_4h = self.ht_data['4h']
            # We must be careful not to introduce lookahead.
            # Calculating rolling on the full series is fine as long as we access only up to current index.
            df_4h['4h_SMA_20'] = df_4h['close'].rolling(20).mean()

    def next(self, current_time: pd.Timestamp, current_row: pd.Series, portfolio: Dict[str, float]) -> Optional[Dict[str, Any]]:
        # 1. Determine Bias (4H)
        bias = "neutral"
        if '4h' in self.ht_data:
            df_4h = self.ht_data['4h']
            # Get latest closed 4h candle
            cutoff_4h = current_time - pd.Timedelta(hours=4)
            valid_indices = df_4h.index[df_4h.index <= cutoff_4h]
            if not valid_indices.empty:
                last_idx = valid_indices[-1]
                row_4h = df_4h.loc[last_idx]
                sma = row_4h.get('4h_SMA_20')
                if pd.notna(sma):
                    if row_4h['close'] > sma:
                        bias = "bullish"
                    else:
                        bias = "bearish"

        # 2. Check 5m FVG and BOS (Recent 3 candles)
        # We need recent history from self.data
        loc = self.data.index.get_loc(current_time)
        if isinstance(loc, slice): loc = loc.stop - 1

        if loc < 3: return None

        # Look at last 3 completed candles (before current)?
        # Actually, let's look at the pattern formed by [i-2, i-1, i]
        # Current row is 'i'.
        # FVG is usually gap between i-2 and i.

        candle_i = current_row # Current closing candle
        candle_i_1 = self.data.iloc[loc-1]
        candle_i_2 = self.data.iloc[loc-2]

        signal = None
        price = candle_i['close']

        # FVG Detection
        # Bullish FVG: Candle i-1 is UP. Gap between High(i-2) and Low(i).
        # Wait, usually FVG is detected AFTER candle i-1 closes.
        # So candle i-1 is the 'gap creator'.
        # At close of candle i (current), we check if i-1 created an FVG? No.
        # FVG is formed by i-2, i-1, i.
        # Let's say current is i.
        # i-1 was the big move.
        # We check gap between i-2 and i (current close/low).
        # Actually, standard ICT definition:
        # Bullish FVG: Candle 2 (middle) is large green. Low(3) > High(1).
        # Here we are at Candle 3 (i).
        # Candle 2 is i-1. Candle 1 is i-2.

        is_fvg_bullish = False
        is_fvg_bearish = False

        # Check if i-1 was Bullish
        if candle_i_1['close'] > candle_i_1['open']:
            # Check gap
            if candle_i['low'] > candle_i_2['high']:
                # There is a gap between 1 (i-2) High and 3 (i) Low.
                # However, this means price hasn't filled it yet.
                # Usually we enter ON the retest.
                # Here we are simulating. If gap exists, we might want to enter limit?
                # Simplified: If FVG exists and bias aligns, enter.
                is_fvg_bullish = True

        # Check if i-1 was Bearish
        if candle_i_1['close'] < candle_i_1['open']:
            if candle_i['high'] < candle_i_2['low']:
                is_fvg_bearish = True

        if bias == "bullish":
            if portfolio['position'] <= 0:
                if is_fvg_bullish:
                    # SL below i-1 Low
                    sl = candle_i_1['low']
                    if sl >= price: sl = price * 0.995 # Fallback
                    tp = price + (price - sl) * 2 # 2R
                    signal = {'action': 'buy', 'quantity': 1, 'sl': sl, 'tp': tp, 'reason': 'Bullish Bias + 5m FVG'}

        elif bias == "bearish":
            if portfolio['position'] >= 0:
                if is_fvg_bearish:
                    # SL above i-1 High
                    sl = candle_i_1['high']
                    if sl <= price: sl = price * 1.005 # Fallback
                    tp = price - (sl - price) * 2 # 2R
                    signal = {'action': 'sell', 'quantity': 1, 'sl': sl, 'tp': tp, 'reason': 'Bearish Bias + 5m FVG'}

        # Exit Logic (Bias Reversal)
        if portfolio['position'] > 0 and bias == "bearish":
             signal = {'action': 'close', 'reason': 'Bias Reversal'}
        if portfolio['position'] < 0 and bias == "bullish":
             signal = {'action': 'close', 'reason': 'Bias Reversal'}

        return signal
