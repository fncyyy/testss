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
                generation_config = {
                    "response_mime_type": "application/json",
                    "max_output_tokens": 250
                }
                self.model = genai.GenerativeModel(self.model_name, generation_config=generation_config)
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
            # Optimize: use CSV for fewer tokens
            recent_str = recent_data[['open', 'high', 'low', 'close', 'volume']].round(2).to_csv(header=True)

            # Calculate Session/Day High/Low for context
            day_high = round(recent_data['high'].max(), 2)
            day_low = round(recent_data['low'].min(), 2)

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
                        # Optimize: use CSV for fewer tokens
                        htf_str = htf_data[['open', 'high', 'low', 'close', 'volume']].round(2).to_csv(header=True)
                    except KeyError:
                        pass

            prompt = f"""
            STRATEGY_LOGIC:
            {self.strategy_text}

            MARKET_DATA:
            Time: {time_str}
            Status: {market_status}
            Session_High: {day_high}
            Session_Low: {day_low}
            Position: {portfolio['position']}

            RECENT_DATA_CSV:
            {recent_str}

            HTF_DATA_CSV:
            {htf_str}

            TASK:
            Analyze the data based on the strategy logic. Return valid JSON only.

            JSON Schema:
            {{
                "action": "buy" | "sell" | "close" | "hold",
                "quantity": 1,
                "sl": float,
                "tp": float,
                "reason": "string",
                "confluences": [
                    {{"type": "line", "label": "string", "price": float}},
                    {{"type": "zone", "label": "string", "top": float, "bottom": float}}
                ]
            }}
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
            except json.JSONDecodeError:
                # Fallback: Try extracting JSON object via regex
                import re
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    try:
                        signal = json.loads(match.group(0))
                    except:
                        signal = None
                else:
                    signal = None

            if isinstance(signal, dict):
                # Normalize keys and values
                signal = {k.lower(): v for k, v in signal.items()}

                if 'action' in signal and isinstance(signal['action'], str):
                    signal['action'] = signal['action'].lower()

                    if signal['action'] in ['buy', 'sell', 'close', 'hold']:
                        self.log(f"Signal: {signal}")
                        return signal

            self.log(f"Invalid Signal or JSON: {text}")
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

    def find_recent_fvg(self, current_time: pd.Timestamp, lookback: int = 10, bias: str = "neutral") -> Optional[Dict]:
        """
        Scans last `lookback` candles for an UNMITIGATED FVG aligned with bias.
        Returns the FVG dict if current price is interacting with it, else None.
        """
        loc = self.data.index.get_loc(current_time)
        if isinstance(loc, slice): loc = loc.stop - 1

        if loc < lookback: return None

        # Iterate backwards from current candle (i)
        # We look for FVG formed at i-k.
        # FVG is formed by candles k-2, k-1, k.
        # We need to check if it's still valid (not filled).

        current_row = self.data.iloc[loc]
        current_low = current_row['low']
        current_high = current_row['high']

        found_fvg = None

        # Scan for FVG formation
        for i in range(loc - 1, loc - lookback, -1):
            # i is the potential 3rd candle of the FVG pattern (the confirmation candle)
            # Pattern: i-2 (1), i-1 (2, displacement), i (3)

            c1 = self.data.iloc[i-2]
            c2 = self.data.iloc[i-1] # Displacement candle
            c3 = self.data.iloc[i]

            # Calculate Body Size for Displacement Check
            body_size = abs(c2['close'] - c2['open'])
            avg_body = abs(self.data['close'].iloc[i-6:i-1] - self.data['open'].iloc[i-6:i-1]).mean()

            # Displacement Filter: Body must be > 1.0x Avg (loose filter)
            is_displacement = body_size > avg_body

            if bias == "bullish":
                # Bullish FVG: C2 Green. Gap between C1 High and C3 Low.
                if c2['close'] > c2['open'] and is_displacement:
                    fvg_top = c3['low']
                    fvg_bottom = c1['high']

                    if fvg_top > fvg_bottom: # Valid Gap
                        # Check if current price (at loc) is inside or touched this zone
                        # Ideally, we want to buy when price dips into [fvg_bottom, fvg_top]
                        if current_low <= fvg_top and current_high >= fvg_bottom:
                            return {
                                'type': 'zone', 'label': 'Bullish FVG',
                                'top': fvg_top, 'bottom': fvg_bottom,
                                'time': c2.name
                            }

            elif bias == "bearish":
                # Bearish FVG: C2 Red. Gap between C1 Low and C3 High.
                if c2['close'] < c2['open'] and is_displacement:
                    fvg_top = c1['low']
                    fvg_bottom = c3['high']

                    if fvg_top > fvg_bottom: # Valid Gap
                         # Check if current price (at loc) is inside or touched this zone
                        if current_high >= fvg_bottom and current_low <= fvg_top:
                            return {
                                'type': 'zone', 'label': 'Bearish FVG',
                                'top': fvg_top, 'bottom': fvg_bottom,
                                'time': c2.name
                            }
        return None

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

        signal = None
        price = current_row['close']

        # 2. Check 5m FVG (Retracement Entry)
        fvg = self.find_recent_fvg(current_time, lookback=15, bias=bias)

        if bias == "bullish":
            if portfolio['position'] <= 0:
                if fvg:
                    # SL below FVG bottom
                    sl = fvg['bottom']
                    if sl >= price: sl = price * 0.995 # Fallback
                    tp = price + (price - sl) * 2 # 2R
                    signal = {
                        'action': 'buy',
                        'quantity': 1,
                        'sl': sl,
                        'tp': tp,
                        'reason': 'Bullish Bias + 5m FVG Retrace',
                        'confluences': [fvg]
                    }

        elif bias == "bearish":
            if portfolio['position'] >= 0:
                if fvg:
                    # SL above FVG top
                    sl = fvg['top']
                    if sl <= price: sl = price * 1.005 # Fallback
                    tp = price - (sl - price) * 2 # 2R
                    signal = {
                        'action': 'sell',
                        'quantity': 1,
                        'sl': sl,
                        'tp': tp,
                        'reason': 'Bearish Bias + 5m FVG Retrace',
                        'confluences': [fvg]
                    }

        # Exit Logic (Bias Reversal)
        if portfolio['position'] > 0 and bias == "bearish":
             signal = {'action': 'close', 'reason': 'Bias Reversal'}
        if portfolio['position'] < 0 and bias == "bullish":
             signal = {'action': 'close', 'reason': 'Bias Reversal'}

        return signal
