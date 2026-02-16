import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str  # 'long' or 'short'
    quantity: int
    ticker: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = 'open'  # 'open', 'closed'
    tp: Optional[float] = None # Take Profit
    sl: Optional[float] = None # Stop Loss

class Backtester:
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.0, multiplier: float = 1.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.multiplier = multiplier # Contract multiplier (e.g. 20 for NQ, 50 for ES)

        self.portfolio: Dict[str, float] = {"cash": initial_capital, "position": 0.0}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # Current market state
        self.current_time = None
        self.current_price = None
        self.current_candle = None

    def run(self, data: pd.DataFrame, strategy, higher_timeframe_data: Dict[str, pd.DataFrame] = None, progress_callback=None, strategy_interval=1):
        """
        Runs the backtest loop chronologically.

        Args:
            data: Main timeframe data.
            strategy: Strategy instance.
            higher_timeframe_data: Dict of HTF dataframes.
            progress_callback: Optional function(current_step, total_steps, current_time, last_signal_info)
            strategy_interval: Check strategy logic every N candles (for AI optimization).
        """
        if data.empty:
            print("No data to backtest.")
            return

        data = data.sort_index()

        if hasattr(strategy, 'initialize'):
            strategy.initialize(data, higher_timeframe_data)

        total_steps = len(data)

        for i in range(total_steps):
            current_idx = data.index[i]
            current_row = data.iloc[i]

            self.current_time = current_idx
            self.current_price = current_row['close']
            self.current_candle = current_row

            # 1. Check Stops/Limits on OPEN trades using High/Low of current candle (ALWAYS CHECK)
            self._check_stops_limits(current_row)

            # 2. Update PnL (Unrealized)
            self.update_pnl(self.current_price, self.current_time)

            # 3. Strategy Decision (OPTIMIZED: Only run every N steps)
            signal = None
            if i % strategy_interval == 0:
                signal = strategy.next(current_idx, current_row, self.portfolio)

            # 4. Execute Trade
            if signal:
                self.execute_trade(signal)

            # Progress Update
            if progress_callback:
                # Get reasoning if available
                reason = signal.get('reason') if signal else "Hold/No Signal"
                # If AI strategy, it might have stored reason even if no signal? No, usually in signal.
                # If strategy has 'last_response', maybe we can parse?
                # But for speed, just pass signal info.

                # Update every iteration or every N
                progress_callback(i + 1, total_steps, current_idx, reason)

        # Close all positions at the end
        if self.portfolio['position'] != 0:
            self._close_all_positions(data.index[-1], data.iloc[-1]['close'], reason="End of Data")
            self.update_pnl(data.iloc[-1]['close'], data.index[-1])

    def _check_stops_limits(self, row):
        """
        Checks if current price hit SL or TP for open trades.
        """
        price_high = row['high']
        price_low = row['low']

        # Iterate over a copy to allow modification
        for trade in self.trades[:]:
            if trade.status == 'open':
                # Check SL
                hit_sl = False
                exec_price = None

                if trade.side == 'long':
                    if trade.sl and price_low <= trade.sl:
                        hit_sl = True
                        exec_price = trade.sl
                elif trade.side == 'short':
                    if trade.sl and price_high >= trade.sl:
                        hit_sl = True
                        exec_price = trade.sl

                if hit_sl:
                    self._close_trade(trade, self.current_time, exec_price, reason="SL Hit")
                    continue

                # Check TP
                hit_tp = False
                if trade.side == 'long':
                    if trade.tp and price_high >= trade.tp:
                        hit_tp = True
                        exec_price = trade.tp
                elif trade.side == 'short':
                    if trade.tp and price_low <= trade.tp:
                        hit_tp = True
                        exec_price = trade.tp

                if hit_tp:
                    self._close_trade(trade, self.current_time, exec_price, reason="TP Hit")

    def execute_trade(self, signal: Dict):
        """
        Executes a trade based on signal.
        """
        action = signal.get('action')
        quantity = signal.get('quantity', 1)
        ticker = signal.get('ticker', 'UNKNOWN')

        if action == 'buy':
            trade = Trade(
                entry_time=self.current_time,
                entry_price=self.current_price,
                side='long',
                quantity=quantity,
                ticker=ticker,
                sl=signal.get('sl'),
                tp=signal.get('tp')
            )
            self.trades.append(trade)
            self.portfolio['position'] += quantity

        elif action == 'sell':
            trade = Trade(
                entry_time=self.current_time,
                entry_price=self.current_price,
                side='short',
                quantity=quantity,
                ticker=ticker,
                sl=signal.get('sl'),
                tp=signal.get('tp')
            )
            self.trades.append(trade)
            self.portfolio['position'] -= quantity

        elif action == 'close':
            self._close_all_positions(self.current_time, self.current_price, reason="Signal Close")

    def _close_trade(self, trade: Trade, time, price, reason=""):
        if trade.status == 'closed':
            return

        trade.exit_time = time
        trade.exit_price = price
        trade.status = 'closed'

        # Calculate PnL
        if trade.side == 'long':
            pnl_points = (price - trade.entry_price)
        else:
            pnl_points = (trade.entry_price - price)

        trade.pnl = (pnl_points * self.multiplier * trade.quantity) - self.commission

        self.portfolio['position'] -= (trade.quantity if trade.side == 'long' else -trade.quantity)
        self.portfolio['cash'] += trade.pnl

        # Debug
        # print(f"Closed {trade.side} on {trade.ticker} at {price}. PnL: {trade.pnl:.2f}. Reason: {reason}")

    def _close_all_positions(self, time, price, reason=""):
        for trade in self.trades:
            if trade.status == 'open':
                self._close_trade(trade, time, price, reason)

    def update_pnl(self, current_price: float, current_time: pd.Timestamp):
        """
        Updates the Unrealized PnL and Equity Curve.
        """
        unrealized_pnl = 0.0

        for trade in self.trades:
            if trade.status == 'open':
                if trade.side == 'long':
                    pnl = (current_price - trade.entry_price)
                else:
                    pnl = (trade.entry_price - current_price)
                unrealized_pnl += pnl * self.multiplier * trade.quantity

        total_equity = self.portfolio['cash'] + unrealized_pnl

        self.equity_curve.append({
            'time': current_time,
            'equity': total_equity,
            'cash': self.portfolio['cash'],
            'unrealized_pnl': unrealized_pnl
        })

    def get_metrics(self) -> Dict:
        """
        Returns performance metrics.
        """
        if not self.trades:
            return {}

        closed_trades = [t for t in self.trades if t.status == 'closed']
        if not closed_trades:
            return {'total_trades': 0}

        total_pnl = sum(t.pnl for t in closed_trades)
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl <= 0]

        win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        # Max Drawdown
        if self.equity_curve:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0

        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_trades': len(closed_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'final_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        }
