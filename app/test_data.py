import pandas as pd
import numpy as np
from app.backtester import Backtester

# Mock Strategy
class MockStrategy:
    def initialize(self, data, higher_timeframe_data):
        self.data = data
        self.count = 0

    def next(self, current_idx, current_row, portfolio):
        self.count += 1
        # Buy on 3rd candle, Sell on 6th
        if self.count == 3:
            return {'action': 'buy', 'quantity': 1, 'ticker': 'TEST', 'sl': current_row['close'] - 5, 'tp': current_row['close'] + 10}
        if self.count == 6:
            return {'action': 'close', 'ticker': 'TEST'}
        return None

def test_backtester():
    print("Testing Backtester...")

    # Create mock data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='5min')
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low':  [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close':[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000] * 10
    }, index=dates)

    # Initialize Backtester
    # Multiplier 1 for simplicity
    bt = Backtester(initial_capital=10000, commission=1.0, multiplier=1.0)
    strategy = MockStrategy()

    bt.run(data, strategy)

    metrics = bt.get_metrics()
    print("Metrics:", metrics)

    # Validation
    # Trade 1: Buy at candle 3 (index 2).
    # current_row at index 2: close=102.
    # Entry price: 102.
    # SL: 97, TP: 112.

    # Trade 2: Close at candle 6 (index 5).
    # current_row at index 5: close=105.
    # Exit price: 105.

    # PnL: (105 - 102) * 1 * 1 - 1.0 = 3 - 1 = 2.0.

    if len(bt.trades) == 1:
        print("Trade Count: PASS")
        trade = bt.trades[0]
        print(f"Trade PnL: {trade.pnl} (Expected 2.0)")
        if abs(trade.pnl - 2.0) < 1e-6:
            print("PnL Check: PASS")
        else:
            print("PnL Check: FAIL")
    else:
        print(f"Trade Count: FAIL (Found {len(bt.trades)})")

    # Test SL Logic
    print("\nTesting SL Logic...")
    bt_sl = Backtester(initial_capital=10000, commission=0.0)

    class SLStrategy:
        def initialize(self, data, ht): pass
        def next(self, idx, row, port):
            if row['close'] == 100: # First candle
                return {'action': 'buy', 'quantity': 1, 'sl': 95}
            return None

    # Data that drops to 90
    data_sl = pd.DataFrame({
        'open': [100, 100, 100],
        'high': [105, 105, 95],
        'low':  [95, 98, 90], # Drops to 90 on 3rd candle
        'close':[100, 99, 92],
        'volume': [100]*3
    }, index=dates[:3])

    bt_sl.run(data_sl, SLStrategy())

    if len(bt_sl.trades) == 1 and bt_sl.trades[0].status == 'closed':
        print("SL Trigger: PASS")
        print(f"Exit Price: {bt_sl.trades[0].exit_price} (Expected 95)")
        # Note: In my logic, if Low <= SL, exec at SL.
        # Candle 1: Low 95. SL 95. Hit?
        # Entry is at Close of Candle 1 (100).
        # Wait, strategy.next is called at end of candle 1.
        # execution happens? In my code, "execute_trade" happens immediately at current price (Close).
        # So trade entered at 100.
        # Next loop (Candle 2): Low 98. SL 95. Not hit.
        # Next loop (Candle 3): Low 90. SL 95. Hit!
        # Should exit at 95.
    else:
        print("SL Trigger: FAIL")
        for t in bt_sl.trades: print(t)

if __name__ == "__main__":
    test_backtester()
