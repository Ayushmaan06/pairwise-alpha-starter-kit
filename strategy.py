import pandas as pd
import numpy as np

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced Multi-Lag Correlation Strategy for MATIC/BTC/ETH
    
    Strategy Logic:
    1. Detects multiple lag patterns (1H, 2H, 4H, 8H)
    2. Uses dynamic thresholds based on volatility
    3. Combines momentum, volume, and price action signals
    4. Implements risk management with SELL signals
    5. Uses weighted scoring system for signal strength
    
    Inputs:
    - candles_target: OHLCV for MATIC (1H)
    - candles_anchor: Merged OHLCV with BTC and ETH data (1H)
    
    Output:
    - DataFrame with ['timestamp', 'signal']
    """
    try:
        # Merge target and anchor data
        df = pd.merge(
            candles_target[['timestamp', 'open', 'high', 'low', 'close', 'volume']],
            candles_anchor,
            on='timestamp',
            how='inner'
        )
        
        # Calculate returns for different lag periods
        for lag in [1, 2, 4, 8]:
            df[f'btc_return_{lag}h'] = df['close_BTC_1H'].pct_change(lag)
            df[f'eth_return_{lag}h'] = df['close_ETH_1H'].pct_change(lag)
            df[f'target_return_{lag}h'] = df['close'].pct_change(lag)
        
        # Calculate rolling volatility for dynamic thresholds
        df['btc_volatility'] = df['close_BTC_1H'].pct_change().rolling(24).std()
        df['eth_volatility'] = df['close_ETH_1H'].pct_change().rolling(24).std()
        df['target_volatility'] = df['close'].pct_change().rolling(24).std()
        
        # Volume indicators
        df['btc_volume_sma'] = df['volume_BTC_1H'].rolling(24).mean()
        df['eth_volume_sma'] = df['volume_ETH_1H'].rolling(24).mean()
        df['target_volume_sma'] = df['volume'].rolling(24).mean()
        
        df['btc_volume_ratio'] = df['volume_BTC_1H'] / df['btc_volume_sma']
        df['eth_volume_ratio'] = df['volume_ETH_1H'] / df['eth_volume_sma']
        df['target_volume_ratio'] = df['volume'] / df['target_volume_sma']
        
        # Technical indicators
        df['target_rsi'] = calculate_rsi(df['close'], 14)
        df['target_sma_short'] = df['close'].rolling(12).mean()
        df['target_sma_long'] = df['close'].rolling(24).mean()
        df['target_momentum'] = (df['target_sma_short'] / df['target_sma_long']) - 1
        
        # Price action patterns
        df['target_range'] = (df['high'] / df['low']) - 1
        df['target_close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Generate signals using sophisticated logic
        signals = []
        
        for i in range(len(df)):
            if i < 24:  # Need sufficient data for calculations
                signals.append('HOLD')
                continue
                
            # Dynamic thresholds based on volatility
            btc_threshold = max(0.015, df['btc_volatility'].iloc[i] * 1.5)
            eth_threshold = max(0.015, df['eth_volatility'].iloc[i] * 1.5)
            
            # Multi-lag signal detection
            signal_score = 0
            
            # BTC signals with different weights for different lags
            for lag, weight in [(1, 0.3), (2, 0.4), (4, 0.5), (8, 0.2)]:
                btc_signal = df[f'btc_return_{lag}h'].iloc[i]
                if abs(btc_signal) > btc_threshold:
                    # Volume confirmation
                    volume_confirm = df['btc_volume_ratio'].iloc[i] > 1.2
                    vol_weight = 1.5 if volume_confirm else 1.0
                    
                    if btc_signal > 0:
                        signal_score += weight * vol_weight * min(btc_signal / btc_threshold, 3.0)
                    else:
                        signal_score -= weight * vol_weight * min(abs(btc_signal) / btc_threshold, 2.0)
            
            # ETH signals with different weights for different lags
            for lag, weight in [(1, 0.3), (2, 0.4), (4, 0.5), (8, 0.2)]:
                eth_signal = df[f'eth_return_{lag}h'].iloc[i]
                if abs(eth_signal) > eth_threshold:
                    # Volume confirmation
                    volume_confirm = df['eth_volume_ratio'].iloc[i] > 1.2
                    vol_weight = 1.5 if volume_confirm else 1.0
                    
                    if eth_signal > 0:
                        signal_score += weight * vol_weight * min(eth_signal / eth_threshold, 3.0)
                    else:
                        signal_score -= weight * vol_weight * min(abs(eth_signal) / eth_threshold, 2.0)
            
            # Additional filters and confirmations
            rsi = df['target_rsi'].iloc[i]
            momentum = df['target_momentum'].iloc[i]
            close_pos = df['target_close_position'].iloc[i]
            
            # RSI-based position sizing
            if rsi < 30:  # Oversold - more aggressive buying
                signal_score *= 1.3
            elif rsi > 70:  # Overbought - reduce buying, favor selling
                signal_score *= 0.7
            
            # Momentum confirmation
            if momentum > 0.02:  # Strong upward momentum
                signal_score *= 1.2
            elif momentum < -0.02:  # Strong downward momentum
                signal_score *= 0.8
            
            # Price position in daily range
            if close_pos > 0.8:  # Close near high
                signal_score *= 0.9
            elif close_pos < 0.2:  # Close near low
                signal_score *= 1.1
            
            # Decision logic
            if signal_score > 1.5:
                signals.append('BUY')
            elif signal_score < -1.0:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        df['signal'] = signals
        return df[['timestamp', 'signal']]
    
    except Exception as e:
        raise RuntimeError(f"Error in generate_signals: {e}")


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_coin_metadata() -> dict:
    """
    Specifies the target and anchor coins used in this strategy.
    
    Target: MATIC (Polygon) - High liquidity altcoin with strong BTC/ETH correlations
    Anchors: BTC and ETH on 1H timeframe for maximum signal resolution
    
    Returns:
    {
        "target": {"symbol": "MATIC", "timeframe": "1H"},
        "anchors": [
            {"symbol": "BTC", "timeframe": "1H"},
            {"symbol": "ETH", "timeframe": "1H"}
        ]
    }
    """
    return {
        "target": {
            "symbol": "MATIC",
            "timeframe": "1H"
        },
        "anchors": [
            {"symbol": "BTC", "timeframe": "1H"},
            {"symbol": "ETH", "timeframe": "1H"}
        ]
    }
