import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
from streamlit_autorefresh import st_autorefresh
import talib
import time
import requests
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Constants
DEFAULT_TIMEFRAMES = {
    '1m': {'interval': '1m', 'period': '1d'},
    '5m': {'interval': '5m', 'period': '5d'},
    '15m': {'interval': '15m', 'period': '15d'},
    '1h': {'interval': '1h', 'period': '30d'},
    '4h': {'interval': '4h', 'period': '60d'},
    '1d': {'interval': '1d', 'period': '180d'}
}

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
    confidence: float  # 0..1
    strength_score: float  # -1..1
    market_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    timestamp: datetime
    key_indicators: Dict
    signal_reasons: List[str]

@dataclass
class MarketRegime:
    trend: str  # BULL, BEAR, SIDEWAYS
    volatility: str  # LOW, NORMAL, HIGH
    volume_trend: str  # INCREASING, DECREASING, STABLE

class TradingStrategy:
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str]]:
        pass

class EnhancedTrendFollowingStrategy(TradingStrategy):
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []
        
        price = indicators.get('current_price', 0)
        sma_20 = indicators.get('bb_middle', price)
        sma_50 = indicators.get('sma_50', price)
        
        # Trend analysis
        adx = indicators.get('adx', 0)
        if adx > 25:
            trend_strength = min(1.0, (adx - 25) / 50)
            score += 0.4 * trend_strength
            reasons.append(f"Strong trend detected (ADX: {adx:.1f})")
        elif adx < 15:
            score -= 0.2
            reasons.append(f"Weak/choppy market (ADX: {adx:.1f})")
            
        # MA alignment
        if price > sma_20 > sma_50:
            score += 0.3
            reasons.append("Bullish MA alignment")
        elif price < sma_20 < sma_50:
            score -= 0.3
            reasons.append("Bearish MA alignment")
            
        # MACD momentum
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            score += 0.25
            reasons.append("MACD bullish momentum")
        elif macd_hist < 0:
            score -= 0.25
            reasons.append("MACD bearish momentum")
            
        return max(-1.0, min(1.0, score)), reasons

class AdvancedMomentumStrategy(TradingStrategy):
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []
        
        rsi = indicators.get('rsi', 50)
        
        # RSI signals
        if 30 < rsi < 45:
            rsi_signal = (45 - rsi) / 15 * 0.4
            score += rsi_signal
            reasons.append(f"RSI oversold recovery ({rsi:.1f})")
        elif 55 < rsi < 70:
            rsi_signal = (rsi - 55) / 15 * 0.3
            score += rsi_signal
            reasons.append(f"RSI momentum ({rsi:.1f})")
        elif rsi > 75:
            score -= 0.3
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 25:
            score += 0.3
            reasons.append(f"RSI oversold ({rsi:.1f})")
            
        # Momentum
        momentum_5 = indicators.get('momentum_5', 0)
        if momentum_5 > 2:
            score += 0.3
            reasons.append(f"Strong bullish momentum ({momentum_5:.1f}%)")
        elif momentum_5 < -2:
            score -= 0.3
            reasons.append(f"Strong bearish momentum ({momentum_5:.1f}%)")
            
        return max(-1.0, min(1.0, score)), reasons

class StatisticalArbitrageStrategy(TradingStrategy):
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []
        
        bb_position = indicators.get('bb_position', 0.5)
        z_score = (bb_position - 0.5) * 4
        
        if z_score < -2:
            score += 0.5 * (abs(z_score) - 2) / 2
            reasons.append(f"Mean reversion buy signal (Z-score: {z_score:.2f})")
        elif z_score > 2:
            score -= 0.5 * (z_score - 2) / 2
            reasons.append(f"Mean reversion sell signal (Z-score: {z_score:.2f})")
            
        return max(-1.0, min(1.0, score)), reasons

class MarketRegimeDetector:
    def detect_regime(self, market_data: Dict) -> MarketRegime:
        indicators = market_data.get('indicators', {})
        
        momentum_10 = indicators.get('momentum_10', 0)
        adx = indicators.get('adx', 0)
        
        if momentum_10 > 5 and adx > 25:
            trend = "BULL"
        elif momentum_10 < -5 and adx > 25:
            trend = "BEAR"
        else:
            trend = "SIDEWAYS"
            
        volatility = indicators.get('volatility', 0)
        if volatility > 30:
            vol_regime = "HIGH"
        elif volatility < 10:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"
            
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.2:
            volume_trend = "INCREASING"
        elif volume_ratio < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
            
        return MarketRegime(
            trend=trend,
            volatility=vol_regime,
            volume_trend=volume_trend
        )

class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 50:
            return {}
            
        indicators = {}
        try:
            close_prices = np.array(df['Close'], dtype=float)
            high_prices = np.array(df['High'], dtype=float)
            low_prices = np.array(df['Low'], dtype=float)
            volumes = np.array(df['Volume'], dtype=float) if 'Volume' in df.columns else None
            
            # Current price
            indicators['current_price'] = float(close_prices[-1])
            
            # Moving averages
            sma_20 = talib.SMA(close_prices, timeperiod=20)
            sma_50 = talib.SMA(close_prices, timeperiod=50)
            indicators['sma_20'] = float(sma_20[-1]) if len(sma_20) > 0 and not np.isnan(sma_20[-1]) else close_prices[-1]
            indicators['sma_50'] = float(sma_50[-1]) if len(sma_50) > 0 and not np.isnan(sma_50[-1]) else close_prices[-1]
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators['bb_upper'] = float(upper[-1]) if len(upper) > 0 and not np.isnan(upper[-1]) else close_prices[-1] * 1.02
            indicators['bb_lower'] = float(lower[-1]) if len(lower) > 0 and not np.isnan(lower[-1]) else close_prices[-1] * 0.98
            indicators['bb_middle'] = float(middle[-1]) if len(middle) > 0 and not np.isnan(middle[-1]) else close_prices[-1]
            
            bb_width = indicators['bb_upper'] - indicators['bb_lower']
            indicators['bb_position'] = (close_prices[-1] - indicators['bb_lower']) / bb_width if bb_width > 0 else 0.5
            
            # RSI
            rsi = talib.RSI(close_prices, timeperiod=14)
            indicators['rsi'] = float(rsi[-1]) if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50.0
            
            # MACD
            macd, signal, histogram = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = float(macd[-1]) if len(macd) > 0 and not np.isnan(macd[-1]) else 0.0
            indicators['macd_histogram'] = float(histogram[-1]) if len(histogram) > 0 and not np.isnan(histogram[-1]) else 0.0
            
            # ADX
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['adx'] = float(adx[-1]) if len(adx) > 0 and not np.isnan(adx[-1]) else 0.0
            
            # ATR
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['atr'] = float(atr[-1]) if len(atr) > 0 and not np.isnan(atr[-1]) else close_prices[-1] * 0.02
            
            # Volatility
            if len(close_prices) >= 20:
                returns = np.diff(close_prices) / close_prices[:-1]
                indicators['volatility'] = float(np.std(returns[-20:]) * np.sqrt(252) * 100)
            else:
                indicators['volatility'] = 20.0
                
            # Momentum
            if len(close_prices) >= 6:
                indicators['momentum_5'] = float((close_prices[-1] - close_prices[-6]) / close_prices[-6] * 100)
            if len(close_prices) >= 11:
                indicators['momentum_10'] = float((close_prices[-1] - close_prices[-11]) / close_prices[-11] * 100)
            
            # Volume
            if volumes is not None:
                volume_sma = talib.SMA(volumes, timeperiod=20)
                if len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) and volume_sma[-1] > 0:
                    indicators['volume_ratio'] = float(volumes[-1] / volume_sma[-1])
                else:
                    indicators['volume_ratio'] = 1.0
            else:
                indicators['volume_ratio'] = 1.0
                
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
            
        return indicators

class DataCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 120):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.access_times = {}
        
    def get(self, key: str) -> Optional[any]:
        if key in self.cache and time.time() - self.access_times[key] < self.ttl:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def set(self, key: str, value: any) -> None:
        if len(self.cache) >= self.max_size and self.access_times:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = value
        self.access_times[key] = time.time()

class PriceDataFetcher:
    @staticmethod
    def get_market_price(symbol: str, fallback_price: float) -> float:
        # Simple price fetcher with small random variation
        variation = np.random.uniform(-0.001, 0.001)
        return fallback_price * (1 + variation)

class TradingEngine:
    def __init__(self):
        self.strategies = [
            EnhancedTrendFollowingStrategy(),
            AdvancedMomentumStrategy(),
            StatisticalArbitrageStrategy()
        ]
        self.strategy_weights = {
            "EnhancedTrendFollowingStrategy": 0.40,
            "AdvancedMomentumStrategy": 0.35,
            "StatisticalArbitrageStrategy": 0.25
        }
        self.regime_detector = MarketRegimeDetector()
        
    def generate_signal(self, symbol: str, market_data: Dict) -> TradingSignal:
        indicators = market_data.get('indicators', {})
        if not indicators:
            return self._create_neutral_signal(symbol, market_data)
        
        # Detect market regime
        market_regime = self.regime_detector.detect_regime(market_data)
        
        # Calculate scores from all strategies
        strategy_scores = {}
        all_reasons = []
        
        for strategy in self.strategies:
            score, reasons = strategy.calculate_score(indicators, market_regime)
            strategy_scores[strategy.__class__.__name__] = score
            all_reasons.extend(reasons)
        
        # Calculate composite score
        composite_score = sum(score * self.strategy_weights.get(name, 0) 
                             for name, score in strategy_scores.items())
        
        # Classify signal
        signal_type, confidence = self._classify_signal(composite_score, indicators)
        
        # Calculate prices
        price = market_data['price']
        atr = indicators.get('atr', price * 0.01)
        volatility = indicators.get('volatility', 20) / 100
        
        # Stop loss and target
        stop_loss = self._calculate_stop_loss(price, atr, signal_type)
        target_price = self._calculate_target(price, stop_loss, signal_type)
        
        # Risk reward ratio
        risk_amount = abs(price - stop_loss)
        reward_amount = abs(target_price - price)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 1.0
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=float(confidence),
            strength_score=float(composite_score),
            market_price=float(price),
            entry_price=float(price),
            target_price=float(target_price),
            stop_loss=float(stop_loss),
            risk_reward_ratio=float(risk_reward_ratio),
            timestamp=datetime.now(),
            key_indicators=indicators,
            signal_reasons=all_reasons[:3] if all_reasons else ["No clear signals"]
        )
    
    def _classify_signal(self, score: float, indicators: Dict) -> Tuple[str, float]:
        if score > 0.7:
            return "STRONG_BUY", 0.85
        elif score > 0.3:
            return "BUY", 0.75
        elif score < -0.7:
            return "STRONG_SELL", 0.85
        elif score < -0.3:
            return "SELL", 0.75
        else:
            return "NEUTRAL", 0.50
    
    def _calculate_stop_loss(self, price: float, atr: float, signal_type: str) -> float:
        stop_distance = atr * 1.5
        
        if signal_type in ["STRONG_BUY", "BUY"]:
            return price - stop_distance
        elif signal_type in ["STRONG_SELL", "SELL"]:
            return price + stop_distance
        else:
            return price
    
    def _calculate_target(self, price: float, stop_loss: float, signal_type: str) -> float:
        risk = abs(price - stop_loss)
        reward = risk * 2.0  # 1:2 risk-reward ratio
        
        if signal_type in ["STRONG_BUY", "BUY"]:
            return price + reward
        elif signal_type in ["STRONG_SELL", "SELL"]:
            return price - reward
        else:
            return price
    
    def _create_neutral_signal(self, symbol: str, market_data: Dict) -> TradingSignal:
        price = market_data['price']
        return TradingSignal(
            symbol=symbol,
            signal_type="NEUTRAL",
            confidence=0.5,
            strength_score=0.0,
            market_price=float(price),
            entry_price=float(price),
            target_price=float(price),
            stop_loss=float(price * 0.98),
            risk_reward_ratio=1.0,
            timestamp=datetime.now(),
            key_indicators={},
            signal_reasons=["No clear signals"]
        )

class DataManager:
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.data_fetcher = PriceDataFetcher()
        self.cache = DataCache(max_size=100, ttl_seconds=120)
        
    def get_market_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        cache_key = f"{symbol}_{timeframe}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            timeframe_config = DEFAULT_TIMEFRAMES.get(timeframe, DEFAULT_TIMEFRAMES['15m'])
            interval, period = timeframe_config['interval'], timeframe_config['period']
            
            hist = yf.Ticker(symbol).history(period=period, interval=interval)
            if hist.empty or len(hist) < 50:
                return None
            
            hist = hist.ffill().bfill()
            if hist.isnull().values.any():
                return None
            
            last_price = float(hist['Close'].iloc[-1])
            market_price = self.data_fetcher.get_market_price(symbol, last_price)
            
            indicators = self.analyzer.calculate_indicators(hist)
            if not indicators:
                return None
            
            result = {
                'symbol': symbol,
                'price': market_price,
                'indicators': indicators,
                'timeframe': timeframe
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

def pretty_asset(symbol: str) -> str:
    """Convert symbol to readable format"""
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-USD", "/USD")
    return symbol

def compute_entry_zone(price: float, atr: float, confidence: float = 0.8) -> Tuple[float, float]:
    """Calculate entry zone"""
    band_multiplier = 0.5 if confidence > 0.8 else 0.3
    band = max(atr * band_multiplier, price * 0.001)
    return (price - band, price + band)

def format_signal(symbol: str, signal: TradingSignal) -> str:
    """Format signal for display"""
    entry_low, entry_high = compute_entry_zone(
        signal.entry_price,
        signal.key_indicators.get('atr', signal.market_price * 0.01),
        signal.confidence
    )
    
    # Determine color based on signal
    if signal.signal_type in ["STRONG_BUY", "BUY"]:
        color = "🟢"
    elif signal.signal_type in ["STRONG_SELL", "SELL"]:
        color = "🔴"
    else:
        color = "🟠"
    
    reasoning = signal.signal_reasons[0] if signal.signal_reasons else "No clear signals"
    
    return f"""{color} **{signal.signal_type.replace('_', ' ').upper()}**

**[Asset]** {pretty_asset(symbol)}
**[Timeframe]** 1m
**[Contract Period]** 15 min
**[Entry Zone]** {entry_low:.2f} – {entry_high:.2f}
**[Target]** {signal.target_price:.2f}
**[Stop Loss]** {signal.stop_loss:.2f}
**[Confidence]** {int(signal.confidence * 100)}%
**[Reasoning]** {reasoning}

---"""

class SimpleTradingApp:
    def __init__(self):
        self.engine = TradingEngine()
        self.data_manager = DataManager()
        
    def render_sidebar(self):
        st.sidebar.header("Configuration")
        
        # Asset selection
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"]
        forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
        crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        
        selected_stocks = st.sidebar.multiselect("Stocks", stock_symbols, default=["AAPL", "TSLA"])
        selected_fx = st.sidebar.multiselect("Forex", forex_symbols, default=["EURUSD=X"])
        selected_crypto = st.sidebar.multiselect("Crypto", crypto_symbols, default=["BTC-USD"])
        
        symbols = selected_stocks + selected_fx + selected_crypto
        
        # Filters
        min_confidence = st.sidebar.slider("Min Confidence (%)", 50, 90, 65) / 100
        
        # Auto refresh
        refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60, step=30)
        st_autorefresh(interval=refresh_sec * 1000, key="simple_auto_refresh")
        
        return {
            'symbols': symbols,
            'min_confidence': min_confidence
        }
    
    def run(self):
        st.set_page_config(
            page_title="Trading Signals",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("Trading Signals")
        
        config = self.render_sidebar()
        
        if not config['symbols']:
            st.warning("Please select at least one asset from the sidebar.")
            return
        
        # Generate signals
        signals = []
        all_data = {}
        
        progress_bar = st.progress(0)
        for i, symbol in enumerate(config['symbols']):
            progress_bar.progress((i + 1) / len(config['symbols']))
            
            data = self.data_manager.get_market_data(symbol, '15m')
            if data:
                signal = self.engine.generate_signal(symbol, data)
                all_data[symbol] = data  # Store data for WAIT signals
                
                if config['show_all_signals']:
                    # Show all signals regardless of confidence
                    signals.append((symbol, signal))
                elif signal.confidence >= config['min_confidence']:
                    # Only show high confidence signals
                    signals.append((symbol, signal))
        
        progress_bar.empty()
        
        # Display signals
        if signals:
            # Sort by confidence
            signals.sort(key=lambda x: x[1].confidence, reverse=True)
            
            for symbol, signal in signals:
                if signal.confidence < config['min_confidence'] and config['show_all_signals']:
                    # Show as WAIT signal if below confidence threshold
                    current_price = all_data.get(symbol, {}).get('price', signal.market_price)
                    st.markdown(format_wait_signal(symbol, current_price))
                else:
                    # Show regular signal
                    st.markdown(format_signal(symbol, signal))
        else:
            # Show WAIT signals for all assets when no signals meet criteria
            for symbol in config['symbols']:
                if symbol in all_data:
                    current_price = all_data[symbol]['price']
                    st.markdown(format_wait_signal(symbol, current_price))

if __name__ == "__main__":
    app = SimpleTradingApp()
    app.run()
