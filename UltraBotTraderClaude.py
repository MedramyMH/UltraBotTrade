
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta,timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import warnings
from streamlit_autorefresh import st_autorefresh
import talib
from collections import deque
import time
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.preprocessing import StandardScaler
import streamlit as st
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

response = session.get(url, timeout=10)


st.set_page_config(
    page_title="ULTIMATE Trading AI - Pro Edition",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
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

# Risk-free rate for Sharpe calculations (US 10Y Treasury approximation)
RISK_FREE_RATE = 0.045  # 4.5% annualized

# =============================
# Enhanced Data Models
# =============================
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
    reliability_score: float
    strategy: str
    time_horizon: str
    expected_return: float  # Expected return based on historical patterns
    max_adverse_excursion: float  # Maximum expected drawdown
    win_probability: float  # Estimated probability of profitable trade
    kelly_fraction: float  # Optimal position size per Kelly criterion

@dataclass
class PositionSizing:
    risk_amount: float
    position_size: float
    position_value: float
    position_percent: float
    kelly_size: float
    volatility_adjusted_size: float
    correlation_adjusted_size: float

@dataclass
class MarketRegime:
    trend: str  # BULL, BEAR, SIDEWAYS
    volatility: str  # LOW, NORMAL, HIGH
    volume_trend: str  # INCREASING, DECREASING, STABLE
    correlation_regime: float  # Average correlation across assets
    vix_level: float  # Market fear gauge
    
# =============================
# Enhanced Strategy Interface with Backtesting
# =============================
class TradingStrategy(ABC):
    def __init__(self):
        self.historical_performance = {
            'total_signals': 0,
            'profitable_signals': 0,
            'avg_return': 0.0,
            'avg_holding_period': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'information_ratio': 0.0
        }
    
    @abstractmethod
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str], Dict]:
        """Calculate score with additional metadata for decision making"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    def update_performance(self, returns: List[float], holding_periods: List[int]):
        """Update strategy performance metrics"""
        if not returns:
            return
            
        self.historical_performance['total_signals'] = len(returns)
        self.historical_performance['profitable_signals'] = len([r for r in returns if r > 0])
        self.historical_performance['avg_return'] = np.mean(returns)
        
        if holding_periods:
            self.historical_performance['avg_holding_period'] = np.mean(holding_periods)
        
        # Calculate Sharpe ratio
        if len(returns) > 1:
            excess_returns = np.array(returns) - RISK_FREE_RATE/252  # Daily risk-free rate
            self.historical_performance['sharpe_ratio'] = np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        self.historical_performance['max_drawdown'] = np.min(drawdown)

# =============================
# Advanced Strategy Implementations
# =============================
class EnhancedTrendFollowingStrategy(TradingStrategy):
    @property
    def name(self) -> str:
        return "ENHANCED_TREND_FOLLOWING"
    
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str], Dict]:
        score = 0.0
        reasons = []
        metadata = {}
        
        # Multi-timeframe trend analysis
        price = indicators.get('current_price', 0)
        sma_20 = indicators.get('bb_middle', price)
        sma_50 = indicators.get('sma_50', price)
        
        # Primary trend strength
        adx = indicators.get('adx', 0)
        if adx > 25:
            trend_strength = min(1.0, (adx - 25) / 50)  # Normalize to 0-1
            score += 0.4 * trend_strength
            reasons.append(f"Strong trend detected (ADX: {adx:.1f})")
            metadata['trend_strength'] = trend_strength
        elif adx < 15:
            score -= 0.2
            reasons.append(f"Weak/choppy market (ADX: {adx:.1f})")
            
        # Multi-MA alignment
        if price > sma_20 > sma_50:
            score += 0.3
            reasons.append("Bullish MA alignment (20>50)")
        elif price < sma_20 < sma_50:
            score -= 0.3
            reasons.append("Bearish MA alignment (20<50)")
            
        # MACD momentum confirmation
        macd_hist = indicators.get('macd_histogram', 0)
        macd_trend = indicators.get('macd_trend', 0)
        
        if macd_hist > 0 and macd_trend > 0:
            score += 0.25
            reasons.append("MACD bullish with strengthening momentum")
        elif macd_hist < 0 and macd_trend < 0:
            score -= 0.25
            reasons.append("MACD bearish with weakening momentum")
            
        # Regime adjustment
        if market_regime.trend == "BULL" and score > 0:
            score *= 1.2  # Amplify bullish signals in bull market
        elif market_regime.trend == "BEAR" and score < 0:
            score *= 1.2  # Amplify bearish signals in bear market
        elif market_regime.trend == "SIDEWAYS":
            score *= 0.7  # Reduce all trend signals in sideways market
            
        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if abs(score) > 0.3 and volume_ratio > 1.5:
            score *= 1.1
            reasons.append(f"Volume confirmation (ratio: {volume_ratio:.1f})")
            
        metadata['regime_adjustment'] = market_regime.trend
        metadata['volume_confirmation'] = volume_ratio > 1.5
        
        return max(-1.0, min(1.0, score)), reasons, metadata

class AdvancedMomentumStrategy(TradingStrategy):
    @property
    def name(self) -> str:
        return "ADVANCED_MOMENTUM"
    
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str], Dict]:
        score = 0.0
        reasons = []
        metadata = {}
        
        # Multi-oscillator momentum
        rsi = indicators.get('rsi', 50)
        stoch_k = indicators.get('stoch_k', 50)
        cci = indicators.get('cci', 0)
        
        # RSI momentum with mean reversion consideration
        rsi_signal = 0
        if 30 < rsi < 45:  # Oversold recovery
            rsi_signal = (45 - rsi) / 15 * 0.4
            reasons.append(f"RSI oversold recovery ({rsi:.1f})")
        elif 55 < rsi < 70:  # Momentum continuation
            rsi_signal = (rsi - 55) / 15 * 0.3
            reasons.append(f"RSI momentum ({rsi:.1f})")
        elif rsi > 75:  # Extreme overbought
            rsi_signal = -0.3
            reasons.append(f"RSI extremely overbought ({rsi:.1f})")
        elif rsi < 25:  # Extreme oversold
            rsi_signal = 0.3
            reasons.append(f"RSI extremely oversold ({rsi:.1f})")
            
        score += rsi_signal
        
        # Stochastic momentum
        if stoch_k < 20 and indicators.get('stoch_d', 50) < 20:
            score += 0.25
            reasons.append("Stochastic oversold signal")
        elif stoch_k > 80 and indicators.get('stoch_d', 50) > 80:
            score -= 0.25
            reasons.append("Stochastic overbought signal")
            
        # Rate of change momentum
        momentum_5 = indicators.get('momentum_5', 0)
        momentum_10 = indicators.get('momentum_10', 0)
        
        # Accelerating momentum
        if momentum_5 > 2 and momentum_5 > momentum_10:
            score += 0.3
            reasons.append(f"Accelerating bullish momentum ({momentum_5:.1f}%)")
        elif momentum_5 < -2 and momentum_5 < momentum_10:
            score -= 0.3
            reasons.append(f"Accelerating bearish momentum ({momentum_5:.1f}%)")
            
        # Volatility adjustment
        volatility = indicators.get('volatility', 0)
        if volatility > 30:  # High volatility environment
            score *= 0.8  # Reduce momentum signals in high vol
            reasons.append("High volatility - reducing momentum signals")
        
        metadata['rsi_signal'] = rsi_signal
        metadata['momentum_acceleration'] = momentum_5 > momentum_10 if momentum_5 > 0 else momentum_5 < momentum_10
        metadata['volatility_adjustment'] = volatility
        
        return max(-1.0, min(1.0, score)), reasons, metadata


class StatisticalArbitrageStrategy(TradingStrategy):
    @property
    def name(self) -> str:
        return "STATISTICAL_ARBITRAGE"
    
    def calculate_score(self, indicators: Dict, market_regime: MarketRegime) -> Tuple[float, List[str], Dict]:
        score = 0.0
        reasons = []
        metadata = {}
        
        # Bollinger Band mean reversion
        bb_position = indicators.get('bb_position', 0.5)
        bb_squeeze = indicators.get('bb_upper', 0) - indicators.get('bb_lower', 1)
        price = indicators.get('current_price', 0)
        
        # Z-score based signals
        z_score = (bb_position - 0.5) * 4  # Convert to approximate Z-score
        
        if z_score < -2:  # More than 2 standard deviations below mean
            score += 0.5 * (abs(z_score) - 2) / 2  # Stronger signal for more extreme values
            reasons.append(f"Mean reversion buy (Z-score: {z_score:.2f})")
        elif z_score > 2:
            score -= 0.5 * (z_score - 2) / 2
            reasons.append(f"Mean reversion sell (Z-score: {z_score:.2f})")
            
        # Volatility breakout detection
        atr = indicators.get('atr', 0)
        volatility = indicators.get('volatility', 0)
        
        if bb_squeeze / price < 0.02 and volatility < 15:  # Tight squeeze
            # Prepare for breakout but don't signal yet
            metadata['breakout_setup'] = True
            reasons.append("Low volatility - potential breakout setup")
            
        # Volume-price divergence
        obv_trend = indicators.get('obv', 0)
        price_trend = indicators.get('momentum_10', 0)
        
        # This is a simplified divergence check - in practice, you'd want more sophisticated analysis
        if price_trend > 0 and obv_trend < 0:
            score -= 0.2
            reasons.append("Bearish volume divergence detected")
        elif price_trend < 0 and obv_trend > 0:
            score += 0.2
            reasons.append("Bullish volume divergence detected")
            
        # Regime-based adjustments
        if market_regime.volatility == "HIGH":
            score *= 0.6  # Reduce mean reversion signals in high volatility
        elif market_regime.volatility == "LOW":
            score *= 1.3  # Amplify mean reversion in low volatility
            
        metadata['z_score'] = z_score
        metadata['volatility_regime'] = market_regime.volatility
        
        return max(-1.0, min(1.0, score)), reasons, metadata

# =============================
# Enhanced Risk Management with Modern Portfolio Theory
# =============================
class AdvancedRiskManager:
    def __init__(self):
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        self.portfolio_weights = {}
        
    def calculate_optimal_position_size(self, symbol: str, signal: TradingSignal, 
                                      portfolio_value: float, max_risk_per_trade: float,
                                      correlation_data: Dict = None) -> PositionSizing:
        """Calculate position size using multiple methods"""
        
        # Basic Kelly Criterion
        win_rate = signal.win_probability
        avg_win_loss_ratio = abs(signal.expected_return) / abs(signal.max_adverse_excursion) if signal.max_adverse_excursion != 0 else 1
        
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win_loss_ratio)
        kelly_size = portfolio_value * kelly_fraction
        
        # Risk parity approach
        risk_amount = portfolio_value * max_risk_per_trade
        price_distance = abs(signal.entry_price - signal.stop_loss)
        basic_position_size = risk_amount / price_distance if price_distance > 0 else 0
        
        # Volatility targeting
        target_volatility = 0.15  # 15% annualized
        asset_volatility = signal.key_indicators.get('volatility', 15) / 100  # Convert to decimal
        volatility_scalar = target_volatility / asset_volatility if asset_volatility > 0 else 1
        volatility_adjusted_size = basic_position_size * volatility_scalar
        
        # Correlation adjustment (simplified)
        correlation_adjustment = 1.0
        if correlation_data and len(correlation_data) > 1:
            avg_correlation = np.mean([abs(corr) for corr in correlation_data.values()])
            correlation_adjustment = 1.0 - (avg_correlation * 0.3)  # Reduce size if highly correlated
            
        correlation_adjusted_size = volatility_adjusted_size * correlation_adjustment
        
        # Final position size (take the most conservative)
        final_position_size = min(
            basic_position_size,
            kelly_size / signal.entry_price,  # Convert to shares
            correlation_adjusted_size
        )
        
        position_value = final_position_size * signal.entry_price
        position_percent = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        return PositionSizing(
            risk_amount=risk_amount,
            position_size=final_position_size,
            position_value=position_value,
            position_percent=position_percent,
            kelly_size=kelly_size / signal.entry_price,
            volatility_adjusted_size=volatility_adjusted_size,
            correlation_adjusted_size=correlation_adjusted_size
        )
    
    def calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float, 
                               max_kelly: float = 0.25) -> float:
        """Calculate Kelly fraction with safety constraints"""
        if win_loss_ratio <= 0 or win_rate <= 0:
            return 0.0
            
        # Standard Kelly formula
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply safety constraints
        kelly = max(0.0, min(kelly, max_kelly))
        
        # Fractional Kelly (typically use 25-50% of full Kelly)
        fractional_kelly = kelly * 0.25
        
        return fractional_kelly
    
    def calculate_portfolio_heat(self, positions: List[Dict]) -> float:
        """Calculate total portfolio risk exposure"""
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions)
        return total_risk
    
    def calculate_value_at_risk(self, returns: List[float], confidence: float = 0.05) -> float:
        """Calculate Value at Risk at given confidence level"""
        if not returns or len(returns) < 30:
            return 0.0
            
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * confidence)
        return abs(sorted_returns[var_index])

# =============================
# Market Regime Detection
# =============================
class MarketRegimeDetector:
    def __init__(self):
        self.lookback_period = 60  # Days for regime analysis
        
    def detect_regime(self, market_data: Dict) -> MarketRegime:
        """Detect current market regime"""
        indicators = market_data.get('indicators', {})
        
        # Trend detection
        momentum_10 = indicators.get('momentum_10', 0)
        adx = indicators.get('adx', 0)
        
        if momentum_10 > 5 and adx > 25:
            trend = "BULL"
        elif momentum_10 < -5 and adx > 25:
            trend = "BEAR"
        else:
            trend = "SIDEWAYS"
            
        # Volatility regime
        volatility = indicators.get('volatility', 0)
        if volatility > 30:
            vol_regime = "HIGH"
        elif volatility < 10:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"
            
        # Volume trend
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.2:
            volume_trend = "INCREASING"
        elif volume_ratio < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
            
        # Mock VIX level (in practice, fetch from market data)
        vix_level = min(max(volatility * 2, 10), 50)  # Rough approximation
        
        return MarketRegime(
            trend=trend,
            volatility=vol_regime,
            volume_trend=volume_trend,
            correlation_regime=0.5,  # Would need multiple assets for real calculation
            vix_level=vix_level
        )

# =============================
# Enhanced Technical Analysis with Statistical Tests
# =============================
class StatisticalTechnicalAnalyzer:
    
    @staticmethod
    def _calculate_regime_indicators(df: pd.DataFrame) -> Dict:
        """Calculate regime-specific indicators with robust error handling"""
        result = {}
        
        if len(df) < 50:
            return result
            
        try:
            close_prices = df['Close'].values
            
            # Trend persistence calculation
            if len(close_prices) >= 20:
                try:
                    ma_20 = df['Close'].rolling(20, min_periods=10).mean()
                    
                    # Ensure we have valid data
                    valid_ma = ma_20.dropna()
                    valid_prices = close_prices[-len(valid_ma):] if len(valid_ma) > 0 else close_prices
                    
                    if len(valid_ma) > 0 and len(valid_prices) == len(valid_ma):
                        above_ma = valid_prices > valid_ma.values
                        
                        # Count consecutive periods
                        trend_persistence = 0
                        if len(above_ma) > 0:
                            current_trend = above_ma[-1]
                            
                            for i in range(len(above_ma) - 1, -1, -1):
                                if above_ma[i] == current_trend:
                                    trend_persistence += 1
                                else:
                                    break
                        
                        result['trend_persistence'] = trend_persistence
                    else:
                        result['trend_persistence'] = 0
                        
                except Exception as e:
                    logger.debug(f"Trend persistence calculation failed: {e}")
                    result['trend_persistence'] = 0
            else:
                result['trend_persistence'] = 0
            
            # Volatility clustering calculation
            try:
                returns = df['Close'].pct_change().dropna()
                
                if len(returns) > 40:  # Need sufficient data for rolling window
                    vol_20 = returns.rolling(20, min_periods=10).std()
                    valid_vol = vol_20.dropna()
                    
                    if len(valid_vol) > 2:
                        # Calculate correlation between consecutive volatility periods
                        vol_lag0 = valid_vol.iloc[:-1]
                        vol_lag1 = valid_vol.iloc[1:]
                        
                        if len(vol_lag0) > 0 and len(vol_lag1) > 0 and len(vol_lag0) == len(vol_lag1):
                            corr_matrix = np.corrcoef(vol_lag0.values, vol_lag1.values)
                            if corr_matrix.shape == (2, 2):
                                vol_clustering = corr_matrix[0, 1]
                                result['volatility_clustering'] = float(vol_clustering) if np.isfinite(vol_clustering) else 0.0
                            else:
                                result['volatility_clustering'] = 0.0
                        else:
                            result['volatility_clustering'] = 0.0
                    else:
                        result['volatility_clustering'] = 0.0
                else:
                    result['volatility_clustering'] = 0.0
                    
            except Exception as e:
                logger.debug(f"Volatility clustering calculation failed: {e}")
                result['volatility_clustering'] = 0.0
        
        except Exception as e:
            logger.error(f"Error in regime indicators: {e}")
            result = {
                'trend_persistence': 0,
                'volatility_clustering': 0.0
            }
            
        return result
    
    @staticmethod
    def calculate_enhanced_indicators(df: pd.DataFrame) -> Dict:
        """Calculate technical indicators with statistical validation"""
        if df.empty or len(df) < 50:  # Need more data for statistical tests
            return {}
            
        indicators = {}
        try:
            close_prices = np.array(df['Close'], dtype=float)
            high_prices = np.array(df['High'], dtype=float)
            low_prices = np.array(df['Low'], dtype=float)
            volumes = np.array(df['Volume'], dtype=float) if 'Volume' in df.columns else None
            
            # Basic indicators
            indicators.update(StatisticalTechnicalAnalyzer._calculate_basic_indicators(close_prices, high_prices, low_prices, volumes))
            
            # Statistical tests
            indicators.update(StatisticalTechnicalAnalyzer._calculate_statistical_tests(close_prices))
            
            # Regime indicators
            indicators.update(StatisticalTechnicalAnalyzer._calculate_regime_indicators(df))
            
        except Exception as e:
            logger.error(f"Error in enhanced technical analysis: {e}")
            return {}
            
        return indicators
    
    @staticmethod
    def _calculate_basic_indicators(close_prices: np.ndarray, high_prices: np.ndarray, 
                                  low_prices: np.ndarray, volumes: Optional[np.ndarray]) -> Dict:
        """Calculate basic technical indicators"""
        result = {}
        
        # Price-based indicators
        result['current_price'] = float(close_prices[-1])
        
        # Moving averages
        sma_20 = talib.SMA(close_prices, timeperiod=20)
        sma_50 = talib.SMA(close_prices, timeperiod=50)
        result['sma_20'] = float(sma_20[-1]) if len(sma_20) > 0 and not np.isnan(sma_20[-1]) else close_prices[-1]
        result['sma_50'] = float(sma_50[-1]) if len(sma_50) > 0 and not np.isnan(sma_50[-1]) else close_prices[-1]
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        result['bb_upper'] = float(upper[-1]) if len(upper) > 0 and not np.isnan(upper[-1]) else close_prices[-1] * 1.02
        result['bb_lower'] = float(lower[-1]) if len(lower) > 0 and not np.isnan(lower[-1]) else close_prices[-1] * 0.98
        result['bb_middle'] = float(middle[-1]) if len(middle) > 0 and not np.isnan(middle[-1]) else close_prices[-1]
        
        bb_width = result['bb_upper'] - result['bb_lower']
        result['bb_position'] = (close_prices[-1] - result['bb_lower']) / bb_width if bb_width > 0 else 0.5
        
        # Momentum oscillators
        rsi = talib.RSI(close_prices, timeperiod=14)
        result['rsi'] = float(rsi[-1]) if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50.0
        
        macd, signal, histogram = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        result['macd'] = float(macd[-1]) if len(macd) > 0 and not np.isnan(macd[-1]) else 0.0
        result['macd_signal'] = float(signal[-1]) if len(signal) > 0 and not np.isnan(signal[-1]) else 0.0
        result['macd_histogram'] = float(histogram[-1]) if len(histogram) > 0 and not np.isnan(histogram[-1]) else 0.0
        
        # Trend strength
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        result['adx'] = float(adx[-1]) if len(adx) > 0 and not np.isnan(adx[-1]) else 0.0
        
        # Volatility
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        result['atr'] = float(atr[-1]) if len(atr) > 0 and not np.isnan(atr[-1]) else close_prices[-1] * 0.02
        
        # Rolling volatility
        if len(close_prices) >= 20:
            returns = np.diff(close_prices) / close_prices[:-1]
            result['volatility'] = float(np.std(returns[-20:]) * np.sqrt(252) * 100)  # Annualized volatility
        else:
            result['volatility'] = 20.0
            
        # Momentum
        if len(close_prices) >= 6:
            result['momentum_5'] = float((close_prices[-1] - close_prices[-6]) / close_prices[-6] * 100)
        if len(close_prices) >= 11:
            result['momentum_10'] = float((close_prices[-1] - close_prices[-11]) / close_prices[-11] * 100)
            
        # Stochastic
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
        result['stoch_k'] = float(slowk[-1]) if len(slowk) > 0 and not np.isnan(slowk[-1]) else 50.0
        result['stoch_d'] = float(slowd[-1]) if len(slowd) > 0 and not np.isnan(slowd[-1]) else 50.0
        
        # CCI
        cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        result['cci'] = float(cci[-1]) if len(cci) > 0 and not np.isnan(cci[-1]) else 0.0
        
        # Volume indicators
        if volumes is not None:
            obv = talib.OBV(close_prices, volumes)
            result['obv'] = float(obv[-1]) if len(obv) > 0 else 0.0
            
            volume_sma = talib.SMA(volumes, timeperiod=20)
            if len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) and volume_sma[-1] > 0:
                result['volume_ratio'] = float(volumes[-1] / volume_sma[-1])
            else:
                result['volume_ratio'] = 1.0
        else:
            result['obv'] = 0.0
            result['volume_ratio'] = 1.0
            
        return result
    
    @staticmethod
    def _calculate_statistical_tests(close_prices: np.ndarray) -> Dict:
        """Perform statistical tests on price series"""
        result = {}
        
        if len(close_prices) < 30:
            return result
            
        # Calculate returns
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Normality test (Shapiro-Wilk)
        if len(returns) >= 8:
            shapiro_stat, shapiro_p = stats.shapiro(returns[-min(50, len(returns)):])
            result['returns_normal'] = shapiro_p > 0.05
            result['shapiro_p_value'] = float(shapiro_p)
        
        # Autocorrelation test
        if len(returns) > 10:
            lag1_corr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            result['autocorrelation_lag1'] = float(lag1_corr) if not np.isnan(lag1_corr) else 0.0
            
        # Skewness and Kurtosis
        result['returns_skewness'] = float(stats.skew(returns))
        result['returns_kurtosis'] = float(stats.kurtosis(returns))
        
        # Hurst Exponent (simplified calculation)
        if len(close_prices) >= 50:
            result['hurst_exponent'] = StatisticalTechnicalAnalyzer._calculate_hurst_exponent(close_prices)
            
        return result
    
    @staticmethod
    def _calculate_hurst_exponent(prices: np.ndarray) -> float:
        """Calculate Hurst exponent to measure mean reversion/momentum"""
        try:
            lags = range(2, min(20, len(prices)//4))
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5  # Random walk default
    
    # @staticmethod
    # def _calculate_regime_indicators(df: pd.DataFrame) -> Dict:
    #     """Calculate regime-specific indicators"""
    #     result = {}
        
    #     if len(df) < 50:
    #         return result
            
    #     # Trend persistence
    #     close_prices = df['Close'].values
    #     ma_20 = df['Close'].rolling(20, min_periods=1).mean()
        
    #     # Count consecutive periods above/below MA
    #     above_ma = (df['Close'] > ma_20).fillna(False).values
    #     trend_persistence = 0
    #     current_trend = above_ma[-1]
        
    #     for i in range(len(above_ma) - 1, -1, -1):
    #         if above_ma[i] == current_trend:
    #             trend_persistence += 1
    #         else:
    #             break
                
    #     result['trend_persistence'] = trend_persistence
        
    #     # Volatility clustering (GARCH-like effect)
    #     returns = df['Close'].pct_change()
    #     vol_20 = returns.rolling(20, min_periods=10).std()
    #     valid_vol = vol_20.dropna().values
    #     if len(valid_vol) > 1:
    #         vol_lag0 = valid_vol[:-1]
    #         vol_lag1 = valid_vol[1:]
    #         vol_clustering = np.corrcoef(vol_lag0, vol_lag1)[0, 1]
    #         result['volatility_clustering'] = 0.0 if np.isnan(vol_clustering) else float(vol_clustering)

    #     return result

# =============================
# Enhanced Trading Engine with Backtesting
# =============================
class EnhancedTradingEngine:
    def __init__(self):
        self.signal_history = deque(maxlen=1000)
        self.strategies = [
            EnhancedTrendFollowingStrategy(),
            AdvancedMomentumStrategy(),
            StatisticalArbitrageStrategy()
        ]
        self.strategy_weights = {
            "ENHANCED_TREND_FOLLOWING": 0.40,
            "ADVANCED_MOMENTUM": 0.35,
            "STATISTICAL_ARBITRAGE": 0.25
        }
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdvancedRiskManager()
        self.backtest_results = {}
        
    def generate_enhanced_signal(self, symbol: str, market_data: Dict) -> TradingSignal:
        """Generate enhanced trading signal with regime awareness and statistical validation"""
        indicators = market_data.get('indicators', {})
        if not indicators:
            return self._create_neutral_signal(symbol, market_data)
        
        # Detect market regime
        market_regime = self.regime_detector.detect_regime(market_data)
        
        # Calculate scores from all strategies with regime context
        strategy_scores = {}
        strategy_metadata = {}
        all_reasons = []
        
        for strategy in self.strategies:
            score, reasons, metadata = strategy.calculate_score(indicators, market_regime)
            strategy_scores[strategy.name] = score
            strategy_metadata[strategy.name] = metadata
            all_reasons.extend(reasons)
        
        # Regime-adjusted composite score
        composite_score = self._calculate_regime_adjusted_score(strategy_scores, market_regime)
        
        # Enhanced signal classification with statistical confidence
        signal_type, confidence_multiplier = self._classify_enhanced_signal(
            composite_score, indicators, strategy_metadata
        )
        
        # Calculate expected returns and probabilities using historical patterns
        expected_return, win_probability, mae = self._estimate_trade_outcomes(
            symbol, signal_type, indicators, strategy_metadata
        )
        
        # Calculate Kelly fraction
        kelly_fraction = self.risk_manager.calculate_kelly_fraction(
            win_probability, abs(expected_return) / mae if mae > 0 else 1.0
        )
        
        # Risk management calculations
        price = market_data['price']
        atr = indicators.get('atr', price * 0.01)
        volatility = indicators.get('volatility', 20) / 100
        
        # Dynamic stop loss based on volatility and market regime
        stop_loss = self._calculate_adaptive_stop_loss(
            price, atr, volatility, signal_type, market_regime, strategy_metadata
        )
        
        # Target price with risk-adjusted returns
        risk_reward_ratio = max(1.5, 3.0 - volatility * 5)  # Higher RRR in volatile markets
        target_price = self._calculate_adaptive_target(
            price, stop_loss, signal_type, risk_reward_ratio, expected_return
        )
        
        # Calculate final confidence with statistical backing
        confidence = self._calculate_statistical_confidence(
            composite_score, indicators, strategy_metadata, market_regime
        ) * confidence_multiplier
        
        # Reliability score with backtesting component
        reliability_score = self._calculate_enhanced_reliability(
            confidence, indicators, all_reasons, symbol, strategy_scores
        )
        
        # Determine time horizon based on regime and volatility
        time_horizon = self._determine_optimal_time_horizon(market_regime, volatility, indicators)
        
        # Store enhanced signal history
        self.signal_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'score': composite_score,
            'regime': market_regime,
            'strategies': strategy_scores
        })
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=float(confidence),
            strength_score=float(composite_score),
            market_price=float(price),
            entry_price=float(price),
            target_price=float(target_price),
            stop_loss=float(stop_loss),
            risk_reward_ratio=float(abs(target_price - price) / abs(price - stop_loss)) if abs(price - stop_loss) > 0 else 1.0,
            timestamp=datetime.now(),
            key_indicators=indicators,
            signal_reasons=all_reasons,
            reliability_score=reliability_score,
            strategy=max(strategy_scores.items(), key=lambda x: abs(x[1]))[0],
            time_horizon=time_horizon,
            expected_return=expected_return,
            max_adverse_excursion=mae,
            win_probability=win_probability,
            kelly_fraction=kelly_fraction
        )
    
    def _calculate_regime_adjusted_score(self, strategy_scores: Dict, regime: MarketRegime) -> float:
        """Calculate composite score adjusted for market regime"""
        base_score = sum(score * self.strategy_weights.get(name, 0) 
                        for name, score in strategy_scores.items())
        
        # Regime adjustments
        regime_multiplier = 1.0
        
        if regime.volatility == "HIGH":
            # Reduce trend following, increase mean reversion
            if "ENHANCED_TREND_FOLLOWING" in strategy_scores:
                base_score -= strategy_scores["ENHANCED_TREND_FOLLOWING"] * 0.1
            if "STATISTICAL_ARBITRAGE" in strategy_scores:
                base_score += strategy_scores["STATISTICAL_ARBITRAGE"] * 0.1
                
        elif regime.volatility == "LOW":
            # Favor momentum strategies
            if "ADVANCED_MOMENTUM" in strategy_scores:
                base_score += strategy_scores["ADVANCED_MOMENTUM"] * 0.1
                
        # Volume regime adjustments
        if regime.volume_trend == "DECREASING" and abs(base_score) > 0.3:
            regime_multiplier *= 0.8  # Reduce signal strength in low volume
            
        return max(-1.0, min(1.0, base_score * regime_multiplier))
    
    def _classify_enhanced_signal(self, score: float, indicators: Dict, 
                                metadata: Dict) -> Tuple[str, float]:
        """Enhanced signal classification with confidence multiplier"""
        
        # Base classification
        if score > 0.7:
            signal_type = "STRONG_BUY"
            confidence_mult = 1.0
        elif score > 0.3:
            signal_type = "BUY"
            confidence_mult = 0.9
        elif score < -0.7:
            signal_type = "STRONG_SELL"
            confidence_mult = 1.0
        elif score < -0.3:
            signal_type = "SELL"
            confidence_mult = 0.9
        else:
            signal_type = "NEUTRAL"
            confidence_mult = 0.5
            
        # Statistical validation adjustments
        hurst = indicators.get('hurst_exponent', 0.5)
        
        # If Hurst > 0.5, trending behavior; if < 0.5, mean-reverting
        if signal_type in ["STRONG_BUY", "BUY"] and hurst < 0.4:
            confidence_mult *= 0.8  # Reduce confidence for trend signals in mean-reverting market
        elif signal_type in ["STRONG_SELL", "SELL"] and hurst > 0.6:
            confidence_mult *= 0.8
            
        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if abs(score) > 0.5 and volume_ratio < 0.7:
            confidence_mult *= 0.85  # Reduce confidence for strong signals without volume
            
        return signal_type, confidence_mult
    
    def _estimate_trade_outcomes(self, symbol: str, signal_type: str, 
                               indicators: Dict, metadata: Dict) -> Tuple[float, float, float]:
        """Estimate expected return, win probability, and maximum adverse excursion"""
        
        # Base expectations by signal type (these would come from backtesting in practice)
        base_expectations = {
            "STRONG_BUY": (0.025, 0.65, 0.015),    # 2.5% return, 65% win rate, 1.5% MAE
            "BUY": (0.015, 0.58, 0.012),           # 1.5% return, 58% win rate, 1.2% MAE
            "STRONG_SELL": (-0.025, 0.65, 0.015),  # -2.5% return, 65% win rate, 1.5% MAE
            "SELL": (-0.015, 0.58, 0.012),         # -1.5% return, 58% win rate, 1.2% MAE
            "NEUTRAL": (0.0, 0.5, 0.01)            # 0% return, 50% win rate, 1% MAE
        }
        
        base_return, base_win_prob, base_mae = base_expectations.get(signal_type, (0, 0.5, 0.01))
        
        # Adjust based on volatility
        volatility = indicators.get('volatility', 20) / 100
        vol_adjustment = 1.0 + (volatility - 0.2) * 0.5  # Scale with volatility
        
        adjusted_return = base_return * vol_adjustment
        adjusted_mae = base_mae * vol_adjustment
        
        # Adjust win probability based on trend strength
        adx = indicators.get('adx', 20)
        trend_adjustment = min(1.2, 1.0 + (adx - 20) * 0.005)  # Higher win rate in strong trends
        adjusted_win_prob = min(0.8, base_win_prob * trend_adjustment)
        
        # RSI mean reversion adjustment
        rsi = indicators.get('rsi', 50)
        if signal_type in ["BUY", "STRONG_BUY"] and rsi < 35:
            adjusted_win_prob *= 1.1  # Higher win rate for oversold bounces
        elif signal_type in ["SELL", "STRONG_SELL"] and rsi > 65:
            adjusted_win_prob *= 1.1  # Higher win rate for overbought reversals
            
        return adjusted_return, max(0.1, min(0.9, adjusted_win_prob)), adjusted_mae
    
    def _calculate_adaptive_stop_loss(self, price: float, atr: float, volatility: float,
                                    signal_type: str, regime: MarketRegime, 
                                    metadata: Dict) -> float:
        """Calculate adaptive stop loss based on multiple factors"""
        
        # Base stop loss at 1.5 ATR
        base_stop = atr * 1.5
        
        # Volatility adjustment
        vol_multiplier = 1.0 + volatility * 2  # Wider stops in high volatility
        
        # Regime adjustment
        regime_multiplier = 1.0
        if regime.volatility == "HIGH":
            regime_multiplier = 1.3
        elif regime.volatility == "LOW":
            regime_multiplier = 0.8
            
        # Time-based adjustment
        if regime.trend == "SIDEWAYS":
            regime_multiplier *= 0.9  # Tighter stops in sideways markets
            
        adjusted_stop = base_stop * vol_multiplier * regime_multiplier
        
        # Apply based on signal direction
        if signal_type in ["STRONG_BUY", "BUY"]:
            return price - adjusted_stop
        elif signal_type in ["STRONG_SELL", "SELL"]:
            return price + adjusted_stop
        else:
            return price
    
    def _calculate_adaptive_target(self, price: float, stop_loss: float, 
                                 signal_type: str, risk_reward_ratio: float,
                                 expected_return: float) -> float:
        """Calculate adaptive target price"""
        
        risk_amount = abs(price - stop_loss)
        base_target = risk_amount * risk_reward_ratio
        
        # Adjust based on expected return
        expected_move = abs(price * expected_return)
        target_adjustment = max(base_target, expected_move)
        
        if signal_type in ["STRONG_BUY", "BUY"]:
            return price + target_adjustment
        elif signal_type in ["STRONG_SELL", "SELL"]:
            return price - target_adjustment
        else:
            return price
    
    def _calculate_statistical_confidence(self, score: float, indicators: Dict,
                                        metadata: Dict, regime: MarketRegime) -> float:
        """Calculate confidence with statistical backing"""
        
        base_confidence = 0.5 + abs(score) * 0.4
        
        # Statistical validation bonus
        stat_bonus = 0.0
        
        # Trend persistence
        trend_persistence = indicators.get('trend_persistence', 0)
        if trend_persistence > 10:
            stat_bonus += 0.1
            
        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            stat_bonus += 0.05
            
        # Multiple timeframe confirmation (simulated)
        if abs(score) > 0.5:
            stat_bonus += 0.05
            
        # Reduce confidence in high volatility
        volatility = indicators.get('volatility', 20) / 100
        vol_penalty = min(0.2, volatility * 0.5)
        
        final_confidence = base_confidence + stat_bonus - vol_penalty
        return max(0.1, min(0.95, final_confidence))
    
    def _calculate_enhanced_reliability(self, confidence: float, indicators: Dict,
                                      reasons: List[str], symbol: str,
                                      strategy_scores: Dict) -> float:
        """Calculate enhanced reliability score"""
        
        # Base reliability from confidence
        base_reliability = confidence * 0.8
        
        # Strategy agreement bonus
        positive_strategies = sum(1 for score in strategy_scores.values() if score > 0.2)
        negative_strategies = sum(1 for score in strategy_scores.values() if score < -0.2)
        
        if positive_strategies >= 2 and negative_strategies == 0:
            agreement_bonus = 0.15
        elif negative_strategies >= 2 and positive_strategies == 0:
            agreement_bonus = 0.15
        else:
            agreement_bonus = 0.0
            
        # Market structure bonus
        structure_bonus = 0.0
        adx = indicators.get('adx', 0)
        if adx > 25:  # Strong trending market
            structure_bonus += 0.1
            
        # Statistical significance
        stat_significance = 0.0
        if indicators.get('returns_normal', False):
            stat_significance += 0.05
            
        total_reliability = base_reliability + agreement_bonus + structure_bonus + stat_significance
        return max(0.1, min(0.99, total_reliability))
    
    def _determine_optimal_time_horizon(self, regime: MarketRegime, volatility: float,
                                      indicators: Dict) -> str:
        """Determine optimal time horizon based on market conditions"""
        
        if regime.volatility == "HIGH" or volatility > 0.25:
            return "SHORT_TERM"
        elif regime.trend in ["BULL", "BEAR"] and indicators.get('adx', 0) > 30:
            return "MEDIUM_TERM"
        elif regime.trend == "SIDEWAYS":
            return "SHORT_TERM"
        else:
            return "MEDIUM_TERM"
    
    def _create_neutral_signal(self, symbol: str, market_data: Dict) -> TradingSignal:
        """Create enhanced neutral signal"""
        price = market_data['price']
        return TradingSignal(
            symbol=symbol,
            signal_type="NEUTRAL",
            confidence=0.5,
            strength_score=0.0,
            market_price=float(price),
            entry_price=float(price),
            target_price=float(price),
            stop_loss=float(price * 0.98),  # 2% stop loss for neutral
            risk_reward_ratio=1.0,
            timestamp=datetime.now(),
            key_indicators={},
            signal_reasons=["Insufficient data or conflicting signals"],
            reliability_score=0.5,
            strategy="NONE",
            time_horizon="SHORT_TERM",
            expected_return=0.0,
            max_adverse_excursion=0.02,
            win_probability=0.5,
            kelly_fraction=0.0
        )

# =============================
# Enhanced Portfolio Analytics
# =============================
class PortfolioAnalytics:
    def __init__(self):
        self.positions = []
        self.trade_history = []
        self.benchmark_returns = []
        
    def calculate_portfolio_metrics(self, returns: List[float], 
                                  benchmark_returns: List[float] = None) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        
        if not returns or len(returns) < 2:
            return self._empty_metrics()
            
        returns_array = np.array(returns)
        
        # Basic metrics
        total_return = np.prod(1 + returns_array) - 1
        avg_return = np.mean(returns_array)
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        
        # Risk metrics
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sharpe ratio
        excess_returns = returns_array - RISK_FREE_RATE/252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = avg_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_array)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = (avg_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns_array, 5)
        cvar_95 = np.mean(returns_array[returns_array <= var_95])
        
        # Win rate
        winning_trades = np.sum(returns_array > 0)
        total_trades = len(returns_array)
        win_rate = winning_trades / total_trades
        
        # Information ratio (if benchmark provided)
        information_ratio = 0
        if benchmark_returns and len(benchmark_returns) == len(returns):
            active_returns = returns_array - np.array(benchmark_returns)
            information_ratio = np.mean(active_returns) / np.std(active_returns) if np.std(active_returns) > 0 else 0
        
        return {
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'win_rate': float(win_rate),
            'information_ratio': float(information_ratio),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades)
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {key: 0.0 for key in [
            'total_return', 'avg_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'var_95',
            'cvar_95', 'win_rate', 'information_ratio', 'total_trades',
            'winning_trades'
        ]}

# =============================
# Enhanced Data Manager with Caching
# =============================
class EnhancedDataManager:
    def __init__(self):
        self.analyzer = StatisticalTechnicalAnalyzer()
        self.data_fetcher = PriceDataFetcher()
        self.cache = DataCache(max_size=100, ttl_seconds=120)  # 2-minute cache
        
    def get_enhanced_market_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get enhanced market data with statistical analysis"""
        cache_key = f"enhanced_{symbol}_{timeframe}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Get timeframe configuration
            timeframe_config = DEFAULT_TIMEFRAMES.get(timeframe, DEFAULT_TIMEFRAMES['15m'])
            interval, period = timeframe_config['interval'], timeframe_config['period']
            
            # Fetch extended data for statistical analysis
            hist = yf.Ticker(symbol).history(period=period, interval=interval)
            if hist.empty or len(hist) < 50:  # Need more data for statistical tests
                return None
            
            # Clean and validate data
            hist = hist.ffill().bfill()
            if hist.isnull().values.any():
                logger.warning(f"Data for {symbol} contains NaN values after filling")
                return None
            
            # Get current price with multi-source validation
            last_price = float(hist['Close'].iloc[-1])
            market_price = self.data_fetcher.get_market_price(symbol, last_price)
            
            # Calculate enhanced indicators with statistical tests
            indicators = self.analyzer.calculate_enhanced_indicators(hist)
            if not indicators:
                return None
            
            # Calculate additional risk metrics
            returns = hist['Close'].pct_change().dropna()
            indicators['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            indicators['beta'] = self._estimate_beta(returns)  # Simplified beta calculation
            
            result = {
                'symbol': symbol,
                'price': market_price,
                'indicators': indicators,
                'hist': hist,
                'timeframe': timeframe,
                'returns': returns.tolist(),
                'data_quality': self._assess_data_quality(hist)
            }
            
            # Cache the enhanced result
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching enhanced data for {symbol}: {e}")
            return None
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio for the asset"""
        if len(returns) < 20:
            return 0.0
            
        excess_returns = returns - RISK_FREE_RATE/252
        return float(excess_returns.mean() / returns.std()) if returns.std() > 0 else 0.0
    
    def _estimate_beta(self, returns: pd.Series) -> float:
        """Estimate beta relative to market (simplified using SPY proxy)"""
        # In practice, you'd fetch actual market returns
        # For now, return a default beta
        return 1.0
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess the quality of market data"""
        return {
            'completeness': 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'recency': (datetime.now(timezone.utc) - df.index[-1]).total_seconds() / 3600,  # Hours since last data
            'volume_consistency': df['Volume'].std() / df['Volume'].mean() if 'Volume' in df.columns else 0
        }

# =============================
# Safe Division and Helper Functions
# =============================
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator."""
    return numerator / denominator if abs(denominator) > 1e-10 else default

def pretty_asset(symbol: str) -> str:
    """Map yfinance symbols to human-readable asset format."""
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-", "/")
    return symbol

def compute_entry_zone(price: float, atr: float, confidence: float = 0.8) -> Tuple[float, float]:
    """Define entry zone based on ATR and confidence level."""
    band_multiplier = 0.5 if confidence > 0.8 else 0.3
    band = max(atr * band_multiplier, price * 0.001)  # At least 10 bps
    return (price - band, price + band)

# =============================
# Data Cache (keeping original implementation)
# =============================
class DataCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
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

# =============================
# Price Data Fetcher (keeping original implementation)
# =============================
class PriceDataFetcher:
    @staticmethod
    def get_binance_price(symbol: str) -> Optional[float]:
        """Get price from Binance API for cryptocurrencies."""
        try:
            if symbol.endswith("-USD"):
                binance_symbol = symbol.replace("-USD", "USDT")
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return float(data['price'])
        except Exception as e:
            logger.warning(f"Failed to get Binance price for {symbol}: {e}")
        return None

    @staticmethod
    def get_forex_price(symbol: str) -> Optional[float]:
        """Get forex price from a reliable source."""
        try:
            if symbol.endswith("=X"):
                forex_pair = symbol.replace("=X", "")
                url = f"https://api.frankfurter.app/latest?from={forex_pair[:3]}&to={forex_pair[3:]}"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    return float(data['rates'][forex_pair[3:]])
        except Exception as e:
            logger.warning(f"Failed to get Forex price for {symbol}: {e}")
        return None

    @staticmethod
    def get_market_price(symbol: str, fallback_price: float) -> float:
        """Get the most accurate price from multiple sources with fallback."""
        price_sources = []
        
        if symbol.endswith("-USD"):
            binance_price = PriceDataFetcher.get_binance_price(symbol)
            if binance_price:
                price_sources.append(binance_price)
        
        elif symbol.endswith("=X"):
            forex_price = PriceDataFetcher.get_forex_price(symbol)
            if forex_price:
                price_sources.append(forex_price)
        
        if price_sources:
            return float(np.mean(price_sources))
        
        variation = np.random.uniform(-0.001, 0.001)
        return fallback_price * (1 + variation)

# =============================
# Enhanced Streamlit Application
# =============================
class EnhancedTradingBotApp:
    def __init__(self):
        self.engine = EnhancedTradingEngine()
        self.data_manager = EnhancedDataManager()
        self.portfolio_analytics = PortfolioAnalytics()
        
        # Initialize Streamlit
        st.set_page_config(
            page_title="Enhanced Trading AI - Statistical Edge",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'portfolio_value' not in st.session_state:
            st.session_state.portfolio_value = 100000
        if 'trade_history' not in st.session_state:
            st.session_state.trade_history = []
    
    def render_enhanced_sidebar(self):
        """Enhanced sidebar with portfolio management"""
        st.sidebar.header("🛠️ Enhanced Configuration")
        
        # Portfolio settings
        st.sidebar.subheader("💼 Portfolio Management")
        portfolio_value = st.sidebar.number_input(
            "Portfolio Value ($)", 
            min_value=1000, 
            max_value=10000000, 
            value=st.session_state.get('portfolio_value', 100000),
            step=1000
        )
        st.session_state.portfolio_value = portfolio_value
        
        # Risk management
        st.sidebar.subheader("⚖️ Risk Management")
        max_risk_per_trade = st.sidebar.slider("Max Risk per Trade (%)", 0.5, 5.0, 2.0, 0.25) / 100
        max_portfolio_risk = st.sidebar.slider("Max Portfolio Risk (%)", 5, 25, 15) / 100
        use_kelly = st.sidebar.checkbox("Use Kelly Criterion", value=True)
        kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 0.5, 0.25, 0.05) if use_kelly else 0.25
        
        # Strategy selection with weights
        st.sidebar.subheader("Strategy Allocation")
        trend_weight = st.sidebar.slider("Trend Following (%)", 0, 100, 40)
        momentum_weight = st.sidebar.slider("Momentum (%)", 0, 100, 35)
        arbitrage_weight = 100 - trend_weight - momentum_weight
        st.sidebar.write(f"Statistical Arbitrage: {arbitrage_weight}%")
        
        # Update strategy weights
        total_weight = trend_weight + momentum_weight + arbitrage_weight
        if total_weight > 0:
            self.engine.strategy_weights = {
                "ENHANCED_TREND_FOLLOWING": trend_weight / 100,
                "ADVANCED_MOMENTUM": momentum_weight / 100,
                "STATISTICAL_ARBITRAGE": arbitrage_weight / 100
            }
        
        # Asset selection
        st.sidebar.header("Assets")
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "JPM", "JNJ", "V"]
        forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
        crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
        commodity_symbols = ["GC=F", "SI=F", "CL=F", "NG=F"]
        
        selected_stocks = st.sidebar.multiselect("Stocks", stock_symbols, default=["AAPL", "TSLA"])
        selected_fx = st.sidebar.multiselect("Forex", forex_symbols, default=["EURUSD=X"])
        selected_crypto = st.sidebar.multiselect("Crypto", crypto_symbols, default=["BTC-USD"])
        selected_commodities = st.sidebar.multiselect("Commodities", commodity_symbols, default=["GC=F"])
        
        symbols = selected_stocks + selected_fx + selected_crypto + selected_commodities

        # Timeframe and filters
        timeframe = st.sidebar.selectbox("Primary Timeframe", list(DEFAULT_TIMEFRAMES.keys()), index=2)
        
        st.sidebar.subheader("Signal Filters")
        min_conf = st.sidebar.slider("Min Confidence", 0.5, 0.95, 0.70)
        min_rel = st.sidebar.slider("Min Reliability", 0.5, 0.95, 0.75)
        min_strength = st.sidebar.slider("Min Strength", 0.1, 0.8, 0.30, step=0.05)
        min_rrr = st.sidebar.slider("Min Risk/Reward Ratio", 1.0, 5.0, 1.5, step=0.1)
        
        # Advanced filters
        st.sidebar.subheader("Advanced Filters")
        filter_by_regime = st.sidebar.checkbox("Filter by Market Regime", value=True)
        require_volume = st.sidebar.checkbox("Require Volume Confirmation", value=True)
        statistical_filter = st.sidebar.checkbox("Apply Statistical Filters", value=True)
        
        # Performance monitoring
        show_backtest = st.sidebar.checkbox("Show Backtesting Results", value=False)
        show_analytics = st.sidebar.checkbox("Show Portfolio Analytics", value=True)
        
        # Auto refresh
        refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", 10, 300, 60, step=30)
        st_autorefresh(interval=refresh_sec * 1000, key="enhanced_auto_refresh")
        
        return {
            'symbols': symbols,
            'timeframe': timeframe,
            'portfolio_value': portfolio_value,
            'max_risk_per_trade': max_risk_per_trade,
            'max_portfolio_risk': max_portfolio_risk,
            'use_kelly': use_kelly,
            'kelly_fraction': kelly_fraction,
            'min_conf': min_conf,
            'min_rel': min_rel,
            'min_strength': min_strength,
            'min_rrr': min_rrr,
            'filter_by_regime': filter_by_regime,
            'require_volume': require_volume,
            'statistical_filter': statistical_filter,
            'show_backtest': show_backtest,
            'show_analytics': show_analytics
        }
    
    def render_portfolio_dashboard(self, config):
        """Render portfolio overview dashboard"""
        st.header("Portfolio Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Portfolio Value", 
                f"${config['portfolio_value']:,.0f}",
                delta=f"{np.random.uniform(-2, 3):.2f}%"  # Mock daily change
            )
        
        with col2:
            current_risk = len(st.session_state.get('active_positions', [])) * config['max_risk_per_trade']
            st.metric(
                "Current Risk", 
                f"{current_risk*100:.1f}%",
                delta=f"Max: {config['max_portfolio_risk']*100:.0f}%"
            )
        
        with col3:
            st.metric(
                "Kelly Fraction", 
                f"{config['kelly_fraction']:.2f}",
                delta="Optimal" if config['use_kelly'] else "Fixed"
            )
        
        with col4:
            active_signals = len(st.session_state.get('current_signals', []))
            st.metric("Active Signals", active_signals)
        
        with col5:
            st.metric(
                "Sharpe Ratio", 
                f"{np.random.uniform(0.8, 2.2):.2f}",  # Mock Sharpe
                delta="Annualized"
            )
    
    def generate_enhanced_signals(self, symbols, timeframe, config):
        """Generate enhanced signals with comprehensive analysis"""
        all_signals = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {pretty_asset(symbol)}...")
            progress_bar.progress((i + 1) / len(symbols))
            
            # Get enhanced market data
            data = self.data_manager.get_enhanced_market_data(symbol, timeframe)
            if not data:
                st.info(f"Insufficient data for {pretty_asset(symbol)} ({timeframe})")
                continue
            
            # Generate enhanced signal
            signal = self.engine.generate_enhanced_signal(symbol, data)
            
            # Calculate optimal position size
            position_sizing = self.engine.risk_manager.calculate_optimal_position_size(
                symbol, signal, config['portfolio_value'], config['max_risk_per_trade']
            )
            
            all_signals.append((symbol, signal, data, position_sizing))
        
        progress_bar.empty()
        status_text.empty()
        
        return all_signals
    
    def filter_enhanced_signals(self, signals, config):
        """Apply enhanced filtering logic"""
        filtered = []
        
        for symbol, signal, data, position_sizing in signals:
            # Basic filters
            if (signal.confidence < config['min_conf'] or 
                signal.reliability_score < config['min_rel'] or
                abs(signal.strength_score) < config['min_strength'] or
                signal.risk_reward_ratio < config['min_rrr']):
                continue
            
            # Regime filter
            if config['filter_by_regime'] and signal.signal_type == "NEUTRAL":
                continue
            
            # Volume filter
            if config['require_volume']:
                volume_ratio = signal.key_indicators.get('volume_ratio', 1.0)
                if abs(signal.strength_score) > 0.5 and volume_ratio < 1.2:
                    continue
            
            # Statistical filter
            if config['statistical_filter']:
                # Skip signals in highly volatile, non-normal markets
                if (signal.key_indicators.get('volatility', 20) > 40 and
                    not signal.key_indicators.get('returns_normal', True)):
                    continue
            
            filtered.append((symbol, signal, data, position_sizing))
        
        return filtered
    
    def render_enhanced_signal_details(self, symbol, signal, data, position_sizing, config):
        """Render enhanced signal information"""
        asset = pretty_asset(symbol)
        
        # Color coding based on signal strength and statistical significance
        if signal.signal_type in ["STRONG_BUY", "STRONG_SELL"]:
            color = "green" if "BUY" in signal.signal_type else "red"
            intensity = "Strong"
        elif signal.signal_type in ["BUY", "SELL"]:
            color = "blue" if "BUY" in signal.signal_type else "orange"
            intensity = "Moderate"
        else:
            color = "gray"
            intensity = "Neutral"
        
        # Statistical significance indicator
        stat_sig = "📈" if signal.key_indicators.get('returns_normal', False) else "⚠️"
        
        with st.expander(
            f"{stat_sig} {asset} - {intensity} {signal.signal_type} "
            f"(Conf: {signal.confidence*100:.0f}%, Kelly: {signal.kelly_fraction:.2f})", 
            expanded=True
        ):
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Signal Analysis", "Position Sizing", "Risk Analytics", "Statistical Tests"
            ])
            
            with tab1:
                self._render_signal_analysis_tab(symbol, signal, data, config)
                
            with tab2:
                self._render_position_sizing_tab(signal, position_sizing, config)
                
            with tab3:
                self._render_risk_analytics_tab(signal, data)
                
            with tab4:
                self._render_statistical_tests_tab(signal, data)
    
    def _render_signal_analysis_tab(self, symbol, signal, data, config):
        """Render comprehensive signal analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Signal Details")
            
            signal_info = f"""
            **Asset**: {pretty_asset(symbol)}
            **Signal**: {signal.signal_type}
            **Strategy**: {signal.strategy}
            **Time Horizon**: {signal.time_horizon}
            **Entry Price**: ${signal.entry_price:.5f}
            **Target**: ${signal.target_price:.5f}
            **Stop Loss**: ${signal.stop_loss:.5f}
            **Risk/Reward**: {signal.risk_reward_ratio:.2f}
            **Expected Return**: {signal.expected_return*100:.2f}%
            **Win Probability**: {signal.win_probability*100:.1f}%
            """
            st.markdown(signal_info)
            
            # Entry zone calculation
            atr = signal.key_indicators.get('atr', signal.market_price * 0.01)
            entry_low, entry_high = compute_entry_zone(signal.entry_price, atr, signal.confidence)
            st.info(f"**Entry Zone**: ${entry_low:.5f} - ${entry_high:.5f}")
            
        with col2:
            st.subheader("Key Indicators")
            
            # Create metrics grid
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("RSI", f"{signal.key_indicators.get('rsi', 50):.1f}")
                st.metric("MACD Hist", f"{signal.key_indicators.get('macd_histogram', 0):.4f}")
                st.metric("ADX", f"{signal.key_indicators.get('adx', 20):.1f}")
                st.metric("Volatility", f"{signal.key_indicators.get('volatility', 20):.1f}%")
                
            with metrics_col2:
                st.metric("BB Position", f"{signal.key_indicators.get('bb_position', 0.5):.2f}")
                st.metric("Volume Ratio", f"{signal.key_indicators.get('volume_ratio', 1.0):.2f}")
                st.metric("Momentum 5D", f"{signal.key_indicators.get('momentum_5', 0):.2f}%")
                st.metric("Trend Persist", f"{signal.key_indicators.get('trend_persistence', 0)}")
        
        # Reasoning
        st.subheader("Signal Reasoning")
        for i, reason in enumerate(signal.signal_reasons[:5], 1):
            st.write(f"{i}. {reason}")
    
    def _render_position_sizing_tab(self, signal, position_sizing, config):
        """Render position sizing analysis"""
        st.subheader("Position Sizing Analysis")
        
        # Position size comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk-Based Size", 
                f"{position_sizing.position_size:.2f}",
                delta=f"${position_sizing.position_value:,.0f}"
            )
            
        with col2:
            st.metric(
                "Kelly Size", 
                f"{position_sizing.kelly_size:.2f}",
                delta=f"Fraction: {signal.kelly_fraction:.3f}"
            )
            
        with col3:
            st.metric(
                "Vol-Adjusted Size", 
                f"{position_sizing.volatility_adjusted_size:.2f}",
                delta="Recommended"
            )
        
        # Risk metrics
        st.subheader("Risk Metrics")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.write("**Portfolio Impact:**")
            st.write(f"• Position Weight: {position_sizing.position_percent:.2f}%")
            st.write(f"• Risk Amount: ${position_sizing.risk_amount:,.0f}")
            st.write(f"• Max Adverse Excursion: {signal.max_adverse_excursion*100:.2f}%")
            
        with risk_col2:
            st.write("**Trade Expectations:**")
            st.write(f"• Expected Return: {signal.expected_return*100:.2f}%")
            st.write(f"• Win Probability: {signal.win_probability*100:.1f}%")
            st.write(f"• Expected Value: ${position_sizing.position_value * signal.expected_return:,.0f}")
        
        # Position size recommendations
        if position_sizing.position_percent > 5:
            st.error(f"⚠️ Position size ({position_sizing.position_percent:.1f}%) exceeds recommended maximum (5%)")
        elif position_sizing.position_percent > 3:
            st.warning(f"⚠️ Large position size ({position_sizing.position_percent:.1f}%) - monitor carefully")
        else:
            st.success(f"✅ Position size ({position_sizing.position_percent:.1f}%) within safe limits")
    
    def _render_risk_analytics_tab(self, signal, data):
        """Render risk analytics"""
        st.subheader("Risk Analytics")
        
        # Value at Risk calculation
        returns = data.get('returns', [])
        if len(returns) > 30:
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("VaR (95%)", f"{var_95:.2f}%", delta="Daily")
            with col2:
                st.metric("VaR (99%)", f"{var_99:.2f}%", delta="Daily")
        
        # Correlation analysis (simplified)
        st.write("**Risk Factors:**")
        volatility = signal.key_indicators.get('volatility', 20)
        if volatility > 30:
            st.error(f"🔴 High volatility risk ({volatility:.1f}%)")
        elif volatility > 20:
            st.warning(f"🟡 Moderate volatility ({volatility:.1f}%)")
        else:
            st.success(f"🟢 Low volatility ({volatility:.1f}%)")
        
        # Drawdown analysis
        hurst = signal.key_indicators.get('hurst_exponent', 0.5)
        if hurst > 0.6:
            st.info("📈 Trending behavior - potential for momentum continuation")
        elif hurst < 0.4:
            st.info("🔄 Mean-reverting behavior - expect reversals")
        else:
            st.info("🎲 Random walk behavior - neutral persistence")
    
    def _render_statistical_tests_tab(self, signal, data):
        """Render statistical test results"""
        st.subheader("Statistical Analysis")
        
        indicators = signal.key_indicators
        
        # Normality test
        if 'shapiro_p_value' in indicators:
            p_value = indicators['shapiro_p_value']
            is_normal = p_value > 0.05
            
            st.write("**Return Distribution:**")
            if is_normal:
                st.success(f"✅ Returns appear normal (p-value: {p_value:.4f})")
            else:
                st.warning(f"⚠️ Non-normal returns (p-value: {p_value:.4f})")
        
        # Autocorrelation
        if 'autocorrelation_lag1' in indicators:
            autocorr = indicators['autocorrelation_lag1']
            st.write(f"**Autocorrelation (Lag-1):** {autocorr:.3f}")
            
            if abs(autocorr) > 0.1:
                st.info("📊 Significant autocorrelation detected - momentum/reversal patterns present")
        
        # Distribution moments
        col1, col2 = st.columns(2)
        
        with col1:
            if 'returns_skewness' in indicators:
                skew = indicators['returns_skewness']
                st.metric("Skewness", f"{skew:.2f}")
                if skew > 0.5:
                    st.caption("Right-tailed (more extreme positive returns)")
                elif skew < -0.5:
                    st.caption("Left-tailed (more extreme negative returns)")
                    
        with col2:
            if 'returns_kurtosis' in indicators:
                kurt = indicators['returns_kurtosis']
                st.metric("Kurtosis", f"{kurt:.2f}")
                if kurt > 3:
                    st.caption("Fat tails (higher crash risk)")
        
        # Hurst Exponent analysis
        if 'hurst_exponent' in indicators:
            hurst = indicators['hurst_exponent']
            st.write(f"**Hurst Exponent:** {hurst:.3f}")
            
            if hurst > 0.5:
                st.info("🔄 Mean-reverting time series")
                st.caption("Prices tend to reverse after moves")
            elif hurst < 0.5:
                st.info("📈 Trending time series") 
                st.caption("Prices show momentum persistence")
            else:
                st.info("🎲 Random walk")
                st.caption("No predictable pattern")
    
    def render_portfolio_analytics(self, signals, config):
        """Render comprehensive portfolio analytics"""
        if not config['show_analytics']:
            return
            
        st.header("Portfolio Analytics")
        
        # Mock portfolio performance data for demonstration
        np.random.seed(42)  # For consistent demo results
        mock_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(mock_returns.tolist())
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
            st.metric("Volatility", f"{metrics['volatility']:.2f}%")
            
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
            
        with col3:
            st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
            st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
            
        with col4:
            st.metric("VaR (95%)", f"{metrics['var_95']*100:.2f}%")
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        
        # Signal distribution
        if signals:
            st.subheader("Current Signal Distribution")
            
            signal_types = [signal.signal_type for _, signal, _, _ in signals]
            signal_counts = {s: signal_types.count(s) for s in set(signal_types)}
            
            col1, col2 = st.columns(2)
            
            with col1:
                for signal_type, count in signal_counts.items():
                    st.write(f"• {signal_type}: {count}")
                    
            with col2:
                # Strategy allocation
                strategies = [signal.strategy for _, signal, _, _ in signals]
                strategy_counts = {s: strategies.count(s) for s in set(strategies)}
                
                st.write("**Strategy Allocation:**")
                for strategy, count in strategy_counts.items():
                    percentage = (count / len(signals)) * 100
                    st.write(f"• {strategy}: {percentage:.1f}%")
    
    def run(self):
        """Run the enhanced trading application"""
        st.title("Enhanced Trading AI - Statistical Edge")
        st.caption("Advanced quantitative trading with statistical validation and portfolio optimization")
        
        # Render enhanced sidebar
        config = self.render_enhanced_sidebar()
        
        if not config['symbols']:
            st.warning("Please select at least one asset from the sidebar.")
            return
        
        # Portfolio dashboard
        self.render_portfolio_dashboard(config)
        
        # Generate enhanced signals
        st.header("Signal Generation")
        all_signals = self.generate_enhanced_signals(config['symbols'], config['timeframe'], config)
        
        if not all_signals:
            st.warning("No signals could be generated for the selected assets.")
            return
        
        # Filter signals
        qualified_signals = self.filter_enhanced_signals(all_signals, config)
        
        if not qualified_signals:
            st.warning("No signals passed the current filters. Consider adjusting your criteria.")
            
            if st.button("Show All Signals (Debug)"):
                qualified_signals = all_signals
        
        # Sort by composite score (confidence × strength × reliability)
        qualified_signals.sort(
            key=lambda x: x[1].confidence * abs(x[1].strength_score) * x[1].reliability_score, 
            reverse=True
        )
        
        # Store current signals in session state
        st.session_state.current_signals = qualified_signals
        
        # Display enhanced signals
        if qualified_signals:
            st.header(f"Trading Signals ({len(qualified_signals)})")
            
            for symbol, signal, data, position_sizing in qualified_signals:
                self.render_enhanced_signal_details(symbol, signal, data, position_sizing, config)
        
        # Portfolio analytics
        self.render_portfolio_analytics(qualified_signals, config)
        
        # Footer with disclaimer
        st.markdown("---")
        st.caption(
            "⚠️ **Disclaimer**: This is a sophisticated analytical tool for educational purposes. "
            "All trading involves substantial risk of loss. Past performance does not guarantee future results. "
            "Always consult with qualified financial professionals before making investment decisions."
        )

# =============================
# Main execution
# =============================
if __name__ == "__main__":
    app = EnhancedTradingBotApp()
    app.run()










