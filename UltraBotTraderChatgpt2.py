
import os
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import talib
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

# ==========================================================
# Logging configuration
# ==========================================================
LOG_DIR = os.environ.get("ULT_BOT_LOG_DIR", ".")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("UltimateTradingAI")
logger.setLevel(logging.INFO)

# File handler with rotation
fh = RotatingFileHandler(os.path.join(LOG_DIR, "trading_bot.log"), maxBytes=1_000_000, backupCount=3)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Avoid duplicate handlers if re-running in Streamlit
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

# ==========================================================
# Constants & Defaults
# ==========================================================
DEFAULT_TIMEFRAMES = {
    '1m': {'interval': '1m',  'period': '1d'},
    '5m': {'interval': '5m',  'period': '5d'},
    '15m': {'interval': '15m','period': '15d'},
    '1h': {'interval': '1h',  'period': '60d'},
    '4h': {'interval': '1h',  'period': '60d'},  # yfinance has limited intraday history
    '1d': {'interval': '1d',  'period': '2y'}
}

EPS = 1e-8

# ==========================================================
# Utility functions
# ==========================================================
def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    """Safe division that guards against zero/None/NaN denominator."""
    try:
        if denom is None:
            return default
        if isinstance(denom, (int, float)) and (denom == 0 or np.isnan(denom)):
            return default
        return numer / denom
    except Exception:
        return default

def pretty_asset(symbol: str) -> str:
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-", "/")
    return symbol

def compute_entry_zone(price: float, atr: float) -> Tuple[float, float]:
    band = max(atr * 0.15, price * 0.0005)  # at least 5 bps
    return (price - band, price + band)

# ==========================================================
# Data models
# ==========================================================
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
    time_horizon: str  # SHORT_TERM / MEDIUM_TERM / LONG_TERM

@dataclass
class PositionSizing:
    risk_amount: float
    position_size: float
    position_value: float
    position_percent: float

# ==========================================================
# Caching
# ==========================================================
class DataCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[float, any]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[any]:
        now = time.time()
        item = self.cache.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts < self.ttl:
            return value
        # expired
        self.cache.pop(key, None)
        return None

    def set(self, key: str, value: any) -> None:
        if len(self.cache) >= self.max_size:
            # remove oldest
            oldest_key = min(self.cache, key=lambda k: self.cache[k][0])
            self.cache.pop(oldest_key, None)
        self.cache[key] = (time.time(), value)

# ==========================================================
# Price fetcher (multi-source)
# ==========================================================
class PriceDataFetcher:
    @staticmethod
    def get_binance_price(symbol: str) -> Optional[float]:
        try:
            if symbol.endswith("-USD"):
                binance_symbol = symbol.replace("-USD", "USDT")
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    return float(r.json()['price'])
        except Exception as e:
            logger.warning(f"Binance price failed for {symbol}: {e}")
        return None

    @staticmethod
    def get_forex_price(symbol: str) -> Optional[float]:
        try:
            if symbol.endswith("=X"):
                pair = symbol.replace("=X", "")
                base, quote = pair[:3], pair[3:]
                url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    return float(data['rates'][quote])
        except Exception as e:
            logger.warning(f"Forex price failed for {symbol}: {e}")
        return None

    @staticmethod
    def get_market_price(symbol: str, fallback_price: float) -> float:
        sources = []
        if symbol.endswith("-USD"):
            p = PriceDataFetcher.get_binance_price(symbol)
            if p: sources.append(p)
        elif symbol.endswith("=X"):
            p = PriceDataFetcher.get_forex_price(symbol)
            if p: sources.append(p)
        if sources:
            return float(np.mean(sources))
        # small random variation to avoid identical charts
        variation = np.random.uniform(-0.001, 0.001)
        return fallback_price * (1 + variation)

# ==========================================================
# Technical Analysis
# ==========================================================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict:
        """Compute key indicators for the latest bar with robust guards."""
        if df.empty or len(df) < 30:
            return {}

        try:
            close = np.asarray(df['Close'], dtype=float)
            high = np.asarray(df['High'], dtype=float)
            low  = np.asarray(df['Low'], dtype=float)
            vol  = np.asarray(df['Volume'], dtype=float) if 'Volume' in df.columns else None

            ind = {}
            ind.update(TechnicalAnalyzer._calc_momentum(close, high, low))
            ind.update(TechnicalAnalyzer._calc_trend(close, high, low))
            ind.update(TechnicalAnalyzer._calc_volatility(close, high, low))
            ind.update(TechnicalAnalyzer._calc_volume(close, high, low, vol))
            ind.update(TechnicalAnalyzer._calc_sr(high, low))

            ind['current_price'] = float(close[-1])
            return ind
        except Exception as e:
            logger.error(f"Indicator calc error: {e}")
            return {}

    @staticmethod
    def _nan_to(value: float, default: float) -> float:
        return float(default if value is None or np.isnan(value) else value)

    @staticmethod
    def _calc_momentum(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict:
        res: Dict[str, float] = {}

        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        rsi_last = rsi[-1] if len(rsi) else np.nan
        res['rsi'] = TechnicalAnalyzer._nan_to(rsi_last, 50.0)
        rsi_prev = rsi[-5] if len(rsi) > 5 else np.nan
        res['rsi_trend'] = TechnicalAnalyzer._nan_to(rsi_last - rsi_prev, 0.0)

        # MACD
        macd, sig, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        res['macd'] = TechnicalAnalyzer._nan_to(macd[-1] if len(macd) else np.nan, 0.0)
        res['macd_signal'] = TechnicalAnalyzer._nan_to(sig[-1] if len(sig) else np.nan, 0.0)
        res['macd_histogram'] = TechnicalAnalyzer._nan_to(hist[-1] if len(hist) else np.nan, 0.0)
        hist_prev = hist[-5] if len(hist) > 5 else np.nan
        res['macd_trend'] = TechnicalAnalyzer._nan_to((hist[-1] - hist_prev) if len(hist) > 5 else np.nan, 0.0)

        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        res['stoch_k'] = TechnicalAnalyzer._nan_to(slowk[-1] if len(slowk) else np.nan, 50.0)
        res['stoch_d'] = TechnicalAnalyzer._nan_to(slowd[-1] if len(slowd) else np.nan, 50.0)

        # Momentum
        if len(close) >= 6 and close[-6] > 0:
            res['momentum_5'] = float((close[-1] - close[-6]) / close[-6] * 100.0)
        else:
            res['momentum_5'] = 0.0
        if len(close) >= 11 and close[-11] > 0:
            res['momentum_10'] = float((close[-1] - close[-11]) / close[-11] * 100.0)
        else:
            res['momentum_10'] = 0.0

        return res

    @staticmethod
    def _calc_trend(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict:
        res: Dict[str, float] = {}

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        last_close = float(close[-1])
        res['bb_upper'] = TechnicalAnalyzer._nan_to(upper[-1] if len(upper) else np.nan, last_close * 1.1)
        res['bb_middle'] = TechnicalAnalyzer._nan_to(middle[-1] if len(middle) else np.nan, last_close)
        res['bb_lower'] = TechnicalAnalyzer._nan_to(lower[-1] if len(lower) else np.nan, last_close * 0.9)
        bb_width = max(res['bb_upper'] - res['bb_lower'], EPS)
        res['bb_position'] = safe_divide(last_close - res['bb_lower'], bb_width, 0.5)

        # ADX & CCI
        adx = talib.ADX(high, low, close, timeperiod=14)
        res['adx'] = TechnicalAnalyzer._nan_to(adx[-1] if len(adx) else np.nan, 0.0)

        cci = talib.CCI(high, low, close, timeperiod=14)
        res['cci'] = TechnicalAnalyzer._nan_to(cci[-1] if len(cci) else np.nan, 0.0)

        return res

    @staticmethod
    def _calc_volatility(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict:
        res: Dict[str, float] = {}
        # ATR
        atr = talib.ATR(high, low, close, timeperiod=14)
        last_close = float(close[-1])
        res['atr'] = TechnicalAnalyzer._nan_to(atr[-1] if len(atr) else np.nan, last_close * 0.01)

        # Realized volatility (decimal, not %)
        if len(close) >= 20 and np.mean(close[-20:]) > 0:
            vol = float(np.std(close[-20:]) / np.mean(close[-20:]))
            res['volatility'] = 0.0 if np.isnan(vol) else max(vol, 0.0)
        else:
            res['volatility'] = 0.0
        return res

    @staticmethod
    def _calc_volume(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: Optional[np.ndarray]) -> Dict:
        res: Dict[str, float] = {}
        if volume is not None and np.nansum(volume) > 0:
            obv = talib.OBV(close, volume)
            res['obv'] = TechnicalAnalyzer._nan_to(obv[-1] if len(obv) else np.nan, 0.0)
            vol_sma = talib.SMA(volume, timeperiod=20)
            vol_sma_last = vol_sma[-1] if len(vol_sma) else np.nan
            if vol_sma_last and not np.isnan(vol_sma_last) and vol_sma_last > 0:
                res['volume_trend'] = float(volume[-1] / vol_sma_last)
            else:
                res['volume_trend'] = 1.0

            if len(volume) >= 5:
                avg5 = float(np.mean(volume[-5:]))
                res['volume_ratio'] = float(volume[-1] / avg5) if avg5 > 0 else 1.0
            else:
                res['volume_ratio'] = 1.0

            # VWAP approximation
            typical = (high + low + close) / 3.0
            cum_vol = np.cumsum(volume)
            vwap = np.cumsum(typical * volume) / np.where(cum_vol == 0, np.nan, cum_vol)
            last_vwap = vwap[-1] if len(vwap) else np.nan
            last_close = float(close[-1])
            res['vwap'] = TechnicalAnalyzer._nan_to(last_vwap, last_close)
            res['vwap_position'] = safe_divide(last_close - res['vwap'], res['vwap'], 0.0)
        else:
            res['obv'] = 0.0
            res['volume_trend'] = 1.0
            res['volume_ratio'] = 1.0
            res['vwap'] = float(close[-1])
            res['vwap_position'] = 0.0
        return res

    @staticmethod
    def _calc_sr(high: np.ndarray, low: np.ndarray) -> Dict:
        res: Dict[str, float] = {}
        if len(low) >= 20:
            res['support'] = float(np.min(low[-20:]))
        else:
            res['support'] = float(low[-1]) * 0.95 if len(low) else 0.0

        if len(high) >= 20:
            res['resistance'] = float(np.max(high[-20:]))
        else:
            res['resistance'] = float(high[-1]) * 1.05 if len(high) else 0.0
        return res

# ==========================================================
# Risk Management
# ==========================================================
class RiskManager:
    @staticmethod
    def calculate_position_size(account_size: float, risk_percent: float, entry: float, stop: float) -> 'PositionSizing':
        risk_amount = max(account_size * max(risk_percent, 0.0), 0.0)
        price_dist = max(abs(entry - stop), entry * 0.0001)  # at least 1bp
        position_size = risk_amount / price_dist
        position_value = position_size * entry
        position_percent = safe_divide(position_value, account_size, 0.0) * 100.0
        return PositionSizing(
            risk_amount=risk_amount,
            position_size=position_size,
            position_value=position_value,
            position_percent=position_percent
        )

    @staticmethod
    def calculate_stop_loss(price: float, atr: float, signal_type: str, volatility: float, strategy: str) -> float:
        # Normalize inputs
        atr = max(atr, price * 0.0005)  # at least 5 bps
        volatility = max(0.0, min(float(volatility), 1.0))  # expect decimal (e.g., 0.02 = 2%)

        base_stop = atr
        vol_factor = 1.0 + min(0.5, volatility * 10.0)  # cap widening at +50%
        adjusted = base_stop * vol_factor

        if strategy == "TREND_FOLLOWING":
            adjusted *= 1.2
        elif strategy == "REVERSAL":
            adjusted *= 0.8

        if signal_type in ["STRONG_BUY", "BUY"]:
            return price - adjusted
        elif signal_type in ["STRONG_SELL", "SELL"]:
            return price + adjusted
        return price

    @staticmethod
    def calculate_target(price: float, stop: float, signal_type: str, rrr: float = 2.0) -> float:
        risk_amt = abs(price - stop)
        if signal_type in ["STRONG_BUY", "BUY"]:
            return price + risk_amt * rrr
        elif signal_type in ["STRONG_SELL", "SELL"]:
            return price - risk_amt * rrr
        return price

# ==========================================================
# Strategies
# ==========================================================

# =============================
# Market Regime Detector (inspired by Claude)
# =============================
class MarketRegimeDetector:
    def __init__(self):
        self.lookback = 60  # bars

    def detect(self, indicators: Dict) -> Dict:
        """Return a simple regime dict: trend, volatility, volume_trend, vix_level"""
        trend = "SIDEWAYS"
        vol_regime = "NORMAL"
        volume_trend = "STABLE"
        vix_level = 20.0

        adx = indicators.get("adx", 0.0)
        mom10 = indicators.get("momentum_10", 0.0)
        volatility = indicators.get("volatility", 0.0)  # decimal or percent depending on analyzer

        if mom10 > 5 and adx > 25:
            trend = "BULL"
        elif mom10 < -5 and adx > 25:
            trend = "BEAR"
        else:
            trend = "SIDEWAYS"

        # volatility can be percent (e.g., 2.0) or decimal; normalize if >1 assume percent
        vol_val = float(volatility)
        if vol_val > 1.0:
            vol_val = vol_val / 100.0
        if vol_val > 0.30:
            vol_regime = "HIGH"
        elif vol_val < 0.10:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"

        vol_ratio = indicators.get("volume_ratio", 1.0)
        if vol_ratio > 1.2:
            volume_trend = "INCREASING"
        elif vol_ratio < 0.8:
            volume_trend = "DECREASING"

        vix_level = min(max(vol_val * 100 * 2, 10.0), 60.0)
        return {
            "trend": trend,
            "volatility": vol_regime,
            "volume_trend": volume_trend,
            "vix_level": vix_level
        }

# =============================
# Advanced Risk Manager (simplified Claude ideas)
# =============================
class AdvancedRiskManager:
    def __init__(self):
        pass

    def calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float, max_kelly: float = 0.25) -> float:
        if win_loss_ratio <= 0 or win_rate <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        kelly = max(0.0, min(kelly, max_kelly))
        # fractional Kelly (25% of full)
        return kelly * 0.25

    def calculate_optimal_position_size(self, signal, portfolio_value: float, max_risk_per_trade: float, correlation_data: Dict = None):
        # Use a mix of risk-based size and Kelly
        win_prob = getattr(signal, "win_probability", 0.55)
        expected_ret = getattr(signal, "expected_return", 0.01)
        mae = getattr(signal, "max_adverse_excursion", 0.01)
        wl_ratio = abs(expected_ret) / mae if mae > 0 else 1.0
        kelly = self.calculate_kelly_fraction(win_prob, wl_ratio)
        kelly_val = portfolio_value * kelly

        # basic risk-based size
        risk_amount = portfolio_value * max_risk_per_trade
        price_dist = max(abs(signal.entry_price - signal.stop_loss), signal.entry_price * 1e-4)
        basic_size = risk_amount / price_dist if price_dist > 0 else 0.0

        # volatility scaling (we expect volatility in decimal or percent)
        vol = signal.key_indicators.get("volatility", 0.02)
        if vol > 1.0:
            vol = vol / 100.0
        target_vol = 0.15
        vol_scalar = target_vol / max(vol, 1e-6)
        vol_adj_size = basic_size * vol_scalar

        final_size = min(basic_size, kelly_val / max(signal.entry_price, 1e-6), vol_adj_size)
        position_value = final_size * signal.entry_price
        position_percent = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0.0
        return {
            "risk_amount": risk_amount,
            "position_size": final_size,
            "position_value": position_value,
            "position_percent": position_percent,
            "kelly_fraction": kelly
        }
class TradingStrategy:
    name: str = "BASE"
    def calculate_score(self, indicators: Dict) -> Tuple[float, List[str]]:
        raise NotImplementedError

class TrendFollowingStrategy(TradingStrategy):
    name = "TREND_FOLLOWING"
    def calculate_score(self, ind: Dict) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        adx = ind.get('adx', 0.0)
        if adx > 25: score += 0.3; reasons.append(f"Strong trend (ADX {adx:.1f})")
        elif adx < 15: score -= 0.1; reasons.append(f"Weak trend (ADX {adx:.1f})")

        vwap_pos = ind.get('vwap_position', 0.0)
        if vwap_pos > 0.01: score += 0.2; reasons.append("Price above VWAP")
        elif vwap_pos < -0.01: score -= 0.2; reasons.append("Price below VWAP")

        price = ind.get('current_price', 0.0)
        sma20 = ind.get('bb_middle', price)
        if price > sma20: score += 0.2; reasons.append("Price above SMA20")
        else: score -= 0.2; reasons.append("Price below SMA20")

        macd_hist = ind.get('macd_histogram', 0.0)
        if macd_hist > 0: score += 0.3; reasons.append("MACD bullish")
        else: score -= 0.3; reasons.append("MACD bearish")

        return max(-1.0, min(1.0, score)), reasons

class MomentumStrategy(TradingStrategy):
    name = "MOMENTUM"
    def calculate_score(self, ind: Dict) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        rsi = ind.get('rsi', 50.0)
        if rsi > 70: score -= 0.3; reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 30: score += 0.3; reasons.append(f"RSI oversold ({rsi:.1f})")

        macd_hist = ind.get('macd_histogram', 0.0)
        score += float(np.tanh(macd_hist * 10.0)) * 0.3
        reasons.append("MACD bullish momentum" if macd_hist > 0 else "MACD bearish momentum")

        k = ind.get('stoch_k', 50.0)
        d = ind.get('stoch_d', 50.0)
        if k > 80 and d > 80: score -= 0.2; reasons.append("Stochastic overbought")
        elif k < 20 and d < 20: score += 0.2; reasons.append("Stochastic oversold")

        mom5 = ind.get('momentum_5', 0.0)
        score += float(np.tanh(mom5 / 10.0)) * 0.2
        reasons.append(f"Positive momentum ({mom5:.1f}%)" if mom5 > 0 else f"Negative momentum ({mom5:.1f}%)")

        return max(-1.0, min(1.0, score)), reasons

class ReversalStrategy(TradingStrategy):
    name = "REVERSAL"
    def calculate_score(self, ind: Dict) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        bb_pos = ind.get('bb_position', 0.5)
        if bb_pos < 0.1: score += 0.4; reasons.append("At lower Bollinger Band (reversal up)")
        elif bb_pos > 0.9: score -= 0.4; reasons.append("At upper Bollinger Band (reversal down)")

        rsi_trend = ind.get('rsi_trend', 0.0)
        price_trend = ind.get('momentum_5', 0.0)
        if rsi_trend > 0 and price_trend < 0: score += 0.3; reasons.append("Bullish RSI divergence")
        elif rsi_trend < 0 and price_trend > 0: score -= 0.3; reasons.append("Bearish RSI divergence")

        return max(-1.0, min(1.0, score)), reasons

# ==========================================================
# Trading Engine
# ==========================================================
class TradingEngine:
    def __init__(self):
        # regime detector and advanced risk manager (added from Claude ideas)
        self.regime_detector = MarketRegimeDetector()
        self.advanced_risk_manager = AdvancedRiskManager()
        self.signal_history: Deque[Tuple[str, float, datetime]] = deque(maxlen=200)
        self.strategies: List[TradingStrategy] = [
            TrendFollowingStrategy(),
            MomentumStrategy(),
            ReversalStrategy()
        ]
        self.strategy_weights: Dict[str, float] = {
            "TREND_FOLLOWING": 0.35,
            "MOMENTUM": 0.35,
            "REVERSAL": 0.30
        }

    def generate_signal(self, symbol: str, market_data: Dict) -> TradingSignal:
        indicators = market_data.get('indicators', {})
        if not indicators:
            return self._neutral_signal(symbol, market_data)

        strategy_scores: Dict[str, float] = {}
        reasons: List[str] = []

        for strat in self.strategies:
            s, r = strat.calculate_score(indicators)
            strategy_scores[strat.name] = s
            reasons.extend(r)

        composite = 0.0
        for name, s in strategy_scores.items():
            composite += s * self.strategy_weights.get(name, 0.0)
        composite = max(-1.0, min(1.0, composite))

        if composite > 0.6: signal_type = "STRONG_BUY"
        elif composite > 0.2: signal_type = "BUY"
        elif composite < -0.6: signal_type = "STRONG_SELL"
        elif composite < -0.2: signal_type = "SELL"
        else: signal_type = "NEUTRAL"

        timeframe = market_data.get('timeframe', '15m')
        if timeframe in ['1m','5m']: horizon = "SHORT_TERM"
        elif timeframe in ['15m','1h']: horizon = "MEDIUM_TERM"
        else: horizon = "LONG_TERM"

        dominant = max(strategy_scores.items(), key=lambda x: abs(x[1]))[0]
        confidence = self._confidence(composite, indicators, reasons)

        price = float(market_data['price'])
        atr = indicators.get('atr', price * 0.005)
        vol = indicators.get('volatility', 0.01)

        # stop = RiskManager.calculate_stop_loss(price, atr, signal_type, vol, dominant)\n# Market regime detection + advanced sizing (Claude-inspired)\n        regime = self.regime_detector.detect(indicators)\n        # enrich signal with simple expected metrics (placeholders derived from heuristics)\n        expected_ret, win_prob, mae = 0.0, 0.55, 0.01\n        # try to estimate using simple heuristics\n        expected_ret = 0.015 if 'BUY' in signal_type else -0.015 if 'SELL' in signal_type else 0.0\n        win_prob = min(0.8, 0.5 + abs(composite) * 0.3)\n        mae = max(0.005, atr / max(price, 1e-6))\n        # attach to local signal object by building a small namespace\n        # Use a simple namespace-like object\n        class _Tmp:\n            pass\n        tmp_sig = _Tmp()\n        tmp_sig.entry_price = price\n        tmp_sig.stop_loss = stop\n        tmp_sig.key_indicators = indicators\n        tmp_sig.expected_return = expected_ret\n        tmp_sig.win_probability = win_prob\n        tmp_sig.max_adverse_excursion = mae\n        # calculate suggested position sizing (user can use it in UI)\n        try:\n            sizing = self.advanced_risk_manager.calculate_optimal_position_size(tmp_sig, portfolio_value=10000, max_risk_per_trade=0.01)\n        except Exception:\n            sizing = {'position_size': 0.0, 'position_percent': 0.0, 'kelly_fraction': 0.0}\n
        stop = RiskManager.calculate_stop_loss(price, atr, signal_type, vol, dominant)
        # Market regime detection + advanced sizing (Claude-inspired)
        regime = self.regime_detector.detect(indicators)

        # enrich signal with simple expected metrics (placeholders derived from heuristics)
        expected_ret, win_prob, mae = 0.0, 0.55, 0.01
        # try to estimate using simple heuristics
        expected_ret = 0.015 if 'BUY' in signal_type else -0.015 if 'SELL' in signal_type else 0.0
        win_prob = min(0.8, 0.5 + abs(composite) * 0.3)
        mae = max(0.005, atr / max(price, 1e-6))

        # attach to local signal object by building a small namespace
        class _Tmp:
            pass

        tmp_sig = _Tmp()
        tmp_sig.entry_price = price
        tmp_sig.stop_loss = stop
        tmp_sig.key_indicators = indicators
        tmp_sig.expected_return = expected_ret
        tmp_sig.win_probability = win_prob
        tmp_sig.max_adverse_excursion = mae

        # calculate suggested position sizing (user can use it in UI)
        try:
            sizing = self.advanced_risk_manager.calculate_optimal_position_size(
                tmp_sig, portfolio_value=10000, max_risk_per_trade=0.01
            )
        except Exception:
            sizing = {'position_size': 0.0, 'position_percent': 0.0, 'kelly_fraction': 0.0}


        target = RiskManager.calculate_target(price, stop, signal_type, 2.0)
        rrr = safe_divide(abs(target - price), abs(price - stop), 1.0)

        reliability = self._reliability(confidence, indicators, reasons)

        self.signal_history.append((symbol, composite, datetime.now()))

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=float(confidence),
            strength_score=float(composite),
            market_price=price,
            entry_price=price,
            target_price=float(target),
            stop_loss=float(stop),
            risk_reward_ratio=float(rrr),
            timestamp=datetime.now(),
            key_indicators=indicators,
            signal_reasons=reasons,
            reliability_score=reliability,
            strategy=dominant,
            time_horizon=horizon
        )

    def _confidence(self, strength: float, ind: Dict, reasons: List[str]) -> float:
        base = min(0.95, 0.5 + abs(strength) * 0.4)
        confirm = min(0.2, len(reasons) * 0.05)
        vol = ind.get('volatility', 0.0)  # decimal
        vol_factor = 1.0 - min(0.3, vol * 5.0)  # 2% vol -> 0.1 reduction
        c = (base + confirm) * vol_factor
        return float(min(0.99, max(0.1, c)))

    def _reliability(self, confidence: float, ind: Dict, reasons: List[str]) -> float:
        vol = ind.get('volatility', 0.0)
        vol_factor = 1.0 - min(0.3, vol * 5.0)
        agreement = min(1.0, len(reasons) / 8.0)
        r = 0.7 * confidence + 0.2 * vol_factor + 0.1 * agreement
        return float(min(0.99, max(0.1, r)))

    def _neutral_signal(self, symbol: str, market_data: Dict) -> TradingSignal:
        price = float(market_data['price'])
        return TradingSignal(
            symbol=symbol,
            signal_type="NEUTRAL",
            confidence=0.5,
            strength_score=0.0,
            market_price=price,
            entry_price=price,
            target_price=price,
            stop_loss=price,
            risk_reward_ratio=1.0,
            timestamp=datetime.now(),
            key_indicators={},
            signal_reasons=["Insufficient data or weak signals"],
            reliability_score=0.5,
            strategy="NONE",
            time_horizon="SHORT_TERM",
        )

# ==========================================================
# Data Manager
# ==========================================================
class DataManager:
    def __init__(self):
        self.cache = DataCache(ttl_seconds=60)

    def fetch_history(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        tf = DEFAULT_TIMEFRAMES.get(timeframe, DEFAULT_TIMEFRAMES['15m'])
        interval, period = tf['interval'], tf['period']
        hist = yf.Ticker(symbol).history(period=period, interval=interval)
        if hist.empty or len(hist) < 30:
            return None
        hist = hist.ffill().bfill()
        if hist.isnull().values.any():
            logger.warning(f"NaNs in history for {symbol} after fill")
            return None
        return hist

    def get_market_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        cache_key = f"{symbol}_{timeframe}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        hist = self.fetch_history(symbol, timeframe)
        if hist is None:
            return None
        last_price = float(hist['Close'].iloc[-1])
        market_price = PriceDataFetcher.get_market_price(symbol, last_price)
        indicators = TechnicalAnalyzer.calculate_indicators(hist)
        if not indicators:
            return None
        data = {'symbol': symbol, 'price': market_price, 'indicators': indicators, 'hist': hist, 'timeframe': timeframe}
        self.cache.set(cache_key, data)
        return data

# ==========================================================
# Performance Tracker
# ==========================================================
class PerformanceTracker:
    def __init__(self):
        self.trades: List[Dict] = []
        self.metrics: Dict[str, float] = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0
        }

    def add_trade(self, signal: TradingSignal, exit_price: float, qty: float, exit_time: datetime):
        pnl = (exit_price - signal.entry_price) * qty if 'BUY' in signal.signal_type else (signal.entry_price - exit_price) * qty if 'SELL' in signal.signal_type else 0.0
        self.trades.append({
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'type': signal.signal_type,
            'entry': signal.entry_price,
            'exit': exit_price,
            'qty': qty,
            'profit': pnl,
            'rrr': signal.risk_reward_ratio,
            'holding_minutes': 0 if signal.timestamp is None else max(0, int((exit_time - signal.timestamp).total_seconds() / 60))
        })
        self._recompute()

    def _recompute(self):
        if not self.trades:
            return
        profits = np.array([t['profit'] for t in self.trades], dtype=float)
        wins = profits[profits > 0]
        losses = profits[profits <= 0]

        self.metrics['total_trades'] = int(len(self.trades))
        self.metrics['winning_trades'] = int(len(wins))
        self.metrics['losing_trades'] = int(len(losses))
        self.metrics['total_profit'] = float(np.nansum(profits))

        win_rate = safe_divide(len(wins), len(self.trades), 0.0)
        avg_win = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) else 0.0

        total_win = float(np.sum(wins)) if len(wins) else 0.0
        total_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
        self.metrics['profit_factor'] = safe_divide(total_win, total_loss, 0.0)

        self.metrics['expectancy'] = (win_rate * avg_win) - ((1.0 - win_rate) * abs(avg_loss))

        equity = np.cumsum(profits)
        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            dd = np.where(peak == 0, 0.0, (equity - peak) / np.where(peak == 0, 1.0, peak)) * 100.0
            self.metrics['max_drawdown'] = float(np.min(dd))

            # returns as diff over lagged equity; guard zeros
            if len(equity) > 1:
                prev = np.where(equity[:-1] == 0, 1.0, equity[:-1])
                rets = (equity[1:] - equity[:-1]) / prev
                if np.std(rets) > 0:
                    self.metrics['sharpe_ratio'] = float(np.mean(rets) / np.std(rets))
                else:
                    self.metrics['sharpe_ratio'] = 0.0

    def report(self) -> Dict[str, float]:
        return self.metrics

# ==========================================================
# Backtesting (simple event-driven hook)
# ==========================================================
class Backtester:
    def __init__(self, engine: TradingEngine):
        self.engine = engine

    def run(self, symbol: str, timeframe: str = '1d', max_hold_bars: int = 20) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Simple walk-forward backtest:
        - Generate signal each bar using history up to that bar (no lookahead)
        - Enter on close of the bar where signal is created
        - Exit on target/stop or after max_hold_bars
        """
        tf = DEFAULT_TIMEFRAMES.get(timeframe, DEFAULT_TIMEFRAMES['1d'])
        hist = yf.Ticker(symbol).history(period=tf['period'], interval=tf['interval'])
        hist = hist.ffill().bfill()
        if hist.empty or len(hist) < 60:
            return pd.DataFrame(), {'error': 'Insufficient data'}

        trades: List[Dict] = []
        i = 50  # warmup for indicators
        while i < len(hist) - 1:
            window = hist.iloc[:i+1].copy()
            data = {
                'symbol': symbol,
                'price': float(window['Close'].iloc[-1]),
                'indicators': TechnicalAnalyzer.calculate_indicators(window),
                'hist': window,
                'timeframe': timeframe
            }
            sig = self.engine.generate_signal(symbol, data)
            if sig.signal_type in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
                direction = 1 if 'BUY' in sig.signal_type else -1
                entry_idx = i
                entry_price = float(window['Close'].iloc[-1])
                atr = sig.key_indicators.get('atr', entry_price * 0.01)
                vol = sig.key_indicators.get('volatility', 0.02)
                stop = RiskManager.calculate_stop_loss(entry_price, atr, sig.signal_type, vol, sig.strategy)
                target = RiskManager.calculate_target(entry_price, stop, sig.signal_type, 2.0)

                exit_price = entry_price
                exit_idx = min(i + max_hold_bars, len(hist) - 1)
                j = i + 1
                while j <= exit_idx:
                    hi = float(hist['High'].iloc[j])
                    lo = float(hist['Low'].iloc[j])
                    if direction == 1:
                        if lo <= stop:
                            exit_price = stop; i = j; break
                        if hi >= target:
                            exit_price = target; i = j; break
                    else:
                        if hi >= stop:
                            exit_price = stop; i = j; break
                        if lo <= target:
                            exit_price = target; i = j; break
                    j += 1
                else:
                    # time exit at close
                    exit_price = float(hist['Close'].iloc[exit_idx])
                    i = exit_idx

                trades.append({
                    'timestamp': hist.index[entry_idx],
                    'symbol': symbol,
                    'type': sig.signal_type,
                    'entry': entry_price,
                    'exit': exit_price,
                    'rrr': sig.risk_reward_ratio,
                    'bars_held': (i - entry_idx)
                })
            i += 1

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return trades_df, {'note': 'No trades generated'}

        # Compute PnL per 1 unit
        trades_df['profit'] = np.where(trades_df['type'].str.contains('BUY'),
                                       trades_df['exit'] - trades_df['entry'],
                                       trades_df['entry'] - trades_df['exit'])
        metrics = {
            'trades': int(len(trades_df)),
            'win_rate': float((trades_df['profit'] > 0).mean()) if len(trades_df) else 0.0,
            'avg_profit': float(trades_df['profit'].mean()) if len(trades_df) else 0.0,
            'profit_factor': float(trades_df.loc[trades_df['profit']>0,'profit'].sum() / max(1e-9, -trades_df.loc[trades_df['profit']<=0,'profit'].sum())) if len(trades_df) else 0.0,
            'max_drawdown': float((trades_df['profit'].cumsum().cummax() - trades_df['profit'].cumsum()).max() * -1)
        }
        return trades_df, metrics

# ==========================================================
# Visualization
# ==========================================================
class ChartVisualizer:
    @staticmethod
    def create_advanced_chart(df: pd.DataFrame, symbol: str, signal: TradingSignal) -> go.Figure:
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=('Price', 'Volume', 'RSI', 'MACD'),
            row_width=[0.2, 0.2, 0.2, 0.4]
        )

        # Candles
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
        ), row=1, col=1)

        # Bollinger Bands (simple)
        bb_period = 20
        bb_std = 2
        mid = df['Close'].rolling(bb_period).mean()
        std = df['Close'].rolling(bb_period).std()
        upper = mid + bb_std * std
        lower = mid - bb_std * std

        fig.add_trace(go.Scatter(x=df.index, y=upper, name='BB Upper', line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=mid,   name='BB Middle', line=dict(width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='BB Lower', line=dict(width=1)), row=1, col=1)

        # Volume
        if 'Volume' in df.columns:
            colors = ['red' if o > c else 'green' for o, c in zip(df['Open'], df['Close'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)

        # RSI
        rsi = talib.RSI(np.asarray(df['Close'], dtype=float), timeperiod=14)
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI'), row=3, col=1)
        # RSI bands via shapes
        fig.add_hrect(y0=70, y1=70, line_width=1, line_dash="dash", row=3, col=1)
        fig.add_hrect(y0=30, y1=30, line_width=1, line_dash="dash", row=3, col=1)
        fig.add_hrect(y0=50, y1=50, line_width=1, line_dash="dot", row=3, col=1)

        # MACD
        macd, macd_sig, macd_hist = talib.MACD(np.asarray(df['Close'], dtype=float))
        fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='MACD Hist'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_sig, name='Signal'), row=4, col=1)

        fig.update_layout(
            title=f"{symbol} - Advanced Technical Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        return fig

# ==========================================================
# Streamlit App
# ==========================================================
class TradingBotApp:
    def __init__(self):
        self.engine = TradingEngine()
        self.data_manager = DataManager()
        self.visualizer = ChartVisualizer()
        self.perf = PerformanceTracker()
        self.backtester = Backtester(self.engine)

        st.set_page_config(
            page_title="ULTIMATE Trading AI - Pro Edition",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def sidebar(self) -> Dict:
        st.sidebar.header("🛠️ Configuration")

        # Assets
        stock_symbols = ["AAPL","GOOGL","MSFT","TSLA","NVDA","AMZN","META","JPM","JNJ","V"]
        forex_symbols = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","USDSEK=X"]
        crypto_symbols = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD","DOT-USD"]
        commodity_symbols = ["GC=F","SI=F","CL=F","NG=F","ZC=F","ZS=F"]

        selected_stocks = st.sidebar.multiselect("Stocks", stock_symbols, default=["AAPL","TSLA"])
        selected_fx = st.sidebar.multiselect("Forex", forex_symbols, default=["EURUSD=X","USDJPY=X"])
        selected_crypto = st.sidebar.multiselect("Crypto", crypto_symbols, default=["BTC-USD","ETH-USD"])
        selected_cmd = st.sidebar.multiselect("Commodities", commodity_symbols, default=["GC=F","CL=F"])
        symbols = selected_stocks + selected_fx + selected_crypto + selected_cmd

        timeframe = st.sidebar.selectbox("Primary Timeframe", list(DEFAULT_TIMEFRAMES.keys()), index=2)
        contract_minutes = st.sidebar.number_input("Contract period (minutes)", min_value=1, max_value=1440, value=15, step=1)

        st.sidebar.subheader("📈 Signal Filters")
        min_conf = st.sidebar.slider("Min Confidence", 0.5, 0.99, 0.65)
        min_rel  = st.sidebar.slider("Min Reliability", 0.5, 0.99, 0.70)
        min_str  = st.sidebar.slider("Min Strength", 0.1, 0.9, 0.25, step=0.05)

        st.sidebar.subheader("⚖️ Risk Management")
        max_pos = st.sidebar.slider("Max Position Size (%)", 1, 20, 5)
        max_daily_loss = st.sidebar.slider("Max Daily Loss (%)", 1, 20, 5)
        default_risk = st.sidebar.slider("Default Risk per Trade (%)", 0.5, 5.0, 1.0, step=0.5) / 100.0

        st.sidebar.subheader("🧪 Backtesting")
        run_bt = st.sidebar.checkbox("Enable Backtesting", value=False)
        bt_symbol = st.sidebar.selectbox("Backtest Symbol", symbols if symbols else ["AAPL"]) if run_bt else None
        bt_timeframe = st.sidebar.selectbox("Backtest Timeframe", ["1d","1h","15m"], index=0) if run_bt else None
        bt_bars = st.sidebar.slider("Max Hold Bars", 5, 100, 20) if run_bt else 0

        debug = st.sidebar.checkbox("Debug Mode", value=False)
        refresh_sec = st.sidebar.slider("Auto-refresh every (seconds)", 10, 300, 30, step=5)
        st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_pro")

        return {
            'symbols': symbols, 'timeframe': timeframe, 'contract_minutes': contract_minutes,
            'min_conf': min_conf, 'min_rel': min_rel, 'min_str': min_str,
            'max_pos': max_pos, 'max_daily_loss': max_daily_loss, 'default_risk': default_risk,
            'debug': debug, 'run_bt': run_bt, 'bt_symbol': bt_symbol, 'bt_timeframe': bt_timeframe, 'bt_bars': bt_bars
        }

    def header(self, symbols, timeframe, contract_minutes, max_pos):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Assets", len(symbols))
        with c2: st.metric("Timeframe", timeframe)
        with c3: st.metric("Contract (min)", contract_minutes)
        with c4: st.metric("Max Position", f"{max_pos}%")
        with c5: st.metric("Last Update", datetime.now().strftime('%H:%M:%S'))

    def generate_signals(self, symbols: List[str], timeframe: str):
        all_signals = []
        bar = st.progress(0)
        text = st.empty()
        for i, sym in enumerate(symbols):
            text.text(f"Loading {pretty_asset(sym)}...")
            bar.progress(int((i+1)/max(1,len(symbols)) * 100))
            data = self.data_manager.get_market_data(sym, timeframe)
            if not data:
                st.info(f"No recent data for {pretty_asset(sym)} ({timeframe}).")
                continue
            sig = self.engine.generate_signal(sym, data)
            all_signals.append((sym, sig, data))
        bar.empty(); text.empty()
        return all_signals

    def filter_signals(self, signals, min_conf, min_rel, min_str, debug):
        qualified = [t for t in signals if t[1].confidence >= min_conf and t[1].reliability_score >= min_rel and abs(t[1].strength_score) >= min_str and t[1].signal_type != "NEUTRAL"]
        if not qualified and debug:
            st.warning("No signals met filters — showing all for review.")
            return signals
        if not qualified:
            st.warning("No signals met filters. Try adjusting thresholds or assets.")
        return qualified

    def render_signal(self, sym: str, sig: TradingSignal, data: Dict, timeframe: str, contract_minutes: int, debug: bool):
        asset = pretty_asset(sym)
        color = "green" if "BUY" in sig.signal_type else "red" if "SELL" in sig.signal_type else "gray"
        intensity = "🟢" if "STRONG" in sig.signal_type else "🟡"
        with st.expander(f"{intensity} {asset} — :{color}[{sig.signal_type}]  (Conf: {sig.confidence*100:.0f}%, Rel: {sig.reliability_score*100:.0f}%)", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Signal", "Technical", "Risk"])
            with tab1:
                self._signal_tab(sym, sig, timeframe, contract_minutes)
            with tab2:
                fig = self.visualizer.create_advanced_chart(data['hist'], asset, sig)
                st.plotly_chart(fig, use_container_width=True)
                if debug:
                    self._debug_tab(sig)
            with tab3:
                self._risk_tab(sig)

    def _signal_tab(self, sym: str, sig: TradingSignal, timeframe: str, contract_minutes: int):
        atr = sig.key_indicators.get('atr', max(sig.market_price * 0.005, EPS))
        low, high = compute_entry_zone(sig.entry_price, atr)
        info = f"""
[Signal] {sig.signal_type}
[Asset] {pretty_asset(sym)}
[Strategy] {sig.strategy}
[Time Horizon] {sig.time_horizon}
[Timeframe] {timeframe}
[Contract Period] {contract_minutes} minutes
[Entry Zone] {low:.5f} – {high:.5f}
[Target] {sig.target_price:.5f}
[Stop Loss] {sig.stop_loss:.5f}
[Risk/Reward] {sig.risk_reward_ratio:.2f}
[Confidence] {sig.confidence*100:.0f}%
[Reliability] {sig.reliability_score*100:.0f}%
[Strength] {sig.strength_score:.2f}
[Reasoning] {', '.join(sig.signal_reasons[:3])}
"""
        st.code(info)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Market Price", f"{sig.market_price:.5f}", delta=f"{sig.key_indicators.get('momentum_5', 0.0):.2f}% (5p)")
        with c2: st.metric("ATR", f"{sig.key_indicators.get('atr', 0.0):.5f}", delta=f"{safe_divide(sig.key_indicators.get('atr', 0.0), sig.market_price, 0.0)*100:.2f}%")
        with c3: st.metric("RRR", f"{sig.risk_reward_ratio:.2f}", delta="1:2+" if sig.risk_reward_ratio >= 2 else "1:1-2")
        with c4: st.metric("Volatility", f"{sig.key_indicators.get('volatility', 0.0)*100:.2f}%")  # decimal→percent

    def _debug_tab(self, sig: TradingSignal):
        st.subheader("Indicator Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write("RSI:", f"{sig.key_indicators.get('rsi', 0.0):.2f}")
            st.write("MACD:", f"{sig.key_indicators.get('macd', 0.0):.4f}")
            st.write("Stoch K:", f"{sig.key_indicators.get('stoch_k', 0.0):.2f}")
        with c2:
            st.write("ADX:", f"{sig.key_indicators.get('adx', 0.0):.2f}")
            st.write("CCI:", f"{sig.key_indicators.get('cci', 0.0):.2f}")
            st.write("OBV:", f"{sig.key_indicators.get('obv', 0.0):.0f}")
        with c3:
            st.write("BB Position:", f"{sig.key_indicators.get('bb_position', 0.0):.2f}")
            st.write("VWAP Position:", f"{sig.key_indicators.get('vwap_position', 0.0)*100:.2f}%")
        with c4:
            st.write("Volume Ratio:", f"{sig.key_indicators.get('volume_ratio', 1.0):.2f}")
            st.write("Support/Resistance:", f"{sig.key_indicators.get('support', 0.0):.2f}/{sig.key_indicators.get('resistance', 0.0):.2f}")

    def _risk_tab(self, sig: TradingSignal):
        st.subheader("Position Sizing")
        acct = st.number_input("Account Size ($)", min_value=100, max_value=5_000_000, value=10_000, step=1000, key=f"acct_{sig.symbol}")
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, step=0.5, key=f"risk_{sig.symbol}") / 100.0
        sizing = RiskManager.calculate_position_size(acct, risk_pct, sig.entry_price, sig.stop_loss)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Risk Amount", f"${sizing.risk_amount:.2f}")
        with c2: st.metric("Position Size", f"{sizing.position_size:.2f} units")
        with c3: st.metric("Position Value", f"${sizing.position_value:.2f} ({sizing.position_percent:.1f}%)")

        max_pos = st.session_state.get('max_pos', 5)
        if sizing.position_percent > max_pos:
            st.error(f"⚠️ Position size ({sizing.position_percent:.1f}%) exceeds maximum ({max_pos}%)")
        elif sizing.position_percent > 0.8 * max_pos:
            st.warning(f"⚠️ Position size ({sizing.position_percent:.1f}%) close to maximum ({max_pos}%)")
        else:
            st.success(f"✅ Position size ({sizing.position_percent:.1f}%) within limits")

        st.subheader("Daily Loss Protection")
        max_daily = st.session_state.get('max_daily_loss', 5)
        st.info(f"Maximum daily loss limit: ${acct * (max_daily/100):.2f} ({max_daily}%)")


    def performance_summary(self, qualified):
        st.subheader("📊 Performance Summary")
        strong_buy = len([s for s in qualified if "STRONG_BUY" in s[1].signal_type])
        buy = len([s for s in qualified if s[1].signal_type == "BUY"])
        strong_sell = len([s for s in qualified if "STRONG_SELL" in s[1].signal_type])
        sell = len([s for s in qualified if s[1].signal_type == "SELL"])

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Total Signals", len(qualified))
        with c2: st.metric("Strong Buy", strong_buy)
        with c3: st.metric("Buy", buy)
        with c4: st.metric("Strong Sell", strong_sell)
        with c5: st.metric("Sell", sell)

        strategies = [s[1].strategy for s in qualified]
        counts = {s: strategies.count(s) for s in set(strategies)}
        st.write("Strategy Distribution:")
        for k, v in counts.items():
            st.write(f"- {k}: {v} signals")

        status = "Bullish" if strong_buy + buy > strong_sell + sell else ("Bearish" if strong_sell + sell > strong_buy + buy else "Neutral")
        st.subheader("🌐 Market Overview")
        st.metric("Overall Market Bias", status)

    def run_backtest_ui(self, symbol: str, timeframe: str, max_hold_bars: int):
        st.subheader("🧪 Backtest Results")
        with st.spinner("Running backtest..."):
            trades_df, metrics = self.backtester.run(symbol, timeframe, max_hold_bars=max_hold_bars)
        if isinstance(metrics, dict) and 'error' in metrics:
            st.error(metrics['error']); return
        if trades_df.empty:
            st.info(metrics.get('note', 'No trades generated.')); return
        st.dataframe(trades_df.tail(20))
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Trades", metrics['trades'])
        with c2: st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
        with c3: st.metric("Avg Profit (1u)", f"{metrics['avg_profit']:.4f}")
        with c4: st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

        eq = trades_df['profit'].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(eq))), y=eq, name='Equity (1u per trade)'))
        fig.update_layout(title=f"Equity Curve — {symbol} ({timeframe})", xaxis_title='Trade #', yaxis_title='PnL')
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        st.title("🚀 ULTIMATE Trading AI — Pro Edition")
        st.caption("Multi-strategy signals, robust risk management, and backtesting hooks.")

        cfg = self.sidebar()
        if not cfg['symbols']:
            st.warning("Select at least one asset from the sidebar."); return

        # expose limits to other tabs
        for k, v in cfg.items():
            if k not in ['symbols']:
                st.session_state[k] = v

        self.header(cfg['symbols'], cfg['timeframe'], cfg['contract_minutes'], cfg['max_pos'])

        signals = self.generate_signals(cfg['symbols'], cfg['timeframe'])
        if not signals:
            st.warning("No signals could be generated."); return

        qualified = self.filter_signals(signals, cfg['min_conf'], cfg['min_rel'], cfg['min_str'], cfg['debug'])
        if not qualified:
            return

        qualified.sort(key=lambda x: (x[1].confidence * abs(x[1].strength_score)), reverse=True)

        st.subheader("🎯 Trading Signals")
        for sym, sig, data in qualified:
            self.render_signal(sym, sig, data, cfg['timeframe'], cfg['contract_minutes'], cfg['debug'])

        self.performance_summary(qualified)

        if cfg['run_bt'] and cfg['bt_symbol']:
            self.run_backtest_ui(cfg['bt_symbol'], cfg['bt_timeframe'], cfg['bt_bars'])


# ==========================================================
# Entrypoint
# ==========================================================
if __name__ == "__main__":
    app = TradingBotApp()
    app.run()
