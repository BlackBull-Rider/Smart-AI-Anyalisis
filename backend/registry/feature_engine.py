import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# EXISTING IMPORTS
# ==============================================================================
from backend.indicators.core.moving_average import ema
from backend.indicators.core.momentum import (
    rsi, adx, macd, roc, momentum,
)
from backend.indicators.core.volatility import (
    atr, supertrend,
)
from backend.indicators.core.statistics import (
    linear_regression,
)
from backend.indicators.core.volume import (
    vwap,
)
from backend.indicators.core.pattern import calculate_patterns
from backend.indicators.core.support_resistance import (
    swing_high, swing_low, bos_level, choch_level, liquidity_pool
)

# ==============================================================================
# SMART MONEY CONCEPTS (L1 IMPORTS)
# ==============================================================================
from backend.indicators.core.smart_money import (
    bos, internal_bos, external_bos, choch, internal_choch, external_choch,
    higher_high, higher_low, market_structure_shift, structure_strength,
    bullish_order_block, bearish_order_block, mitigation_block, breaker_block,
    order_block_age, order_block_strength, mitigated_order_block,
    bullish_fvg, bearish_fvg, inverse_fvg, fvg_width, fvg_age, fvg_strength,
    filled_fvg, liquidity_sweep, liquidity_score, equal_high, equal_low,
    internal_liquidity, external_liquidity, liquidity_strength, stop_hunt,
    institutional_score, displacement_candle, active_fvg
)

# ==============================================================================
# MISSING IMPORTS (CANDLE & SMC)
# ==============================================================================
try:
    from backend.indicators.core.candle import (
        bullish_engulfing, bearish_engulfing, doji, marubozu,
        morning_star, evening_star, three_white_soldiers, three_black_crows,
        inside_bar, outside_bar
    )
except ImportError:
    pass

try:
    from backend.indicators.core.smc import (
        fair_value_gap, mitigation, order_block, breaker
    )
except ImportError:
    pass

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =========================
    # EMA
    # =========================
    df["ema_20"] = ema(df, length=20)
    df["ema_50"] = ema(df, length=50)

    # =========================
    # RSI
    # =========================
    df["rsi"] = rsi(df, length=14)

    # =========================
    # ADX
    # =========================
    adx_df = adx(df, length=14)
    if isinstance(adx_df, pd.DataFrame):
        df["adx"] = adx_df["adx"] if "adx" in adx_df.columns else adx_df.iloc[:, 0]
    else:
        df["adx"] = adx_df

    # =========================
    # ATR
    # =========================
    df["atr_14"] = atr(df, length=14)
    df["atr"] = df["atr_14"]  # Fix for missing hook

    # =========================
    # MACD
    # =========================
    macd_df = macd(df)
    if isinstance(macd_df, pd.DataFrame):
        df["macd_line"] = macd_df["macd"] if "macd" in macd_df.columns else macd_df.iloc[:, 0]
        df["macd_signal"] = macd_df["signal"] if "signal" in macd_df.columns else macd_df.iloc[:, 1]
        if "histogram" in macd_df.columns:
            df["macd_histogram"] = macd_df["histogram"]

    # =========================
    # ROC
    # =========================
    df["roc"] = roc(df, length=14)

    # =========================
    # MOMENTUM
    # =========================
    df["momentum"] = momentum(df, length=10)

    # =========================
    # SUPERTREND
    # =========================
    st = supertrend(df, length=10, multiplier=3.0)
    if isinstance(st, pd.DataFrame):
        df["supertrend"] = st.iloc[:, 0]
    else:
        df["supertrend"] = st

    # =========================
    # VWAP
    # =========================
    vw = vwap(df)
    if isinstance(vw, pd.DataFrame):
        df["vwap"] = vw.iloc[:, 0]
    else:
        df["vwap"] = vw

    # =========================
    # Linear Regression
    # =========================
    lr = linear_regression(df, length=100)
    if isinstance(lr, pd.DataFrame):
        df["linreg_slope"] = lr["slope"] if "slope" in lr.columns else lr.iloc[:, 0]
        df["linreg_r2"] = lr["r2"] if "r2" in lr.columns else lr.iloc[:, 1]

    # =========================
    # TREND CONTEXT
    # =========================
    df['trend_direction'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)

    # =========================
    # CANDLESTICK PATTERNS & ANATOMY
    # =========================
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    c_range = h - l + 1e-9

    # Close Location Value (-1 to +1)
    df['clv'] = ((c - l) - (h - c)) / c_range
    abs_body = (c - o).abs()
    
    # === RENAMED: body_percent to body_pct ===
    df['body_pct'] = (abs_body / c_range) * 100.0

    try:
        df["bullish_engulfing"] = bullish_engulfing(df)
        df["bearish_engulfing"] = bearish_engulfing(df)
        df["doji"] = doji(df)
        df["marubozu"] = marubozu(df)
        df["morning_star"] = morning_star(df)
        df["evening_star"] = evening_star(df)
        df["three_white_soldiers"] = three_white_soldiers(df)
        df["three_black_crows"] = three_black_crows(df)
        df["inside_bar"] = inside_bar(df)
        df["outside_bar"] = outside_bar(df)
    except NameError:
        df['doji'] = ((abs_body / c_range) < 0.1).astype(int)
        df['inside_bar'] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
        df['outside_bar'] = ((h > h.shift(1)) & (l < l.shift(1))).astype(int)

    # =========================
    # GAPS
    # =========================
    prev_h = df['high'].shift(1)
    prev_l = df['low'].shift(1)
    df['gap_up'] = (df['open'] > prev_h).astype(int)
    df['gap_down'] = (df['open'] < prev_l).astype(int)
    df['gap_size'] = (df['open'] - df['close'].shift(1)).abs()

    # =========================
    # SMART MONEY CONCEPTS (SMC)
    # =========================
    try:
        df["bos"] = bos(df)
        df["choch"] = choch(df)
        df["liquidity_sweep"] = liquidity_sweep(df)
        df["fair_value_gap"] = fair_value_gap(df)
        df["mitigation"] = mitigation(df)
        df["order_block"] = order_block(df)
        df["breaker"] = breaker(df)
    except NameError:
        bull_fvg = l > h.shift(2)
        bear_fvg = h < l.shift(2)
        df['fair_value_gap'] = np.where(bull_fvg, 1, np.where(bear_fvg, -1, 0))
        df['mitigation'] = df['fair_value_gap'].shift(1).fillna(0)

        swing_h = (h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2))
        swing_l = (l < l.shift(1)) & (l < l.shift(2)) & (l < l.shift(-1)) & (l < l.shift(-2))
        sh_val = h.where(swing_h).ffill()
        sl_val = l.where(swing_l).ffill()
        
        bull_sweep = (l < sl_val.shift(1)) & (c > sl_val.shift(1))
        bear_sweep = (h > sh_val.shift(1)) & (c < sh_val.shift(1))
        df['liquidity_sweep'] = np.where(bull_sweep, 1, np.where(bear_sweep, -1, 0))

        df['bos'] = np.where((c > sh_val.shift(1)) & (c.shift(1) <= sh_val.shift(2)), 1,
                    np.where((c < sl_val.shift(1)) & (c.shift(1) >= sl_val.shift(2)), -1, 0))
        df['choch'] = df['bos']
        df['order_block'] = df['liquidity_sweep']
        df['breaker'] = df['bos'].shift(1).fillna(0)

    # =========================
    # ADVANCED MISSING FEATURES (For Candle Analyzer)
    # =========================
    # 1. Volume Ratio
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)

    # 2. Normalized Volatility
    df['normalized_volatility'] = df['atr'] / (df['close'] + 1e-9)
    
    # 3. Efficiency Ratio (Kaufman's ER logic - 10 period)
    direction = (df['close'] - df['close'].shift(10)).abs()
    volatility = (df['close'] - df['close'].shift(1)).abs().rolling(10).sum()
    df['efficiency_ratio'] = direction / (volatility + 1e-9)

    # 4. Bull Sequence
    is_bull = df['close'] > df['close'].shift(1)
    df['bull_sequence'] = is_bull.groupby((~is_bull).cumsum()).cumsum()

    # 5. Bear Sequence
    is_bear = df['close'] < df['close'].shift(1)
    df['bear_sequence'] = is_bear.groupby((~is_bear).cumsum()).cumsum()

    # =========================
    # VOLUME ANALYZER FEATURES (NEWLY ADDED)
    # =========================
    v = df['volume']
    
    # OBV
    df['obv'] = np.where(c > c.shift(1), v, np.where(c < c.shift(1), -v, 0)).cumsum()
    
    # Money Flow Multiplier & CMF & ADL
    mfm = ((c - l) - (h - c)) / c_range
    df['money_flow'] = mfm * v
    df['adl'] = df['money_flow'].cumsum()
    df['accdist'] = df['adl']
    df['cmf'] = df['money_flow'].rolling(20).sum() / (v.rolling(20).sum() + 1e-9)

    # MFI (14 period)
    tp = (h + l + c) / 3.0
    rmf = tp * v
    pos_mf = np.where(tp > tp.shift(1), rmf, 0.0)
    neg_mf = np.where(tp < tp.shift(1), rmf, 0.0)
    pos_mf_sum = pd.Series(pos_mf).rolling(14).sum()
    neg_mf_sum = pd.Series(neg_mf).rolling(14).sum()
    mfr = pos_mf_sum / (neg_mf_sum + 1e-9)
    df['mfi'] = 100.0 - (100.0 / (1.0 + mfr))

    # VPT
    df['vpt'] = (v * (c - c.shift(1)) / (c.shift(1) + 1e-9)).cumsum()

    # Elder's Force Index
    df['force_index'] = ((c - c.shift(1)) * v).ewm(span=13, adjust=False).mean()

    # NVI & PVI
    price_roc = c.pct_change()
    vol_down = v < v.shift(1)
    vol_up = v > v.shift(1)
    df['nvi'] = 1000.0 * (1.0 + np.where(vol_down, price_roc, 0.0)).cumprod()
    df['pvi'] = 1000.0 * (1.0 + np.where(vol_up, price_roc, 0.0)).cumprod()
    
    # Volume Z-Score & Percentile
    vol_20_mean = v.rolling(20).mean()
    vol_20_std = v.rolling(20).std()
    df['volume_zscore'] = (v - vol_20_mean) / (vol_20_std + 1e-9)
    df['volume_percentile'] = v.rolling(20).rank(pct=True) * 100.0

    # Delivery Proxies (Fallback)
    if 'delivery_percent' not in df.columns:
        df['delivery_percent'] = 50.0 + (df['body_pct'] * 0.4)
    if 'delivery_quantity' not in df.columns:
        df['delivery_quantity'] = v * (df['delivery_percent'] / 100.0)

    # =========================================================
    # 5. PATTERN ANALYZER HOOKS (SOVEREIGN L1 GEOMETRY ENGINE)
    # =========================================================
    try:
        pattern_features = calculate_patterns(df)
        overlap_cols = [col for col in pattern_features.columns if col in df.columns]
        if overlap_cols:
            df.drop(columns=overlap_cols, inplace=True)
        df = pd.concat([df, pattern_features], axis=1)
    except Exception as e:
        print(f"⚠️ [Layer-1 Pattern Engine Error]: {e}")
        pass

    # ==============================================================================
    # S&R ANALYZER L1_FEATURES PATCH
    # ==============================================================================
    df['swing_high'] = df['high'].rolling(5, center=True).max().ffill()
    df['swing_low'] = df['low'].rolling(5, center=True).min().ffill()
    df['support_strength'] = (df['volume'] * (df['close'] <= df['low'].rolling(10).min())).rolling(10).mean()
    df['resistance_strength'] = (df['volume'] * (df['close'] >= df['high'].rolling(10).max())).rolling(10).mean()
    df['compression_pct'] = (df['high'] - df['low']) / (df['atr_14'] + 1e-9)
    df['breakout_pressure'] = (df['volume'] / df['volume'].rolling(20).mean()) * (df['close'].diff().abs() / (df['atr_14'] + 1e-9))
    df['volume_confirmation'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
    
    # Metadata defaults
    df['pattern_family'] = 'None'
    df['market_phase'] = 'Neutral'
    df['pattern_confidence'] = 0.0
    df['market_regime'] = 'Neutral'
    df['neckline'] = 0.0

    # Geometry placeholders
    for col in ['triangle_upper', 'triangle_lower', 'channel_upper', 'channel_lower',
                'channel_width', 'rectangle_upper', 'rectangle_lower', 'rectangle_width']:
        if col not in df.columns:
            df[col] = 0.0

    # Force Inject 31 Mandatory L1_FEATURES
    required_31_features = [
        'high', 'low', 'close', 'volume', 'swing_high', 'swing_low',
        'support_strength', 'resistance_strength', 'triangle_upper',
        'triangle_lower', 'channel_upper', 'channel_lower', 'channel_width',
        'rectangle_upper', 'rectangle_lower', 'rectangle_width', 'neckline',
        'market_phase', 'breakout_pressure', 'compression_pct',
        'pattern_family', 'pattern_confidence', 'trend_direction',
        'trend_strength', 'volume_confirmation', 'market_regime',
        'bos', 'choch', 'order_block', 'fvg', 'liquidity_sweep'
    ]

    for col in required_31_features:
        if col not in df.columns:
            if col in ['market_phase', 'pattern_family', 'market_regime']:
                df[col] = "Neutral"
            else:
                df[col] = 0.0

    df['liquidity_sweep'] = df['liquidity_sweep'].astype(float)
    df['order_block'] = df['order_block'].astype(float)
    df['fvg'] = df.get('fair_value_gap', 0.0).astype(float) # Patch for fvg naming

    # ==============================================================================
    # SMC LAYER-1 INJECTION: REAL DATA FUSION FOR SMARTMONEY ANALYZER
    # ==============================================================================
    try:
        df['bos'] = bos(df)
        df['internal_bos'] = internal_bos(df)
        df['external_bos'] = external_bos(df)
        df['major_bos'] = df['external_bos']
        df['minor_bos'] = df['internal_bos']
        df['choch'] = choch(df)
        df['internal_choch'] = internal_choch(df)
        df['external_choch'] = external_choch(df)
        df['displacement_strength'] = displacement_candle(df).astype(float) * 100.0
        df['protected_high'] = higher_high(df)
        df['protected_low'] = higher_low(df)
        df['swing_failure'] = market_structure_shift(df)
        df['structure_score'] = structure_strength(df)
        df['violation_level'] = df['close']
        
        # Order Blocks
        bull_ob = bullish_order_block(df)
        bear_ob = bearish_order_block(df)
        df['bullish_ob_detected'] = bull_ob['active']
        df['bearish_ob_detected'] = bear_ob['active']
        df['active_order_block'] = bull_ob['active'] | bear_ob['active']
        df['mitigation_block'] = mitigation_block(df)
        df['breaker_block'] = breaker_block(df)
        df['ob_age'] = order_block_age(df)
        df['ob_efficiency'] = order_block_strength(df)
        df['reaction_count'] = mitigated_order_block(df).astype(int)

        # Fair Value Gaps (FVG)
        bull_fvg_data = bullish_fvg(df)
        bear_fvg_data = bearish_fvg(df)
        df['fair_value_gap'] = np.where(bull_fvg_data['active'], 1.0, np.where(bear_fvg_data['active'], -1.0, 0.0))
        df['inverse_fvg'] = inverse_fvg(df)['active']
        df['fvg_width'] = fvg_width(df)
        df['fvg_age'] = fvg_age(df)
        df['fvg_fill_ratio'] = np.where(filled_fvg(df), 1.0, np.where(active_fvg(df), 0.5, 0.0))
        df['imbalance_strength'] = fvg_strength(df)

        # Liquidity & Volume
        df['liquidity_sweep'] = liquidity_sweep(df)
        df['sweep_strength'] = liquidity_score(df)
        df['equal_highs'] = equal_high(df)
        df['equal_lows'] = equal_low(df)
        df['equal_highs_strength'] = df['equal_highs'].astype(float) * 100.0
        df['equal_lows_strength'] = df['equal_lows'].astype(float) * 100.0
        df['internal_liquidity'] = internal_liquidity(df).notna()
        df['external_liquidity'] = external_liquidity(df).notna()
        df['liquidity_pool_strength'] = liquidity_strength(df)
        df['inducement'] = stop_hunt(df)
        df['institutional_volume_score'] = institutional_score(df)

        # Advanced Context
        df['nested_fvg'] = 0.0
        df['fvg_stack'] = 0.0
        df['resting_liquidity'] = df['equal_highs'] | df['equal_lows']
        df['liquidity_void'] = 0.0
        df['liquidity_exhaustion'] = 0.0
        df['liquidity_consumption'] = df['liquidity_sweep'] != 0.0
        df['liquidity_age'] = 10.0
        df['htf_alignment'] = "Neutral"

    except Exception as e:
        logger.error(f"SMC Layer-1 Extraction Failed: {e}")
        # fallback safety initialization if calculation fails (optional handled above)

    # =========================
    # CLEAN
    # =========================
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df = df.ffill().bfill()
    return df
