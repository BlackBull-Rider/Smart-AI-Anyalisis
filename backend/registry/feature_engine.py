import pandas as pd
import numpy as np

# ==============================================================================
# EXISTING IMPORTS
# ==============================================================================
from backend.indicators.core.moving_average import ema
from backend.indicators.core.momentum import (
    rsi,
    adx,
    macd,
    roc,
    momentum,
)
from backend.indicators.core.volatility import (
    atr,
    supertrend,
)
from backend.indicators.core.statistics import (
    linear_regression,
)
from backend.indicators.core.volume import (
    vwap,
)

from backend.indicators.core.pattern import calculate_patterns 


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
        bos, choch, liquidity_sweep, fair_value_gap, mitigation, order_block, breaker
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
        df["macd_line"] = (
            macd_df["macd"] if "macd" in macd_df.columns else macd_df.iloc[:, 0]
        )
        df["macd_signal"] = (
            macd_df["signal"] if "signal" in macd_df.columns else macd_df.iloc[:, 1]
        )
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
        df["linreg_slope"] = (
            lr["slope"] if "slope" in lr.columns else lr.iloc[:, 0]
        )
        df["linreg_r2"] = (
            lr["r2"] if "r2" in lr.columns else lr.iloc[:, 1]
        )

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

        swing_high = (h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2))
        swing_low = (l < l.shift(1)) & (l < l.shift(2)) & (l < l.shift(-1)) & (l < l.shift(-2))
        sh_val = h.where(swing_high).ffill()
        sl_val = l.where(swing_low).ffill()

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
    # কলিং দ্য প্রোডাকশন-গ্রেড প্যাটার্ন ডিটেক্টর
    
    try:
        # প্যাটার্ন ইঞ্জিন থেকে ৩৯টি পিওর জ্যামিতিক কলাম নিয়ে আসা হচ্ছে
        pattern_features = calculate_patterns(df)
        
        # সেফটি চেক: যদি df-এ আগে থেকেই কোনো প্যাটার্ন কলাম ইনিশিয়ালাইজ করা থাকে, 
        # তবে সেগুলো ড্রপ করে দিচ্ছি যাতে মার্জ করার সময় কলাম ডুপ্লিকেট (যেমন: _x, _y) না হয়।
        overlap_cols = [col for col in pattern_features.columns if col in df.columns]
        if overlap_cols:
            df.drop(columns=overlap_cols, inplace=True)
            
        # মেইন ডেটাফ্রেমের সাথে নতুন প্যাটার্ন কলামগুলো নিখুঁতভাবে জুড়ে দেওয়া হলো (Vectorized Concat)
        df = pd.concat([df, pattern_features], axis=1)
        
    except Exception as e:
        print(f"⚠️ [Layer-1 Pattern Engine Error]: {e}")
        # যদি কোনো কারণে ইঞ্জিন ক্র্যাশ করে (যেটা হওয়ার কথা না), তবে পাইপলাইন যেন না থামে
        pass

    # =========================
    # CLEAN
    # =========================
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df = df.ffill().bfill()

    return df
