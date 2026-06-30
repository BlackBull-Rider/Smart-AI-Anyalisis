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

    # === CRITICAL FIX: CLV HOOK ===
    # Close Location Value (-1 to +1)
    df['clv'] = ((c - l) - (h - c)) / c_range

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
        abs_body = (c - o).abs()
        df['doji'] = ((abs_body / c_range) < 0.1).astype(int)
        df['inside_bar'] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)

    # =========================
    # GAPS
    # =========================
    prev_h = df['high'].shift(1)
    prev_l = df['low'].shift(1)
    df['gap_up'] = (df['open'] > prev_h).astype(int)
    df['gap_down'] = (df['open'] < prev_l).astype(int)

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

    # =========================
    # CLEAN
    # =========================
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df = df.ffill().bfill()

    return df
