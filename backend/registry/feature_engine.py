"""
GREEN BULL RIDER V6
Feature Engine

This module is responsible ONLY for building the feature dataframe.

Pipeline
    ↓
Feature Engine
    ↓
Indicator Libraries
    ↓
Enriched DataFrame
"""

from __future__ import annotations

import logging
import pandas as pd

from backend.indicators.core import (
    moving_average,
    momentum,
    volume,
    volatility,
    statistics,
    candle,
    support_resistance,
    smart_money,
    pattern,
)

logger = logging.getLogger(__name__)


class FeatureEngine:

    def __init__(self):

        logger.info(
            "Feature Engine initialized."
        )

    def build(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df is None:
            raise ValueError("Input dataframe is None.")

        if df.empty:
            raise ValueError("Input dataframe is empty.")

        required = (
            "open",
            "high",
            "low",
            "close",
            "volume",
        )

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise KeyError(
                f"Missing columns : {missing}"
            )

        logger.info(
            "Starting Feature Engine..."
        )

        ####################################################################
        # MOVING AVERAGE
        ####################################################################

        #
        # EMA
        #
        df["EMA_5"] = moving_average.ema(df, 5)
        df["EMA_9"] = moving_average.ema(df, 9)
        df["EMA_10"] = moving_average.ema(df, 10)
        df["EMA_20"] = moving_average.ema(df, 20)
        df["EMA_21"] = moving_average.ema(df, 21)
        df["EMA_34"] = moving_average.ema(df, 34)
        df["EMA_50"] = moving_average.ema(df, 50)
        df["EMA_55"] = moving_average.ema(df, 55)
        df["EMA_100"] = moving_average.ema(df, 100)
        df["EMA_144"] = moving_average.ema(df, 144)
        df["EMA_200"] = moving_average.ema(df, 200)

        #
        # SMA
        #
        df["SMA_5"] = moving_average.sma(df, 5)
        df["SMA_10"] = moving_average.sma(df, 10)
        df["SMA_20"] = moving_average.sma(df, 20)
        df["SMA_50"] = moving_average.sma(df, 50)
        df["SMA_100"] = moving_average.sma(df, 100)
        df["SMA_200"] = moving_average.sma(df, 200)

        #
        # WMA
        #
        df["WMA_20"] = moving_average.wma(df, 20)
        df["WMA_50"] = moving_average.wma(df, 50)

        #
        # HMA
        #
        df["HMA_20"] = moving_average.hma(df, 20)
        df["HMA_50"] = moving_average.hma(df, 50)


        print("DF index type:", type(df.index))
        print("DF index:", df.index[:5])

        print("Close index:", df["close"].index[:5])
        print("Volume index:", df["volume"].index[:5])

        print("Equal before VWMA:", df["close"].index.equals(df["volume"].index))
        #
        # VWMA
        #
        df["VWMA_20"] = moving_average.vwma(df, length=20)
        #
        # DEMA
        #
        df["DEMA_20"] = moving_average.dema(df, 20)

        #
        # TEMA
        #
        df["TEMA_20"] = moving_average.tema(df, 20)

        #
        # TRIMA
        #
        df["TRIMA_20"] = moving_average.trima(df, 20)

        #
        # KAMA
        #
        df["KAMA_10"] = moving_average.kama(df, 10)

        #
        # ALMA
        #
        df["ALMA_20"] = moving_average.alma(df, 20)

        #
        # ZLEMA
        #
        df["ZLEMA_20"] = moving_average.zlema(df, 20)

        #
        # McGinley Dynamic
        #
        df["MCGINLEY_20"] = moving_average.mcginley_dynamic(
          df,
          length=20,
        )
        logger.info(
            "Moving Average completed."
        )

        ####################################################################
        # MOMENTUM
        ####################################################################

        #
        # RSI
        #
        df["RSI_14"] = momentum.rsi(df, 14)

        #
        # MACD
        #
        macd = momentum.macd(df)

        df["MACD"] = macd["MACD"]
        df["MACD_SIGNAL"] = macd["Signal"]
        df["MACD_HIST"] = macd["Histogram"]

        #
        # ADX
        #
        adx = momentum.adx(df, 14)

        df["ADX_14"] = adx["ADX"]
        df["PLUS_DI"] = adx["+DI"]
        df["MINUS_DI"] = adx["-DI"]

        #
        # ROC
        #
        df["ROC_12"] = momentum.roc(df, 12)

        #
        # CCI
        #
        df["CCI_20"] = momentum.cci(df, 20)

        #
        # MOMENTUM
        #
        df["MOM_10"] = momentum.momentum(df, 10)

        #
        # TRIX
        #
        df["TRIX_18"] = momentum.trix(df, 18)

        #
        # DPO
        #
        df["DPO_20"] = momentum.dpo(df, 20)

        #
        # STOCHASTIC
        #
        stoch = momentum.stochastic(df)

        df["STOCH_K"] = stoch["%K"]
        df["STOCH_D"] = stoch["%D"]

        #
        # WILLIAMS %R
        #
        df["WILLIAMS_R"] = momentum.williams_r(df, 14)

        #
        # ULTIMATE OSCILLATOR
        #
        df["ULTIMATE_OSC"] = momentum.ultimate_oscillator(df)

        logger.info(
            "Momentum completed."
        )

        ####################################################################
        # VOLUME
        ####################################################################

        #
        # OBV
        #
        df["OBV"] = volume.obv(df)

        #
        # VWAP
        #
        df["VWAP"] = volume.vwap(df)

        #
        # ADL
        #
        df["ADL"] = volume.adl(df)

        #
        # CMF
        #
        df["CMF_20"] = volume.cmf(df, 20)

        #
        # MFI
        #
        df["MFI_14"] = volume.mfi(df, 14)

        #
        # FORCE INDEX
        #
        df["FORCE_INDEX"] = volume.force_index(df)

        #
        # EASE OF MOVEMENT
        #
        df["EOM_14"] = volume.ease_of_movement(df, 14)

        #
        # PVO
        #
        pvo = volume.pvo(df)

        df["PVO"] = pvo["PVO"]
        df["PVO_SIGNAL"] = pvo["Signal"]
        df["PVO_HIST"] = pvo["Histogram"]

        #
        # KLINGER
        #
        klinger = volume.klinger(df)

        df["KLINGER"] = klinger["KVO"]
        df["KLINGER_SIGNAL"] = klinger["Signal"]

        #
        # VOLUME PROFILE
        #
        profile = volume.volume_profile(df)

        df["POC"] = profile["POC"]
        df["VAH"] = profile["VAH"]
        df["VAL"] = profile["VAL"]
        df["HVN"] = profile["HVN"]
        df["LVN"] = profile["LVN"]

        logger.info(
            "Volume completed."
        )

        ####################################################################
        # VOLATILITY
        ####################################################################

        #
        # TRUE RANGE
        #
        df["TR"] = volatility.true_range(df)

        #
        # ATR
        #
        df["ATR_14"] = volatility.atr(df, 14)

        #
        # NATR
        #
        df["NATR_14"] = volatility.natr(df, 14)

        #
        # BOLLINGER BANDS
        #
        bb = volatility.bollinger_bands(df)

        df["BB_MIDDLE"] = bb["Middle"]
        df["BB_UPPER"] = bb["Upper"]
        df["BB_LOWER"] = bb["Lower"]

        #
        # DONCHIAN CHANNEL
        #
        dc = volatility.donchian_channel(df)

        df["DONCHIAN_UPPER"] = dc["Upper"]
        df["DONCHIAN_MIDDLE"] = dc["Middle"]
        df["DONCHIAN_LOWER"] = dc["Lower"]

        #
        # KELTNER CHANNEL
        #
        kc = volatility.keltner_channel(df)

        df["KC_UPPER"] = kc["Upper"]
        df["KC_MIDDLE"] = kc["Middle"]
        df["KC_LOWER"] = kc["Lower"]

        #
        # HISTORICAL VOLATILITY
        #
        df["HV"] = volatility.historical_volatility(df)

        #
        # CHAIKIN VOLATILITY
        #
        df["CHAIKIN_VOL"] = volatility.chaikin_volatility(df)

        #
        # ULCER INDEX
        #
        df["ULCER_INDEX"] = volatility.ulcer_index(df)

        #
        # MASS INDEX
        #
        df["MASS_INDEX"] = volatility.mass_index(df)

        #
        # VOLATILITY RATIO
        #
        df["VOLATILITY_RATIO"] = volatility.volatility_ratio(df)

        #
        # CHOPPINESS INDEX
        #
        df["CHOP"] = volatility.choppiness_index(df)

        #
        # VHF
        #
        df["VHF"] = volatility.vertical_horizontal_filter(df)

        #
        # RANGE EXPANSION INDEX
        #
        df["REI"] = volatility.range_expansion_index(df)

        logger.info(
            "Volatility completed."
        )

        ####################################################################
        # STATISTICS
        ####################################################################

        #
        # MEAN
        #
        df["STAT_MEAN"] = statistics.mean(df)

        #
        # MEDIAN
        #
        df["STAT_MEDIAN"] = statistics.median(df)

        #
        # MODE
        #
        df["STAT_MODE"] = statistics.mode(df)

        #
        # VARIANCE
        #
        df["STAT_VARIANCE"] = statistics.variance(df)

        #
        # STANDARD DEVIATION
        #
        df["STAT_STD"] = statistics.standard_deviation(df)

        #
        # Z SCORE
        #
        df["STAT_ZSCORE"] = statistics.z_score(df)

        #
        # SKEWNESS
        #
        df["STAT_SKEWNESS"] = statistics.skewness(df)

        #
        # KURTOSIS
        #
        df["STAT_KURTOSIS"] = statistics.kurtosis(df)

        #
        # COEFFICIENT OF VARIATION
        #
        df["STAT_CV"] = statistics.coefficient_of_variation(df)

        #
        # ENTROPY
        #
        df["STAT_ENTROPY"] = statistics.entropy(df)

        #
        # LINEAR REGRESSION
        #
        lr = statistics.linear_regression(df)

        df["LR_SLOPE"] = lr["Slope"]
        df["LR_INTERCEPT"] = lr["Intercept"]
        df["LR_R2"] = lr["R2"]
        df["LR_VALUE"] = lr["Regression"]

        #
        # ROLLING LINEAR REGRESSION
        #
        rlr = statistics.rolling_regression(df)

        df["ROLL_SLOPE"] = rlr["Slope"]
        df["ROLL_INTERCEPT"] = rlr["Intercept"]
        df["ROLL_R2"] = rlr["R2"]
        df["ROLL_REG"] = rlr["Regression"]

        #
        # ROLLING MEAN
        #
        df["ROLL_MEAN"] = statistics.rolling_mean(df)

        #
        # ROLLING STD
        #
        df["ROLL_STD"] = statistics.rolling_standard_deviation(df)

        #
        # ROLLING VARIANCE
        #
        df["ROLL_VARIANCE"] = statistics.rolling_variance(df)

        logger.info(
            "Statistics completed."
        )

        ####################################################################
        # CANDLE
        ####################################################################

        #
        # BODY
        #
        df["BODY_SIZE"] = candle.body_size(df)
        df["BODY_PERCENT"] = candle.body_percent(df)
        df["BODY_RATIO"] = candle.body_ratio(df)
        df["BODY_STRENGTH"] = candle.body_strength(df)
        df["BODY_POSITION"] = candle.body_position(df)

        #
        # BODY TYPE
        #
        df["BULLISH_BODY"] = candle.bullish_body(df)
        df["BEARISH_BODY"] = candle.bearish_body(df)
        df["SMALL_BODY"] = candle.small_body(df)
        df["LARGE_BODY"] = candle.large_body(df)

        #
        # WICKS
        #
        df["UPPER_WICK"] = candle.upper_wick(df)
        df["LOWER_WICK"] = candle.lower_wick(df)
        df["UPPER_WICK_RATIO"] = candle.upper_wick_ratio(df)
        df["LOWER_WICK_RATIO"] = candle.lower_wick_ratio(df)

        #
        # RANGE
        #
        df["CANDLE_RANGE"] = candle.candle_range(df)
        df["TRUE_BODY"] = candle.real_body(df)

        #
        # PATTERNS
        #
        df["DOJI"] = candle.doji(df)
        df["HAMMER"] = candle.hammer(df)
        df["HANGING_MAN"] = candle.hanging_man(df)
        df["SHOOTING_STAR"] = candle.shooting_star(df)
        df["INVERTED_HAMMER"] = candle.inverted_hammer(df)
        df["MARUBOZU"] = candle.marubozu(df)
        df["SPINNING_TOP"] = candle.spinning_top(df)

        #
        # ENGULFING
        #
        df["BULLISH_ENGULFING"] = candle.bullish_engulfing(df)
        df["BEARISH_ENGULFING"] = candle.bearish_engulfing(df)

        #
        # MORNING / EVENING STAR
        #
        df["MORNING_STAR"] = candle.morning_star(df)
        df["EVENING_STAR"] = candle.evening_star(df)

        logger.info(
            "Candle completed."
        )

        ####################################################################
        # SUPPORT / RESISTANCE
        ####################################################################

        #
        # SWING LEVELS
        #
        swings = support_resistance.swing_levels(df)

        df["SWING_HIGH"] = swings["SwingHigh"]
        df["SWING_LOW"] = swings["SwingLow"]
        df["LAST_SWING_HIGH"] = swings["LastSwingHigh"]
        df["LAST_SWING_LOW"] = swings["LastSwingLow"]

        #
        # SUPPORT / RESISTANCE
        #
        df["SUPPORT"] = support_resistance.support(df)
        df["RESISTANCE"] = support_resistance.resistance(df)

        #
        # PRICE CLUSTER
        #
        df["PRICE_CLUSTER"] = support_resistance.price_cluster(df)

        #
        # EQUAL HIGHS / LOWS
        #
        equal = support_resistance.equal_levels(df)

        df["EQUAL_HIGH"] = equal["EqualHigh"]
        df["EQUAL_LOW"] = equal["EqualLow"]
        df["CLUSTER_SIZE"] = equal["ClusterSize"]
        df["CLUSTER_STRENGTH"] = equal["ClusterStrength"]

        #
        # FAIR VALUE GAP
        #
        fvg = support_resistance.fair_value_gap(df)

        df["FVG_TOP"] = fvg["Top"]
        df["FVG_BOTTOM"] = fvg["Bottom"]
        df["FVG_BULLISH"] = fvg["Bullish"]
        df["FVG_MITIGATED"] = fvg["Mitigated"]

        #
        # ORDER BLOCK
        #
        ob = support_resistance.order_block(df)

        df["OB_HIGH"] = ob["High"]
        df["OB_LOW"] = ob["Low"]
        df["OB_FRESH"] = ob["Fresh"]
        df["OB_MITIGATED"] = ob["Mitigated"]
        df["OB_INVALIDATED"] = ob["Invalidated"]

        #
        # VOLUME PROFILE
        #
        vp = support_resistance.volume_profile(df)

        df["POC_SR"] = vp["POC"]
        df["VAH_SR"] = vp["VAH"]
        df["VAL_SR"] = vp["VAL"]

        logger.info(
            "Support / Resistance completed."
        )

        ####################################################################
        # SMART MONEY
        ####################################################################

        #
        # MARKET STRUCTURE
        #
        structure = smart_money.market_structure(df)

        df["SWING_HIGH_SM"] = structure["SwingHigh"]
        df["SWING_LOW_SM"] = structure["SwingLow"]
        df["HH"] = structure["HH"]
        df["HL"] = structure["HL"]
        df["LH"] = structure["LH"]
        df["LL"] = structure["LL"]
        df["BOS"] = structure["BOS"]
        df["CHOCH"] = structure["CHOCH"]
        df["TREND"] = structure["Trend"]

        #
        # LIQUIDITY
        #
        liquidity = smart_money.liquidity(df)

        df["BUY_SIDE_LIQUIDITY"] = liquidity["BSL"]
        df["SELL_SIDE_LIQUIDITY"] = liquidity["SSL"]
        df["EQUAL_HIGHS"] = liquidity["EQH"]
        df["EQUAL_LOWS"] = liquidity["EQL"]
        df["BSL_SWEEP"] = liquidity["BSL_Sweep"]
        df["SSL_SWEEP"] = liquidity["SSL_Sweep"]

        #
        # ORDER BLOCK
        #
        sm_ob = smart_money.order_blocks(df)

        df["SM_OB_HIGH"] = sm_ob["High"]
        df["SM_OB_LOW"] = sm_ob["Low"]
        df["SM_OB_STATE"] = sm_ob["State"]

        #
        # FAIR VALUE GAP
        #
        sm_fvg = smart_money.fair_value_gap(df)

        df["SM_FVG_TOP"] = sm_fvg["Top"]
        df["SM_FVG_BOTTOM"] = sm_fvg["Bottom"]
        df["SM_FVG_STATE"] = sm_fvg["State"]

        #
        # OPTIMAL TRADE ENTRY
        #
        ote = smart_money.optimal_trade_entry(df)

        df["OTE_LOW"] = ote["Low"]
        df["OTE_HIGH"] = ote["High"]
        df["OTE_ACTIVE"] = ote["Active"]

        logger.info(
            "Smart Money completed."
        )

        ####################################################################
        # PATTERN
        ####################################################################

        pattern_df = pattern.calculate_patterns(df)

        for column in pattern_df.columns:
            if column not in df.columns:
                df[column] = pattern_df[column]
            else:
                df[column] = pattern_df[column]

        logger.info(
            "Pattern completed."
        )

        logger.info(
            "Feature Engine completed successfully."
        )

        return df


feature_engine = FeatureEngine()


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return feature_engine.build(df)

