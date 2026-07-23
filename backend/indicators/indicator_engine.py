"""
GREEN BULL AI
Ultimate Safe & Aligned Indicator Engine (Plugin Mode - Enterprise Logging & Tracing)
Automatically aligns and cushions Series lengths to prevent shape mismatch crashes.
No Database I/O. Runs directly on Memory DataFrame for ultra-low latency.
"""
import time
import logging
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

from backend.indicators.core import moving_average
from backend.indicators.core import momentum
from backend.indicators.core import volume
from backend.indicators.core import volatility
from backend.indicators.core import candle
from backend.indicators.core import pattern
from backend.indicators.core import support_resistance
from backend.indicators.core import statistics
from backend.indicators.core import smart_money

logger = logging.getLogger("IndicatorEngine")


def run(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure Functional Indicator Engine with Full Performance Tracing & Diagnostic Logging.
    """
    overall_start_time = time.perf_counter()

    # 1. Input Validation
    if df is None or df.empty or len(df) < 50:
        logger.warning(
            f"[{symbol}] Input DataFrame rejected. Empty or insufficient rows (Count: {0 if df is None else len(df)})."
        )
        return pd.DataFrame()

    # 2. Anti-Glitch Preparation & Strict Sorting
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df = df[~df.duplicated(subset=["date"], keep="last")]
        df.set_index("date", inplace=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
    else:
        logger.error(f"[{symbol}] No valid Datetime index or 'date' column found.")
        return pd.DataFrame()

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"[{symbol}] Missing mandatory OHLCV columns: {missing_cols}")
        return pd.DataFrame()

    df = df.tail(350)
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")

    features = pd.DataFrame(index=df.index)
    features["symbol"] = symbol
    features["date"] = df.index

    # RAW OHLCV
    features["OPEN"] = df["open"]
    features["HIGH"] = df["high"]
    features["LOW"] = df["low"]
    features["CLOSE"] = df["close"]
    features["VOLUME"] = df["volume"]

    def align(series, name=None):
        if series is None:
            return pd.Series(index=df.index, dtype="float64")
        if isinstance(series, pd.DataFrame):
            return series.reindex(df.index)
        if isinstance(series, pd.Series):
            s = series.reindex(df.index)
        else:
            s = pd.Series(series, index=df.index)
        if name:
            s.name = name
        return s

    # Metrics dictionary to track 9 module timings
    module_timings = {}

    def run_module(module_name, func):
        m_start = time.perf_counter()
        try:
            func()
            elapsed = (time.perf_counter() - m_start) * 1000
            module_timings[module_name] = round(elapsed, 2)
        except Exception as e:
            elapsed = (time.perf_counter() - m_start) * 1000
            module_timings[module_name] = f"FAILED ({round(elapsed, 2)}ms)"
            logger.error(f"[{symbol}] Module '{module_name}' failed: {e}", exc_info=True)

    # ==========================
    # 1. MOVING AVERAGES
    # ==========================
    def _ma():
        features["ALMA_9"] = align(moving_average.alma(df))
        features["DEMA_20"] = align(moving_average.dema(df))
        features["EMA_20"] = align(moving_average.ema(df, 20))
        features["EMA_50"] = align(moving_average.ema(df, 50))
        features["EMA_100"] = align(moving_average.ema(df, 100))
        features["EMA_200"] = align(moving_average.ema(df, 200))
        features["HMA_20"] = align(moving_average.hma(df, 20))
        features["KAMA_10"] = align(moving_average.kama(df))
        features["LSMA_25"] = align(moving_average.lsma(df, 25))
        features["MCGINLEY_14"] = align(moving_average.mcginley_dynamic(df))
        features["SMA_20"] = align(moving_average.sma(df, 20))
        features["SMA_50"] = align(moving_average.sma(df, 50))
        features["SMA_100"] = align(moving_average.sma(df, 100))
        features["SMA_200"] = align(moving_average.sma(df, 200))
        features["SMMA_20"] = align(moving_average.smma(df))
        features["T3_5"] = align(moving_average.t3(df))
        features["TEMA_20"] = align(moving_average.tema(df, 20))
        features["TRIMA_20"] = align(moving_average.trima(df, 20))
        features["VIDYA_9"] = align(moving_average.vidya(df))
        features["VWMA_20"] = align(moving_average.vwma(df))
        features["WMA_20"] = align(moving_average.wma(df, 20))
        features["ZLEMA_20"] = align(moving_average.zlema(df, 20))

    run_module("moving_average", _ma)

    # ==========================
    # 2. MOMENTUM
    # ==========================
    def _momentum():
        features["RSI_14"] = align(momentum.rsi(df))
        features["RSI_SLOPE"] = align(momentum.rsi_slope(df))

        macd = momentum.macd(df)
        features["MACD"] = align(macd["macd"])
        features["MACD_SIGNAL"] = align(macd["signal"])
        features["MACD_HIST"] = align(macd["histogram"])

        adx = momentum.adx(df)
        features["ADX_14"] = align(adx["adx"])
        features["PLUS_DI"] = align(adx["plus_di"])
        features["MINUS_DI"] = align(adx["minus_di"])
        features["DX"] = align(adx["dx"])

        features["ROC_12"] = align(momentum.roc(df))
        features["CCI_20"] = align(momentum.cci(df))
        features["MOM_10"] = align(momentum.momentum(df))
        features["TRIX_18"] = align(momentum.trix(df))

        ppo = momentum.ppo(df)
        features["PPO"] = align(ppo["ppo"])
        features["PPO_SIGNAL"] = align(ppo["signal"])
        features["PPO_HIST"] = align(ppo["histogram"])

        features["DPO_20"] = align(momentum.dpo(df))

        stoch = momentum.stochastic(df)
        features["STOCH_K"] = align(stoch["k"])
        features["STOCH_D"] = align(stoch["d"])

        stoch_rsi = momentum.stochastic_rsi(df)
        features["STOCH_RSI_K"] = align(stoch_rsi["k"])
        features["STOCH_RSI_D"] = align(stoch_rsi["d"])

        features["WILLIAMS_R"] = align(momentum.williams_r(df))
        features["ULTIMATE_OSC"] = align(momentum.ultimate_oscillator(df))

    run_module("momentum", _momentum)

    # ==========================
    # 3. VOLUME
    # ==========================
    def _volume():
        features["AVG_VOLUME_20"] = align(volume.average_volume(df))
        features["ROLLING_VOLUME_20"] = align(volume.rolling_volume(df))
        features["RVOL_20"] = align(volume.relative_volume(df))
        features["VOLUME_RATIO"] = align(volume.volume_ratio(df))
        features["VOL_EMA_20"] = align(volume.volume_ema(df))
        features["VOL_ZSCORE"] = align(volume.volume_zscore(df))
        features["VOL_PERCENTILE"] = align(volume.volume_percentile(df))
        features["VOL_ROC"] = align(volume.volume_roc(df))
        features["VWAP"] = align(volume.vwap(df))
        features["ROLLING_VWAP"] = align(volume.rolling_vwap(df))
        features["VWEMA"] = align(volume.vwema(df))
        features["MFI_14"] = align(volume.money_flow_index(df))
        features["CMF_20"] = align(volume.chaikin_money_flow(df))
        features["ADL"] = align(volume.accumulation_distribution_line(df))
        features["ADOSC"] = align(volume.accumulation_distribution_oscillator(df))
        features["OBV"] = align(volume.obv(df))
        features["OBV_EMA"] = align(volume.obv_ema(df))
        features["OBV_ROC"] = align(volume.obv_roc(df))
        features["FORCE_INDEX"] = align(volume.force_index(df))
        features["EFI_13"] = align(volume.force_index_ema(df))
        features["EOM"] = align(volume.ease_of_movement(df))
        features["SMOOTHED_EOM"] = align(volume.smoothed_eom(df))
        features["EVWMA"] = align(
            volume.elastic_volume_weighted_momentum(df)
        )
        features["VOLUME_OSC"] = align(volume.volume_oscillator(df))

        pvo = volume.percentage_volume_oscillator(df)
        features["PVO"] = align(pvo["pvo"])
        features["PVO_SIGNAL"] = align(pvo["signal"])
        features["PVO_HIST"] = align(pvo["histogram"])

        vmacd = volume.volume_macd(df)
        features["VOL_MACD"] = align(vmacd["macd"])
        features["VOL_MACD_SIGNAL"] = align(vmacd["signal"])
        features["VOL_MACD_HIST"] = align(vmacd["histogram"])

        klinger = volume.klinger_oscillator(df)
        features["KLINGER"] = align(klinger["klinger"])
        features["KLINGER_SIGNAL"] = align(klinger["signal"])
        features["KLINGER_HIST"] = align(klinger["histogram"])

        features["NVI"] = align(volume.negative_volume_index(df))
        features["PVI"] = align(volume.positive_volume_index(df))
        features["VOLUME_TREND"] = align(volume.volume_trend(df))
        features["PVT"] = align(volume.price_volume_trend(df))
        features["SMART_MONEY_INDEX"] = align(volume.smart_money_index(df))
        features["EFFORT_RESULT"] = align(volume.effort_vs_result(df))
        features["STOPPING_VOLUME"] = align(volume.stopping_volume(df))

        ndns = volume.no_demand_no_supply(df)
        features["NO_DEMAND"] = align(ndns["no_demand"])
        features["NO_SUPPLY"] = align(ndns["no_supply"])

        features["BUY_VOLUME"] = align(volume.buy_volume(df))
        features["SELL_VOLUME"] = align(volume.sell_volume(df))
        features["DELTA_VOLUME"] = align(volume.delta_volume(df))
        features["AMIHUD"] = align(volume.amihud_illiquidity(df))
        features["VPIN"] = align(volume.vpin(df))

    run_module("volume", _volume)

    # ==========================
    # 4. VOLATILITY
    # ==========================
    def _volatility():
        features["TR"] = align(volatility.true_range(df))
        features["ATR_14"] = align(volatility.atr(df))
        features["NATR_14"] = align(volatility.natr(df))

        bb = volatility.bollinger_bands(df)
        features["BB_MIDDLE"] = align(bb["middle"])
        features["BB_UPPER"] = align(bb["upper"])
        features["BB_LOWER"] = align(bb["lower"])
        features["BB_WIDTH"] = align(volatility.bollinger_width(df))
        features["BB_PERCENT_B"] = align(volatility.bollinger_percent_b(df))

        dc = volatility.donchian_channel(df)
        features["DONCHIAN_UPPER"] = align(dc["upper"])
        features["DONCHIAN_MIDDLE"] = align(dc["middle"])
        features["DONCHIAN_LOWER"] = align(dc["lower"])

        kc = volatility.keltner_channel(df)
        features["KELTNER_MIDDLE"] = align(kc["middle"])
        features["KELTNER_UPPER"] = align(kc["upper"])
        features["KELTNER_LOWER"] = align(kc["lower"])

        features["HV_21"] = align(volatility.historical_volatility(df))
        features["CHAIKIN_VOL"] = align(volatility.chaikin_volatility(df))
        features["ULCER_INDEX"] = align(volatility.ulcer_index(df))
        features["STD_20"] = align(volatility.standard_deviation(df))
        features["VAR_20"] = align(volatility.variance(df))
        features["MASS_INDEX"] = align(volatility.mass_index(df))
        features["VOLATILITY_RATIO"] = align(volatility.volatility_ratio(df))
        features["ATR_PERCENTILE"] = align(volatility.atr_percentile(df))
        features["BB_SQUEEZE"] = align(volatility.bollinger_squeeze(df))
        features["EXPANSION_INDEX"] = align(volatility.expansion_index(df))
        features["VOLATILITY_OSC"] = align(volatility.volatility_oscillator(df))
        features["ADAPTIVE_ATR"] = align(volatility.adaptive_atr(df))
        features["PARKINSON_VOL"] = align(volatility.parkinson_volatility(df))
        features["GARMAN_KLASS"] = align(volatility.garman_klass_volatility(df))
        features["ROGERS_SATCHELL"] = align(
            volatility.rogers_satchell_volatility(df)
        )
        features["YANG_ZHANG"] = align(volatility.yang_zhang_volatility(df))
        features["CHOPPINESS"] = align(volatility.choppiness_index(df))
        features["VHF"] = align(volatility.vhf(df))
        features["BB_SQZ_MOM"] = align(volatility.bb_squeeze_momentum(df))
        features["ATR_STOP_DIST"] = align(volatility.atr_stop_distance(df))

        vstop = volatility.volatility_stop(df)
        features["VOL_STOP"] = align(vstop["stop_price"])
        features["VOL_STOP_LONG"] = align(vstop["is_long"])

        features["STANDARD_ERROR"] = align(volatility.standard_error(df))
        features["REI"] = align(volatility.rei(df))

        st = volatility.supertrend(df)
        features["SUPERTREND"] = align(st["supertrend"])
        features["SUPERTREND_TREND"] = align(st["trend"])
        features["SUPERTREND_UPPER"] = align(st["upper_band"])
        features["SUPERTREND_LOWER"] = align(st["lower_band"])

    run_module("volatility", _volatility)

    # ==========================
    # 5. CANDLE
    # ==========================
    def _candle():
        features["BODY_SIZE"] = align(candle.body_size(df))
        features["REAL_BODY"] = align(candle.real_body(df))
        features["BODY_PERCENT"] = align(candle.body_percent(df))
        features["BODY_MIDPOINT"] = align(candle.body_midpoint(df))
        features["BODY_RATIO"] = align(candle.body_ratio(df))
        features["BODY_STRENGTH"] = align(candle.body_strength(df))
        features["BODY_POSITION"] = align(candle.body_position(df))
        features["BULLISH_BODY"] = align(candle.bullish_body(df))
        features["BEARISH_BODY"] = align(candle.bearish_body(df))
        features["SMALL_BODY"] = align(candle.small_body(df))
        features["LARGE_BODY"] = align(candle.large_body(df))
        features["BODY_CHANGE"] = align(candle.body_change(df))
        features["BODY_AVG"] = align(candle.body_average(df))
        features["BODY_EXPANSION"] = align(candle.body_expansion(df))
        features["BODY_CONTRACTION"] = align(candle.body_contraction(df))
        features["UPPER_WICK"] = align(candle.upper_wick(df))
        features["LOWER_WICK"] = align(candle.lower_wick(df))
        features["WICK_SIZE"] = align(candle.wick_size(df))
        features["WICK_RATIO"] = align(candle.wick_ratio(df))
        features["WICK_BALANCE"] = align(candle.wick_balance(df))
        features["WICK_STRENGTH"] = align(candle.wick_strength(df))
        features["WICK_PERCENT"] = align(candle.wick_percent(df))
        features["LONG_UPPER_WICK"] = align(candle.long_upper_wick(df))
        features["LONG_LOWER_WICK"] = align(candle.long_lower_wick(df))
        features["SMALL_WICK"] = align(candle.small_wick(df))
        features["CANDLE_RANGE"] = align(candle.candle_range(df))
        features["TRUE_RANGE_CANDLE"] = align(candle.true_range(df))
        features["BODY_TO_RANGE"] = align(candle.body_to_range(df))
        features["RANGE_EXPANSION"] = align(candle.range_expansion(df))
        features["RANGE_CONTRACTION"] = align(candle.range_contraction(df))
        features["AVERAGE_RANGE"] = align(candle.average_range(df))
        features["ROLLING_RANGE"] = align(candle.rolling_range(df))
        features["RANGE_PERCENTILE"] = align(candle.range_percentile(df))
        features["GAP_UP"] = align(candle.gap_up(df))
        features["GAP_DOWN"] = align(candle.gap_down(df))
        features["BREAKAWAY_GAP_UP"] = align(candle.breakaway_gap_up(df))
        features["BREAKAWAY_GAP_DOWN"] = align(candle.breakaway_gap_down(df))
        features["GAP_SIZE"] = align(candle.gap_size(df))
        features["GAP_PERCENT"] = align(candle.gap_percent(df))
        features["GAP_FILL"] = align(candle.gap_fill(df))
        features["INSIDE_GAP"] = align(candle.inside_gap(df))
        features["DOJI"] = align(candle.doji(df))
        features["DRAGONFLY_DOJI"] = align(candle.dragonfly_doji(df))
        features["GRAVESTONE_DOJI"] = align(candle.gravestone_doji(df))
        features["LONG_LEGGED_DOJI"] = align(candle.long_legged_doji(df))
        features["MARUBOZU"] = align(candle.marubozu(df))
        features["BULLISH_MARUBOZU"] = align(candle.bullish_marubozu(df))
        features["BEARISH_MARUBOZU"] = align(candle.bearish_marubozu(df))
        features["SPINNING_TOP"] = align(candle.spinning_top(df))
        features["HIGH_WAVE"] = align(candle.high_wave(df))
        features["HAMMER_SHAPE"] = align(candle.hammer_shape(df))
        features["HANGING_MAN_SHAPE"] = align(candle.hanging_man_shape(df))
        features["INVERTED_HAMMER_SHAPE"] = align(candle.inverted_hammer_shape(df))
        features["SHOOTING_STAR_SHAPE"] = align(candle.shooting_star_shape(df))
        features["BELT_HOLD"] = align(candle.belt_hold(df))
        features["SHAVEN_HEAD"] = align(candle.shaven_head(df))
        features["SHAVEN_BOTTOM"] = align(candle.shaven_bottom(df))
        features["BULL_POWER"] = align(candle.bull_power(df))
        features["BEAR_POWER"] = align(candle.bear_power(df))
        features["BUYING_PRESSURE"] = align(candle.buying_pressure(df))
        features["SELLING_PRESSURE"] = align(candle.selling_pressure(df))
        features["CANDLE_STRENGTH"] = align(candle.candle_strength(df))
        features["DIRECTION_STRENGTH"] = align(candle.direction_strength(df))
        features["DOMINANCE_SCORE"] = align(candle.dominance_score(df))
        features["PRESSURE_SCORE"] = align(candle.pressure_score(df))
        features["BALANCE_SCORE"] = align(candle.balance_score(df))
        features["CLV"] = align(candle.close_location_value(df))
        features["CLOSE_PERCENT"] = align(candle.close_percent(df))
        features["CLOSE_TO_HIGH"] = align(candle.close_to_high(df))
        features["CLOSE_TO_LOW"] = align(candle.close_to_low(df))
        features["CLOSE_POSITION"] = align(candle.close_position(df))
        features["OPEN_LOCATION"] = align(candle.open_location(df))
        features["OPEN_POSITION"] = align(candle.open_position(df))
        features["BULLISH_CANDLE"] = align(candle.bullish_candle(df))
        features["BEARISH_CANDLE"] = align(candle.bearish_candle(df))
        features["NEUTRAL_CANDLE"] = align(candle.neutral_candle(df))
        features["INSIDE_BAR"] = align(candle.inside_bar(df))
        features["OUTSIDE_BAR"] = align(candle.outside_bar(df))
        features["ENGULFING_BODY"] = align(candle.engulfing_body(df))
        features["BODY_OVERLAP"] = align(candle.body_overlap(df))
        features["RANGE_OVERLAP"] = align(candle.range_overlap(df))
        features["EXPANSION_CANDLE"] = align(candle.expansion_candle(df))
        features["COMPRESSION_CANDLE"] = align(candle.compression_candle(df))
        features["IMPULSE_CANDLE"] = align(candle.impulse_candle(df))
        features["INDECISION_CANDLE"] = align(candle.indecision_candle(df))
        features["ROLLING_BODY_MEAN"] = align(candle.rolling_body_mean(df))
        features["ROLLING_BODY_STD"] = align(candle.rolling_body_std(df))
        features["ROLLING_BODY_ZSCORE"] = align(candle.rolling_body_zscore(df))
        features["ROLLING_RANGE_MEAN"] = align(candle.rolling_range_mean(df))
        features["ROLLING_RANGE_STD"] = align(candle.rolling_range_std(df))
        features["ROLLING_WICK_MEAN"] = align(candle.rolling_wick_mean(df))
        features["ROLLING_WICK_STD"] = align(candle.rolling_wick_std(df))
        features["ROLLING_BODY_PERCENTILE"] = align(
            candle.rolling_body_percentile(df)
        )
        features["ROLLING_RANGE_PERCENTILE"] = align(
            candle.rolling_range_percentile(df)
        )
        features["ROLLING_WICK_PERCENTILE"] = align(
            candle.rolling_wick_percentile(df)
        )
        features["INSTITUTIONAL_BODY"] = align(candle.institutional_body(df))
        features["INSTITUTIONAL_WICK"] = align(candle.institutional_wick(df))
        features["INSTITUTIONAL_IMBALANCE"] = align(
            candle.institutional_imbalance(df)
        )
        features["INSTITUTIONAL_PRESSURE"] = align(
            candle.institutional_pressure(df)
        )
        features["ABSORPTION_CANDLE"] = align(candle.absorption_candle(df))
        features["REJECTION_CANDLE"] = align(candle.rejection_candle(df))
        features["ACCEPTANCE_CANDLE"] = align(candle.acceptance_candle(df))
        features["LIQUIDITY_SWEEP_CANDLE"] = align(
            candle.liquidity_sweep_candle(df)
        )
        features["SMART_MONEY_CANDLE"] = align(candle.smart_money_candle(df))

    run_module("candle", _candle)

    # ==========================
    # 6. PATTERN
    # ==========================
    def _pattern():
        nonlocal features
        pat = pattern.calculate_patterns(df)
        pat.columns = [c.upper() for c in pat.columns]
        cols_to_use = pat.columns.difference(features.columns)
        pat = pat.loc[~pat.index.isna()]
        pat = pat.loc[~pat.index.duplicated(keep="last")]
        pat = pat.reindex(features.index)
        features = features.join(pat[cols_to_use], how="left")

    run_module("pattern", _pattern)

    # ==========================
    # 7. SUPPORT & RESISTANCE
    # ==========================
    def _sr():
        piv = support_resistance.classic_pivot_levels(df)
        features["PIVOT"] = align(piv["PP"])
        features["PIVOT_R1"] = align(piv["R1"])
        features["PIVOT_R2"] = align(piv["R2"])
        features["PIVOT_R3"] = align(piv["R3"])
        features["PIVOT_S1"] = align(piv["S1"])
        features["PIVOT_S2"] = align(piv["S2"])
        features["PIVOT_S3"] = align(piv["S3"])

        features["WOODIE_PIVOT"] = align(support_resistance.woodie_pivot(df))
        features["CAMARILLA_PIVOT"] = align(
            support_resistance.camarilla_pivot(df)
        )
        features["DEMARK_PIVOT"] = align(support_resistance.demark_pivot(df))
        features["SWING_HIGH"] = align(support_resistance.swing_high(df))
        features["SWING_LOW"] = align(support_resistance.swing_low(df))
        features["LAST_SWING_HIGH"] = align(
            support_resistance.last_swing_high(df)
        )
        features["LAST_SWING_LOW"] = align(
            support_resistance.last_swing_low(df)
        )
        features["CONFIRMED_SWING_HIGH"] = align(
            support_resistance.confirmed_swing_high(df)
        )
        features["CONFIRMED_SWING_LOW"] = align(
            support_resistance.confirmed_swing_low(df)
        )
        features["HIGHEST_SWING"] = align(support_resistance.highest_swing(df))
        features["LOWEST_SWING"] = align(support_resistance.lowest_swing(df))
        features["HH_BREAKOUT"] = align(
            support_resistance.highest_high_breakout(df)
        )
        features["LL_BREAKDOWN"] = align(
            support_resistance.lowest_low_breakdown(df)
        )
        features["PREV_HIGH_BREAKOUT"] = align(
            support_resistance.previous_high_breakout(df)
        )
        features["PREV_LOW_BREAKDOWN"] = align(
            support_resistance.previous_low_breakdown(df)
        )
        features["DONCHIAN_BREAKOUT"] = align(
            support_resistance.donchian_breakout(df)
        )
        features["RANGE_BREAKOUT"] = align(
            support_resistance.range_breakout(df)
        )
        features["FRACTAL_HIGH"] = align(
            support_resistance.bill_williams_fractal_high(df)
        )
        features["FRACTAL_LOW"] = align(
            support_resistance.bill_williams_fractal_low(df)
        )
        features["FRACTAL_PIVOT"] = align(
            support_resistance.fractal_pivot(df)
        )
        features["FRACTAL_SUPPORT"] = align(
            support_resistance.fractal_support(df)
        )
        features["FRACTAL_RESISTANCE"] = align(
            support_resistance.fractal_resistance(df)
        )

        dc_sr = support_resistance.donchian_channel(df)
        features["SR_DONCHIAN_UPPER"] = align(dc_sr["upper"])
        features["SR_DONCHIAN_MIDDLE"] = align(dc_sr["middle"])
        features["SR_DONCHIAN_LOWER"] = align(dc_sr["lower"])

        features["ROLLING_HIGH"] = align(
            support_resistance.rolling_highest_high(df)
        )
        features["ROLLING_LOW"] = align(
            support_resistance.rolling_lowest_low(df)
        )
        features["DYNAMIC_SUPPORT"] = align(
            support_resistance.dynamic_support(df)
        )
        features["DYNAMIC_RESISTANCE"] = align(
            support_resistance.dynamic_resistance(df)
        )
        features["ADAPTIVE_SUPPORT"] = align(
            support_resistance.adaptive_support(df)
        )
        features["ADAPTIVE_RESISTANCE"] = align(
            support_resistance.adaptive_resistance(df)
        )

        sz = support_resistance.support_zone(df)
        features["SUP_ZONE_UPPER"] = align(sz["zone_upper"])
        features["SUP_ZONE_LOWER"] = align(sz["zone_lower"])

        rz = support_resistance.resistance_zone(df)
        features["RES_ZONE_UPPER"] = align(rz["zone_upper"])
        features["RES_ZONE_LOWER"] = align(rz["zone_lower"])

        features["DEMAND_ZONE"] = align(support_resistance.demand_zone(df))
        features["SUPPLY_ZONE"] = align(support_resistance.supply_zone(df))

        reaction = support_resistance.reaction_zone(df)
        features["REACTION_DEMAND"] = align(reaction["demand"])
        features["REACTION_SUPPLY"] = align(reaction["supply"])

        features["CONGESTION_ZONE"] = align(
            support_resistance.congestion_zone(df)
        )

        bull = support_resistance.bullish_fvg(df)
        features["BULLISH_FVG"] = align(bull["is_fvg"])

        bear = support_resistance.bearish_fvg(df)
        features["BEARISH_FVG"] = align(bear["is_fvg"])

        fvg = support_resistance.mitigated_fvg(df)
        features["FVG_MITIGATED"] = align(fvg["is_mitigated"])
        features["FVG_ACTIVE"] = align(fvg["is_active"])
        features["BPR"] = align(support_resistance.balanced_price_range(df))

        ob = support_resistance.order_blocks(df)
        features["FRESH_OB"] = align(ob["fresh_ob"])
        features["MITIGATED_OB"] = align(ob["mitigated_ob"])
        features["INVALIDATED_OB"] = align(ob["invalidated_ob"])
        features["OB_HIGH"] = align(ob["ob_high"])
        features["OB_LOW"] = align(ob["ob_low"])

        features["BREAKER_BLOCK"] = align(
            support_resistance.breaker_block(df)
        )
        features["FLIP_ZONE"] = align(support_resistance.flip_zone(df))

        smc = support_resistance.smc_structure(df)
        features["BOS_UP"] = align(smc["bos_up"])
        features["BOS_DOWN"] = align(smc["bos_dn"])
        features["CHOCH_UP"] = align(smc["choch_up"])
        features["CHOCH_DOWN"] = align(smc["choch_dn"])
        features["SMC_TREND"] = align(smc["trend"])

        mss = support_resistance.market_structure_shift(df)
        features["MSS_BULLISH"] = align(mss["mss_bullish"])
        features["MSS_BEARISH"] = align(mss["mss_bearish"])

        pdz = support_resistance.premium_discount_zone(df)
        features["PREMIUM_LEVEL"] = align(pdz["premium_level"])
        features["EQUILIBRIUM"] = align(pdz["equilibrium"])
        features["DISCOUNT_LEVEL"] = align(pdz["discount_level"])

        features["BUY_SIDE_LIQUIDITY"] = align(
            support_resistance.buy_side_liquidity(df)
        )
        features["SELL_SIDE_LIQUIDITY"] = align(
            support_resistance.sell_side_liquidity(df)
        )
        features["LIQUIDITY_SWEEP"] = align(
            support_resistance.liquidity_sweep_level(df)
        )
        features["TURTLE_SOUP"] = align(support_resistance.turtle_soup(df))
        features["JUDAS_SWING"] = align(support_resistance.judas_swing(df))
        features["SUP_TOUCHES"] = align(
            support_resistance.support_touch_count(df)
        )
        features["RES_TOUCHES"] = align(
            support_resistance.resistance_touch_count(df)
        )
        features["LEVEL_STRENGTH"] = align(
            support_resistance.level_strength(df)
        )
        features["LEVEL_CONFIDENCE"] = align(
            support_resistance.level_confidence(df)
        )
        features["ZONE_WIDTH"] = align(support_resistance.zone_width(df))
        features["BOUNCE_COUNT"] = align(support_resistance.bounce_count(df))
        features["DISTANCE_TO_SUPPORT"] = align(
            support_resistance.distance_to_support(df)
        )
        features["DISTANCE_TO_RESISTANCE"] = align(
            support_resistance.distance_to_resistance(df)
        )
        features["NEAREST_SUPPORT"] = align(
            support_resistance.nearest_support(df)
        )
        features["NEAREST_RESISTANCE"] = align(
            support_resistance.nearest_resistance(df)
        )
        features["RISK_DISTANCE"] = align(support_resistance.risk_distance(df))
        features["REWARD_DISTANCE"] = align(
            support_resistance.reward_distance(df)
        )
        features["CONFIRMED_BREAKOUT"] = align(
            support_resistance.confirmed_breakout(df)
        )
        features["CONFIRMED_BREAKDOWN"] = align(
            support_resistance.confirmed_breakdown(df)
        )
        features["FALSE_BREAKOUT"] = align(
            support_resistance.false_breakout(df)
        )
        features["FALSE_BREAKDOWN"] = align(
            support_resistance.false_breakdown(df)
        )
        features["RETEST_LEVEL"] = align(support_resistance.retest_level(df))
        features["BREAK_STRENGTH"] = align(
            support_resistance.break_strength(df)
        )
        features["ROLLING_SUPPORT"] = align(
            support_resistance.rolling_support(df)
        )
        features["ROLLING_RESISTANCE"] = align(
            support_resistance.rolling_resistance(df)
        )
        features["SUP_PERCENTILE"] = align(
            support_resistance.support_percentile(df)
        )
        features["RES_PERCENTILE"] = align(
            support_resistance.resistance_percentile(df)
        )
        features["SUP_ZSCORE"] = align(support_resistance.support_zscore(df))
        features["RES_ZSCORE"] = align(support_resistance.resistance_zscore(df))
        features["ATR_SUPPORT"] = align(support_resistance.atr_support(df))
        features["ATR_RESISTANCE"] = align(
            support_resistance.atr_resistance(df)
        )
        features["VOL_SUPPORT"] = align(
            support_resistance.volatility_support(df)
        )
        features["VOL_RESISTANCE"] = align(
            support_resistance.volatility_resistance(df)
        )

        ab = support_resistance.adaptive_breakout(df)
        features["ADAPTIVE_BREAKOUT_UPPER"] = align(ab["breakout_upper"])
        features["ADAPTIVE_BREAKOUT_LOWER"] = align(ab["breakout_lower"])

        daily = support_resistance.daily_levels(df)
        features["DAILY_HIGH"] = align(daily["upper"])
        features["DAILY_MID"] = align(daily["middle"])
        features["DAILY_LOW"] = align(daily["lower"])

        weekly = support_resistance.weekly_levels(df)
        features["WEEKLY_HIGH"] = align(weekly["upper"])
        features["WEEKLY_MID"] = align(weekly["middle"])
        features["WEEKLY_LOW"] = align(weekly["lower"])

        monthly = support_resistance.monthly_levels(df)
        features["MONTHLY_HIGH"] = align(monthly["upper"])
        features["MONTHLY_MID"] = align(monthly["middle"])
        features["MONTHLY_LOW"] = align(monthly["lower"])

        yearly = support_resistance.yearly_levels(df)
        features["YEARLY_HIGH"] = align(yearly["upper"])
        features["YEARLY_MID"] = align(yearly["middle"])
        features["YEARLY_LOW"] = align(yearly["lower"])

        features["MERGED_SUPPORT"] = align(
            support_resistance.merged_support(df)
        )
        features["MERGED_RESISTANCE"] = align(
            support_resistance.merged_resistance(df)
        )
        features["PRICE_CLUSTER"] = align(
            support_resistance.price_cluster(df)
        )
        features["CLUSTER_SUPPORT"] = align(
            support_resistance.cluster_support(df)
        )
        features["CLUSTER_RESISTANCE"] = align(
            support_resistance.cluster_resistance(df)
        )
        features["CLUSTER_DENSITY"] = align(
            support_resistance.cluster_density(df)
        )
        features["CLUSTER_STRENGTH"] = align(
            support_resistance.cluster_strength(df)
        )
        features["MS_HIGH"] = align(
            support_resistance.market_structure_high(df)
        )
        features["MS_LOW"] = align(support_resistance.market_structure_low(df))
        features["STRUCTURE_SUPPORT"] = align(
            support_resistance.structure_support(df)
        )
        features["STRUCTURE_RESISTANCE"] = align(
            support_resistance.structure_resistance(df)
        )
        features["HIGHER_HIGH"] = align(support_resistance.higher_high(df))
        features["HIGHER_LOW"] = align(support_resistance.higher_low(df))
        features["LOWER_HIGH"] = align(support_resistance.lower_high(df))
        features["LOWER_LOW"] = align(support_resistance.lower_low(df))

        avp = support_resistance.adaptive_volume_profile(df)
        features["AVP_POC"] = align(avp["POC"])
        features["AVP_VAH"] = align(avp["VAH"])
        features["AVP_VAL"] = align(avp["VAL"])

        features["SWING_HIGH_VWAP"] = align(
            support_resistance.swing_high_vwap(df)
        )
        features["SWING_LOW_VWAP"] = align(
            support_resistance.swing_low_vwap(df)
        )

        fib = support_resistance.fibonacci_levels(df)
        features["FIB_0"] = align(fib["Fib_0"])
        features["FIB_236"] = align(fib["Fib_236"])
        features["FIB_382"] = align(fib["Fib_382"])
        features["FIB_500"] = align(fib["Fib_500"])
        features["FIB_618"] = align(fib["Fib_618"])
        features["FIB_786"] = align(fib["Fib_786"])
        features["FIB_100"] = align(fib["Fib_100"])
        features["GOLDEN_ZONE_UPPER"] = align(fib["GoldenZone_Upper"])
        features["GOLDEN_ZONE_LOWER"] = align(fib["GoldenZone_Lower"])

        gann = support_resistance.gann_levels(df)
        features["GANN_0_8"] = align(gann["Gann_0_8"])
        features["GANN_1_8"] = align(gann["Gann_1_8"])
        features["GANN_2_8"] = align(gann["Gann_2_8"])
        features["GANN_3_8"] = align(gann["Gann_3_8"])
        features["GANN_4_8"] = align(gann["Gann_4_8"])
        features["GANN_5_8"] = align(gann["Gann_5_8"])
        features["GANN_6_8"] = align(gann["Gann_6_8"])
        features["GANN_7_8"] = align(gann["Gann_7_8"])
        features["GANN_8_8"] = align(gann["Gann_8_8"])

        murrey = support_resistance.murrey_math_levels(df)
        features["MURREY_0_8"] = align(murrey["MM_0_8"])
        features["MURREY_1_8"] = align(murrey["MM_1_8"])
        features["MURREY_2_8"] = align(murrey["MM_2_8"])
        features["MURREY_3_8"] = align(murrey["MM_3_8"])
        features["MURREY_4_8"] = align(murrey["MM_4_8"])
        features["MURREY_5_8"] = align(murrey["MM_5_8"])
        features["MURREY_6_8"] = align(murrey["MM_6_8"])
        features["MURREY_7_8"] = align(murrey["MM_7_8"])
        features["MURREY_8_8"] = align(murrey["MM_8_8"])

        features["PSYCHOLOGICAL_LEVEL"] = align(
            support_resistance.psychological_levels(df)
        )

        gap = support_resistance.gap_levels(df)
        features["GAP_SUPPORT"] = align(gap["gap_support"])
        features["GAP_RESISTANCE"] = align(gap["gap_resistance"])

        prev = support_resistance.previous_session_levels(df)
        features["PREV_DAY_HIGH"] = align(prev["PDH"])
        features["PREV_DAY_LOW"] = align(prev["PDL"])

        features["INST_CONFLUENCE"] = align(
            support_resistance.institutional_confluence_score(df)
        )
        features["MTF_CONFLUENCE"] = align(
            support_resistance.mtf_confluence_score(df)
        )
        features["LEVEL_RANK"] = align(
            support_resistance.level_reliability_ranking(df)
        )

        features["LIQUIDITY_POOL"] = align(
            support_resistance.liquidity_pool(df)
        )
        features["INSTITUTIONAL_SUPPORT"] = align(
            support_resistance.institutional_support(df)
        )
        features["INSTITUTIONAL_RESISTANCE"] = align(
            support_resistance.institutional_resistance(df)
        )
        features["SMART_MONEY_LEVEL"] = align(
            support_resistance.smart_money_level(df)
        )
        features["STOP_HUNT_ZONE"] = align(
            support_resistance.stop_hunt_zone(df)
        )
        features["INTERNAL_LIQUIDITY"] = align(
            support_resistance.internal_liquidity(df)
        )
        features["EXTERNAL_LIQUIDITY"] = align(
            support_resistance.external_liquidity(df)
        )

        eqh = support_resistance.equal_highs(df)
        features["EQUAL_HIGHS"] = align(eqh["is_equal"])

        eql = support_resistance.equal_lows(df)
        features["EQUAL_LOWS"] = align(eql["is_equal"])

    run_module("support_resistance", _sr)

    # ==========================
    # 8. SMART MONEY
    # ==========================
    def _smart_money():
        sm = smart_money
        features["SM_SWING_HIGH"] = align(sm.swing_high(df))
        features["SM_SWING_LOW"] = align(sm.swing_low(df))
        features["SM_HIGHER_HIGH"] = align(sm.higher_high(df))
        features["SM_HIGHER_LOW"] = align(sm.higher_low(df))
        features["SM_LOWER_HIGH"] = align(sm.lower_high(df))
        features["SM_LOWER_LOW"] = align(sm.lower_low(df))
        features["SM_BOS"] = align(sm.bos(df))
        features["SM_CHOCH"] = align(sm.choch(df))
        features["SM_MSS"] = align(sm.market_structure_shift(df))
        features["SM_BUY_SIDE_LIQUIDITY"] = align(sm.buy_side_liquidity(df))
        features["SM_SELL_SIDE_LIQUIDITY"] = align(sm.sell_side_liquidity(df))
        features["SM_FRESH_OB"] = align(sm.fresh_order_block(df))
        features["SM_MITIGATED_OB"] = align(sm.mitigated_order_block(df))
        features["SM_ACTIVE_FVG"] = align(sm.active_fvg(df))
        features["SM_MITIGATED_FVG"] = align(sm.mitigated_fvg(df))
        features["SM_PREMIUM_ZONE"] = align(sm.premium_zone(df))
        features["SM_DISCOUNT_ZONE"] = align(sm.discount_zone(df))
        features["SM_EQUILIBRIUM"] = align(sm.equilibrium(df))
        features["SM_BOS_SCORE"] = align(sm.bos_score(df))
        features["SM_CHOCH_SCORE"] = align(sm.choch_score(df))
        features["SM_LIQUIDITY_SCORE"] = align(sm.liquidity_score(df))
        features["SM_FVG_SCORE"] = align(sm.fvg_score(df))
        features["SM_TREND_SCORE"] = align(sm.trend_score(df))
        features["SM_INSTITUTIONAL_SCORE"] = align(sm.institutional_score(df))
        features["SM_SMART_MONEY_SCORE"] = align(sm.smart_money_score(df))

    run_module("smart_money", _smart_money)

    # ==========================
    # 9. STATISTICS
    # ==========================
    def _stats():
        features["STAT_MEAN"] = align(statistics.mean(df))
        features["STAT_ROLLING_MEAN"] = align(statistics.rolling_mean(df))
        features["STAT_MEDIAN"] = align(statistics.median(df))
        features["STAT_ROLLING_MEDIAN"] = align(statistics.rolling_median(df))
        features["STAT_MODE"] = align(statistics.mode(df))
        features["STAT_VARIANCE"] = align(statistics.variance(df))
        features["STAT_ROLLING_VARIANCE"] = align(
            statistics.rolling_variance(df)
        )
        features["STAT_STD"] = align(statistics.standard_deviation(df))
        features["STAT_ROLLING_STD"] = align(
            statistics.rolling_standard_deviation(df)
        )
        features["STAT_MAD"] = align(statistics.mean_absolute_deviation(df))
        features["STAT_MEDIAN_MAD"] = align(
            statistics.median_absolute_deviation(df)
        )
        features["STAT_RMS"] = align(statistics.root_mean_square(df))
        features["STAT_CV"] = align(statistics.coefficient_of_variation(df))
        features["STAT_RANGE"] = align(statistics.range_stat(df))
        features["STAT_IQR"] = align(statistics.interquartile_range(df))
        features["STAT_QUANTILE"] = align(statistics.quantile(df))
        features["STAT_PERCENTILE"] = align(statistics.percentile(df))
        features["STAT_ROLLING_PERCENTILE"] = align(
            statistics.rolling_percentile(df)
        )
        features["STAT_PERCENTILE_RANK"] = align(statistics.percentile_rank(df))
        features["STAT_ZSCORE"] = align(statistics.z_score(df))
        features["STAT_ROLLING_ZSCORE"] = align(statistics.rolling_z_score(df))
        features["STAT_MODIFIED_ZSCORE"] = align(
            statistics.modified_z_score(df)
        )
        features["STAT_MINMAX"] = align(statistics.min_max_scaling(df))
        features["STAT_NORMALIZED"] = align(statistics.normalization(df))
        features["STAT_ROBUST_SCALE"] = align(statistics.robust_scaling(df))
        features["STAT_WINSORIZED"] = align(statistics.winsorization(df))
        features["STAT_SKEWNESS"] = align(statistics.skewness(df))
        features["STAT_ROLLING_SKEWNESS"] = align(
            statistics.rolling_skewness(df)
        )
        features["STAT_KURTOSIS"] = align(statistics.kurtosis(df))
        features["STAT_ROLLING_KURTOSIS"] = align(
            statistics.rolling_kurtosis(df)
        )
        features["STAT_ENTROPY"] = align(statistics.entropy(df))
        features["STAT_SHANNON_ENTROPY"] = align(statistics.shannon_entropy(df))
        features["STAT_JARQUE_BERA"] = align(statistics.jarque_bera(df))
        features["STAT_NORMALITY"] = align(statistics.normality_score(df))
        features["STAT_AUTOCORR"] = align(statistics.autocorrelation(df))

        reg = statistics.rolling_regression(df)
        features["REGRESSION_VALUE"] = align(reg["value"])
        features["REGRESSION_SLOPE"] = align(reg["slope"])
        features["REGRESSION_INTERCEPT"] = align(reg["intercept"])
        features["REGRESSION_R2"] = align(reg["r2"])

        lr = statistics.linear_regression(df)
        features["LINEAR_REGRESSION_VALUE"] = align(lr["value"])
        features["LINEAR_REGRESSION_SLOPE"] = align(lr["slope"])
        features["LINEAR_REGRESSION_INTERCEPT"] = align(lr["intercept"])
        features["LINEAR_REGRESSION_R2"] = align(lr["r2"])
        features["REGRESSION_LINE"] = align(statistics.regression_line(df))
        features["REGRESSION_VALUE_ONLY"] = align(statistics.regression_value(df))
        features["REGRESSION_INTERCEPT_ONLY"] = align(
            statistics.regression_intercept(df)
        )
        features["REGRESSION_SLOPE_ONLY"] = align(
            statistics.regression_slope(df)
        )
        features["ROLLING_SLOPE"] = align(statistics.rolling_slope(df))
        features["R_SQUARED"] = align(statistics.r_squared(df))
        features["ADJUSTED_R_SQUARED"] = align(
            statistics.adjusted_r_squared(df)
        )
        features["REGRESSION_RESIDUAL"] = align(statistics.residual(df))
        features["REGRESSION_RSE"] = align(
            statistics.residual_standard_error(df)
        )

        rc = statistics.regression_channel(df)
        features["REG_CHANNEL_MIDDLE"] = align(rc["middle"])
        features["REG_CHANNEL_UPPER"] = align(rc["upper"])
        features["REG_CHANNEL_LOWER"] = align(rc["lower"])

        features["OUTLIER_DETECTION"] = align(statistics.outlier_detection(df))
        features["THREE_SIGMA_RULE"] = align(statistics.three_sigma_rule(df))
        features["MODIFIED_Z_OUTLIER"] = align(
            statistics.modified_z_outlier(df)
        )
        features["LINEAR_TREND_STRENGTH"] = align(
            statistics.linear_trend_strength(df)
        )
        features["TREND_ANGLE"] = align(statistics.trend_angle(df))
        features["SLOPE_PERCENTAGE"] = align(statistics.slope_percentage(df))
        features["COEFF_DISPERSION"] = align(
            statistics.coefficient_of_dispersion(df)
        )
        features["RELATIVE_STD_DEV"] = align(
            statistics.relative_standard_deviation(df)
        )

        features["PEARSON_CORR"] = align(statistics.pearson_correlation(df, df))
        features["ROLLING_PEARSON"] = align(
            statistics.rolling_pearson(df, df)
        )
        features["LAG_CORRELATION"] = align(statistics.lag_correlation(df, df))
        features["CROSS_CORRELATION"] = align(
            statistics.cross_correlation(df, df)
        )
        features["COVARIANCE"] = align(statistics.covariance(df, df))
        features["ROLLING_COVARIANCE"] = align(
            statistics.rolling_covariance(df, df)
        )
        features["BETA"] = align(statistics.beta(df, df))
        features["ROLLING_BETA"] = align(statistics.rolling_beta(df, df))
        features["ALPHA"] = align(statistics.alpha(df, df))
        features["ROLLING_ALPHA"] = align(statistics.rolling_alpha(df, df))
        features["TRACKING_ERROR"] = align(statistics.tracking_error(df, df))
        features["INFORMATION_RATIO"] = align(
            statistics.information_ratio(df, df)
        )

    run_module("statistics", _stats)

    # Post-Processing & Quality Diagnostics
    features = features.loc[~features.index.isna()].copy()
    features = features.loc[~features.index.duplicated(keep="last")]
    features.sort_index(inplace=True)
    features["date"] = features.index

    # Quality Calculations (Last row inspection)
    latest_row = features.iloc[-1]
    total_features = len(features.columns) - 2  # Exclude 'symbol' and 'date'
    null_count = latest_row.isna().sum()
    valid_count = total_features - null_count
    health_percentage = round((valid_count / total_features) * 100, 2)

    overall_time_ms = round((time.perf_counter() - overall_start_time) * 1000, 2)

    # Print Tracing Logs
    logger.info(
        f"[{symbol}] Range: {start_date} -> {end_date} | Total Time: {overall_time_ms}ms"
    )
    logger.info(
        f"[{symbol}] Features: Total={total_features} | Success={valid_count} | Null/Failed={null_count} | Health={health_percentage}%"
    )
    logger.info(
        f"[{symbol}] Module Timings (ms): {module_timings}"
    )

    return features
