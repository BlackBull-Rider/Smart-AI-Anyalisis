"""
Moving Average Indicator Engine
"""

import pandas as pd
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)
from backend.data.data_fetcher import fetch_ohlcv
from backend.indicators.core import moving_average
from backend.indicators.core import momentum
from backend.indicators.core import volume
from backend.indicators.core import volatility
from backend.indicators.core import candle
from backend.indicators.core import pattern
from backend.indicators.core import support_resistance
from backend.indicators.core import statistics
from backend.indicators.core import smart_money

def run(symbol: str) -> pd.DataFrame:

    df = fetch_ohlcv(symbol)

    features = pd.DataFrame(index=df.index)

    features["ALMA_9"] = moving_average.alma(df)
    features["DEMA_20"] = moving_average.dema(df)

    features["EMA_20"] = moving_average.ema(df, 20)
    features["EMA_50"] = moving_average.ema(df, 50)
    features["EMA_100"] = moving_average.ema(df, 100)
    features["EMA_200"] = moving_average.ema(df, 200)

    features["HMA_20"] = moving_average.hma(df, 20)
    features["KAMA_10"] = moving_average.kama(df)
    features["LSMA_25"] = moving_average.lsma(df, 25)
    features["MCGINLEY_14"] = moving_average.mcginley_dynamic(df)

    features["SMA_20"] = moving_average.sma(df, 20)
    features["SMA_50"] = moving_average.sma(df, 50)
    features["SMA_100"] = moving_average.sma(df, 100)
    features["SMA_200"] = moving_average.sma(df, 200)

    features["SMMA_20"] = moving_average.smma(df)

    features["T3_5"] = moving_average.t3(df)
    features["TEMA_20"] = moving_average.tema(df, 20)
    features["TRIMA_20"] = moving_average.trima(df, 20)

    features["VIDYA_9"] = moving_average.vidya(df)
    features["VWMA_20"] = moving_average.vwma(df)
    features["WMA_20"] = moving_average.wma(df, 20)
    features["ZLEMA_20"] = moving_average.zlema(df, 20)

    features.insert(0, "date", features.index)
    features.insert(0, "symbol", symbol)

# ==========================
# MOMENTUM
# ==========================

    features["RSI_14"] = momentum.rsi(df)
    features["RSI_SLOPE"] = momentum.rsi_slope(df)

    macd = momentum.macd(df)
    features["MACD"] = macd["macd"]
    features["MACD_SIGNAL"] = macd["signal"]
    features["MACD_HIST"] = macd["histogram"]

    adx = momentum.adx(df)
    features["ADX_14"] = adx["adx"]
    features["PLUS_DI"] = adx["plus_di"]
    features["MINUS_DI"] = adx["minus_di"]
    features["DX"] = adx["dx"]

    features["ROC_12"] = momentum.roc(df)
    features["CCI_20"] = momentum.cci(df)
    features["MOM_10"] = momentum.momentum(df)
    features["TRIX_18"] = momentum.trix(df)

    ppo = momentum.ppo(df)
    features["PPO"] = ppo["ppo"]
    features["PPO_SIGNAL"] = ppo["signal"]
    features["PPO_HIST"] = ppo["histogram"]

    features["DPO_20"] = momentum.dpo(df)

    stoch = momentum.stochastic(df)
    features["STOCH_K"] = stoch["k"]
    features["STOCH_D"] = stoch["d"]

    stoch_rsi = momentum.stochastic_rsi(df)
    features["STOCH_RSI_K"] = stoch_rsi["k"]
    features["STOCH_RSI_D"] = stoch_rsi["d"]

    features["WILLIAMS_R"] = momentum.williams_r(df)
    features["ULTIMATE_OSC"] = momentum.ultimate_oscillator(df)

    # ==========================
    # VOLUME
    # ==========================

    features["VOLUME"] = volume.volume(df)
    features["AVG_VOLUME_20"] = volume.average_volume(df)
    features["ROLLING_VOLUME_20"] = volume.rolling_volume(df)
    features["RVOL_20"] = volume.relative_volume(df)
    features["VOLUME_RATIO"] = volume.volume_ratio(df)

    features["VOL_EMA_20"] = volume.volume_ema(df)
    features["VOL_ZSCORE"] = volume.volume_zscore(df)
    features["VOL_PERCENTILE"] = volume.volume_percentile(df)
    features["VOL_ROC"] = volume.volume_roc(df)

    features["VWAP"] = volume.vwap(df)
    features["ROLLING_VWAP"] = volume.rolling_vwap(df)
    features["VWEMA"] = volume.vwema(df)

    features["MFI_14"] = volume.money_flow_index(df)
    features["CMF_20"] = volume.chaikin_money_flow(df)
    features["ADL"] = volume.accumulation_distribution_line(df)
    features["ADOSC"] = volume.accumulation_distribution_oscillator(df)

    features["OBV"] = volume.obv(df)
    features["OBV_EMA"] = volume.obv_ema(df)
    features["OBV_ROC"] = volume.obv_roc(df)

    features["FORCE_INDEX"] = volume.force_index(df)
    features["EFI_13"] = volume.force_index_ema(df)

    features["EOM"] = volume.ease_of_movement(df)
    features["SMOOTHED_EOM"] = volume.smoothed_eom(df)
    features["EVWMA"] = volume.elastic_volume_weighted_momentum(df)

    features["VOLUME_OSC"] = volume.volume_oscillator(df)

    pvo = volume.percentage_volume_oscillator(df)
    features["PVO"] = pvo["pvo"]
    features["PVO_SIGNAL"] = pvo["signal"]
    features["PVO_HIST"] = pvo["histogram"]

    vmacd = volume.volume_macd(df)
    features["VOL_MACD"] = vmacd["macd"]
    features["VOL_MACD_SIGNAL"] = vmacd["signal"]
    features["VOL_MACD_HIST"] = vmacd["histogram"]

    klinger = volume.klinger_oscillator(df)
    features["KLINGER"] = klinger["klinger"]
    features["KLINGER_SIGNAL"] = klinger["signal"]
    features["KLINGER_HIST"] = klinger["histogram"]

    features["NVI"] = volume.negative_volume_index(df)
    features["PVI"] = volume.positive_volume_index(df)
    features["VOLUME_TREND"] = volume.volume_trend(df)
    features["PVT"] = volume.price_volume_trend(df)
    features["SMART_MONEY_INDEX"] = volume.smart_money_index(df)
    features["EFFORT_RESULT"] = volume.effort_vs_result(df)
    features["STOPPING_VOLUME"] = volume.stopping_volume(df)

    ndns = volume.no_demand_no_supply(df)
    features["NO_DEMAND"] = ndns["no_demand"]
    features["NO_SUPPLY"] = ndns["no_supply"]

    features["BUY_VOLUME"] = volume.buy_volume(df)
    features["SELL_VOLUME"] = volume.sell_volume(df)
    features["DELTA_VOLUME"] = volume.delta_volume(df)

    features["AMIHUD"] = volume.amihud_illiquidity(df)
    features["VPIN"] = volume.vpin(df)


# ==========================
# VOLATILITY
# ==========================

    features["TR"] = volatility.true_range(df)
    features["ATR_14"] = volatility.atr(df)
    features["NATR_14"] = volatility.natr(df)

    bb = volatility.bollinger_bands(df)
    features["BB_MIDDLE"] = bb["middle"]
    features["BB_UPPER"] = bb["upper"]
    features["BB_LOWER"] = bb["lower"]

    features["BB_WIDTH"] = volatility.bollinger_width(df)
    features["BB_PERCENT_B"] = volatility.bollinger_percent_b(df)

    dc = volatility.donchian_channel(df)
    features["DONCHIAN_UPPER"] = dc["upper"]
    features["DONCHIAN_MIDDLE"] = dc["middle"]
    features["DONCHIAN_LOWER"] = dc["lower"]

    kc = volatility.keltner_channel(df)
    features["KELTNER_MIDDLE"] = kc["middle"]
    features["KELTNER_UPPER"] = kc["upper"]
    features["KELTNER_LOWER"] = kc["lower"]

    features["HV_21"] = volatility.historical_volatility(df)
    features["CHAIKIN_VOL"] = volatility.chaikin_volatility(df)
    features["ULCER_INDEX"] = volatility.ulcer_index(df)
    features["STD_20"] = volatility.standard_deviation(df)
    features["VAR_20"] = volatility.variance(df)
    features["MASS_INDEX"] = volatility.mass_index(df)
    features["VOLATILITY_RATIO"] = volatility.volatility_ratio(df)
    features["ATR_PERCENTILE"] = volatility.atr_percentile(df)
    features["BB_SQUEEZE"] = volatility.bollinger_squeeze(df)
    features["EXPANSION_INDEX"] = volatility.expansion_index(df)
    features["VOLATILITY_OSC"] = volatility.volatility_oscillator(df)
    features["ADAPTIVE_ATR"] = volatility.adaptive_atr(df)
    features["PARKINSON_VOL"] = volatility.parkinson_volatility(df)
    features["GARMAN_KLASS"] = volatility.garman_klass_volatility(df)
    features["ROGERS_SATCHELL"] = volatility.rogers_satchell_volatility(df)
    features["YANG_ZHANG"] = volatility.yang_zhang_volatility(df)
    features["CHOPPINESS"] = volatility.choppiness_index(df)
    features["VHF"] = volatility.vhf(df)
    features["BB_SQZ_MOM"] = volatility.bb_squeeze_momentum(df)
    features["ATR_STOP_DIST"] = volatility.atr_stop_distance(df)

    vstop = volatility.volatility_stop(df)
    features["VOL_STOP"] = vstop["stop_price"]
    features["VOL_STOP_LONG"] = vstop["is_long"]

    features["STANDARD_ERROR"] = volatility.standard_error(df)
    features["REI"] = volatility.rei(df)

    st = volatility.supertrend(df)
    features["SUPERTREND"] = st["supertrend"]
    features["SUPERTREND_TREND"] = st["trend"]
    features["SUPERTREND_UPPER"] = st["upper_band"]
    features["SUPERTREND_LOWER"] = st["lower_band"]


# ==========================
# CANDLE
# ==========================

    features["BODY_SIZE"] = candle.body_size(df)
    features["REAL_BODY"] = candle.real_body(df)
    features["BODY_PERCENT"] = candle.body_percent(df)
    features["BODY_MIDPOINT"] = candle.body_midpoint(df)
    features["BODY_RATIO"] = candle.body_ratio(df)
    features["BODY_STRENGTH"] = candle.body_strength(df)
    features["BODY_POSITION"] = candle.body_position(df)

    features["BULLISH_BODY"] = candle.bullish_body(df)
    features["BEARISH_BODY"] = candle.bearish_body(df)
    features["SMALL_BODY"] = candle.small_body(df)
    features["LARGE_BODY"] = candle.large_body(df)

    features["BODY_CHANGE"] = candle.body_change(df)
    features["BODY_AVG"] = candle.body_average(df)
    features["BODY_EXPANSION"] = candle.body_expansion(df)
    features["BODY_CONTRACTION"] = candle.body_contraction(df)

    features["UPPER_WICK"] = candle.upper_wick(df)
    features["LOWER_WICK"] = candle.lower_wick(df)
    features["WICK_SIZE"] = candle.wick_size(df)
    features["WICK_RATIO"] = candle.wick_ratio(df)
    features["WICK_BALANCE"] = candle.wick_balance(df)
    features["WICK_STRENGTH"] = candle.wick_strength(df)
    features["WICK_PERCENT"] = candle.wick_percent(df)

    features["LONG_UPPER_WICK"] = candle.long_upper_wick(df)
    features["LONG_LOWER_WICK"] = candle.long_lower_wick(df)
    features["SMALL_WICK"] = candle.small_wick(df)

    features["CANDLE_RANGE"] = candle.candle_range(df)
    features["TRUE_RANGE_CANDLE"] = candle.true_range(df)
    features["BODY_TO_RANGE"] = candle.body_to_range(df)
    features["RANGE_EXPANSION"] = candle.range_expansion(df)
    features["RANGE_CONTRACTION"] = candle.range_contraction(df)
    features["AVERAGE_RANGE"] = candle.average_range(df)
    features["ROLLING_RANGE"] = candle.rolling_range(df)
    features["RANGE_PERCENTILE"] = candle.range_percentile(df)

    features["GAP_UP"] = candle.gap_up(df)
    features["GAP_DOWN"] = candle.gap_down(df)
    features["BREAKAWAY_GAP_UP"] = candle.breakaway_gap_up(df)
    features["BREAKAWAY_GAP_DOWN"] = candle.breakaway_gap_down(df)
    features["GAP_SIZE"] = candle.gap_size(df)
    features["GAP_PERCENT"] = candle.gap_percent(df)
    features["GAP_FILL"] = candle.gap_fill(df)
    features["INSIDE_GAP"] = candle.inside_gap(df)

    features["DOJI"] = candle.doji(df)
    features["DRAGONFLY_DOJI"] = candle.dragonfly_doji(df)
    features["GRAVESTONE_DOJI"] = candle.gravestone_doji(df)
    features["LONG_LEGGED_DOJI"] = candle.long_legged_doji(df)

    features["MARUBOZU"] = candle.marubozu(df)
    features["BULLISH_MARUBOZU"] = candle.bullish_marubozu(df)
    features["BEARISH_MARUBOZU"] = candle.bearish_marubozu(df)

    features["SPINNING_TOP"] = candle.spinning_top(df)
    features["HIGH_WAVE"] = candle.high_wave(df)
    features["HAMMER_SHAPE"] = candle.hammer_shape(df)
    features["HANGING_MAN_SHAPE"] = candle.hanging_man_shape(df)
    features["INVERTED_HAMMER_SHAPE"] = candle.inverted_hammer_shape(df)
    features["SHOOTING_STAR_SHAPE"] = candle.shooting_star_shape(df)

    features["BELT_HOLD"] = candle.belt_hold(df)
    features["SHAVEN_HEAD"] = candle.shaven_head(df)
    features["SHAVEN_BOTTOM"] = candle.shaven_bottom(df)

    features["BULL_POWER"] = candle.bull_power(df)
    features["BEAR_POWER"] = candle.bear_power(df)
    features["BUYING_PRESSURE"] = candle.buying_pressure(df)
    features["SELLING_PRESSURE"] = candle.selling_pressure(df)
    features["CANDLE_STRENGTH"] = candle.candle_strength(df)
    features["DIRECTION_STRENGTH"] = candle.direction_strength(df)
    features["DOMINANCE_SCORE"] = candle.dominance_score(df)
    features["PRESSURE_SCORE"] = candle.pressure_score(df)
    features["BALANCE_SCORE"] = candle.balance_score(df)

    features["CLV"] = candle.close_location_value(df)
    features["CLOSE_PERCENT"] = candle.close_percent(df)
    features["CLOSE_TO_HIGH"] = candle.close_to_high(df)
    features["CLOSE_TO_LOW"] = candle.close_to_low(df)
    features["CLOSE_POSITION"] = candle.close_position(df)
    features["OPEN_LOCATION"] = candle.open_location(df)
    features["OPEN_POSITION"] = candle.open_position(df)

    features["BULLISH_CANDLE"] = candle.bullish_candle(df)
    features["BEARISH_CANDLE"] = candle.bearish_candle(df)
    features["NEUTRAL_CANDLE"] = candle.neutral_candle(df)
    features["INSIDE_BAR"] = candle.inside_bar(df)
    features["OUTSIDE_BAR"] = candle.outside_bar(df)
    features["ENGULFING_BODY"] = candle.engulfing_body(df)
    features["BODY_OVERLAP"] = candle.body_overlap(df)
    features["RANGE_OVERLAP"] = candle.range_overlap(df)

    features["EXPANSION_CANDLE"] = candle.expansion_candle(df)
    features["COMPRESSION_CANDLE"] = candle.compression_candle(df)
    features["IMPULSE_CANDLE"] = candle.impulse_candle(df)
    features["INDECISION_CANDLE"] = candle.indecision_candle(df)

    features["ROLLING_BODY_MEAN"] = candle.rolling_body_mean(df)
    features["ROLLING_BODY_STD"] = candle.rolling_body_std(df)
    features["ROLLING_BODY_ZSCORE"] = candle.rolling_body_zscore(df)
    features["ROLLING_RANGE_MEAN"] = candle.rolling_range_mean(df)
    features["ROLLING_RANGE_STD"] = candle.rolling_range_std(df)
    features["ROLLING_WICK_MEAN"] = candle.rolling_wick_mean(df)
    features["ROLLING_WICK_STD"] = candle.rolling_wick_std(df)
    features["ROLLING_BODY_PERCENTILE"] = candle.rolling_body_percentile(df)
    features["ROLLING_RANGE_PERCENTILE"] = candle.rolling_range_percentile(df)
    features["ROLLING_WICK_PERCENTILE"] = candle.rolling_wick_percentile(df)

    features["INSTITUTIONAL_BODY"] = candle.institutional_body(df)
    features["INSTITUTIONAL_WICK"] = candle.institutional_wick(df)
    features["INSTITUTIONAL_IMBALANCE"] = candle.institutional_imbalance(df)
    features["INSTITUTIONAL_PRESSURE"] = candle.institutional_pressure(df)
    features["ABSORPTION_CANDLE"] = candle.absorption_candle(df)
    features["REJECTION_CANDLE"] = candle.rejection_candle(df)
    features["ACCEPTANCE_CANDLE"] = candle.acceptance_candle(df)
    features["LIQUIDITY_SWEEP_CANDLE"] = candle.liquidity_sweep_candle(df)
    features["SMART_MONEY_CANDLE"] = candle.smart_money_candle(df)


# ==========================
# PATTERN
# ==========================

    pat = pattern.calculate_patterns(df)

    pat.columns = [c.upper() for c in pat.columns]

    features = features.join(pat)



# ==========================
# SUPPORT & RESISTANCE
# ==========================



# ==========================
# SUPPORT & RESISTANCE
# ==========================

    piv = support_resistance.classic_pivot_levels(df)
    features["PIVOT"] = piv["PP"]
    features["PIVOT_R1"] = piv["R1"]
    features["PIVOT_R2"] = piv["R2"]
    features["PIVOT_R3"] = piv["R3"]
    features["PIVOT_S1"] = piv["S1"]
    features["PIVOT_S2"] = piv["S2"]
    features["PIVOT_S3"] = piv["S3"]

    features["WOODIE_PIVOT"] = support_resistance.woodie_pivot(df)
    features["CAMARILLA_PIVOT"] = support_resistance.camarilla_pivot(df)
    features["DEMARK_PIVOT"] = support_resistance.demark_pivot(df)

    features["SWING_HIGH"] = support_resistance.swing_high(df)
    features["SWING_LOW"] = support_resistance.swing_low(df)
    features["LAST_SWING_HIGH"] = support_resistance.last_swing_high(df)
    features["LAST_SWING_LOW"] = support_resistance.last_swing_low(df)
    features["CONFIRMED_SWING_HIGH"] = support_resistance.confirmed_swing_high(df)
    features["CONFIRMED_SWING_LOW"] = support_resistance.confirmed_swing_low(df)

    features["HIGHEST_SWING"] = support_resistance.highest_swing(df)
    features["LOWEST_SWING"] = support_resistance.lowest_swing(df)

    features["HH_BREAKOUT"] = support_resistance.highest_high_breakout(df)
    features["LL_BREAKDOWN"] = support_resistance.lowest_low_breakdown(df)
    features["PREV_HIGH_BREAKOUT"] = support_resistance.previous_high_breakout(df)
    features["PREV_LOW_BREAKDOWN"] = support_resistance.previous_low_breakdown(df)

    features["DONCHIAN_BREAKOUT"] = support_resistance.donchian_breakout(df)
    features["RANGE_BREAKOUT"] = support_resistance.range_breakout(df)

    features["FRACTAL_HIGH"] = support_resistance.bill_williams_fractal_high(df)
    features["FRACTAL_LOW"] = support_resistance.bill_williams_fractal_low(df)
    features["FRACTAL_PIVOT"] = support_resistance.fractal_pivot(df)
    features["FRACTAL_SUPPORT"] = support_resistance.fractal_support(df)
    features["FRACTAL_RESISTANCE"] = support_resistance.fractal_resistance(df)

    dc = support_resistance.donchian_channel(df)
    features["SR_DONCHIAN_UPPER"] = dc["upper"]
    features["SR_DONCHIAN_MIDDLE"] = dc["middle"]
    features["SR_DONCHIAN_LOWER"] = dc["lower"]



    features["ROLLING_HIGH"] = support_resistance.rolling_highest_high(df)
    features["ROLLING_LOW"] = support_resistance.rolling_lowest_low(df)

    features["DYNAMIC_SUPPORT"] = support_resistance.dynamic_support(df)
    features["DYNAMIC_RESISTANCE"] = support_resistance.dynamic_resistance(df)
    features["ADAPTIVE_SUPPORT"] = support_resistance.adaptive_support(df)
    features["ADAPTIVE_RESISTANCE"] = support_resistance.adaptive_resistance(df)

    sz = support_resistance.support_zone(df)
    features["SUP_ZONE_UPPER"] = sz["zone_upper"]
    features["SUP_ZONE_LOWER"] = sz["zone_lower"]

    rz = support_resistance.resistance_zone(df)
    features["RES_ZONE_UPPER"] = rz["zone_upper"]
    features["RES_ZONE_LOWER"] = rz["zone_lower"]

    features["DEMAND_ZONE"] = support_resistance.demand_zone(df)
    features["SUPPLY_ZONE"] = support_resistance.supply_zone(df)

    reaction = support_resistance.reaction_zone(df)
    features["REACTION_DEMAND"] = reaction["demand"]
    features["REACTION_SUPPLY"] = reaction["supply"]

    features["CONGESTION_ZONE"] = support_resistance.congestion_zone(df)

    bull = support_resistance.bullish_fvg(df)
    features["BULLISH_FVG"] = bull["is_fvg"]

    bear = support_resistance.bearish_fvg(df)
    features["BEARISH_FVG"] = bear["is_fvg"]

    fvg = support_resistance.mitigated_fvg(df)
    features["FVG_MITIGATED"] = fvg["is_mitigated"]
    features["FVG_ACTIVE"] = fvg["is_active"]

    features["BPR"] = support_resistance.balanced_price_range(df)



    ob = support_resistance.order_blocks(df)
    features["FRESH_OB"] = ob["fresh_ob"]
    features["MITIGATED_OB"] = ob["mitigated_ob"]
    features["INVALIDATED_OB"] = ob["invalidated_ob"]
    features["OB_HIGH"] = ob["ob_high"]
    features["OB_LOW"] = ob["ob_low"]

    features["BREAKER_BLOCK"] = support_resistance.breaker_block(df)
    features["FLIP_ZONE"] = support_resistance.flip_zone(df)

    smc = support_resistance.smc_structure(df)
    features["BOS_UP"] = smc["bos_up"]
    features["BOS_DOWN"] = smc["bos_dn"]
    features["CHOCH_UP"] = smc["choch_up"]
    features["CHOCH_DOWN"] = smc["choch_dn"]
    features["SMC_TREND"] = smc["trend"]

    mss = support_resistance.market_structure_shift(df)
    features["MSS_BULLISH"] = mss["mss_bullish"]
    features["MSS_BEARISH"] = mss["mss_bearish"]

    pdz = support_resistance.premium_discount_zone(df)
    features["PREMIUM_LEVEL"] = pdz["premium_level"]
    features["EQUILIBRIUM"] = pdz["equilibrium"]
    features["DISCOUNT_LEVEL"] = pdz["discount_level"]

    features["BUY_SIDE_LIQUIDITY"] = support_resistance.buy_side_liquidity(df)
    features["SELL_SIDE_LIQUIDITY"] = support_resistance.sell_side_liquidity(df)
    features["LIQUIDITY_SWEEP"] = support_resistance.liquidity_sweep_level(df)

    features["TURTLE_SOUP"] = support_resistance.turtle_soup(df)
    features["JUDAS_SWING"] = support_resistance.judas_swing(df)

    # amd_cycle() depends on initial_balance(), which is not implemented.
    # Disabled temporarily.


    features["SUP_TOUCHES"] = support_resistance.support_touch_count(df)
    features["RES_TOUCHES"] = support_resistance.resistance_touch_count(df)

    features["LEVEL_STRENGTH"] = support_resistance.level_strength(df)
    features["LEVEL_CONFIDENCE"] = support_resistance.level_confidence(df)
    features["ZONE_WIDTH"] = support_resistance.zone_width(df)
    features["BOUNCE_COUNT"] = support_resistance.bounce_count(df)

    features["DISTANCE_TO_SUPPORT"] = support_resistance.distance_to_support(df)
    features["DISTANCE_TO_RESISTANCE"] = support_resistance.distance_to_resistance(df)
    features["NEAREST_SUPPORT"] = support_resistance.nearest_support(df)
    features["NEAREST_RESISTANCE"] = support_resistance.nearest_resistance(df)

    features["RISK_DISTANCE"] = support_resistance.risk_distance(df)
    features["REWARD_DISTANCE"] = support_resistance.reward_distance(df)

    features["CONFIRMED_BREAKOUT"] = support_resistance.confirmed_breakout(df)
    features["CONFIRMED_BREAKDOWN"] = support_resistance.confirmed_breakdown(df)
    features["FALSE_BREAKOUT"] = support_resistance.false_breakout(df)
    features["FALSE_BREAKDOWN"] = support_resistance.false_breakdown(df)

    features["RETEST_LEVEL"] = support_resistance.retest_level(df)
    features["BREAK_STRENGTH"] = support_resistance.break_strength(df)

    features["ROLLING_SUPPORT"] = support_resistance.rolling_support(df)
    features["ROLLING_RESISTANCE"] = support_resistance.rolling_resistance(df)

    features["SUP_PERCENTILE"] = support_resistance.support_percentile(df)
    features["RES_PERCENTILE"] = support_resistance.resistance_percentile(df)

    features["SUP_ZSCORE"] = support_resistance.support_zscore(df)
    features["RES_ZSCORE"] = support_resistance.resistance_zscore(df)

    features["ATR_SUPPORT"] = support_resistance.atr_support(df)
    features["ATR_RESISTANCE"] = support_resistance.atr_resistance(df)

    features["VOL_SUPPORT"] = support_resistance.volatility_support(df)
    features["VOL_RESISTANCE"] = support_resistance.volatility_resistance(df)



    ab = support_resistance.adaptive_breakout(df)
    features["ADAPTIVE_BREAKOUT_UPPER"] = ab["breakout_upper"]
    features["ADAPTIVE_BREAKOUT_LOWER"] = ab["breakout_lower"]

    daily = support_resistance.daily_levels(df)
    features["DAILY_HIGH"] = daily["upper"]
    features["DAILY_MID"] = daily["middle"]
    features["DAILY_LOW"] = daily["lower"]

    weekly = support_resistance.weekly_levels(df)
    features["WEEKLY_HIGH"] = weekly["upper"]
    features["WEEKLY_MID"] = weekly["middle"]
    features["WEEKLY_LOW"] = weekly["lower"]

    monthly = support_resistance.monthly_levels(df)
    features["MONTHLY_HIGH"] = monthly["upper"]
    features["MONTHLY_MID"] = monthly["middle"]
    features["MONTHLY_LOW"] = monthly["lower"]

    yearly = support_resistance.yearly_levels(df)
    features["YEARLY_HIGH"] = yearly["upper"]
    features["YEARLY_MID"] = yearly["middle"]
    features["YEARLY_LOW"] = yearly["lower"]

    features["MERGED_SUPPORT"] = support_resistance.merged_support(df)
    features["MERGED_RESISTANCE"] = support_resistance.merged_resistance(df)

    features["PRICE_CLUSTER"] = support_resistance.price_cluster(df)
    features["CLUSTER_SUPPORT"] = support_resistance.cluster_support(df)
    features["CLUSTER_RESISTANCE"] = support_resistance.cluster_resistance(df)
    features["CLUSTER_DENSITY"] = support_resistance.cluster_density(df)
    features["CLUSTER_STRENGTH"] = support_resistance.cluster_strength(df)

    features["MS_HIGH"] = support_resistance.market_structure_high(df)
    features["MS_LOW"] = support_resistance.market_structure_low(df)

    features["STRUCTURE_SUPPORT"] = support_resistance.structure_support(df)
    features["STRUCTURE_RESISTANCE"] = support_resistance.structure_resistance(df)

    features["HIGHER_HIGH"] = support_resistance.higher_high(df)
    features["HIGHER_LOW"] = support_resistance.higher_low(df)
    features["LOWER_HIGH"] = support_resistance.lower_high(df)
    features["LOWER_LOW"] = support_resistance.lower_low(df)



    avp = support_resistance.adaptive_volume_profile(df)
    features["AVP_POC"] = avp["POC"]
    features["AVP_VAH"] = avp["VAH"]
    features["AVP_VAL"] = avp["VAL"]

    features["SWING_HIGH_VWAP"] = support_resistance.swing_high_vwap(df)
    features["SWING_LOW_VWAP"] = support_resistance.swing_low_vwap(df)

    fib = support_resistance.fibonacci_levels(df)
    features["FIB_0"] = fib["Fib_0"]
    features["FIB_236"] = fib["Fib_236"]
    features["FIB_382"] = fib["Fib_382"]
    features["FIB_500"] = fib["Fib_500"]
    features["FIB_618"] = fib["Fib_618"]
    features["FIB_786"] = fib["Fib_786"]
    features["FIB_100"] = fib["Fib_100"]
    features["GOLDEN_ZONE_UPPER"] = fib["GoldenZone_Upper"]
    features["GOLDEN_ZONE_LOWER"] = fib["GoldenZone_Lower"]

    gann = support_resistance.gann_levels(df)
    features["GANN_0_8"] = gann["Gann_0_8"]
    features["GANN_1_8"] = gann["Gann_1_8"]
    features["GANN_2_8"] = gann["Gann_2_8"]
    features["GANN_3_8"] = gann["Gann_3_8"]
    features["GANN_4_8"] = gann["Gann_4_8"]
    features["GANN_5_8"] = gann["Gann_5_8"]
    features["GANN_6_8"] = gann["Gann_6_8"]
    features["GANN_7_8"] = gann["Gann_7_8"]
    features["GANN_8_8"] = gann["Gann_8_8"]

    murrey = support_resistance.murrey_math_levels(df)
    features["MURREY_0_8"] = murrey["MM_0_8"]
    features["MURREY_1_8"] = murrey["MM_1_8"]
    features["MURREY_2_8"] = murrey["MM_2_8"]
    features["MURREY_3_8"] = murrey["MM_3_8"]
    features["MURREY_4_8"] = murrey["MM_4_8"]
    features["MURREY_5_8"] = murrey["MM_5_8"]
    features["MURREY_6_8"] = murrey["MM_6_8"]
    features["MURREY_7_8"] = murrey["MM_7_8"]
    features["MURREY_8_8"] = murrey["MM_8_8"]

    features["PSYCHOLOGICAL_LEVEL"] = support_resistance.psychological_levels(df)

    gap = support_resistance.gap_levels(df)
    features["GAP_SUPPORT"] = gap["gap_support"]
    features["GAP_RESISTANCE"] = gap["gap_resistance"]

    prev = support_resistance.previous_session_levels(df)
    features["PREV_DAY_HIGH"] = prev["PDH"]
    features["PREV_DAY_LOW"] = prev["PDL"]

    features["INST_CONFLUENCE"] = support_resistance.institutional_confluence_score(df)
    features["MTF_CONFLUENCE"] = support_resistance.mtf_confluence_score(df)
    features["LEVEL_RANK"] = support_resistance.level_reliability_ranking(df)



    features["LIQUIDITY_POOL"] = support_resistance.liquidity_pool(df)
    features["INSTITUTIONAL_SUPPORT"] = support_resistance.institutional_support(df)
    features["INSTITUTIONAL_RESISTANCE"] = support_resistance.institutional_resistance(df)

    features["SMART_MONEY_LEVEL"] = support_resistance.smart_money_level(df)
    features["STOP_HUNT_ZONE"] = support_resistance.stop_hunt_zone(df)

    features["INTERNAL_LIQUIDITY"] = support_resistance.internal_liquidity(df)
    features["EXTERNAL_LIQUIDITY"] = support_resistance.external_liquidity(df)

    eqh = support_resistance.equal_highs(df)
    features["EQUAL_HIGHS"] = eqh["is_equal"]

    eql = support_resistance.equal_lows(df)
    features["EQUAL_LOWS"] = eql["is_equal"]




    # ==========================
    # SMART MONEY
    # ==========================

    sm = smart_money

    features["SM_SWING_HIGH"] = sm.swing_high(df)
    features["SM_SWING_LOW"] = sm.swing_low(df)

    features["SM_HIGHER_HIGH"] = sm.higher_high(df)
    features["SM_HIGHER_LOW"] = sm.higher_low(df)
    features["SM_LOWER_HIGH"] = sm.lower_high(df)
    features["SM_LOWER_LOW"] = sm.lower_low(df)

    features["SM_BOS"] = sm.bos(df)
    features["SM_CHOCH"] = sm.choch(df)
    features["SM_MSS"] = sm.market_structure_shift(df)

    features["SM_BUY_SIDE_LIQUIDITY"] = sm.buy_side_liquidity(df)
    features["SM_SELL_SIDE_LIQUIDITY"] = sm.sell_side_liquidity(df)

    features["SM_FRESH_OB"] = sm.fresh_order_block(df)
    features["SM_MITIGATED_OB"] = sm.mitigated_order_block(df)

    features["SM_ACTIVE_FVG"] = sm.active_fvg(df)
    features["SM_MITIGATED_FVG"] = sm.mitigated_fvg(df)

    features["SM_PREMIUM_ZONE"] = sm.premium_zone(df)
    features["SM_DISCOUNT_ZONE"] = sm.discount_zone(df)
    features["SM_EQUILIBRIUM"] = sm.equilibrium(df)

    features["SM_BOS_SCORE"] = sm.bos_score(df)
    features["SM_CHOCH_SCORE"] = sm.choch_score(df)
    features["SM_LIQUIDITY_SCORE"] = sm.liquidity_score(df)
    features["SM_FVG_SCORE"] = sm.fvg_score(df)
    features["SM_TREND_SCORE"] = sm.trend_score(df)
    features["SM_INSTITUTIONAL_SCORE"] = sm.institutional_score(df)
    features["SM_SMART_MONEY_SCORE"] = sm.smart_money_score(df)


# ==========================
# STATISTICS (PART-1)
# ==========================

    features["STAT_MEAN"] = statistics.mean(df)
    features["STAT_ROLLING_MEAN"] = statistics.rolling_mean(df)

    features["STAT_MEDIAN"] = statistics.median(df)
    features["STAT_ROLLING_MEDIAN"] = statistics.rolling_median(df)

    features["STAT_MODE"] = statistics.mode(df)

    features["STAT_VARIANCE"] = statistics.variance(df)
    features["STAT_ROLLING_VARIANCE"] = statistics.rolling_variance(df)

    features["STAT_STD"] = statistics.standard_deviation(df)
    features["STAT_ROLLING_STD"] = statistics.rolling_standard_deviation(df)

    features["STAT_MAD"] = statistics.mean_absolute_deviation(df)
    features["STAT_MEDIAN_MAD"] = statistics.median_absolute_deviation(df)

    features["STAT_RMS"] = statistics.root_mean_square(df)

    features["STAT_CV"] = statistics.coefficient_of_variation(df)

    features["STAT_RANGE"] = statistics.range_stat(df)
    features["STAT_IQR"] = statistics.interquartile_range(df)

    features["STAT_QUANTILE"] = statistics.quantile(df)
    features["STAT_PERCENTILE"] = statistics.percentile(df)
    features["STAT_ROLLING_PERCENTILE"] = statistics.rolling_percentile(df)
    features["STAT_PERCENTILE_RANK"] = statistics.percentile_rank(df)

    features["STAT_ZSCORE"] = statistics.z_score(df)
    features["STAT_ROLLING_ZSCORE"] = statistics.rolling_z_score(df)
    features["STAT_MODIFIED_ZSCORE"] = statistics.modified_z_score(df)

    features["STAT_MINMAX"] = statistics.min_max_scaling(df)
    features["STAT_NORMALIZED"] = statistics.normalization(df)
    features["STAT_ROBUST_SCALE"] = statistics.robust_scaling(df)
    features["STAT_WINSORIZED"] = statistics.winsorization(df)




# ==========================
# STATISTICS (PART-2)
# ==========================

    features["STAT_SKEWNESS"] = statistics.skewness(df)
    features["STAT_ROLLING_SKEWNESS"] = statistics.rolling_skewness(df)

    features["STAT_KURTOSIS"] = statistics.kurtosis(df)
    features["STAT_ROLLING_KURTOSIS"] = statistics.rolling_kurtosis(df)

    features["STAT_ENTROPY"] = statistics.entropy(df)
    features["STAT_SHANNON_ENTROPY"] = statistics.shannon_entropy(df)

    features["STAT_JARQUE_BERA"] = statistics.jarque_bera(df)
    features["STAT_NORMALITY"] = statistics.normality_score(df)

    features["STAT_AUTOCORR"] = statistics.autocorrelation(df)

    reg = statistics.rolling_regression(df)
    features["REGRESSION_VALUE"] = reg["value"]
    features["REGRESSION_SLOPE"] = reg["slope"]
    features["REGRESSION_INTERCEPT"] = reg["intercept"]
    features["REGRESSION_R2"] = reg["r2"]

    lr = statistics.linear_regression(df)
    features["LINEAR_REGRESSION_VALUE"] = lr["value"]
    features["LINEAR_REGRESSION_SLOPE"] = lr["slope"]
    features["LINEAR_REGRESSION_INTERCEPT"] = lr["intercept"]
    features["LINEAR_REGRESSION_R2"] = lr["r2"]
    features["REGRESSION_LINE"] = statistics.regression_line(df)
    features["REGRESSION_VALUE_ONLY"] = statistics.regression_value(df)
    features["REGRESSION_INTERCEPT_ONLY"] = statistics.regression_intercept(df)
    features["REGRESSION_SLOPE_ONLY"] = statistics.regression_slope(df)
    features["ROLLING_SLOPE"] = statistics.rolling_slope(df)

    features["R_SQUARED"] = statistics.r_squared(df)
    features["ADJUSTED_R_SQUARED"] = statistics.adjusted_r_squared(df)

    features["REGRESSION_RESIDUAL"] = statistics.residual(df)
    features["REGRESSION_RSE"] = statistics.residual_standard_error(df)

    rc = statistics.regression_channel(df)
    features["REG_CHANNEL_MIDDLE"] = rc["middle"]
    features["REG_CHANNEL_UPPER"] = rc["upper"]
    features["REG_CHANNEL_LOWER"] = rc["lower"]




# ==========================
# STATISTICS (PART-3)
# ==========================

    features["OUTLIER_DETECTION"] = statistics.outlier_detection(df)
    features["THREE_SIGMA_RULE"] = statistics.three_sigma_rule(df)
    features["MODIFIED_Z_OUTLIER"] = statistics.modified_z_outlier(df)

    features["LINEAR_TREND_STRENGTH"] = statistics.linear_trend_strength(df)
    features["TREND_ANGLE"] = statistics.trend_angle(df)
    features["SLOPE_PERCENTAGE"] = statistics.slope_percentage(df)

    features["COEFF_DISPERSION"] = statistics.coefficient_of_dispersion(df)
    features["RELATIVE_STD_DEV"] = statistics.relative_standard_deviation(df)

    # ---------------------------------
    # Benchmark dependent statistics
    # Using self benchmark to keep engine standalone.
    # Replace with index dataframe later if needed.
    # ---------------------------------

    bench = df

    features["PEARSON_CORR"] = statistics.pearson_correlation(df, bench)
    features["ROLLING_PEARSON"] = statistics.rolling_pearson(df, bench)

    # scipy not installed
    # features["SPEARMAN_CORR"] = statistics.spearman_correlation(df, bench)
    # scipy not installed
    # features["ROLLING_SPEARMAN"] = statistics.rolling_spearman(df, bench)

    # scipy not installed
    # features["KENDALL_CORR"] = statistics.kendall_correlation(df, bench)
    # scipy not installed
    # features["ROLLING_KENDALL"] = statistics.rolling_kendall(df, bench)

    features["LAG_CORRELATION"] = statistics.lag_correlation(df, bench)
    features["CROSS_CORRELATION"] = statistics.cross_correlation(df, bench)

    features["COVARIANCE"] = statistics.covariance(df, bench)
    features["ROLLING_COVARIANCE"] = statistics.rolling_covariance(df, bench)

    features["BETA"] = statistics.beta(df, bench)
    features["ROLLING_BETA"] = statistics.rolling_beta(df, bench)

    features["ALPHA"] = statistics.alpha(df, bench)
    features["ROLLING_ALPHA"] = statistics.rolling_alpha(df, bench)

    features["TRACKING_ERROR"] = statistics.tracking_error(df, bench)
    features["INFORMATION_RATIO"] = statistics.information_ratio(df, bench)


    return features


if __name__ == "__main__":

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    result = run("RELIANCE")

    print(result.tail())

