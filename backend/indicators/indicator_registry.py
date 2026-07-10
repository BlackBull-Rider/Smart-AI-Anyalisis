"""
GREEN BULL DATA ENGINE
Indicator Registry

Phase-1
Modules:
    - Moving Average
    - Momentum
    - Candle
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


# ============================================================
# MODULE CONFIG
# ============================================================

MODULES = {
    "moving_average": "backend.indicators.core.moving_average",
    "momentum": "backend.indicators.core.momentum",
    "candle": "backend.indicators.core.candle",
}


# ============================================================
# FEATURE
# ============================================================

@dataclass(slots=True)
class Feature:

    name: str
    module: str
    category: str
    function: Callable
    enabled: bool = True


# ============================================================
# STORE FEATURES
# ============================================================

STORE_FEATURES = {

    "moving_average": [

        "alma",
        "dema",
        "ema",
        "hma",
        "kama",
        "lsma",
        "mcginley_dynamic",
        "sma",
        "smma",
        "t3",
        "tema",
        "trima",
        "vidya",
        "vwma",
        "wma",
        "zlema",

    ],

    "momentum": [

        "adx",
        "cci",
        "dpo",
        "macd",
        "momentum",
        "ppo",
        "roc",
        "rsi",
        "rsi_slope",
        "stochastic",
        "stochastic_rsi",
        "trix",
        "ultimate_oscillator",
        "williams_r",

    ],

    "candle": [

        "absorption_candle",
        "acceptance_candle",
        "average_range",
        "balance_score",
        "bear_power",
        "bearish_body",
        "bearish_candle",
        "bearish_marubozu",
        "belt_hold",
        "body_average",
        "body_change",
        "body_contraction",
        "body_expansion",
        "body_midpoint",
        "body_overlap",
        "body_percent",
        "body_position",
        "body_ratio",
        "body_size",
        "body_strength",
        "body_to_range",
        "breakaway_gap_down",
        "breakaway_gap_up",
        "bull_power",
        "bullish_body",
        "bullish_candle",
        "bullish_marubozu",
        "buying_pressure",
        "candle_range",
        "candle_strength",
        "close_location_value",
        "close_percent",
        "close_position",
        "close_to_high",
        "close_to_low",
        "closing_gap",
        "compression_candle",
        "direction_strength",
        "doji",
        "dominance_score",
        "dragonfly_doji",
        "engulfing_body",
        "expansion_candle",
        "gap_down",
        "gap_fill",
        "gap_percent",
        "gap_size",
        "gap_up",
        "gravestone_doji",
        "hammer_shape",
        "hanging_man_shape",
        "high_wave",
        "impulse_candle",
        "indecision_candle",
        "inside_bar",
        "inside_gap",
        "institutional_body",
        "institutional_imbalance",
        "institutional_pressure",
        "institutional_wick",
        "inverted_hammer_shape",
        "large_body",
        "liquidity_sweep_candle",
        "long_legged_doji",
        "long_lower_wick",
        "long_upper_wick",
        "lower_shadow",
        "lower_wick",
        "marubozu",
        "neutral_candle",
        "open_location",
        "open_position",
        "opening_gap",
        "outside_bar",
        "pressure_score",
        "range_contraction",
        "range_expansion",
        "range_overlap",
        "range_percentile",
        "real_body",
        "rejection_candle",
        "selling_pressure",
        "shaven_bottom",
        "shaven_head",
        "shooting_star_shape",
        "small_body",
        "small_wick",
        "smart_money_candle",
        "spinning_top",
        "true_range",
        "upper_shadow",
        "upper_wick",
        "wick_balance",
        "wick_percent",
        "wick_ratio",
        "wick_size",
        "wick_strength",

    ],

}

# ================================
# PHASE 2
# Volatility + Volume + Statistics
# ================================

STORE_FEATURES.update({

    "volatility": [

        "adaptive_atr",
        "atr",
        "atr_percentile",
        "atr_stop_distance",
        "atr_trailing_stop",
        "bb_squeeze_momentum",
        "bollinger_bands",
        "bollinger_lower",
        "bollinger_middle",
        "bollinger_percent_b",
        "bollinger_squeeze",
        "bollinger_upper",
        "bollinger_width",
        "chaikin_volatility",
        "choppiness_index",
        "donchian_channel",
        "donchian_lower",
        "donchian_middle",
        "donchian_upper",
        "expansion_index",
        "garman_klass_volatility",
        "historical_volatility",
        "keltner_channel",
        "keltner_lower",
        "keltner_middle",
        "keltner_upper",
        "mass_index",
        "natr",
        "parkinson_volatility",
        "rei",
        "rogers_satchell_volatility",
        "standard_deviation",
        "standard_error",
        "supertrend",
        "true_range",
        "ulcer_index",
        "variance",
        "vhf",
        "volatility_oscillator",
        "volatility_ratio",
        "volatility_stop",
        "yang_zhang_volatility",

    ],

    "volume": [

        "accumulation_distribution_line",
        "accumulation_distribution_oscillator",
        "amihud_illiquidity",
        "anchored_vwap",
        "anchored_vwap_bands",
        "average_volume",
        "buy_volume",
        "chaikin_money_flow",
        "delta_volume",
        "ease_of_movement",
        "effort_vs_result",
        "elastic_volume_weighted_momentum",
        "force_index",
        "force_index_ema",
        "intraday_rvol",
        "klinger_histogram",
        "klinger_oscillator",
        "klinger_signal",
        "money_flow_index",
        "money_flow_multiplier",
        "money_flow_volume",
        "negative_money_flow",
        "negative_volume_index",
        "no_demand_no_supply",
        "obv",
        "obv_ema",
        "obv_oscillator",
        "obv_roc",
        "obv_sma",
        "percentage_volume_oscillator",
        "positive_money_flow",
        "positive_volume_index",
        "price_volume_rank",
        "price_volume_trend",
        "relative_volume",
        "rolling_volume",
        "rolling_vwap",
        "sell_volume",
        "session_volume_profile",
        "smart_money_index",
        "smoothed_eom",
        "stopping_volume",
        "time_segmented_volume",
        "typical_price_volume",
        "volume",
        "volume_ema",
        "volume_macd",
        "volume_oscillator",
        "volume_percentile",
        "volume_profile",
        "volume_ratio",
        "volume_roc",
        "volume_sma",
        "volume_trend",
        "volume_zscore",
        "vpin",
        "vwap",
        "vwap_bands",
        "vwema",
        "vwma",
        "weighted_close_volume",

    ],

    "statistics": [

        "adjusted_r_squared",
        "alpha",
        "autocorrelation",
        "beta",
        "coefficient_of_dispersion",
        "coefficient_of_variation",
        "covariance",
        "cross_correlation",
        "entropy",
        "information_ratio",
        "interquartile_range",
        "jarque_bera",
        "kendall_correlation",
        "kurtosis",
        "lag_correlation",
        "linear_regression",
        "linear_trend_strength",
        "mean",
        "mean_absolute_deviation",
        "median",
        "median_absolute_deviation",
        "min_max_scaling",
        "mode",
        "modified_z_outlier",
        "modified_z_score",
        "normality_score",
        "normalization",
        "outlier_detection",
        "pearson_correlation",
        "percentile",
        "percentile_rank",
        "quantile",
        "r_squared",
        "range_stat",
        "regression_channel",
        "regression_intercept",
        "regression_line",
        "regression_slope",
        "regression_value",
        "relative_standard_deviation",
        "residual",
        "residual_standard_error",
        "robust_scaling",
        "rolling_alpha",
        "rolling_beta",
        "rolling_covariance",
        "rolling_kendall",
        "rolling_percentile",
        "rolling_regression",
        "rolling_slope",
        "rolling_spearman",
        "rolling_standard_deviation",
        "root_mean_square",
        "shannon_entropy",
        "skewness",
        "slope_percentage",
        "spearman_correlation",
        "standard_deviation",
        "three_sigma_rule",
        "tracking_error",
        "trend_angle",
        "variance",
        "winsorization",
        "z_score",

    ],

})


# ===================================
# PHASE 3
# Smart Money + Pattern + S/R
# ===================================

STORE_FEATURES.update({

    "smart_money": [

        "swing_high",
        "swing_low",
        "higher_high",
        "higher_low",
        "lower_high",
        "lower_low",
        "bos",
        "internal_bos",
        "external_bos",
        "choch",
        "internal_choch",
        "external_choch",
        "market_structure_shift",
        "trend_state",
        "equal_high",
        "equal_low",
        "buy_side_liquidity",
        "sell_side_liquidity",
        "buy_side_liquidity_sweep",
        "sell_side_liquidity_sweep",
        "resting_buy_side_liquidity",
        "resting_sell_side_liquidity",
        "fair_value_gap",
        "bullish_fvg",
        "bearish_fvg",
        "order_block",
        "bullish_order_block",
        "bearish_order_block",
        "breaker_block",
        "mitigation_block",
        "optimal_trade_entry",
        "premium_discount_zone",
        "displacement",
        "balanced_price_range",
        "market_efficiency_gap",
        "liquidity_void",
        "inducement",
        "smart_money_bias",

    ],

    "pattern": [

        "calculate_patterns",

    ],

    "support_resistance": [

        "pivot_point",
        "pivot_support_1",
        "pivot_support_2",
        "pivot_support_3",
        "pivot_resistance_1",
        "pivot_resistance_2",
        "pivot_resistance_3",

        "floor_pivot",
        "floor_s1",
        "floor_s2",
        "floor_s3",
        "floor_r1",
        "floor_r2",
        "floor_r3",

        "woodie_pivot",
        "woodie_r1",
        "woodie_r2",
        "woodie_r3",
        "woodie_s1",
        "woodie_s2",
        "woodie_s3",

        "camarilla_pivot",
        "camarilla_h1",
        "camarilla_h2",
        "camarilla_h3",
        "camarilla_h4",
        "camarilla_l1",
        "camarilla_l2",
        "camarilla_l3",
        "camarilla_l4",

        "demark_pivot",
        "fibonacci_pivot",

        "swing_high",
        "swing_low",
        "support_zone",
        "resistance_zone",
        "price_cluster",
        "equal_high",
        "equal_low",
        "volume_profile",
        "point_of_control",
        "value_area_high",
        "value_area_low",
        "high_volume_node",
        "low_volume_node",
        "fair_value_gap",
        "bullish_fvg",
        "bearish_fvg",
        "order_block",
        "bullish_order_block",
        "bearish_order_block",
        "breaker_block",
        "mitigation_block",
        "dynamic_support",
        "dynamic_resistance",
        "trendline_support",
        "trendline_resistance",
        "channel_support",
        "channel_resistance",

    ],

})



# ============================================================
# INDICATOR REGISTRY
# ============================================================

class IndicatorRegistry:

    def __init__(self) -> None:

        self.features: dict[str, Feature] = {}

        self.modules: dict[str, object] = {}

        self.load_all()

    # --------------------------------------------------------

    def load_all(self) -> None:

        for category, module_path in MODULES.items():

            module = importlib.import_module(module_path)

            self.modules[category] = module

            self._register_module(category, module)

        logger.info(
            "Indicator Registry Loaded | %d Indicators",
            len(self.features),
        )

    # --------------------------------------------------------

    def _register_module(
        self,
        category: str,
        module,
    ) -> None:

        feature_names = STORE_FEATURES.get(category, [])

        for name in feature_names:

            func = getattr(module, name, None)

            if func is None:

                logger.warning(
                    "[%s] Missing indicator : %s",
                    category,
                    name,
                )

                continue

            if name in self.features:

                logger.warning(
                    "Duplicate Indicator : %s",
                    name,
                )

                continue

            self.features[name] = Feature(

                name=name,

                module=module.__name__,

                category=category,

                function=func,

            )

    # --------------------------------------------------------

    def get(
        self,
        name: str,
    ) -> Feature:

        if name not in self.features:

            raise KeyError(
                f"Indicator '{name}' not registered."
            )

        return self.features[name]

    # --------------------------------------------------------

    def get_function(
        self,
        name: str,
    ):

        return self.get(name).function

    # --------------------------------------------------------

    def exists(
        self,
        name: str,
    ) -> bool:

        return name in self.features

    # --------------------------------------------------------

    def by_category(
        self,
        category: str,
    ) -> list[Feature]:

        return [

            feature

            for feature in self.features.values()

            if feature.category == category

        ]

    # --------------------------------------------------------

    def names(self) -> list[str]:

        return sorted(

            self.features.keys()

        )

    # --------------------------------------------------------

    def categories(self) -> list[str]:

        return sorted(

            STORE_FEATURES.keys()

        )

    # --------------------------------------------------------

    def count(self) -> int:

        return len(

            self.features

        )

    # --------------------------------------------------------

    def summary(self) -> dict[str, int]:

        return {

            category: len(

                STORE_FEATURES.get(

                    category,

                    [],

                )

            )

            for category in STORE_FEATURES

        }

    # --------------------------------------------------------

    def __contains__(
        self,
        item: str,
    ) -> bool:

        return item in self.features

    # --------------------------------------------------------

    def __len__(self):

        return len(

            self.features

        )

    # --------------------------------------------------------

    def __iter__(self):

        return iter(

            self.features.values()

        )


# ============================================================
# GLOBAL REGISTRY
# ============================================================

registry = IndicatorRegistry()


def get_indicator(
    name: str,
):

    return registry.get_function(name)


def list_indicators():

    return registry.names()


def indicator_exists(
    name: str,
):

    return registry.exists(name)


