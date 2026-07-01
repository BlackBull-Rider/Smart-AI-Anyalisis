"""
backend/indicators/core/pattern.py

Green Bull Rider V109 - Sovereign Layer-1 Structural Geometry Engine
Strictly Layer-1 Geometric Pattern Detection Engine.
Calculates structural chart primitives purely from mathematical logic.
"""

import numpy as np
import pandas as pd

# Scipy ছাড়াই লিনিয়ার রিগ্রেশন (টার্মাক্সে মেমোরি ক্র্যাশ এড়ানোর জন্য)
def linregress(x, y):
    if len(x) == 0 or len(y) == 0: 
        return type('obj', (object,), {'slope': 0, 'intercept': 0, 'rvalue': 0})()
    slope, intercept = np.polyfit(x, y, 1)
    r_val = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    return type('obj', (object,), {'slope': slope, 'intercept': intercept, 'rvalue': r_val})()

EPSILON = 1e-9

CONFIG = {
    "pivot_window": 5,
    "r2_threshold": 0.85,
    "flat_slope_threshold": 0.0015,
    "parallel_threshold": 0.002,
    "extrema_match_pct": 0.015,
    "pole_momentum_pct": 0.04,
    "flag_retracement_limit": 0.5,
    "volume_decay_ratio": 0.8,
    "cup_min_depth": 0.10,
    "handle_max_depth_ratio": 0.50,
    "min_handle_duration": 3
}

def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # 1. PRE-ALLOCATE OUTPUT ARRAYS (লেয়ার-১ এবং লেয়ার-২ এর জন্য সব কলাম এখানে আছে)
    float_cols = [
        "triangle_detected", "triangle_upper", "triangle_lower", "apex_distance", 
        "compression_pct", "breakout_pressure", "channel_detected", "channel_upper", 
        "channel_lower", "channel_width", "rectangle_detected", "rectangle_upper", 
        "rectangle_lower", "rectangle_width", "flag_detected", "pennant_detected", 
        "flag_quality", "pole_length", "retracement_depth", "volume_decay",
        "wedge_detected", "cup_detected", "handle_detected", "rounding_top_detected", 
        "rounding_bottom_detected", "neckline", "hs_detected", "ihs_detected", 
        "hs_neckline_slope", "hs_breakout_confirmed", "double_top_detected", 
        "double_bottom_detected", "triple_top_detected", "triple_bottom_detected",
        # Layer-2 Advanced Metrics support columns
        "resistance_strength", "volume_confirmation", "support_strength",
        "pattern_confidence", "trend_strength", "breakout_strength", "swing_low", "swing_high"
    ]
    
    out = {col: np.full(n, np.nan) for col in float_cols}
    out["triangle_type"] = np.full(n, "None", dtype=object)
    out["channel_type"] = np.full(n, "None", dtype=object)
    out["wedge_type"] = np.full(n, "None", dtype=object)
    out["pattern_family"] = np.full(n, "None", dtype=object)
    out["market_phase"] = np.full(n, "Neutral", dtype=object)
    out["market_regime"] = np.full(n, "Neutral", dtype=object)

    high, low, close = df['high'].values, df['low'].values, df['close'].values
    vol_sma_5 = df['volume'].rolling(5).mean().values
    vol_sma_20 = df['volume'].rolling(20).mean().values

    w = CONFIG["pivot_window"]
    sh_indices, sl_indices = [], []
    
    phase_priority = {"Neutral": 0, "Accumulation/Distribution": 1, "Trending": 2, "Compression": 3, "Breakout": 4}
    current_phase_prio = np.zeros(n)

    def _set_phase(idx: int, phase: str):
        if phase_priority[phase] > current_phase_prio[idx]:
            out["market_phase"][idx] = phase
            current_phase_prio[idx] = phase_priority[phase]
            
    def _set_family(idx: int, family: str):
        curr = out["pattern_family"][idx]
        if curr == "None" or family == "Reversal":
            out["pattern_family"][idx] = family

    # 2. SINGLE-PASS GEOMETRIC EVALUATION
    for i in range(w * 2, n):
        pivot_idx = i - w
        
        # Pivot Confirmation
        is_sh, is_sl = True, True
        for j in range(1, w + 1):
            if high[pivot_idx] <= high[pivot_idx - j] or high[pivot_idx] <= high[pivot_idx + j]: is_sh = False
            if low[pivot_idx] >= low[pivot_idx - j] or low[pivot_idx] >= low[pivot_idx + j]: is_sl = False

        if is_sh: 
            sh_indices.append(pivot_idx)
            out["swing_high"][i] = high[pivot_idx]
        if is_sl: 
            sl_indices.append(pivot_idx)
            out["swing_low"][i] = low[pivot_idx]

        # Tops & Bottoms
        if is_sh and len(sh_indices) >= 2:
            h1, h2 = high[sh_indices[-2]], high[sh_indices[-1]]
            if abs(h1 - h2) / (h1 + EPSILON) < CONFIG["extrema_match_pct"]:
                out["double_top_detected"][i] = 1.0
                out["resistance_strength"][i] = 85.0
                _set_family(i, "Reversal")
                if len(sh_indices) >= 3 and abs(h2 - high[sh_indices[-3]]) / (h2 + EPSILON) < CONFIG["extrema_match_pct"]:
                    out["triple_top_detected"][i] = 1.0
                    out["resistance_strength"][i] = 95.0

        if is_sl and len(sl_indices) >= 2:
            l1, l2 = low[sl_indices[-2]], low[sl_indices[-1]]
            if abs(l1 - l2) / (l1 + EPSILON) < CONFIG["extrema_match_pct"]:
                out["double_bottom_detected"][i] = 1.0
                out["support_strength"][i] = 85.0
                _set_family(i, "Reversal")
                if len(sl_indices) >= 3 and abs(l2 - low[sl_indices[-3]]) / (l2 + EPSILON) < CONFIG["extrema_match_pct"]:
                    out["triple_bottom_detected"][i] = 1.0
                    out["support_strength"][i] = 95.0

        # Head & Shoulders & IHS
        if is_sh and len(sh_indices) >= 3:
            ls, hd, rs = high[sh_indices[-3]], high[sh_indices[-2]], high[sh_indices[-1]]
            if hd > ls and hd > rs and abs(ls - rs) / (ls + EPSILON) < 0.03:
                valid_sls = [idx for idx in sl_indices if idx > sh_indices[-3] and idx < sh_indices[-1]]
                if len(valid_sls) >= 2:
                    out["hs_detected"][i] = 1.0
                    nl_y1, nl_y2 = low[valid_sls[-2]], low[valid_sls[-1]]
                    slope = (nl_y2 - nl_y1) / (valid_sls[-1] - valid_sls[-2] + EPSILON)
                    current_neckline = nl_y1 + slope * (i - valid_sls[-2])
                    out["neckline"][i] = current_neckline
                    out["hs_neckline_slope"][i] = slope
                    out["hs_breakout_confirmed"][i] = 1.0 if close[i] < current_neckline else 0.0
                    out["pattern_confidence"][i] = 80.0
                    _set_family(i, "Reversal")

        if is_sl and len(sl_indices) >= 3:
            ls, hd, rs = low[sl_indices[-3]], low[sl_indices[-2]], low[sl_indices[-1]]
            if hd < ls and hd < rs and abs(ls - rs) / (ls + EPSILON) < 0.03:
                valid_shs = [idx for idx in sh_indices if idx > sl_indices[-3] and idx < sl_indices[-1]]
                if len(valid_shs) >= 2:
                    out["ihs_detected"][i] = 1.0
                    nl_y1, nl_y2 = high[valid_shs[-2]], high[valid_shs[-1]]
                    slope = (nl_y2 - nl_y1) / (valid_shs[-1] - valid_shs[-2] + EPSILON)
                    current_neckline = nl_y1 + slope * (i - valid_shs[-2])
                    out["neckline"][i] = current_neckline
                    out["hs_neckline_slope"][i] = slope
                    out["hs_breakout_confirmed"][i] = 1.0 if close[i] > current_neckline else 0.0
                    out["pattern_confidence"][i] = 80.0
                    _set_family(i, "Reversal")

        # Triangles, Channels, Rectangles
        if len(sh_indices) >= 3 and len(sl_indices) >= 3 and (is_sh or is_sl):
            sh3, sl3 = np.array(sh_indices[-3:]), np.array(sl_indices[-3:])
            res = linregress(sh3, high[sh3])
            sup = linregress(sl3, low[sl3])
            m_res, m_sup = res.slope, sup.slope

            if res.rvalue**2 > CONFIG["r2_threshold"] and sup.rvalue**2 > CONFIG["r2_threshold"]:
                u_val = m_res * i + res.intercept
                l_val = m_sup * i + sup.intercept
                
                if u_val > l_val + EPSILON:
                    out["triangle_upper"][i] = u_val
                    out["triangle_lower"][i] = l_val
                    
                    is_parallel = abs(m_res - m_sup) < CONFIG["parallel_threshold"]
                    is_flat_res = abs(m_res) < CONFIG["flat_slope_threshold"]
                    is_flat_sup = abs(m_sup) < CONFIG["flat_slope_threshold"]
                    width = u_val - l_val

                    if is_parallel:
                        out["triangle_detected"][i] = np.nan
                        out["wedge_detected"][i] = np.nan
                        out["triangle_type"][i] = "None"
                        out["wedge_type"][i] = "None"
                        
                        if is_flat_res and is_flat_sup:
                            out["rectangle_detected"][i] = 1.0
                            out["channel_detected"][i] = np.nan
                            out["channel_type"][i] = "None"
                            out["rectangle_upper"][i] = u_val
                            out["rectangle_lower"][i] = l_val
                            out["rectangle_width"][i] = width
                            out["pattern_confidence"][i] = 75.0
                            _set_phase(i, "Accumulation/Distribution")
                        else:
                            out["channel_detected"][i] = 1.0
                            out["rectangle_detected"][i] = np.nan
                            out["channel_type"][i] = "Ascending" if m_res > 0 else "Descending"
                            out["channel_upper"][i] = u_val
                            out["channel_lower"][i] = l_val
                            out["channel_width"][i] = width
                            out["pattern_confidence"][i] = 80.0
                            _set_phase(i, "Trending")

                    elif abs(m_res - m_sup) > CONFIG["parallel_threshold"]:
                        out["channel_detected"][i] = np.nan
                        out["rectangle_detected"][i] = np.nan
                        out["channel_type"][i] = "None"
                        
                        detected_poly = False
                        if m_res < 0 and m_sup > 0:
                            out["triangle_detected"][i] = 1.0
                            out["triangle_type"][i] = "Symmetrical"
                            out["wedge_type"][i] = "None"
                            detected_poly = True
                        elif m_res < 0 and is_flat_sup:
                            out["triangle_detected"][i] = 1.0
                            out["triangle_type"][i] = "Descending"
                            out["wedge_type"][i] = "None"
                            detected_poly = True
                        elif is_flat_res and m_sup > 0:
                            out["triangle_detected"][i] = 1.0
                            out["triangle_type"][i] = "Ascending"
                            out["wedge_type"][i] = "None"
                            detected_poly = True
                        elif (m_res > 0 and m_sup > 0) or (m_res < 0 and m_sup < 0):
                            out["wedge_detected"][i] = 1.0
                            out["wedge_type"][i] = "Rising" if m_res > 0 else "Falling"
                            out["triangle_type"][i] = "None"
                            detected_poly = True

                        if detected_poly:
                            out["pattern_confidence"][i] = 85.0
                            _set_family(i, "Continuation")
                            _set_phase(i, "Compression")
                            
                            start_idx = min(sh3[0], sl3[0])
                            initial_upper = m_res * start_idx + res.intercept
                            initial_lower = m_sup * start_idx + sup.intercept
                            initial_width = initial_upper - initial_lower
                            
                            if initial_width > EPSILON:
                                out["compression_pct"][i] = np.clip((1.0 - (width / initial_width)) * 100.0, 0.0, 100.0)
                            
                            if width > EPSILON:
                                out["breakout_pressure"][i] = np.clip(((close[i] - l_val) / width) * 100.0, 0.0, 100.0)
                            
                            if m_res != m_sup:
                                intersect_x = (sup.intercept - res.intercept) / (m_res - m_sup)
                                out["apex_distance"][i] = max(0.0, intersect_x - i)

                    if close[i] > u_val or close[i] < l_val:
                        _set_phase(i, "Breakout")

        # Flags & Pennants
        if i >= 15:
            pole_val = close[i] - close[i-10]
            if abs(pole_val) / (close[i-10] + EPSILON) > CONFIG["pole_momentum_pct"]:
                ret_range = max(high[i-5:i]) - min(low[i-5:i])
                if ret_range < (abs(pole_val) * CONFIG["flag_retracement_limit"]):
                    v_decay = vol_sma_5[i] / (vol_sma_20[i] + EPSILON)
                    if v_decay < CONFIG["volume_decay_ratio"]:
                        out["flag_detected"][i] = 1.0
                        out["pole_length"][i] = abs(pole_val)
                        out["retracement_depth"][i] = np.clip((ret_range / (abs(pole_val) + EPSILON)) * 100.0, 0.0, 100.0)
                        out["volume_decay"][i] = v_decay
                        out["flag_quality"][i] = np.clip((1.0 - v_decay) * 100.0, 0.0, 100.0)
                        out["pattern_confidence"][i] = 90.0
                        _set_family(i, "Continuation")
                        if out["triangle_detected"][i] == 1.0: out["pennant_detected"][i] = 1.0

        # Institutional Cup & Handle / Rounding Tops & Bottoms
        if i >= 30:
            left_rim = np.max(high[i-30:i-20])
            mid_bowl = np.min(low[i-20:i-10])
            right_rim = np.max(high[i-10:i])
            
            if left_rim > mid_bowl * 1.05 and right_rim > mid_bowl * 1.05 and right_rim > left_rim * 0.98:
                if abs(left_rim - right_rim) / (left_rim + EPSILON) < 0.05:
                    out["rounding_bottom_detected"][i] = 1.0
                    depth = (left_rim - mid_bowl) / (left_rim + EPSILON)
                    
                    if depth >= CONFIG["cup_min_depth"]:
                        out["cup_detected"][i] = 1.0
                        out["pattern_confidence"][i] = 85.0
                        _set_family(i, "Reversal")
                        
                        handle_low_idx = i - 10 + np.argmin(low[i-10:i])
                        handle_low = low[handle_low_idx]
                        handle_duration = i - handle_low_idx
                        handle_depth = (right_rim - handle_low) / (right_rim + EPSILON)
                        
                        if handle_duration >= CONFIG["min_handle_duration"] and handle_depth < (depth * CONFIG["handle_max_depth_ratio"]):
                            if close[i] < right_rim and close[i] > handle_low:
                                if vol_sma_5[i] < vol_sma_20[i]:
                                    out["handle_detected"][i] = 1.0
                                    out["pattern_confidence"][i] = 95.0

            l_rim_t = np.min(low[i-30:i-20])
            m_bowl_t = np.max(high[i-20:i-10])
            r_rim_t = np.min(low[i-10:i])
            if l_rim_t < m_bowl_t * 0.95 and r_rim_t < m_bowl_t * 0.95:
                if abs(l_rim_t - r_rim_t) / (l_rim_t + EPSILON) < 0.05:
                    out["rounding_top_detected"][i] = 1.0
                    out["pattern_confidence"][i] = 85.0
                    _set_family(i, "Reversal")

    # 3. BUILD DATAFRAME
    pat_df = pd.DataFrame(out, index=df.index)
    for col in pat_df.columns:
        if pat_df[col].dtype == 'object':
            pat_df[col] = pat_df[col].replace(np.nan, 'None')
            
    return pat_df
