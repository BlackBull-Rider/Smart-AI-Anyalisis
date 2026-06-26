import pandas as pd
import numpy as np
from backend.indicators.smart_money import (
    calculate_market_structure, calculate_fvg, 
    calculate_liquidity_zones, calculate_order_blocks
)

def analyze_smart_money(df: pd.DataFrame) -> dict:
    """
    SMC Intelligence: BOS, CHOCH, FVG, and Liquidity Context.
    """
    struct = calculate_market_structure(df)
    fvg = calculate_fvg(df)
    liq = calculate_liquidity_zones(df)
    blocks = calculate_order_blocks(df)
    
    # 1. Market Structure Context (BOS/CHOCH)
    # যদি প্রাইস সুইং হাই ব্রেক করে তাহলে BOS, আর যদি প্রিভিয়াস সুইং লো ব্রেক করে তাহলে CHOCH
    price = df['close']
    last_high = df['high'].rolling(10).max().iloc[-1]
    last_low = df['low'].rolling(10).min().iloc[-1]
    
    bos_bullish = price.iloc[-1] > last_high
    choch_bearish = price.iloc[-1] < last_low # Trend reversal context
    
    # 2. FVG Analysis (Status of Imbalance)
    fvg_status = "BULLISH_IMBALANCE" if fvg['bullish_fvg'].iloc[-1] else \
                 ("BEARISH_IMBALANCE" if fvg['bearish_fvg'].iloc[-1] else "NONE")
    
    # 3. Liquidity Status
    liquidity_threat = "EQUAL_HIGHS_SWEEP_RISK" if liq['equal_high'].iloc[-1] else \
                       ("EQUAL_LOWS_SWEEP_RISK" if liq['equal_low'].iloc[-1] else "CLEAR")
    
    # 4. Order Block Analysis (Mitigation Logic)
    # ব্লকের কাছে প্রাইস আসলে সেটাকে 'Mitigation' বলে
    is_mitigated = (abs(price.iloc[-1] - last_low) < (df['high'].iloc[-1] - df['low'].iloc[-1]))
    block_status = "MITIGATED" if (blocks['order_block'].iloc[-1] and is_mitigated) else "FRESH"
    
    return {
        "structure": {
            "bos": bos_bullish,
            "choch": choch_bearish,
            "bias": "BULLISH" if bos_bullish else "BEARISH"
        },
        "fvg": {
            "status": fvg_status,
            "needs_mitigation": fvg_status != "NONE"
        },
        "liquidity": liquidity_threat,
        "order_block": {
            "validity": block_status
        },
        "metrics": {
            "structure_integrity": struct['swing_high'].sum() + struct['swing_low'].sum()
        }
    }
