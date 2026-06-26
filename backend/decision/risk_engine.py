import pandas as pd
from backend.analyzers.volatility_analyzer import analyze_volatility

def calculate_risk_metrics(df: pd.DataFrame, entry_price: float, stop_loss: float, account_balance: float, risk_per_trade_pct: float = 0.02) -> dict:
    """
    The Insurance Policy: Calculates Capital Risk, Position Sizing, and Volatility Risk.
    
    Args:
        risk_per_trade_pct: এক ট্রেডে কত শতাংশ রিস্ক নিতে চাও (ডিফল্ট ২%)
    """
    # 1. Gather Volatility Data
    v_data = analyze_volatility(df)
    
    # 2. Trade Risk Calculation
    # রিস্ক কত টাকা (Entry - SL)
    risk_per_share = abs(entry_price - stop_loss)
    risk_percentage = risk_per_share / entry_price
    
    # 3. Position Sizing (The Core of Risk Management)
    # কতগুলো শেয়ার কিনবে? (Kelly-lite approach)
    max_risk_amount = account_balance * risk_per_trade_pct
    position_size = max_risk_amount / risk_per_share
    
    # 4. Drawdown & Capital Risk
    # ড্রডাউন রিস্ক = হাই ভোল্যাটিলিটি মার্কেট + বড় স্টপ লস
    drawdown_risk = "HIGH" if (v_data['regime'] == "HIGH_VOLATILITY" and risk_percentage > 0.05) else "MANAGED"
    
    # 5. Decision: Is this trade safe?
    is_safe = (risk_percentage < 0.10) # যদি রিস্ক ১০% এর বেশি হয়, তবেই 'Unsafe'
    
    return {
        "risk_score": round(risk_percentage * 100, 2), # ট্রেড রিস্ক %
        "position_size": round(position_size, 2), # কতগুলো শেয়ার কিনতে হবে
        "max_risk_amount": round(max_risk_amount, 2), # রিস্ক নেওয়া টাকার পরিমাণ
        "drawdown_risk": drawdown_risk,
        "is_safe": is_safe,
        "meta": {
            "volatility_regime": v_data['regime'],
            "risk_per_share": risk_per_share
        }
    }
