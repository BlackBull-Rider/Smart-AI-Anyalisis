import math

def calculate_position_size(account_balance: float, risk_per_trade_pct: float, entry_price: float, stop_loss: float) -> dict:
    """
    The Math Shield: Calculates exact quantity to trade based on Risk Management rules.
    Prevents over-leveraging and keeps risk per trade consistent.
    """
    # 1. Edge Case Handling (Division by Zero prevention)
    if entry_price == stop_loss:
        return {"error": "Entry and Stop Loss cannot be the same.", "quantity": 0}
        
    # 2. Risk Amount Calculation (Total $ allowed to lose in this trade)
    risk_amount = account_balance * risk_per_trade_pct
    
    # 3. Risk Per Share (Distance between Entry and SL)
    risk_per_share = abs(entry_price - stop_loss)
    
    # 4. Quantity Calculation (Rounding down to ensure we never exceed risk limit)
    quantity = math.floor(risk_amount / risk_per_share)
    
    # 5. Exposure Calculation
    total_exposure = quantity * entry_price
    
    # 6. Risk-to-Balance Ratio Check
    # যদি ট্রেডটি ক্যাপিটালের ৫% এর বেশি এক্সপোজার নেয়, তবে সাবধান হতে হবে
    exposure_pct = (total_exposure / account_balance) * 100
    
    return {
        "quantity": int(quantity),
        "risk_amount": round(risk_amount, 2),
        "total_exposure": round(total_exposure, 2),
        "exposure_pct_of_capital": round(exposure_pct, 2),
        "is_safe": exposure_pct <= 50, # 50% এর বেশি এক্সপোজার সাধারণত ঝুঁকিপূর্ণ
        "meta": {
            "risk_per_share": round(risk_per_share, 2),
            "risk_rule": f"{risk_per_trade_pct*100}%_account_risk"
        }
    }
