import pandas as pd

def calculate_allocation(total_capital: float, current_holdings: list, new_trade_sector: str, risk_score: float) -> dict:
    """
    The Capital Guard: Manages Portfolio, Sector, and Cash Allocation.
    
    current_holdings: List of dicts e.g., [{'sector': 'BANKING', 'value': 20000}, ...]
    risk_score: 0-100 (Based on system engines)
    """
    
    # 1. Cash Allocation (Reserve 20% as Dry Powder)
    cash_buffer_pct = 0.20
    available_capital = total_capital * (1 - cash_buffer_pct)
    
    # 2. Sector Allocation Check
    # কোনো সেক্টরে ২৫% এর বেশি এক্সপোজার রাখা যাবে না
    sector_cap = 0.25 * total_capital
    current_sector_exposure = sum([h['value'] for h in current_holdings if h['sector'] == new_trade_sector])
    
    remaining_sector_capacity = max(0, sector_cap - current_sector_exposure)
    
    # 3. Portfolio Allocation (Dynamic Sizing)
    # Risk Score অনুযায়ী অ্যালোকেশন (Score 80+ হলে ফুল ক্যাপাসিটি, কম হলে হাফ)
    base_allocation = (available_capital * 0.10) # ট্রেড প্রতি ১০% এর বেশি নয়
    dynamic_scaling = min(risk_score / 100, 1.0)
    
    final_allocation = min(base_allocation * dynamic_scaling, remaining_sector_capacity)
    
    # 4. Decision
    is_allowed = final_allocation > (total_capital * 0.02) # যদি বরাদ্দ ২% এর কম হয়, ট্রেড করার দরকার নেই
    
    return {
        "allocated_capital": round(final_allocation, 2),
        "cash_buffer_retained": round(total_capital * cash_buffer_pct, 2),
        "sector_headroom": round(remaining_sector_capacity, 2),
        "is_trade_allowed": is_allowed,
        "diversification_score": round((1 - (current_sector_exposure / total_capital)) * 100, 2)
    }
