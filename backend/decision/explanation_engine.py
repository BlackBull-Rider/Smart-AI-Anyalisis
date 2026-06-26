import pandas as pd

def generate_explanation(decision_report: dict, scores: dict) -> dict:
    """
    The Translator: Turns complex engine data into human-readable insights.
    """
    
    # 1. Buy/Sell Reasoning
    if decision_report.get('DECISION') == "BUY":
        reason = f"Trend strength at {scores['trend']}% and Smart Money inflow detected. The trade meets the {decision_report.get('CONVICTION')} criteria."
    elif decision_report.get('DECISION') == "SELL":
        reason = "Trend exhaustion detected or Stop Loss threshold breached. Protecting capital based on Volatility-Adjusted Exit strategy."
    else:
        reason = f"Market conditions are currently {decision_report.get('reason', 'UNSTABLE')}. Maintaining cash position for safety."

    # 2. Risk Context
    risk_exp = f"Risk is managed at {decision_report['RISK_MANAGEMENT']['risk_amount']} per trade. Exposure is within acceptable limits ({decision_report['POSITION_SIZE']} units)."
    
    # 3. Confidence Context
    conf_exp = f"Confidence level is {decision_report['SIGNAL_RELIABILITY']}. Strategy aligned with {decision_report['CONVICTION']} metrics."
    
    # 4. Final AI Summary
    summary = f"SUMMARY: {decision_report['DECISION']} signal on this asset. Rationale: {reason} | Risk: {risk_exp} | Confidence: {conf_exp}"
    
    return {
        "reason": reason,
        "risk_explanation": risk_exp,
        "confidence_explanation": conf_exp,
        "ai_summary": summary
    }
