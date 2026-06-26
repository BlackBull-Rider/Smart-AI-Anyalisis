import pandas as pd
import logging
from datetime import datetime

# Layer 2: Analysis
from backend.analyzers import (
    trend_analyzer, momentum_analyzer, volume_analyzer, 
    volatility_analyzer, smart_money_analyzer, 
    institutional_analyzer, fundamental_analyzer
)

# Layer 3: Scoring
from backend.engines import (
    compounder_engine, swing_engine, long_term_engine,
    trend_engine, momentum_engine, volume_engine, 
    volatility_engine, smart_money_engine, institutional_engine, fundamental_engine
)

# Layer 4: Decision
from backend.decision import (
    market_regime_engine, entry_engine, exit_engine, 
    stoploss_engine, target_engine, risk_engine, 
    reward_engine, holding_engine, allocation_engine, 
    position_size_engine, confidence_engine, 
    conviction_engine, recommendation_engine, explanation_engine
)

def generate_final_master_report(df: pd.DataFrame, account_balance: float) -> dict:
    """
    Enterprise Master AI: The Uncompressed Version.
    This Engine ensures that every single variable from the Analysis Layer 
    to the Execution Layer is calculated and reported explicitly.
    """
    try:
        # 1. MARKET REGIME CHECK (Gatekeeping)
        regime = market_regime_engine.decide_market_permission(df)
        if regime['trade_permission'] != "ALLOWED":
            return {"STATUS": "BLOCKED", "REASON": regime['reasoning'], "SYSTEM": "IDLE"}

        # 2. ANALYSIS LAYER (Layer 2)
        analysis_data = {
            "Trend": trend_analyzer.analyze_trend(df),
            "Momentum": momentum_analyzer.analyze_momentum(df),
            "Volume": volume_analyzer.analyze_volume(df),
            "Volatility": volatility_analyzer.analyze_volatility(df),
            "SmartMoney": smart_money_analyzer.analyze_smart_money(df),
            "Institutional": institutional_analyzer.analyze_institutional(df),
            "Fundamental": fundamental_analyzer.analyze_fundamentals(df)
        }
        
        # 3. SCORING LAYER (Layer 3)
        scoring_data = {
            "Compounder": compounder_engine.calculate_compounder_score(df),
            "Swing": swing_engine.calculate_swing_score(df),
            "LongTerm": long_term_engine.calculate_long_term_score(df),
            "TrendScore": trend_engine.calculate_trend_score(df),
            "MomentumScore": momentum_engine.calculate_momentum_score(df),
            "VolumeScore": volume_engine.calculate_volume_score(df),
            "VolatilityScore": volatility_engine.calculate_volatility_score(df)
        }
        # Final Master Score (Weighted Logic)
        master_score = (scoring_data['Compounder']['score'] * 0.3 + 
                        scoring_data['Swing']['score'] * 0.25 + 
                        scoring_data['LongTerm']['score'] * 0.15 +
                        scoring_data['TrendScore']['score'] * 0.1 +
                        scoring_data['MomentumScore']['score'] * 0.1 +
                        scoring_data['VolumeScore']['score'] * 0.05 +
                        scoring_data['VolatilityScore']['score'] * 0.05)
        
        # 4. DECISION & EXECUTION LAYER (Layer 4)
        entry = entry_engine.get_entry_strategy(df)
        if entry['action'] != "EXECUTE_ENTRY":
            return {"STATUS": "WAIT", "REASON": entry.get('reason', 'NO_ENTRY_SIGNAL')}
            
        sl = stoploss_engine.calculate_stoploss_levels(df, entry['entry_price'])
        targets = target_engine.calculate_targets(entry['entry_price'], sl['initial_sl'])
        
        # 5. RISK & MANAGEMENT LAYER
        risk = risk_engine.calculate_risk_metrics(df, entry['entry_price'], sl['initial_sl'], account_balance)
        reward = reward_engine.calculate_reward_metrics(entry['entry_price'], sl['initial_sl'], 10)
        conf = confidence_engine.calculate_confidence_score(scoring_data)
        conv = conviction_engine.calculate_conviction_score(entry, risk, reward)
        pos = position_size_engine.calculate_position_size(account_balance, 0.02, entry['entry_price'], sl['initial_sl'])
        alloc = allocation_engine.calculate_allocation(account_balance, [], "GENERAL", master_score)
        holding = holding_engine.calculate_holding_strategy(df)
        
        # 6. VERDICT LAYER
        rec = recommendation_engine.get_recommendation(master_score, conf['confidence_score'])
        expl = explanation_engine.generate_explanation({'DECISION': rec['recommendation'], 'CONVICTION': conv['rating'], 'RISK_MANAGEMENT': risk}, analysis_data['Trend'])

        # 7. FINAL ENTERPRISE REPORT
        return {
            "metadata": {"timestamp": datetime.now().isoformat(), "version": "PRO-1.0"},
            "analysis_layer": analysis_data,
            "scoring_matrix": scoring_data,
            "decision_metrics": {
                "MasterScore": master_score,
                "Risk": risk,
                "Reward": reward,
                "Confidence": conf,
                "Conviction": conv
            },
            "execution_plan": {
                "EntryZone": entry['entry_zone'],
                "EntryPrice": entry['entry_price'],
                "StopLoss": sl['initial_sl'],
                "Targets": targets['targets']
            },
            "management_layer": {
                "PositionSize": pos['quantity'],
                "CapitalAllocation": alloc['allocated_capital'],
                "HoldingPeriod": holding['expected_duration_days']
            },
            "final_verdict": {
                "Recommendation": rec['recommendation'],
                "AI_Reasoning": expl['ai_summary'],
                "WarningSignals": "NONE" if rec['is_actionable'] else "HIGH_RISK_AVOID"
            }
        }

    except Exception as e:
        logging.error(f"CRITICAL MASTER AI FAILURE: {e}")
        return {"STATUS": "SYSTEM_FAILURE", "ERROR": str(e)}
