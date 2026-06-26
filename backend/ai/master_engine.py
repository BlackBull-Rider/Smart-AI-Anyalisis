import pandas as pd
import logging
from datetime import datetime

# Import every single module to ensure full coverage
from backend.analyzers import (trend_analyzer, momentum_analyzer, volume_analyzer, 
                               volatility_analyzer, smart_money_analyzer, 
                               institutional_analyzer, fundamental_analyzer)
from backend.engines import (compounder_engine, swing_engine, long_term_engine)
from backend.decision import (market_regime_engine, entry_engine, exit_engine, 
                             stoploss_engine, target_engine, risk_engine, 
                             reward_engine, holding_engine, allocation_engine, 
                             position_size_engine, confidence_engine, 
                             conviction_engine, recommendation_engine, 
                             explanation_engine)

def generate_final_master_report(df: pd.DataFrame, account_balance: float) -> dict:
    """
    Enterprise-Grade Master AI: Orchestrator.
    Maps every single requirement to a specific engine execution.
    """
    # 1. INITIALIZATION & GATEKEEPING
    try:
        regime = market_regime_engine.decide_market_permission(df)
        if regime['trade_permission'] != "ALLOWED":
            return {"STATUS": "BLOCKED", "REASON": regime['reasoning']}

        # 2. DATA COLLECTION: ANALYSIS LAYER (The "What")
        analysis_data = {
            "Trend": trend_analyzer.analyze_trend(df),
            "Momentum": momentum_analyzer.analyze_momentum(df),
            "Volume": volume_analyzer.analyze_volume(df),
            "Volatility": volatility_analyzer.analyze_volatility(df),
            "SmartMoney": smart_money_analyzer.analyze_smart_money(df),
            "Institutional": institutional_analyzer.analyze_institutional(df),
            "Fundamental": fundamental_analyzer.analyze_fundamentals(df)
        }
        
        # 3. SCORING LAYER (The "Quality")
        scoring_data = {
            "Compounder": compounder_engine.calculate_compounder_score(df),
            "Swing": swing_engine.calculate_swing_score(df),
            "LongTerm": long_term_engine.calculate_long_term_score(df)
        }
        # Composite Master Score
        master_score = (scoring_data['Compounder']['compounder_score'] + 
                        scoring_data['Swing']['swing_score'] + 
                        scoring_data['LongTerm']['long_term_score']) / 3
        
        # 4. DECISION LAYER (The "Action Plan")
        entry = entry_engine.get_entry_strategy(df)
        if entry['action'] != "EXECUTE_ENTRY":
            return {"STATUS": "WAIT", "REASON": entry.get('reason', 'NO_ENTRY_SIGNAL')}
            
        sl = stoploss_engine.calculate_stoploss_levels(df, entry['entry_price'])
        targets = target_engine.calculate_targets(entry['entry_price'], sl['initial_sl'])
        
        # 5. RISK & MANAGEMENT LAYER (The "Safety")
        risk = risk_engine.calculate_risk_metrics(df, entry['entry_price'], sl['initial_sl'], account_balance)
        reward = reward_engine.calculate_reward_metrics(entry['entry_price'], sl['initial_sl'], 10)
        pos_size = position_size_engine.calculate_position_size(account_balance, 0.02, entry['entry_price'], sl['initial_sl'])
        allocation = allocation_engine.calculate_allocation(account_balance, [], "GENERAL", master_score)
        holding = holding_engine.calculate_holding_strategy(df)
        
        # 6. CONVICTION & FINAL AI VERDICT (The "Brain")
        conf = confidence_engine.calculate_confidence_score({'trend': analysis_data['Trend']['trend_score'], 'momentum': analysis_data['Momentum']['momentum_score']})
        conv = conviction_engine.calculate_conviction_score(entry, risk, reward)
        rec = recommendation_engine.get_recommendation(master_score, conf['confidence_score'])
        expl = explanation_engine.generate_explanation({'DECISION': rec['recommendation'], 'CONVICTION': conv['rating'], 'RISK_MANAGEMENT': risk}, {'trend': analysis_data['Trend']['trend_score']})

        # 7. FINAL ENTERPRISE REPORT CONSTRUCTION (The "Final Output")
        final_report = {
            "metadata": {"timestamp": datetime.now().isoformat(), "asset": "NIFTY/STOCK"},
            "market_intelligence": analysis_data,
            "scoring_matrix": scoring_data,
            "risk_and_confidence": {"Risk": risk, "Reward": reward, "Confidence": conf, "Conviction": conv},
            "execution_plan": {
                "EntryZone": entry['entry_zone'], 
                "EntryPrice": entry['entry_price'], 
                "StopLoss": sl['initial_sl'], 
                "Targets": targets['targets']
            },
            "position_management": {
                "PositionSize": pos_size['quantity'], 
                "CapitalAllocation": allocation['allocated_capital'], 
                "HoldingPeriod": holding['expected_duration_days']
            },
            "final_decision": {
                "Recommendation": rec['recommendation'],
                "ReasonToBuy": expl['reason'],
                "AI_Summary": expl['ai_summary'],
                "WarningSignals": "NONE" if rec['is_actionable'] else "MONITOR_RISK",
                "ActionPlan": rec['action_plan']
            }
        }
        
        return final_report

    except Exception as e:
        logging.error(f"SYSTEM CRITICAL ERROR: {e}")
        return {"STATUS": "SYSTEM_CRASH", "LOG": str(e)}
