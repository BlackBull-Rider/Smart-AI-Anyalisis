import pandas as pd
import numpy as np

def analyze_institutional_flow(inst_df: pd.DataFrame) -> dict:
    """
    Institutional Intelligence: FII/DII Net Flow, Delivery, and Block Deals.
    inst_df তে থাকতে হবে: [date, fii_net, dii_net, delivery_pct, bulk_block_volume]
    """
    last = inst_df.iloc[-1]
    
    # 1. FII/DII Flow Analysis
    total_flow = last['fii_net'] + last['dii_net']
    market_sentiment = "BULLISH_INSTITUTIONAL" if total_flow > 0 else "BEARISH_INSTITUTIONAL"
    
    # 2. Delivery Analysis (The Smart Money Proxy)
    # ডেলিভারি > 50% মানে ভালো কনভিকশন বা একুমুলেশন
    delivery_status = "ACCUMULATION" if last['delivery_pct'] > 50 else \
                      ("DISTRIBUTION" if last['delivery_pct'] < 30 else "NEUTRAL")
    
    # 3. Block/Bulk Deal Alert
    # যদি ব্লকের ভলিউম ২০ দিনের অ্যাভারেজের চেয়ে অনেক বেশি হয়
    avg_block = inst_df['bulk_block_volume'].rolling(20).mean().iloc[-1]
    is_block_deal = last['bulk_block_volume'] > (avg_block * 2)
    
    # 4. Promoter Tracking (Static Check)
    # এটা শুধু এলার্ট দেওয়ার জন্য যে প্রমোটার কি সেল করছে কি না
    promoter_alert = "WARNING_PROMOTER_SELLING" if last.get('promoter_sold', 0) > 0 else "SAFE"
    
    return {
        "institutional_bias": market_sentiment,
        "delivery_signal": delivery_status,
        "block_deal_alert": is_block_deal,
        "promoter_status": promoter_alert,
        "metrics": {
            "net_flow": total_flow,
            "delivery_percentage": last['delivery_pct']
        }
    }
