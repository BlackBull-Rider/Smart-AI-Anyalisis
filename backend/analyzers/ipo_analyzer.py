import pandas as pd
import numpy as np

def analyze_ipo(subscription_data: dict, gmp: float, anchor_quality: str) -> dict:
    """
    IPO Intelligence: Subscription ratios, Listing strength, and Opportunity score.
    
    subscription_data format: {'qib': float, 'hni': float, 'retail': float}
    gmp: Grey Market Premium in percentage
    anchor_quality: 'HIGH', 'MEDIUM', 'LOW'
    """
    
    # 1. Subscription Analysis (Weighted Score)
    # Smart Money (QIB) কে সবচেয়ে বেশি গুরুত্ব দেওয়া হয়
    qib_score = subscription_data.get('qib', 1) * 0.5
    hni_score = subscription_data.get('hni', 1) * 0.3
    retail_score = subscription_data.get('retail', 1) * 0.2
    
    total_sub_score = qib_score + hni_score + retail_score
    sub_strength = "AGGRESSIVE" if total_sub_score > 50 else "MODERATE" if total_sub_score > 10 else "WEAK"
    
    # 2. Listing Strength (GMP & Subscription Correlation)
    # GMP 20% এর বেশি হলে স্ট্রং লিস্টিং গেইন পসিবল
    listing_gain = "HIGH_PROBABILITY" if (gmp > 20 and total_sub_score > 20) else "NEUTRAL"
    
    # 3. IPO Quality (Anchor Investor Analysis)
    anchor_map = {'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.1}
    quality_score = anchor_map.get(anchor_quality, 0)
    
    # 4. Final Verdict
    # একটি কনফ্লুয়েন্স স্কোর (০ থেকে ১০০)
    confluence_score = min((total_sub_score * 0.6) + (gmp * 0.3) + (quality_score * 10), 100)
    
    return {
        "subscription_strength": sub_strength,
        "listing_opportunity": listing_gain,
        "anchor_confidence": anchor_quality,
        "listing_probability_score": round(confluence_score, 2),
        "verdict": "APPLY" if confluence_score > 70 else "WATCH_LIST" if confluence_score > 40 else "AVOID"
    }
