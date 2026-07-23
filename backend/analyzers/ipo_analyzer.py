import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==============================================================================
# DYNAMIC INSTITUTIONAL CONFIGURATION
# ==============================================================================

IPO_CONFIG = {
    "scalars": {
        "bayesian_damping_power": 0.45,
        "base_prior": 0.5,
        "risk_free_rate": 0.07,
        "equity_risk_premium": 0.05,
        "terminal_growth": 0.04,
        "ideal_qib_subscription": 50.0,
        "ideal_hni_subscription": 30.0,
        "monte_carlo_sims": 5000
    },
    "thresholds": {
        "roce_excellent": 20.0, "roce_good": 12.0,
        "debt_equity_safe": 0.5, "debt_equity_risk": 1.5,
        "gmp_premium_excellent": 30.0, "gmp_premium_good": 10.0,
        "anchor_concentration_max": 40.0,
        "sovereign_mf_min_pct": 30.0
    },
    "base_weights": {
        "valuation_pe": 0.25, "valuation_ev_ebitda": 0.25,
        "valuation_ev_sales": 0.20, "valuation_dcf": 0.30,
        "composite_quality": 0.25, "composite_demand": 0.25,
        "composite_listing": 0.15, "composite_opportunity": 0.20,
        "composite_risk": 0.15
    },
    "reliabilities": {
        "financials": 0.95, "market_data": 0.98, "subscription": 0.99,
        "anchor_data": 0.92, "gmp_data": 0.70, "valuation_models": 0.85
    },
    "penalties": {
        "missing_core_feature": 4.0,
        "negative_equity": 30.0,
        "promoter_dump": 40.0,
        "gmp_collapse": 25.0
    },
    "action_thresholds": {
        "strong_apply": 80.0, "apply": 60.0, "watch": 45.0, "avoid": 25.0
    }
}

# ==============================================================================
# SCHEMAS
# ==============================================================================

class EvidenceItem(TypedDict):
    category: str; feature: str; weight: float; polarity: int
    reliability: float; likelihood_ratio: float; explanation: str

class IPOQualityResult(TypedDict):
    issue_quality: float; business_quality: float; management_quality: float
    financial_quality: float; capital_allocation: float; evidence: List[EvidenceItem]

class GMPAnalysisResult(TypedDict):
    gmp_premium: float; gmp_momentum: float; gmp_reliability: float
    gmp_persistence: float; gmp_premium_score: float; evidence: List[EvidenceItem]

class ListingStrengthResult(TypedDict):
    listing_gain: float; opening_auction_strength: float; vwap_premium: float
    intraday_high_retention: float; closing_strength: float; relative_volume: float
    evidence: List[EvidenceItem]

class SubscriptionAnalysisResult(TypedDict):
    overall_subscription: float; oversubscription_velocity: float; last_day_spike: float
    qib_score: float; hni_score: float; retail_score: float
    category_concentration: float; smart_money_participation: float; evidence: List[EvidenceItem]

class AnchorAnalysisResult(TypedDict):
    anchor_allocation: float; top_anchor_concentration: float; domestic_vs_foreign_ratio: float
    mf_sovereign_pct: float; lockin_expiry_impact: float; anchor_quality_score: float
    evidence: List[EvidenceItem]

class ListingOpportunityResult(TypedDict):
    dcf_value: float; relative_pe_value: float; ev_ebitda_value: float; ev_sales_value: float
    weighted_fair_value: float; expected_upside: float; margin_of_safety: float
    p10_listing_estimate: float; p50_listing_estimate: float; p90_listing_estimate: float
    evidence: List[EvidenceItem]

class RiskAnalysisResult(TypedDict):
    sector_risk: float; market_sentiment_risk: float; grey_market_collapse_risk: float
    float_turnover_risk: float; post_lockin_supply_shock: float; valuation_dispersion: float
    macro_liquidity_risk: float; overall_ipo_risk: float; evidence: List[EvidenceItem]

class CompositeScoreResult(TypedDict):
    quality_score: float; demand_score: float; opportunity_score: float; listing_score: float
    risk_score: float; base_weighted_score: float; penalty_deduction: float
    bayesian_confidence_multiplier: float; overall_ipo_score: float; investment_grade: str

class SummaryResult(TypedDict):
    recommended_action: str; why_apply: List[str]; why_avoid: List[str]
    expected_listing_range: str; expected_1_month_return: float; expected_6_month_return: float
    expected_cagr: float; probability_of_success: float; probability_of_failure: float

class IPOAnalysisResult(TypedDict):
    ipo_quality: IPOQualityResult; gmp_analysis: GMPAnalysisResult
    subscription_analysis: SubscriptionAnalysisResult; anchor_analysis: AnchorAnalysisResult
    listing_strength: ListingStrengthResult; listing_opportunity: ListingOpportunityResult
    risk_analysis: RiskAnalysisResult; composite_score: CompositeScoreResult; summary: SummaryResult

# ==============================================================================
# MAIN ENGINE
# ==============================================================================

class IPOAnalyzer:
    # Segregated Schema for Robust Integrity Scoring
    CORE_SCHEMA = [
        'issue_price', 'ipo_size', 'market_cap', 'float_shares', 'gmp',
        'subscription_qib', 'subscription_hni', 'subscription_retail',
        'roe', 'roce', 'roic', 'eps', 'book_value', 'pe_ratio', 'sector_pe',
        'sales', 'net_profit', 'debt_to_equity', 'promoter_holding_pre', 'promoter_holding_post'
    ]

    ADVANCED_SCHEMA = [
        'gmp_momentum', 'gmp_reliability', 'gmp_persistence',
        'subscription_velocity', 'last_day_spike', 'category_concentration',
        'anchor_allocation', 'top_anchor_concentration', 'domestic_vs_foreign_ratio',
        'mf_anchor_pct', 'sovereign_anchor_pct', 'lockin_days',
        'ebitda', 'sector_ev_ebitda', 'sector_ev_sales', 'sales_growth', 'profit_growth',
        'operating_cash_flow', 'free_cash_flow', 'current_ratio', 'cash_equivalents', 'total_debt',
        'listing_price', 'current_price', 'vwap', 'intraday_high', 'opening_auction_volume',
        'listing_volume', 'close', 'volume', 'beta', 'market_sentiment_score', 'sector_risk_score',
        'macro_liquidity_index', 'shares_outstanding' # <-- FIXED: Added shares_outstanding
    ]

    EXPECTED_SCHEMA = CORE_SCHEMA + ADVANCED_SCHEMA # <-- FIXED: For Dynamic Registry

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or IPO_CONFIG

    def _safe_div(self, num: float, den: float, default: float = 0.0) -> float:
        return num / den if den and not math.isnan(den) and den != 0 else default

    def _normalize(self, value: float, min_val: float, max_val: float, invert: bool = False) -> float:
        if math.isnan(value) or math.isinf(value) or max_val == min_val:
            return 50.0
        clamped = np.clip(value, min_val, max_val)
        score = ((clamped - min_val) / (max_val - min_val)) * 100.0
        return 100.0 - score if invert else score

    def _stable_sigmoid(self, log_odds: float) -> float:
        if log_odds >= 0:
            return 1.0 / (1.0 + math.exp(-log_odds))
        return math.exp(log_odds) / (1.0 + math.exp(log_odds))

    def _logistic_calibration(self, prob: float) -> float:
        k = 6.0
        midpoint = 0.5
        logistic_val = 1.0 / (1.0 + math.exp(-k * (prob - midpoint)))
        return 0.5 + (0.7 * logistic_val)

    def _bayesian_aggregation(self, evidence: List[EvidenceItem]) -> Tuple[float, float]:
        prior = self.config['scalars']['base_prior']
        log_prior = math.log(prior / (1.0 - prior))
        cat_log_lrs = defaultdict(list)

        for e in evidence:
            val = e['reliability'] * math.log(max(e['likelihood_ratio'], 1e-5))
            cat_log_lrs[e['category']].append(val)

        damped_log_lr = 0.0
        d_power = self.config['scalars']['bayesian_damping_power']

        for cat, vals in cat_log_lrs.items():
            damping = 1.0 / (len(vals) ** d_power) if vals else 1.0
            damped_log_lr += sum(vals) * damping

        log_post = log_prior + damped_log_lr
        prob = self._stable_sigmoid(log_post)
        return prob, log_post

    def _validate_integrity(self, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        missing_core = [f for f in self.CORE_SCHEMA if f not in df.columns]
        penalty = len(missing_core) * self.config['penalties']['missing_core_feature']
        integrity = np.clip(100.0 - penalty, 0.0, 100.0)

        latest = df.iloc[-1] if not df.empty else pd.Series()
        l1 = {f: float(latest.get(f, np.nan)) for f in self.EXPECTED_SCHEMA}
        return integrity, l1

    def analyze(self, df, corporate_actions=None, **kwargs) -> IPOAnalysisResult:
        if df.empty:
            raise ValueError("IPOAnalyzer: Empty DataFrame")

        integrity_score, l1 = self._validate_integrity(df)
        
        qual = self._analyze_quality(l1)
        gmp_res = self._analyze_gmp(l1)
        sub = self._analyze_subscription(l1)
        anc = self._analyze_anchors(l1)
        lst = self._analyze_listing_strength(l1)
        opp = self._analyze_valuation_multi_model(l1, qual, sub, gmp_res)
        risk = self._analyze_risk(l1, gmp_res, anc, sub, opp)

        all_ev = qual['evidence'] + gmp_res['evidence'] + sub['evidence'] + anc['evidence'] + lst['evidence'] + opp['evidence'] + risk['evidence']
        comp = self._generate_composite_score(qual, sub, opp, lst, risk, all_ev, integrity_score)
        summary = self._generate_summary(comp, opp, all_ev, l1)

        return {
            "ipo_quality": qual,
            "gmp_analysis": gmp_res,
            "subscription_analysis": sub,
            "anchor_analysis": anc,
            "listing_strength": lst,
            "listing_opportunity": opp,
            "risk_analysis": risk,
            "composite_score": comp,
            "summary": summary
        }

    # ---------------------------------------------------------
    # 1. IPO QUALITY 
    # ---------------------------------------------------------
    def _analyze_quality(self, l1: Dict[str, float]) -> IPOQualityResult:
        roic = np.nan_to_num(l1['roic'], nan=0.0)
        debt = np.nan_to_num(l1['debt_to_equity'], nan=0.0)
        cfo = np.nan_to_num(l1['operating_cash_flow'], nan=0.0)
        pat = np.nan_to_num(l1['net_profit'], nan=1e-9)
        cfo_pat = self._safe_div(cfo, pat)
        pro_pre = np.nan_to_num(l1['promoter_holding_pre'], nan=100.0)
        pro_post = np.nan_to_num(l1['promoter_holding_post'], nan=75.0)
        retention = self._safe_div(pro_post, pro_pre) * 100.0

        th = self.config['thresholds']
        rel = self.config['reliabilities']['financials']

        fin_q = np.clip(100.0 - (debt * 20.0), 0, 100)
        biz_q = self._normalize(roic, 0, th['roce_excellent'])
        cap_q = np.clip(cfo_pat * 50.0, 0, 100)
        mgmt_q = np.clip((retention * 0.4) + (cap_q * 0.3) + (biz_q * 0.3), 0, 100)
        issue_quality = np.mean([fin_q, biz_q, cap_q, mgmt_q])

        ev = []
        if roic > th['roce_good']:
            ev.append({"category": "Quality", "feature": "ROIC", "weight": 15.0, "polarity": 1, "reliability": rel, "likelihood_ratio": 1.8, "explanation": f"Strong Capital Return (ROIC: {roic:.1f}%)"})
        if retention < 50.0:
            ev.append({"category": "Quality", "feature": "Promoter Selloff", "weight": 20.0, "polarity": -1, "reliability": rel, "likelihood_ratio": 0.2, "explanation": f"Heavy Promoter Dilution (Retained only {retention:.1f}%)"})

        return {
            "issue_quality": issue_quality, "business_quality": biz_q, 
            "management_quality": mgmt_q, "financial_quality": fin_q, 
            "capital_allocation": cap_q, "evidence": ev
        }

    # ---------------------------------------------------------
    # 2. GMP INTELLIGENCE
    # ---------------------------------------------------------
    def _analyze_gmp(self, l1: Dict[str, float]) -> GMPAnalysisResult:
        issue = np.nan_to_num(l1['issue_price'], nan=1e-9)
        gmp = np.nan_to_num(l1['gmp'], nan=0.0)
        prem = self._safe_div(gmp, issue) * 100.0 if issue > 0 else 0.0
        momentum = np.nan_to_num(l1['gmp_momentum'], nan=0.0)
        reliab = np.nan_to_num(l1['gmp_reliability'], nan=50.0)
        persist = np.nan_to_num(l1['gmp_persistence'], nan=50.0)
        score = np.clip((prem * 0.4) + (momentum * 0.3) + (reliab * 0.3), 0, 100)

        ev = []
        if prem > 20 and reliab > 70:
            ev.append({"category": "GMP", "feature": "Premium", "weight": 15.0, "polarity": 1, "reliability": 0.8, "likelihood_ratio": 1.5, "explanation": f"Reliable GMP Premium ({prem:.1f}%)"})
        elif momentum < -10:
            ev.append({"category": "GMP", "feature": "Momentum Collapse", "weight": 20.0, "polarity": -1, "reliability": 0.85, "likelihood_ratio": 0.2, "explanation": "Grey Market Premium collapsing."})

        return {"gmp_premium": prem, "gmp_momentum": momentum, "gmp_reliability": reliab, "gmp_persistence": persist, "gmp_premium_score": score, "evidence": ev}

    # ---------------------------------------------------------
    # 3. SUBSCRIPTION FLOW
    # ---------------------------------------------------------
    def _analyze_subscription(self, l1: Dict[str, float]) -> SubscriptionAnalysisResult:
        qib = np.nan_to_num(l1['subscription_qib'], nan=0.0)
        hni = np.nan_to_num(l1['subscription_hni'], nan=0.0)
        ret = np.nan_to_num(l1['subscription_retail'], nan=0.0)
        vel = np.nan_to_num(l1['subscription_velocity'], nan=0.0)
        spike = np.nan_to_num(l1['last_day_spike'], nan=0.0)
        cat_conc = np.nan_to_num(l1['category_concentration'], nan=0.0)
        overall = qib + hni + ret
        smart_money = (qib * 0.7) + (hni * 0.3)

        ev = []
        if qib > 30 and spike > 50:
            ev.append({"category": "Subscription", "feature": "Institutional Spike", "weight": 18.0, "polarity": 1, "reliability": 0.99, "likelihood_ratio": 2.5, "explanation": "Massive Institutional Bid Velocity on Day 3."})
        if overall > 0 and qib < 1:
            ev.append({"category": "Subscription", "feature": "QIB Avoidance", "weight": 25.0, "polarity": -1, "reliability": 0.99, "likelihood_ratio": 0.1, "explanation": "Smart money completely avoided the issue."})

        return {
            "overall_subscription": overall, "oversubscription_velocity": vel, 
            "last_day_spike": spike, "qib_score": qib, "hni_score": hni, 
            "retail_score": ret, "category_concentration": cat_conc, 
            "smart_money_participation": smart_money, "evidence": ev
        }

    # ---------------------------------------------------------
    # 4. ANCHOR INTELLIGENCE
    # ---------------------------------------------------------
    def _analyze_anchors(self, l1: Dict[str, float]) -> AnchorAnalysisResult:
        alloc = np.nan_to_num(l1['anchor_allocation'], nan=0.0)
        top_conc = np.nan_to_num(l1['top_anchor_concentration'], nan=100.0)
        mf_sov = np.nan_to_num(l1['mf_anchor_pct'], nan=0.0) + np.nan_to_num(l1['sovereign_anchor_pct'], nan=0.0)
        dom_vs_for = np.nan_to_num(l1['domestic_vs_foreign_ratio'], nan=1.0)
        lockin = np.nan_to_num(l1['lockin_days'], nan=30.0)
        qual = np.clip((mf_sov * 0.6) + ((100-top_conc) * 0.4), 0, 100)
        lockin_imp = 100.0 - self._normalize(lockin, 30, 90)

        ev = []
        if mf_sov > 50.0:
            ev.append({"category": "Anchor", "feature": "High Quality", "weight": 12.0, "polarity": 1, "reliability": 0.95, "likelihood_ratio": 1.6, "explanation": f"Bluechip Mutual/Sovereign Backing ({mf_sov:.1f}%)"})

        return {"anchor_allocation": alloc, "top_anchor_concentration": top_conc, "domestic_vs_foreign_ratio": dom_vs_for, "mf_sovereign_pct": mf_sov, "lockin_expiry_impact": lockin_imp, "anchor_quality_score": qual, "evidence": ev}

    # ---------------------------------------------------------
    # 5. LISTING STRENGTH
    # ---------------------------------------------------------
    def _analyze_listing_strength(self, l1: Dict[str, float]) -> ListingStrengthResult:
        if math.isnan(l1.get('listing_price', np.nan)):
            return {"listing_gain": 0.0, "opening_auction_strength": 0.0, "vwap_premium": 0.0, "intraday_high_retention": 0.0, "closing_strength": 0.0, "relative_volume": 0.0, "evidence": []}

        issue = l1['issue_price']
        listing = l1['listing_price']
        vwap = np.nan_to_num(l1['vwap'], nan=listing)
        close = l1.get('close', listing)
        high = np.nan_to_num(l1['intraday_high'], nan=listing)

        gain = self._safe_div(close - issue, issue) * 100.0
        vwap_prem = self._safe_div(vwap - listing, listing) * 100.0
        retention = self._safe_div(close - listing, high - listing) * 100.0 if high > listing else 0.0
        auc_vol = np.nan_to_num(l1['opening_auction_volume'], nan=0.0)
        rel_vol = self._safe_div(auc_vol, np.nan_to_num(l1['float_shares'], nan=1e9)) * 100.0

        ev = []
        if retention > 80.0 and gain > 10.0:
            ev.append({"category": "Listing", "feature": "Closing Strength", "weight": 15.0, "polarity": 1, "reliability": 0.99, "likelihood_ratio": 1.8, "explanation": "Sustained institutional buying into close."})

        return {"listing_gain": gain, "opening_auction_strength": rel_vol, "vwap_premium": vwap_prem, "intraday_high_retention": retention, "closing_strength": retention, "relative_volume": rel_vol, "evidence": ev}

    # ---------------------------------------------------------
    # 6. MONTE CARLO VALUATION & RANGE 
    # ---------------------------------------------------------
    def _analyze_valuation_multi_model(self, l1: Dict[str, float], qual: IPOQualityResult, sub: SubscriptionAnalysisResult, gmp: GMPAnalysisResult) -> ListingOpportunityResult:
        issue = np.nan_to_num(l1['issue_price'], nan=1e-9)
        shares = np.nan_to_num(l1.get('shares_outstanding', 1e6), nan=1e6) # FIXED: Use get with fallback
        mcap = issue * shares
        debt = np.nan_to_num(l1['total_debt'], nan=0.0)
        cash = np.nan_to_num(l1['cash_equivalents'], nan=0.0)

        eps = np.nan_to_num(l1['eps'], nan=0.0)
        sec_pe = np.nan_to_num(l1['sector_pe'], nan=15.0)
        pe_val = (eps * sec_pe) if eps > 0 else 0.0

        ebitda = np.nan_to_num(l1['ebitda'], nan=0.0)
        sec_ev_ebitda = np.nan_to_num(l1['sector_ev_ebitda'], nan=10.0)
        ev_ebitda_val = self._safe_div((ebitda * sec_ev_ebitda) - debt + cash, shares) if ebitda > 0 else 0.0

        sales = np.nan_to_num(l1['sales'], nan=0.0)
        sec_ev_sales = np.nan_to_num(l1['sector_ev_sales'], nan=2.0)
        ev_sales_val = self._safe_div((sales * sec_ev_sales) - debt + cash, shares) if sales > 0 else 0.0

        fcf = np.nan_to_num(l1['free_cash_flow'], nan=0.0)
        beta = np.nan_to_num(l1['beta'], nan=1.0)
        rf = self.config['scalars']['risk_free_rate']
        erp = self.config['scalars']['equity_risk_premium']
        coe = rf + (beta * erp)
        g = np.clip(np.nan_to_num(l1['profit_growth'], nan=5.0) / 100.0, 0, 0.15)
        dcf_val = self._safe_div((fcf * (1 + g)) / (coe - g), shares) if fcf > 0 and coe > g else 0.0

        w = self.config['base_weights']
        valid_models = []
        if pe_val > 0:
            valid_models.append((pe_val, w['valuation_pe'] * (1.0 if not math.isnan(l1.get('sector_pe', np.nan)) else 0.5)))
        if ev_ebitda_val > 0:
            valid_models.append((ev_ebitda_val, w['valuation_ev_ebitda'] * (1.0 if not math.isnan(l1.get('sector_ev_ebitda', np.nan)) else 0.5)))
        if ev_sales_val > 0:
            valid_models.append((ev_sales_val, w['valuation_ev_sales'] * (1.0 if not math.isnan(l1.get('sector_ev_sales', np.nan)) else 0.5)))
        if dcf_val > 0:
            valid_models.append((dcf_val, w['valuation_dcf']))

        fair_val = sum(v * wt for v, wt in valid_models) / sum(wt for _, wt in valid_models) if valid_models else issue
        upside = self._safe_div(fair_val - issue, issue) * 100.0
        mos = self._safe_div(fair_val - issue, fair_val) * 100.0 if fair_val > 0 else 0.0

        n_sims = self.config['scalars']['monte_carlo_sims']
        base_prem = np.nan_to_num(l1['gmp'], nan=0.0)
        
        # FIXED: Prevent scale from becoming 0 or negative
        vol_calc = issue * (0.05 + (0.10 / (math.log10(max(sub['overall_subscription'] + 2, 2.0)))))
        volatility = max(vol_calc, 1e-5) 
        
        sim_listing_prices = np.random.normal(loc=(issue + base_prem), scale=volatility, size=n_sims)
        sim_listing_prices = np.maximum(sim_listing_prices, issue * 0.5)

        p10 = float(np.percentile(sim_listing_prices, 10))
        p50 = float(np.percentile(sim_listing_prices, 50))
        p90 = float(np.percentile(sim_listing_prices, 90))

        ev = []
        if mos > 20:
            ev.append({"category": "Valuation", "feature": "Multi-Model MOS", "weight": 20.0, "polarity": 1, "reliability": 0.85, "likelihood_ratio": 1.9, "explanation": f"Deep Value via Multi-Model Framework. Fair Value: {fair_val:.2f}"})
        elif mos < -15:
            ev.append({"category": "Valuation", "feature": "Overpriced", "weight": 20.0, "polarity": -1, "reliability": 0.85, "likelihood_ratio": 0.25, "explanation": f"Aggressively Priced vs Sector. Fair Value: {fair_val:.2f}"})

        return {
            "dcf_value": dcf_val, "relative_pe_value": pe_val, "ev_ebitda_value": ev_ebitda_val, 
            "ev_sales_value": ev_sales_val, "weighted_fair_value": fair_val, "expected_upside": upside, 
            "margin_of_safety": mos, "p10_listing_estimate": p10, "p50_listing_estimate": p50, 
            "p90_listing_estimate": p90, "evidence": ev
        }

    # ---------------------------------------------------------
    # 7. INSTITUTIONAL RISK ENGINE
    # ---------------------------------------------------------
    def _analyze_risk(self, l1: Dict[str, float], gmp: GMPAnalysisResult, anc: AnchorAnalysisResult, sub: SubscriptionAnalysisResult, opp: ListingOpportunityResult) -> RiskAnalysisResult:
        sector_risk = np.nan_to_num(l1['sector_risk_score'], nan=50.0)
        sentiment = np.nan_to_num(l1['market_sentiment_score'], nan=50.0)
        float_shares = np.nan_to_num(l1['float_shares'], nan=1e6) + 1e-9
        shares = np.nan_to_num(l1.get('shares_outstanding', 1e7), nan=1e7) # FIXED
        
        float_risk = self._safe_div(float_shares, shares) * 100.0
        lockin_days = np.nan_to_num(l1.get('lockin_days', 30.0), nan=30.0)
        lockin_shock = (anc['anchor_allocation'] / float_shares) * 100.0 if lockin_days < 45 else 0.0

        implied_pe = np.nan_to_num(l1['pe_ratio'], nan=15.0)
        sec_pe = np.nan_to_num(l1['sector_pe'], nan=15.0)
        dispersion = self._safe_div(abs(implied_pe - sec_pe), sec_pe) * 100.0

        macro_liq = np.nan_to_num(l1.get('macro_liquidity_index', 50.0), nan=50.0)
        gmp_col = 100.0 if gmp['gmp_momentum'] < -15.0 else 0.0

        overall = np.clip((sector_risk * 0.15) + ((100-sentiment) * 0.2) + (gmp_col * 0.2) + (lockin_shock * 0.15) + (dispersion * 0.15) + ((100-macro_liq) * 0.15), 0, 100)

        ev = []
        if lockin_shock > 30.0:
            ev.append({"category": "Risk", "feature": "Supply Shock", "weight": 15.0, "polarity": -1, "reliability": 0.9, "likelihood_ratio": 0.2, "explanation": f"High Post-Lockin Dumping Risk ({lockin_shock:.1f}% float expansion expected)"})

        return {"sector_risk": sector_risk, "market_sentiment_risk": 100.0-sentiment, "grey_market_collapse_risk": gmp_col, "float_turnover_risk": float_risk, "post_lockin_supply_shock": lockin_shock, "valuation_dispersion": dispersion, "macro_liquidity_risk": 100.0-macro_liq, "overall_ipo_risk": overall, "evidence": ev}

    # ---------------------------------------------------------
    # 8. COMPOSITE SCORING
    # ---------------------------------------------------------
    def _generate_composite_score(self, qual, sub, opp, lst, risk, all_ev, integrity) -> CompositeScoreResult:
        w = self.config['base_weights']
        q_s = qual['issue_quality']
        d_s = np.clip(sub['smart_money_participation'] * 2, 0, 100)
        o_s = np.clip(50.0 + opp['margin_of_safety'], 0, 100)
        l_s = np.clip(50.0 + lst['listing_gain'], 0, 100)
        r_s = 100.0 - risk['overall_ipo_risk']

        base = (q_s*w['composite_quality'] + d_s*w['composite_demand'] + o_s*w['composite_opportunity'] + l_s*w['composite_listing'] + r_s*w['composite_risk'])

        prob, _ = self._bayesian_aggregation(all_ev)
        conf_mult = self._logistic_calibration(prob)

        penalty = 0.0
        if qual['financial_quality'] < 20:
            penalty += self.config['penalties']['negative_equity']
        if risk['grey_market_collapse_risk'] > 80:
            penalty += self.config['penalties']['gmp_collapse']

        final_score = np.clip((base * conf_mult) - penalty, 0, 100)

        ig = "AAA" if final_score >= 85 else "AA" if final_score >= 70 else "A" if final_score >= 60 else "BBB" if final_score >= 45 else "BB" if final_score >= 35 else "JUNK"

        return {
            "quality_score": q_s, "demand_score": d_s, "opportunity_score": o_s, "listing_score": l_s, 
            "risk_score": 100.0-r_s, "base_weighted_score": base, "penalty_deduction": penalty, 
            "bayesian_confidence_multiplier": conf_mult, "overall_ipo_score": final_score, "investment_grade": ig
        }

    # ---------------------------------------------------------
    # 9. SUMMARY
    # ---------------------------------------------------------
    def _generate_summary(self, comp, opp, all_ev, l1) -> SummaryResult:
        score = comp['overall_ipo_score']
        th = self.config['action_thresholds']

        if score >= th['strong_apply']: action = "STRONG APPLY"
        elif score >= th['apply']: action = "APPLY"
        elif score >= th['watch']: action = "WATCH"
        else: action = "AVOID"

        prob_success, _ = self._bayesian_aggregation(all_ev)
        prob_failure = 1.0 - prob_success

        sorted_ev = sorted(all_ev, key=lambda x: x['weight']*x['reliability']*abs(math.log(max(x['likelihood_ratio'], 1e-5))), reverse=True)
        why_apply = [e['explanation'] for e in sorted_ev if e['polarity'] > 0][:3]
        why_avoid = [e['explanation'] for e in sorted_ev if e['polarity'] < 0][:3]

        rng = f"₹{opp['p10_listing_estimate']:.1f} - ₹{opp['p90_listing_estimate']:.1f} (Base: ₹{opp['p50_listing_estimate']:.1f})"

        return {
            "recommended_action": action, 
            "why_apply": why_apply if why_apply else ["Neutral Framework"], 
            "why_avoid": why_avoid if why_avoid else ["No major flags"], 
            "expected_listing_range": rng, 
            "expected_1_month_return": opp['margin_of_safety'] * 0.8, 
            "expected_6_month_return": opp['margin_of_safety'] * 1.5, 
            "expected_cagr": l1.get('profit_growth', 0.0), 
            "probability_of_success": prob_success * 100.0, 
            "probability_of_failure": prob_failure * 100.0
        }
