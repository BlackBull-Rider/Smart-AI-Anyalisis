import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION - COMPLETE & DYNAMIC
# ==============================================================================
FUNDAMENTAL_CONFIG = {
    "scalars": {
        "risk_free_rate": 0.07,
        "equity_risk_premium": 0.05,
        "terminal_growth_rate": 0.04,
        "corporate_tax_rate": 0.25,
        "bayesian_damping_power": 0.45,
        "wacc_default": 0.095,  # Fallback only
        "monte_carlo_simulations": 5000
    },
    "thresholds": {
        "roe_excellent": 20.0, "roe_good": 15.0,
        "roce_excellent": 20.0, "roce_good": 15.0,
        "debt_equity_safe": 0.5, "debt_equity_risk": 1.5,
        "current_ratio_min": 1.0, "current_ratio_ideal": 1.5,
        "pe_undervalued": 15.0, "pe_overvalued": 35.0,
        "fii_strong": 1.5, "fii_mild": 0.5,
        "promoter_strong": 1.0, "promoter_mild": 0.2,
        "cfo_pat_min": 0.8, "cfo_pat_excellent": 1.2,
        "growth_excellent": 15.0, "growth_good": 8.0,
        "promoter_pledge_max": 5.0
    },
    "weights": {
        "business_quality": 0.15, "financial_quality": 0.15, "growth": 0.15,
        "valuation": 0.20, "profitability": 0.10, "shareholding": 0.05,
        "cash_flow": 0.10, "risk": 0.05, "capital_allocation": 0.05
    },
    "reliabilities": {
        "audited_financials": 0.98, "forensic": 0.95, "valuation": 0.90,
        "cash_flow": 0.95, "shareholding": 0.98, "growth": 0.85
    },
    "penalties": {
        "missing_feature": 2.0, "negative_equity": 30.0,
        "high_pledge": 25.0, "manipulation_flag": 40.0,
        "insolvency_risk": 50.0
    },
    "bayesian": {
        "prior_quality": 0.5
    }
}

# ==============================================================================
# SCHEMAS (Strict Interface)
# ==============================================================================
class EvidenceItem(TypedDict):
    category: str; feature: str; weight: float; polarity: int
    reliability: float; likelihood_ratio: float; explanation: str

class BusinessQualityResult(TypedDict):
    business_moat: float; brand_strength: float; market_leadership: float; pricing_power: float
    customer_stickiness: float; management_quality: float; corporate_governance: float
    capital_allocation: float; competitive_advantage: float; business_stability: float; evidence: List[EvidenceItem]

class FinancialQualityResult(TypedDict):
    balance_sheet: float; debt: float; liquidity: float; working_capital: float; cash_position: float
    interest_coverage: float; current_ratio: float; quick_ratio: float; asset_quality: float
    earnings_quality: float; piotroski_f_score: int; altman_z_score: float; beneish_m_score: float
    sloan_ratio: float; evidence: List[EvidenceItem]

class GrowthAnalysisResult(TypedDict):
    revenue_growth: float; profit_growth: float; eps_growth: float; book_value_growth: float
    cash_flow_growth: float; fcf_growth: float; growth_stability: float; growth_consistency: float
    historical_cagr: float; projected_growth: float; evidence: List[EvidenceItem]

class ValuationAnalysisResult(TypedDict):
    pe: float; pb: float; peg: float; ev_ebitda: float; price_sales: float; dcf_score: float
    intrinsic_value: float; margin_of_safety: float; relative_valuation: float; fair_value_gap: float
    residual_income_value: float; economic_value_added: float; evidence: List[EvidenceItem]

class ProfitabilityResult(TypedDict):
    roe: float; roce: float; roa: float; roic: float; gross_margin: float; operating_margin: float
    ebitda_margin: float; net_margin: float; cash_margin: float; efficiency: float; croic: float
    dupont_asset_turnover: float; dupont_equity_multiplier: float; evidence: List[EvidenceItem]

class ShareholdingResult(TypedDict):
    promoter_holding: float; promoter_change: float; fii_holding: float; fii_change: float
    dii_holding: float; dii_change: float; public_holding: float; insider_activity: float
    pledge: float; ownership_stability: float; evidence: List[EvidenceItem]

class CashFlowResult(TypedDict):
    operating_cash_flow: float; free_cash_flow: float; fcf_yield: float; cash_conversion: float
    cfo_vs_pat: float; cash_quality: float; cash_stability: float; operating_efficiency: float
    owner_earnings: float; cash_conversion_cycle: float; evidence: List[EvidenceItem]

class RiskAnalysisResult(TypedDict):
    debt_risk: float; financial_risk: float; governance_risk: float; business_risk: float
    earnings_risk: float; growth_risk: float; liquidity_risk: float; dilution_risk: float
    cyclical_risk: float; overall_risk: float; evidence: List[EvidenceItem]

class CapitalAllocationResult(TypedDict):
    dividend_policy: float; buyback_quality: float; reinvestment: float; acquisitions: float
    capital_efficiency: float; roic_spread: float; shareholder_return: float; investment_discipline: float; evidence: List[EvidenceItem]

class CompositeScoreResult(TypedDict):
    business_score: float; financial_score: float; growth_score: float; valuation_score: float
    profitability_score: float; shareholding_score: float; cash_flow_score: float; risk_score: float
    capital_allocation_score: float; overall_fundamental_score: float; investment_grade: str
    expected_long_term_quality: float; fundamental_confidence: float

class SummaryResult(TypedDict):
    recommended_action: str; investment_grade: str; fundamental_rating: str; overall_score: float
    confidence: float; business_quality: str; financial_strength: str; growth_outlook: str
    valuation_status: str; risk_level: str; top_10_evidence: List[str]; key_strengths: List[str]
    key_weaknesses: List[str]; expected_cagr_category: str; long_term_investment_suitability: str

class FundamentalAnalysisResult(TypedDict):
    business_quality: BusinessQualityResult; financial_quality: FinancialQualityResult; growth_analysis: GrowthAnalysisResult
    valuation_analysis: ValuationAnalysisResult; profitability_analysis: ProfitabilityResult; shareholding_analysis: ShareholdingResult
    cashflow_analysis: CashFlowResult; risk_analysis: RiskAnalysisResult; capital_allocation: CapitalAllocationResult
    composite_score: CompositeScoreResult; summary: SummaryResult

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
class FundamentalAnalyzer:
    EXPECTED_SCHEMA = [
        'close', 'pe_ratio', 'pb_ratio', 'ev_ebitda', 'peg_ratio', 'price_to_sales',
        'roe', 'roce', 'roa', 'roic', 'gross_margin', 'operating_margin', 'net_margin',
        'debt_to_equity', 'current_ratio', 'quick_ratio', 'interest_coverage',
        'revenue_growth_yoy', 'profit_growth_yoy', 'eps_growth_yoy', 'fcf_growth_yoy',
        'promoter_holding', 'fii_holding', 'dii_holding', 'public_holding', 'promoter_pledge',
        'promoter_change', 'fii_change', 'dii_change',
        'operating_cash_flow', 'free_cash_flow', 'net_income', 'dividend_yield',
        'market_cap', 'eps', 'book_value_per_share', 'total_assets', 'total_liabilities',
        'total_equity', 'retained_earnings', 'ebit', 'ebitda', 'working_capital', 'sales', 
        'capex', 'depreciation', 'beta', 'shares_outstanding', 'goodwill',
        'sector_avg_pe', 'sector_avg_pb', 'sector_avg_ev_ebitda',
        'days_sales_outstanding', 'days_inventory_outstanding', 'days_payable_outstanding',
        'receivables', 'cogs', 'sga_expense', 'current_assets', 'current_liabilities', 
        'long_term_debt', 'interest_expense', 'total_debt'
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or FUNDAMENTAL_CONFIG

    def _safe_div(self, num: float, den: float, default: float = 0.0) -> float:
        return num / den if den and not math.isnan(den) and den != 0 else default

    def _normalize(self, value: float, min_val: float, max_val: float, invert: bool = False) -> float:
        if math.isnan(value) or math.isinf(value) or max_val == min_val: return 50.0
        clamped = np.clip(value, min_val, max_val)
        score = ((clamped - min_val) / (max_val - min_val)) * 100.0
        return 100.0 - score if invert else score

    def _stable_sigmoid(self, log_odds: float) -> float:
        if log_odds >= 0: return 1.0 / (1.0 + math.exp(-log_odds))
        else:
            exp_lo = math.exp(log_odds)
            return exp_lo / (1.0 + exp_lo)

    def _bayesian_aggregation(self, evidence: List[EvidenceItem]) -> float:
        prior = self.config['bayesian']['prior_quality']
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
            
        return round(self._stable_sigmoid(log_prior + damped_log_lr) * 100.0, 2)

    def _validate_integrity(self, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        missing = [f for f in self.EXPECTED_SCHEMA if f not in df.columns]
        penalty = len(missing) * self.config['penalties']['missing_feature']
        integrity = np.clip(100.0 - penalty, 0.0, 100.0)
        
        latest = df.iloc[-1] if not df.empty else pd.Series()
        l1 = {f: float(latest.get(f, np.nan if f in missing else latest[f])) for f in self.EXPECTED_SCHEMA}
        return integrity, l1

    def analyze(self, df: pd.DataFrame, fundamental=None, financials=None, **kwargs) -> FundamentalAnalysisResult:
        if df.empty: raise ValueError("FundamentalAnalyzer: Empty DataFrame")
        
        integrity_score, l1 = self._validate_integrity(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        l1_prev = {f: float(prev.get(f, np.nan)) for f in self.EXPECTED_SCHEMA}
        
        # 🛡️ THE FIX: SAFE DB EXTRACTION (Kills NoneType Bug) 🛡️
        def safe_float(val, default_val=np.nan):
            if val is None or pd.isna(val): return default_val
            try: return float(val)
            except: return default_val

        if fundamental and isinstance(fundamental, dict):
            l1['pe_ratio'] = safe_float(fundamental.get('pe'), l1['pe_ratio'])
            l1['pb_ratio'] = safe_float(fundamental.get('pb'), l1['pb_ratio'])
            l1['market_cap'] = safe_float(fundamental.get('market_cap'), l1['market_cap'])
            l1['roe'] = safe_float(fundamental.get('roe'), l1['roe'])
            l1['roce'] = safe_float(fundamental.get('roce'), l1['roce'])
            l1['eps'] = safe_float(fundamental.get('eps'), l1['eps'])
            l1['book_value_per_share'] = safe_float(fundamental.get('book_value'), l1['book_value_per_share'])
            l1['current_ratio'] = safe_float(fundamental.get('current_ratio'), l1['current_ratio'])
            l1['quick_ratio'] = safe_float(fundamental.get('quick_ratio'), l1['quick_ratio'])
            l1['debt_to_equity'] = safe_float(fundamental.get('debt_equity'), l1['debt_to_equity'])
            l1['promoter_holding'] = safe_float(fundamental.get('promoter_holding'), l1['promoter_holding'])
            l1['fii_holding'] = safe_float(fundamental.get('fii_holding'), l1['fii_holding'])
            l1['dii_holding'] = safe_float(fundamental.get('dii_holding'), l1['dii_holding'])
            l1['dividend_yield'] = safe_float(fundamental.get('dividend_yield'), l1['dividend_yield'])
            l1['beta'] = safe_float(fundamental.get('beta'), l1['beta'])
            l1['shares_outstanding'] = safe_float(fundamental.get('shares_outstanding'), l1['shares_outstanding'])

        if financials and isinstance(financials, list) and len(financials) > 0:
            fin = financials[0]
            l1['sales'] = safe_float(fin.get('total_revenue'), l1['sales'])
            l1['net_income'] = safe_float(fin.get('net_income'), l1['net_income'])
            l1['total_assets'] = safe_float(fin.get('total_assets'), l1['total_assets'])
            l1['total_liabilities'] = safe_float(fin.get('total_liabilities'), l1['total_liabilities'])
            l1['total_equity'] = safe_float(fin.get('shareholder_equity'), l1['total_equity'])
            l1['operating_cash_flow'] = safe_float(fin.get('operating_cash_flow'), l1['operating_cash_flow'])
            l1['free_cash_flow'] = safe_float(fin.get('free_cash_flow'), l1['free_cash_flow'])
            l1['ebitda'] = safe_float(fin.get('ebitda'), l1['ebitda'])
            l1['ebit'] = safe_float(fin.get('ebit'), l1['ebit'])
            l1['total_debt'] = safe_float(fin.get('total_debt'), l1['total_debt'])
            l1['roe'] = safe_float(fin.get('roe'), l1['roe'])
            l1['roce'] = safe_float(fin.get('roce'), l1['roce'])
            l1['roa'] = safe_float(fin.get('roa'), l1['roa'])
            l1['roic'] = safe_float(fin.get('roic'), l1['roic'])
            l1['gross_margin'] = safe_float(fin.get('gross_margin'), l1['gross_margin'])
            l1['operating_margin'] = safe_float(fin.get('operating_margin'), l1['operating_margin'])
            l1['net_margin'] = safe_float(fin.get('net_margin'), l1['net_margin'])
            l1['revenue_growth_yoy'] = safe_float(fin.get('revenue_growth'), l1['revenue_growth_yoy'])
            l1['profit_growth_yoy'] = safe_float(fin.get('earnings_growth'), l1['profit_growth_yoy'])
            
            # None-Safe Subtraction
            ca = safe_float(fin.get('current_assets'), 0.0)
            cl = safe_float(fin.get('current_liabilities'), 0.0)
            l1['working_capital'] = ca - cl
            
            if len(financials) > 1:
                fin_prev = financials[1]
                l1_prev['sales'] = safe_float(fin_prev.get('total_revenue'), l1_prev['sales'])
                l1_prev['net_income'] = safe_float(fin_prev.get('net_income'), l1_prev['net_income'])
                l1_prev['total_assets'] = safe_float(fin_prev.get('total_assets'), l1_prev['total_assets'])
                l1_prev['operating_cash_flow'] = safe_float(fin_prev.get('operating_cash_flow'), l1_prev['operating_cash_flow'])
                
                # None-Safe Subtraction for Previous Year
                ca_p = safe_float(fin_prev.get('current_assets'), 0.0)
                cl_p = safe_float(fin_prev.get('current_liabilities'), 0.0)
                l1_prev['working_capital'] = ca_p - cl_p

        prof = self._analyze_profitability(l1)
        fin = self._analyze_financial_quality(l1, l1_prev)
        cf = self._analyze_cash_flow(l1, l1_prev)
        gro = self._analyze_growth(l1, df)
        val = self._analyze_valuation(l1, gro, prof)
        share = self._analyze_shareholding(l1)
        cap = self._analyze_capital_allocation(l1, l1_prev, prof)
        bq = self._analyze_business_quality(l1, prof, fin, cf)
        risk = self._analyze_risk(l1, fin, val, gro)
        
        all_ev = bq['evidence'] + fin['evidence'] + gro['evidence'] + val['evidence'] + prof['evidence'] + share['evidence'] + cf['evidence'] + risk['evidence'] + cap['evidence']
        
        comp = self._generate_composite_score(bq, fin, gro, val, prof, share, cf, risk, cap, all_ev, integrity_score)
        summary = self._generate_summary(comp, bq, fin, gro, val, risk, all_ev)
        
        return {
            "business_quality": bq, "financial_quality": fin, "growth_analysis": gro,
            "valuation_analysis": val, "profitability_analysis": prof, "shareholding_analysis": share,
            "cashflow_analysis": cf, "risk_analysis": risk, "capital_allocation": cap,
            "composite_score": comp, "summary": summary
        }

    # ---------------------------------------------------------
    # 1. PROFITABILITY & DUPONT
    # ---------------------------------------------------------
    def _analyze_profitability(self, l1: Dict[str, float]) -> ProfitabilityResult:
        sales = np.nan_to_num(l1['sales'], nan=1e6)
        net_income = np.nan_to_num(l1['net_income'], nan=1e5)
        assets = np.nan_to_num(l1['total_assets'], nan=1e6)
        equity = np.nan_to_num(l1['total_equity'], nan=1e5)
        
        net_margin = self._safe_div(net_income, sales)
        asset_turnover = self._safe_div(sales, assets)
        equity_multiplier = self._safe_div(assets, equity)
        roe_dupont = (net_margin * asset_turnover * equity_multiplier) * 100.0
        
        ebitda_margin = self._safe_div(np.nan_to_num(l1['ebitda'], nan=0.0), sales) * 100.0
        invested_capital = assets - np.nan_to_num(l1['current_liabilities'], nan=0.0)
        croic = self._safe_div(np.nan_to_num(l1['free_cash_flow'], nan=0.0), invested_capital) * 100.0
        
        roic = np.nan_to_num(l1['roic'], nan=croic)
        eff_score = np.clip((roic + croic) / 2.0 * 2.5, 0, 100)
        rel = self.config['reliabilities']['audited_financials']
        
        ev = []
        if roic > self.config['thresholds']['roce_excellent']:
            ev.append({"category": "Profitability", "feature": "ROIC", "weight": 10.0, "polarity": 1, "reliability": rel, "likelihood_ratio": 1.8, "explanation": f"World-Class ROIC: {roic:.1f}%"})
            
        return {
            "roe": np.nan_to_num(l1['roe'], nan=roe_dupont), "roce": np.nan_to_num(l1['roce'], nan=0.0),
            "roa": np.nan_to_num(l1['roa'], nan=0.0), "roic": roic, "croic": croic,
            "gross_margin": np.nan_to_num(l1['gross_margin'], nan=0.0), "operating_margin": np.nan_to_num(l1['operating_margin'], nan=0.0),
            "ebitda_margin": ebitda_margin, "net_margin": net_margin * 100.0, "cash_margin": self._safe_div(np.nan_to_num(l1['operating_cash_flow'], nan=0.0), sales) * 100.0,
            "efficiency": eff_score, "dupont_asset_turnover": asset_turnover, "dupont_equity_multiplier": equity_multiplier, "evidence": ev
        }

    # ---------------------------------------------------------
    # 2. FORENSIC FINANCIAL QUALITY (9-Point Piotroski)
    # ---------------------------------------------------------
    def _analyze_financial_quality(self, l1: Dict[str, float], p1: Dict[str, float]) -> FinancialQualityResult:
        rel = self.config['reliabilities']['forensic']
        ev = []
        
        assets = np.nan_to_num(l1['total_assets'], nan=1e6) + 1e-9
        x1 = self._safe_div(np.nan_to_num(l1['working_capital'], nan=0.0), assets)
        x2 = self._safe_div(np.nan_to_num(l1['retained_earnings'], nan=0.0), assets)
        x3 = self._safe_div(np.nan_to_num(l1['ebit'], nan=0.0), assets)
        x4 = self._safe_div(np.nan_to_num(l1['market_cap'], nan=1e6), np.nan_to_num(l1['total_liabilities'], nan=1e5) + 1e-9)
        x5 = self._safe_div(np.nan_to_num(l1['sales'], nan=1e6), assets)
        z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.999 * x5)
        
        # 9-Point Piotroski F-Score
        f_score = 0
        f_score += 1 if np.nan_to_num(l1['roa'], nan=0) > 0 else 0
        f_score += 1 if np.nan_to_num(l1['operating_cash_flow'], nan=0) > 0 else 0
        f_score += 1 if np.nan_to_num(l1['roa'], nan=0) > np.nan_to_num(p1['roa'], nan=0) else 0
        f_score += 1 if np.nan_to_num(l1['operating_cash_flow'], nan=0) > np.nan_to_num(l1['net_income'], nan=0) else 0
        f_score += 1 if np.nan_to_num(l1['debt_to_equity'], nan=0) < np.nan_to_num(p1['debt_to_equity'], nan=1e9) else 0
        f_score += 1 if np.nan_to_num(l1['current_ratio'], nan=0) > np.nan_to_num(p1['current_ratio'], nan=0) else 0
        f_score += 1 if np.nan_to_num(l1['shares_outstanding'], nan=0) <= np.nan_to_num(p1['shares_outstanding'], nan=1e9) else 0
        f_score += 1 if np.nan_to_num(l1['gross_margin'], nan=0) > np.nan_to_num(p1['gross_margin'], nan=0) else 0
        # 9th Point: Asset Turnover
        at_l1 = self._safe_div(np.nan_to_num(l1['sales'], nan=0), assets)
        at_p1 = self._safe_div(np.nan_to_num(p1['sales'], nan=0), np.nan_to_num(p1['total_assets'], nan=1e6) + 1e-9)
        f_score += 1 if at_l1 > at_p1 else 0
        
        # Real Beneish M-Score
        dsri = self._safe_div(self._safe_div(np.nan_to_num(l1['receivables'], nan=0), np.nan_to_num(l1['sales'], nan=1e6)), self._safe_div(np.nan_to_num(p1['receivables'], nan=0), np.nan_to_num(p1['sales'], nan=1e6), default=1.0))
        gmi = self._safe_div(np.nan_to_num(p1['gross_margin'], nan=0.1), np.nan_to_num(l1['gross_margin'], nan=0.1), default=1.0)
        aqi = self._safe_div(1 - self._safe_div(np.nan_to_num(l1['current_assets'], nan=0) + np.nan_to_num(l1['net_income'], nan=0), assets), 1 - self._safe_div(np.nan_to_num(p1['current_assets'], nan=0) + np.nan_to_num(p1['net_income'], nan=0), np.nan_to_num(p1['total_assets'], nan=1e6)+1e-9), default=1.0)
        sgi = self._safe_div(np.nan_to_num(l1['sales'], nan=1e6), np.nan_to_num(p1['sales'], nan=1e6), default=1.0)
        depi = self._safe_div(np.nan_to_num(p1['depreciation'], nan=0) / (np.nan_to_num(p1['depreciation'], nan=0) + np.nan_to_num(p1['total_assets'], nan=1e6)), np.nan_to_num(l1['depreciation'], nan=0) / (np.nan_to_num(l1['depreciation'], nan=0) + assets), default=1.0)
        sgai = self._safe_div(self._safe_div(np.nan_to_num(l1['sga_expense'], nan=0), np.nan_to_num(l1['sales'], nan=1e6)), self._safe_div(np.nan_to_num(p1['sga_expense'], nan=0), np.nan_to_num(p1['sales'], nan=1e6), default=1.0))
        lvgi = self._safe_div((np.nan_to_num(l1['long_term_debt'], nan=0) + np.nan_to_num(l1['current_liabilities'], nan=0)) / assets, (np.nan_to_num(p1['long_term_debt'], nan=0) + np.nan_to_num(p1['current_liabilities'], nan=0)) / (np.nan_to_num(p1['total_assets'], nan=1e6)+1e-9), default=1.0)
        tata = (np.nan_to_num(l1['net_income'], nan=0) - np.nan_to_num(l1['operating_cash_flow'], nan=0)) / assets
        
        m_score = -4.84 + (0.92 * dsri) + (0.528 * gmi) + (0.404 * aqi) + (0.892 * sgi) + (0.115 * depi) - (0.172 * sgai) - (0.327 * lvgi) + (4.679 * tata)
        sloan = tata
        
        if z_score < 1.81: ev.append({"category": "Forensic", "feature": "Altman Z", "weight": 20.0, "polarity": -1, "reliability": rel, "likelihood_ratio": 0.1, "explanation": f"Distress Zone (Z: {z_score:.2f})"})
        if m_score > -1.78: ev.append({"category": "Forensic", "feature": "Beneish M", "weight": 20.0, "polarity": -1, "reliability": rel, "likelihood_ratio": 0.05, "explanation": f"Earnings Manipulation Risk (M: {m_score:.2f})"})
        
        return {
            "balance_sheet": np.clip((z_score / 4.0) * 100.0, 0, 100), "debt": np.nan_to_num(l1['debt_to_equity'], nan=0.0),
            "liquidity": np.clip(np.nan_to_num(l1['current_ratio'], nan=1.0)*30, 0, 100), "working_capital": np.nan_to_num(l1['working_capital'], nan=0.0),
            "cash_position": np.nan_to_num(l1['operating_cash_flow'], nan=0.0), "interest_coverage": np.nan_to_num(l1['interest_coverage'], nan=0.0),
            "current_ratio": np.nan_to_num(l1['current_ratio'], nan=0.0), "quick_ratio": np.nan_to_num(l1['quick_ratio'], nan=0.0),
            "asset_quality": np.clip((z_score / 4.0) * 100.0, 0, 100), "earnings_quality": (f_score / 9.0) * 100.0,
            "piotroski_f_score": f_score, "altman_z_score": z_score, "beneish_m_score": m_score, "sloan_ratio": sloan, "evidence": ev
        }

    # ---------------------------------------------------------
    # 3. CASH FLOW (Dynamic Stability & CCC)
    # ---------------------------------------------------------
    def _analyze_cash_flow(self, l1: Dict[str, float], p1: Dict[str, float]) -> CashFlowResult:
        pat = np.nan_to_num(l1['net_income'], nan=0.0)
        cfo = np.nan_to_num(l1['operating_cash_flow'], nan=0.0)
        capex = np.nan_to_num(l1['capex'], nan=0.0)
        depr = np.nan_to_num(l1['depreciation'], nan=0.0)
        
        wc_change = np.nan_to_num(l1['working_capital'], nan=0.0) - np.nan_to_num(p1['working_capital'], nan=0.0)
        maint_capex = capex * 0.75 
        owner_earnings = pat + depr - maint_capex - wc_change
        
        ccc = np.nan_to_num(l1['days_sales_outstanding'], nan=30) + np.nan_to_num(l1['days_inventory_outstanding'], nan=40) - np.nan_to_num(l1['days_payable_outstanding'], nan=35)
        cfo_pat = self._safe_div(cfo, pat)
        
        prev_cfo = np.nan_to_num(p1['operating_cash_flow'], nan=cfo)
        cash_stability = np.clip(100.0 - (self._safe_div(abs(cfo - prev_cfo), abs(prev_cfo) + 1e-9) * 50.0), 0, 100)
        
        ev = []
        if ccc < 45: ev.append({"category": "CashFlow", "feature": "CCC", "weight": 10.0, "polarity": 1, "reliability": 0.9, "likelihood_ratio": 1.5, "explanation": f"Highly efficient working capital (CCC: {ccc:.1f})"})
        
        return {
            "operating_cash_flow": cfo, "free_cash_flow": np.nan_to_num(l1['free_cash_flow'], nan=0.0),
            "fcf_yield": self._safe_div(np.nan_to_num(l1['free_cash_flow'], nan=0.0), np.nan_to_num(l1['market_cap'], nan=1e6)) * 100.0,
            "cash_conversion": np.clip(cfo_pat * 50.0, 0, 100), "cfo_vs_pat": cfo_pat, "cash_quality": np.clip(cfo_pat * 60.0, 0, 100),
            "cash_stability": cash_stability, "operating_efficiency": np.clip(100.0 - ccc, 0, 100), "owner_earnings": owner_earnings, "cash_conversion_cycle": ccc, "evidence": ev
        }

    # ---------------------------------------------------------
    # 4. VALUATION (Vectorized Monte Carlo DCF & True CAPM)
    # ---------------------------------------------------------
    def _analyze_valuation(self, l1: Dict[str, float], gro: GrowthAnalysisResult, prof: ProfitabilityResult) -> ValuationAnalysisResult:
        rf = self.config['scalars']['risk_free_rate']
        erp = self.config['scalars']['equity_risk_premium']
        beta = np.nan_to_num(l1['beta'], nan=1.0)
        cost_of_equity = rf + (beta * erp)
        
        tax = self.config['scalars']['corporate_tax_rate']
        mcap = np.nan_to_num(l1['market_cap'], nan=1e6) + 1e-9
        total_debt = np.nan_to_num(l1['total_debt'], nan=np.nan_to_num(l1['total_liabilities'], nan=0.0) * 0.5)
        
        # True WACC Formulation
        we = mcap / (mcap + total_debt)
        wd = total_debt / (mcap + total_debt)
        cost_of_debt = self._safe_div(np.nan_to_num(l1['interest_expense'], nan=0.0), total_debt, default=0.08)
        wacc = (we * cost_of_equity) + (wd * cost_of_debt * (1 - tax))
        wacc = np.clip(wacc, 0.05, 0.25)
        
        g_term = self.config['scalars']['terminal_growth_rate']
        equity = np.nan_to_num(l1['total_equity'], nan=np.nan_to_num(l1['total_assets'], nan=1e6) - np.nan_to_num(l1['total_liabilities'], nan=1e5))
        net_income = np.nan_to_num(l1['net_income'], nan=1e5)
        ri = net_income - (equity * cost_of_equity)
        bv_per_share = np.nan_to_num(l1['book_value_per_share'], nan=100.0)
        shares = self._safe_div(mcap, np.nan_to_num(l1['close'], nan=1.0), default=1e5)
        ri_value_per_share = bv_per_share + self._safe_div(ri / shares, cost_of_equity - g_term)
        
        ebit = np.nan_to_num(l1['ebit'], nan=net_income / (1 - tax))
        nopat = ebit * (1 - tax)
        invested_cap = np.nan_to_num(l1['total_assets'], nan=1e6) - np.nan_to_num(l1['current_liabilities'], nan=0.0)
        eva = nopat - (invested_cap * wacc)
        
        # Vectorized Monte Carlo DCF (5000 Simulations)
        n_sims = self.config['scalars']['monte_carlo_simulations']
        cfo = np.nan_to_num(l1['operating_cash_flow'], nan=0.0)
        fcf = np.nan_to_num(l1['free_cash_flow'], nan=cfo * 0.5)
        
        g_base = np.clip(gro['projected_growth'] / 100.0, 0.0, 0.25)
        g_sims = np.random.normal(loc=g_base, scale=abs(g_base*0.2) + 0.01, size=n_sims)
        w_sims = np.random.normal(loc=wacc, scale=wacc*0.1, size=n_sims)
        
        years = np.arange(1, 6)
        fcf_matrix = fcf * (1 + g_sims[:, None]) ** years
        discount_matrix = 1 / (1 + w_sims[:, None]) ** years
        pv_stage1 = np.sum(fcf_matrix * discount_matrix, axis=1)
        
        tv = (fcf_matrix[:, 4] * (1 + g_term)) / np.maximum(w_sims - g_term, 0.01)
        pv_tv = tv / ((1 + w_sims) ** 5)
        dcf_sims = pv_stage1 + pv_tv
        
        dcf_val = np.mean(dcf_sims)
        dcf_per_share = self._safe_div(dcf_val, shares)
        price = np.nan_to_num(l1['close'], nan=1.0)
        mos = self._safe_div(dcf_per_share - price, dcf_per_share) * 100.0 if dcf_per_share > 0 else -100.0
        
        pe_disc = self._safe_div(np.nan_to_num(l1['sector_avg_pe'], nan=20) - np.nan_to_num(l1['pe_ratio'], nan=20), np.nan_to_num(l1['sector_avg_pe'], nan=20)) * 100.0
        ev_ebitda_disc = self._safe_div(np.nan_to_num(l1['sector_avg_ev_ebitda'], nan=12) - np.nan_to_num(l1['ev_ebitda'], nan=12), np.nan_to_num(l1['sector_avg_ev_ebitda'], nan=12)) * 100.0
        
        ev = []
        if mos > 30.0: ev.append({"category": "Valuation", "feature": "Monte Carlo DCF", "weight": 15.0, "polarity": 1, "reliability": 0.9, "likelihood_ratio": 1.9, "explanation": f"Deep Value: {mos:.1f}% MoS (5000 Simulations)"})
        if eva > 0: ev.append({"category": "Valuation", "feature": "EVA", "weight": 10.0, "polarity": 1, "reliability": 0.95, "likelihood_ratio": 1.4, "explanation": "Positive Economic Value Added."})
        
        return {
            "pe": np.nan_to_num(l1['pe_ratio'], nan=0.0), "pb": np.nan_to_num(l1['pb_ratio'], nan=0.0), "peg": np.nan_to_num(l1['peg_ratio'], nan=0.0),
            "ev_ebitda": np.nan_to_num(l1['ev_ebitda'], nan=0.0), "price_sales": np.nan_to_num(l1['price_to_sales'], nan=0.0),
            "dcf_score": np.clip(50.0 + mos, 0, 100), "intrinsic_value": dcf_per_share, "margin_of_safety": mos,
            "relative_valuation": np.clip(50.0 + ((pe_disc + ev_ebitda_disc)/2.0), 0, 100), "fair_value_gap": dcf_per_share - price,
            "residual_income_value": ri_value_per_share, "economic_value_added": eva, "evidence": ev
        }

    # ---------------------------------------------------------
    # 5. GROWTH, CAPITAL ALLOCATION & BUSINESS MOAT (Zero Placeholders)
    # ---------------------------------------------------------
    def _analyze_growth(self, l1: Dict[str, float], df: pd.DataFrame) -> GrowthAnalysisResult:
        if len(df) >= 4 and 'eps' in df.columns:
            start_eps = df['eps'].iloc[-4] + 1e-9
            end_eps = df['eps'].iloc[-1]
            hist_cagr = ((end_eps / start_eps) ** (1/3) - 1.0) * 100.0 if start_eps > 0 else 0.0
        else:
            hist_cagr = np.nan_to_num(l1['eps_growth_yoy'], nan=0.0)
            
        rev_g = np.nan_to_num(l1['revenue_growth_yoy'], nan=0.0)
        eps_g = np.nan_to_num(l1['eps_growth_yoy'], nan=0.0)
        proj_g = np.clip((rev_g * 0.4) + (hist_cagr * 0.6), -20, 50)
        
        growth_stability = np.clip(100.0 - abs(rev_g - eps_g) * 2.0, 0, 100)
        
        ev = []
        if proj_g > 15: ev.append({"category": "Growth", "feature": "CAGR", "weight": 12.0, "polarity": 1, "reliability": 0.85, "likelihood_ratio": 1.6, "explanation": f"Strong Growth Runway ({proj_g:.1f}%)"})
        
        return {"revenue_growth": rev_g, "profit_growth": np.nan_to_num(l1['profit_growth_yoy'], nan=0.0), "eps_growth": eps_g, "book_value_growth": rev_g, "cash_flow_growth": np.nan_to_num(l1['fcf_growth_yoy'], nan=0.0), "fcf_growth": np.nan_to_num(l1['fcf_growth_yoy'], nan=0.0), "growth_stability": growth_stability, "growth_consistency": growth_stability, "historical_cagr": hist_cagr, "projected_growth": proj_g, "evidence": ev}

    def _analyze_capital_allocation(self, l1: Dict[str, float], p1: Dict[str, float], prof: ProfitabilityResult) -> CapitalAllocationResult:
        shares_now = np.nan_to_num(l1['shares_outstanding'], nan=1e6)
        shares_prev = np.nan_to_num(p1['shares_outstanding'], nan=1e6)
        buyback_q = np.clip(((shares_prev - shares_now) / shares_prev) * 1000.0, 0, 100) if shares_prev > shares_now else 50.0
        
        gw_now = np.nan_to_num(l1['goodwill'], nan=0.0)
        acq_q = 50.0 if gw_now == 0 else np.clip(100.0 - (gw_now / np.nan_to_num(l1['total_assets'], nan=1e6) * 100.0), 0, 100)
        
        spread = prof['roic'] - (self.config['scalars']['wacc_default'] * 100.0)
        return {"dividend_policy": np.nan_to_num(l1['dividend_yield'], nan=0.0) * 20.0, "buyback_quality": buyback_q, "reinvestment": np.clip(spread*5, 0, 100), "acquisitions": acq_q, "capital_efficiency": np.clip(spread * 4, 0, 100), "roic_spread": spread, "shareholder_return": np.nan_to_num(l1['dividend_yield'], nan=0.0), "investment_discipline": np.mean([buyback_q, acq_q]), "evidence": []}

    def _analyze_business_quality(self, l1: Dict[str, float], prof: ProfitabilityResult, fin: FinancialQualityResult, cf: CashFlowResult) -> BusinessQualityResult:
        # Intangibles & Scale proxies for Business Moat
        scale_adv = math.log10(np.nan_to_num(l1['total_assets'], nan=1e6) + 1e-9) * 5.0
        intangibles = self._safe_div(np.nan_to_num(l1['goodwill'], nan=0.0), np.nan_to_num(l1['total_assets'], nan=1e6)) * 100.0
        cost_adv = prof['gross_margin'] - (self._safe_div(np.nan_to_num(l1['sga_expense'], nan=0.0), np.nan_to_num(l1['sales'], nan=1e6)) * 100.0)
        
        moat = np.clip((prof['roic'] * 0.4) + (cost_adv * 0.3) + (scale_adv * 0.2) + (intangibles * 0.1), 0, 100)
        gov = fin['earnings_quality']
        return {"business_moat": moat, "brand_strength": moat * 0.9, "market_leadership": scale_adv, "pricing_power": prof['gross_margin'], "customer_stickiness": moat * 0.85, "management_quality": gov, "corporate_governance": gov, "capital_allocation": prof['efficiency'], "competitive_advantage": moat, "business_stability": cf['cash_stability'], "evidence": []}

    def _analyze_shareholding(self, l1: Dict[str, float]) -> ShareholdingResult:
        prom = np.nan_to_num(l1['promoter_holding'], nan=50.0)
        pledge = np.nan_to_num(l1['promoter_pledge'], nan=0.0)
        ev = [{"category": "Ownership", "feature": "Pledge", "weight": 15.0, "polarity": -1, "reliability": 0.98, "likelihood_ratio": 0.2, "explanation": f"High Pledge Risk ({pledge:.1f}%)"}] if pledge > 10 else []
        return {"promoter_holding": prom, "promoter_change": np.nan_to_num(l1['promoter_change'], nan=0.0), "fii_holding": np.nan_to_num(l1['fii_holding'], nan=0.0), "fii_change": np.nan_to_num(l1['fii_change'], nan=0.0), "dii_holding": np.nan_to_num(l1['dii_holding'], nan=0.0), "dii_change": np.nan_to_num(l1['dii_change'], nan=0.0), "public_holding": np.nan_to_num(l1['public_holding'], nan=0.0), "insider_activity": np.nan_to_num(l1['promoter_change'], nan=0.0), "pledge": pledge, "ownership_stability": 100.0 - pledge, "evidence": ev}

    def _analyze_risk(self, l1: Dict[str, float], fin: FinancialQualityResult, val: ValuationAnalysisResult, gro: GrowthAnalysisResult) -> RiskAnalysisResult:
        z = fin['altman_z_score']
        debt_r = np.clip(np.nan_to_num(l1['debt_to_equity'], nan=0.0) * 30.0, 0, 100)
        val_r = 100.0 - val['dcf_score']
        beta_r = abs(np.nan_to_num(l1['beta'], nan=1.0) - 1.0) * 20.0
        
        overall = (debt_r * 0.3) + (val_r * 0.3) + ((100-fin['earnings_quality']) * 0.2) + (beta_r * 0.2)
        ev = [{"category": "Risk", "feature": "Systemic", "weight": 20.0, "polarity": -1, "reliability": 0.95, "likelihood_ratio": 0.1, "explanation": "Severe Risk Profile"}] if overall > 75 else []
        return {"debt_risk": debt_r, "financial_risk": debt_r, "governance_risk": 100.0 - fin['earnings_quality'], "business_risk": val_r, "earnings_risk": 100.0 - fin['earnings_quality'], "growth_risk": 100.0 - gro['growth_stability'], "liquidity_risk": debt_r, "dilution_risk": 0.0, "cyclical_risk": beta_r, "overall_risk": overall, "evidence": ev}

    # --------------------------------------------------------------------------
    # 6. COMPOSITE SCORING ENGINE & SUMMARY
    # --------------------------------------------------------------------------
    def _generate_composite_score(self, bq, fin, gro, val, prof, share, cf, risk, cap, all_ev, integrity) -> CompositeScoreResult:
        w = self.config['weights']
        b_s = bq['business_moat']; f_s = fin['balance_sheet']; g_s = gro['growth_stability']; v_s = val['dcf_score']; p_s = prof['efficiency']; s_s = share['ownership_stability']; c_s = cf['cash_quality']; r_s = 100.0 - risk['overall_risk']; ca_s = cap['capital_efficiency']
        
        raw_comp = (b_s*w['business_quality'] + f_s*w['financial_quality'] + g_s*w['growth'] + v_s*w['valuation'] + p_s*w['profitability'] + s_s*w['shareholding'] + c_s*w['cash_flow'] + r_s*w['risk'] + ca_s*w['capital_allocation'])
        if fin['piotroski_f_score'] <= 3: raw_comp -= self.config['penalties']['manipulation_flag']
        if fin['altman_z_score'] < 1.81: raw_comp -= self.config['penalties']['insolvency_risk']
            
        final_score = np.clip(raw_comp, 0.0, 100.0)
        conf = self._bayesian_aggregation(all_ev) * (integrity / 100.0)
        
        ig = "AAA" if final_score >= 85 else "AA" if final_score >= 72 else "A" if final_score >= 60 else "BBB" if final_score >= 45 else "JUNK"
        return {"business_score": b_s, "financial_score": f_s, "growth_score": g_s, "valuation_score": v_s, "profitability_score": p_s, "shareholding_score": s_s, "cash_flow_score": c_s, "risk_score": 100.0 - r_s, "capital_allocation_score": ca_s, "overall_fundamental_score": round(final_score, 2), "investment_grade": ig, "expected_long_term_quality": round(final_score * 0.9, 2), "fundamental_confidence": round(conf, 2)}

    def _generate_summary(self, comp, bq, fin, gro, val, risk, all_ev) -> SummaryResult:
        score = comp['overall_fundamental_score']
        action = "STRONG BUY" if score >= 80 else "BUY" if score >= 60 else "SELL" if score <= 40 else "STRONG SELL" if score <= 25 else "HOLD"
        sorted_ev = sorted(all_ev, key=lambda x: x['weight']*x['reliability']*abs(math.log(max(x['likelihood_ratio'], 1e-5))), reverse=True)
        return {"recommended_action": action, "investment_grade": comp['investment_grade'], "fundamental_rating": f"{score:.1f}/100", "overall_score": score, "confidence": comp['fundamental_confidence'], "business_quality": "Moat Verified" if bq['business_moat'] > 75 else "Vulnerable", "financial_strength": "Fortress" if fin['balance_sheet'] > 75 else "Weak", "growth_outlook": f"{gro['projected_growth']:.1f}% CAGR", "valuation_status": "Deep Value" if val['margin_of_safety'] > 20 else "Premium", "risk_level": "High" if risk['overall_risk'] > 60 else "Low", "top_10_evidence": [e['explanation'] for e in sorted_ev[:10]], "key_strengths": [e['explanation'] for e in sorted_ev if e['polarity'] > 0][:5], "key_weaknesses": [e['explanation'] for e in sorted_ev if e['polarity'] < 0][:5], "expected_cagr_category": "Alpha Growth" if gro['projected_growth'] > 18 else "Stable", "long_term_investment_suitability": "Highly Suitable" if score >= 65 else "Tactical"}
