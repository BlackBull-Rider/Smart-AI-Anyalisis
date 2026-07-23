import math
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, TypedDict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INSTITUTIONAL_CONFIG = {
    "scalars": {
        "pressure_multiplier": 20.0,
        "dii_score_multiplier": 25.0,
        "promoter_control_norm": 75.0,
        "conviction_multiplier": 25.0,
        "float_absorption_threshold": 0.05,
        "delivery_trend_weight": 10.0,
        "block_market_cap_ratio": 0.005,
        "bayesian_damping_power": 0.5
    },
    "thresholds": {
        "fii_strong": 1.5, "fii_mild": 0.5,
        "dii_strong": 1.5, "dii_mild": 0.5,
        "promoter_strong": 1.0, "promoter_mild": 0.2,
        "promoter_control_min": 50.0, 
        "delivery_high_pct": 65.0,
        "delivery_trend_strong": 1.2, 
        "heavy_dist_limit": -2.0,
        "smart_money_strong": 70.0,
        "smart_money_weak": 30.0
    },
    "weights": {
        "fii": 0.30, "dii": 0.20, "promoter": 0.25,
        "delivery": 0.10, "block": 0.05, "bulk": 0.05,
        "smart_money": 0.05
    },
    "reliabilities": {
        "promoter": 0.98, "fii": 0.92, "dii": 0.90,
        "block": 0.88, "bulk": 0.80, "delivery": 0.85,
        "smart_money": 0.85
    },
    "hierarchy_multipliers": {
        "Promoter": 2.0, "FII": 1.5, "DII": 1.2,
        "SmartMoney": 1.1, "Block": 1.0, "Bulk": 0.8, "Delivery": 0.5
    },
    "penalties": {
        "missing_feature": 10.0,
        "conflict": 15.0
    },
    "action_thresholds": {
        "strong_acc": 75.0, "acc": 55.0,
        "dist": 55.0, "strong_dist": 75.0
    },
    "bayesian": {
        "prior_accumulation": 0.5,
        "prior_distribution": 0.5
    }
}

# ==============================================================================
# SCHEMAS (TypedDicts)
# ==============================================================================
class EvidenceItem(TypedDict):
    category: str; feature: str; weight: float; polarity: int
    reliability: float; likelihood_ratio: float; institutional_explanation: str

class FIIResult(TypedDict):
    status: str; fii_confidence: float; fii_conviction: float; institutional_pressure: float
    historical_strength: float; evidence: List[EvidenceItem]

class DIIResult(TypedDict):
    status: str; dii_score: float; participation: float; reliability: float; evidence: List[EvidenceItem]

class PromoterResult(TypedDict):
    status: str; stable_holding: bool; increasing_ownership: bool; ownership_risk: bool
    control_strength: float; promoter_confidence: float; promoter_reliability: float; evidence: List[EvidenceItem]

class DeliveryResult(TypedDict):
    delivery_percent: float; delivery_trend: float; long_term_accumulation: bool
    speculative_volume: bool; institutional_participation: bool; delivery_quality: float
    delivery_confidence: float; evidence: List[EvidenceItem]

class BlockDealResult(TypedDict):
    large_institutional_buy: bool; large_institutional_sell: bool; fresh_position: bool
    exit_position: bool; continuation: bool; reliability: float; probability: float; evidence: List[EvidenceItem]

class BulkDealResult(TypedDict):
    bulk_buying: bool; bulk_selling: bool; repeated_buying: bool; repeated_selling: bool
    reliability: float; probability: float; evidence: List[EvidenceItem]

class SmartMoneyResult(TypedDict):
    status: str; flow_score: float; evidence: List[EvidenceItem]

class RoutingResult(TypedDict):
    primary_institutional_event: str; dominant_buyer: str; dominant_seller: str
    institutional_bias: str; participation_quality: float; ownership_quality: float
    consensus_score: float; conflict_score: float; institutional_conviction: float

class ProbabilitiesResult(TypedDict):
    accumulation_probability: float; distribution_probability: float
    institutional_presence_probability: float; ownership_shift_probability: float; trend_continuation_probability: float

class AdvancedMetricsResult(TypedDict):
    institutional_score: float; ownership_quality: float; buying_pressure: float
    selling_pressure: float; accumulation_score: float; distribution_score: float
    delivery_quality: float; ownership_stability: float; institutional_confidence: float; market_quality: float

class SummaryResult(TypedDict):
    recommended_action: str; dominant_institutional_event: str; dominant_buyer: str
    dominant_seller: str; institutional_bias: str; confidence: float; top_5_evidence: List[str]; risk_level: str

class InstitutionalAnalysisResult(TypedDict):
    fii_analysis: FIIResult; dii_analysis: DIIResult; promoter_analysis: PromoterResult
    delivery_analysis: DeliveryResult; block_deal_analysis: BlockDealResult; bulk_deal_analysis: BulkDealResult
    smart_money_analysis: SmartMoneyResult; routing_engine: RoutingResult
    probabilities: ProbabilitiesResult; advanced_metrics: AdvancedMetricsResult; summary: SummaryResult

# ==============================================================================
# ANALYZER ENGINE
# ==============================================================================
class InstitutionalAnalyzer:
    EXPECTED_SCHEMA = [
        'institutional_holding', 'fii_holding', 'dii_holding', 'promoter_holding',
        'promoter_change', 'fii_change', 'dii_change', 'delivery_percent',
        'delivery_trend', 'block_deal', 'block_deal_value', 'block_deal_buy',
        'block_deal_sell', 'bulk_deal', 'bulk_deal_buy', 'bulk_deal_sell',
        'smart_money_flow', 'ownership_concentration', 'institutional_volume_score',
        'market_cap', 'float_shares', 'free_float', 'volume_confirmation',
        'trend_direction', 'trend_strength', 'market_regime'
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or INSTITUTIONAL_CONFIG

    def _validate_integrity(self, df: pd.DataFrame) -> float:
        missing = [f for f in self.EXPECTED_SCHEMA if f not in df.columns]
        penalty = len(missing) * self.config['penalties']['missing_feature']
        return np.clip(100.0 - penalty, 0.0, 100.0)

    def _weighted_confidence(self, evidence: List[EvidenceItem]) -> float:
        if not evidence: return 0.0
        total_w = sum(e['weight'] for e in evidence)
        if total_w == 0: return 0.0
        return sum(e['weight'] * e['reliability'] for e in evidence) / total_w

    def _stable_sigmoid(self, log_odds: float) -> float:
        """Math Overflow Safe Sigmoid for Bayesian Posterior."""
        if log_odds >= 0:
            return 1.0 / (1.0 + math.exp(-log_odds))
        else:
            exp_lo = math.exp(log_odds)
            return exp_lo / (1.0 + exp_lo)

    def analyze(self, df, shareholding=None, **kwargs) -> InstitutionalAnalysisResult:
        if df is None or df.empty:
            raise ValueError("InstitutionalAnalyzer received empty DataFrame.")
        
        integrity_score = self._validate_integrity(df)
        
        # Multi-Period Context Extraction
        latest = df.iloc[-1]
        l1 = {f: latest.get(f, 0.0) for f in self.EXPECTED_SCHEMA}
        
        # 20-Day and 5-Day Historical Context
        hist_20d = df.tail(20).mean(numeric_only=True) if len(df) >= 20 else latest
        hist_5d = df.tail(5).mean(numeric_only=True) if len(df) >= 5 else latest
        
        # Safe Multi-period volume extraction
        vol_20d = df['volume'].tail(20).mean() if 'volume' in df.columns and len(df) >= 20 else latest.get('volume_confirmation', 0.0)*1000
        
        # Core Processors
        fii = self._proc_fii(l1, hist_20d)
        dii = self._proc_dii(l1, hist_20d)
        prom = self._proc_promoter(l1)
        deliv = self._proc_delivery(l1, hist_5d, vol_20d)
        block = self._proc_block(l1)
        bulk = self._proc_bulk(l1)
        smart_money = self._proc_smart_money(l1, hist_20d)
        
        all_evidence = fii['evidence'] + dii['evidence'] + prom['evidence'] + \
                       deliv['evidence'] + block['evidence'] + bulk['evidence'] + smart_money['evidence']
        
        # Hierarchical Routing
        routing = self._run_routing(fii, dii, prom, deliv, block, bulk, smart_money, l1)
        
        # Category-Aware Bayesian Engine
        probs, bayesian_log_odds = self._run_bayesian(all_evidence, l1, hist_20d)
        
        # Holistic Metrics
        metrics = self._run_metrics(routing, probs, all_evidence, deliv, l1)
        
        # Final Confidence Calibration
        bayesian_cert = np.clip(abs(bayesian_log_odds) * 33.3, 0.0, 100.0)
        evidence_cert = self._weighted_confidence(all_evidence) * 100.0
        routing_cert = routing['consensus_score']
        
        final_confidence = np.clip(
            (integrity_score * 0.1) + (routing_cert * 0.35) + (bayesian_cert * 0.35) + (evidence_cert * 0.2) - routing['conflict_score'], 
            0.0, 100.0
        )
        
        # Action Matrix
        th = self.config['action_thresholds']
        if metrics['accumulation_score'] >= th['strong_acc'] and final_confidence > 60: action = "STRONG BUY"
        elif metrics['accumulation_score'] >= th['acc']: action = "BUY"
        elif metrics['distribution_score'] >= th['strong_dist'] and final_confidence > 60: action = "STRONG SELL"
        elif metrics['distribution_score'] >= th['dist']: action = "SELL"
        else: action = "WAIT"
        
        sorted_ev = sorted(all_evidence, key=lambda x: x['weight'] * x['reliability'] * abs(math.log(max(x['likelihood_ratio'], 1e-5))), reverse=True)
        
        return {
            "fii_analysis": fii, "dii_analysis": dii, "promoter_analysis": prom,
            "delivery_analysis": deliv, "block_deal_analysis": block, "bulk_deal_analysis": bulk,
            "smart_money_analysis": smart_money, "routing_engine": routing,
            "probabilities": probs, "advanced_metrics": metrics,
            "summary": {
                "recommended_action": action,
                "dominant_institutional_event": routing['primary_institutional_event'],
                "dominant_buyer": routing['dominant_buyer'],
                "dominant_seller": routing['dominant_seller'],
                "institutional_bias": routing['institutional_bias'],
                "confidence": round(final_confidence, 2),
                "top_5_evidence": [e['institutional_explanation'] for e in sorted_ev[:5]],
                "risk_level": "High" if metrics['selling_pressure'] > metrics['buying_pressure'] else "Low"
            }
        }

    def _proc_fii(self, l1: Dict[str, Any], hist_20d: pd.Series) -> FIIResult:
        change = l1.get('fii_change', 0.0)
        hist_change = hist_20d.get('fii_change', 0.0)
        th = self.config['thresholds']
        rel = self.config['reliabilities']['fii']
        w = self.config['weights']['fii']
        
        if change >= th['fii_strong']: status, lr = "Strong Accumulation", 2.5
        elif change >= th['fii_mild']: status, lr = "Mild Accumulation", 1.5
        elif change <= -th['fii_strong']: status, lr = "Heavy Distribution", 0.2
        elif change <= -th['fii_mild']: status, lr = "Distribution", 0.5
        else: status, lr = "Neutral", 1.0
        
        # Boost LR if multi-period trend confirms
        if change > 0 and hist_change > 0: lr *= 1.2
        elif change < 0 and hist_change < 0: lr *= 0.8
        
        pressure = np.clip(change * self.config['scalars']['pressure_multiplier'], -100, 100)
        ev = [{"category": "FII", "feature": "Change", "weight": w, "polarity": 1 if change > 0 else -1, "reliability": rel, "likelihood_ratio": lr, "institutional_explanation": f"FII Net Flow: {change:.2f}%"}] if change != 0 else []
        return {"status": status, "fii_confidence": 90.0 if change != 0 else 0.0, "fii_conviction": abs(pressure), "institutional_pressure": pressure, "historical_strength": l1.get('fii_holding', 0.0), "evidence": ev}

    def _proc_dii(self, l1: Dict[str, Any], hist_20d: pd.Series) -> DIIResult:
        change = l1.get('dii_change', 0.0)
        score = np.clip(change * self.config['scalars']['dii_score_multiplier'], -100, 100)
        rel = self.config['reliabilities']['dii']
        
        lr = 2.0 if change > 1.0 else 1.3 if change > 0 else 0.4 if change < -1.0 else 0.7 if change < 0 else 1.0
        ev = [{"category": "DII", "feature": "Change", "weight": self.config['weights']['dii'], "polarity": 1 if change > 0 else -1, "reliability": rel, "likelihood_ratio": lr, "institutional_explanation": f"DII Net Flow: {change:.2f}%"}] if change != 0 else []
        return {"status": "Accumulation" if score > 0 else "Distribution" if score < 0 else "Neutral", "dii_score": score, "participation": l1.get('dii_holding', 0.0), "reliability": rel * 100.0, "evidence": ev}

    def _proc_promoter(self, l1: Dict[str, Any]) -> PromoterResult:
        h = l1.get('promoter_holding', 0.0)
        change = l1.get('promoter_change', 0.0)
        th = self.config['thresholds']
        rel = self.config['reliabilities']['promoter']
        
        control = np.clip((h / self.config['scalars']['promoter_control_norm']) * 100, 0, 100)
        stable = abs(change) < th['promoter_mild']
        lr = 3.0 if change >= th['promoter_strong'] else 1.8 if change > 0 else 0.1 if change <= -th['promoter_strong'] else 0.4 if change < 0 else 1.0
        
        ev = [{"category": "Promoter", "feature": "Change", "weight": self.config['weights']['promoter'], "polarity": 1 if change > 0 else -1, "reliability": rel, "likelihood_ratio": lr, "institutional_explanation": f"Promoter Shift: {change:.2f}%"}] if not stable else []
        return {"status": "Stable" if stable else "Active", "stable_holding": stable, "increasing_ownership": change > 0, "ownership_risk": h < th['promoter_control_min'], "control_strength": control, "promoter_confidence": 95.0, "promoter_reliability": rel * 100.0, "evidence": ev}

    def _proc_delivery(self, l1: Dict[str, Any], hist_5d: pd.Series, vol_20d: float) -> DeliveryResult:
        pct = l1.get('delivery_percent', 0.0)
        trend_5d = hist_5d.get('delivery_trend', 0.0)
        free_float = l1.get('free_float', 1e6) + 1e-9
        
        delivery_volume = vol_20d * (pct / 100.0)
        absorption = (delivery_volume / free_float) * 100.0 
        
        th_abs = self.config['scalars']['float_absorption_threshold'] * 100.0
        lr = 1.6 if absorption >= th_abs else 0.7 if pct < 30.0 else 1.0
        qual = np.clip(pct + (trend_5d * self.config['scalars']['delivery_trend_weight']), 0.0, 100.0)
        
        ev = []
        if absorption > th_abs or pct < 30.0:
            ev.append({"category": "Delivery", "feature": "Absorption", "weight": self.config['weights']['delivery'], "polarity": 1 if absorption >= th_abs else -1, "reliability": self.config['reliabilities']['delivery'], "likelihood_ratio": lr, "institutional_explanation": f"Float Absorption: {absorption:.2f}%"})
            
        return {"delivery_percent": pct, "delivery_trend": trend_5d, "long_term_accumulation": absorption > th_abs, "speculative_volume": pct < 30.0, "institutional_participation": pct > 50.0, "delivery_quality": qual, "delivery_confidence": qual * 0.9, "evidence": ev}

    def _proc_block(self, l1: Dict[str, Any]) -> BlockDealResult:
        buy = l1.get('block_deal_buy', 0) > 0
        sell = l1.get('block_deal_sell', 0) > 0
        mcap = l1.get('market_cap', 1e9) + 1e-9
        impact = (l1.get('block_deal_value', 0.0) / mcap) * 100.0
        is_sig = impact > (self.config['scalars']['block_market_cap_ratio'] * 100)
        
        lr = 2.2 if (buy and is_sig) else 0.3 if (sell and is_sig) else 1.0
        ev = [{"category": "Block", "feature": "Impact", "weight": self.config['weights']['block'], "polarity": 1 if buy else -1, "reliability": self.config['reliabilities']['block'], "likelihood_ratio": lr, "institutional_explanation": f"Block Deal Impact: {impact:.3f}%"}] if (is_sig and (buy or sell)) else []
        return {"large_institutional_buy": buy and is_sig, "large_institutional_sell": sell and is_sig, "fresh_position": buy and is_sig, "exit_position": sell and is_sig, "continuation": False, "reliability": 88.0, "probability": 85.0 if is_sig else 0.0, "evidence": ev}

    def _proc_bulk(self, l1: Dict[str, Any]) -> BulkDealResult:
        buy = l1.get('bulk_deal_buy', 0) > 0
        sell = l1.get('bulk_deal_sell', 0) > 0
        lr = 1.4 if buy else 0.6 if sell else 1.0
        ev = [{"category": "Bulk", "feature": "Deal", "weight": self.config['weights']['bulk'], "polarity": 1 if buy else -1, "reliability": self.config['reliabilities']['bulk'], "likelihood_ratio": lr, "institutional_explanation": "Bulk Buying" if buy else "Bulk Selling"}] if (buy or sell) else []
        return {"bulk_buying": buy, "bulk_selling": sell, "repeated_buying": False, "repeated_selling": False, "reliability": 80.0, "probability": 65.0 if (buy or sell) else 0.0, "evidence": ev}

    def _proc_smart_money(self, l1: Dict[str, Any], hist_20d: pd.Series) -> SmartMoneyResult:
        score = l1.get('smart_money_flow', 50.0)
        trend = score - hist_20d.get('smart_money_flow', 50.0)
        th = self.config['thresholds']
        
        lr = 1.8 if score >= th['smart_money_strong'] else 0.4 if score <= th['smart_money_weak'] else 1.0
        if trend > 10: lr *= 1.1
        elif trend < -10: lr *= 0.9
        
        ev = [{"category": "SmartMoney", "feature": "Flow", "weight": self.config['weights']['smart_money'], "polarity": 1 if score > 50 else -1, "reliability": self.config['reliabilities']['smart_money'], "likelihood_ratio": lr, "institutional_explanation": f"SMC Score: {score:.1f}"}] if score != 50 else []
        return {"status": "Accumulating" if score > 60 else "Distributing" if score < 40 else "Neutral", "flow_score": score, "evidence": ev}

    def _run_routing(self, fii, dii, prom, deliv, block, bulk, smc, l1) -> RoutingResult:
        # Hierarchical Arbitration with Multipliers
        w = self.config['weights']
        h_mult = self.config['hierarchy_multipliers']
        
        p_force = (1.0 if prom['increasing_ownership'] else -1.0 if prom['ownership_risk'] else 0.0) * w['promoter'] * h_mult['Promoter']
        f_force = np.clip(l1.get('fii_change', 0), -5, 5) / 5.0 * w['fii'] * h_mult['FII']
        d_force = np.clip(l1.get('dii_change', 0), -5, 5) / 5.0 * w['dii'] * h_mult['DII']
        smc_force = ((smc['flow_score'] - 50) / 50.0) * w['smart_money'] * h_mult['SmartMoney']
        blk_force = (1.0 if block['large_institutional_buy'] else -1.0 if block['large_institutional_sell'] else 0.0) * w['block'] * h_mult['Block']
        blk_force += (1.0 if bulk['bulk_buying'] else -1.0 if bulk['bulk_selling'] else 0.0) * w['bulk'] * h_mult['Bulk']
        del_force = (1.0 if deliv['long_term_accumulation'] else -1.0 if deliv['speculative_volume'] else 0.0) * w['delivery'] * h_mult['Delivery']
        
        total_bull = sum(f for f in [p_force, f_force, d_force, smc_force, blk_force, del_force] if f > 0)
        total_bear = abs(sum(f for f in [p_force, f_force, d_force, smc_force, blk_force, del_force] if f < 0))
        net_force = total_bull - total_bear
        
        forces = {"Promoter": p_force, "FII": f_force, "DII": d_force, "Block": blk_force, "SmartMoney": smc_force}
        dom_buyer = max(forces, key=forces.get) if max(forces.values()) > 0 else "None"
        dom_seller = min(forces, key=forces.get) if min(forces.values()) < 0 else "None"
        bias = "Bullish" if net_force > 0.1 else "Bearish" if net_force < -0.1 else "Neutral"
        
        return {
            "primary_institutional_event": "Accumulation" if net_force > 0 else "Distribution" if net_force < 0 else "Rotational",
            "dominant_buyer": dom_buyer, "dominant_seller": dom_seller, "institutional_bias": bias,
            "participation_quality": np.clip(l1.get('institutional_holding', 0.0), 0, 100),
            "ownership_quality": np.clip(l1.get('ownership_concentration', 50.0), 0, 100),
            "consensus_score": np.clip((abs(net_force) / (total_bull + total_bear + 1e-9)) * 100, 0, 100),
            "conflict_score": np.clip((min(total_bull, total_bear) / (max(total_bull, total_bear) + 1e-9)) * 100, 0, 100),
            "institutional_conviction": np.clip(abs(net_force) * 30, 0, 100) # Scaled dynamically
        }

    def _run_bayesian(self, ev: List[EvidenceItem], l1: Dict[str, Any], hist_20d: pd.Series) -> Tuple[ProbabilitiesResult, float]:
        """Category-Aware Damped Log-Odds Formulation with Reliability Weighting."""
        prior_acc = self.config['bayesian']['prior_accumulation']
        log_prior = math.log(prior_acc / (1.0 - prior_acc))
        
        cat_log_lrs = defaultdict(list)
        for e in ev:
            # log(LR) * reliability
            val = e['reliability'] * math.log(max(e['likelihood_ratio'], 1e-5))
            cat_log_lrs[e['category']].append(val)
            
        damped_log_lr = 0.0
        d_power = self.config['scalars']['bayesian_damping_power']
        for cat, vals in cat_log_lrs.items():
            # Apply Damping 1 / N^power per category
            damping = 1.0 / (len(vals) ** d_power) if len(vals) > 0 else 1.0
            damped_log_lr += sum(vals) * damping
            
        log_posterior = log_prior + damped_log_lr
        post_prob = self._stable_sigmoid(log_posterior)
        
        # Dynamic Presence & Continuation Probabilities
        inst_vol = l1.get('institutional_volume_score', 0.0)
        inst_presence = np.clip((inst_vol * 0.5) + (l1.get('smart_money_flow', 50.0) * 0.5), 0.0, 100.0)
        
        trend = l1.get('trend_strength', 50.0)
        trend_20d = hist_20d.get('trend_strength', 50.0)
        trend_cont = np.clip((trend * 0.7) + ((trend - trend_20d) * 2.0), 0.0, 100.0)
        
        own_shift = np.clip((abs(l1.get('promoter_change', 0))*3 + abs(l1.get('fii_change', 0))*1.5) * 10, 0, 100)
        
        return {
            "accumulation_probability": round(post_prob * 100.0, 2),
            "distribution_probability": round((1.0 - post_prob) * 100.0, 2),
            "institutional_presence_probability": round(inst_presence, 2),
            "ownership_shift_probability": round(own_shift, 2),
            "trend_continuation_probability": round(trend_cont, 2)
        }, log_posterior

    def _run_metrics(self, r: RoutingResult, p: ProbabilitiesResult, ev: List[EvidenceItem], deliv: DeliveryResult, l1: Dict[str, Any]) -> AdvancedMetricsResult:
        total_possible_weight = sum(self.config['weights'].values())
        buy_w_sum = sum(e['weight'] * e['reliability'] for e in ev if e['polarity'] > 0)
        sell_w_sum = sum(e['weight'] * e['reliability'] for e in ev if e['polarity'] < 0)
        
        buy_press = np.clip((buy_w_sum / total_possible_weight) * 100, 0, 100)
        sell_press = np.clip((sell_w_sum / total_possible_weight) * 100, 0, 100)
        
        # Holistic Ownership Quality
        total_inst = l1.get('promoter_holding',0) + l1.get('fii_holding',0) + l1.get('dii_holding',0)
        own_qual = np.clip((total_inst * (1 + (l1.get('promoter_change',0)/100))) * (1 - (l1.get('free_float', 1e7)/1e9)), 0, 100)
        
        # Composite Institutional Score
        inst_score = np.clip((r['institutional_conviction'] * 0.3) + (p['accumulation_probability'] * 0.3) + (buy_press * 0.2) + (own_qual * 0.2), 0, 100)
        
        return {
            "institutional_score": inst_score, 
            "ownership_quality": own_qual, 
            "buying_pressure": buy_press, 
            "selling_pressure": sell_press, 
            "accumulation_score": np.clip(p['accumulation_probability'] * (buy_press/100.0) * 1.5, 0, 100), 
            "distribution_score": np.clip(p['distribution_probability'] * (sell_press/100.0) * 1.5, 0, 100), 
            "delivery_quality": deliv['delivery_quality'], 
            "ownership_stability": own_qual, 
            "institutional_confidence": r['consensus_score'], 
            "market_quality": np.clip((own_qual * 0.5) + (r['participation_quality'] * 0.5), 0, 100)
        }
