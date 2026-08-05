from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EPS = 1e-12
MIN_LR = 0.15
MAX_LR = 8.0

REQUIRED_FEATURES = (
    "fii_holding_pct", "fii_holding", "fii_change", "fpi_holding_pct",
    "dii_holding_pct", "dii_holding", "dii_change",
    "promoter_holding_pct", "promoter_holding", "promoter_change",
    "promoter_pledge_pct", "promoter_pledge", "promoter_pledge_change",
    "institutional_holding_pct", "institutional_holding", "institutional_change",
    "public_holding_pct", "public_holding",
    "delivery_percentage", "delivery_pct", "delivery_change", "delivery",
    "accumulation", "distribution",
    "close", "open", "high", "low", "volume", 
    "rvol_20", "volume_ratio", "vol_zscore", 
    "price_change", "return_1d", "breakout_pressure", "trend_angle",
    "smart_money_candle", "liquidity_sweep_candle", "institutional_body", "absorption_candle", "rejection_candle",
    "bullish_candle", "bearish_candle"
)

PCT_KEYS = {
    "fii": ("fii_holding_pct", "fii_holding_percentage", "fpi_holding_pct", "fpi_holding_percentage"),
    "dii": ("dii_holding_pct", "dii_holding_percentage"),
    "inst": ("institutional_holding_pct", "institutional_holding_percentage"),
    "promoter": ("promoter_holding_pct", "promoter_holding_percentage"),
    "pledge": ("promoter_pledge_pct", "promoter_pledge_percentage"),
    "public": ("public_holding_pct", "public_holding_percentage"),
    "delivery": ("delivery_percentage", "delivery_pct")
}

CHG_KEYS = {
    "fii": ("fii_holding_change", "fii_change", "fpi_change"),
    "dii": ("dii_holding_change", "dii_change"),
    "inst": ("institutional_change", "institutional_holding_change"),
    "promoter": ("promoter_holding_change", "promoter_change"),
    "pledge": ("promoter_pledge_change",),
    "public": ("public_holding_change", "public_change"),
    "delivery": ("delivery_change",)
}

@dataclass
class EvidenceNode:
    domain: str
    direction: float
    magnitude: float
    weight: float
    message: str
    reliability: float

def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool): return None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "").replace("%", "")
            if not value: return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")

def _safe_float(value: Any, default: float = 0.0) -> float:
    n = _num(value)
    if n is None or not math.isfinite(n): return default
    return float(n)

def _raw(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        val = row.get(_key(name))
        if val is not None: return val
    return None

def _value(row: Mapping[str, Any], *names: str) -> Optional[float]:
    return _num(_raw(row, *names))

def _lr(weight: float) -> float:
    return _clip(math.exp(_clip(weight, -1.9, 2.08)), MIN_LR, MAX_LR)


class InstitutionalAnalyzer:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze(self, data: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            rows = self._rows(data)
            if not rows: return self._empty()

            current_market = self._get_latest_market_row(rows)
            if current_market is None:
                result = self._empty()
                result["institutional_analyzer"]["evidence"].append(
                    self._evidence("Institutional", "No valid market data snapshot found in payload.", 0.99, 0.35)
                )
                return result

            nodes: List[EvidenceNode] = []
            
            # --- PROMOTER DOMAIN ---
            prom_pct, prom_delta = self._extract_value_and_delta(rows, "promoter")
            prom_status = "unknown"
            if prom_delta is not None:
                if abs(prom_delta) < 0.01:
                    prom_status = "stable"
                    nodes.append(EvidenceNode("promoter", 0.0, 1.0, 1.0, "Promoter holding remains structurally stable.", 0.85))
                else:
                    direction = 1.0 if prom_delta > 0 else -1.0
                    magnitude = _clip(abs(prom_delta) / 2.0, 0.1, 1.0)
                    prom_status = "buying" if direction > 0 else "selling"
                    action = "accumulation" if direction > 0 else "reduction"
                    nodes.append(EvidenceNode("promoter", direction, magnitude, 1.8, f"Promoter {action} detected ({prom_delta:+.2f}%).", 0.95))
            elif prom_pct is not None:
                prom_status = "stable"

            # --- PLEDGE DOMAIN ---
            pledge_pct, pledge_delta = self._extract_value_and_delta(rows, "pledge")
            if pledge_pct is not None and pledge_pct > 15.0:
                magnitude = _clip((pledge_pct - 15.0) / 35.0, 0.1, 1.0)
                nodes.append(EvidenceNode("pledge", -1.0, magnitude, 1.2, f"Significant promoter pledge ({pledge_pct:.1f}%) acts as corporate overhang.", 0.90))
            if pledge_delta is not None and abs(pledge_delta) > 0.05:
                direction = -1.0 if pledge_delta > 0 else 1.0 
                magnitude = _clip(abs(pledge_delta) / 5.0, 0.1, 1.0)
                action = "increase" if direction < 0 else "decrease"
                nodes.append(EvidenceNode("pledge", direction, magnitude, 1.5, f"Promoter pledge {action} by {abs(pledge_delta):.2f}%.", 0.90))

            # --- INSTITUTIONAL/FII/DII DOMAIN ---
            inst_pct, inst_delta = self._extract_value_and_delta(rows, "inst")
            fii_pct, fii_delta = self._extract_value_and_delta(rows, "fii")
            dii_pct, dii_delta = self._extract_value_and_delta(rows, "dii")
            
            combined_inst_delta = 0.0
            inst_delta_found = False
            
            if fii_delta is not None or dii_delta is not None:
                combined_inst_delta = (fii_delta or 0.0) + (dii_delta or 0.0)
                inst_delta_found = True
            elif inst_delta is not None:
                combined_inst_delta = inst_delta
                inst_delta_found = True

            has_inst_data = (inst_pct is not None or fii_pct is not None or dii_pct is not None or inst_delta_found)

            if inst_delta_found:
                if abs(combined_inst_delta) < 0.02:
                    nodes.append(EvidenceNode("institutional", 0.0, 1.0, 1.0, "Institutional ownership structure remains largely stable.", 0.85))
                else:
                    direction = 1.0 if combined_inst_delta > 0 else -1.0
                    magnitude = _clip(abs(combined_inst_delta) / 1.5, 0.1, 1.0)
                    action = "expanded" if direction > 0 else "contracted"
                    nodes.append(EvidenceNode("institutional", direction, magnitude, 1.6, f"Institutional ownership {action} by {abs(combined_inst_delta):.2f}%.", 0.92))

            # --- MARKET & CONTEXT DOMAIN ---
            c, o = _value(current_market, "close", "last_price"), _value(current_market, "open")
            price_chg = _value(current_market, "return_1d", "price_change")
            if price_chg is None and c is not None and o is not None and o > 0:
                price_chg = (c - o) / o * 100.0
            
            p_dir = 1.0 if (price_chg and price_chg > 0) else -1.0 if (price_chg and price_chg < 0) else 0.0
            p_mag = _clip(abs(price_chg or 0.0) / 3.0, 0.1, 1.0)

            bullish_c = _value(current_market, "bullish_candle") == 1.0
            bearish_c = _value(current_market, "bearish_candle") == 1.0
            micro_dir = 1.0 if bullish_c else -1.0 if bearish_c else p_dir

            # --- DELIVERY DOMAIN ---
            delivery_pct, _ = self._extract_value_and_delta(rows, "delivery")
            if delivery_pct is not None and price_chg is not None:
                if delivery_pct > 50.0:
                    magnitude = _clip((delivery_pct - 50.0) / 40.0, 0.1, 1.0) * p_mag
                    if p_dir > 0:
                        nodes.append(EvidenceNode("delivery", 1.0, magnitude, 1.0, f"High delivery ({delivery_pct:.1f}%) validates upside price accumulation.", 0.88))
                    elif p_dir < 0:
                        nodes.append(EvidenceNode("delivery", -1.0, magnitude, 1.0, f"High delivery ({delivery_pct:.1f}%) validates downside price distribution.", 0.88))
                elif delivery_pct < 30.0:
                    nodes.append(EvidenceNode("delivery", 0.0, 0.5, 0.8, f"Low delivery ({delivery_pct:.1f}%) implies largely speculative volume.", 0.80))

            # --- VOLUME DOMAIN ---
            rvol = _value(current_market, "rvol_20", "volume_ratio")
            vol_z = _value(current_market, "vol_zscore")
            vol_magnitude = 0.0
            
            if vol_z is not None and vol_z > 1.0:
                vol_magnitude = _clip((vol_z - 1.0) / 3.0, 0.2, 1.0)
            elif rvol is not None and rvol > 1.2:
                vol_magnitude = _clip((rvol - 1.2) / 2.0, 0.1, 0.8)

            if vol_magnitude > 0 and p_dir != 0:
                nodes.append(EvidenceNode("volume", p_dir, vol_magnitude, 1.3, "Significant volume expansion confirms current directional pressure.", 0.85))

            # --- SMART MONEY / MICRO DOMAIN ---
            smc = _value(current_market, "smart_money_candle")
            inst_body = _value(current_market, "institutional_body")
            absorption = _value(current_market, "absorption_candle")
            liq_sweep = _value(current_market, "liquidity_sweep_candle")
            rejection = _value(current_market, "rejection_candle")

            if (smc == 1.0 or inst_body == 1.0) and micro_dir != 0:
                nodes.append(EvidenceNode("micro", micro_dir, 0.8, 1.5, "Candle footprint matches institutional block execution.", 0.85))
            if absorption == 1.0 and micro_dir != 0:
                action_str = "supply" if micro_dir > 0 else "demand"
                nodes.append(EvidenceNode("micro", micro_dir, 0.7, 1.4, f"Absorption pattern suggests consumption of opposing {action_str}.", 0.82))
            if liq_sweep == 1.0 and micro_dir != 0:
                nodes.append(EvidenceNode("micro", micro_dir, 0.75, 1.6, "Liquidity sweep indicates trapping of retail participants.", 0.85))
            if rejection == 1.0 and micro_dir != 0:
                nodes.append(EvidenceNode("micro", micro_dir, 0.6, 1.2, "Strong rejection confirms boundary defense.", 0.80))

            # --- STRUCTURAL (PRECALC) DOMAIN ---
            precalc_acc = _value(current_market, "accumulation")
            precalc_dist = _value(current_market, "distribution")
            if precalc_acc is not None and precalc_acc > 0.5:
                nodes.append(EvidenceNode("structure", 1.0, precalc_acc, 1.0, "Underlying algorithmic structure signals accumulation bias.", 0.80))
            if precalc_dist is not None and precalc_dist > 0.5:
                nodes.append(EvidenceNode("structure", -1.0, precalc_dist, 1.0, "Underlying algorithmic structure signals distribution bias.", 0.80))

            # --- CONFLUENCE AGGREGATION ENGINE ---
            total_weight = sum(n.weight for n in nodes) if nodes else 1.0
            
            buy_power_raw = sum(n.direction * n.magnitude * n.weight for n in nodes if n.direction > 0)
            acc_score = _clip((buy_power_raw / max(total_weight, 1.0)) * 100.0 * 2.0, 0.0, 100.0) 
            
            sell_power_raw = sum(abs(n.direction) * n.magnitude * n.weight for n in nodes if n.direction < 0)
            dist_score = _clip((sell_power_raw / max(total_weight, 1.0)) * 100.0 * 2.0, 0.0, 100.0)

            shareholding_quality = 0.0
            if prom_pct is not None: 
                shareholding_quality += _clip(prom_pct, 0.0, 75.0)
            if inst_pct is not None: 
                shareholding_quality += _clip(inst_pct, 0.0, 40.0)
            elif fii_pct is not None or dii_pct is not None: 
                shareholding_quality += _clip((fii_pct or 0.0) + (dii_pct or 0.0), 0.0, 40.0)
            if pledge_pct is not None: 
                shareholding_quality -= _clip(pledge_pct * 0.5, 0.0, 30.0)
            shareholding_quality = _clip(shareholding_quality, 0.0, 100.0)

            inst_buy_nodes = sum(n.magnitude * n.weight for n in nodes if n.domain in ["institutional", "delivery", "volume", "micro"] and n.direction > 0)
            inst_sell_nodes = sum(n.magnitude * n.weight for n in nodes if n.domain in ["institutional", "delivery", "volume", "micro"] and n.direction < 0)
            
            inst_buying = _clip(inst_buy_nodes * 25.0, 0.0, 100.0)
            inst_selling = _clip(inst_sell_nodes * 25.0, 0.0, 100.0)
            
            inst_score = max(inst_buying, inst_selling)
            if not has_inst_data and delivery_pct is None:
                inst_status = "unknown"
                inst_score = 0.0
            elif inst_buying > inst_selling * 1.2:
                inst_status = "buying"
            elif inst_selling > inst_buying * 1.2:
                inst_status = "selling"
            else:
                inst_status = "neutral"

            inst_delta_abs = abs(combined_inst_delta) if inst_delta_found else 0.0
            prom_delta_abs = abs(prom_delta) if prom_delta is not None else 0.0
            variance_penalty = _clip((inst_delta_abs + prom_delta_abs) * 15.0, 0.0, 80.0)
            ownership_stability = _clip(100.0 - variance_penalty, 0.0, 100.0)

            promoter_score = _clip(sum(n.magnitude * 100.0 for n in nodes if n.domain == "promoter"), 0.0, 100.0)
            promoter_conf = 100.0 if prom_pct is not None else 0.0

            acc_status = "strong" if acc_score > 60.0 else "weak" if acc_score > 20.0 else "neutral"
            dist_status = "strong" if dist_score > 60.0 else "weak" if dist_score > 20.0 else "neutral"
            
            delivery_val = delivery_pct or 0.0
            del_status = "high" if delivery_val > 55.0 else "low" if (delivery_val > 0 and delivery_val < 35.0) else "neutral"
            
            if prom_pct is None and not has_inst_data:
                shareholding_status = "unknown"
            else:
                shareholding_status = "stable" if ownership_stability > 50.0 else "unstable"

            evidence_output = []
            for n in nodes:
                lr = _lr(n.magnitude * n.weight)
                if n.direction != 0: lr = 1.0 + (lr - 1.0)
                evidence_output.append(self._evidence("Institutional", n.message, n.reliability, lr))

            if acc_score > 60 and dist_score > 60:
                evidence_output.append(self._evidence("Institutional", "Severe contradiction between aggressive buying and selling signals creates a high-churn/neutral state.", 0.95, 0.8))

            coverage_domains = 0
            if prom_pct is not None or prom_delta is not None: coverage_domains += 1
            if inst_delta_found or inst_pct is not None: coverage_domains += 1
            if delivery_pct is not None: coverage_domains += 1
            if rvol is not None: coverage_domains += 1
            if smc is not None or liq_sweep is not None: coverage_domains += 1
            
            coverage_ratio = coverage_domains / 5.0
            base_conf = (coverage_ratio * 60.0) + (min(len(nodes), 5) * 8.0)
            
            if acc_score > 40 and dist_score > 40:
                base_conf -= 20.0

            final_conf = _clip(base_conf, 0.0, 100.0)

            return {
                "institutional_analyzer": {
                    "confidence": _safe_float(round(final_conf, 4)),
                    "institutional_buying": _safe_float(round(inst_buying, 4)),
                    "institutional_selling": _safe_float(round(inst_selling, 4)),
                    "promoter_confidence": _safe_float(round(promoter_conf, 4)),
                    "ownership_stability": _safe_float(round(ownership_stability, 4)),
                    "accumulation": _safe_float(round(acc_score, 4)),
                    "accumulation_status": acc_status,
                    "distribution": _safe_float(round(dist_score, 4)),
                    "distribution_status": dist_status,
                    "institutional": _safe_float(round(inst_score, 4)),
                    "institutional_status": inst_status,
                    "promoter": _safe_float(round(promoter_score, 4)),
                    "promoter_status": prom_status,
                    "delivery": _safe_float(round(delivery_val, 4)),
                    "delivery_status": del_status,
                    "shareholding": _safe_float(round(shareholding_quality, 4)), 
                    "shareholding_status": shareholding_status,
                    "evidence": self._unique_evidence(evidence_output)
                }
            }

        except Exception as e:
            self.logger.exception(f"Institutional Analyzer Critical Failure: {e}")
            result = self._empty()
            result["institutional_analyzer"]["evidence"].append(self._evidence("Institutional", "Execution encountered a critical failure.", 0.96, 0.42))
            return result

    def _empty(self) -> Dict[str, Any]:
        return {
            "institutional_analyzer": {
                "confidence": 0.0, "institutional_buying": 0.0, "institutional_selling": 0.0,
                "promoter_confidence": 0.0, "ownership_stability": 0.0,
                "accumulation": 0.0, "accumulation_status": "neutral",
                "distribution": 0.0, "distribution_status": "neutral",
                "institutional": 0.0, "institutional_status": "unknown",
                "promoter": 0.0, "promoter_status": "unknown",
                "delivery": 0.0, "delivery_status": "neutral",
                "shareholding": 0.0, "shareholding_status": "unknown",
                "evidence": []
            }
        }

    def _flatten_db_payload(self, data: Any) -> Dict[str, Any]:
        flat = {}
        if not isinstance(data, dict): return flat
        for k, v in data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[_key(sub_k)] = sub_v
            else:
                flat[_key(k)] = v
        return flat

    def _rows(self, data: Any) -> List[Mapping[str, Any]]:
        raw_rows = []
        if data is None: return []

        if hasattr(data, "to_dict") and hasattr(data, "columns"):
            try: raw_rows = data.to_dict(orient="records")
            except Exception: return []
        elif isinstance(data, Mapping):
            found_nested = False
            for name in ("features", "data", "payload", "rows", "historical_data", "fundamental_data", "shareholding_data"):
                nested = data.get(name)
                if isinstance(nested, list):
                    raw_rows.extend(nested)
                    found_nested = True
                elif isinstance(nested, dict):
                    raw_rows.append(nested)
                    found_nested = True
            if not found_nested: raw_rows = [self._flatten_db_payload(data)]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            raw_rows = [self._flatten_db_payload(item) if isinstance(item, dict) else item for item in data if isinstance(item, Mapping)]

        normalized_rows = []
        for row in raw_rows:
            normalized = {}
            for k, v in row.items(): normalized[_key(k)] = v
            normalized_rows.append(normalized)
            
        return normalized_rows

    def _get_latest_market_row(self, rows: List[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        for i in range(len(rows) - 1, -1, -1):
            c = _value(rows[i], "close", "last_price", "cmp")
            if c is not None and c > 0:
                return rows[i]
        return rows[-1] if rows else None

    def _extract_value_and_delta(self, rows: List[Mapping[str, Any]], domain: str) -> Tuple[Optional[float], Optional[float]]:
        pct_names = PCT_KEYS.get(domain, [])
        chg_names = CHG_KEYS.get(domain, [])
        
        latest_pct, prev_pct = None, None
        explicit_delta = None

        for i in range(len(rows)-1, -1, -1):
            d_val = _value(rows[i], *chg_names)
            if d_val is not None and abs(d_val) <= 100.0:
                explicit_delta = d_val
                break

        for i in range(len(rows)-1, -1, -1):
            p_val = _value(rows[i], *pct_names)
            if p_val is not None and abs(p_val) <= 100.0:
                if latest_pct is None:
                    latest_pct = p_val
                elif p_val != latest_pct and prev_pct is None:
                    prev_pct = p_val
                    break
                    
        delta = explicit_delta
        if delta is None and latest_pct is not None and prev_pct is not None:
            delta = latest_pct - prev_pct
            
        return latest_pct, delta

    def _evidence(self, category: str, message: str, reliability: float, likelihood_ratio: float) -> Dict[str, Any]:
        return {
            "category": str(category),
            "message": str(message),
            "reliability": _safe_float(round(_clip(reliability, 0.0, 1.0), 6)),
            "likelihood_ratio": _safe_float(round(_clip(likelihood_ratio, MIN_LR, MAX_LR), 6)),
        }

    def _unique_evidence(self, evidence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen = set()
        for item in evidence:
            key = (item.get("category"), item.get("message"))
            if key in seen: continue
            seen.add(key)
            result.append(item)
        
        result.sort(key=lambda x: x.get("likelihood_ratio", 1.0), reverse=True)
        return result[:8]


def analyze(data: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    return InstitutionalAnalyzer().analyze(data, *args, **kwargs)

__all__ = ["InstitutionalAnalyzer", "analyze"]
