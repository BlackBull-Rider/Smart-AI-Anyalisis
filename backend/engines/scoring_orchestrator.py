"""
GREEN BULL RIDER V6
Layer-3: Master Scoring Orchestrator
Module: scoring_orchestrator.py
"""

import time
import logging
from typing import Dict, Any, Union
from dataclasses import dataclass, asdict

from backend.engines.base_engine import (
    BaseEngine,
    EngineConfig,
    EvidenceGraph,
    OutputStatus,
    PipelineTrace
)

from backend.engines.trend_engine import calculate_trend_score
from backend.engines.momentum_engine import calculate_momentum_score
from backend.engines.volatility_engine import calculate_volatility_score
from backend.engines.volume_engine import calculate_volume_score
from backend.engines.fundamental_engine import calculate_fundamental_score
from backend.engines.institutional_engine import calculate_institutional_score
from backend.engines.ipo_engine import calculate_ipo_score
from backend.engines.swing_engine import calculate_swing_score
from backend.engines.long_term_engine import calculate_long_term_score
from backend.engines.compounder_engine import calculate_compounder_score
from backend.engines.smart_money_engine import calculate_smart_money_score

logger = logging.getLogger(__name__)

def get_orchestrator_profile() -> EngineConfig:
    return EngineConfig(
        profile_name="Master_Scoring_Orchestrator",
        version="2.1.6", 
        stage="Layer-3: Final Fusion",
        schema_version="1.0",
        api_version="v6",
        thresholds={
            "base_evidence": 50.0,
            "pos_multiplier": 5.0,
            "neg_multiplier": 5.0,
            "conflict_penalty": 15.0,
            "critical_success_ratio": 0.6 
        }
    )

class ScoringOrchestrator(BaseEngine):
    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_orchestrator_profile())
        
        self.isolated_engines = {
            "trend": calculate_trend_score,
            "momentum": calculate_momentum_score,
            "volatility": calculate_volatility_score,
            "volume": calculate_volume_score,
            "fundamental": calculate_fundamental_score,
            "institutional": calculate_institutional_score,
            "ipo": calculate_ipo_score,
            "smart_money": calculate_smart_money_score
        }
        
        self.fusion_engines = {
            "swing": calculate_swing_score,
            "long_term": calculate_long_term_score,
            "compounder": calculate_compounder_score
        }

    def calculate(self, l2_raw_data: Union[Dict[str, Any], None]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace = self._generate_trace(l2_raw_data, start_time, "ORCHESTRATOR")

        if not isinstance(l2_raw_data, dict) or not l2_raw_data:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            aggregated_results = {}
            master_evidence = []
            master_warnings = []
            
            total_pos = 0
            total_neg = 0
            total_conflicts = 0
            total_coverage = 0.0
            valid_engines = 0

            for name, engine_func in self.isolated_engines.items():
                data = l2_raw_data.get(name, {})
                aggregated_results[name] = engine_func(data)

            for name, engine_func in self.fusion_engines.items():
                aggregated_results[name] = engine_func(l2_raw_data)

            for name, res in aggregated_results.items():
                if "explanations" in res:
                    master_evidence.extend(res["explanations"].get("evidence", []))
                    master_warnings.extend(res["explanations"].get("warnings", []))
                
                if "evidence_graph" in res:
                    eg = res["evidence_graph"]
                    total_pos += eg.get("positive_count", 0)
                    total_neg += eg.get("negative_count", 0)
                    total_conflicts += eg.get("conflict_count", 0)
                    total_coverage += eg.get("evidence_coverage", 0.0)
                
                if res.get("status", {}).get("status") == "SUCCESS":
                    valid_engines += 1
                else:
                    total_conflicts += 1

            t = self.config.thresholds
            total_all_engines = len(self.isolated_engines) + len(self.fusion_engines)
            
            avg_coverage = (total_coverage / valid_engines) if valid_engines > 0 else 0.0
            master_evidence_score = self._normalize(
                t.get("base_evidence", 50.0) + 
                (total_pos * t.get("pos_multiplier", 5.0)) - 
                (total_neg * t.get("neg_multiplier", 5.0)) - 
                (total_conflicts * t.get("conflict_penalty", 15.0))
            )

            success_ratio = (valid_engines / total_all_engines) if total_all_engines > 0 else 0.0
            
            if success_ratio == 1.0:
                final_status = "SUCCESS"
                final_quality = "VALID"
            elif success_ratio >= t.get("critical_success_ratio", 0.6):
                final_status = "PARTIAL_SUCCESS"
                final_quality = "DEGRADED"
            else:
                final_status = "FAILED"
                final_quality = "INVALID"

            evidence_graph = EvidenceGraph(
                evidence_score=round(master_evidence_score, 2),
                positive_count=total_pos,
                negative_count=total_neg,
                conflict_count=total_conflicts,
                evidence_coverage=round(avg_coverage, 2)
            )

            payload = {
                "status": asdict(OutputStatus(status=final_status, quality=final_quality)),
                "master_scores": self._aggregate_scores(aggregated_results),
                "master_explanations": {
                    "evidence": sorted(list(set(master_evidence))),
                    "warnings": sorted(list(set(master_warnings)))
                },
                "engine_health": {k: v.get("status", {}).get("status") for k, v in aggregated_results.items()},
                "engine_outputs": aggregated_results,
                "evidence_graph": asdict(evidence_graph),
                "trace": asdict(trace)
            }

            return self._sanitize_json(payload)

        except Exception as e:
            logger.error(f"Orchestrator failure: {e}", exc_info=True)
            return self._sanitize_json(self._build_fallback(trace))

    def _aggregate_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Strictly fetches proper score keys containing 'overall' or 'investment_score'."""
        scores = {}
        for name, res in results.items():
            if res.get("status", {}).get("status") == "SUCCESS":
                engine_scores = res.get("scores", {})
                if not engine_scores:
                    continue
                
                primary_key = None
                
                for k in engine_scores.keys():
                    if 'overall' in k or 'investment_score' == k:
                        primary_key = k
                        break
                        
                if not primary_key:
                    numeric_keys = [k for k, v in engine_scores.items() if isinstance(v, (int, float))]
                    primary_key = numeric_keys[0] if numeric_keys else list(engine_scores.keys())[0]

                try:
                    scores[name] = float(engine_scores[primary_key])
                except:
                    scores[name] = 0.0
                    
        return scores

    def _build_fallback(self, trace: PipelineTrace) -> Dict[str, Any]:
        return {
            "status": asdict(OutputStatus(status="FAILED", quality="INVALID")),
            "master_scores": {},
            "trace": asdict(trace)
        }

def get_master_scorecard(l2_raw_data: Dict[str, Any]) -> Dict[str, Any]:
    return ScoringOrchestrator().calculate(l2_raw_data)
