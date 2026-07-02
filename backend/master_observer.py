import warnings
import time
import traceback
import importlib
import inspect
import pkgutil
import contextvars
import concurrent.futures
import graphlib
import tracemalloc
import numpy as np
import pandas as pd
from backend.data.data_fetcher import fetch_ohlcv
from typing import Dict, List, Any, Tuple, Type, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from backend.registry.feature_engine import build_features

warnings.filterwarnings("ignore")

# ==========================================================
# 1. RUNTIME AUDITOR (Monkey Patch for 100% True 'Used' Count)
# ==========================================================
_access_log_var = contextvars.ContextVar('access_log', default=None)

_orig_df_getitem = pd.DataFrame.__getitem__
_orig_df_get = pd.DataFrame.get
_orig_series_getitem = pd.Series.__getitem__
_orig_series_get = pd.Series.get

def _log_key_access(obj, key, is_series=False):
    log = _access_log_var.get()
    if log is not None:
        keys = [key] if isinstance(key, str) else (key if isinstance(key, list) else [])
        for k in keys:
            if isinstance(k, str) and k not in ['open', 'high', 'low', 'close', 'volume']:
                available_keys = obj.index if is_series else obj.columns
                if k in available_keys:
                    log["used"].add(k)
                else:
                    log["missing"].add(k)

def _thread_safe_df_getitem(self, key):
    _log_key_access(self, key, is_series=False)
    return _orig_df_getitem(self, key)

def _thread_safe_df_get(self, key, default=None):
    _log_key_access(self, key, is_series=False)
    return _orig_df_get(self, key, default)

def _thread_safe_series_getitem(self, key):
    _log_key_access(self, key, is_series=True)
    return _orig_series_getitem(self, key)

def _thread_safe_series_get(self, key, default=None):
    _log_key_access(self, key, is_series=True)
    return _orig_series_get(self, key, default)

class DataFrameMonkeyPatch:
    def __enter__(self):
        pd.DataFrame.__getitem__ = _thread_safe_df_getitem
        pd.DataFrame.get = _thread_safe_df_get
        pd.Series.__getitem__ = _thread_safe_series_getitem
        pd.Series.get = _thread_safe_series_get
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pd.DataFrame.__getitem__ = _orig_df_getitem
        pd.DataFrame.get = _orig_df_get
        pd.Series.__getitem__ = _orig_series_getitem
        pd.Series.get = _orig_series_get

class ThreadSafeFeatureTracker:
    def __init__(self):
        self.log_dict = {"used": set(), "missing": set()}
        self.token = None

    def __enter__(self) -> Dict[str, Set[str]]:
        self.token = _access_log_var.set(self.log_dict)
        return self.log_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _access_log_var.reset(self.token)

# ==========================================================
# 2. DATA STRUCTURES & BATCH SUMMARY
# ==========================================================
class ValidationSeverity(str, Enum):
    PASS, WARNING, SOFT_FAIL, HARD_FAIL, BLOCK, CRITICAL = "PASS", "WARNING", "SOFT_FAIL", "HARD_FAIL", "BLOCK", "CRITICAL"

@dataclass
class MasterReport:
    symbol: str
    timestamp: float = field(default_factory=time.time)
    health: float = 100.0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    lineage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_features_count: int = 0
    analyzer_status: Dict[str, str] = field(default_factory=dict)

    def add_issue(self, layer: str, severity: ValidationSeverity, msg: str):
        self.issues.append({"layer": layer, "severity": severity.value, "message": msg})
        penalties = {"WARNING": 2.0, "SOFT_FAIL": 10.0, "HARD_FAIL": 25.0, "BLOCK": 50.0, "CRITICAL": 100.0}
        self.health = max(0.0, self.health - penalties.get(severity.value, 0.0))

class BatchSummary:
    def __init__(self):
        self.total_stocks = 0
        self.analyzer_passes = defaultdict(int)
        self.health_scores = []
        self.total_req_feature = 0
        self.total_feature_used = 0
        self.total_missing_feature = 0
        self.total_listed_feature = 0

    def add_report(self, report: MasterReport, req: int, used: int, l1_miss: int, unused: int):
        self.total_stocks += 1
        self.health_scores.append(report.health)
        
        for name, status in report.analyzer_status.items():
            if status == "PASS":
                self.analyzer_passes[name] += 1
                
        self.total_req_feature += req
        self.total_feature_used += used
        self.total_missing_feature += (l1_miss + unused)
        self.total_listed_feature += report.total_features_count


# ==========================================================
# 3. REGISTRY (Absolute Explicit Contracts Only)
# ==========================================================
class DynamicRegistry:
    def __init__(self):
        self.analyzers: Dict[str, Type] = {}
        self.contracts: Dict[str, List[str]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.discover()

    def discover(self):
        try:
            from backend import analyzers
            for _, module_name, _ in pkgutil.iter_modules(analyzers.__path__):
                mod = importlib.import_module(f"backend.analyzers.{module_name}")
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if name.endswith("Analyzer") and hasattr(obj, "analyze"):
                        clean_name = name.replace("Analyzer", "")
                        self.analyzers[clean_name] = obj
                        
                        # একমাত্র সত্য: EXPECTED_SCHEMA
                        # কোনো ম্যাজিক স্ক্যানার নেই। না থাকলে 0 হবে।
                        schema = getattr(obj, "EXPECTED_SCHEMA", [])
                        self.contracts[clean_name] = schema
                        self.dependencies[clean_name] = getattr(obj, "DEPENDS_ON", [])
        except Exception as e:
            print(f"CRITICAL: Discovery Engine Failed: {e}")


# ==========================================================
# 4. DATA QA
# ==========================================================
class FeatureQA:
    @staticmethod
    def validate(df: pd.DataFrame, report: MasterReport):
        if df is None or df.empty: return
        report.total_features_count = len(df.columns)


# ==========================================================
# 5. DAG EXECUTION & 80% COMPLIANCE CHECK
# ==========================================================
class AnalyzerDAG:
    def __init__(self, registry: DynamicRegistry):
        self.registry = registry

    def _instantiate_safely(self, cls: Type) -> Any:
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters if p != 'self']
        return cls(config={}) if params else cls()

    def _run_single(self, name: str, analyzer_cls: Type, df: pd.DataFrame) -> Tuple[str, Dict, Dict[str, Set[str]], float, float, str]:
        t0 = time.perf_counter()
        tracemalloc.start()
        mem_before, _ = tracemalloc.get_traced_memory()

        error, output, feat_log = "", {}, {"used": set(), "missing": set()}
        try:
            instance = self._instantiate_safely(analyzer_cls)
            with ThreadSafeFeatureTracker() as log_dict:
                output = instance.analyze(df)
            feat_log = log_dict
        except Exception:
            error = traceback.format_exc().splitlines()[-1]

        mem_after, _ = tracemalloc.get_traced_memory()
        exec_time_ms = (time.perf_counter() - t0) * 1000.0
        ram_mb = max(0.0, (mem_after - mem_before) / 10**6)
        tracemalloc.stop()

        return name, output, feat_log, exec_time_ms, ram_mb, error

    def _execute_wrapper(self, ctx, *args, **kwargs):
        return ctx.run(self._run_single, *args, **kwargs)

    def execute_all(self, df: pd.DataFrame, report: MasterReport) -> Dict[str, Any]:
        results = {}
        graph = {name: set(deps) for name, deps in self.registry.dependencies.items()}
        try:
            sorter = graphlib.TopologicalSorter(graph)
            sorter.prepare()
        except: return results

        with DataFrameMonkeyPatch(), concurrent.futures.ThreadPoolExecutor() as executor:
            while sorter.is_active():
                ready_nodes = sorter.get_ready()
                if not ready_nodes: break
                
                futures = {}
                for node in ready_nodes:
                    cls = self.registry.analyzers.get(node)
                    if not cls: continue
                    
                    expected = set(self.registry.contracts.get(node, []))
                    available = set(df.columns)
                    
                    present_features = expected.intersection(available)
                    l1_missing = expected - available
                    
                    # যদি কন্ট্রাক্টে কিছু না থাকে, আমরা কভারেজ 1.0 ধরে রান হতে দেব (যাতে আটকে না যায়)
                    coverage = (len(present_features) / len(expected)) if expected else 1.0
                    
                    if coverage >= 0.8:
                        ctx = contextvars.copy_context()
                        futures[executor.submit(self._execute_wrapper, ctx, node, cls, df)] = (node, expected, l1_missing)
                    else:
                        report.lineage[node] = {
                            "req": len(expected),
                            "used": [],
                            "l1_missing": list(l1_missing),
                            "unused_contract": list(present_features) 
                        }
                        report.add_issue(node, ValidationSeverity.BLOCK, f"Coverage {coverage*100:.1f}% (<80%). Missing: {list(l1_missing)}")
                        report.analyzer_status[node] = "BLOCKED"
                        results[node] = {"summary": {"action": f"BLOCKED (<80% Data)"}}
                        sorter.done(node)

                for future in concurrent.futures.as_completed(futures):
                    node, expected, l1_missing = futures[future]
                    name, output, feat_log, t_ms, ram_mb, err = future.result()

                    report.execution_metrics[name] = {"time_ms": round(t_ms, 2), "memory_mb": round(ram_mb, 4)}

                    # রানটাইমে যা যা আসল ইউজ হয়েছে (MonkeyPatch থেকে প্রাপ্ত)
                    runtime_used = (set(feat_log["used"]) - {'open', 'high', 'low', 'close', 'volume'}).intersection(expected)
                    
                    present_features = expected - l1_missing
                    unused_contract = present_features - runtime_used

                    report.lineage[name] = {
                        "req": len(expected),
                        "used": list(runtime_used),
                        "l1_missing": list(l1_missing),
                        "unused_contract": list(unused_contract)
                    }

                    if err:
                        report.add_issue(name, ValidationSeverity.HARD_FAIL, err)
                        report.analyzer_status[name] = "FAIL"
                    else:
                        if l1_missing: report.add_issue(name, ValidationSeverity.WARNING, f"Missing {len(l1_missing)} L1 features")
                        report.analyzer_status[name] = "PASS"

                    results[name] = output
                    sorter.done(node)
        return results


# ==========================================================
# 6. MASTER OS ORCHESTRATOR & UI
# ==========================================================
class MasterObserver:
    def __init__(self):
        self.registry = DynamicRegistry()
        self.dag_engine = AnalyzerDAG(self.registry)
        self.batch_summary = BatchSummary()

    def run_pipeline(self, symbol: str) -> MasterReport:
        report = MasterReport(symbol=symbol)
        try:
            df = fetch_ohlcv(symbol, limit=500)
            if df.empty: raise ValueError("No Market Data")
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [str(x).lower() for x in df.columns]

            try: df = build_features(df)
            except Exception: pass

            FeatureQA.validate(df, report)
            l2_results = self.dag_engine.execute_all(df, report)

            # TERMINAL UI
            print(f"\n{'='*115}")
            print(f" {symbol}")
            print(f"{'Analyzer':<18} | {'Req':<4} | {'Used':<6} | {'L1 Miss':<8} | {'Unused':<7} | {'Analyzer Res':<18} | {'Missing List'}")
            print(f"{'-'*115}")

            total_req, total_used, total_l1_missing, total_unused = 0, 0, 0, 0

            for analyzer_name, result in l2_results.items():
                name_title = analyzer_name.title()
                lineage = report.lineage.get(analyzer_name, {})

                req = lineage.get("req", 0)
                fullfill = len(lineage.get("used", []))
                l1_miss_list = lineage.get("l1_missing", [])
                unused_list = lineage.get("unused_contract", [])

                l1_missing_cnt = len(l1_miss_list)
                unused_cnt = len(unused_list)

                missing_str = f"({','.join(l1_miss_list)})" if l1_miss_list else ""
                if len(missing_str) > 35: missing_str = missing_str[:32] + "...)"

                action = "N/A"
                status = report.analyzer_status.get(analyzer_name)

                if status == "FAIL": action = "CRASHED"
                elif status == "BLOCKED": action = "BLOCKED (<80%)"
                elif isinstance(result, dict):
                    if 'summary' in result: action = result['summary'].get('action', 'N/A')
                    else:
                        for k, v in result.items():
                            if isinstance(v, dict) and any(x in v for x in ['status', 'regime', 'dominance']):
                                action = str(v.get('status', v.get('regime', v.get('dominance')))); break

                if len(action) > 18: action = action[:15] + "..."

                print(f"{name_title:<18} | {req:<4} | {fullfill:<6} | {l1_missing_cnt:<8} | {unused_cnt:<7} | {action:<18} | {missing_str}")

                total_req += req
                total_used += fullfill
                total_l1_missing += l1_missing_cnt
                total_unused += unused_cnt

            print(f"{'='*115}")
            self.batch_summary.add_report(report, total_req, total_used, total_l1_missing, total_unused)

        except Exception as e:
            print(f"[{symbol}] FATAL: {e}")

        return report

    def print_final_summary(self):
        print(f"\n{'='*70}")
        print(f"{' FINAL GLOBAL QA SUMMARY ':^70}")
        print(f"{'='*70}")
        print(f"Stocks Processed     : {self.batch_summary.total_stocks}")
        print(f"{'-'*70}")

        for name in self.registry.analyzers.keys():
            passes = self.batch_summary.analyzer_passes.get(name, 0)
            print(f"{name:<20} : {passes}/{self.batch_summary.total_stocks} Executed")

        print(f"{'-'*70}")

        req = self.batch_summary.total_req_feature
        used = self.batch_summary.total_feature_used
        missing = self.batch_summary.total_missing_feature
        listed = self.batch_summary.total_listed_feature

        global_coverage = (used / req * 100) if req > 0 else 0.0

        print(f"Total Listed Feature : {listed}")
        print(f"Total Req Feature    : {req}")
        print(f"Feature Used         : {used}")
        print(f"Missing Feature      : {missing}")
        print(f"Global Coverage      : {global_coverage:.1f}%")

        avg_health = np.mean(self.batch_summary.health_scores) if self.batch_summary.health_scores else 0.0
        print(f"Avg System Health    : {avg_health:.1f}%")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    test_stocks = ["RELIANCE", "TCS", "INFY"]
    os_engine = MasterObserver()

    for stock in test_stocks:
        os_engine.run_pipeline(stock)
        time.sleep(0.1)

    os_engine.print_final_summary()
