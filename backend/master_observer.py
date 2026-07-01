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
from typing import Dict, List, Any, Optional, Tuple, Type, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from backend.registry.feature_engine import build_features

warnings.filterwarnings("ignore")
EPSILON = 1e-9

# ==========================================================
# THREAD-SAFE DEPENDENCY & MISSING FEATURE TRACKER
# ==========================================================
_access_log_var = contextvars.ContextVar('access_log', default=None)

_orig_getitem = pd.DataFrame.__getitem__
_orig_get = pd.DataFrame.get
_orig_getattr = pd.DataFrame.__getattr__

def _thread_safe_getitem(self, key):
    log = _access_log_var.get()
    if log is not None:
        keys = [key] if isinstance(key, str) else key if isinstance(key, list) else []
        for k in keys:
            if isinstance(k, str):
                if k in self.columns: log["used"].add(k)
                else: log["missing"].add(k)
    return _orig_getitem(self, key)

def _thread_safe_get(self, key, default=None):
    log = _access_log_var.get()
    if log is not None and isinstance(key, str):
        if key in self.columns: log["used"].add(key)
        else: log["missing"].add(key)
    return _orig_get(self, key, default)

def _thread_safe_getattr(self, name):
    log = _access_log_var.get()
    if log is not None and name in self.columns:
        log["used"].add(name)
    return _orig_getattr(self, name)

pd.DataFrame.__getitem__ = _thread_safe_getitem
pd.DataFrame.get = _thread_safe_get
pd.DataFrame.__getattr__ = _thread_safe_getattr

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
# DATA STRUCTURES & BATCH SUMMARY
# ==========================================================
class ValidationSeverity(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    SOFT_FAIL = "SOFT_FAIL"
    HARD_FAIL = "HARD_FAIL"
    BLOCK = "BLOCK"
    CRITICAL = "CRITICAL"

@dataclass
class MasterReport:
    symbol: str
    timestamp: float = field(default_factory=time.time)
    health: float = 100.0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    lineage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    final_decision: Dict[str, Any] = field(default_factory=dict)
    total_features_count: int = 0
    analyzer_status: Dict[str, str] = field(default_factory=dict)

    def add_issue(self, layer: str, severity: ValidationSeverity, msg: str):
        self.issues.append({"layer": layer, "severity": severity.value, "message": msg})
        penalties = {"WARNING": 5.0, "SOFT_FAIL": 15.0, "HARD_FAIL": 30.0, "BLOCK": 50.0, "CRITICAL": 100.0}
        self.health = max(0.0, self.health - penalties.get(severity.value, 0.0))

class BatchSummary:
    def __init__(self):
        self.total_stocks = 0
        self.analyzer_passes = defaultdict(int)
        self.global_used_features = set()
        self.global_missing_features = set()
        self.total_feature_space = 0
        self.health_scores = []

    def add_report(self, report: MasterReport):
        self.total_stocks += 1
        self.health_scores.append(report.health)
        self.total_feature_space = max(self.total_feature_space, report.total_features_count)

        for name, status in report.analyzer_status.items():
            if status == "PASS":
                self.analyzer_passes[name] += 1

        for data in report.lineage.values():
            self.global_used_features.update(data["used"])
            self.global_missing_features.update(data["missing"])

# ==========================================================
# AUTO-DISCOVERY ENGINE (Contract via __getattr__)
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

                        # Dynamic Contract Loading
                        self.contracts[clean_name] = getattr(obj, "EXPECTED_SCHEMA", [])
                        self.dependencies[clean_name] = getattr(obj, "DEPENDS_ON", [])
        except Exception:
            self._setup_mock_registry()

    def _setup_mock_registry(self):
        class MockTrendAnalyzer:
            EXPECTED_SCHEMA = ["direction.status", "strength.status", "phase.status"]
            def analyze(self, df): return {"direction": {"status": "Bullish"}, "strength": {"status": "Strong"}, "phase": {"status": "Markup"}}
        class MockMomentumAnalyzer:
            EXPECTED_SCHEMA = ["momentum_strength.status", "acceleration.status"]
            def analyze(self, df): return {"momentum_strength": {"status": "Bullish"}, "acceleration": {"status": "Expanding"}}

        self.analyzers = {"Trend": MockTrendAnalyzer, "Momentum": MockMomentumAnalyzer}
        self.contracts = {k: getattr(v, "EXPECTED_SCHEMA") for k, v in self.analyzers.items()}
        self.dependencies = {"Trend": [], "Momentum": []}

# ==========================================================
# LAYER 1: DATA QA & FEATURE FRESHNESS
# ==========================================================
class FeatureQA:
    @staticmethod
    def validate(df: pd.DataFrame, report: MasterReport):
        if df is None or df.empty:
            report.add_issue("L1_Feature", ValidationSeverity.CRITICAL, "DataFrame is completely empty.")
            return

        report.total_features_count = len(df.columns)
        num_df = df.select_dtypes(include=[np.number])

        # Inf Check
        inf_count = np.isinf(num_df).sum().sum()
        if inf_count > 0:
            report.add_issue("L1_Feature", ValidationSeverity.HARD_FAIL, f"Found {inf_count} INF values.")

        # Stale / Freshness Check
        if len(num_df) > 10:
            tail_df = num_df.tail(10)
            stds = tail_df.std()
            stale_cols = []

            for col in tail_df.columns:
                if col in ['open', 'high', 'low', 'close', 'volume']: continue

                # CRITICAL FIX: Ignore binary/categorical flags (if the column has 3 or fewer unique values like -1, 0, 1)
                if df[col].nunique(dropna=True) <= 3:
                    continue

                if stds[col] == 0.0:  # Continuous feature unchanged for 10 bars
                    stale_cols.append(col)

            if stale_cols:
                report.add_issue("L1_Feature", ValidationSeverity.WARNING, f"STALE FEATURE (Flatlined): {stale_cols[:5]}...")



# ==========================================================
# LAYER 2: DAG EXECUTION & RAM/TIME PROFILER
# ==========================================================
class AnalyzerDAG:
    def __init__(self, registry: DynamicRegistry):
        self.registry = registry
        tracemalloc.start()

    def _instantiate_safely(self, cls: Type) -> Any:
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters if p != 'self']
        if params: return cls(config={})
        return cls()

    def _run_single(self, name: str, analyzer_cls: Type, df: pd.DataFrame) -> Tuple[str, Dict, Dict[str, Set[str]], float, float, str]:
        t0 = time.perf_counter()
        mem_before, _ = tracemalloc.get_traced_memory()

        error, output, feat_log = "", {}, {"used": set(), "missing": set()}
        try:
            instance = self._instantiate_safely(analyzer_cls)
            with ThreadSafeFeatureTracker() as log_dict:
                output = instance.analyze(df)
            feat_log = log_dict
        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]

        mem_after, _ = tracemalloc.get_traced_memory()
        exec_time_ms = (time.perf_counter() - t0) * 1000.0
        ram_mb = max(0.0, (mem_after - mem_before) / 10**6)

        return name, output, feat_log, exec_time_ms, ram_mb, error

    def _execute_wrapper(self, ctx, *args, **kwargs):
        return ctx.run(self._run_single, *args, **kwargs)

    def execute_all(self, df: pd.DataFrame, report: MasterReport) -> Dict[str, Any]:
        results = {}
        graph = {name: set(deps) for name, deps in self.registry.dependencies.items()}
        
        # Pre-flight check: Analyzer gula ki ki feature chay ta check kora
        for name, cls in self.registry.analyzers.items():
            instance = self._instantiate_safely(cls)
            # Jodio analyzer e 'req_cols' thake, check kore nibe
            if hasattr(instance, 'req_cols'):
                missing = [c for c in instance.req_cols if c not in df.columns]
                if missing:
                    report.add_issue("L2_PreFlight", ValidationSeverity.BLOCK, f"[{name}] Missing Critical Features: {missing}")
                    # Missing list e add korlam jate radar e dhara pore
                    report.lineage[name] = {"used": [], "missing": missing}

        try:
            sorter = graphlib.TopologicalSorter(graph)
            sorter.prepare()
        except: return results

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while sorter.is_active():
                ready_nodes = sorter.get_ready()
                if not ready_nodes: break
                
                futures = {}
                for node in ready_nodes:
                    cls = self.registry.analyzers.get(node)
                    if cls:
                        # Jodi agei block hoye thake tobe skip korbe
                        if report.analyzer_status.get(node) == "FAIL":
                            sorter.done(node)
                            continue
                            
                        ctx = contextvars.copy_context()
                        futures[executor.submit(self._execute_wrapper, ctx, node, cls, df)] = node

                for future in concurrent.futures.as_completed(futures):
                    node = futures[future]
                    name, output, feat_log, t_ms, ram_mb, err = future.result()
                    
                    report.execution_metrics[name] = {"time_ms": round(t_ms, 2), "ram_mb": round(ram_mb, 4)}
                    # Merge tracker logs with pre-flight logs
                    existing_missing = report.lineage.get(name, {}).get("missing", [])
                    report.lineage[name] = {
                        "used": list(feat_log["used"]), 
                        "missing": list(set(list(feat_log["missing"]) + existing_missing))
                    }
                    
                    if err:
                        report.add_issue("L2_Analyzer", ValidationSeverity.HARD_FAIL, f"[{name}] Crashed: {err}")
                        report.analyzer_status[name] = "FAIL"
                    else:
                        report.analyzer_status[name] = "PASS"
                    
                    results[name] = output
                    sorter.done(node)
        return results


    def _flatten_keys(self, d: Dict, prefix="") -> Set[str]:
        keys = set()
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            if isinstance(v, dict): keys.update(self._flatten_keys(v, full_key))
        return keys

# ==========================================================
# MASTER OS ORCHESTRATOR
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

            try:
                from backend.registry.feature_engine import build_features
                df = build_features(df)
            except Exception:
                pass # Proceed with raw data if engine missing for tests

            FeatureQA.validate(df, report)

            # Layer-2: DAG Execution (Observations Only)
            l2_results = self.dag_engine.execute_all(df, report)

            # Layer-3: Awaiting Scoring Engines. Force HOLD.
            report.final_decision = {"signal": "HOLD", "reason": "Layer-3 Scoring Engine pending."}

            if report.health < 40.0:
                report.final_decision = {"signal": "BLOCK", "reason": "System health critical."}

        except Exception as e:
            report.add_issue("OS_Core", ValidationSeverity.CRITICAL, f"Fatal Crash: {str(e)}")
            report.final_decision = {"signal": "BLOCK", "reason": str(e)}

        self.batch_summary.add_report(report)
        self._print_single_report(report)
        return report

    def _print_single_report(self, report: MasterReport):
        print(f"\n[{report.symbol}] QA Health: {report.health}% | Final Decision: {report.final_decision['signal']}")
        for name, metrics in report.execution_metrics.items():
            lineage = report.lineage.get(name, {})
            used_feats = lineage.get("used", [])
            missing_feats = lineage.get("missing", [])
            
            # RAM, Time এবং Used Features প্রিন্ট
            print(f"  -> {name:<12}: {len(used_feats):<3} Features | {metrics['time_ms']:<6.2f} ms | RAM: {metrics['ram_mb']:.4f} MB")
            
            # মিসিং ফিচার সবসময় দেখাবে (0 হলেও)
            if missing_feats:
                print(f"     [!] MISSING ({len(missing_feats)}): {list(missing_feats)}")
            else:
                print(f"     [!] MISSING: 0")

        if report.issues:
            for iss in report.issues[:3]:
                print(f"     [!] {iss['severity']} - {iss['layer']}: {iss['message']}")

    def print_final_summary(self):
        print(f"\n{'='*40}")
        print("QA SUMMARY")
        print(f"Stocks = {self.batch_summary.total_stocks}")

        for name in self.registry.analyzers.keys():
            passes = self.batch_summary.analyzer_passes.get(name, 0)
            print(f"{name} Pass = {passes}")

        used_count = len(self.batch_summary.global_used_features)
        total_feats = self.batch_summary.total_feature_space
        unused_count = max(0, total_feats - used_count)
        coverage = (used_count / total_feats * 100) if total_feats > 0 else 0.0

        print("\nFeature Coverage")
        print(f"{coverage:.1f}%")
        print("Unused Features")
        print(f"{unused_count}")
        print("Missing Features")
        print(f"{len(self.batch_summary.global_missing_features)}")
        print("System Health")
        avg_health = np.mean(self.batch_summary.health_scores) if self.batch_summary.health_scores else 0.0
        print(f"{avg_health:.1f}%")
        print(f"{'='*40}\n")

# ==========================================================
# RUNNER
# ==========================================================
if __name__ == "__main__":
    test_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    os_engine = MasterObserver()

    for stock in test_stocks:
        os_engine.run_pipeline(stock)
        time.sleep(0.1)

    os_engine.print_final_summary()
