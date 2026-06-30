from __future__ import annotations

import ast
import inspect
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Optional


# ============================================================
# NODE TYPES
# ============================================================

FEATURE_NODE = "feature"
PARAMETER_NODE = "parameter"
CALL_NODE = "call"
RETURN_NODE = "return"


# ============================================================
# DETECTED FEATURE
# ============================================================

@dataclass(slots=True)
class FeatureRequest:

    feature: str

    source: str

    line: int

    column: int

    optional: bool = False

    default: Any = None

    access_type: str = "getitem"


# ============================================================
# DETECTED CALL
# ============================================================

@dataclass(slots=True)
class FunctionCall:

    function: str

    line: int

    kwargs: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# DETECTED PARAMETER
# ============================================================

@dataclass(slots=True)
class Parameter:

    name: str

    default: Any


# ============================================================
# DETECTED CONTRACT
# ============================================================

@dataclass(slots=True)
class AnalyzerContract:

    analyzer: str

    file: str

    class_name: str

    analyze_function: str

    hash: str

    required_features: Dict[str, FeatureRequest] = field(default_factory=dict)

    function_calls: List[FunctionCall] = field(default_factory=list)

    parameters: Dict[str, Parameter] = field(default_factory=dict)

    outputs: Set[str] = field(default_factory=set)

    aliases: Dict[str, str] = field(default_factory=dict)

    imports: Set[str] = field(default_factory=set)

    dependencies: Set[str] = field(default_factory=set)

    globals: Set[str] = field(default_factory=set)

# ============================================================
# AST DETECTOR
# ============================================================

class AnalyzerVisitor(ast.NodeVisitor):

    def __init__(self):

        self.contract = None

        self.current_class = ""

        self.current_function = ""

    # -------------------------------------------------------

    def visit_Import(self, node):

        for n in node.names:

            self.contract.imports.add(n.name)

    # -------------------------------------------------------

    def visit_ImportFrom(self, node):

        if node.module:

            self.contract.imports.add(node.module)

    # -------------------------------------------------------

    def visit_ClassDef(self, node):

        self.current_class = node.name

        self.generic_visit(node)

    # -------------------------------------------------------

    def visit_FunctionDef(self, node):

        self.current_function = node.name

        if node.name == "analyze":

            for arg in node.args.args:

                if arg.arg != "self":

                    self.contract.parameters[arg.arg] = Parameter(

                        name=arg.arg,

                        default=None,

                    )

        self.generic_visit(node)

# -------------------------------------------------------

    def visit_Subscript(self, node):

        """
        df["rsi"]
        row["atr"]
        data["ema_20"]
        """

        try:

            base = None

            if isinstance(node.value, ast.Name):

                base = node.value.id

            if isinstance(node.slice, ast.Constant):

                if isinstance(node.slice.value, str):

                    feature = node.slice.value

                    self.contract.required_features[feature] = FeatureRequest(

                        feature=feature,

                        source=base,

                        line=node.lineno,

                        column=node.col_offset,

                        access_type="getitem",

                    )

        except Exception:

            pass

        self.generic_visit(node)

# -------------------------------------------------------
    # df.get("rsi")
    # features.get("atr")
    # -------------------------------------------------------

    def visit_Call(self, node):

        try:

            # ---------------------------------------------
            # dict.get(...)
            # ---------------------------------------------

            if isinstance(node.func, ast.Attribute):

                if node.func.attr == "get":

                    base = None

                    if isinstance(node.func.value, ast.Name):

                        base = node.func.value.id

                    if node.args:

                        arg = node.args[0]

                        if isinstance(arg, ast.Constant):

                            if isinstance(arg.value, str):

                                feature = arg.value

                                self.contract.required_features[feature] = FeatureRequest(

                                    feature=feature,

                                    source=base,

                                    line=node.lineno,

                                    column=node.col_offset,

                                    access_type="get",

                                    optional=True,

                                )

            # ---------------------------------------------
            # remember every function call
            # ---------------------------------------------

            func_name = None

            if isinstance(node.func, ast.Name):

                func_name = node.func.id

            elif isinstance(node.func, ast.Attribute):

                func_name = node.func.attr

            if func_name:

                kwargs = {}

                for kw in node.keywords:

                    try:

                        kwargs[kw.arg] = ast.literal_eval(kw.value)

                    except Exception:

                        kwargs[kw.arg] = "<dynamic>"

                self.contract.function_calls.append(

                    FunctionCall(

                        function=func_name,

                        line=node.lineno,

                        kwargs=kwargs,

                    )

                )

        except Exception:

            pass

        self.generic_visit(node)

    # -------------------------------------------------------
    # a = df["rsi"]
    # current = features["atr"]
    # -------------------------------------------------------

    def visit_Assign(self, node):

        try:

            if isinstance(node.value, ast.Subscript):

                if isinstance(node.value.slice, ast.Constant):

                    value = node.value.slice.value

                    if isinstance(value, str):

                        for target in node.targets:

                            if isinstance(target, ast.Name):

                                self.contract.aliases[target.id] = value

        except Exception:

            pass

        self.generic_visit(node)

    # -------------------------------------------------------
    # global variable usage
    # -------------------------------------------------------

    def visit_Name(self, node):

        if isinstance(node.ctx, ast.Load):

            self.contract.globals.add(node.id)

        self.generic_visit(node)

# -------------------------------------------------------
    # detect output schema
    # -------------------------------------------------------

    def visit_Return(self, node):

        try:

            if isinstance(node.value, ast.Dict):

                for key in node.value.keys:

                    if isinstance(key, ast.Constant):

                        if isinstance(key.value, str):

                            self.contract.outputs.add(key.value)

        except Exception:

            pass

        self.generic_visit(node)

    # -------------------------------------------------------
    # detect dependency
    # -------------------------------------------------------

    def visit_Attribute(self, node):

        """
        self.trend
        self.volume
        self.market
        """

        try:

            if isinstance(node.value, ast.Name):

                if node.value.id == "self":

                    self.contract.dependencies.add(node.attr)

        except Exception:

            pass

        self.generic_visit(node)

# ============================================================
# DETECTOR MANAGER
# ============================================================

class DetectorManager:

    def __init__(self):

        self.contracts: Dict[str, AnalyzerContract] = {}

    # -------------------------------------------------------

    def _file_hash(self, path: Path):

        return hashlib.sha256(

            path.read_bytes()

        ).hexdigest()

    # -------------------------------------------------------

    def scan(self, file):

        path = Path(file)

        source = path.read_text(

            encoding="utf-8"

        )

        tree = ast.parse(source)

        contract = AnalyzerContract(

            analyzer=path.stem,

            file=str(path),

            class_name="",

            analyze_function="analyze",

            hash=self._file_hash(path),

        )

        visitor = AnalyzerVisitor()

        visitor.contract = contract

        visitor.visit(tree)

        self.contracts[path.stem] = contract

        return contract

# -------------------------------------------------------

    def scan_directory(self, directory):

        directory = Path(directory)

        result = {}

        for file in sorted(directory.glob("*.py")):

            if file.name.startswith("_"):

                continue

            result[file.stem] = self.scan(file)

        return result

    # -------------------------------------------------------

    def all_required_features(self):

        output = set()

        for contract in self.contracts.values():

            output.update(

                contract.required_features.keys()

            )

        return sorted(output)

    # -------------------------------------------------------

    def analyzer_dependencies(self):

        graph = {}

        for analyzer, contract in self.contracts.items():

            graph[analyzer] = sorted(

                contract.dependencies

            )

        return graph

    # -------------------------------------------------------

    def export(self):

        data = {}

        for name, contract in self.contracts.items():

            data[name] = {

                "features": sorted(

                    contract.required_features.keys()

                ),

                "outputs": sorted(

                    contract.outputs

                ),

                "dependencies": sorted(

                    contract.dependencies

                ),

                "imports": sorted(

                    contract.imports

                ),

                "aliases": contract.aliases,

                "functions": [

                    {

                        "name": x.function,

                        "kwargs": x.kwargs,

                        "line": x.line,

                    }

                    for x in contract.function_calls

                ],

            }

        return data

# -------------------------------------------------------
    # resolve
    # features = ["rsi","adx"]
    # need = {"a":"atr","b":"ema_20"}
    # -------------------------------------------------------

    def visit_List(self, node):

        try:

            for item in node.elts:

                if isinstance(item, ast.Constant):

                    if isinstance(item.value, str):

                        self.contract.required_features.setdefault(

                            item.value,

                            FeatureRequest(

                                feature=item.value,

                                source="list",

                                line=node.lineno,

                                column=node.col_offset,

                                access_type="constant",

                            ),

                        )

        except Exception:

            pass

        self.generic_visit(node)

    # -------------------------------------------------------

    def visit_Tuple(self, node):

        try:

            for item in node.elts:

                if isinstance(item, ast.Constant):

                    if isinstance(item.value, str):

                        self.contract.required_features.setdefault(

                            item.value,

                            FeatureRequest(

                                feature=item.value,

                                source="tuple",

                                line=node.lineno,

                                column=node.col_offset,

                                access_type="constant",

                            ),

                        )

        except Exception:

            pass

        self.generic_visit(node)

    # -------------------------------------------------------

    def visit_Dict(self, node):

        try:

            for key in node.values:

                if isinstance(key, ast.Constant):

                    if isinstance(key.value, str):

                        self.contract.required_features.setdefault(

                            key.value,

                            FeatureRequest(

                                feature=key.value,

                                source="dict",

                                line=node.lineno,

                                column=node.col_offset,

                                access_type="constant",

                            ),

                        )

        except Exception:

            pass

        self.generic_visit(node)

# -------------------------------------------------------
    # for feature in REQUIRED_FEATURES:
    #     df[feature]
    # -------------------------------------------------------

    def visit_For(self, node):

        try:

            if isinstance(node.iter, ast.Name):

                self.contract.dependencies.add(

                    f"LOOP::{node.iter.id}"

                )

        except Exception:

            pass

        self.generic_visit(node)

    # -------------------------------------------------------
    # if "rsi" in df:
    # -------------------------------------------------------

    def visit_Compare(self, node):

        try:

            if len(node.ops):

                if isinstance(node.ops[0], ast.In):

                    if isinstance(node.left, ast.Constant):

                        if isinstance(node.left.value, str):

                            self.contract.required_features.setdefault(

                                node.left.value,

                                FeatureRequest(

                                    feature=node.left.value,

                                    source="compare",

                                    line=node.lineno,

                                    column=node.col_offset,

                                    access_type="contains",

                                ),

                            )

        except Exception:

            pass

        self.generic_visit(node)

# -------------------------------------------------------
    # df.loc[:, "rsi"]
    # df.loc[last, "atr"]
    # -------------------------------------------------------

    def visit_Attribute(self, node):

        try:

            if node.attr in {

                "loc",

                "iloc",

                "iat",

                "at",

            }:

                self.contract.dependencies.add(

                    f"ACCESS::{node.attr}"

                )

        except Exception:

            pass

        self.generic_visit(node)

# ============================================================
# VALIDATION
# ============================================================

    def validate(self):

        errors = []

        for analyzer, contract in self.contracts.items():

            if not contract.required_features:

                errors.append(

                    f"{analyzer}: no required feature detected"

                )

            if not contract.function_calls:

                errors.append(

                    f"{analyzer}: no function call detected"

                )

            duplicated = set()

            seen = set()

            for feature in contract.required_features:

                if feature in seen:

                    duplicated.add(feature)

                seen.add(feature)

            if duplicated:

                errors.append(

                    f"{analyzer}: duplicate feature {duplicated}"

                )

        return errors

# ============================================================
# REPORT
# ============================================================

    def report(self):

        print()

        print("=" * 80)

        print("ANALYZER DETECTION REPORT")

        print("=" * 80)

        for analyzer, contract in self.contracts.items():

            print()

            print(analyzer)

            print("-" * 80)

            print("Features")

            for x in sorted(contract.required_features):

                print(" ", x)

            print()

            print("Dependencies")

            for x in sorted(contract.dependencies):

                print(" ", x)

            print()

            print("Outputs")

            for x in sorted(contract.outputs):

                print(" ", x)

            print()

            print("Functions")

            for fn in contract.function_calls:

                print(

                    f"  {fn.function} {fn.kwargs}"

                )


