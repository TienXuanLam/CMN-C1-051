# PB-4: Import Isolation — CMN-C1-051

import ast
import os

_L0_PROHIBITED = ["agenticstar", "agenticstar_agentcore"]
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _scan(filepath):
    with open(filepath) as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return []
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for p in _L0_PROHIBITED:
                    if alias.name == p or alias.name.startswith(f"{p}."):
                        violations.append(f"{filepath}:{node.lineno} — import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for p in _L0_PROHIBITED:
                if node.module == p or node.module.startswith(f"{p}."):
                    violations.append(f"{filepath}:{node.lineno} — from {node.module} import ...")
    return violations


class TestImportIsolation:
    def test_no_l0_imports_in_src(self):
        src_dir = os.path.join(_root, "src")
        violations = []
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith(".py"):
                    violations.extend(_scan(os.path.join(root, f)))
        assert violations == [], "L0 violations:\n" + "\n".join(violations)

    def test_src_uses_framework_imports(self):
        src_dir = os.path.join(_root, "src")
        found = False
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if not f.endswith(".py"):
                    continue
                with open(os.path.join(root, f)) as fh:
                    try:
                        tree = ast.parse(fh.read())
                    except SyntaxError:
                        continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("framework."):
                        found = True
                        break
                if found:
                    break
        assert found, "No framework.* imports found in src/"
