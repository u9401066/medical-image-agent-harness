"""Prevent syntactically valid but silently uncollected regression tests."""

import ast
from pathlib import Path


def test_no_nested_or_shadowed_test_definitions():
    failures = []
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for path in Path(__file__).resolve().parents[1].rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for parent in ast.walk(tree):
            if isinstance(parent, functions):
                for child in ast.walk(parent):
                    if (child is not parent and isinstance(child, functions)
                            and child.name.startswith("test_")):
                        failures.append(f"{path.name}:{child.lineno}: nested test")
            if isinstance(parent, (ast.Module, ast.ClassDef)):
                names = set()
                for child in parent.body:
                    if (isinstance(child, (*functions, ast.ClassDef))
                            and child.name.startswith(("test_", "Test"))):
                        if child.name in names:
                            failures.append(f"{path.name}:{child.lineno}: shadowed test")
                        names.add(child.name)
    assert failures == []
