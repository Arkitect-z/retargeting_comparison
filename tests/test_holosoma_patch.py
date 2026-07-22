from __future__ import annotations

import ast
from pathlib import Path

import pytest


SOURCE = Path(
    "external/holosoma/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py"
)
pytestmark = pytest.mark.skipif(not SOURCE.is_file(), reason="frozen Holosoma checkout is not present")


def _attribute_names(node: ast.AST) -> set[str]:
    return {
        item.attr
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == "self"
    }


def test_full_no_hard_constraint_graph_has_both_runtime_gates() -> None:
    source = SOURCE.read_text()
    tree = ast.parse(source)
    solve = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "solve_single_iteration"
    )
    nonpenetration_gate = next(
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.If) and "activate_obj_non_penetration" in _attribute_names(node.test)
    )
    assert "_update_jacobians_and_phis_from_q" in {
        item.func.attr
        for item in ast.walk(nonpenetration_gate)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
    }
    foot_gate = next(
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "apply_foot_sticking" for target in node.targets)
    )
    assert "activate_foot_sticking" in _attribute_names(foot_gate.value)
