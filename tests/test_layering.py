"""The package layering is a rule, not a diagram in a README.

Each subpackage may import from the layers below it and never from those above.
This is checked by reading the imports rather than trusting them, so a change
that quietly turns the structure back into a tangle fails here first.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "sentinels" / "backtest"

# Lower number = deeper foundation. A layer may import from equal or lower
# numbers only. The orchestration modules at the top may import anything.
LAYERS: Dict[str, int] = {
    "core": 0,  # config, risk limits, action schema, records, protocols
    "data": 1,  # the feed and its indicators
    "analysis": 1,  # metrics and equity reshaping
    "storage": 1,  # run layout, decision cache, registry
    "execution": 2,  # account, fills, broker
    "decisions": 3,  # context, validator
    "agents": 4,  # the strategies
    "(top)": 5,  # runner.py, manager.py, __init__.py
}


def subpackage_of(path: Path) -> str:
    rel = path.relative_to(PACKAGE)
    return rel.parts[0] if len(rel.parts) > 1 else "(top)"


def runtime_imports(path: Path) -> Set[str]:
    """Backtest subpackages this module imports at run time.

    Imports inside `if TYPE_CHECKING:` are skipped: they cost nothing at run
    time and are how a protocol refers to the types that implement it without
    depending on them.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    type_checking_nodes: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_tc:
                for child in ast.walk(node):
                    type_checking_nodes.add(id(child))

    found: Set[str] = set()
    for node in ast.walk(tree):
        if id(node) in type_checking_nodes:
            continue
        modules: List[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        for module in modules:
            if not module.startswith("sentinels.backtest"):
                continue
            parts = module.split(".")
            target = parts[2] if len(parts) > 2 else "(top)"
            if target in LAYERS:
                found.add(target)
    return found


def all_modules() -> List[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py"))


def test_every_module_lives_in_a_declared_layer():
    unknown = {subpackage_of(p) for p in all_modules()} - set(LAYERS)
    assert not unknown, f"subpackages missing from the layer map: {sorted(unknown)}"


def test_no_module_imports_from_a_layer_above_it():
    violations: List[Tuple[str, str, str]] = []
    for path in all_modules():
        source = subpackage_of(path)
        for target in runtime_imports(path):
            if target != source and LAYERS[target] > LAYERS[source]:
                violations.append((str(path.relative_to(PACKAGE)), source, target))
    assert not violations, "upward imports break the layering:\n" + "\n".join(
        f"  {mod}: {src} (layer {LAYERS[src]}) imports {dst} (layer {LAYERS[dst]})"
        for mod, src, dst in violations
    )


def test_the_dependency_graph_has_no_cycles():
    edges: Dict[str, Set[str]] = defaultdict(set)
    for path in all_modules():
        source = subpackage_of(path)
        for target in runtime_imports(path):
            if target != source:
                edges[source].add(target)

    visiting: Set[str] = set()
    done: Set[str] = set()
    trail: List[str] = []

    def walk(node: str) -> None:
        if node in done:
            return
        if node in visiting:
            cycle = " -> ".join(trail[trail.index(node) :] + [node])
            pytest.fail(f"import cycle between subpackages: {cycle}")
        visiting.add(node)
        trail.append(node)
        for nxt in sorted(edges[node]):
            walk(nxt)
        trail.pop()
        visiting.discard(node)
        done.add(node)

    for node in sorted(LAYERS):
        walk(node)


def test_core_depends_on_nothing_at_run_time():
    """The foundation has to actually be one."""
    for path in all_modules():
        if subpackage_of(path) != "core":
            continue
        upward = runtime_imports(path) - {"core"}
        assert not upward, f"{path.name} pulls in {sorted(upward)} at run time"


def test_every_subpackage_documents_itself():
    for directory in sorted(p for p in PACKAGE.iterdir() if p.is_dir()):
        if directory.name == "__pycache__":
            continue
        init = directory / "__init__.py"
        assert init.exists(), f"{directory.name}/ is not a package"
        doc = ast.get_docstring(ast.parse(init.read_text(encoding="utf-8")))
        assert doc and len(doc) > 40, f"{directory.name}/__init__.py needs a real docstring"
