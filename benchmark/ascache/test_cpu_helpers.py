"""Differential checks of actual optimized helpers, without engine/GPU imports."""

import ast
import unittest
from array import array
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]


def execute(nodes, namespace, filename):
    """Compile selected AST nodes into an isolated helper-test namespace.

    Args:
        nodes: AST definitions extracted from actual implementation source.
        namespace: Namespace receiving the executed helper definitions.
        filename: Source path used for compilation diagnostics.

    Returns:
        None; namespace receives the executed definitions.

    Raises:
        SyntaxError: The extracted helper module cannot be compiled.
    """
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(filename), "exec"), namespace)


def radix_namespace():
    """Load the real radix-key helpers without importing the serving engine.

    Returns:
        dict: Isolated namespace containing RadixKey and its native comparator.
    """
    path = ROOT / "python/sglang/srt/mem_cache/radix_cache.py"
    tree = ast.parse(path.read_text())
    native = next(
        node
        for node in tree.body
        if isinstance(node, ast.Try) and "_radix_memcmp" in ast.unparse(node)
    )
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RadixKey"
    )
    ns = {"array": array}
    execute([native, cls], ns, path)
    return ns


class Prefix:
    """Represent a cached prefix using only its configured token length."""
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def request_namespace():
    """Load actual request length and suffix helpers into a CPU test namespace.

    Returns:
        dict: Request fixtures and helpers extracted from production source.
    """
    path = ROOT / "python/sglang/srt/managers/schedule_batch.py"
    tree = ast.parse(path.read_text())
    req = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Req"
    )
    selected = [
        node
        for node in req.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("get_fill_ids", "_get_fill_ids_len")
        or isinstance(node, ast.Assign)
        and "_GET_FILL_IDS_FOR_LENGTH" in ast.unparse(node)
    ]
    cls = ast.ClassDef(
        name="Req", bases=[], keywords=[], body=selected, decorator_list=[]
    )
    alias = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node).startswith("_REQ_GET_FILL_IDS =")
    )
    suffix = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_extend_input_ids"
    )
    common = ROOT / "python/sglang/srt/utils/common.py"
    range_node = next(
        node
        for node in ast.parse(common.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Range"
    )
    ns = {
        "array": array,
        "MethodType": MethodType,
        "NamedTuple": NamedTuple,
        "torch": SimpleNamespace(Tensor=Prefix),
    }
    execute([range_node, cls, alias, suffix], ns, path)
    return ns


class PrefixChecks(unittest.TestCase):
    """Compare native and legacy prefix matching against a token oracle."""
    def test_native_and_fallback_match_independent_long_prefix_oracle(self):
        """Native and fallback match independent long prefix oracle.

        Returns:
            None.

        Raises:
            AssertionError: The implementation violates the tested contract.
        """
        ns = radix_namespace()
        native = ns["_radix_memcmp"]
        for accelerator in (native, None):
            ns["_radix_memcmp"] = accelerator
            for length in (0, 1, 2047, 16383, 61440, 1000000):
                left = array("q", (i % 101 for i in range(length)))
                for mismatch in sorted(
                    {0, min(2048, length), min(58880, length), length}
                ):
                    right = array("q", [-9, -8, -7]) + left
                    if mismatch < length:
                        right[3 + mismatch] = -999
                    for bigram in (False, True):
                        a = ns["RadixKey"](left, is_bigram=bigram)
                        b = ns["RadixKey"](right, is_bigram=bigram)
                        if 3 > len(b):
                            with self.assertRaises(IndexError):
                                a.match_at(b, 3)
                            continue
                        expected = max(0, mismatch - int(bigram))
                        for page in (1, 256):
                            with self.subTest(
                                length=length,
                                mismatch=mismatch,
                                bigram=bigram,
                                page=page,
                                native=accelerator is not None,
                            ):
                                self.assertEqual(
                                    a.match_at(b, 3, page), expected // page * page
                                )

    def test_namespace_and_offset_validation(self):
        """Namespace and offset validation.

        Returns:
            None.

        Raises:
            AssertionError: The implementation violates the tested contract.
        """
        ns = radix_namespace()
        a = ns["RadixKey"](array("q", [1, 2]), extra_key="a")
        with self.assertRaises(ValueError):
            a.match(ns["RadixKey"](array("q", [1, 2]), extra_key="b"))
        with self.assertRaises(IndexError):
            a.match_at(a, -1)


class RequestChecks(unittest.TestCase):
    """Check request length and owned-suffix behavior against public getters."""
    def test_owned_suffix_and_length_match_public_snapshot(self):
        """Owned suffix and length match public snapshot.

        Returns:
            None.

        Raises:
            AssertionError: The implementation violates the tested contract.
        """
        ns = request_namespace()
        for length in (0, 1, 256, 61440):
            for end in (-length - 1, -1, 0, 1, length, length + 10, 10**30):
                for prefix in (0, 1, length // 2, length, length + 3):
                    req = ns["Req"]()
                    req.full_untruncated_fill_ids = array("q", range(length))
                    req.extend_range = ns["Range"](0, end)
                    req.prefix_indices = Prefix(prefix)
                    snapshot = req.get_fill_ids()
                    actual = ns["_get_extend_input_ids"](req)
                    with self.subTest(length=length, end=end, prefix=prefix):
                        self.assertEqual(actual, snapshot[prefix:])
                        self.assertEqual(req._get_fill_ids_len(), len(snapshot))
                        self.assertIsNot(actual, req.full_untruncated_fill_ids)
                        actual.append(-1)
                        self.assertEqual(len(req.full_untruncated_fill_ids), length)

    def test_overridden_getter_uses_legacy_semantics(self):
        """Overridden getter uses legacy semantics.

        Returns:
            None.

        Raises:
            AssertionError: The implementation violates the tested contract.
        """
        ns = request_namespace()
        req = ns["Req"]()
        req.full_untruncated_fill_ids = array("q", range(100))
        req.extend_range = ns["Range"](0, 90)
        req.prefix_indices = Prefix(1)
        req.get_fill_ids = lambda: array("q", [8, 9, 10])
        self.assertEqual(req._get_fill_ids_len(), 3)
        self.assertEqual(ns["_get_extend_input_ids"](req), array("q", [9, 10]))


if __name__ == "__main__":
    unittest.main()
