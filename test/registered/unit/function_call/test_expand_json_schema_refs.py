"""Unit tests for tool-parameter schema $ref inlining."""

import unittest

from jsonschema import Draft202012Validator

from sglang.srt.function_call.utils import expand_json_schema_refs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestExpandJsonSchemaRefs(CustomTestCase):
    def _assert_accepts(self, schema: dict) -> None:
        Draft202012Validator.check_schema(schema)

    def test_defs_ref_inlined_and_container_dropped(self):
        """A $ref into $defs is replaced by the target; $defs is removed once
        nothing points into it."""
        schema = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            },
            "properties": {
                "billing_address": {"$ref": "#/$defs/Address"},
                "shipping_address": {"$ref": "#/$defs/Address"},
            },
        }
        result = expand_json_schema_refs(schema)

        address = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        self.assertEqual(result["properties"]["billing_address"], address)
        self.assertEqual(result["properties"]["shipping_address"], address)
        self.assertNotIn("$defs", result)
        self._assert_accepts(result)

    def test_definitions_keyword_supported(self):
        """draft-07 style `definitions` (not `$defs`) resolves the same way."""
        schema = {
            "type": "object",
            "definitions": {"Id": {"type": "integer"}},
            "properties": {"id": {"$ref": "#/definitions/Id"}},
        }
        result = expand_json_schema_refs(schema)
        self.assertEqual(result["properties"]["id"], {"type": "integer"})
        self.assertNotIn("definitions", result)

    def test_input_not_mutated(self):
        """Expansion returns a new tree and leaves the caller's dict intact."""
        schema = {
            "$defs": {"S": {"type": "string"}},
            "properties": {"a": {"$ref": "#/$defs/S"}},
        }
        expand_json_schema_refs(schema)
        self.assertEqual(schema["properties"]["a"], {"$ref": "#/$defs/S"})
        self.assertIn("$defs", schema)

    def test_nested_ref_chain_fully_expanded(self):
        """A def that references another def is transitively inlined."""
        schema = {
            "$defs": {
                "Inner": {"type": "string"},
                "Outer": {
                    "type": "object",
                    "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                },
            },
            "properties": {"outer": {"$ref": "#/$defs/Outer"}},
        }
        result = expand_json_schema_refs(schema)
        self.assertEqual(
            result["properties"]["outer"],
            {"type": "object", "properties": {"inner": {"type": "string"}}},
        )
        self.assertNotIn("$defs", result)

    def test_sibling_keyword_overrides_target(self):
        """A keyword beside $ref (2020-12) overrides the referenced value."""
        schema = {
            "$defs": {"Base": {"type": "string", "description": "base"}},
            "properties": {
                "x": {"$ref": "#/$defs/Base", "description": "overridden"}
            },
        }
        result = expand_json_schema_refs(schema)
        self.assertEqual(
            result["properties"]["x"],
            {"type": "string", "description": "overridden"},
        )

    def test_recursive_schema_stays_finite_and_keeps_defs(self):
        """A self-recursive def can't be fully inlined; the back-edge $ref is
        left in place and $defs is retained so the schema stays valid."""
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Node"},
                        }
                    },
                }
            },
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }
        result = expand_json_schema_refs(schema)
        # First level inlined, the recursive edge preserved as a $ref.
        leaf = result["properties"]["root"]["properties"]["children"]["items"]
        self.assertEqual(leaf, {"$ref": "#/$defs/Node"})
        self.assertIn("$defs", result)
        self._assert_accepts(result)

    def test_external_ref_left_untouched(self):
        """A non-local $ref (URI) is not something we can inline; leave it."""
        schema = {
            "type": "object",
            "properties": {
                "loc": {"$ref": "https://example.com/geo.schema.json"}
            },
        }
        result = expand_json_schema_refs(schema)
        self.assertEqual(
            result["properties"]["loc"],
            {"$ref": "https://example.com/geo.schema.json"},
        )

    def test_unresolvable_local_ref_left_untouched(self):
        """A $ref pointing at a missing def is preserved rather than crashing."""
        schema = {
            "$defs": {"A": {"type": "string"}},
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
        }
        result = expand_json_schema_refs(schema)
        self.assertEqual(result["properties"]["x"], {"$ref": "#/$defs/Missing"})
        # An unresolved local ref remains, so $defs is retained.
        self.assertIn("$defs", result)

    def test_non_dict_input_returned_as_is(self):
        self.assertEqual(expand_json_schema_refs(None), None)
        self.assertEqual(expand_json_schema_refs("string"), "string")

    def test_fragment_bearing_id_dropped(self):
        """A draft-07 anchor-style `$id` (e.g. "#user") is rejected by the
        2020-12 metaschema; expansion drops `$id` so the schema validates.
        Mirrors Kimi-Vendor-Verifier TestID cases 3/5/6."""
        for id_value in (
            "#user",
            "https://example.com/schemas/user#/definitions/name",
            "https://example.com/schemas/user#/$defs/name",
        ):
            schema = {
                "type": "object",
                "required": ["value"],
                "additionalProperties": False,
                "properties": {"value": {"type": "object"}},
                "$id": id_value,
            }
            result = expand_json_schema_refs(schema)
            self.assertNotIn("$id", result)
            self._assert_accepts(result)

    def test_anchor_dropped(self):
        """`$anchor` is identity metadata with no meaning in a flattened tree."""
        schema = {
            "type": "object",
            "$anchor": "user",
            "properties": {"name": {"type": "string"}},
        }
        result = expand_json_schema_refs(schema)
        self.assertNotIn("$anchor", result)
        self._assert_accepts(result)

    def test_nested_id_dropped_but_refs_still_inlined(self):
        """A `$id` on a def entry is dropped while its body is still inlined."""
        schema = {
            "$defs": {
                "Addr": {
                    "$id": "https://x/addr#frag",
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
            "properties": {"a": {"$ref": "#/$defs/Addr"}},
        }
        result = expand_json_schema_refs(schema)
        self.assertEqual(
            result["properties"]["a"],
            {"type": "object", "properties": {"city": {"type": "string"}}},
        )
        self.assertNotIn("$defs", result)
        self._assert_accepts(result)


if __name__ == "__main__":
    unittest.main()
