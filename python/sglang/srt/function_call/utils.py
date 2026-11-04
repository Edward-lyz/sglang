"""Utilities shared by function-call and streaming output parsers."""

import copy
import re
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import orjson
import partial_json_parser
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice

_STANDARD_JSON_SCHEMA_TYPES = {
    "null",
    "boolean",
    "object",
    "array",
    "number",
    "string",
    "integer",
}

# Non-standard ``type`` values commonly emitted by DB/ORM-driven tool-schema
# generators. Mapped to the closest JSON Schema 2020-12 primitive so that
# ``Draft202012Validator.check_schema`` does not reject an otherwise-usable
# tool definition.
_JSON_SCHEMA_TYPE_ALIASES: Dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "uuid": "string",
    "date": "string",
    "datetime": "string",
    "time": "string",
    "timestamp": "string",
    "binary": "string",
    "blob": "string",
    "bytea": "string",
    "bytes": "string",
    "varbinary": "string",
    "bool": "boolean",
    "bigint": "integer",
    "smallint": "integer",
    "tinyint": "integer",
    "double": "number",
    "decimal": "number",
    "real": "number",
    "numeric": "number",
    "arr": "array",
    "tuple": "array",
    "set": "array",
    "map": "object",
}

# Prefix-based matching so that parameterised names like ``int32`` /
# ``float64`` / ``list[str]`` / ``dict[str, int]`` resolve. A prefix only
# matches when it spans the entire token or is followed by a non-identifier
# char, so "int" does not swallow "internal" and "list" does not swallow
# "list_price".
_PREFIX_BOUNDARY_CHARS = frozenset("0123456789[<( \t")
_PREFIX_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("int", "uint", "long", "short", "unsigned"), "integer"),
    (("num", "float"), "number"),
    (("list",), "array"),
    (("dict",), "object"),
)


def _matches_type_prefix(base: str, prefixes: Tuple[str, ...]) -> bool:
    for p in prefixes:
        if base == p:
            return True
        if (
            len(base) > len(p)
            and base.startswith(p)
            and base[len(p)] in _PREFIX_BOUNDARY_CHARS
        ):
            return True
    return False


def _normalize_single_type(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    if raw in _STANDARD_JSON_SCHEMA_TYPES:
        return raw
    # ``split("(", 1)[0]`` strips parenthesized params like ``varchar(255)``
    # or ``decimal(10,2)`` without the overhead of a regex per call.
    base = raw.split("(", 1)[0].strip().lower()
    if base in _STANDARD_JSON_SCHEMA_TYPES:
        return base
    mapped = _JSON_SCHEMA_TYPE_ALIASES.get(base)
    if mapped is not None:
        return mapped
    for prefixes, target in _PREFIX_RULES:
        if _matches_type_prefix(base, prefixes):
            return target
    return raw


def _normalize_type_list(raw_items: List[Any]) -> List[Any]:
    normalized_items: List[Any] = []
    for item in raw_items:
        normalized_item = _normalize_single_type(item)
        if normalized_item not in normalized_items:
            normalized_items.append(normalized_item)
    return normalized_items


_SCHEMA_MAP_KEYWORDS = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
)
_SCHEMA_LIST_KEYWORDS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_KEYWORDS = (
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
)


def _walk_subschemas(schema: Any) -> Iterator[dict]:
    """
    Yield every schema object reachable from ``schema``, outermost first.

    Only positions JSON Schema defines as subschemas are followed, so
    data-carrying keywords (``const``, ``enum``, ``default``, ``examples``) are
    never mistaken for schemas and their contents are left alone. Boolean
    subschemas (``items: false``) and other scalars are skipped.

    Callers may rewrite keywords on a yielded node; adding or removing
    subschema containers mid-walk is not supported.
    """
    if isinstance(schema, list):
        for item in schema:
            yield from _walk_subschemas(item)
        return
    if not isinstance(schema, dict):
        return

    yield schema

    for key in _SCHEMA_MAP_KEYWORDS:
        nested = schema.get(key)
        if isinstance(nested, dict):
            for value in nested.values():
                yield from _walk_subschemas(value)

    for key in _SCHEMA_LIST_KEYWORDS:
        nested = schema.get(key)
        if isinstance(nested, list):
            for value in nested:
                yield from _walk_subschemas(value)

    for key in _SCHEMA_KEYWORDS:
        if key in schema:
            yield from _walk_subschemas(schema[key])


def normalize_json_schema_types(schema: Any) -> None:
    """
    Walk a JSON Schema in place and rewrite non-standard ``"type"`` values
    (e.g. ``"varchar"``, ``"enum"``, ``"int"``) to their standard JSON Schema
    equivalents.

    Acts as a compatibility layer for tool ``parameters`` schemas exported
    from database / ORM tooling, which often uses DB type names rather than
    JSON Schema types. Unknown types are left untouched so that downstream
    validation can still surface genuine errors.

    Mutates the input dict in place; the rewritten schema is also what gets
    rendered into the model prompt, so e.g. a user-supplied ``"varchar"``
    reaches the model as ``"string"``. ``$ref`` values are not resolved;
    callers pass tree-shaped schemas (HTTP JSON input is always a tree).
    """
    for node in _walk_subschemas(schema):
        t = node.get("type")
        if isinstance(t, str):
            node["type"] = _normalize_single_type(t)
        elif isinstance(t, list):
            node["type"] = _normalize_type_list(t)


# Draft 2020-12 constrains ``$id`` to ``^[^#]*#?$``: a non-empty fragment is a
# schema error, not a warning. ``$anchor`` is the replacement for the draft-07
# plain-name form, and it carries the same ``#name`` reference syntax.
_ANCHOR_RE = re.compile(r"^[A-Za-z_][-A-Za-z0-9._]*$")


def normalize_json_schema_ids(schema: Any) -> None:
    """
    Walk a JSON Schema in place and move the fragment out of every ``$id``.

    Two draft-07 idioms survive in the wild and make
    ``Draft202012Validator.check_schema`` reject an otherwise usable tool
    definition outright (the request never reaches grammar compilation):

    - ``{"$id": "#user"}`` — a plain-name anchor. Re-expressed as
      ``{"$anchor": "user"}``, which keeps a matching ``{"$ref": "#user"}``
      resolvable; dropping the identifier instead would leave a dangling ref.
    - ``{"$id": "https://x/user#/definitions/name"}`` — a pointer fragment. It
      has no 2020-12 equivalent, so only the base URI is kept.

    An ``$id`` that ends up empty is removed rather than left as ``""``. An
    existing ``$anchor`` is never overwritten. Mutates the input in place.
    """
    for node in _walk_subschemas(schema):
        identifier = node.get("$id")
        if not isinstance(identifier, str):
            continue
        base, _, fragment = identifier.partition("#")
        if not fragment:
            continue
        if base:
            node["$id"] = base
        else:
            del node["$id"]
        if "$anchor" not in node and _ANCHOR_RE.match(fragment):
            node["$anchor"] = fragment


def _resolve_json_pointer(root: Any, pointer: str) -> Any:
    """
    Resolve an RFC 6901 JSON pointer (the part after ``#``) against ``root``.

    Returns the referenced node, or ``None`` if the pointer does not resolve
    (missing key, out-of-range index, or a scalar encountered mid-path).
    """
    if pointer in ("", "/"):
        return root
    node = root
    for raw_token in pointer.lstrip("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                return None
            node = node[token]
        elif isinstance(node, list):
            try:
                idx = int(token)
            except ValueError:
                return None
            if not 0 <= idx < len(node):
                return None
            node = node[idx]
        else:
            return None
    return node


def _has_local_ref(node: Any) -> bool:
    """True if any ``$ref`` pointing inside the document (``#...``) remains."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#"):
            return True
        return any(_has_local_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_local_ref(item) for item in node)
    return False


# Identifier keywords that only matter for cross-document / anchor-based $ref
# resolution. Once a schema is flattened into a self-contained tree they carry
# no meaning, and a non-empty-fragment ``$id`` (draft-07 anchor style, e.g.
# ``"#user"``) is outright rejected by the JSON Schema 2020-12 metaschema, so
# they are dropped during expansion.
_DROPPED_IDENTIFIER_KEYWORDS = frozenset({"$id", "$anchor"})


def _expand_refs_node(node: Any, root: Any, active: frozenset) -> Any:
    """
    Return a copy of ``node`` with local ``$ref`` pointers inlined and
    identifier keywords (``$id`` / ``$anchor``) dropped.

    ``active`` is the set of ``$ref`` strings currently being expanded on this
    branch; a ref already in it is a cycle and is left in place so expansion
    stays finite.
    """
    if isinstance(node, list):
        return [_expand_refs_node(item, root, active) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    is_local_ref = isinstance(ref, str) and ref.startswith("#")

    if is_local_ref and ref not in active:
        target = _resolve_json_pointer(root, ref[1:])
        if isinstance(target, dict):
            expanded = _expand_refs_node(copy.deepcopy(target), root, active | {ref})
            result = dict(expanded)
            # JSON Schema 2020-12 permits keywords beside ``$ref``; a sibling
            # (e.g. an overriding ``description``) wins over the target's value.
            for key, value in node.items():
                if key == "$ref" or key in _DROPPED_IDENTIFIER_KEYWORDS:
                    continue
                result[key] = _expand_refs_node(value, root, active)
            return result
        # Unresolved / non-object target: fall through and keep the ``$ref``.

    return {
        key: (value if key == "$ref" else _expand_refs_node(value, root, active))
        for key, value in node.items()
        if key not in _DROPPED_IDENTIFIER_KEYWORDS
    }


def expand_json_schema_refs(schema: Any) -> Any:
    """
    Inline local ``$ref`` pointers so a tool ``parameters`` schema becomes
    self-contained.

    Each ``{"$ref": "#/$defs/Name"}`` (or ``#/definitions/Name``, or any deeper
    RFC 6901 pointer into the document) is replaced by a copy of the referenced
    subschema, recursively. Refs on a cyclic branch (recursive schemas) and
    external refs (``$ref`` values not starting with ``#``) are left untouched
    so the result stays finite and valid. The ``$defs`` / ``definitions``
    containers are dropped once nothing points into them anymore, and identifier
    keywords (``$id`` / ``$anchor``) are dropped throughout — they are
    meaningless in a flattened tree and a fragment-bearing ``$id`` would
    otherwise be rejected by the 2020-12 metaschema.

    Returns a new tree; the input is not mutated. Intended for tree-shaped HTTP
    JSON input, gated behind ``SGLANG_ENABLE_TOOL_SCHEMA_REF_EXPANSION``.
    """
    if not isinstance(schema, dict):
        return schema

    expanded = _expand_refs_node(schema, schema, frozenset())

    remainder = {
        key: value
        for key, value in expanded.items()
        if key not in ("$defs", "definitions")
    }
    if not _has_local_ref(remainder):
        return remainder
    return expanded


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise
    except AssertionError as e:
        # partial_json_parser.fix_fast() asserts on some partial/ambiguous inputs
        # (e.g. trailing non-whitespace after an otherwise-fixable prefix) instead
        # of signaling "incomplete". Convert to JSONDecodeError so streaming
        # callers treat it as not-yet-complete (wait for more tokens) rather than
        # raising and failing the request.
        raise JSONDecodeError(
            "partial_json_parser assertion (treat as incomplete)", input_str, 0
        ) from e


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


def _get_tool_schema_defs(tools: List[Tool]) -> dict:
    """
    Get consolidated $defs from all tools, validating for conflicts.

    Args:
        tools: List of tools to process

    Returns:
        Dictionary of consolidated $defs from all tools

    Raises:
        ValueError: If conflicting $defs are found
    """
    all_defs = {}
    for tool in tools:
        if tool.function.parameters is None:
            continue
        defs = tool.function.parameters.get("$defs", {})
        for def_name, def_schema in defs.items():
            if def_name in all_defs and all_defs[def_name] != def_schema:
                raise ValueError(
                    f"Tool definition '{def_name}' has "
                    "multiple schemas, which is not "
                    "supported."
                )
            else:
                all_defs[def_name] = def_schema
    return all_defs


def _get_tool_schema(tool: Tool) -> dict:
    return {
        "properties": {
            "name": {"type": "string", "enum": [tool.function.name]},
            "parameters": (
                tool.function.parameters
                if tool.function.parameters
                else {"type": "object", "properties": {}}
            ),
        },
        "required": ["name", "parameters"],
    }


def infer_type_from_json_schema(schema: Dict[str, Any]) -> Optional[str]:
    """
    Infer the primary type of a parameter from JSON Schema.

    Supports complex JSON Schema structures including:
    - Direct type field (including type arrays)
    - anyOf/oneOf: parameter can be any of multiple types
    - enum: parameter must be one of enum values
    - allOf: parameter must satisfy all type definitions
    - properties: inferred as object type
    - items: inferred as array type

    Args:
        schema: JSON Schema definition

    Returns:
        Inferred type ('string', 'number', 'object', 'array', etc.) or None
    """
    if not isinstance(schema, dict):
        return None

    # Priority 1: Direct type field (including type arrays)
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        elif isinstance(type_value, list) and type_value:
            # Handle type arrays: return first non-null type
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"  # If only null, default to string

    # Priority 2: Handle anyOf/oneOf
    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []

        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(sub_schema)
                if inferred_type:
                    types.append(inferred_type)

            if types:
                # If all types are the same, return unified type
                if len(set(types)) == 1:
                    return types[0]
                # If it's an optional type, return original type.
                if len(set(types)) == 2 and "null" in types:
                    return [t for t in types if t != "null"][0]
                # When types differ, prioritize string (safest)
                if "string" in types:
                    return "string"
                # Otherwise return first type
                return types[0]

    # Priority 3: Handle enum (infer type from enum values)
    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"

        # Infer type from enum values
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")

        # If type is uniform, return that type
        if len(enum_types) == 1:
            return enum_types.pop()
        # Mixed types, prioritize string
        return "string"

    # Priority 4: Handle allOf (must satisfy all types)
    if "allOf" in schema and isinstance(schema["allOf"], list):
        schemas = schema["allOf"]
        for sub_schema in schemas:
            inferred_type = infer_type_from_json_schema(sub_schema)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    # Priority 5: Infer object type
    if "properties" in schema:
        return "object"

    # Priority 6: Infer array type
    if "items" in schema:
        return "array"

    return None


def get_json_schema_constraint(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["required"]],
    parallel_tool_calls: bool = True,
) -> Optional[dict]:
    """
    Get the JSON schema constraint for the specified tool choice.

    Args:
        tool_choice: The tool choice specification
        parallel_tool_calls: If False, constrain to exactly one tool call (maxItems=1)

    Returns:
        JSON schema dict, or None if no valid tools found
    """

    if isinstance(tool_choice, ToolChoice):
        # For specific function choice, return the user's parameters schema directly
        fn_name = tool_choice.function.name
        for tool in tools:
            if tool.function.name == fn_name:
                schema = {
                    "type": "array",
                    "minItems": 1,
                    "items": _get_tool_schema(tool),
                }
                if not parallel_tool_calls:
                    schema["maxItems"] = 1
                return schema
        return None
    elif tool_choice == "required":
        json_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [_get_tool_schema(tool) for tool in tools],
            },
        }
        if not parallel_tool_calls:
            json_schema["maxItems"] = 1
        json_schema_defs = _get_tool_schema_defs(tools)
        if json_schema_defs:
            json_schema["$defs"] = json_schema_defs
        return json_schema

    return None


# Canonical streaming input boundaries shared by live parsing and continuation replay.
_MARKER_ATTRS = (
    "bot_token",
    "eot_token",
    "bot",
    "suffix",
    "tool_call_separator",
    "invoke_end_token",
    "tool_call_begin",
    "tool_call_end",
    "tool_call_start_token",
    "tool_call_end_token",
    "tool_call_argument_begin_token",
    "tool_call_prefix",
    "tool_call_function_end_token",
    "function_end_token",
    "parameter_prefix",
    "parameter_end_token",
    "tool_sep_token",
    "arg_key_start_token",
    "arg_key_end_token",
    "arg_value_start_token",
    "arg_value_end_token",
    "arg_val_sep_token",
    "arg_key_end",
    "arg_value_end",
    "think_start_token",
    "think_end_token",
    "tool_start_token",
    "_tool_start_token",
    "_tool_end_token",
    "TEXT_START_TOKEN",
    "TEXT_END_TOKEN",
    "ACTION_START_TOKEN",
)


def collect_streaming_markers(detector, tokenizer=None) -> list[str]:
    """Collect known delimiters without importing any model-specific parser.

    Single-character delimiters do not need protection from partial input and
    would unnecessarily turn ordinary JSON punctuation into separate chunks.
    Tokenizer metadata is accepted only as a concrete list/tuple of strings;
    this also avoids treating attributes synthesized by mocks as token lists.
    """
    markers = {
        value
        for attr in _MARKER_ATTRS
        if isinstance(value := getattr(detector, attr, None), str) and len(value) > 1
    }
    think_start = getattr(detector, "think_start_token", None)
    self_label = getattr(detector, "think_start_self_label", None)
    if isinstance(think_start, str) and isinstance(self_label, str):
        if think_start and self_label:
            markers.add(think_start + self_label)

    for tokens in (
        getattr(detector, "think_excluded_tokens", None),
        getattr(tokenizer, "all_special_tokens", None),
    ):
        if isinstance(tokens, (list, tuple)):
            markers.update(
                token for token in tokens if isinstance(token, str) and len(token) > 1
            )
    return sorted(markers, key=lambda marker: (-len(marker), marker))


class StreamingInputBuffer:
    """Emit canonical ordinary-text blocks and atomic structural markers.

    The emitted sequence depends only on the text, never on the boundaries of
    ``feed`` calls. Ordinary text is emitted in fixed-size blocks, with a short
    block emitted before each marker. An incomplete block or ambiguous marker
    prefix remains buffered across request suspension and history replay.

    ``finish`` flushes a true terminal EOF. It must not be called merely because
    a resumable request reached its output-token limit.
    """

    def __init__(self, markers: Iterable[str] = (), block_size: int = 32):
        """Initialize the buffer with structural markers and a plain-text block size."""
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self._block_size = block_size
        self._plain = ""
        self._pending_marker = ""
        self._markers_by_start: dict[str, list[str]] = {}
        for marker in sorted(set(markers), key=lambda value: (-len(value), value)):
            if marker:
                self._markers_by_start.setdefault(marker[0], []).append(marker)
        self._marker_start = (
            re.compile("[" + re.escape("".join(self._markers_by_start)) + "]")
            if self._markers_by_start
            else None
        )

    def feed(self, text: str) -> list[str]:
        """Consume live text and return complete canonical streaming chunks."""
        if not text:
            return []
        return self._consume(text, final=False)

    def finish(self) -> list[str]:
        """Flush buffered text at the true end of the stream."""
        return self._consume("", final=True)

    def _append_plain(self, text: str, start: int, end: int, output: list[str]):
        """Append plain text to fixed-size chunks while preserving any remainder."""
        if self._plain and start < end:
            take = min(self._block_size - len(self._plain), end - start)
            self._plain += text[start : start + take]
            start += take
            if len(self._plain) == self._block_size:
                output.append(self._plain)
                self._plain = ""
        while end - start >= self._block_size:
            output.append(text[start : start + self._block_size])
            start += self._block_size
        if start < end:
            self._plain = text[start:end]

    def _flush_plain(self, output: list[str]):
        """Flush the buffered plain-text remainder into the output."""
        if self._plain:
            output.append(self._plain)
            self._plain = ""

    def _consume(self, text: str, final: bool) -> list[str]:
        """Consume text while keeping structural markers atomic across feed calls."""
        # Only an unresolved marker prefix is prepended, so copying never grows
        # with the full replay history. Ordinary-text slices are block-sized.
        if self._pending_marker:
            text = self._pending_marker + text
            self._pending_marker = ""
        output: list[str] = []
        plain_start = pos = 0
        while self._marker_start is not None:
            candidate = self._marker_start.search(text, pos)
            if candidate is None:
                break
            pos = candidate.start()
            matched = None
            incomplete = False
            remaining = len(text) - pos
            for marker in self._markers_by_start[text[pos]]:
                if remaining < len(marker):
                    if not final and marker.startswith(text[pos:]):
                        incomplete = True
                        break
                elif text.startswith(marker, pos):
                    matched = marker
                    break

            if incomplete:
                self._append_plain(text, plain_start, pos, output)
                self._pending_marker = text[pos:]
                return output
            if matched is not None:
                self._append_plain(text, plain_start, pos, output)
                self._flush_plain(output)
                output.append(matched)
                pos += len(matched)
                plain_start = pos
            else:
                pos += 1

        self._append_plain(text, plain_start, len(text), output)
        if final:
            self._flush_plain(output)
        return output
