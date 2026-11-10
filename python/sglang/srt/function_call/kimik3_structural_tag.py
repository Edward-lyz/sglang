import logging
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    AnyTokensFormat,
    ConstStringFormat,
    ExcludeTokenFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    CALL_OPEN,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)

logger = logging.getLogger(__name__)

_JSON_TYPES = (
    "string",
    "number",
    "integer",
    "boolean",
    "array",
    "object",
    "null",
)
_CLOSE_TOKEN = "<|close|>"
_ARGUMENT_CLOSE_SUFFIX = ARGUMENT_CLOSE.removeprefix(_CLOSE_TOKEN)
_JSON_TO_XTML_TYPE = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "null": "null",
}
_STRING_KEYWORDS = {"format", "maxLength", "minLength", "pattern"}
_NUMBER_KEYWORDS = {
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "minimum",
    "multipleOf",
}
_ARRAY_KEYWORDS = {
    "contains",
    "items",
    "maxContains",
    "maxItems",
    "minContains",
    "minItems",
    "prefixItems",
    "uniqueItems",
}
_OBJECT_KEYWORDS = {
    "additionalProperties",
    "dependentRequired",
    "dependentSchemas",
    "maxProperties",
    "minProperties",
    "patternProperties",
    "properties",
    "propertyNames",
    "required",
}


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _matches_json_type(value: Any, json_type: str) -> bool:
    value_type = _json_type(value)
    return value_type == json_type or (
        json_type == "number" and value_type == "integer"
    )


def _resolve_local_ref(
    ref: str, root_schema: Dict[str, Any]
) -> Optional[Union[bool, Dict[str, Any]]]:
    if not ref.startswith("#/"):
        return None
    value: Any = root_schema
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, (bool, dict)):
        return value
    return None


def _schema_types(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    seen_refs: Optional[Set[str]] = None,
) -> List[str]:
    if schema is False:
        return []
    if schema is True:
        return list(_JSON_TYPES)

    ref = schema.get("$ref")
    if isinstance(ref, str):
        seen_refs = set() if seen_refs is None else set(seen_refs)
        if ref not in seen_refs:
            target = _resolve_local_ref(ref, root_schema)
            if target is not None:
                seen_refs.add(ref)
                return _schema_types(target, root_schema, seen_refs)

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type] if schema_type in _JSON_TYPES else list(_JSON_TYPES)
    if isinstance(schema_type, list):
        return [item for item in _JSON_TYPES if item in schema_type]

    for keyword in ("anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list):
            option_types = {
                item
                for option in options
                if isinstance(option, (bool, dict))
                for item in _schema_types(option, root_schema, seen_refs)
            }
            return [item for item in _JSON_TYPES if item in option_types]

    options = schema.get("allOf")
    if isinstance(options, list):
        type_sets = [
            set(_schema_types(option, root_schema, seen_refs))
            for option in options
            if isinstance(option, (bool, dict))
        ]
        constrained = [types for types in type_sets if types != set(_JSON_TYPES)]
        if constrained:
            result = constrained[0] | (
                {"integer"} if "number" in constrained[0] else set()
            )
            for types in constrained[1:]:
                result &= types | ({"integer"} if "number" in types else set())
            # number survives the intersection only if every branch allows it,
            # making the widened integer redundant rather than the other way.
            if "number" in result and "integer" in result:
                result.remove("integer")
            return [item for item in _JSON_TYPES if item in result]

    if "const" in schema:
        return [_json_type(schema["const"])]
    enum = schema.get("enum")
    if isinstance(enum, list):
        enum_types = {_json_type(value) for value in enum}
        return [item for item in _JSON_TYPES if item in enum_types]
    if _OBJECT_KEYWORDS.intersection(schema):
        return ["object"]
    if _ARRAY_KEYWORDS.intersection(schema):
        return ["array"]
    if _STRING_KEYWORDS.intersection(schema):
        return ["string"]
    if _NUMBER_KEYWORDS.intersection(schema):
        return ["number"]
    return list(_JSON_TYPES)


_DEFINITION_KEYWORDS = ("$defs", "definitions")
_HOISTED_DEF_PREFIX = "__sglang_ref_"


def _with_root_definitions(
    schema: Union[bool, Dict[str, Any]], root_schema: Dict[str, Any]
) -> Union[bool, Dict[str, Any]]:
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    for key in _DEFINITION_KEYWORDS:
        root_definitions = root_schema.get(key)
        own_definitions = result.get(key)
        if root_definitions is None:
            continue
        if own_definitions is None:
            result[key] = root_definitions
        elif isinstance(own_definitions, dict) and isinstance(root_definitions, dict):
            # The argument's own definitions win on a name clash (previous
            # behavior); root names are only added so root refs resolve.
            result[key] = {**root_definitions, **own_definitions}
    return result


def _is_hoistable_ref(ref: Any, root_schema: Dict[str, Any]) -> bool:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return False
    if ref[2:].split("/", 1)[0] in _DEFINITION_KEYWORDS:
        # Already reachable: root definitions are copied into each argument.
        return False
    return _resolve_local_ref(ref, root_schema) is not None


def _rewrite_hoistable_refs(
    node: Any, root_schema: Dict[str, Any], names: Dict[str, str]
) -> Any:
    if isinstance(node, list):
        return [_rewrite_hoistable_refs(item, root_schema, names) for item in node]
    if not isinstance(node, dict):
        return node
    result = {
        key: _rewrite_hoistable_refs(value, root_schema, names)
        for key, value in node.items()
    }
    ref = node.get("$ref")
    if _is_hoistable_ref(ref, root_schema):
        name = names.setdefault(ref, f"{_HOISTED_DEF_PREFIX}{len(names)}")
        result["$ref"] = f"#/$defs/{name}"
    return result


def _hoist_root_pointer_refs(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Make root-relative pointers survive per-argument schema emission.

    Each argument value is compiled as its own JSON schema document, so a
    nested ``$ref`` such as ``#/properties/value/properties/a`` (valid against
    the tool parameters) would point at nothing there. Such targets are hoisted
    into root ``$defs`` under synthetic names and the refs rewritten to them;
    root ``$defs`` are then carried into every argument schema. Refs that do
    not resolve against the parameters (e.g. pointers relative to the
    argument's own sub-schema) are left unchanged.
    """
    names: Dict[str, str] = {}
    rewritten = _rewrite_hoistable_refs(parameters, parameters, names)
    if not names:
        return parameters
    hoisted: Dict[str, Any] = {}
    # Hoisted targets may reference further root pointers; iterate to a fixpoint.
    while len(hoisted) < len(names):
        for ref, name in list(names.items()):
            if name not in hoisted:
                target = _resolve_local_ref(ref, parameters)
                hoisted[name] = _rewrite_hoistable_refs(target, parameters, names)
    root_definitions = rewritten.get("$defs")
    if not isinstance(root_definitions, dict):
        root_definitions = {}
    return {**rewritten, "$defs": {**root_definitions, **hoisted}}


def _restrict_schema_type(
    schema: Union[bool, Dict[str, Any]],
    json_type: str,
    root_schema: Dict[str, Any],
) -> Union[bool, Dict[str, Any]]:
    if not isinstance(schema, dict):
        return {"type": json_type} if schema else False

    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve_local_ref(ref, root_schema)
        if target is not None:
            return _with_root_definitions(
                _restrict_schema_type(target, json_type, root_schema), root_schema
            )

    result = dict(schema)
    schema_type = result.get("type")
    if isinstance(schema_type, list):
        if json_type not in schema_type:
            return False
        result["type"] = json_type
    elif isinstance(schema_type, str):
        if schema_type != json_type:
            return False
    else:
        result["type"] = json_type

    for keyword in ("anyOf", "oneOf"):
        options = result.get(keyword)
        if not isinstance(options, list):
            continue
        restricted = [
            _restrict_schema_type(option, json_type, root_schema)
            for option in options
            if isinstance(option, (bool, dict))
            and json_type in _schema_types(option, root_schema)
        ]
        if not restricted:
            return False
        if len(restricted) == 1 and isinstance(restricted[0], dict):
            result.pop(keyword)
            result.update(restricted[0])
        else:
            result[keyword] = restricted

    enum = result.get("enum")
    if isinstance(enum, list):
        result["enum"] = [
            value for value in enum if _matches_json_type(value, json_type)
        ]
        if not result["enum"]:
            return False
    if "const" in result and not _matches_json_type(result["const"], json_type):
        return False
    return _with_root_definitions(result, root_schema)


# xgrammar 0.2.1 miscompiles the bounded integer/number grammar in exactly one
# shape: a negative lower bound (minimum/exclusiveMinimum < 0) with no upper
# bound. There it accepts a bare "-" as a complete value while rejecting real
# integers ("-1", "-1000000"); the model then emits {"value":"-"} and schema
# validation fails with "'-' is not of type 'integer'". Dropping just the lower
# bound in that shape yields a well-formed signed integer/number grammar; the
# (now unenforced) lower limit is still checked by schema validation. All other
# bound shapes compile correctly and keep their constraints.
_LOWER_BOUND_KEYWORDS = ("minimum", "exclusiveMinimum")
_UPPER_BOUND_KEYWORDS = ("maximum", "exclusiveMaximum")


def _has_negative_lower_bound(schema: Dict[str, Any]) -> bool:
    return any(
        isinstance(schema.get(key), (int, float)) and schema[key] < 0
        for key in _LOWER_BOUND_KEYWORDS
    )


def _strip_miscompiled_lower_bound(
    schema: Union[bool, Dict[str, Any]],
) -> Union[bool, Dict[str, Any]]:
    if (
        not isinstance(schema, dict)
        or any(key in schema for key in _UPPER_BOUND_KEYWORDS)
        or not _has_negative_lower_bound(schema)
    ):
        return schema
    return {
        key: value
        for key, value in schema.items()
        if key not in _LOWER_BOUND_KEYWORDS
    }


def _value_format(
    schema: Union[bool, Dict[str, Any]],
    json_type: str,
    loose_string: bool = False,
) -> Format:
    if loose_string and json_type == "string":
        return AnyTextFormat()
    if json_type in ("integer", "number"):
        schema = _strip_miscompiled_lower_bound(schema)
    return JSONSchemaFormat(
        json_schema=schema,
        style="qwen_xml" if json_type == "string" else "json",
    )


def _argument_value_variants(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    loose_strings: bool = False,
) -> List[Tuple[str, Format]]:
    return [
        (
            json_type,
            _value_format(restricted, json_type, loose_string=loose_strings),
        )
        for json_type in _schema_types(schema, root_schema)
        if (restricted := _restrict_schema_type(schema, json_type, root_schema))
        is not False
    ]


def _known_argument_format(
    key: str,
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
) -> Optional[Format]:
    escaped_key = _escape_attr(key)
    variants = [
        TagFormat(
            begin=(
                f'<|open|>argument key="{escaped_key}" '
                f'type="{_JSON_TO_XTML_TYPE[json_type]}"<|sep|>'
            ),
            content=value_format,
            end=ARGUMENT_CLOSE,
        )
        for json_type, value_format in _argument_value_variants(schema, root_schema)
    ]
    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    return OrFormat(elements=variants)


def _dynamic_argument_format(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    loose_strings: bool = False,
) -> Format:
    variants = [
        SequenceFormat(
            elements=[
                RegexFormat(pattern=r'[^"& \t\r\n\f\v=<>]+'),
                ConstStringFormat(
                    value=(f'" type="{_JSON_TO_XTML_TYPE[json_type]}"<|sep|>')
                ),
                value_format,
            ]
        )
        for json_type, value_format in _argument_value_variants(
            schema, root_schema, loose_strings=loose_strings
        )
    ]
    if not variants:
        raise ValueError("Kimi K3 additional parameter schema accepts no values")
    content = variants[0] if len(variants) == 1 else OrFormat(elements=variants)
    return TagFormat(
        begin='<|open|>argument key="',
        content=content,
        end=ARGUMENT_CLOSE,
    )


def _strict_arguments_format(parameters: Dict[str, Any]) -> Format:
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Kimi K3 tool parameters 'properties' must be an object")
    required = parameters.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise ValueError("Kimi K3 tool parameters 'required' must be a string list")

    required_set = set(required)
    missing = required_set.difference(properties)
    if missing:
        raise ValueError(
            f"Kimi K3 required parameters are missing schemas: {sorted(missing)!r}"
        )

    elements: List[Format] = []
    for key, schema in properties.items():
        if not isinstance(key, str) or not isinstance(schema, (bool, dict)):
            raise ValueError("Kimi K3 tool property schemas must be JSON schemas")
        argument = _known_argument_format(key, schema, parameters)
        if argument is None:
            if key in required_set:
                raise ValueError(
                    f"Kimi K3 required parameter {key!r} accepts no values"
                )
            continue
        elements.append(
            argument if key in required_set else OptionalFormat(content=argument)
        )

    # OpenAI strict semantics default additionalProperties to False; a strict
    # tool that omits it takes no extra arguments. Defaulting to True appended an
    # unbounded (extra argument)* to the call, which let the model open argument
    # tags forever without ever having to emit <|close|>call<|sep|> -- kimi_k3
    # only surfaces tool_calls once the call closes, so the request looked hung.
    additional = parameters.get("additionalProperties", False)
    if additional is True:
        elements.append(StarFormat(content=_dynamic_argument_format(True, parameters)))
    elif isinstance(additional, dict):
        elements.append(
            StarFormat(content=_dynamic_argument_format(additional, parameters))
        )
    elif additional is not False:
        raise ValueError(
            "Kimi K3 tool parameters 'additionalProperties' must be a schema"
        )
    if not elements:
        return ConstStringFormat(value="")
    return SequenceFormat(elements=elements)


def _tool_arguments_format(tool: Tool) -> Format:
    parameters = tool.function.parameters
    if not tool.function.strict:
        root_schema = parameters if isinstance(parameters, dict) else {}
        return StarFormat(
            content=_dynamic_argument_format(True, root_schema, loose_strings=True)
        )
    if parameters is None:
        # Server-side strict levels mark tools without parameters strict too;
        # they take no arguments rather than failing the whole constraint.
        return ConstStringFormat(value="")
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Kimi K3 strict tool {tool.function.name!r} must define parameters"
        )
    schema_types = _schema_types(parameters, parameters)
    if "object" not in schema_types:
        raise ValueError(
            f"Kimi K3 tool {tool.function.name!r} parameters must be an object schema"
        )
    return _strict_arguments_format(_hoist_root_pointer_refs(parameters))


def _tool_call_tag(tool: Tool, arguments_format: Optional[Format] = None) -> TagFormat:
    name = _escape_attr(tool.function.name)
    if arguments_format is None:
        arguments_format = _tool_arguments_format(tool)
    return TagFormat(
        begin=f'{CALL_OPEN} tool="{name}" index="',
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[1-9][0-9]*"),
                ConstStringFormat(value='"<|sep|>'),
                arguments_format,
            ]
        ),
        end=CALL_CLOSE,
    )


def _tool_calls_tag(call_tags: List[TagFormat], parallel_tool_calls: bool) -> TagFormat:
    if parallel_tool_calls:
        content: Format = TagsWithSeparatorFormat(
            tags=call_tags, separator="", at_least_one=True
        )
    elif len(call_tags) == 1:
        content = call_tags[0]
    else:
        content = OrFormat(elements=call_tags)
    return TagFormat(begin=TOOLS_OPEN, content=content, end=TOOLS_CLOSE)


def _auto_suffix(tools_tag: TagFormat) -> Format:
    # tool_choice="auto": the assistant always emits a response channel and may
    # additionally open the tools section, then closes the message. The K3 encoder
    # (_render_assistant_segments) always wraps the turn in
    # <|open|>response<|sep|>...<|close|>response<|sep|> [tools] <|close|>message<|sep|>,
    # so MESSAGE_CLOSE terminates every assistant turn whether or not a tool is
    # called.
    #
    # The earlier TriggeredTagsFormat rendered as ``free (tags)* free`` (see
    # structural_tag_to_lark._emit_triggered_tags): the trailing free region and
    # its unbounded ``(text | response_channel)*`` never modelled the terminator
    # and never forced EOS, so after a completed call the model kept sampling
    # legal non-EOS tokens — re-opening response channels in a loop until it hit
    # max_tokens. Mirroring _required_suffix (response channel, then the tools
    # section, then a mandatory MESSAGE_CLOSE) forces a clean stop; the only
    # difference is that the tools section is optional here. The single-call-vs-
    # parallel limit already lives inside tools_tag (_tool_calls_tag).
    return SequenceFormat(
        elements=[
            _response_channel(),
            OptionalFormat(content=tools_tag),
            ConstStringFormat(value=MESSAGE_CLOSE),
        ]
    )


def _with_reasoning(suffix: Format, thinking_mode: bool) -> Format:
    if not thinking_mode:
        return suffix
    return SequenceFormat(
        elements=[
            TagFormat(begin="", content=AnyTextFormat(), end=THINK_CLOSE),
            suffix,
        ]
    )


def _response_channel() -> OptionalFormat:
    # The K3 response channel that precedes the tools section:
    # ``<|open|>response<|sep|>...<|close|>response<|sep|>``. Reasoning is peeled
    # off ahead of this by the ``think`` wrapper. The whole channel is optional and
    # its body may be empty. The response-open marker is optional because the
    # generation prompt may already have emitted it (non-thinking mode) or the
    # model emits it itself (reasoning handled out of band, so thinking_mode is
    # False here even though the model reasons). The body excludes every structural
    # marker so free text cannot smuggle a second channel, a tools section, or the
    # terminator into the reply.
    response_body = AnyTextFormat(
        excludes=[
            RESPONSE_OPEN,
            RESPONSE_CLOSE,
            TOOLS_OPEN,
            TOOLS_CLOSE,
            THINK_OPEN,
            THINK_CLOSE,
            CALL_OPEN,
            MESSAGE_CLOSE,
        ]
    )
    return OptionalFormat(
        content=SequenceFormat(
            elements=[
                OptionalFormat(content=ConstStringFormat(value=RESPONSE_OPEN)),
                response_body,
                ConstStringFormat(value=RESPONSE_CLOSE),
            ]
        )
    )


def _required_suffix(tools_tag: TagFormat) -> Format:
    # tool_choice="required": the assistant emits its response channel, then
    # MUST open the tools section, then closes the message. The old grammar
    # let a leading AnyTextFormat absorb everything before the tools section,
    # which let the model ramble unbounded free text / plain JSON and never
    # open <|open|>tools<|sep|> (hitting max_tokens with no parseable call),
    # and had no terminator so a completed call was followed by
    # <|close|>message<|sep|> spam instead of EOS.
    #
    # Modelling the response channel explicitly fixes both: text can only
    # appear inside the response wrapper, so once it closes the sole legal
    # continuation is the tools section; and appending MESSAGE_CLOSE after the
    # tools section forces a clean stop.
    return SequenceFormat(
        elements=[
            _response_channel(),
            tools_tag,
            ConstStringFormat(value=MESSAGE_CLOSE),
        ]
    )


def _single_xtml_type(
    schema: Union[bool, Dict[str, Any]], root_schema: Dict[str, Any]
) -> Optional[str]:
    schema_types = _schema_types(schema, root_schema)
    if len(schema_types) != 1:
        return None
    return _JSON_TO_XTML_TYPE[schema_types[0]]


def _nonempty_argument_format(key: str, xtml_type: str) -> Format:
    # A token-based end keeps the first close token out of both content formats.
    argument = TagFormat(
        begin=(
            f'<|open|>argument key="{_escape_attr(key)}" ' f'type="{xtml_type}"<|sep|>'
        ),
        content=SequenceFormat(elements=[ExcludeTokenFormat(), AnyTokensFormat()]),
        end=TokenFormat(token=_CLOSE_TOKEN),
    )
    return SequenceFormat(
        elements=[
            argument,
            ConstStringFormat(value=_ARGUMENT_CLOSE_SUFFIX),
        ]
    )


def _auto_tool_arguments_format(tool: Tool) -> Format:
    parameters = tool.function.parameters
    if not isinstance(parameters, dict):
        return AnyTextFormat()
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return AnyTextFormat()

    required_tags: List[Format] = []
    for key in required:
        schema = properties.get(key)
        if not isinstance(key, str) or not isinstance(schema, (bool, dict)):
            return AnyTextFormat()
        xtml_type = _single_xtml_type(schema, parameters)
        if xtml_type is None:
            return AnyTextFormat()
        required_tags.append(_nonempty_argument_format(key, xtml_type))

    if required_tags:
        elements = required_tags
    else:
        alternatives = [
            _nonempty_argument_format(key, xtml_type)
            for key, schema in properties.items()
            if isinstance(key, str)
            and isinstance(schema, (bool, dict))
            and (xtml_type := _single_xtml_type(schema, parameters)) is not None
        ]
        if not alternatives:
            return AnyTextFormat()
        elements = [
            (
                alternatives[0]
                if len(alternatives) == 1
                else OrFormat(elements=alternatives)
            )
        ]

    return SequenceFormat(
        elements=[*elements, AnyTextFormat(excludes=[CALL_OPEN, CALL_CLOSE])]
    )


def get_kimik3_auto_tool_call_structural_tag(
    tools: List[Tool],
    thinking_mode: bool = False,
    parallel_tool_calls: bool = True,
) -> Optional[StructuralTag]:
    if not tools:
        return None

    call_tags = [
        _tool_call_tag(tool, _auto_tool_arguments_format(tool)) for tool in tools
    ]
    suffix = _auto_suffix(
        _tool_calls_tag(call_tags, parallel_tool_calls)
    )
    return StructuralTag(format=_with_reasoning(suffix, thinking_mode))


def promote_tools_to_strict(tools: List[Tool]) -> List[Tool]:
    """Mark every tool strict, leaving behind the ones whose schema cannot compile.

    ``SGLANG_TOOL_STRICT_LEVEL=PARAMETER`` asks for parameter-level constraints on
    tools that never opted into ``strict``. The strict argument builder rejects
    schemas the loose builder tolerates (``required`` naming an undeclared
    property, a non-object root, ...), and callers send such schemas routinely, so
    a rejected tool keeps the loose format instead of failing the whole request.

    A tool whose request explicitly sets ``"strict": false`` is left loose
    (unlike the community, which overrides it). The strict format pins arguments
    to ``properties`` order, so a caller that opted out keeps any-order arguments.
    """
    promoted: List[Tool] = []
    for tool in tools:
        if tool.function.strict or "strict" in tool.function.model_fields_set:
            promoted.append(tool)
            continue
        candidate = tool.model_copy(
            update={"function": tool.function.model_copy(update={"strict": True})}
        )
        try:
            _tool_arguments_format(candidate)
        except ValueError as error:
            logger.warning(
                "Kimi K3 tool %r stays loosely constrained: %s",
                tool.function.name,
                error,
            )
            promoted.append(tool)
        else:
            promoted.append(candidate)
    return promoted


def _select_tools(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["auto", "required"]],
) -> Tuple[List[Tool], bool]:
    if not isinstance(tool_choice, ToolChoice):
        return tools, tool_choice == "required"
    name = tool_choice.function.name
    selected = [tool for tool in tools if tool.function.name == name]
    if not selected:
        raise ValueError(f"Kimi K3 tool choice {name!r} is not in the tools list")
    return selected, True


def get_kimik3_structural_tag(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
    thinking_mode: bool = False,
    parallel_tool_calls: bool = True,
) -> StructuralTag:
    selected_tools, at_least_one = _select_tools(tools, tool_choice)
    if not selected_tools:
        raise ValueError("Kimi K3 structural tags require at least one tool")

    call_tags = [_tool_call_tag(tool) for tool in selected_tools]
    tools_tag = _tool_calls_tag(call_tags, parallel_tool_calls)
    if at_least_one:
        suffix: Format = _required_suffix(tools_tag)
    else:
        suffix = _auto_suffix(tools_tag)
    return StructuralTag(format=_with_reasoning(suffix, thinking_mode))
