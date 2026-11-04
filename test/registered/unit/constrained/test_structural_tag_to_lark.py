"""Kimi-K3 xgrammar-native structural tags compile to a valid llguidance Lark
grammar.

Kimi-K3 emits its tool-call constraint as the xgrammar-native
``{"type": "structural_tag", "format": {...}}`` shape. The llguidance backend
used to only accept the *legacy* (``structures``/``triggers``) shape and dropped
the native tree into an ``InvalidGrammarObject``, so K3 tool-call constraints
never compiled on ``--grammar-backend llguidance``. ``structural_tag_to_lark``
walks the serialized Format tree and produces an llguidance Lark grammar; these
tests pin that every strict-path Format node the K3 generator can emit yields a
grammar that ``LLMatcher.validate_grammar`` accepts (empty error string), plus
the raw-value semantics that make the emitted values round-trip.
"""

import unittest

from llguidance import LLMatcher, grammar_from

from sglang.srt.constrained.structural_tag_to_lark import structural_tag_to_lark
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_structural_tag,
    promote_tools_to_strict,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tool(name, parameters, strict=True):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description="d",
            parameters=parameters,
            strict=strict,
        ),
    )


def _object(properties, required=None, **extra):
    schema = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    schema.update(extra)
    return schema


def _lark_for(tools, **kwargs):
    promoted = promote_tools_to_strict(tools)
    tag = get_kimik3_structural_tag(promoted, tool_choice="required", **kwargs)
    return structural_tag_to_lark(tag.model_dump(by_alias=True))


def _validate(lark):
    return LLMatcher.validate_grammar(grammar_from("lark", lark))


def _synthetic_k3_tokenizer():
    """A byte-level tiktoken model with K3's three markers as atomic special tokens.

    Mirrors the fixed backend (``_llguidance_tokenizer_from_tiktoken``): the
    markers are declared to ``LLTokenizer.from_tiktoken`` as real special tokens,
    so the atomic marker id -- the id ``enc.encode(..., allowed_special=...)``
    emits and the id the grammar's bare ``<|open|>`` reference asks for -- is the
    id the mask admits. Folding them into the byte encoder instead
    re-canonicalizes each marker into its bytes and drops the atomic id from the
    mask (the bug under repair).
    """
    import tiktoken
    from llguidance import LLTokenizer

    from sglang.srt.tokenizer.tiktoken_tokenizer import PAT_STR_B

    specials = {"<|open|>": 256, "<|sep|>": 257, "<|close|>": 258, "<|eos|>": 259}
    enc = tiktoken.Encoding(
        name="k3-syn",
        pat_str=PAT_STR_B,
        mergeable_ranks={bytes([b]): b for b in range(256)},
        special_tokens=specials,
    )
    lltok = LLTokenizer.from_tiktoken(
        encoder={bytes([b]): b for b in range(256)},
        special_tokens=dict(specials),
        pattern=PAT_STR_B,
        eos_token=specials["<|eos|>"],
        n_vocab=enc.n_vocab,
    )
    return enc, lltok, specials


def _bit_allowed(mask: bytes, token_id: int) -> bool:
    """Read one token's admit bit from an llguidance ``compute_bitmask`` blob."""
    return bool(mask[token_id // 8] & (1 << (token_id % 8)))


class TestStructuralTagToLark(unittest.TestCase):
    # Each fixture exercises a distinct Format-node path the strict K3 generator
    # can emit; a translator regression on any node type turns its case red.
    def test_string_argument(self):
        lark = _lark_for([_tool("a", _object({"x": {"type": "string"}}, ["x"]))])
        self.assertEqual(_validate(lark), "")

    def test_number_argument(self):
        lark = _lark_for([_tool("a", _object({"x": {"type": "number"}}, ["x"]))])
        self.assertEqual(_validate(lark), "")

    def test_integer_argument(self):
        lark = _lark_for([_tool("a", _object({"x": {"type": "integer"}}, ["x"]))])
        self.assertEqual(_validate(lark), "")

    def test_boolean_argument(self):
        lark = _lark_for([_tool("a", _object({"x": {"type": "boolean"}}, ["x"]))])
        self.assertEqual(_validate(lark), "")

    def test_enum_string_argument(self):
        lark = _lark_for(
            [_tool("a", _object({"x": {"type": "string", "enum": ["p", "q"]}}, ["x"]))]
        )
        self.assertEqual(_validate(lark), "")

    def test_const_argument(self):
        lark = _lark_for(
            [_tool("a", _object({"x": {"type": "string", "const": "fixed"}}, ["x"]))]
        )
        self.assertEqual(_validate(lark), "")

    def test_nested_object_argument(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object(
                        {"x": {"type": "object", "properties": {"y": {"type": "integer"}}}},
                        ["x"],
                    ),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_array_items_argument(self):
        lark = _lark_for(
            [_tool("a", _object({"x": {"type": "array", "items": {"type": "string"}}}, ["x"]))]
        )
        self.assertEqual(_validate(lark), "")

    def test_array_prefix_items_argument(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object(
                        {
                            "x": {
                                "type": "array",
                                "prefixItems": [
                                    {"type": "string"},
                                    {"type": "integer"},
                                ],
                            }
                        },
                        ["x"],
                    ),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_recursive_ref_argument(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object(
                        {"x": {"$ref": "#/$defs/N"}},
                        ["x"],
                        **{
                            "$defs": {
                                "N": {
                                    "type": "object",
                                    "properties": {"c": {"$ref": "#/$defs/N"}},
                                }
                            }
                        },
                    ),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_one_of_argument(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object(
                        {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
                        ["x"],
                    ),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_any_of_argument(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object(
                        {"x": {"anyOf": [{"type": "string"}, {"type": "boolean"}]}},
                        ["x"],
                    ),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_all_of_argument(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object(
                        {"x": {"allOf": [{"type": "integer"}, {"minimum": 0}]}},
                        ["x"],
                    ),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_additional_properties_true(self):
        lark = _lark_for(
            [_tool("a", _object({"x": {"type": "string"}}, ["x"], additionalProperties=True))]
        )
        self.assertEqual(_validate(lark), "")

    def test_additional_properties_schema(self):
        lark = _lark_for(
            [_tool("a", _object({}, additionalProperties={"type": "string"}))]
        )
        self.assertEqual(_validate(lark), "")

    def test_additional_properties_false(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object({"x": {"type": "string"}}, ["x"], additionalProperties=False),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_optional_argument(self):
        lark = _lark_for(
            [
                _tool(
                    "a",
                    _object(
                        {"x": {"type": "string"}, "y": {"type": "integer"}},
                        ["x"],
                    ),
                )
            ]
        )
        self.assertEqual(_validate(lark), "")

    def test_multiple_tools_parallel(self):
        lark = _lark_for(
            [
                _tool("a", _object({"x": {"type": "string"}}, ["x"])),
                _tool("b", _object({"y": {"type": "integer"}}, ["y"])),
            ],
            parallel_tool_calls=True,
        )
        self.assertEqual(_validate(lark), "")

    def test_thinking_mode_wrapper(self):
        lark = _lark_for(
            [_tool("a", _object({"x": {"type": "string"}}, ["x"]))],
            thinking_mode=True,
        )
        self.assertEqual(_validate(lark), "")


class TestStructuralTagToLarkSemantics(unittest.TestCase):
    # Derived-property guards on the raw-value rendering — the crux that makes
    # qwen_xml string values round-trip unquoted through KimiK3Detector.
    def test_qwen_xml_enum_values_are_raw_and_unquoted(self):
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "json_schema": {"type": "string", "enum": ["p", "q"]},
                "style": "qwen_xml",
            },
        }
        lark = structural_tag_to_lark(tag)
        self.assertIn('"p"', lark)
        self.assertIn('"q"', lark)
        # A raw string value must not be JSON-quoted (no escaped inner quotes).
        self.assertNotIn('"\\"p\\""', lark)
        self.assertEqual(_validate(lark), "")

    def test_json_style_embeds_inline_json_schema(self):
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "json_schema": {"type": "integer"},
                "style": "json",
            },
        }
        lark = structural_tag_to_lark(tag)
        self.assertIn("%json", lark)
        self.assertEqual(_validate(lark), "")

    def test_begin_end_double_quotes_are_escaped(self):
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": '<|open|>call tool="a" index="',
                "content": {"type": "regex", "pattern": "[0-9]+"},
                "end": '"<|sep|>',
            },
        }
        lark = structural_tag_to_lark(tag)
        # Embedded " survives as an escaped Lark literal quote.
        self.assertIn('\\"a\\"', lark)
        self.assertEqual(_validate(lark), "")

    def test_control_char_enum_values_compile(self):
        # Regression: an enum/const string value carrying a control character
        # (newline, tab) was emitted as that literal byte inside a Lark "..."
        # terminal, which is a lexer error -- the whole structural_tag failed to
        # compile ("Failed to compile structural_tag grammar"). Such values must
        # be rendered as escape sequences so the grammar validates. Seen with a
        # real TestBasicTypes case: enum members "", " ", "\n", "\t", "a\\b".
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "enum": ["", " ", "\n", "\t", "special\\chars"],
                    "type": "string",
                },
                "style": "qwen_xml",
            },
        }
        lark = structural_tag_to_lark(tag)
        self.assertEqual(_validate(lark), "")

    def test_unknown_node_type_raises(self):
        tag = {"type": "structural_tag", "format": {"type": "no_such_node"}}
        with self.assertRaises(ValueError):
            structural_tag_to_lark(tag)


class TestAutoPathTermination(unittest.TestCase):
    """Auto path (``tool_choice=auto``) terminates like the strict path.

    ``kimik3_structural_tag._auto_suffix`` builds the auto turn as
    ``response_channel? tools_tag? <|close|>message<|sep|>`` -- the same
    Lark-expressible nodes as the strict ``_required_suffix``, differing only in
    that the tools section is optional. ``MESSAGE_CLOSE`` is *mandatory* and
    closes every assistant turn (the K3 encoder wraps every turn that way, and
    ``KimiK3Detector`` strips it), so after it the grammar is exhausted and
    llguidance forces EOS. This replaced the earlier non-terminating
    ``TriggeredTagsFormat`` (``free (tags)* free``): fully nullable/starred, it
    left EOS competing with ~180 other legal tokens at every accepting point, so
    under temp sampling the model looped to the token cap
    (``finish_reason=length``, observed live on ``tool_choice=auto``).

    Free text lives only inside the one response channel, whose body is an
    ``any_text`` node -- the translator renders it ``/[^\x00]*/``, a run of
    content bytes that MAY contain a literal ``<`` (K3 markers arrive as atomic
    special tokens, and a byte-regex terminal can never match a special token, so
    admitting ``<`` bytes cannot reconstruct a bare ``<|open|>tools``). The
    channel therefore does not shadow the tool-call tag; the tag's inner facets
    stay decode-enforced on the auto path too, pinned by
    ``test_auto_path_enforces_inner_maxlength`` (alongside the strict-path
    ``test_strict_path_enforces_inner_maxlength``)."""

    def _tiktoken(self):
        return _synthetic_k3_tokenizer()

    def test_auto_path_accepts_response_channel_before_tools(self):
        enc, lltok, specials = self._tiktoken()
        tools = [_tool("Note", _object({"text": {"type": "string"}}, ["text"]))]
        tag = get_kimik3_structural_tag(
            promote_tools_to_strict(tools),
            tool_choice="auto",
            thinking_mode=False,
            parallel_tool_calls=True,
        )
        lark = structural_tag_to_lark(tag.model_dump(by_alias=True))
        self.assertEqual(_validate(lark), "")

        matcher = LLMatcher(lltok, grammar_from("lark", lark))
        # A response channel with free text, then the tools section and the
        # mandatory message close -- the whole auto turn round-trips. The reply
        # carries literal ``<`` bytes (``List<int>``, ``i < n``): the content run
        # is ``[^\x00]*``, so ``<`` is legal. Under the earlier ``[^<]`` run these
        # tokens were masked out and this consume returned 0 -- the charset
        # regression guard.
        wire = (
            "<|open|>response<|sep|>Use List<int> when i < n.<|close|>response<|sep|>"
            '<|open|>tools<|sep|><|open|>call tool="Note" index="1"<|sep|>'
            '<|open|>argument key="text" type="string"<|sep|>hi<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
        )
        for token_id in enc.encode(wire, allowed_special=set(specials)):
            self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
        self.assertTrue(matcher.is_accepting())
        self.assertFalse(matcher.is_error())

    def test_content_only_reply_can_terminate_on_eos(self):
        # A text-only reply (response channel, no tools section) must reach
        # ``is_accepting()`` and consume the turn's EOS marker. The tools section
        # is optional (``response tools? MESSAGE_CLOSE``), but MESSAGE_CLOSE is
        # mandatory and closes the turn, so the wire is
        # ``<|open|>response<|sep|>...<|close|>response<|sep|><|close|>message<|sep|>``.
        # K3 ends a turn with an EOS marker that must be grammar-legal, so if the
        # turn did not leave the grammar exhausted the plain-text reply would
        # abort. The reply body carries literal ``<`` (``a < b``, ``List<T>``):
        # the content run is ``[^\x00]*`` so those bytes are legal -- under the
        # earlier ``[^<]`` run they were masked and the reply could not be typed.
        # Asserting both parallel modes pins each tools-optional branch.
        enc, lltok, specials = self._tiktoken()
        tools = [_tool("Note", _object({"text": {"type": "string"}}, ["text"]))]
        content = (
            "<|open|>response<|sep|>Use `a < b` and List<T>.<|close|>response<|sep|>"
            "<|close|>message<|sep|>"
        )
        for parallel in (True, False):
            with self.subTest(parallel=parallel):
                tag = get_kimik3_structural_tag(
                    promote_tools_to_strict(tools),
                    tool_choice="auto",
                    thinking_mode=False,
                    parallel_tool_calls=parallel,
                )
                lark = structural_tag_to_lark(tag.model_dump(by_alias=True))
                matcher = LLMatcher(lltok, grammar_from("lark", lark))
                for token_id in enc.encode(content, allowed_special=set(specials)):
                    self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
                # A text-only turn must be terminable and consume the EOS marker.
                self.assertTrue(matcher.is_accepting())
                self.assertTrue(matcher.consume_token(specials["<|eos|>"]))
                self.assertFalse(matcher.is_error())

    def test_auto_path_terminates_after_tool_call(self):
        # Bug regression: on ``tool_choice=auto`` a completed tool call never
        # forced EOS, so the model looped to ``max_tokens``
        # (``finish_reason=length``, 0 usable calls -- observed live on
        # 10.79.184.88:8000 with fc.json). Root cause: the old ``free (tags)* free``
        # triggered_tags shape was fully nullable, so after a tool call the grammar
        # was ``is_accepting`` with ~180 non-EOS tokens still legal; under temp
        # sampling EOS almost never won. The fix converges the auto path to a
        # forced close like the strict path: the tools section must end with
        # ``MESSAGE_CLOSE`` and nothing may follow it.
        #
        # Two black-box forcing signals, each red on the pre-fix shape:
        #   (a) right after the tools tag (``<|close|>tools<|sep|>``) the grammar
        #       is NOT accepting -- ``MESSAGE_CLOSE`` is mandatory (pre-fix the
        #       trailing free region left it accepting);
        #   (b) after ``MESSAGE_CLOSE`` the only legal token is EOS: a free-text
        #       byte and a fresh ``<|open|>`` channel are both rejected, so the
        #       model cannot loop (pre-fix both were legal at the accepting point).
        enc, lltok, specials = self._tiktoken()
        tools = [_tool("Note", _object({"text": {"type": "string"}}, ["text"]))]
        call = (
            '<|open|>tools<|sep|><|open|>call tool="Note" index="1"<|sep|>'
            '<|open|>argument key="text" type="string"<|sep|>hi<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|>"
        )
        message_close = "<|close|>message<|sep|>"
        (free_byte,) = enc.encode("x", allowed_special=set(specials))
        for parallel in (True, False):
            with self.subTest(parallel=parallel):
                tag = get_kimik3_structural_tag(
                    promote_tools_to_strict(tools),
                    tool_choice="auto",
                    thinking_mode=False,
                    parallel_tool_calls=parallel,
                )
                lark = structural_tag_to_lark(tag.model_dump(by_alias=True))
                self.assertEqual(_validate(lark), "")

                matcher = LLMatcher(lltok, grammar_from("lark", lark))
                for token_id in enc.encode(call, allowed_special=set(specials)):
                    self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
                # (a) MESSAGE_CLOSE is mandatory after the tools tag.
                self.assertFalse(matcher.is_accepting())
                for token_id in enc.encode(message_close, allowed_special=set(specials)):
                    self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
                # (b) grammar exhausted: EOS is the only continuation.
                self.assertTrue(matcher.is_accepting())
                self.assertEqual(matcher.try_consume_tokens([free_byte]), 0)
                self.assertEqual(matcher.try_consume_tokens([specials["<|open|>"]]), 0)
                self.assertTrue(matcher.consume_token(specials["<|eos|>"]))
                self.assertFalse(matcher.is_error())

    def test_strict_path_enforces_inner_maxlength(self):
        # The strict path (tool_choice=required) carries the tool call as explicit
        # tag rules with the markers as atomic special-token references and *no*
        # ambiguous free-text region, so the inner /[^\x00]{0,1}/ from a maxLength=1
        # string binds deterministically -- a 2nd content byte is rejected at the
        # mask. This is the facet-enforcement guarantee the whole strict path
        # exists for; a regression that wrapped the strict tags in an ambiguous
        # free region (or dropped the length bound from the value regex) would let
        # the 2nd byte through and turn this red. The auto-path twin,
        # ``test_auto_path_enforces_inner_maxlength``, pins the same guarantee
        # for the auto suffix (response channel + optional tag + MESSAGE_CLOSE).
        enc, lltok, specials = self._tiktoken()
        tool = _tool(
            "write", _object({"content": {"type": "string", "maxLength": 1}}, ["content"])
        )
        tag = get_kimik3_structural_tag(
            promote_tools_to_strict([tool]),
            tool_choice="required",
            thinking_mode=False,
            parallel_tool_calls=False,
        )
        lark = structural_tag_to_lark(tag.model_dump(by_alias=True))
        self.assertEqual(_validate(lark), "")

        matcher = LLMatcher(lltok, grammar_from("lark", lark))
        prefix = (
            '<|open|>tools<|sep|><|open|>call tool="write" index="1"<|sep|>'
            '<|open|>argument key="content" type="string"<|sep|>A'
        )
        for token_id in enc.encode(prefix, allowed_special=set(specials)):
            self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
        # maxLength=1 is exhausted after "A"; a 2nd content byte must be rejected.
        (over,) = enc.encode("B", allowed_special=set(specials))
        self.assertEqual(matcher.try_consume_tokens([over]), 0)

    def test_auto_path_enforces_inner_maxlength(self):
        # Auto-path twin of the strict-path test, and the direct regression for
        # the reported "maxLength 失效" bug. The auto suffix is now
        # ``response_channel? tool_tag? <|close|>message<|sep|>`` with the tool
        # call carried as the same explicit tag rules the strict path uses -- no
        # ambiguous free-text region wraps the call. The response channel body is
        # an AnyText rendered ``/[^\x00]*/`` (a content-byte run that may include
        # ``<``), but the K3 markers are atomic special tokens and a byte-regex
        # run can never match one, so the body cannot reconstruct a bare
        # ``<|open|>tools<|sep|>...``; the tools tag is reachable only through the
        # tag rule, so the inner /[^\x00]{0,1}/ length bound re-binds at decode.
        # The old triggered_tags free region (bare ``<|open|>`` alternative +
        # content runs) let that region parse the whole call as free text, keeping
        # both parses live so the mask union admitted an unbounded ``content`` past
        # maxLength -- that shape is gone.
        # thinking_mode=True mirrors the fc_len1.json payload that surfaced the bug.
        # Asserting both parallel modes pins the optional-tag branch either way.
        enc, lltok, specials = self._tiktoken()
        tool = _tool(
            "write", _object({"content": {"type": "string", "maxLength": 1}}, ["content"])
        )
        opening = "reasoning<|close|>think<|sep|>"
        prefix = (
            '<|open|>tools<|sep|><|open|>call tool="write" index="1"<|sep|>'
            '<|open|>argument key="content" type="string"<|sep|>A'
        )
        (over,) = enc.encode("B", allowed_special=set(specials))
        for parallel in (True, False):
            with self.subTest(parallel=parallel):
                tag = get_kimik3_structural_tag(
                    promote_tools_to_strict([tool]),
                    tool_choice="auto",
                    thinking_mode=True,
                    parallel_tool_calls=parallel,
                )
                lark = structural_tag_to_lark(tag.model_dump(by_alias=True))
                self.assertEqual(_validate(lark), "")
                matcher = LLMatcher(lltok, grammar_from("lark", lark))
                for token_id in enc.encode(
                    opening + prefix, allowed_special=set(specials)
                ):
                    self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
                # maxLength=1 exhausted after "A"; the 2nd content byte must be
                # rejected even though the response channel precedes the call.
                self.assertEqual(matcher.try_consume_tokens([over]), 0)


class TestMaskAdmitsAtomicMarkers(unittest.TestCase):
    """Bug regression (root cause of the DSPARK accept-rate collapse): the K3
    markers were folded into the byte encoder (``special_tokens={}``) and rendered
    in the grammar as quoted byte literals. llguidance enforces canonical
    tokenization, so at a marker boundary the mask admitted only the first
    canonical byte segment (``<``) and masked *out* the atomic special-token id --
    the id the model and the speculative draft actually emit. Nearly every drafted
    marker was rejected, the accept rate cratered, and requests ran away to the
    length cap.

    This probe queries the *mask* (not ``consume``): consuming a byte-literal
    marker succeeds even under the bug (the byte path is legal), so a consume test
    is blind to it. The mask is the object DSPARK reads to score drafts. With the
    fix (markers declared as special tokens + bare ``<|open|>`` references) the
    atomic id is admitted at every marker boundary; and at boundaries *inside* the
    tool-call tag structure (no content run in scope) the bare ``<`` byte is *not*
    admitted, proving the marker takes the atomic-id path, not the byte-literal
    path. Pre-fix the atomic id would be absent from the mask, turning this red.

    (At the very start the strict suffix may instead open the response channel,
    whose ``any_text`` body is a ``[^\x00]*`` content run that legitimately admits
    ``<``, so the bare-``<`` rejection is only asserted at the tag-internal
    boundaries.)"""

    def test_atomic_marker_id_is_in_mask_at_each_boundary(self):
        enc, lltok, specials = _synthetic_k3_tokenizer()
        tools = [_tool("a", _object({"x": {"type": "string"}}, ["x"]))]
        tag = get_kimik3_structural_tag(
            promote_tools_to_strict(tools),
            tool_choice="required",
            thinking_mode=False,
            parallel_tool_calls=False,
        )
        lark = structural_tag_to_lark(tag.model_dump(by_alias=True))
        self.assertEqual(_validate(lark), "")

        bare_lt = enc.encode_ordinary("<")[0]

        # At each atomic-marker boundary the mask must admit the next marker id.
        # ``prefix`` is the wire consumed before the boundary; ``marker`` is the
        # atomic id the grammar demands next. ``no_bare_lt`` marks boundaries with
        # no content run in scope (inside the tag structure): there the bare ``<``
        # byte must be rejected -- the discriminator against the byte-literal path.
        # The start boundary omits it: the response channel's content run admits
        # ``<`` there legitimately.
        boundaries = [
            ("", "<|open|>", False),
            ('<|open|>tools', "<|sep|>", True),
            ('<|open|>tools<|sep|><|open|>call tool="a" index="1"', "<|sep|>", True),
        ]
        for prefix, marker, no_bare_lt in boundaries:
            matcher = LLMatcher(lltok, grammar_from("lark", lark))
            for token_id in enc.encode(prefix, allowed_special=set(specials)):
                self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
            mask = matcher.compute_bitmask()
            self.assertTrue(
                _bit_allowed(mask, specials[marker]),
                f"atomic {marker} id must be admitted after {prefix!r}",
            )
            if no_bare_lt:
                self.assertFalse(
                    _bit_allowed(mask, bare_lt),
                    f"bare '<' byte must not be admitted after {prefix!r}",
                )


class TestJsonStyleLlguidanceRelaxation(unittest.TestCase):
    """Bug regression: llguidance 1.8.0's ``%json{...}`` inline compiler fails
    *closed* on draft keywords it hasn't implemented (``not``, ``uniqueItems``,
    ...), raising ``Unimplemented keys: [...]`` -- which the backend turns into an
    ``InvalidGrammarObject`` and the request into an HTTP 400. A prod L2 +
    llguidance trace showed ~55k such occurrences across tool schemas carrying
    ``not`` / ``uniqueItems``, every one a 400. xgrammar (the default backend)
    never hits this -- it hands the schema verbatim to its C++ engine, which
    ignores ``uniqueItems`` and enforces ``not``. The translator now strips the
    unsupported keywords out of the embedded ``%json`` schema so the surviving
    constraints still compile and enforce (fail-open, matching xgrammar).

    Pre-fix both cases are red: ``grammar_from("lark", ...)`` / the ``LLMatcher``
    constructor raises ``Unimplemented keys``. Post-fix the stripped grammar
    validates and the surviving keywords stay enforced."""

    def _json_tag(self, json_schema):
        return {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "json_schema": json_schema,
                "style": "json",
            },
        }

    def test_unsupported_keywords_no_longer_break_compile(self):
        # Every strip branch at once: scalar drop (uniqueItems), standalone
        # subschema drop (not, propertyNames), group drop (if/then/else), and the
        # oneOf -> anyOf conversion -- nested so the recursive walker is exercised.
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "choice": {
                    "oneOf": [{"type": "string"}, {"type": "integer"}],
                },
                "guarded": {
                    "type": "string",
                    "not": {"const": "forbidden"},
                },
            },
            "required": ["tags"],
            "propertyNames": {"pattern": "^[a-z]+$"},
            "if": {"properties": {"choice": {"type": "string"}}},
            "then": {"required": ["guarded"]},
            "else": {"required": ["tags"]},
        }
        lark = structural_tag_to_lark(self._json_tag(schema))
        self.assertEqual(_validate(lark), "")
        self.assertIn("%json", lark)

    def test_surviving_keywords_still_enforced(self):
        # Strip is surgical, not a flatten: stripped keywords disappear but their
        # surviving siblings (type/required/enum/additionalProperties) stay bound
        # at decode. (a) a value satisfying the survivors -- including a duplicate
        # array item, proving uniqueItems was genuinely relaxed -- consumes fully
        # and is accepting; (b) a value violating a *surviving* keyword (enum) is
        # rejected at the mask.
        enc, lltok, specials = _synthetic_k3_tokenizer()
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "mode": {"type": "string", "enum": ["fast", "slow"]},
            },
            "required": ["items", "mode"],
            "additionalProperties": False,
            "not": {"required": ["forbidden"]},
        }
        lark = structural_tag_to_lark(self._json_tag(schema))
        self.assertEqual(_validate(lark), "")

        # (a) duplicate array item is accepted (uniqueItems relaxed), enum value
        # is a legal member, no extra property -> the whole value round-trips.
        ok = '{"items": ["x", "x"], "mode": "fast"}'
        matcher = LLMatcher(lltok, grammar_from("lark", lark))
        for token_id in enc.encode(ok, allowed_special=set(specials)):
            self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
        self.assertTrue(matcher.is_accepting())
        self.assertFalse(matcher.is_error())

        # (b) a value violating the surviving enum is rejected -- the first byte
        # that diverges from both {"fast","slow"} cannot be consumed.
        bad = '{"items": ["x"], "mode": "'
        matcher = LLMatcher(lltok, grammar_from("lark", lark))
        for token_id in enc.encode(bad, allowed_special=set(specials)):
            self.assertEqual(matcher.try_consume_tokens([token_id]), 1)
        (violating,) = enc.encode("z", allowed_special=set(specials))
        self.assertEqual(matcher.try_consume_tokens([violating]), 0)


class TestOversizedStringLengthBound(unittest.TestCase):
    """Bug regression (as-u3tzp1bnd6): a ``write_file`` tool whose ``content``
    has ``maxLength: 100000`` compiled to the counted repetition
    ``/[^\\x00]{0,100000}/``. llguidance lexes ``{0,N}`` as a fresh DFA state per
    generated byte, so a long file body exhausted ``max_lexer_states``
    (``lexer error: too many states``) mid-generation and the request was
    aborted after minutes of decoding. Bounds above
    ``max_enforced_string_length`` are now dropped, on both the qwen_xml raw
    string path and the embedded ``%json`` path. Each case feeds a value longer
    than the old state budget allows under a reduced ``max_lexer_states``; on
    the pre-fix grammar it errors after a few KB."""

    _MAX_ENFORCED = 4096
    _BODY = "x" * 10000

    def _feed(self, tool, value_wire, max_enforced):
        from llguidance import LLParserLimits

        enc, lltok, specials = _synthetic_k3_tokenizer()
        tag = get_kimik3_structural_tag(
            promote_tools_to_strict([tool]), tool_choice="required"
        )
        lark = structural_tag_to_lark(
            tag.model_dump(by_alias=True), max_enforced_string_length=max_enforced
        )
        matcher = LLMatcher(
            lltok,
            grammar_from("lark", lark),
            limits=LLParserLimits(max_lexer_states=20000),
        )
        wire = (
            '<|open|>tools<|sep|><|open|>call tool="write" index="1"<|sep|>'
            + value_wire
        )
        for token_id in enc.encode(wire, allowed_special=set(specials)):
            matcher.compute_bitmask()
            if matcher.is_error() or matcher.try_consume_tokens([token_id]) != 1:
                return matcher.get_error() or "token rejected"
        return ""

    def test_large_bound_does_not_exhaust_lexer_states(self):
        string_tool = _tool(
            "write",
            _object(
                {"content": {"type": "string", "maxLength": 100000}}, ["content"]
            ),
        )
        string_wire = '<|open|>argument key="content" type="string"<|sep|>' + self._BODY
        json_tool = _tool(
            "write",
            _object(
                {
                    "meta": _object(
                        {"body": {"type": "string", "maxLength": 100000}}, ["body"]
                    )
                },
                ["meta"],
            ),
        )
        json_wire = (
            '<|open|>argument key="meta" type="object"<|sep|>{"body": "' + self._BODY
        )
        for name, tool, wire in (
            ("qwen_xml", string_tool, string_wire),
            ("json", json_tool, json_wire),
        ):
            with self.subTest(name):
                self.assertEqual(self._feed(tool, wire, self._MAX_ENFORCED), "")

    def test_bound_at_threshold_is_still_enforced(self):
        # The threshold itself stays enforced (drop only *above* it): with
        # maxLength == max_enforced the value closes at that length.
        tool = _tool(
            "write",
            _object({"content": {"type": "string", "maxLength": 8}}, ["content"]),
        )
        wire = '<|open|>argument key="content" type="string"<|sep|>'
        self.assertEqual(self._feed(tool, wire + "x" * 8, max_enforced=8), "")
        self.assertEqual(
            self._feed(tool, wire + "x" * 9, max_enforced=8), "token rejected"
        )


class TestOversizedMaxItemsClamped(unittest.TestCase):
    """Bug regression (python/req.json, L2 + llguidance scheduler OOM).

    A tool parameter ``{"type": "array", "maxItems": 2147483647}`` is embedded
    verbatim in ``%json{...}``. llguidance 1.8.0 unrolls ``maxItems`` one symbol
    per item before its grammar-size check, so the compile grows past several GB
    within seconds. The translator now clamps ``maxItems`` above 100000 while
    leaving smaller bounds intact. Only the emitted Lark is checked here:
    compiling the pre-fix grammar would OOM the test runner."""

    def test_huge_max_items_clamped_small_kept(self):
        tool = _tool(
            "delete",
            _object(
                {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 2147483647,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                    },
                },
                ["ids"],
            ),
        )
        lark = _lark_for([tool])
        self.assertNotIn("2147483647", lark)
        self.assertIn('"maxItems": 100000', lark)
        self.assertIn('"maxItems": 3', lark)


class TestParameterLevelHonorsExplicitNonStrict(unittest.TestCase):
    """Bug regression (req_case.json, TaskUpdate loop on SGLANG_TOOL_STRICT_LEVEL=2).

    PARAMETER level promoted a tool the request declared ``"strict": false`` to
    strict. The strict format pins arguments to ``properties`` order, and
    TaskUpdate declares ``taskId`` last, so once the model wrote ``taskId`` first
    the only legal continuation was closing the call: every call came out as
    ``{"taskId": "8"}``, the tool answered "No updates provided" and the model
    retried forever. An explicit ``false`` now keeps the loose any-order format;
    a tool that omits ``strict`` is still promoted.
    """

    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["pending", "completed"]},
            "taskId": {"type": "string"},
        },
        "required": ["taskId"],
    }

    def _feed(self, function, arguments):
        """Feed one auto-path TaskUpdate turn; return '' or the rejection."""
        from sglang.srt.environ import ToolStrictLevel, envs
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tool = Tool.model_validate({"type": "function", "function": function})
        with envs.SGLANG_TOOL_STRICT_LEVEL.override(ToolStrictLevel.PARAMETER):
            constraint = FunctionCallParser(
                tool_call_parser="kimi_k3", tools=[tool], tool_choice="auto"
            ).get_structure_constraint()
        lark = structural_tag_to_lark(constraint[1].model_dump(by_alias=True))
        enc, lltok, specials = _synthetic_k3_tokenizer()
        matcher = LLMatcher(lltok, grammar_from("lark", lark))
        wire = (
            "<|open|>response<|sep|><|close|>response<|sep|>"
            '<|open|>tools<|sep|><|open|>call tool="TaskUpdate" index="1"<|sep|>'
            + "".join(
                f'<|open|>argument key="{key}" type="string"<|sep|>{value}'
                "<|close|>argument<|sep|>"
                for key, value in arguments
            )
            + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
        )
        for token_id in enc.encode(wire, allowed_special=set(specials)):
            if matcher.try_consume_tokens([token_id]) != 1:
                return "token rejected"
        return "" if matcher.is_accepting() else "not accepting"

    def _function(self, **strict):
        return {"name": "TaskUpdate", "parameters": self._SCHEMA, **strict}

    def test_explicit_false_admits_required_argument_first(self):
        self.assertEqual(
            self._feed(
                self._function(strict=False),
                [("taskId", "8"), ("status", "completed")],
            ),
            "",
        )

    def test_omitted_strict_is_still_promoted(self):
        # Negative branch: only an explicit ``false`` opts out. Promotion shows
        # as the enum being enforced (the loose format takes any string).
        bogus = [("status", "bogus"), ("taskId", "8")]
        self.assertEqual(self._feed(self._function(), bogus), "token rejected")
        self.assertEqual(self._feed(self._function(strict=False), bogus), "")


if __name__ == "__main__":
    unittest.main()