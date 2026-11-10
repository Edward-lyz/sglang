# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constrained decoding with llguidance backend."""

import json
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from typing import Iterable, List, Optional, Tuple, Union

import torch
from llguidance import (
    LLExecutor,
    LLMatcher,
    LLParserLimits,
    LLTokenizer,
    StructTag,
    grammar_from,
)
from llguidance.hf import from_tokenizer
from llguidance.torch import (
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
    fill_next_token_bitmask,
    fill_next_token_bitmask_par,
)

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarRow,
    InvalidGrammarObject,
)
from sglang.srt.constrained.structural_tag_to_lark import structural_tag_to_lark
from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.srt.environ import envs
from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

logger = logging.getLogger(__name__)


@cache
def _get_or_init_mask_executor() -> LLExecutor:
    return LLExecutor()


@cache
def _get_or_init_compile_executor() -> ThreadPoolExecutor:
    # LLMatcher construction releases the GIL, so per-request compiles run here,
    # off the scheduler thread.
    return ThreadPoolExecutor(thread_name_prefix="llguidance-compile")


def _new_ll_matcher(
    llguidance_tokenizer: LLTokenizer,
    serialized_grammar: str,
    limits: Optional[LLParserLimits],
) -> LLMatcher:
    return LLMatcher(
        llguidance_tokenizer,
        serialized_grammar,
        log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        limits=limits,
    )


def fill_token_bitmask_batched(
    entries: List[GrammarRow],
    vocab_mask: torch.Tensor,
) -> None:
    """Fill regular-decode mask rows with llguidance's native parallel kernel."""
    if not entries:
        return
    matchers = [(e.grammar.ll_matcher, e.row) for e in entries]
    fill_next_token_bitmask_par(_get_or_init_mask_executor(), matchers, vocab_mask)


def _normalize_eos_token_ids(
    eos_token_ids: Optional[Union[int, Iterable[int]]],
) -> Optional[Union[int, List[int]]]:
    if eos_token_ids is None or isinstance(eos_token_ids, int):
        return eos_token_ids
    return list(eos_token_ids)


def _tiktoken_encoding_for_llguidance(tokenizer):
    """Return the inner ``tiktoken.Encoding`` of a tiktoken-backed tokenizer.

    Two unrelated classes wrap a tiktoken model for K3, and neither is a fast HF
    tokenizer that ``from_tokenizer`` accepts:

      * sglang's ``TiktokenTokenizer`` (loaded only when the tokenizer path ends
        in ``.json``) keeps the ``Encoding`` on ``.tokenizer``;
      * Kimi's HF ``TikTokenTokenizer`` (loaded via ``trust_remote_code`` from a
        model directory, module ``tokenization_kimi``, a *slow*
        ``PreTrainedTokenizer``) keeps it on ``.model``. This is the class a real
        K3 deployment loads, and ``from_tokenizer`` rejects it with "Only fast
        tokenizers are supported" -- so it must be routed through the tiktoken
        bridge too, not just the ``.json`` path.

    Returns ``None`` for any other tokenizer so the caller falls back to
    ``from_tokenizer``. Both wrappers expose a genuine ``tiktoken.Encoding`` whose
    ids are byte-matchable by a grammar literal. The Kimi class is detected by
    name/module (mirroring ``patch_tokenizer._is_kimi_tiktoken_tokenizer``) to
    avoid importing the trust_remote_code module here.
    """
    if isinstance(tokenizer, TiktokenTokenizer):
        return tokenizer.tokenizer
    cls = type(tokenizer)
    if cls.__name__ == "TikTokenTokenizer" and "tokenization_kimi" in (
        cls.__module__ or ""
    ):
        return tokenizer.model
    return None


def _llguidance_tokenizer_from_tiktoken(
    enc,
    *,
    fallback_eos_token_id: Optional[int],
    n_vocab: Optional[int],
    eos_token_ids: Optional[Union[int, Iterable[int]]],
) -> LLTokenizer:
    """Build an llguidance tokenizer from a tiktoken ``Encoding``.

    ``from_tokenizer`` only accepts a HF ``PreTrainedTokenizerFast``; K3 ships a
    tiktoken model. Declare the tiktoken special tokens to llguidance (the same
    ``encoder``/``special_tokens``/``pattern`` triple the official
    ``llguidance.tiktoken.lltokenizer_from_encoding`` helper forwards), rather
    than folding their strings into the byte encoder.

    Folding them in with ``special_tokens={}`` looked equivalent -- every id was
    still byte-decodable -- but it silently broke the *mask*. llguidance enforces
    canonical tokenization, so at a marker boundary a byte-literal ``<|open|>``
    (five bytes ``<`` ``|`` ``open`` ``|`` ``>``) only admits the first canonical
    segment ``<``; the atomic marker id -- the token the model, and the
    speculative draft, actually emit -- is off that canonical path and gets
    masked out. Under DSPARK that rejects nearly every drafted marker, collapsing
    the accept rate and letting a request run away to the length cap. Declaring
    the markers as real special tokens makes the atomic id the sole legal
    continuation at a boundary; ``structural_tag_to_lark`` emits bare ``<|open|>``
    references (not quoted byte literals) so the grammar asks for exactly that id.
    """
    eos_token = _normalize_eos_token_ids(eos_token_ids)
    if eos_token is None:
        eos_token = fallback_eos_token_id
    return LLTokenizer.from_tiktoken(
        encoder=dict(enc._mergeable_ranks),
        special_tokens=dict(enc._special_tokens),
        pattern=enc._pat_str,
        eos_token=eos_token,
        n_vocab=n_vocab or enc.n_vocab,
    )


class GuidanceGrammar(BaseGrammarObject):

    def __init__(
        self,
        llguidance_tokenizer: LLTokenizer,
        serialized_grammar: str,
        *,
        limits: Optional[LLParserLimits] = None,
        pending_matcher: Optional[Future] = None,
    ):
        super().__init__()
        self.llguidance_tokenizer = llguidance_tokenizer
        self.serialized_grammar = serialized_grammar
        self.limits = limits

        # A request copy gets its matcher from a background compile (see copy());
        # it is resolved on first use. Otherwise compile now, so an invalid grammar
        # raises here and the backend turns it into an InvalidGrammarObject.
        self._pending_matcher = pending_matcher
        self._ll_matcher: Optional[LLMatcher] = None
        if pending_matcher is None:
            self._ll_matcher = _new_ll_matcher(
                llguidance_tokenizer, serialized_grammar, limits
            )
            self._raise_if_error()

        self.eos_tokens = set(self.llguidance_tokenizer.eos_tokens)

    @property
    def ll_matcher(self) -> LLMatcher:
        """The underlying LL matcher"""
        if self._ll_matcher is None:
            self._ll_matcher = self._pending_matcher.result()
            self._pending_matcher = None
        return self._ll_matcher

    # Runtime error policy: a matcher error (e.g. "lexer error: too many states")
    # is sticky and leaves llguidance writing an EOS-only mask row. Only
    # accept_token raises -- its callers abort just that request -- so the mask
    # fill and rollback paths, which run inside the forward step, never raise.
    def accept_token(self, token: int):
        if self.finished:
            return
        self._raise_if_error()
        if self.ll_matcher.is_stopped() and token in self.eos_tokens:
            self.finished = True
            return
        self.ll_matcher.consume_token(token)
        self._raise_if_error()

    def rollback(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        if self.finished:
            self.finished = False
            # EOS token after stop isn't tracked in ll_matcher
            num_tokens -= 1
        self.ll_matcher.rollback(num_tokens)

    def is_terminated(self):
        return self.finished

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        fill_next_token_bitmask(self.ll_matcher, vocab_mask, idx)

    @staticmethod
    def fill_vocab_mask_batched(
        entries: List[GrammarRow], vocab_mask: torch.Tensor
    ) -> None:
        """Use the native parallel fill when every entry is a Guidance grammar."""
        if all(isinstance(entry.grammar, GuidanceGrammar) for entry in entries):
            fill_token_bitmask_batched(entries, vocab_mask)
            return
        BaseGrammarObject.fill_vocab_mask_batched(entries, vocab_mask)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, self.llguidance_tokenizer.vocab_size)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        # Not ll_matcher.deep_copy(): that is a shallow Rust clone sharing one
        # lexer DFA across every copy of a cached template. DFA states only grow,
        # so a hot grammar accumulates all its requests' states until it hits
        # max_lexer_states, and the sticky error then fails every later request.
        # Each request compiles its own matcher in the background instead.
        pending_matcher = _get_or_init_compile_executor().submit(
            _new_ll_matcher,
            self.llguidance_tokenizer,
            self.serialized_grammar,
            self.limits,
        )
        return GuidanceGrammar(
            llguidance_tokenizer=self.llguidance_tokenizer,
            serialized_grammar=self.serialized_grammar,
            limits=self.limits,
            pending_matcher=pending_matcher,
        )

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        ff_tokens = self.ll_matcher.compute_ff_tokens()
        if ff_tokens:
            return ff_tokens, ""
        else:
            return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        return "", -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        pass

    def _raise_if_error(self) -> None:
        if self.ll_matcher.is_error():
            raise ValueError(self.ll_matcher.get_error())


class GuidanceBackend(BaseGrammarBackend):

    def __init__(
        self,
        tokenizer,
        any_whitespace: bool = True,
        whitespace_pattern: Optional[str] = None,
        n_vocab: Optional[int] = None,
        eos_token_ids: Optional[Union[int, Iterable[int]]] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.any_whitespace = any_whitespace
        self.whitespace_pattern = whitespace_pattern
        max_enforced = envs.SGLANG_LLGUIDANCE_MAX_ENFORCED_STRING_LENGTH.get()
        self.max_enforced_string_length = None if max_enforced < 0 else max_enforced
        tiktoken_encoding = _tiktoken_encoding_for_llguidance(tokenizer)
        if tiktoken_encoding is not None:
            self.llguidance_tokenizer = _llguidance_tokenizer_from_tiktoken(
                tiktoken_encoding,
                fallback_eos_token_id=tokenizer.eos_token_id,
                n_vocab=n_vocab,
                eos_token_ids=eos_token_ids,
            )
        else:
            self.llguidance_tokenizer = from_tokenizer(
                self.tokenizer,
                n_vocab,
                eos_token=_normalize_eos_token_ids(eos_token_ids),
            )
        # Warm the shared executor here so the first batched mask fill does not
        # pay its one-time setup cost on the request path.
        _get_or_init_mask_executor()

    # Backend-level mask ops, mirroring GuidanceGrammar's object-level ones.
    # ReasonerGrammarBackend wraps grammars for reasoning models (e.g. kimi_k3)
    # and reads these off the backend to build the wrapper; without them the
    # wrapper construction raises AttributeError. XGrammarGrammarBackend exposes
    # the same trio at the backend level for the same reason.
    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        """ allocate_vocab_mask """
        return allocate_token_bitmask(
            batch_size, self.llguidance_tokenizer.vocab_size
        )

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        """ move_vocab_mask """
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        """ apply_vocab_mask """
        apply_token_bitmask_inplace(logits, vocab_mask)

    def _from_serialized(self, serialized_grammar) -> BaseGrammarObject:
        try:
            return GuidanceGrammar(
                llguidance_tokenizer=self.llguidance_tokenizer,
                serialized_grammar=serialized_grammar,
            )
        except Exception as e:
            logger.error(f"Hit invalid grammar: {serialized_grammar=}, {e=}")
            return InvalidGrammarObject(str(e))

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        try:
            serialized_grammar = LLMatcher.grammar_from_json_schema(
                key_string,
                defaults={
                    "whitespace_flexible": self.any_whitespace,
                    "whitespace_pattern": self.whitespace_pattern,
                },
            )
        except Exception as e:
            logger.error(f"Hit invalid json_schema: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))
        return self._from_serialized(serialized_grammar)

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        serialized_grammar = grammar_from("regex", key_string)
        return self._from_serialized(serialized_grammar)

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        try:
            serialized_grammar = grammar_from("ebnf", key_string)
            return self._from_serialized(serialized_grammar)
        except ValueError as e:
            logger.error(f"Hit invalid ebnf: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        try:
            structural_tag = json.loads(key_string)
            if is_legacy_structural_tag(structural_tag):
                return self._from_legacy_structural_tag(structural_tag)
            # xgrammar-native ``{type, format}`` shape (e.g. Kimi-K3 tool calls):
            # xgrammar's token-level TagDispatch has no GBNF/Lark equivalent, so
            # translate the serialized Format tree into an llguidance Lark grammar.
            lark = structural_tag_to_lark(
                structural_tag,
                max_enforced_string_length=self.max_enforced_string_length,
            )
            return self._from_serialized(grammar_from("lark", lark))
        except Exception as e:
            logger.error(f"Hit invalid structural_tag: {key_string=}, {e=}")
            return InvalidGrammarObject(str(e))

    def _from_legacy_structural_tag(
        self, structural_tag: dict
    ) -> BaseGrammarObject:
        # Pair each structure with a trigger that prefixes its own
        # ``begin`` — StructTag asserts begin.startswith(trigger), and
        # detectors with per-tool triggers (e.g. Inkling's
        # <|message_model|>{name}<|content_invoke_tool_json|>) emit a
        # distinct trigger per tool, so triggers[0] matches only one of
        # them and multi-tool grammars fail to compile.
        triggers = structural_tag["triggers"]
        tags = [
            StructTag(
                begin=structure["begin"],
                grammar=structure["schema"],
                end=structure["end"],
                trigger=next(
                    (t for t in triggers if structure["begin"].startswith(t)),
                    triggers[0],
                ),
            )
            for structure in structural_tag["structures"]
        ]
        return self._from_serialized(StructTag.to_grammar(tags))
