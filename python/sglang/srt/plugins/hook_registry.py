"""
Hook registry for SGLang plugins.

Provides before/after/around/replace hooks that can be applied to any
function, method, or class in the sglang codebase. Hooks are registered
during plugin loading and applied before the engine starts.

Usage:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def my_timer(original_fn, *args, **kwargs):
        start = time.perf_counter()
        result = original_fn(*args, **kwargs)
        print(f"Elapsed: {time.perf_counter() - start:.3f}s")
        return result

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.schedule",
        my_timer,
        HookType.AROUND,
    )
"""

import contextvars
import functools
import hashlib
import inspect
import linecache
import logging
import os
import pkgutil
import sys
import tempfile
import textwrap
import types
import __future__ as _future_mod
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


# Bitmask of all `__future__` compiler flags. Used by statement hooks to
# inherit features like `from __future__ import annotations` (PEP 563) when
# recompiling a patched function body, so that annotations referencing
# names only available under TYPE_CHECKING (e.g. `Scheduler`) remain
# strings instead of being eagerly evaluated.
_ALL_FUTURE_FLAGS = 0
for _name in _future_mod.all_feature_names:
    _ALL_FUTURE_FLAGS |= getattr(_future_mod, _name).compiler_flag


class HookSource(NamedTuple):
    """Identifies which plugin registered a hook."""

    plugin_name: str  # entry_point name, e.g. "xpu_hooks"
    dist_name: str | None  # distribution name, e.g. "sglang_xpu_platform"


# Set by load_plugins() around each plugin's func() call, read by register().
_current_plugin_source: contextvars.ContextVar[HookSource | None] = (
    contextvars.ContextVar("_current_plugin_source", default=None)
)


def _format_source(source: HookSource | None) -> str:
    """Format source info for log messages."""
    if source is None:
        return "unknown"
    if source.dist_name:
        return f"plugin={source.plugin_name}, dist={source.dist_name}"
    return f"plugin={source.plugin_name}"


class HookType(Enum):
    """Types of hooks that can be applied to functions or classes."""

    BEFORE = "before"  # Execute before original; can modify args
    AFTER = "after"  # Execute after original; can modify return value
    AROUND = "around"  # Wrap original; full control over execution
    REPLACE = "replace"  # Replace the original function, class, or statement


class ReplaceHook(NamedTuple):
    """Describes a REPLACE hook target — either whole-function or statement-level."""

    hook: Callable | None  # The replacement callable (None for statement replace)
    anchor: str | None  # If set, statement-level replace
    code: str | None  # Replacement source code (for statement replace)
    occurrence: int  # Which occurrence of anchor to replace (0-based)


class HookRegistry:
    """
    Global registry for function/method/class hooks.

    Thread safety: All registration should happen during load_plugins()
    phase (single-threaded). apply_hooks() should be called once before the
    engine starts serving requests.
    """

    _hooks: dict[
        str, list[tuple[HookType, Callable | ReplaceHook, HookSource | None]]
    ] = defaultdict(list)
    _patched: set[str] = set()

    @classmethod
    def register(
        cls,
        target: str,
        hook: Callable | None = None,
        hook_type: HookType = HookType.AFTER,
        *,
        anchor: str | None = None,
        code: str | None = None,
        occurrence: int = 0,
        source: HookSource | None = None,
    ):
        """
        Register a hook on a target function, method, or class.

        Args:
            target: Fully-qualified dotted path to the target.
                    e.g. "sglang.srt.managers.scheduler.Scheduler.schedule"
                    or   "sglang.srt.managers.scheduler.Scheduler" (class)
            hook: The hook callable (function or class). Signature depends on hook_type:
                - BEFORE:    fn(*args, **kwargs) -> (args, kwargs) or None
                - AFTER:     fn(result, *args, **kwargs) -> new_result or None
                - AROUND:    fn(original_fn, *args, **kwargs) -> result
                - REPLACE:   fn(*args, **kwargs) -> result   (function replacement)
                             MyClass                         (class replacement)
                For statement-level REPLACE (anchor is set), hook can be None.
            hook_type: Type of hook (default: AFTER).
            anchor: For statement-level REPLACE — the source text to locate.
            code: For statement-level REPLACE — the replacement source code.
            occurrence: Zero-based match index when anchor appears multiple times.
            source: Optional source info. If None, auto-read from context var
                set by ``load_plugins()``.

        Raises:
            TypeError: If a class is passed with a hook_type other than REPLACE.
            ValueError: If anchor is set but code is empty.
        """
        if isinstance(hook, type) and hook_type != HookType.REPLACE:
            raise TypeError(
                f"Class {hook.__name__} can only be used with HookType.REPLACE, "
                f"got HookType.{hook_type.name}. "
                f"Use a function for BEFORE/AFTER/AROUND hooks."
            )
        resolved_source = source or _current_plugin_source.get()

        # Statement-level REPLACE (anchor-based)
        if anchor is not None:
            if hook_type != HookType.REPLACE:
                raise ValueError(
                    f"anchor parameter requires HookType.REPLACE, got {hook_type.name}"
                )
            if occurrence < 0:
                raise ValueError("Statement hook occurrence must be non-negative")
            if not code or not code.strip():
                raise ValueError("Statement hook replacement code cannot be empty")
            replace_hook = ReplaceHook(
                hook=None, anchor=anchor, code=code, occurrence=occurrence
            )
            cls._hooks[target].append((HookType.REPLACE, replace_hook, resolved_source))
            logger.debug(
                "Registered statement REPLACE on %s anchor occurrence %d [%s]",
                target,
                occurrence,
                _format_source(resolved_source),
            )
            return

        # Warn on duplicate whole-function REPLACE for the same target
        if hook_type == HookType.REPLACE:
            existing_replace = [
                (h, src)
                for ht, h, src in cls._hooks[target]
                if ht == HookType.REPLACE and not isinstance(h, ReplaceHook)
            ]
            if existing_replace:
                prev, prev_src = existing_replace[-1]
                prev_name = getattr(prev, "__qualname__", None) or repr(prev)
                new_name = getattr(hook, "__qualname__", None) or repr(hook)
                logger.warning(
                    "Multiple REPLACE hooks on '%s': previous (%s [%s]) will be "
                    "overridden by (%s [%s]). The last registered REPLACE takes effect.",
                    target,
                    prev_name,
                    _format_source(prev_src),
                    new_name,
                    _format_source(resolved_source),
                )
        cls._hooks[target].append((hook_type, hook, resolved_source))
        logger.debug(
            "Registered %s hook on %s [%s]",
            hook_type.value,
            target,
            _format_source(resolved_source),
        )

    @classmethod
    def apply_hooks(cls):
        """
        Apply all registered hooks to their target functions/classes.

        This performs the actual monkey-patching. Should be called once after
        all plugins have been loaded and before the engine starts.

        Targets with class REPLACE hooks are applied first, so that
        subsequent method-level hooks (AROUND, BEFORE, AFTER) on child
        attributes resolve against the *replaced* class rather than the
        original.
        """
        sorted_items = sorted(cls._hooks.items(), key=cls._target_sort_key)
        for target, hooks in sorted_items:
            if target in cls._patched:
                continue
            try:
                cls._apply_target(target, hooks)
                cls._patched.add(target)
            except Exception:
                logger.exception("Failed to apply hooks to %s", target)

    @staticmethod
    def _target_sort_key(item):
        """Sort key: class REPLACE targets (tier 0) before all others (tier 1).

        This ensures that when a class is replaced, subsequent method-level
        hooks on ``ClassName.method`` resolve against the replacement class.
        """
        _target, hooks = item
        has_class_replace = any(
            isinstance(h, type) and ht == HookType.REPLACE for ht, h, _ in hooks
        )
        return (0 if has_class_replace else 1, _target)

    @classmethod
    def _apply_target(cls, target: str, hooks: list):
        """Resolve target, build wrapper chain, and replace the original."""
        parts = target.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid target path (need at least module.attr): {target}"
            )

        obj_path, attr_name = parts
        obj = pkgutil.resolve_name(obj_path)

        # Check if the original is a classmethod or staticmethod by
        # inspecting __dict__ before getattr() triggers the descriptor
        # protocol (which would lose the wrapper type for classmethod).
        original = getattr(obj, attr_name)
        is_classmethod = False
        is_staticmethod = False
        if isinstance(obj, type):
            raw_attr = obj.__dict__.get(attr_name)
            if isinstance(raw_attr, classmethod):
                is_classmethod = True
                original = raw_attr.__func__
            elif isinstance(raw_attr, staticmethod):
                is_staticmethod = True
                original = raw_attr.__func__

        # Cross-target conflict detection: if the parent object is a class
        # that was already class-REPLACE'd, and the replacement class defines
        # its own version of this method, a method REPLACE here will silently
        # override the replacement class's implementation.
        if isinstance(obj, type) and obj_path in cls._patched:
            has_method_replace = any(ht == HookType.REPLACE for ht, _, _ in hooks)
            if has_method_replace and attr_name in obj.__dict__:
                replace_sources = [
                    _format_source(src)
                    for ht, _, src in hooks
                    if ht == HookType.REPLACE
                ]
                logger.warning(
                    "Method REPLACE on '%s' will override the class REPLACE's "
                    "own implementation of '%s'. If this is unintended, remove "
                    "the method REPLACE and modify the replacement class "
                    "directly, or use AROUND to wrap it. (from: %s)",
                    target,
                    attr_name,
                    ", ".join(replace_sources),
                )

        # Guard: if the target is a class, only REPLACE is safe. Wrapping a
        # class in a function would break isinstance/issubclass/inheritance.
        if isinstance(original, type):
            bad = [ht for ht, _, _ in hooks if ht != HookType.REPLACE]
            if bad:
                raise TypeError(
                    f"Target '{target}' is a class. Only HookType.REPLACE is "
                    f"allowed for class targets (got {bad[0].value}). "
                    f"To hook a method, use '{target}.<method_name>' instead."
                )

        # Warn about risky hook combinations
        hook_types = [ht for ht, _, _ in hooks]
        around_count = hook_types.count(HookType.AROUND)
        has_replace = HookType.REPLACE in hook_types
        has_others = any(ht != HookType.REPLACE for ht in hook_types)

        if around_count > 1:
            around_sources = [
                _format_source(src) for ht, _, src in hooks if ht == HookType.AROUND
            ]
            logger.warning(
                "Multiple AROUND hooks on '%s' (%d hooks, from: %s). If any AROUND hook "
                "skips calling original_fn, inner hooks will be bypassed.",
                target,
                around_count,
                ", ".join(around_sources),
            )
        if has_replace and has_others:
            logger.info(
                "Target '%s' has both REPLACE and %s hooks. "
                "REPLACE will be applied first, then wrapped by other hooks.",
                target,
                ", ".join(
                    sorted({ht.value for ht in hook_types if ht != HookType.REPLACE})
                ),
            )

        # Build the wrapper chain.
        # Sort: REPLACE hooks first (whole-function, then statement-level), then wrappers
        # (stable sort preserves registration order within the same type).
        # This ensures wrappers observe the source-injected implementation.
        def _hook_sort_key(h):
            ht, hook_obj, _ = h
            if ht == HookType.REPLACE:
                if isinstance(hook_obj, ReplaceHook) and hook_obj.anchor:
                    return 1  # statement-level REPLACE after whole-function REPLACE
                return 0  # whole-function/class REPLACE first
            return 2  # BEFORE/AFTER/AROUND last

        sorted_hooks = sorted(hooks, key=_hook_sort_key)
        wrapped = original
        for hook_type, hook, _src in sorted_hooks:
            if hook_type == HookType.REPLACE:
                if isinstance(hook, ReplaceHook) and hook.anchor:
                    # Statement-level REPLACE
                    wrapped = _wrap_statement(wrapped, hook)
                elif isinstance(hook, type):
                    # Class replacement
                    wrapped = hook
                else:
                    # Whole-function REPLACE
                    wrapped = _wrap_fn(wrapped, hook, hook_type)
            else:
                wrapped = _wrap_fn(wrapped, hook, hook_type)

        # Restore classmethod/staticmethod decorator if the original had one.
        if is_classmethod:
            wrapped = classmethod(wrapped)
            logger.debug("Preserved @classmethod decorator for %s", target)
        elif is_staticmethod:
            wrapped = staticmethod(wrapped)
            logger.debug("Preserved @staticmethod decorator for %s", target)

        setattr(obj, attr_name, wrapped)

        # Propagate the patch to all other modules that imported the original
        # via ``from source_module import name``.  Python's ``from X import Y``
        # copies the reference at import time; patching X alone leaves
        # importers with a stale binding.
        if wrapped is not original:
            extra = _propagate_patch(original, wrapped, obj)
            if extra:
                logger.debug(
                    "Propagated patch for %s to %d additional module(s)",
                    target,
                    extra,
                )

        sources = sorted({_format_source(src) for _, _, src in hooks})
        logger.info(
            "Applied %d hook(s) to %s (from: %s)",
            len(hooks),
            target,
            ", ".join(sources),
        )

    @classmethod
    def reset(cls):
        """Reset all hooks and patches. Primarily for testing."""
        cls._hooks.clear()
        cls._patched.clear()


def _get_function_body_source(fn: Callable) -> str:
    """Return a decorated function's body source for statement injection."""
    source = textwrap.dedent(inspect.getsource(fn))
    lines = source.splitlines()
    def_index = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("def ")),
        None,
    )
    if def_index is None:
        raise ValueError(f"Cannot find function body for statement hook {fn!r}")
    body = "\n".join(lines[def_index + 1 :])
    return textwrap.dedent(body).strip("\n")


# Flow-control placeholders that allow writing continue/break/return in
# decorator-style statement hooks without triggering SyntaxError at compile time.
_FLOW_PLACEHOLDERS = {
    "__CONTINUE__": "continue",
    "__BREAK__": "break",
}


def _replace_flow_placeholders(code: str) -> str:
    """Replace flow-control placeholders with real statements."""
    for placeholder, replacement in _FLOW_PLACEHOLDERS.items():
        code = code.replace(placeholder, replacement)
    return code


def _indent_source(code: str, indent: str) -> list[str]:
    """Indent each line of code with the given prefix, return as list of lines."""
    dedented = textwrap.dedent(code).strip("\n")
    return [indent + line if line.strip() else line for line in dedented.splitlines()]


# Statement-hook recompiled sources are materialized here so that runtime
# exceptions carry a real filename + correct line numbers (and tracebacks can
# render the *actual executed* source instead of the upstream file's mismatched
# lines).
_HOOK_SOURCE_DIR = os.path.join(tempfile.gettempdir(), "sglang_statement_hooks")


def _materialize_hook_source(qualname: str, source: str) -> str:
    """Write the combined (anchor + injected) source to a temp file and register it
    with ``linecache``, returning the filename to use as the code object's
    ``co_filename``.

    Why: statement hooks recompile a patched function body via ``compile()``. If we
    keep the upstream file as ``co_filename`` (the old behavior), a runtime error
    inside the injected code points at the wrong lines of the real file and the
    traceback shows misleading source. Compiling against a real on-disk file makes
    tracebacks/pdb/py-spy show the exact executed code at the right line.

    Falls back to a synthetic ``<statement-hook:...>`` name if the file cannot be
    written (e.g. read-only fs); linecache registration still enables source
    rendering in that case.
    """
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in qualname) or "hook"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    filename = os.path.join(_HOOK_SOURCE_DIR, f"{safe}.{digest}.py")
    try:
        os.makedirs(_HOOK_SOURCE_DIR, exist_ok=True)
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(source)
    except OSError:
        filename = f"<statement-hook:{safe}>"
    # Register with linecache so traceback/inspect can render the source even if the
    # file is missing or outside sys.path. (size, mtime=None, lines, fullname)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    return filename


def _wrap_statement(original_fn: Callable, hook: ReplaceHook) -> Callable:
    """Wrap source code around a stable statement anchor."""
    # If original_fn was itself produced by a prior STATEMENT hook on the same
    # target, reuse its stashed (already-patched) source and the upstream
    # real_fn it was compiled against. This lets multiple STATEMENT hooks
    # compose, instead of inspect.unwrap() walking back to the upstream
    # source and clobbering earlier hook modifications.
    prior_source = getattr(original_fn, "__statement_hook_source__", None)
    prior_real_fn = getattr(original_fn, "__statement_hook_real_fn__", None)
    if prior_source is not None and prior_real_fn is not None:
        real_fn = prior_real_fn
        source = prior_source
    else:
        # Unwrap decorators (e.g. @torch.no_grad()) so we operate on the real
        # user-defined function: its __globals__, __code__.co_flags (for future
        # flags such as `from __future__ import annotations`), and source file.
        real_fn = inspect.unwrap(original_fn)
        try:
            source = inspect.getsource(real_fn)
        except (OSError, TypeError) as exc:
            raise ValueError(
                f"Cannot inspect source for statement hook target {original_fn!r}"
            ) from exc

    lines = textwrap.dedent(source).splitlines()
    anchor = textwrap.dedent(hook.anchor).strip()
    if not anchor:
        raise ValueError("Statement hook anchor cannot be empty")

    anchor_lines = textwrap.dedent(hook.anchor).strip("\n").splitlines()
    anchor_lines = [line.rstrip() for line in anchor_lines]
    matches = _find_statement_anchor_matches(lines, anchor_lines)
    if hook.occurrence >= len(matches):
        raise ValueError(
            f"Statement hook anchor occurrence {hook.occurrence} is out of range "
            f"in {real_fn.__qualname__}; found {len(matches)} matches: {anchor!r}"
        )

    match_index = matches[hook.occurrence]
    anchor_line = lines[match_index]
    anchor_indent = anchor_line[: len(anchor_line) - len(anchor_line.lstrip())]
    replacement_lines = _indent_source(
        _replace_flow_placeholders(hook.code), anchor_indent
    )
    patched_lines = (
        lines[:match_index]
        + replacement_lines
        + lines[match_index + len(anchor_lines) :]
    )
    patched_source = "\n".join(patched_lines) + "\n"

    global_ns = real_fn.__globals__
    local_ns: dict[str, object] = {}
    # Inherit future flags (e.g. PEP 563 `annotations`) from the real function
    # so type annotations remain strings rather than being eagerly evaluated.
    future_flags = real_fn.__code__.co_flags & _ALL_FUTURE_FLAGS
    # Materialize the patched source to a real file so runtime errors from the
    # injected code produce a precise, readable traceback (filename + line + source)
    # instead of pointing at mismatched lines in the upstream file.
    hook_filename = _materialize_hook_source(real_fn.__qualname__, patched_source)
    exec(
        compile(
            patched_source,
            hook_filename,
            "exec",
            flags=future_flags,
            dont_inherit=True,
        ),
        global_ns,
        local_ns,
    )
    replacement = local_ns.get(real_fn.__name__)
    if not callable(replacement):
        raise ValueError(
            f"Statement hook did not recreate callable {real_fn.__qualname__}"
        )

    functools.update_wrapper(replacement, original_fn)
    replacement.__wrapped__ = original_fn
    # Stash the patched source and the underlying real_fn so a subsequent
    # STATEMENT hook on the same target can compose against this version
    # instead of inspect.unwrap()-ing back to the upstream original.
    replacement.__statement_hook_source__ = patched_source
    replacement.__statement_hook_real_fn__ = real_fn
    return replacement


def _find_statement_anchor_matches(lines: list[str], anchor_lines: list[str]) -> list[int]:
    """Find zero-based line indices whose dedented block matches anchor_lines."""
    matches: list[int] = []
    if not anchor_lines:
        return matches
    anchor_len = len(anchor_lines)
    for index in range(0, len(lines) - anchor_len + 1):
        block = lines[index : index + anchor_len]
        block_indent = block[0][: len(block[0]) - len(block[0].lstrip())]
        normalized_block = [
            line[len(block_indent) :].rstrip()
            if line.startswith(block_indent)
            else line.rstrip()
            for line in block
        ]
        if normalized_block == anchor_lines:
            matches.append(index)
    return matches


def _propagate_patch(original: object, wrapped: object, source_module: object) -> int:
    """Propagate a monkey-patch to all modules holding a stale ``from X import Y`` binding.

    After ``setattr(source_module, name, wrapped)`` updates the defining module,
    other modules that did ``from source_module import name`` still hold a direct
    reference to the old *original* object.  This walks ``sys.modules`` and
    replaces every such stale binding with *wrapped*.

    Returns the number of additional module attributes that were patched.
    """
    patched_count = 0
    for mod in list(sys.modules.values()):
        if mod is source_module or mod is None:
            continue
        if not isinstance(mod, types.ModuleType):
            continue
        try:
            mod_vars = vars(mod)
        except TypeError:
            continue
        for attr_name, attr_value in list(mod_vars.items()):
            if attr_value is original:
                try:
                    setattr(mod, attr_name, wrapped)
                    patched_count += 1
                except (AttributeError, TypeError):
                    pass
    return patched_count


def _wrap_fn(original_fn: Callable, hook: Callable, hook_type: HookType) -> Callable:
    """Create a wrapper function based on the hook type."""
    if hook_type == HookType.REPLACE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.BEFORE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = hook(*args, **kwargs)
            if result is not None:
                args, kwargs = result
            return original_fn(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AFTER:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            modified = hook(result, *args, **kwargs)
            return modified if modified is not None else result

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AROUND:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook(original_fn, *args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    else:
        raise ValueError(f"Unknown hook type: {hook_type}")


def plugin_hook(
    target: str,
    type: HookType = HookType.AFTER,
    *,
    anchor: str | None = None,
    occurrence: int = 0,
) -> Callable:
    """Decorator that registers a function/class hook, or statement replacement.

    Usage::

        # Function hook (AROUND)
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler.schedule",
                      type=HookType.AROUND)
        def my_timer(original_fn, *args, **kwargs):
            start = time.perf_counter()
            result = original_fn(*args, **kwargs)
            print(f"Elapsed: {time.perf_counter() - start:.3f}s")
            return result

        # Statement-level REPLACE (anchor-based)
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler.schedule",
                      anchor="result = self.forward_batch(...)")
        def wrap_forward():
            print("before forward")
            result = self.forward_batch(...)
            print("after forward")

        # Class replacement (REPLACE)
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler",
                      type=HookType.REPLACE)
        class MyScheduler(Scheduler):
            ...

    The decorated function/class is returned unchanged so it can still be
    used directly if needed.
    """

    def decorator(hook: Callable) -> Callable:
        if anchor is None:
            HookRegistry.register(target, hook, type)
        else:
            HookRegistry.register(
                target,
                hook_type=HookType.REPLACE,
                anchor=anchor,
                code=_get_function_body_source(hook),
                occurrence=occurrence,
            )
        return hook

    return decorator