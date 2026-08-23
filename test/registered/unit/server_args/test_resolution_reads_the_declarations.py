"""Resolution reads its own decisions, not the record's fields.

`declare_resolution` records a decision in the declaration stash and writes
nothing. The fields keep what the caller passed, so a resolver that reads a
field another resolver may have decided reads the raw input -- silently, and
only on the configurations where that other resolver fires. The whole pipeline
therefore reads through `resolving_view` (or `ServerArgs._resolved()`, which is
the same view spelled as the record's own member), and this pins that there is
nothing left reading a field directly.

Subjects: every function in `arg_groups/` that takes a config, and every
`ServerArgs` handler the dispatcher reaches. Both are derived -- a new hook file
or a new handler is covered the moment it is written. Readers *outside* those
two -- the platform defaults, `ModelConfig`, the spec-algo hook -- are reached by
resolution too and have moved to the view as well, but enumerating them needs
the call-graph derivation `test_resolution_reads_no_bag` owns; this file pins
the two scopes it can derive exactly.
"""

import ast
import dataclasses
import pathlib

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"
_FIELDS = frozenset(field.name for field in dataclasses.fields(ServerArgs))

# Names a config travels under. `args` is included because the platform hooks
# use it; a false positive would be a function taking an argparse Namespace and
# reading an attribute that happens to be a ServerArgs field name, which the
# allowlist below would then have to carry.
_HOLDER_NAMES = frozenset({"server_args", "sa", "args"})


def _holders(fn):
    names = {
        arg.arg
        for arg in list(fn.args.posonlyargs)
        + list(fn.args.args)
        + list(fn.args.kwonlyargs)
        if arg.arg in _HOLDER_NAMES
    }
    for arg in (
        list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)
    ):
        annotation = arg.annotation
        text = (
            annotation.value
            if isinstance(annotation, ast.Constant)
            else (
                annotation.id
                if isinstance(annotation, ast.Name)
                else annotation.attr if isinstance(annotation, ast.Attribute) else None
            )
        )
        if text == "ServerArgs":
            names.add(arg.arg)
    return names


def _field_reads(fn, holders):
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _FIELDS
            and isinstance(node.value, ast.Name)
            and node.value.id in holders
            and isinstance(node.ctx, ast.Load)
        ):
            yield node.lineno, node.attr


def _resolution_handlers():
    """The `ServerArgs` methods the dispatcher reaches, transitively."""
    tree = ast.parse((_SRT / "server_args.py").read_text(encoding="utf-8-sig"))
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_resolution_pipeline" in methods, "the dispatcher was renamed"
    seen, stack = set(), ["_run_resolution_pipeline"]
    while stack:
        name = stack.pop()
        if name in seen or name not in methods:
            continue
        seen.add(name)
        for node in ast.walk(methods[name]):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                stack.append(node.func.attr)
    return {name: methods[name] for name in seen}


class TestResolutionReadsTheDeclarations(CustomTestCase):
    def test_no_hook_reads_a_field_off_the_record(self):
        offenders = []
        files = sorted((_SRT / "arg_groups").glob("*.py"))
        self.assertGreater(len(files), 5, "the hook scan found almost nothing")
        for path in files:
            rel = f"arg_groups/{path.name}"
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                holders = _holders(fn)
                if not holders:
                    continue
                for lineno, field in _field_reads(fn, holders):
                    offenders.append(f"{rel}:{lineno} {fn.name} reads .{field}")
        self.assertEqual(
            offenders,
            [],
            "a resolution hook reads a field off the record; the field holds the "
            "raw input, so this decides from what was typed rather than from "
            "what resolution decided. Read `resolving_view(server_args)`:\n  "
            + "\n  ".join(offenders),
        )

    def test_no_handler_reads_a_field_off_self(self):
        handlers = _resolution_handlers()
        self.assertGreater(
            len(handlers), 50, f"only {len(handlers)} handlers were reached"
        )
        offenders = []
        for name, fn in sorted(handlers.items()):
            for lineno, field in _field_reads(fn, {"self"}):
                offenders.append(f"server_args.py:{lineno} {name} reads self.{field}")
        self.assertEqual(
            offenders,
            [],
            "a resolution handler reads its own field; the field holds the raw "
            "input. Bind `cfg = resolving_view(self)` and read that:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
