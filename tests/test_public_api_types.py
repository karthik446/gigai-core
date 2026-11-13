"""The public API of every core module is fully and precisely typed, documented, and checked strict by pyright.

An AST walk over every ``.py`` under ``src/gigai_core`` except ``_vendor``. "Public" is a top-level name without a
leading underscore (also inside a module-level ``if``/``try``/``with``), a method or nested class of a public class
whose name is public or a dunder, and a public field annotation of a public class. Each rule has a self-test: one
snippet it must flag and one it must accept.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import REPO, SRC

VENDOR = SRC / "_vendor"
#: Names that are a container or callable type only once parametrised: the builtins and their ``typing`` and
#: ``collections.abc`` spellings (``typing.Dict``, ``Mapping``, ``collections.abc.Sequence`` ...).
BARE_GENERICS = frozenset(
    {
        "dict", "list", "set", "frozenset", "tuple", "type",
        "Dict", "List", "Set", "FrozenSet", "Tuple", "Type", "Deque", "DefaultDict", "OrderedDict", "Counter", "ChainMap",
        "Mapping", "MutableMapping", "Sequence", "MutableSequence", "AbstractSet", "MutableSet", "Collection", "Container",
        "Iterable", "Iterator", "Generator", "AsyncIterable", "AsyncIterator", "AsyncGenerator", "Awaitable", "Coroutine",
        "Callable", "KeysView", "ItemsView", "ValuesView", "Pattern", "Match",
    }
)  # fmt: skip
#: The only ignore comments allowed outside ``_vendor``: three private names of the vendored inspect_ai parser.
ALLOWED_IGNORES = {
    "src/gigai_core/graders/lifted.py": (
        "_GRADE_WORD_VALUES,  # pyright: ignore[reportPrivateUsage]",
        "_PERMISSIVE_GRADE_PATTERN,  # pyright: ignore[reportPrivateUsage]",
        "_trailing_verdict_disagrees,  # pyright: ignore[reportPrivateUsage]",
    )
}
#: ``# pyright: ignore``/``# type: ignore`` with or without a rule, a file-level ``# pyright: basic``/``standard``,
#: and a file-level rule override ``# pyright: reportX=false``.
IGNORE_COMMENT = re.compile(r"#\s*(pyright:\s*(basic|standard|ignore|\w+\s*=)|type:\s*ignore)(\[[^\]]*\])?")
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
BlockNode = ast.If | ast.Try | ast.With


def _core_files() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if VENDOR not in path.parents)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _annotation_tree(annotation: ast.expr) -> ast.expr | None:
    """The annotation itself, or a string annotation parsed once as an expression (``None`` if it does not parse)."""
    if not (isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)):
        return annotation
    try:
        return ast.parse(annotation.value, mode="eval").body
    except SyntaxError:
        return None


def _is_any(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Any"
    return isinstance(node, ast.Attribute) and node.attr == "Any"


def _generic_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name) and node.id in BARE_GENERICS:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in BARE_GENERICS:
        return node.attr
    return None


def _annotation_problems(annotation: ast.expr) -> list[str]:
    """Rule 1 (``Any``, also as ``dict[str, Any]``) and rule 2 (an unparametrised container or callable type)."""
    tree = _annotation_tree(annotation)
    if tree is None:
        return []
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.expr)]
    subscripted = {id(node.value) for node in nodes if isinstance(node, ast.Subscript)}
    problems: list[str] = []
    for node in nodes:
        if _is_any(node):
            problems.append("Any")
        elif (generic := _generic_name(node)) is not None and id(node) not in subscripted:
            problems.append(f"bare {generic}")
    return problems


def _function_problems(function: FunctionNode, *, method: bool, want_docstring: bool) -> list[str]:
    """Rules 1 to 6 on one def: typed and precise parameters and return, no untyped or ``Any`` star arguments, a docstring."""
    problems: list[str] = []
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    if method and positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    for argument in positional + arguments.kwonlyargs:
        if argument.annotation is None:
            problems.append(f"parameter {argument.arg} has no annotation")
        else:
            problems += [f"parameter {argument.arg}: {why}" for why in _annotation_problems(argument.annotation)]
    for star, prefix in ((arguments.vararg, "*"), (arguments.kwarg, "**")):
        if star is None:
            continue
        if star.annotation is None:
            problems.append(f"untyped {prefix}{star.arg}")
        else:
            problems += [f"{prefix}{star.arg}: {why}" for why in _annotation_problems(star.annotation)]
    if function.returns is None:
        problems.append("no return annotation")
    else:
        problems += [f"return: {why}" for why in _annotation_problems(function.returns)]
    overloaded = any(isinstance(d, ast.Name) and d.id == "overload" for d in function.decorator_list)
    if want_docstring and not overloaded and ast.get_docstring(function) is None:
        problems.append("no docstring")
    return problems


def _import_problems(node: ast.ImportFrom, module_dotted: str) -> list[str]:
    """Rule 7: no underscore name imported from another core module (``_vendor`` exempt), wherever the import sits."""
    if node.level:
        base = module_dotted.rsplit(".", node.level)[0] if node.level <= module_dotted.count(".") else ""
        module = f"{base}.{node.module}".strip(".") if node.module else base
    else:
        module = node.module or ""
    if not module.startswith("gigai_core") or "_vendor" in module.split("."):
        return []
    return [f"imports private {alias.name} from {module}" for alias in node.names if alias.name.startswith("_")]


def _statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements of ``body``, descending into ``if``/``try``/``with`` blocks (a def there is just as public)."""
    for node in body:
        if isinstance(node, BlockNode):
            blocks = [node.body, node.orelse] if isinstance(node, ast.If) else [node.body]
            if isinstance(node, ast.Try):
                blocks += [handler.body for handler in node.handlers] + [node.orelse, node.finalbody]
            for block in blocks:
                yield from _statements(block)
        else:
            yield node


def violations(source: str, relpath: str) -> list[str]:
    """Every rule violation in ``source`` (a module at ``relpath`` under ``src/``), as ``relpath:line: why``."""
    tree = ast.parse(source, filename=relpath)
    module_dotted = relpath.removesuffix(".py").replace("/", ".")
    found: list[str] = []

    def report(node: ast.stmt, problems: list[str]) -> None:
        found.extend(f"{relpath}:{node.lineno}: {why}" for why in problems)

    def check_class(node: ast.ClassDef) -> None:
        if ast.get_docstring(node) is None:
            report(node, ["no docstring"])
        for member in _statements(node.body):
            if isinstance(member, FunctionNode) and (_is_public(member.name) or _is_dunder(member.name)):
                report(member, _function_problems(member, method=True, want_docstring=_is_public(member.name)))
            elif isinstance(member, ast.ClassDef) and _is_public(member.name):
                check_class(member)
            elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name) and _is_public(member.target.id):
                report(member, [f"field {member.target.id}: {why}" for why in _annotation_problems(member.annotation)])

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            report(node, _import_problems(node, module_dotted))
    for node in _statements(tree.body):
        if isinstance(node, FunctionNode) and _is_public(node.name):
            report(node, _function_problems(node, method=False, want_docstring=True))
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            check_class(node)
    return found


def test_every_public_def_field_and_import_of_core_follows_the_rules() -> None:
    found: list[str] = []
    for path in _core_files():
        found += violations(path.read_text(encoding="utf-8"), str(path.relative_to(SRC.parent)))
    assert found == [], "\n" + "\n".join(found)


FLAGGED = {
    "Any": "from typing import Any\ndef f(x: Any) -> int:\n    '''d'''\n    return 1\n",
    "typing.Any": "import typing\ndef f(x: int) -> typing.Any:\n    '''d'''\n    return x\n",
    "dict[str, Any]": "from typing import Any\ndef f(x: dict[str, Any]) -> int:\n    '''d'''\n    return 1\n",
    "bare dict": "def f(x: dict) -> int:\n    '''d'''\n    return 1\n",
    "bare list": "def f() -> list:\n    '''d'''\n    return []\n",
    "bare set": "def f(x: set) -> int:\n    '''d'''\n    return 1\n",
    "bare tuple": "class C:\n    '''d'''\n    field: tuple\n",
    "unparametrised typing.Dict": "import typing\ndef f(x: typing.Dict) -> int:\n    '''d'''\n    return 1\n",
    "unparametrised List": "from typing import List\ndef f() -> List:\n    '''d'''\n    return []\n",
    "unparametrised Mapping": "from collections.abc import Mapping\ndef f(x: Mapping) -> int:\n    '''d'''\n    return 1\n",
    "unparametrised collections.abc.Sequence": "import collections.abc\nclass C:\n    '''d'''\n    field: collections.abc.Sequence\n",
    "unparametrised Callable": "from collections.abc import Callable\ndef f(x: Callable) -> int:\n    '''d'''\n    return 1\n",
    "string annotation hiding Any": "def f(x: 'Any') -> int:\n    '''d'''\n    return 1\n",
    "string annotation hiding a bare list": "def f(x: 'list') -> int:\n    '''d'''\n    return 1\n",
    "**kwargs: Any": "from typing import Any\ndef f(**kwargs: Any) -> int:\n    '''d'''\n    return 1\n",
    "untyped **kwargs": "def f(**kwargs) -> int:\n    '''d'''\n    return 1\n",
    "untyped *args": "def f(*args) -> int:\n    '''d'''\n    return 1\n",
    "parameter without annotation": "def f(x) -> int:\n    '''d'''\n    return 1\n",
    "no return annotation": "def f(x: int):\n    '''d'''\n    return 1\n",
    "function without docstring": "def f(x: int) -> int:\n    return 1\n",
    "class without docstring": "class C:\n    field: int\n",
    "method without docstring": "class C:\n    '''d'''\n    def m(self, x: int) -> int:\n        return x\n",
    "method parameter without annotation": "class C:\n    '''d'''\n    def m(self, x) -> int:\n        '''d'''\n        return 1\n",
    "dunder without a return annotation": "class C:\n    '''d'''\n    def __init__(self, x: int):\n        self.x = x\n",
    "public field typed Any": "from typing import Any\nclass C:\n    '''d'''\n    field: Any\n",
    "public class nested in a public class": "class C:\n    '''d'''\n    class Inner:\n        '''d'''\n        def m(self, x) -> int:\n            '''d'''\n            return 1\n",
    "public def inside a module-level if": "import sys\nif sys.platform == 'darwin':\n    def f(x) -> int:\n        '''d'''\n        return 1\n",
    "public def inside a module-level try": "try:\n    import fast\nexcept ImportError:\n    def f(x: int):\n        '''d'''\n        return 1\n",
    "public class inside a module-level else": "import sys\nif sys.platform == 'win32':\n    pass\nelse:\n    class C:\n        field: int\n",
    "private name from a sibling module": "from .other import _hidden\n",
    "private name from an absolute core module": "from gigai_core.other import _hidden\n",
    "private name imported inside a function body": "def f(x: int) -> int:\n    '''d'''\n    from gigai_core.other import _hidden\n    return _hidden(x)\n",
    "private name imported inside a method body": "class C:\n    '''d'''\n    def m(self) -> int:\n        '''d'''\n        from .other import _hidden\n        return _hidden()\n",
}
ACCEPTED = {
    "typed and documented function": "def f(x: int, *, y: str = '') -> dict[str, int]:\n    '''d'''\n    return {}\n",
    "typed star arguments": "def f(*args: int, **fields: str) -> tuple[int, ...]:\n    '''d'''\n    return ()\n",
    "a parametrised typing or abc generic": "import typing\nfrom collections.abc import Callable, Mapping\ndef f(x: typing.Dict[str, int], g: Callable[[int], str]) -> Mapping[str, int]:\n    '''d'''\n    return x\n",
    "a private function is not held to the rules": "def _f(x, **kwargs):\n    return x\n",
    "a private class is not held to the rules": "class _C:\n    field: dict\n    def m(self, x):\n        return x\n",
    "a private method and a private field are not held to the rules": "class C:\n    '''d'''\n    _hidden: dict\n    def _m(self, x):\n        return x\n",
    "a private nested class is not held to the rules": "class C:\n    '''d'''\n    class _Inner:\n        field: dict\n",
    "a documented class with typed public fields and methods": "from collections.abc import Mapping\nclass C:\n    '''d'''\n    field: Mapping[str, int]\n    def __init__(self, x: int) -> None:\n        self.x = x\n    @property\n    def value(self) -> int:\n        '''d'''\n        return self.x\n",
    "an overload needs no docstring": "from typing import overload\n@overload\ndef f(x: int) -> int: ...\n@overload\ndef f(x: str) -> str: ...\ndef f(x: int | str) -> int | str:\n    '''d'''\n    return x\n",
    "a string annotation that is precise": "def f(x: 'Mapping[str, int]') -> 'C':\n    '''d'''\n    return C()\n",
    "a string annotation of a parametrised builtin": "def f(x: 'list[int]') -> 'dict[str, tuple[int, ...]]':\n    '''d'''\n    return {}\n",
    "a typed def inside a module-level try": "try:\n    import fast\nexcept ImportError:\n    def f(x: int) -> int:\n        '''d'''\n        return 1\n",
    "a public name from a sibling module": "from .other import visible\n",
    "a private name from the vendored tree": "from ._vendor.pkg import _hidden\n",
    "a private name from the vendored tree inside a function": "def f() -> int:\n    '''d'''\n    from gigai_core._vendor.pkg import _hidden\n    return _hidden()\n",
    "a nested def is implementation, not API": "def f(x: int) -> int:\n    '''d'''\n    def inner(y):\n        return y\n    return inner(x)\n",
}


@pytest.mark.parametrize("case", sorted(FLAGGED))
def test_the_rules_flag_each_violation(case: str) -> None:
    assert violations(FLAGGED[case], "gigai_core/example.py") != [], case


@pytest.mark.parametrize("case", sorted(ACCEPTED))
def test_the_rules_accept_each_compliant_shape(case: str) -> None:
    assert violations(ACCEPTED[case], "gigai_core/example.py") == [], case


FLAGGED_COMMENTS = (
    "x = 1  # pyright: ignore",
    "x = 1  # pyright: ignore[reportPrivateUsage]",
    "x = 1  # type: ignore",
    "x = 1  # type: ignore[attr-defined]",
    "# pyright: basic",
    "# pyright: standard",
    "# pyright: reportPrivateUsage=false",
    "# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false",
)
ACCEPTED_COMMENTS = ("x = 1  # the parser's pieces are private upstream", "# pyright: strict")


@pytest.mark.parametrize("line", FLAGGED_COMMENTS)
def test_the_ignore_pattern_sees_each_way_of_switching_pyright_off(line: str) -> None:
    assert IGNORE_COMMENT.search(line), line


@pytest.mark.parametrize("line", ACCEPTED_COMMENTS)
def test_the_ignore_pattern_leaves_plain_comments_and_strict_alone(line: str) -> None:
    assert not IGNORE_COMMENT.search(line), line


def test_pyright_is_strict_over_every_core_file_and_the_only_ignores_are_the_three_documented_ones() -> None:
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pyright"]
    assert config["typeCheckingMode"] == "strict"
    included = [REPO / entry for entry in config["include"]]
    excluded = [REPO / entry for entry in config.get("exclude", [])]
    found: dict[str, list[str]] = {}
    for path in _core_files():
        assert any(root == path or root in path.parents for root in included), f"{path} is not under pyright include"
        assert not any(root == path or root in path.parents for root in excluded), f"{path} is excluded from pyright"
        for line in path.read_text(encoding="utf-8").splitlines():
            if IGNORE_COMMENT.search(line):
                found.setdefault(str(path.relative_to(REPO)), []).append(line.strip())
    assert found == {file: list(lines) for file, lines in ALLOWED_IGNORES.items()}
