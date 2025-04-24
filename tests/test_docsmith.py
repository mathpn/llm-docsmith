import textwrap
from unittest.mock import Mock, patch

import libcst as cst
import pytest

from docsmith import (
    Argument,
    ChangedEntities,
    Docstring,
    DocstringTransformer,
    Documentation,
    Return,
    docstring_to_str,
    extract_signatures,
    find_docstring_by_name,
    get_changed_entities,
    get_changed_lines,
    get_context,
    has_docstring,
    llm_docstring_generator,
    modify_docstring,
    wrap_text,
)


@pytest.fixture
def sample_python_code():
    return textwrap.dedent(
        """
        def greet(name: str, times: int = 1) -> str:
            return "Hello " * times + name


        class Calculator:
            def add(self, a: int, b: int) -> int:
                return a + b

            def subtract(self, a: int, b: int) -> int:
                return a - b
        """
    ).strip()


@pytest.fixture
def sample_docstring():
    return Docstring(
        node_type="function",
        name="greet",
        docstring="Greets someone multiple times.",
        args=[
            Argument(name="name", description="The name to greet", annotation="str"),
            Argument(
                name="times",
                description="Number of times to repeat greeting",
                annotation="int",
                default="1",
            ),
        ],
        ret=Return(description="The greeting message", annotation="str"),
    )


def test_wrap_text_basic():
    text = "This is a long text that should be wrapped at a specific width to ensure readability."
    wrapped = wrap_text(text, max_width=20)
    lines = wrapped.split("\n")
    assert all(len(line) <= 20 for line in lines)
    assert wrapped.replace("\n", " ") == text


def test_wrap_text_with_indentation():
    text = "This is an indented text."
    indent = "    "
    wrapped = wrap_text(text, indent=indent, max_width=20)
    lines = wrapped.split("\n")
    assert all([line.startswith(indent) for line in lines[1:] if line])
    assert all(len(line) <= 20 for line in lines)


def test_wrap_text_with_newlines():
    text = "First paragraph.\n\nSecond paragraph."
    wrapped = wrap_text(text, max_width=20)
    assert len(wrapped.split("\n\n")) == 2


def test_docstring_to_str_basic(sample_docstring):
    result = docstring_to_str(sample_docstring)
    assert "Greets someone multiple times." in result
    assert "Parameters:" in result
    assert "name (str):" in result
    assert "times (int):" in result
    assert "(default 1)" in result
    assert "Returns:" in result
    assert "str:" in result


def test_docstring_to_str_no_args():
    docstring = Docstring(
        node_type="function",
        name="simple_func",
        docstring="A simple function.",
        args=None,
        ret=None,
    )
    result = docstring_to_str(docstring)
    assert "Parameters:" not in result
    assert "Returns:" not in result
    assert result.strip() == "A simple function."


def test_docstring_to_str_with_long_descriptions():
    docstring = Docstring(
        node_type="function",
        name="func",
        docstring="A function with a very long description that should be wrapped properly to maintain readability and formatting.",
        args=[
            Argument(
                name="param",
                description="A parameter with a very long description that should also be wrapped properly to maintain readability.",
                annotation="str",
            )
        ],
        ret=Return(
            description="A return value with a very long description that should be wrapped as well.",
            annotation="str",
        ),
    )
    result = docstring_to_str(docstring)
    assert all(len(line) <= 88 for line in result.split("\n"))


def test_find_docstring_by_name_basic(sample_docstring):
    doc = Documentation(entries=[sample_docstring])
    found = find_docstring_by_name(doc, "greet")
    assert found == sample_docstring


def test_find_docstring_by_name_not_found(sample_docstring):
    doc = Documentation(entries=[sample_docstring])
    found = find_docstring_by_name(doc, "nonexistent")
    assert found is None


def test_find_docstring_by_name_multiple_entries():
    docstrings = [
        Docstring(node_type="function", name="func1", docstring="First function"),
        Docstring(node_type="function", name="func2", docstring="Second function"),
        Docstring(
            node_type="function", name="func1", docstring="Another first function"
        ),
    ]
    doc = Documentation(entries=docstrings)
    found = find_docstring_by_name(doc, "func1")
    assert found == docstrings[0]


def test_extract_signatures_basic(sample_python_code):
    import libcst as cst

    module = cst.parse_module(sample_python_code)
    doc = extract_signatures(module, module)

    assert len(doc.entries) == 4

    # Check class
    class_entry = next(e for e in doc.entries if e.name == "Calculator")
    assert class_entry.node_type == "class"
    assert class_entry.docstring == "<SLOT>"

    # Check standalone function
    greet_entry = next(e for e in doc.entries if e.name == "greet")
    assert greet_entry.node_type == "function"
    assert greet_entry.args is not None
    assert len(greet_entry.args) == 2
    assert greet_entry.args[0].name == "name"
    assert greet_entry.args[0].annotation == "str"
    assert greet_entry.args[1].name == "times"
    assert greet_entry.args[1].annotation == "int"
    assert greet_entry.args[1].default == "1"
    assert greet_entry.ret is not None
    assert greet_entry.ret.annotation == "str"


def test_extract_signatures_incomplete_type_hints():
    import libcst as cst

    module = cst.parse_module(
        textwrap.dedent(
            """
        def foo(a: int, b, c: str="foo", d=3, e=None) -> bool:
            return False
        """
        )
    )
    doc = extract_signatures(module, module)
    foo_entry = next(e for e in doc.entries if e.name == "foo")
    assert foo_entry.args is not None
    assert foo_entry.args[0].name == "a"
    assert foo_entry.args[0].annotation == "int"
    assert foo_entry.args[1].name == "b"
    assert foo_entry.args[1].annotation is None
    assert foo_entry.args[2].name == "c"
    assert foo_entry.args[2].annotation == "str"
    assert foo_entry.args[2].default == "'foo'"
    assert foo_entry.args[3].name == "d"
    assert foo_entry.args[3].annotation is None
    assert foo_entry.args[3].default == "3"
    assert foo_entry.args[4].name == "e"
    assert foo_entry.args[4].annotation is None
    assert foo_entry.args[4].default == "None"


def test_extract_signatures_with_complex_types():
    code = textwrap.dedent(
        """
        from typing import List, Dict, Optional

        def complex_func(
            items: List[int],
            mapping: Dict[str, Optional[int]],
            *args: str,
            **kwargs: int
        ) -> Optional[List[Dict[str, int]]]:
            return None
        """
    ).strip()
    import libcst as cst

    module = cst.parse_module(code)
    doc = extract_signatures(module, module)
    print(doc)

    func = doc.entries[0]
    assert func.name == "complex_func"
    assert func.args is not None
    assert len(func.args) == 4
    assert func.args[0].annotation == "List[int]"
    assert func.args[1].annotation == "Dict[str, Optional[int]]"
    assert func.args[2].name == "*args"
    assert func.args[2].annotation == "str"
    assert func.args[3].name == "**kwargs"
    assert func.args[3].annotation == "int"
    assert func.ret is not None
    assert func.ret.annotation == "Optional[List[Dict[str, int]]]"


def test_extract_signatures_private_methods():
    code = textwrap.dedent(
        """
        class Test:
            def _private(self):
                pass

            def __dunder__(self):
                pass

            def public(self):
                pass
        """
    ).strip()
    import libcst as cst

    module = cst.parse_module(code)
    doc = extract_signatures(module, module)

    # Should only include public methods
    assert len(doc.entries) == 2  # Test class + public method
    assert all(not entry.name.startswith("_") for entry in doc.entries)


def test_has_docstring_regular_function():
    """Test a regular function with a docstring."""
    code = '''
def foo():
    """This is a docstring."""
    pass
'''
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is True


def test_has_docstring_no_docstring():
    """Test a function without a docstring."""
    code = """
def foo():
    pass
"""
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is False


def test_has_docstring_empty_function():
    """Test an empty function."""
    code = """
def foo():
    pass
"""
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is False


def test_has_docstring_oneliner():
    """Test a one-liner function."""
    code = "def foo(): return 42"
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is False


def test_has_docstring_class():
    """Test a class with a docstring."""
    code = '''
class Foo:
    """This is a class docstring."""
    pass
'''
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is True


def test_has_docstring_class_no_docstring():
    """Test a class without a docstring."""
    code = """
class Foo:
    pass
"""
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is False


def test_has_docstring_concatenated_string():
    """Test a function with a concatenated string docstring."""
    code = '''
def foo():
    """This is a """    """concatenated docstring."""
    pass
'''
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is True


def test_has_docstring_first_statement_not_string():
    """Test a function where first statement is not a string."""
    code = '''
def foo():
    x = 42
    """This is not a docstring."""
    pass
'''
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is False


def test_has_docstring_empty_class():
    """Test an empty class."""
    code = """
class Foo:
    pass
"""
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is False


def test_has_docstring_oneliner_class():
    """Test a one-liner class."""
    code = "class Foo: pass"
    node = cst.parse_module(code).body[0]
    assert has_docstring(node) is False


def test_docstring_transformer_without_changed_entities():
    source = textwrap.dedent("""
    def greet():
        return "hello"

    class MyClass:
        def my_method(self):
            pass
    """)

    def mock_generator(input_code, context, template):
        return Documentation(
            entries=[
                Docstring(
                    node_type="function", name="greet", docstring="A test function."
                ),
                Docstring(node_type="class", name="MyClass", docstring="A test class."),
                Docstring(
                    node_type="function", name="my_method", docstring="A test method."
                ),
            ]
        )

    module = cst.parse_module(source)
    transformer = DocstringTransformer(mock_generator, module)
    modified = module.visit(transformer)

    assert "A test function." in modified.code
    assert "A test class." in modified.code
    assert "A test method." in modified.code


def test_docstring_transformer_with_changed_entities():
    source = textwrap.dedent("""
    def changed_function():
        return True

    def unchanged_function():
        return False

    class ChangedClass:
        def changed_method(self):
            pass

        def unchanged_method(self):
            pass

    class UnchangedClass:
        def another_method(self):
            pass
    """)

    def mock_generator(input_code, context, template):
        return Documentation(
            entries=[
                Docstring(
                    node_type="function",
                    name="changed_function",
                    docstring="A changed function.",
                ),
                Docstring(
                    node_type="function",
                    name="unchanged_function",
                    docstring="An unchanged function.",
                ),
                Docstring(
                    node_type="class", name="ChangedClass", docstring="A changed class."
                ),
                Docstring(
                    node_type="function",
                    name="changed_method",
                    docstring="A changed method.",
                ),
                Docstring(
                    node_type="function",
                    name="unchanged_method",
                    docstring="An unchanged method.",
                ),
            ]
        )

    changed_entities = ChangedEntities(
        functions={"changed_function"},
        classes={"ChangedClass"},
        methods={"ChangedClass.changed_method"},
    )

    module = cst.parse_module(source)
    transformer = DocstringTransformer(mock_generator, module, changed_entities)
    modified = module.code_for_node(module.visit(transformer))

    assert "A changed function." in modified
    assert "A changed class." in modified
    assert "A changed method." in modified
    assert "An unchanged function." not in modified
    assert "An unchanged method." not in modified


@patch("subprocess.run")
def test_get_changed_lines_basic(mock_run):
    mock_run.return_value = Mock(
        stdout=textwrap.dedent(
            """
            @@ -1,3 +1,4 @@
            +def new_function():
             def old_function():
                 pass
            -    return None
            +    return True
            """
        ).strip()
    )

    lines = get_changed_lines("test.py")
    assert lines == [1, 2, 3, 4]
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_get_changed_lines_with_git_base(mock_run):
    mock_run.return_value = Mock(stdout="@@ -1 +1 @@\n-old\n+new\n")
    get_changed_lines("test.py", git_base="main")
    assert mock_run.call_args[0][0][5] == "main"


@patch("subprocess.run")
def test_get_changed_lines_no_changes(mock_run):
    mock_run.return_value = Mock(stdout="")
    lines = get_changed_lines("test.py")
    assert lines == []


@patch("subprocess.run")
def test_get_changed_entities_basic(mock_run, sample_python_code):
    mock_run.return_value = Mock(
        stdout=textwrap.dedent(
            """
            @@ -7,2 +7 @@ class Calculator:
            -        add = a + b
            -        return add
            +        return a + b
            """
        ).strip()
    )

    with patch("docsmith.read_source", return_value=sample_python_code):
        entities = get_changed_entities("test.py")

        assert "Calculator" in entities.classes
        assert "Calculator.add" in entities.methods
        assert "Calculator.subtract" not in entities.methods
        assert not entities.functions


@patch("subprocess.run")
def test_get_changed_entities_multiple_changes(mock_run):
    code = textwrap.dedent(
        """
        def func1():
            print("changed")
            return True


        class TestClass:
            def method1(self):
                print("changed")

            def method2(self):
                pass


        def func2():
            pass
        """
    ).strip()

    mock_run.return_value = Mock(
        stdout=textwrap.dedent(
            """
            @@ -2 +2,2 @@ def func1():
            -    pass
            +    print("changed")
            +    return True
            @@ -7 +8 @@ class TestClass:
            -        pass
            +        print("changed")
            """
        ).strip()
    )

    with patch("docsmith.read_source", return_value=code):
        entities = get_changed_entities("test.py")

        assert "func1" in entities.functions
        assert "TestClass" in entities.classes
        assert "TestClass.method1" in entities.methods
        assert "func2" not in entities.functions
        assert "TestClass.method2" not in entities.methods


@patch("llm.get_model")
def test_llm_docstring_generator(mock_get_model):
    mock_model = Mock()
    mock_model.prompt.return_value = Mock(
        text=lambda: '{"entries": [{"node_type": "function", "name": "test", "docstring": "Test function"}]}'
    )
    mock_get_model.return_value = mock_model

    result = llm_docstring_generator(
        "def test(): pass",
        "",
        Documentation(entries=[]),
        model_id="test-model",
        verbose=False,
    )

    assert len(result.entries) == 1
    assert result.entries[0].name == "test"
    assert result.entries[0].docstring == "Test function"


def test_modify_docstring_basic():
    code = "def test(): pass"
    mock_generator = Mock(
        return_value=Documentation(
            entries=[
                Docstring(
                    node_type="function",
                    name="test",
                    docstring="Test function",
                    args=None,
                    ret=None,
                )
            ]
        )
    )

    result = modify_docstring(code, mock_generator)
    print(result)
    assert "Test function" in result
    assert "def test():" in result


def test_modify_docstring_with_existing_docstring():
    code = textwrap.dedent(
        '''
        def test():
            """Old docstring."""
            pass
        '''
    ).strip()
    mock_generator = Mock(
        return_value=Documentation(
            entries=[
                Docstring(
                    node_type="function",
                    name="test",
                    docstring="New docstring",
                    args=None,
                    ret=None,
                )
            ]
        )
    )

    result = modify_docstring(code, mock_generator)
    assert "New docstring" in result
    assert "Old docstring." not in result


def test_modify_docstring_with_changed_entities():
    code = textwrap.dedent(
        """
        def func1():
            pass

        def func2():
            pass
        """
    ).strip()
    mock_generator = Mock(
        return_value=Documentation(
            entries=[
                Docstring(
                    node_type="function",
                    name="func1",
                    docstring="Updated func1",
                    args=None,
                    ret=None,
                ),
                Docstring(
                    node_type="function",
                    name="func2",
                    docstring="Updated func2",
                    args=None,
                    ret=None,
                ),
            ]
        )
    )

    changed_entities = ChangedEntities(functions={"func1"})
    result = modify_docstring(code, mock_generator, changed_entities)

    assert "Updated func1" in result
    assert "Updated func2" not in result


@pytest.mark.parametrize(
    "code,expected_context",
    [
        (
            textwrap.dedent(
                """
                def helper(x: int) -> str:
                    return str(x)

                def main(y: int) -> str:
                    return helper(y)
                """
            ).strip(),
            "def helper(x: int) -> str:\n    return str(x)",
        ),
        (
            textwrap.dedent(
                """
                def unused() -> None:
                    pass

                def main() -> None:
                    pass
                """
            ).strip(),
            "",
        ),
    ],
)
def test_get_context_function_references(code, expected_context):
    import libcst as cst

    module = cst.parse_module(code)
    main_func = next(
        node
        for node in module.body
        if isinstance(node, cst.FunctionDef) and node.name.value == "main"
    )
    context = get_context(module, main_func)
    assert context.strip() == expected_context.strip()
