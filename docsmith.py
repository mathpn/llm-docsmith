import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Literal,
    Protocol,
    runtime_checkable,
)

import click
import libcst as cst
import llm
from pydantic import BaseModel

SYSTEM_PROMPT = """
You are a coding assistant whose task is to generate docstrings for existing Python code.
You will receive code without any docstrings.
Generate the appropiate docstrings for each function, class or method.

Do not return any code. Use the context only to learn about the code.
Write documentation only for the code provided as input code.

The docstring for a function or method should summarize its behavior, side effects, exceptions raised,
and restrictions on when it can be called (all if applicable).
Only mention exceptions if there is at least one _explicitly_ raised or reraised exception inside the function or method.
The docstring prescribes the function or method's effect as a command, not as a description; e.g. don't write "Returns the pathname ...".
Do not explain implementation details, do not include information about arguments and return here.
If the docstring is multiline, the first line should be a very short summary, followed by a blank line and a more ellaborate description.
Write single-line docstrings if the function is simple.
The docstring for a class should summarize its behavior and list the public methods (one by line) and instance variables.

In the Argument object, describe each argument. In the return object, describe the returned values of the function, if any.

You will receive a JSON template. Fill the slots marked with <SLOT> with the appropriate description. Return as JSON.
"""

PROMPT_TEMPLATE = """
{CONTEXT}

Input code:

```python
{CODE}
```

Output template:

```json
{TEMPLATE}
```
"""


INDENT = "    "


class Argument(BaseModel):
    name: str
    description: str
    annotation: str | None = None
    default: str | None = None


class Return(BaseModel):
    description: str
    annotation: str | None


class Docstring(BaseModel):
    node_type: Literal["class", "function"]
    name: str
    docstring: str
    args: list[Argument] | None = None
    ret: Return | None = None


class Documentation(BaseModel):
    entries: list[Docstring]


class DocstringGenerator(Protocol):
    def __call__(
        self, input_code: str, context: str, template: Documentation
    ) -> Documentation: ...


def create_docstring_node(docstring_text: str, indent: str) -> cst.BaseStatement:
    lines = docstring_text.strip().split("\n")

    indented_lines = []
    for line in lines:
        indented_lines.append(indent + line if line.strip() else line)

    return cst.SimpleStatementLine(
        body=[
            cst.Expr(
                value=cst.SimpleString(
                    value=f'"""\n{"\n".join(indented_lines)}\n{indent}"""'
                )
            )
        ]
    )


@dataclass
class ChangedEntities:
    functions: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)


def has_docstring(node: cst.CSTNode) -> bool:
    """
    Check if a node has a docstring.

    A docstring is the first statement in a module, function or class body and must be a string literal.
    The node can have different types of bodies (IndentedBlock or SimpleStatementSuite) depending on
    whether it's a compound statement or a simple one-liner.
    """
    # Handle simple one-liner functions/classes that use SimpleStatementSuite
    if isinstance(node.body, cst.SimpleStatementSuite):
        return False  # One-liners can't have docstrings

    # Handle regular functions/classes with IndentedBlock
    if isinstance(node.body, cst.IndentedBlock):
        body_statements = node.body.body
    else:
        body_statements = node.body

    if not body_statements:
        return False

    first_stmt = body_statements[0]
    if not isinstance(first_stmt, cst.SimpleStatementLine):
        return False

    if not first_stmt.body:
        return False

    first_expr = first_stmt.body[0]
    if not isinstance(first_expr, cst.Expr):
        return False

    return isinstance(first_expr.value, (cst.SimpleString, cst.ConcatenatedString))


class DocstringTransformer(cst.CSTTransformer):
    def __init__(
        self,
        docstring_generator: DocstringGenerator,
        module: cst.Module,
        changed_entities: ChangedEntities | None = None,
        only_missing: bool = False,
    ):
        self._class_stack: list[str] = []
        self._doc: Documentation | None = None
        self.module: cst.Module = module
        self.docstring_gen = docstring_generator
        self.indentation_level = 0
        self.changed_entities = changed_entities
        self.only_missing = only_missing

    @property
    def _current_class(self) -> str | None:
        """Get the current class name from the top of the stack."""
        return self._class_stack[-1] if self._class_stack else None

    def visit_Module(self, node):
        self.module = node
        return True

    def visit_FunctionDef(self, node):
        self.indentation_level += 1

    def visit_ClassDef(self, node) -> bool | None:
        self.indentation_level += 1
        self._class_stack.append(node.name.value)

        if (
            self.changed_entities is None
            or node.name.value in self.changed_entities.classes
        ):
            source_lines = cst.Module([node]).code
            template = extract_signatures(self.module, node)
            context = get_context(self.module, node)
            doc = self.docstring_gen(source_lines, context, template)
            self._doc = doc

        return super().visit_ClassDef(node)

    def _modify_docstring(self, body, new_docstring):
        # If body is an IndentedBlock, extract its body
        if isinstance(body, cst.IndentedBlock):
            body_statements = list(body.body)
        elif not isinstance(body, list):
            # Create an IndentedBlock if body is not already one
            indent = INDENT * (self.indentation_level + 1)
            new_docstring_node = create_docstring_node(new_docstring, indent)
            return cst.IndentedBlock(body=[new_docstring_node, body])
        else:
            body_statements = list(body)

        indent = INDENT * (self.indentation_level + 1)
        # Check if first statement is a docstring
        if (
            body_statements
            and isinstance(body_statements[0], cst.SimpleStatementLine)
            and isinstance(body_statements[0].body[0], cst.Expr)
            and isinstance(body_statements[0].body[0].value, cst.SimpleString)
        ):
            # Replace existing docstring
            new_docstring_node = create_docstring_node(new_docstring, indent)
            body_statements[0] = new_docstring_node

        # No existing docstring - add new one if provided
        elif new_docstring:
            new_docstring_node = create_docstring_node(new_docstring, indent)
            body_statements.insert(0, new_docstring_node)

        # Reconstruct the body
        if isinstance(body, cst.IndentedBlock):
            return body.with_changes(body=tuple(body_statements))
        return tuple(body_statements)

    def leave_FunctionDef(self, original_node, updated_node):
        self.indentation_level -= 1

        if self.changed_entities is not None:
            if (
                self._current_class
                and f"{self._current_class}.{updated_node.name.value}"
                not in self.changed_entities.methods
            ):
                return updated_node
            if (
                not self._current_class
                and updated_node.name.value not in self.changed_entities.functions
            ):
                return updated_node

        if self.only_missing and has_docstring(updated_node):
            return updated_node

        source_lines = cst.Module([updated_node]).code
        name = updated_node.name.value

        doc = None
        if self._current_class is None:
            template = extract_signatures(self.module, updated_node)
            context = get_context(self.module, updated_node)
            doc = self.docstring_gen(source_lines, context, template)
        elif self._doc is not None:
            doc = self._doc
        else:
            return updated_node

        new_docstring = find_docstring_by_name(doc, name)
        if new_docstring is None:
            return updated_node

        new_body = self._modify_docstring(
            updated_node.body, docstring_to_str(new_docstring)
        )

        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(self, original_node, updated_node):
        self.indentation_level -= 1
        self._class_stack.pop()

        if (
            self.changed_entities is not None
            and updated_node.name.value not in self.changed_entities.classes
        ):
            return updated_node

        if self.only_missing and has_docstring(updated_node):
            return updated_node

        if self._doc is None:
            return updated_node

        new_docstring = find_docstring_by_name(self._doc, updated_node.name.value)

        if new_docstring is None:
            return updated_node

        new_body = self._modify_docstring(
            updated_node.body, docstring_to_str(new_docstring)
        )

        return updated_node.with_changes(body=new_body)


def find_function_definitions(tree) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    function_defs = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_defs.append(node)

    return function_defs


def find_class_definitions(tree) -> list[ast.ClassDef]:
    function_defs = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            function_defs.append(node)

    return function_defs


def find_top_level_definitions(
    tree,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
    return definitions


def collect_entities(
    node,
    definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef],
) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    entities = set()

    for node in ast.walk(node):
        match node:
            case ast.Call(func=ast.Name(name)):
                entities.add(definitions.get(name))
            case (
                ast.AnnAssign(annotation=ast.Name(name))
                | ast.arg(annotation=ast.Name(name))
            ):
                entities.add(definitions.get(name))
            case (
                ast.AnnAssign(
                    annotation=ast.Subscript(
                        value=ast.Name(subs_name), slice=ast.Name(name)
                    )
                )
                | ast.arg(
                    annotation=ast.Subscript(
                        value=ast.Name(subs_name), slice=ast.Name(name)
                    )
                )
            ):
                entities.add(definitions.get(name))
                entities.add(definitions.get(subs_name))

    return list(e for e in entities if e is not None)


def get_context(module: cst.Module, node: cst.CSTNode) -> str:
    source = module.code

    tree = ast.parse(source)
    definitions = find_top_level_definitions(tree)

    node_source = module.code_for_node(node)
    node_tree = ast.parse(node_source)
    referenced_functions = collect_entities(node_tree, definitions)

    out = "\n".join(ast.unparse(func) for func in referenced_functions)
    return out


def has_return_stmt(node):
    return any(
        isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)
    )


def extract_signatures(module: cst.Module, node: cst.CSTNode) -> Documentation:
    source = module.code_for_node(node)

    tree = ast.parse(source)
    function_defs = find_function_definitions(tree)
    # TODO argument
    function_defs = filter(lambda x: not is_private(x), function_defs)
    function_defs = filter(lambda x: not is_dunder(x), function_defs)

    class_defs = find_class_definitions(tree)
    class_defs = filter(lambda x: not is_private(x), class_defs)

    function_entries = [extract_signature(node) for node in function_defs]
    class_entries = [
        Docstring(node_type="class", name=node.name, docstring="<SLOT>")
        for node in class_defs
    ]

    return Documentation(entries=[*class_entries, *function_entries])


def is_private(node):
    name = node.name
    return name.startswith("_") and not is_dunder(node)


def is_dunder(node):
    name = node.name
    return name.startswith("__") and name.endswith("__")


def extract_signature(function_node: ast.FunctionDef | ast.AsyncFunctionDef):
    function_name = function_node.name

    arguments = []
    for arg in function_node.args.args:
        arg_name = arg.arg

        if arg_name in {"self", "cls"}:
            continue

        arg_type = ast.unparse(arg.annotation) if arg.annotation else None

        default_value = None
        if function_node.args.defaults:
            num_defaults = len(function_node.args.defaults)

            # Align defaults with arguments
            default_index = len(function_node.args.args) - num_defaults
            if function_node.args.args.index(arg) >= default_index:
                default_value = ast.unparse(
                    function_node.args.defaults[
                        function_node.args.args.index(arg) - default_index
                    ]
                )

        arguments.append(
            Argument(
                name=arg_name,
                annotation=arg_type,
                default=default_value,
                description="<SLOT>",
            )
        )

    # Handle *args
    if function_node.args.vararg:
        arguments.append(
            Argument(
                name=f"*{function_node.args.vararg.arg}",
                annotation=ast.unparse(function_node.args.vararg.annotation)
                if function_node.args.vararg.annotation
                else None,
                description="<SLOT>",
            )
        )

    # Handle **kwargs
    if function_node.args.kwarg:
        arguments.append(
            Argument(
                name=f"**{function_node.args.kwarg.arg}",
                annotation=ast.unparse(function_node.args.kwarg.annotation)
                if function_node.args.kwarg.annotation
                else None,
                description="<SLOT>",
            )
        )

    # Extract return type
    ret = None
    if has_return_stmt(function_node):
        return_type = (
            ast.unparse(function_node.returns) if function_node.returns else None
        )
        ret = Return(description="<SLOT>", annotation=return_type)

    return Docstring(
        node_type="function",
        name=function_name,
        docstring="<SLOT>",
        args=arguments,
        ret=ret,
    )


def find_docstring_by_name(doc: Documentation, name: str) -> Docstring | None:
    entries = [entry for entry in doc.entries if entry.name == name]
    return entries[0] if entries else None


def wrap_text(
    text: str, indent: str = "", initial_indent: str = "", max_width: int = 88
) -> str:
    """Wrap text to max_width, respecting indentation and breaking only between words."""
    # Split by newlines first to preserve them
    text = text.replace("\\n", "\n")
    paragraphs = text.split("\n")
    result = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            # Empty line, preserve it
            result.append("")
            continue

        lines = []
        current_line = initial_indent

        for word in words:
            # Check if adding this word would exceed max_width
            if (
                len(current_line) + len(word) + 1 <= max_width
                or not current_line.strip()
            ):
                # Add word with a space if not the first word on the line
                if current_line.strip():
                    current_line += " " + word
                else:
                    current_line += word
            else:
                # Start a new line
                lines.append(current_line)
                current_line = indent + word

        # Add the last line if it's not empty
        if current_line:
            lines.append(current_line)

        result.append("\n".join(lines))

    # Join all paragraphs with newlines
    return "\n".join(result)


def docstring_to_str(docstring: Docstring) -> str:
    wrapped_docstring = wrap_text(docstring.docstring.strip())
    string = f"{wrapped_docstring}\n"

    args_strings = []
    for arg in docstring.args or []:
        if arg.annotation is not None:
            prefix = f"    - {arg.name} ({arg.annotation}):"
        else:
            prefix = f"    - {arg.name}:"

        description = arg.description
        if arg.default is not None:
            description += f" (default {arg.default})"

        # Wrap the argument description with proper indentation
        wrapped_arg = wrap_text(
            description.strip(), indent=" " * 6, initial_indent=prefix
        )
        args_strings.append(wrapped_arg)

    if args_strings:
        string += f"""\nParameters:
-----------

{"\n".join(args_strings)}
"""

    # Process return value
    if docstring.ret is not None and (
        docstring.ret.description or docstring.ret.annotation
    ):
        if docstring.ret.annotation:
            prefix = f"    - {docstring.ret.annotation}:"
            description = docstring.ret.description
            indent = " " * 6
        else:
            prefix = "    "
            description = docstring.ret.description
            indent = prefix

        # Wrap the return description with proper indentation
        wrapped_return = wrap_text(description, indent=indent, initial_indent=prefix)

        string += f"""\nReturns:
--------

{wrapped_return}
"""
    return string


def llm_docstring_generator(
    input_code: str, context: str, template: Documentation, model_id: str, verbose: bool
) -> Documentation:
    context = f"Important context:\n\n```python\n{context}\n```" if context else ""
    model = llm.get_model(model_id)
    prompt = PROMPT_TEMPLATE.strip().format(
        CONTEXT=context,
        CODE=input_code,
        TEMPLATE=template.model_dump_json(),
    )

    if verbose:
        click.echo(
            click.style(f"System:\n{SYSTEM_PROMPT}", fg="yellow", bold=True), err=True
        )
        click.echo(click.style(f"Prompt:\n{prompt}", fg="yellow", bold=True), err=True)

    response = model.prompt(
        prompt=prompt, schema=Documentation, system=SYSTEM_PROMPT.strip()
    )
    if verbose:
        click.echo(click.style(response, fg="green", bold=True), err=True)

    return Documentation.model_validate_json(response.text())


def read_source(fpath: str):
    with open(fpath, "r", encoding="utf-8") as f:
        source = f.read()
    return source


def modify_docstring(
    source_code,
    docstring_generator: DocstringGenerator,
    changed_entities: ChangedEntities | None = None,
    only_missing: bool = False,
):
    module = cst.parse_module(source_code)
    modified_module = module.visit(
        DocstringTransformer(
            docstring_generator, module, changed_entities, only_missing
        )
    )
    return modified_module.code


def get_changed_lines(file_path: str, git_base: str = "HEAD"):
    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    result = subprocess.run(
        ["git", "-C", file_dir, "diff", "-U0", git_base, "--", file_path],
        stdout=subprocess.PIPE,
        text=True,
    )

    lines = result.stdout.splitlines()
    line_change_regex = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    modified_lines = []

    for line in lines:
        match = line_change_regex.match(line)
        if match:
            start_line = int(match.group(1))
            num_lines = int(match.group(2) or "1")

            # Collect all affected line numbers
            for i in range(num_lines):
                modified_lines.append(start_line + i)

    return modified_lines


class ParentNodeVisitor(ast.NodeVisitor):
    """
    Custom AST node visitor that tracks parent-child relationships.
    """

    def __init__(self):
        self.parent_map = {}

    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
        super().visit(node)


@runtime_checkable
class ASTNodeWithLines(Protocol):
    lineno: int
    end_lineno: int | None


def get_node_line_range(node: ASTNodeWithLines) -> tuple[int, int]:
    """
    Get the line range (start_line, end_line) for an AST node.

    Args:
        node: The AST node to get the line range for

    Returns:
        A tuple containing the start and end line numbers
    """
    start_line = node.lineno
    end_line = getattr(node, "end_lineno", start_line)
    return start_line, end_line


def is_node_in_lines(node: ast.AST, changed_lines: list[int]) -> bool:
    """
    Check if an AST node has any lines that were changed.

    Args:
        node: The AST node to check
        changed_lines: List of line numbers that were changed

    Returns:
        True if any line in the node was changed, False otherwise
    """
    if isinstance(node, ASTNodeWithLines):
        start_line, end_line = get_node_line_range(node)
        return any(start_line <= line <= end_line for line in changed_lines)
    return False


def get_parent_class(
    node: ast.AST, parent_map: dict[ast.AST, ast.AST]
) -> ast.ClassDef | None:
    """
    Get the parent class of a node if it exists.

    Args:
        node: The AST node to check
        parent_map: Dictionary mapping nodes to their parents

    Returns:
        The parent ClassDef node if the node is a method, None otherwise
    """
    parent = parent_map.get(node)
    if parent and isinstance(parent, ast.ClassDef):
        return parent
    return None


def get_changed_entities(file_path: str, git_base: str = "HEAD") -> ChangedEntities:
    """
    Get a dictionary of changed entities (functions, methods, classes) in a file.

    Args:
        file_path: Path to the Python file
        git_base: Git reference to compare against (default: HEAD)

    Returns:
        ChangedEntities containing sets of changed entity names
    """
    changed_lines = get_changed_lines(file_path, git_base)

    if not changed_lines:
        return ChangedEntities()

    source = read_source(file_path)
    tree = ast.parse(source)

    visitor = ParentNodeVisitor()
    visitor.visit(tree)
    parent_map = visitor.parent_map

    changed_functions = set()
    changed_classes = set()
    changed_methods = set()

    classes_with_changed_methods = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_node_in_lines(node, changed_lines):
                parent_class = get_parent_class(node, parent_map)

                if parent_class:
                    method_name = f"{parent_class.name}.{node.name}"
                    changed_methods.add(method_name)
                    classes_with_changed_methods.add(parent_class.name)
                else:
                    changed_functions.add(node.name)

        elif isinstance(node, ast.ClassDef):
            if is_node_in_lines(node, changed_lines):
                changed_classes.add(node.name)

    changed_classes.update(classes_with_changed_methods)

    return ChangedEntities(
        functions=changed_functions,
        classes=changed_classes,
        methods=changed_methods,
    )


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("file_path")
    @click.option("model_id", "-m", "--model", help="Model to use")
    @click.option(
        "-o",
        "--output",
        help="Only show the modified code, without modifying the file",
        is_flag=True,
    )
    @click.option(
        "-v", "--verbose", help="Verbose output of prompt and response", is_flag=True
    )
    @click.option(
        "--git",
        help="Only update docstrings for functions and classes that have been changed since the last commit",
        is_flag=True,
    )
    @click.option(
        "--git-base",
        help="Git reference to compare against (default: HEAD)",
        default="HEAD",
    )
    @click.option(
        "--only-missing",
        help="Only add docstrings to entities that don't have them",
        is_flag=True,
    )
    def docsmith(file_path, model_id, output, verbose, git, git_base, only_missing):
        """Generate and write docstrings to a Python file.

        Example usage:

            llm docsmith ./scripts/main.py
            llm docsmith ./scripts/main.py --git
            llm docsmith ./scripts/main.py --git --git-base HEAD~1
            llm docsmith ./scripts/main.py --only-missing
        """
        source = read_source(file_path)
        docstring_generator = partial(
            llm_docstring_generator, model_id=model_id, verbose=verbose
        )

        changed_entities = None
        if git:
            changed_entities = get_changed_entities(file_path, git_base)
            if verbose:
                click.echo(f"Changed functions: {changed_entities.functions}")
                click.echo(f"Changed classes: {changed_entities.classes}")
                click.echo(f"Changed methods: {changed_entities.methods}")

        modified_source = modify_docstring(
            source, docstring_generator, changed_entities, only_missing
        )

        if output:
            click.echo(modified_source)
            return

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modified_source)
