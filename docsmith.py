import ast
from functools import partial
from typing import Literal, Protocol, Set, Dict, List, Optional, Tuple

import click
import libcst as cst
import llm
from pydantic import BaseModel
import subprocess
import re

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


def get_changed_functions(
    file_path: str, base_ref: str = "HEAD"
) -> Dict[str, Set[str]]:
    """
    Get a dictionary of changed functions and classes in a file compared to a base reference.

    Args:
        file_path: Path to the Python file
        base_ref: Git reference to compare against (default: HEAD)

    Returns:
        Dictionary with keys 'functions' and 'classes' containing sets of changed names
    """
    # Get absolute path to the file
    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    # Try to find the git repository for the file
    repo_check = subprocess.run(
        ["git", "-C", file_dir, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )

    if repo_check.returncode != 0:
        # Not in a git repository or git not installed
        click.echo(
            click.style(
                f"Warning: File '{file_path}' is not in a git repository or git is not installed. Processing all functions and classes.",
                fg="yellow",
            ),
            err=True,
        )
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        functions = {node.name for node in find_function_definitions(tree)}
        classes = {node.name for node in find_class_definitions(tree)}
        return {"functions": functions, "classes": classes}

    # Get the repository root
    repo_root = repo_check.stdout.strip()

    # Check if file is inside the git repository
    rel_file_path = os.path.relpath(abs_file_path, repo_root)
    if rel_file_path.startswith(".."):
        # File is outside the repository
        click.echo(
            click.style(
                f"Warning: File '{file_path}' is outside any git repository. Processing all functions and classes.",
                fg="yellow",
            ),
            err=True,
        )
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        functions = {node.name for node in find_function_definitions(tree)}
        classes = {node.name for node in find_class_definitions(tree)}
        return {"functions": functions, "classes": classes}

    # Get the diff of the file
    result = subprocess.run(
        ["git", "-C", file_dir, "diff", base_ref, "--", rel_file_path],
        capture_output=True,
        text=True,
        check=True,
    )
    diff = result.stdout
    print(diff)

    # If the file is new, all functions are considered changed
    if not diff:
        # Check if the file is new (not tracked by git)
        result = subprocess.run(
            ["git", "-C", file_dir, "ls-files", rel_file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        if not result.stdout.strip():
            # File is new, get all functions and classes
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            functions = {node.name for node in find_function_definitions(tree)}
            classes = {node.name for node in find_class_definitions(tree)}
            return {"functions": functions, "classes": classes}

    # Parse the diff to find changed functions and classes
    changed_functions = set()
    changed_classes = set()

    # Regular expressions to match function and class definitions
    func_pattern = re.compile(r"^\s[+-]?def\s+([a-zA-Z_][a-zA-Z0-9_]*)")
    class_pattern = re.compile(r"^\s[+-]?class\s+([a-zA-Z_][a-zA-Z0-9_]*)")

    # Track context to identify modified functions
    current_function = None
    current_class = None
    in_function = False
    in_class = False
    in_method = False
    current_method = None

    # Track method signatures to detect changes
    method_signatures = {}

    # Track method bodies to detect changes
    method_bodies = {}

    for line in diff.split("\n"):
        # Check for function definitions
        func_match = func_pattern.match(line)
        if func_match:
            func_name = func_match.group(1)
            if line.startswith("+"):
                # New function added
                changed_functions.add(func_name)
            elif line.startswith("-"):
                # Function removed or modified
                changed_functions.add(func_name)
            current_function = func_name
            in_function = True
            in_class = False
            in_method = False
            current_method = None
            continue

        # Check for class definitions
        class_match = class_pattern.match(line)
        if class_match:
            class_name = class_match.group(1)
            if line.startswith("+"):
                # New class added
                changed_classes.add(class_name)
            elif line.startswith("-"):
                # Class removed or modified
                changed_classes.add(class_name)
            current_class = class_name
            in_class = True
            in_function = False
            in_method = False
            current_method = None
            continue

        # Check for method definitions
        if in_class and current_class and "def " in line:
            method_match = re.match(r"^[+-]\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
            if method_match:
                method_name = method_match.group(1)
                current_method = method_name
                in_method = True

                # Extract the signature part
                sig_start = line.find("(")
                if sig_start > 0:
                    signature = line[sig_start:]
                    key = f"{current_class}.{method_name}"

                    if line.startswith("+"):
                        if (
                            key in method_signatures
                            and method_signatures[key] != signature
                        ):
                            # Method signature changed
                            changed_classes.add(current_class)
                        method_signatures[key] = signature
                    elif line.startswith("-"):
                        # Method removed or modified
                        changed_classes.add(current_class)

        # Check for changes in method bodies
        if in_method and current_method and current_class:
            raise ValueError()
            if line.startswith("+") or line.startswith("-"):
                # Method body changed
                changed_classes.add(current_class)

        # Check for changes within functions or classes
        if line.startswith("+") or line.startswith("-"):
            if in_function and current_function:
                changed_functions.add(current_function)
            elif in_class and current_class and not in_method:
                changed_classes.add(current_class)

        # Check for end of function, class, or method
        if line.startswith(" ") and (in_function or in_class or in_method):
            # If we see a line that's not indented, we're out of the function/class/method
            if not line.startswith("    "):
                in_function = False
                in_class = False
                in_method = False
                current_method = None

    return {"functions": changed_functions, "classes": changed_classes}
    # except subprocess.CalledProcessError:
    #     raise
    #     # If git command fails, return empty sets
    #     return {"functions": set(), "classes": set()}


class DocstringTransformer(cst.CSTTransformer):
    def __init__(self, docstring_generator: DocstringGenerator, module: cst.Module, changed_entities: Optional[Dict[str, Set[str]]] = None):
        self._current_class: str | None = None
        self._doc: Documentation | None = None
        self.module: cst.Module = module
        self.docstring_gen = docstring_generator
        self.indentation_level = 0
        self.changed_entities = changed_entities or {"functions": set(), "classes": set()}

    def visit_Module(self, node):
        self.module = node
        return True

    def visit_FunctionDef(self, node):
        self.indentation_level += 1

    def visit_ClassDef(self, node) -> bool | None:
        self.indentation_level += 1
        self._current_class = node.name.value

        if self.changed_entities and node.name.value in self.changed_entities["classes"]:
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
            return body
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
        
        # Skip if function is not in changed entities
        if self.changed_entities and updated_node.name.value not in self.changed_entities["functions"]:
            return updated_node
            
        source_lines = cst.Module([updated_node]).code

        name = updated_node.name.value
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
        self._current_class = None

        # Skip if class is not in changed entities
        if self.changed_entities and updated_node.name.value not in self.changed_entities["classes"]:
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
            # TODO double check
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


def modify_docstring(source_code, docstring_generator: DocstringGenerator, changed_entities: Optional[Dict[str, Set[str]]] = None):
    module = cst.parse_module(source_code)
    modified_module = module.visit(DocstringTransformer(docstring_generator, module, changed_entities))
    return modified_module.code


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
    def docsmith(file_path, model_id, output, verbose, git, git_base):
        """Generate and write docstrings to a Python file.

        Example usage:

            llm docsmith ./scripts/main.py
            llm docsmith ./scripts/main.py --git
        """
        source = read_source(file_path)
        docstring_generator = partial(
            llm_docstring_generator, model_id=model_id, verbose=verbose
        )
        
        changed_entities = None
        if git:
            changed_entities = get_changed_functions(file_path, git_base)
            if verbose:
                click.echo(f"Changed functions: {changed_entities['functions']}")
                click.echo(f"Changed classes: {changed_entities['classes']}")
        
        modified_source = modify_docstring(source, docstring_generator, changed_entities)

        if output:
            click.echo(modified_source)
            return

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modified_source)
