import ast
import sys
from typing import Literal, Protocol

import libcst as cst
from ollama import ChatResponse, chat
from pydantic import BaseModel

PROMPT_FILL = """
You are a coding assistant whose task is to generate docstrings for existing code. You will receive code without any docstrings.
Generate the appropiate docstrings for each function, class or method. Important context:

```python
{CONTEXT}
```

Input code:

```python
{CODE}
```

Do not return any code. Return concise documentation only. Follow each programming language conventions for documentation.
For classes, the documentation should state the purpose of the class.

Use the context only to learn about the code. Write documentation only for the code provided as input code.
You will receive JSON template below. Fill the slots marked with <SLOT> with the appropriate description. Return as JSON.

In the Docstring object, do not explain implementation details, focus on the purpose of the function or class. Do not include
information about arguments and return here.

In the Argument object, describe each argument. In the return object, describe the returned values of the function, if any.

Template:

```json
{TEMPLATE}
```
"""


class Argument(BaseModel):
    name: str
    description: str
    annotation: str | None = None
    default: str | None = None


class Return(BaseModel):
    description: str
    annotation: str = "Any"


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


class DocstringTransformer(cst.CSTTransformer):
    def __init__(self, docstring_generator: DocstringGenerator, module: cst.Module):
        self._current_class: str | None = None
        self._doc: Documentation | None = None
        self.module: cst.Module = module
        self.docstring_gen = docstring_generator

    def visit_Module(self, node):
        self.module = node
        return True

    def visit_ClassDef(self, node) -> bool | None:
        self._current_class = node.name.value
        source_lines = cst.Module([node]).code
        # TODO add context
        template = extract_signatures(self.module, node)
        doc = self.docstring_gen(source_lines, "", template)
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

        # Check if first statement is a docstring
        if (
            body_statements
            and isinstance(body_statements[0], cst.SimpleStatementLine)
            and isinstance(body_statements[0].body[0], cst.Expr)
            and isinstance(body_statements[0].body[0].value, cst.SimpleString)
        ):
            # Replace existing docstring
            new_docstring_node = cst.SimpleStatementLine(
                body=[cst.Expr(value=cst.SimpleString(f'"""{new_docstring}"""'))]
            )
            body_statements[0] = new_docstring_node

        # No existing docstring - add new one if provided
        elif new_docstring:
            new_docstring_node = cst.SimpleStatementLine(
                body=[cst.Expr(value=cst.SimpleString(f'"""{new_docstring}"""'))]
            )
            body_statements.insert(0, new_docstring_node)

        # Reconstruct the body
        if isinstance(body, cst.IndentedBlock):
            return body.with_changes(body=tuple(body_statements))
        return tuple(body_statements)

    def leave_FunctionDef(self, original_node, updated_node):
        source_lines = cst.Module([updated_node]).code

        name = updated_node.name.value
        # TODO add context
        if self._current_class is None:
            template = extract_signatures(self.module, updated_node)
            doc = self.docstring_gen(source_lines, "", template)
        elif self._doc is not None:
            doc = self._doc
        else:
            return updated_node

        new_docstring = find_docstring_by_name(doc, name)
        if new_docstring is None:
            print("oh no")
            print(updated_node.name)
            print(doc)
            return updated_node

        new_body = self._modify_docstring(
            updated_node.body, docstring_to_str(new_docstring)
        )
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(self, original_node, updated_node):
        self._current_class = None

        if self._doc is None:
            return updated_node

        new_docstring = find_docstring_by_name(self._doc, updated_node.name.value)

        if new_docstring is None:
            print("oh no 2")
            print(updated_node.name)
            print(self._doc)
            return updated_node

        new_body = self._modify_docstring(
            updated_node.body, docstring_to_str(new_docstring)
        )

        return updated_node.with_changes(body=new_body)


def find_function_defs(node):
    function_defs = []

    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
        function_defs.append(node)

    for child_node in ast.iter_child_nodes(node):
        function_defs.extend(find_function_defs(child_node))

    return function_defs


def find_class_defs(node):
    class_defs = []

    if isinstance(node, ast.ClassDef):
        class_defs.append(node)

    for child_node in ast.iter_child_nodes(node):
        class_defs.extend(find_class_defs(child_node))

    return class_defs


def extract_signatures(module: cst.Module, node: cst.CSTNode) -> Documentation:
    source = module.code_for_node(node)

    tree = ast.parse(source)
    function_defs = find_function_defs(tree)
    # TODO argument
    function_defs = filter(lambda x: not is_private(x), function_defs)
    function_defs = filter(lambda x: not is_dunder(x), function_defs)

    class_defs = find_class_defs(tree)
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
    return_type = ast.unparse(function_node.returns) if function_node.returns else None

    # TODO check if there is any return
    ret = (
        Return(description="<SLOT>", annotation=return_type)
        if return_type is not None
        else None
    )

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


def docstring_to_str(docstring: Docstring) -> str:
    args_strings = []
    for arg in docstring.args or []:
        if arg.annotation is not None:
            arg_string = f"    {arg.name} ({arg.annotation}): {arg.description}"
        else:
            arg_string = f"    {arg.name}: {arg.description}"
        if arg.default is not None:
            arg_string += f" (default {arg.default})"
        args_strings.append(arg_string)

    string = f"{docstring.docstring}"

    if args_strings:
        string += f"""\n\nParameters:
-----------

{"\n".join(args_strings)}"""

    if docstring.ret is not None:
        string += f"""\n\nReturns:
--------

    {docstring.ret.annotation} : {docstring.ret.description}
"""
    return string


def generate_docstring(
    input_code: str, context: str, template: Documentation
) -> Documentation:
    response: ChatResponse = chat(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": PROMPT_FILL.format(
                    CONTEXT=context,
                    CODE=input_code,
                    TEMPLATE=template.model_dump_json(),
                ),
            }
        ],
        format=Documentation.model_json_schema(),
    )
    return Documentation.model_validate_json(response.message.content)


def read_source(fpath: str):
    with open(fpath, "r") as f:
        source = f.read()
    return source


def modify_docstring(source_code, docstring_generator: DocstringGenerator):
    module = cst.parse_module(source_code)
    modified_module = module.visit(DocstringTransformer(docstring_generator, module))
    return modified_module.code


def main():
    source = read_source(sys.argv[1])

    modified_source = modify_docstring(source, generate_docstring)

    with open("out.py", "w") as f:
        f.write(modified_source)


if __name__ == "__main__":
    main()
