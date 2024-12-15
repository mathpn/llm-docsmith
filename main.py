import sys
from typing import Protocol

import libcst as cst
from ollama import ChatResponse, chat
from pydantic import BaseModel

PROMPT = """
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
The documentation should always precisely indicate the arguments and return types. For classes, the documentation should
state the purpose of the class.

Use the context only to learn about the code. Write documentation only for the code provided as input code.
Do not explain how each function is implemented, focus on its purpose, arguments and returns.
"""


class Docstring(BaseModel):
    name: str
    docstring: str


class Documentation(BaseModel):
    entries: list[Docstring]


class DocstringGenerator(Protocol):
    def __call__(self, input_code: str, context: str) -> Documentation: ...


class DocstringTransformer(cst.CSTTransformer):
    def __init__(self, docstring_generator: DocstringGenerator):
        self.docstring_gen = docstring_generator

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
        # TODO add context
        doc = self.docstring_gen(source_lines, "")
        new_docstring = find_docstring_by_name(doc, updated_node.name.value)
        new_body = self._modify_docstring(updated_node.body, new_docstring)
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(self, original_node, updated_node):
        source_lines = cst.Module([updated_node]).code
        # TODO add context
        doc = self.docstring_gen(source_lines, "")
        new_docstring = find_docstring_by_name(doc, updated_node.name.value)
        new_body = self._modify_docstring(updated_node.body, new_docstring)
        return updated_node.with_changes(body=new_body)


def find_docstring_by_name(doc: Documentation, name: str) -> str | None:
    entries = [entry for entry in doc.entries if entry.name == name]
    return entries[0].docstring if entries else None


def generate_docstring(input_code: str, context: str) -> Documentation:
    response: ChatResponse = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(CONTEXT=context, CODE=input_code),
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
    modified_module = module.visit(DocstringTransformer(docstring_generator))
    return modified_module.code


def main():
    source = read_source(sys.argv[1])

    modified_source = modify_docstring(source, generate_docstring)

    with open("out.py", "w") as f:
        f.write(modified_source)


if __name__ == "__main__":
    main()
