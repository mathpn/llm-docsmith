import ast
import sys

from ollama import GenerateResponse, generate

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

Do not explain how each function is implemented, focus on its purpose, arguments and returns.

Your response should be a JSON following this format:

```json
{{"<function_1>": "<docstring_1>", "<function_2>": "<docstring_2>"}}
```

Never nest JSON entries.
"""


def read_source(fpath: str):
    with open(fpath, "r") as f:
        source = f.read()
    return source


def get_top_level_nodes(source_code):
    tree = ast.parse(source_code)

    top_level_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef)
    ]

    return top_level_functions


def main():
    source = read_source(sys.argv[1])

    functions = get_top_level_nodes(source)

    functions = [ast.unparse(func) for func in functions]

    for func in functions:
        print(func)
        response: GenerateResponse = generate(
            model="qwen2.5-coder",
            prompt=PROMPT.format(CONTEXT=source, CODE=func),
            format="json",
        )
        print(response.response)


if __name__ == "__main__":
    main()
