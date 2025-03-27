# llm-docsmith

## WIP!

Generate Python docstrings automatically with LLM and syntax trees.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/en/stable/).

```bash
llm install llm-docsmith
```

## Usage

Pass a Python file as argument to `llm docsmith`:

```bash
llm docsmith ./scripts/main.py
```

The file will be edited to include the generated docstrings.

Options:

- `-m/--model`: Use a model other than the configured LLM default model
- `-o/--output`: Only show the modified code, without modifying the file
- `-v/--verbose`: Verbose output of prompt and response
