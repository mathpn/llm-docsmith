# llm-docsmith

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
- `--git`: Only update docstrings for functions and classes that have been changed since the last commit
- `--git-base`: Git reference to compare against (default: HEAD)

## Git Integration

The `--git` flag enables Git integration, which will only update docstrings for functions and classes that have been changed since the last commit.
This is useful for large codebases where you only want to update docstrings for modified code.

```bash
# Update docstrings only for changed functions and classes
llm docsmith ./scripts/main.py --git
```

It's also possible to compare against a different git reference to find changed, for instance the main branch:

```bash
# Compare against a specific Git reference
llm docsmith ./scripts/main.py --git --git-base main
```
