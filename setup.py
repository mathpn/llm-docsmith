from setuptools import setup, find_packages

setup(
    name="docsmith",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "libcst",
        "ollama",
        "pydantic",
    ],
    entry_points={
        "console_scripts": [
            "docsmith=docsmith.main:main",
        ],
    },
    author="Matheus Pedroni",
    # TODO email
    author_email="your.email@example.com",
    description="A tool to generate Python docstrings automatically using LLMs and syntax trees.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mathpn/docsmith/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
