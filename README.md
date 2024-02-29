# LangEvals

A Python package for aggregating various language model evaluation tools into a single platform. This package allows developers to evaluate the quality of language models using a variety of evaluators, which can be run locally or through an API.

# Contributing

LangEvals is a monorepo and has many subpackages with different dependencies for each evaluator library or provider. We use poetry to install all dependencies and create a virtual env for each sub-package to make sure they are fully isolated. Given this complexity, to make it easier to contribute to LangEvals we recomend using VS Code for the development. Before opening up on VS Code though, you need to make sure to install all dependencies, generating thus the .venv for each package:

```
make install
```

This will also generate the `langevals.code-workspace` file, creating a different workspace per evaluator and telling VS Code which venv to use for each. Then, open this file on vscode and click the "Open Workspace" button