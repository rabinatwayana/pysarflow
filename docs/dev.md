# Development Setup

Welcome to the development guide for pysarflow!

## 1. Create Conda Environment and Link with SNAP software



## 2. Clone the github repository

```bash
git clone https://github.com/rabinatwayana/pysarflow.git
```

##  3. Install package in editable mode
Go to the directory containing pyproject.toml and run this command to install your local Python package in editable mode, allowing your code changes to take effect right away without needing to reinstall.

```bash
pip install -e .
```
This command reads project's configuration (i.e. `pyproject.toml`) and ensures that all required libraries are installed with the correct versions.

## 4. Install Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) hooks to ensure code quality and consistency across the project. Our pre-commit configuration includes:

- **Ruff Hooks (linter and formatter)**: Ruff is used for linting and formatting. It helps catch issues and enforces a consistent code style.
- **Commitizen**: Helps enforce conventional commit messages for better project history.

It should be installed already. If not run:

```bash
pre-commit install
```
Then, run follwing command to run all the pre-commit hooks, configured in your .pre-commit-config.yaml file

```bash
pre-commit run --all-files
```

## 5. Build Documentation Github Page
#### Build the Static Site
Generate the static HTML files for your documentation inside the site/ directory by running:
```bash
mkdocs build
```
#### Preview the Documentation Locally
To serve the documentation locally and preview it in your browser, run:
```bash
mkdocs serve
```
By default, this will make the site available at: http://127.0.0.1:8000/pysarflow/

#### Publish to GitHub Pages
Your GitHub repository is configured to serve the documentation site from the gh-pages branch.
To deploy your site, run:

```bash
mkdocs gh-deploy
```
This command will build the site, commit the generated files to the gh-pages branch, and push it to GitHub.

Once deployed, your documentation will be available at:
https://rabinatwayana.github.io/pysarflow/

## 6. Getting Started

Once you have environment setup, installed dependencies and pre-commit hooks set, you’re ready for development. A typical workflow might look like:

- Work on a **feature** or **bug fix**. Just tell other people what you will be working on in issues
- **Run your tests** – our project uses Pytest for testing.
- **Commit your changes** – pre-commit hooks ensure that your code meets our quality standards and that your commit messages follow the Conventional Commits guidelines.
- **Submit your PR** - Create a branch with suitable name as per as your changes and raise PR
