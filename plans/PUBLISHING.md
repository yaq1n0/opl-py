# Publishing opl-py to PyPI

This guide covers the one-time setup and the ongoing release process for publishing `opl-py` to PyPI via GitHub Actions using **Trusted Publishers** (OIDC) — no API tokens stored in secrets.

---

## One-time setup

### 1. Create a GitHub Actions workflow

Create the file `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev,analytics]"
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/
      - run: pyright src/
      - run: pytest tests/ -v --cov=src/opl --cov-report=term-missing

  publish:
    needs: check
    runs-on: ubuntu-latest
    permissions:
      id-token: write # required for OIDC trusted publishing

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

### 2. Configure a Trusted Publisher on PyPI

This lets PyPI verify the release came from your GitHub Actions workflow without needing an API token.

1. Log in to [pypi.org](https://pypi.org)
2. Go to **Your projects → opl-py** (or, if the package doesn't exist yet, go to **Publishing → Add a new pending publisher**)
3. Under **Trusted Publishers**, click **Add a new publisher** and fill in:

   | Field             | Value                       |
   | ----------------- | --------------------------- |
   | PyPI Project Name | `opl-py`                    |
   | Owner             | your GitHub username or org |
   | Repository        | `opl-py`                    |
   | Workflow filename | `publish.yml`               |
   | Environment name  | *(Any)*                     |

> **First release only:** If `opl-py` doesn't exist on PyPI yet, use the **pending publisher** flow at `pypi.org/manage/account/publishing/` — this creates the project on first publish.

---

## Releasing a new version

### 1. Bump the version

Edit `src/opl/__init__.py` and update `__version__`:

```python
__version__ = "0.2.0"  # was 0.1.0
```

Follow [Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`

- **PATCH** — bug fixes, no API changes
- **MINOR** — new features, backwards compatible
- **MAJOR** — breaking changes

### 2. Run quality checks locally

```bash
make check
```

All lint, type, and test checks must pass before tagging.

### 3. Commit, tag, and push

```bash
git add src/opl/__init__.py
git commit -m "chore: bump version to 0.2.0"
git tag v0.2.0
git push origin main --tags
```

Pushing the tag triggers the `publish.yml` workflow: it runs `make check`, builds the wheel and sdist, then uploads to PyPI automatically.

### 4. Verify the release

- Check the Actions tab on GitHub to confirm the workflow passed
- Confirm the new version is live at `https://pypi.org/project/opl-py/`

---

## Building locally (optional)

To inspect the build artifacts before releasing:

```bash
pip install build
python -m build
```

This produces `dist/opl_py-<version>-py3-none-any.whl` and `dist/opl_py-<version>.tar.gz`.

To do a dry-run upload against the [TestPyPI](https://test.pypi.org) index:

```bash
pip install twine
twine upload --repository testpypi dist/*
```

TestPyPI requires a separate account and API token (create one at `test.pypi.org/manage/account/token/`).
