---
name: code-validation-audit
description: Comprehensive verification, linting, formatting, docstring auditing, testing, and Git commit workflow to execute after implementing or modifying code.
---

# Code Validation, Quality Audit & Git Commit Workflow

This skill outlines the standard, rigorous quality assurance protocol to follow whenever code is written, refactored, or reviewed in this repository.

---

## 📋 The 5-Step Code Quality Checklist

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Logic & Completeness Audit                               │
├─────────────────────────────────────────────────────────────┤
│ 2. Docstrings & Type Annotations Audit                      │
├─────────────────────────────────────────────────────────────┤
│ 3. Automated Linter, Formatter & Unit Tests                 │
├─────────────────────────────────────────────────────────────┤
│ 4. End-to-End Real-World Smoke Test                         │
├─────────────────────────────────────────────────────────────┤
│ 5. Clean Semantic Git Commit & Remote Push                  │
└─────────────────────────────────────────────────────────────┘
```

---

### **Step 1: Logic & Completeness Audit**
Verify that the implementation handles real-world edge cases:
- **PyTorch Tensor / CUDA Handling**: Are tensors properly detached and moved to CPU before NumPy conversions (`tensor.detach().cpu().numpy()`)?
- **Evaluation Mode & Gradients**: Are inference functions decorated with `@torch.no_grad()` and models put into `model.eval()`?
- **Memory Efficiency**: Avoid in-loop array concatenations (`np.concatenate`); collect batches in Python lists and concatenate once at the end.
- **Resource Lifecycle**: Are file handles, TensorBoard `SummaryWriter` instances, and streams properly closed?

---

### **Step 2: Docstrings & Type Annotations Audit**
- **Docstrings**: Ensure all public functions, classes, and modules have complete Google-style docstrings (`Args`, `Returns`, `Raises`).
- **Type Hints**: Verify modern Python 3.12+ type annotations (`int | None`, `list[int]`, `Literal[...]`).
- **Unused Imports**: Remove stale imports and verify package exports in `__init__.py` (`__all__`).

---

### **Step 3: Automated Linter, Formatter & Test Suite**
Run the automated test and linting pipeline:

```bash
# 1. Lint and auto-fix code issues
uv run ruff check . --fix

# 2. Format all files to 100-character standard
uv run ruff format .

# 3. Execute unit test suite
uv run pytest -q
```
*All tests must pass with 100% success rate and 0 linting warnings before proceeding.*

---

### **Step 4: End-to-End Real-World Smoke Test**
- For training features: Run a fast smoke run (e.g. `uv run python scripts/train.py --epochs 2`) and verify artifacts.
- For inference/preprocessor features: Test on sample real-world images from `data/test_images/`.
- For UI features: Verify Streamlit execution without runtime tracebacks.

---

### **Step 5: Semantic Git Commit & Remote Push**
Stage and commit changes using Conventional Commits:
```bash
# 1. Review status
git status

# 2. Stage changes
git add -A

# 3. Commit with semantic message
# Types: feat, fix, docs, refactor, test, perf, chore
git commit -m "<type>(<scope>): <clear descriptive message>"

# 4. Push to remote
git push origin main
```
