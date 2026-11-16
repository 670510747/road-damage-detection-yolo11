"""Validate notebook structure and Python syntax without running training/downloads."""
import ast
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

root = Path(__file__).resolve().parents[1]
shell = InteractiveShell.instance()
count = 0
for path in sorted((root / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            ast.parse(shell.input_transformer_manager.transform_cell(cell.source), filename=f"{path.name}:cell-{index}")
            assert not cell.outputs, f"Unexpected saved output: {path}:{index}"
            count += 1
for path in (root / "examples").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"PASS: {count} code cells; notebook schema and Python syntax checked. Training was not executed.")
