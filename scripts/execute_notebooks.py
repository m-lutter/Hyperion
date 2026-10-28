"""Run all deliverable notebooks and save their figures and execution report.

Run from the repository root after installing requirements.txt. An error stops
the run; a notebook is saved only after every cell and figure check succeeds.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hyperion.verification import baseline_id


def execute_notebook(path, kernel_directory):
    """Execute one notebook, verify its plots, and return a short run record."""
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    manager = AsyncKernelManager(
        kernel_name="python3",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[kernel_directory]),
    )

    def progress(cell, cell_index):
        if cell.cell_type == "code":
            print(
                f"{path.name}: cell {cell_index + 1}/{len(notebook.cells)}",
                flush=True,
            )

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=1200,
        startup_timeout=60,
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        on_cell_start=progress,
    )
    client.execute(cleanup_kc=True)
    nbformat.validate(notebook)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    figures = 0
    for cell in code_cells:
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                figures += 1

    if path.name.startswith("Hyperion_"):
        required_figures = 5
    elif path.name.startswith(("01_", "02_")):
        required_figures = 1
    else:
        required_figures = 0
    if figures < required_figures:
        raise RuntimeError(
            f"{path.name}: expected at least {required_figures} rendered figures, "
            f"found {figures}"
        )

    nbformat.write(notebook, path)
    return {
        "notebook": path.name,
        "code_cells": len(code_cells),
        "figures": figures,
        "seconds": round(time.perf_counter() - started, 2),
        "errors": 0,
        "baseline_id": baseline_id(),
    }


def main():
    paths = [ROOT / "Hyperion_6DOF_RK45_Annotated.ipynb"]
    paths += sorted(ROOT.glob("0[1-4]_*.ipynb"))
    logs = []

    # Use this Python environment, even if Jupyter has another default kernel.
    with tempfile.TemporaryDirectory(prefix="hyperion_kernel_") as kernel_directory:
        spec_directory = Path(kernel_directory) / "python3"
        spec_directory.mkdir()
        kernel_spec = {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "Python 3",
            "language": "python",
            "env": {
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "MPLBACKEND": "Agg",
            },
        }
        (spec_directory / "kernel.json").write_text(json.dumps(kernel_spec))
        for path in paths:
            logs.append(execute_notebook(path, kernel_directory))

    report = ROOT / "reports/notebook_execution.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(logs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()
