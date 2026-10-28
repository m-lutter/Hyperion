"""Execute all deliverable notebooks with the active Python environment.

Run from the repository root after installing requirements.txt. Errors abort the
run; outputs are saved only after that notebook succeeds.
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
sys.path.insert(0,str(ROOT))
from hyperion.verification import baseline_id
paths = [ROOT / "Hyperion_6DOF_RK45_Annotated.ipynb"] + sorted(ROOT.glob("0[1-4]_*.ipynb"))
logs = []
with tempfile.TemporaryDirectory(prefix="hyperion_kernel_") as tmp:
    spec = Path(tmp) / "python3"
    spec.mkdir()
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3", "language": "python",
        "env": {"PYTHONPATH": os.environ.get("PYTHONPATH", ""), "MPLBACKEND": "Agg"},
    }))
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        started = time.perf_counter()
        manager = AsyncKernelManager(kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
        def progress(cell, cell_index):
            if cell.cell_type == "code":
                print(f"{path.name}: cell {cell_index+1}/{len(nb.cells)}", flush=True)
        client = NotebookClient(nb, km=manager, timeout=1200, startup_timeout=60,
            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False, on_cell_start=progress)
        client.execute(cleanup_kc=True)
        nbformat.validate(nb)
        cells = [c for c in nb.cells if c.cell_type == "code"]
        figures = sum("image/png" in o.get("data",{}) for c in cells for o in c.get("outputs",[]))
        required_figures = 5 if path.name.startswith('Hyperion_') else 1 if path.name.startswith(('01_','02_')) else 0
        if figures < required_figures:
            raise RuntimeError(f'{path.name}: expected at least {required_figures} rendered figures, found {figures}')
        nbformat.write(nb, path)
        logs.append({"notebook":path.name, "code_cells":len(cells),
            "figures":figures,
            "seconds":round(time.perf_counter()-started,2), "errors":0,
            "baseline_id":baseline_id()})
report = ROOT / "reports/notebook_execution.json"
report.parent.mkdir(exist_ok=True)
report.write_text(json.dumps(logs,indent=2)+"\n",encoding="utf-8")
print(json.dumps(logs,indent=2))
