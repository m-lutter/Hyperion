"""Regenerate the activity list and the inputs needed by each calculation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hyperion.verification import activities, catalog, input_contracts


def main():
    table = activities()
    functions, _ = catalog()
    code_routes = []
    route_types = []
    for kernel in table.kernel:
        if isinstance(kernel, str):
            function = functions[kernel]
            code_routes.append(f"{function.__module__}.{function.__name__}")
            route_types.append("calculation")
        else:
            code_routes.append("hyperion.verification.check_evidence / measured_limits")
            route_types.append("external evidence validation")
    table["code_route"] = code_routes
    table["route_type"] = route_types
    table.to_csv(ROOT / "data/activities.csv", index=False)
    input_contracts().to_csv(ROOT / "data/calculation_inputs.csv", index=False)
    assigned = table.kernel.notna().sum()
    print(f"Indexed {len(table)} activities; {assigned} have calculation assignments.")


if __name__ == "__main__":
    main()
