from pathlib import Path
import json
import re

repo = Path(r"E:\Github-assessment\uncertainty-aware-or-scheduling")
path = repo / "optimization" / "milp_engine.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

patterns = [
    r"beta",
    r"balance",
    r"imbalance",
    r"workload",
    r"0\.10",
    r"0\.1",
]

regex = re.compile("|".join(patterns), re.IGNORECASE)

print("=" * 80)
print("WORKLOAD-BALANCE LOCATOR")
print("=" * 80)

for i, cell in enumerate(nb.get("cells", [])):
    src = cell.get("source", [])
    if isinstance(src, list):
        src = "".join(src)

    lines = src.splitlines()

    matching = [
        (j + 1, line)
        for j, line in enumerate(lines)
        if regex.search(line)
    ]

    if matching:
        print(f"\nCELL {i} ({cell.get('cell_type')})")
        print("-" * 80)

        for line_no, line in matching:
            print(f"{line_no:03d}: {line}")