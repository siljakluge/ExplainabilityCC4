import torch
from pathlib import Path
import pprint

LOG_DIR = Path("logs")

LOG_FILES = [
    "contractoractive.pt",
    "contractorinactive.pt",
    "m1_contractorinactive.pt",
]

pp = pprint.PrettyPrinter(indent=2, width=120)

for fname in LOG_FILES:
    path = LOG_DIR / fname
    print("\n" + "=" * 80)
    print(f"Inspecting: {path}")
    print("=" * 80)

    if not path.exists():
        print("❌ File does not exist")
        continue

    log = torch.load(path, map_location="cpu", weights_only=False)

    print(f"Top-level type: {type(log)}")
    try:
        print(f"Length: {len(log)}")
    except TypeError:
        print("Not a sized object")

    if isinstance(log, list) and len(log) > 0:
        first = log[0]
        print("\nFirst entry type:", type(first))

        if isinstance(first, (tuple, list)):
            print("Tuple/list length:", len(first))
            print("Raw value:")
            pp.pprint(first)

        elif isinstance(first, dict):
            print("Dict keys:", list(first.keys()))
            print("Raw value:")
            pp.pprint(first)

        else:
            print("Raw value:")
            pp.pprint(first)

        # Explicit checks for loss components
        if isinstance(first, dict):
            print("\nLoss fields present:")
            for k in first.keys():
                if "loss" in k.lower():
                    print("  ", k)
        else:
            print("\nNo named loss fields (tuple-based log).")

    else:
        print("Log is empty or not a list.")
