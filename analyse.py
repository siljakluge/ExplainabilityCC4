import numpy as np
import csv
from pathlib import Path
import ast

def read_csv(path: str, has_header: bool = True):
    # Tries NumPy first (fast), falls back to csv for mixed types
    try:
        if has_header:
            return np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        else:
            return np.genfromtxt(path, delimiter=',', dtype=None, encoding='utf-8')
    except ValueError:
        with open(path, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f)) if has_header else list(csv.reader(f))

def get_actions(data):
    actions = [event["Actions"] for event in data if "server_host_0" in str(event['Actions']) and "Restore" in str(event["Actions"])]
    acts = [ast.literal_eval(event["Actions"]) for event in data]
    for action in actions:
        for action in event["Actions"].keys():
            if "server_host_0" in str(event['Actions'][agent]) and "Restore" in str(event["Actions"][agent]):
                actions.append(event['Actions'][agent])
    return actions

def save_list_to_csv(path: str | Path, data, header: list[str] | None = None) -> None:
    """
    Save a Python list to a CSV file.
    - list[dict]: writes Dict rows (header inferred from keys unless provided).
    - list[list|tuple|ndarray]: each inner sequence is a row.
    - 1D list: writes one value per row.
    """
    from collections.abc import Mapping, Sequence

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Support NumPy arrays
    if isinstance(data, np.ndarray):
        data = data.tolist()

    with p.open('w', newline='', encoding='utf-8') as f:
        # Empty data: write only header if given
        if not data:
            if header:
                csv.writer(f).writerow(header)
            return

        first = data[0]

        # List of dicts
        if isinstance(first, Mapping):
            # Infer header if not provided (union of keys)
            if header is None:
                keys = set()
                for row in data:
                    keys.update(row.keys())
                header = sorted(keys)
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)
            return

        # List of rows or 1D list
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)

        # 1D list (avoid treating strings as sequences)
        is_sequence_row = isinstance(first, Sequence) and not isinstance(first, (str, bytes))
        if not is_sequence_row:
            for item in data:
                writer.writerow([item])
        else:
            for row in data:
                writer.writerow(row)

if __name__ == '__main__':
    here = Path(__file__).parent
    csv_path = here / 'Results/Heuristic_0.4_20251015_1533.csv/Log_20251015_1533.csv'  # change this
    save_path = here / 'Results/Heuristic_0.4_20251015_1533.csv/Actions.csv'  # change this
    data = read_csv(csv_path, has_header=True)
    actions = get_actions(data)
    save_list_to_csv(save_path, actions)



