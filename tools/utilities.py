import math
import re
from pathlib import Path


class Utilities:
    MOCK_PATTERN = re.compile(
        r"^(?P<model>[A-Za-z0-9]+)"
        r"A(?P<A>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
        r"B(?P<B>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
        r"nQSO(?P<nQSO>\d+)"
        r"_(?P<obs>da|dl)_dataset\.txt$",
        re.IGNORECASE,
    )

    @staticmethod
    def parse_mock_filename(name):
        match = Utilities.MOCK_PATTERN.match(name)
        if not match:
            return None
        data = match.groupdict()
        return {
            "model": data["model"],
            "A_str": data["A"],
            "A": float(data["A"]),
            "B_str": data["B"],
            "B": float(data["B"]),
            "n_QSO": int(data["nQSO"]),
            "obs": data["obs"].lower(),
        }

    @staticmethod
    def scan_mock_data(mock_dir="mock_data"):
        path = Path(mock_dir)
        if not path.exists():
            return [], []

        records = []
        unmatched = []
        for file_path in sorted(path.glob("*.txt")):
            info = Utilities.parse_mock_filename(file_path.name)
            if info is None:
                unmatched.append(file_path.name)
                continue
            info["file"] = str(file_path)
            records.append(info)

        return records, unmatched

    @staticmethod
    def get_mock_index(mock_dir="mock_data"):
        records, unmatched = Utilities.scan_mock_data(mock_dir)
        index = {}
        for record in records:
            # Key now includes B_str
            key = (record["model"], record["A_str"], record["B_str"], record["n_QSO"])
            entry = index.setdefault(key, {"da": None, "dl": None})
            entry[record["obs"]] = record["file"]
        return index, unmatched

    @staticmethod
    def list_mock_datasets(mock_dir="mock_data"):
        index, unmatched = Utilities.get_mock_index(mock_dir)
        datasets = []
        for i, ((model, A_str, B_str, n_QSO), paths) in enumerate(sorted(index.items())):
            datasets.append(
                {
                    "index": i,
                    "model": model,
                    "A": float(A_str),
                    "B": float(B_str),
                    "n_QSO": n_QSO,
                    "settings": f"{model}A{A_str}B{B_str}nQSO{n_QSO}",
                }
            )
        return datasets, unmatched

    @staticmethod
    def print_mock_datasets(datasets, unmatched=None):
        if not datasets:
            print("No matching mock datasets found.")
            return

        for row in datasets:
            print(f"[{row['index']}] {row['model']} A={row['A']} B={row['B']} n_QSO={row['n_QSO']}")

        if unmatched:
            print("\nUnmatched files:")
            for name in unmatched:
                print(name)

    @staticmethod
    def select_mock_dataset(index=None, mock_dir="mock_data", datasets=None):
        if datasets is None:
            datasets, _ = Utilities.list_mock_datasets(mock_dir)

        if not datasets:
            raise ValueError("No matching mock datasets found.")

        if index is None:
            raise IndexError('You must select a dataset to use by its index.')

        if index < 0 or index >= len(datasets):
            raise IndexError(f"Index {index} out of range for {len(datasets)} datasets.")

        selected = dict(datasets[index])
        selected.pop("index", None)
        return selected

    @staticmethod
    def print_mock_index(index, unmatched=None):
        if not index:
            print("No matching mock datasets found.")
            return

        for (model, A_str, B_str, n_QSO), paths in sorted(index.items()):
            da_path = paths.get("da")
            dl_path = paths.get("dl")
            # Fixed typo n_QSP -> n_QSO
            print(f"{model} A={A_str} B={B_str} n_QSO={n_QSO} | da: {da_path} | dl: {dl_path}")

        if unmatched:
            print("\nUnmatched files:")
            for name in unmatched:
                print(name)

    @staticmethod
    def load_mock_dataset(path):
        import numpy as np

        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return {"z": data[:, 0], "value": data[:, 1], "error": data[:, 2]}

    @staticmethod
    def _val_matches(val, val_str, val_float):
        # Renamed from _A_matches to a generic _val_matches so it works for A and B
        if isinstance(val, str):
            return val == val_str
        try:
            return math.isclose(float(val), val_float, rel_tol=1e-9, abs_tol=0.0)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def load_mock_pair(model, A, B, n_QSO, mock_dir="mock_data", index=None):
        if index is None:
            index, _ = Utilities.get_mock_index(mock_dir)

        model_lower = str(model).lower()
        target_n = int(n_QSO)

        for (m, A_str, B_str, n), paths in index.items():
            if m.lower() != model_lower:
                continue
            if n != target_n:
                continue
            if not Utilities._val_matches(A, A_str, float(A_str)):
                continue
            # Added check for B
            if not Utilities._val_matches(B, B_str, float(B_str)):
                continue

            out = {}
            if paths.get("da"):
                out["da"] = Utilities.load_mock_dataset(paths["da"])
            if paths.get("dl"):
                out["dl"] = Utilities.load_mock_dataset(paths["dl"])
            return out

        return {}

    @staticmethod
    def load_all(mock_dir="mock_data", index=None):
        if index is None:
            index, _ = Utilities.get_mock_index(mock_dir)

        out = {}
        for key, paths in index.items():
            out[key] = {}
            for obs in ("da", "dl"):
                if paths.get(obs):
                    out[key][obs] = Utilities.load_mock_dataset(paths[obs])
        return out
