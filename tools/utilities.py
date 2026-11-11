import re
from pathlib import Path

class Utilities:
    MOCK_PATTERN = re.compile(
        r"^(?P<model>.+?)"                                     # Non-greedy match for the model name
        r"alpha(?P<alpha>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"     # Matches the alpha float
        r"beta(?P<beta>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"       # Matches the beta float
        r"nQSO(?P<nQSO>\d+)"                                   # Matches the nQSO integer
        r"(?:_(?P<scenario>[A-Za-z]+))?"                       # Optional error-budget scenario
        r"_(?:(?P<obs>da|dl)_dataset\.txt"                     # legacy text pair
        r"|(?P<paired>paired)_dataset\.npz)\s*$",              # paired NPZ
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
            "alpha_str": data["alpha"],
            "alpha": float(data["alpha"]),
            "beta_str": data["beta"],
            "beta": float(data["beta"]),
            "n_QSO": int(data["nQSO"]),
            # Older mocks predate the scenario tag and carry no error budget
            "scenario": (data["scenario"] or "unspecified").lower(),
            "obs": "npz" if data["paired"] else data["obs"].lower(),
        }


    @staticmethod
    def scan_mock_data(mock_dir="mock_data"):
        path = Path(mock_dir)
        if not path.exists():
            return [], []

        records = []
        unmatched = []
        candidates = sorted(list(path.rglob("*.txt")) + list(path.rglob("*.npz")))
        for file_path in candidates:
            info = Utilities.parse_mock_filename(file_path.name)
            if info is None:
                unmatched.append(file_path.name)
                continue
            info["file"] = str(file_path)
            relative_parent = file_path.parent.relative_to(path)
            info["subfolder"] = "" if str(relative_parent) == "." else str(relative_parent)
            records.append(info)

        return records, unmatched

    @staticmethod
    def get_mock_index(mock_dir="mock_data"):
        records, unmatched = Utilities.scan_mock_data(mock_dir)
        index = {}
        for record in records:
            # Key updated to use alpha/beta strings
            key = (record["model"], record["subfolder"], record["alpha_str"],
                   record["beta_str"], record["n_QSO"], record["scenario"])
            entry = index.setdefault(key, {"da": None, "dl": None, "npz": None})
            entry[record["obs"]] = record["file"]
        return index, unmatched

    @staticmethod
    def list_mock_datasets(mock_dir="mock_data"):
        index, unmatched = Utilities.get_mock_index(mock_dir)
        datasets = []
        for i, ((model, subfolder, alpha_str, beta_str, n_QSO, scenario), paths) in enumerate(sorted(index.items())):
            suffix = "" if scenario == "unspecified" else f"_{scenario}"
            name = f"{model}alpha{alpha_str}beta{beta_str}nQSO{n_QSO}{suffix}"
            datasets.append(
                {
                    "index": i,
                    "model": model,
                    "subfolder": subfolder,
                    "alpha": float(alpha_str),
                    "beta": float(beta_str),
                    "n_QSO": n_QSO,
                    "scenario": scenario,
                    "settings": (
                        f"{subfolder}/{name}" if subfolder else name
                    ),
                }
            )

        return datasets, unmatched

    @staticmethod
    def print_mock_datasets(datasets, unmatched=None):
        if not datasets:
            print("No matching mock datasets found.")
            return

        for row in datasets:
            location = f"{row['subfolder']} | " if row["subfolder"] else ""
            print(
                f"[{row['index']}] "
                f"{location}"
                f"{row['model']} "
                f"α={row['alpha']} "
                f"β={row['beta']} "
                f"n_QSO={row['n_QSO']} "
                f"errors={row['scenario']}"
            )

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

        for (model, subfolder, alpha_str, beta_str, n_QSO, scenario), paths in sorted(index.items()):
            da_path = paths.get("da")
            dl_path = paths.get("dl")
            location = f"{subfolder} | " if subfolder else ""
            print(f"{location}{model} α={alpha_str} β={beta_str} n_QSO={n_QSO} "
                  f"errors={scenario} | da: {da_path} | dl: {dl_path}")

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

    # DDR model -> parameters that actually enter eta(z) and should be sampled
    MODEL_PARAMS = {"M1": ["alpha"], "M2": ["alpha", "beta"], "M3": ["alpha"]}

    # Fixed prior / proposal / reference-width template, shared by all runs.
    # ref_scale sets the spread of the walker starting points.
    PARAM_TEMPLATE = {
        "alpha":     {"prior": {"min": -0.2, "max": 0.2}, "proposal": 0.05,  "ref_scale": 0.1,  "latex": r"\alpha"},
        "beta":      {"prior": {"min": -0.1, "max": 0.1}, "proposal": 0.005, "ref_scale": 0.05, "latex": r"\beta"},
    }

    @staticmethod
    def sampler_config(dataset, likelihood):
        """
        Build a Cobaya `info` dictionary for sampling a mock dataset.

        Priors and MCMC settings are fixed; only the set of sampled parameters
        varies with the DDR violation model (M1/M3: alpha; M2: alpha, beta).
        Walker starting points (`ref`) are seeded at the injected fiducial
        truth taken from the dataset.

        Parameters
        ----------
        dataset : dict
            A dataset record from `select_mock_dataset` / `list_mock_datasets`,
            providing 'model', 'alpha', 'beta' and 'settings'.
        likelihood : class
            The Cobaya likelihood class (e.g. DDRlike). Its __name__ is used as
            the likelihood key.

        Returns
        -------
        dict
            A Cobaya `info` dictionary ready to pass to `cobaya.run.run`.
        """
        names = list(Utilities.MODEL_PARAMS[dataset["model"]])

        params = {}
        for name in names:
            spec = dict(Utilities.PARAM_TEMPLATE[name])
            ref_scale = spec.pop("ref_scale")
            spec["ref"] = {"dist": "norm", "loc": dataset[name], "scale": ref_scale}
            params[name] = spec

        return {
            "params": params,
            "sampler": {"mcmc": {"max_tries": 10000, "Rminus1_stop": 0.1}},
            "likelihood": {
                likelihood.__name__: {
                    "external": likelihood,
                    "model": dataset["model"],
                    "datapath": "./mock_data/" + dataset["settings"],
                }
            },
            "output": "chains/" + dataset["settings"],
            "force": False,
        }
