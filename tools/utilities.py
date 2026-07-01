import math
import re
from pathlib import Path

class Utilities:
    MOCK_PATTERN = re.compile(
        r"^(?P<model>.+?)"                                     # Non-greedy match for the model name
        r"alpha(?P<alpha>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"     # Matches the alpha float
        r"beta(?P<beta>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"       # Matches the beta float
        r"nQSO(?P<nQSO>\d+)"                                   # Matches the nQSO integer
        r"_(?P<obs>da|dl)_dataset\.txt\s*$",                   # Matches the observation type and extension
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
            # Key updated to use alpha/beta strings
            key = (record["model"], record["alpha_str"], record["beta_str"], record["n_QSO"])
            entry = index.setdefault(key, {"da": None, "dl": None})
            entry[record["obs"]] = record["file"]
        return index, unmatched

    @staticmethod
    def list_mock_datasets(mock_dir="mock_data"):
        index, unmatched = Utilities.get_mock_index(mock_dir)
        datasets = []
        for i, ((model, alpha_str, beta_str, n_QSO), paths) in enumerate(sorted(index.items())):
            datasets.append(
                {
                    "index": i,
                    "model": model,
                    "alpha": float(alpha_str),
                    "beta": float(beta_str),
                    "n_QSO": n_QSO,
                    "settings": f"{model}alpha{alpha_str}beta{beta_str}nQSO{n_QSO}",
                }
            )
        return datasets, unmatched

    @staticmethod
    def print_mock_datasets(datasets, unmatched=None):
        if not datasets:
            print("No matching mock datasets found.")
            return

        for row in datasets:
            print(f"[{row['index']}] {row['model']} α={row['alpha']} β={row['beta']} n_QSO={row['n_QSO']}")

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

        for (model, alpha_str, beta_str, n_QSO), paths in sorted(index.items()):
            da_path = paths.get("da")
            dl_path = paths.get("dl")
            print(f"{model} α={alpha_str} β={beta_str} n_QSO={n_QSO} | da: {da_path} | dl: {dl_path}")

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
        if isinstance(val, str):
            return val == val_str
        try:
            return math.isclose(float(val), val_float, rel_tol=1e-9, abs_tol=0.0)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def load_mock_pair(model, alpha, beta, n_QSO, mock_dir="mock_data", index=None):
        if index is None:
            index, _ = Utilities.get_mock_index(mock_dir)

        model_lower = str(model).lower()
        target_n = int(n_QSO)

        for (m, alpha_str, beta_str, n), paths in index.items():
            if m.lower() != model_lower:
                continue
            if n != target_n:
                continue
            if not Utilities._val_matches(alpha, alpha_str, float(alpha_str)):
                continue
            if not Utilities._val_matches(beta, beta_str, float(beta_str)):
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

    # DDR model -> parameters that actually enter eta(z) and should be sampled
    MODEL_PARAMS = {"M1": ["alpha"], "M2": ["alpha", "beta"], "M3": ["alpha"]}

    # Fixed prior / proposal / reference-width template, shared by all runs.
    # ref_scale sets the spread of the walker starting points.
    PARAM_TEMPLATE = {
        "alpha":     {"prior": {"min": -0.2, "max": 0.2}, "proposal": 0.05,  "ref_scale": 0.1,  "latex": r"\alpha"},
        "beta":      {"prior": {"min": -0.1, "max": 0.1}, "proposal": 0.005, "ref_scale": 0.05, "latex": r"\beta"},
        "sigma_int": {"prior": {"min": 0.0,  "max": 0.5}, "proposal": 0.02,  "ref_scale": 0.05, "latex": r"\sigma_{\rm int}"},
    }

    @staticmethod
    def sampler_config(dataset, likelihood, sample_sigma_int=True):
        """
        Build a Cobaya `info` dictionary for sampling a mock dataset.

        Priors and MCMC settings are fixed; only the set of sampled parameters
        varies with the DDR violation model (M1/M3: alpha; M2: alpha, beta).
        sigma_int is appended when `sample_sigma_int` is True. Walker starting
        points (`ref`) are seeded at the injected fiducial truth where it is
        known (alpha, beta from the dataset); sigma_int has no injected truth,
        so it starts at a small interior value rather than at the prior edge.

        Parameters
        ----------
        dataset : dict
            A dataset record from `select_mock_dataset` / `list_mock_datasets`,
            providing 'model', 'alpha', 'beta' and 'settings'.
        likelihood : class
            The Cobaya likelihood class (e.g. DDRlike). Its __name__ is used as
            the likelihood key.
        sample_sigma_int : bool, optional
            Whether to sample the intrinsic-scatter nuisance parameter.

        Returns
        -------
        dict
            A Cobaya `info` dictionary ready to pass to `cobaya.run.run`.
        """
        names = list(Utilities.MODEL_PARAMS[dataset["model"]])
        if sample_sigma_int:
            names.append("sigma_int")

        ref_truth = {
            "alpha": dataset["alpha"],
            "beta": dataset["beta"],
            "sigma_int": 0.1,  # no injected truth; start inside the prior
        }

        params = {}
        for name in names:
            spec = dict(Utilities.PARAM_TEMPLATE[name])
            ref_scale = spec.pop("ref_scale")
            spec["ref"] = {"dist": "norm", "loc": ref_truth[name], "scale": ref_scale}
            params[name] = spec

        return {
            "params": params,
            "sampler": {"mcmc": {"max_tries": 10000, "Rminus1_stop": 0.1}},
            "likelihood": {
                likelihood.__name__: {
                    "external": likelihood,
                    "model": dataset["model"],
                    "datapath": "./mock_data/" + dataset["settings"],
                    "sample_sigma_int": sample_sigma_int,
                }
            },
            "output": "chains/" + dataset["settings"],
            "force": True,
        }
