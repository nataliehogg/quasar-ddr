import numpy as np


class QSOSpecs:
    """
    Specifications for QSO observations.

    This class provides utilities for generating observation specifications
    for QSO surveys.
    """

    @staticmethod
    def redshift_distribution(specs, rng=None, mode="random"):
        """
        Generate redshift distribution for SMBH observations.

        Parameters
        ----------
        specs : dict
            Specifications dictionary with keys:
            - 'z_min': Minimum redshift
            - 'z_max': Maximum redshift
            - 'n_QSO': Number of QSO observations

        Returns
        -------
        array_like
            Array of redshifts. Random catalogues are sorted uniform draws;
            Asimov catalogues use deterministic midpoint quantiles.

        Notes
        -----
        Currently uses uniform random distribution. This should be updated
        to use a physically motivated redshift distribution based on
        expected QSO survey selection functions.
        
        """
        
        if mode not in {"random", "asimov"}:
            raise ValueError("redshift mode must be 'random' or 'asimov'")

        if 'redshifts' in specs:
            redshifts = np.asarray(specs['redshifts'], dtype=float)
            if redshifts.ndim != 1 or redshifts.size == 0 or np.any(redshifts < 0):
                raise ValueError("redshifts must be a non-empty non-negative array")
            return np.sort(redshifts.copy())

        required = {'z_min', 'z_max', 'n_QSO'}
        missing = required.difference(specs)
        if missing:
            raise ValueError(f"missing redshift specifications: {sorted(missing)}")
        z_min = float(specs['z_min'])
        z_max = float(specs['z_max'])
        n_qso = int(specs['n_QSO'])
        if z_min < 0 or z_max <= z_min or n_qso <= 0:
            raise ValueError("require 0 <= z_min < z_max and n_QSO > 0")
        if mode == "asimov":
            quantiles = (np.arange(n_qso, dtype=float) + 0.5) / n_qso
            return z_min + quantiles * (z_max - z_min)
        rng = np.random.default_rng() if rng is None else rng
        return np.sort(rng.uniform(z_min, z_max, n_qso))
