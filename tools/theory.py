import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .constants import SPEED_OF_LIGHT_KMS


def _checked_redshift(redshift):
    z = np.asarray(redshift, dtype=float)
    if z.ndim != 1 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("redshift must be a finite one-dimensional non-negative array")
    return z


def dimensionless_expansion_rate(
    redshift,
    cosmology_model,
    omega_m,
    omega_k=0.0,
    w0=-1.0,
):
    z = np.asarray(redshift, dtype=float)
    omega_m = float(omega_m)
    omega_k = float(omega_k)
    w0 = float(w0)
    if cosmology_model == "flat_lcdm":
        omega_k = 0.0
        omega_de = 1.0 - omega_m
        dark_energy = omega_de
    elif cosmology_model == "lcdm":
        omega_de = 1.0 - omega_m - omega_k
        dark_energy = omega_de
    elif cosmology_model == "flat_wcdm":
        omega_k = 0.0
        omega_de = 1.0 - omega_m
        dark_energy = omega_de * (1.0 + z) ** (3.0 * (1.0 + w0))
    else:
        raise ValueError(
            "cosmology_model must be flat_lcdm, lcdm, or flat_wcdm"
        )
    if not all(np.isfinite([omega_m, omega_k, omega_de, w0])):
        raise ValueError("cosmology parameters must be finite")
    if omega_m < 0 or omega_de < 0:
        raise ValueError("matter and dark-energy densities must be non-negative")
    e2 = (
        omega_m * (1.0 + z) ** 3
        + omega_k * (1.0 + z) ** 2
        + dark_energy
    )
    if np.any(e2 <= 0) or np.any(~np.isfinite(e2)):
        raise ValueError("cosmology produces a non-positive expansion rate")
    return np.sqrt(e2)


def angular_diameter_distance(
    redshift,
    h0,
    omega_m,
    cosmology_model="flat_lcdm",
    omega_k=0.0,
    w0=-1.0,
):
    z = _checked_redshift(redshift)
    h0 = float(h0)
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError("H0 must be finite and positive")
    effective_omega_k = 0.0 if cosmology_model != "lcdm" else float(omega_k)
    hubble_distance = SPEED_OF_LIGHT_KMS / h0

    def inverse_e(value):
        return 1.0 / float(
            dimensionless_expansion_rate(
                np.array([value]),
                cosmology_model,
                omega_m,
                omega_k=omega_k,
                w0=w0,
            )[0]
        )

    chi = np.array([quad(inverse_e, 0.0, value)[0] for value in z])
    if effective_omega_k > 0:
        root = np.sqrt(effective_omega_k)
        transverse = np.sinh(root * chi) / root
    elif effective_omega_k < 0:
        root = np.sqrt(-effective_omega_k)
        transverse = np.sin(root * chi) / root
    else:
        transverse = chi
    return hubble_distance * transverse / (1.0 + z)


def ddr_eta(redshift, model, alpha, beta=0.0):
    z = _checked_redshift(redshift)
    alpha = float(alpha)
    beta = float(beta)
    if model == "M1":
        eta = np.full(z.shape, 1.0 + alpha)
    elif model == "M2":
        eta = 1.0 + alpha + beta * z
    elif model == "M3":
        eta = (1.0 + z) ** alpha
    else:
        raise ValueError("DDR model must be M1, M2, or M3")
    if np.any(~np.isfinite(eta)) or np.any(eta <= 0):
        raise ValueError("eta(z) must be finite and strictly positive")
    return eta


class Theory:
    """Compatibility wrapper for M1/M2/M3 electromagnetic luminosity distance.

    New mock workflows should call ``angular_diameter_distance`` and ``ddr_eta``
    directly. The historical constructor signature is retained for repository
    compatibility.
    """

    def __init__(self, z, d_C, d_A, H, Om, H0, alpha, beta, xi0, model):

        self.z = z
        self.d_A = d_A
        self.H0 = H0
        self.eta = ddr_eta(z, model=model, alpha=alpha, beta=beta)

    def EMlumdist(self):
        """
        Compute modified electromagnetic luminosity distance.

        Returns an interpolation function for the luminosity distance
        in the electromagnetic sector, including DDR violation effects.

        Returns
        -------
        callable
            Interpolation function for the dimensionless distance H0 d_L/c.

        Notes
        -----
        The returned dimensionless distance is:
        H0 d_L^EM(z)/c = (H0/c) * (1+z)^2 * d_A(z) * eta(z)

        where eta(z) encodes the DDR violation.
        """
        lumdist = self.H0 / SPEED_OF_LIGHT_KMS * (1 + self.z)**2 * self.d_A(self.z) * self.eta

        return interp1d(self.z, lumdist, kind='cubic')
