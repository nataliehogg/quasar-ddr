"""Named, unit-explicit observational error scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .ddr_statistics import uvx_residual_dex_to_log10_distance


@dataclass(frozen=True)
class ErrorScenario:
    """Effective random-error model for a paired UV--X/SARM forecast.

    Parameters
    ----------
    uvx_residual_dex
        Residual scatter in the log10(F_X) direction of the UV--X relation.
    sarm_sigma_ln
        Effective one-object natural-log scatter of the SARM angular distance.
        For errors below about 30 per cent this is close to the quoted
        fractional distance uncertainty.
    """

    name: str
    uvx_residual_dex: float
    sarm_sigma_ln: float
    gamma: float = 0.6
    calibration_sigma_ln: float = 0.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must not be empty")
        if not math.isfinite(self.uvx_residual_dex) or self.uvx_residual_dex < 0:
            raise ValueError("uvx_residual_dex must be finite and non-negative")
        if not math.isfinite(self.sarm_sigma_ln) or self.sarm_sigma_ln < 0:
            raise ValueError("sarm_sigma_ln must be finite and non-negative")
        if not math.isfinite(self.gamma) or self.gamma >= 1:
            raise ValueError("gamma must be finite and smaller than 1")
        if (
            not math.isfinite(self.calibration_sigma_ln)
            or self.calibration_sigma_ln < 0
        ):
            raise ValueError(
                "calibration_sigma_ln must be finite and non-negative"
            )

    @property
    def sigma_log10_dl(self):
        return float(
            uvx_residual_dex_to_log10_distance(
                self.uvx_residual_dex, gamma=self.gamma
            )
        )

    @property
    def sigma_ln_dl(self):
        return self.sigma_log10_dl * math.log(10.0)

    @property
    def sigma_ln_eta(self):
        return math.hypot(self.sigma_ln_dl, self.sarm_sigma_ln)


@dataclass(frozen=True)
class ResolvedDistanceErrors:
    name: str
    sigma_ln_dl: float
    sigma_ln_da: float
    uvx_residual_dex: float | None
    errors_include_intrinsic: bool = True


UVX_EFFECTIVE_OPTIONS = {
    "current": 0.24,
    "future": 0.17,
    "targeted": 0.14,
}

SARM_ERROR_OPTIONS = {
    "current": 0.30,
    "future": 0.20,
    "targeted": 0.15,
}


@dataclass(frozen=True)
class ResolvedUVXError:
    option: str
    sigma_ln_dl: float
    sigma_uvx_dex: float | None
    gamma: float
    gamma_used: bool
    errors_include_intrinsic: bool = True


@dataclass(frozen=True)
class ResolvedSARMError:
    option: str
    sigma_ln_da: float


def resolve_uvx_error(option, *, gamma=0.60, custom_sigma_ln_dl=None):
    """Resolve an effective UV--X scatter into natural-log distance scatter."""
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("UVX_GAMMA must satisfy 0 <= UVX_GAMMA < 1")
    if option == "custom":
        if custom_sigma_ln_dl is None:
            raise ValueError("CUSTOM_SIGMA_LN_DL must be finite and non-negative")
        sigma_ln_dl = float(custom_sigma_ln_dl)
        if not math.isfinite(sigma_ln_dl) or sigma_ln_dl < 0:
            raise ValueError("CUSTOM_SIGMA_LN_DL must be finite and non-negative")
        return ResolvedUVXError("custom", sigma_ln_dl, None, gamma, False)
    if option not in UVX_EFFECTIVE_OPTIONS:
        valid = "current, future, targeted, custom"
        raise KeyError(
            f"unknown effective UV--X option {option!r}; choose one of: {valid}"
        )
    residual = UVX_EFFECTIVE_OPTIONS[option]
    sigma_ln_dl = math.log(10.0) * residual / (2.0 * (1.0 - gamma))
    return ResolvedUVXError(option, sigma_ln_dl, residual, gamma, True)


def resolve_sarm_error(option, *, custom_sigma_ln_da=None):
    """Resolve an effective SARM angular-distance scatter."""
    if option == "custom":
        if custom_sigma_ln_da is None:
            raise ValueError("CUSTOM_SIGMA_LN_DA must be finite and non-negative")
        sigma_ln_da = float(custom_sigma_ln_da)
        if not math.isfinite(sigma_ln_da) or sigma_ln_da < 0:
            raise ValueError("CUSTOM_SIGMA_LN_DA must be finite and non-negative")
        return ResolvedSARMError("custom", sigma_ln_da)
    if option not in SARM_ERROR_OPTIONS:
        valid = "current, future, targeted, custom"
        raise KeyError(f"unknown SARM option {option!r}; choose one of: {valid}")
    return ResolvedSARMError(option, SARM_ERROR_OPTIONS[option])


ERROR_SCENARIOS = {
    name: ErrorScenario(
        name,
        uvx_residual_dex=resolve_uvx_error(name, gamma=0.60).sigma_uvx_dex,
        sarm_sigma_ln=resolve_sarm_error(name).sigma_ln_da,
        gamma=0.60,
    )
    for name in UVX_EFFECTIVE_OPTIONS
}


def get_error_scenario(name):
    """Return a named scenario, raising an informative error if it is unknown."""
    try:
        return ERROR_SCENARIOS[name]
    except KeyError as exc:
        valid = ", ".join(ERROR_SCENARIOS)
        raise KeyError(f"unknown error scenario {name!r}; choose one of: {valid}") from exc


def resolve_distance_errors(
    name,
    custom_sigma_ln_dl=None,
    custom_sigma_ln_da=None,
):
    """Resolve a named tier or explicit natural-log distance scatters."""
    if name == "custom":
        values = (custom_sigma_ln_dl, custom_sigma_ln_da)
        if any(value is None for value in values):
            raise ValueError(
                "custom errors require CUSTOM_SIGMA_LN_DL and CUSTOM_SIGMA_LN_DA"
            )
        sigma_ln_dl, sigma_ln_da = map(float, values)
        if not (
            math.isfinite(sigma_ln_dl)
            and math.isfinite(sigma_ln_da)
            and sigma_ln_dl >= 0
            and sigma_ln_da >= 0
        ):
            raise ValueError(
                "custom natural-log distance scatters must be finite and non-negative"
            )
        return ResolvedDistanceErrors(
            name="custom",
            sigma_ln_dl=sigma_ln_dl,
            sigma_ln_da=sigma_ln_da,
            uvx_residual_dex=None,
        )

    scenario = get_error_scenario(name)
    return ResolvedDistanceErrors(
        name=scenario.name,
        sigma_ln_dl=scenario.sigma_ln_dl,
        sigma_ln_da=scenario.sarm_sigma_ln,
        uvx_residual_dex=scenario.uvx_residual_dex,
    )
