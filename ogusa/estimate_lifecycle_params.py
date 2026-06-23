"""
Joint SMM calibration of beta, chi_b, and chi_n parameters.

This module estimates the preference parameters that govern savings,
bequests, and labor supply in one steady-state SMM problem.  It is designed
as a standalone calibration layer; callers can wire the returned parameter
values into the standard OG-USA calibration flow after validating the fit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
import ogcore
from ogcore import SS
from ogcore.utils import Inequality

from ogusa import compute_moments, wealth

ogcore.config.VERBOSE = False
logger = logging.getLogger(__name__)

WeightingMethod = Literal["identity", "diagonal", "optimal"]
TailMethod = Literal["scaled_default", "flat"]
WealthProfileMoment = Literal["anchor_window", "level", "mean_normalized"]
MomentDistanceMethod = Literal["absolute", "relative"]
WEALTH_MOMENT_BIN_WEIGHTS = np.array(
    [0.25, 0.25, 0.20, 0.10, 0.10, 0.09, 0.01]
)
SAVINGS_RATE_DATA_LABEL = r"Gross savings rate $(S/Y)$"


@dataclass(frozen=True)
class LifecycleCalibrationConfig:
    """
    Configuration for the joint lifecycle preference calibration.

    The default labor age window is 20 through 79.  Wealth-profile moments
    use ages 21 through 79 and are normalized by the mean wealth level from
    ages 20 through 24.  chi_n is parameterized as a cubic B-spline in log
    space over all model ages, with chi_n_n_spline_knots basis functions.
    That gives 2*J + chi_n_n_spline_knots parameters and ~130 default
    moments when J=10 and chi_n_n_spline_knots=10 (30 parameters total).
    """

    min_age: int = 20
    max_age: int = 79
    wealth_profile_min_age: int = 21
    wealth_profile_max_age: int = 79
    wealth_anchor_min_age: int = 20
    wealth_anchor_max_age: int = 24
    estimate_chi_n_min_age: int = 20
    estimate_chi_n_max_age: int = 79
    chi_n_tail_method: TailMethod = "scaled_default"
    wealth_profile_moment: WealthProfileMoment = "anchor_window"
    include_labor_profile: bool = True
    include_wealth_profile: bool = True
    include_income_gini: bool = True
    include_savings_rate: bool = True
    include_wealth_distribution: bool = True
    include_inheritance_moments: bool = False
    macro_year: int = 2025
    cps_years: tuple[int, ...] = (2023, 2022)
    scf_yrs_list: tuple[int, ...] = (2019,)
    cps_directory: str | None = None
    scf_directory: str | None = None
    scf_web: bool = False
    bootstrap_iterations: int = 1000
    weighting_method: WeightingMethod = "identity"
    weighting_ridge: float = 1e-8
    n_starts: int = 3
    start_radius: float = 0.2
    dfols_rhoend: float = 1e-6
    dfols_maxfun: int | None = None
    bound_epsilon: float = 1e-4
    chi_n_n_spline_knots: int = 10
    chi_n_spline_degree: int = 3
    moment_distance_method: MomentDistanceMethod = "relative"
    moment_distance_floor: float = 1e-8
    use_ss_solver_restart: bool = True
    log_optimizer_progress: bool = True

    @property
    def moment_ages(self) -> np.ndarray:
        """Return age labels used in labor-profile moments."""
        return np.arange(self.min_age, self.max_age + 1)

    @property
    def wealth_profile_ages(self) -> np.ndarray:
        """Return age labels used in wealth-profile moments."""
        return np.arange(
            self.wealth_profile_min_age,
            self.wealth_profile_max_age + 1,
        )

    @property
    def wealth_anchor_ages(self) -> np.ndarray:
        """Return age labels used to normalize wealth-profile moments."""
        return np.arange(
            self.wealth_anchor_min_age, self.wealth_anchor_max_age + 1
        )

    @property
    def estimated_chi_n_ages(self) -> np.ndarray:
        """Return the age labels for directly estimated chi_n values."""
        return np.arange(
            self.estimate_chi_n_min_age,
            self.estimate_chi_n_max_age + 1,
        )

    def validate(self, p) -> None:
        """Validate age and dimension settings against an OG-Core spec."""
        starting_age = int(getattr(p, "starting_age", 20))
        ending_age = int(getattr(p, "ending_age", 100))
        model_ages = np.arange(starting_age, ending_age)
        min_model_age = int(model_ages[0])
        max_model_age = int(model_ages[-1])
        requested_ages = np.concatenate(
            [
                self.moment_ages,
                self.wealth_profile_ages,
                self.wealth_anchor_ages,
                self.estimated_chi_n_ages,
            ]
        )
        if requested_ages.min() < min_model_age:
            raise ValueError("Requested ages start before model ages.")
        if requested_ages.max() > max_model_age:
            raise ValueError("Requested ages extend beyond model ages.")
        if self.estimate_chi_n_max_age < self.estimate_chi_n_min_age:
            raise ValueError(
                "estimate_chi_n_max_age must be at least min age."
            )
        if self.max_age < self.min_age:
            raise ValueError("max_age must be at least min_age.")
        if self.wealth_profile_max_age < self.wealth_profile_min_age:
            raise ValueError(
                "wealth_profile_max_age must be at least min age."
            )
        if self.wealth_anchor_max_age < self.wealth_anchor_min_age:
            raise ValueError("wealth_anchor_max_age must be at least min age.")


@dataclass(frozen=True)
class MomentSet:
    """Named vector of moments."""

    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        object.__setattr__(self, "values", values)
        if len(self.names) != values.size:
            raise ValueError("Moment names and values have different lengths.")


@dataclass
class LifecycleCalibrationResult:
    """Container for joint lifecycle preference calibration results."""

    beta_annual: np.ndarray
    chi_b: np.ndarray
    chi_n: np.ndarray
    objective_value: float
    optimizer_result: object
    data_moments: MomentSet
    model_moments: MomentSet
    weighting_matrix: np.ndarray
    best_start_index: int = 0
    all_start_results: list = field(default_factory=list)

    @property
    def parameter_dict(self) -> dict[str, list[float]]:
        """Return calibrated values in OG-Core update_specifications format."""
        return {
            "beta_annual": self.beta_annual.tolist(),
            "chi_b": self.chi_b.tolist(),
            "chi_n": self.chi_n.tolist(),
        }


@dataclass
class SSSolutionCache:
    """Mutable cache for warm-starting repeated SS solves."""

    previous_output: dict | None = None
    use_ss_solver: bool = True

    def reset(self) -> None:
        """Clear the cached SS output."""
        self.previous_output = None


def _as_vector(values) -> np.ndarray:
    """Return values as a one-dimensional float array."""
    return np.asarray(values, dtype=float).reshape(-1)


def _lambdas(p) -> np.ndarray:
    """Return lifetime-income weights as a one-dimensional array."""
    return _as_vector(p.lambdas)


def _ss_chi_n(p) -> np.ndarray:
    """Return the steady-state chi_n age profile from a spec object."""
    chi_n = np.asarray(p.chi_n, dtype=float)
    if chi_n.ndim == 1:
        return chi_n.copy()
    return chi_n[-1, :].copy()


def _age_to_index(age: int, p) -> int:
    """Map an age label to its model age index."""
    return int(age) - int(getattr(p, "starting_age", 20))


def _age_indices(ages: np.ndarray, p) -> np.ndarray:
    """Map age labels to model indices."""
    return np.array([_age_to_index(age, p) for age in ages], dtype=int)


def _build_chi_n_spline_basis(
    ages: np.ndarray,
    n_basis: int,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a B-spline design matrix for the chi_n age profile.

    The spline is defined in log space: evaluating ``B @ gamma`` gives
    ``log(chi_n)`` at each age, so ``chi_n = exp(B @ gamma)`` is always
    positive regardless of the coefficient values.

    Args:
        ages: Age values at which to evaluate the basis (length N).
        n_basis: Number of B-spline basis functions (= number of free
            coefficients).  Must satisfy n_basis >= degree + 1.
        degree: Polynomial degree of the spline (default 3 = cubic).

    Returns:
        B: Design matrix of shape (N, n_basis).
        knots: Full knot vector used to construct the basis.
    """
    from scipy.interpolate import BSpline

    ages_f = np.asarray(ages, dtype=float)
    age_min, age_max = ages_f[0], ages_f[-1]
    n_internal = n_basis - degree - 1
    if n_internal < 0:
        raise ValueError(
            f"n_basis={n_basis} is too small for degree={degree}. "
            f"Need n_basis >= degree + 1 = {degree + 1}."
        )
    internal = (
        np.linspace(age_min, age_max, n_internal + 2)[1:-1]
        if n_internal > 0
        else np.array([], dtype=float)
    )
    knots = np.concatenate(
        [
            np.repeat(age_min, degree + 1),
            internal,
            np.repeat(age_max, degree + 1),
        ]
    )
    n_cols = len(knots) - degree - 1
    B = np.zeros((len(ages_f), n_cols))
    for i in range(n_cols):
        c = np.zeros(n_cols)
        c[i] = 1.0
        B[:, i] = BSpline(knots, c, degree)(ages_f)
    return B, knots


def _weighted_mean(values, weights) -> float:
    """Return a weighted mean after dropping nonfinite observations."""
    data = pd.DataFrame({"value": values, "weight": weights})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["weight"] > 0]
    if data.empty:
        return np.nan
    return float((data["value"] * data["weight"]).sum() / data["weight"].sum())


def _weighted_mean_by_age(
    data: pd.DataFrame,
    value_col: str,
    weight_col: str | None,
    ages: np.ndarray,
    age_col: str = "age",
) -> np.ndarray:
    """Compute weighted means by single-year age."""
    columns = [age_col, value_col]
    if weight_col is not None:
        columns.append(weight_col)
    age_data = data[columns].copy()
    age_data[age_col] = pd.to_numeric(age_data[age_col], errors="coerce")
    age_data[value_col] = pd.to_numeric(age_data[value_col], errors="coerce")
    age_data = age_data.replace([np.inf, -np.inf], np.nan).dropna()
    age_data = age_data[age_data[age_col].isin(ages)].copy()
    age_data[age_col] = age_data[age_col].astype(int)

    if weight_col is None:
        profile = age_data.groupby(age_col)[value_col].mean()
    else:
        age_data[weight_col] = pd.to_numeric(
            age_data[weight_col], errors="coerce"
        )
        age_data = age_data[age_data[weight_col] > 0].copy()
        age_data["weighted_value"] = age_data[value_col] * age_data[weight_col]
        by_age = age_data.groupby(age_col)[
            ["weighted_value", weight_col]
        ].sum()
        profile = by_age["weighted_value"] / by_age[weight_col]

    return profile.reindex(ages).to_numpy(dtype=float)


def _require_finite(values: np.ndarray, label: str) -> np.ndarray:
    """Validate that all values are finite."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains missing or nonfinite values.")
    return values


def _normalize_wealth_profile(
    profile: np.ndarray,
    anchor_profile: np.ndarray,
    method: WealthProfileMoment,
) -> np.ndarray:
    """Scale a wealth age profile according to the requested convention."""
    profile = np.asarray(profile, dtype=float)
    if method == "level":
        return profile
    if method == "mean_normalized":
        mean = np.nanmean(profile)
        if not np.isfinite(mean) or np.isclose(mean, 0.0):
            raise ValueError("Cannot normalize wealth profile with zero mean.")
        return profile / mean
    if method == "anchor_window":
        anchor_mean = np.nanmean(np.asarray(anchor_profile, dtype=float))
        if not np.isfinite(anchor_mean) or np.isclose(anchor_mean, 0.0):
            raise ValueError(
                "Cannot normalize wealth profile with zero anchor mean."
            )
        return profile / anchor_mean
    raise ValueError(f"Unsupported wealth profile moment: {method}")


def _wealth_distribution_moment_names() -> tuple[str, ...]:
    """Return names for the nine SCF/model wealth distribution moments."""
    return (
        "wealth_share_0_25",
        "wealth_share_25_50",
        "wealth_share_50_70",
        "wealth_share_70_80",
        "wealth_share_80_90",
        "wealth_share_90_99",
        "wealth_share_99_100",
        "wealth_gini",
        "wealth_var_log",
    )


def _model_wealth_distribution_moments(ss_output: dict, p) -> np.ndarray:
    """Compute the model moments matching wealth.compute_wealth_moments."""
    b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
    wealth_ineq = Inequality(b_sp1, p.omega_SS, _lambdas(p), p.S, p.J)
    return np.array(
        [
            1 - wealth_ineq.top_share(0.75),
            wealth_ineq.top_share(0.75) - wealth_ineq.top_share(0.50),
            wealth_ineq.top_share(0.50) - wealth_ineq.top_share(0.30),
            wealth_ineq.top_share(0.30) - wealth_ineq.top_share(0.20),
            wealth_ineq.top_share(0.20) - wealth_ineq.top_share(0.10),
            wealth_ineq.top_share(0.10) - wealth_ineq.top_share(0.01),
            wealth_ineq.top_share(0.01),
            wealth_ineq.gini(),
            wealth_ineq.var_of_logs(),
        ],
        dtype=float,
    )


def load_cps_hours_data(
    cps_years: tuple[int, ...] = (2023, 2022),
    cps_directory: str | None = None,
) -> pd.DataFrame:
    """Load packaged CPS ASEC hours data used for labor moments."""
    if cps_directory is None:
        cps_directory = compute_moments.CPS_DATA_DIR
    data = []
    for year in cps_years:
        path = f"{cps_directory}/cps_asec_hours_{year}.csv"
        data.append(pd.read_csv(path))
    return pd.concat(data, ignore_index=True)


def load_scf_wealth_data(config: LifecycleCalibrationConfig) -> pd.DataFrame:
    """Load SCF wealth data with ages for wealth moments."""
    return wealth.get_wealth_data(
        scf_yrs_list=list(config.scf_yrs_list),
        web=config.scf_web,
        directory=config.scf_directory,
        include_age=True,
    )


def labor_profile_from_cps(
    cps: pd.DataFrame,
    config: LifecycleCalibrationConfig,
) -> np.ndarray:
    """Compute labor supply by age from CPS data."""
    cps = cps.copy()
    if "hours" not in cps:
        if "hours_per_week" in cps:
            cps["hours"] = cps["hours_per_week"]
        else:
            raise ValueError("CPS data must include hours or hours_per_week.")
    weight_col = None
    for possible_weight in ("weight", "wtsupp", "s006", "wgt"):
        if possible_weight in cps:
            weight_col = possible_weight
            break
    hours = _weighted_mean_by_age(
        cps,
        "hours",
        weight_col,
        config.moment_ages,
    )
    labor = hours / ((24 - 8) * 7)  # scale so fraction of time endowment
    return _require_finite(labor, "labor profile")


def wealth_profile_from_scf(
    scf: pd.DataFrame,
    config: LifecycleCalibrationConfig,
) -> np.ndarray:
    """Compute net-wealth age profile moments from SCF data."""
    profile = _weighted_mean_by_age(
        scf,
        "networth_infadj",
        "wgt",
        config.wealth_profile_ages,
    )
    anchor_profile = _weighted_mean_by_age(
        scf,
        "networth_infadj",
        "wgt",
        config.wealth_anchor_ages,
    )
    profile = _normalize_wealth_profile(
        profile,
        anchor_profile,
        config.wealth_profile_moment,
    )
    return _require_finite(profile, "wealth profile")


def income_gini_data_moment(
    income_year: int | None = None,
) -> float:
    """Compute the before-tax income Gini data moment."""
    moments = compute_moments._taxcalc_cps_income_ginis(income_year)
    return float(moments["Gini coefficient, income"])


def savings_rate_data_moment(macro_year: int = 2025) -> float:
    """Compute the aggregate savings-rate data moment."""
    moments = compute_moments.get_macro_moments(year=macro_year)
    return float(moments[SAVINGS_RATE_DATA_LABEL])


def model_savings_rate_moment(ss_output: dict, p) -> float:
    """Compute the model gross aggregate savings rate."""
    output = float(ss_output["Y"])
    if np.isclose(output, 0.0):
        raise ValueError("Cannot compute savings rate with zero output.")
    growth = (1 + p.g_n_ss) * np.exp(p.g_y)
    gross_saving_flow = (growth - 1.0) * float(ss_output["B"]) + (
        p.delta * float(ss_output["K_d"])
    )
    return gross_saving_flow / output


def compute_inheritance_moments_from_scf(
    scf: pd.DataFrame,
    amount_col: str,
    received_col: str | None = None,
    weight_col: str = "wgt",
    networth_col: str = "networth_infadj",
) -> MomentSet:
    """
    Compute optional inherited-transfer moments from full SCF extracts.

    The trimmed SCF files packaged in OG-USA do not include the inheritance
    variables needed here.  Pass a full SCF extract with the relevant Section X
    variables and specify the amount and receipt indicator columns.
    """
    if amount_col not in scf:
        raise ValueError(f"SCF data are missing {amount_col}.")
    if weight_col not in scf:
        raise ValueError(f"SCF data are missing {weight_col}.")

    data = scf.copy()
    data[amount_col] = pd.to_numeric(data[amount_col], errors="coerce")
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[amount_col, weight_col]
    )
    data = data[data[weight_col] > 0].copy()
    if received_col is None:
        received = data[amount_col] > 0
    else:
        if received_col not in data:
            raise ValueError(f"SCF data are missing {received_col}.")
        received = pd.to_numeric(data[received_col], errors="coerce") > 0

    names = ["inheritance_received_rate"]
    values = [_weighted_mean(received.astype(float), data[weight_col])]

    recipients = data[received].copy()
    if not recipients.empty:
        names.append("inheritance_amount_conditional_mean")
        values.append(
            _weighted_mean(recipients[amount_col], recipients[weight_col])
        )

    if networth_col in data:
        data[networth_col] = pd.to_numeric(data[networth_col], errors="coerce")
        positive_networth = data[networth_col] > 0
        ratio_data = data[positive_networth].copy()
        if not ratio_data.empty:
            names.append("inheritance_to_networth_mean")
            values.append(
                _weighted_mean(
                    ratio_data[amount_col] / ratio_data[networth_col],
                    ratio_data[weight_col],
                )
            )

    return MomentSet(tuple(names), np.asarray(values, dtype=float))


def compute_data_moments(
    p,
    config: LifecycleCalibrationConfig | None = None,
    cps: pd.DataFrame | None = None,
    scf: pd.DataFrame | None = None,
    income_year: int | None = None,
    inheritance_moments: MomentSet | None = None,
    savings_rate: float | None = None,
) -> MomentSet:
    """Compute the stacked data moment vector for the SMM objective."""
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    names: list[str] = []
    values: list[float] = []

    if config.include_labor_profile:
        if cps is None:
            cps = load_cps_hours_data(config.cps_years, config.cps_directory)
        labor = labor_profile_from_cps(cps, config)
        names.extend(f"labor_supply_age_{age}" for age in config.moment_ages)
        values.extend(labor)

    if config.include_wealth_profile:
        if scf is None:
            scf = load_scf_wealth_data(config)
        wealth_profile = wealth_profile_from_scf(scf, config)
        names.extend(
            f"net_wealth_age_{age}" for age in config.wealth_profile_ages
        )
        values.extend(wealth_profile)

    if config.include_income_gini:
        names.append("income_gini")
        values.append(income_gini_data_moment(income_year=income_year))

    if config.include_savings_rate:
        if savings_rate is None:
            savings_rate = savings_rate_data_moment(config.macro_year)
        names.append("savings_rate")
        values.append(float(savings_rate))

    if config.include_wealth_distribution:
        if scf is None:
            scf = load_scf_wealth_data(config)
        wealth_dist = wealth.compute_wealth_moments(
            scf.copy(),
            WEALTH_MOMENT_BIN_WEIGHTS,
        )
        names.extend(_wealth_distribution_moment_names())
        values.extend(wealth_dist)

    if config.include_inheritance_moments:
        if inheritance_moments is None:
            raise ValueError(
                "inheritance_moments must be supplied when "
                "include_inheritance_moments is True."
            )
        names.extend(inheritance_moments.names)
        values.extend(inheritance_moments.values)

    return MomentSet(tuple(names), np.asarray(values, dtype=float))


def compute_model_moments(
    ss_output: dict,
    p,
    config: LifecycleCalibrationConfig | None = None,
    inheritance_moments: MomentSet | None = None,
) -> MomentSet:
    """Compute model moments in the same order as compute_data_moments."""
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    lambdas = _lambdas(p)
    names: list[str] = []
    values: list[float] = []

    if config.include_labor_profile:
        age_idx = _age_indices(config.moment_ages, p)
        n = np.asarray(ss_output["n"], dtype=float)
        labor = (n[age_idx, :] * lambdas.reshape(1, p.J)).sum(axis=1)
        names.extend(f"labor_supply_age_{age}" for age in config.moment_ages)
        values.extend(labor)

    if config.include_wealth_profile:
        wealth_age_idx = _age_indices(config.wealth_profile_ages, p)
        anchor_age_idx = _age_indices(config.wealth_anchor_ages, p)
        b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
        factor = float(ss_output.get("factor", 1.0))
        wealth_profile = (
            b_sp1[wealth_age_idx, :] * factor * lambdas.reshape(1, p.J)
        ).sum(axis=1)
        anchor_profile = (
            b_sp1[anchor_age_idx, :] * factor * lambdas.reshape(1, p.J)
        ).sum(axis=1)
        wealth_profile = _normalize_wealth_profile(
            wealth_profile,
            anchor_profile,
            config.wealth_profile_moment,
        )
        names.extend(
            f"net_wealth_age_{age}" for age in config.wealth_profile_ages
        )
        values.extend(wealth_profile)

    if config.include_income_gini:
        income = np.asarray(ss_output["before_tax_income"], dtype=float)
        income_ineq = Inequality(income, p.omega_SS, lambdas, p.S, p.J)
        names.append("income_gini")
        values.append(income_ineq.gini())

    if config.include_savings_rate:
        names.append("savings_rate")
        values.append(model_savings_rate_moment(ss_output, p))

    if config.include_wealth_distribution:
        names.extend(_wealth_distribution_moment_names())
        values.extend(_model_wealth_distribution_moments(ss_output, p))

    if config.include_inheritance_moments:
        if inheritance_moments is None:
            raise ValueError(
                "Model inheritance moments must be supplied when "
                "include_inheritance_moments is True."
            )
        names.extend(inheritance_moments.names)
        values.extend(inheritance_moments.values)

    return MomentSet(tuple(names), np.asarray(values, dtype=float))


def build_chi_n_profile(
    estimated_chi_n: np.ndarray,
    base_chi_n: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
) -> np.ndarray:
    """Build a full-S chi_n profile from directly estimated values."""
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    full_chi_n = np.asarray(base_chi_n, dtype=float).reshape(-1).copy()
    if full_chi_n.size != p.S:
        raise ValueError("base_chi_n length must equal p.S.")

    est_values = np.asarray(estimated_chi_n, dtype=float).reshape(-1)
    est_ages = config.estimated_chi_n_ages
    if est_values.size != est_ages.size:
        raise ValueError("estimated_chi_n length does not match config ages.")

    est_idx = _age_indices(est_ages, p)
    full_chi_n[est_idx] = est_values
    tail_start = int(est_idx[-1] + 1)
    if tail_start >= p.S:
        return full_chi_n

    if config.chi_n_tail_method == "flat":
        full_chi_n[tail_start:] = est_values[-1]
    elif config.chi_n_tail_method == "scaled_default":
        base_anchor = base_chi_n[tail_start - 1]
        if np.isclose(base_anchor, 0.0):
            scale = 1.0
        else:
            scale = est_values[-1] / base_anchor
        full_chi_n[tail_start:] = base_chi_n[tail_start:] * scale
    else:
        raise ValueError(
            f"Unsupported chi_n tail method: {config.chi_n_tail_method}"
        )

    return full_chi_n


def pack_lifecycle_params(
    beta_annual: np.ndarray,
    chi_b: np.ndarray,
    chi_n: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    transform: bool = True,
) -> np.ndarray:
    """Pack natural lifecycle parameters into an optimizer vector.

    chi_n (length p.S) is projected onto the B-spline basis in log space.
    The returned vector layout is [beta_trans, chi_b_trans, gamma] where
    gamma holds the chi_n spline coefficients (unconstrained when transform
    is True, since the log-space spline already enforces positivity via exp).
    """
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    beta_annual = _as_vector(beta_annual)
    chi_b = _as_vector(chi_b)
    chi_n = _as_vector(chi_n)

    starting_age = int(getattr(p, "starting_age", 20))
    all_ages = np.arange(starting_age, starting_age + p.S)
    B, _ = _build_chi_n_spline_basis(
        all_ages, config.chi_n_n_spline_knots, config.chi_n_spline_degree
    )

    if not transform:
        # Project chi_n directly (no log); coefficients are in natural space.
        gamma, _, _, _ = np.linalg.lstsq(B, chi_n, rcond=None)
        return np.concatenate([beta_annual, chi_b, gamma])

    if np.any((beta_annual <= 0) | (beta_annual >= 1)):
        raise ValueError(
            "beta_annual values must be strictly between 0 and 1."
        )
    if np.any(chi_b <= 0):
        raise ValueError("chi_b values must be positive.")
    if np.any(chi_n <= 0):
        raise ValueError("chi_n values must be positive.")

    beta_trans = np.log(beta_annual / (1 - beta_annual))
    chi_b_trans = np.log(chi_b)
    # Fit spline to log(chi_n) over all model ages; gamma is unconstrained.
    gamma, _, _, _ = np.linalg.lstsq(B, np.log(chi_n), rcond=None)
    return np.concatenate([beta_trans, chi_b_trans, gamma])


def unpack_lifecycle_params(
    theta: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    base_chi_n: np.ndarray | None = None,
    transform: bool = True,
) -> dict[str, np.ndarray]:
    """Unpack an optimizer vector into natural lifecycle parameters.

    The chi_n block of theta holds B-spline coefficients (gamma).  When
    transform=True the spline operates in log space and chi_n = exp(B @ gamma).
    The base_chi_n argument is accepted for backward compatibility but is no
    longer used; the spline covers all model ages directly.
    """
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)

    theta = _as_vector(theta)
    n_beta = p.J
    n_chi_b = p.J
    n_gamma = config.chi_n_n_spline_knots
    expected = n_beta + n_chi_b + n_gamma
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}.")

    beta_raw = theta[:n_beta]
    chi_b_raw = theta[n_beta : n_beta + n_chi_b]
    gamma = theta[n_beta + n_chi_b :]

    starting_age = int(getattr(p, "starting_age", 20))
    all_ages = np.arange(starting_age, starting_age + p.S)
    B, _ = _build_chi_n_spline_basis(
        all_ages, config.chi_n_n_spline_knots, config.chi_n_spline_degree
    )

    if transform:
        beta_annual = 1 / (1 + np.exp(-beta_raw))
        chi_b = np.exp(chi_b_raw)
        chi_n = np.exp(B @ gamma)
    else:
        beta_annual = beta_raw
        chi_b = chi_b_raw
        chi_n = B @ gamma

    return {
        "beta_annual": beta_annual,
        "chi_b": chi_b,
        "chi_n": chi_n,
    }


def initial_lifecycle_theta(
    p,
    config: LifecycleCalibrationConfig | None = None,
    transform: bool = True,
) -> np.ndarray:
    """Build an optimizer vector from the current spec values."""
    if config is None:
        config = LifecycleCalibrationConfig()
    return pack_lifecycle_params(
        p.beta_annual,
        p.chi_b,
        _ss_chi_n(p),
        p,
        config,
        transform=transform,
    )


def apply_lifecycle_params(
    p,
    beta_annual: np.ndarray,
    chi_b: np.ndarray,
    chi_n: np.ndarray,
) -> None:
    """Apply natural lifecycle parameters to a spec object in-place."""
    p.update_specifications(
        {
            "beta_annual": _as_vector(beta_annual).tolist(),
            "chi_b": _as_vector(chi_b).tolist(),
            "chi_n": _as_vector(chi_n).tolist(),
        }
    )


def solve_ss_with_cache(
    p,
    client=None,
    ss_cache: SSSolutionCache | None = None,
) -> dict:
    """
    Solve SS, optionally warm-starting from the previous SS output.

    The direct SS_solver path keeps p.baseline unchanged, so baseline solves
    still update the model scaling factor.  If the warm start fails, fall back
    to SS.run_SS and refresh the cache with that solution.
    """
    use_cache = (
        ss_cache is not None
        and ss_cache.use_ss_solver
        and ss_cache.previous_output is not None
    )
    if use_cache:
        previous = ss_cache.previous_output
        try:
            ig_baseline = (
                previous.get("I_g")
                if getattr(p, "baseline_spending", False)
                else None
            )
            ss_output = SS.SS_solver(
                previous["b_sp1"],
                previous["n"],
                float(previous["r_p"]),
                float(previous["r"]),
                float(previous["w"]),
                previous["p_m"],
                float(previous["Y"]),
                previous["BQ"],
                float(previous["TR"]),
                ig_baseline,
                float(previous["factor"]),
                p,
                client,
            )
        except (
            AssertionError,
            FloatingPointError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            ss_output = SS.run_SS(p, client=client)
    else:
        ss_output = SS.run_SS(p, client=client)

    if ss_cache is not None:
        ss_cache.previous_output = ss_output
    return ss_output


def weighting_matrix(
    moment_count: int,
    method: WeightingMethod = "identity",
    bootstrap_moments: np.ndarray | None = None,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Construct a weighting matrix for the SMM objective."""
    if method == "identity":
        return np.eye(moment_count)
    if bootstrap_moments is None:
        raise ValueError("bootstrap_moments are required for this method.")

    boot = np.asarray(bootstrap_moments, dtype=float)
    if boot.ndim != 2 or boot.shape[1] != moment_count:
        raise ValueError("bootstrap_moments must be n x moment_count.")
    vcv = np.cov(boot.T)
    vcv = np.atleast_2d(vcv)
    if method == "diagonal":
        diag = np.diag(vcv).copy()
        diag[diag < ridge] = ridge
        return np.diag(1 / diag)
    if method == "optimal":
        vcv = vcv + ridge * np.eye(moment_count)
        return np.linalg.pinv(vcv)
    raise ValueError(f"Unsupported weighting method: {method}")


def bootstrap_data_moments(
    p,
    config: LifecycleCalibrationConfig | None = None,
    cps: pd.DataFrame | None = None,
    scf: pd.DataFrame | None = None,
    seed: int | None = None,
    savings_rate: float | None = None,
) -> np.ndarray:
    """Bootstrap the data moments available from CPS and SCF microdata."""
    if config is None:
        config = LifecycleCalibrationConfig()
    if cps is None and config.include_labor_profile:
        cps = load_cps_hours_data(config.cps_years, config.cps_directory)
    if scf is None and (
        config.include_wealth_profile or config.include_wealth_distribution
    ):
        scf = load_scf_wealth_data(config)

    point_moments = compute_data_moments(
        p,
        config,
        cps=cps,
        scf=scf,
        savings_rate=savings_rate,
    )
    resampled_config = replace(
        config,
        include_income_gini=False,
        include_savings_rate=False,
        include_inheritance_moments=False,
    )
    rng = np.random.default_rng(seed)
    boot = np.zeros((config.bootstrap_iterations, point_moments.values.size))
    for i in range(config.bootstrap_iterations):
        cps_boot = None
        scf_boot = None
        if cps is not None:
            cps_boot = cps.iloc[
                rng.integers(0, len(cps), size=len(cps))
            ].reset_index(drop=True)
        if scf is not None:
            scf_boot = scf.iloc[
                rng.integers(0, len(scf), size=len(scf))
            ].reset_index(drop=True)
        resampled_moments = compute_data_moments(
            p,
            resampled_config,
            cps=cps_boot,
            scf=scf_boot,
        )
        resampled_values = dict(
            zip(resampled_moments.names, resampled_moments.values)
        )
        boot[i, :] = [
            resampled_values.get(name, value)
            for name, value in zip(point_moments.names, point_moments.values)
        ]
    return boot


def smm_distance(
    model_moments: MomentSet,
    data_moments: MomentSet,
    W: np.ndarray,
    method: MomentDistanceMethod = "relative",
    floor: float = 1e-8,
) -> float:
    """Compute the quadratic SMM distance.

    Args:
        model_moments: Simulated model moments.
        data_moments: Empirical data moments.
        W: Positive semi-definite weighting matrix.
        method: ``"relative"`` divides each residual by ``|data_moment|``
            (floored at ``floor``) so all moments are on a comparable
            percentage-deviation scale.  ``"absolute"`` uses raw levels.
        floor: Minimum absolute value used as the denominator when
            method="relative", preventing division by zero for near-zero
            moments such as bottom wealth shares.
    """
    if model_moments.names != data_moments.names:
        raise ValueError("Model and data moments are not aligned.")
    m = model_moments.values
    d = data_moments.values
    if method == "relative":
        safe_denom = np.where(np.abs(d) > floor, np.abs(d), floor)
        diff = (m - d) / safe_denom
    else:
        diff = m - d
    if not np.all(np.isfinite(diff)):
        return np.inf
    return float(diff.T @ W @ diff)


def smm_objective(
    theta: np.ndarray,
    data_moments: MomentSet,
    W: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    base_chi_n: np.ndarray | None = None,
    client=None,
    transform: bool = True,
    ss_cache: SSSolutionCache | None = None,
) -> float:
    """Evaluate the joint lifecycle SMM objective."""
    if config is None:
        config = LifecycleCalibrationConfig()
    if base_chi_n is None:
        base_chi_n = _ss_chi_n(p)
    try:
        params = unpack_lifecycle_params(
            theta,
            p,
            config,
            base_chi_n=base_chi_n,
            transform=transform,
        )
        apply_lifecycle_params(
            p,
            params["beta_annual"],
            params["chi_b"],
            params["chi_n"],
        )
        ss_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
        model_moments = compute_model_moments(ss_output, p, config)
        distance = smm_distance(
            model_moments,
            data_moments,
            W,
            method=config.moment_distance_method,
            floor=config.moment_distance_floor,
        )
    except (
        AssertionError,
        FloatingPointError,
        ValueError,
        RuntimeError,
        KeyError,
    ):
        distance = np.inf
    if not np.isfinite(distance):
        return 1e30
    return distance


def _apply_weight_sqrt(residuals: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return L.T @ residuals where W = L @ L.T, so ||result||^2 = r^T W r.

    For identity W this is a no-op.  For diagonal W it multiplies element-wise
    by sqrt of the diagonal.  For a full positive definite W it uses the
    Cholesky factor, falling back to the diagonal if W is not PD.
    """
    if np.allclose(W, np.eye(len(W))):
        return residuals
    try:
        L = np.linalg.cholesky(W)
        return L.T @ residuals
    except np.linalg.LinAlgError:
        return np.sqrt(np.maximum(np.diag(W), 0.0)) * residuals


def smm_residual(
    theta: np.ndarray,
    data_moments: MomentSet,
    W: np.ndarray,
    p,
    config: LifecycleCalibrationConfig,
    base_chi_n: np.ndarray | None = None,
    client=None,
    transform: bool = True,
    ss_cache: SSSolutionCache | None = None,
) -> np.ndarray:
    """Return the weighted moment residual vector for DFO-LS.

    DFO-LS minimises ``||f(x)||^2``.  Pre-multiplying by ``W^{1/2}`` ensures
    that ``||result||^2 == r^T W r``, matching the SMM objective.  On solver
    failure returns a large constant vector so DFO-LS avoids that region.
    """
    n_moments = data_moments.values.size
    try:
        params = unpack_lifecycle_params(
            theta, p, config, base_chi_n=base_chi_n, transform=transform
        )
        apply_lifecycle_params(
            p, params["beta_annual"], params["chi_b"], params["chi_n"]
        )
        ss_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
        model_moments = compute_model_moments(ss_output, p, config)
        m = model_moments.values
        d = data_moments.values
        if config.moment_distance_method == "relative":
            safe_denom = np.where(
                np.abs(d) > config.moment_distance_floor,
                np.abs(d),
                config.moment_distance_floor,
            )
            residuals = (m - d) / safe_denom
        else:
            residuals = m - d
        weighted = _apply_weight_sqrt(residuals, W)
        if not np.all(np.isfinite(weighted)):
            return np.full(n_moments, 1e15)
        return weighted
    except (
        AssertionError,
        FloatingPointError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return np.full(n_moments, 1e15)


def _extract_dfols_bounds(
    p,
    config: LifecycleCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build DFO-LS bound arrays from the ParamTools validators in p.

    Bounds are read from ``p._data[param]['validators']['range']`` and then
    mapped into the same transformed space used by the optimizer:

    * ``beta_annual`` — logit transform; validator min/max become the
      logit of the natural bounds (with ``bound_epsilon`` as a floor so
      log(0) is avoided).
    * ``chi_b`` — log transform; same epsilon floor on the natural minimum.
    * ``chi_n`` spline ``gamma`` — the B-spline has the convex-hull
      property: if every coefficient satisfies
      ``log(chi_n_min) ≤ γᵢ ≤ log(chi_n_max)`` then the evaluated
      ``chi_n = exp(B @ gamma)`` is guaranteed to stay within
      ``[chi_n_min, chi_n_max]`` at every model age.

    Returns:
        lower: 1-D array of length n_beta + n_chi_b + n_gamma.
        upper: 1-D array of the same length.
    """
    eps = config.bound_epsilon

    def _range(param_name):
        validators = p._data.get(param_name, {}).get("validators", {})
        r = validators.get("range", {})
        lo = float(r.get("min", -np.inf))
        hi = float(r.get("max", np.inf))
        return lo, hi

    # --- beta_annual (logit space) ---
    b_lo_nat, b_hi_nat = _range("beta_annual")
    b_lo_nat = max(b_lo_nat, eps)
    b_hi_nat = min(b_hi_nat, 1.0 - eps)
    beta_lo = np.full(p.J, np.log(b_lo_nat / (1.0 - b_lo_nat)))
    beta_hi = np.full(p.J, np.log(b_hi_nat / (1.0 - b_hi_nat)))

    # --- chi_b (log space) ---
    cb_lo_nat, cb_hi_nat = _range("chi_b")
    cb_lo_nat = max(cb_lo_nat, eps)
    chi_b_lo = np.full(p.J, np.log(cb_lo_nat))
    chi_b_hi = np.full(p.J, np.log(cb_hi_nat))

    # --- chi_n spline coefficients (log space, convex-hull argument) ---
    cn_lo_nat, cn_hi_nat = _range("chi_n")
    cn_lo_nat = max(cn_lo_nat, eps)
    gamma_lo = np.full(config.chi_n_n_spline_knots, np.log(cn_lo_nat))
    gamma_hi = np.full(config.chi_n_n_spline_knots, np.log(cn_hi_nat))

    lower = np.concatenate([beta_lo, chi_b_lo, gamma_lo])
    upper = np.concatenate([beta_hi, chi_b_hi, gamma_hi])
    return lower, upper


def _generate_starts(
    theta0: np.ndarray,
    n_starts: int,
    radius: float,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Return n_starts starting vectors centred on theta0.

    The first element is always theta0 itself (the warm start from prior
    calibration values).  Additional points are drawn by adding additive
    noise scaled by ``radius * max(|theta0_i|, 1.0)`` in each dimension,
    so near-zero transformed parameters still receive meaningful perturbations.
    All generated points are clipped to ``[lower, upper]`` when provided.
    """
    def _clip(x):
        if lower is not None:
            x = np.maximum(x, lower)
        if upper is not None:
            x = np.minimum(x, upper)
        return x

    starts = [_clip(theta0.copy())]
    if n_starts <= 1:
        return starts
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(theta0), 1.0)
    for _ in range(n_starts - 1):
        noise = rng.uniform(-radius, radius, size=theta0.shape) * scale
        starts.append(_clip(theta0 + noise))
    return starts


def estimate_lifecycle_params(
    p,
    config: LifecycleCalibrationConfig | None = None,
    theta0: np.ndarray | None = None,
    data_moments: MomentSet | None = None,
    W: np.ndarray | None = None,
    bootstrap_moments: np.ndarray | None = None,
    client=None,
    transform: bool = True,
    savings_rate: float | None = None,
) -> LifecycleCalibrationResult:
    """Estimate beta_annual, chi_b, and chi_n jointly by SMM using DFO-LS."""
    import dfols

    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    base_chi_n = _ss_chi_n(p)
    if theta0 is None:
        theta0 = initial_lifecycle_theta(p, config, transform=transform)
    if data_moments is None:
        data_moments = compute_data_moments(
            p,
            config,
            savings_rate=savings_rate,
        )
    if W is None:
        W = weighting_matrix(
            data_moments.values.size,
            method=config.weighting_method,
            bootstrap_moments=bootstrap_moments,
            ridge=config.weighting_ridge,
        )

    lower, upper = _extract_dfols_bounds(p, config)
    starts = _generate_starts(
        theta0,
        config.n_starts,
        config.start_radius,
        lower=lower,
        upper=upper,
    )
    best_result = None
    best_obj = np.inf
    best_start_index = 0
    all_start_results = []

    for i, theta_start in enumerate(starts):
        logger.info(
            "Lifecycle SMM: DFO-LS start %d/%d", i + 1, config.n_starts
        )
        ss_cache = SSSolutionCache(use_ss_solver=config.use_ss_solver_restart)

        def residual_fn(theta, _cache=ss_cache):
            return smm_residual(
                theta,
                data_moments,
                W,
                p,
                config,
                base_chi_n=base_chi_n,
                client=client,
                transform=transform,
                ss_cache=_cache,
            )

        maxfun = config.dfols_maxfun
        dfols_result = dfols.solve(
            residual_fn,
            theta_start,
            bounds=(lower, upper),
            rhoend=config.dfols_rhoend,
            maxfun=maxfun,
            do_logging=config.log_optimizer_progress,
            print_progress=False,
        )
        all_start_results.append(dfols_result)
        obj = float(dfols_result.cost)
        logger.info(
            "Lifecycle SMM: start %d/%d finished — objective=%.6e, "
            "evals=%d, msg=%s",
            i + 1,
            config.n_starts,
            obj,
            dfols_result.nf,
            dfols_result.msg,
        )
        if obj < best_obj:
            best_obj = obj
            best_result = dfols_result
            best_start_index = i

    params = unpack_lifecycle_params(
        best_result.x,
        p,
        config,
        base_chi_n=base_chi_n,
        transform=transform,
    )
    apply_lifecycle_params(
        p,
        params["beta_annual"],
        params["chi_b"],
        params["chi_n"],
    )
    ss_cache = SSSolutionCache(use_ss_solver=config.use_ss_solver_restart)
    ss_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
    model_moments = compute_model_moments(ss_output, p, config)

    return LifecycleCalibrationResult(
        beta_annual=params["beta_annual"],
        chi_b=params["chi_b"],
        chi_n=params["chi_n"],
        objective_value=best_obj,
        optimizer_result=best_result,
        data_moments=data_moments,
        model_moments=model_moments,
        weighting_matrix=W,
        best_start_index=best_start_index,
        all_start_results=all_start_results,
    )


def compute_parameter_vcv(
    theta_hat: np.ndarray,
    W: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    base_chi_n: np.ndarray | None = None,
    h: float = 1e-4,
    client=None,
    transform: bool = True,
) -> np.ndarray:
    """Compute a numerical GMM parameter VCV matrix."""
    if config is None:
        config = LifecycleCalibrationConfig()
    if base_chi_n is None:
        base_chi_n = _ss_chi_n(p)
    ss_cache = SSSolutionCache(use_ss_solver=config.use_ss_solver_restart)
    theta_hat = _as_vector(theta_hat)
    params = unpack_lifecycle_params(
        theta_hat,
        p,
        config,
        base_chi_n=base_chi_n,
        transform=transform,
    )
    apply_lifecycle_params(
        p,
        params["beta_annual"],
        params["chi_b"],
        params["chi_n"],
    )
    ss_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
    base_moments = compute_model_moments(ss_output, p, config)
    deriv = np.zeros((base_moments.values.size, theta_hat.size))

    for i in range(theta_hat.size):
        step = h * max(1.0, abs(theta_hat[i]))
        high = theta_hat.copy()
        low = theta_hat.copy()
        high[i] += step
        low[i] -= step

        high_params = unpack_lifecycle_params(
            high,
            p,
            config,
            base_chi_n=base_chi_n,
            transform=transform,
        )
        apply_lifecycle_params(
            p,
            high_params["beta_annual"],
            high_params["chi_b"],
            high_params["chi_n"],
        )
        high_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
        high_moments = compute_model_moments(high_output, p, config)

        low_params = unpack_lifecycle_params(
            low,
            p,
            config,
            base_chi_n=base_chi_n,
            transform=transform,
        )
        apply_lifecycle_params(
            p,
            low_params["beta_annual"],
            low_params["chi_b"],
            low_params["chi_n"],
        )
        low_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
        low_moments = compute_model_moments(low_output, p, config)
        deriv[:, i] = (high_moments.values - low_moments.values) / (2 * step)

    return np.linalg.pinv(deriv.T @ W @ deriv)
