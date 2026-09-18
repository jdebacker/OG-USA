"""
Nested calibration of household preference parameters in OG-USA.

This module implements the calibration path described in
``LIFECYCLE_CALIBRATION_PLAN.md``.  Phase 2 (this file's first block)
provides a household-only steady-state solve: given prices, taxes, bequests,
transfers, and the income scaling factor from a general-equilibrium
steady-state solution, it re-solves only the household Euler equations for
every lifetime-income type.  That is the inner workhorse for the ``chi_n``
inversion and the ``beta`` / ``chi_b`` fixed point in later phases, which
would otherwise need a full general-equilibrium solve on every evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ogcore import SS, aggregates as aggr, household

logger = logging.getLogger(__name__)

_HOUSEHOLD_INPUT_KEYS = ("p_i", "p_tilde", "bq", "rm", "tr", "ubi")


@dataclass(frozen=True)
class HouseholdEnvironment:
    """
    Everything the household block takes as given in the steady state.

    Attributes:
        r_p: return on the household portfolio.
        w: wage rate.
        p_tilde: composite consumption good price.
        p_i: consumption good prices, length I.
        bq: bequests received by age and type, shape (S, J).
        rm: remittances received by age and type, shape (S, J).
        tr: government transfers received by age and type, shape (S, J).
        ubi: universal basic income by age and type, shape (S, J).
        factor: scaling factor from model units to dollars.
    """

    r_p: float
    w: float
    p_tilde: float
    p_i: np.ndarray
    bq: np.ndarray
    rm: np.ndarray
    tr: np.ndarray
    ubi: np.ndarray
    factor: float

    @classmethod
    def from_ss_output(cls, ss_output: dict, p) -> "HouseholdEnvironment":
        """Build the environment from an OG-Core steady-state output dict.

        The output dictionary written by ``SS.run_SS`` already carries the
        household-level arrays (``bq``, ``rm``, ``tr``, ``ubi``) and prices
        (``p_i``, ``p_tilde``).  When any of those are missing, for example
        from an older pickle, they are rebuilt from the aggregates ``BQ``,
        ``RM``, ``TR``, ``Y``, and ``p_m`` with the same OG-Core helpers the
        steady-state inner loop uses.
        """
        factor = float(ss_output["factor"])
        if all(key in ss_output for key in _HOUSEHOLD_INPUT_KEYS):
            p_i = np.atleast_1d(np.asarray(ss_output["p_i"], dtype=float))
            p_tilde = float(np.squeeze(ss_output["p_tilde"]))
            bq = np.asarray(ss_output["bq"], dtype=float)
            rm = np.asarray(ss_output["rm"], dtype=float)
            tr = np.asarray(ss_output["tr"], dtype=float)
            ubi = np.asarray(ss_output["ubi"], dtype=float)
        else:
            p_m = np.atleast_1d(np.asarray(ss_output["p_m"], dtype=float))
            p_i = _consumption_good_prices(p_m, p)
            p_tilde = float(
                np.squeeze(aggr.get_ptilde(p_i, p.tau_c[-1, :], p.alpha_c))
            )
            BQ = np.asarray(ss_output["BQ"], dtype=float)
            if "RM" in ss_output:
                RM = np.asarray(ss_output["RM"], dtype=float)
            else:
                RM = np.asarray(
                    aggr.get_RM(float(ss_output["Y"]), p, "SS"), dtype=float
                )
            bq = np.asarray(household.get_bq(BQ, None, p, "SS"), dtype=float)
            rm = np.asarray(household.get_rm(RM, None, p, "SS"), dtype=float)
            tr = np.asarray(
                household.get_tr(float(ss_output["TR"]), None, p, "SS"),
                dtype=float,
            )
            ubi = np.asarray(p.ubi_nom_array[-1, :, :], dtype=float) / factor
        return cls(
            r_p=float(ss_output["r_p"]),
            w=float(ss_output["w"]),
            p_tilde=p_tilde,
            p_i=p_i,
            bq=bq,
            rm=rm,
            tr=tr,
            ubi=ubi,
            factor=factor,
        )


@dataclass
class HouseholdSolution:
    """Household savings and labor supply from a partial-equilibrium solve."""

    b_sp1: np.ndarray
    n: np.ndarray
    euler_errors: np.ndarray
    success: np.ndarray

    @property
    def max_abs_euler_error(self) -> float:
        """Largest absolute Euler equation error across ages and types."""
        return float(np.max(np.abs(self.euler_errors)))

    @property
    def all_converged(self) -> bool:
        """Whether every type's root finder reported success."""
        return bool(np.all(self.success))


def _consumption_good_prices(p_m: np.ndarray, p) -> np.ndarray:
    """Consumption good prices from production good prices.

    Newer OG-Core releases expose ``aggregates.get_io_prices``; older ones
    compute the prices inline as ``io_matrix @ p_m``.
    """
    get_io_prices = getattr(aggr, "get_io_prices", None)
    if get_io_prices is not None:
        p_i, _, _ = get_io_prices(p_m, p, "SS")
        return np.atleast_1d(np.asarray(p_i, dtype=float))
    return np.atleast_1d(np.dot(p.io_matrix, p_m).astype(float))


def _scatter_params(p, client):
    """Scatter the parameters object to Dask workers once.

    Uses ``SS.scatter_params`` when the installed OG-Core provides it.
    Otherwise it mirrors the steady-state inner loop: the ParamTools schema
    objects are unpicklable, so they are detached before scattering and
    restored afterwards.
    """
    scatter = getattr(SS, "scatter_params", None)
    if scatter is not None:
        return scatter(p, client)
    backup = {}
    for attr in ("_defaults_schema", "_validator_schema", "sel"):
        if hasattr(p, attr):
            backup[attr] = getattr(p, attr)
            try:
                delattr(p, attr)
            except Exception:  # pragma: no cover - defensive, as in OG-Core
                pass
    try:
        return client.scatter(p, broadcast=True)
    finally:
        for attr, value in backup.items():
            try:
                setattr(p, attr, value)
            except Exception:  # pragma: no cover - defensive, as in OG-Core
                pass


def _solve_one_type(env, p_or_future, guesses, j):
    """Call ``SS.solve_for_j`` for one type with the environment unpacked."""
    return SS.solve_for_j(
        guesses,
        env.r_p,
        env.w,
        env.p_tilde,
        env.p_i,
        env.bq[:, j],
        env.rm[:, j],
        env.tr[:, j],
        env.ubi[:, j],
        env.factor,
        j,
        p_or_future,
    )


def solve_households(
    env: HouseholdEnvironment,
    p,
    b_guess: np.ndarray,
    n_guess: np.ndarray,
    client=None,
    scattered_p=None,
) -> HouseholdSolution:
    """Solve every type's lifecycle problem at a fixed environment.

    Args:
        env: prices, transfers, and scaling factor held fixed.
        p: OG-Core Specifications object (current preference parameters).
        b_guess: initial savings guesses, shape (S, J).
        n_guess: initial labor guesses, shape (S, J).
        client: optional Dask client; types are solved in parallel when
            given, with a serial fallback if the parallel run fails.
        scattered_p: optional Dask future for ``p`` already scattered to the
            workers, to avoid re-scattering on repeated calls.

    Returns:
        HouseholdSolution with savings, labor, Euler errors (2S x J), and
        per-type convergence flags.
    """
    b_guess = np.asarray(b_guess, dtype=float)
    n_guess = np.asarray(n_guess, dtype=float)
    if b_guess.shape != (p.S, p.J) or n_guess.shape != (p.S, p.J):
        raise ValueError("b_guess and n_guess must have shape (S, J).")
    guesses = [np.append(b_guess[:, j], n_guess[:, j]) for j in range(p.J)]

    results = None
    if client is not None:
        try:
            p_future = (
                scattered_p
                if scattered_p is not None
                else _scatter_params(p, client)
            )
            futures = [
                client.submit(_solve_one_type, env, p_future, guesses[j], j)
                for j in range(p.J)
            ]
            results = client.gather(futures)
        except Exception as err:  # noqa: BLE001 - mirror OG-Core fallback
            logger.warning(
                "Parallel household solve failed (%s: %s); solving types "
                "serially.",
                type(err).__name__,
                err,
            )
            results = None
    if results is None:
        results = [_solve_one_type(env, p, guesses[j], j) for j in range(p.J)]

    b_sp1 = np.zeros((p.S, p.J))
    n = np.zeros((p.S, p.J))
    euler_errors = np.zeros((2 * p.S, p.J))
    success = np.zeros(p.J, dtype=bool)
    for j, result in enumerate(results):
        b_sp1[:, j] = result.x[: p.S]
        n[:, j] = result.x[p.S :]
        euler_errors[:, j] = result.fun
        success[j] = bool(getattr(result, "success", True))
    if not np.all(success):
        logger.warning(
            "Household root finder did not report success for types %s.",
            np.flatnonzero(~success).tolist(),
        )
    return HouseholdSolution(
        b_sp1=b_sp1, n=n, euler_errors=euler_errors, success=success
    )


def partial_equilibrium_ss(
    ss_output: dict,
    p,
    client=None,
    b_guess: np.ndarray | None = None,
    n_guess: np.ndarray | None = None,
    scattered_p=None,
) -> tuple[dict, HouseholdSolution]:
    """Re-solve the household block at the prices in ``ss_output``.

    Returns a copy of ``ss_output`` with ``b_sp1``, ``b_s``, and ``n``
    replaced by the new household solution (aggregates such as ``BQ``,
    ``Y``, and ``factor`` are left at their general-equilibrium values), so
    the result can be passed straight to
    :func:`ogusa.estimate_lifecycle_params.compute_model_moments`.  The
    second return value carries Euler errors and convergence flags.
    """
    env = HouseholdEnvironment.from_ss_output(ss_output, p)
    if b_guess is None:
        b_guess = np.asarray(ss_output["b_sp1"], dtype=float)
    if n_guess is None:
        n_guess = np.asarray(ss_output["n"], dtype=float)
    solution = solve_households(
        env, p, b_guess, n_guess, client=client, scattered_p=scattered_p
    )
    updated = dict(ss_output)
    updated["b_sp1"] = solution.b_sp1
    b_s = np.vstack([np.zeros((1, p.J)), solution.b_sp1[:-1, :]])
    updated["b_s"] = b_s
    updated["n"] = solution.n
    updated["before_tax_income"] = np.asarray(
        household.get_y(env.r_p, env.w, b_s, solution.n, p, "SS"),
        dtype=float,
    )
    return updated, solution


# ---------------------------------------------------------------------------
# Phase 3: concentrate out chi_n by inverting the labor first-order condition
# ---------------------------------------------------------------------------


@dataclass
class ChiNInversionResult:
    """Outcome of the chi_n inversion at fixed prices."""

    chi_n: np.ndarray
    ages: np.ndarray
    labor_model: np.ndarray
    labor_target: np.ndarray
    iterations: int
    converged: bool
    max_abs_log_gap: float
    history: list
    capped_ages: np.ndarray
    ss_output: dict
    solution: HouseholdSolution


def aggregate_labor_by_age(n: np.ndarray, p, ages: np.ndarray) -> np.ndarray:
    """Population-weighted mean labor supply at each requested age."""
    from ogusa import estimate_lifecycle_params as elp

    n = np.asarray(n, dtype=float)
    weights = elp._type_weights_by_age(p)
    idx = elp._age_indices(np.asarray(ages), p)
    return (n[idx, :] * weights[idx, :]).sum(axis=1)


def chi_n_update(
    chi_n_values: np.ndarray,
    labor_model: np.ndarray,
    labor_target: np.ndarray,
    p,
    damping: float = 1.0,
) -> np.ndarray:
    """One inversion step of the labor first-order condition.

    The steady-state labor FOC is ``chi_n[s] * MDU(n) = LHS[s]`` where the
    right-hand side depends on wages, taxes, and consumption.  Holding that
    side fixed, the ``chi_n`` that delivers the target hours is
    ``chi_n * MDU(n_model) / MDU(n_target)``.  ``damping`` raises the ratio
    to a power: 1 is the full step, below 1 damps, above 1 over-relaxes to
    offset the consumption response that makes hours move less than the
    fixed-LHS step predicts.
    """
    chi_n_values = np.asarray(chi_n_values, dtype=float)
    ratio = household.marg_ut_labor(
        np.asarray(labor_model, dtype=float), 1.0, p
    ) / household.marg_ut_labor(np.asarray(labor_target, dtype=float), 1.0, p)
    return chi_n_values * np.asarray(ratio, dtype=float) ** damping


def apply_chi_n(p, chi_n: np.ndarray) -> None:
    """Set the steady-state chi_n age profile on the spec (validated)."""
    p.update_specifications(
        {"chi_n": np.asarray(chi_n, dtype=float).reshape(-1).tolist()}
    )


def _chi_n_bounds(p, chi_n_min: float | None, chi_n_max: float | None):
    """Natural bounds for chi_n from the validators unless overridden."""
    from ogusa import estimate_lifecycle_params as elp

    lo, hi = elp._validator_range(p, "chi_n")
    lo = max(lo, 1e-8) if chi_n_min is None else float(chi_n_min)
    hi = hi if chi_n_max is None else float(chi_n_max)
    return lo, hi


def invert_chi_n(
    ss_output: dict,
    p,
    labor_target: np.ndarray,
    config=None,
    max_iter: int = 30,
    tol: float = 1e-3,
    damping: float = 1.0,
    chi_n_min: float | None = None,
    chi_n_max: float | None = None,
    client=None,
) -> ChiNInversionResult:
    """Choose chi_n by age so model hours match ``labor_target`` at fixed prices.

    Iterates: solve the household block at the prices in ``ss_output``,
    form population-weighted hours at each target age, update ``chi_n`` at
    those ages with :func:`chi_n_update`, and repeat until the largest
    absolute log gap between model and target hours is below ``tol``.
    Ages above the last target age are filled with
    :func:`ogusa.estimate_lifecycle_params.build_chi_n_profile` using
    ``config.chi_n_tail_method`` (default: the initial profile's tail
    rescaled to join the last estimated value).  Values are clipped to the
    ParamTools range for ``chi_n`` unless narrower bounds are given, and the
    ages where the cap binds are reported.

    On return ``p`` carries the final ``chi_n`` and ``result.ss_output`` is
    the household solution at that profile, ready for
    :func:`ogusa.estimate_lifecycle_params.compute_model_moments`.
    """
    from dataclasses import replace

    from ogusa import estimate_lifecycle_params as elp

    if config is None:
        config = elp.LifecycleCalibrationConfig()
    config = replace(
        config,
        estimate_chi_n_min_age=config.min_age,
        estimate_chi_n_max_age=config.max_age,
    )
    config.validate(p)
    ages = config.moment_ages
    labor_target = np.asarray(labor_target, dtype=float).reshape(-1)
    if labor_target.size != ages.size:
        raise ValueError("labor_target must have one value per target age.")
    if np.any(labor_target <= 0) or np.any(labor_target >= p.ltilde):
        raise ValueError("labor_target must lie strictly inside (0, ltilde).")
    lo, hi = _chi_n_bounds(p, chi_n_min, chi_n_max)

    base_chi_n = elp._ss_chi_n(p)
    est_idx = elp._age_indices(ages, p)
    chi_n = base_chi_n.copy()
    b_guess = np.asarray(ss_output["b_sp1"], dtype=float)
    n_guess = np.asarray(ss_output["n"], dtype=float)
    history: list[float] = []
    converged = False
    capped = np.zeros(ages.size, dtype=bool)
    updated = ss_output
    solution = None
    labor_model = np.full(ages.size, np.nan)
    iterations = 0

    for iterations in range(1, max_iter + 1):
        apply_chi_n(p, chi_n)
        updated, solution = partial_equilibrium_ss(
            ss_output, p, client=client, b_guess=b_guess, n_guess=n_guess
        )
        labor_model = aggregate_labor_by_age(updated["n"], p, ages)
        gap = np.log(labor_model / labor_target)
        max_gap = float(np.max(np.abs(gap)))
        history.append(max_gap)
        logger.info(
            "chi_n inversion iteration %d: max |log(n_model/n_target)| = %.3e",
            iterations,
            max_gap,
        )
        if max_gap < tol:
            converged = True
            break
        if iterations == max_iter:
            break
        new_values = chi_n_update(
            chi_n[est_idx], labor_model, labor_target, p, damping=damping
        )
        capped = new_values >= hi
        new_values = np.clip(new_values, lo, hi)
        chi_n = elp.build_chi_n_profile(new_values, base_chi_n, p, config)
        chi_n = np.clip(chi_n, lo, hi)
        b_guess, n_guess = solution.b_sp1, solution.n

    if not converged:
        logger.warning(
            "chi_n inversion did not converge in %d iterations "
            "(max |log gap| = %.3e).",
            iterations,
            history[-1],
        )
    if np.any(capped):
        logger.warning(
            "chi_n hit its upper bound %.0f at ages %s.",
            hi,
            ages[capped].tolist(),
        )
    return ChiNInversionResult(
        chi_n=chi_n,
        ages=ages,
        labor_model=labor_model,
        labor_target=labor_target,
        iterations=iterations,
        converged=converged,
        max_abs_log_gap=history[-1],
        history=history,
        capped_ages=ages[capped],
        ss_output=updated,
        solution=solution,
    )


# ---------------------------------------------------------------------------
# Phase 4: beta by type and chi_b from wealth shares, level, and bequest flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferenceCalibrationOptions:
    """
    Options for the beta / chi_b calibration at fixed prices.

    Attributes:
        chi_b_mode: ``"common_scale"`` moves every type's ``chi_b`` by one
            common factor, identified by the aggregate bequest-flow ratio.
            ``"by_type"`` gives each type group its own ``chi_b`` factor,
            identified by the old-age wealth tilt of the matching wealth
            percentile bin, with the bequest-flow ratio as an additional
            residual.
        bottom_share: types whose cumulative population share lies in the
            bottom ``bottom_share`` form the bottom group.
        top_share: types inside the top ``top_share`` share one ``chi_b``
            factor in ``by_type`` mode (each still has its own ``beta``).
        beta_annual_max, chi_b_max: optional ceilings tighter than the
            ParamTools validators (the validator caps are 0.9999 and
            10,000).
        exclude_bottom: when True (default), the bottom group's wealth-share
            and tilt bins are dropped from the targets and the bottom types
            share their ``beta`` (and ``chi_b``) factor with the next type
            up.  A deterministic model with no within-type heterogeneity
            cannot deliver the SCF bottom-half share of about one percent,
            because young households of every type fill the bottom
            percentiles, so targeting it drives the bottom betas to zero.
            When False the bottom group gets its own factor and one merged
            share target.
        bequest_flow_weight: weight on the bequest-flow residual.
        failure_residual: value of every residual when the household solve
            fails to converge, in log units.
        max_nfev: cap on least-squares iterations counted as function
            evaluations by SciPy; finite-difference Jacobian columns are
            not counted, so total household solves are about
            ``max_nfev * (1 + n_params)``.
        diff_step: relative finite-difference step for the Jacobian.
        ftol, xtol: least-squares tolerances.
    """

    chi_b_mode: str = "by_type"
    bottom_share: float = 0.5
    top_share: float = 0.01
    exclude_bottom: bool = True
    beta_annual_max: float | None = None
    chi_b_max: float | None = None
    bequest_flow_weight: float = 1.0
    failure_residual: float = 3.0
    max_nfev: int = 150
    diff_step: float = 1e-3
    ftol: float = 1e-8
    xtol: float = 1e-8


@dataclass
class PreferenceCalibrationResult:
    """Outcome of the beta / chi_b calibration at fixed prices."""

    beta_annual: np.ndarray
    chi_b: np.ndarray
    theta: np.ndarray
    residuals: np.ndarray
    residual_names: tuple
    data_values: np.ndarray
    model_values: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str
    ss_output: dict
    solution: HouseholdSolution | None

    def to_frame(self):
        """Data, model, and log residual for each target."""
        import pandas as pd

        return pd.DataFrame(
            {
                "target": self.residual_names,
                "data": self.data_values,
                "model": self.model_values,
                "log_residual": self.residuals,
            }
        )


def _group_bounds(base, groups, lo, hi, transform):
    """Per-group bounds on an additive shift in transformed space."""
    lower = np.empty(len(groups))
    upper = np.empty(len(groups))
    for g, members in enumerate(groups):
        t_base = transform(base[members])
        lower[g] = np.max(transform(lo) - t_base)
        upper[g] = np.min(transform(hi) - t_base)
    return lower, upper


def _logit(x):
    x = np.asarray(x, dtype=float)
    return np.log(x / (1.0 - x))


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class _PreferenceParameterization:
    """Map a free parameter vector to beta_annual and chi_b by type."""

    def __init__(self, p, options: PreferenceCalibrationOptions):
        from ogusa import estimate_lifecycle_params as elp

        self.base_beta = np.asarray(p.beta_annual, dtype=float).copy()
        self.base_chi_b = np.asarray(p.chi_b, dtype=float).copy()
        lambdas = elp._lambdas(p)
        groups = _type_groups(lambdas, options)
        bottom = groups[0]
        # beta: the bottom group shares one factor, every other type is free.
        self.beta_groups = [bottom] + [
            [j] for j in range(p.J) if j not in bottom
        ]
        if options.chi_b_mode == "common_scale":
            self.chi_b_groups = [list(range(p.J))]
        elif options.chi_b_mode == "by_type":
            self.chi_b_groups = groups
        else:
            raise ValueError(f"Unsupported chi_b_mode: {options.chi_b_mode}")
        self.n_beta = len(self.beta_groups)
        self.n_chi_b = len(self.chi_b_groups)
        eps = 1e-4
        b_lo, b_hi = elp._validator_range(p, "beta_annual")
        b_lo, b_hi = max(b_lo, eps), min(b_hi, 1.0 - eps)
        if options.beta_annual_max is not None:
            b_hi = min(b_hi, float(options.beta_annual_max))
        c_lo, c_hi = elp._validator_range(p, "chi_b")
        c_lo = max(c_lo, eps)
        if options.chi_b_max is not None:
            c_hi = min(c_hi, float(options.chi_b_max))
        # Base values may already sit on a bound; keep them inside it.
        self.base_beta = np.clip(self.base_beta, b_lo, b_hi)
        self.base_chi_b = np.clip(self.base_chi_b, c_lo, c_hi)
        self.beta_bounds = (b_lo, b_hi)
        self.chi_b_bounds = (c_lo, c_hi)
        beta_lo, beta_hi = _group_bounds(
            self.base_beta, self.beta_groups, b_lo, b_hi, _logit
        )
        chi_lo, chi_hi = _group_bounds(
            self.base_chi_b, self.chi_b_groups, c_lo, c_hi, np.log
        )
        self.lower = np.concatenate([beta_lo, chi_lo])
        self.upper = np.concatenate([beta_hi, chi_hi])

    @property
    def size(self) -> int:
        return self.n_beta + self.n_chi_b

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=float)
        beta = self.base_beta.copy()
        for g, members in enumerate(self.beta_groups):
            beta[members] = _logistic(
                _logit(self.base_beta[members]) + theta[g]
            )
        chi_b = self.base_chi_b.copy()
        for g, members in enumerate(self.chi_b_groups):
            chi_b[members] = self.base_chi_b[members] * np.exp(
                theta[self.n_beta + g]
            )
        # Round-tripping through logit/log at a bound can overshoot it by
        # floating-point error, which ParamTools rejects; clip to be safe.
        beta = np.clip(beta, *self.beta_bounds)
        chi_b = np.clip(chi_b, *self.chi_b_bounds)
        return beta, chi_b


def _type_groups(lambdas, options: PreferenceCalibrationOptions):
    """Type groups for parameters: bottom merged (plus next type when the
    bottom bin is excluded), top merged, others single."""
    from ogusa import estimate_lifecycle_params as elp

    groups = elp.merged_type_groups(
        lambdas, options.bottom_share, options.top_share
    )
    if options.exclude_bottom and len(groups) > 1:
        bottom = groups[0] + groups[1]
        groups = [bottom] + groups[2:]
    return groups


def _merge_bins(values, groups):
    """Sum per-type values over groups."""
    values = np.asarray(values, dtype=float)
    return np.array([values[g].sum() for g in groups])


def preference_targets(
    data_moments, p, config, options: PreferenceCalibrationOptions
):
    """Select and merge the data moments the beta / chi_b calibration uses.

    Returns names and values for: wealth shares by type bin (the bottom
    group's bins merged into one target, or dropped when
    ``options.exclude_bottom``), the wealth-to-income ratio, the by-bin
    old-age tilts in ``by_type`` mode (bottom tilt bin dropped when
    ``exclude_bottom``), and the bequest-flow ratio.  The third return value
    is the selection needed to compute matching model values.
    """
    from ogusa import estimate_lifecycle_params as elp

    lookup = dict(zip(data_moments.names, data_moments.values))
    lambdas = elp._lambdas(p)
    share_names = elp.wealth_share_bin_names(lambdas)
    shares = np.array([lookup[name] for name in share_names])
    raw_groups = elp.merged_type_groups(
        lambdas, options.bottom_share, options.top_share
    )
    bottom = raw_groups[0]
    if options.exclude_bottom:
        share_bins = [[j] for j in range(p.J) if j not in bottom]
    else:
        share_bins = [bottom] + [[j] for j in range(p.J) if j not in bottom]
    names = []
    values = []
    cum = np.cumsum(lambdas)
    for members in share_bins:
        if len(members) == 1:
            names.append(share_names[members[0]])
        else:
            lo = _as_pct(cum[members[0]] - lambdas[members[0]])
            hi = _as_pct(cum[members[-1]])
            names.append(f"wealth_share_{lo}_{hi}")
        values.append(shares[members].sum())
    names.append("wealth_income_ratio")
    values.append(lookup["wealth_income_ratio"])
    tilt_idx = []
    if options.chi_b_mode == "by_type":
        tilt_names = elp.tilt_moment_names(config, p)
        start = 1 if options.exclude_bottom else 0
        for k in range(start, len(tilt_names)):
            names.append(tilt_names[k])
            values.append(lookup[tilt_names[k]])
            tilt_idx.append(k)
    names.append("bequest_flow_ratio")
    values.append(lookup["bequest_flow_ratio"])
    selection = {"share_bins": share_bins, "tilt_idx": tilt_idx}
    return tuple(names), np.asarray(values, dtype=float), selection


def _as_pct(share):
    from ogusa import estimate_lifecycle_params as elp

    return elp._percent_label(share)


def _preference_model_values(ss_output, p, config, options, selection):
    from ogusa import estimate_lifecycle_params as elp

    shares = elp.model_wealth_shares(ss_output, p)
    values = list(_merge_bins(shares, selection["share_bins"]))
    values.append(elp.model_wealth_income_ratio(ss_output, p))
    if options.chi_b_mode == "by_type":
        tilt = elp.model_old_age_ratio_by_type(ss_output, p, config)
        values.extend(tilt[selection["tilt_idx"]])
    values.append(elp.model_bequest_flow_ratio(ss_output, p))
    return np.asarray(values, dtype=float)


def _preference_weights(n_targets, options):
    weights = np.ones(n_targets)
    weights[-1] = options.bequest_flow_weight
    return weights


def calibrate_beta_chi_b(
    ss_output: dict,
    p,
    data_moments,
    config=None,
    options: PreferenceCalibrationOptions | None = None,
    client=None,
) -> PreferenceCalibrationResult:
    """Calibrate beta by type and chi_b at fixed prices.

    Solves a bounded nonlinear least-squares problem over additive shifts
    to ``logit(beta_annual)`` by type (bottom types tied) and to
    ``log(chi_b)`` by group, with log residuals between model and data for
    the merged wealth shares, the wealth-to-income ratio, and the bequest
    flow ratio (plus by-bin old-age tilts in ``by_type`` mode).  Every
    residual evaluation is a household-only solve at the prices in
    ``ss_output``; failed solves return ``options.failure_residual``.

    On return ``p`` carries the calibrated parameters and
    ``result.ss_output`` the matching household solution.
    """
    from scipy import optimize

    from ogusa import estimate_lifecycle_params as elp

    if config is None:
        config = elp.LifecycleCalibrationConfig()
    if options is None:
        options = PreferenceCalibrationOptions()
    if (
        options.chi_b_mode == "by_type"
        and not config.include_old_age_ratio_by_type
    ):
        raise ValueError(
            "chi_b_mode='by_type' needs config.include_old_age_ratio_by_type."
        )
    for needed in ("wealth_income_ratio", "bequest_flow_ratio"):
        if needed not in data_moments.names:
            raise ValueError(f"data_moments must include {needed}.")

    names, data_values, selection = preference_targets(
        data_moments, p, config, options
    )
    weights = _preference_weights(len(names), options)
    param = _PreferenceParameterization(p, options)
    state = {
        "b_guess": np.asarray(ss_output["b_sp1"], dtype=float),
        "n_guess": np.asarray(ss_output["n"], dtype=float),
        "last": None,
        "nfev": 0,
    }

    initial_guesses = (state["b_guess"], state["n_guess"])

    def _solve(theta):
        """Household solve at theta, retrying from the initial guesses."""
        beta, chi_b = param.unpack(theta)
        p.update_specifications(
            {"beta_annual": beta.tolist(), "chi_b": chi_b.tolist()}
        )
        guess_sets = [(state["b_guess"], state["n_guess"])]
        if state["b_guess"] is not initial_guesses[0]:
            guess_sets.append(initial_guesses)
        for b_guess, n_guess in guess_sets:
            updated, solution = partial_equilibrium_ss(
                ss_output, p, client=client, b_guess=b_guess, n_guess=n_guess
            )
            state["nfev"] += 1
            if solution.all_converged:
                return updated, solution
        return updated, solution

    updated0, solution0 = _solve(theta0 := np.zeros(param.size))
    if not solution0.all_converged:
        raise RuntimeError(
            "The household solve did not converge at the starting parameters "
            "with the guesses in ss_output. Pass an ss_output whose b_sp1 "
            "and n were solved at the current chi_n (for example the "
            "ss_output returned by invert_chi_n)."
        )

    def residuals(theta):
        updated, solution = _solve(theta)
        if not solution.all_converged:
            logger.warning(
                "Household solve failed at theta=%s; penalizing.",
                np.round(theta, 4).tolist(),
            )
            return np.full(len(names), options.failure_residual)
        state["b_guess"], state["n_guess"] = solution.b_sp1, solution.n
        model_values = _preference_model_values(
            updated, p, config, options, selection
        )
        state["last"] = (updated, solution, model_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = weights * np.log(model_values / data_values)
        if not np.all(np.isfinite(res)):
            return np.full(len(names), options.failure_residual)
        return np.clip(
            res, -options.failure_residual, options.failure_residual
        )

    result = optimize.least_squares(
        residuals,
        theta0,
        bounds=(param.lower, param.upper),
        method="trf",
        diff_step=options.diff_step,
        max_nfev=options.max_nfev,
        ftol=options.ftol,
        xtol=options.xtol,
    )
    # Re-evaluate at the solution so p and the cached state match result.x.
    final_res = residuals(result.x)
    beta, chi_b = param.unpack(result.x)
    if state["last"] is None:
        raise RuntimeError("No converged household solve during calibration.")
    updated, solution, model_values = state["last"]
    logger.info(
        "beta/chi_b calibration: cost=%.3e, nfev=%d, success=%s, %s",
        result.cost,
        state["nfev"],
        result.success,
        result.message,
    )
    return PreferenceCalibrationResult(
        beta_annual=beta,
        chi_b=chi_b,
        theta=result.x,
        residuals=final_res,
        residual_names=names,
        data_values=data_values,
        model_values=model_values,
        cost=float(result.cost),
        nfev=state["nfev"],
        success=bool(result.success),
        message=str(result.message),
        ss_output=updated,
        solution=solution,
    )


# ---------------------------------------------------------------------------
# Phase 5: outer general-equilibrium loop
# ---------------------------------------------------------------------------


def _ss_solver_has_G() -> bool:
    """Whether the installed OG-Core steady-state solver carries G."""
    import inspect

    return "G" in inspect.signature(SS.SS_solver).parameters


def _ss_guesses_from_solution(previous: dict, p) -> list:
    """Outer-loop guess vector for ``SS.SS_fsolve`` from a prior solution.

    Layout follows ``SS.run_SS`` for a baseline solve: ``[r_p, r, w]``,
    then ``p_m``, ``Y``, the bequest items, ``G`` on OG-Core versions whose
    solver carries it, ``TR``, and ``factor``.
    """
    BQ = np.atleast_1d(np.asarray(previous["BQ"], dtype=float))
    bq_items = [float(BQ.sum())] if p.use_zeta else BQ.tolist()
    guesses = (
        [float(previous["r_p"]), float(previous["r"]), float(previous["w"])]
        + np.atleast_1d(np.asarray(previous["p_m"], dtype=float)).tolist()
        + [float(previous["Y"])]
        + bq_items
    )
    if _ss_solver_has_G():
        guesses.append(float(previous["G"]))
    guesses.append(float(previous["TR"]))
    guesses.append(float(previous["factor"]))
    return guesses


def _unpack_ss_solution(x: np.ndarray, p) -> dict:
    """Split the root-finder solution into named outer-loop variables."""
    x = np.asarray(x, dtype=float)
    has_G = _ss_solver_has_G()
    out = {
        "r_p": float(x[0]),
        "r": float(x[1]),
        "w": float(x[2]),
        "p_m": x[3 : 3 + p.M],
        "Y": float(x[3 + p.M]),
    }
    tail = 3 if has_G else 2
    out["BQ"] = x[3 + p.M + 1 : -tail]
    if has_G:
        out["G"] = float(x[-3])
    out["TR"] = float(x[-2])
    out["factor"] = float(x[-1])
    if not p.budget_balance and not p.baseline_spending:
        out["Y"] = out["TR"] / p.alpha_T[-1]
    return out


def solve_ge_steady_state(
    p, previous: dict | None = None, client=None
) -> dict:
    """Solve the baseline general-equilibrium steady state.

    With ``previous`` (an earlier OG-Core steady-state output), the outer
    root finder starts from that solution's prices, aggregates, and
    household arrays instead of OG-Core's cold guesses, which matters when
    the calibrated parameters move the equilibrium far from the defaults.
    If the warm start fails to converge the solve falls back to
    ``SS.run_SS``.  Serial solves (``client=None``) are much faster than
    Dask on a single machine because the steady state is dominated by
    parameter-scattering overhead.
    """
    from scipy import optimize

    if not p.baseline:
        raise ValueError("solve_ge_steady_state supports baseline solves.")
    if previous is None or p.baseline_spending:
        return SS.run_SS(p, client=client)

    guesses = _ss_guesses_from_solution(previous, p)
    b_guess = np.asarray(previous["b_sp1"], dtype=float)
    n_guess = np.asarray(previous["n"], dtype=float)
    args = [b_guess, n_guess, None, None, None, p, client]
    if _ss_solver_has_G():
        args.append(None)  # scattered_p
    try:
        sol = optimize.root(
            SS.SS_fsolve,
            guesses,
            args=tuple(args),
            method=p.SS_root_method,
            tol=p.mindist_SS,
        )
    except _WARM_START_ERRORS as err:
        logger.warning(
            "Warm-started GE solve raised %s: %s; using SS.run_SS.",
            type(err).__name__,
            err,
        )
        return SS.run_SS(p, client=client)
    if not sol.success:
        logger.warning(
            "Warm-started GE solve did not converge (%s); using SS.run_SS.",
            sol.message,
        )
        return SS.run_SS(p, client=client)

    vals = _unpack_ss_solution(sol.x, p)
    kwargs = {
        "bmat": b_guess,
        "nmat": n_guess,
        "r_p": vals["r_p"],
        "r": vals["r"],
        "w": vals["w"],
        "p_m": vals["p_m"],
        "Y": vals["Y"],
        "BQ": vals["BQ"],
        "TR": vals["TR"],
        "Ig_baseline": None,
        "factor": vals["factor"],
        "p": p,
        "client": client,
        "fsolve_flag": True,
    }
    if "G" in vals:
        kwargs["G"] = vals["G"]
    return SS.SS_solver(**kwargs)


_WARM_START_ERRORS = (
    AssertionError,
    FloatingPointError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class OuterIterationRecord:
    """Diagnostics for one pass of the outer calibration loop."""

    iteration: int
    param_change: float
    price_change: float
    damping: float
    chi_n_iterations: int
    pref_nfev: int
    pref_cost: float
    ge_seconds: float
    prices: dict
    residuals: dict


@dataclass
class LifecycleCalibrationOutcome:
    """Result of the full nested preference calibration."""

    beta_annual: np.ndarray
    chi_b: np.ndarray
    chi_n: np.ndarray
    ss_output: dict
    iterations: int
    converged: bool
    history: list
    data_moments: object
    model_moments: object
    chi_n_result: ChiNInversionResult | None
    pref_result: PreferenceCalibrationResult | None

    @property
    def parameter_dict(self) -> dict:
        """Calibrated values in ``update_specifications`` format."""
        return {
            "beta_annual": np.asarray(self.beta_annual).tolist(),
            "chi_b": np.asarray(self.chi_b).tolist(),
            "chi_n": np.asarray(self.chi_n).tolist(),
        }

    def to_frame(self):
        """Data versus model moments at the final general equilibrium."""
        return self.data_moments.to_frame(self.model_moments)


def _theta_from_p(p) -> np.ndarray:
    """Stack transformed preference parameters for change tracking."""
    from ogusa import estimate_lifecycle_params as elp

    beta = np.asarray(p.beta_annual, dtype=float)
    return np.concatenate(
        [
            _logit(beta),
            np.log(np.asarray(p.chi_b, dtype=float)),
            np.log(elp._ss_chi_n(p)),
        ]
    )


def _apply_theta(theta: np.ndarray, p) -> None:
    """Inverse of :func:`_theta_from_p`, applied to the spec."""
    J = p.J
    beta = _logistic(theta[:J])
    chi_b = np.exp(theta[J : 2 * J])
    chi_n = np.exp(theta[2 * J :])
    p.update_specifications(
        {
            "beta_annual": beta.tolist(),
            "chi_b": chi_b.tolist(),
            "chi_n": chi_n.tolist(),
        }
    )


def _price_change(new: dict, old: dict) -> tuple[float, dict]:
    """Largest relative change across the outer-loop prices."""
    keys = ("r_p", "r", "w", "factor", "TR")
    changes = {}
    for key in keys:
        a, b = float(np.squeeze(old[key])), float(np.squeeze(new[key]))
        changes[key] = abs(b - a) / max(abs(a), 1e-12)
    bq_old = np.asarray(old["BQ"], dtype=float).sum()
    bq_new = np.asarray(new["BQ"], dtype=float).sum()
    changes["BQ"] = abs(bq_new - bq_old) / max(abs(bq_old), 1e-12)
    return max(changes.values()), changes


def calibrate_lifecycle_preferences(
    p,
    config=None,
    options: PreferenceCalibrationOptions | None = None,
    data_moments=None,
    initial_ss: dict | None = None,
    max_outer: int = 15,
    param_tol: float = 1e-3,
    price_tol: float = 1e-3,
    outer_damping: float = 1.0,
    adaptive_damping: bool = True,
    reinvert_chi_n: bool = True,
    client=None,
    ge_client=None,
) -> LifecycleCalibrationOutcome:
    """Calibrate chi_n, beta by type, and chi_b to joint GE convergence.

    Each outer pass: solve (or reuse) the general-equilibrium steady state,
    invert the labor FOC for ``chi_n`` at those prices, calibrate ``beta``
    and ``chi_b`` at those prices, optionally re-invert ``chi_n`` so hours
    stay on target after the preference change, blend the new parameters
    with the old ones by ``outer_damping`` in transformed space, and
    re-solve the general equilibrium warm-started from the previous
    solution.  Stops when the largest transformed-parameter change and the
    largest relative price change both fall below their tolerances.

    With ``adaptive_damping`` the damping factor halves whenever the
    parameter change fails to shrink by at least ten percent from one pass
    to the next, which guards against
    the oscillation that strong general-equilibrium feedback (saving down,
    interest rate up, hours up) can produce.

    On return ``p`` carries the calibrated parameters and the outcome holds
    the final steady state and a data-versus-model moment table.
    """
    import time

    from ogusa import estimate_lifecycle_params as elp

    if config is None:
        config = elp.LifecycleCalibrationConfig()
    if options is None:
        options = PreferenceCalibrationOptions()
    config.validate(p)
    if data_moments is None:
        data_moments = elp.compute_data_moments(p, config)
    labor_target = np.array(
        [
            dict(zip(data_moments.names, data_moments.values))[
                f"labor_supply_age_{age}"
            ]
            for age in config.moment_ages
        ]
    )

    t = time.time()
    ss = (
        initial_ss
        if initial_ss is not None
        else solve_ge_steady_state(p, client=ge_client)
    )
    logger.info("Outer loop: initial GE solve took %.0f s", time.time() - t)

    history: list[OuterIterationRecord] = []
    damping = float(outer_damping)
    converged = False
    chi_n_result = None
    pref_result = None
    previous_change = np.inf
    for iteration in range(1, max_outer + 1):
        theta_old = _theta_from_p(p)
        chi_n_result = invert_chi_n(
            ss, p, labor_target, config=config, client=client
        )
        pref_result = calibrate_beta_chi_b(
            chi_n_result.ss_output,
            p,
            data_moments,
            config=config,
            options=options,
            client=client,
        )
        if reinvert_chi_n:
            chi_n_result = invert_chi_n(
                pref_result.ss_output,
                p,
                labor_target,
                config=config,
                client=client,
            )
        theta_new = _theta_from_p(p)
        param_change = float(np.max(np.abs(theta_new - theta_old)))
        if adaptive_damping and param_change > 0.9 * previous_change:
            # Not shrinking fast enough (or growing): damp harder.
            damping = max(damping / 2.0, 0.05)
            logger.info("Outer loop: damping reduced to %.3f", damping)
        previous_change = param_change
        if damping < 1.0:
            _apply_theta(theta_old + damping * (theta_new - theta_old), p)

        t = time.time()
        ss_new = solve_ge_steady_state(p, previous=ss, client=ge_client)
        ge_seconds = time.time() - t
        price_change, changes = _price_change(ss_new, ss)
        model_moments = elp.compute_model_moments(ss_new, p, config)
        residuals = dict(
            zip(
                data_moments.names,
                elp.moment_residuals(model_moments, data_moments, "relative"),
            )
        )
        history.append(
            OuterIterationRecord(
                iteration=iteration,
                param_change=param_change,
                price_change=price_change,
                damping=damping,
                chi_n_iterations=chi_n_result.iterations,
                pref_nfev=pref_result.nfev,
                pref_cost=pref_result.cost,
                ge_seconds=ge_seconds,
                prices={
                    k: float(np.squeeze(ss_new[k]))
                    for k in ("r_p", "r", "w", "factor", "TR")
                },
                residuals=residuals,
            )
        )
        logger.info(
            "Outer loop %d: max param change %.3e, max price change %.3e "
            "(%s), GE %.0f s, pref cost %.3e",
            iteration,
            param_change,
            price_change,
            max(changes, key=changes.get),
            ge_seconds,
            pref_result.cost,
        )
        ss = ss_new
        if param_change < param_tol and price_change < price_tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "Outer loop did not converge in %d passes (param change %.3e, "
            "price change %.3e).",
            max_outer,
            history[-1].param_change,
            history[-1].price_change,
        )
    model_moments = elp.compute_model_moments(ss, p, config)
    return LifecycleCalibrationOutcome(
        beta_annual=np.asarray(p.beta_annual, dtype=float).copy(),
        chi_b=np.asarray(p.chi_b, dtype=float).copy(),
        chi_n=elp._ss_chi_n(p),
        ss_output=ss,
        iterations=len(history),
        converged=converged,
        history=history,
        data_moments=data_moments,
        model_moments=model_moments,
        chi_n_result=chi_n_result,
        pref_result=pref_result,
    )
