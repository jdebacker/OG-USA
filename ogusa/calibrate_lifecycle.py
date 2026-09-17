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
    updated["b_s"] = np.vstack([np.zeros((1, p.J)), solution.b_sp1[:-1, :]])
    updated["n"] = solution.n
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
