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
