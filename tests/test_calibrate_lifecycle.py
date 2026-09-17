"""
Tests for the household-only steady-state solve used in calibration.
"""

import json
import logging
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pytest

from ogusa import calibrate_lifecycle as cl


class MockParams:
    """
    Minimal parameter object for the household solver wrapper.
    """

    S = 4
    J = 2
    FOC_root_method = "hybr"


def _ss_output(p, scale=1.0):
    """
    Steady-state output with the household-level inputs already present.
    """
    return {
        "r_p": 0.04,
        "w": 1.2,
        "factor": 100000.0,
        "p_m": np.array([1.0]),
        "p_i": np.array([1.0]),
        "p_tilde": np.array(1.0),
        "BQ": np.array([0.1, 0.2]),
        "RM": 0.0,
        "TR": 0.3,
        "Y": 10.0,
        "bq": np.ones((p.S, p.J)) * 0.01,
        "rm": np.zeros((p.S, p.J)),
        "tr": np.ones((p.S, p.J)) * 0.02,
        "ubi": np.zeros((p.S, p.J)),
        "b_sp1": np.ones((p.S, p.J)) * scale,
        "n": np.ones((p.S, p.J)) * 0.3 * scale,
    }


def _fake_solve_for_j(offset):
    """
    Return a fake SS.solve_for_j whose root is the guess plus an offset.
    """

    def fake(
        guesses, r_p, w, p_tilde, p_i, bq_j, rm_j, tr_j, ubi_j, factor, j, p
    ):
        assert bq_j.shape == (p.S,)
        assert tr_j.shape == (p.S,)
        assert np.isclose(r_p, 0.04)
        return SimpleNamespace(
            x=np.asarray(guesses) + offset + j,
            fun=np.full(2 * p.S, 1e-12 * (j + 1)),
            success=j == 0,
        )

    return fake


def test_household_environment_from_stored_household_inputs():
    """
    Household-level arrays in the SS output are used directly.
    """
    p = MockParams()
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    assert env.r_p == 0.04
    assert env.w == 1.2
    assert env.p_tilde == 1.0
    assert env.p_i.shape == (1,)
    assert env.bq.shape == (p.S, p.J)
    assert np.allclose(env.tr, 0.02)
    assert env.factor == 100000.0


def test_solve_households_serial_collects_results(monkeypatch, caplog):
    """
    Serial solves return savings, labor, Euler errors, and success flags.
    """
    p = MockParams()
    monkeypatch.setattr(cl.SS, "solve_for_j", _fake_solve_for_j(0.5))
    caplog.set_level(logging.WARNING, logger=cl.logger.name)
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)

    solution = cl.solve_households(
        env, p, np.ones((p.S, p.J)), np.ones((p.S, p.J)) * 0.3
    )

    assert solution.b_sp1.shape == (p.S, p.J)
    assert np.allclose(solution.b_sp1[:, 0], 1.5)
    assert np.allclose(solution.b_sp1[:, 1], 2.5)
    assert np.allclose(solution.n[:, 0], 0.8)
    assert solution.euler_errors.shape == (2 * p.S, p.J)
    assert np.isclose(solution.max_abs_euler_error, 2e-12)
    assert solution.success.tolist() == [True, False]
    assert not solution.all_converged
    assert "types [1]" in caplog.text


def test_solve_households_rejects_bad_guess_shapes():
    """
    Guess arrays must be (S, J).
    """
    p = MockParams()
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    with pytest.raises(ValueError, match=r"\(S, J\)"):
        cl.solve_households(env, p, np.ones(p.S), np.ones((p.S, p.J)))


def test_solve_households_parallel_falls_back_to_serial(monkeypatch, caplog):
    """
    A failing Dask client triggers a warning and a serial solve.
    """
    p = MockParams()
    monkeypatch.setattr(cl.SS, "solve_for_j", _fake_solve_for_j(0.0))
    monkeypatch.setattr(cl, "_scatter_params", lambda p, client: "future")

    class FailingClient:
        def submit(self, *args, **kwargs):
            raise RuntimeError("no workers")

        def gather(self, futures):  # pragma: no cover - never reached
            return futures

    caplog.set_level(logging.WARNING, logger=cl.logger.name)
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    solution = cl.solve_households(
        env,
        p,
        np.ones((p.S, p.J)),
        np.ones((p.S, p.J)),
        client=FailingClient(),
    )
    assert np.allclose(solution.b_sp1[:, 0], 1.0)
    assert "solving types serially" in caplog.text


def test_solve_households_uses_client_when_available(monkeypatch):
    """
    With a working client, each type is submitted once and gathered.
    """
    p = MockParams()
    fake = _fake_solve_for_j(0.0)
    monkeypatch.setattr(cl.SS, "solve_for_j", fake)
    monkeypatch.setattr(cl, "_scatter_params", lambda p, client: p)

    class Client:
        def __init__(self):
            self.submitted = []

        def submit(self, fn, *args):
            self.submitted.append(args[-1])
            return fn(*args)

        def gather(self, futures):
            return futures

    client = Client()
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    solution = cl.solve_households(
        env, p, np.ones((p.S, p.J)), np.ones((p.S, p.J)), client=client
    )
    assert client.submitted == [0, 1]
    assert np.allclose(solution.b_sp1[:, 1], 2.0)


def test_partial_equilibrium_ss_updates_household_arrays_only(monkeypatch):
    """
    The returned dict swaps b_sp1, b_s, and n and keeps aggregates.
    """
    p = MockParams()
    monkeypatch.setattr(cl.SS, "solve_for_j", _fake_solve_for_j(1.0))
    ss_output = _ss_output(p)

    updated, solution = cl.partial_equilibrium_ss(ss_output, p)

    assert updated is not ss_output
    assert np.allclose(updated["b_sp1"][:, 0], 2.0)
    assert np.allclose(updated["b_s"][0, :], 0.0)
    assert np.allclose(updated["b_s"][1:, 0], 2.0)
    assert np.allclose(updated["n"][:, 1], 0.3 + 2.0)
    assert updated["Y"] == 10.0
    assert np.allclose(ss_output["b_sp1"], 1.0)
    assert solution.all_converged is False


@pytest.mark.local
def test_partial_equilibrium_reproduces_general_equilibrium_households():
    """
    At equilibrium prices, the household-only solve returns the GE solution.

    This solves the full OG-USA steady state, so it is local only.
    """
    import ogcore
    from ogcore.parameters import Specifications
    from ogusa import estimate_lifecycle_params as elp

    ogcore.config.VERBOSE = False
    p = Specifications(baseline=True, num_workers=1)
    with (
        resources.files("ogusa")
        .joinpath("ogusa_default_parameters.json")
        .open() as file
    ):
        p.update_specifications(json.load(file))

    ss_output = cl.SS.run_SS(p, client=None)
    updated, solution = cl.partial_equilibrium_ss(ss_output, p)

    assert solution.all_converged
    assert solution.max_abs_euler_error < 1e-6
    assert np.allclose(updated["b_sp1"], ss_output["b_sp1"], rtol=1e-5)
    assert np.allclose(updated["n"], ss_output["n"], rtol=1e-5)

    config = elp.LifecycleCalibrationConfig()
    ge_moments = elp.compute_model_moments(ss_output, p, config)
    pe_moments = elp.compute_model_moments(updated, p, config)
    assert np.allclose(ge_moments.values, pe_moments.values, rtol=1e-5)

    # Rebuilding the environment from aggregates matches the stored arrays.
    trimmed = {
        k: v for k, v in ss_output.items() if k not in cl._HOUSEHOLD_INPUT_KEYS
    }
    env_stored = cl.HouseholdEnvironment.from_ss_output(ss_output, p)
    env_rebuilt = cl.HouseholdEnvironment.from_ss_output(trimmed, p)
    assert np.allclose(env_stored.p_i, env_rebuilt.p_i)
    assert np.isclose(env_stored.p_tilde, env_rebuilt.p_tilde)
    assert np.allclose(env_stored.bq, env_rebuilt.bq)
    assert np.allclose(env_stored.tr, env_rebuilt.tr)
    assert np.allclose(env_stored.ubi, env_rebuilt.ubi)


# ---------------------------------------------------------------------------
# Phase 3: chi_n inversion
# ---------------------------------------------------------------------------


class MockLaborParams:
    """
    Parameter object with the elliptical utility fields chi_n updates need.
    """

    S = 80
    J = 2
    starting_age = 20
    ending_age = 100
    ltilde = 1.0
    b_ellipse = 0.573
    upsilon = 2.856
    lambdas = np.array([0.6, 0.4]).reshape(2, 1)
    omega_SS = np.ones(80) / 80
    FOC_root_method = "hybr"
    _data = {"chi_n": {"validators": {"range": {"min": 0.0, "max": 1e4}}}}

    def __init__(self):
        self.chi_n = np.tile(np.linspace(20.0, 80.0, self.S), (2, 1))

    def update_specifications(self, revision):
        chi_n = np.asarray(revision["chi_n"], dtype=float)
        if np.any(chi_n > 1e4) or np.any(chi_n < 0):
            raise ValueError("chi_n out of range")
        self.chi_n = np.tile(chi_n, (2, 1))


def _hours_from_foc(chi_n, lhs, p):
    """
    Solve chi_n * MDU(n) = lhs for n, age by age, by bisection.
    """
    from ogcore import household

    n = np.zeros_like(chi_n)
    for s in range(chi_n.size):
        lo, hi = 1e-6, p.ltilde - 1e-6
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            value = float(household.marg_ut_labor(np.array([mid]), 1.0, p))
            if chi_n[s] * value > lhs[s]:
                hi = mid
            else:
                lo = mid
        n[s] = 0.5 * (lo + hi)
    return n


def test_aggregate_labor_by_age_uses_type_weights():
    """
    Hours are averaged over types with within-age population weights.
    """
    p = MockLaborParams()
    n = np.zeros((p.S, p.J))
    n[:, 0] = 0.2
    n[:, 1] = 0.5
    ages = np.array([20, 50, 79])
    labor = cl.aggregate_labor_by_age(n, p, ages)
    assert np.allclose(labor, 0.6 * 0.2 + 0.4 * 0.5)


def test_chi_n_update_is_exact_with_fixed_lhs():
    """
    One full step hits the target when the FOC left-hand side is fixed.
    """
    p = MockLaborParams()
    chi_n = np.array([20.0, 40.0, 60.0])
    lhs = np.array([1.5, 1.5, 1.5])
    n_model = _hours_from_foc(chi_n, lhs, p)
    n_target = np.array([0.3, 0.2, 0.05])
    new_chi_n = cl.chi_n_update(chi_n, n_model, n_target, p)
    assert np.allclose(_hours_from_foc(new_chi_n, lhs, p), n_target, atol=1e-6)
    # Higher target hours require lower chi_n and vice versa.
    assert np.all((new_chi_n > chi_n) == (n_target < n_model))
    # No step with zero damping.
    assert np.allclose(
        cl.chi_n_update(chi_n, n_model, n_target, p, 0.0), chi_n
    )


def test_invert_chi_n_converges_on_fixed_lhs_household_block(monkeypatch):
    """
    With a fixed FOC left-hand side the inversion converges in a few passes.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockLaborParams()
    lhs = np.linspace(2.0, 1.0, p.S)  # falls with age like w*e*(1-mtr)*c^-s
    calls = []

    def fake_partial_equilibrium_ss(
        ss_output, params, client=None, b_guess=None, n_guess=None
    ):
        calls.append(1)
        chi_n = params.chi_n[-1, :]
        n_one = _hours_from_foc(chi_n, lhs, params)
        n = np.tile(n_one.reshape(-1, 1), (1, params.J))
        updated = dict(ss_output)
        updated["n"] = n
        updated["b_sp1"] = ss_output["b_sp1"]
        return updated, cl.HouseholdSolution(
            b_sp1=updated["b_sp1"],
            n=n,
            euler_errors=np.zeros((2 * params.S, params.J)),
            success=np.ones(params.J, dtype=bool),
        )

    monkeypatch.setattr(
        cl, "partial_equilibrium_ss", fake_partial_equilibrium_ss
    )
    config = elp.LifecycleCalibrationConfig(chi_n_tail_method="flat")
    ages = config.moment_ages
    target = np.interp(ages, [20, 30, 60, 79], [0.18, 0.30, 0.25, 0.04])
    ss_output = {
        "b_sp1": np.ones((p.S, p.J)),
        "n": np.ones((p.S, p.J)) * 0.3,
    }

    result = cl.invert_chi_n(ss_output, p, target, config=config, tol=1e-6)

    assert result.converged
    assert result.iterations <= 3
    assert len(calls) == result.iterations
    assert np.allclose(result.labor_model, target, rtol=1e-5)
    assert result.chi_n.shape == (p.S,)
    assert np.allclose(p.chi_n[-1, :], result.chi_n)
    # Flat tail in levels beyond the last target age.
    assert np.allclose(result.chi_n[60:], result.chi_n[59])
    assert result.capped_ages.size == 0
    # chi_n rises steeply where target hours are tiny.
    assert result.chi_n[59] > 10 * result.chi_n[30]


def test_invert_chi_n_reports_capped_ages(monkeypatch):
    """
    Ages whose required chi_n exceeds the bound are clipped and reported.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockLaborParams()
    lhs = np.full(p.S, 3.0)

    def fake_partial_equilibrium_ss(
        ss_output, params, client=None, b_guess=None, n_guess=None
    ):
        n_one = _hours_from_foc(params.chi_n[-1, :], lhs, params)
        n = np.tile(n_one.reshape(-1, 1), (1, params.J))
        updated = dict(ss_output)
        updated["n"] = n
        return updated, cl.HouseholdSolution(
            b_sp1=ss_output["b_sp1"],
            n=n,
            euler_errors=np.zeros((2 * params.S, params.J)),
            success=np.ones(params.J, dtype=bool),
        )

    monkeypatch.setattr(
        cl, "partial_equilibrium_ss", fake_partial_equilibrium_ss
    )
    config = elp.LifecycleCalibrationConfig()
    target = np.full(config.moment_ages.size, 0.3)
    target[-5:] = 0.001  # needs chi_n far above the cap
    ss_output = {"b_sp1": np.ones((p.S, p.J)), "n": np.ones((p.S, p.J)) * 0.3}

    result = cl.invert_chi_n(
        ss_output, p, target, config=config, chi_n_max=500.0, max_iter=5
    )

    assert not result.converged
    assert result.iterations == 5
    assert np.array_equal(result.capped_ages, config.moment_ages[-5:])
    assert np.all(result.chi_n <= 500.0)
    assert np.allclose(result.labor_model[:-5], 0.3, rtol=1e-3)


def test_invert_chi_n_validates_targets():
    """
    Targets must be one per age and strictly inside the time endowment.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockLaborParams()
    config = elp.LifecycleCalibrationConfig()
    ss_output = {"b_sp1": np.ones((p.S, p.J)), "n": np.ones((p.S, p.J))}
    with pytest.raises(ValueError, match="one value per target age"):
        cl.invert_chi_n(ss_output, p, np.ones(5) * 0.3, config=config)
    bad = np.full(config.moment_ages.size, 0.3)
    bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly inside"):
        cl.invert_chi_n(ss_output, p, bad, config=config)
