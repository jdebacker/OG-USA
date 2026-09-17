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
