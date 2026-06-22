"""
Tests for the joint lifecycle preference calibration helpers.
"""

import logging

import numpy as np
import pandas as pd

from ogusa import estimate_lifecycle_params as elp


class MockParams:
    """
    Minimal parameter object for lifecycle calibration helper tests.
    """

    S = 80
    J = 10
    starting_age = 20
    ending_age = 100
    lambdas = np.array(
        [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.0009, 0.0001]
    ).reshape(10, 1)
    omega_SS = np.ones(80) / 80
    g_n_ss = 0.01
    g_y = 0.02
    delta = 0.05
    beta_annual = np.linspace(0.91, 0.995, 10)
    chi_b = np.ones(10) * 80
    chi_n = np.linspace(20, 80, 80)


def _mock_ss_output(p, scale=1.0):
    """
    Build the subset of SS output used to warm-start repeated solves.
    """
    return {
        "b_sp1": np.ones((p.S, p.J)) * scale,
        "n": np.ones((p.S, p.J)) * 0.4 * scale,
        "r_p": 0.04 * scale,
        "r": 0.04 * scale,
        "w": 1.2 * scale,
        "p_m": np.ones(1) * scale,
        "B": 2.0 * scale,
        "K_d": 3.0 * scale,
        "Y": 10.0 * scale,
        "BQ": np.ones(p.J) * scale,
        "TR": 0.2 * scale,
        "factor": scale,
    }


def test_default_config_dimensions():
    """
    Test that the default setup gives 30 parameters and 130 moments.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    assert config.estimated_chi_n_ages.size == 60
    assert config.wealth_profile_ages.size == 59
    assert config.chi_n_n_spline_knots == 10
    assert config.chi_n_spline_degree == 3
    assert config.moment_distance_method == "relative"
    assert p.J + p.J + config.chi_n_n_spline_knots == 30

    ss_output = {
        "n": np.ones((p.S, p.J)) * 0.35,
        "b_sp1": np.arange(1, p.S * p.J + 1).reshape(p.S, p.J),
        "before_tax_income": (
            np.arange(1, p.S * p.J + 1).reshape(p.S, p.J) + 1000
        ),
        "B": 2.0,
        "K_d": 3.0,
        "Y": 10.0,
        "factor": 2.0,
    }

    moments = elp.compute_model_moments(ss_output, p, config)

    assert len(moments.names) == 130
    assert moments.values.shape == (130,)
    assert moments.names[0] == "labor_supply_age_20"
    assert moments.names[59] == "labor_supply_age_79"
    assert moments.names[60] == "net_wealth_age_21"
    assert moments.names[118] == "net_wealth_age_79"
    assert moments.names[119] == "income_gini"
    assert moments.names[120] == "savings_rate"
    growth = (1 + p.g_n_ss) * np.exp(p.g_y)
    expected_savings_rate = ((growth - 1.0) * 2.0 + p.delta * 3.0) / 10.0
    assert np.allclose(moments.values[120], expected_savings_rate)
    assert moments.names[-1] == "wealth_var_log"


def test_pack_unpack_lifecycle_params_roundtrip():
    """
    Test transformed packing and unpacking of lifecycle parameters.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    beta = np.linspace(0.92, 0.99, p.J)
    chi_b = np.linspace(50, 100, p.J)
    ages = np.arange(p.starting_age, p.starting_age + p.S)
    basis, _ = elp._build_chi_n_spline_basis(
        ages,
        config.chi_n_n_spline_knots,
        config.chi_n_spline_degree,
    )
    gamma = np.linspace(np.log(10), np.log(70), config.chi_n_n_spline_knots)
    chi_n = np.exp(basis @ gamma)

    theta = elp.pack_lifecycle_params(beta, chi_b, chi_n, p, config)
    unpacked = elp.unpack_lifecycle_params(
        theta,
        p,
        config,
        base_chi_n=p.chi_n,
    )

    assert theta.size == 30
    assert np.allclose(unpacked["beta_annual"], beta)
    assert np.allclose(unpacked["chi_b"], chi_b)
    assert np.allclose(unpacked["chi_n"], chi_n)


def test_build_chi_n_profile_scales_default_tail():
    """
    Test that the default chi_n tail shape is preserved and scaled.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    base_chi_n = np.arange(1, p.S + 1, dtype=float)
    estimated = np.ones(config.estimated_chi_n_ages.size) * 10

    full = elp.build_chi_n_profile(estimated, base_chi_n, p, config)

    assert np.allclose(full[:60], 10)
    expected_scale = 10 / base_chi_n[59]
    assert np.allclose(full[60:], base_chi_n[60:] * expected_scale)


def test_weighting_matrix_handles_singular_bootstrap_vcv():
    """
    Test pseudo-inverse/ridge handling for singular bootstrap VCV matrices.
    """
    boot = np.ones((5, 3))

    W = elp.weighting_matrix(
        3,
        method="optimal",
        bootstrap_moments=boot,
        ridge=1e-6,
    )

    assert W.shape == (3, 3)
    assert np.all(np.isfinite(W))
    assert np.allclose(W, W.T)


def test_smm_distance_supports_relative_and_absolute_residuals():
    """
    Test percentage-deviation and level-deviation distance calculations.
    """
    model = elp.MomentSet(("a", "b", "c"), np.array([2.0, 6.0, 2e-9]))
    data = elp.MomentSet(("a", "b", "c"), np.array([1.0, 3.0, 0.0]))
    W = np.eye(3)

    absolute = elp.smm_distance(model, data, W, method="absolute")
    relative = elp.smm_distance(
        model,
        data,
        W,
        method="relative",
        floor=1e-8,
    )

    assert np.allclose(absolute, 1.0**2 + 3.0**2 + (2e-9) ** 2)
    assert np.allclose(relative, 1.0**2 + 1.0**2 + 0.2**2)


def test_compute_inheritance_moments_from_scf():
    """
    Test optional inherited-transfer moments from full SCF-like data.
    """
    scf = pd.DataFrame(
        {
            "inheritance": [0.0, 100.0, 200.0],
            "received": [0, 1, 1],
            "networth_infadj": [50.0, 1000.0, 2000.0],
            "wgt": [1.0, 1.0, 2.0],
        }
    )

    moments = elp.compute_inheritance_moments_from_scf(
        scf,
        amount_col="inheritance",
        received_col="received",
    )

    assert moments.names == (
        "inheritance_received_rate",
        "inheritance_amount_conditional_mean",
        "inheritance_to_networth_mean",
    )
    assert np.allclose(moments.values[0], 0.75)
    assert np.allclose(moments.values[1], (100 + 2 * 200) / 3)


def test_compute_data_moments_with_synthetic_microdata():
    """
    Test data moment construction without Tax-Calculator or web access.
    """
    p = MockParams()
    ages = np.arange(20, 80)
    cps = pd.DataFrame(
        {
            "age": ages,
            "hours_per_week": np.linspace(20, 40, ages.size),
            "weight": np.ones(ages.size),
        }
    )
    scf = pd.DataFrame(
        {
            "age": ages,
            "networth_infadj": np.linspace(1000, 100000, ages.size),
            "networth": np.linspace(1000, 100000, ages.size),
            "wgt": np.ones(ages.size),
        }
    )
    config = elp.LifecycleCalibrationConfig(include_income_gini=False)

    moments = elp.compute_data_moments(
        p,
        config,
        cps=cps,
        scf=scf,
        savings_rate=0.17,
    )

    assert len(moments.names) == 129
    assert moments.values.shape == (129,)
    anchor_mean = scf.loc[scf["age"].between(20, 24), "networth_infadj"].mean()
    age_21_wealth = scf.loc[scf["age"] == 21, "networth_infadj"].iloc[0]
    assert np.allclose(moments.values[60], age_21_wealth / anchor_mean)
    assert moments.names[119] == "savings_rate"
    assert np.allclose(moments.values[119], 0.17)
    assert moments.names[-9:] == elp._wealth_distribution_moment_names()


def test_savings_rate_data_moment_uses_macro_moment(monkeypatch):
    """
    Test that the savings-rate data moment comes from macro moments.
    """

    def fake_get_macro_moments(year):
        assert year == 2030
        return {elp.SAVINGS_RATE_DATA_LABEL: 0.18}

    monkeypatch.setattr(
        elp.compute_moments,
        "get_macro_moments",
        fake_get_macro_moments,
    )

    assert np.allclose(elp.savings_rate_data_moment(macro_year=2030), 0.18)


def test_estimate_lifecycle_params_logs_optimizer_progress(
    monkeypatch, caplog
):
    """
    Test that optimizer progress logging uses cached objective values.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(
        include_labor_profile=False,
        include_wealth_profile=False,
        include_income_gini=False,
        include_savings_rate=False,
        include_wealth_distribution=False,
        optimizer_options={"maxiter": 2, "maxfun": 4},
        log_optimizer_progress=True,
    )
    data_moments = elp.MomentSet(("moment",), np.array([0.0]))
    W = np.eye(1)
    theta0 = np.array([0.1])
    objective_values = [4.0, 3.0]

    def fake_smm_objective(theta, *args):
        return objective_values.pop(0)

    def fake_minimize(
        fun,
        x0,
        method=None,
        tol=None,
        options=None,
        callback=None,
    ):
        first = fun(x0)
        assert np.isclose(first, 4.0)
        callback(x0)
        second = fun(x0)
        assert np.isclose(second, 3.0)
        callback(x0)
        return elp.opt.OptimizeResult(
            x=x0,
            fun=second,
            success=True,
            message="ok",
            nit=2,
            nfev=2,
        )

    def fake_unpack_lifecycle_params(*args, **kwargs):
        return {
            "beta_annual": np.ones(p.J) * 0.95,
            "chi_b": np.ones(p.J),
            "chi_n": np.ones(p.S),
        }

    monkeypatch.setattr(elp, "smm_objective", fake_smm_objective)
    monkeypatch.setattr(elp.opt, "minimize", fake_minimize)
    monkeypatch.setattr(
        elp, "unpack_lifecycle_params", fake_unpack_lifecycle_params
    )
    monkeypatch.setattr(elp, "apply_lifecycle_params", lambda *args: None)
    monkeypatch.setattr(
        elp,
        "solve_ss_with_cache",
        lambda *args, **kwargs: _mock_ss_output(p),
    )
    monkeypatch.setattr(
        elp,
        "compute_model_moments",
        lambda *args, **kwargs: data_moments,
    )
    caplog.set_level(logging.INFO, logger=elp.logger.name)

    result = elp.estimate_lifecycle_params(
        p,
        config=config,
        theta0=theta0,
        data_moments=data_moments,
        W=W,
    )

    assert result.objective_value == 3.0
    assert objective_values == []
    assert "Lifecycle SMM iteration 1/2" in caplog.text
    assert "objective=4.000000e+00" in caplog.text
    assert "Lifecycle SMM iteration 2/2" in caplog.text
    assert "objective=3.000000e+00" in caplog.text
    assert "function_evals=2" in caplog.text


def test_solve_ss_with_cache_uses_run_ss_then_ss_solver(monkeypatch):
    """
    Test that repeated SS solves use the previous solution as a warm start.
    """
    p = MockParams()
    calls = []

    def fake_run_ss(params, client=None):
        calls.append(("run", client))
        return _mock_ss_output(params, scale=1.0)

    def fake_ss_solver(
        b_sp1,
        n,
        r_p,
        r,
        w,
        p_m,
        y,
        bq,
        tr,
        ig_baseline,
        factor,
        params,
        client=None,
    ):
        calls.append(("solver", factor, client))
        assert np.allclose(b_sp1, np.ones((params.S, params.J)))
        assert np.allclose(n, np.ones((params.S, params.J)) * 0.4)
        assert np.isclose(r_p, 0.04)
        assert np.isclose(r, 0.04)
        assert np.isclose(w, 1.2)
        assert np.allclose(p_m, np.ones(1))
        assert np.isclose(y, 10.0)
        assert np.allclose(bq, np.ones(params.J))
        assert np.isclose(tr, 0.2)
        assert ig_baseline is None
        return _mock_ss_output(params, scale=2.0)

    monkeypatch.setattr(elp.SS, "run_SS", fake_run_ss)
    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)

    cache = elp.SSSolutionCache(use_ss_solver=True)
    first = elp.solve_ss_with_cache(p, client="client", ss_cache=cache)
    second = elp.solve_ss_with_cache(p, client="client", ss_cache=cache)

    assert [call[0] for call in calls] == ["run", "solver"]
    assert first["factor"] == 1.0
    assert second["factor"] == 2.0
    assert cache.previous_output is second


def test_solve_ss_with_cache_falls_back_to_run_ss(monkeypatch):
    """
    Test fallback to SS.run_SS if the direct solver restart fails.
    """
    p = MockParams()
    calls = []

    def fake_run_ss(params, client=None):
        calls.append("run")
        return _mock_ss_output(params, scale=3.0)

    def fake_ss_solver(*args, **kwargs):
        calls.append("solver")
        raise RuntimeError("failed warm start")

    monkeypatch.setattr(elp.SS, "run_SS", fake_run_ss)
    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)

    cache = elp.SSSolutionCache(
        previous_output=_mock_ss_output(p, scale=1.0),
        use_ss_solver=True,
    )
    output = elp.solve_ss_with_cache(p, ss_cache=cache)

    assert calls == ["solver", "run"]
    assert output["factor"] == 3.0
    assert cache.previous_output is output


def test_solve_ss_with_cache_can_disable_ss_solver(monkeypatch):
    """
    Test that the restart path can be disabled for robustness checks.
    """
    p = MockParams()
    calls = []

    def fake_run_ss(params, client=None):
        calls.append("run")
        return _mock_ss_output(params, scale=4.0)

    def fake_ss_solver(*args, **kwargs):
        calls.append("solver")
        return _mock_ss_output(p, scale=5.0)

    monkeypatch.setattr(elp.SS, "run_SS", fake_run_ss)
    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)

    cache = elp.SSSolutionCache(
        previous_output=_mock_ss_output(p, scale=1.0),
        use_ss_solver=False,
    )
    output = elp.solve_ss_with_cache(p, ss_cache=cache)

    assert calls == ["run"]
    assert output["factor"] == 4.0
