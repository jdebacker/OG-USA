# Plan: calibrating beta, chi_b, and chi_n in OG-USA

Status as of 2026-09-16. Branch: `smm`.

## Background

The `smm` branch added `ogusa/estimate_lifecycle_params.py`, a joint
steady-state simulated method of moments (SMM) estimator for `beta_annual`
(one per lifetime-income type), `chi_b` (one per type), and a 10-knot
log B-spline for the `chi_n` age profile, against roughly 130 moments.
Each objective evaluation solved the full general-equilibrium steady state.

Review findings that motivate this plan:

- About three quarters of the objective came from hours at ages 70 to 79.
  The CPS decline there is almost entirely participation, which the model
  lacks. With a Frisch elasticity of 0.4 the elliptical disutility has
  upsilon of about 2.86, so moving hours from 0.30 to 0.035 needs `chi_n`
  to rise by a factor of roughly 55. A 10-knot spline cannot produce that
  cliff without distorting ages 55 to 70.
- `beta_j` and `chi_b_j` both raise wealth for type `j` and are not
  separately identified by aggregate age profiles. The hard-coded seven
  wealth-share bins put the four top-1% types behind a single moment.
- Relative deviations explode on near-zero data moments: the bottom-25%
  wealth share is negative in the SCF, and the wealth profile was anchored
  to mean net worth at ages 20 to 24, a thin and noisy cell.
- Income Gini and variance of log wealth do not depend on the preference
  parameters to first order and cannot be matched by a deterministic model
  with ten types.
- The committed DFO-LS driver had never run (bounds bug, `dfo-ls` not
  installed). Saved results came from an earlier L-BFGS-B version that
  stopped after three iterations.

## On J

Data and model moments always agree with each other; nothing uses J = 7
when the model has J = 10. But the wealth-share bins were hard-coded to
seven cuts on both sides, so the moment set did not scale with J. Using
`p.lambdas` as the bins gives one share moment per type. SCF 2019 shares
with the ten default lambdas (row-bootstrap standard errors understate the
true ones, especially in the top bin, which rests on about 87 households;
the SCF also excludes the Forbes 400):

| Bin (percentile) | Lambda | SCF share | Row-bootstrap SE |
|---|---|---|---|
| 0 to 25 | 0.25 | -0.005 | 0.0002 |
| 25 to 50 | 0.25 | 0.020 | 0.0005 |
| 50 to 70 | 0.20 | 0.055 | 0.0009 |
| 70 to 80 | 0.10 | 0.056 | 0.0009 |
| 80 to 90 | 0.10 | 0.110 | 0.0015 |
| 90 to 99 | 0.09 | 0.392 | 0.0047 |
| 99 to 99.5 | 0.005 | 0.092 | 0.0016 |
| 99.5 to 99.9 | 0.004 | 0.140 | 0.0026 |
| 99.9 to 99.99 | 0.0009 | 0.089 | 0.0020 |
| 99.99 to 100 | 0.0001 | 0.051 | 0.0042 |

The bottom bin is negative and cannot be matched, so `beta` for type 1 is
tied to type 2 and the bottom 50% is treated as one target.

## Phases

### Phase 0. Make the branch runnable and decide what to keep

Status: done 2026-09-16 (all items below).

- Install `dfo-ls` into the `.venv` (`uv sync`) and fix the failing test.
- Keep the data-side functions, `MomentSet`, and `compute_model_moments`.
- Fix the off-by-one in the wealth-profile age index (`b_sp1[s]` is wealth
  held at age `s + 1`).
- Replace the silent warm-start fallback with a logged warning and call
  `SS.SS_solver` by keyword so signature changes across OG-Core versions
  fail loudly instead of silently.
- Replace the 1e15 failure penalty with a bounded value so it does not
  poison the DFO-LS interpolation model.
- Retire the 30-parameter DFO-LS driver as the main path; it becomes the
  optional inference layer in Phase 6.
- Remove `ogusa/calibrate_chi_n.py`, which targets an OG-Core API that no
  longer exists.

### Phase 1. Redesign the moment set

Status: done 2026-09-16. Defaults in `LifecycleCalibrationConfig` now give
71 moments for J = 10 (60 hours ages, 10 shares, 1 old-age ratio). The SCF
old-age wealth ratio (75-79 over 60-64, household weighted) is 0.80. The
ability profiles are hourly-wage based (CWHS earnings with imputed hours,
see `docs/book/content/calibration/earnings.md`), so hours targets do not
double count. While fixing the bins, `wealth.compute_wealth_moments` was
found to drop the wealthiest observation from the top bin; fixed.

- Wealth shares use `p.lambdas` as bins on both sides; generalize the model
  side from seven hard-coded cuts to cumulative lambdas.
- Hours targets stay at single years of age 20 to 79 (the `chi_n` inversion
  needs them) with a light smoother on the CPS profile.
- Add one `chi_b` target: the ratio of mean SCF net worth at ages 75 to 79
  to that at 60 to 64. Report aggregate bequests over GDP as a check.
- Drop income Gini, variance of log wealth, and the anchored wealth profile
  as targets. Keep a wealth-by-age profile normalized by its mean over ages
  21 to 79 for diagnostics only.
- Confirm the ability profiles `e` are hourly-wage based; if they embed
  hours, the labor targets double count.

### Phase 2. Household-only solver wrapper

Status: done 2026-09-16 in `ogusa/calibrate_lifecycle.py`
(`HouseholdEnvironment`, `solve_households`, `partial_equilibrium_ss`).
Validated against a cold general-equilibrium solve with the default
parameters: the household-only re-solve reproduces `b_sp1` and `n` to
about 1e-12 relative error. Timings on the development machine:

| Solve | Time |
|---|---|
| General-equilibrium steady state, cold, 5 Dask workers | 1028 s |
| Household-only re-solve, serial | 0.5 s |
| Household-only re-solve, 5 Dask workers | 14 s |

The household block itself is cheap; the Dask path is dominated by
scattering the parameters object. Use `client=None` for the inner loops in
Phases 3 and 4. The general-equilibrium solve's cost is therefore mostly
outer-loop iterations and per-iteration Dask overhead, which is worth
revisiting in Phase 5 (a serial general-equilibrium solve may be faster).

A function that takes a steady-state output dictionary and solves only the
household block at fixed prices, taxes, bequests, transfers, and scaling
factor, using `SS.solve_for_j`, `aggregates.get_io_prices` (or
`io_matrix @ p_m` on older OG-Core), `aggregates.get_ptilde`, and the
`household.get_bq` / `get_tr` / `get_rm` helpers. Parallelize across types
with the existing Dask pattern, with a serial fallback.

### Phase 3. Concentrate out chi_n

Iterate: solve households at fixed prices, form lambda-weighted mean hours
by age, update each age's `chi_n` by the ratio of marginal disutility at
model hours to that at data hours, repeat. Hold `chi_n` flat in logs for
ages 80 to 99 at the age-79 value. Report where `chi_n` lands relative to
the 10,000 validator cap. If ages 75 to 79 need values near the cap, that
is the signal to switch the target at those ages to hours conditional on
working; that decision is the maintainer's.

### Phase 4. Fixed point on beta and chi_b

For types 2 through J, update `beta` in logit space toward each type's own
share with a damped diagonal secant; tie type 1 to type 2. Update the
common `chi_b` with a one-dimensional secant on the old-age wealth ratio.
Both run inside the Phase 2 wrapper. Converge when every share is within
one bootstrap standard error or within one percent, whichever is looser.

### Phase 5. Outer general-equilibrium loop

Solve the full steady state, run Phases 3 and 4, re-solve warm-started from
the previous solution, repeat until prices, bequests, transfers, and the
scaling factor stop moving. Requires one OG-Core change: let `run_SS`
accept initial guesses for the baseline case. Open that as a separate
OG-Core pull request early. Expect five to fifteen steady-state solves in
total (unmeasured; report actual timings from the first run).

### Phase 6. Integrate and validate

- Wire the routine into the `Calibration` class behind a flag; `get_dict`
  returns `beta_annual`, `chi_b`, and `chi_n`.
- Replace `check_smm.py` with an example script that runs the calibration,
  writes the moment comparison, and plots hours and wealth profiles in
  matching units.
- Validate with a baseline and one reform time path, since large `chi_n`
  at old ages is the most likely place for the time-path solver to struggle.
- Optionally run DFO-LS from the calibrated point over the beta and chi_b
  parameters with household-only evaluations for standard errors and an
  overidentification test (`compute_parameter_vcv`).

## Tests

- Unit tests for the `chi_n` and `beta` update rules on synthetic data.
- Lambdas-based data shares sum to one and match the model-side cuts.
- The household-only wrapper reproduces general-equilibrium savings and
  hours when handed equilibrium prices.
- The full calibration gets a `local` marker.

## Order and dependencies

Phases 0 and 1 are independent and can start immediately. Phase 2 does not
depend on Phase 1. Phases 3 and 4 depend on 2. Phase 5 depends on the
OG-Core change. Phase 6 depends on everything before it.

## Environment

Use `uv run python ...` for everything in this repo. The conda environments
on the development machine are stale (OG-Core 0.16.1) and cannot load the
current default parameters.
