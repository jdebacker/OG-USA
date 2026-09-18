"""
Compare lifecycle labor supply and chi_n between two parameterizations.

Produces two figures:

* ``labor_supply_profiles.png``: steady-state labor supply by age for the
  current OG-USA default parameters and for a newly calibrated parameter
  set, with the CPS data profile (survey-weighted mean weekly hours by age
  as a share of the 112-hour time endowment) drawn with
  ``ogcore.output_plots.ss_profiles``.
* ``chi_n_comparison.png`` and ``chi_n_comparison_log.png``: the chi_n age
  profiles from ``ogcore.parameter_plots.plot_chi_n`` in levels and on a
  log scale (the calibrated profile spans roughly 40 to 7,000).

Steady states are solved serially unless saved pickles are supplied, and
any newly solved steady state is saved next to the figures for reuse.

Example::

    uv run python examples/plot_lifecycle_calibration.py \
        --params examples/lifecycle_calibration/calibrated_params_total_income.json \
        --current-ss examples/lifecycle_calibration/ss_current_params.pkl \
        --new-ss examples/lifecycle_calibration/ss_calibrated_total_income.pkl
"""

import argparse
import json
import logging
import os
import pickle
import time
import warnings
from importlib import resources

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ogcore  # noqa: E402
from ogcore import SS  # noqa: E402
from ogcore import output_plots as op  # noqa: E402
from ogcore import parameter_plots as pp  # noqa: E402
from ogcore.parameters import Specifications  # noqa: E402

from ogusa import compute_moments as cm  # noqa: E402

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(CUR_DIR, "lifecycle_calibration")
HOURS_IN_TIME_ENDOWMENT = (24 - 8) * 7


def load_default_spec():
    """OG-USA default parameters as a baseline Specifications object."""
    p = Specifications(baseline=True, num_workers=1)
    with (
        resources.files("ogusa")
        .joinpath("ogusa_default_parameters.json")
        .open() as file
    ):
        p.update_specifications(json.load(file))
    return p


def load_or_solve_ss(p, pickle_path, save_path, label):
    """Load a steady state from ``pickle_path`` or solve and save it."""
    if pickle_path is not None and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    print(f"Solving steady state for {label} parameters (serial)...")
    start = time.time()
    ss_output = SS.run_SS(p, client=None)
    print(f"  done in {time.time() - start:.0f}s")
    with open(save_path, "wb") as file:
        pickle.dump(ss_output, file)
    return ss_output


def cps_hours_profile(min_age, max_age):
    """Survey-weighted mean weekly hours by age as a share of the endowment.

    ``compute_moments.get_age_profile_moments`` returns the profile indexed
    by model age 20 through 99 with NaN outside ``[min_age, max_age]``,
    already scaled by the 112-hour weekly time endowment.
    """
    profile = cm.get_age_profile_moments(
        "hours", min_age=min_age, max_age=max_age, hours_source="cps"
    )
    return profile.to_numpy(dtype=float)


def plot_labor_profiles(
    ss_current,
    p_current,
    ss_new,
    p_new,
    data_profile,
    labels,
    output_dir,
):
    """Labor supply by age for both parameterizations plus the CPS data.

    ``ss_profiles`` labels the two model series "Baseline" and "Reform" and
    rescales the data series so its first point equals the baseline model's
    first point.  Both are undone here: the lines are relabeled, and the
    data line is reset to the survey levels so the level gap is visible.
    """
    fig = op.ss_profiles(
        ss_current,
        p_current,
        reform_ss=ss_new,
        reform_params=p_new,
        by_j=False,
        var="n",
        plot_data=data_profile,
        plot_title="Labor supply by age",
    )
    ax = fig.gca()
    relabel = {"Baseline": labels[0], "Reform": labels[1], "Data": "CPS data"}
    for line in ax.get_lines():
        name = line.get_label()
        if name == "Data":
            line.set_ydata(data_profile)
        if name in relabel:
            line.set_label(relabel[name])
    ax.relim()
    ax.autoscale_view()
    ax.set_ylabel("Labor supply, share of time endowment")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "labor_supply_profiles.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_chi_n_profiles(p_current, p_new, labels, output_dir):
    """chi_n by age for both parameterizations, in levels and log scale."""
    year = int(p_current.start_year)
    paths = []
    for suffix, log_scale in (("", False), ("_log", True)):
        fig = pp.plot_chi_n(
            [p_current, p_new], labels=list(labels), years_to_plot=[year]
        )
        ax = fig.gca()
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper left")
        fig.tight_layout()
        path = os.path.join(output_dir, f"chi_n_comparison{suffix}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return paths


def aggregate_labor(ss_output, p):
    """Population-weighted labor supply by age, matching the moments code."""
    omega = np.asarray(p.omega_SS, dtype=float)
    lambdas = np.asarray(p.lambdas, dtype=float).reshape(-1)
    if omega.ndim == 1:
        weights = omega.reshape(-1, 1) * lambdas.reshape(1, -1)
    else:
        weights = omega
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (np.asarray(ss_output["n"]) * weights).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        required=True,
        help="JSON file with beta_annual, chi_b, and chi_n for the new "
        "parameterization (update_specifications format).",
    )
    parser.add_argument(
        "--current-ss",
        default=None,
        help="Pickle of the steady state at the current default parameters "
        "(solved and saved if missing).",
    )
    parser.add_argument(
        "--new-ss",
        default=None,
        help="Pickle of the steady state at the new parameters (solved and "
        "saved if missing).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and any newly solved steady states.",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=("Current parameters", "Calibrated parameters"),
        metavar=("CURRENT", "NEW"),
        help="Legend labels for the two parameterizations.",
    )
    parser.add_argument("--min-age", type=int, default=20)
    parser.add_argument("--max-age", type=int, default=79)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("ogcore").setLevel(logging.WARNING)
    ogcore.config.VERBOSE = False
    os.makedirs(args.output_dir, exist_ok=True)

    p_current = load_default_spec()
    p_new = load_default_spec()
    with open(args.params, "r", encoding="utf-8") as file:
        new_params = json.load(file)
    p_new.update_specifications(
        {k: new_params[k] for k in ("beta_annual", "chi_b", "chi_n")}
    )

    ss_current = load_or_solve_ss(
        p_current,
        args.current_ss,
        os.path.join(args.output_dir, "ss_current_params.pkl"),
        "current",
    )
    ss_new = load_or_solve_ss(
        p_new,
        args.new_ss,
        os.path.join(args.output_dir, "ss_new_params.pkl"),
        "new",
    )

    data_profile = cps_hours_profile(args.min_age, args.max_age)
    labor_path = plot_labor_profiles(
        ss_current,
        p_current,
        ss_new,
        p_new,
        data_profile,
        args.labels,
        args.output_dir,
    )
    chi_paths = plot_chi_n_profiles(
        p_current, p_new, args.labels, args.output_dir
    )

    ages = np.arange(
        p_current.starting_age, p_current.starting_age + p_current.S
    )
    table = pd.DataFrame(
        {
            "age": ages,
            "cps_data": data_profile,
            args.labels[0]: aggregate_labor(ss_current, p_current),
            args.labels[1]: aggregate_labor(ss_new, p_new),
            "chi_n_" + args.labels[0]: p_current.chi_n[-1, :],
            "chi_n_" + args.labels[1]: p_new.chi_n[-1, :],
        }
    )
    table_path = os.path.join(args.output_dir, "labor_supply_and_chi_n.csv")
    table.to_csv(table_path, index=False)
    print(table.iloc[::5].round(4).to_string(index=False))
    print("\nWrote:")
    for path in [labor_path, *chi_paths, table_path]:
        print("  ", path)


if __name__ == "__main__":
    main()
