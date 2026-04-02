import multiprocessing
from distributed import Client
import os
import json
import importlib.resources
from ogusa.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore.utils import params_to_json


def main():
    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 7)
    client = Client(n_workers=num_workers, threads_per_worker=1)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))

    """
    ---------------------------------------------------------------------------
    Run baseline policy
    ---------------------------------------------------------------------------
    """
    # Set up baseline parameterization
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
    )
    # Update parameters for baseline from default json file
    with importlib.resources.open_text(
        "ogusa", "ogusa_default_parameters.json"
    ) as file:
        defaults = json.load(file)
    p.update_specifications(defaults)
    c = Calibration(
        p, estimate_tax_functions=True, estimate_pop=True, client=client
    )
    p.update_specifications(c.get_dict())
    # save to json file
    params_to_json(p, os.path.join(CUR_DIR, "ogusa_default_parameters.json"))
