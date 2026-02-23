from . import uncertainties
from . import sampling
from . import plotting
from . import utils

# gui is NOT imported eagerly to avoid pulling in matplotlib.pyplot
# at package load time (breaks headless / server environments).
# Use:  from perturbed_equilibria import gui

# Public API re-exports
from .sampling import (
    generate_perturbed_equilibria,
    generate_perturbed_GPR,
    verify_gpr_statistics,
    perturb_kinetic_equilibrium,
)

from .uncertainties import (
    new_uncertainty_profiles
)

from .plotting import plot_family
from .utils import (
    initialize_equilibrium_database,
    store_equilibrium,
    load_equilibrium,
    store_baseline_profiles,
    load_baseline_profiles,
    discover_scan_values,
    count_equilibria,
)
