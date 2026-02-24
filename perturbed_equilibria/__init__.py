from . import uncertainties
from . import sampling
from . import plotting
from . import utils

# gui is NOT imported eagerly to avoid pulling in matplotlib.pyplot
# at package load time (breaks headless / server environments).
# Use:  from perturbed_equilibria import gui

# Public API re-exports
from .sampling import (
    GPRProfilePerturber,
    generate_perturbed_GPR,
    verify_gpr_statistics,
    calc_cylindrical_li_proxy,
    perturb_kinetic_equilibrium,
    generate_perturbed_equilibria,
)

from .uncertainties import (
    new_uncertainty_profiles,
)

from .plotting import (
    draw_kinetic_profiles,
    draw_pressure_profiles,
    draw_jphi_profiles,
    plot_family,
    plot_kinetic_profiles,
    plot_jphi_profiles,
)

from .utils import (
    Hmode_profiles,
    Ip_flux_integral_vs_target,
    initialize_equilibrium_database,
    store_equilibrium,
    load_equilibrium,
    load_equilibrium_by_path,
    store_baseline_profiles,
    load_baseline_profiles,
    discover_scan_values,
    count_equilibria,
    read_eqdsk_from_bytes,
)
