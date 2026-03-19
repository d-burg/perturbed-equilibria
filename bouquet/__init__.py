from . import uncertainties
from . import sampling
from . import TokaMaker_interface
from . import plotting
from . import utils
from . import io

# gui is NOT imported eagerly to avoid pulling in matplotlib.pyplot
# at package load time (breaks headless / server environments).
# Use:  from bouquet import gui

# Public API re-exports
from .sampling import (
    GPRProfilePerturber,
    generate_perturbed_GPR,
    sigmoid_length_scale,
    verify_gpr_statistics,
    calc_cylindrical_li_proxy,
)

from .TokaMaker_interface import (
    fit_inductive_profile,
    perturb_kinetic_equilibrium,
    generate_bouquet,
    reconstruct_equilibrium,
)

from .uncertainties import (
    new_uncertainty_profiles,
)

from .plotting import (
    draw_kinetic_profiles,
    draw_pressure_profiles,
    draw_jphi_total,
    draw_jphi_components,
    draw_jphi_profiles,
    plot_bouquet,
    plot_tokamaker_comparison,
    plot_geqdsk_bouquet,
    plot_pfile_bouquet,
    plot_coil_currents,
    plot_kinetic_profiles,
    plot_jphi_profiles,
)

from .io import (
    GEQDSKEquilibrium,
    read_geqdsk,
    PFile,
    read_pfile,
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
