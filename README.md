# bouquet

**BO**otstrap **U**ncertainty **QU**antified **E**quilibrium **T**oolkit

GP-sampled perturbed equilibria for uncertainty quantification with TokaMaker.

Bouquet generates families ("bouquets") of perturbed tokamak equilibria from a
baseline kinetic equilibrium (g-file + p-file) by drawing correlated profile
perturbations from Gaussian process regression posteriors, solving the
Grad–Shafranov equation for each sample, and archiving all results to a single
HDF5 database.

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue" alt="Python 3.9+"/>
<img src="https://img.shields.io/badge/version-0.1.0-green" alt="v0.1.0"/>
</p>

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)
- [IO Modules](#io-modules)
- [Plotting](#plotting)
- [HDF5 Database](#hdf5-database)
- [Examples](#examples)
- [Testing](#testing)
- [Architecture and Assumptions](#architecture-and-assumptions)
- [API Reference](#api-reference)

---

## Features

- **Gaussian Process perturbation** of kinetic profiles (n_e, T_e, n_i, T_i)
  with user-supplied uncertainty envelopes and spatially-varying correlation
  lengths (Gibbs non-stationary kernels).
- **Pressure and l_i matching**: perturbed profiles are constrained to match
  the baseline volume-averaged pressure and internal inductance.
- **Current decomposition**: explicit bootstrap (Sauter model) + inductive
  separation with iterative l_i convergence.
- **COCOS-aware GEQDSK reader** with full flux-surface geometry (κ, δ,
  squareness), safety factor, current density, and exact outboard-midplane
  profiles — all computed independently of external tools.
- **COCOS conversion and Bt/Ip flip**: convert g-file data between any two
  COCOS conventions (`cocosify`) and reverse the toroidal field / plasma
  current directions (`flip_Bt_Ip`), with save-to-disk and HDF5 round-trip
  support.  Verified field-by-field against OMFIT's `OMFITgeqdsk`.
- **P-file reader/writer** supporting the Osborne format with 24+ profile
  types, diamagnetic rotation computation, E×B decomposition, and radial
  electric field.
- **Self-contained HDF5 archive**: raw g-file and p-file bytes, 1-D profiles,
  scalar diagnostics, and coil currents — everything needed to reconstruct and
  visualise each equilibrium.
- **Comprehensive plotting**: multi-panel kinetic, current, geometry, and
  rotation diagnostics from a single function call.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/bouquet.git
cd bouquet

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| scipy | Interpolation, GPR kernels, optimisation |
| matplotlib | Plotting |
| h5py | HDF5 database I/O |
| [TokaMaker](https://github.com/hansec/OpenFUSIONToolkit) | Grad–Shafranov solver (required for equilibrium generation, not for IO/plotting) |

TokaMaker must be installed separately following the
[OpenFUSIONToolkit instructions](https://github.com/hansec/OpenFUSIONToolkit).
The IO and plotting modules work without TokaMaker.

---

## Quick Start

### Reading equilibrium files (no TokaMaker required)

```python
from bouquet import GEQDSKEquilibrium, read_pfile

# Load a g-file
eq = GEQDSKEquilibrium("g123456.01000")
print(f"Ip = {eq.Ip/1e6:.3f} MA")
print(f"q95 = {eq.q_profile[-1]:.2f}")
print(f"li(1) = {eq.li['li(1)']:.3f}")

# Flux-surface-averaged current density (from p' and FF')
j_phi = eq.j_tor_averaged            # <Jt/R> / <1/R>
j_phi_direct = eq.j_tor_averaged_direct  # literal <Jt> from GS equation

# Access flux-surface geometry
geo = eq.geometry
print(f"Elongation at boundary: {geo['kappa'][-1]:.2f}")

# Exact outboard-midplane profiles
mid = eq.midplane
print(f"R_mid at boundary: {mid['R'][-1]:.4f} m")

# Load a p-file
pf = read_pfile("p123456.01000")
ne = pf.get("ne")       # returns (psinorm, values, derivatives) tuple
Te = pf.get("te")
```

### COCOS conversion and Bt/Ip flip

```python
from bouquet import GEQDSKEquilibrium, read_geqdsk

# Load as COCOS 7, convert to COCOS 1, flip Bt/Ip
eq = read_geqdsk("g123456.01000", cocos=7)
eq.cocosify(1)       # in-place; use copy=True to get a new object
eq.flip_Bt_Ip()      # in-place

# Save to disk or serialise for HDF5
eq.save("modified.geqdsk")
raw_bytes = eq.to_bytes()          # for HDF5 storage
eq2 = GEQDSKEquilibrium.from_bytes(raw_bytes, cocos=1)  # reconstruct
```

### Generating a bouquet (requires TokaMaker)

```python
import os, sys
tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, 'python'))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker

from bouquet import (
    new_uncertainty_profiles,
    generate_bouquet,
    plot_bouquet,
    plot_geqdsk_bouquet,
    plot_pfile_bouquet,
)

# 1. Define uncertainties on the baseline profiles
sigma_ne, sigma_te, sigma_ni, sigma_ti, sigma_jphi = new_uncertainty_profiles(
    psi_N, ne, te, ni, ti, j_phi
)

# 2. Generate perturbed equilibria
generate_bouquet(
    mygs=mygs,                          # TokaMaker instance
    header="my_run",
    N=10,                               # number of perturbations
    psi_N=psi_N,
    ne=ne, te=te, ni=ni, ti=ti,
    j_phi=j_phi, j_BS=j_BS,
    sigma_ne=sigma_ne, sigma_te=sigma_te,
    sigma_ni=sigma_ni, sigma_ti=sigma_ti,
    sigma_jphi=sigma_jphi,
    Ip_target=Ip_target,
    l_i_target=li_target,
    eqdsk_bytes=open("baseline.geqdsk", "rb").read(),
    pfile_bytes=open("baseline.pfile", "rb").read(),
)

# 3. Visualise
plot_bouquet(h5path="my_run.h5")
plot_geqdsk_bouquet(h5path="my_run.h5", x_coord="psi_N")
plot_pfile_bouquet(h5path="my_run.h5", x_coord="psi_N")
```

---

## Workflow Overview

```
Baseline g-file + p-file
        │
        ▼
┌───────────────────────┐
│  Define uncertainties  │  new_uncertainty_profiles()
│  σ_ne, σ_Te, σ_ni, …  │  or user-supplied arrays
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Draw GPR perturbation │  GPRProfilePerturber
│  ne±δne, Te±δTe, …    │  (Gibbs kernel, monotonicity enforced)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Match pressure & li   │  Pressure rescaling + secant iteration
│  Decompose j = jBS+jI  │  on inductive amplitude
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Solve Grad–Shafranov  │  TokaMaker
│  Export g-file bytes   │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Build perturbed p-file│  Recompute ptot, diamagnetic,
│  Export p-file bytes   │  Er, ω_HB from perturbed profiles
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Store to HDF5         │  store_equilibrium()
│  (g-file + p-file raw  │
│   bytes + diagnostics) │
└───────────────────────┘
```

### What is perturbed vs. held fixed

| Quantity | Perturbed? | Notes |
|----------|:----------:|-------|
| n_e, T_e, n_i, T_i | ✓ | Drawn from GPR posterior |
| Total pressure (p_tot) | ✓ | Recomputed from perturbed kinetics |
| Bootstrap current (j_BS) | ✓ | Sauter model (optional) |
| Inductive current (j_ind) | ✓ | Scaled to match l_i |
| Coil currents | ✓ | Adjusted by TokaMaker to match I_p |
| Diamagnetic rotations | ✓ | Recomputed from perturbed n, T |
| E_r, ω_HB | ✓ | Recomputed using exact midplane B-fields |
| n_z1 (impurity density) | ✗ | Preserved from baseline |
| n_b, p_b (beam) | ✗ | Preserved from baseline |
| Toroidal/poloidal rotation | ✗ | Preserved from baseline |

---

## IO Modules

### GEQDSK Reader

```python
from bouquet import GEQDSKEquilibrium

eq = GEQDSKEquilibrium("g123456.01000", cocos=1)
```

| Property / Method | Description |
|-------------------|-------------|
| `psi_N` | Normalised poloidal flux grid (0 → 1) |
| `psi_N_RZ` | 2-D normalised poloidal flux on the (R, Z) grid |
| `psi_axis`, `psi_boundary` | Axis and boundary flux (Wb) |
| `Ip` | Plasma current (A, sign-corrected) |
| `q_profile` | Safety factor on psi_N |
| `j_tor_averaged` | `<Jt/R>/<1/R>` — standard convention (OMFIT, TRANSP) |
| `j_tor_averaged_direct` | Literal `<Jt>` from p' and FF' (GS equation) |
| `geometry` | Dict with R, Z, a, κ, δ, squareness per surface |
| `midplane` | Exact outboard-midplane R, Bp, Bt, Btot |
| `rhovn` | Normalised toroidal flux coordinate |
| `li` | Internal inductance dict (li(1), li(2), li(3), …) |
| `betas` | Plasma beta values (beta_t, beta_p, beta_n) |
| `cocos` | Current COCOS convention index |
| `cocosify(out)` | Convert between COCOS conventions (in-place or copy) |
| `flip_Bt_Ip()` | Reverse Bt and Ip signs (in-place or copy) |
| `save(path)` | Write modified g-file to disk |
| `to_bytes()` | Serialise to bytes for HDF5 storage |
| `from_bytes()` | Construct from in-memory bytes |

COCOS conventions 1–8 and 11–18 are fully supported.  See
[`bouquet/io/GEQDSK_QUANTITIES.md`](bouquet/io/GEQDSK_QUANTITIES.md)
for a complete reference of every derived quantity.

### P-File Reader/Writer

```python
from bouquet import PFile, read_pfile

pf = read_pfile("p123456.01000")

# Access profiles
psi, ne, dne = pf.get("ne")          # (psinorm, values, derivatives)
psi_grid = pf.psinorm_for("ne")      # just the grid

# Modify profiles
pf.set_profile("ne", new_psi, new_ne)

# Recompute derived quantities
pf.compute_pressure()
pf.compute_diamagnetic_rotations(psi_Wb)
pf.compute_rotation_decomposition(R=R_mid, Bp=Bp_mid, Bt=Bt_mid, psi=psi_Wb)

# Serialise
raw_bytes = pf.to_bytes()
pf2 = PFile.from_bytes(raw_bytes)
```

Supported profiles: `ne`, `te`, `ni`, `ti`, `nb`, `pb`, `ptot`, `nz1`,
`omeg`, `omegp`, `omgvb`, `omgpp`, `omgeb`, `er`, `ommvb`, `ommpp`,
`omevb`, `omepp`, `kpol`, `omghb`, `vtor1`, `vpol1`, and more.

---

## Plotting

All plotting functions return `(fig, axes)` and work in two modes:

1. **File mode** — pass file paths directly
2. **HDF5 mode** — pass `h5path` to load from a bouquet database

### Available functions

```python
from bouquet import (
    plot_bouquet,               # Full overview (kinetics + jphi + coils)
    plot_geqdsk_bouquet,        # 3×3 grid: pressure, current, q, geometry, li, flux surfaces
    plot_pfile_bouquet,         # Multi-panel: densities, temperatures, rotations
    plot_coil_currents,         # Bar chart of coil currents
    plot_tokamaker_comparison,  # TokaMaker vs source geqdsk comparison
    draw_kinetic_profiles,      # ne, Te, ni, Ti on existing axes
    draw_pressure_profiles,     # Pressure + perturbed ensemble
    draw_jphi_total,            # j_phi with uncertainty band
    draw_jphi_components,       # Bootstrap + inductive decomposition
)
```

### Filtering by scan value

When an HDF5 database contains multiple scan values, use `scan_val` and
`count` to select subsets:

```python
# All perturbations for scan_val=0 (+ baseline in black)
plot_pfile_bouquet(h5path="run.h5", scan_val=0, x_coord="psi_N")

# A single specific equilibrium
plot_pfile_bouquet(h5path="run.h5", scan_val=0, count=3, x_coord="psi_N")

# Discover available scan values
from bouquet import discover_scan_values
discover_scan_values("run.h5")  # e.g. ['0', '1', '2']
```

### Styling

In HDF5 mode, the **baseline** is plotted in black (lw=1.5) and **perturbed**
equilibria in orange (lw=1.5, alpha=0.7). In file-list mode (no baseline/perturbed
distinction), all entries use the `tab10` colormap uniformly.

---

## HDF5 Database

### Schema

```
run.h5
├── scan/
│   └── {scan_val}/
│       ├── _baseline/
│       │   ├── baseline.eqdsk       # raw geqdsk bytes
│       │   ├── baseline.pfile       # raw p-file bytes
│       │   ├── ne, te, ni, ti, …    # baseline 1-D profiles
│       │   └── sigma_ne, sigma_te … # uncertainty envelopes
│       ├── 0/
│       │   ├── header_sv_0.eqdsk    # perturbed geqdsk bytes
│       │   ├── header_sv_0.pfile    # perturbed p-file bytes
│       │   ├── psi_N, j_phi, ne, …  # perturbed 1-D profiles
│       │   └── li1, li3, Zeff       # scalar diagnostics
│       ├── 1/ …
│       └── N/ …
└── (flat layout also supported without scan/ prefix)
```

### Utilities

```python
from bouquet import (
    initialize_equilibrium_database,
    store_equilibrium,
    load_equilibrium,
    store_baseline_profiles,
    load_baseline_profiles,
    discover_scan_values,
    count_equilibria,
)

# Inspect a database
svs = discover_scan_values("run.h5")
for sv in svs:
    n = count_equilibria("run.h5", scan_value=sv)
    print(f"scan_val={sv}: {n} equilibria")

# Load a specific equilibrium
data = load_equilibrium("run.h5", scan_value="0", count=0)
```

---

## Examples

Example notebooks are in the `examples/` directory:

| Notebook | Description |
|----------|-------------|
| `basic_example.ipynb` | Fundamental workflow walkthrough |
| `g-file_p-file_example.ipynb` | GEQDSK and p-file I/O demonstration |
| `D3D-like/` | DIII-D-like tokamak equilibrium perturbation |
| `COCOS_Bt_Ip/cocos_and_save_example.ipynb` | COCOS conversion, Bt/Ip flip, save to disk/HDF5 |
| `COCOS_Bt_Ip/omfit_cocos_comparison.ipynb` | Field-by-field validation against OMFIT (requires `omfit_classes`) |
| `omfit-comparison/` | Verification against OMFIT reference values |

---

## Testing

```bash
pytest tests/
```

Tests cover:

- **`test_core.py`** — Uncertainty envelopes, H-mode profile generation
- **`test_geqdsk.py`** — GEQDSK parsing, COCOS conventions, flux-surface
  geometry, current density, safety factor
- **`test_pfile.py`** — P-file parsing, rotation decomposition, byte
  serialisation round-trip

Test data (sample g-files and p-files) is in `tests/data/`.

---

## Architecture and Assumptions

A detailed document covering all physics assumptions, numerical approximations,
coordinate conventions, known limitations, and planned future work is maintained
in [`architecture.md`](architecture.md). Key topics include:

- COCOS sign conventions
- Quasi-neutrality handling (baseline n_z1 preserved, not recomputed)
- Current decomposition (Sauter bootstrap model)
- Rotation profile computation (exact midplane, Savitzky–Golay smoothing for ω_HB)
- Numerical floors for axis/edge singularities
- Pressure matching and l_i iteration tolerances
- Unit conventions (bouquet SI vs. p-file units)

---

## API Reference

### Core Workflow

| Function | Description |
|----------|-------------|
| `generate_bouquet()` | Batch driver: draw N perturbations, solve GS, archive to HDF5 |
| `perturb_kinetic_equilibrium()` | Single perturbation: draw profiles, match pressure and l_i |
| `reconstruct_equilibrium()` | Reconstruct one GS equilibrium from geqdsk + profiles |
| `fit_inductive_profile()` | Spline fit of inductive current scaled to target l_i |

### Sampling

| Function / Class | Description |
|------------------|-------------|
| `GPRProfilePerturber` | Gaussian process profile perturbation engine |
| `generate_perturbed_GPR()` | One-call wrapper for perturbing a 1-D profile |
| `sigmoid_length_scale()` | Spatially-varying correlation length for Gibbs kernels |
| `verify_gpr_statistics()` | Monte Carlo validation of GPR sampling statistics |
| `calc_cylindrical_li_proxy()` | Fast cylindrical l_i proxy (no GS solve required) |

### Uncertainties

| Function | Description |
|----------|-------------|
| `new_uncertainty_profiles()` | Build 1-D uncertainty envelopes (power-law or flat+tail) |

### IO

| Class / Function | Description |
|------------------|-------------|
| `GEQDSKEquilibrium` | Full-featured GEQDSK reader with flux-surface analysis |
| `read_geqdsk()` | Parse a GEQDSK file (returns GEQDSKEquilibrium) |
| `write_geqdsk()` | Write a raw g-file dict to disk |
| `PFile` | P-file reader/writer with rotation computation |
| `read_pfile()` | Parse a p-file (returns PFile object) |

### Database

| Function | Description |
|----------|-------------|
| `initialize_equilibrium_database()` | Create/open HDF5 database |
| `store_equilibrium()` | Write one perturbed equilibrium to HDF5 |
| `load_equilibrium()` | Retrieve one equilibrium from HDF5 |
| `store_baseline_profiles()` | Store baseline profiles and uncertainties |
| `load_baseline_profiles()` | Load baseline profiles from HDF5 |
| `discover_scan_values()` | List all scan values in a database |
| `count_equilibria()` | Count equilibria for a given scan value |

### Plotting

| Function | Description |
|----------|-------------|
| `plot_bouquet()` | Notebook-friendly overview plot |
| `plot_geqdsk_bouquet()` | GEQDSK multi-panel (9 panels: pressure, current, q, geometry, …) |
| `plot_pfile_bouquet()` | P-file multi-panel (densities, temperatures, rotations) |
| `plot_coil_currents()` | Coil current bar chart |
| `plot_tokamaker_comparison()` | TokaMaker reconstruction comparison |

---

## License

See [LICENSE](LICENSE).
