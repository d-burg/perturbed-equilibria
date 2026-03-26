# Bouquet: Architecture, Assumptions, and Caveats

This document catalogues the physics assumptions, numerical approximations,
data-format conventions, and known limitations of the bouquet perturbed-equilibrium
toolkit.  It is intended as a reference for developers and for users who need to
understand exactly what is — and is not — guaranteed by the code.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Coordinate Systems and Sign Conventions](#2-coordinate-systems-and-sign-conventions)
3. [Equilibrium Perturbation Methodology](#3-equilibrium-perturbation-methodology)
4. [Quasi-Neutrality and Impurity Handling](#4-quasi-neutrality-and-impurity-handling)
5. [Current Density Decomposition](#5-current-density-decomposition)
6. [Rotation Profile Computation](#6-rotation-profile-computation)
7. [Pressure and Beta Calculations](#7-pressure-and-beta-calculations)
8. [Flux-Surface Geometry and Averaging](#8-flux-surface-geometry-and-averaging)
9. [Internal Inductance](#9-internal-inductance)
10. [Numerical Floors and Clamps](#10-numerical-floors-and-clamps)
11. [Edge Extrapolation](#11-edge-extrapolation)
12. [Data Format Assumptions](#12-data-format-assumptions)
13. [HDF5 Storage Schema](#13-hdf5-storage-schema)
14. [Unit Conventions](#14-unit-conventions)
15. [Known Limitations and Future Work](#15-known-limitations-and-future-work)

---

## 1. Overview

Bouquet generates families ("bouquets") of perturbed tokamak equilibria by:

1. Starting from a baseline kinetic equilibrium (g-file + p-file).
2. Drawing correlated perturbations of ne, Te, ni, Ti from Gaussian
   process regression (GPR) posteriors, respecting user-supplied
   uncertainty envelopes.
3. Matching the perturbed pressure profile to the baseline volume-averaged
   pressure (within a configurable tolerance).
4. Decomposing the perturbed current density into bootstrap and inductive
   components and iterating the inductive profile to match the baseline
   internal inductance li.
5. Solving the Grad-Shafranov equation via TokaMaker for each accepted
   perturbation.
6. Archiving all results (geqdsk bytes, p-file bytes, scalar diagnostics)
   to a single HDF5 database.

Each of these steps carries assumptions documented below.

---

## 2. Coordinate Systems and Sign Conventions

### 2.1 COCOS Convention

The geqdsk reader implements the COCOS (COordinate COnventionS) framework
[O. Sauter and S.Yu. Medvedev, Computer Physics Communications **184**
(2013) 293].  Eight sign/exponent parameters fully determine the
orientation:

| Symbol | Meaning | COCOS 1 (EFIT default) |
|--------|---------|----------------------|
| `sigma_Bp` | Sign of poloidal field relative to psi gradient | +1 |
| `sigma_RpZ` | Handedness of (R, phi, Z) | +1 |
| `sigma_rhotp` | Sign of (theta_pol × phi_tor) | +1 |
| `exp_Bp` | 2pi exponent (0 for COCOS 1-8, 1 for 11-18) | 0 |

**Assumption:** The default is COCOS 1 (standard EFIT).  An incorrect
choice propagates sign errors through all flux-surface-averaged
quantities, safety factor, and current density.

**User control:** The `cocos` parameter on `GEQDSKEquilibrium`.

### 2.1b COCOS Conversion (`cocosify`)

`GEQDSKEquilibrium.cocosify(cocos_out)` converts the raw g-file data
from the current COCOS to any target COCOS, following the transformation
rules in Sauter & Medvedev (2013), Eq. 14/23.

The effective transformation parameters between `cocos_in` and
`cocos_out` are:

```
sigma_Bp_eff    = sigma_Bp_out    * sigma_Bp_in
sigma_RpZ_eff   = sigma_RpZ_out   * sigma_RpZ_in
sigma_rhotp_eff = sigma_rhotp_out * sigma_rhotp_in
exp_Bp_eff      = exp_Bp_out      - exp_Bp_in
```

The multiplicative factors applied to each g-file field:

| Field | Factor |
|-------|--------|
| PSIRZ, SIMAG, SIBRY | `sigma_RpZ_eff * sigma_Bp_eff * (2π)^exp_Bp_eff` |
| PPRIME, FFPRIM | `sigma_RpZ_eff * sigma_Bp_eff / (2π)^exp_Bp_eff` |
| FPOL, BCENTR | `sigma_RpZ_eff` |
| CURRENT | `sigma_RpZ_eff` |
| QPSI | `sigma_rhotp_eff` |

These factors have been verified field-by-field against OMFIT's
`OMFITgeqdsk.cocosify()` for COCOS 1→7, 1→11, 7→1, and 7→11
(see `examples/COCOS_Bt_Ip/omfit_cocos_comparison.ipynb`).

**Caveat:** `cocosify` transforms the raw numerical arrays but does
not re-derive any cached quantities.  The cache is cleared on every
call, so subsequent property accesses (q_profile, geometry, etc.)
will recompute using the new COCOS parameters.

### 2.1c Bt/Ip Flip (`flip_Bt_Ip`)

`GEQDSKEquilibrium.flip_Bt_Ip()` reverses the direction of both
the toroidal field and the plasma current.  This negates:

- `BCENTR`, `FPOL` (toroidal field direction)
- `CURRENT` (plasma current direction)
- `SIMAG`, `SIBRY`, `PSIRZ` (poloidal flux)
- `PPRIME`, `FFPRIM` (per-psi derivatives)

The safety factor `QPSI` is **unchanged** because q ∝ Bt/Ip and
both signs cancel.

**Use case:** Converting between experiments with opposite field
and current directions (e.g. forward vs. reversed Bt operation on
DIII-D).

### 2.1d Serialisation (`save`, `to_bytes`)

Modified equilibria can be written back to standard GEQDSK format
via `save(filename)` or serialised to bytes via `to_bytes()` for
HDF5 storage.  The writer uses the same fixed-format (5 values per
line, 16 chars each) as the parser expects, ensuring exact
round-trip fidelity.

### 2.2 Normalised Poloidal Flux (psi_N)

Defined as:

```
psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)
```

Ranges from 0 (magnetic axis) to 1 (last closed flux surface).
All 1-D profile grids in bouquet use this normalisation.

### 2.3 Toroidal Flux Coordinate (rho)

`rhovn` (normalised square-root toroidal flux) is computed by
integrating the safety factor:

```
rho(psi_N) = sqrt( integral_0^psi_N q dpsi_N  /  integral_0^1 q dpsi_N )
```

The code explicitly recomputes this from `QPSI` rather than trusting
the RHOVN array stored in many g-files, because some equilibrium solvers
write a placeholder `RHOVN = sqrt(psi_N)` which is only correct when q
is spatially constant.

**Caveat:** A ~22% discrepancy against OMFIT's rhovn has been observed
in testing and has not yet been fully resolved.  See
[Section 15, Future Work](#15-known-limitations-and-future-work).

### 2.4 Midplane Convention

The outboard midplane is defined following OMFIT:

- R_mid = geometric centre + minor radius (outboard intersection)
- Z_mid = Z of the magnetic axis
- Br, Bz interpolated from the 2-D psi grid via `RectBivariateSpline`
- Bp = signed `sqrt(Br^2 + Bz^2)`, sign from
  `-sigma_rhotp * sigma_RpZ * sign(Bz)`
- Bt = F(psi) / R_mid

This is an **exact** midplane evaluation (no flux-surface averaging).
It is used in the rotation decomposition (Er, Hahm-Burrell rate) and
has been verified against OMFIT to <0.01% in R, <0.01% in Bt, and
<0.3% in Bp (excluding the degenerate magnetic axis).

---

## 3. Equilibrium Perturbation Methodology

### 3.1 Profile Sampling

Perturbations of ne, Te, ni, Ti are drawn from a zero-mean Gaussian
process whose covariance kernel is built from user-supplied uncertainty
envelopes sigma(psi_N).  Supported kernels include RBF (squared
exponential) and Matern, with either scalar or spatially-varying
(Gibbs non-stationary) length scales.

**Covariance matrix conditioning:** The eigendecomposition uses
`np.linalg.eigh` for symmetric matrices.  Small negative eigenvalues
from floating-point round-off are clamped to zero.

**Monotonicity enforcement:** For profiles that are physically
monotonically decreasing (ne, Te, ni, Ti in a standard tokamak), the
sampler rejects draws that violate monotonicity up to a maximum of
10 000 attempts before raising an error.

**Assumption:** Profile uncertainties are Gaussian-distributed and
fully characterised by the covariance kernel.  Non-Gaussian tails
(e.g. from ELMs or sawtooth crashes) are not captured.

### 3.2 Pressure Matching

After drawing perturbed profiles, the total kinetic pressure is computed
as:

```
P = e_charge * (ne * Te + ni * Ti)    [Pa, with ne in m^-3 and Te in eV]
```

The perturbed pressure is rescaled so that its volume average matches
the baseline to within `p_thresh` (default 0.5%).

**Assumption:** The rescaling adjusts all four profiles uniformly.
This preserves the shape of the perturbation but slightly alters its
magnitude.

**Iteration limit:** 100 000 attempts (effectively unlimited).

### 3.3 li Matching

The internal inductance li(3) of each perturbed current profile is
matched to the baseline value through a two-stage process:

1. **Cylindrical proxy filter:** A fast 1-D proxy for li is computed
   without solving Grad-Shafranov, using pre-computed geometry (volume,
   area, Bp integrals from the baseline).  Draws whose proxy li falls
   outside `l_i_proxy_threshold` (default 5%) of the target are
   rejected cheaply.

2. **Secant iteration on the inductive profile:** For draws that pass
   the proxy filter, a secant method scales the inductive current-density
   amplitude to match the true (Grad-Shafranov) li to within
   `l_i_tolerance` (default 0.05, i.e. 5%).

**Proxy correction:** After each GS solve, the proxy target is
adaptively updated based on the observed proxy-vs-reality offset,
blended as 70% new correction / 30% old target.

**Assumption:** The cylindrical proxy is a monotonic function of the
scaling amplitude.  This is generally true for broad inductive profiles
but can break down for highly hollow or peaked current configurations.

**Iteration limit:** 20 secant steps.  If not converged, the
equilibrium is rejected.

**Step clamping:** Secant steps are clamped to ±15% of the current
amplitude to prevent runaway oscillation.

### 3.4 Safety Factor Constraint

By default (`constrain_sawteeth=True`), any equilibrium with q(0) < 1
is rejected.  This prevents the generation of sawtoothing equilibria
that would require additional physics (reconnection, island evolution)
to model self-consistently.

**Caveat:** The constraint is checked *after* the GS solve, so
rejected equilibria still cost a TokaMaker call.

### 3.5 What Is Perturbed vs. Held Fixed

| Quantity | Perturbed? | Notes |
|----------|-----------|-------|
| ne, Te, ni, Ti | Yes | Drawn from GPR posterior |
| Total pressure (ptot) | Yes | Recomputed from perturbed kinetics |
| Bootstrap current (j_BS) | Yes (optional) | Recomputed from Sauter model if `recalculate_j_BS=True` |
| Inductive current (j_ind) | Yes | Scaled to match li |
| Coil currents | Yes | Adjusted by TokaMaker to match Ip |
| nz1 (impurity density) | **No** | Kept from baseline; see [Section 4](#4-quasi-neutrality-and-impurity-handling) |
| nb (beam density) | **No** | Preserved from baseline p-file |
| pb (beam pressure) | **No** | Preserved from baseline p-file |
| Toroidal rotation (omeg) | **No** | Preserved from baseline p-file |
| Poloidal rotation (omegp) | **No** | Preserved from baseline p-file |
| kpol | **No** | Preserved from baseline p-file |
| E×B rotation (w_ExB) | **No** | Zero placeholder stored; see [Section 15](#15-known-limitations-and-future-work) |
| Diamagnetic rotations | Yes | Recomputed from perturbed ne, Te, ni, Ti + baseline nz1 |
| Er, Hahm-Burrell rate | Yes | Recomputed using exact midplane B-fields |

---

## 4. Quasi-Neutrality and Impurity Handling

### 4.1 The Problem

Quasi-neutrality requires:

```
ne = ni * Z_main + nz1 * Z_imp + nb * Z_beam
```

When bouquet perturbs ne and ni independently, this constraint can be
violated.  Recomputing `nz1 = (ne - ni - nb) / Z_imp` from
independently perturbed profiles can yield **negative impurity density**
— an unphysical result.

### 4.2 Bouquet's Approach

**`generate_bouquet()` does not recompute nz1.**  The baseline impurity
density is preserved in the perturbed p-file.  This means:

- The perturbed p-file does **not** satisfy exact quasi-neutrality.
- The impurity contribution to pressure and diamagnetic rotation uses
  the original (self-consistent) nz1 profile.
- The total pressure `ptot` is recomputed as
  `ne*Te + (ni + nz1_baseline)*Ti + pb`, which is self-consistent with
  the stored profiles.

**Rationale:** Bouquet perturbs the *thermal* species (ne, ni, Te, Ti)
within their measurement uncertainties.  The impurity content is not
being perturbed — it is a separate measurement (e.g. from charge-exchange
spectroscopy) with its own uncertainty.  Forcing quasi-neutrality by
adjusting nz1 would couple the impurity density to the thermal profile
uncertainties in a physically unmotivated way.

### 4.3 `compute_quasineutrality()` Standalone Behaviour

The `PFile.compute_quasineutrality()` method is still available for
users who want to enforce charge balance explicitly.  If the result
contains negative values, a `UserWarning` is emitted with the number
of affected grid points and the minimum value.  The negative values
are **not** clamped — the caller decides how to handle them.

### 4.4 Downstream Consequences

Because baseline nz1 is preserved:

- `omgpp` (impurity diamagnetic) reflects the original impurity
  pressure gradient, not a quasi-neutrality-derived one.
- Zeff computed from the stored profiles will differ slightly from
  `(ni + nz1*Z^2 + nb) / ne` because quasi-neutrality is not exact.
- These differences are small for typical bouquet perturbation
  magnitudes (5-10% profile variations) and are within the measurement
  uncertainty of the impurity density itself.

---

## 5. Current Density Decomposition

### 5.1 Bootstrap Current

The bootstrap current density `j_BS` is computed using the Sauter
model [O. Sauter, C. Angioni, and Y.R. Lin-Liu, Physics of Plasmas
**6** (1999) 2834], which is the standard analytic model used by most
transport codes.

**Assumption:** The Sauter model is a fit to numerical neoclassical
calculations and is accurate to ~10% for standard tokamak conditions.
It can be less accurate for:

- Very low aspect ratio (spherical tokamaks)
- Strong rotation
- Non-Maxwellian distributions
- Very steep edge pedestals

### 5.2 Inductive Current

The inductive current `j_ind` is defined as the residual:

```
j_ind = j_total - j_BS
```

It is fitted with a `scipy.interpolate.UnivariateSpline` using a
smoothing factor of `len(psi) * var(residual) * 0.01`.

**Assumption:** The inductive profile shape is smooth and can be
represented by a low-order spline.  No cross-validation is performed
on the smoothing parameter.

### 5.3 Bootstrap Shelf Option

If `shelf_psi_N > 0`, the bootstrap current is flattened for
psi_N < shelf_psi_N (i.e. in the core), using the value at the shelf
location.  This separates core and edge bootstrap contributions and
can be useful when the Sauter model produces unphysical core structure.

**Caveat:** The shelf introduces a discontinuity in the j_BS derivative
at psi_N = shelf_psi_N.

---

## 6. Rotation Profile Computation

### 6.1 Diamagnetic Rotation

The diamagnetic rotation frequency for species *s* is:

```
omega_dia,s = (1 / n_s Z_s e) * d(n_s T_s) / dpsi
```

In p-file units (n in 10^20/m^3, T in keV, psi in Wb), this gives
kRad/s directly.

**Sign convention (enforced by code):**
- Impurity and main-ion diamagnetic: negative (counter-current)
- Electron diamagnetic: positive (co-current)

The code uses `np.abs()` to compute the magnitude and then applies
the sign by convention.

**Assumption:** All ion species share the same temperature unless
an explicit impurity temperature `TI` is provided.

### 6.2 ExB Decomposition

```
omega_ExB  = omega_VxB + omega_dia_impurity
omega_VxB(main)  = omega_ExB - omega_dia_main
omega_VxB(elec)  = omega_ExB - omega_dia_electron
```

If `omega_VxB` (the impurity VxB term, from measured toroidal
rotation) is not present in the p-file, it defaults to zero.

### 6.3 Radial Electric Field and Hahm-Burrell Rate

```
Er    = omega_ExB * R_mid * Bp_mid        [kV/m]
omghb = (R_mid * Bp_mid)^2 / Bt_mid * d(omega_ExB)/dpsi
```

These use exact outboard-midplane values of R, Bp, Bt (not
flux-surface averages), following the OMFIT convention.

**Hahm-Burrell derivative method:** The `d(omega_ExB)/dpsi` term is
computed using a Savitzky-Golay filter with `deriv=1` (window ≈ 3%
of the grid, minimum 7 points, polynomial order 3).  This is
mathematically equivalent to fitting a local cubic polynomial and
analytically differentiating it, which is optimal for noisy data.
A naive `np.gradient()` on `omega_ExB` amplifies grid-scale noise
because `omghb` is effectively a **second derivative** of the
kinetic profiles (`omega_ExB ~ d(nT)/dpsi`, `omghb ~ d^2(nT)/dpsi^2`).

**Baseline consistency:** When storing the baseline p-file to HDF5,
`generate_bouquet()` recomputes the baseline rotation profiles
(diamagnetic, ExB decomposition, Er, omghb) using the same midplane
method as the perturbed p-files.  This ensures the baseline and
perturbed curves in `plot_pfile_bouquet()` are directly comparable.
Without this recomputation, the baseline `omghb` (from the original
p-file creation tool) can differ from the perturbed `omghb` by an
order of magnitude purely due to methodological differences.

### 6.4 Numerical Safeguards

Near the magnetic axis, `dpsi = gradient(psi)` approaches zero and
produces division-by-zero spikes.  At the plasma edge, densities
approach zero with the same effect.  Flooring is applied:

- **dpsi floor:** `max(1e-4 * max|dpsi|, 1e-30)`
- **Density floor:** `1e-4 * max|ne|` (applied to ne, ni, nz1 in
  the denominator only)
- **Bt floor:** `sign(Bt) * 1e-6` where |Bt| < 1e-6 T

These floors are small enough to preserve the physics everywhere
except the degenerate axis/edge points.  Any remaining NaN/inf values
are replaced with zero via `np.nan_to_num`.

---

## 7. Pressure and Beta Calculations

### 7.1 Total Pressure (P-file)

```
ptot = 16.022 * (ne * Te + (ni + nz1) * Ti) + pb     [kPa]
```

where the constant 16.022 kPa/(10^20 m^-3 keV) = e_charge * 1e20.

**Assumption:** Main ions are singly charged (Z_main = 1).  The
formula is correct for arbitrary Z_main but the p-file convention
stores `ni` as the main-ion density (not `ni * Z_main`).

### 7.2 Total Pressure (Bouquet SI)

```
P = EC * (ne * Te + ni * Ti)    [Pa]
```

where `EC = 1.6022e-19` J/eV, ne in m^-3, Te in eV.

**Caveat:** The bouquet SI pressure does not include the impurity or
beam contributions.  The p-file pressure does.  When comparing the
two, the difference is `EC * nz1 * Ti + pb`.

### 7.3 Beta

```
beta_t = 2 mu0 <p> / Bt_vac^2
beta_p = 2 mu0 <p> / <Bp>^2
beta_N = beta_t / (Ip / a Bt)    [in %·m·T/MA]
```

where `Bt_vac = B0 * R0 / R_boundary`.

**Caveat:** Beta quantities are computed in the geqdsk reader but
are not yet fully validated.  See [Section 15](#15-known-limitations-and-future-work).

---

## 8. Flux-Surface Geometry and Averaging

### 8.1 Contour Tracing

Flux surfaces are traced using `contourpy` on the 2-D psi(R,Z) grid
and resampled to 257 arc-length-uniform points using periodic (interior)
or non-periodic (near-separatrix) cubic splines.

**Threshold for periodic vs. non-periodic:** psi_N = 0.99.

**Rationale:** The X-point introduces a cusp in the separatrix contour.
Periodic splines would smooth it out; non-periodic splines preserve
the cusp but may oscillate slightly.

### 8.2 X-Point Detection

The X-point is located as the sharpest bend in the separatrix contour
(minimum cosine of the angle between adjacent tangent vectors).

**Assumption:** Single lower X-point.  Double-null or upper-single-null
configurations may not be handled correctly.

### 8.3 Flux-Surface Averaging

Averages use flux-expansion weighting:

```
<Q> = (1/V') * oint Q * dl / |Bp|
V'  = oint dl / |Bp|
```

where `dl` is the arc-length element along the contour.

**Bp flooring:** `max(1e-6 * max|Bp|, 1e-14)` prevents X-point-adjacent
spikes from dominating the average.

**Caveat:** This smooths the true geometric singularity at the X-point.
Flux-surface averages on surfaces very close to the separatrix
(psi_N > 0.995) should be treated with caution.

### 8.4 Separatrix Treatment

At psi_N = 1, the code uses the stored RBBBS/ZBBBS boundary from the
g-file rather than tracing a contour.  If the boundary has fewer than
4 points, a degenerate contour at the magnetic axis is substituted
with no error raised.

### 8.5 Jt Averaging

Two definitions are computed:

- **`j_tor_averaged`** (OMFIT standard): `<Jt/R> / <1/R>`.  This is
  the current density that, when multiplied by the cross-sectional
  area, gives the total toroidal current.

- **`j_tor_averaged_direct`**: literal `<Jt>`.

These differ by a Jensen inequality correction proportional to
`p' * [<R> - 1/<1/R>]`.  The difference is small for circular
cross-sections but can be significant for highly shaped plasmas.

---

## 9. Internal Inductance

### 9.1 Definitions

Multiple definitions of li are computed:

| Name | Formula | Notes |
|------|---------|-------|
| li(1)_EFIT | `<Bp^2> / Bp_edge^2` | Standard EFIT definition |
| li(1)_TLUCE | li(1)_EFIT × shape correction | Rarely used |
| li(2) | Volume-weighted variant | |
| li(3) | Alternative magnetic energy | Used by bouquet for matching |

The shape correction for li(1)_TLUCE is `(1 + kappa^2) / (2 * kappa_a)`
where `kappa_a = V / (2 pi R0 pi a^2)`.

**Default:** `li(1)` returns the EFIT definition.

**Risk:** Different transport codes and databases use different li
definitions.  Users must verify which definition their workflow expects.

---

## 10. Numerical Floors and Clamps

| Location | Floor | Purpose |
|----------|-------|---------|
| Diamagnetic dpsi | `1e-4 * max\|dpsi\|` | Prevent axis singularity |
| Diamagnetic density | `1e-4 * max\|ne\|` | Prevent edge/negative-density singularity |
| Rotation Bt | `1e-6` T | Prevent Er/omghb blowup |
| Flux-surface Bp | `1e-6 * max\|Bp\|` | Prevent X-point weight blowup |
| Secant step | ±15% of current value | Prevent li iteration runaway |
| Ip scaling | 50% of Ip_desired | Safety floor for TokaMaker scaling |
| GPR eigenvalues | max(lambda, 0) | Numerical PSD enforcement |

These floors are empirically chosen to be small enough that they do not
affect the physics on resolved grid points.  They are not user-configurable.

---

## 11. Edge Extrapolation

Many EFIT and TokaMaker equilibria force p'(1) = FF'(1) = 0 as a
free-boundary condition.  This creates a spurious dip in the current
density at the separatrix.

When `extrapolate_edge=True` (the default in the geqdsk reader), the
last 3-4 non-zero points of p' and FF' are quadratically extrapolated
to the boundary.  A sign-protection clip prevents the extrapolation
from reversing sign.

**Caveat:** This creates artificial structure at the plasma edge and
may distort the near-edge j_phi and pressure gradient profiles.
Users working on pedestal physics should consider disabling this
feature.

---

## 12. Data Format Assumptions

### 12.1 GEQDSK (G-file)

- Standard fixed-format: 5 values per line, 16 characters per value.
- The parser handles SOLPS variants (extra whitespace) and FIESTA
  row-by-row PSIRZ output.
- **Assumption:** The grid is uniform and rectangular (guaranteed by
  the GEQDSK specification).
- Br and Bz are computed from 2-D `np.gradient()` (second-order
  finite differences), which can amplify numerical noise at grid
  edges.

### 12.2 P-file (Osborne Format)

- Header lines match the regex:
  `(\d+)\s+(\S+)\s+(\S+)\(([^)]*)\)\s+(.*?)\s*$`
- Each profile block: count, x-name, y-name(units), description.
- The `"N Z A of ION SPECIES"` block is parsed separately and encodes
  species ordering: [impurity, main ion, beam ion].
- **Assumption:** All profiles use `psinorm` as the radial coordinate.
  Non-standard p-files using other normalisations will silently produce
  incorrect results.

### 12.3 HDF5 Database

See [Section 13](#13-hdf5-storage-schema).

---

## 13. HDF5 Storage Schema

Each bouquet run produces a single HDF5 file with the structure:

```
header.h5
+-- scan/
|   +-- {scan_val_key}/
|   |   +-- _baseline/
|   |   |   +-- baseline.eqdsk          # raw geqdsk bytes
|   |   |   +-- baseline.pfile          # raw p-file bytes (optional)
|   |   |   +-- ne, te, ni, ti [...]    # baseline 1-D profiles
|   |   +-- 0/
|   |   |   +-- {header}_{sv}_{0}.eqdsk # raw geqdsk bytes
|   |   |   +-- {header}_{sv}_{0}.pfile # raw p-file bytes (optional)
|   |   |   +-- psi_N, j_phi, ne, te, ni, ti [...]
|   |   |   +-- li1, li3, Zeff, coil_currents [A], coil_names
|   |   +-- 1/ ...
|   +-- {another_scan_val}/ ...
```

**Key design choices:**

- Raw geqdsk and p-file bytes are stored as opaque binary datasets.
  This enables exact round-tripping: `GEQDSKEquilibrium.from_bytes()`
  and `PFile.from_bytes()` reconstruct the original objects.
- Scalar diagnostics (li1, li3) are stored as attributes or scalar
  datasets.
- Coil currents are stored as a 1-D array with coil names in a
  companion string attribute.
- The `_baseline` group stores the unperturbed equilibrium for
  comparison.

---

## 14. Unit Conventions

### 14.1 Bouquet Internal (SI)

| Quantity | Unit |
|----------|------|
| ne, ni | m^-3 |
| Te, Ti | eV |
| Pressure | Pa |
| Current density | A/m^2 |
| Psi | Wb (Weber) |
| B-field | T (Tesla) |
| R, Z | m |

### 14.2 P-file

| Quantity | Unit |
|----------|------|
| ne, ni, nz1, nb | 10^20 m^-3 |
| Te, Ti | keV |
| Pressure (ptot, pb) | kPa |
| Rotation (omeg, omgeb, ...) | kRad/s |
| Er | kV/m |

### 14.3 Conversion Constants

| Conversion | Value | Used in |
|------------|-------|---------|
| EC (eV to J) | 1.6022e-19 | Bouquet pressure = EC * n * T |
| _NT_TO_KPA | 16.02176634 | P-file ptot = _NT_TO_KPA * (ne*Te + ...) |
| m^-3 to 10^20/m^3 | 1e-20 | P-file density |
| eV to keV | 1e-3 | P-file temperature |
| Pa to kPa | 1e-3 | P-file pressure |

---

## 15. Known Limitations and Future Work

### 15.1 E×B Rotation (w_ExB) Not Computed from Equilibrium

The `w_ExB` field stored in the HDF5 is a **zero placeholder**.
A self-consistent E×B rotation would require solving the radial
force balance for each perturbed equilibrium, which is not yet
implemented.  The diamagnetic decomposition (omgpp, ommpp, omepp)
and derived quantities (omgeb, Er, omghb) in the perturbed p-file
*are* computed, but they use the baseline VxB rotation (`omgvb`)
if present, or zero if absent.

### 15.2 rhovn Discrepancy

A ~22% discrepancy between bouquet's `rhovn` and OMFIT's `rhovn`
has been observed.  The bouquet computation integrates q over psi_N;
OMFIT may use a different integration scheme or include additional
corrections.  This affects any analysis using rho as the radial
coordinate.

### 15.3 Beta Quantities

`beta_t`, `beta_p`, `beta_N` are computed in the geqdsk reader
but have not been validated against EFIT or other reference codes.

### 15.4 Double-Null and Upper-Single-Null Equilibria

The X-point detection assumes a single lower X-point.  Double-null
or upper-single-null configurations may produce incorrect contour
cropping, flux-surface averaging, and geometric quantities (kappa,
delta, squareness).

### 15.5 Near-Axis Shaping

Elongation (kappa) and triangularity (delta) are computed from the
half-widths and extrema of each flux surface.  Near the magnetic
axis, these quantities become noisy because the contours are nearly
circular and the extrema are poorly defined.  No smoothing is
currently applied.

### 15.6 Uncertainty Envelope Model

The default uncertainty envelope is a power-law profile:

```
sigma(psi_N) = sigma_0 * (1 - psi_N)^n
```

with a flat core, smooth tanh transition, and a minimum floor.  This
is a simple parametric model.  Users with experimentally-derived
profile uncertainties should supply them directly rather than relying
on this model.

### 15.7 Non-Gaussian Profile Uncertainties

The GPR sampling assumes Gaussian-distributed profile uncertainties.
Non-Gaussian features (e.g. pedestal bifurcation, ELM-induced
transients, sawtooth mixing) are not captured.

### 15.8 Single Ion Temperature

The diamagnetic rotation calculation assumes all ion species
(main + impurity) share the same temperature `Ti` unless a separate
impurity temperature `TI` is explicitly provided.  This is a common
assumption in tokamak transport modelling but breaks down when strong
ion-impurity temperature decoupling exists (e.g. during NBI heating).

### 15.9 Beam and Impurity Perturbation

Currently, bouquet does not perturb:

- Fast-ion density (nb) or pressure (pb)
- Impurity density (nz1)
- Toroidal/poloidal rotation profiles (omeg, omegp, kpol)

Adding uncertainty-driven perturbation of these quantities is a
natural extension but would require additional uncertainty
specifications and potentially different sampling strategies (e.g.
rotation profiles are not necessarily monotonic).

### 15.10 P-file Regeneration for Example Files

The example p-files shipped with the repository have rotation profiles
computed from a specific baseline equilibrium.  If the baseline
equilibrium is changed (e.g. different bootstrap current amplitude),
the rotation profiles in the shipped p-file will be stale.
Regeneration of rotation profiles for all example files is planned.

### 15.11 Flux-Surface Average vs. Midplane for Diagnostics

Some diagnostics (e.g. Thomson scattering, charge-exchange
recombination) measure profiles at specific poloidal locations
(typically the outboard midplane), not flux-surface averages.
Bouquet perturbs flux-surface-averaged profiles.  For strongly
up-down asymmetric plasmas, this distinction matters.

### 15.12 Bootstrap Current Model Alternatives

Only the Sauter model is currently implemented.  Alternative models
(e.g. Sauter with Redl corrections, NEO, or direct drift-kinetic
solvers) may give significantly different bootstrap current profiles,
especially in the pedestal region.  Adding pluggable bootstrap
current models is planned.
