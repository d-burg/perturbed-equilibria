# GEQDSKEquilibrium — Derived Quantities Reference

Complete reference for every property, method, and dict key accessible
from a `GEQDSKEquilibrium` object.

```python
from bouquet import GEQDSKEquilibrium, read_geqdsk
eq = GEQDSKEquilibrium("g123456.01000", cocos=1)
# or
eq = read_geqdsk("g123456.01000", cocos=1)
```

---

## Construction

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(filename, cocos=1, nlevels=None, resample="theta", extrapolate_edge=True)` | Load from a g-file on disk |
| `from_bytes` | `(raw_bytes, cocos=1, ...)` | Construct from in-memory bytes (e.g. from HDF5) |
| `from_raw` | `(raw_dict, cocos=1, ...)` | Construct from a raw g-file dict (no file I/O) |

**Parameters:**
- `cocos` — COCOS convention index (1–8 or 11–18; default 1 = standard EFIT)
- `nlevels` — number of psi_N levels for flux-surface analysis (default: NW from g-file)
- `resample` — `"theta"` (OMFIT angular resampling) or `"arc_length"` (raw contourpy)
- `extrapolate_edge` — if `True`, extrapolate p' and FF' where the g-file forces them to zero at the boundary

---

## Raw G-File Data (no flux-surface tracing required)

These are direct reads from the parsed g-file arrays.  They are
available immediately with no computation beyond parsing.

| Property | Type | Units | Description |
|----------|------|-------|-------------|
| `eq.R_grid` | 1-D array (NW,) | m | Uniform radial grid |
| `eq.Z_grid` | 1-D array (NH,) | m | Uniform vertical grid |
| `eq.psi_RZ` | 2-D array (NH, NW) | Wb/rad (COCOS 1–8) or Wb (COCOS 11–18) | Poloidal flux on the (R, Z) grid |
| `eq.psi_N_RZ` | 2-D array (NH, NW) | — | Normalised poloidal flux on the (R, Z) grid: `(psi - psi_axis) / (psi_boundary - psi_axis)`.  0 at axis, 1 at LCFS, >1 outside separatrix |
| `eq.psi_axis` | float | Wb/rad | Poloidal flux at the magnetic axis |
| `eq.psi_boundary` | float | Wb/rad | Poloidal flux at the LCFS |
| `eq.psi_N` | 1-D array (nlevels,) | — | Normalised poloidal flux grid, 0 (axis) to 1 (boundary) |
| `eq.fpol` | 1-D array (NW,) | T m | F = R Bt, on uniform psi_N |
| `eq.pres` | 1-D array (NW,) | Pa | Pressure profile on uniform psi_N |
| `eq.pprime` | 1-D array (NW,) | Pa / (Wb/rad) | dP/dpsi on uniform psi_N |
| `eq.ffprim` | 1-D array (NW,) | T m / (Wb/rad) | F dF/dpsi on uniform psi_N |
| `eq.qpsi` | 1-D array (NW,) | — | Safety factor from g-file (on uniform psi_N) |
| `eq.Ip` | float | A | Plasma current |
| `eq.R_mag` | float | m | R of magnetic axis |
| `eq.Z_mag` | float | m | Z of magnetic axis |
| `eq.R_center` | float | m | Reference geometric center R |
| `eq.B_center` | float | T | Vacuum toroidal field at R_center |
| `eq.boundary_R` | 1-D array | m | R coordinates of plasma boundary |
| `eq.boundary_Z` | 1-D array | m | Z coordinates of plasma boundary |
| `eq.limiter_R` | 1-D array | m | R coordinates of limiter |
| `eq.limiter_Z` | 1-D array | m | Z coordinates of limiter |
| `eq.rhovn` | 1-D array (NW,) | — | Normalised toroidal flux sqrt(Phi_tor/Phi_edge), recomputed from QPSI |
| `eq.cocos` | int | — | Current COCOS convention index |

---

## Flux-Surface-Averaged Quantities

These trigger lazy flux-surface tracing on first access.  All arrays
are on the `eq.psi_N` grid (length `nlevels`).

### Current Density

| Property | Type | Units | Description |
|----------|------|-------|-------------|
| `eq.j_tor_averaged` | 1-D array | A/m^2 | `<Jt/R> / <1/R>` — standard OMFIT/TRANSP convention |
| `eq.j_tor_averaged_direct` | 1-D array | A/m^2 | Literal `<Jt>` from p'`<R>` + FF'`<1/R>`/mu_0 (GS equation) |
| `eq.j_tor_averaged_numerical` | 1-D array | A/m^2 | Numerical contour-average of 2-D curl(B)/mu_0 field |
| `eq.j_tor_over_R` | 1-D array | A/m^3 | `<Jt/R>` from Grad-Shafranov equation |

### Safety Factor

| Property | Type | Units | Description |
|----------|------|-------|-------------|
| `eq.q_profile` | 1-D array | — | Safety factor from flux-surface averaging (more accurate than `qpsi` near edge) |

### Internal Inductance

`eq.li` returns a dict with multiple definitions:

| Key | Description |
|-----|-------------|
| `li(1)` | EFIT standard: `<Bp^2> / Bp_edge^2` (same as `li(1)_EFIT`) |
| `li(1)_EFIT` | Identical to `li(1)` |
| `li(1)_TLUCE` | Shape-corrected: li(1) * (1+kappa^2) / (2 kappa_a) |
| `li(2)` | Volume-weighted variant |
| `li(3)` | Alternative magnetic energy definition |
| `li_from_definition` | Direct `V <Bp^2> / (mu0^2 Ip^2 R0)` |

**Usage:** `eq.li['li(1)']`, `eq.li['li(3)']`, etc.

### Beta

`eq.betas` returns a dict:

| Key | Description |
|-----|-------------|
| `beta_t` | Toroidal beta: 2 mu0 `<p>` / Bt_vac^2 |
| `beta_p` | Poloidal beta: 2 mu0 `<p>` / `<Bp>^2` |
| `beta_n` | Normalised beta: beta_t / (Ip [MA] / a [m] Bt [T]) |

**Usage:** `eq.betas['beta_t']`, etc.

---

## Geometry (per flux surface)

`eq.geometry` returns a dict of 1-D arrays (length `nlevels`):

| Key | Units | Description |
|-----|-------|-------------|
| `R` | m | Geometric center R of each surface |
| `Z` | m | Geometric center Z of each surface |
| `a` | m | Minor radius (half-width) |
| `kappa` | — | Elongation (average of upper + lower) |
| `kapu` | — | Upper elongation |
| `kapl` | — | Lower elongation |
| `delta` | — | Triangularity (average of upper + lower) |
| `delu` | — | Upper triangularity |
| `dell` | — | Lower triangularity |
| `eps` | — | Inverse aspect ratio a/R |
| `perimeter` | m | Poloidal perimeter of each surface |
| `surfArea` | m^2 | Toroidal surface area (2pi R-weighted perimeter) |
| `vol` | m^3 | Enclosed volume |
| `cxArea` | m^2 | Poloidal cross-section area |

**Usage:** `eq.geometry['kappa'][-1]` for boundary elongation.

---

## Outboard Midplane

`eq.midplane` returns a dict of 1-D arrays (length `nlevels`):

| Key | Units | Description |
|-----|-------|-------------|
| `R` | m | R at outboard midplane (geo center + a) |
| `Z` | m | Z at outboard midplane (= Z_axis) |
| `Br` | T | Radial field at midplane |
| `Bz` | T | Vertical field at midplane |
| `Bp` | T | Poloidal field magnitude at midplane (signed) |
| `Bt` | T | Toroidal field at midplane: F(psi) / R |
| `Btot` | T | Total field magnitude at midplane |

**Sign convention for Bp:**
`-sigma_rhotp * sigma_RpZ * sign(Bz) * sqrt(Br^2 + Bz^2)`

**Usage:** `eq.midplane['Bt']` for the midplane toroidal field profile.

---

## Flux-Surface Averages (full dict)

`eq.averages` returns a dict of 1-D arrays (length `nlevels`):

| Key | Units | Description |
|-----|-------|-------------|
| `R` | m | `<R>` |
| `1/R` | m^-1 | `<1/R>` |
| `1/R**2` | m^-2 | `<1/R^2>` |
| `R**2` | m^2 | `<R^2>` |
| `Bp` | T | `<Bp>` (poloidal field) |
| `Bp**2` | T^2 | `<Bp^2>` |
| `Bt` | T | `<Bt>` (toroidal field) |
| `Bt**2` | T^2 | `<Bt^2>` |
| `Btot**2` | T^2 | `<B_total^2>` |
| `Jt` | A/m^2 | `<Jt>` numerical (from 2-D curl B / mu0) |
| `Jt/R` | A/m^3 | `<Jt/R>` analytical (from GS equation) |
| `Jt_GS` | A/m^2 | `<Jt>` analytical: p'`<R>` + FF'`<1/R>`/mu0 |
| `vp` | m^3/Wb | `V' = dV/dpsi` (flux-surface volume element) |
| `q` | — | Safety factor |
| `ip` | A | Enclosed toroidal current |
| `F` | T m | F = R Bt (interpolated to each surface) |
| `PPRIME` | Pa/(Wb/rad) | p' (interpolated to each surface) |
| `FFPRIM` | T m/(Wb/rad) | FF' (interpolated to each surface) |

---

## Contours

| Property | Type | Description |
|----------|------|-------------|
| `eq.contours` | list of (N,2) arrays | Flux-surface contour (R,Z) points for each psi_N level |

**Usage:** `eq.contours[i]` gives the (R, Z) contour for `eq.psi_N[i]`.

---

## Integration Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `volume_integral` | `(what)` | 1-D array (nlevels,) | Cumulative volume integral: integral(V' * what * dpsi) |
| `surface_integral` | `(what)` | 1-D array (nlevels,) | Cumulative cross-section integral: integral(V' / R * what * dpsi) / (2pi) |
| `flux_integral` | `(psi_N_val, profile)` | float | Total volume integral of `profile` evaluated at scalar `psi_N_val` |

**Example:** Total plasma current from j_phi:
```python
Ip_check = eq.flux_integral(1.0, eq.j_tor_averaged)
```

---

## COCOS Conversion and Sign-Flip Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `cocosify` | `(cocos_out, copy=False)` | self or new obj | Convert raw data to target COCOS convention |
| `flip_Bt_Ip` | `(copy=False)` | self or new obj | Negate Bt and Ip (plus psi fields and derivatives) |

**`copy=False`** (default): modifies in place, clears cache, returns `self`.
**`copy=True`**: returns a new `GEQDSKEquilibrium`; original is unchanged.

---

## Serialisation

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `save` | `(filename)` | — | Write to standard GEQDSK file |
| `to_bytes` | `()` | bytes | Serialise to ASCII bytes (for HDF5 `np.void()` storage) |

**Round-trip example:**
```python
raw = eq.to_bytes()
eq2 = GEQDSKEquilibrium.from_bytes(raw, cocos=eq.cocos)
assert np.allclose(eq2.psi_RZ, eq.psi_RZ)
```
