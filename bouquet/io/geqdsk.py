"""
Standalone GEQDSK reader with flux-surface analysis.

Reads standard GEQDSK (g-file) equilibrium files and computes
flux-surface-averaged quantities without OMFIT or TokaMaker dependencies.

Dependencies: numpy, scipy, contourpy (ships with matplotlib).
"""

import io
import tempfile
import warnings

import contourpy
import numpy as np
from scipy import constants, integrate, interpolate


# ---------------------------------------------------------------------------
# Section 1: COCOS parameter table
# ---------------------------------------------------------------------------

def _cocos_params(cocos_index):
    """Return COCOS sign/exponent parameters for a given COCOS index.

    Parameters
    ----------
    cocos_index : int
        COCOS convention index (1-8 or 11-18).

    Returns
    -------
    dict with keys: sigma_Bp, sigma_RpZ, sigma_rhotp, exp_Bp
    """
    if cocos_index < 1 or cocos_index > 18 or cocos_index in (9, 10):
        raise ValueError(f"Invalid COCOS index: {cocos_index}")

    exp_Bp = 0 if cocos_index < 10 else 1

    # Base index in 1-8 range
    base = cocos_index if cocos_index < 10 else cocos_index - 10

    # sigma_Bp: sign of poloidal flux (psi increasing outward = +1)
    sigma_Bp = +1 if base in (1, 2, 5, 6) else -1

    # sigma_RpZ: right-hand (R,phi,Z) vs (R,Z,phi) orientation
    sigma_RpZ = +1 if base in (1, 2, 7, 8) else -1

    # sigma_rhotp: sign of theta_pol*phi_tor product
    sigma_rhotp = +1 if base in (1, 3, 5, 7) else -1

    return {
        "sigma_Bp": sigma_Bp,
        "sigma_RpZ": sigma_RpZ,
        "sigma_rhotp": sigma_rhotp,
        "exp_Bp": exp_Bp,
    }


# ---------------------------------------------------------------------------
# Section 2: GEQDSK parser
# ---------------------------------------------------------------------------

def _read_geqdsk(filename):
    """Parse a GEQDSK (g-file) into a plain dict.

    Parameters
    ----------
    filename : str or path-like
        Path to the g-file.

    Returns
    -------
    dict
        Keys match the standard GEQDSK field names (NW, NH, RDIM, ...,
        FPOL, PRES, FFPRIM, PPRIME, PSIRZ, QPSI, RBBBS, ZBBBS, etc.).
    """

    def splitter(text, step=16):
        return [text[step * k : step * (k + 1)] for k in range(len(text) // step)]

    def merge(lines):
        if not lines:
            return ""
        # Handle SOLPS-style g-files that add spaces between numbers
        if len(lines[0]) > 80:
            return "".join(lines).replace(" ", "")
        return "".join(lines)

    with open(filename, "r") as f:
        EQDSK = f.read().splitlines()

    g = {}

    # --- Header line: CASE string, IDUM, NW, NH ---
    g["CASE"] = np.array(splitter(EQDSK[0][:48], 8))
    try:
        tmp = [x for x in EQDSK[0][48:].split() if x]
        _idum, g["NW"], g["NH"] = (int(x) for x in tmp[:3])
    except ValueError:
        _idum = int(EQDSK[0][48:52])
        g["NW"] = int(EQDSK[0][52:56])
        g["NH"] = int(EQDSK[0][56:60])
    offset = 1

    # --- 20 scalar values (4 lines x 5 values) ---
    scalars = list(map(float, splitter(merge(EQDSK[offset : offset + 4]))))
    (
        g["RDIM"], g["ZDIM"], g["RCENTR"], g["RLEFT"], g["ZMID"],
        g["RMAXIS"], g["ZMAXIS"], g["SIMAG"], g["SIBRY"], g["BCENTR"],
        g["CURRENT"], g["SIMAG"], _, g["RMAXIS"], _,
        g["ZMAXIS"], _, g["SIBRY"], _, _,
    ) = scalars
    offset += 4

    NW, NH = int(g["NW"]), int(g["NH"])
    nlNW = int(np.ceil(NW / 5.0))

    # --- 1-D profiles (each NW long) ---
    for name in ("FPOL", "PRES", "FFPRIM", "PPRIME"):
        g[name] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
        offset += nlNW

    # --- 2-D poloidal flux: PSIRZ (NH x NW) ---
    try:
        nlNWNH = int(np.ceil(NW * NH / 5.0))
        flat = np.fromiter(splitter(merge(EQDSK[offset : offset + nlNWNH])),
                           dtype=np.float64)[: NH * NW]
        g["PSIRZ"] = flat.reshape((NH, NW))
        offset += nlNWNH
    except ValueError:
        # Some codes (e.g. FIESTA) write row-by-row
        nlNWNH = NH * nlNW
        flat = np.fromiter(splitter(merge(EQDSK[offset : offset + nlNWNH])),
                           dtype=np.float64)[: NH * NW]
        g["PSIRZ"] = flat.reshape((NH, NW))
        offset += nlNWNH

    # --- Safety factor ---
    g["QPSI"] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
    offset += nlNW

    # --- Boundary and limiter ---
    if len(EQDSK) > offset + 1:
        parts = [x for x in EQDSK[offset].split() if x]
        g["NBBBS"] = int(parts[0])
        g["LIMITR"] = int(parts[1])
        offset += 1

        nlNBBBS = int(np.ceil(g["NBBBS"] * 2 / 5.0))
        bnd_vals = list(map(float, splitter(merge(EQDSK[offset : offset + nlNBBBS]))))
        g["RBBBS"] = np.array(bnd_vals[0::2])[: g["NBBBS"]]
        g["ZBBBS"] = np.array(bnd_vals[1::2])[: g["NBBBS"]]
        offset += max(nlNBBBS, 1)

        try:
            nlLIMITR = int(np.ceil(g["LIMITR"] * 2 / 5.0))
            lim_vals = list(map(float, splitter(merge(EQDSK[offset : offset + nlLIMITR]))))
            g["RLIM"] = np.array(lim_vals[0::2])[: g["LIMITR"]]
            g["ZLIM"] = np.array(lim_vals[1::2])[: g["LIMITR"]]
            offset += nlLIMITR
        except ValueError:
            # Fallback: construct rectangular limiter
            g["LIMITR"] = 5
            dd = g["RDIM"] / 10.0
            R = np.linspace(0, g["RDIM"], 2) + g["RLEFT"]
            Z = np.linspace(0, g["ZDIM"], 2) - g["ZDIM"] / 2.0 + g["ZMID"]
            rmin = max(R[0], np.min(g["RBBBS"]) - dd)
            rmax = min(R[1], np.max(g["RBBBS"]) + dd)
            zmin = max(Z[0], np.min(g["ZBBBS"]) - dd)
            zmax = min(Z[1], np.max(g["ZBBBS"]) + dd)
            g["RLIM"] = np.array([rmin, rmax, rmax, rmin, rmin])
            g["ZLIM"] = np.array([zmin, zmin, zmax, zmax, zmin])
    else:
        g["NBBBS"] = 0
        g["LIMITR"] = 0
        g["RBBBS"] = np.array([])
        g["ZBBBS"] = np.array([])
        g["RLIM"] = np.array([])
        g["ZLIM"] = np.array([])

    # --- Optional extended data (RHOVN, PCURRT, etc.) ---
    try:
        parts = [float(x) for x in EQDSK[offset].split() if x]
        g["KVTOR"], g["RVTOR"], g["NMASS"] = parts[:3]
        offset += 1

        if g["KVTOR"] > 0:
            g["PRESSW"] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
            offset += nlNW
            g["PWPRIM"] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
            offset += nlNW

        if g["NMASS"] > 0:
            g["DMION"] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
            offset += nlNW

        g["RHOVN"] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
        offset += nlNW
    except Exception:
        pass

    # Add RHOVN if missing
    if "RHOVN" not in g or len(g.get("RHOVN", [])) == 0:
        g["RHOVN"] = np.sqrt(np.linspace(0, 1, NW))

    # Fix missing PRES (e.g. some EAST g-files)
    if not np.any(g["PRES"]):
        pres = integrate.cumulative_trapezoid(
            g["PPRIME"],
            np.linspace(g["SIMAG"], g["SIBRY"], len(g["PPRIME"])),
            initial=0,
        )
        g["PRES"] = pres - pres[-1]

    return g


# ---------------------------------------------------------------------------
# Section 2b: GEQDSK writer
# ---------------------------------------------------------------------------

def _write_fortran_block(stream, values):
    """Write *values* in standard GEQDSK format (5 per line, 16 chars each)."""
    for i, v in enumerate(values):
        stream.write(f"{v:16.9E}")
        if (i + 1) % 5 == 0:
            stream.write("\n")
    if len(values) % 5 != 0:
        stream.write("\n")


def _write_geqdsk_to_stream(g, stream):
    """Serialise a raw g-file dict *g* to an open text *stream*."""
    NW = int(g["NW"])
    NH = int(g["NH"])

    # Header
    case_arr = g.get("CASE", np.array([" " * 8] * 6))
    case_str = "".join(c.ljust(8)[:8] for c in case_arr)[:48]
    stream.write(f"{case_str}{0:4d}{NW:4d}{NH:4d}\n")

    # 20 scalars
    scalars = [
        g["RDIM"], g["ZDIM"], g["RCENTR"], g["RLEFT"], g["ZMID"],
        g["RMAXIS"], g["ZMAXIS"], g["SIMAG"], g["SIBRY"], g["BCENTR"],
        g["CURRENT"], g["SIMAG"], 0.0, g["RMAXIS"], 0.0,
        g["ZMAXIS"], 0.0, g["SIBRY"], 0.0, 0.0,
    ]
    _write_fortran_block(stream, scalars)

    # 1-D profiles
    for name in ("FPOL", "PRES", "FFPRIM", "PPRIME"):
        _write_fortran_block(stream, g[name])

    # 2-D poloidal flux
    _write_fortran_block(stream, g["PSIRZ"].ravel())

    # Safety factor
    _write_fortran_block(stream, g["QPSI"])

    # Boundary and limiter
    nbbbs = int(g.get("NBBBS", len(g.get("RBBBS", []))))
    nlim = int(g.get("LIMITR", len(g.get("RLIM", []))))
    stream.write(f" {nbbbs:5d} {nlim:5d}\n")

    if nbbbs > 0:
        bnd = np.empty(2 * nbbbs)
        bnd[0::2] = g["RBBBS"][:nbbbs]
        bnd[1::2] = g["ZBBBS"][:nbbbs]
        _write_fortran_block(stream, bnd)

    if nlim > 0:
        lim = np.empty(2 * nlim)
        lim[0::2] = g["RLIM"][:nlim]
        lim[1::2] = g["ZLIM"][:nlim]
        _write_fortran_block(stream, lim)


def _write_geqdsk(g, filename):
    """Write a raw g-file dict to *filename*."""
    with open(filename, "w") as f:
        _write_geqdsk_to_stream(g, f)


# ---------------------------------------------------------------------------
# Section 3: Contour tracing
# ---------------------------------------------------------------------------

def _trace_contours(R, Z, PSI, levels):
    """Extract contour lines of PSI at given levels using contourpy.

    Parameters
    ----------
    R, Z : 1-D arrays
        Grid coordinates.
    PSI : 2-D array (len(Z), len(R))
        Poloidal flux on the grid.
    levels : 1-D array
        PSI values at which to extract contours.

    Returns
    -------
    list of list of ndarray
        For each level, a list of (N, 2) arrays of (R, Z) points.
    """
    cg = contourpy.contour_generator(R, Z, PSI, name="serial", line_type="Separate")
    all_contours = []
    for lev in levels:
        segments = cg.lines(lev)
        all_contours.append(segments)
    return all_contours


def _select_main_contour(segments, R0, Z0, sigma_RpZ, sigma_rhotp):
    """Select the main closed contour encircling the magnetic axis.

    Uses the winding/angular-coverage criterion: the contour whose
    double integral of angle vs. arc-fraction has the largest amplitude
    is most likely the one that wraps around the axis.

    Parameters
    ----------
    segments : list of (N, 2) arrays
        Candidate contour segments at a single PSI level.
    R0, Z0 : float
        Magnetic axis position.
    sigma_RpZ, sigma_rhotp : int
        COCOS sign factors for orientation.

    Returns
    -------
    ndarray (N, 2) or None
        The selected contour with correct orientation, or None if no
        valid contour found.
    """
    if not segments:
        return None

    sign_theta = sigma_RpZ * sigma_rhotp

    best = None
    best_score = -1.0

    for seg in segments:
        r, z = seg[:, 0], seg[:, 1]
        if len(r) < 4 or np.any(np.isnan(r * z)):
            continue

        # Close the contour exactly
        r = r.copy()
        z = z.copy()
        r[0] = r[-1] = 0.5 * (r[0] + r[-1])
        z[0] = z[-1] = 0.5 * (z[0] + z[-1])

        # Angle from axis, unwrapped
        theta = np.unwrap(np.arctan2(z - Z0, r - R0))
        theta -= np.mean(theta)
        s = np.linspace(0, 1, len(theta))

        # Winding score: peak of double integral
        score = np.max(np.abs(
            integrate.cumulative_trapezoid(
                integrate.cumulative_trapezoid(theta, s, initial=0), s
            )
        ))

        if score > best_score:
            best_score = score
            # Determine orientation and flip if needed
            orient = int(np.sign(
                (z[0] - Z0) * (r[1] - r[0]) - (r[0] - R0) * (z[1] - z[0])
            ))
            if orient != 0:
                best = seg[::sign_theta * orient, :]
            else:
                best = seg

    return best


def _crop_at_xpoint(seg, R0, Z0):
    """Crop an open separatrix contour at the X-point to produce a closed LCFS.

    For a lower single null (or upper), contourpy returns an open contour
    that goes through the X-point and into the divertor legs.  This
    function finds the two points where the contour is closest to the
    X-point (one per pass) and keeps only the segment between them that
    encircles the magnetic axis.

    Parameters
    ----------
    seg : ndarray (N, 2)
        Open contour segment with columns (R, Z).
    R0, Z0 : float
        Magnetic axis position.

    Returns
    -------
    ndarray (M, 2)
        Closed contour (first point == last point).
    """
    r, z = seg[:, 0], seg[:, 1]
    n = len(r)

    # Check if already closed
    gap = np.sqrt((r[0] - r[-1])**2 + (z[0] - z[-1])**2)
    perimeter = np.sum(np.sqrt(np.diff(r)**2 + np.diff(z)**2))
    if gap < 0.01 * perimeter:
        out = seg.copy()
        out[-1] = out[0]
        return out

    # Approximate X-point from the gap midpoint
    r_xpt = 0.5 * (r[0] + r[-1])
    z_xpt = 0.5 * (z[0] + z[-1])

    # Distance of every contour point from the X-point
    dist = np.sqrt((r - r_xpt)**2 + (z - z_xpt)**2)

    # Find local minima in the distance profile (passes near X-point).
    # Smooth slightly to avoid noise.
    from scipy.ndimage import uniform_filter1d
    dist_smooth = uniform_filter1d(dist, size=max(5, n // 50))

    # Find all local minima
    local_min = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if dist_smooth[i] < dist_smooth[i-1] and dist_smooth[i] < dist_smooth[i+1]:
            local_min[i] = True

    min_indices = np.where(local_min)[0]

    if len(min_indices) < 2:
        # Can't find two X-point passes — just close it
        out = seg.copy()
        out[-1] = out[0]
        return out

    # Sort minima by distance and take the two closest to the X-point
    sorted_by_dist = min_indices[np.argsort(dist[min_indices])]

    # We need two minima that are well separated (not adjacent)
    idx1 = sorted_by_dist[0]
    idx2 = None
    for candidate in sorted_by_dist[1:]:
        separation = abs(candidate - idx1)
        if separation > n // 10:  # at least 10% of contour apart
            idx2 = candidate
            break

    if idx2 is None:
        out = seg.copy()
        out[-1] = out[0]
        return out

    # Ensure idx1 < idx2
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    # The plasma-encircling portion is between idx1 and idx2
    plasma_seg = seg[idx1:idx2+1]

    # Verify it encircles the axis
    theta = np.unwrap(np.arctan2(plasma_seg[:, 1] - Z0,
                                  plasma_seg[:, 0] - R0))
    winding = abs(theta[-1] - theta[0])

    if winding < np.pi:
        # Wrong half — use the complement
        plasma_seg = np.vstack([seg[idx2:], seg[:idx1+1]])

    # Close the contour
    closed = np.vstack([plasma_seg, plasma_seg[:1]])
    return closed


# ---------------------------------------------------------------------------
# Section 4: Flux-surface geometry helper
# ---------------------------------------------------------------------------

def _flux_geometry(R, Z):
    """Compute geometric properties of a single flux surface contour.

    Parameters
    ----------
    R, Z : 1-D arrays
        Contour points (should be closed: first == last, or nearly so).

    Returns
    -------
    dict with keys: R (geometric center), Z, a (minor radius),
        kappa, delta, perimeter, surfArea
    """
    geo = {}

    # --- Parabolic sub-grid extremum finder (matches OMFIT's
    # ``parabolaMaxCycle`` in utils_math.py) ---------------------------
    def _parabola_extremum(idx, main, other):
        """Parabolic fit through 3 points around *idx* on a closed curve.

        Returns ``(other_at_extremum, main_at_extremum)`` — the
        coordinates of the refined extremum in both arrays.
        """
        n = len(main)
        if n < 3:
            return other[idx], main[idx]

        # Cyclic neighbour indices (closed contour: first == last)
        im = (idx - 1) % (n - 1) if main[0] == main[-1] else max(idx - 1, 0)
        ip = (idx + 1) % (n - 1) if main[0] == main[-1] else min(idx + 1, n - 1)

        ym1, y0, yp1 = main[im], main[idx], main[ip]
        xm1, x0, xp1 = other[im], other[idx], other[ip]

        denom = 2.0 * (ym1 - 2.0 * y0 + yp1)
        if abs(denom) < 1e-30:
            return x0, y0
        frac = (ym1 - yp1) / denom  # fractional index offset
        frac = np.clip(frac, -0.5, 0.5)
        x_ext = x0 + frac * 0.5 * (xp1 - xm1)
        y_ext = y0 + frac * 0.5 * (yp1 - ym1)
        return x_ext, y_ext

    # Four principal extrema via parabolic refinement
    imaxr = np.argmax(R)
    z_at_max_r, max_r = _parabola_extremum(imaxr, R, Z)

    iminr = np.argmin(R)
    z_at_min_r, min_r = _parabola_extremum(iminr, R, Z)

    imaxz = np.argmax(Z)
    r_at_max_z, max_z = _parabola_extremum(imaxz, Z, R)

    iminz = np.argmin(Z)
    r_at_min_z, min_z = _parabola_extremum(iminz, Z, R)

    # Closed-contour arc-length: include the closing segment.
    dR = np.diff(R, append=R[0])
    dZ = np.diff(Z, append=Z[0])
    dl_segs = np.sqrt(dR**2 + dZ**2)
    # Trapezoidal weights: each point gets half of its two adjacent segments
    dl = 0.5 * (dl_segs + np.roll(dl_segs, 1))

    geo["R"] = 0.5 * (max_r + min_r)
    geo["Z"] = 0.5 * (max_z + min_z)
    geo["a"] = 0.5 * (max_r - min_r)
    if geo["a"] > 0:
        geo["kappa"] = 0.5 * (max_z - min_z) / geo["a"]
        geo["kapu"] = (max_z - z_at_max_r) / geo["a"]
        geo["kapl"] = (z_at_max_r - min_z) / geo["a"]
        geo["delu"] = (geo["R"] - r_at_max_z) / geo["a"]
        geo["dell"] = (geo["R"] - r_at_min_z) / geo["a"]
        geo["delta"] = 0.5 * (geo["delu"] + geo["dell"])
    else:
        geo["kappa"] = 1.0
        geo["kapu"] = 1.0
        geo["kapl"] = 1.0
        geo["delu"] = 0.0
        geo["dell"] = 0.0
        geo["delta"] = 0.0
    geo["perimeter"] = np.sum(dl)
    geo["surfArea"] = 2 * np.pi * np.sum(R * dl)
    geo["eps"] = geo["a"] / geo["R"] if geo["R"] > 0 else 0.0

    return geo


def _resample_contour(R, Z, npts=257, periodic=True):
    """Resample a closed contour to *npts* equally-spaced-in-arc-length points.

    Parameters
    ----------
    R, Z : 1-D arrays
        Contour coordinates (should be nearly closed: first ≈ last).
    npts : int
        Number of output points (including the repeated closing point).
    periodic : bool
        If *True* (default), use a periodic cubic spline so the result is
        smooth and closed by construction.  Set to *False* for the separatrix
        surface: the X-point cusp produces a derivative discontinuity that a
        periodic spline would smooth out, distorting the contour.  A
        non-periodic spline preserves the cusp while still providing uniform
        arc-length spacing for good quadrature.

    Returns
    -------
    R_new, Z_new : 1-D arrays of length *npts*
    """
    # Ensure exact closure before fitting spline
    R = np.asarray(R, dtype=float).copy()
    Z = np.asarray(Z, dtype=float).copy()
    R[-1], Z[-1] = R[0], Z[0]

    # Cumulative arc-length parameter
    ds = np.sqrt(np.diff(R) ** 2 + np.diff(Z) ** 2)
    s = np.empty(len(R))
    s[0] = 0.0
    s[1:] = np.cumsum(ds)
    s_total = s[-1]

    if s_total < 1e-14:
        return R, Z  # degenerate contour

    # Cubic spline — periodic for interior surfaces, non-periodic for
    # the separatrix so the X-point cusp is preserved.
    tck_R = interpolate.splrep(s, R, k=3, per=periodic)
    tck_Z = interpolate.splrep(s, Z, k=3, per=periodic)

    # Evaluate at equally-spaced arc-length positions
    s_new = np.linspace(0, s_total, npts)
    R_new = interpolate.splev(s_new, tck_R)
    Z_new = interpolate.splev(s_new, tck_Z)

    if not periodic:
        # Force exact closure for the non-periodic case
        R_new[-1], Z_new[-1] = R_new[0], Z_new[0]

    return R_new, Z_new


def _detect_xpoint_angle(R, Z, R0, Z0):
    """Detect the X-point angle on a separatrix contour.

    Finds the location of the sharpest bend (minimum cosine of angle
    between adjacent tangent vectors) and returns the corresponding
    θ = arctan2(Z − Z₀, R − R₀).  Adapted from OMFIT fluxSurface.py
    lines 1115-1124.

    Parameters
    ----------
    R, Z : 1-D arrays
        Separatrix contour (should be nearly closed: first ≈ last).
    R0, Z0 : float
        Magnetic axis coordinates.

    Returns
    -------
    float
        Poloidal angle of the X-point, or ``None`` if detection fails.
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = len(R)
    if n < 6:
        return None

    n1 = n - 1  # number of unique points (last = first)
    dR_a = np.gradient(R[:n1])
    dZ_a = np.gradient(Z[:n1])
    dR_b = np.gradient(np.concatenate([R[1:n1], R[0:1]]))
    dZ_b = np.gradient(np.concatenate([Z[1:n1], Z[0:1]]))
    dot = dR_a * dR_b + dZ_a * dZ_b
    norm = np.sqrt(dR_a**2 + dZ_a**2) * np.sqrt(dR_b**2 + dZ_b**2)
    norm = np.maximum(norm, 1e-30)
    cos_angle = dot / norm
    idx_xpt = np.argmin(cos_angle)

    return float(np.arctan2(Z[idx_xpt] - Z0, R[idx_xpt] - R0))


def _resample_contour_theta(R, Z, R0, Z0, npts=257, theta_xpt=None):
    """Resample a closed contour to *npts* equally-spaced-in-angle points.

    Matches OMFIT's ``fluxSurface._resample`` method (fluxSurface.py
    lines 1091-1169): fits periodic cubic splines R(θ) and Z(θ) where
    θ = arctan2(Z − Z₀, R − R₀), sorts points starting from the
    X-point angle, and evaluates on a uniform θ grid with OMFIT-style
    domain wrapping.

    Following OMFIT's convention, the X-point angle should be detected
    once on the separatrix and reused for all near-separatrix surfaces.
    If *theta_xpt* is not provided the sharpest bend on *this* contour
    is used as a fallback.

    Parameters
    ----------
    R, Z : 1-D arrays
        Contour coordinates (should be nearly closed: first ≈ last).
    R0, Z0 : float
        Magnetic axis coordinates.
    npts : int
        Number of output points (including the repeated closing point).
    theta_xpt : float or None
        Pre-computed X-point angle (from ``_detect_xpoint_angle`` on the
        separatrix).  If ``None``, detected from this contour.

    Returns
    -------
    R_new, Z_new : 1-D arrays of length *npts*
    """
    R = np.asarray(R, dtype=float).copy()
    Z = np.asarray(Z, dtype=float).copy()
    R[-1], Z[-1] = R[0], Z[0]
    n = len(R)

    if n < 6:
        return R, Z  # too few points to fit a spline

    n1 = n - 1  # number of unique points (last = first)

    # --- X-point angle: use pre-computed value or detect from this contour
    if theta_xpt is None:
        theta_xpt = _detect_xpoint_angle(R, Z, R0, Z0)
        if theta_xpt is None:
            return R, Z

    # --- Compute theta from magnetic axis, sort starting from X-point.
    #     This mirrors OMFIT fluxSurface.py lines 1139-1161.
    t_raw = np.arctan2(Z - Z0, R - R0)

    # Sort contour points by theta relative to X-point angle
    t_rel = (t_raw[:n1] - theta_xpt) % (2.0 * np.pi)
    idx = np.argsort(t_rel)
    idx = np.concatenate([idx, idx[0:1]])  # re-close

    t_sorted = np.unwrap(t_raw[idx])
    R_sorted = R[idx]
    Z_sorted = Z[idx]

    # Force monotonically increasing (OMFIT lines 1148-1152)
    if t_sorted[0] > t_sorted[1]:
        t_sorted = -t_sorted

    # Guard against degenerate theta span
    t_span = t_sorted[-1] - t_sorted[0]
    if abs(t_span) < 1e-6:
        return R, Z

    # --- Build uniform evaluation grid starting at X-point angle, then
    #     sort it the same way (OMFIT lines 1130, 1154-1156).
    theta0 = np.linspace(0, 2 * np.pi, npts) + theta_xpt
    # Sort evaluation grid relative to X-point (same wrapping as data)
    idx_eval = np.argsort((theta0[:-1] + theta_xpt) % (2 * np.pi) - theta_xpt)
    idx_eval = np.concatenate([idx_eval, idx_eval[0:1]])
    theta_eval = np.unwrap(theta0[idx_eval])
    # Match sign convention of t_sorted
    if t_sorted[0] > 0 and theta_eval[0] < 0:
        theta_eval = -theta_eval
    elif t_sorted[0] < 0 and theta_eval[0] > 0:
        theta_eval = -theta_eval

    # --- Periodic cubic spline in theta-space.
    #
    #     OMFIT fluxSurface.py line 1167 uses per=1 (periodic) for all
    #     surfaces.  While a comment ``# per=per)`` suggests per=0 was
    #     intended for the separatrix, the running code is per=1.  We
    #     match OMFIT's running code: the periodic spline combined with
    #     the evaluation wrapping below handles the X-point topology
    #     well in practice.
    tck_R = interpolate.splrep(t_sorted, R_sorted, k=3, per=True)
    tck_Z = interpolate.splrep(t_sorted, Z_sorted, k=3, per=True)

    # Evaluate with OMFIT-style domain wrapping (line 1168):
    #   (theta - t[0]) % (t[-1] - t[0]) + t[0]
    t_range = t_sorted[-1] - t_sorted[0]
    theta_wrapped = (theta_eval - t_sorted[0]) % t_range + t_sorted[0]
    R_new = interpolate.splev(theta_wrapped, tck_R, ext=0)
    Z_new = interpolate.splev(theta_wrapped, tck_Z, ext=0)

    # Enforce exact closure
    R_new[-1], Z_new[-1] = R_new[0], Z_new[0]
    return R_new, Z_new


# ---------------------------------------------------------------------------
# Section 5: GEQDSKEquilibrium class
# ---------------------------------------------------------------------------

class GEQDSKEquilibrium:
    """GEQDSK equilibrium with flux-surface-averaged quantities.

    Reads a standard g-file and lazily computes magnetic fields,
    flux-surface contours, averages, geometry, and inductance.

    Parameters
    ----------
    filename : str or path-like
        Path to the g-file.
    cocos : int
        COCOS convention index (default 1, standard EFIT).
    nlevels : int or None
        Number of normalised-psi levels for flux-surface analysis.
        Defaults to NW from the g-file.
    resample : ``"theta"`` or ``"arc_length"``
        Contour resampling method for near-separatrix surfaces
        (ψ_N ≥ 0.99).  ``"theta"`` (default) resamples in angular
        space from the magnetic axis using the OMFIT "method of
        angles", which preserves the X-point cusp via a non-periodic
        spline.  ``"arc_length"`` falls back to the raw contourpy
        points without resampling.
    """

    def __init__(self, filename, cocos=1, nlevels=None, resample="theta",
                 extrapolate_edge=True):
        self._raw = _read_geqdsk(filename)
        self._cocos_index = int(cocos)
        self._cocos = _cocos_params(cocos)
        self._nlevels = nlevels if nlevels is not None else int(self._raw["NW"])
        if resample not in ("theta", "arc_length"):
            raise ValueError(
                f"resample must be 'theta' or 'arc_length', got {resample!r}"
            )
        self._resample_method = resample
        self._extrapolate_edge = bool(extrapolate_edge)
        self._cache = {}

    @classmethod
    def from_bytes(cls, raw_bytes, cocos=1, nlevels=None, resample="theta",
                   extrapolate_edge=True):
        """Construct from in-memory bytes (e.g. from HDF5 storage).

        Parameters
        ----------
        raw_bytes : bytes
            Raw content of a g-file.
        cocos : int
            COCOS index.
        nlevels : int
            Number of psi_N levels.
        resample : str
            Contour resampling method (``"theta"`` or ``"arc_length"``).
        extrapolate_edge : bool
            If ``True`` (default), extrapolate p' and FF' at the
            separatrix when the g-file has them forced to zero.
        """
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".geqdsk",
                                         delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        return cls(tmp_path, cocos=cocos, nlevels=nlevels, resample=resample,
                   extrapolate_edge=extrapolate_edge)

    @classmethod
    def from_raw(cls, raw_dict, cocos=1, nlevels=None, resample="theta",
                 extrapolate_edge=True):
        """Construct directly from a raw g-file dict (no file I/O).

        Parameters
        ----------
        raw_dict : dict
            Dictionary with standard GEQDSK keys (as returned by
            ``_read_geqdsk``).  A shallow copy is made internally.
        cocos : int
            COCOS index for the data.
        nlevels, resample, extrapolate_edge :
            Same as ``__init__``.
        """
        obj = object.__new__(cls)
        obj._raw = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in raw_dict.items()}
        obj._cocos_index = int(cocos)
        obj._cocos = _cocos_params(cocos)
        obj._nlevels = nlevels if nlevels is not None else int(obj._raw["NW"])
        if resample not in ("theta", "arc_length"):
            raise ValueError(
                f"resample must be 'theta' or 'arc_length', got {resample!r}"
            )
        obj._resample_method = resample
        obj._extrapolate_edge = bool(extrapolate_edge)
        obj._cache = {}
        return obj

    # --- Raw data properties ---

    @property
    def R_grid(self):
        """1-D R grid."""
        if "R_grid" not in self._cache:
            NW = int(self._raw["NW"])
            self._cache["R_grid"] = np.linspace(
                self._raw["RLEFT"],
                self._raw["RLEFT"] + self._raw["RDIM"],
                NW,
            )
        return self._cache["R_grid"]

    @property
    def Z_grid(self):
        """1-D Z grid."""
        if "Z_grid" not in self._cache:
            NH = int(self._raw["NH"])
            self._cache["Z_grid"] = np.linspace(
                self._raw["ZMID"] - self._raw["ZDIM"] / 2.0,
                self._raw["ZMID"] + self._raw["ZDIM"] / 2.0,
                NH,
            )
        return self._cache["Z_grid"]

    @property
    def psi_RZ(self):
        """2-D poloidal flux array (NH x NW)."""
        return self._raw["PSIRZ"]

    @property
    def psi_N_RZ(self):
        """2-D normalised poloidal flux on the (R, Z) grid (NH x NW).

        ``(psi - psi_axis) / (psi_boundary - psi_axis)``, ranging from
        0 at the magnetic axis to 1 at the LCFS.
        """
        return (self._raw["PSIRZ"] - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    @property
    def psi_axis(self):
        """Poloidal flux at the magnetic axis."""
        return self._raw["SIMAG"]

    @property
    def psi_boundary(self):
        """Poloidal flux at the last closed flux surface."""
        return self._raw["SIBRY"]

    @property
    def psi_N(self):
        """Normalised psi levels used for flux-surface analysis."""
        return np.linspace(0, 1, self._nlevels)

    @property
    def fpol(self):
        """F = R*Bt poloidal current function, on uniform psi_N grid."""
        return self._raw["FPOL"]

    @property
    def pres(self):
        """Pressure profile on uniform psi_N grid."""
        return self._raw["PRES"]

    @property
    def pprime(self):
        """dP/dpsi on uniform psi_N grid."""
        return self._raw["PPRIME"]

    @property
    def ffprim(self):
        """F*dF/dpsi on uniform psi_N grid."""
        return self._raw["FFPRIM"]

    @property
    def qpsi(self):
        """Safety factor from the g-file on uniform psi_N grid."""
        return self._raw["QPSI"]

    @property
    def Ip(self):
        """Plasma current [A]."""
        return self._raw["CURRENT"]

    @property
    def R_mag(self):
        """R of the magnetic axis [m]."""
        return self._raw["RMAXIS"]

    @property
    def Z_mag(self):
        """Z of the magnetic axis [m]."""
        return self._raw["ZMAXIS"]

    @property
    def R_center(self):
        """Reference geometric center R [m]."""
        return self._raw["RCENTR"]

    @property
    def B_center(self):
        """Vacuum toroidal field at R_center [T]."""
        return self._raw["BCENTR"]

    @property
    def boundary_R(self):
        """R coordinates of the plasma boundary."""
        return self._raw["RBBBS"]

    @property
    def boundary_Z(self):
        """Z coordinates of the plasma boundary."""
        return self._raw["ZBBBS"]

    @property
    def limiter_R(self):
        """R coordinates of the limiter."""
        return self._raw["RLIM"]

    @property
    def limiter_Z(self):
        """Z coordinates of the limiter."""
        return self._raw["ZLIM"]

    @property
    def rhovn(self):
        r"""Normalised toroidal flux coordinate
        :math:`\rho = \sqrt{\Phi_{\rm tor}/\Phi_{\rm tor,edge}}`.

        Always computed from the safety factor profile:
        :math:`\Phi(\psi) = \int_{\psi_{\rm axis}}^{\psi} q\,d\psi'`.

        Many equilibrium solvers (including TokaMaker) write a
        placeholder ``RHOVN = sqrt(psi_N)`` to the g-file, which is
        only correct when q is constant.  This property ignores any
        stored ``RHOVN`` and recomputes from ``QPSI`` for accuracy.
        """
        NW = int(self._raw["NW"])
        psi_grid = np.linspace(self.psi_axis, self.psi_boundary, NW)
        phi = integrate.cumulative_trapezoid(self._raw["QPSI"], psi_grid, initial=0)
        phi_edge = phi[-1]
        if abs(phi_edge) < 1e-30:
            return np.sqrt(np.linspace(0, 1, NW))
        return np.sqrt(np.abs(phi / phi_edge))

    # --- COCOS property ---

    @property
    def cocos(self):
        """Current COCOS convention index."""
        return self._cocos_index

    # --- COCOS conversion and sign-flip methods ---

    def cocosify(self, cocos_out, copy=False):
        """Convert the raw g-file data from the current COCOS to *cocos_out*.

        Applies the multiplicative sign/2π factors to every relevant
        field following Sauter & Medvedev, Comput. Phys. Commun. 184
        (2013) 293, Eq. 14/23.

        Parameters
        ----------
        cocos_out : int
            Target COCOS convention index (1-8 or 11-18).
        copy : bool
            If ``True``, return a **new** ``GEQDSKEquilibrium`` with the
            converted data, leaving this object unchanged.  If ``False``
            (default), convert **in place** and return ``self``.

        Returns
        -------
        GEQDSKEquilibrium
            The converted object (``self`` when *copy=False*).
        """
        cc_in = _cocos_params(self._cocos_index)
        cc_out = _cocos_params(cocos_out)

        # Effective transformation parameters (Eq. 23)
        sigma_Bp_eff = cc_out["sigma_Bp"] * cc_in["sigma_Bp"]
        sigma_RpZ_eff = cc_out["sigma_RpZ"] * cc_in["sigma_RpZ"]
        sigma_rhotp_eff = cc_out["sigma_rhotp"] * cc_in["sigma_rhotp"]
        exp_Bp_eff = cc_out["exp_Bp"] - cc_in["exp_Bp"]

        # sigma_Ip_eff = sigma_B0_eff = sigma_RpZ_eff
        twopi_exp = (2.0 * np.pi) ** exp_Bp_eff

        # Multiplicative factors
        psi_factor = sigma_RpZ_eff * sigma_Bp_eff * twopi_exp
        dpsi_factor = sigma_RpZ_eff * sigma_Bp_eff / twopi_exp
        bt_factor = sigma_RpZ_eff
        ip_factor = sigma_RpZ_eff
        q_factor = sigma_rhotp_eff  # = sigma_Ip*sigma_B0*sigma_rhotp_eff

        target = self._copy_for_mutation() if copy else self

        # Apply to raw dict
        g = target._raw
        for key in ("SIMAG", "SIBRY"):
            g[key] *= psi_factor
        g["PSIRZ"] = g["PSIRZ"] * psi_factor
        g["PPRIME"] = g["PPRIME"] * dpsi_factor
        g["FFPRIM"] = g["FFPRIM"] * dpsi_factor
        g["FPOL"] = g["FPOL"] * bt_factor
        g["BCENTR"] *= bt_factor
        g["CURRENT"] *= ip_factor
        g["QPSI"] = g["QPSI"] * q_factor

        # Update COCOS bookkeeping
        target._cocos_index = int(cocos_out)
        target._cocos = cc_out
        target._cache.clear()
        return target

    def flip_Bt_Ip(self, copy=False):
        """Reverse the signs of Bt and Ip in the raw g-file data.

        This negates ``BCENTR``, ``FPOL``, ``CURRENT``, ``PSIRZ``,
        ``SIMAG``, ``SIBRY``, ``PPRIME``, and ``FFPRIM`` — equivalent
        to flipping the direction of both the toroidal field and the
        plasma current while keeping the COCOS index unchanged.

        Parameters
        ----------
        copy : bool
            If ``True``, return a new object; otherwise modify in place.

        Returns
        -------
        GEQDSKEquilibrium
        """
        target = self._copy_for_mutation() if copy else self
        g = target._raw

        g["BCENTR"] *= -1
        g["FPOL"] = g["FPOL"] * -1
        g["CURRENT"] *= -1
        g["SIMAG"] *= -1
        g["SIBRY"] *= -1
        g["PSIRZ"] = g["PSIRZ"] * -1
        g["PPRIME"] = g["PPRIME"] * -1
        g["FFPRIM"] = g["FFPRIM"] * -1

        target._cache.clear()
        return target

    def _copy_for_mutation(self):
        """Return a deep-enough copy for mutation (cocosify / flip)."""
        return GEQDSKEquilibrium.from_raw(
            self._raw, cocos=self._cocos_index,
            nlevels=self._nlevels, resample=self._resample_method,
            extrapolate_edge=self._extrapolate_edge,
        )

    # --- Save / serialise ---

    def save(self, filename):
        """Write the (possibly modified) g-file data to *filename*.

        Parameters
        ----------
        filename : str or path-like
            Output path.
        """
        _write_geqdsk(self._raw, filename)

    def to_bytes(self):
        """Serialise to in-memory bytes (round-trips with ``from_bytes``).

        Returns
        -------
        bytes
        """
        buf = io.StringIO()
        _write_geqdsk_to_stream(self._raw, buf)
        return buf.getvalue().encode("ascii")

    # --- Lazy field computation ---

    def _compute_fields(self):
        """Compute Br, Bz, Jt on the full (R, Z) grid."""
        if "Br" in self._cache:
            return

        cc = self._cocos
        R = self.R_grid
        Z = self.Z_grid
        PSI = self.psi_RZ
        RR, _ZZ = np.meshgrid(R, Z)

        dR = R[1] - R[0]
        dZ = Z[1] - Z[0]
        dPSIdZ, dPSIdR = np.gradient(PSI, dZ, dR)

        twopi_exp = (2.0 * np.pi) ** cc["exp_Bp"]
        self._cache["Br"] = cc["sigma_RpZ"] * cc["sigma_Bp"] * dPSIdZ / (RR * twopi_exp)
        self._cache["Bz"] = -cc["sigma_RpZ"] * cc["sigma_Bp"] * dPSIdR / (RR * twopi_exp)

        Br = self._cache["Br"]
        Bz = self._cache["Bz"]
        dBrdZ, dBrdR = np.gradient(Br, dZ, dR)
        _dBzdZ, dBzdR = np.gradient(Bz, dZ, dR)
        self._cache["Jt"] = cc["sigma_RpZ"] * (dBrdZ - dBzdR) / constants.mu_0

    # --- Lazy flux-surface tracing and averaging ---

    def _trace_surfaces(self):
        """Trace flux surfaces and compute all averaged quantities."""
        if "avg" in self._cache:
            return

        self._compute_fields()

        cc = self._cocos
        R = self.R_grid
        Z = self.Z_grid

        psi_N_levels = self.psi_N
        dpsi = self.psi_boundary - self.psi_axis
        psi_levels = psi_N_levels * dpsi + self.psi_axis

        # Interpolators for fields
        Br_interp = interpolate.RectBivariateSpline(Z, R, self._cache["Br"])
        Bz_interp = interpolate.RectBivariateSpline(Z, R, self._cache["Bz"])
        Jt_interp = interpolate.RectBivariateSpline(Z, R, self._cache["Jt"])

        # Interpolator for F(psi_N)
        NW = int(self._raw["NW"])
        psi_N_raw = np.linspace(0, 1, NW)
        F_interp = interpolate.InterpolatedUnivariateSpline(psi_N_raw, self._raw["FPOL"])

        # Many equilibrium solvers (EFIT, TokaMaker) force p' = FF' = 0 at
        # the last grid point (ψ_N = 1) as a free-boundary condition.  This
        # creates a discontinuous jump that makes the direct-GS Jt profile
        # artificially zero at the separatrix.  When extrapolate_edge is
        # True, detect and undo this: if the last value is zero while the
        # preceding points have significant magnitude, extrapolate from the
        # neighbours instead.
        pprime_raw = self._raw["PPRIME"].copy()
        ffprim_raw = self._raw["FFPRIM"].copy()
        if self._extrapolate_edge:
            for prof in (pprime_raw, ffprim_raw):
                if (
                    NW >= 4
                    and prof[-1] == 0.0
                    and abs(prof[-2]) > 1e-30
                ):
                    # Quadratic extrapolation from last three non-zero points
                    prof[-1] = 3.0 * prof[-2] - 3.0 * prof[-3] + prof[-4]
                    # Clip to zero if the extrapolation overshoots past
                    # zero (sign reversal relative to the neighbour).
                    # Use np.sign rather than a product test to avoid
                    # float underflow when both values are tiny.
                    if np.sign(prof[-1]) != np.sign(prof[-2]):
                        prof[-1] = 0.0
        pprime_interp = interpolate.InterpolatedUnivariateSpline(psi_N_raw, pprime_raw)
        ffprim_interp = interpolate.InterpolatedUnivariateSpline(psi_N_raw, ffprim_raw)

        # Trace contours
        contours = _trace_contours(R, Z, self.psi_RZ, psi_levels)

        nc = len(psi_N_levels)
        R0 = self.R_mag
        Z0 = self.Z_mag

        # Allocate averaged quantities
        avg = {key: np.zeros(nc) for key in [
            "R", "1/R", "1/R**2", "R**2",
            "Bp", "Bp**2", "Bt", "Bt**2", "Btot**2",
            "Jt", "Jt/R",
            "vp", "q", "ip",
            "F", "PPRIME", "FFPRIM",
        ]}
        geo_arrays = {key: np.zeros(nc) for key in [
            "R", "Z", "a", "kappa", "kapu", "kapl",
            "delta", "delu", "dell", "perimeter", "surfArea", "eps",
        ]}
        contour_data = []  # store (R, Z) for each surface

        # Pre-detect X-point angle from the separatrix contour.
        # Following OMFIT convention, this angle is detected once on the
        # separatrix and reused for all near-separatrix surfaces so that
        # the spline domain boundary is always placed at the true X-point
        # location, even for surfaces where the cusp is too subtle for
        # reliable per-contour detection.
        theta_xpt = None
        if self._resample_method == "theta" and nc > 1:
            sep_seg = _select_main_contour(
                contours[nc - 1], R0, Z0,
                cc["sigma_RpZ"], cc["sigma_rhotp"],
            )
            if sep_seg is not None and len(sep_seg) >= 10:
                sep_seg = _crop_at_xpoint(sep_seg, R0, Z0)
            if sep_seg is not None and len(sep_seg) >= 6:
                theta_xpt = _detect_xpoint_angle(
                    sep_seg[:, 0], sep_seg[:, 1], R0, Z0
                )

        # Precompute dpsi spacing (once, outside the loop)
        dpsi_arr = np.abs(np.gradient(psi_levels))

        # Per-surface loop
        Bp2_vol = 0.0
        for k in range(nc):
            pn = psi_N_levels[k]
            F_k = float(F_interp(pn))
            pprime_k = float(pprime_interp(pn))
            ffprim_k = float(ffprim_interp(pn))

            avg["F"][k] = F_k
            avg["PPRIME"][k] = pprime_k
            avg["FFPRIM"][k] = ffprim_k

            if pn == 0:
                # Magnetic axis: create a tiny elliptical contour
                # following OMFIT's approach (fluxSurface.py lines
                # 731-757).  The axis is a degenerate point, but a
                # tiny contour around it allows the averaging loop to
                # process it consistently.  The shape (kappa) is
                # borrowed from the first interior surface (k=1) once
                # that surface has been processed.  For now, store
                # placeholder values; the axis will be revisited
                # AFTER the main loop using k=1's geometry.
                contour_data.append(np.array([[R0, Z0]]))
                avg["R"][k] = R0
                avg["1/R"][k] = 1.0 / R0
                avg["1/R**2"][k] = 1.0 / R0**2
                avg["R**2"][k] = R0**2
                Bt_axis = F_k / R0
                avg["Bp"][k] = 0.0
                avg["Bp**2"][k] = 0.0
                avg["Bt"][k] = Bt_axis
                avg["Bt**2"][k] = Bt_axis**2
                avg["Btot**2"][k] = Bt_axis**2
                Jt0 = float(Jt_interp.ev(Z0, R0))
                avg["Jt"][k] = Jt0
                avg["Jt/R"][k] = Jt0 / R0
                avg["vp"][k] = 0.0
                avg["ip"][k] = 0.0
                for gk in geo_arrays:
                    geo_arrays[gk][k] = 0.0
                geo_arrays["R"][k] = R0
                geo_arrays["Z"][k] = Z0
                continue

            # Select main contour.  At the separatrix (psi_N == 1),
            # contourpy often returns an open contour that includes
            # divertor legs.  Use the g-file's stored LCFS boundary
            # instead — it is already properly closed.
            if pn == 1.0 and len(self.boundary_R) >= 4:
                seg = np.column_stack([self.boundary_R, self.boundary_Z])
                # Ensure closure
                if np.sqrt((seg[0,0]-seg[-1,0])**2 + (seg[0,1]-seg[-1,1])**2) > 1e-6:
                    seg = np.vstack([seg, seg[:1]])
            else:
                seg = _select_main_contour(
                    contours[k], R0, Z0,
                    cc["sigma_RpZ"], cc["sigma_rhotp"],
                )
                if seg is not None and len(seg) >= 10:
                    seg = _crop_at_xpoint(seg, R0, Z0)

            if seg is None or len(seg) < 4:
                contour_data.append(np.array([[R0, Z0]]))
                avg["R"][k] = R0
                avg["1/R"][k] = 1.0 / R0
                avg["1/R**2"][k] = 1.0 / R0**2
                avg["R**2"][k] = R0**2
                continue

            # Resample to equally-spaced arc-length using cubic spline.
            # This smooths contourpy discretisation artefacts and gives
            # uniform quadrature weights.
            #
            # Interior surfaces (psi_N < 0.99): periodic arc-length spline.
            # Near-separatrix (0.99 <= psi_N < 1.0): non-periodic spline
            #   so the X-point cusp is preserved.
            # Separatrix (psi_N == 1.0): use the g-file boundary raw
            #   (no resampling) — this matches OMFIT which uses
            #   RBBBS/ZBBBS directly.  Resampling would smooth the
            #   X-point cusp and distort Bp/q averages.
            if pn == 1.0:
                # Separatrix: use raw boundary points (OMFIT parity)
                r_s, z_s = seg[:, 0].copy(), seg[:, 1].copy()
                r_s[-1], z_s[-1] = r_s[0], z_s[0]
            elif pn >= 0.99 and len(seg) >= 20:
                # Near-separatrix: non-periodic arc-length resampling
                r_s, z_s = _resample_contour(
                    seg[:, 0], seg[:, 1], npts=257, periodic=False,
                )
            elif len(seg) >= 20:
                # Interior: periodic arc-length resampling
                r_s, z_s = _resample_contour(seg[:, 0], seg[:, 1], npts=257)
            else:
                # Fallback: raw contourpy points
                r_s, z_s = seg[:, 0].copy(), seg[:, 1].copy()
                r_s[-1], z_s[-1] = r_s[0], z_s[0]
            contour_data.append(np.column_stack([r_s, z_s]))

            # Arc length — include closing segment (append wraps back to
            # first point so diff gives N segments for N+1-style indexing).
            dR = np.diff(r_s, append=r_s[0])
            dZ = np.diff(z_s, append=z_s[0])
            dl_segs = np.sqrt(dR**2 + dZ**2)
            # Assign each point the average of its two adjacent segments
            # (trapezoidal quadrature weight for closed-curve integrals).
            dl = 0.5 * (dl_segs + np.roll(dl_segs, 1))

            # Sample fields on contour
            Br_s = Br_interp.ev(z_s, r_s)
            Bz_s = Bz_interp.ev(z_s, r_s)
            Jt_s = Jt_interp.ev(z_s, r_s)

            Bp2_s = Br_s**2 + Bz_s**2
            Bp_mod = np.sqrt(Bp2_s)

            # Signed Bp for orientation
            signBp = (
                cc["sigma_rhotp"] * cc["sigma_RpZ"]
                * np.sign((z_s - Z0) * Br_s - (r_s - R0) * Bz_s)
            )
            Bp_signed = signBp * Bp_mod

            Bt_s = F_k / r_s
            B2_s = Bp2_s + Bt_s**2

            # Flux expansion: dl / |Bp|
            # Floor |Bp| at a fraction of its surface-max to prevent
            # X-point-adjacent spikes from dominating the average.
            Bp_floor = max(1e-6 * np.max(Bp_mod), 1e-14)
            Bp_safe = np.maximum(Bp_mod, Bp_floor)
            fe_dl = dl / Bp_safe
            int_fe_dl = np.sum(fe_dl)

            def flx_avg(quantity):
                return np.sum(fe_dl * quantity) / int_fe_dl

            # Averages
            avg["R"][k] = flx_avg(r_s)
            avg["1/R"][k] = flx_avg(1.0 / r_s)
            avg["1/R**2"][k] = flx_avg(1.0 / r_s**2)
            avg["R**2"][k] = flx_avg(r_s**2)
            avg["Bp"][k] = flx_avg(Bp_signed)
            avg["Bp**2"][k] = flx_avg(Bp2_s)
            avg["Bt"][k] = flx_avg(Bt_s)
            avg["Bt**2"][k] = flx_avg(Bt_s**2)
            avg["Btot**2"][k] = flx_avg(B2_s)
            avg["Jt"][k] = flx_avg(Jt_s)
            avg["Jt/R"][k] = flx_avg(Jt_s / r_s)

            # Volume element
            avg["vp"][k] = (
                cc["sigma_rhotp"] * cc["sigma_Bp"]
                * np.sign(avg["Bp"][k])
                * int_fe_dl
                * (2.0 * np.pi) ** (1.0 - cc["exp_Bp"])
            )

            # Safety factor from averaging
            avg["q"][k] = (
                cc["sigma_rhotp"] * cc["sigma_Bp"]
                * avg["vp"][k] * F_k * avg["1/R**2"][k]
                / ((2 * np.pi) ** (2.0 - cc["exp_Bp"]))
            )

            # Enclosed current
            avg["ip"][k] = (
                cc["sigma_rhotp"]
                * np.sum(dl * Bp_signed)
                / (constants.mu_0)
            )

            # Geometry (computed from resampled contour)
            geo_k = _flux_geometry(r_s, z_s)
            for gk in geo_arrays:
                if gk in geo_k:
                    geo_arrays[gk][k] = geo_k[gk]

            # Accumulate for li calculation: sum(|Bp| * dl * 2*pi) * dpsi_k
            Bpl = np.sum(Bp_mod * dl * 2 * np.pi)
            Bp2_vol += Bpl * dpsi_arr[k]

        # Fix q on axis by quadratic extrapolation
        if psi_N_levels[0] == 0 and nc > 3:
            x = psi_N_levels[1:4]
            y = avg["q"][1:4]
            coeffs = np.polyfit(x, y, 2)
            avg["q"][0] = coeffs[2]  # p(0) = c
        elif psi_N_levels[0] == 0 and nc > 2:
            x = psi_N_levels[1:3]
            y = avg["q"][1:3]
            avg["q"][0] = y[0] - (y[1] - y[0]) / (x[1] - x[0]) * x[0]

        # Fix near-axis geometry following OMFIT's approach
        # (fluxSurface.py lines 1627-1641):
        #  - kapu/kapl: linear extrapolation from first 2 interior surfaces
        #  - kappa = 0.5*(kapu + kapl)
        #  - delta/delu/dell = 0 at the axis (circle has zero triangularity)
        if psi_N_levels[0] == 0 and nc > 2:
            x = psi_N_levels[1:]
            for gk in ["kapu", "kapl"]:
                y = geo_arrays[gk][1:]
                geo_arrays[gk][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]
            geo_arrays["kappa"][0] = 0.5 * (geo_arrays["kapu"][0] + geo_arrays["kapl"][0])
            for gk in ["delta", "delu", "dell"]:
                geo_arrays[gk][0] = 0.0

        # Analytic Grad-Shafranov current densities — more accurate than
        # numerically averaging curl(B)/μ₀, especially at the edge where
        # finite-difference noise from double-differentiating ψ is large.
        #
        # <Jt/R> = -(p' + FF'·<1/R²>/μ₀) × (2π)^exp_Bp
        #
        # The ``+ 0.0`` prevents IEEE-754 negative zero when p' = FF' = 0
        # at the separatrix (standard EFIT boundary condition).
        avg["Jt/R"] = (
            -cc["sigma_Bp"]
            * (avg["PPRIME"] + avg["FFPRIM"] * avg["1/R**2"] / constants.mu_0)
            * (2.0 * np.pi) ** cc["exp_Bp"]
        ) + 0.0
        # <Jt>  = -(p'·<R> + FF'·<1/R>/μ₀) × (2π)^exp_Bp   (direct average)
        # This differs from the OMFIT convention <Jt/R>/<1/R> by a Jensen
        # inequality term p'·[<R> - 1/<1/R>] which is positive when p' > 0,
        # making <Jt> larger in the H-mode pedestal region.
        avg["Jt_GS"] = (
            -cc["sigma_Bp"]
            * (avg["PPRIME"] * avg["R"] + avg["FFPRIM"] * avg["1/R"] / constants.mu_0)
            * (2.0 * np.pi) ** cc["exp_Bp"]
        ) + 0.0

        # Warn if the non-extrapolated portion of the Jt profile changes
        # sign.  Exclude the last surface (potentially extrapolated) when
        # extrapolate_edge is active so only genuinely computed values are
        # checked.
        jt_check = avg["Jt_GS"][:-1] if self._extrapolate_edge else avg["Jt_GS"]
        jt_nz = jt_check[jt_check != 0]
        if len(jt_nz) > 1 and np.any(jt_nz > 0) and np.any(jt_nz < 0):
            warnings.warn(
                "The direct Grad-Shafranov Jt profile changes sign across "
                "flux surfaces; this may indicate unusual equilibrium data.",
                stacklevel=2,
            )

        # Geometry: volume and cross-section area
        psi_arr = psi_N_levels * dpsi + self.psi_axis
        geo_arrays["vol"] = np.abs(
            integrate.cumulative_trapezoid(avg["vp"], psi_arr, initial=0)
        )
        geo_arrays["cxArea"] = np.abs(
            integrate.cumulative_trapezoid(
                avg["vp"] * avg["1/R"], psi_arr, initial=0
            ) / (2.0 * np.pi)
        )

        # Internal inductance
        ip = self.Ip
        r_0 = self.R_center if self.R_center else R0

        # Use g-file boundary (RBBBS/ZBBBS) for LCFS geometry in li
        # calculations.  The g-file boundary is EFIT's own LCFS and gives
        # circum/vol consistent with EFIT's own li values, whereas our
        # contourpy-traced contour at psi_N=1 can differ in resolution.
        Rb = self._raw["RBBBS"]
        Zb = self._raw["ZBBBS"]
        if len(Rb) > 3:
            dRb = np.diff(Rb, append=Rb[0])
            dZb = np.diff(Zb, append=Zb[0])
            circum = np.sum(np.sqrt(dRb**2 + dZb**2))
            # Toroidal volume via Pappus / Green's theorem:
            # V = pi * |oint R^2 dZ|
            vol = np.pi * np.abs(np.sum(Rb**2 * dZb))
            a_bdry = 0.5 * (np.max(Rb) - np.min(Rb))
            kappa_bdry = 0.5 * (np.max(Zb) - np.min(Zb)) / a_bdry if a_bdry > 0 else 1.0
        else:
            # Fallback to traced-contour geometry
            circum = geo_arrays["perimeter"][-1] if geo_arrays["perimeter"][-1] > 0 else 1.0
            vol = geo_arrays["vol"][-1] if geo_arrays["vol"][-1] > 0 else 1.0
            a_bdry = geo_arrays["a"][-1] if geo_arrays["a"][-1] > 0 else 1.0
            kappa_bdry = geo_arrays["kappa"][-1] if geo_arrays["kappa"][-1] > 0 else 1.0

        kappa_a = vol / (2.0 * np.pi * r_0 * np.pi * a_bdry * a_bdry) if a_bdry > 0 else 1.0
        correction_factor = (1 + kappa_bdry**2) / (2.0 * kappa_a) if kappa_a > 0 else 1.0

        if abs(ip) > 0:
            li_def = Bp2_vol / vol / constants.mu_0**2 / ip**2 * circum**2
        else:
            li_def = 0.0

        li_info = {
            "li_from_definition": li_def,
            # li(1)_EFIT = li_from_definition (OMFIT convention)
            "li(1)_EFIT": li_def,
            # li(1)_TLUCE applies a shape correction factor
            "li(1)_TLUCE": li_def / circum**2 * 2 * vol / r_0 * correction_factor if circum > 0 else 0.0,
            # li(1) defaults to EFIT definition (most widely used)
            "li(1)": li_def,
            "li(2)": li_def / circum**2 * 2 * vol / R0 if circum > 0 else 0.0,
            "li(3)": 2 * Bp2_vol / r_0 / ip**2 / constants.mu_0**2 if abs(ip) > 0 else 0.0,
        }

        # Betas (use boundary-derived vol, circum, a for consistency)
        betas = {}
        if np.any(self._raw["PRES"]):
            P_interp = interpolate.InterpolatedUnivariateSpline(psi_N_raw, self._raw["PRES"])
            P_on_levels = np.array([float(P_interp(pn)) for pn in psi_N_levels])
            Btvac = self.B_center * self.R_center / geo_arrays["R"][-1]
            P_vol = integrate.cumulative_trapezoid(avg["vp"] * P_on_levels, psi_arr, initial=0)
            if vol > 0 and abs(Btvac) > 0:
                betas["beta_t"] = abs(P_vol[-1] / (Btvac**2 / 2.0 / constants.mu_0) / vol)
                i_MA = ip / 1e6
                betas["beta_n"] = betas["beta_t"] / abs(i_MA / a_bdry / Btvac) * 100 if abs(i_MA * a_bdry * Btvac) > 0 else 0.0
            Bpave = ip * constants.mu_0 / circum if circum > 0 else 1.0
            if vol > 0 and abs(Bpave) > 0:
                betas["beta_p"] = abs(P_vol[-1] / (Bpave**2 / 2.0 / constants.mu_0) / vol)

        # -----------------------------------------------------------
        # Outboard midplane profiles (following OMFIT fluxSurface.py)
        # -----------------------------------------------------------
        # R_mid = geo_center + minor_radius  (outboard intersection)
        # Z_mid = Z0 (magnetic axis)
        # Br, Bz interpolated on the 2-D grid at (R_mid, Z_mid).
        # Bp  = signed sqrt(Br² + Bz²)
        # Bt  = F / R_mid
        # -----------------------------------------------------------
        mid = {}
        mid["R"] = geo_arrays["R"] + geo_arrays["a"]
        mid["Z"] = np.full(nc, Z0)

        mid["Br"] = Br_interp.ev(mid["Z"], mid["R"])
        mid["Bz"] = Bz_interp.ev(mid["Z"], mid["R"])

        signBp = (
            -cc["sigma_rhotp"] * cc["sigma_RpZ"]
            * np.sign(mid["Bz"])
        )
        mid["Bp"] = signBp * np.sqrt(mid["Br"] ** 2 + mid["Bz"] ** 2)
        mid["Bp"][0] = 0.0  # axis: Bp = 0 by definition

        mid["Bt"] = np.array([
            float(F_interp(psi_N_levels[k])) / mid["R"][k]
            for k in range(nc)
        ])
        mid["Btot"] = np.sqrt(mid["Bp"] ** 2 + mid["Bt"] ** 2)
        self._cache["midplane"] = mid

        # Store everything
        self._cache["avg"] = avg
        self._cache["geo"] = geo_arrays
        self._cache["contours"] = contour_data
        self._cache["li"] = li_info
        self._cache["betas"] = betas

    # --- Public analysis properties ---

    @property
    def j_tor_averaged(self):
        r"""Flux-surface-averaged toroidal current density [A/m²].

        Standard convention used by OMFIT, TRANSP, and GS solvers:

        .. math::
            j_\mathrm{tor} = \frac{\langle J_t / R \rangle}{\langle 1/R \rangle}

        where ``<Jt/R>`` is computed analytically from the GS equation
        using the smooth 1-D profiles p' and FF'.
        """
        self._trace_surfaces()
        avg = self._cache["avg"]
        return avg["Jt/R"] / avg["1/R"]

    @property
    def j_tor_averaged_direct(self):
        r"""Direct flux-surface average of Jt from GS [A/m²].

        .. math::
            \langle J_t \rangle = p' \langle R \rangle
            + FF' \langle 1/R \rangle / \mu_0

        This is the literal ``<Jt>`` and differs from
        :attr:`j_tor_averaged` by a Jensen-inequality term
        ``p' [<R> - 1/<1/R>]``, making it slightly larger when
        ``p' > 0``.  Not the standard convention but useful for
        understanding R-weighting effects.
        """
        self._trace_surfaces()
        return self._cache["avg"]["Jt_GS"]

    @property
    def j_tor_averaged_numerical(self):
        """Numerically flux-surface-averaged <Jt> [A/m²].

        Computed by averaging the 2-D ``Jt = curl(B)/μ₀`` field over
        each contour.  Less accurate than :attr:`j_tor_averaged` near
        the edge due to finite-difference noise, but useful as a
        cross-check.
        """
        self._trace_surfaces()
        return self._cache["avg"]["Jt"]

    @property
    def j_tor_over_R(self):
        """<Jt/R> from Grad-Shafranov equation [A/m³]."""
        self._trace_surfaces()
        return self._cache["avg"]["Jt/R"]

    @property
    def q_profile(self):
        """Safety factor from flux-surface averaging."""
        self._trace_surfaces()
        return self._cache["avg"]["q"]

    @property
    def li(self):
        """Internal inductance dict.

        Keys: li_from_definition, li(1) [=EFIT], li(1)_EFIT, li(1)_TLUCE,
              li(2), li(3).
        """
        self._trace_surfaces()
        return self._cache["li"]

    @property
    def geometry(self):
        """Per-surface geometric quantities dict.

        Keys: R, Z, a, kappa, kapu, kapl, delta, delu, dell,
              perimeter, surfArea, eps, vol, cxArea
        """
        self._trace_surfaces()
        return self._cache["geo"]

    @property
    def averages(self):
        """All flux-surface-averaged quantities dict.

        Keys: R, 1/R, 1/R**2, R**2, Bp, Bp**2, Bt, Bt**2, Btot**2,
              Jt, Jt/R, vp, q, ip, F, PPRIME, FFPRIM
        """
        self._trace_surfaces()
        return self._cache["avg"]

    @property
    def midplane(self):
        """Outboard midplane quantities on the psi_N grid.

        Computed following the OMFIT convention: the outboard midplane
        point of each flux surface is at ``R = geo_center + a``,
        ``Z = Z_axis``.  Magnetic field components are interpolated from
        the 2-D ``(R, Z)`` grid at that point.

        Keys: R, Z, Br, Bz, Bp, Bt, Btot
        """
        self._trace_surfaces()
        return self._cache["midplane"]

    @property
    def betas(self):
        """Plasma beta values: beta_t, beta_p, beta_n."""
        self._trace_surfaces()
        return self._cache["betas"]

    @property
    def contours(self):
        """List of (N, 2) contour arrays for each psi_N level."""
        self._trace_surfaces()
        return self._cache["contours"]

    # --- Integration methods ---

    def volume_integral(self, what):
        """Volume integral of a quantity on the psi_N grid.

        Parameters
        ----------
        what : array-like (nlevels,)
            Quantity to integrate, sampled at self.psi_N.

        Returns
        -------
        ndarray (nlevels,)
            Cumulative integral from core to each surface.
        """
        self._trace_surfaces()
        dpsi = self.psi_boundary - self.psi_axis
        psi_arr = self.psi_N * dpsi + self.psi_axis
        return integrate.cumulative_trapezoid(
            self._cache["avg"]["vp"] * np.asarray(what), psi_arr, initial=0
        )

    def surface_integral(self, what):
        """Cross-section integral of a quantity on the psi_N grid.

        Parameters
        ----------
        what : array-like (nlevels,)
            Quantity to integrate.

        Returns
        -------
        ndarray (nlevels,)
            Cumulative integral.
        """
        self._trace_surfaces()
        dpsi = self.psi_boundary - self.psi_axis
        psi_arr = self.psi_N * dpsi + self.psi_axis
        return integrate.cumulative_trapezoid(
            self._cache["avg"]["vp"] * self._cache["avg"]["1/R"] * np.asarray(what),
            psi_arr, initial=0,
        ) / (2.0 * np.pi)

    def flux_integral(self, psi_N_val, profile):
        """Total flux integral at a given psi_N value (scalar).

        Interpolates the cumulative volume integral of `profile` to
        the requested normalised psi.

        Parameters
        ----------
        psi_N_val : float
            Normalised poloidal flux location (0 = axis, 1 = boundary).
        profile : array-like (nlevels,)
            Profile to integrate.

        Returns
        -------
        float
            Value of the volume integral at psi_N_val.
        """
        cum = self.volume_integral(profile)
        return float(np.interp(psi_N_val, self.psi_N, cum))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def read_geqdsk(filename, cocos=1, nlevels=None, resample="theta",
                extrapolate_edge=True):
    """Read a GEQDSK file and return a GEQDSKEquilibrium object.

    Parameters
    ----------
    filename : str or path-like
        Path to the g-file.
    cocos : int
        COCOS convention index (default 1).
    nlevels : int
        Number of psi_N levels for flux-surface analysis.
    resample : str
        Contour resampling method for near-separatrix surfaces
        (``psi_N >= 0.99``).  ``"theta"`` (default) uses OMFIT-style
        angular resampling that preserves the X-point cusp.
        ``"arc_length"`` falls back to raw contourpy points (no
        resampling) for these surfaces.
    extrapolate_edge : bool
        If ``True`` (default), extrapolate p' and FF' at the separatrix
        when the g-file has them forced to zero.  Many solvers (EFIT,
        TokaMaker) clamp these to zero at ψ_N = 1 as a boundary
        condition, which makes ``j_tor_averaged_direct`` artificially
        zero there.  Set to ``False`` to use the raw g-file values.

    Returns
    -------
    GEQDSKEquilibrium
    """
    return GEQDSKEquilibrium(
        filename, cocos=cocos, nlevels=nlevels, resample=resample,
        extrapolate_edge=extrapolate_edge,
    )
