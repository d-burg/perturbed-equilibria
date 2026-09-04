"""
HDF5 archive helpers and eqdsk I/O utilities for perturbed equilibria.
"""

import contextlib
import os
import sys
import tempfile
import warnings

import h5py
import numpy as np

from .schema import write_profile

#: The single l_i estimator the whole reconstruction/draw chain runs on.
#:
#: ``"iter(li3)"`` == TokaMaker ``li_normalization='iter'``
#: == ``2 * int(Bp^2 dV) / ((mu0 * Ip)^2 * R_axis)``
#: == the g-file reader's ``li["li(2)"]`` key (whose name is historical and
#: misleading -- it is not Jackson's li(2)).
#:
#: This is the ONLY estimator bouquet and TokaMaker agree on (0.17% across the
#: DIII-D 169510 beta-scan g-files); the li(1)/EFIT pair differs by +3.3%
#: because TokaMaker projects the padded surface onto the true separatrix
#: before summing perimeter.  Targeting one and measuring the other is
#: issue #20.  Written into every archive's ``_baseline`` group as
#: ``l_i_scale`` so a reader never has to guess.
LI_SCALE = "iter(li3)"


class OpenLCFSContourWarning(UserWarning):
    """No closed contour was found at the LCFS flux level.

    Emitted by :func:`select_closed_lcfs` when every candidate segment is
    open, so the caller is falling back to the longest one and any
    boundary metric derived from it is unreliable.
    """


#: Endpoint gap, as a fraction of a segment's own bounding-box diagonal,
#: under which :func:`select_closed_lcfs` calls that segment closed.
#:
#: matplotlib repeats the first vertex exactly when a contour closes, so a
#: genuinely closed LCFS scores 0.0 and any positive threshold would do.
#: The bar is kept this tight so that an OPEN branch cannot sneak through:
#: to be misread as closed it would have to return to within 0.1% of its own
#: extent, at which point it is closed in every sense that matters here.
_LCFS_CLOSURE_RTOL = 1e-3


def select_closed_lcfs(segs, context=""):
    r"""Pick the longest **closed** contour segment from *segs*.

    On a diverted equilibrium the :math:`\psi = \psi_\mathrm{LCFS}` level set
    contains the open separatrix branch running down to the divertor as well
    as the closed LCFS.  The open branch spans the full vessel height, so it
    frequently carries *more* points than the closed boundary -- and a plain
    ``max(segs, key=len)`` then silently returns the wrong curve.  Measured on
    a diverted lower-single-null case, that mis-selection reported a boundary
    RMS of 891.86 mm (open branch, n=1070) where the true closed-LCFS value is
    2.06 mm (n=969), which flipped the acceptance verdict PASS -> CHECK.  A
    second case selected the closed branch only because it happened to be
    longer, so the defect is general rather than case-specific (issue #33).

    Closedness is tested directly -- first vertex coincident with last, within
    :data:`_LCFS_CLOSURE_RTOL` of the segment's own bounding-box diagonal --
    rather than inferred from length.

    Parameters
    ----------
    segs : sequence of ndarray, shape (N, 2)
        Candidate contour segments, as returned by matplotlib ``allsegs``.
    context : str, optional
        Caller label, used only in the fallback warning message.

    Returns
    -------
    ndarray, shape (N, 2) or None
        The longest closed segment; the longest segment overall (with an
        :class:`OpenLCFSContourWarning`) if none closes; ``None`` when no
        candidate survives -- either *segs* was empty, or every entry had
        4 or fewer vertices and was discarded as degenerate.  Callers must
        treat ``None`` as "no usable contour" and fall back accordingly.
    """
    segs = [np.asarray(s, dtype=float) for s in segs if len(s) > 4]
    if not segs:
        return None

    closed = []
    for s in segs:
        span = np.ptp(s, axis=0)
        diag = float(np.hypot(*span))
        gap = float(np.hypot(*(s[0] - s[-1])))
        if diag > 0.0 and gap <= _LCFS_CLOSURE_RTOL * diag:
            closed.append(s)

    if closed:
        return max(closed, key=len)

    warnings.warn(
        f"no CLOSED contour at the LCFS flux level"
        f"{' in ' + context if context else ''} -- falling back to the "
        f"longest of {len(segs)} open segment(s).  Any boundary RMS/max "
        f"derived from it is unreliable (issue #33).",
        OpenLCFSContourWarning,
        stacklevel=2,
    )
    return max(segs, key=len)


class DerivativeSanityWarning(UserWarning):
    """Data-sanity warning category for :func:`pchip_derivative`.

    Raised (as a warning, or as ``ValueError`` when ``strict=True``) for
    conditions that are not fatal to computing a derivative but are strong
    signals of upstream data problems: a corrupted/duplicated ``psi_N``
    grid, an unsorted grid, or a spike in the fitted derivative that looks
    like an axis-mapping artifact rather than real pedestal/edge physics.
    """


@contextlib.contextmanager
def capture_native_output(enabled=True):
    """Capture *all* stdout/stderr — both Python-level and OS-level.

    Two layers are needed to fully silence a TokaMaker solve:

    * :func:`contextlib.redirect_stdout`/``redirect_stderr`` catch Python
      ``print()`` (the homotopy / boundary / li-match diagnostics), which under
      a Jupyter kernel go straight to the cell via ``sys.stdout``, bypassing the
      file descriptors.
    * an ``os.dup2`` file-descriptor redirect catches output from the compiled
      extension (the Fortran/C solver: ``DLSODE``, ``gs_get_qprof`` warnings)
      that bypasses Python's ``sys.stdout``.

    Yields a dict whose ``"text"`` key holds the combined captured output once
    the block exits (also populated if the block raises). ``enabled=False`` is a
    no-op pass-through, so callers can gate on a ``verbose`` flag.
    """
    import io

    holder = {"text": ""}
    if not enabled:
        yield holder
        return
    sys.stdout.flush()
    sys.stderr.flush()
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    py_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(py_buf), contextlib.redirect_stderr(py_buf):
            yield holder
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tmp.seek(0)
        holder["text"] = py_buf.getvalue() + tmp.read().decode("utf-8", "replace")
        tmp.close()


# ====================================================================
#  Internal helpers
# ====================================================================

def safe_trace_surf(mygs, psi):
    r'''Snapshot/restore-wrapped trace_surf.

    The Fortran-side `trace_surf` has been observed to perturb subsequent
    `get_stats` / boundary measurements via mutation of state hanging off
    the `gs_equil` struct (mesh-cell caches, tracer step state, etc.).
    Pre-OFT-PR-248 this was hard to undo cleanly; with PR #248 we now have
    `mygs.copy_eq()` / `mygs.replace_eq()` which atomically swap the
    `gs_equil` pointer.  This wrapper snapshots the equilibrium before
    the trace, copies the returned LCFS into an owned numpy array, then
    restores the original equilibrium.

    Whatever state lives outside `gs_equil` (eg. module-level
    `active_tracer` in `tracing_2d`) is *not* restored — but that state
    is reset at the start of each `trace_surf` call anyway, so the only
    risk surface is mid-trace interaction with concurrent
    `get_stats`-like calls, which bouquet does not do.

    Parameters
    ----------
    mygs : OpenFUSIONToolkit.TokaMaker.TokaMaker
        Active TokaMaker instance.  Requires PR #248+ (`copy_eq` /
        `replace_eq` available).
    psi : float
        Normalized psi value to trace (eg. ``1.0 - psi_pad``).

    Returns
    -------
    numpy.ndarray or None
        ``(N, 2)`` array of (R, Z) points along the traced surface,
        owned by the caller (a copy, decoupled from any internal
        TokaMaker buffers).  Returns ``None`` if the trace fails.
    '''
    if not hasattr(mygs, 'copy_eq') or not hasattr(mygs, 'replace_eq'):
        # legacy OFT build: fall through to bare trace_surf (no protection)
        result = mygs.trace_surf(psi)
        return None if result is None else np.asarray(result).copy()
    saved = mygs.copy_eq()
    try:
        result = mygs.trace_surf(psi)
        if result is not None:
            result = np.asarray(result).copy()
        return result
    finally:
        mygs.replace_eq(source_eq=saved)


def safe_save_eqdsk(mygs, filename, **kwargs):
    r'''Snapshot/restore-wrapped save_eqdsk.

    `mygs.save_eqdsk` internally re-runs the q-profile tracer (which
    sets `active_tracer` state and may mutate cached `<R>` / `<1/R>`
    geometry on the `gs_equil` struct) and has been empirically
    observed to shift `mygs.get_globals()[0]` (Ip integral) by ~0.5-
    0.8% when called against a converged equilibrium.  This wrapper
    snapshots the equilibrium via `mygs.copy_eq()` before the save,
    then restores via `mygs.replace_eq()` after -- so the .geqdsk
    file is written from the unmodified state AND downstream
    diagnostics see the same state recon converged to.

    On legacy OFT builds without `copy_eq` / `replace_eq` (pre-PR
    #248), falls through to a bare `mygs.save_eqdsk(...)` with no
    protection.

    Parameters
    ----------
    mygs : OpenFUSIONToolkit.TokaMaker.TokaMaker
        Active TokaMaker instance.  Requires PR #248+ for the
        protected snapshot/restore path.
    filename : str
        Same as `mygs.save_eqdsk` filename arg.
    **kwargs
        Passed through to `mygs.save_eqdsk(...)`.
    '''
    if not hasattr(mygs, 'copy_eq') or not hasattr(mygs, 'replace_eq'):
        return mygs.save_eqdsk(filename, **kwargs)
    saved = mygs.copy_eq()
    try:
        return mygs.save_eqdsk(filename, **kwargs)
    finally:
        mygs.replace_eq(source_eq=saved)


def pchip_derivative(x, y, x_eval=None, strict=False):
    r'''Analytic derivative dy/dx via PCHIP on the native (x, y) grid.

    Replaces the ``np.gradient(y) / np.gradient(x)`` central-difference
    pattern used throughout bouquet to build P'(psi_N) for TokaMaker's
    ``pp_prof``. A :class:`scipy.interpolate.PchipInterpolator` is fit to
    the *native* (unresampled) grid and differentiated analytically, which:

    * avoids the smearing a resample-then-``np.gradient`` pipeline
      introduces at sharp features (e.g. an H-mode pedestal), and
    * is shape-preserving / no-overshoot by construction -- monotone input
      data can never yield a derivative of the "wrong" sign, unlike an
      interpolating cubic spline (:class:`scipy.interpolate.CubicSpline`),
      which can ring on noisy data. On smooth (e.g. GPR) profiles PCHIP and
      CubicSpline derivatives are numerically indistinguishable, so this
      guranteed-safe behavior at sharp/noisy features is a pure win.

    Two tiers of data-sanity checks guard against silently producing a
    plausible-looking but wrong P' from corrupted input:

    **Tier 1 -- hard errors** (always ``raise ValueError``, regardless of
    *strict*):

    1. Non-finite (``NaN``/``inf``) values in *x* or *y*.
    2. Fewer than 2 unique *x* points after de-duplication (a degenerate
       grid cannot support a derivative -- this must not silently return
       zeros).
    3. Non-finite values in the computed derivative.
    4. Shape-guarantee tripwire: PCHIP's monotonicity invariant guarantees
       that monotone input data can never yield a derivative of the wrong
       sign. If the de-duplicated, sorted *y* is monotone (within
       ``rtol=1e-12*max|y|``) but the fitted derivative disagrees in sign
       beyond ``1e-12*max|secant slope|``, that invariant has been broken
       -- almost certainly by scrambled ``(x, y)`` pairing rather than a
       genuine numerical edge case -- so this raises rather than warns.

    **Tier 2 -- aggressive warnings** (:class:`DerivativeSanityWarning` by
    default; promoted to ``ValueError`` when ``strict=True``):

    5. Duplicate *x* points were dropped (count + first few duplicated
       locations reported) -- a duplicated ``psi_N`` grid is a suspected
       upstream cause of stepwise ``j_BS`` artifacts, so this is surfaced
       loudly rather than silently fixed.
    6. Input *x* was not sorted (ascending sort was applied).
    7. More than 5% of points were dropped as duplicates ("grid likely
       corrupted").
    8. Spike detector: the fitted derivative's peak magnitude exceeds
       ``50 *`` the median secant-slope magnitude (guarded on
       ``median > 0``) -- flagged as a possible unphysical feature (e.g.
       an axis-mapping artifact) rather than real edge/pedestal physics,
       with the spike's *x* location and the ratio reported.

    Parameters
    ----------
    x : array_like
        1-D abscissa (e.g. ``psi_N``). Need not be strictly increasing or
        unique on input -- see the duplicate-handling note below.
    y : array_like
        1-D ordinate (e.g. pressure), same length as *x*.
    x_eval : array_like or None, optional
        Points at which to evaluate the derivative. Defaults to *x* itself
        (i.e. the derivative is returned on the native grid).
    strict : bool, optional
        When ``True``, every Tier-2 condition that would normally emit a
        :class:`DerivativeSanityWarning` instead raises ``ValueError``.
        Default ``False`` (warn only); bouquet's call sites keep the
        default so a merely-suspicious profile doesn't abort a solve.

    Returns
    -------
    numpy.ndarray
        ``dy/dx`` evaluated at *x_eval* (or at the original, full-length
        *x* when *x_eval* is ``None`` -- so the returned array matches the
        input grid's shape even when duplicate points were dropped for
        fitting; a repeated abscissa simply gets the same derivative value
        at each of its occurrences).

    Notes
    -----
    ``PchipInterpolator`` requires strictly increasing abscissas.
    Duplicated or non-monotonic ``psi_N`` grid points are a suspected
    cause of stepwise ``j_BS`` artifacts downstream, so this guard is
    load-bearing, not cosmetic: repeated ``x`` values are collapsed via
    ``np.unique`` (keeping the first occurrence of each distinct value,
    then sorting), silently making the interpolant well-posed rather than
    raising on real-world profile data that occasionally carries a
    repeated grid point -- but see Tier 2 items 5-7 above: "silently"
    well-posed does not mean "silently" as far as the user is concerned.
    '''
    from scipy.interpolate import PchipInterpolator

    def _flag(msg):
        """Tier-2 report: warn (default) or raise (``strict=True``).

        ``stacklevel=3`` walks past this closure and ``pchip_derivative``
        itself so the warning is attributed to the code that called
        ``pchip_derivative``, not to a line inside this helper.
        """
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, DerivativeSanityWarning, stacklevel=3)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ---- Tier 1.1: non-finite inputs -----------------------------------
    if not np.all(np.isfinite(x)):
        n_bad = int(np.count_nonzero(~np.isfinite(x)))
        raise ValueError(
            f"pchip_derivative: x contains {n_bad} non-finite value(s) "
            "(NaN/inf) -- cannot fit a PCHIP interpolant.")
    if not np.all(np.isfinite(y)):
        n_bad = int(np.count_nonzero(~np.isfinite(y)))
        raise ValueError(
            f"pchip_derivative: y contains {n_bad} non-finite value(s) "
            "(NaN/inf) -- cannot fit a PCHIP interpolant.")

    # ---- Tier 2.6: unsorted input ---------------------------------------
    if x.size >= 2 and np.any(np.diff(x) < 0):
        _flag(
            "pchip_derivative: input x is not sorted ascending; an "
            "ascending sort was applied before fitting. If this x grid "
            "(e.g. psi_N) is expected to be monotonic, check upstream "
            "for a scrambled profile.")

    # ---- de-duplicate + sort -------------------------------------------
    x_unique, idx, counts = np.unique(x, return_index=True, return_counts=True)
    y_unique = y[idx]
    n_total = x.size
    n_dropped = n_total - x_unique.size

    # ---- Tier 2.5 / 2.7: duplicate x points dropped ---------------------
    if n_dropped > 0:
        dup_locs = x_unique[counts > 1]
        loc_str = ", ".join(f"{v:.6g}" for v in dup_locs[:5])
        if dup_locs.size > 5:
            loc_str += f", ... ({dup_locs.size} total distinct dup locations)"
        frac_dropped = n_dropped / n_total
        if frac_dropped > 0.05:
            _flag(
                f"pchip_derivative: {n_dropped}/{n_total} points "
                f"({100*frac_dropped:.1f}%) dropped as duplicate x values "
                f"(at x = {loc_str}) -- grid likely corrupted. A duplicated "
                "psi_N grid is a suspected cause of stepwise j_BS "
                "artifacts; check the upstream profile/grid construction.")
        else:
            _flag(
                f"pchip_derivative: dropped {n_dropped} duplicate x "
                f"point(s) (at x = {loc_str}) before fitting. A duplicated "
                "psi_N grid is a suspected cause of stepwise j_BS "
                "artifacts; check the upstream profile/grid construction.")

    # ---- Tier 1.2: degenerate grid ---------------------------------------
    if x_unique.size < 2:
        raise ValueError(
            f"pchip_derivative: only {x_unique.size} unique x point(s) "
            "after de-duplication -- at least 2 are required to fit a "
            "derivative; refusing to silently return zeros.")

    interp = PchipInterpolator(x_unique, y_unique)
    deriv_unique = interp.derivative()(x_unique)   # for the sanity checks below
    x_eval = x if x_eval is None else np.asarray(x_eval, dtype=float)
    result = interp.derivative()(x_eval)

    # ---- Tier 1.3: non-finite output -------------------------------------
    if not np.all(np.isfinite(result)):
        n_bad = int(np.count_nonzero(~np.isfinite(result)))
        raise ValueError(
            f"pchip_derivative: computed derivative contains {n_bad} "
            "non-finite value(s) -- check the input profile for "
            "near-degenerate spacing or extreme dynamic range.")

    # ---- secant slopes on the native (unique, sorted) grid ---------------
    dx = np.diff(x_unique)
    dy = np.diff(y_unique)
    secants = dy / dx
    abs_secants = np.abs(secants)

    # ---- Tier 1.4: shape-guarantee tripwire ------------------------------
    max_y = np.max(np.abs(y_unique)) if y_unique.size else 0.0
    y_tol = 1e-12 * max_y
    max_secant = np.max(abs_secants) if abs_secants.size else 0.0
    slope_tol = 1e-12 * max_secant
    non_increasing = np.all(dy <= y_tol)
    non_decreasing = np.all(dy >= -y_tol)
    if non_increasing and not non_decreasing:
        bad = deriv_unique > slope_tol
        if np.any(bad):
            raise ValueError(
                "pchip_derivative: internal invariant violated -- y is "
                "monotone non-increasing but the PCHIP derivative has "
                f"{int(np.count_nonzero(bad))} positive-sign entrie(s) "
                "beyond tolerance. PCHIP's shape-preservation guarantee "
                "should make this impossible; this likely indicates "
                "scrambled (x, y) pairing rather than a numerical edge "
                "case.")
    elif non_decreasing and not non_increasing:
        bad = deriv_unique < -slope_tol
        if np.any(bad):
            raise ValueError(
                "pchip_derivative: internal invariant violated -- y is "
                "monotone non-decreasing but the PCHIP derivative has "
                f"{int(np.count_nonzero(bad))} negative-sign entrie(s) "
                "beyond tolerance. PCHIP's shape-preservation guarantee "
                "should make this impossible; this likely indicates "
                "scrambled (x, y) pairing rather than a numerical edge "
                "case.")

    # ---- Tier 2.8: spike detector -----------------------------------------
    median_secant = np.median(abs_secants) if abs_secants.size else 0.0
    if median_secant > 0:
        peak_idx = int(np.argmax(np.abs(deriv_unique)))
        peak_val = float(np.abs(deriv_unique[peak_idx]))
        ratio = peak_val / median_secant
        if ratio > 50.0:
            _flag(
                f"pchip_derivative: derivative peak |dy/dx|={peak_val:.4g} "
                f"at x={x_unique[peak_idx]:.6g} is {ratio:.1f}x the median "
                f"secant-slope magnitude ({median_secant:.4g}) -- possible "
                "unphysical feature in input data (e.g. an axis mapping "
                "artifact) rather than real profile physics.")

    return result


# =====================================================================
#  Flux-surface-averaged plasma-current integral
# =====================================================================
#
# ``TokaMaker.compute_flux_integral`` is NOT ``int_plasma f dA``.  Measured on
# the synthetic D3D-like example (see ``tests/test_fsa_current_integral.py``):
#
#   * it integrates over the whole ``reg == 1`` (limiter) region, and the
#     flux-function interpolator returns the profile's EDGE value everywhere
#     outside the LCFS (``gs_prof_interp_apply`` CASE(4) returns 0 -- the LCFS
#     end of the internal psi coordinate -- off the plasma, and ``gs_flux_int``
#     then evaluates the profile there).  ``compute_flux_integral(1.0)`` is
#     therefore 2.83853 m^2, the LIMITER-region area, against a true plasma
#     cross-section of 1.79005 m^2;
#   * so for a profile with a finite edge value the excess area is charged at
#     ``f(psi_N=1)``.  On the archived total that is
#     ``1.36e5 A/m^2 * 1.05 m^2 = 1.43e5 A``, i.e. +11.9 % of I_p -- almost the
#     whole of the +12.9 % "representation bias" 7dc254b calibrated away.
#
# The measure below never uses the mesh integral.  It is the textbook
# axisymmetric current integral,
#
#     I_p = int j_phi dA = int dpsi (V'/2pi) <j_phi/R>,
#
# with ``V' = dV/dpsi`` and the flux-surface averages taken from ``get_q``.
#
# CONVENTION.  Two readings of a bouquet current array are in play and they
# differ by O(1 %) on a DIII-D-like case:
#
#   ``"fsa"``           J = <j_phi/R>/<1/R>, the FSA toroidal density
#                       => I_p = int dpsi (V'/2pi) <1/R> J
#   ``"jphi-linterp"``  J = <R> P' + <1/R> FF'/mu0, TokaMaker's own
#                       ``jphi-linterp`` profile variable (``jphi_update`` in
#                       ``grad_shaf_prof_phys.F90``), i.e. j_phi evaluated at
#                       the surface's mean major radius
#                       => FF'/mu0 = (J - <R> P')/<1/R> and hence
#                          I_p = int dpsi (V'/2pi) [ J <1/R^2>/<1/R>
#                                                    + P' (1 - <R><1/R^2>/<1/R>) ]
#
# bouquet's arrays are ``jphi-linterp`` values -- that is the variable they are
# handed to ``set_profiles`` as, and the variable the reconstruction fits.  On
# the D3D-like anchor the ``"jphi-linterp"`` reading recovers the true I_p of
# the archived total to +0.072 %, against +0.927 % for ``"fsa"``, and it agrees
# to 0.011 % with the solver's own internal renormalisation factor (measured
# independently as ``<w J_eq>/<w J_archived>``).  Both readings integrate the
# equilibrium's OWN profile to its true I_p to better than 0.01 %, which is the
# validation that the V'/<1/R> plumbing and the psi_N -> psi Jacobian are right.

_FSA_PSI_PAD = 1.0e-3
_FSA_MU0 = 4.0e-7 * np.pi


def fsa_current_geometry(eq, psi_N, psi_pad=_FSA_PSI_PAD, want_pprime=True):
    r"""Per-surface FSA geometry for :func:`Ip_fsa_integral`, from ``get_q``.

    *eq* is anything exposing TokaMaker's ``get_q`` / ``get_profiles`` /
    ``psi_bounds`` -- a live ``TokaMaker`` or a ``copy_eq()`` snapshot
    (``TokaMaker_equilibrium``); both were verified to return bit-identical
    ``ravgs`` for the same state, and evaluating them on a snapshot does not
    perturb the live solver.

    Two traps this function exists to close:

    * **exact endpoints silently collapse ``get_q``.**  ``get_q(psi=...)`` with
      ``psi_N`` containing 0.0 (the magnetic axis) returns the AXIS values on
      *every* surface -- ``<R>`` constant to 2e-15, ``dV/dPsi`` constant --
      with no error and no warning; the surface tracer starts from the axis and
      never leaves it.  The sampling grid is therefore clipped to
      ``[psi_pad, 1 - psi_pad]`` and the collapse is asserted against.
    * **``dV/dPsi`` is per DIMENSIONAL psi.**  ``int dV/dPsi dpsi`` recovers
      ``get_stats()['vol']`` to -0.25 % (edge/axis truncation) while the
      ``dpsi_N`` reading is out by +291 %.  The Jacobian is
      ``|psi_bounds[1] - psi_bounds[0]|``, applied here once.

    Returns a dict with ``psi_N``, ``R_avg``, ``inv_R``, ``inv_R2`` (``None``
    on OFT builds whose ``ravgs`` is the legacy 3-entry positional array),
    ``dV_dpsi`` (magnitude), ``dpsi_dpsiN``, ``pprime`` (``None`` if not
    requested) and ``dA_dpsiN = (V'/2pi) <1/R> |dpsi/dpsi_N|``.
    """
    from .physics import q_ravg

    psi_N = np.asarray(psi_N, dtype=float)
    psi_q = np.ascontiguousarray(
        np.clip(psi_N, float(psi_pad), 1.0 - float(psi_pad)), dtype=float)
    ravgs = eq.get_q(psi=psi_q)[2]
    R_avg = np.asarray(q_ravg(ravgs, "<R>"), dtype=float)
    inv_R = np.asarray(q_ravg(ravgs, "<1/R>"), dtype=float)
    dV_dpsi = np.abs(np.asarray(q_ravg(ravgs, "dV/dPsi"), dtype=float))
    inv_R2 = None
    if isinstance(ravgs, dict) and "<1/R^2>" in ravgs:
        inv_R2 = np.asarray(ravgs["<1/R^2>"], dtype=float)

    if R_avg.size > 1 and np.ptp(R_avg) <= 1.0e-9 * float(np.mean(R_avg)):
        raise RuntimeError(
            "fsa_current_geometry: get_q returned a CONSTANT <R> across all "
            f"{R_avg.size} surfaces ({float(R_avg[0]):.6f} m) -- the surface "
            "tracer collapsed onto the magnetic axis.  This is the silent "
            "failure mode triggered by sampling psi_N = 0 exactly; raise "
            "psi_pad.")
    if not (np.all(np.isfinite(R_avg)) and np.all(np.isfinite(inv_R))
            and np.all(np.isfinite(dV_dpsi))):
        raise RuntimeError("fsa_current_geometry: get_q returned non-finite "
                           "flux-surface averages")

    bounds = np.asarray(eq.psi_bounds, dtype=float)
    dpsi_dpsiN = abs(float(bounds[1]) - float(bounds[0]))
    if not np.isfinite(dpsi_dpsiN) or dpsi_dpsiN <= 0.0:
        raise RuntimeError(f"fsa_current_geometry: bad psi_bounds {bounds}")

    if inv_R2 is not None and not np.all(np.isfinite(inv_R2)):
        raise RuntimeError("fsa_current_geometry: get_q returned non-finite "
                           "<1/R^2> averages")

    pprime = None
    if want_pprime:
        prof = eq.get_profiles(psi=psi_q.copy())
        pprime = np.asarray(prof[4], dtype=float)
        if not np.all(np.isfinite(pprime)):
            raise RuntimeError("fsa_current_geometry: get_profiles returned "
                               "non-finite P' -- a NaN here would poison the "
                               "affine c term and pass every downstream "
                               "threshold gate silently")

    return {
        "psi_N": psi_N,
        "psi_q": psi_q,
        "R_avg": R_avg,
        "inv_R": inv_R,
        "inv_R2": inv_R2,
        "dV_dpsi": dV_dpsi,
        "dpsi_dpsiN": dpsi_dpsiN,
        "pprime": pprime,
        "dA_dpsiN": dV_dpsi / (2.0 * np.pi) * inv_R * dpsi_dpsiN,
    }


def Ip_fsa_weights(geom, convention="jphi-linterp", pprime_sign=1.0):
    r"""``(w, c)`` such that ``I_p[J] = trapezoid(w * J, psi_N) + c``.

    The measure is affine, not linear: in the ``jphi-linterp`` convention the
    ``P'`` term of the Grad-Shafranov source is independent of *J* and lands in
    the constant *c* (it is -3.3 % of I_p on the D3D-like anchor -- large
    enough that dropping it costs +3.4 %).  In the ``fsa`` convention ``c`` is
    exactly 0.

    ``pprime_sign`` flips ``get_profiles``' ``P'`` if the case's flux/current
    sign convention needs it; :class:`~bouquet.TokaMaker_interface._AnchorIpRenorm`
    determines it from the anchor rather than assuming.
    """
    g = geom["dV_dpsi"] / (2.0 * np.pi) * geom["dpsi_dpsiN"]
    if convention == "fsa":
        return g * geom["inv_R"], 0.0
    if convention != "jphi-linterp":
        raise ValueError(f"unknown convention {convention!r} "
                         "(want 'jphi-linterp' or 'fsa')")
    inv_R2 = geom["inv_R2"]
    if inv_R2 is None:
        raise ValueError(
            "the 'jphi-linterp' measure needs <1/R^2>, which this OFT build's "
            "get_q does not return (legacy positional ravgs).  Use "
            "convention='fsa' on that build.")
    pprime = geom["pprime"]
    if pprime is None:
        raise ValueError("the 'jphi-linterp' measure needs P'; rebuild the "
                         "geometry with want_pprime=True")
    ratio = inv_R2 / geom["inv_R"]
    w = g * ratio
    c = float(_trapezoid(g * float(pprime_sign) * pprime
                         * (1.0 - geom["R_avg"] * ratio), geom["psi_N"]))
    return w, c


def _trapezoid(y, x):
    from scipy.integrate import trapezoid
    return float(trapezoid(np.asarray(y, dtype=float),
                           np.asarray(x, dtype=float)))


def close_ip(channel, Ip_target_signed, c_affine, ip_ind, ip_bs, ip_fix,
             scale_bounds=(0.2, 5.0)):
    """Channel scales closing the affine Ip measure on a hybrid baseline.

    Solves ``s_ohm*ip_ind + s_bs*ip_bs + ip_fix + c = Ip`` for the one free
    scale, where ``ip_*`` are the LINEAR parts (``trapezoid(w * j)``) of each
    component under :func:`Ip_fsa_weights` and ``c`` is the affine P' term
    carried exactly once.  ``channel`` picks which component absorbs the
    deficit: ``"bootstrap"`` keeps ``s_ohm = 1`` and rescales the bootstrap;
    ``"ohmic"`` keeps ``s_bs = 1`` and rescales the inductive.  Returns
    ``(ohm_scale, bs_scale)``.

    Refuses (``RuntimeError``) when the rescaled component's linear part is
    ~0 relative to the target (the closure would be a division by noise) or
    when the resulting scale falls outside ``scale_bounds`` -- the hybrid
    components then simply do not add up to Ip and hiding that behind a
    rescale would be a lie.  Unknown channels raise ``ValueError``.
    """
    Ip_t = abs(float(Ip_target_signed))
    deficit = float(Ip_target_signed) - float(c_affine) - float(ip_fix)
    if channel == "bootstrap":
        if abs(ip_bs) < 1e-6 * Ip_t:
            raise RuntimeError("close_ip/bootstrap channel: j_BS integrates "
                               "to ~0; cannot close Ip on it")
        scales = (1.0, (deficit - float(ip_ind)) / float(ip_bs))
    elif channel == "ohmic":
        if abs(ip_ind) < 1e-6 * Ip_t:
            raise RuntimeError("close_ip/ohmic channel: j_inductive "
                               "integrates to ~0; cannot close Ip on it")
        scales = ((deficit - float(ip_bs)) / float(ip_ind), 1.0)
    else:
        raise ValueError(f"unknown closure_channel {channel!r} "
                         "(expected 'ohmic' or 'bootstrap')")
    lo, hi = scale_bounds
    for name, s in zip(("ohm_scale", "bs_scale"), scales):
        if not (lo < s < hi):
            raise RuntimeError(
                f"close_ip/{channel} channel: {name} {s:.3f} is outside "
                f"[{lo:g}, {hi:g}] -- the hybrid components do not add up "
                "to Ip; refusing to hide that behind a rescale")
    return scales


def close_ip_q0(Ip_target_signed, c_affine, ip_ind, ip_bs, ip_fix,
                j_ind0, j_bs0, j_fix0, j_ref0,
                scale_bounds=(0.2, 5.0), det_rtol=1e-6):
    r"""Both hybrid scales from Ip **and** an on-axis-current (q0) constraint.

    The ``"sawtooth_bootstrap"`` channel's predictor.  Two unknowns
    (``s_ohm``, ``s_bs``), two targets, one 2x2 linear system and **no GS
    solve**:

    .. code-block:: text

        s_ohm*j_ind0 + s_bs*j_bs0 = j_ref0 - j_fix0      (axis current -> q0)
        s_ohm*ip_ind + s_bs*ip_bs = Ip_signed - c - ip_fix   (exact Ip, affine)

    The first row is the q0 constraint *linearised*: at frozen anchor geometry
    (kappa0, B0, R0 fixed with the snapshot) the on-axis safety factor is
    ``q0 = 2 B0 (1 + kappa0^2) / (2 kappa0 mu0 R0 j0)``, i.e. ``q0 ~ 1/j0``, so
    matching q0 to the reference equilibrium is -- to first order -- matching
    its on-axis toroidal current density ``j_ref0``.  All the ``j*0`` are the
    profile values at the SAME psi_N sample, the psi_pad-clipped axis of
    :func:`fsa_current_geometry` (never psi_N = 0 exactly; see that docstring's
    collapse trap).  The second row is untouched from :func:`close_ip`: the
    LINEAR parts of the affine FSA measure with the P' term ``c`` carried once,
    so Ip stays exact by construction wherever this succeeds.

    Degenerate by design where the recomputed bootstrap has no core content:
    ``j_bs0 ~ 0`` makes row 1 pin ``s_ohm = (j_ref0 - j_fix0)/j_ind0`` (~1 when
    the source's own axis current is reproduced) and row 2 hand the whole Ip
    deficit to ``s_bs`` -- the channel reduces to ``"bootstrap"`` at zero cost.
    That is not the singular case; the singular case is the two rows becoming
    proportional, which is refused against a RELATIVE determinant floor
    (``det_rtol``) rather than an absolute one, because ``ip_*`` are amps and
    ``j*0`` are A/m^2 and no absolute epsilon is meaningful across both.

    Raises ``RuntimeError`` on a singular system, on non-finite inputs, or when
    either scale leaves ``scale_bounds`` -- the three sources then simply do not
    admit a common (Ip, q0) solution, which is a finding to report, not
    something to hide behind a rescale.  Returns ``(ohm_scale, bs_scale)``.
    """
    vals = (Ip_target_signed, c_affine, ip_ind, ip_bs, ip_fix,
            j_ind0, j_bs0, j_fix0, j_ref0)
    if not all(np.isfinite(float(v)) for v in vals):
        raise RuntimeError("close_ip_q0: non-finite input "
                           f"{tuple(float(v) for v in vals)}")
    b_axis = float(j_ref0) - float(j_fix0)
    b_ip = float(Ip_target_signed) - float(c_affine) - float(ip_fix)
    det = float(j_ind0) * float(ip_bs) - float(j_bs0) * float(ip_ind)
    floor = det_rtol * max(abs(float(j_ind0)), abs(float(j_bs0))) \
        * max(abs(float(ip_ind)), abs(float(ip_bs)))
    if abs(det) <= floor:
        raise RuntimeError(
            f"close_ip_q0: singular 2x2 (det {det:.4e} <= relative floor "
            f"{floor:.4e}) -- the inductive and bootstrap components are "
            "proportional in (axis current, Ip); Ip and q0 cannot both be "
            "imposed on this split")
    ohm_scale = (b_axis * float(ip_bs) - b_ip * float(j_bs0)) / det
    bs_scale = (float(j_ind0) * b_ip - float(ip_ind) * b_axis) / det
    lo, hi = scale_bounds
    for name, s in (("ohm_scale", ohm_scale), ("bs_scale", bs_scale)):
        if not (lo < s < hi):
            raise RuntimeError(
                f"close_ip_q0: {name} {s:.3f} is outside [{lo:g}, {hi:g}] -- "
                "no (Ip, q0)-consistent split exists within the scale bounds; "
                "refusing to hide that behind a rescale")
    return ohm_scale, bs_scale


def Ip_fsa_integral(eq, psi_N, j_profile, convention="jphi-linterp",
                    psi_pad=_FSA_PSI_PAD, pprime_sign=1.0, geom=None):
    r"""Plasma current [A] carried by a bouquet current profile.

    The physically-exact axisymmetric current integral, evaluated on *eq*'s
    flux-surface geometry -- see the module comment above for the two profile
    conventions and the measured accuracy of each.  Pass a *geom* from
    :func:`fsa_current_geometry` to reuse a frozen snapshot's geometry (and to
    avoid re-tracing every surface on each call).
    """
    if geom is None:
        geom = fsa_current_geometry(eq, psi_N, psi_pad=psi_pad,
                                    want_pprime=(convention == "jphi-linterp"))
    w, c = Ip_fsa_weights(geom, convention=convention, pprime_sign=pprime_sign)
    return _trapezoid(w * np.asarray(j_profile, dtype=float),
                      geom["psi_N"]) + c


def eq_jphi_profile(geom, convention="jphi-linterp", eq=None, psi_N=None,
                    pprime_sign=1.0):
    r"""The equilibrium's OWN current profile on ``geom``'s grid, in *convention*.

    Reconstructed from the Grad-Shafranov source functions -- ``jphi-linterp``
    gives ``<R> P' + <1/R> FF'/mu0``, ``fsa`` gives
    ``(P' + <1/R^2> FF'/mu0)/<1/R>`` -- so that
    ``Ip_fsa_integral(..., this profile) == the equilibrium's true I_p``.  That
    round trip is the self-consistency check the measure is validated by
    (measured: +0.0055 % on the D3D-like anchor).

    Needs ``F`` and ``F'`` as well as ``P'``, so it takes *eq* and re-reads
    ``get_profiles`` unless *geom* already carries ``FFp``.
    """
    FFp = geom.get("FFp")
    if FFp is None:
        if eq is None:
            raise ValueError("eq_jphi_profile needs eq (or geom['FFp'])")
        prof = eq.get_profiles(psi=geom["psi_q"].copy())
        FFp = np.asarray(prof[1], dtype=float) * np.asarray(prof[2], dtype=float)
    FFp = float(pprime_sign) * np.asarray(FFp, dtype=float) / _FSA_MU0
    pprime = float(pprime_sign) * np.asarray(geom["pprime"], dtype=float)
    if convention == "jphi-linterp":
        return geom["R_avg"] * pprime + geom["inv_R"] * FFp
    if convention == "fsa":
        if geom["inv_R2"] is None:
            raise ValueError("the 'fsa' equilibrium profile needs <1/R^2>")
        return (pprime + geom["inv_R2"] * FFp) / geom["inv_R"]
    raise ValueError(f"unknown convention {convention!r}")


def Ip_flux_integral_vs_target(alpha, mygs, jtor_prof, spike_profile, psi_N, Ip_target):
    r'''! Compute difference between integrated a*j_tor+j_spike profile and Ip_target

    @param alpha Scaling factor to solve for
    @param jtor_prof Input j_inductive profile
    @param spike_profile Isolated j_bootstrap spike (a Gaussian), 0.0 everywhere else
    @param my_psi_N Local psi_N grid
    @param my_Ip_target Ip target
    '''
    prof = alpha*jtor_prof + spike_profile
    Ip_computed = mygs.flux_integral(psi_N, prof)
    return Ip_computed - Ip_target

def Hmode_profiles(edge=0.08, ped=0.4, core=2.5, rgrid=201, expin=1.5, expout=1.5, widthp=0.04, xphalf=None):
    r'''! This function generates H-mode density and temperature profiles evenly spaced in your favorite 
    radial coordinate. Copied from https://omfit.io/_modules/omfit_classes/utils_fusion.html

    @param edge Separatrix height (float)
    @param ped Pedestal height (float)
    @param core On-axis profile height (float)
    @param rgrid Number of radial grid points (int)
    @param expin Inner core exponent for H-mode pedestal profile (float)
    @param expout Outer core exponent for H-mode pedestal profile (float)
    @param widthp Width of pedestal (float)
    @param xphalf Position of tanh (float, optional)
    @result H-mode profile array over radial grid
    '''

    w_E1 = 0.5 * widthp  # width as defined in eped
    if xphalf is None:
        xphalf = 1.0 - w_E1

    xped = xphalf - w_E1

    pconst = 1.0 - np.tanh((1.0 - xphalf) / w_E1)
    a_t = 2.0 * (ped - edge) / (1.0 + np.tanh(1.0) - pconst)

    coretanh = 0.5 * a_t * (1.0 - np.tanh(-xphalf / w_E1) - pconst) + edge

    xpsi = np.linspace(0, 1, rgrid)
    ones = np.ones(rgrid)

    val = 0.5 * a_t * (1.0 - np.tanh((xpsi - xphalf) / w_E1) - pconst) + edge * ones

    xtoped = xpsi / xped
    for i in range(0, rgrid):
        if xtoped[i] ** expin < 1.0:
            val[i] = val[i] + (core - coretanh) * (1.0 - xtoped[i] ** expin) ** expout

    return val

def _scan_key(scan_key):
    """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

    Returns ``None`` when *scan_key* is ``None`` (flat layout).
    """
    if scan_key is None:
        return None
    return str(scan_key)


def _resolve_h5(h5path_or_header):
    """Resolve an archive reference to an absolute ``.h5`` path.

    Accepts (duck-typed, so no import cycle):
      * a ``BouquetArchive`` (uses its ``.path``);
      * a ``Bouquet`` (uses ``config.output_header``);
      * a full path ending in ``.h5`` (returned unchanged);
      * a bare *header* stem (``<header>.h5`` resolved to absolute).

    The single place archive references are normalized, so every reader accepts
    a run object, an archive, a header, or a path interchangeably (F1).
    """
    p = getattr(h5path_or_header, "path", None)          # BouquetArchive
    if isinstance(p, str) and p.endswith(".h5"):
        return p
    cfg = getattr(h5path_or_header, "config", None)      # Bouquet
    ref = getattr(cfg, "output_header", None) if cfg is not None else h5path_or_header
    s = str(ref)
    return s if s.endswith(".h5") else os.path.abspath(f"{s}.h5")


def _default_scan_key(ref, scan_key):
    """Fill a missing ``scan_key`` from a ``Bouquet`` reference's config."""
    if scan_key is None:
        cfg = getattr(ref, "config", None)
        if cfg is not None:
            return getattr(cfg.generation, "scan_key", None)
    return scan_key


def _group_path(scan_key, count):
    """Return the internal HDF5 group path for a given entry."""
    bkey = _scan_key(scan_key)
    if bkey is not None:
        return f"scan/{bkey}/{int(count)}"
    return str(int(count))


# ====================================================================
#  Database lifecycle
# ====================================================================
def initialize_equilibrium_database(header):
    """
    Create (or open) the top-level HDF5 database file on disk.

    Stamps ``schema_version`` / ``bouquet_version`` / ``created`` at creation
    (the single chokepoint every archive passes through), so the ABSENCE of
    ``schema_version`` reliably identifies a pre-v2 legacy file to readers.
    ``config_json`` provenance is added separately by :func:`write_provenance`.

    Parameters
    ----------
    header : str
        Base name for the database.  File will be ``<header>.h5``.

    Returns
    -------
    db_path : str
        Absolute path to the HDF5 file.
    """
    import datetime
    from . import __version__
    db_path = os.path.abspath(f"{header}.h5")
    with h5py.File(db_path, "a") as hf:
        hf.attrs["schema_version"] = int(SCHEMA_VERSION)
        hf.attrs["bouquet_version"] = str(__version__)
        if "created" not in hf.attrs:
            hf.attrs["created"] = datetime.datetime.now().isoformat(
                timespec="seconds")
    return db_path


# HDF5 archive schema version -- imported from the schema module (now v2:
# bare dataset names + units attrs, fixed eqdsk/pfile names, unified coils).
from .schema import SCHEMA_VERSION


def write_provenance(h5path_or_header, config=None, scan_key=None):
    """Stamp provenance onto an archive: schema/version/timestamp + config JSON.

    File-level attrs ``schema_version``, ``bouquet_version``, ``created`` (set
    once) and ``updated``; and, when a :class:`~bouquet.BouquetConfig` is given,
    a ``config_json`` dataset at the file root mirrored into the
    ``scan/<key>/`` group when ``scan_key`` is provided (each slice can carry its
    own config). The per-scan copies are AUTHORITATIVE; the root copy is only
    the most recent write and is overwritten by each slice of a multi-slice
    sweep -- :func:`load_config` therefore disambiguates by scan and never
    silently serves a stale root copy. Called by ``Bouquet.generate`` /
    ``run_shard`` / ``merge_archives``; safe to call repeatedly.
    """
    import datetime
    from . import __version__
    path = _resolve_h5(h5path_or_header)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with h5py.File(path, "a") as hf:
        hf.attrs["schema_version"] = int(SCHEMA_VERSION)
        hf.attrs["bouquet_version"] = str(__version__)
        if "created" not in hf.attrs:
            hf.attrs["created"] = now
        hf.attrs["updated"] = now
        if config is not None:
            cj = config.to_json()
            targets = [hf]
            bkey = _scan_key(scan_key)
            if bkey is not None:
                targets.append(hf.require_group(f"scan/{bkey}"))
            for grp in targets:
                if "config_json" in grp:
                    del grp["config_json"]
                grp.create_dataset("config_json", data=cj)


def load_config(h5path_or_header, scan_key=None):
    """Reconstruct the :class:`~bouquet.BouquetConfig` stored in an archive.

    The per-scan ``config_json`` copies are authoritative (the root copy is
    just the most recent write). With an explicit ``scan_key``, that scan's
    copy is read. With ``scan_key=None``: a single stored per-scan config is
    served directly, but a MULTI-scan archive raises and lists the keys --
    each slice of a sweep carried its own config, so "the" config is
    ambiguous and the (last-writer-wins) root copy would silently describe
    only the final slice. Raises a clear error on pre-provenance archives.
    """
    from .config import BouquetConfig
    path = _resolve_h5(h5path_or_header)
    with h5py.File(path, "r") as hf:
        node = None
        bkey = _scan_key(scan_key)
        scan_cfgs = ([k for k in hf["scan"] if f"scan/{k}/config_json" in hf]
                     if "scan" in hf else [])
        if bkey is not None:
            if f"scan/{bkey}/config_json" in hf:
                node = hf[f"scan/{bkey}/config_json"]
        elif len(scan_cfgs) == 1:
            node = hf[f"scan/{scan_cfgs[0]}/config_json"]
        elif len(scan_cfgs) > 1:
            raise KeyError(
                f"{path} holds per-scan configs for {len(scan_cfgs)} scan keys "
                f"({scan_cfgs}); pass scan_key=<one of these> -- each slice of "
                "a sweep carries its own config, so there is no single file "
                "config to return.")
        elif "config_json" in hf:
            node = hf["config_json"]
        if node is None:
            raise KeyError(
                f"{path} has no stored config for scan_key={scan_key!r} "
                f"(schema_version={hf.attrs.get('schema_version', 'absent')}; "
                f"per-scan configs present: {scan_cfgs or 'none'}). Only files "
                "written by a provenance-aware Bouquet carry a config.")
        raw = node[()]
    return BouquetConfig.from_json(raw.decode() if isinstance(raw, bytes) else str(raw))


# ====================================================================
#  Per-equilibrium storage
# ====================================================================
_PROFILE_KEYS = [
    "psi_N",
    "j_phi",
    "j_BS",
    "j_BS,edge",
    "j_inductive",
    "n_e",
    "T_e",
    "n_i",
    "T_i",
    "w_ExB",
]


def _read_coil_names(grp):
    """Coil-name list from the ``coil_names`` string dataset (schema v2; F15)."""
    if "coil_names" not in grp:
        return []
    return [n.decode() if isinstance(n, bytes) else str(n)
            for n in grp["coil_names"][()]]


def store_equilibrium(
    header,
    count,
    eqdsk_filepath,
    psi_N,
    j_phi,
    j_BS,
    j_inductive,
    n_e,
    T_e,
    n_i,
    T_i,
    w_ExB,
    li1,
    li3,
    scan_key=None,
    pressure=None,
    pressure_thermal=None,
    j_BS_edge=None,
    pfile_bytes=None,
    Zeff=None,
    coil_currents=None,
    psi_N_kinetic=None,
    homotopy_pass=None,
    homotopy_F_lim=None,
    homotopy_VSC_lim=None,
    max_F_drift_pct=None,
    max_VSC_drift_pct=None,
    in_spec=None,
    inspec_F_max=None,
    inspec_VSC_max=None,
    perturbed_lcfs_ref=None,
    l_i_target_used=None,
    l_i_uncertainty=None,
    x_points=None,
    diverted=None,
    aux=None,
    eq_fsa=None,
):
    """
    Write one perturbed equilibrium into the HDF5 database.

    Parameters
    ----------
    header : str
        Base name (same string passed to ``initialize_equilibrium_database``).
    count : int
        Perturbation index (typically 0 -- N-1).
    eqdsk_filepath : str
        Path to the ``.geqdsk`` / ``.eqdsk`` file.  Read as raw bytes so
        the Fortran-namelist formatting is preserved exactly.
    psi_N, j_phi, j_BS, j_inductive,
    n_e, T_e, n_i, T_i, w_ExB : array_like, 1-D
        Profile arrays.
    li1 : float
        Internal inductance l_i(1).
    li3 : float
        Internal inductance l_i(3).
    scan_key : str, float, int, or None
        Scan-point label.  When provided, an extra ``scan/{label}/``
        group layer is inserted.  ``None`` gives the flat layout.
    pressure : array_like or None
        1-D total pressure.
    j_BS_edge : array_like or None
        1-D isolated edge bootstrap current [A m^-2].
    pfile_bytes : bytes or None
        Raw p-file content to store alongside the g-file bytes.
    Zeff : array_like or None
        1-D effective charge profile (dimensionless).
    coil_currents : dict or None
        Coil currents {name: current_A} from TokaMaker.
    """
    db_path = os.path.abspath(f"{header}.h5")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Database '{db_path}' not found.  "
            f"Call initialize_equilibrium_database('{header}') first."
        )

    with open(eqdsk_filepath, "rb") as fh:
        eqdsk_bytes = fh.read()

    grp_path = _group_path(scan_key, count)

    with h5py.File(db_path, "a") as hf:
        # clean slate if this entry already exists
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        # ---- raw eqdsk (opaque binary -- bit-perfect; schema-v2 fixed
        # name, the group path carries the coordinates) --------------------
        from .schema import EQDSK_DS
        grp.create_dataset(EQDSK_DS, data=np.void(eqdsk_bytes))

        # ---- 1-D profiles -----------------------------------------------
        write_profile(grp, "psi_N", psi_N)
        write_profile(grp, "j_phi", j_phi)
        write_profile(grp, "j_BS", j_BS)
        write_profile(grp, "j_inductive", j_inductive)

        if j_BS_edge is not None:
            write_profile(grp, "j_BS,edge", j_BS_edge)
        write_profile(grp, "n_e", n_e)
        write_profile(grp, "T_e", T_e)
        write_profile(grp, "n_i", n_i)
        write_profile(grp, "T_i", T_i)
        write_profile(grp, "w_ExB", w_ExB)

        # ---- optional: kinetic profile grid (when different from psi_N) ----
        if psi_N_kinetic is not None:
            write_profile(grp, "psi_N_kinetic", psi_N_kinetic)

        if pressure is not None:
            write_profile(grp, "pressure", pressure)
        # Thermal (main-ion + electron) pressure alongside the total stored above,
        # so plots can show what the kinetic profiles contribute vs the impurity +
        # fast-ion pressure the GS solve actually used (total - thermal).
        if pressure_thermal is not None:
            write_profile(grp, "pressure_thermal", pressure_thermal)

        # ---- optional: auxiliary perturbed profiles (the switchboard) ----
        # Stored on psi_N_kinetic. Named "aux_<name>" (e.g. aux_omega_tor,
        # aux_chi_e, aux_zeff) for downstream (transport) consumers.
        if aux:
            for _name, _arr in aux.items():
                grp.create_dataset(f"aux_{_name}",
                                   data=np.asarray(_arr, dtype=np.float64))

        # ---- scalars (group attributes) ----------------------------------
        grp.attrs["l_i(1)"] = float(li1)
        grp.attrs["l_i(3)"] = float(li3)
        grp.attrs["count"]  = int(count)
        if scan_key is not None:
            grp.attrs["scan_key"] = scan_key

        # ---- optional: p-file bytes ----------------------------------------
        # Per-draw pfile blobs are only stored for TEXT p-files (rewritten with
        # this draw's perturbed kinetics -- genuinely draw-specific data). A
        # binary IDA .cdf source cannot be draw-perturbed and would only be
        # duplicated verbatim into every draw (~190 MB x n_draws with the new
        # IDA-database files); it lives ONCE in _baseline and per-draw readers
        # (BouquetArchive.DrawView.pfile_bytes, load_equilibrium) fall back.
        from .schema import is_binary_profile_source
        if pfile_bytes is not None and not is_binary_profile_source(pfile_bytes):
            pf_ds = "pfile"
            grp.create_dataset(pf_ds, data=np.void(pfile_bytes))

        # ---- optional: Zeff profile ----------------------------------------
        if Zeff is not None:
            write_profile(grp, "Zeff", Zeff)

        # ---- optional: coil currents ---------------------------------------
        if coil_currents is not None:
            import json
            names = list(coil_currents.keys())
            values = np.array([coil_currents[n] for n in names], dtype=np.float64)
            grp.create_dataset("coil_currents", data=values)
            grp.create_dataset("coil_names",
                               data=np.array(names, dtype=h5py.string_dtype()))

        # ---- homotopy / in-spec metadata (per-draw) -----------------------
        if homotopy_pass is not None:
            grp.attrs["homotopy_pass"] = int(homotopy_pass)
        if homotopy_F_lim is not None:
            grp.attrs["homotopy_F_lim"] = float(homotopy_F_lim)
        if homotopy_VSC_lim is not None:
            grp.attrs["homotopy_VSC_lim"] = float(homotopy_VSC_lim)
        if max_F_drift_pct is not None:
            grp.attrs["max_F_drift_pct"] = float(max_F_drift_pct)
        if max_VSC_drift_pct is not None:
            grp.attrs["max_VSC_drift_pct"] = float(max_VSC_drift_pct)
        if in_spec is not None:
            grp.attrs["in_spec"] = bool(in_spec)
        if inspec_F_max is not None:
            grp.attrs["inspec_F_max"] = float(inspec_F_max)
        if inspec_VSC_max is not None:
            grp.attrs["inspec_VSC_max"] = float(inspec_VSC_max)
        # The l_i_target this draw actually converged to.  Equal to the
        # bouquet's l_i_target when l_i_uncertainty=0; otherwise a per-
        # draw sample from N(l_i_target, l_i_uncertainty * l_i_target).
        # Stored so post-run analysis can verify the sampled
        # distribution + diagnose any draw whose realised l_i diverged
        # from its (intentionally perturbed) target.
        if l_i_target_used is not None:
            grp.attrs["l_i_target_used"] = float(l_i_target_used)
        if l_i_uncertainty is not None:
            grp.attrs["l_i_uncertainty"] = float(l_i_uncertainty)

        # ---- High-resolution LCFS reference for this draw.
        # Captured by perturb_kinetic_equilibrium via safe_trace_surf at
        # the same mygs state save_eqdsk was called from -- so this and
        # the baseline.recon_lcfs_ref are method-consistent (both 10k-pt
        # trace_surf at psi = 1 - psi_pad).  Eliminates the ~4 mm
        # sampling-noise floor that the 100-pt eqdsk RBBBS introduces
        # when used as the perturbed-side boundary in plot_traces.
        if perturbed_lcfs_ref is not None:
            grp.create_dataset(
                "perturbed_lcfs_ref",
                data=np.asarray(perturbed_lcfs_ref, dtype=np.float64),
            )

        # ---- TokaMaker-computed X-point(s) for this draw ------------------
        # The poloidal-field nulls returned by mygs.get_xpoints(), captured
        # at the SAME solver state the eqdsk was saved from.  This is the
        # authoritative X-point location (a true B_p=0 saddle), used by
        # plot_boundary_point_traces in place of any geometric corner guess.
        # Shape (N, 2) = [[R, Z], ...]; the active (boundary-defining) null
        # is the last row when ``diverted`` is True.
        if x_points is not None:
            xp_arr = np.asarray(x_points, dtype=np.float64).reshape(-1, 2)
            if xp_arr.size:
                grp.create_dataset("x_points", data=xp_arr)
        if diverted is not None:
            grp.attrs["diverted"] = bool(diverted)

        # ---- Live-equilibrium FSA block (optional subgroup) --------------
        # Captured from the converged TokaMaker equilibrium at the same state
        # the eqdsk was saved from, so this draw's own flux geometry enables
        # an exact toroidal<->parallel conversion at IMAS export
        # (physics.capture_equilibrium_fsa -> physics.toroidal_to_parallel).
        if eq_fsa:
            from .schema import EQ_FSA_GROUP, EQ_FSA_UNITS
            fsa_grp = grp.create_group(EQ_FSA_GROUP)
            for _name, _arr in eq_fsa.items():
                if _arr is None:
                    continue
                ds = fsa_grp.create_dataset(
                    _name, data=np.asarray(_arr, dtype=np.float64))
                _u = EQ_FSA_UNITS.get(_name)
                if _u:
                    ds.attrs["units"] = _u


def load_eq_fsa(header, count, scan_key=None):
    """Load one draw's live-equilibrium FSA block, or ``None`` if not captured.

    Returns a dict of 1-D arrays (``psi_N``, ``F``, ``avg_inv_R``,
    ``avg_inv_R2`` (present only when the exact quadrature succeeded),
    ``avg_B2``, ``q``, ``dV_dpsi``, ``f_trap``, ``B_avg``) -- the geometry
    :func:`bouquet.physics.toroidal_to_parallel` needs for an exact IMAS
    write-back. ``None`` for archives written without live capture (fall back
    to the baseline-ratio reconstruction).
    """
    from .schema import EQ_FSA_GROUP
    h5path = _resolve_h5(header)
    gp = _group_path(scan_key, count)
    with h5py.File(h5path, "r") as hf:
        if gp not in hf or EQ_FSA_GROUP not in hf[gp]:
            return None
        g = hf[gp][EQ_FSA_GROUP]
        return {k: np.array(g[k], dtype=float) for k in g}


def load_equilibrium(header, count, scan_key=None, eqdsk_out_dir=None):
    """
    Retrieve one equilibrium entry from the HDF5 database.

    Parameters
    ----------
    header : str
        Base name of the database.
    count : int
        Perturbation index.
    scan_key : str, float, int, or None
        Scan-point label (must match what was used at write time).
    eqdsk_out_dir : str or None, optional
        If given, the raw eqdsk is written to a file in this directory.

    Returns
    -------
    result : dict
        Keys: ``"eqdsk_filepath"``, ``"eqdsk_bytes"``,
        the 1-D array names, ``"l_i(1)"``, ``"l_i(3)"``,
        and optionally ``"pressure"``, ``"Zeff"``,
        ``"coil_currents"``, ``"pfile_bytes"``.
    """
    from .schema import EQDSK_DS

    db_path  = os.path.abspath(f"{header}.h5")
    grp_path = _group_path(scan_key, count)

    result = {}

    with h5py.File(db_path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {db_path}"
            )
        grp = hf[grp_path]

        # ---- eqdsk raw bytes (schema-v2 fixed name) ----------------------
        if EQDSK_DS not in grp:
            legacy = [k for k in grp if k.endswith(".eqdsk")]
            raise KeyError(
                f"no '{EQDSK_DS}' dataset in {grp_path} of {db_path}"
                + (f" -- found legacy-named {legacy}: this is a pre-v2 "
                   "archive; regenerate it with the current bouquet (or "
                   "read it via BouquetArchive, whose suffix scan still "
                   "resolves legacy eqdsk names)." if legacy else "."))
        eqdsk_bytes = bytes(grp[EQDSK_DS][()])
        result["eqdsk_bytes"] = eqdsk_bytes

        if eqdsk_out_dir is not None:
            os.makedirs(eqdsk_out_dir, exist_ok=True)
            # coordinate-carrying FILENAME (the fixed dataset name would
            # make every extracted draw clobber the same 'eqdsk' file)
            bkey = _scan_key(scan_key)
            stem = os.path.basename(header) + (f"_{bkey}" if bkey else "")
            out_path = os.path.join(eqdsk_out_dir, f"{stem}_{int(count)}.eqdsk")
            with open(out_path, "wb") as fh:
                fh.write(eqdsk_bytes)
            result["eqdsk_filepath"] = os.path.abspath(out_path)
        else:
            result["eqdsk_filepath"] = None

        # ---- 1-D arrays ------------------------------------------------
        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure" in grp:
            result["pressure"] = np.array(grp["pressure"])
        if "pressure_thermal" in grp:
            result["pressure_thermal"] = np.array(grp["pressure_thermal"])

        # ---- scalars ----------------------------------------------------
        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

        # ---- optional: Zeff -----------------------------------------------
        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        # ---- optional: p-file bytes ----------------------------------------
        # Text p-file sources are stored per draw (draw-perturbed); binary IDA
        # .cdf sources live ONCE in _baseline -- fall back there when the draw
        # carries no blob (see store_equilibrium / is_binary_profile_source).
        pf_ds = "pfile"
        if pf_ds in grp:
            result["pfile_bytes"] = bytes(grp[pf_ds][()])
        else:
            _bl = grp.parent.get("_baseline") if grp.parent is not None else None
            if _bl is not None and pf_ds in _bl:
                result["pfile_bytes"] = bytes(_bl[pf_ds][()])

        # ---- optional: coil currents ---------------------------------------
        if "coil_currents" in grp:
            import json
            values = np.array(grp["coil_currents"])
            names = _read_coil_names(grp)
            result["coil_currents"] = dict(zip(names, values))

    return result


# ====================================================================
#  Baseline (input) profile storage
# ====================================================================
def store_baseline_profiles(
    header,
    psi_N,
    ne,
    te,
    ni,
    ti,
    pressure,
    j_phi,
    sigma_ne,
    sigma_te,
    sigma_ni,
    sigma_ti,
    sigma_jphi,
    Ip_target,
    l_i_target,
    scan_key=None,
    l_i_scale=LI_SCALE,
    pressure_thermal=None,
    eqdsk_bytes=None,
    pfile_bytes=None,
    psi_N_kinetic=None,
    coil_currents=None,
    coil_names=None,
    recon_lcfs_ref=None,
    x_points=None,
    diverted=None,
    aux_baselines=None,
    aux_sigmas=None,
    j_BS=None,
    j_inductive=None,
    source_kind=None,
):
    """
    Store the input (baseline) profiles and their uncertainties.

    For hierarchical layout (*scan_key* is not ``None``), these are
    stored in ``scan/{label}/_baseline/``.  For flat layout they go
    in ``_baseline/``.

    Parameters
    ----------
    eqdsk_bytes : bytes or None
        Raw baseline geqdsk file content.  Stored so that
        ``plot_geqdsk_bouquet`` can distinguish the true baseline
        from perturbed equilibria.
    pfile_bytes : bytes or None
        Raw baseline p-file content.

    This data is written once per scan-point and is required by the
    plotting GUI to be fully self-contained.
    """
    db_path = os.path.abspath(f"{header}.h5")
    bkey = _scan_key(scan_key)

    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    with h5py.File(db_path, "a") as hf:
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        write_profile(grp, "psi_N", psi_N)
        write_profile(grp, "n_e", ne)
        write_profile(grp, "T_e", te)
        write_profile(grp, "n_i", ni)
        write_profile(grp, "T_i", ti)
        write_profile(grp, "pressure", pressure)
        if pressure_thermal is not None:
            write_profile(grp, "pressure_thermal", pressure_thermal)
        write_profile(grp, "j_phi", j_phi)
        if j_BS is not None:
            write_profile(grp, "j_BS", j_BS)
        if j_inductive is not None:
            write_profile(grp, "j_inductive", j_inductive)
        write_profile(grp, "sigma_ne", sigma_ne)
        write_profile(grp, "sigma_te", sigma_te)
        write_profile(grp, "sigma_ni", sigma_ni)
        write_profile(grp, "sigma_ti", sigma_ti)
        write_profile(grp, "sigma_jphi", sigma_jphi)

        if psi_N_kinetic is not None:
            write_profile(grp, "psi_N_kinetic", psi_N_kinetic)

        # ---- auxiliary-profile baselines + sigmas (on the kinetic
        # grid), so plot_transport_profiles can draw input +/- sigma bands
        if aux_baselines:
            for _n, _a in aux_baselines.items():
                grp.create_dataset(f"aux_{_n}",
                                   data=np.asarray(_a, dtype=np.float64))
        if aux_sigmas:
            for _n, _a in aux_sigmas.items():
                grp.create_dataset(f"sigma_aux_{_n}",
                                   data=np.asarray(_a, dtype=np.float64))

        grp.attrs["Ip_target"]  = float(Ip_target)
        grp.attrs["l_i_target"] = float(l_i_target)
        # Which l_i estimator `l_i_target` is on, so the archive is
        # self-describing and a reader can pick the matching per-draw attr
        # (`l_i(3)` for "iter(li3)", `l_i(1)` for the legacy "std(li1)").
        # Archives written before issue #20 lack this attr entirely --
        # readers must default to "std(li1)".
        grp.attrs["l_i_scale"] = str(l_i_scale)
        # Provenance marker for robust path detection in plotting (independent of
        # the source-decoupled aux switchboard): "imas" or "geqdsk".
        if source_kind is not None:
            grp.attrs["source_kind"] = str(source_kind)

        if eqdsk_bytes is not None:
            grp.create_dataset("eqdsk", data=np.void(eqdsk_bytes))
        if pfile_bytes is not None:
            grp.create_dataset("pfile", data=np.void(pfile_bytes))

        # ---- Recon-LCFS reference for downstream boundary-deviation
        # measurements.  Captured by the caller via mygs.trace_surf() at
        # the SAME mygs state where the baseline.eqdsk was saved, giving
        # a method-consistent reference (~10000 points) for plot_traces
        # and the per-stage bnd-diag inside generate_bouquet.  Using this
        # instead of the eqdsk's 100-pt boundary eliminates ~2-3 mm of
        # save_eqdsk sampling noise (see Probe 2 in save_eqdsk_probe.py).
        if recon_lcfs_ref is not None:
            grp.create_dataset(
                "recon_lcfs_ref",
                data=np.asarray(recon_lcfs_ref, dtype=np.float64),
            )

        # ---- TokaMaker-computed X-point(s) for the recon baseline --------
        # Same provenance as the per-draw ``x_points`` (mygs.get_xpoints()
        # at the recon-converged state).  plot_boundary_point_traces uses
        # this to decide whether the top/bottom traces are tracking a true
        # X-point and to anchor the deviation reference.
        if x_points is not None:
            xp_arr = np.asarray(x_points, dtype=np.float64).reshape(-1, 2)
            if xp_arr.size:
                grp.create_dataset("x_points", data=xp_arr)
        if diverted is not None:
            grp.attrs["diverted"] = bool(diverted)

        # Recon's converged coil currents (the perturbation reference).
        # Saved alongside profiles so post-processors can compute
        # absolute coil drift per draw without re-running recon.
        if coil_currents is not None:
            if coil_names is not None:
                names = list(coil_names)
                values = np.array([float(coil_currents[n]) for n in names],
                                  dtype=np.float64)
            elif isinstance(coil_currents, dict):
                names = list(coil_currents.keys())
                values = np.array([float(coil_currents[n]) for n in names],
                                  dtype=np.float64)
            else:
                names = [f"coil_{i}" for i in range(len(coil_currents))]
                values = np.asarray(coil_currents, dtype=np.float64)
            grp.create_dataset("coil_currents", data=values)
            grp.create_dataset("coil_names",
                               data=np.array(names, dtype=h5py.string_dtype()))


# ====================================================================
#  Introspection helpers (used by GUI and notebook API)
# ====================================================================
def discover_scan_keys(h5path_or_header):
    """
    Discover all scan values in an HDF5 equilibrium database.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- either a full ``.h5`` path or a bare header
        stem (``<header>.h5`` is resolved).

    Returns
    -------
    scan_keys : list[str] or None
        Sorted list of scan-value keys, or ``None`` if the file uses
        the flat layout (no ``scan/`` group).
    """
    h5path = _resolve_h5(h5path_or_header)
    with h5py.File(h5path, "r") as hf:
        if "scan" not in hf:
            return None
        keys = list(hf["scan"].keys())

    # Sort numerically when all keys look like numbers, otherwise
    # fall back to lexicographic order.
    try:
        return sorted(keys, key=float)
    except (ValueError, TypeError):
        return sorted(keys)


def count_equilibria(h5path_or_header, scan_key=None):
    """
    Count the number of perturbed equilibria stored for a scan value.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    n : int
    """
    return len(list_equilibrium_indices(h5path_or_header, scan_key=scan_key))


def list_equilibrium_indices(h5path_or_header, scan_key=None):
    """Return the sorted list of integer draw indices actually stored.

    Band-rejected / failed draws leave GAPS in the index sequence (e.g.
    ``[0, 1, 2, 3, 4, 5, 7, ...]`` with draw 6 missing), so callers must
    iterate these indices rather than ``range(count_equilibria(...))`` --
    the latter assumes a contiguous ``0..n-1`` and KeyErrors on the gap.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    list of int
        Sorted stored draw indices.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
    with h5py.File(h5path, "r") as hf:
        parent = hf[f"scan/{bkey}"] if bkey is not None else hf
        return sorted(
            int(k) for k in parent.keys()
            if k not in ("_baseline", "scan") and str(k).lstrip("-").isdigit()
        )


def load_baseline_profiles(h5path_or_header, scan_key=None):
    """
    Load the baseline profiles and uncertainties for a given scan value.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        ``None`` for flat-layout files.

    Returns
    -------
    result : dict
        All stored baseline arrays and scalar attributes.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Baseline group '{grp_path}' not found in {h5path}.  "
                f"Was store_baseline_profiles() called?"
            )
        grp = hf[grp_path]
        for key in grp.keys():
            result[key] = np.array(grp[key])
        for attr in grp.attrs:
            result[attr] = grp.attrs[attr]

    return result


def load_equilibrium_by_path(h5path_or_header, count, scan_key=None):
    """
    Load one perturbed equilibrium (profiles + scalars only).

    Like :func:`load_equilibrium`, but addressed by *scan_key* and does
    **not** extract the raw eqdsk bytes (use :func:`load_equilibrium` if
    you need those).  Accepts either a full ``.h5`` path or a bare header
    stem for *h5path_or_header*.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
    if bkey is not None:
        grp_path = f"scan/{bkey}/{int(count)}"
    else:
        grp_path = str(int(count))

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {h5path}"
            )
        grp = hf[grp_path]

        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure" in grp:
            result["pressure"] = np.array(grp["pressure"])
        if "pressure_thermal" in grp:
            result["pressure_thermal"] = np.array(grp["pressure_thermal"])

        if "psi_N_kinetic" in grp:
            result["psi_N_kinetic"] = np.array(grp["psi_N_kinetic"])

        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])

        # auxiliary ("switchboard") perturbed profiles -- aux_zeff, aux_omega_tor,
        # aux_chi_e, ... on psi_N_kinetic -- so per-draw plots (draw_zeff, the aux
        # figure) can overlay the draws, not just the baseline.
        for _k in grp.keys():
            if str(_k).startswith("aux_"):
                result[str(_k)] = np.array(grp[_k])

        if "coil_currents" in grp:
            import json
            values = np.array(grp["coil_currents"])
            names = _read_coil_names(grp)
            result["coil_currents"] = dict(zip(names, values))

        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

    return result


# ====================================================================
#  kinetic-profile regridding helper
# ====================================================================
def pchip_interp(x_src, y_src, x_out):
    r'''Shape-preserving (PCHIP) regrid of a profile onto a new grid.

    The single shared kin->eq regridding used everywhere a kinetic profile
    (measured on the IDA/kinetic ``psi_N`` grid) is resampled onto the
    equilibrium grid. Replaces the piecewise-LINEAR ``np.interp`` /
    ``interp1d(kind='linear')`` pattern: linear regrid leaves a slope kink
    at every source knot, so ANY subsequent derivative (including OFT's
    PCHIP-derivative bootstrap) is a staircase with plateaus between knots
    -- visible as stepped j_BS between the kinetic knots (verified on a
    weak-pedestal case: |d2 j_BS| correlates 0.99 with |d2 dTe/dpsi| of the
    linear regrid, identical in np.gradient and PCHIP OFT builds; direct
    PCHIP regrid cuts the step energy ~7x).

    PCHIP passes through the same source knot values (no alteration of the
    measured points), is monotone/shape-preserving between them (no spline
    ringing at a pedestal), and is C1 -- so downstream gradients are
    smooth. Duplicate source-x points are dropped (first occurrence kept,
    matching :func:`pchip_derivative`); outside the source range the
    endpoint values are held constant (the fill_value=(y[0], y[-1])
    convention of the interp1d calls this replaces).

    All regrid sites must use this helper (or none): mixing linear and
    PCHIP regrids between the recon and draw paths would break the
    sigma=0 consistency that verify_sigma0_consistency() guards.
    '''
    from scipy.interpolate import PchipInterpolator

    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_out = np.asarray(x_out, dtype=float)
    x_u, idx = np.unique(x_src, return_index=True)
    y_u = y_src[idx]
    if x_u.size < 2:
        raise ValueError("pchip_interp: fewer than 2 unique source points")
    out = PchipInterpolator(x_u, y_u, extrapolate=False)(x_out)
    # hold endpoint values outside the source range (no poly extrapolation)
    out = np.where(x_out < x_u[0], y_u[0], out)
    out = np.where(x_out > x_u[-1], y_u[-1], out)
    return out


# ====================================================================
#  LCFS shape helper
# ====================================================================
def _shape_from_boundary(boundary_RZ):
    """LCFS shape params (R0, Z0, a, kappa, delta) from boundary (R,Z) points.

    Lives here rather than in ``bouquet.run`` because both the orchestrator and
    ``TokaMaker_interface`` need it, and importing it from ``run`` into
    ``TokaMaker_interface`` at runtime would close a dependency cycle (``run``
    already imports from ``TokaMaker_interface``).  ``bouquet.run`` re-exports
    it for callers that use its original location.
    """
    rz = np.asarray(boundary_RZ, dtype=float)
    R, Z = rz[:, 0], rz[:, 1]
    Rmax, Rmin, Zmax, Zmin = R.max(), R.min(), Z.max(), Z.min()
    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    kappa = (Zmax - Zmin) / (Rmax - Rmin)
    R_upper = R[int(np.argmax(Z))]
    R_lower = R[int(np.argmin(Z))]
    delta = 0.5 * ((R0 - R_upper) + (R0 - R_lower)) / a
    return R0, Z0, a, kappa, delta


# ====================================================================
#  eqdsk byte-stream helper
# ====================================================================
def read_eqdsk_from_bytes(raw_bytes, reader_func):
    """
    Call an existing eqdsk reader that expects a filename,
    but feed it in-memory bytes instead of a file on disk.
    """
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".eqdsk",
        delete=False,
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        result = reader_func(tmp_path)
    finally:
        os.remove(tmp_path)

    return result
