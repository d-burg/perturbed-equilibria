"""TokaMaker interface – perturbed Grad-Shafranov equilibrium workflow
=====================================================================

Routines that interact directly with TokaMaker to iterate perturbed
kinetic / current-density profiles toward target :math:`I_p` and
:math:`l_i` values.

Provides:
  - ``fit_inductive_profile`` – spline-based fit of the inductive
    current-density profile, scaled to match a target :math:`l_i` proxy.
  - ``perturb_kinetic_equilibrium`` – perturbs kinetic + current-density
    profiles and iterates to match :math:`I_p` and :math:`l_i` targets
    via TokaMaker.
  - ``generate_bouquet`` – batch driver that archives perturbed
    equilibria to HDF5.
  - ``reconstruct_equilibrium`` – reconstruct a single equilibrium from
    a geqdsk reference and kinetic profiles, matching :math:`l_i(1)`
    via secant iteration.
"""

import os
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

from .sampling import (
    generate_perturbed_GPR,
    make_rng,
    calc_cylindrical_li_proxy,
    get_li_proxy_geometry,
    calc_cylindrical_li_proxy_fast,
    calc_realgeom_li_proxy_fast,
    _draw_monotonic_perturbation,
    EC,
    _MAX_PRESSURE_ITER,
    _MAX_LI_ITER,
)
from .utils import (
    Ip_flux_integral_vs_target,
    Ip_fsa_weights,
    eq_jphi_profile,
    fsa_current_geometry,
    pchip_derivative,
    pchip_interp,
    safe_save_eqdsk,
    safe_trace_surf,
    select_closed_lcfs,
    store_equilibrium,
    store_baseline_profiles,
    _scan_key,
    _shape_from_boundary,
    read_eqdsk_from_bytes,
)
from .io.geqdsk import read_geqdsk
from .physics import q_ravg

# ---- Masked anchor-solve failure counter (issue #24) ------------------------
# The per-draw anchor solves swallow failures (fallback: `pass`; band
# resampling: `continue`).  Those masks are deliberate control flow, but the
# failures were invisible, so the converged-on-entry degeneracy (see
# verify_sigma0_consistency and issue #24) cannot be sized on real campaigns.
# This counter ONLY observes: control flow is unchanged, and each increment
# prints one line.
#
# WHAT THIS TALLY IS AND IS NOT.  Read before relying on it:
#
#  * It is a plain module-level dict and is NEVER reset.  Its value is
#    therefore "increments since this interpreter imported the module", not
#    "failures in this run".  The per-unit reading of an archived genlog holds
#    only because a campaign unit is its own PROCESS; two runs in one
#    interpreter share -- and keep accumulating into -- the same counters.
#  * It has NO consumer beyond the print below.  Nothing archives it, nothing
#    asserts on it, and nothing reads it at the end of a run.  The evidence a
#    campaign keeps is the printed `[anchor-masked-failure]` lines in the
#    captured genlog, not this dict.
#  * Under `bouquet.parallel` it is per-WORKER.  That path uses a spawn
#    ProcessPoolExecutor, so every worker imports its own copy; the parent's
#    counters stay at zero no matter how many failures the workers mask, and
#    the per-worker tallies are only visible in each worker's captured output.
#
# Deliberately NOT archived into the run diagnostics: the parent-side value is
# zero under the parallel backend (so it would archive a confidently wrong
# number), and adding a key to the archived diagnostics would move the
# goldens, which this non-physics work must not do.  Sizing the degeneracy
# across a real campaign should aggregate the printed lines from the genlogs.
ANCHOR_MASKED_FAILURES = {"recon_anchor_fallback": 0, "band_resample": 0,
                          # third site (issue #24): the unperturbed
                          # jphi-linterp baseline solve, whose failure silently
                          # reverts every per-draw boundary/l_i diagnostic to
                          # recon's inverse-mode reference.
                          "jphi_baseline": 0}


def sigma0_reference_scale(jBS_scale_range):
    """Bootstrap scale of the sigma=0 delta-mode reference spike.

    The cached reference must sit at the CENTER of the per-draw scale
    distribution so the delta composition telescopes at sigma=0: the
    midpoint of ``jBS_scale_range`` (uniform draws -> mean == midpoint;
    run.py has already re-centred the range on ``bl.bs_scale``), and 1.0
    when no range is configured (every draw then runs at scale 1.0).
    """
    if jBS_scale_range is None:
        return 1.0
    return 0.5 * (float(jBS_scale_range[0]) + float(jBS_scale_range[1]))


def _count_masked_anchor_failure(site, exc):
    ANCHOR_MASKED_FAILURES[site] += 1
    print(f"  [anchor-masked-failure] {site} #{ANCHOR_MASKED_FAILURES[site]}: "
          f"{type(exc).__name__}: {exc}", flush=True)


# ---- Adaptive corrective iteration ----
def _renormalize_target_to_Ip(mygs, psi_N, target_jphi, Ip_target, psi_pad,
                              label="jphi_corr"):
    """Scale a corrector target so it carries ``Ip_target`` (issue #29).

    The corrective iteration hands TokaMaker a ``jphi-linterp`` input and the
    solver renormalises that input's AMPLITUDE to hit ``Ip_target`` on every
    iterate.  A target whose own current integral is not ``Ip_target`` is
    therefore a shape the solver can reproduce but a total it is constrained
    to refuse -- and the Newton update ``input += target - output`` keeps
    re-injecting that refused current.  Measured (#29) on eight cases, the
    reconstruction's target carried **-3.9 % to +18.7 % of Ip** (golden
    +6.0 %; its amplitude comes from the l_i secant, which never sees Ip),
    while every solver output sat at Ip to <= 0.06 %.

    Uniform scaling is the right operation: ``l_i`` depends on the shape, not
    the amplitude, so the step-6 match is preserved in shape; and a
    uniformly-scaled target is exactly what ``jphi-linterp`` hands back.

    The measure is the physical FSA current integral on the LIVE geometry
    (self-validated to 0.01-0.04 % on real cases, #35) -- never the
    limiter-area ``flux_integral``, which reads the same target +10..+44 %
    high.  It is AFFINE, ``Ip[J] = int(w*J) + c`` (the ``P'`` term lands in
    ``c``, -3.3 % of Ip on the golden), so the scale is solved exactly,
    ``s = (Ip_target - c) / int(w*J)`` -- a plain ratio ``Ip_target/Ip[J]``
    would miss by ``(1-s)*c``, measured -0.19..-0.26 % of Ip.

    Returns ``(scaled_target, factor)``; factor is 1.0 (target untouched) if
    the measure is unavailable, with a loud note, so the corrector still runs
    rather than not at all.
    """
    from scipy.integrate import trapezoid
    from .utils import (fsa_current_geometry, Ip_fsa_weights,
                        eq_jphi_profile)
    psi = np.asarray(psi_N, dtype=float)
    t = np.asarray(target_jphi, dtype=float)
    try:
        geom = fsa_current_geometry(mygs, psi)
        probe = eq_jphi_profile(geom, "jphi-linterp", eq=mygs)
        sign = 1.0 if float(np.dot(probe, t)) > 0.0 else -1.0
        w, c = Ip_fsa_weights(geom, convention="jphi-linterp",
                              pprime_sign=sign)
        lin = float(trapezoid(w * t, psi))
        Ip_t = lin + c
    except Exception as exc:          # measure unavailable: do not block
        print(f"  [{label}] WARN: target Ip measure failed ({exc}); "
              f"corrector target NOT renormalised (issue #29)", flush=True)
        return t.copy(), 1.0
    if not (np.isfinite(lin) and np.isfinite(c) and lin != 0.0
            and np.isfinite(Ip_target)):
        print(f"  [{label}] WARN: target Ip measure non-finite "
              f"(lin={lin}, c={c}); corrector target NOT renormalised "
              f"(issue #29)", flush=True)
        return t.copy(), 1.0
    factor = (float(Ip_target) - c) / lin
    if not np.isfinite(factor) or factor <= 0.0:
        print(f"  [{label}] WARN: target Ip scale {factor} is not positive; "
              f"corrector target NOT renormalised (issue #29)", flush=True)
        return t.copy(), 1.0
    print(f"  [{label}] target renormalised to Ip: carried "
          f"{100.0 * (Ip_t / float(Ip_target) - 1.0):+.3f}% of Ip_target, "
          f"scaled x{factor:.6f} (affine-exact; issue #29)", flush=True)
    return t * factor, factor


def _corrective_jphi_iteration(mygs, psi_N, target_jphi, pp_prof,
                                Ip_target, pax_target, psi_pad,
                                min_iters=2, max_iters=8,
                                rtol=0.05, verbose=True,
                                damping=1.0, protect_state=False):
    r"""Iterate TokaMaker input j_phi until the output matches a target.

    Uses Newton correction: ``input += (target - output)`` each step.
    Starts with *min_iters*, then checks whether the edge spike RMS
    is still improving by more than *rtol* relative per step.  Stops
    when converged or *max_iters* is reached.

    Parameters
    ----------
    mygs : TokaMaker
        GS solver (in a solved state with current profiles set).
    psi_N : ndarray
        Normalised flux grid.
    target_jphi : ndarray
        Target j_phi profile [A/m²] (e.g. j_inductive + spike_profile).
    pp_prof : dict
        Pressure gradient profile dict for ``set_profiles``.
    Ip_target : float
        Plasma current target [A].
    pax_target : float
        On-axis pressure target [Pa].
    psi_pad : float
        LCFS padding.
    min_iters : int
        Minimum iterations before checking convergence (default 2).
    max_iters : int
        Maximum iterations (default 8).
    rtol : float
        Relative improvement threshold — stop if
        ``|rms_new - rms_old| / rms_old < rtol`` (default 0.05 = 5%).
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    j_phi_output : ndarray
        Converged GS output j_phi [A/m²].
    n_iters : int
        Number of iterations performed.
    edge_rms_history : list of float
        Edge RMS per iteration [A/m²].
    """
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS

    npsi = len(psi_N)
    edge_mask = psi_N > 0.9
    j_phi_input = target_jphi.copy()
    edge_rms_history = []
    # keep-best bookkeeping (protect_state=True): the imas anchor target comes
    # from ANOTHER code's flux geometry and may not be exactly achievable, so
    # undamped Newton steps can oscillate/diverge; track the best full-profile
    # RMS state and restore it at the end instead of trusting the last iterate.
    # Gate on BOTH halves of the snapshot/restore pair.  Every snapshot taken
    # here is eventually consumed by `replace_eq` (the per-iteration
    # solve-failure restore and the final keep-best restore), so a solver
    # object exposing only `copy_eq` would pass a copy_eq-only gate and then
    # raise AttributeError on the restore -- turning a recoverable solve
    # failure into a hard crash.  Matches the both-methods gate already used
    # at the warm-start snapshot site further down this module.
    _can_snap = (protect_state and hasattr(mygs, "copy_eq")
                 and hasattr(mygs, "replace_eq"))
    best = {"rms": np.inf, "eq": None, "out": None}
    full_rms_history = []
    # Seed the output with the UNCORRECTED input.  If the very first solve
    # raises, the handler below breaks out before `j_phi_output` is ever
    # assigned, and the return statement then raised NameError -- masking a
    # solve failure behind an unrelated-looking crash.  Seeding it means that
    # case degrades to "no correction was applied", which is the truthful
    # answer and matches the non-fatal intent of the break; the empty
    # `edge_rms_history` and the warning below tell the caller it happened.
    # (A later-iteration failure is unaffected: it keeps the last good
    # iterate, exactly as before.)
    j_phi_output = j_phi_input.copy()
    it = -1

    for it in range(max_iters):
        ffp = {"type": "jphi-linterp", "y": j_phi_input.copy(), "x": psi_N}
        mygs.set_targets(Ip=Ip_target, pax=pax_target)
        mygs.set_profiles(pp_prof=pp_prof, ffp_prof=ffp)
        _snap = mygs.copy_eq() if _can_snap else None
        try:
            mygs.solve()
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"  [jphi_corr iter {it+1}] solve failed: {e}")
            if it == 0:
                # Not verbose-gated: no iterate ever succeeded, so the caller is
                # getting its own input back with no correction applied at all.
                # That is a materially different result from a converged one and
                # must not be inferable only from an empty RMS history.
                print(f"  [jphi_corr] WARNING: the FIRST corrective solve "
                      f"failed ({e}); returning the uncorrected input j_phi "
                      f"-- no corrective iteration was applied")
            if _snap is not None:
                mygs.replace_eq(source_eq=_snap)   # do not leave the diverged state
            break

        _, f, fp, _, pp = mygs.get_profiles(npsi=npsi, psi_pad=psi_pad)
        _, _, ravgs, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
        j_phi_output = get_jphi_from_GS(f * fp, pp, q_ravg(ravgs, "<R>"), q_ravg(ravgs, "<1/R>"))

        diff = j_phi_output - target_jphi
        rms_edge = float(np.sqrt(np.mean(diff[edge_mask]**2)))
        edge_rms_history.append(rms_edge)
        _kept = None
        if _can_snap:
            rms_full = float(np.sqrt(np.mean(diff**2)))
            full_rms_history.append(rms_full)
            _kept = rms_full < best["rms"]
            if _kept:
                best.update(rms=rms_full, eq=mygs.copy_eq(), out=j_phi_output.copy())

        if verbose:
            # Report the FULL-domain RMS too when keep-best is on: the stopping
            # rule is on the edge, but the state that gets kept is chosen on the
            # full domain, so a log showing only the edge cannot explain which
            # iterate was landed on (issue #25).
            if _can_snap:
                print(f"  [jphi_corr iter {it+1}] edge RMS = "
                      f"{rms_edge/1e6:.6f} MA/m², full RMS = "
                      f"{full_rms_history[-1]/1e6:.6f} MA/m² "
                      f"({'KEPT (new best)' if _kept else 'discarded'}; "
                      f"best {best['rms']/1e6:.6f})")
            else:
                print(f"  [jphi_corr iter {it+1}] edge RMS = {rms_edge/1e6:.6f} MA/m²")

        # Check convergence after min_iters
        if it >= min_iters - 1 and len(edge_rms_history) >= 2:
            prev_rms = edge_rms_history[-2]
            if prev_rms > 0:
                rel_change = abs(rms_edge - prev_rms) / prev_rms
                if rel_change < rtol:
                    if verbose:
                        print(f"  [jphi_corr] converged at iter {it+1} "
                              f"(rel_change={rel_change:.4f} < {rtol})")
                    break

        # Newton correction (optionally damped)
        j_phi_input = j_phi_input + damping * (target_jphi - j_phi_output)
        j_phi_input = np.maximum(j_phi_input, 0.0)

    if _can_snap and best["eq"] is not None:
        mygs.replace_eq(source_eq=best["eq"])      # land on the best state seen
        j_phi_output = best["out"]
        if verbose:
            # The kept-RMS sequence is monotone non-increasing BY CONSTRUCTION;
            # printing it is what lets a run be checked rather than trusted.
            _kept_traj = np.minimum.accumulate(np.asarray(full_rms_history))
            print(f"  [jphi_corr] keep-best landed on full RMS "
                  f"{best['rms']/1e6:.6f} MA/m² (per-iterate "
                  + " -> ".join(f"{r/1e6:.6f}" for r in full_rms_history)
                  + "; kept "
                  + " -> ".join(f"{r/1e6:.6f}" for r in _kept_traj) + ")")

    return j_phi_output, it + 1, edge_rms_history


# ---- j_phi profile classifier ----
def classify_jphi_profile(psi_N, eqdsk_jphi, spike_profile,
                          edge_psi_min=0.5, prominence_frac=0.15):
    r"""Classify the edge current profile to determine reconstruction strategy.

    Parameters
    ----------
    psi_N : ndarray
        Normalised poloidal flux grid.
    eqdsk_jphi : ndarray
        Toroidal current density from the geqdsk [A/m²].
    spike_profile : ndarray
        Isolated edge bootstrap spike from ``analyze_bootstrap_edge_spike``
        [A/m²].  Flat shelf in the core, rising spike at the edge.
    edge_psi_min : float
        Inner boundary of the edge search window (default 0.5).
    prominence_frac : float
        Minimum peak prominence as a fraction of the edge range
        (default 0.15).

    Returns
    -------
    mode : str
        One of ``'H_mode'``, ``'Lmode_like_jphi'``, ``'L_mode'``.
    metrics : dict
        Edge spike metrics for reconstruction quality tracking.
    """
    from scipy.signal import find_peaks

    metrics = {}
    edge_mask = psi_N >= edge_psi_min

    # --- Check Sauter spike ---
    spike_edge = spike_profile[edge_mask]
    psi_edge = psi_N[edge_mask]

    # Robust two-pass edge-peak detection (hardened 2026-07, user-approved).
    # The height reference used to be spike_profile[0] -- meaningful when the
    # profile was analyze_bootstrap_edge_spike's flat-shelf output
    # (isolate_edge_jBS=True), but with the full-Sauter default it is the
    # numerically fragile collapsed axis point, and on weak-pedestal shots
    # (weak-pedestal case: peak within ~2 permille of it) detection flipped on
    # run-to-run jitter.  Mirror analyze_bootstrap_edge_spike instead:
    #   pass 1 -- locate candidate edge peaks with a liberal prominence-only
    #             search (5% of the edge maximum, as in OFT);
    #   pass 2 -- take the VALLEY (profile minimum between the edge-window
    #             start and the tallest candidate -- the core-hump/edge-spike
    #             saddle, OFT's shelf level) as the height reference, and
    #             re-detect with the original height + prominence criteria
    #             measured from that valley.
    edge_max = float(np.max(spike_edge)) if spike_edge.size else 0.0
    if edge_max > 0:
        cand, _ = find_peaks(spike_edge, prominence=0.05 * edge_max)
    else:
        cand = np.array([], dtype=int)
    if len(cand):
        _tallest = cand[np.argmax(spike_edge[cand])]
        _between = (psi_N >= edge_psi_min) & (psi_N < psi_edge[_tallest])
        shelf_val = (float(np.min(spike_profile[_between]))
                     if _between.any() else float(spike_edge[0]))
        spike_range = edge_max - shelf_val
        if spike_range > 0:
            peaks_s, _ = find_peaks(spike_edge, height=shelf_val,
                                    prominence=prominence_frac * spike_range)
        else:
            peaks_s = np.array([], dtype=int)
    else:
        peaks_s = np.array([], dtype=int)

    if len(peaks_s) == 0:
        # No Sauter EDGE spike found.  Distinguish truly negligible
        # bootstrap (classic L-mode: the split is zeroed downstream) from
        # a profile with real bootstrap current whose edge peak merely
        # failed the height criterion.  Zeroing the split for the latter
        # puts the bootstrap into j_inductive AND lets the per-draw SWB
        # recompute add it again -- a double-counted bootstrap in every
        # draw.  Caught by the sigma=0 guard on a weak-pedestal case whose
        # edge peak (~0.108 MA/m^2) sits within ~2 permille of
        # the height threshold (shelf_val = spike[0], the numerically
        # fragile collapsed axis point), so run-to-run jitter in the axis
        # point flips the edge-peak detection -- the old-vs-new j_BS
        # profiles themselves are essentially identical (np.gradient vs
        # PCHIP OFT builds agree; see jbs_pchip_vs_gradient_5307.png).
        sauter_max = float(np.max(spike_profile))
        jphi_scale = float(np.max(np.abs(eqdsk_jphi)))
        if jphi_scale > 0 and sauter_max >= 0.05 * jphi_scale:
            metrics['spike_height_sauter'] = sauter_max
            metrics['spike_psiN_sauter'] = float(
                psi_N[int(np.argmax(spike_profile))])
            metrics['spike_height_geqdsk'] = None
            metrics['spike_psiN_geqdsk'] = None
            metrics['spike_height_ratio'] = None
            metrics['spike_psiN_offset'] = None
            print(f"[classify] Lmode_like_jphi — no Sauter EDGE spike, but "
                  f"the bootstrap profile is significant (max "
                  f"{sauter_max/1e6:.4f} MA/m² = "
                  f"{100*sauter_max/jphi_scale:.1f}% of peak j_phi at "
                  f"psi_N={metrics['spike_psiN_sauter']:.3f}); keeping the "
                  f"full Sauter profile in the split")
            return 'Lmode_like_jphi', metrics
        metrics['spike_height_sauter'] = 0.0
        metrics['spike_psiN_sauter'] = None
        metrics['spike_height_geqdsk'] = None
        metrics['spike_psiN_geqdsk'] = None
        metrics['spike_height_ratio'] = None
        metrics['spike_psiN_offset'] = None
        print(f"[classify] L_mode — no Sauter edge spike and negligible "
              f"bootstrap profile")
        return 'L_mode', metrics

    # Sauter spike exists — record its peak
    best_s = peaks_s[np.argmax(spike_edge[peaks_s])]
    metrics['spike_height_sauter'] = float(spike_edge[best_s])
    metrics['spike_psiN_sauter'] = float(psi_edge[best_s])

    # --- Check geqdsk for an edge peak ---
    geqdsk_edge = eqdsk_jphi[edge_mask]
    geqdsk_baseline = geqdsk_edge[0]  # value at edge_psi_min
    geqdsk_range = np.max(geqdsk_edge) - geqdsk_baseline

    # Use prominence only — no height filter. The edge spike may be
    # below the core value (j_phi decreases from core to edge) but
    # still be a significant local peak.
    geqdsk_max = np.max(geqdsk_edge)
    if geqdsk_max > 0:
        min_prom_g = prominence_frac * geqdsk_max
        peaks_g, props_g = find_peaks(geqdsk_edge, prominence=min_prom_g)
        # Filter to psi_N > 0.85 (true edge peaks, not shoulder of core)
        if len(peaks_g) > 0:
            far_edge = psi_edge[peaks_g] > 0.85
            if np.any(far_edge):
                peaks_g = peaks_g[far_edge]
            else:
                peaks_g = np.array([], dtype=int)
    else:
        peaks_g = np.array([], dtype=int)

    if len(peaks_g) > 0:
        # geqdsk has an edge peak → H_mode
        best_g = peaks_g[np.argmax(geqdsk_edge[peaks_g])]
        metrics['spike_height_geqdsk'] = float(geqdsk_edge[best_g])
        metrics['spike_psiN_geqdsk'] = float(psi_edge[best_g])
        metrics['spike_height_ratio'] = (
            metrics['spike_height_sauter'] / metrics['spike_height_geqdsk']
        )
        metrics['spike_psiN_offset'] = (
            metrics['spike_psiN_sauter'] - metrics['spike_psiN_geqdsk']
        )
        print(f"[classify] H_mode — geqdsk peak at psi_N={metrics['spike_psiN_geqdsk']:.4f} "
              f"({metrics['spike_height_geqdsk']/1e6:.4f} MA/m²), "
              f"Sauter peak at psi_N={metrics['spike_psiN_sauter']:.4f} "
              f"({metrics['spike_height_sauter']/1e6:.4f} MA/m²), "
              f"height ratio={metrics['spike_height_ratio']:.2f}, "
              f"psiN offset={metrics['spike_psiN_offset']:.4f}")
        return 'H_mode', metrics
    else:
        # geqdsk has no edge peak → Lmode_like_jphi
        metrics['spike_height_geqdsk'] = None
        metrics['spike_psiN_geqdsk'] = None
        metrics['spike_height_ratio'] = None
        metrics['spike_psiN_offset'] = None
        print(f"[classify] Lmode_like_jphi — no geqdsk edge peak, "
              f"Sauter spike at psi_N={metrics['spike_psiN_sauter']:.4f} "
              f"({metrics['spike_height_sauter']/1e6:.4f} MA/m²)")
        return 'Lmode_like_jphi', metrics


# ---- Coil-current drift metric (in-spec) -------------------------------------
# Additive coil-current measurement+eddy-current uncertainty for the VSC in-spec
# channel metric, carried as a denominator floor [A]. ASSUMPTION: no published
# DIII-D coil-current measurement-noise figure was found; the literature puts
# Rogowski/PF current measurement at ~0.1-1%, with eddy currents in the vessel/
# passive structure contributing a current-INDEPENDENT (additive) term. So this
# is a deliberately conservative placeholder -- at the 2% inspec gain it
# corresponds to a ~200 A additive tolerance. Replace with the real transducer
# offset + eddy-equivalent figure when available (see README "VSC drift metric").
_COIL_DRIFT_DENOM_FLOOR_A = 10000.0


def _coil_drift_pct(cur, baseline):
    r"""Per-coil current drift [%] = ``100 * (I_draw - I_base) / max(|I_base|, 1)``.

    Used for the SOLVE-time homotopy bounds/saturation (per coil, per power
    supply) and for the non-VSC F-coil in-spec metric -- there the measurement
    uncertainty is gain-dominated (~2% of 35-180 kA), so a relative % is the
    right scale. The anti-series VSC pair is scored separately by
    :func:`_vsc_channel_drift_pct` (per-coil % breaks on its near-zero coil).
    Signed; caller takes abs()/max().
    """
    return {n: (float(cur[n]) - baseline[n]) / max(abs(baseline[n]), 1.0) * 100.0
            for n in baseline}


def _vsc_channel_drift_pct(cur, baseline, vsc_set,
                           denom_floor_A=_COIL_DRIFT_DENOM_FLOOR_A):
    r"""VSC-pair in-spec drift [%] via common-mode + differential channels.

    The anti-series VSC pair (F9A, F9B) carries a vertical-CONTROL common-mode
    current ``I_cm = (F9A - F9B)/2`` and a shaping differential
    ``I_df = (F9A + F9B)/2``. Both are linear combinations of two INDEPENDENTLY-
    measured coils, so each channel's measurement tolerance PROPAGATES IN
    QUADRATURE from the per-coil uncertainties ``sigma_coil = gain*|I_coil|``
    (the ±1/2 channel coefficients give the same sigma for both channels)::

        sigma_VSC = sqrt(|F9A|^2 + |F9B|^2) / 2  (+ denom_floor_A, the offset/gain)
        drift     = 100 * max(|dI_cm|, |dI_df|) / sigma_VSC

    (the gain cancels into the inspec_VSC_max threshold, so it is not needed
    here). Because ``sigma_VSC`` is built from the COIL magnitudes, it never goes
    near zero -- robust to BOTH a near-zero coil (F9A on the FUSE/IMAS baseline)
    AND a near-zero common-mode baseline (a co-current pair, e.g. the geqdsk
    baseline where F9A≈F9B). Gating either channel against its OWN baseline
    blows up in one of those cases; the propagated sigma does not. The
    differential channel still catches a genuinely asymmetric/same-sign
    excursion. Falls back to the per-coil max when the pair is not two coils.
    """
    if len(vsc_set) != 2:
        d = _coil_drift_pct(cur, baseline)
        return float(max((abs(d[n]) for n in vsc_set), default=0.0))
    a, b = vsc_set
    dA = float(cur[a]) - baseline[a]
    dB = float(cur[b]) - baseline[b]
    cm = abs((dA - dB) / 2.0)
    df = abs((dA + dB) / 2.0)
    sigma = float(np.hypot(baseline[a], baseline[b])) / 2.0 + denom_floor_A
    return float(100.0 * max(cm, df) / sigma) if sigma > 0 else 0.0


# ---- Shelf-blend decomposition helper ----
def _shelf_blend_decompose(psi_N, j_phi_total, spike_profile,
                           eqdsk_jphi=None):
    r"""Decompose j_phi into j_inductive + spike_profile.

    In the core (where the spike shelf is flat), j_inductive = j_phi - spike.
    At the edge (beyond the shelf), j_inductive is replaced by an
    optimised cubic Hermite that:

    - matches value and derivative of the core j_inductive at the
      exact shelf-end index (C1 join, no blend window),
    - minimises ``||j_ind_bridge + spike - eqdsk_jphi||²`` in the
      edge region by optimising the two free slopes,
    - is constrained to be monotonically decreasing and non-negative,
    - arrives at ``max(eqdsk_jphi[-1] - spike_profile[-1], 0)``
      at psi_N = 1.

    Parameters
    ----------
    psi_N : ndarray
        Normalised flux grid.
    j_phi_total : ndarray
        Total toroidal current density [A/m²].
    spike_profile : ndarray
        Isolated edge bootstrap spike [A/m²] (flat shelf + edge peak).
    eqdsk_jphi : ndarray or None
        Original geqdsk j_phi [A/m²].  Used to set the edge boundary
        value and as the optimisation target.  If ``None``, a simple
        linear taper is used.

    Returns
    -------
    j_inductive : ndarray
        Inductive component (non-negative, smooth edge taper).
    shelf_psi : float
        psi_N location where the shelf ends.
    """
    j_ind_raw = j_phi_total - spike_profile

    # Find where the shelf ends
    shelf_val = spike_profile[0]
    shelf_end = 0
    for i in range(1, len(spike_profile)):
        if abs(spike_profile[i] - shelf_val) / max(abs(shelf_val), 1e-30) < 1e-6:
            shelf_end = i
        else:
            break
    shelf_psi = psi_N[shelf_end]

    # Edge target: spike[-1] + j_ind[-1] = eqdsk[-1], unless spike dominates
    if eqdsk_jphi is not None:
        edge_target = max(eqdsk_jphi[-1] - spike_profile[-1], 0.0)
    else:
        edge_target = 0.0

    # Value at shelf end (from core subtraction)
    val_at_shelf = max(j_ind_raw[shelf_end], 0.0)

    # Estimate the slope that j_inductive needs at the shelf end so that
    # the TOTAL j_phi = j_ind + spike has a smooth derivative there.
    #
    # On the core side: dj_phi/dpsi = dj_ind/dpsi + 0 (spike is flat).
    # On the edge side: dj_phi/dpsi = dj_ind/dpsi + dspike/dpsi.
    # For C1 in total j_phi: the Hermite's dj_ind/dpsi at t=0 should
    # equal the core dj_phi/dpsi minus the spike's derivative just
    # past the shelf end.
    #
    # Core total j_phi slope (5-point stencil):
    n_stencil = min(5, shelf_end)
    if n_stencil >= 2:
        _sl_idx = shelf_end - n_stencil
        _sl_dy = j_ind_raw[shelf_end] - j_ind_raw[_sl_idx]
        _sl_dx = psi_N[shelf_end] - psi_N[_sl_idx]
        core_jphi_slope = _sl_dy / _sl_dx if _sl_dx > 0 else 0.0
    else:
        core_jphi_slope = 0.0

    # Spike derivative just past the shelf end
    if shelf_end < len(psi_N) - 2:
        spike_slope_at_edge = ((spike_profile[shelf_end + 2] - spike_profile[shelf_end])
                               / (psi_N[shelf_end + 2] - psi_N[shelf_end]))
    else:
        spike_slope_at_edge = 0.0

    # j_inductive start slope = core total slope - spike slope
    # so that dj_phi/dpsi = dj_ind/dpsi + dspike/dpsi = core total slope
    core_slope_est = core_jphi_slope - spike_slope_at_edge

    interval = 1.0 - shelf_psi

    if interval <= 0 or eqdsk_jphi is None:
        # Fallback: linear taper
        j_ind = j_ind_raw.copy()
        for i in range(len(psi_N)):
            if psi_N[i] > shelf_psi:
                t = (psi_N[i] - shelf_psi) / max(interval, 1e-30)
                j_ind[i] = val_at_shelf * (1 - t) + edge_target * t
        return np.maximum(j_ind, 0.0), shelf_psi

    # Hermite bridge: 4 DOF, 3 fixed, 1 optimised.
    #   Start value = val_at_shelf (C0 match, fixed)
    #   Start slope = core_slope_est (C1 match, fixed)
    #   End value   = edge_target (fixed)
    #   End slope   = optimised to minimise ||j_ind + spike - eqdsk||²
    edge_mask = psi_N >= shelf_psi
    psi_edge = psi_N[edge_mask]
    spike_edge = spike_profile[edge_mask]
    eqdsk_edge = eqdsk_jphi[edge_mask]

    m0_fixed = core_slope_est * interval  # scaled start slope (exact C1)

    def _build_hermite_arr(m1_scaled):
        """Build Hermite on edge grid with fixed m0."""
        t = (psi_edge - shelf_psi) / interval
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * val_at_shelf + h10 * m0_fixed + h01 * edge_target + h11 * m1_scaled

    def _cost(m1_scaled):
        bridge = np.maximum(_build_hermite_arr(m1_scaled), 0.0)
        total = bridge + spike_edge
        residual = total - eqdsk_edge
        cost = np.mean(residual**2)
        # Penalty for non-monotonic bridge
        dbridge = np.diff(bridge)
        violations = np.sum(np.maximum(dbridge, 0.0)**2)
        return cost + 10.0 * violations

    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        _cost, bounds=(-5.0 * val_at_shelf, 5.0 * val_at_shelf),
        method='bounded',
    )
    m1_opt = res.x
    m0_opt = m0_fixed

    # Build final bridge on the full grid (inclusive of shelf_end index)
    j_ind = j_ind_raw.copy()
    for i in range(len(psi_N)):
        if psi_N[i] >= shelf_psi:
            t = (psi_N[i] - shelf_psi) / interval
            t = min(t, 1.0)
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            j_ind[i] = (h00 * val_at_shelf + h10 * m0_opt
                        + h01 * edge_target + h11 * m1_opt)

    return np.maximum(j_ind, 0.0), shelf_psi


# ---- Spline-based inductive profile fitting ----
def fit_inductive_profile(mygs, eqdsk_jtor, j_BS_isolated, psi_N, psi_pad,
                            baseline_li_proxy,
                            k=3, psi_bridge=0.99,
                            rescale_j_BS=False,
                            shelf_psi_N=0.0,
                            core_exact_psi=0.30):
    r"""Fit a smooth inductive current profile and scale it to match
    a target cylindrical :math:`l_i` proxy.

    Fits a ``scipy.interpolate.UnivariateSpline`` to
    ``eqdsk_jtor - j_BS_isolated``, enforcing a zero edge anchor and
    optionally bridging over the edge region with *psi_bridge*.  Scales
    the resulting inductive basis profile (and optionally the bootstrap
    current) so that the total :math:`j_\phi` reproduces
    *baseline_li_proxy*.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker Grad-Shafranov solver object.
    eqdsk_jtor : ndarray
        1-D target total :math:`j_{\rm tor}` from the geqdsk [A m\ :sup:`-2`].
    j_BS_isolated : ndarray
        1-D isolated bootstrap current profile [A m\ :sup:`-2`].
    psi_N : ndarray
        1-D normalised poloidal flux grid.
    psi_pad : float
        Padding inside the LCFS for the :math:`l_i` proxy calculation.
    baseline_li_proxy : float
        Target cylindrical :math:`l_i` proxy value.
    k : int
        Spline order (default 3).
    psi_bridge : float
        :math:`\hat{\psi}` above which data are replaced by the edge
        anchor point (default 0.99).
    rescale_j_BS : bool
        If ``True``, jointly optimise a bootstrap rescaling factor to
        minimise the RMS residual against *eqdsk_jtor*.
        ``False`` (default) scales the inductive profile only.
    shelf_psi_N : float
        If > 0, apply a flat shelf to *j_BS_isolated* for
        :math:`\hat{\psi} <` *shelf_psi_N*, using the value of
        *j_BS_isolated* at that location.  ``0`` disables the shelf.
    core_exact_psi : float
        For :math:`\hat{\psi} <` *core_exact_psi*, take the inductive
        basis from the RAW residual ``eqdsk_jtor - j_BS`` (dense
        subsample, smooth blend to the smoothing spline at the seam)
        instead of the globally-smoothed spline.  The core residual is
        smooth (both inputs are), but its anti-hump -- the mirror of the
        Sauter core bump peaking near :math:`\hat{\psi} \sim 0.05` -- is
        washed out by the global smoothing factor, which left a ±2-3%
        S-wiggle in the total core :math:`j_\phi` vs the g-file.  ``0``
        disables (previous behaviour).

    Returns
    -------
    dict
        ``'j_inductive_fit'`` -- fitted inductive profile (scaled)
            [A m\ :sup:`-2`].
        ``'j_phi_fit'`` -- total :math:`j_\phi = j_{\rm ind} + b_{\rm scale}\,j_{\rm BS}`
            [A m\ :sup:`-2`].
        ``'fit_li'`` -- :math:`l_i` proxy of ``j_phi_fit``.
        ``'ind_scale'`` -- inductive scaling factor applied.
        ``'bs_scale'`` -- bootstrap scaling factor (1.0 when
            ``rescale_j_BS=False``).
        ``'j_BS_used'`` -- *j_BS_isolated* after optional shelving.
        ``'spline'`` -- the fitted ``UnivariateSpline`` object.
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import brentq, minimize_scalar

    j_BS_work = j_BS_isolated.copy()

    # ---- Optional shelf on j_BS_isolated ----
    if shelf_psi_N > 0.0:
        shelf_idx = np.searchsorted(psi_N, shelf_psi_N)
        shelf_idx = min(shelf_idx, len(psi_N) - 1)
        shelf_val = j_BS_work[shelf_idx]
        j_BS_work[:shelf_idx] = shelf_val

    # ---- Build the spline basis (at bs_scale = 1) ----
    residual = eqdsk_jtor - j_BS_work
    mask_core = psi_N <= psi_bridge

    edge_target = eqdsk_jtor[-1] - j_BS_work[-1]  # used when rescale_j_BS
    psi_trusted = np.concatenate([psi_N[mask_core], [1.0]])
    res_trusted = np.concatenate([residual[mask_core],
                                    [edge_target if rescale_j_BS else 0.0]])

    # Use a smoothing spline followed by PCHIP to eliminate ringing.
    # Step 1: smooth the residual with a generous smoothing factor
    # to remove high-frequency noise while preserving the overall shape.
    # Step 2: evaluate on a coarser grid and use PchipInterpolator
    # (shape-preserving, monotonicity-respecting, C1) for the final
    # profile on the full psi_N grid.
    from scipy.interpolate import PchipInterpolator

    _s_factor = len(psi_trusted) * np.var(res_trusted) * 0.1
    _smooth_spline = UnivariateSpline(psi_trusted, res_trusted, k=k,
                                       s=_s_factor)

    # Subsample to ~32 points for PCHIP (enough to capture the shape,
    # few enough to avoid oscillation)
    _n_sub = min(32, len(psi_N))
    _psi_sub = np.linspace(psi_N[0], psi_N[-1], _n_sub)
    if core_exact_psi and core_exact_psi > 0.0:
        # Densify the core and take its values from the raw residual: the
        # heavy global smoothing exists for pedestal/edge noise, but in the
        # core it flattens the (smooth, well-resolved) anti-hump that must
        # mirror the Sauter j_BS bump, leaving an S-wiggle in the total.
        # Blend raw -> spline over the outer ~third of the core window so
        # the PCHIP sees no seam.
        _psi_core = np.linspace(psi_N[0], core_exact_psi, 16)
        _psi_sub = np.unique(np.concatenate([_psi_core, _psi_sub]))
        _raw_sub = np.interp(_psi_sub, psi_trusted, res_trusted)
        _spl_sub = _smooth_spline(_psi_sub)
        _w_core = np.clip((core_exact_psi - _psi_sub)
                          / (0.35 * core_exact_psi), 0.0, 1.0)
        _res_sub = _w_core * _raw_sub + (1.0 - _w_core) * _spl_sub
    else:
        _res_sub = _smooth_spline(_psi_sub)
    _res_sub = np.maximum(_res_sub, 0.0)

    _pchip = PchipInterpolator(_psi_sub, _res_sub)
    j_inductive_basis = _pchip(psi_N)
    j_inductive_basis = np.maximum(j_inductive_basis, 0.0)

    # ---- Helper: solve ind_scale for a given bs_scale via brentq ----
    def _solve_ind_scale(bs_scale):
        def _li_residual(scale):
            j_phi = scale * j_inductive_basis + bs_scale * j_BS_work
            return calc_cylindrical_li_proxy(mygs, j_phi, psi_pad) - baseline_li_proxy

        s_lo, s_hi = 0.5, 2.0
        f_lo, f_hi = _li_residual(s_lo), _li_residual(s_hi)
        for _ in range(10):
            if f_lo * f_hi < 0:
                break
            s_lo *= 0.5
            s_hi *= 2.0
            f_lo, f_hi = _li_residual(s_lo), _li_residual(s_hi)

        if f_lo * f_hi < 0:
            return brentq(_li_residual, s_lo, s_hi, xtol=1e-6)
        else:
            return 1.0  # fallback

    if not rescale_j_BS:
        # ---- v1: scale inductive profile only ----
        ind_scale = _solve_ind_scale(1.0)
        bs_scale_out = 1.0
    else:
        # ---- v2: jointly optimise bs_scale and ind_scale ----
        def _rms_for_bs_scale(bs_scale):
            isc = _solve_ind_scale(bs_scale)
            j_phi = isc * j_inductive_basis + bs_scale * j_BS_work
            return np.sqrt(np.mean((j_phi - eqdsk_jtor)**2))

        result = minimize_scalar(_rms_for_bs_scale, bounds=(0.0, 4.0),
                                    method='bounded', options={'xatol': 1e-4})
        bs_scale_out = result.x
        ind_scale = _solve_ind_scale(bs_scale_out)

    j_inductive_fit = ind_scale * j_inductive_basis
    j_phi_fit = j_inductive_fit + bs_scale_out * j_BS_work
    fit_li = calc_cylindrical_li_proxy(mygs, j_phi_fit, psi_pad)

    return {
        'j_inductive_fit': j_inductive_fit,
        'j_phi_fit': j_phi_fit,
        'fit_li': fit_li,
        'ind_scale': ind_scale,
        'bs_scale': bs_scale_out,
        'j_BS_used': j_BS_work,
        'spline': _pchip,
    }

def _achieved_jphi_fsa(mygs, psi_N, psi_pad=1e-3, sign_ref=None):
    """ACHIEVED flux-surface-averaged toroidal current of the CONVERGED solve.

    ``get_jphi_from_GS`` on the live equilibrium (the same formula the recon
    corrective loop matches), interpolated onto ``psi_N``. This is the current
    the stored eqdsk actually carries -- as opposed to the prescribed target
    profile, which a single-pass jphi-linterp solve reproduces only
    approximately (measured +3.2-3.4%% high over psi_N 0.2-0.9 on the IMAS
    path). ``sign_ref`` aligns the sign convention to the profile it replaces.
    """
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS
    psi_N = np.asarray(psi_N, dtype=float)
    ps, f, fp, _, pp = mygs.get_profiles(npsi=len(psi_N), psi_pad=psi_pad)
    _, _, ravgs, _, _, _ = mygs.get_q(npsi=len(psi_N), psi_pad=psi_pad)
    ps = np.asarray(ps, dtype=float)
    if ps.size and (ps.max() > 1.5 or ps.min() < -0.5):
        ps = (ps - ps.min()) / (ps.max() - ps.min())
    from .physics import q_ravg
    j = get_jphi_from_GS(np.asarray(f, float) * np.asarray(fp, float),
                         np.asarray(pp, float),
                         np.asarray(q_ravg(ravgs, "<R>"), float),
                         np.asarray(q_ravg(ravgs, "<1/R>"), float))
    j = np.interp(psi_N, ps, np.asarray(j, dtype=float))
    if sign_ref is not None and \
            np.nanmedian(j) * np.nanmedian(np.asarray(sign_ref, float)) < 0:
        j = -j
    return j


def _swb_debug():
    """Single switch for the SWB debugging instrumentation.

    ``BOUQUET_SWB_DEBUG=1`` enables, in one place, what used to be separate
    SWB_STATE_DUMP / SWB_VERBOSE / PROFILE toggles: the per-draw ``[SWB-diag]``
    state prints, verbose solve_with_bootstrap output, SWB wall-time, and the
    pre/post-state ``.npz`` dumps (written under the system temp dir, see
    :func:`_swb_dump_path`) that let a failing draw be replayed offline.
    Read at call time so it can be flipped mid-session in a notebook.
    """
    return os.environ.get('BOUQUET_SWB_DEBUG', '0') == '1'


def _swb_dump_path(name):
    """Path for an SWB debug dump: ``<tempdir>/bouquet_<name>.npz``."""
    return os.path.join(tempfile.gettempdir(), f"bouquet_{name}.npz")


def _ket_stage_diag(mygs, tag, extra=""):
    """Guarded per-stage trace (KET_STAGE_DIAG=1): log F9B coil, magnetic axis,
    and Ip at a draw-pipeline stage, to localize where the systematic coil /
    boundary drift is injected (recon-anchor vs find_optimal_scale vs corrective).
    No-op unless the env var is set, so it is inert in normal runs."""
    if os.environ.get("KET_STAGE_DIAG", "0") != "1":
        return
    try:
        _cc, _ = mygs.get_coil_currents()
        _op = mygs.o_point
        print(f"  [STAGE-DIAG {tag}] F9B={_cc.get('F9B', float('nan')):+.1f}A "
              f"axis=({_op[0]:.4f},{_op[1]:+.5f}) Ip={mygs.get_globals()[0]:+.0f}{extra}",
              flush=True)
    except Exception as _e:
        print(f"  [STAGE-DIAG {tag}] failed: {_e}", flush=True)


def _swb_jbs_to_toroidal(mygs, j_bs_swb, psi_pad):
    """Convert ``solve_with_bootstrap``'s j_BS output to toroidal convention.

    SWB computes the Redl/Sauter bootstrap as the FSA *parallel* current
    ``<j_BS.B>`` and projects it to A/m^2 by a zeroth-order division by the
    toroidal field at the average radius (``j_BS_neo * R_avg/F``, with its own
    ``# to-do: project j_BS_parallel to j_phi more accurately?``). Every other
    current profile in bouquet is the FSA toroidal density
    ``j_tor = <j_phi/R>/<1/R>`` (what TokaMaker's jphi-linterp / flux_integral
    consume), so mixing the two conventions misallocates the j_BS / j_inductive
    split, mostly in the pedestal where the spike lives.

    This undoes SWB's crude factor to recover ``<j_BS.B>`` and applies the
    field-aligned projection (see :func:`bouquet.physics.parallel_to_toroidal`,
    analytic method). The net factor is ``1/(<R><1/R>)`` (<= 1 by
    Cauchy-Schwarz, ~ 1 - eps^2 at the edge), evaluated on the same
    ``mygs``/grid the SWB call just used -- call this IMMEDIATELY after
    ``solve_with_bootstrap``, before any further mygs solve.
    """
    from .physics import parallel_to_toroidal

    j_bs_swb = np.asarray(j_bs_swb, dtype=float)
    npsi = len(j_bs_swb)
    _, F, _, _, _ = mygs.get_profiles(npsi=npsi, psi_pad=psi_pad)
    # <R>, <1/R> from get_q -- the SAME quantities SWB used for its R_avg/F
    # projection, so the undo is exact; <B^2> from sauter_fc.
    _, _, ravgs, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
    _, _, _, modb_avgs = mygs.sauter_fc(npsi=npsi, psi_pad=psi_pad)
    j_dot_B = j_bs_swb * F / q_ravg(ravgs, "<R>")   # undo SWB's R_avg/F projection
    return parallel_to_toroidal(
        j_dot_B,
        geom={"F": F, "avg_inv_R": q_ravg(ravgs, "<1/R>"), "avg_B2": modb_avgs[1]},
    )


#: Default R2 Ip measure.  ``exact`` (the physical FSA current integral) as
#: of 2026-08-04, per the package author: with ``ratio`` the sigma=0
#: invariant is true by construction, whereas ``exact`` makes |s-1| a
#: MEASUREMENT of the archive's internal self-consistency (typically a few
#: 1e-3; pinned at 5e-3 in the golden test with its derivation).  See the
#: "WHICH MODE IS THE DEFAULT" section of :class:`_AnchorIpRenorm`.
_R2_IP_MODE_DEFAULT = 'exact'


def _r2_ip_mode():
    """Resolved ``BOUQUET_R2_IP_MODE``.  See :class:`_AnchorIpRenorm`."""
    mode = os.environ.get(
        'BOUQUET_R2_IP_MODE', _R2_IP_MODE_DEFAULT).strip().lower()
    if mode in ('ratio', 'anchor'):   # 'anchor' was 7dc254b's name for it
        # RETIRED 2026-08-17, per maintainer decision on issue #35.  The
        # seven-archive decomposition showed the ratio calibration is blind
        # by construction to the dominant sigma=0 defect (the archived-split
        # amplitude bias: kappa absorbs it into the demand) while remaining
        # hypersensitive to the other (SWB bootstrap non-reproduction)
        # through compute_flux_integral -- the limiter-area measure this
        # class's own defect-3 note discredits.  Its passes were false clean
        # bills; its failures were real signal through a bad lens.  Raising,
        # not silently mapping to 'exact': a run that asked for the retired
        # measure should fail loudly, not report numbers on a different one.
        raise ValueError(
            f"BOUQUET_R2_IP_MODE={mode!r}: the ratio calibration was retired "
            "(issue #35; blind to the archived-split amplitude bias, "
            "hypersensitive to jBS non-reproduction through the limiter-area "
            "integral).  Use 'exact' (default), 'fsa' (diagnostic A/B), or "
            "'legacy'.")
    if mode not in ('exact', 'fsa', 'legacy'):
        raise ValueError(
            f"BOUQUET_R2_IP_MODE={mode!r} is not one of "
            "'exact' (default), 'fsa', 'legacy'")
    return mode


class _AnchorIpRenorm:
    r"""Frozen anchor-geometry evaluator for the R2 :math:`I_p` renormalisation.

    Route R2 (``perturb_jind_in_anchor=True``) sets the inductive AMPLITUDE by
    rooting on ``s*j_ind + j_BS``, holding the bootstrap fixed -- the
    physically-correct bookkeeping, since an :math:`I_p` constraint should move
    the ohmic drive only, and the Sauter bootstrap is an absolute current
    density that must survive the constraint unscaled.  THREE defects made that
    root untrustworthy.  Defects 1 and 2 were fixed in 7dc254b; defect 3 is why
    2's fix was a calibration rather than a measure, and is now fixed properly.

    **1. Geometry.**  The root used to run AFTER ``solve_with_bootstrap``, so
    ``mygs.flux_integral`` was evaluated on SWB's *landed* equilibrium instead
    of the anchor.  Measured on the synthetic D3D-like example at
    :math:`\sigma=0`: the SWB-landed geometry gives ``s = 0.8373`` against the
    anchor geometry's ``0.8524`` -- a 1.80 % error in the inductive amplitude
    for no physical reason.  Here the anchor equilibrium is captured with
    ``copy_eq()`` immediately after the state-anchor solve and every flux
    integral is taken on that snapshot, via the equilibrium object's own
    ``compute_flux_integral``.  ``mygs`` is never mutated, so nothing
    downstream of the SWB call sees a different state.

    **2. Convention normalisation.**  Rooting the raw ``Ip_target`` against
    ``mygs.flux_integral`` mis-splits ohmic vs bootstrap even once the geometry
    is right: on the D3D-like example the archived total flux-integrates to
    ``+12.92 %`` of :math:`I_p`, in ANY geometry, and that bias is by far the
    larger part of the R2 error.  7dc254b cancelled it by CALIBRATING the
    demand on the anchor -- target ``FI(reference_total)*Ip_target/Ip_anchor``
    instead of ``Ip_target`` -- which makes the :math:`\sigma=0` invariant
    ``s == 1.000`` true by construction.  It works, but it is a ratio fix for a
    measurement error, and it is only first-order accurate away from the
    reference shape.  RETIRED 2026-08-17 (issue #35): the seven-archive
    decomposition showed the calibration absorbs the archived-split amplitude
    bias into the demand (blind to the dominant real defect, e.g. a PASS at
    71 % of bar on the case with the WORST +4.1 % bias) while reading SWB
    bootstrap non-reproduction through ``compute_flux_integral`` (defect 3
    below).  Asking for it now raises.

    **3. The measure (the actual defect behind 2).**  Two separate errors were
    hiding inside that +12.92 %, and 7dc254b attributed all of it to the
    profile convention.  Measured on the D3D-like anchor:

    * ``compute_flux_integral`` is NOT :math:`\int_{\rm plasma} f\,dA`.  It
      integrates over the whole ``reg == 1`` limiter region, with the flux
      function evaluated at its LCFS value everywhere outside the plasma
      (``gs_prof_interp_apply`` CASE(4) returns 0 -- the LCFS end of the
      internal flux coordinate -- off the plasma, and ``gs_flux_int`` then
      reads the profile there).  ``FI(1) = 2.83853`` is the LIMITER area, not
      the plasma area, which is ``1.79005``; 7dc254b's note that ``FI(1) ==``
      plasma area is wrong.  For the archived total the excess area is charged
      at the edge value, ``1.36e5 A/m^2 * 1.05 m^2 = +1.43e5 A``, i.e. +11.9 %
      of :math:`I_p` -- essentially the whole bias.
    * the residual ~1 % is the profile convention, and it is not the one
      7dc254b assumed either.  bouquet's arrays are TokaMaker
      ``jphi-linterp`` values, :math:`J = \langle R\rangle P' +
      \langle 1/R\rangle FF'/\mu_0` (``jphi_update``), not the FSA density
      :math:`\langle j_\phi/R\rangle/\langle 1/R\rangle`.  The two differ by
      :math:`\langle R\rangle\langle 1/R^2\rangle/\langle 1/R\rangle`, up to
      11 % per surface at the edge.

    The fix is the measure itself: the textbook axisymmetric current integral

    .. math::
        I_p = \int j_\phi\,dA
            = \int d\psi\,\frac{V'}{2\pi}\,\langle j_\phi/R\rangle ,

    with :math:`V' = dV/d\psi` and :math:`\langle\cdot\rangle` straight from
    ``get_q``'s ``ravgs``, and with the ``jphi-linterp`` conversion folded in
    (:func:`bouquet.utils.Ip_fsa_integral`).  No mesh integral, no calibration,
    no reference profile: the demand is ``Ip_target``, full stop.  Validated on
    the solved anchor by integrating the equilibrium's OWN current profile
    (:func:`bouquet.utils.eq_jphi_profile`) and comparing with
    ``compute_area_integral(calc_jtor_plasma)``: **+0.0071 %**, against a
    required 0.1 %.  The same measure recovers the true :math:`I_p` of the
    ARCHIVED total to +0.068 %, and agrees to 0.011 % with the solver's own
    internal ``jphi_norm`` -- an independent cross-check, since TokaMaker
    renormalises a ``jphi-linterp`` profile by exactly that factor.

    Consequence for the golden invariant: at :math:`\sigma=0` the honest answer
    is no longer *exactly* 1.  The archived split, taken at face value, carries
    +0.068 % more current than ``Ip_target`` (the reconstruction's own
    :math:`j_\phi` self-consistency residual), so holding :math:`j_{BS}` fixed
    -- which is the point of route R2 -- requires shaving that off the
    inductive amplitude.  The measure is right; ``s`` is allowed to differ from
    1 by the reconstruction's own residual, and does.

    ``BOUQUET_R2_IP_MODE`` selects the measure, for A/B:

    ``exact``   (DEFAULT, since f48cd24) the ``jphi-linterp`` FSA current
                integral above;
    ``fsa``     the same integral read as an FSA density,
                :math:`\int d\psi (V'/2\pi)\langle 1/R\rangle J` -- correct
                only if the arrays really are FSA densities, which they are
                not (+0.927 % on the archived total);
    ``legacy``  pre-7dc254b: bracketed root against ``mygs`` in whatever state
                it is in, target ``Ip_target``.

    (``ratio`` -- 7dc254b's calibration, alias ``anchor`` -- was a fourth mode
    until 2026-08-17; retired per issue #35, see defect 2 above.  Its rows in
    the tables below are kept as historical measurements.)

    **WHICH MODE IS THE DEFAULT, AND WHY IT IS ``exact``.**  Measured at
    :math:`\sigma=0` on the synthetic golden fixture, one call per mode, twice
    for ``exact`` (bit-identical):

    ===========  ==========  ==========  ==============================
    mode         ``s``       ``|s-1|``   ``l_i`` vs recon
    ===========  ==========  ==========  ==============================
    ``exact``    0.996962    3.0e-3      -0.117 %
    ``fsa``      0.986045    1.4e-2      -0.287 %
    ``ratio``    0.999111    8.9e-4      -0.083 %
    ``legacy``   0.825917    1.7e-1      -3.114 %
    ===========  ==========  ==========  ==============================

    Provenance (table refreshed for issue #28): re-measured on main @ 4ad4894,
    on the synthetic golden fixture used throughout this module, production
    defaults, single-threaded BLAS.
    ``exact``/``ratio``/``legacy`` are from the stock R2 probe
    (``python tests/test_seeded_reproducibility.py r2 <outdir>``, seed
    20260804).  ``fsa`` is NOT exercised by that probe; it was run separately
    through the same ``perturb_kinetic_equilibrium`` call with
    ``BOUQUET_R2_IP_MODE=fsa``, so its row is measured, not carried over.

    THE ``l_i`` COLUMN IS l_i(3)/``iter`` AGAINST THE STEP-6 MATCHED TARGET,
    and it is not comparable to the values this table carried before.  Two
    re-denominations sit between them, neither of them a physics event: #20
    moved probe and target from l_i(1)/``std`` to l_i(3)/``iter`` (functionals
    ~29 % apart here), and #27 repointed ``l_i_target`` from the post-step-7
    read to ``result['li_final']``.  The second matters for THIS table in
    particular, because route R2 skips the corrective iteration (see the note
    on the standard loop below), so the old column charged every row for a
    +0.342 % step-7 drift the route never applies.  ``legacy`` moves most in
    absolute terms -- it is the uncorrected control, and what it is
    uncorrected *against* moved too; it remains at ``|s-1| = 1.7e-1``, two
    orders above every live mode, so the control has not gone vacuous.

    The correct measure does NOT reproduce the ``s == 1.000`` invariant more
    closely -- it reproduces it LESS closely, and for an understood reason.
    ``ratio`` is exact at :math:`\sigma=0` *by construction*: it asks the draw
    to carry the same (mis-measured) integral as the archived total, so every
    representation error cancels.  ``exact`` asks the draw to carry
    ``Ip_target`` in real amperes, so it also picks up the two ways the
    archived split fails to be the anchor:

    * the reconstruction's own :math:`j_\phi` residual -- the archived total
      differs from the anchor's own current profile by 1.6 % of peak in SHAPE,
      worth +0.1717 % of :math:`I_p` at the R2 state anchor (+0.068 % at the
      baseline recon), i.e. -0.25 % of inductive amplitude; plus
    * the :math:`\sigma=0` ``solve_with_bootstrap`` residual, -0.085 %, the
      same term ``ratio`` is left with.

    Both are real; neither is a bug in the measure, whose runtime self-check
    against the anchor's own profile is +0.0173 % here.  But it means the
    pinned invariant ``|s-1| <= 1e-3`` is not attainable by ANY honest measure
    on this case -- even a perfectly self-consistent archive would leave
    ~1.1e-3.  Flipping the default was therefore an acceptance-criterion
    decision rather than a code change, and it was taken as one.

    (Numbers in the two bullets and this paragraph: the runtime self-check
    ``+0.0173 %`` and the archived-total bias ``+0.1717 %`` are re-measured on
    main @ 4ad4894 -- the probe prints both on the ``[R2-anchor]`` line in
    ``exact`` mode -- and were recorded as ``+0.014 %`` / ``+0.193 %`` when
    this section was written in ``5a424d1``.  The ``-0.085 %`` SWB residual,
    the ``1.6 %`` shape figure, the ``+0.068 %`` baseline-recon value and the
    ``~1.1e-3`` floor derived from them are AS RECORDED AT ``5a424d1`` and
    were not re-measured here; issue #32 is the place that needs them
    refreshed, because they are what the ratio-path floor is built out of.)

    **THE DECISION WAS TAKEN, 2026-08-04.**  This section deferred it to the
    package author ("everything needed for it is here: set
    ``_R2_IP_MODE_DEFAULT = 'exact'``"); ``f48cd24``, later the same day, did
    exactly that, per package-author decision.  ``exact`` has been the default
    ever since and nothing has flipped it back.  The argument above is kept
    because it is the derivation BEHIND that decision and is still true -- but
    read it as the record of a choice already made, not as a live proposal.
    What the flip bought, in the author's framing: under ``ratio`` the
    :math:`\sigma=0` invariant is true by construction, whereas under
    ``exact`` ``|s-1|`` becomes a MEASUREMENT of the archive's internal
    self-consistency.  The acceptance moved with it, and is now stated on the
    Ip-space product ``|s-1|*f_ind`` rather than on ``|s-1|`` (issue #23,
    PR #26); see ``_S_FIND_ATOL_EXACT`` in
    ``tests/test_seeded_reproducibility.py``.

    **THE SECOND DECISION WAS TAKEN, 2026-08-17 (issue #35).**  ``ratio``
    itself was retired, and ``_S_ATOL = 1e-3`` -- which by then governed the
    ratio path only -- was retired with it, per explicit maintainer approval.
    The seven-archive decomposition settled #32's open question: the 8.9e-4
    the fixture showed was not a noise floor but the low end of a residual
    reaching 3.2e-2 on real geometry (a factor-965 span), driven entirely by
    sigma=0 SWB bootstrap non-reproduction seen through the limiter-area
    integral, while the calibration hid the archived-split amplitude bias
    that ``exact`` correctly reports.  ``exact`` is the sole sigma=0
    invariant (``_S_FIND_ATOL_EXACT`` on the Ip-space product); ``fsa`` is
    kept as a diagnostic A/B that isolates the affine ``P'`` term, with no
    acceptance bar of its own.

    Only route R2 uses this; the standard :math:`l_i` loop
    (``perturb_jind_in_anchor=False``, the production ensemble path) is
    untouched and bit-identical, because its root is followed by
    ``find_optimal_scale`` + the corrective iteration, which re-derive the
    amplitude from the solved equilibrium anyway.

    Parameters
    ----------
    mygs : TokaMaker
        Solver, positioned at the state-anchor equilibrium.  Snapshotted, not
        modified -- ``copy_eq()`` supports ``get_q``/``get_profiles`` and was
        verified to return ``ravgs`` bit-identical to the live solver's while
        leaving the live :math:`I_p` unchanged to the last bit.
    psi_N : ndarray
        Equilibrium flux grid the profiles live on.
    reference_total : ndarray
        The archived total :math:`j_\phi` (``input_j_phi``).  The demand no
        longer depends on it in ``exact``/``fsa`` mode; it is still used to fix
        the sign convention of ``get_profiles``' :math:`P'` and to report the
        measured bias.
    Ip_target : float
        Requested plasma current [A].
    psi_pad : float
        LCFS padding for the anchor ``get_stats`` call.
    mode : str
        Resolved :func:`_r2_ip_mode` value (``exact``/``fsa``).
    """

    __slots__ = ("_eq", "_psi_N", "_mode", "_target", "_Ip_target",
                 "_Ip_anchor", "_w", "_c", "_pprime_sign",
                 "_self_check", "_ref_bias")

    def __init__(self, mygs, psi_N, reference_total, Ip_target, psi_pad,
                 mode="exact"):
        # 'ratio' was retired 2026-08-17 (issue #35) -- see _r2_ip_mode for
        # the rationale.  Guarded here too because tests construct the class
        # directly, bypassing the env-var resolver.
        if mode not in ("exact", "fsa"):
            raise ValueError(
                f"_AnchorIpRenorm mode {mode!r}: only 'exact' and 'fsa' "
                "exist ('ratio' retired per issue #35; 'legacy' never "
                "constructs an anchor)")
        self._eq = mygs.copy_eq()
        self._mode = str(mode)
        self._psi_N = np.asarray(psi_N, dtype=float)
        self._Ip_target = float(Ip_target)
        reference_total = np.asarray(reference_total, dtype=float)
        try:
            self._Ip_anchor = float(mygs.get_stats(lcfs_pad=psi_pad)["Ip"])
        except Exception:
            self._Ip_anchor = self._Ip_target
        if not np.isfinite(self._Ip_anchor) or self._Ip_anchor == 0.0:
            self._Ip_anchor = self._Ip_target

        convention = "jphi-linterp" if self._mode == "exact" else "fsa"
        # Every FSA getter is evaluated on the FROZEN snapshot, and the
        # per-surface weights are cached as plain arrays -- after __init__
        # the root needs no solver call at all, so nothing downstream of
        # solve_with_bootstrap can move the demand.
        geom = fsa_current_geometry(self._eq, self._psi_N)
        # get_profiles' P' sign follows the case's flux convention; take it
        # from the anchor instead of assuming (the P' term is -3.3 % of Ip,
        # so a sign slip would be a 6.6 % error).
        probe = eq_jphi_profile(geom, "jphi-linterp", eq=self._eq)
        self._pprime_sign = (
            1.0 if float(np.dot(probe, reference_total)) > 0.0 else -1.0)
        self._w, self._c = Ip_fsa_weights(
            geom, convention=convention, pprime_sign=self._pprime_sign)
        # Runtime validation of the measure ON THIS CASE: integrate the
        # anchor's own current profile and compare with its true Ip.
        # Measured 0.009-0.043 % across the seven DIII-D demo archives
        # (issue #35), so a self_check far above that is a geometry problem,
        # not noise.
        j_eq = eq_jphi_profile(geom, convention, eq=self._eq,
                               pprime_sign=self._pprime_sign)
        self._self_check = self._Ip_of(j_eq) / self._Ip_anchor - 1.0
        self._ref_bias = self._Ip_of(reference_total) / self._Ip_anchor - 1.0
        self._target = self._Ip_target

    def _Ip_of(self, profile):
        """FSA plasma current [A] of *profile* on the frozen anchor geometry."""
        from scipy.integrate import trapezoid
        return float(trapezoid(
            self._w * np.asarray(profile, dtype=float), self._psi_N)) + self._c

    @property
    def mode(self):
        return self._mode

    @property
    def self_check(self):
        """Relative error of the measure on the anchor's own current profile.
        The step-1 validation, at runtime.  Measured 0.009-0.043 % across the
        seven DIII-D demo archives (issue #35)."""
        return self._self_check

    @property
    def reference_bias(self):
        """``I_p[reference_total]/I_p(anchor) - 1``: how far the archived total
        is from the current the anchor actually carries.  On the D3D-like
        golden this is +0.00068; on the seven DIII-D demo archives it is the
        +1.1 to +4.1 % archived-split amplitude bias of issue #35 / #15."""
        return self._ref_bias

    def inductive_share(self, j_ind):
        r"""``f_ind``: the LINEAR part of the measure on *j_ind*, over the
        :math:`I_p` demand -- i.e. what fraction of the demanded current the
        unscaled inductive profile carries.

        .. math::
            f_{\rm ind} = \frac{\int w\,j_{\rm ind}\,d\psi_N}{I_p^{\rm demand}}

        (``exact``/``fsa``: demand ``Ip_target``, so this is literally
        ``\int w j_ind / Ip_target``; ``ratio``: demand
        ``kappa*Ip_target``, the calibrated one that mode roots against, so the
        identity below holds in every mode.)

        **Why this is on the class.**  ``solve_scale`` returns ``s``, and the
        :math:`\sigma=0` invariant is stated on ``|s-1|``.  But ``s`` is not the
        residual -- it is the residual DIVIDED by this number, because the whole
        :math:`I_p` miss is charged to the inductive amplitude alone (route R2
        holds :math:`j_{BS}` fixed, by design).  Writing
        :math:`\Delta = I_p[j_{\rm ind}+j_{\rm other}] - I_p^{\rm demand}`, the
        affine measure gives exactly

        .. math::
            s - 1 = \frac{-\Delta}{\int w\,j_{\rm ind}}
                  \qquad\Longrightarrow\qquad
            |s - 1|\,f_{\rm ind} = \left|\Delta / I_p^{\rm demand}\right| .

        So ``|s-1|`` reported alone is not comparable across operating points:
        the same Ip-space residual reads ~2x larger on a high-bootstrap
        (low ``f_ind``) archive than on a low-beta one.  Reporting the pair is
        what makes the number interpretable, and the product is the quantity
        the acceptance budget is actually derived in -- see issue #23 and
        ``tests/test_seeded_reproducibility._S_FIND_ATOL_EXACT``.

        Parameters
        ----------
        j_ind : ndarray
            The inductive profile handed to :meth:`solve_scale` (UNSCALED --
            the same array, not ``s*j_ind``).

        Returns
        -------
        float
            ``f_ind``.  Dimensionless; ~0.45-0.95 over a single-machine beta
            ramp.

            Two values are quoted for the D3D-like golden, and they are
            measurements at DIFFERENT points, not a disagreement:

            * **0.772** -- the issue-#23-derived constant, measured at the
              *baseline-recon* geometry with an independent integrator.  This
              is the provenance of the ``3.86e-3 = 5e-3 * 0.772`` acceptance
              bar and is frozen as the reviewed number.
            * **0.7976** -- what THIS method returns in the fixture, measured
              at the :math:`\sigma=0` R2 anchor in exact mode.

            The few-percent gap between them is the baseline -> anchor step
            that #23 explicitly flags as not yet decomposed.  Carrying 0.772
            in the bar makes that bar slightly TIGHTER at the golden than a
            re-derivation would (3.86e-3 vs 5e-3*0.7976 = 3.99e-3), so the
            discrepancy relaxes nothing.  See
            ``tests/test_seeded_reproducibility._S_FIND_ATOL_EXACT``.
        """
        j_ind = np.asarray(j_ind, dtype=float)
        i_ind = self._Ip_of(j_ind) - self._c            # linear part only
        target = self._target
        if not np.isfinite(target) or target == 0.0:
            return float("nan")
        return float(i_ind / target)

    def solve_scale(self, j_ind, j_other):
        r"""Inductive scale ``s`` putting ``s*j_ind + j_other`` at the
        :math:`I_p` demand, in the frozen anchor geometry.

        Both measures are affine in the profile -- the FSA integral is
        ``trapezoid(w*J) + c`` with a cached weight array, and
        ``compute_flux_integral`` is linear -- so the root is analytic: two
        integrals instead of Brent's ~30.  Affinity is VERIFIED rather than
        assumed; a third evaluation checks the result and the call falls back
        to ``brentq`` if it does not hold.
        """
        j_ind = np.asarray(j_ind, dtype=float)
        j_other = np.asarray(j_other, dtype=float)
        eval_ip = self._Ip_of
        i_ind = eval_ip(j_ind) - self._c            # linear part only
        i_oth = eval_ip(j_other)
        scale = None
        if np.isfinite(i_ind) and i_ind != 0.0:
            _s = (self._target - i_oth) / i_ind
            resid = abs(eval_ip(_s * j_ind + j_other) - self._target)
            if np.isfinite(_s) and _s > 0.0 and \
                    resid <= 1.0e-9 * abs(self._target):
                scale = float(_s)
        if scale is None:
            # Non-affine or degenerate: fall back to the bracketed root.
            from scipy.optimize import root_scalar

            def _resid(a, *_args):
                return eval_ip(a * j_ind + j_other) - self._target

            scale = float(root_scalar(
                _resid,
                bracket=[1.0e-10 * self._Ip_target, 1.0e1 * self._Ip_target],
                method="brentq", rtol=1e-6).root)
        return scale


def _r2_ip_scale(anchor_ip, mygs, j_ind, j_other, psi_N, Ip_target):
    """Inductive scale for route R2, on the anchor geometry when available.

    ``anchor_ip`` is the :class:`_AnchorIpRenorm` captured before the SWB
    call.  ``None`` (capture failed, or ``BOUQUET_R2_IP_MODE=legacy``) falls
    back to the historical bracketed root against the live ``mygs`` state.
    """
    if anchor_ip is not None:
        return anchor_ip.solve_scale(j_ind, j_other)
    from scipy.optimize import root_scalar
    return float(root_scalar(
        Ip_flux_integral_vs_target,
        args=(mygs, j_ind, j_other, psi_N, Ip_target),
        bracket=[1.0e-10 * Ip_target, 1.0e1 * Ip_target],
        method="brentq", rtol=1e-6).root)


def _floored_zone_note(input_jinductive):
    """Annotation for the ``[R2-invariant]`` line on floored baselines.

    Where ``floor_inductive_split`` clamped the baseline ``j_inductive`` to 0,
    the archived ``j_BS`` absorbed the deficit, so a sigma=0 draw CANNOT
    reproduce the archived split there -- the amplitude root redistributes
    that current and ``s`` carries it.  Real and expected, not measurement
    error, so it is annotated rather than carved out (measured on the worst
    demo archive: 0.873 % of Ip inside a 6-point zone, 0.006 % elsewhere;
    issue #35 item 1).  The sigma0 guard's carve-out verifies a DIFFERENT
    contract (SWB-context reproducibility) and does not apply here.

    Returns '' when nothing is floored, so the QC line is unchanged on the
    (typical) un-floored case.
    """
    if input_jinductive is None:
        return ""
    n = int(np.sum(np.asarray(input_jinductive, dtype=float) <= 0.0))
    if not n:
        return ""
    return (f"  [{n} floored j_ind pts: the invariant carries an expected "
            f"floor here; issue #35]")


def _fmt_s_and_find(s, f_ind, mode=None):
    """The R2 QC fragment: ``|s-1|``, ``f_ind`` and their product, together.

    ``s`` on its own is a ratio whose denominator is the inductive share, so a
    bare ``|s-1|`` cannot be compared between a low-beta and a high-bootstrap
    archive.  Every place that reports the scale reports this fragment with it,
    so the Ip-space residual ``|s-1|*f_ind`` -- the quantity the acceptance
    budget is derived in -- is readable off the log without a second lookup.
    See :meth:`_AnchorIpRenorm.inductive_share` and issue #23.

    ``mode`` STAMPS THE MEASURE, and is not cosmetic.  ``f_ind`` is normalised
    by the mode's own Ip demand, and two modes can print different ``f_ind``
    (hence different products) for identical physics, purely because of the
    measure.  (The worked example was ``ratio`` vs ``exact`` -- 0.7809 vs
    0.7976 on the golden, the calibrated demand reading ~2.1 % high -- and
    although ``ratio`` is retired (issue #35), ``exact`` vs ``fsa`` differ by
    the affine ``P'`` constant for the same reason.)  Emitting them under one
    unlabelled ``[R2-invariant]`` tag invites exactly the
    cross-operating-point comparison issue #23 exists to prevent, so the
    label travels with the number.
    """
    out = f", |s-1|={abs(float(s) - 1.0):.3e}"
    tag = f" [mode={mode}]" if mode else ""
    if f_ind is None or not np.isfinite(f_ind):
        return out + ", f_ind=n/a" + tag
    return (out + f", f_ind={float(f_ind):.4f}"
            f", |s-1|*f_ind={abs(float(s) - 1.0) * float(f_ind):.3e}" + tag)


def _r2_f_ind(anchor_ip, j_ind):
    """Inductive share for the scale returned by :func:`_r2_ip_scale`.

    ``None`` when there is no frozen anchor (legacy/fallback root): the bracketed
    root against the live solver has no cached measure to read the share off, and
    a number computed on a DIFFERENT state would silently mis-normalise ``s``.
    Reporting nothing is the honest answer there.
    """
    if anchor_ip is None:
        return None
    try:
        return float(anchor_ip.inductive_share(j_ind))
    except Exception:                       # diagnostic only -- never fatal
        return None


def smooth_jbs_transition(j_BS):
    """Gaussian-smooth the shelf->spike transition / near-axis zone of a
    (toroidal-converted) ``solve_with_bootstrap`` j_BS profile.

    The innermost-surface Sauter evaluation is numerically fragile: SWB's
    raw axis point collapses ~2x below its neighbours (verified identical
    across recon/draw call contexts to <0.001 MA/m^2 -- it is an
    evaluation artifact of the tiny ``psi_pad`` surface, not state noise).
    The reconstruction has always repaired this with a localised Gaussian
    filter around the flat-shelf end (with no shelf the window is the
    near-axis zone, indices 0..10); until 2026-07 the per-draw path did
    NOT, so every draw's spike kept the raw collapsed axis point while
    riding on an inductive fit made against the smoothed split -- a 1-2
    grid-point axis divot in every draw target (-9% j_phi(0), q0 +12%
    wholesale at sigma=0). This helper is the single shared treatment:
    apply it immediately after EVERY ``_swb_jbs_to_toroidal`` conversion
    so recon and draws stay sigma=0-consistent.

    Detection + window + weights are bit-identical to the original recon
    inline block: find the leading flat shelf (values equal to j_BS[0]
    within 1e-6 relative), Gaussian-filter (sigma=3 grid indices) a
    +/-10-index window around the shelf end, and blend with a triangular
    weight (1 at the shelf end, 0 at the window edges) so the exact
    shelf value and the spike beyond the window are preserved.
    """
    from scipy.ndimage import gaussian_filter1d

    j_BS = np.asarray(j_BS, dtype=float)
    shelf_val = j_BS[0]
    shelf_end = 0
    for i in range(1, len(j_BS)):
        if abs(j_BS[i] - shelf_val) / max(abs(shelf_val), 1e-30) < 1e-6:
            shelf_end = i
        else:
            break

    half = 10
    lo = max(0, shelf_end - half)
    hi = min(len(j_BS), shelf_end + half + 1)
    smoothed = gaussian_filter1d(j_BS[lo:hi], sigma=3.0)

    out = j_BS.copy()
    for i in range(lo, hi):
        w = max(0.0, 1.0 - abs(i - shelf_end) / half)
        out[i] = w * smoothed[i - lo] + (1 - w) * j_BS[i]
    return out


# ====================================================================
#  Core perturbation routine
# ====================================================================
def perturb_kinetic_equilibrium(
    mygs,
    psi_N,
    pressure,
    ne,
    te,
    ni,
    ti,
    input_j_phi,
    sigma_ne,
    sigma_te,
    sigma_ni,
    sigma_ti,
    sigma_jphi,
    n_ls,
    t_ls,
    j_ls,
    Ip_target,
    l_i_target,
    Zeff,
    npsi,
    p_thresh=0.005,
    input_jinductive=None,
    # Default tightened from 5.0 -> 1.0 (% real-equilibrium l_i error).
    # The outer loop's proxy_target correction is now a pure Newton step
    # so 1% is reachable in 2-3 iters.  The notebook previously
    # overrode to 10.0 to compensate for the conservative 0.7/0.3 blend
    # accepting the proxy-biased equilibrium at iter 1; with Newton +
    # tighter tolerance the equilibrium tracks recon l_i instead of
    # locking in at the +5% proxy-bias offset that produced ~10 mm
    # systematic boundary shift across all draws.
    l_i_tolerance=0.01,
    psi_pad=1e-3,
    constrain_sawteeth=True,
    recalculate_j_BS=True,
    isolate_edge_jBS=True,
    scale_jBS=1.0,
    floor_j_BS=True,
    jBS_diff=None,
    accept_anchor_inband=False,
    perturb_jind_in_anchor=False,
    swb_iterations=3,
    diagnostic_plots=False,
    max_pressure_iter=_MAX_PRESSURE_ITER,
    max_li_iter=_MAX_LI_ITER,
    psi_N_kinetic=None,
    p_fast=None,
    j_NBI=None,
    j_RF=None,
    aux_sigmas=None,
    aux_baselines=None,
    aux_length_scales=None,
    ni_from_zeff=True,
    max_proxy_draws=500,
    bnd_diag_callback=None,
    # Differential bootstrap (DIFF_BS=1 mode):
    #   recon_eq_snapshot      -- TokaMaker_equilibrium snapshot of mygs
    #                             in recon state (from copy_eq()).  When
    #                             provided, mygs is restored to this
    #                             state before the per-draw SWB call so
    #                             SWB sees the same context as the
    #                             cached recon SWB call did.
    #   spike_profile_recon_cached -- numpy array of isolated_j_BS from
    #                                 SWB(recon kinetics), computed once
    #                                 before the per-draw loop.  The
    #                                 per-draw spike is subtracted to
    #                                 get delta_spike, which is added to
    #                                 input_j_phi to form new_jphi.  At
    #                                 sigma->0 delta_spike->0 and
    #                                 new_jphi->input_j_phi exactly
    #                                 (= PIN_JPHI behaviour).
    recon_eq_snapshot=None,
    spike_profile_recon_cached=None,
    # Delta composition (GenerationConfig.jbs_delta_mode) -- unlike DIFF_BS
    # this composes with the full l_i band / GPR j_ind machinery:
    #   spike_delta_ref      -- RAW sigma=0 spike (toroidal-converted, NO
    #                           smoothing), cached once per run in the same
    #                           pre-draw anchor context.
    #   spike_delta_baseline -- the baseline j_BS split the draws must
    #                           reproduce at sigma=0 (bl.j_BS [+ jBS_diff]).
    # When both are set, the standard branch builds the per-draw spike as
    #   spike_delta_baseline + (SWB_raw(perturbed) - spike_delta_ref)
    # so common-mode evaluation artifacts (the collapsed innermost-surface
    # point) cancel exactly and the per-draw Sauter response is unfiltered.
    spike_delta_ref=None,
    spike_delta_baseline=None,
    proxy_bias_warmstart=None,
    pin_jphi=False,
    Z_imp=None,
    p_diff=None,
    jphi_diff=None,
    rng=None,
):
    r"""Perturb kinetic and current-density profiles and iterate to
    match :math:`I_p` and :math:`l_i` targets.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker Grad-Shafranov solver object.
    psi_N : ndarray
        1-D normalised poloidal flux grid :math:`\hat{\psi}`.
    pressure : ndarray
        1-D baseline total pressure.
    ne : ndarray
        1-D electron density [m\ :sup:`-3`].
    te : ndarray
        1-D electron temperature [eV].
    ni : ndarray
        1-D ion density [m\ :sup:`-3`].
    ti : ndarray
        1-D ion temperature [eV].
    input_j_phi : ndarray
        1-D toroidal current density [A/m\ :sup:`2`]; must be the
        *inductive* component when ``recalculate_j_BS=True``.
    sigma_ne : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_e` [m\ :sup:`-3`].
    sigma_te : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_e` [eV].
    sigma_ni : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_i` [m\ :sup:`-3`].
    sigma_ti : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_i` [eV].
    sigma_jphi : ndarray
        1-D experimental :math:`1\sigma` for :math:`j_\phi` [A/m\ :sup:`2`].
    n_ls : float
        GPR length-scale for density profiles.
    t_ls : float
        GPR length-scale for temperature profiles.
    j_ls : float or ndarray
        GPR length-scale for :math:`j_\phi`.  A 1-D array gives a
        non-stationary Gibbs kernel (see ``sigmoid_length_scale``).
    Ip_target : float
        Target plasma current [A].
    l_i_target : float
        Target internal inductance.
    Zeff : ndarray
        Effective ion charge profile on ``psi_N`` (scalar accepted and
        broadcast). With the active zeff channel this is re-drawn per draw.
    npsi : int
        Normalised poloidal flux grid size.
    p_thresh : float
        Acceptable :math:`\langle P \rangle` mismatch as a FRACTION
        (e.g. ``0.05`` == 5 %).
    input_jinductive : ndarray or None
        Dimensionless inductive :math:`j_\phi` shape (required when
        ``recalculate_j_BS=True``).
    l_i_tolerance : float
        :math:`l_i` matching tolerance as a FRACTION of ``l_i_target``
        (e.g. ``0.01`` == 1 %).
    psi_pad : float
        Padding inside the LCFS for profile queries.
    constrain_sawteeth : bool
        Reject equilibria with :math:`q_0 < 1`.
    recalculate_j_BS : bool
        Recompute bootstrap current for perturbed profiles.
    isolate_edge_jBS : bool
        Separate the edge bootstrap-current spike from the core
        contribution inside ``solve_with_bootstrap``.
    scale_jBS : float
        Multiplicative scale factor applied to :math:`j_{\rm BS}` in
        ``solve_with_bootstrap``.  A value of 1.0 applies no scaling.
    swb_iterations : int
        H-mode self-consistency iterations inside ``solve_with_bootstrap``
        (its ``iterations`` argument). 2 is usually enough when trading
        accuracy for speed.
    diagnostic_plots : bool
        Show diagnostic matplotlib figures (including inside
        ``solve_with_bootstrap`` and ``find_optimal_scale``).
    max_pressure_iter : int
        Safety cap on pressure-matching loop.
    max_li_iter : int
        Safety cap on :math:`l_i`-matching loop.
    psi_N_kinetic : ndarray or None
        Kinetic-profile grid (may extend past psi_N = 1 into the SOL).
        If provided, ``ne``, ``te``, ``ni``, ``ti`` and their sigmas
        must be on this grid.  GPR sampling is done on this grid;
        perturbed profiles are then interpolated onto ``psi_N`` for
        pressure matching and equilibrium solving.  Returned
        perturbed profiles are on ``psi_N_kinetic``.  If ``None``,
        ``psi_N`` is used for everything (original behaviour).
    perturb_jind_in_anchor : bool
        Route R2: GPR-perturb the inductive current in the recon-anchor
        block and accept that equilibrium (band-conditioned), rather than
        running the downstream ``find_optimal_scale`` + corrective loop.
        :math:`I_p` then sets the inductive AMPLITUDE only, holding the
        bootstrap fixed -- the physically-correct bookkeeping.  The
        renormalisation is evaluated on the ANCHOR geometry, pinned before
        the ``solve_with_bootstrap`` call (see :class:`_AnchorIpRenorm`), so
        at :math:`\sigma=0` it returns exactly 1.000 and reproduces the
        archived split.  Default False (the standard :math:`l_i` loop).
    rng : numpy.random.Generator, int, or None
        The Generator EVERY GPR draw in this call is taken from -- the
        kinetic channels, the aux channels and the :math:`j_\phi` draws
        alike.  ``generate_bouquet`` passes the run's single Generator (see
        :func:`bouquet.sampling.make_rng`) so one seed governs the whole
        ensemble.  ``None`` (default) draws from fresh OS entropy, i.e. the
        call is NOT reproducible; an ``int`` is promoted to a Generator.

    Returns
    -------
    tuple
        ``(ne_perturb, te_perturb, ni_perturb, ti_perturb,
        w_ExB, output_jphi, diagnostics)``

        When ``psi_N_kinetic`` is provided, the kinetic profiles
        (``ne_perturb`` etc.) are on the ``psi_N_kinetic`` grid.
    """

    # ----------------------------------------------------------------
    #  1.  Lazy OFT imports (deferred so GPR-only use works without OFT)
    # ----------------------------------------------------------------
    from scipy.optimize import root_scalar
    from scipy.interpolate import interp1d
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS
    from OpenFUSIONToolkit.TokaMaker.bootstrap import (
        solve_with_bootstrap,
        find_optimal_scale,
    )

    # ----------------------------------------------------------------
    #  2.  Validate inputs and set up dual grids
    # ----------------------------------------------------------------
    if recalculate_j_BS and input_jinductive is None:
        raise ValueError(
            "input_jinductive must be provided when recalculate_j_BS=True"
        )

    # One Generator for every draw in this call.  Promoting here (rather than
    # defaulting each call site to None) is what makes the seed reach the GPR:
    # a per-call-site ``np.random.default_rng()`` would re-seed from OS entropy
    # at every draw and silently discard the run's seed.
    rng = make_rng(rng)

    # Kinetic grid: either the user-supplied extended grid or psi_N
    psi_kin = psi_N_kinetic if psi_N_kinetic is not None else psi_N
    _dual_grid = psi_N_kinetic is not None

    def _kin_to_eq(arr_kin):
        """Regrid a profile from the kinetic onto the equilibrium grid.

        Shape-preserving PCHIP via the shared helper (linear regrid leaves
        slope kinks at every kinetic knot -> stepped Sauter j_BS; see
        utils.pchip_interp). Must match every other kin->eq site."""
        if not _dual_grid:
            return arr_kin
        return pchip_interp(psi_kin, arr_kin, psi_N)

    # Fixed additive currents (NBI + RF), held constant across draws. They are
    # treated exactly like the bootstrap spike (additive, non-scaled) in the
    # j_phi assembly below. They contribute to the total only when the inductive
    # base (input_jinductive) excludes them; with recalculate_j_BS=False the base
    # is input_j_phi, which already contains them, so j_fixed_eff is zero there.
    # Defaults (None -> zero) reproduce the original behaviour exactly.
    _jfix = np.zeros_like(psi_N)
    if j_NBI is not None:
        _jfix = _jfix + np.asarray(j_NBI, dtype=float)
    if j_RF is not None:
        _jfix = _jfix + np.asarray(j_RF, dtype=float)
    j_fixed_eff = _jfix if recalculate_j_BS else np.zeros_like(psi_N)
    # Total-current anchor: fold jphi_diff (= equilibrium.j_tor - core_profiles
    # total) into the fixed additive so it rides under EVERY downstream new_jphi
    # build (recon-anchor / l_i-match / corrective; all use j_fixed_eff), exactly
    # like jBS_diff in spike_profile. The perturbed j_inductive + SWB bootstrap
    # move underneath this fixed offset; at sigma=0 the total == equilibrium.j_tor.
    # jphi_diff integrates to ~0, so the Ip renorm (which holds spike+j_fixed_eff
    # fixed while scaling j_inductive) is unaffected.
    if jphi_diff is not None and recalculate_j_BS:
        j_fixed_eff = j_fixed_eff + np.asarray(jphi_diff, dtype=float)

    # ----------------------------------------------------------------
    #  3.  Perturb kinetic profiles to match <P>
    # ----------------------------------------------------------------
    inp_avg = mygs.flux_integral(psi_N, pressure)

    p_err = np.inf
    p_iter = 0
    # --- Zeff-primary main-ion derivation (active zeff channel) -----------
    # When the zeff aux channel is enabled AND the baseline carries dilution
    # information (ni < ne), ni is DERIVED per draw from the drawn (ne, Zeff)
    # via single-impurity quasineutrality instead of drawn independently:
    # one mutually consistent (ne, ni, Zeff, nz) set per draw, used by the
    # bootstrap, the archived profiles, and the per-draw p-file alike.
    # sigma_ni is not used in this mode. See physics.main_ion_density_from_zeff.
    # ni_from_zeff=False opts out (ni drawn from its own sigma_ni, e.g. a
    # measured IDA ni_source envelope); Zeff is still drawn below as a normal
    # aux channel, bounded by the same Z_imp, and drives the bootstrap either way.
    _zeff_chan = bool(aux_sigmas) and ('zeff' in aux_sigmas) \
        and (aux_baselines or {}).get('zeff') is not None
    _zeff_active = bool(ni_from_zeff) and _zeff_chan
    _Z_imp = None
    _zeff_draw = None
    if _zeff_chan:
        # Prefer the baseline's own Z_imp (the charge its ni was actually built
        # with, e.g. the IDA impurity_Z); invert (ne, ni, Zeff) only if absent.
        # Resolved for the whole channel, not just the active mode: it also
        # bounds the passive draw below.
        from .physics import effective_impurity_charge
        _Z_imp = float(Z_imp) if Z_imp else effective_impurity_charge(
            ne, ni, np.asarray(aux_baselines['zeff'], dtype=float))
        if _zeff_active and _Z_imp is None:
            print("  [zeff] baseline has no ne-ni dilution (ni ~= ne): Zeff "
                  "draws still drive the bootstrap, but ni remains an "
                  "independent channel")

    # p_thresh is a FRACTION (e.g. 0.05 == 5%); p_err is computed in percent.
    _p_thresh_pct = float(p_thresh) * 100.0
    print("Searching for pressure profile match...")

    while p_err > _p_thresh_pct:
        p_iter += 1
        if p_iter > max_pressure_iter:
            raise RuntimeError(
                f"Pressure match not found within {max_pressure_iter} iterations "
                f"(last error {p_err:.2f}% vs threshold {_p_thresh_pct:.2f}%)"
            )

        # GPR sampling on psi_kin (kinetic grid, may include SOL)
        ne_perturb = _draw_monotonic_perturbation(
            psi_kin, ne / ne[0], sigma_ne / ne[0], n_ls, rng=rng
        ) * ne[0]

        te_perturb = _draw_monotonic_perturbation(
            psi_kin, te / te[0], sigma_te / te[0], t_ls, rng=rng
        ) * te[0]

        if _zeff_active and _Z_imp is not None:
            # draw Zeff, derive ni (quasineutrality): ni and the pressure it
            # feeds stay inside the pressure-match loop with the other draws
            from .physics import main_ion_density_from_zeff
            _zb = np.asarray(aux_baselines['zeff'], dtype=float)
            _zs = np.asarray(aux_sigmas['zeff'], dtype=float)
            _z0 = float(np.max(np.abs(_zb))) or 1.0
            _zeff_draw = np.atleast_1d(np.asarray(np.squeeze(
                generate_perturbed_GPR(
                    psi_kin, _zb / _z0, _zs / _z0,
                    length_scale=(aux_length_scales or {}).get('zeff', 0.4),
                    n_samples=1, rng=rng)) * _z0, dtype=float))
            # 1 <= Zeff <= Z_imp guarantees 0 <= ni <= ne and nz >= 0
            _zeff_draw = np.clip(_zeff_draw, 1.0, _Z_imp * (1.0 - 1e-9))
            ni_perturb = main_ion_density_from_zeff(ne_perturb, _zeff_draw, _Z_imp)
        else:
            ni_perturb = _draw_monotonic_perturbation(
                psi_kin, ni / ni[0], sigma_ni / ni[0], n_ls, rng=rng
            ) * ni[0]

        ti_perturb = _draw_monotonic_perturbation(
            psi_kin, ti / ti[0], sigma_ti / ti[0], t_ls, rng=rng
        ) * ti[0]

        # Pressure matching on equilibrium grid (psi_N, confined only)
        ne_eq = _kin_to_eq(ne_perturb)
        te_eq = _kin_to_eq(te_perturb)
        ni_eq = _kin_to_eq(ni_perturb)
        ti_eq = _kin_to_eq(ti_perturb)

        pres_tmp = EC * (ne_eq * te_eq + ni_eq * ti_eq)
        tmp_avg = mygs.flux_integral(psi_N, pres_tmp)
        p_err = np.mean(np.abs(inp_avg - tmp_avg) / inp_avg) * 100.0

    # Add the fixed (fast-ion) pressure -- constant across draws, never perturbed
    # -- to the thermal pressure for the GS solve. The pressure-match diagnostic
    # above stays thermal-only (comparable to the thermal baseline inp_avg).
    if p_fast is not None:
        pres_tmp = pres_tmp + _kin_to_eq(np.asarray(p_fast, dtype=float))

    # Impurity (carbon) thermal pressure: single-impurity model on the SAME
    # (ne, ni, Z_imp) set that derived the main ion (one Zeff). Computed from the
    # converged draw's (ne_eq, ni_eq) so it perturbs with the kinetics; the match
    # loop above stays thermal-D-only. Single-ion e*(ne*Te + ni*Ti) omits this.
    if Z_imp:
        from .physics import impurity_pressure
        pres_tmp = pres_tmp + impurity_pressure(ne_eq, ni_eq, ti_eq, Z_imp)
    # Pressure-diff anchor: fixed offset (= equilibrium.pressure - reconstructed
    # baseline) added to baseline AND every draw, mirroring jBS_diff, so the solve
    # pressure anchors to FUSE exactly while the reconstructed thermal delta tracks
    # per-draw kinetics. At sigma=0 pres_tmp == dd equilibrium.pressure.
    if p_diff is not None:
        pres_tmp = pres_tmp + np.asarray(p_diff, dtype=float)

    # --- switchboard: perturb the auxiliary profiles (rotation / transport /
    # impurity). GPR-sample each enabled channel once per draw (sigma
    # presence = on). 'zeff'
    # is ACTIVE: the perturbed Zeff is reassigned so every downstream SWB
    # bootstrap call uses it. Passive aux (omega_tor, e_r, chi_*) are carried
    # out for storage only. Baselines/sigmas are on the kinetic grid.
    aux_out = {}
    if aux_sigmas:
        for _en, _es in aux_sigmas.items():
            if _en == 'zeff' and _zeff_draw is not None:
                continue          # already drawn inside the pressure loop
            _eb = (aux_baselines or {}).get(_en)
            if _eb is None:
                continue
            _eb = np.asarray(_eb, dtype=float)
            _es = np.asarray(_es, dtype=float)
            _els = (aux_length_scales or {}).get(_en, 0.4)
            # normalize by PEAK magnitude (robust: some aux, e.g. E_r, are
            # ~0 on axis -> a denormal _eb[0] would make _eb/_e0 overflow)
            _e0 = float(np.max(np.abs(_eb)))
            if not (_e0 > 0):
                _e0 = 1.0
            _ep = np.squeeze(generate_perturbed_GPR(
                psi_kin, _eb / _e0, _es / _e0, length_scale=_els, n_samples=1,
                rng=rng)) * _e0
            if _en == 'zeff':
                # Same bound as the active path: a draw outside [1, Z_imp] is
                # outside the single-impurity model that Z_imp / p_imp assume.
                _ep = np.clip(_ep, 1.0,
                              None if _Z_imp is None else _Z_imp * (1.0 - 1e-9))
            aux_out[_en] = np.atleast_1d(np.asarray(_ep, dtype=float))
        if _zeff_draw is not None:
            aux_out['zeff'] = _zeff_draw      # the draw ni was derived from
        if "zeff" in aux_out:                 # active -> drives the bootstrap
            Zeff = np.clip(_kin_to_eq(aux_out["zeff"]), 1.0, None)

    mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])

    # ----------------------------------------------------------------
    #  3b. Optional diagnostic plots for kinetic profiles
    # ----------------------------------------------------------------
    if diagnostic_plots:
        fig, ax = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
        _pairs = [
            #  axis       orig  pert        scale  σ_phys     color        label       ylabel
            (ax[0, 0], ne, ne_perturb, 1.0,  sigma_ne, "tab:red",    r"$n_e$", r"n [m$^{-3}$]"),
            (ax[0, 1], ni, ni_perturb, 1.0,  sigma_ni, "tab:orange", r"$n_i$", None),
            (ax[1, 0], te, te_perturb, 1e-3, sigma_te, "tab:blue",   r"$T_e$", r"T [keV]"),
            (ax[1, 1], ti, ti_perturb, 1e-3, sigma_ti, "tab:cyan",   r"$T_i$", None),
        ]
        for a, orig, pert, scale, sig, clr, lbl, ylabel in _pairs:
            a.plot(psi_N, pert * scale, c=clr, ls="--", alpha=0.5)
            a.plot(psi_N, orig * scale, c=clr, lw=2, label=f"input {lbl}")
            a.fill_between(
                psi_N,
                (orig - sig) * scale,
                (orig + sig) * scale,
                alpha=0.3, color=clr,
                label=r"$\pm\,1\sigma_{\rm exp}$",
            )
            a.legend(loc="best")
            a.grid(ls=":")
            if ylabel:
                a.set_ylabel(ylabel)
        ax[1, 0].set_xlabel(r"$\hat{\psi}$")
        ax[1, 1].set_xlabel(r"$\hat{\psi}$")
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------
    #  4.  Bootstrap-current recalculation (optional)
    # ----------------------------------------------------------------
    j0_scales = []
    Ip_scales = []
    iteration_l_is = []
    iteration_Ips = []
    # R2 inductive Ip-renorm scale actually applied (None off route R2);
    # surfaced on diagnostics['r2_ip_scale'] -- at sigma=0 it must be 1.000.
    _r2_scale_used = None
    # Inductive share f_ind at the same call (None off route R2).  `s` alone is
    # not interpretable across operating points: |s-1| = |Delta/Ip| / f_ind, so
    # the SAME Ip-space residual reads ~2x larger on a high-bootstrap archive.
    # Surfaced on diagnostics['r2_f_ind'] and printed next to the scale so the
    # pair travels together.  See _AnchorIpRenorm.inductive_share, issue #23.
    _r2_f_ind_used = None

    # pin_jphi: pin j_phi to recon's converged shape (only pressure perturbs
    # per draw).  Set via the function argument; the PIN_JPHI env var is kept
    # as a back-compat override.  Read here so it's in scope whether or not
    # the recalculate_j_BS branch below runs (see recon-anchor block).
    _pin_jphi = bool(pin_jphi) or os.environ.get('PIN_JPHI', '0') == '1'

    # DIFF_BS (differential bootstrap) is structurally similar to
    # PIN_JPHI but instead of fully bypassing SWB, it runs SWB on the
    # perturbed kinetics from a restored recon state, subtracts a
    # pre-cached SWB(recon kinetics) result, and adds the resulting
    # delta to input_j_phi.  At sigma->0 this reproduces PIN_JPHI
    # exactly (delta=0).  Requires recon_eq_snapshot and
    # spike_profile_recon_cached kwargs to be populated by
    # generate_bouquet before the per-draw loop.
    _diff_bs = (os.environ.get('DIFF_BS', '0') == '1'
                and recon_eq_snapshot is not None
                and spike_profile_recon_cached is not None)
    if (os.environ.get('DIFF_BS', '0') == '1'
            and (recon_eq_snapshot is None
                 or spike_profile_recon_cached is None)):
        print("  [DIFF_BS] WARNING: env var set but cache/snapshot kwargs "
              "missing; falling back to standard SWB mode")

    # ---- PIN_JPHI: bypass SWB entirely ----
    # When PIN_JPHI=1, j_phi is pinned to recon's exact converged
    # profile (input_j_phi), so we don't need SWB's Sauter recompute
    # of the bootstrap -- the bootstrap is implicitly embedded in
    # input_j_phi already.  Skipping SWB removes a major failure
    # source (DLSODE / Picard maxits inside Sauter), making this
    # diagnostic mode actually reach the recon-anchor solve where
    # the PIN_JPHI logic takes effect.
    # ---- PIN_JPHI / DIFF_BS pipeline probe ----
    # Enabled by env var PINJ_PROBE=1.  Logs (l_i, Ip, axis) at every
    # stage where mygs state could be mutated.  Used to localise the
    # source of l_i drift between warmstart-restored recon state and
    # the post-perturb state recorded by store_equilibrium.
    _pinj_probe = (os.environ.get('PINJ_PROBE', '0') == '1'
                   and (_pin_jphi or _diff_bs))
    def _probe(label):
        if not _pinj_probe:
            return
        try:
            _st = mygs.get_stats(li_normalization='iter',
                                 lcfs_pad=psi_pad)
            _op = mygs.o_point
            print(f"    [probe {label:38s}] "
                  f"l_i={float(_st['l_i']):.5f}  "
                  f"Ip={float(_st['Ip']):.0f}  "
                  f"axis=({float(_op[0]):.5f},{float(_op[1]):+.5f})")
        except Exception as _pexc:
            print(f"    [probe {label:38s}] failed ({_pexc})")
    _probe("entry to perturb_kinetic_equilibrium")
    if _pin_jphi and recalculate_j_BS:
        print(f"  [PIN_JPHI] bypassing SWB call; using recon j_phi "
              f"as fixed forward-mode target")
        # Provide the variables that the recon-anchor and l_i loop
        # expect from SWB's results: spike_profile is the bootstrap
        # implied by recon (input_j_phi - input_jinductive), full_j_BS
        # = same (we don't track separate components here).
        spike_profile = (input_j_phi - input_jinductive).copy()
        full_j_BS = spike_profile.copy()
        _probe("after PIN_JPHI bypass setup")
        baseline_li_proxy = calc_cylindrical_li_proxy(
            mygs, input_j_phi, psi_pad)
        # Don't append SWB scale factors -- nothing to scale here
        eq_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)

        # ---- PIN_JPHI recon-anchor solve ----
        # The if/elif/elif chain (this branch | DIFF_BS | recalculate_j_BS)
        # used to skip the recon-anchor solve at line ~1156 for PIN_JPHI
        # because that solve lived inside the `elif recalculate_j_BS:`
        # branch.  Result (bug fixed 2026-05): with PIN_JPHI=1 +
        # nonzero kinetic σ, perturbed kinetics were drawn (ne_perturb,
        # te_perturb, ...) and stored in H5 correctly, but they NEVER
        # reached mygs.set_profiles / mygs.solve.  mygs stayed in the
        # warmstart-restored recon state for every draw, so per-draw
        # Ip, l_i, axis position, q profile, and LCFS were all bit-
        # identical despite ±5-10% pressure perturbations on paper.
        #
        # Fix: inline a recon-anchor solve here, mirroring the
        # DIFF_BS branch's solve at line ~956-977 and the standard
        # SWB branch's solve at line ~1135-1156.  Use perturbed
        # pres_tmp for PP' + pax, and pinned input_j_phi for the
        # jphi-linterp FFP shape (= "PIN_JPHI" semantics: j_phi
        # shape locked to recon, pressure free to perturb).  The
        # dead-code `if _pin_jphi: new_jphi = input_j_phi.copy()` at
        # line ~1135 (inside the elif branch) is now genuinely dead
        # and could be cleaned up, but leaving it preserves symmetry
        # with the other branches.
        _psi_range_pin = mygs.psi_bounds[1] - mygs.psi_bounds[0]
        _pp_pin = {"type": "linterp",
                   "y": pchip_derivative(psi_N, pres_tmp) / _psi_range_pin,
                   "x": psi_N}
        _pp_pin["y"][-1] = 0.0
        _ffp_pin = {"type": "jphi-linterp",
                    "y": input_j_phi.copy(),
                    "x": psi_N}
        mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])
        _probe("PIN_JPHI: after set_targets(Ip,pax)")
        mygs.set_profiles(pp_prof=_pp_pin, ffp_prof=_ffp_pin)
        _probe("PIN_JPHI: after set_profiles(pp,ffp)")
        try:
            mygs.solve()
            _probe("PIN_JPHI: after solve()")
            print(f"  [recon-anchor] forward-solved with PINNED "
                  f"recon j_phi (PIN_JPHI=1, only pressure perturbs)")
            if bnd_diag_callback is not None:
                bnd_diag_callback("after PIN_JPHI recon-anchor")
        except (ValueError, RuntimeError) as _pin_anchor_exc:
            # Solve failed -- mygs is in an indeterminate state.
            # Keep going so the downstream Ip-align / save_eqdsk
            # path at least produces something to compare; the
            # boundary will reflect whatever state the failed
            # solve left mygs in.
            print(f"  [recon-anchor] PIN_JPHI solve failed "
                  f"({_pin_anchor_exc}); proceeding with current "
                  f"mygs state -- per-draw boundary may not "
                  f"reflect perturbed pressure")
        # Refresh eq_stats after the anchor solve so the PIN_JPHI
        # short-circuit at line ~1237 reports the perturbed-pressure
        # equilibrium's l_i / Ip, not the pre-solve warmstart state.
        eq_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
        baseline_li_proxy = calc_cylindrical_li_proxy(
            mygs, input_j_phi, psi_pad)

    elif _diff_bs and recalculate_j_BS:
        # ---- DIFF_BS: differential bootstrap mode -----------------------
        # Restore mygs to the cached recon equilibrium state so SWB sees
        # the same context the cached SWB(recon kinetics) call did, then
        # call SWB on perturbed kinetics and subtract the cache to get a
        # delta that's applied on top of input_j_phi (recon's exact
        # j_phi).  At sigma->0 the perturbed kinetics == recon kinetics
        # so SWB output is identical to the cache, delta = 0, and
        # new_jphi = input_j_phi (= PIN_JPHI reproduction).
        print(f"  [DIFF_BS] restoring mygs to recon snapshot before SWB")
        mygs.replace_eq(source_eq=recon_eq_snapshot)
        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
        _swb_seed = create_power_flux_fun(npsi, 1.5, 1.5)['y']
        _stashed_bounds = getattr(mygs, '_coil_drift_bounds', None)
        if _stashed_bounds is not None:
            mygs.set_coil_bounds(None)
        try:
            _results_diff = solve_with_bootstrap(
                mygs,
                ne_eq, te_eq, ni_eq, ti_eq,
                Zeff, Ip_target, _swb_seed,
                scale_jBS=scale_jBS,
                isolate_edge_jBS=isolate_edge_jBS,
                diagnostic_plots=False,
                verbose=False,
            )
        finally:
            if _stashed_bounds is not None:
                mygs.set_coil_bounds(_stashed_bounds)
        # Convert SWB's parallel-projected j_BS to toroidal convention on the
        # SWB-landed equilibrium -- BEFORE the snapshot restore below changes
        # mygs. The cached recon spike was converted (and axis-smoothed) the
        # same way at cache time, so the delta is consistently toroidal.
        _spike_perturbed = smooth_jbs_transition(_swb_jbs_to_toroidal(
            mygs, _results_diff["isolated_j_BS"], psi_pad))
        _full_j_BS_tor = smooth_jbs_transition(_swb_jbs_to_toroidal(
            mygs, _results_diff["j_BS"], psi_pad))
        delta_spike = _spike_perturbed - spike_profile_recon_cached
        _delta_rms = float(np.sqrt(np.mean(delta_spike**2)))
        _delta_max = float(np.max(np.abs(delta_spike)))
        print(f"  [DIFF_BS] delta_spike rms={_delta_rms:.3e} A/m² "
              f"max={_delta_max:.3e} A/m² (-> 0 at sigma=0)")
        # Restore the recon snapshot a SECOND time so the recon-anchor
        # solve below operates from the same pristine state PIN_JPHI
        # sees.  Without this, the recon-anchor inherits SWB's landed
        # state (l_i ~ 0.86, off-target geometry) which gives the GS
        # solver a poor warm-start and produces large boundary shifts.
        mygs.replace_eq(source_eq=recon_eq_snapshot)
        # Build new_jphi as input_j_phi (recon exact) + delta_spike
        spike_profile = delta_spike
        full_j_BS = _full_j_BS_tor
        # ---- DIFF_BS recon-anchor solve (mirrors regular SWB branch's
        # recon-anchor at line ~1067 but with new_jphi = input_j_phi +
        # delta_spike).  Without this explicit solve, mygs stays in the
        # restored snapshot state (= recon equilibrium with recon j_phi)
        # which is identical for every draw -- the per-draw delta_spike
        # never reaches the equilibrium, so all draws produce bit-
        # identical output.
        new_jphi_diff = input_j_phi + delta_spike
        _psi_range_diff = mygs.psi_bounds[1] - mygs.psi_bounds[0]
        _pp_diff = {"type": "linterp",
                    "y": pchip_derivative(psi_N, pres_tmp) / _psi_range_diff,
                    "x": psi_N}
        _pp_diff["y"][-1] = 0.0
        _ffp_diff = {"type": "jphi-linterp", "y": new_jphi_diff, "x": psi_N}
        mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])
        mygs.set_profiles(pp_prof=_pp_diff, ffp_prof=_ffp_diff)
        try:
            mygs.solve()
            print(f"  [DIFF_BS recon-anchor] solved with input_j_phi + "
                  f"delta_spike (perturbed pressure)")
            if bnd_diag_callback is not None:
                bnd_diag_callback("after DIFF_BS recon-anchor")
        except (ValueError, RuntimeError) as _diff_anchor_exc:
            print(f"  [DIFF_BS recon-anchor] WARN: solve failed "
                  f"({_diff_anchor_exc}); state may be inconsistent")
        baseline_li_proxy = calc_cylindrical_li_proxy(
            mygs, new_jphi_diff, psi_pad)
        eq_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)

    elif recalculate_j_BS:
        # ---- SWB call hygiene -------------------------------------------
        # solve_with_bootstrap is sensitive to two things beyond kinetics:
        #
        #   1. mygs state (q profile, FSA quantities) -- drifts between
        #      calls if perturb is invoked from a non-recon state.
        #   2. inductive_jphi seed shape -- SWB amplitude-only-scales its
        #      seed and the bootstrap iteration's convergence path
        #      depends on the seed.  Feeding recon's *converged* peaky
        #      ``j_inductive_fit`` produces a ~22% bootstrap drift vs
        #      feeding the broad ``create_power_flux_fun`` seed that
        #      ``reconstruct_equilibrium`` itself uses.
        #
        # Both fixes here:
        #   (a) State anchor: one solve at recon's exact j_phi + targets
        #       to put mygs in recon's converged state.
        #   (b) Synthesise the SAME broad SWB seed that recon uses
        #       internally.  Decouples the SWB seed (a bootstrap-
        #       iteration starting point) from ``input_jinductive`` (the
        #       GPR baseline shape used downstream).
        # Clear coil_drift bounds for the ENTIRE state-anchor + SWB block.
        # If the state anchor solves under hard bounds and the QP lands
        # at a constrained corner (forward mode wants slightly different
        # coils than recon's inverse-mode-with-Ip-secant solution), mygs
        # ends up in a degenerate state and SWB inherits it -- producing
        # a warped j_BS even though SWB itself runs unconstrained.
        # Restore bounds after SWB so the rest of the perturbation
        # iteration (l_i match, jphi correction) continues constrained.
        _stashed_bounds = getattr(mygs, '_coil_drift_bounds', None)
        if _stashed_bounds is not None:
            mygs.set_coil_bounds(None)

        # ---- Weaken coil reg for the SWB exploration phase ----
        # generate_bouquet installs a STRONG soft-reg (weight ~1e4 targeting
        # recon coils) before the draw loop to keep coils near recon during
        # the post-perturb bounded/in-spec solve.  But SWB's internal
        # *inverse* GS solves (state-anchor + find_optimal_scale + H-mode
        # iteration) must EXPLORE the natural perturbed equilibrium, where
        # coils legitimately move off recon.  Leaving the strong reg active
        # makes the reg (pull to recon) fight the isoflux constraint (settle
        # the perturbed boundary), oscillating the fixed-point iteration ->
        # "Exceeded maxits" for any draw that pushes coils meaningfully off
        # recon (verified 2026-05: identical state+kinetics converge with
        # weak reg, diverge with 1e4 reg).  Stash the strong reg, install a
        # weak recon-like reg for the whole SWB block, restore in finally.
        _stashed_reg = getattr(mygs, '_strong_coil_reg', None)
        if _stashed_reg is not None:
            try:
                _weak_rt = []
                for _rn in mygs.coil_sets:
                    _weak_rt.append(mygs.coil_reg_term(
                        {_rn: 1.0}, target=0.0, weight=1.0))
                _weak_rt.append(mygs.coil_reg_term(
                    {'#VSC': 1.0}, target=0.0, weight=1e-2))
                mygs.set_coil_reg(reg_terms=_weak_rt)
            except Exception as _wreg_exc:
                print(f"  [SWB-hygiene] weak-reg install failed "
                      f"({_wreg_exc}); SWB runs under strong reg")
                _stashed_reg = None

        # Anchor at pres_tmp -- the full solve pressure (thermal + p_fast +
        # impurity + p_diff) that every OTHER solve site in this function
        # uses (the draw solve, PIN_JPHI, and the diff path all set
        # pax=pres_tmp[0]).  This was the one site still on the thermal-only
        # `pressure` argument, which made the anchor a genuinely different
        # equilibrium from the reconstruction that produced input_j_phi
        # (issue #35 Defect 1): on a 27 kPa-p_fast case the missing pressure
        # shrinks the Shafranov shift enough that the archived j_phi
        # integrates +4.1 % of Ip high on the anchor geometry.  The
        # positional `pressure` argument CANNOT simply carry the full
        # pressure instead: it doubles as the kinetics pressure-match target,
        # which is thermal by construction (feeding it full pressure fails
        # that loop by exactly the p_fast fraction -- measured 42.7 % on the
        # same case).  Note pres_tmp is the DRAW's own perturbed pressure, so
        # the anchor now tracks the draw it anchors, per maintainer decision
        # (2026-08-18): at sigma=0 this equals the baseline full pressure
        # bitwise; at sigma>0 it is the state the draw actually solves.
        _pre_pp = {"type": "linterp",
                    "y": pchip_derivative(psi_N, pres_tmp) /
                         (mygs.psi_bounds[1] - mygs.psi_bounds[0]),
                    "x": psi_N}
        _pre_pp["y"][-1] = 0.0
        _pre_ffp = {"type": "jphi-linterp",
                     "y": input_j_phi.copy(),
                     "x": psi_N}
        mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])
        mygs.set_profiles(pp_prof=_pre_pp, ffp_prof=_pre_ffp)
        try:
            mygs.solve()
        except (ValueError, RuntimeError):
            # Anchor failed (rare); fall through and let SWB cope.
            pass

        # ---- capture the ANCHOR geometry for the R2 Ip renormalisation ----
        # mygs is in recon's converged state right here and nowhere later:
        # solve_with_bootstrap (below) moves it to its own landed equilibrium,
        # which is what the R2 root used to be evaluated on.  See
        # _AnchorIpRenorm for the two defects that fixes and the measured
        # numbers.  Built ONLY on the R2 path, so the production ensemble path
        # (perturb_jind_in_anchor=False) pays nothing and is bit-identical.
        _anchor_ip = None
        _r2_mode = _r2_ip_mode()
        if perturb_jind_in_anchor and _r2_mode != 'legacy':
            try:
                _anchor_ip = _AnchorIpRenorm(
                    mygs, psi_N, input_j_phi, Ip_target, psi_pad,
                    mode=_r2_mode)
                print(f"  [R2-anchor] Ip renorm on the anchor geometry "
                      f"(mode={_r2_mode}, FSA measure self-check "
                      f"{100 * _anchor_ip.self_check:+.4f}%, archived "
                      f"total {100 * _anchor_ip.reference_bias:+.4f}% "
                      f"vs anchor Ip)", flush=True)
            except Exception as _aip_exc:
                print(f"  [R2-anchor] WARN: anchor capture failed "
                      f"({_aip_exc}); Ip renorm falls back to the "
                      f"SWB-landed geometry")

        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
        _swb_seed = create_power_flux_fun(npsi, 1.5, 1.5)['y']

        # ---- SWB debug instrumentation (BOUQUET_SWB_DEBUG=1) ----
        # State prints + pre/post .npz dumps so a failing draw can be replayed
        # offline from exactly this state. Dumps land in the system temp dir
        # (see _swb_dump_path) and are overwritten each draw; on failure the
        # pre-state is preserved separately so a later draw can't clobber it.
        if _swb_debug():
            try:
                _diag_axis = (float(mygs.o_point[0]), float(mygs.o_point[1]))
                _diag_Ip = float(mygs.get_globals()[0])
                _ped = (psi_N >= 0.85) & (psi_N <= 1.0)
                _coils_now, _ = mygs.get_coil_currents()
                print(f"  [SWB-diag] axis=({_diag_axis[0]:.4f},{_diag_axis[1]:+.5f}) "
                      f"Ip={_diag_Ip:+.0f}  bounds_cleared={_stashed_bounds is not None}")
                print(f"  [SWB-diag] te_eq[0]={te_eq[0]:.0f} eV (baseline {te[0]:.0f}), "
                      f"ne_eq[0]={ne_eq[0]:.2e} m^-3 (baseline {ne[0]:.2e})")
                print(f"  [SWB-diag] te_eq pedestal psi=[0.85,1]: "
                      f"min={te_eq[_ped].min():.0f} max={te_eq[_ped].max():.0f}  "
                      f"(monotone? {bool(np.all(np.diff(te_eq[_ped]) <= 0))})")
                # Largest coil drifts vs the stashed bounds midpoints
                # (bounds = [base - delta, base + delta] => base = mean).
                _stashed = getattr(mygs, '_coil_drift_bounds', None) or {}
                _drifts = sorted(
                    ((abs(float(_coils_now[_cn]) - 0.5 * (_b[0] + _b[1])),
                      _cn, 0.5 * (_b[0] + _b[1]))
                     for _cn, _b in _stashed.items() if _cn in _coils_now),
                    reverse=True)
                for _, _cn, _base in _drifts[:2]:
                    print(f"  [SWB-diag] {_cn}={float(_coils_now[_cn]):+.0f} A "
                          f"(recon {_base:+.0f}, drift "
                          f"{float(_coils_now[_cn]) - _base:+.0f})")
                _ps_coils, _ = mygs.get_coil_currents()
                np.savez(
                    _swb_dump_path('swb_prestate'),
                    psi=mygs.get_psi(False),
                    coil_names=np.array(list(_ps_coils.keys())),
                    coil_vals=np.array([float(v) for v in _ps_coils.values()]),
                    psi_N=psi_N,
                    ne_eq=ne_eq, te_eq=te_eq, ni_eq=ni_eq, ti_eq=ti_eq,
                    Zeff=np.atleast_1d(np.asarray(Zeff)),
                    Ip_target=np.array([Ip_target]),
                    swb_seed=_swb_seed,
                    scale_jBS=np.array([scale_jBS]),
                    isolate_edge_jBS=np.array([bool(isolate_edge_jBS)]),
                )
            except Exception as _diag_exc:
                print(f"  [SWB-diag] pre-SWB capture failed: {_diag_exc}")

        _t_swb0 = time.perf_counter()
        try:
            results = solve_with_bootstrap(
                mygs,
                ne_eq, te_eq, ni_eq, ti_eq,
                Zeff, Ip_target, _swb_seed,
                scale_jBS=scale_jBS,
                isolate_edge_jBS=isolate_edge_jBS,
                diagnostic_plots=False,
                verbose=_swb_debug(),
                iterations=swb_iterations,
            )
            if _swb_debug():
                print(f"  [SWB-diag] SWB call: {time.perf_counter()-_t_swb0:.1f}s")
                # Preserve this draw's kinetics as an in-spec control for
                # failing-vs-succeeding spike-shape comparison.
                try:
                    np.savez(
                        _swb_dump_path('swb_success'),
                        psi_N=psi_N,
                        ne_eq=ne_eq, te_eq=te_eq, ni_eq=ni_eq, ti_eq=ti_eq,
                        Zeff=np.atleast_1d(np.asarray(Zeff)),
                        Ip_target=np.array([Ip_target]),
                        swb_seed=_swb_seed,
                        scale_jBS=np.array([scale_jBS]),
                        isolated_j_BS=results.get('isolated_j_BS'),
                        j_inductive=results.get('j_inductive'),
                        total_j_phi=results.get('total_j_phi'),
                    )
                except Exception as _dump_exc:
                    print(f"  [SWB-diag] success dump failed: {_dump_exc}")
        except (TypeError, ValueError, RuntimeError):
            if _swb_debug():
                # Dump the SWB inputs for offline replay, and preserve the
                # pre-SWB state capture from being overwritten by later draws.
                try:
                    import shutil as _sh
                    np.savez(
                        _swb_dump_path('swb_failure'),
                        psi_N=psi_N,
                        ne_eq=ne_eq, te_eq=te_eq, ni_eq=ni_eq, ti_eq=ti_eq,
                        Zeff=np.atleast_1d(np.asarray(Zeff)),
                        Ip_target=np.array([Ip_target]),
                        swb_seed=_swb_seed,
                        scale_jBS=np.array([scale_jBS]),
                    )
                    _sh.copyfile(_swb_dump_path('swb_prestate'),
                                 _swb_dump_path('swb_prestate_FAILED'))
                    print(f"  [SWB-diag] failure inputs + pre-state dumped to "
                          f"{_swb_dump_path('swb_failure')} / "
                          f"{_swb_dump_path('swb_prestate_FAILED')}")
                except Exception as _dump_exc:
                    print(f"  [SWB-diag] failure dump failed: {_dump_exc}")
            raise
        finally:
            if _stashed_bounds is not None:
                mygs.set_coil_bounds(_stashed_bounds)
        # NOTE: the weak reg installed for SWB is intentionally LEFT ACTIVE
        # through the rest of perturb_kinetic_equilibrium -- the recon-anchor
        # solve and the l_i band loop are ALSO exploratory inverse solves
        # that must let coils settle the perturbed equilibrium.  Restoring
        # the strong reg here (as a first cut did) just handed the same
        # reg-vs-isoflux oscillation to those downstream solves (verified:
        # SWB then completed but the recon-anchor solve displaced ~35 mm and
        # the l_i-loop solve hit maxits).  generate_bouquet restores the
        # strong reg AFTER perturb returns, for the post-perturb bounded /
        # homotopy / in-spec phase where keeping coils near recon is the
        # point.
        # ---- Recon-anchored baseline (do NOT use SWB's j_inductive) ----
        # SWB seeds with create_power_flux_fun(1.5, 1.5) and alpha-scales
        # to match Ip, but produces a matched_j_inductive whose SHAPE
        # differs structurally from recon's eqdsk-fit j_inductive_fit.
        # Empirically (DIII-D reference case probe):
        # recon/SWB ratio = 3.4× at axis vs 1.4× at mid-radius -- not a
        # uniform scaling, so post-SWB ind_factor anchoring cannot reach
        # recon's l_i.  At σ=0 the SWB-natural baseline lives at l_i≈0.89
        # vs recon's 1.10 -- a ~19% systematic offset that breaks the
        # σ→0 reproducibility test (the pipeline is supposed to be a
        # forward operator returning recon's equilibrium when no
        # perturbation is applied).
        #
        # Fix: keep SWB's bootstrap profile (j_BS, isolated_j_BS),
        # which legitimately depends on the perturbed kinetics via
        # Sauter, but DROP its matched_j_inductive.  Use recon's
        # converged j_inductive_fit (passed in as input_jinductive)
        # as the inductive baseline.  At σ=0 the kinetics are
        # unchanged so j_BS recompute ≈ recon's stored j_BS, and
        # combined with input_jinductive the total j_phi recovers
        # recon's exactly -> l_i = recon's l_i, bnd_RMS ≈ 0.
        # Convert SWB's parallel-projected bootstrap (<j.B> R_avg/F) to
        # bouquet's toroidal convention <j_phi/R>/<1/R>, evaluated on the
        # SWB-landed equilibrium (no solves between the call and here).
        _use_spike_delta = (spike_delta_ref is not None
                            and spike_delta_baseline is not None)
        if _use_spike_delta:
            # Delta composition: baseline split + raw SWB delta.  Both SWB
            # terms are RAW (no smoothing of perturbed profiles): the
            # collapsed innermost-surface point is a deterministic
            # common-mode artifact (measured 0.44-0.51x its neighbours on
            # all 56 draws of the 2026-07 reference case) and cancels in
            # the difference, while the per-draw Sauter response passes
            # through unfiltered.  At sigma=0 the spike equals the baseline
            # split exactly.
            _spike_raw = _swb_jbs_to_toroidal(
                mygs, results["isolated_j_BS"], psi_pad)
            _full_raw = _swb_jbs_to_toroidal(mygs, results["j_BS"], psi_pad)
            _delta_bl = np.asarray(spike_delta_baseline, dtype=float)
            _delta_ref = np.asarray(spike_delta_ref, dtype=float)
            spike_profile = _delta_bl + (_spike_raw - _delta_ref)
            full_j_BS = _delta_bl + (_full_raw - _delta_ref)
            print(f"  [jBS-delta] spike = baseline + raw SWB delta "
                  f"(|delta| rms={np.sqrt(np.mean((_spike_raw - _delta_ref)**2))/1e3:.1f} kA/m²)")
        else:
            # smooth_jbs_transition repairs SWB's collapsed near-axis point
            # with the IDENTICAL treatment the recon split received --
            # without it, every draw target carries a 1-2 grid-point axis
            # divot vs the recon baseline (hollow core, q0 shifted +12%
            # wholesale at sigma=0).
            full_j_BS = smooth_jbs_transition(
                _swb_jbs_to_toroidal(mygs, results["j_BS"], psi_pad))
            spike_profile = smooth_jbs_transition(
                _swb_jbs_to_toroidal(mygs, results["isolated_j_BS"], psi_pad))

        # Floor the SWB bootstrap at 0 (drop unphysical negative excursions)
        # before it enters j_phi. Then, in Case-B "diff" mode, add the fixed
        # correction jBS_diff = FUSE_jBS - SWB_baseline so the bootstrap term
        # becomes SWB(perturbed) + (FUSE_jBS - SWB_baseline): at sigma=0 it
        # reduces to FUSE_jBS exactly, and per-draw it tracks the SWB delta.
        # spike_profile feeds EVERY downstream j_phi build (recon-anchor,
        # l_i-match, corrective), so this single reassignment covers them all.
        # (Not re-floored after the diff: a negative excursion there is the
        # genuine Case-B edge-misalignment signal we want to surface, not hide.)
        if floor_j_BS:
            _fdp = os.environ.get('FLOOR_DIAG', '')
            if _fdp:
                _nc = int((spike_profile < 0).sum())
                with open(_fdp, 'a') as _fh:
                    if _nc:
                        _wn = psi_N[spike_profile < 0]
                        _fh.write(f"clipped {_nc} pts min={spike_profile.min()/1e3:+.1f}kA "
                                  f"psiN=[{_wn.min():.3f},{_wn.max():.3f}]\n")
                    else:
                        _fh.write(f"noop min={spike_profile.min()/1e3:+.1f}kA\n")
            full_j_BS = np.clip(full_j_BS, 0.0, None)
            spike_profile = np.clip(spike_profile, 0.0, None)
        if jBS_diff is not None and not _use_spike_delta:
            # (In delta mode spike_delta_baseline is run.py's
            # bl.j_BS + jBS_diff, so the diff is already inside the spike --
            # adding it again would double-count.)
            spike_profile = spike_profile + np.asarray(jBS_diff, dtype=float)

        # Anchor: forward-solve with recon's inductive shape + SWB's
        # Sauter-recomputed bootstrap spike, using the perturbed pressure
        # profile (pres_tmp).  The SWB recompute is a core feature of
        # the workflow -- the bootstrap legitimately responds to the
        # per-draw perturbed kinetics via Sauter, and this is the only
        # path by which the bootstrap shape changes between draws.
        # Pinning to recon's stored bootstrap (the implied-bootstrap
        # approach) gives σ=0 exact recovery but kills the per-draw
        # Sauter response, which is the wrong trade for this study.
        #
        # PIN_JPHI=1 env var: diagnostic mode that pins j_phi to recon's
        # exact converged profile (no SWB spike, no GPR perturbation),
        # leaving only the perturbed pressure (P', pax) to drive the
        # per-draw equilibrium response.  Useful for isolating whether
        # the per-draw j_phi shape is causing systematic boundary shift.
        # _pin_jphi is read above (outside the recalculate_j_BS branch).
        if _pin_jphi:
            new_jphi = input_j_phi.copy()
        elif _diff_bs:
            # spike_profile already = delta_spike (perturbed - cached recon)
            new_jphi = input_j_phi + spike_profile
        else:
            _anchor_jind = input_jinductive
            if perturb_jind_in_anchor:
                # Fix C: GPR-perturb the inductive HERE (then accept the anchor),
                # so each draw carries a genuinely different j_ind without the
                # downstream find_optimal_scale/corrective that homogenize it.
                _j0a = input_jinductive[0]
                _candA = input_jinductive
                for _tryA in range(20):
                    _c = generate_perturbed_GPR(
                        psi_N, input_jinductive / _j0a,
                        sigma_profile=sigma_jphi / _j0a, length_scale=j_ls,
                        n_samples=1, rng=rng, diag_plot=False) * _j0a
                    if np.all(_c >= 0.0):
                        _candA = _c
                        break
                # Ip renorm on the ANCHOR geometry (see _AnchorIpRenorm), not
                # on the equilibrium SWB happened to land on.
                _sA = _r2_ip_scale(_anchor_ip, mygs, _candA,
                                   spike_profile + j_fixed_eff, psi_N,
                                   Ip_target)
                _anchor_jind = _sA * _candA
                _r2_scale_used = float(_sA)
                _r2_f_ind_used = _r2_f_ind(_anchor_ip, _candA)
                print(f"  [perturb-anchor] GPR-perturbed j_ind in anchor "
                      f"(Ip-renorm scale={_sA:.4f}"
                      + _fmt_s_and_find(_sA, _r2_f_ind_used, _r2_mode) + ")")
            new_jphi = _anchor_jind + spike_profile + j_fixed_eff
        _psi_range_anchor = mygs.psi_bounds[1] - mygs.psi_bounds[0]
        _pp_anchor = {"type": "linterp",
                      "y": pchip_derivative(psi_N, pres_tmp) / _psi_range_anchor,
                      "x": psi_N}
        _pp_anchor["y"][-1] = 0.0
        _ffp_anchor = {"type": "jphi-linterp", "y": new_jphi, "x": psi_N}

        _probe("entry to recon-anchor block")
        mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])
        _probe("after set_targets(Ip,pax)")
        mygs.set_profiles(pp_prof=_pp_anchor, ffp_prof=_ffp_anchor)
        _probe("after set_profiles(pp,ffp)")
        try:
            mygs.solve()
            _probe("after solve()")
            if _pin_jphi:
                print(f"  [recon-anchor] forward-solved with PINNED "
                      f"recon j_phi (PIN_JPHI=1, only pressure perturbs)")
            elif _diff_bs:
                print(f"  [recon-anchor] forward-solved with input_j_phi + "
                      f"differential-SWB delta (DIFF_BS=1)")
            else:
                print(f"  [recon-anchor] forward-solved with recon's "
                      f"j_inductive_fit + SWB Sauter spike "
                      f"(σ-perturbed kinetics)")
            if bnd_diag_callback is not None:
                bnd_diag_callback("after recon-anchor solve")
            _ket_stage_diag(mygs, "1-recon-anchor")
        except (ValueError, RuntimeError) as _anchor_exc:
            # Anchor failed -- fall back to SWB's natural total_j_phi
            # so the rest of the loop has a workable baseline.
            print(f"  [recon-anchor] WARN: solve failed ({_anchor_exc}); "
                  f"falling back to SWB total_j_phi")
            new_jphi = results["total_j_phi"]
            _ffp_fb = {"type": "jphi-linterp", "y": new_jphi, "x": psi_N}
            mygs.set_profiles(pp_prof=_pp_anchor, ffp_prof=_ffp_fb)
            try:
                mygs.solve()
            except Exception as _fb_exc:
                # deliberate mask (see issue #24) -- now counted, not silent
                _count_masked_anchor_failure("recon_anchor_fallback", _fb_exc)

        eq_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
        baseline_li_proxy = calc_cylindrical_li_proxy(mygs, new_jphi, psi_pad)

        # Fix C band-conditioning: an UNCONDITIONAL accept passed pathological
        # GPR draws on hard/high-l_i cases (l_i -30% accepted -> garbage).
        # Resample the GPR inductive (re-solving the anchor) until its l_i is in
        # band or max_li_iter is hit; on exhaustion REJECT the draw (raise,
        # caught per-draw by generate_bouquet) rather than accept it.
        if perturb_jind_in_anchor:
            _tolp = float(l_i_tolerance) * 100.0
            _j0a = input_jinductive[0]
            _nr = 0
            while (100.0 * abs(float(eq_stats['l_i']) - l_i_target) / l_i_target > _tolp
                   and _nr < int(max_li_iter)):
                _nr += 1
                for _t in range(20):
                    _c = generate_perturbed_GPR(
                        psi_N, input_jinductive / _j0a, sigma_profile=sigma_jphi / _j0a,
                        length_scale=j_ls, n_samples=1, rng=rng,
                        diag_plot=False) * _j0a
                    if np.all(_c >= 0.0):
                        break
                _sA = _r2_ip_scale(_anchor_ip, mygs, _c,
                                   spike_profile + j_fixed_eff, psi_N,
                                   Ip_target)
                new_jphi = _sA * _c + spike_profile + j_fixed_eff
                _r2_scale_used = float(_sA)
                _r2_f_ind_used = _r2_f_ind(_anchor_ip, _c)
                mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])
                mygs.set_profiles(pp_prof=_pp_anchor,
                                  ffp_prof={"type": "jphi-linterp", "y": new_jphi, "x": psi_N})
                try:
                    mygs.solve()
                except Exception as _rs_exc:
                    # deliberate mask (see issue #24) -- now counted, not silent
                    _count_masked_anchor_failure("band_resample", _rs_exc)
                    continue
                eq_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
            _erp = 100.0 * abs(float(eq_stats['l_i']) - l_i_target) / l_i_target
            print(f"  [perturb-anchor] band-conditioned: {_nr} resample(s), "
                  f"l_i={float(eq_stats['l_i']):.4f} ({_erp:.2f}% vs band {_tolp:.2f}%)",
                  flush=True)
            # QC line for the sigma=0 s-invariant: NEVER |s-1| on its own.
            if _r2_scale_used is not None:
                print(f"  [R2-invariant] s={_r2_scale_used:.6f}"
                      + _fmt_s_and_find(_r2_scale_used, _r2_f_ind_used,
                                        _r2_mode)
                      + "  (bound is on the product; issue #23)"
                      + _floored_zone_note(input_jinductive), flush=True)
            if _erp > _tolp:
                raise RuntimeError(
                    f"perturb_jind_in_anchor: no in-band draw in {int(max_li_iter)} "
                    f"resamples (last l_i err {_erp:.1f}%)")
            baseline_li_proxy = calc_cylindrical_li_proxy(mygs, new_jphi, psi_pad)
        # ---- DIAG: recon-anchor (SWB) l_i vs target, BEFORE the sampling
        # loop runs.  Quantifies how much of the per-draw l_i shift is the
        # PHYSICAL SWB-bootstrap response (this value) vs the downstream
        # GPR sampler.  If this already sits near l_i_target, the sampling
        # loop is unnecessary in free-jphi mode.
        try:
            _ra_li1 = float(eq_stats['l_i'])
            print(f"  [recon-anchor l_i] SWB equilibrium l_i(1)={_ra_li1:.5f} "
                  f"vs target {l_i_target:.5f} "
                  f"({100.0*(_ra_li1 - l_i_target)/l_i_target:+.2f}%); "
                  f"baseline_li_proxy={baseline_li_proxy:.5f}")
        except Exception:
            pass

        j0_scales.append(results["scale_j0"])
        Ip_scales.append(results["scale_Ip"])
        iteration_l_is.append(eq_stats["l_i"])
        iteration_Ips.append(eq_stats["Ip"])
    else:
        # When bootstrap is not recalculated there is no edge spike
        full_j_BS = np.zeros_like(psi_N)
        spike_profile = np.zeros_like(psi_N)
        baseline_li_proxy = calc_cylindrical_li_proxy(mygs, input_j_phi, psi_pad)

    # ----------------------------------------------------------------
    #  5.  l_i matching loop
    # ----------------------------------------------------------------
    l_i = np.inf
    final_scale_j0 = 1.0
    # Use recon's j_inductive_fit (input_jinductive) instead of SWB's
    # matched_j_inductive -- see [recon-anchor] block above for rationale.
    matched_j_inductive = (
        input_jinductive.copy() if recalculate_j_BS else input_j_phi.copy()
    )

    # The proxy target starts at the baseline proxy value but is
    # adaptively corrected after each TokaMaker solve to account for
    # the systematic offset between the cylindrical proxy and the
    # actual equilibrium l_i.  This makes the proxy filter select
    # profiles that land near l_i_target in equilibrium space rather
    # than in proxy space.
    # Initialize proxy_target.  If the caller passed a warmstart bias
    # (proxy_bias_warmstart = proxy / real_l_i from the previous draw),
    # use it -- so the very first outer iter of this draw lands at
    # ~l_i_target, converging in 1 iter instead of 2.  The cylindrical
    # proxy bias is empirically very stable across draws (~7%) so
    # warmstart is essentially free; without it every draw has to
    # waste one outer iter (~30s) re-discovering the bias.
    if proxy_bias_warmstart is not None and np.isfinite(proxy_bias_warmstart):
        proxy_target = l_i_target * proxy_bias_warmstart
        print(f"  [proxy-warmstart] proxy_target = "
              f"{proxy_target:.4f} (bias={proxy_bias_warmstart:.4f} from "
              f"previous draw)")
    else:
        proxy_target = baseline_li_proxy

    # ---- PIN_JPHI diagnostic short-circuit ----
    # When PIN_JPHI=1, skip the GPR-sampling l_i match loop entirely
    # and accept the recon-anchor's equilibrium (which used input_j_phi
    # as the fixed FF' shape) as the per-draw output.  Only pressure
    # (pres_tmp -> PP', pax) varies per draw; j_phi shape and Ip
    # target stay locked to recon.  Diagnostic mode for isolating
    # whether per-draw j_phi shape variation drives the boundary shift.
    if _pin_jphi or _diff_bs:
        _pinned_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
        l_i = float(_pinned_stats['l_i'])
        Ip = float(_pinned_stats['Ip'])
        # DIFF_BS: output = input_j_phi + delta_spike (delta=0 at sigma=0)
        # PIN_JPHI: output = input_j_phi exactly
        if _diff_bs:
            output_jphi = input_j_phi + spike_profile  # spike_profile = delta_spike
            _tag = "DIFF_BS"
        else:
            output_jphi = input_j_phi.copy()
            _tag = "PIN_JPHI"
        iteration_l_is.append(l_i)
        iteration_Ips.append(Ip)
        j0_scales.append(1.0)
        Ip_scales.append(1.0)
        final_li_proxy = calc_cylindrical_li_proxy(mygs, output_jphi, psi_pad)
        print(f"  [{_tag}] using {'input_j_phi+delta' if _diff_bs else 'recon j_phi'} "
              f"as output_jphi  (l_i={l_i:.4f}, Ip={Ip:.0f}); skipping l_i match loop")

    # ---- l_i band-conditioning loop ---------------------------------
    # Rejection sampling from the GPR prior conditioned on the measured
    # l_i: each iteration GPR-samples one jphi perturbation, solves the
    # equilibrium, and ACCEPTS it iff the resulting equilibrium l_i lands
    # within the measured band (`l_i_tolerance`, expressed as % of recon
    # l_i) of `l_i_target`.  Out-of-band draws are rejected and the next
    # iteration redraws (up to `max_li_iter`).  This preserves the
    # flagship GPR current-profile perturbation while keeping only draws
    # consistent with the magnetics -- it does NOT force every draw to a
    # single point l_i (the old behavior, which fought the perturbation
    # and was unreachable at large sigma_jphi).
    #
    # `l_i` is initialised to np.inf before the loop, so li_iter=1 always
    # draws+solves; the band check below then accepts/rejects.  The
    # accept-break sits at the TOP so an in-band draw exits before a
    # wasted extra draw.
    #
    # ---- Real-geometry l_i pre-screen geometry (built ONCE per draw) ----
    # Without a pre-screen, every GPR draw pays the full find_optimal_scale
    # + corrective refinement (~35 s) before the band check rejects it --
    # and at full sigma only ~30-40% of draws land in the +/-band, so ~3 of
    # every 4 draws are expensive rejects (~105 s/draw wasted).  The cheap
    # cylindrical proxy can't safely pre-screen them (its ~10% bias is
    # peakedness-correlated -> would bias the accepted ensemble).
    #
    # The REAL-geometry proxy (snapshot flux-surface perimeters L_p in the
    # li(1) formula) is ~unbiased (validated 0.994 vs solved l_i(1), ~1.2%
    # scatter, device-agnostic -- the geometry IS the calibration).  We
    # build it ONCE here, frozen on the recon-anchor equilibrium (this
    # draw's perturbed-P' geometry), and use it to SKIP draws whose
    # estimate is well outside the band; the real solved l_i remains the
    # sole acceptance criterion (see end of loop body).  Conservative
    # margin (PRESCREEN_MARGIN, default 3%) >> the proxy's bias+scatter so
    # no in-band draw is skipped.  Set PRESCREEN=0 to disable (solve every
    # draw, as before).
    _prescreen_geo = None
    # Default OFF: the interior-surface perimeter tracing in the geometry
    # build leaves the module-level tracer state dirty, which corrupts the
    # subsequent l_i-loop free-boundary solve (verified A/B same-seed:
    # 2/3 without vs 0/3 with).  A final LCFS-trace reset (below) is the
    # attempted fix; keep PRESCREEN opt-in until it's confirmed clean.
    _prescreen_on = (os.environ.get('PRESCREEN', '0') == '1'
                     and not (_pin_jphi or _diff_bs))
    if _prescreen_on:
        try:
            _pg = get_li_proxy_geometry(mygs, npsi, psi_pad)
            _n_trace = int(os.environ.get('PRESCREEN_NTRACE', '20'))
            _lev = np.linspace(0.06, 1.0 - psi_pad, _n_trace)
            # Trace each perimeter with safe_trace_surf (per-call copy_eq /
            # replace_eq).  Raw consecutive trace_surf calls accumulate
            # gs_equil mutations that a single end-of-loop replace_eq does
            # NOT fully undo, corrupting the subsequent l_i-loop GS solve
            # (verified: same-seed draw succeeds without the build, maxits
            # with raw traces).  safe_trace_surf is the validated-clean
            # wrapper used elsewhere before solves.
            _Lp_lev = []
            for _L in _lev:
                _c = safe_trace_surf(mygs, _L)
                if _c is None or len(_c) < 4:
                    _Lp_lev.append(np.nan)
                    continue
                _c = np.asarray(_c)
                _d = np.diff(_c, axis=0)
                _Lp_lev.append(float(np.sum(np.hypot(_d[:, 0], _d[:, 1]))))
            _Lp_lev = np.array(_Lp_lev)
            # Drop any failed traces before interpolation.
            _ok = np.isfinite(_Lp_lev)
            _lev = _lev[_ok]; _Lp_lev = _Lp_lev[_ok]
            _xi = np.concatenate([[0.0], _lev])
            _yi = np.concatenate([[0.0], _Lp_lev])
            from scipy.interpolate import interp1d as _interp1d_ps
            _pg['L_p'] = _interp1d_ps(_xi, _yi, kind='linear',
                                      bounds_error=False,
                                      fill_value=(0.0, _Lp_lev[-1]))(psi_N)
            # Reset the module-level tracer to the LCFS: the interior traces
            # above leave it on an interior surface, which corrupts the
            # downstream free-boundary solve's LCFS search.  A final LCFS
            # trace leaves the tracer where the solve expects it (this is
            # the state the bnd-diag LCFS trace normally leaves it in).
            _ = safe_trace_surf(mygs, 1.0 - psi_pad)
            _prescreen_geo = _pg
            print(f"  [prescreen] real-geom geometry built "
                  f"({_n_trace} perimeter traces; L_p edge={_pg['L_p'][-1]:.2f} m)")
        except Exception as _ps_exc:
            print(f"  [prescreen] geometry build failed ({_ps_exc}); "
                  f"pre-screen disabled (solving every draw)")
            _prescreen_geo = None
    _prescreen_margin = float(os.environ.get('PRESCREEN_MARGIN', '3.0'))
    _prescreen_verify = os.environ.get('PRESCREEN_VERIFY', '0') == '1'

    # l_i_tolerance is a FRACTION of l_i_target (e.g. 0.05 == 5%).  Internally
    # the acceptance band, pre-screen window and reporting are in percent, so
    # convert once here.  (_prescreen_margin stays in percent.)
    _li_tol_pct = float(l_i_tolerance) * 100.0

    # Fix B/C: accept the recon-anchor and skip find_optimal_scale + corrective
    # (which overshoot l_i and drift degenerate coils off baseline). B accepts
    # only when the anchor l_i is already in-band; C accepts always (the
    # perturbed-in-anchor j_ind IS the draw; l_i floats with the bounded GPR).
    _accept_anchor = False
    if (recalculate_j_BS and not (_pin_jphi or _diff_bs)
            and (perturb_jind_in_anchor or accept_anchor_inband)):
        try:
            _anchor_li = float(eq_stats['l_i'])
            _inband = 100.0 * abs(_anchor_li - l_i_target) / l_i_target <= _li_tol_pct
            if perturb_jind_in_anchor or _inband:
                _accept_anchor = True
                l_i = _anchor_li
                Ip = float(eq_stats['Ip'])
                output_jphi = new_jphi.copy()
                iteration_l_is.append(l_i)
                iteration_Ips.append(Ip)
                j0_scales.append(1.0)
                Ip_scales.append(1.0)
                final_li_proxy = calc_cylindrical_li_proxy(mygs, output_jphi, psi_pad)
                _why = "C/perturb-anchor" if perturb_jind_in_anchor else "B/inband"
                print(f"  [ACCEPT-ANCHOR {_why}] l_i={l_i:.4f} "
                      f"({100.0*abs(_anchor_li-l_i_target)/l_i_target:.2f}% vs "
                      f"band {_li_tol_pct:.2f}%); skip find_optimal_scale+corrective",
                      flush=True)
        except Exception as _e:
            print(f"  [ACCEPT-ANCHOR] check failed: {_e!r}", flush=True)

    for li_iter in range(1, max_li_iter + 1):
        if _pin_jphi or _diff_bs or _accept_anchor:
            break  # PIN_JPHI / DIFF_BS / accept-anchor shortcut handled above
        if 100.0 * abs(l_i - l_i_target) / l_i_target <= _li_tol_pct:
            break  # last draw's equilibrium l_i is within the measured band

        t_phase = time.perf_counter()

        # ---- 5a. Draw a GPR j_phi perturbation (flagship), pre-screened --
        # GPR-perturb the inductive current shape around recon's
        # j_inductive_fit with envelope sigma_jphi (the core bouquet
        # current-profile uncertainty sampling).
        #
        # Band conditioning: a draw is ACCEPTED downstream iff the *real*
        # solved equilibrium l_i lands within `l_i_tolerance` of `l_i_target`
        # (see end of loop body) -- rejection sampling from the GPR prior
        # conditioned on the magnetics.  To avoid paying the full
        # find_optimal_scale + corrective solve on draws that are obviously
        # out-of-band, we pre-screen each GPR draw with the cheap
        # REAL-geometry l_i proxy (frozen snapshot geometry built above) and
        # skip the solve when its estimate is outside band + PRESCREEN_MARGIN.
        # The margin (>> the proxy's ~0.6% bias + ~1.2% scatter) guarantees
        # no in-band draw is skipped, so the accepted ensemble is unchanged
        # (the proxy is a speed-only gate, NOT the acceptance test).
        step_j_phi = (
            input_jinductive if recalculate_j_BS else input_j_phi
        )
        j_phi_0 = step_j_phi[0]
        _geo = _prescreen_geo  # frozen recon-anchor geometry (or None)
        # Floor zone: where the GPR mean is at/near zero (<= sigma), e.g. the
        # zero-anchored edge or a floored pedestal residual, a Gaussian sample
        # is negative with O(50%) probability PER POINT -- hard-rejecting those
        # draws can exhaust all tries (observed 0/500 on a strong-pedestal case). There a
        # non-negative quantity is properly half-Gaussian: CLIP to zero instead.
        # Negative excursions where the mean is materially positive (> sigma)
        # still reject the draw -- that remains a genuinely pathological sample.
        _floor_zone = np.asarray(step_j_phi, dtype=float) <= np.asarray(
            sigma_jphi, dtype=float)

        jphi_perturb = None
        a_optimal = None
        matched_jphi_perturb = None
        _n_skipped = 0
        _ps_window = _li_tol_pct + _prescreen_margin  # % of l_i_target
        for _gpr_try in range(1, max_proxy_draws + 1):
            _cand = generate_perturbed_GPR(
                psi_N,
                step_j_phi / j_phi_0,
                sigma_profile=sigma_jphi / j_phi_0,   # normalised σ
                length_scale=j_ls,
                n_samples=1,
                rng=rng,
                diag_plot=False,
            ) * j_phi_0
            if np.any((_cand < 0.0) & ~_floor_zone):
                continue  # non-physical (negative current where mean >> 0)
            _cand = np.clip(_cand, 0.0, None)   # half-Gaussian at the floor
            _root = root_scalar(
                Ip_flux_integral_vs_target,
                args=(mygs, _cand, spike_profile + j_fixed_eff, psi_N, Ip_target),
                bracket=[1.0e-10 * Ip_target, 1.0e1 * Ip_target],
                method="brentq", rtol=1e-6,
            )
            _a = _root.root
            _matched = _a * _cand + spike_profile + j_fixed_eff
            # Cheap real-geom pre-screen: skip if confidently out-of-band.
            if _prescreen_geo is not None:
                _est = calc_realgeom_li_proxy_fast(_matched, _prescreen_geo)
                _est_err = 100.0 * abs(_est - l_i_target) / l_i_target
                if _est_err > _ps_window:
                    _n_skipped += 1
                    if _prescreen_verify and _n_skipped <= 3:
                        print(f"    [prescreen-verify] skipped draw: "
                              f"real-geom est l_i={_est:.4f} "
                              f"({_est_err:.1f}% > window {_ps_window:.1f}%)")
                    continue
            # Passed pre-screen (or pre-screen disabled): take this draw.
            jphi_perturb = _cand
            a_optimal = _a
            matched_jphi_perturb = _matched
            break
        if jphi_perturb is None:
            raise RuntimeError(
                f"No GPR j_phi draw passed the pre-screen in "
                f"{max_proxy_draws} tries (window +/-{_ps_window:.1f}%); "
                f"widen l_i_tolerance/PRESCREEN_MARGIN or check sigma_jphi")

        dt_proxy = time.perf_counter() - t_phase
        print(f"  [li_iter={li_iter}] GPR draw "
              f"({_gpr_try} tries, {_n_skipped} pre-screen-skipped, "
              f"{dt_proxy:.1f}s)")

        # ---- 5b. Set up GS profiles --------------------------------
        psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
        pprime_tmp = pchip_derivative(psi_N, pres_tmp) / psi_range
        pprime_tmp[-1] = 0.0

        pp_prof = {"type": "linterp", "y": pprime_tmp, "x": psi_N}
        ffp_prof = {
            "type": "jphi-linterp",
            "y": matched_jphi_perturb,
            "x": psi_N,
        }

        matched_j_inductive = a_optimal * jphi_perturb

        # ---- 5c. Find optimal scale factors -------------------------
        t_scale = time.perf_counter()
        final_scale_j0, final_jphi = find_optimal_scale(
            mygs, psi_N, pres_tmp, ffp_prof, pp_prof,
            matched_j_inductive, Ip_target, psi_pad,
            spike_prof=spike_profile + j_fixed_eff,
            diagnostic_plots=False, verbose=False,
        )

        # Preliminary q_0 check: the j_phi scale solve has already
        # converged, so we can reject before the more expensive Ip
        # scale solve.  A definitive check follows after Ip scaling.
        if constrain_sawteeth:
            _, q_pre, _, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
            if q_pre[0] < 1.0:
                dt_scale = time.perf_counter() - t_scale
                print(f"  [li_iter={li_iter}] find_optimal_scale: {dt_scale:.1f}s")
                print("Skipping this equilibrium, q_0 < 1.0 (pre-check)")
                l_i = np.inf
                continue

        # Only the core-j0 scale is applied above; the OFT
        # solver holds Ip to target natively, so Ip_target is used unscaled
        # downstream (no Ip-scale secant).
        dt_scale = time.perf_counter() - t_scale
        print(f"  [li_iter={li_iter}] find_optimal_scale (j0 only): {dt_scale:.1f}s")
        _ket_stage_diag(mygs, f"2-after-find_optimal_scale[iter{li_iter}]",
                        extra=f" j0_scale={final_scale_j0:.4f}")

        # ---- 5d. Definitive sawtooth constraint (after Ip scaling) --
        if constrain_sawteeth:
            _, q, _, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
            if q[0] < 1.0:
                print("Skipping this equilibrium, q_0 < 1.0")
                l_i = np.inf
                continue

        j0_scales.append(final_scale_j0)
        Ip_scales.append(1.0)   # Ip held natively; no Ip-scale applied

        # ---- 5e. Adaptive corrective iteration ----------------------
        # Iterate TokaMaker until its output j_phi matches the intended
        # input (j_inductive*scale + spike).  This compensates for
        # geometry coupling that distorts the edge profile.
        #
        # SPIKE SOURCE: SWB's Sauter-recomputed isolated_j_BS spike,
        # matching the recon-anchor block above.  The Sauter recompute
        # legitimately responds to the per-draw perturbed kinetics --
        # this is the core feature of the workflow, not an artifact to
        # be removed.  At σ=0 the SWB-vs-recon bootstrap mismatch leaves
        # a ~5-13 mm structural floor in boundary RMS; that's the
        # honest cost of having per-draw Sauter response.
        pprime_tmp = pchip_derivative(psi_N, pres_tmp) / psi_range
        pprime_tmp[-1] = 0.0
        pp_prof = {"type": "linterp", "y": pprime_tmp, "x": psi_N}

        target_jphi_perturb = (
            matched_j_inductive * final_scale_j0 + spike_profile + j_fixed_eff
        )
        # Issue #29 (second site): the inductive amplitude above was rooted on
        # the limiter-area flux integral (#15), so the assembled target does
        # not carry Ip_target in the physical measure either.  Same uniform
        # renormalisation as the reconstruction site -- the solver will scale
        # the input to Ip regardless; make the target the profile it can
        # actually return, so the Newton update stops re-injecting refused
        # current.
        target_jphi_perturb, _corr_ip_factor = _renormalize_target_to_Ip(
            mygs, psi_N, target_jphi_perturb, Ip_target, psi_pad,
            label="jphi_corr/draw")

        output_jphi, _n_corr, _corr_hist = _corrective_jphi_iteration(
            mygs, psi_N, target_jphi_perturb, pp_prof,
            Ip_target, pres_tmp[0], psi_pad,
            min_iters=2,
            # Corrective-iteration cap (default 8; observed to converge ~5).
            # Env CORR_MAX_ITERS lets us trim for speed (4 saves ~1 solve).
            max_iters=int(os.environ.get('CORR_MAX_ITERS', '8')),
            rtol=0.05, verbose=False,
        )
        if _n_corr > 2:
            print(f"  [jphi correction] {_n_corr} iterations, "
                  f"edge RMS: {_corr_hist[0]/1e6:.4f} → {_corr_hist[-1]/1e6:.4f} MA/m²")

        if diagnostic_plots:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(psi_N, matched_jphi_perturb, label=r"Input $j_\phi$")
            ax.plot(psi_N, output_jphi, label=r"Converged $j_\phi$")
            ax.fill_between(
                psi_N,
                input_j_phi - sigma_jphi,
                input_j_phi + sigma_jphi,
                alpha=0.3, label=r"$\pm\,1\sigma_{\rm exp}$ envelope",
            )
            ax.set_ylim(
                0.0,
                max(input_j_phi[0], (input_j_phi + sigma_jphi)[0]),
            )

            ax.legend(loc="best")
            ax.set_xlabel(r"$\hat{\psi}$")
            ax.set_ylabel(r"$j_\phi$ [A/m$^2$]")
            ax.set_title(f"jphi-linterp  |  l_i iter {li_iter}")
            plt.tight_layout()
            plt.show()

        eq_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
        Ip = eq_stats["Ip"]
        l_i = eq_stats["l_i"]

        # Compute cylindrical proxy on the FINAL converged j_phi purely to
        # report the proxy-vs-TokaMaker l_i offset (diagnostic; the proxy no
        # longer gates draw acceptance -- that is the equilibrium-l_i band).
        # _geo may be None if the pre-screen geometry build failed or
        # PRESCREEN=0; build a one-off cylindrical cache for the diagnostic.
        _geo_diag = _geo if _geo is not None else get_li_proxy_geometry(
            mygs, npsi, psi_pad)
        _ket_stage_diag(mygs, f"3-after-corrective[iter{li_iter}]")
        final_li_proxy = calc_cylindrical_li_proxy_fast(output_jphi, _geo_diag)
        proxy_vs_real = 100.0 * (final_li_proxy - l_i) / l_i if l_i != 0 else 0.0

        Ip_err = 100.0 * abs(Ip - Ip_target) / Ip_target

        # Adaptive proxy target correction: use the observed
        # proxy-to-equilibrium mapping to predict what proxy value
        # would produce l_i_target in the actual equilibrium.
        # Pure-Newton blend (1.0 / 0.0).  The cylindrical proxy has a
        # structurally fixed bias (~7%) vs the TokaMaker equilibrium
        # l_i for a given j_phi shape; the linearization is accurate,
        # so a full Newton step converges in 2-3 iters where the old
        # 0.7/0.3 blend was conservative enough that the outer loop
        # broke on l_i_tolerance=10 default and never corrected.
        if l_i > 0 and np.isfinite(l_i):
            proxy_target = final_li_proxy * (l_i_target / l_i)

        print(f"  l_i target (equil):   {l_i_target:.4f}")
        print(f"  proxy target:         {proxy_target:.4f}  (corrected)")
        print(f"  matched l_i (equil):  {l_i:.4f}")
        print(f"  matched l_i (proxy):  {final_li_proxy:.4f}")
        print(f"  Ip error vs target:   {Ip_err:.3f}%")
        print(f"  proxy vs real l_i:    {proxy_vs_real:+.2f}%")
        _li_pct_err = 100.0 * abs(l_i - l_i_target) / l_i_target if l_i_target != 0 else float('inf')
        print(f"  l_i error:            {_li_pct_err:.2f}% (tolerance: {_li_tol_pct:.2f}%)")

        iteration_l_is.append(l_i)
        iteration_Ips.append(Ip)
    else:
        # Fired only if no GPR draw landed within the l_i band in
        # max_li_iter attempts.
        raise RuntimeError(
            f"No GPR j_phi draw landed within the l_i band after "
            f"{max_li_iter} attempts "
            f"(last l_i error = {100.0 * abs(l_i - l_i_target) / l_i_target:.2f}%, "
            f"band = +/-{_li_tol_pct:.2f}%, "
            f"target={l_i_target:.4f}, last={l_i:.4f}).\n"
            f"Try widening the l_i band (l_i_tolerance) or increasing "
            f"max_li_iter."
        )

    # ----------------------------------------------------------------
    #  6.  Package outputs
    # ----------------------------------------------------------------
    if bnd_diag_callback is not None:
        bnd_diag_callback("after l_i match loop")
    # NOTE: w_ExB (E×B rotation) is not yet computed from the
    # perturbed equilibrium.  A zero placeholder is stored so the
    # output tuple and HDF5 schema remain forward-compatible.
    w_ExB = np.zeros_like(psi_N)

    # j_inductive decomposition. Two regimes:
    #   isolate_edge_jBS=True  (DIII-D g-file edge-spike work): the bootstrap is
    #     an isolated flat-shelf + edge spike, so j_inductive tapers to zero at
    #     the edge where the spike dominates -- the shelf-blend Hermite handles
    #     the C1 join and edge taper.
    #   isolate_edge_jBS=False (FUSE/IMAS full bootstrap): spike_profile is NOT a
    #     flat-shelf spike (it is a full Sauter profile / its delta), so the
    #     shelf-blend mis-detects the shelf and mangles the core. Use the clean
    #     residual j_inductive = j_phi - j_BS - j_NBI - j_RF instead -- it sums
    #     exactly and mirrors read_imas_baseline's baseline decomposition.
    if isolate_edge_jBS:
        # Closing decomposition (option A), replacing the non-closing shelf-blend
        # Hermite: j_inductive = j_phi - j_BS,edge - j_fixed, so the stored
        # components sum exactly to j_phi (the old shelf-blend substituted a
        # Hermite taper past the shelf, so j_ind + spike != j_phi at the edge).
        # The core bootstrap stays folded into j_inductive -- the isolated-edge-
        # spike construct -- only the non-closing edge substitution is removed.
        _jfix_iso = np.zeros_like(psi_N)
        if j_NBI is not None:
            _jfix_iso = _jfix_iso + np.asarray(j_NBI, dtype=float)
        if j_RF is not None:
            _jfix_iso = _jfix_iso + np.asarray(j_RF, dtype=float)
        j_inductive_consistent = output_jphi - spike_profile - _jfix_iso
        # Where the edge spike locally exceeds the available current (near the
        # spike peak -- what the Hermite used to smooth), floor j_inductive at 0
        # and cap the stored edge spike so both stay >= 0 and closure holds.
        _neg = j_inductive_consistent < 0.0
        if np.any(_neg):
            j_inductive_consistent = np.maximum(j_inductive_consistent, 0.0)
            spike_profile = np.where(_neg, output_jphi - _jfix_iso, spike_profile)
    else:
        _jfix_store = np.zeros_like(psi_N)
        if j_NBI is not None:
            _jfix_store = _jfix_store + np.asarray(j_NBI, dtype=float)
        if j_RF is not None:
            _jfix_store = _jfix_store + np.asarray(j_RF, dtype=float)
        # j_BS is the PHYSICAL bootstrap that was summed into the solve:
        # spike_profile == Sauter(perturbed kinetics) * scale_jBS + jBS_diff
        # (the recomputed Sauter on the per-draw kinetics, anchored by the kept
        # baseline diff) -- a forward-COMPUTED quantity, not a residual. The ohmic
        # is then j_phi - j_BS - j_fixed, which recovers exactly the GPR-perturbed,
        # Ip-renormalised inductive that went into the anchor (output ~= anchor
        # total = _anchor_jind + spike_profile + j_fixed), so it keeps its physical
        # edge foot and its genuine per-draw spread.
        #
        # The previous code subtracted `full_j_BS` (Sauter WITHOUT the diff)
        # instead, so jBS_diff leaked into the ohmic residual and dragged it to 0 /
        # negative at the pedestal -- the unphysical early cutoff. Using
        # spike_profile (WITH the diff) fixes both: physical j_BS, physical ohmic.
        full_j_BS = spike_profile.copy()
        j_inductive_consistent = output_jphi - full_j_BS - _jfix_store
        # Safety: the GPR Ip-renorm + edge realisation can leave a sub-kA negative
        # sliver in the ohmic at the very separatrix; floor it and absorb into j_BS
        # (bootstrap fraction -> 1 there) so both stay >= 0 and closure holds.
        _neg = j_inductive_consistent < 0.0
        if np.any(_neg):
            j_inductive_consistent = np.maximum(j_inductive_consistent, 0.0)
            full_j_BS = output_jphi - j_inductive_consistent - _jfix_store

    # Compute the cylindrical-proxy / real-l_i ratio at the converged
    # state.  Used by generate_bouquet to warmstart the next draw's
    # proxy_target so it converges in 1 outer iter instead of 2.
    final_l_i_for_bias = (iteration_l_is[-1]
                          if iteration_l_is else float('nan'))
    if (final_l_i_for_bias and np.isfinite(final_l_i_for_bias)
            and final_l_i_for_bias > 0
            and 'final_li_proxy' in dir()):
        proxy_bias_observed = final_li_proxy / final_l_i_for_bias
    else:
        proxy_bias_observed = None

    diagnostics = {
        "j0_scales": j0_scales,
        "Ip_scales": Ip_scales,
        "iteration_l_is": iteration_l_is,
        "iteration_Ips": iteration_Ips,
        "j_inductive": j_inductive_consistent,
        "j_BS": full_j_BS,
        # Only an isolated edge spike is a meaningful separate "j_BS,edge"; in
        # full-bootstrap mode spike_profile == j_BS (or its delta), so storing it
        # would just draw a redundant/mislabelled curve. Drop it there.
        "j_BS_edge": spike_profile if isolate_edge_jBS else None,
        "proxy_bias_observed": proxy_bias_observed,
        # Route-R2 inductive Ip-renormalisation scale (None off R2).  The
        # golden invariant: at sigma=0 the archived split is reproduced, so
        # this must be 1.000.  See _AnchorIpRenorm.
        "r2_ip_scale": _r2_scale_used,
        # ...and the inductive share it was normalised by, WITHOUT which
        # `r2_ip_scale` is not comparable between operating points:
        # |s-1| = |Delta/Ip| / f_ind.  The acceptance bound is on the product
        # (issue #23); see _AnchorIpRenorm.inductive_share.
        "r2_f_ind": _r2_f_ind_used,
        "aux": aux_out,
    }

    return (
        ne_perturb,
        te_perturb,
        ni_perturb,
        ti_perturb,
        w_ExB,
        output_jphi,
        diagnostics,
    )


# ====================================================================
#  Top-level scan driver
# ====================================================================
def generate_bouquet(
    mygs,
    psi_N,
    n_equils,
    header,
    input_j_phi,
    ne,
    te,
    ni,
    ti,
    sigma_ne,
    sigma_te,
    sigma_ni,
    sigma_ti,
    sigma_jphi,
    n_ls,
    t_ls,
    j_ls,
    initial_Ip_target,
    l_i_target,
    Zeff,
    input_jinductive=None,
    baseline_j_BS=None,
    l_i_tolerance=0.01,
    psi_pad=1e-3,
    constrain_sawteeth=True,
    recalculate_j_BS=True,
    isolate_edge_jBS=True,
    floor_j_BS=True,
    jBS_diff=None,
    accept_anchor_inband=False,
    perturb_jind_in_anchor=False,
    jBS_scale_range=None,
    # Delta composition (GenerationConfig.jbs_delta_mode): per-draw spike =
    # baseline_j_BS + (SWB_raw(perturbed) - SWB_raw(sigma=0)); the sigma=0
    # reference is cached once below in the same pre-draw anchor context.
    jbs_delta_mode=False,
    swb_iterations=3,
    diagnostic_plots=True,
    scan_key=None,
    pfile_bytes=None,
    Zeff_profile=None,
    baseline_eqdsk_bytes=None,
    baseline_pfile_bytes=None,
    psi_N_kinetic=None,
    max_proxy_draws=500,
    coil_drift=0.01,
    coil_drift_floor_A=50.0,
    vsc_coils=('F9A', 'F9B'),
    vsc_budget_frac=0.5,
    coil_drift_hard_factor=None,
    soft_reg_weight=1.0e4,
    vsc_soft_reg_weight=1.0,
    p_thresh=0.05,
    homotopy_passes=None,
    inspec_F_max=0.025,
    inspec_VSC_max=0.10,
    recon_lcfs_ref=None,
    l_i_uncertainty=0.0,
    save_truncate_eq=True,
    jphi_baseline=True,
    seed=None,
    pin_jphi=False,
    p_fast=None,
    Z_imp=None,
    p_diff=None,
    jphi_diff=None,
    j_NBI=None,
    j_RF=None,
    aux_sigmas=None,
    aux_baselines=None,
    aux_length_scales=None,
    ni_from_zeff=True,
    progress_callback=None,
    source_kind=None,
    capture_live_eq=True,
    capture_npsi=257,
    capture_exact_inv_R2=True,
    # Archive the ACHIEVED FSA j_phi of each converged solve (baseline + every
    # draw) instead of the prescribed target profile. Set on the IMAS path,
    # where the single-pass jphi-linterp solve lands a few % off its anchor, so
    # the stored 1-D j_phi matches the stored eqdsk (option-A semantics);
    # j_inductive is recomputed as the residual so closure stays exact. The
    # geqdsk path leaves this False: its corrective iteration already drives
    # achieved ~= target, and its baseline stores the corrective output.
    store_achieved_jphi=False,
):
    r"""Generate a batch of perturbed equilibria and archive to HDF5.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker GS solver object.
    psi_N : ndarray
        1-D normalised flux grid :math:`\hat{\psi}`.
    n_equils : int
        Number of perturbed equilibria to generate.
    header : str
        Base name for the HDF5 database.
    input_j_phi : ndarray
        1-D baseline toroidal current density [A/m\ :sup:`2`].
    ne : ndarray
        1-D electron density [m\ :sup:`-3`].
    te : ndarray
        1-D electron temperature [eV].
    ni : ndarray
        1-D ion density [m\ :sup:`-3`].
    ti : ndarray
        1-D ion temperature [eV].
    sigma_ne : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_e` [m\ :sup:`-3`].
    sigma_te : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_e` [eV].
    sigma_ni : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_i` [m\ :sup:`-3`].
    sigma_ti : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_i` [eV].
    sigma_jphi : ndarray
        1-D experimental :math:`1\sigma` for :math:`j_\phi` [A/m\ :sup:`2`].
    n_ls : float
        GPR length-scale for density profiles.
    t_ls : float
        GPR length-scale for temperature profiles.
    j_ls : float or ndarray
        GPR length-scale for :math:`j_\phi`.  A 1-D array gives a
        non-stationary Gibbs kernel (see ``sigmoid_length_scale``).
    initial_Ip_target : float
        Target plasma current [A].
    l_i_target : float
        Target internal inductance.
    Zeff : ndarray
        Effective ion charge profile on the equilibrium grid ``psi_N``
        (consumed by ``solve_with_bootstrap``). A scalar is accepted and
        broadcast, but the class API always passes the per-draw profile.
    input_jinductive : ndarray or None
        Dimensionless inductive :math:`j_\phi` shape.
    l_i_tolerance : float
        :math:`l_i` tolerance as a FRACTION of ``l_i_target`` (e.g.
        ``0.01`` == 1 %).
    jphi_baseline : bool
        When True (default), solve the recon profiles ONCE through the
        same jphi-linterp machinery the perturbed draws use and
        reference all per-draw diagnostics (boundary, :math:`l_i`) to
        THIS baseline rather than recon's inverse-mode LCFS.  This
        removes the constant jphi-linterp edge-representation offset so
        an unperturbed (:math:`\sigma=0`) draw lands ~0 mm / ~0 % from
        baseline and perturbations show their true incremental response.
    seed : int or None
        The run's single random seed.  It is consumed once, into one
        ``numpy.random.Generator`` (see
        :func:`bouquet.sampling.make_rng`) that is threaded explicitly into
        every draw site -- the GPR kinetic/aux/:math:`j_\phi` draws, the
        per-draw ``scale_jBS`` sample and the per-draw :math:`l_i` target
        sample.  Two runs with the same seed, inputs and solver therefore
        produce bitwise-identical archives.  ``None`` (default) draws from
        fresh OS entropy, i.e. the run is deliberately not regenerable.
    pin_jphi : bool
        When True, pin :math:`j_\phi` to recon's converged shape so only the
        pressure profile perturbs per draw (the bootstrap/SWB call is
        bypassed).  At :math:`\sigma=0` every draw then reproduces the
        baseline exactly -- the strongest no-systematic / no-bias check.
        Default False (full free-:math:`j_\phi` production mode).  The
        ``PIN_JPHI`` env var remains a back-compat override.
    psi_pad : float
        LCFS padding for profile queries.
    constrain_sawteeth : bool
        Reject equilibria with :math:`q_0 < 1`.
    recalculate_j_BS : bool
        Recompute bootstrap current each iteration.
    isolate_edge_jBS : bool
        Separate the edge bootstrap-current spike from the core
        contribution inside ``solve_with_bootstrap``.
    jBS_scale_range : list of two floats, or None
        Bounds ``[lo, hi]`` for a uniformly distributed multiplicative
        scale factor applied to :math:`j_{\rm BS}`.  For example,
        ``[0.8, 1.2]`` draws from :math:`\mathcal{U}(0.8, 1.2)`.
        When ``None``, no additional scaling is applied
        (``scale_jBS = 1.0`` for every sample).
    swb_iterations : int
        H-mode self-consistency iterations inside ``solve_with_bootstrap``
        (its ``iterations`` argument); 2 trades a little accuracy for speed.
    diagnostic_plots : bool
        Show diagnostic matplotlib figures.
    scan_key : str, float, int, or None
        Optional scan-point label for nested HDF5 storage.
        ``None`` gives the flat layout.
    pfile_bytes : bytes or None
        Raw p-file content to store alongside each equilibrium.
    Zeff_profile : array-like or None
        1-D effective charge profile to store in HDF5.
    coil_drift : float or None
        Symmetric ``+/-coil_drift * |I_baseline|`` hard bounds installed
        on every coil from the reconstructed (baseline) currents, via
        ``mygs.set_coil_bounds``.  Default 0.01 (one percent of the
        reconstructed value -- DIII-D coil-current measurement
        uncertainty).  Pass ``None`` to disable bounds entirely.

        Bounds remain installed for every perturbed draw.  When a
        perturbation cannot be solved within the bounds, TokaMaker
        raises and the draw is skipped; ``mygs`` is reset to the
        baseline ``(psi, coils)`` snapshot before the next draw.
    coil_drift_floor_A : float
        Absolute floor in amperes applied when
        ``coil_drift * |I_baseline|`` falls below this value (e.g. for
        a coil that happens to have a tiny baseline current).  Default
        50.0 A.
    vsc_coils : tuple of str
        Names of the coils that participate in the VSC pair (those
        with non-zero ``coil_vcont`` after ``mygs.set_coil_vsc``).
        Default ``('F9A', 'F9B')``.  These coils get split-budget
        bounds so the *total* current (bare + VSC contribution)
        stays within ``coil_drift``.
    vsc_budget_frac : float
        Fraction of ``coil_drift`` allocated to the ``#VSC`` channel
        for the VSC-pair coils; the remainder goes to the bare-coil
        bound.  Default 0.5 (equal split).
        - 0.0 → all budget on bare; ``#VSC`` pinned to 0 (no vertical
          control slack; may break solver convergence).
        - 1.0 → all budget on ``#VSC``; bare F-coils pinned at
          baseline (only the VSC channel can drift).
        Worst-case total drift on a VSC-pair coil with this split
        is exactly ``coil_drift × |I_baseline|`` for the smaller-
        magnitude coil of the pair, slightly less for the larger.
    coil_drift_hard_factor : float or None
        Optional hard inequality bound at ``±coil_drift_hard_factor *
        coil_drift * |I_baseline|`` per coil and on ``#VSC`` (sized by
        the smaller of the VSC pair).  Default ``None`` -- no hard
        bound is installed; the soft regularization toward baseline
        is the only constraint, matching PR1's ``lock_coils=True``
        behavior.  Set to a number (e.g. 20 or 50) to add a safety
        ceiling that catches catastrophic drift; values < ~10 tend to
        clip Picard-iteration transients and break solver convergence
        on perturbed draws.  Drift relative to ``coil_drift`` can be
        post-checked from the stored ``coil_currents`` per draw.
    soft_reg_weight : float
        Weight of the soft regularization toward baseline coil
        currents (default 1e4, calibrated for ~1% drift on DIII-D).
        Higher → tighter pull, lower → looser.
    vsc_soft_reg_weight : float
        Soft-reg weight for the ``#VSC`` channel (default 1.0).  Kept
        much lower than ``soft_reg_weight`` so the VSC has freedom to
        do vertical-mode control work without being heavily penalized.

    Returns
    -------
    list[dict]
        Diagnostics from each equilibrium.
    """
    # ---- the reproducibility contract: ONE Generator per run ----------
    # `seed` is consumed exactly here and nowhere else.  The resulting
    # Generator is threaded explicitly into every draw site -- the per-draw
    # scale_jBS and l_i-target samples below, and (via the `rng=` argument)
    # all nine GPR draw sites inside perturb_kinetic_equilibrium.  Nothing in
    # the sampling path reads global RNG state, so two runs with the same
    # seed produce bitwise-identical archives and seed=None keeps the
    # OS-entropy behaviour.
    #
    # Before this was threaded, `seed` only reached np.random's legacy global
    # state, which governed the two draws below but NOT the GPR: every draw
    # site called np.random.default_rng() with fresh OS entropy, so seeded
    # ensembles were not regenerable.
    rng = make_rng(seed)
    # The legacy global RNG is still seeded so that any third-party code in
    # the solve path that samples from np.random stays deterministic too.
    # bouquet's own draws no longer read it.
    if seed is not None:
        np.random.seed(int(seed))

    all_diagnostics = []

    # self-consistent pressure for baseline <P>
    # When kinetic profiles are on a different grid, interpolate
    # onto the equilibrium grid for pressure/GS calculations.
    if psi_N_kinetic is not None:
        # PCHIP regrid (shared helper) -- must match _kin_to_eq in the draws
        _kin2eq = lambda arr: pchip_interp(psi_N_kinetic, arr, psi_N)
        pressure = EC * (_kin2eq(ne) * _kin2eq(te) + _kin2eq(ni) * _kin2eq(ti))
    else:
        pressure = EC * (ne * te + ni * ti)

    # Fixed fast-ion pressure on the equilibrium grid. The baseline jphi-linterp
    # solve below (the per-draw boundary/l_i/coil reference) must include p_fast
    # so it is consistent with the perturbed draws (which add p_fast in
    # perturb_kinetic_equilibrium); otherwise p_fast's whole equilibrium response
    # appears as spurious, systematic coil/boundary drift in every draw. The
    # thermal-only `pressure` is kept for the perturbed-vs-baseline pressure match.
    if p_fast is not None:
        _p_fast_eq = (_kin2eq(np.asarray(p_fast, dtype=float))
                      if psi_N_kinetic is not None
                      else np.asarray(p_fast, dtype=float))
    else:
        _p_fast_eq = np.zeros_like(psi_N)
    # Impurity (carbon) + pressure-diff anchor on the equilibrium grid, matching
    # perturb_kinetic_equilibrium so the baseline reference solve equals every
    # draw at sigma=0 and anchors to the dd equilibrium.pressure. (Single-ion
    # `pressure` above is kept thermal-only for the perturbed-vs-baseline match.)
    if Z_imp:
        from .physics import impurity_pressure
        if psi_N_kinetic is not None:
            _p_imp_eq = impurity_pressure(_kin2eq(ne), _kin2eq(ni),
                                          _kin2eq(ti), Z_imp)
        else:
            _p_imp_eq = impurity_pressure(ne, ni, ti, Z_imp)
    else:
        _p_imp_eq = np.zeros_like(psi_N)
    _p_diff_eq = (np.asarray(p_diff, dtype=float) if p_diff is not None
                  else np.zeros_like(psi_N))
    pressure_solve = pressure + _p_imp_eq + _p_fast_eq + _p_diff_eq

    npsi = len(psi_N)

    # --- Auto-override constrain_sawteeth for sawtoothing baselines ---
    # If the baseline equilibrium already has q_0 < 1, constraining
    # perturbed equilibria to q_0 >= 1 is incompatible and will cause
    # every candidate to be rejected.  Detect this and override.
    # Re-solve with the baseline profiles first so the check reflects
    # the reconstruction state (not a corrective-iteration state that
    # may have altered q_0).
    # ----------------------------------------------------------------
    # Forward-mode baseline + coil_drift bounds.
    # ----------------------------------------------------------------
    # Mirrors PR1's lock_coils flow: do a forward solve at recon's
    # profiles + abs(eqdsk.Ip), THEN capture (psi, coils) from the
    # post-solve state.  This is the *forward-mode baseline* -- the
    # state forward-mode perturbations naturally land at in the sigma=0
    # limit -- which differs slightly from the inverse-mode recon's
    # coils because recon used its own Ip-secant correction.  Centering
    # coil_drift bounds around this forward-mode baseline (not around
    # recon's coils) is what makes ±1% bounds physically attainable for
    # forward-mode perturbations.
    _baseline_psi = None
    _baseline_coils = None
    _recon_Ip = None

    # Sawtooth check.  We use mygs.get_q on the current (recon) state
    # rather than re-solving here -- a fresh forward solve at
    # abs(eqdsk.Ip) shifts the equilibrium slightly off the recon's
    # converged state and ends up confusing the soft-reg lock.  Recon
    # already has a populated q profile.
    if constrain_sawteeth:
        try:
            _, q_baseline_check, _, _, _, _ = mygs.get_q(
                npsi=len(psi_N), psi_pad=psi_pad
            )
            if q_baseline_check[0] < 1.0:
                print(
                    f"NOTE: Baseline equilibrium has q(0) = {q_baseline_check[0]:.4f} < 1.0 "
                    f"(sawtoothing plasma).\n"
                    f"      Overriding constrain_sawteeth = False so perturbed "
                    f"equilibria are not rejected."
                )
                constrain_sawteeth = False
        except Exception as _q_exc:
            print(f"WARNING: q-profile check failed ({_q_exc}); "
                  f"leaving constrain_sawteeth as is.")

    if coil_drift is not None:
        # Capture mygs's current (recon) coil currents -- these are the
        # soft-lock target.  Caller must have just returned from
        # reconstruct_equilibrium, so mygs is at the recon's converged
        # state on entry to generate_bouquet.
        _initial_coils_dict, _ = mygs.get_coil_currents()
        _initial_coils = {k: float(v) for k, v in _initial_coils_dict.items()}

        # Also capture the recon's *converged* Ip (post-Ip-secant inside
        # reconstruct_equilibrium).  This is what each perturbed-equilibrium
        # actual_Ip should be aligned to by the post-perturb Ip secant
        # (jphi-linterp doesn't enforce Ip exactly, so passing a target
        # alone isn't enough -- need to iterate).
        _recon_Ip = float(abs(mygs.get_globals()[0]))

        # ---- Unperturbed jphi-linterp baseline solve (jphi_baseline flag) ---
        # WHY: recon converges in INVERSE mode using its exact internal
        # FF'/P' profiles.  Every ensemble member, by contrast, is solved
        # in jphi-linterp mode (we specify <j_phi> and the solver back-
        # solves FF').  jphi-linterp's edge realization overshoots the
        # specified <j_phi> near the separatrix (psi_N~0.996: input 0.20 ->
        # realized 0.29 MA/m^2, ~5% of peak), which lands the equilibrium
        # ~1.6 mm / l_i(3) ~-0.9% off recon's inverse-mode LCFS.  This
        # offset is CONSTANT across draws (verified: sigma=0 draws are bit-
        # identical), so it is a representation bias, not a perturbation
        # response -- but referencing per-draw diagnostics to recon's
        # inverse LCFS bakes it into every draw as a spurious floor.
        #
        # Fix: solve recon's profiles ONCE through the same jphi-linterp
        # machinery the draws use, and reference all per-draw diagnostics
        # (boundary, l_i) to THIS baseline.  An unperturbed (sigma=0) draw
        # then lands ~0 mm / ~0% from baseline by construction, and
        # perturbations show their true incremental response.  Coils are
        # left free (inverse + isoflux) exactly as in a draw.
        _baseline_li3 = float(l_i_target)
        if jphi_baseline:
            _psi_range_b = mygs.psi_bounds[1] - mygs.psi_bounds[0]
            _pp_b = {"type": "linterp",
                     "y": pchip_derivative(psi_N, pressure_solve) / _psi_range_b,
                     "x": psi_N}
            _pp_b["y"][-1] = 0.0
            # Anchor the baseline reference equilibrium (this solve is re-saved as
            # baseline.eqdsk, the profile GPEC reads) to equilibrium.j_tor by
            # adding the fixed jphi_diff, matching every draw's total at sigma=0.
            _jphi_b = (input_j_phi + np.asarray(jphi_diff, dtype=float)
                       if jphi_diff is not None else input_j_phi.copy())
            _ffp_b = {"type": "jphi-linterp",
                      "y": _jphi_b, "x": psi_N}
            # ---- psi re-initialisation before the baseline solve (issue #24) --
            # This is the THIRD site of the converged-on-entry degeneracy, and
            # the same treatment as the sigma=0 state anchor got in #22.
            # `mygs` arrives here on the reconstruction's own converged state;
            # once the reconstruction solves the SAME pressure the draws do,
            # that state sits essentially ON this forward jphi-linterp solve's
            # fixed point.  The under-relaxed Picard iteration then has no
            # gradient to descend, parks in a small limit cycle just above
            # nl_tol, burns all `maxits` iterations and raises
            # 'Exceeded "maxits"' -- a hard failure reported for a state that is
            # physically converged.  Re-initialising psi from the LCFS shape
            # starts the iteration far enough away that it converges normally.
            #
            # The shape is taken from the state we are about to leave (the same
            # LCFS the recon landed on), which is the local equivalent of the
            # run.py hunk's g-file boundary; `safe_trace_surf` snapshots and
            # restores, so the trace itself perturbs nothing.  Failure to get a
            # usable contour is not fatal: skip the re-init and solve warm, i.e.
            # exactly the previous behaviour.
            #
            # STATE GUARD.  `init_psi` DISCARDS the state we arrive on (recon's
            # converged equilibrium) and installs a cold analytic psi.  The
            # handler below advertises "REVERTING to the recon (inverse)
            # reference", and before the re-init existed that was literally
            # true -- a failed solve left mygs untouched on recon's converged
            # state, which is why #24 classed the fallback as benign.  With an
            # unguarded init_psi it is no longer true: the message would be
            # printed while mygs sits on a cold, non-converged psi, which then
            # poisons the warm start of every subsequent draw.  Snapshot before
            # the re-init so the revert the message promises actually happens.
            _bl_snap = None
            _bl_can_snap = (hasattr(mygs, "copy_eq")
                            and hasattr(mygs, "replace_eq"))
            _init_lcfs_exc = None
            try:
                _init_lcfs = safe_trace_surf(mygs, 1.0 - psi_pad)
            except Exception as _exc:
                _init_lcfs = None
                _init_lcfs_exc = _exc
            if _init_lcfs is not None and len(np.asarray(_init_lcfs)) >= 4:
                _bR0, _bZ0, _ba, _bkap, _bdel = _shape_from_boundary(_init_lcfs)
                if _bl_can_snap:
                    _bl_snap = mygs.copy_eq()
                mygs.init_psi(_bR0, _bZ0, _ba, _bkap, _bdel)
                print(f"  [jphi-baseline] psi re-initialised from the landed "
                      f"LCFS (R0={_bR0:.4f} a={_ba:.4f} kappa={_bkap:.4f} "
                      f"delta={_bdel:.4f}) before the forward solve "
                      f"(converged-on-entry guard, issue #24)")
            else:
                # Say WHY the trace was unusable.  The fallback is deliberately
                # non-fatal, so this line is the only record a run keeps of it;
                # dropping the exception detail makes a trace failure and a
                # merely-too-short contour indistinguishable after the fact.
                if _init_lcfs_exc is not None:
                    _why = (f"safe_trace_surf raised "
                            f"{type(_init_lcfs_exc).__name__}: {_init_lcfs_exc}")
                elif _init_lcfs is None:
                    _why = "safe_trace_surf returned None"
                else:
                    _why = (f"traced contour has only "
                            f"{len(np.asarray(_init_lcfs))} points (need >= 4)")
                print(f"  [jphi-baseline] WARN: could not trace an LCFS to "
                      f"re-initialise psi from ({_why}); solving warm (the "
                      f"converged-on-entry stall of issue #24 is possible here)")
            mygs.set_targets(Ip=initial_Ip_target, pax=float(pressure_solve[0]))
            mygs.set_profiles(pp_prof=_pp_b, ffp_prof=_ffp_b)
            try:
                mygs.solve()
                _recon_Ip = float(abs(mygs.get_globals()[0]))
                _baseline_li3 = float(mygs.get_stats(
                    lcfs_pad=psi_pad, li_normalization='iter')['l_i'])
                _bl_lcfs = safe_trace_surf(mygs, 1.0 - psi_pad)
                if _bl_lcfs is not None and len(_bl_lcfs) >= 4:
                    # Override the trace reference so bnd-diag + plot_traces
                    # measure against the jphi-linterp baseline, not recon.
                    recon_lcfs_ref = np.asarray(_bl_lcfs)
                # Re-capture coils from the baseline solve (these become the
                # soft-reg target + drift reference, == the comment at the
                # soft-reg block which wants the jphi-linterp baseline coils,
                # not recon's inverse-mode coils).
                _initial_coils_dict, _ = mygs.get_coil_currents()
                _initial_coils = {k: float(v) for k, v in
                                  _initial_coils_dict.items()}
                print(f"  [jphi-baseline] unperturbed jphi-linterp baseline "
                      f"solved: l_i(3)={_baseline_li3:.5f} Ip={_recon_Ip:.0f}; "
                      f"per-draw boundary/l_i now reference THIS baseline")
            except (ValueError, RuntimeError) as _bl_exc:
                # Deliberate mask (issue #24, third site) -- now COUNTED, and
                # the revert spelled out.  Previously this printed one line and
                # moved on, so an archive could not tell you whether its
                # per-draw boundary/l_i diagnostics were referenced to the
                # jphi-linterp baseline the draws live in or to recon's
                # inverse-mode LCFS -- two references ~1.6 mm / ~0.9 % in l_i(3)
                # apart, which is the whole reason this baseline solve exists.
                # Make the revert real BEFORE announcing it (and before the
                # count, so the counted event is the fully-handled one): put
                # mygs back on the state init_psi discarded, or every
                # subsequent draw warm-starts from a cold, non-converged psi.
                if _bl_snap is not None:
                    mygs.replace_eq(source_eq=_bl_snap)
                    _bl_reverted = "solver state restored to recon's"
                elif _bl_can_snap:
                    # psi was never re-initialised (no usable LCFS trace), so
                    # the state was never discarded -- nothing to restore.
                    _bl_reverted = "solver state untouched (no psi re-init)"
                else:
                    _bl_reverted = ("solver state NOT restored: this object "
                                    "exposes no copy_eq/replace_eq pair")
                _count_masked_anchor_failure("jphi_baseline", _bl_exc)
                print(f"  [jphi-baseline] solve failed ({_bl_exc}); REVERTING "
                      f"to the recon (inverse) reference [{_bl_reverted}] -- "
                      f"this run's "
                      f"per-draw boundary and l_i diagnostics carry the "
                      f"~1.6 mm / ~0.9 % representation offset the "
                      f"jphi-linterp baseline exists to remove, and the "
                      f"soft-reg coil target stays recon's inverse-mode set "
                      f"(l_i(3) reference {_baseline_li3:.5f} = l_i_target, "
                      f"NOT a solved baseline)")

        # ---- Boundary-shift diagnostic ----
        # The reference LCFS for per-stage boundary-deviation reporting
        # must match the snapshot that plot_traces uses as its baseline,
        # otherwise the two diagnostics measure from different references
        # and disagree by the state shift between snapshots.
        #
        # Preferred path: the caller passes `recon_lcfs_ref` (an Nx2
        # numpy array captured via mygs.trace_surf at the SAME state
        # where baseline.eqdsk was saved) so bnd-diag and plot_traces
        # are method-consistent.  Falls back to a fresh trace_surf at
        # generate_bouquet entry if no reference was provided.
        # Toggle off entirely with env var BNDDIAG=0.
        _bnd_diag_on = os.environ.get('BNDDIAG', '1') != '0'
        _bnd_diag_tree = None
        _bnd_diag_npts = 0
        _bnd_diag_source = "(disabled)"
        if _bnd_diag_on:
            try:
                from scipy.spatial import cKDTree as _cKDTree_bd
                if recon_lcfs_ref is not None and len(recon_lcfs_ref) >= 4:
                    _bd_ref = np.asarray(recon_lcfs_ref)
                    _bnd_diag_source = "caller-supplied"
                else:
                    _bd_ref = safe_trace_surf(mygs, 1.0 - psi_pad)
                    if _bd_ref is None or len(_bd_ref) < 4:
                        _bd_ref = safe_trace_surf(mygs, 1.0 - 1e-4)
                    _bnd_diag_source = "trace_surf at entry (snapshot-protected)"
                if _bd_ref is not None and len(_bd_ref) >= 4:
                    _bnd_diag_tree = _cKDTree_bd(np.asarray(_bd_ref))
                    _bnd_diag_npts = len(_bd_ref)
                    print(f"  [bnd-diag] using recon LCFS reference: "
                          f"{_bnd_diag_npts} pts ({_bnd_diag_source})")
                    # Promote the fresh-trace_surf reference to the
                    # caller's recon_lcfs_ref slot if the caller didn't
                    # pass one explicitly.  This makes the in-loop bnd-
                    # diag reference (built here at generate_bouquet
                    # entry) AND the H5 _baseline/recon_lcfs_ref that
                    # plot_traces consumes identical.  Without this,
                    # plot_traces would fall back to the ~100-pt
                    # baseline.eqdsk boundary against per-draw 9992-pt
                    # perturbed_lcfs_ref, yielding a ~mm-level
                    # sampling-noise floor that masks bit-identical
                    # equilibria (observed 2026-05 PIN_JPHI σ=0 +
                    # SKIP_HOMOTOPY: in-loop bnd-diag 0.00 mm
                    # all draws, but plot_traces showed ~few-mm spread
                    # because the comparison sampling was mismatched).
                    if recon_lcfs_ref is None:
                        recon_lcfs_ref = np.asarray(_bd_ref)
            except Exception as _bd_init_exc:
                print(f"  [bnd-diag] startup capture failed "
                      f"({_bd_init_exc}); per-stage reporting disabled")
                _bnd_diag_tree = None

        def _report_bnd(stage):
            """Print boundary deviation from the recon LCFS captured at
            generate_bouquet entry.  No-op if diagnostic disabled or
            LCFS trace fails."""
            if _bnd_diag_tree is None:
                return
            try:
                _lcfs = safe_trace_surf(mygs, 1.0 - psi_pad)
                if _lcfs is None or len(_lcfs) < 4:
                    _lcfs = safe_trace_surf(mygs, 1.0 - 1e-4)
                if _lcfs is not None and len(_lcfs) >= 4:
                    _devs, _ = _bnd_diag_tree.query(np.asarray(_lcfs))
                    _rms_mm = float(np.sqrt(np.mean(_devs**2)) * 1e3)
                    _max_mm = float(np.max(_devs) * 1e3)
                    print(f"  [bnd-diag] {stage:24s} "
                          f"vs recon LCFS: rms={_rms_mm:6.2f} mm  "
                          f"max={_max_mm:6.2f} mm")
            except Exception as _bd_exc:
                print(f"  [bnd-diag] {stage} trace failed ({_bd_exc})")

        # Hybrid coil-drift control:
        #
        #   (i)  HARD outer bounds at +/- (coil_drift * coil_drift_hard_factor)
        #        on every coil and the #VSC channel.  Prevents catastrophic
        #        drift but is loose enough that Picard iteration has
        #        wiggle room and converges (a hard +/-coil_drift bound
        #        sits exactly on the QP optimum and triggers iteration
        #        ping-pong against the constraint, exceeding maxits).
        #   (ii) SOFT regularization with target=baseline_coils,
        #        weight=soft_reg_weight, that pulls coil currents back
        #        toward baseline.  Typical drift settles near
        #        +/- coil_drift * |I_baseline| with weight=1e4 on DIII-D.
        #        The #VSC soft reg uses a much lower weight so the
        #        vertical-stability channel has room to do its work.
        #
        # The forward-solve "baseline" (post-Ip-secant-corrected forward
        # equilibrium at recon profiles + abs(eqdsk.Ip)) is what we
        # measure drift relative to -- not the inverse-mode recon
        # coils, which differ slightly from forward-mode equilibrium.
        # Step 2: install soft regularization targeting the post-q-check
        # forward-mode coils.  Replaces whatever soft reg the user
        # installed before generate_bouquet (typically target=0 weight=1.0
        # from the recon setup, which is too loose for perturbed solves).
        _rt = []
        for _name in mygs.coil_sets:
            _target = float(_initial_coils.get(_name, 0.0))
            _rt.append(mygs.coil_reg_term(
                {_name: 1.0}, target=_target,
                weight=float(soft_reg_weight)))
        _rt.append(mygs.coil_reg_term(
            {'#VSC': 1.0}, target=0.0,
            weight=float(vsc_soft_reg_weight)))
        mygs.set_coil_reg(reg_terms=_rt)
        # Stash the strong reg so the SWB-hygiene block in
        # perturb_kinetic_equilibrium can swap to a weak recon-like reg for
        # the exploratory SWB solves (which inverse-solve the *perturbed*
        # equilibrium and oscillate under the strong reg) and restore this
        # strong reg afterward for the constrained downstream phase.
        mygs._strong_coil_reg = _rt

        # Step 3: capture baseline (psi, coils) for the perturbation
        # reset path.  No re-solve here -- mygs is already at recon's
        # converged state from the caller.
        _baseline_psi = mygs.get_psi(False).copy()
        _baseline_coils = dict(_initial_coils)

        # Hoist SKIP_HARD / SKIP_ISO / SKIP_HOMOTOPY env-var reads
        # to function scope so the per-draw loop's pre-solve coil-pin
        # block can reference them (it runs BEFORE the Step-3 homotopy
        # block where these were originally defined).  Reading at
        # function entry also gives a consistent view across all
        # draws -- if a user toggles an env var mid-run we use the
        # value captured here, not a per-draw race.
        _skip_hard = os.environ.get('SKIP_HARD', '0') == '1'
        _skip_iso  = os.environ.get('SKIP_ISO',  '0') == '1'
        _skip_homotopy = os.environ.get('SKIP_HOMOTOPY', '0') == '1'

        # Warm-start snapshot for per-draw state restoration.
        # Captured AT THE END of the first successful draw (not at
        # startup): that draw's fully-converged forward-mode state
        # (psi, coils, isoflux from iso-update) is a much better
        # Picard initial guess for subsequent draws than the recon
        # state itself (which forward-mode physics doesn't satisfy
        # exactly -- the SWB pedestal spike shifts the boundary
        # ~10 mm from recon).  Without the snapshot, each draw's
        # Picard starts from the previous draw's converged state
        # (non-deterministic warm-chain) -- with it, draws 2..N
        # all start from draw 1's snapshot (deterministic warm
        # start), so identical perturbation inputs produce
        # repeatable outputs.
        # ---- Warmstart capture: snapshot RECON state, not draw-1 ----
        # mygs is in the recon's converged state on entry to
        # generate_bouquet (the caller just ran reconstruct_equilibrium
        # and re-set the recon isoflux).  Snapshot psi+coils+isoflux
        # NOW so every per-draw iteration starts from the same clean
        # baseline -- no draw-to-draw state pollution, no drift
        # accumulation.  Prior versions captured the first successful
        # draw's converged state, which locked in any small offset
        # that draw landed at and propagated it forever (manifested
        # as bit-identical clustering of draws 2..N at the draw-1
        # offset, with VSC drift fighting the inherited mismatch).
        # PR #248: prefer the full equilibrium-object snapshot via
        # copy_eq() over the partial set_psi/set_coil_currents/
        # set_isoflux restore path.  copy_eq grabs the entire
        # gs_equil struct (psi, FFP/PP profiles, plasma_bounds,
        # coil_currents, isoflux_constraints, Itor_target, ffp_scale,
        # p_scale, psiscale, ...).  replace_eq is a Fortran-level
        # pointer swap that restores ALL of it atomically -- no
        # leakage of derived state (axis cache, FSA quantities,
        # cached <R>/<1/R>, etc.) that the manual restore path
        # silently inherits from the prior draw.
        # Falls back to the manual restore on legacy OFT builds.
        _warmstart_eq_snap = None
        try:
            if hasattr(mygs, 'copy_eq') and hasattr(mygs, 'replace_eq'):
                _warmstart_eq_snap = mygs.copy_eq()
            _warmstart_psi = mygs.get_psi(False).copy()
            _wcoils, _ = mygs.get_coil_currents()
            _warmstart_coils = {k: float(v) for k, v in _wcoils.items()}
            # Isoflux capture (used for legacy fallback path only):
            _wiso = None
            _tm_eq = getattr(mygs, '_tMaker_equil', None)
            if _tm_eq is not None:
                _wiso = getattr(_tm_eq, '_isoflux_constraints', None)
            if _wiso is None:
                _wiso = getattr(mygs, '_isoflux_targets', None)
            if _wiso is None:
                _wiso = getattr(mygs, '_isoflux', None)
            if _wiso is not None and len(_wiso) >= 4:
                _warmstart_iso_pts = np.asarray(_wiso).copy()
                _warmstart_iso_w = (np.ones(
                    len(_warmstart_iso_pts)) * 200.0)
            else:
                _warmstart_iso_pts = None
                _warmstart_iso_w = None
            _warmstart_captured = True
            _snap_tag = ("FULL eq object via copy_eq"
                         if _warmstart_eq_snap is not None
                         else "psi+coils+iso (legacy)")
            print(f"  [warmstart] snapshot of RECON state captured "
                  f"({_snap_tag}, "
                  f"{len(_warmstart_iso_pts) if _warmstart_iso_pts is not None else 0} "
                  f"isoflux pts) -- every draw will restore from this "
                  f"baseline, not propagate prior draws' state")

            # ---- Patch the H5 top-level recon "0" group's l_i attrs
            # to match the warmstart-converged state.
            #
            # Background: the typical notebook flow calls
            # store_equilibrium(header, count=0, ...) once BEFORE
            # generate_bouquet, writing l_i from mygs.get_stats at
            # the post-recon / pre-generate_bouquet moment to the
            # top-level "0" group.  But generate_bouquet's own
            # internal recon flow (q-baseline check, jphi_corr,
            # warmstart-setup re-solve) settles mygs into a slightly
            # different state -- l_i typically shifts by ~0.5-2%.
            # Per-draw equilibria are then measured against THIS new
            # state, not the notebook's pre-bouquet snapshot.
            #
            # plot_traces uses top-level "0"/attrs/l_i(1) as the
            # denominator for its "% l_i deviation" panel.  If that
            # value is from the pre-bouquet snapshot but every per-
            # draw l_i is from the post-recon-flow state, the plot
            # shows a fixed ~1% offset that does NOT reflect any
            # actual per-draw variation -- it reflects a state
            # mismatch between the saved baseline and the
            # measurement reference.
            #
            # Fix: at the warmstart-capture point (= the canonical
            # "recon converged" state from generate_bouquet's
            # perspective), overwrite the top-level "0" group's
            # l_i attrs with the values mygs reports right now.
            # Per-draw l_i values are computed at moments that are
            # state-equivalent to here (warmstart-restored before
            # each draw's own work), so the denominator and numerator
            # are now consistent.
            #
            # The store_equilibrium pre-call from the notebook is
            # what creates the top-level "0" group; if that wasn't
            # done (group missing), this patch is a no-op.  We use
            # an 'a' file open + soft-fail try block so a missing
            # H5, missing group, or read-only file doesn't break
            # the run -- this is a cosmetic plotting fix, not
            # physics.
            try:
                import h5py as _h5_patch
                import tempfile as _tmp_patch
                _gs_recon = mygs.get_stats(
                    li_normalization='std', lcfs_pad=psi_pad)
                _gs_recon_iter = mygs.get_stats(
                    li_normalization='iter', lcfs_pad=psi_pad)
                _li1_recon = float(_gs_recon['l_i'])
                _li3_recon = float(_gs_recon_iter['l_i'])
                # mygs.get_globals()[0] = gs_comp_globals Ip (physical
                # integral).  For Ip_target patching we instead want
                # the gs_itor_nl-based Ip that save_eqdsk writes,
                # since plot_traces compares per-draw eq.Ip (from
                # eqdsk bytes) against _baseline/Ip_target attr --
                # both must use the same integration to agree.  We
                # capture both: Ip_target attr gets gs_itor_nl via the
                # re-saved baseline.eqdsk (read back after save), and
                # we also record gs_comp_globals Ip in a separate attr
                # for diagnostic comparison.
                _Ip_recon_compglob = float(abs(mygs.get_globals()[0]))
                _h5_baseline_path = os.path.abspath(f"{header}.h5")
                if os.path.isfile(_h5_baseline_path):
                    with _h5_patch.File(_h5_baseline_path, 'a') as _hfp:
                        # --- Top-level "0" l_i patch (existing). ---
                        # Notebook convention: top-level "0" is the
                        # baseline-recon entry written before
                        # generate_bouquet ran.  Hierarchical scans
                        # also write per-scan baselines under
                        # scan/{key}/_baseline; those don't need this
                        # patch because they use _baseline/attrs/
                        # l_i_target as the reference, which is set
                        # at recon time and is stable.
                        if '0' in _hfp and hasattr(_hfp['0'], 'attrs'):
                            _old_li1 = float(_hfp['0'].attrs.get(
                                'l_i(1)', float('nan')))
                            _hfp['0'].attrs['l_i(1)'] = _li1_recon
                            _hfp['0'].attrs['l_i(3)'] = _li3_recon
                            if abs(_li1_recon - _old_li1) > 1e-4:
                                print(f"  [warmstart] patched H5 top-"
                                      f"level '0' l_i(1) "
                                      f"{_old_li1:.5f} -> "
                                      f"{_li1_recon:.5f} (state "
                                      f"shift between notebook's "
                                      f"pre-bouquet capture and "
                                      f"generate_bouquet's recon-"
                                      f"converged state; "
                                      f"plot_traces denominator now "
                                      f"matches per-draw numerators)")

            except Exception as _li_patch_exc:
                # Cosmetic plot-consistency fix; never block the run.
                print(f"  [warmstart] H5 top-level l_i patch skipped "
                      f"({_li_patch_exc})")

            # --- Re-save baseline.eqdsk + retarget Ip_target. ---
            # Background: cell 22 of the typical notebook saves
            # baseline.eqdsk from mygs's pre-bouquet state and passes
            # those bytes + eqdsk.Ip into generate_bouquet as
            # baseline_eqdsk_bytes / initial_Ip_target.
            # generate_bouquet's internal recon flow (Ip-match loop,
            # jphi_corr, warmstart-setup re-solve, OFT outer loop
            # convergence) then settles mygs at a slightly different
            # state -- typically ~0.3-0.5% shift in gs_itor_nl Ip (the
            # integral save_eqdsk uses).  Per-draw equilibria are
            # saved from the post-recon-flow state, so per-draw eq.Ip
            # and per-draw eqdsk.boundary_R/Z all carry the recon-
            # converged geometry, but the H5 baseline does not.
            # Result in plot_traces / plot_boundary_point_traces: a
            # fixed ~0.4% Ip offset and ~1 mm boundary shift between
            # every draw and the baseline, despite the underlying
            # mygs state being bit-identical draw-to-draw.  This is
            # a baseline-reference mismatch, not physics.
            #
            # Fix: at this warmstart-capture moment (= canonical
            # recon-converged state used by all per-draw work),
            # re-save baseline.eqdsk to disk via mygs.save_eqdsk()
            # and REASSIGN the local baseline_eqdsk_bytes and
            # initial_Ip_target that store_baseline_profiles (called
            # a few lines below) consumes.  That writes the recon-
            # converged eqdsk into H5 _baseline.baseline.eqdsk and
            # Ip_target.  The original input eqdsk on disk is
            # untouched; only the H5 baseline shifts to "the
            # canonical reference per-draw equilibria are measured
            # against," which makes plot_traces /
            # plot_boundary_point_traces show flat 0% / 0 mm at σ=0
            # + PIN_JPHI + SKIP_HOMOTOPY.
            #
            # IMPORTANT: this MUST run AFTER the l_i patch's H5 'a'
            # context exits (otherwise we'd hold the file open while
            # save_eqdsk also writes), and MUST run BEFORE the
            # store_baseline_profiles call below (which deletes and
            # recreates the _baseline group, obliterating any direct
            # H5 patch we'd try).  Reassigning the function-local
            # baseline_eqdsk_bytes / initial_Ip_target propagates
            # cleanly through the existing store_baseline_profiles
            # call without needing schema changes.
            #
            # Cost: one extra save_eqdsk call per scan_key (a few
            # hundred ms).  Soft-fails -- cosmetic plot fix, never
            # blocks the physics run.
            try:
                import tempfile as _tmp_patch2
                with _tmp_patch2.NamedTemporaryFile(
                        suffix='.geqdsk', delete=False) as _tf:
                    _tmp_eqdsk_path = _tf.name
                # nr/nz=257 matches the per-draw save_eqdsk call
                # inside the per-draw loop further down, so the
                # baseline and per-draw eqdsks have the same grid
                # resolution.
                mygs.save_eqdsk(
                    _tmp_eqdsk_path,
                    nr=257, nz=257,
                    truncate_eq=save_truncate_eq,
                    lcfs_pad=psi_pad)
                with open(_tmp_eqdsk_path, 'rb') as _ef:
                    _new_eqdsk_bytes = _ef.read()
                _new_eq_obj = read_eqdsk_from_bytes(
                    _new_eqdsk_bytes, read_geqdsk)
                _new_eq_Ip = float(abs(_new_eq_obj.Ip))
                _old_initial_Ip = float(initial_Ip_target)
                # Reassign function-local variables so the upcoming
                # store_baseline_profiles call writes the recon-
                # converged eqdsk + Ip into the H5 _baseline group.
                baseline_eqdsk_bytes = _new_eqdsk_bytes
                initial_Ip_target = _new_eq_Ip
                if abs(_new_eq_Ip - _old_initial_Ip) > 1.0:
                    _shift_pct = (
                        100.0 * (_new_eq_Ip - _old_initial_Ip)
                        / max(abs(_old_initial_Ip), 1.0))
                    print(f"  [warmstart] re-saved baseline.eqdsk at "
                          f"recon-converged state: Ip "
                          f"{_old_initial_Ip:.0f} -> {_new_eq_Ip:.0f} A "
                          f"({_shift_pct:+.3f}%, was the gs_itor_nl "
                          f"shift between notebook's pre-bouquet "
                          f"capture and generate_bouquet's recon-"
                          f"converged state; plot_traces Ip% and "
                          f"plot_boundary_point_traces now reference "
                          f"the same state per-draw equilibria "
                          f"settle at)")
                try:
                    os.unlink(_tmp_eqdsk_path)
                except Exception:
                    pass
            except Exception as _eq_patch_exc:
                print(f"  [warmstart] baseline.eqdsk re-save skipped "
                      f"({_eq_patch_exc}); H5 baseline will retain "
                      f"the pre-bouquet snapshot")
            if os.environ.get('PINJ_PROBE', '0') == '1':
                try:
                    _gs0 = mygs.get_stats(li_normalization='iter',
                                          lcfs_pad=psi_pad)
                    _op0 = mygs.o_point
                    print(f"    [probe RECON reference (warmstart src)  ] "
                          f"l_i={float(_gs0['l_i']):.5f}  "
                          f"Ip={float(_gs0['Ip']):.0f}  "
                          f"axis=({float(_op0[0]):.5f},{float(_op0[1]):+.5f})")
                except Exception:
                    pass
        except Exception as _ws_init_exc:
            print(f"  [warmstart] recon-state capture failed "
                  f"({_ws_init_exc}); draws will inherit prior state")
            _warmstart_psi = None
            _warmstart_coils = None
            _warmstart_iso_pts = None
            _warmstart_iso_w = None
            _warmstart_captured = False

        # Step 5: optional hard outer bounds at
        # +/-(coil_drift * hard_factor) around the lock-mode coils.
        # When hard_factor is None (default), no hard bounds are
        # installed -- soft reg alone shapes the QP optimum.  Hard
        # bounds at any factor can break Picard iteration when the
        # iteration's transients exceed the bound (the QP clips and
        # the iteration ping-pongs).  If you want a safety ceiling,
        # set hard_factor large (e.g. 20-50) so it only catches
        # catastrophic drift; leave None for PR1-equivalent behavior.
        _vsc_in_set = tuple(c for c in vsc_coils if c in _baseline_coils)
        if coil_drift_hard_factor is not None:
            _hard = float(coil_drift_hard_factor) * coil_drift
            _bounds = {}
            for _name, _base in _baseline_coils.items():
                _delta = max(_hard * abs(_base), float(coil_drift_floor_A))
                _bounds[_name] = [_base - _delta, _base + _delta]
            if len(_vsc_in_set) > 0:
                _vsc_min_base = min(abs(_baseline_coils[c]) for c in _vsc_in_set)
                _vsc_delta = max(_hard * _vsc_min_base, float(coil_drift_floor_A))
                _bounds['#VSC'] = [-_vsc_delta, _vsc_delta]
            mygs.set_coil_bounds(_bounds)
            mygs._coil_drift_bounds = _bounds
        else:
            # Don't call set_coil_bounds at all when no hard bounds are
            # requested -- even set_coil_bounds(None) (which uses ±1e98)
            # may put the underlying QP into bounded-mode and subtly
            # change the iteration path.
            if hasattr(mygs, '_coil_drift_bounds'):
                delattr(mygs, '_coil_drift_bounds')

        _hard_msg = (f"hard +/-{float(coil_drift_hard_factor)*coil_drift*100:.1f}% "
                     f"(factor={coil_drift_hard_factor:g}x) + "
                     if coil_drift_hard_factor is not None else "soft-reg-only, ")
        print(
            f"[coil-bounds hybrid] target +/-{coil_drift*100:.2f}% drift, "
            f"{_hard_msg}"
            f"soft reg (weight={soft_reg_weight:.0e}, "
            f"vsc_weight={vsc_soft_reg_weight:.0e}); "
            f"psi+coils snapshot captured (recon baseline, "
            f"{len(_baseline_coils)} coils)."
        )
    # Pre-compute jBS scale factors for the whole batch (if requested).
    # Uses a uniform distribution within the specified range so that
    # extreme values (which can make l_i matching very difficult) are
    # strictly bounded.
    if jBS_scale_range is not None:
        lo, hi = jBS_scale_range
        jBS_scales = rng.uniform(lo, hi, size=n_equils)
    else:
        jBS_scales = np.ones(n_equils)

    # Store baseline profiles and uncertainties so the .h5 file is
    # self-contained (the plotting GUI only needs the file path).
    #
    # Recompute the baseline p-file's rotation profiles using the same
    # midplane method we use for perturbed p-files.  This ensures that
    # baseline and perturbed omghb / Er are computed consistently and
    # can be compared directly in plots.
    stored_pfile_bytes = baseline_pfile_bytes
    from .schema import is_binary_profile_source as _is_bin_src
    if (baseline_pfile_bytes is not None and baseline_eqdsk_bytes is not None
            and not _is_bin_src(baseline_pfile_bytes)):
        # (binary IDA .cdf sources carry no p-file rotation blocks to rebuild;
        # attempting to parse them as text only produced a decode warning)
        try:
            from .io.pfile import PFile as _PFile
            from .io import GEQDSKEquilibrium as _GEQDSK
            from scipy.interpolate import interp1d

            pf_bl = _PFile.from_bytes(baseline_pfile_bytes)
            eq_bl = _GEQDSK.from_bytes(baseline_eqdsk_bytes)
            psi_pf = pf_bl.psinorm_for("ne")
            dpsi_bl = eq_bl.psi_boundary - eq_bl.psi_axis
            psi_Wb_bl = psi_pf * dpsi_bl + eq_bl.psi_axis

            pf_bl.compute_diamagnetic_rotations(psi_Wb_bl)

            mid_bl = eq_bl.midplane
            psi_eq = eq_bl.psi_N
            R_bl = interp1d(
                psi_eq, mid_bl["R"],
                fill_value="extrapolate")(psi_pf)
            Bp_bl = interp1d(
                psi_eq, mid_bl["Bp"],
                fill_value="extrapolate")(psi_pf)
            Bt_bl = interp1d(
                psi_eq, mid_bl["Bt"],
                fill_value="extrapolate")(psi_pf)

            pf_bl.compute_rotation_decomposition(
                R=R_bl, Bp=Bp_bl, Bt=Bt_bl, psi=psi_Wb_bl)
            stored_pfile_bytes = pf_bl.to_bytes()
        except Exception as exc:
            import traceback
            print(f"  WARNING: could not recompute baseline rotations: {exc}")
            traceback.print_exc()

    # Capture recon's coil currents at this point (mygs is in recon's
    # converged state on entry to generate_bouquet).  Saved as the
    # absolute reference for per-draw coil-drift analysis.
    try:
        _bl_coil_dict, _ = mygs.get_coil_currents()
        _bl_coil_dict = {k: float(v) for k, v in _bl_coil_dict.items()}
    except Exception:
        _bl_coil_dict = None

    # TokaMaker's built-in X-point finder, captured at the recon-converged
    # state (the same state baseline.eqdsk was saved from).  Stored so
    # plot_boundary_point_traces tracks true B_p=0 nulls instead of a
    # geometric corner guess on the saved boundary polyline.
    try:
        _bl_xpts, _bl_div = mygs.get_xpoints()
        _bl_xpts = (np.asarray(_bl_xpts, dtype=float)
                    if _bl_xpts is not None else None)
    except Exception as _xexc:
        print(f"  WARN: baseline get_xpoints() failed ({_xexc}); "
              f"plot_boundary_point_traces will fall back to the "
              f"axis-line intersection for top/bottom")
        _bl_xpts, _bl_div = None, None

    # ---- What to archive as the baseline currents -----------------------
    # Default (geqdsk path / gate off): the anchored target profile
    # (input_j_phi + jphi_diff = equilibrium.j_tor) with the model split.
    # store_achieved_jphi (IMAS path): the ACHIEVED FSA j_phi of the converged
    # solve (mygs is in the baseline-converged state here), so the 1-D j_phi
    # dataset matches the stored baseline eqdsk instead of sitting a few %
    # off it; j_inductive is recomputed as the residual against the physical
    # bootstrap so closure (j_phi = j_ind + j_BS + fixed) stays exact, with
    # any sub-zero sliver floored and absorbed into j_BS (same convention as
    # the per-draw split).
    _bl_jphi_store = (input_j_phi + np.asarray(jphi_diff, dtype=float)
                      if jphi_diff is not None else input_j_phi)
    _bl_jBS_store = baseline_j_BS
    _bl_jind_store = input_jinductive
    if store_achieved_jphi:
        try:
            _bl_jphi_store = _achieved_jphi_fsa(
                mygs, psi_N, psi_pad, sign_ref=_bl_jphi_store)
            if baseline_j_BS is not None:
                _fx = np.zeros_like(np.asarray(psi_N, dtype=float))
                if j_NBI is not None:
                    _fx = _fx + np.asarray(j_NBI, dtype=float)
                if j_RF is not None:
                    _fx = _fx + np.asarray(j_RF, dtype=float)
                _bl_jind_store = (_bl_jphi_store
                                  - np.asarray(baseline_j_BS, dtype=float) - _fx)
                if np.any(_bl_jind_store < 0.0):
                    _bl_jind_store = np.maximum(_bl_jind_store, 0.0)
                    _bl_jBS_store = _bl_jphi_store - _bl_jind_store - _fx
            print("  [archive] baseline j_phi = achieved FSA current "
                  "(matches baseline eqdsk)")
        except Exception as _aexc:
            print(f"  WARN: achieved-jphi baseline archival failed ({_aexc}); "
                  f"storing the anchored target instead")

    store_baseline_profiles(
        header, psi_N,
        ne, te, ni, ti,
        # Store the ACTUAL solved quantities, not the un-anchored inputs, so the
        # archived _baseline diagnostics match what was solved (== dd equilibrium):
        #  - pressure_solve = thermal + impurity + fast + p_diff anchor (not thermal-D)
        #  - with store_achieved_jphi, j_phi is the ACHIEVED FSA current of the
        #    converged solve (matches the stored eqdsk); otherwise the anchored
        #    target input_j_phi + jphi_diff (= equilibrium.j_tor).
        pressure_solve,
        _bl_jphi_store,
        sigma_ne, sigma_te, sigma_ni, sigma_ti, sigma_jphi,
        initial_Ip_target, l_i_target,
        scan_key=scan_key,
        # thermal-only part, so plots can separate it from the impurity+fast
        # the GS solve added (pressure_solve - pressure).
        pressure_thermal=pressure,
        eqdsk_bytes=baseline_eqdsk_bytes,
        pfile_bytes=stored_pfile_bytes,
        psi_N_kinetic=psi_N_kinetic,
        coil_currents=_bl_coil_dict,
        recon_lcfs_ref=recon_lcfs_ref,
        x_points=_bl_xpts,
        diverted=_bl_div,
        aux_baselines=aux_baselines,
        aux_sigmas=aux_sigmas,
        j_BS=_bl_jBS_store,
        j_inductive=_bl_jind_store,
        source_kind=source_kind,
    )

    # ---- Purge stale draws for THIS scan value -------------------------
    # The database is opened append-mode (multi-scan runs accumulate scan
    # values across calls), so draws from a previous run of the SAME header
    # + scan value would survive wherever this run has failures -- a mixed
    # archive whose ghost draws carry a different baseline/target. Delete
    # the numeric draw groups under this scan value up front; _baseline and
    # other scan values are untouched.
    import h5py as _h5_purge
    from .utils import _scan_key as _svk
    with _h5_purge.File(f"{header}.h5", "a") as _hf_purge:
        _bk = _svk(scan_key)
        _parent = (_hf_purge.get(f"scan/{_bk}") if _bk is not None
                   else _hf_purge)
        if _parent is not None:
            _stale = [k for k in _parent.keys() if k.isdigit()]
            for _k in _stale:
                del _parent[_k]
            if _stale:
                print(f"  purged {len(_stale)} stale draw group(s) from a "
                      f"previous run of this header/scan")

    # ---- DIFF_BS cache: snapshot mygs + cache SWB(recon kinetics) ----
    # If DIFF_BS=1 is set, capture the recon-state equilibrium and run
    # SWB once on the unperturbed recon kinetics.  Both are passed into
    # perturb_kinetic_equilibrium for each draw, which restores mygs to
    # the snapshot, calls SWB on perturbed kinetics, subtracts the
    # cached isolated_j_BS, and applies the delta on top of input_j_phi.
    # At sigma->0 delta -> 0 and the output exactly equals PIN_JPHI.
    # The same cache also provides the sigma=0 RAW reference spike for the
    # delta composition mode (jbs_delta_mode): spike = baseline_j_BS +
    # (SWB_raw(perturbed) - SWB_raw(sigma=0)), computed in this identical
    # pre-draw anchor context so common-mode evaluation artifacts (the
    # collapsed innermost-surface point) cancel exactly.
    _diff_bs_env = os.environ.get('DIFF_BS', '0') == '1'
    _diff_recon_eq_snap = None
    _diff_spike_recon = None
    _delta_spike0_raw = None
    if (_diff_bs_env or jbs_delta_mode) and recalculate_j_BS:
        print("\n" + "=" * 60)
        print("  [%s] Pre-loop setup: caching SWB(recon kinetics)"
              % ("DIFF_BS" if _diff_bs_env else "jBS-delta"))
        print("=" * 60)
        try:
            from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
            from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap as _swb
            from scipy.interpolate import interp1d as _interp1d
            _swb_seed_cache = create_power_flux_fun(npsi, 1.5, 1.5)['y']
            # Interpolate recon kinetic profiles to equilibrium grid if
            # caller is using a dual-grid (mirrors `_kin_to_eq` inside
            # perturb_kinetic_equilibrium).  SWB expects the kinetic
            # arrays on the same grid as `_swb_seed_cache` (npsi=len(psi_N)).
            if psi_N_kinetic is not None:
                def _k2e(a):
                    # PCHIP regrid -- must match _kin_to_eq in the draws
                    return pchip_interp(psi_N_kinetic, a, psi_N)
                ne_cache = _k2e(ne); te_cache = _k2e(te)
                ni_cache = _k2e(ni); ti_cache = _k2e(ti)
            else:
                ne_cache, te_cache, ni_cache, ti_cache = ne, te, ni, ti
            # Stash bounds (SWB expects unbounded coils, see SWB hygiene block)
            _cache_stash = getattr(mygs, '_coil_drift_bounds', None)
            if _cache_stash is not None:
                mygs.set_coil_bounds(None)
            # State-anchor solve before SWB.  Mirrors per-draw flow at
            # line ~870 -- without this, SWB sometimes inherits a stale
            # mygs state and hits maxits.  Uses pressure_solve (thermal +
            # p_fast + impurity + p_diff, the same assembly the baseline
            # jphi-linterp solve uses) + input_j_phi so this is recon's
            # natural equilibrium re-solved -- post-#22 the reconstruction
            # solves at the FULL pressure, so anchoring this cache at the
            # thermal-only `pressure` re-solved a different, lower-pressure
            # equilibrium (issue #35 Defect 1, fifth site).  This is a
            # BASELINE cache, so the baseline assembly is the consistent
            # choice here (the per-draw anchor tracks pres_tmp instead).
            try:
                _cache_pp = {"type": "linterp",
                             "y": pchip_derivative(psi_N, pressure_solve) /
                                  (mygs.psi_bounds[1] - mygs.psi_bounds[0]),
                             "x": psi_N}
                _cache_pp["y"][-1] = 0.0
                _cache_ffp = {"type": "jphi-linterp",
                              "y": input_j_phi.copy(), "x": psi_N}
                mygs.set_targets(Ip=initial_Ip_target,
                                 pax=float(pressure_solve[0]))
                mygs.set_profiles(pp_prof=_cache_pp, ffp_prof=_cache_ffp)
                mygs.solve()
                print(f"  [DIFF_BS] state-anchor solve OK; entering SWB")
            except (ValueError, RuntimeError) as _anch_exc:
                print(f"  [DIFF_BS] state-anchor solve failed "
                      f"({_anch_exc}); SWB may inherit stale state")
            try:
                # The sigma=0 reference MUST carry the CENTER of the per-draw
                # scale distribution: OFT applies scale_jBS INSIDE SWB, so a
                # 1.0 reference makes the delta
                #   bs*SWB0 + bs*SWB_pert - 1.0*SWB0
                # i.e. every draw loses (1-bs)/bs of the pedestal bootstrap.
                # Exactly zero at bs=1 (why the original validation passed)
                # and ~10% of Ip at bs=0.70.  Full diagnosis in issue #44.
                # See sigma0_reference_scale for the center definition.  NOTE
                # the per-draw `scale_jBS` local does NOT exist yet here (it
                # is bound inside the draw loop below); referencing it raises
                # UnboundLocalError, which the enclosing except used to
                # swallow into a silent cache-disable fallback.
                _scale_ref = sigma0_reference_scale(jBS_scale_range)
                _cache_results = _swb(
                    mygs, ne_cache, te_cache, ni_cache, ti_cache, Zeff,
                    initial_Ip_target, _swb_seed_cache,
                    scale_jBS=_scale_ref,
                    isolate_edge_jBS=isolate_edge_jBS,
                    diagnostic_plots=False, verbose=False,
                )
                # Toroidal conversion on the cache-time SWB equilibrium, so
                # the per-draw delta (also converted) is convention-consistent.
                # RAW profile for delta mode (artifacts cancel in the delta);
                # smoothed version for DIFF_BS (whose per-draw spikes are also
                # smoothed).
                _delta_spike0_raw = _swb_jbs_to_toroidal(
                    mygs, _cache_results["isolated_j_BS"], psi_pad)
                _diff_spike_recon = smooth_jbs_transition(_delta_spike0_raw)
                # Snapshot AFTER the SWB call -- this is the state from
                # which we'll re-launch SWB on perturbed kinetics each
                # draw, so it must match what the cached SWB saw.
                _diff_recon_eq_snap = mygs.copy_eq()
                print(f"  [DIFF_BS] cache populated: "
                      f"isolated_j_BS rms="
                      f"{float(np.sqrt(np.mean(_diff_spike_recon**2))):.3e} A/m², "
                      f"len={len(_diff_spike_recon)}; "
                      f"snapshot held in TokaMaker_equilibrium")
            finally:
                if _cache_stash is not None:
                    mygs.set_coil_bounds(_cache_stash)
        except Exception as _cache_exc:
            print(f"  [DIFF_BS] WARNING: cache setup failed ({_cache_exc}); "
                  f"falling back to standard SWB per draw")
            import traceback as _tb
            _tb.print_exc()
            _diff_recon_eq_snap = None
            _diff_spike_recon = None
            _delta_spike0_raw = None
    if jbs_delta_mode and _delta_spike0_raw is None:
        print("  [jBS-delta] WARNING: sigma=0 reference unavailable; draws "
              "fall back to the shared-smoothing spike treatment")
    if jbs_delta_mode and baseline_j_BS is None:
        print("  [jBS-delta] WARNING: baseline_j_BS not provided; draws "
              "fall back to the shared-smoothing spike treatment")
        _delta_spike0_raw = None

    # Tracks the cylindrical-proxy / real-l_i ratio observed at the
    # end of the most recent successful draw.  Passed into the next
    # draw's perturb_kinetic_equilibrium as proxy_bias_warmstart so its
    # initial proxy_target lands at l_i_target * bias -> 1 outer iter
    # convergence instead of 2 (saves ~30s/draw).  Set to None until
    # the first successful draw establishes a baseline.
    _proxy_bias_warmstart = None

    # One-time notice for the Zeff-primary mode (the per-draw mechanics live
    # in perturb_kinetic_equilibrium; see physics.main_ion_density_from_zeff).
    # Gate and Z_imp source both mirror that function's _zeff_active branch:
    # the mode needs a zeff sigma AND a zeff baseline, and uses the baseline's
    # declared Z_imp when it has one, else inverts (ne, ni, Zeff).
    if (ni_from_zeff and aux_sigmas and 'zeff' in aux_sigmas
            and (aux_baselines or {}).get('zeff') is not None):
        from .physics import effective_impurity_charge
        _zimp_note = float(Z_imp) if Z_imp else effective_impurity_charge(
            ne, ni, np.asarray(aux_baselines['zeff'], dtype=float))
        if _zimp_note is not None:
            print(f"NOTE: zeff channel active -> ni is DERIVED per draw from "
                  f"the drawn (ne, Zeff) via single-impurity quasineutrality "
                  f"at Z_imp = {_zimp_note:.2f}; sigma_ni is not used.")

    t_batch_start = time.perf_counter()
    elapsed_times = []

    try:
        # Plain text bar on stderr: tqdm.auto would emit a Jupyter widget,
        # which breaks under capture_native_output's fd redirect.
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    pbar = (
        _tqdm(range(n_equils), desc="Bouquet", unit="eq")
        if _tqdm is not None
        else None
    )
    eq_iter = pbar if pbar is not None else range(n_equils)

    for count in eq_iter:
        if progress_callback is not None:
            # one tick per draw attempt -- fed to a parent aggregate bar in the
            # process-parallel path (worker tqdm/stderr is suppressed there).
            try:
                progress_callback(count)
            except Exception:
                pass
        scale_jBS = float(jBS_scales[count])
        # ---- Per-draw l_i_target sampling ----
        # If l_i_uncertainty > 0, draw a perturbed target from
        # N(l_i_target, l_i_uncertainty * l_i_target) so the bouquet
        # spans a physical l_i distribution (e.g. 5% to mimic DIII-D's
        # measurement uncertainty).  Each draw then converges TIGHTLY
        # to its own sampled target (l_i_tolerance governs *that*
        # convergence, not the spread).  l_i_uncertainty=0 (default)
        # pins every draw to the recon's l_i exactly, as before.
        if l_i_uncertainty > 0.0:
            l_i_target_draw = float(rng.normal(
                l_i_target, l_i_uncertainty * l_i_target))
            # Guard against pathological negative samples (4σ events
            # for any reasonable uncertainty); clamp to a small fraction
            # of recon to keep find_optimal_scale's Brent search valid.
            if l_i_target_draw < 0.1 * l_i_target:
                l_i_target_draw = 0.1 * l_i_target
        else:
            l_i_target_draw = float(l_i_target)
        # Per-draw homotopy/spec defaults (overwritten by Step 3 below).
        _final_drifts = None
        _final_pass_idx = -1
        _final_drift_F_lim = (coil_drift if coil_drift is not None else 0.0)
        _final_drift_VSC_lim = (coil_drift if coil_drift is not None else 0.0)
        _max_f_drift = float('nan')
        _max_vsc_drift = float('nan')
        _in_spec = False
        eta_str = ""
        if elapsed_times:
            avg_s = np.mean(elapsed_times)
            remaining = avg_s * (n_equils - count)
            eta_min = remaining / 60.0
            eta_str = f"  ETA: {eta_min:.1f} min"
        print(f"\n{'='*60}")
        print(f"  Equilibrium {count+1}/{n_equils}  "
              f"(scale_jBS={scale_jBS:.4f}){eta_str}")
        if l_i_uncertainty > 0.0:
            _dev_pct = 100.0 * (l_i_target_draw - l_i_target) / l_i_target
            print(f"  l_i_target sampled: {l_i_target_draw:.4f} "
                  f"({_dev_pct:+.2f}% vs recon, σ={100*l_i_uncertainty:.1f}%)")
        print(f"{'='*60}")
        t_start = time.perf_counter()

        # ---- Warm-start restore ----
        # On draw 0, this is a no-op (snapshot not yet captured).
        # On draws 1..N-1, restore mygs's psi/coils/isoflux to the
        # snapshot taken at end of draw 0, so each subsequent draw's
        # Picard begins from the same warm state.  Ip target is
        # deliberately NOT restored -- it's overwritten by
        # perturb_kinetic_equilibrium's internal set_targets call
        # anyway, and resetting it independently has been observed
        # to perturb that Picard's behaviour.
        if _warmstart_captured:
            try:
                if _warmstart_eq_snap is not None:
                    # Full atomic equilibrium restore via PR #248
                    # replace_eq -- Fortran-level pointer swap that
                    # restores the entire gs_equil struct (psi, FFP/PP
                    # profiles, plasma_bounds, coil_currents, isoflux,
                    # Itor_target, scale factors, axis cache, FSA
                    # quantities, ...).  No leakage of derived state
                    # from prior draws.
                    mygs.replace_eq(source_eq=_warmstart_eq_snap)
                else:
                    # Legacy partial restore (PR #248 unavailable)
                    mygs.set_coil_currents(_warmstart_coils)
                    mygs.set_psi(_warmstart_psi, update_bounds=True)
                    if _warmstart_iso_pts is not None:
                        mygs.set_isoflux(_warmstart_iso_pts,
                                         weights=_warmstart_iso_w)
            except Exception as _ws_rs_exc:
                print(f"  [warmstart] restore failed "
                      f"({_ws_rs_exc}); draw inherits prior state")
        # PIN_JPHI/DIFF_BS probe just after warmstart restore -- shows
        # the state perturb_kinetic_equilibrium will see on entry.
        if os.environ.get('PINJ_PROBE', '0') == '1':
            try:
                _gs = mygs.get_stats(li_normalization='iter',
                                     lcfs_pad=psi_pad)
                _op = mygs.o_point
                print(f"    [probe just after warmstart restore     ] "
                      f"l_i={float(_gs['l_i']):.5f}  "
                      f"Ip={float(_gs['Ip']):.0f}  "
                      f"axis=({float(_op[0]):.5f},{float(_op[1]):+.5f})")
            except Exception as _pp_exc:
                print(f"    [probe warmstart restore] failed ({_pp_exc})")

        # ---- SKIP_HOMOTOPY: install Tier-2 phantom-VSC coil pin ----
        # SKIP_HOMOTOPY's original semantics were "skip the hard-bound
        # homotopy tightening passes" -- written when the per-draw work
        # didn't include a perturbed-pressure solve (PIN_JPHI's recon-
        # anchor bug, fixed 2026-05).  With that bug fixed, the recon-
        # anchor solve inside perturb_kinetic_equilibrium DOES run, and
        # the QP behind mygs.solve freely optimizes coil currents
        # subject to whatever bounds are in place at solve time.  With
        # SKIP_HOMOTOPY skipping the bound install, coils are
        # unconstrained: the QP can drift F9A (VSC) by ~5% per draw to
        # match isoflux constraints against perturbed pressure,
        # producing ~cm-scale boundary motion that doesn't match the
        # intended "phantom VSC" (coils locked, boundary responds
        # naturally) semantics.
        #
        # Fix: install tight hard bounds at recon ± COIL_DRIFT_FLOOR_A
        # (a few tens of amps, essentially zero per-coil drift) so the
        # recon-anchor + Ip-align solves can't move any coil from its
        # recon value.  This is what Tier-2 phantom-VSC means: the QP
        # is told "you cannot use coil currents to compensate the
        # perturbation; whatever boundary the equilibrium settles at
        # given recon coils + perturbed pressure + pinned j_phi is the
        # answer."  Boundary motion now reflects the actual physical
        # response, not coil-optimization artifact.
        #
        # IMPORTANT: this MUST happen AFTER warmstart restore (which
        # may itself reset bounds via replace_eq) and BEFORE
        # perturb_kinetic_equilibrium (which contains the per-draw
        # solve).  The downstream SKIP_HOMOTOPY block in step 3 then
        # measures the actual drift and reports it.
        if _skip_homotopy and _baseline_coils is not None:
            _pin_bounds = {}
            _floor = float(coil_drift_floor_A)
            for _n, _b in _baseline_coils.items():
                _pin_bounds[_n] = [_b - _floor, _b + _floor]
            try:
                mygs.set_coil_bounds(_pin_bounds)
                if os.environ.get('PINJ_PROBE', '0') == '1':
                    print(f"    [SKIP_HOMOTOPY pre-solve] installed pin "
                          f"bounds = recon +/- {_floor:.0f} A on "
                          f"{len(_pin_bounds)} coils")
            except Exception as _pin_b_exc:
                print(f"  [SKIP_HOMOTOPY] tight-bound install failed "
                      f"({_pin_b_exc}); coils may drift via QP "
                      f"optimisation in the recon-anchor solve")

        try:
            (
                ne_perturb,
                te_perturb,
                ni_perturb,
                ti_perturb,
                w_ExB,
                jphi_perturb,
                diagnostics,
            ) = perturb_kinetic_equilibrium(
                mygs,
                psi_N,
                pressure,
                ne, te, ni, ti,
                input_j_phi,
                sigma_ne,
                sigma_te,
                sigma_ni,
                sigma_ti,
                sigma_jphi,
                n_ls, t_ls, j_ls,
                initial_Ip_target,
                l_i_target_draw,
                Zeff,
                npsi,
                input_jinductive=input_jinductive,
                l_i_tolerance=l_i_tolerance,
                psi_pad=psi_pad,
                constrain_sawteeth=constrain_sawteeth,
                recalculate_j_BS=recalculate_j_BS,
                isolate_edge_jBS=isolate_edge_jBS,
                floor_j_BS=floor_j_BS,
                jBS_diff=jBS_diff,
                Z_imp=Z_imp,
                p_diff=p_diff,
                jphi_diff=jphi_diff,
                accept_anchor_inband=accept_anchor_inband,
                perturb_jind_in_anchor=perturb_jind_in_anchor,
                scale_jBS=scale_jBS,
                swb_iterations=swb_iterations,
                diagnostic_plots=diagnostic_plots,
                psi_N_kinetic=psi_N_kinetic,
                p_fast=p_fast,
                j_NBI=j_NBI,
                j_RF=j_RF,
                aux_sigmas=aux_sigmas,
                aux_baselines=aux_baselines,
                aux_length_scales=aux_length_scales,
                ni_from_zeff=ni_from_zeff,
                max_proxy_draws=max_proxy_draws,
                p_thresh=p_thresh,
                # the run's single Generator -- every GPR draw in this draw
                # comes off it, so `seed` governs the whole ensemble
                rng=rng,
                bnd_diag_callback=_report_bnd,
                recon_eq_snapshot=_diff_recon_eq_snap,
                spike_profile_recon_cached=_diff_spike_recon,
                spike_delta_ref=(_delta_spike0_raw if jbs_delta_mode else None),
                spike_delta_baseline=(np.asarray(baseline_j_BS, dtype=float)
                                      if (jbs_delta_mode
                                          and _delta_spike0_raw is not None
                                          and baseline_j_BS is not None)
                                      else None),
                proxy_bias_warmstart=_proxy_bias_warmstart,
                pin_jphi=pin_jphi,
            )
        except Exception as e:
            # Catch ANY exception during a perturbed solve -- ValueError
            # / RuntimeError from the GS solver, TypeError from OFT's
            # bootstrap-spike analyzer when it returns None on a
            # pathological draw, ArithmeticError, etc.  The user's spec
            # is "if a perturbation fails for any reason, reset mygs to
            # the recon baseline and try the next perturbation".  We
            # deliberately swallow the broad Exception class here
            # (KeyboardInterrupt and SystemExit derive from
            # BaseException so are not caught -- ctrl-c still aborts).
            import traceback as _tb
            _err_short = str(e).strip().splitlines()[-1] if str(e) else type(e).__name__
            print(f"\n  STOPPED: {type(e).__name__}: {_err_short}")
            print(f"  Skipping equilibrium {count+1}/{n_equils}.")
            _skl = os.environ.get('BQ_SKIPLOG')
            if _skl:
                with open(_skl, 'a') as _skf:
                    _skf.write(f"draw {count+1}/{n_equils}: {type(e).__name__}: {_err_short}\n")

            # Restore the recon baseline state -- (psi, coils, bounds) --
            # so the next draw starts from a known-good state rather
            # than whatever partial / failed state the perturbation left
            # behind.  Bounds were stashed on mygs at install time and
            # may have been cleared by perturb_kinetic_equilibrium's
            # try/finally around solve_with_bootstrap; restore them too.
            if coil_drift is not None and _baseline_psi is not None:
                try:
                    # Prefer full equilibrium-object restore when
                    # available (PR #248).  Falls back to manual
                    # restore with update_bounds=True for legacy OFT.
                    if _warmstart_eq_snap is not None:
                        mygs.replace_eq(source_eq=_warmstart_eq_snap)
                    else:
                        mygs.set_coil_currents(_baseline_coils)
                        mygs.set_psi(_baseline_psi, update_bounds=True)
                    _stashed = getattr(mygs, '_coil_drift_bounds', None)
                    if _stashed is not None:
                        mygs.set_coil_bounds(_stashed)
                    print(f"  reset mygs to recon baseline "
                          f"(psi + coils + bounds re-applied).")
                except Exception as _reset_exc:
                    print(f"  WARN: baseline reset failed: {_reset_exc}")
                    print(_tb.format_exc())
            # Restore the strong soft-reg before moving to the next draw
            # (perturb left the weak SWB-exploration reg active; the reset
            # above doesn't touch coil reg).  Harmless if absent.
            _sreg = getattr(mygs, '_strong_coil_reg', None)
            if _sreg is not None:
                try:
                    mygs.set_coil_reg(reg_terms=_sreg)
                except Exception:
                    pass
            if pbar is not None:
                pbar.update(1)
            continue

        # ---- Restore strong coil soft-reg for the post-perturb phase ----
        # perturb_kinetic_equilibrium runs its whole exploratory phase (SWB
        # + recon-anchor + l_i band loop) under a WEAK reg so coils can
        # settle the perturbed equilibrium.  The post-perturb Ip-align /
        # in-spec / homotopy phase below is where we WANT coils held near
        # recon, so re-install the strong reg here.
        _sreg = getattr(mygs, '_strong_coil_reg', None)
        if _sreg is not None:
            try:
                mygs.set_coil_reg(reg_terms=_sreg)
            except Exception as _sreg_exc:
                print(f"  [reg] strong-reg restore failed ({_sreg_exc})")

        # ---- Post-perturb: iso-update + hard coil bounds --------------
        # The perturbed equilibrium out of perturb_kinetic_equilibrium is
        # accepted as-is (Ip is held to <0.05% of target natively by the
        # backend -- no per-draw Ip alignment).  We then:
        #   Step 2: update the isoflux target to the perturbed LCFS, and
        #   Step 3: install hard +/-coil_drift bounds and re-solve (with
        #           optional homotopy tightening).
        #   If the QP is feasible the draw is accepted; if infeasible the
        #   draw is rejected and the loop moves on (standard baseline
        #   reset).
        # Boundary diagnostic: state coming OUT of perturb_kinetic_equilibrium
        # (post-recon-anchor + post-GPR-loop + post-corrective-iteration)
        _report_bnd("after perturb_kinetic_eq")

        if coil_drift is not None and _recon_Ip is not None:
            _ip_aligned = False
            _post_align_failed = False
            # Per-draw Ip-secant alignment was removed (2026-05): the OFT
            # jphi-linterp cut-cell fix + the gs solver outer loop hold Ip to
            # <0.05% of target natively, so re-solving each draw to nudge Ip
            # was redundant and only added a per-draw boundary spread
            # (~0.3-0.5 mm) even at sigma=0.  Accept the perturbed
            # equilibrium as-is and run the iso-update + hard-bound homotopy
            # below against it.
            try:
                _ip_aligned = True

                # Step 2: update isoflux to the *post-Ip-aligned* boundary
                # (still free VSC).  This is the boundary the perturbed
                # equilibrium naturally settles into; using it as the new
                # isoflux target makes the next solve self-consistent --
                # otherwise tightening F9 bounds against the recon
                # boundary forces the QP to relax isoflux dramatically
                # (we've seen the plasma drop ~2 m vertically).
                #
                # GATE: only run iso-update if hard bounds are going to
                # be installed.  In SKIP_HARD mode, the post-Ip-align
                # equilibrium uses recon's isoflux directly.
                # Independent SKIP_ISO=1 env var also disables iso-update
                # even when hard bounds are active, useful for diagnosing
                # whether the iso-shifted boundary is what forces F9
                # drift > spec at low σ.
                #
                # SKIP_HOMOTOPY=1 (added 2026-05) -- diagnostic gate that
                # bypasses the coil-bound tightening passes (Step 3 below)
                # while STILL running iso-update.  Intended for the
                # PIN_JPHI / Tier-2 phantom-VSC use-case ("all coils
                # pinned"), where any homotopy-driven coil motion is
                # outside the experiment's premise.  In that mode the
                # Ip-aligned equilibrium is accepted as final; coil
                # currents stay exactly at recon values, so the only
                # remaining LCFS error is whatever the OFT solver
                # produces from the recon-pinned j_phi (i.e. the
                # jphi-linterp Ip-discretization residual ~ 0.04% after
                # the OFT outer loop fix).  Empirical observation
                # (2026-05 PIN_JPHI σ=0 sweep): homotopy pass 2 drifts
                # F-coils 0.17% / VSC 1.75% and contributes ~1.9 mm RMS
                # / ~7 mm max LCFS displacement vs recon -- entirely a
                # regularization artifact, not physics.  SKIP_HOMOTOPY
                # zeros it.  Do NOT use for the production Tier-1/3
                # studies where bounded coil variation is part of the
                # uncertainty model -- those need the homotopy.
                # SKIP_HARD / SKIP_ISO / SKIP_HOMOTOPY hoisted to
                # function scope (see line ~1965) so the pre-solve
                # coil-pin block can see them.  Reads here removed
                # to avoid shadowing.
                _iso_updated = False
                if _ip_aligned and not _skip_hard and not _skip_iso:
                    try:
                        # Iso-update is PHYSICS (LCFS feeds the next solve's
                        # isoflux constraints), not a diagnostic — do NOT
                        # use safe_trace_surf here; rolling back the
                        # equilibrium afterwards undoes state needed
                        # downstream and was empirically observed to
                        # degrade in-spec yield.
                        _new_lcfs = mygs.trace_surf(1.0 - psi_pad)
                        if _new_lcfs is None or len(_new_lcfs) < 4:
                            # Try a slightly larger pad for the trace
                            _new_lcfs = mygs.trace_surf(1.0 - 1e-4)
                        if _new_lcfs is not None and len(_new_lcfs) >= 4:
                            # Reuse the recon's isoflux weights pattern
                            # (constant 200 across all points by default)
                            _new_w = (np.ones(len(_new_lcfs))
                                       * float(np.mean(np.abs(
                                            mygs.psi_bounds[1] - mygs.psi_bounds[0]))) * 0
                                       + 200.0)
                            mygs.set_isoflux(_new_lcfs, weights=_new_w)
                            _iso_updated = True
                            print(f"  [iso-update] new isoflux from perturbed LCFS "
                                  f"({len(_new_lcfs)} pts)")
                        else:
                            print(f"  [iso-update] LCFS trace failed; keeping recon isoflux")
                    except Exception as _iso_exc:
                        print(f"  [iso-update] failed: {_iso_exc}")

                # Step 3: install hard coil bounds (with optional
                # homotopy: progressive tightening across multiple passes
                # warm-started from prior pass's psi).  _skip_hard already
                # fetched above.
                #
                # Single-pass mode (homotopy_passes=None): use uniform
                # ``coil_drift`` for all coils + ``#VSC``.  Backward
                # compatible with prior behaviour.
                #
                # Homotopy mode: each pass is a (drift_F, drift_VSC) tuple.
                # Passes run sequentially.  On success the last good
                # (psi, coils) is snapshotted and we advance.  On failure
                # we restore the last good state and stop tightening.
                # The final accepted draw uses the tightest pass that
                # converged.
                _vsc_in_set = tuple(c for c in vsc_coils if c in _baseline_coils)
                # Use MIN baseline magnitude so the VSC channel bound is
                # conservative for BOTH coils.  Per-coil drift = |VSC_value|
                # / |baseline_coil|, so max drift = VSC_bound / min_baseline.
                # Using max would let the smaller coil drift more than spec.
                _vsc_min_base = (min(abs(_baseline_coils[c]) for c in _vsc_in_set)
                                  if _vsc_in_set else 0.0)
                _vsc_max_base = _vsc_min_base  # back-compat alias

                def _build_bounds(drift_F, drift_VSC):
                    """Build per-coil bare bounds + #VSC channel bound.

                    Pure per-coil semantics:
                    - Non-VSC F-coils: bare ≤ drift_F → total drift ≤ drift_F
                    - F9A/F9B (VSC pair): bare ≤ drift_F (same per-coil bound)
                    - #VSC channel: ≤ drift_VSC × min(|F9A|, |F9B|)
                      so VSC contribution to either coil ≤ drift_VSC

                    Total F9 drift = bare ± VSC contribution.  Worst case:
                    drift_F + drift_VSC.  Caller must pick drift_F + drift_VSC
                    ≤ user spec for F9 (e.g. drift_F=1%, drift_VSC=1% for
                    a global ≤2% spec).
                    """
                    _b = {}
                    for _name, _base in _baseline_coils.items():
                        _delta = max(drift_F * abs(_base),
                                      float(coil_drift_floor_A))
                        _b[_name] = [_base - _delta, _base + _delta]
                    if _vsc_in_set:
                        _vsc_delta = max(drift_VSC * _vsc_min_base,
                                          float(coil_drift_floor_A))
                        _b['#VSC'] = [-_vsc_delta, _vsc_delta]
                    return _b

                _final_drifts = None  # last successful pass's drifts
                _final_pass_idx = -1
                _final_drift_F_lim = coil_drift
                _final_drift_VSC_lim = coil_drift

                # SKIP_HOMOTOPY=1 short-circuits the entire pass loop:
                # accept the Ip-aligned (and optionally iso-updated)
                # equilibrium as final, leaving coil currents at their
                # recon values.  Populate the drift bookkeeping with
                # zeros so downstream IN_SPEC and HDF5 diagnostics
                # reflect the "all coils pinned" reality rather than
                # NaN/False (which would suppress them from plots).
                if _skip_homotopy and _ip_aligned and not _skip_hard:
                    # Measure actual per-coil drift from the recon-anchor
                    # and Ip-align solves above.  Pin bounds installed
                    # just after warmstart restore (recon +/-
                    # coil_drift_floor_A) should hold drift to
                    # essentially zero, but if the QP saturated against
                    # the tight bounds we want the diagnostic to show
                    # it -- not a hard-coded zero.  Previously this
                    # block populated _final_drifts = {n: 0.0 for ...}
                    # unconditionally, which hid genuine coil motion
                    # (observed 2026-05: F9A VSC drifted ~5% per draw
                    # under perturbed pressure when SKIP_HOMOTOPY skipped
                    # the bound install -- reported as 0% in the H5,
                    # and plot_traces correctly drew nothing because
                    # there was nothing to show).  The pin-bounds
                    # install at warmstart fixes the physical drift;
                    # this measurement makes sure the diagnostic
                    # reflects reality.
                    try:
                        _cur_skip, _ = mygs.get_coil_currents()
                        _final_drifts = _coil_drift_pct(
                            _cur_skip, _baseline_coils)
                        _max_d = max((abs(d)
                                      for d in _final_drifts.values()),
                                     default=0.0)
                    except Exception as _meas_exc:
                        # Fallback: zero -- but mark the failure so
                        # downstream readers know.
                        print(f"  [homotopy] coil-drift measurement "
                              f"failed in SKIP_HOMOTOPY mode "
                              f"({_meas_exc}); reporting zeros")
                        _final_drifts = {n: 0.0
                                          for n in _baseline_coils}
                        _max_d = 0.0
                    print(f"  [homotopy] SKIPPED (SKIP_HOMOTOPY=1 -- "
                          f"coils pinned at recon +/- "
                          f"{coil_drift_floor_A:.0f} A; "
                          f"measured max drift = {_max_d:.3f}%)")
                    _final_pass_idx = 0
                    # The "pass limit" here = 0% by design (we asked the
                    # QP to keep coils at recon).  The MEASURED drift
                    # goes into _final_drifts above.  If a coil
                    # saturated the floor (e.g. drifted exactly to the
                    # floor cap), measured drift will report it.
                    _final_drift_F_lim = 0.0
                    _final_drift_VSC_lim = 0.0
                    _report_bnd("after homotopy (SKIPPED)")

                if _ip_aligned and not _skip_hard and not _skip_homotopy:
                    _passes = (homotopy_passes if homotopy_passes is not None
                               else [(coil_drift, coil_drift)])
                    _last_good_psi   = mygs.get_psi(False).copy()
                    _last_good_coils = dict(_baseline_coils)  # fallback only

                    for _p_idx, (_dF, _dVSC) in enumerate(_passes):
                        _bounds_p = _build_bounds(_dF, _dVSC)
                        mygs.set_coil_bounds(_bounds_p)
                        _label = (f"pass {_p_idx+1}/{len(_passes)}"
                                   if len(_passes) > 1 else "single-pass")
                        try:
                            mygs.solve()
                            # Snapshot success
                            _cur, _ = mygs.get_coil_currents()
                            _all_drifts = _coil_drift_pct(
                                _cur, _baseline_coils)
                            _f_only = {n: d for n, d in _all_drifts.items()
                                        if n.startswith('F') and n not in _vsc_in_set}
                            _vsc_only = {n: d for n, d in _all_drifts.items()
                                          if n in _vsc_in_set}
                            _max_f = max((abs(d) for d in _f_only.values()),
                                          default=0.0)
                            _max_vsc = max((abs(d) for d in _vsc_only.values()),
                                            default=0.0)
                            print(f"  [homotopy {_label}] F=+/-{_dF*100:.1f}%  "
                                  f"VSC=+/-{_dVSC*100:.1f}% -> SOLVED "
                                  f"(max non-VSC F-coil drift={_max_f:.2f}%, "
                                  f"max VSC drift={_max_vsc:.2f}%)")

                            # ---- QP saturation guard ----
                            # If the QP solution sits at the bound (active
                            # constraint), the optimiser wanted to go
                            # further but couldn't.  Accepting that state
                            # as the draw's final answer is the trigger
                            # for the cascading-plasma-loss bug: at tight
                            # bounds (esp. Pass 3) the flux topology
                            # touches the constraint surface, save_eqdsk's
                            # q-profile trace fails across most psi, and
                            # mygs's internal solver state is silently
                            # left degenerate -- the NEXT draw's anchor
                            # solve then collapses to axis=(0.5,0), Ip=0,
                            # and the rest of the sweep cascades.
                            # Treating saturation as infeasible and
                            # rolling back to the last good (looser) pass
                            # eliminates the cascade.  The factor 0.99
                            # absorbs QP numerical slop without missing
                            # the active-constraint case (saturation is
                            # ~exact in practice; non-saturation usually
                            # leaves several % of headroom).
                            _sat_F   = (_max_f   >= 0.99 * _dF   * 100.0)
                            _sat_VSC = (_max_vsc >= 0.99 * _dVSC * 100.0)
                            if _sat_F or _sat_VSC:
                                _which = []
                                if _sat_F:
                                    _which.append(
                                        f"F ({_max_f:.2f}% vs cap "
                                        f"{_dF*100:.1f}%)")
                                if _sat_VSC:
                                    _which.append(
                                        f"VSC ({_max_vsc:.2f}% vs cap "
                                        f"{_dVSC*100:.1f}%)")
                                print(f"  [homotopy {_label}] saturation "
                                      f"detected on {'; '.join(_which)} "
                                      f"-> treating as infeasible to "
                                      f"avoid mygs state corruption")
                                if _final_pass_idx < 0:
                                    # No prior good pass to fall back to
                                    # (even Pass 1 saturated).  Reject
                                    # the draw entirely.
                                    _post_align_failed = True
                                else:
                                    # Roll back to last good pass and
                                    # re-solve so mygs's FF'/P' state
                                    # matches the restored psi.
                                    try:
                                        _lg_dF, _lg_dVSC = _passes[
                                            _final_pass_idx]
                                        mygs.set_psi(_last_good_psi,
                                                     update_bounds=True)
                                        mygs.set_coil_currents(
                                            _last_good_coils)
                                        mygs.set_coil_bounds(
                                            _build_bounds(_lg_dF, _lg_dVSC))
                                        mygs.solve()
                                    except Exception as _rb_exc:
                                        print(f"  [homotopy] WARN: "
                                              f"rollback re-solve failed "
                                              f"({_rb_exc}); stats may "
                                              f"be stale")
                                    print(f"  [homotopy] rolled back to "
                                          f"pass {_final_pass_idx + 1} "
                                          f"(saturation)")
                                break  # stop tightening

                            _last_good_psi   = mygs.get_psi(False).copy()
                            _last_good_coils = dict(_cur)
                            _final_drifts    = _all_drifts
                            _final_pass_idx  = _p_idx
                            _final_drift_F_lim   = _dF
                            _final_drift_VSC_lim = _dVSC

                            # ---- Early stop: skip a tighter pass the current
                            # solution already can't satisfy ----
                            # This pass solved with HEADROOM (not saturated), so
                            # _max_f/_max_vsc are the minimal coil drift needed to
                            # hold the boundary. If the NEXT pass's bound is
                            # tighter than that minimum, it cannot hold the
                            # boundary within bound -- it will saturate or burn a
                            # full maxits solve and roll right back here. Stop now
                            # and keep this solution (the in-spec verdict is
                            # already determined by these drifts). This also
                            # covers the "coils unchanged vs the previous pass"
                            # case (bound not binding), in its general form.
                            if _p_idx + 1 < len(_passes):
                                _nF, _nVSC = _passes[_p_idx + 1]
                                if (_max_f > _nF * 100.0
                                        or _max_vsc > _nVSC * 100.0):
                                    print(f"  [homotopy {_label}] natural drift "
                                          f"(F={_max_f:.2f}%, VSC={_max_vsc:.2f}%) "
                                          f"exceeds next-pass bound "
                                          f"(F=+/-{_nF*100:.1f}%, "
                                          f"VSC=+/-{_nVSC*100:.1f}%) -> stop "
                                          f"(skip infeasible tighter pass)")
                                    break
                        except (ValueError, RuntimeError) as _hb_exc:
                            print(f"  [homotopy {_label}] F=+/-{_dF*100:.1f}%  "
                                  f"VSC=+/-{_dVSC*100:.1f}% -> infeasible "
                                  f"({_hb_exc})")
                            if _final_pass_idx < 0:
                                # First pass failed -> draw is rejected
                                _post_align_failed = True
                            else:
                                # Roll back to last successful pass and
                                # re-solve so mygs's internal FF'/P'
                                # state is consistent with the restored
                                # psi.  Without this re-solve,
                                # mygs.get_stats() returns inf because
                                # set_psi alone doesn't recompute the
                                # flux-surface-averaged quantities.
                                try:
                                    _lg_dF, _lg_dVSC = _passes[_final_pass_idx]
                                    mygs.set_psi(_last_good_psi,
                                                 update_bounds=True)
                                    mygs.set_coil_currents(_last_good_coils)
                                    mygs.set_coil_bounds(
                                        _build_bounds(_lg_dF, _lg_dVSC))
                                    mygs.solve()
                                except Exception as _rb_exc:
                                    print(f"  [homotopy] WARN: rollback "
                                          f"re-solve failed ({_rb_exc}); "
                                          f"stats may be stale")
                                print(f"  [homotopy] rolled back to pass "
                                      f"{_final_pass_idx + 1}")
                            break  # stop tightening
                    mygs.set_coil_bounds(None)
                    _report_bnd("after homotopy")

                # Compute in-spec status from final drifts (if any)
                _in_spec = False
                _max_f_drift = float('nan')
                _max_vsc_drift = float('nan')
                if _final_drifts is not None:
                    _f_only = {n: d for n, d in _final_drifts.items()
                                if n.startswith('F') and n not in _vsc_in_set}
                    _max_f_drift = float(max((abs(d) for d in _f_only.values()),
                                              default=0.0))
                    # VSC SELECTION metric: common-mode + differential channel
                    # gate (NOT the per-coil drift -- F9A's near-zero baseline
                    # makes a per-coil % meaningless for the anti-series pair).
                    # Computed from the post-homotopy coil currents; falls back
                    # to the per-coil max if the query fails.
                    try:
                        _cur_fin, _ = mygs.get_coil_currents()
                        _max_vsc_drift = _vsc_channel_drift_pct(
                            _cur_fin, _baseline_coils, _vsc_in_set)
                    except Exception:
                        _vsc_only = {n: d for n, d in _final_drifts.items()
                                      if n in _vsc_in_set}
                        _max_vsc_drift = float(max(
                            (abs(d) for d in _vsc_only.values()), default=0.0))
                    _in_spec = (_max_f_drift <= inspec_F_max * 100.0
                                 and _max_vsc_drift <= inspec_VSC_max * 100.0)
                    _spec_msg = ('IN_SPEC' if _in_spec else 'OUT_OF_SPEC')
                    print(f"  [in-spec] {_spec_msg}: max non-VSC F-drift="
                          f"{_max_f_drift:.2f}% (limit {inspec_F_max*100:.1f}%), "
                          f"max VSC drift={_max_vsc_drift:.2f}% "
                          f"(limit {inspec_VSC_max*100:.1f}%)")

            except Exception as _post_exc:
                print(f"  [post-perturb] unexpected failure: {_post_exc}")
                _post_align_failed = True

            if _post_align_failed:
                # Reset to baseline and skip this draw (treat as rejected).
                try:
                    # Prefer full equilibrium-object restore when
                    # available (PR #248).  Falls back to manual
                    # restore with update_bounds=True for legacy OFT.
                    if _warmstart_eq_snap is not None:
                        mygs.replace_eq(source_eq=_warmstart_eq_snap)
                    else:
                        mygs.set_coil_currents(_baseline_coils)
                        mygs.set_psi(_baseline_psi, update_bounds=True)
                except Exception:
                    pass
                if pbar is not None:
                    pbar.update(1)
                continue

        elapsed = time.perf_counter() - t_start
        elapsed_times.append(elapsed)
        total_elapsed = time.perf_counter() - t_batch_start
        print(f"  Wall-clock time: {elapsed:.1f}s  "
              f"(total: {total_elapsed/60:.1f} min, "
              f"avg: {np.mean(elapsed_times):.1f}s/eq)")

        if pbar is not None:
            pbar.set_postfix_str(
                f"avg={np.mean(elapsed_times):.0f}s/eq, "
                f"total={total_elapsed/60:.1f}min"
            )

        diagnostics['time'] = elapsed
        # Homotopy + in-spec bookkeeping (always present so downstream
        # H5 schema is consistent across draws; NaN/-1 when hard bounds
        # weren't installed e.g. SKIP_HARD=1).
        diagnostics['homotopy_pass']     = int(_final_pass_idx)
        diagnostics['homotopy_F_lim']    = float(_final_drift_F_lim)
        diagnostics['homotopy_VSC_lim']  = float(_final_drift_VSC_lim)
        diagnostics['max_F_drift_pct']   = float(_max_f_drift)
        diagnostics['max_VSC_drift_pct'] = float(_max_vsc_drift)
        diagnostics['in_spec']           = bool(_in_spec)
        diagnostics['inspec_F_max']      = float(inspec_F_max)
        diagnostics['inspec_VSC_max']    = float(inspec_VSC_max)
        diagnostics['l_i_target_used']   = float(l_i_target_draw)
        diagnostics['l_i_uncertainty']   = float(l_i_uncertainty)
        # The bootstrap scale this draw was sampled at (from jBS_scale_range).
        # Not archived -- it is a sampler input, recorded here so a run can be
        # audited against its seed without re-deriving the draw stream.
        diagnostics['scale_jBS']         = float(scale_jBS)

        # ---- save geqdsk to a temporary file, archive, delete -------
        eqdsk_filename = f"{header}_count={count}.geqdsk"
        full_path = os.path.abspath(eqdsk_filename)
        print(f"  Saving to: {full_path}")

        # safe_save_eqdsk: snapshot mygs equilibrium before save, restore
        # after.  Prevents save_eqdsk's q-profile tracer from shifting
        # mygs.get_globals()[0] by ~0.5-0.8% between this draw and the
        # next draw's warmstart capture (the empirical save_eqdsk
        # Ip-mutation pathology).
        safe_save_eqdsk(
            mygs,
            eqdsk_filename,
            nr=257, nz=257,
            truncate_eq=save_truncate_eq,
            lcfs_pad=psi_pad,
        )

        # Capture a high-resolution LCFS trace at the SAME mygs state
        # we just saved the eqdsk from.  The eqdsk's RBBBS/ZBBBS is only
        # ~100 pts (save_eqdsk samples coarsely), so comparing it
        # against the 10k-pt recon_lcfs_ref in plot_traces inflates RMS
        # by ~4 mm of pure sampling noise.  Storing this high-res trace
        # per draw lets plot_traces do an apples-to-apples comparison
        # (10k vs 10k) and matches the in-loop bnd-diag value.
        try:
            from .utils import safe_trace_surf as _safe_trace_lcfs
            perturbed_lcfs_ref = _safe_trace_lcfs(mygs, 1.0 - psi_pad)
            if (perturbed_lcfs_ref is None
                    or len(perturbed_lcfs_ref) < 4):
                # try a smaller pad if the standard one fails
                perturbed_lcfs_ref = _safe_trace_lcfs(mygs, 1.0 - 1e-4)
        except Exception as _lcfs_exc:
            print(f"  WARN: high-res LCFS capture failed "
                  f"({_lcfs_exc}); plot_traces will fall back to "
                  f"the eqdsk's coarse boundary for this draw")
            perturbed_lcfs_ref = None

        # TokaMaker's built-in X-point finder at this same (post-save)
        # solver state -- the authoritative B_p=0 saddle location for this
        # draw, stored for plot_boundary_point_traces.
        try:
            _draw_xpts, _draw_div = mygs.get_xpoints()
            _draw_xpts = (np.asarray(_draw_xpts, dtype=float)
                          if _draw_xpts is not None else None)
        except Exception as _xexc:
            print(f"  WARN: get_xpoints() failed ({_xexc}); "
                  f"this draw's top/bottom traces fall back to the "
                  f"axis-line intersection")
            _draw_xpts, _draw_div = None, None
        diagnostics['x_points'] = _draw_xpts
        diagnostics['diverted'] = _draw_div

        # Live-equilibrium FSA block at the same (post-save) state, so this
        # draw's own flux geometry travels into the archive for an exact
        # toroidal<->parallel current conversion at IMAS export. Defensive:
        # a capture failure never sinks a draw -- IMAS export just falls back
        # to the baseline-ratio reconstruction for it.
        diagnostics['eq_fsa'] = None
        if capture_live_eq:
            try:
                from .physics import capture_equilibrium_fsa
                diagnostics['eq_fsa'] = capture_equilibrium_fsa(
                    mygs, npsi=int(capture_npsi), psi_pad=psi_pad,
                    exact_inv_R2=bool(capture_exact_inv_R2))
            except Exception as _fsa_exc:
                print(f"  WARN: live-equilibrium FSA capture failed "
                      f"({_fsa_exc}); IMAS export for this draw will use the "
                      f"baseline-ratio reconstruction")

        # Guard get_stats: a degenerate draw (Ip->0 / collapsed plasma, e.g.
        # after Sauter "corrector convergence failed") makes OFT's l_i
        # normalization divide by zero. Don't let one bad draw crash the whole
        # run (which would discard every already-stored draw's filtering/figs):
        # set l_i to nan so the draw is filtered out downstream and continue.
        try:
            eq_stats_std = mygs.get_stats(li_normalization="std", lcfs_pad=psi_pad)
            li1 = eq_stats_std["l_i"]
            eq_stats_iter = mygs.get_stats(li_normalization="iter", lcfs_pad=psi_pad)
            li3 = eq_stats_iter["l_i"]
        except Exception as _stats_exc:
            print(f"  WARN: per-draw get_stats failed ({_stats_exc}); "
                  f"degenerate equilibrium -> l_i=nan (draw filtered out, "
                  f"run continues)")
            li1 = float('nan'); li3 = float('nan')

        # Pressure on the equilibrium grid (for storage and plotting).
        # Interpolate kinetic profiles onto psi_N if on a different grid.
        if psi_N_kinetic is not None:
            # PCHIP regrid (shared helper), matching _kin_to_eq
            _to_eq = lambda arr: pchip_interp(psi_N_kinetic, arr, psi_N)
            _ne_eqp, _te_eqp = _to_eq(ne_perturb), _to_eq(te_perturb)
            _ni_eqp, _ti_eqp = _to_eq(ni_perturb), _to_eq(ti_perturb)
        else:
            _ne_eqp, _te_eqp, _ni_eqp, _ti_eqp = (
                ne_perturb, te_perturb, ni_perturb, ti_perturb)
        # Thermal (main-ion + electron) pressure -- the part that perturbs with
        # the kinetic draw.
        pressure_perturb = EC * (_ne_eqp * _te_eqp + _ni_eqp * _ti_eqp)
        # Total pressure the GS solve actually used for this draw: thermal +
        # impurity(carbon) + fast + p_diff anchor, recomputed exactly as
        # perturb_kinetic_equilibrium built its solve pressure (same components
        # and grid). Stored as "pressure"; the thermal part is stored
        # separately so plots show the impurity+fast the solve added.
        pressure_total_perturb = pressure_perturb.copy()
        if Z_imp:
            from .physics import impurity_pressure as _impP
            pressure_total_perturb = pressure_total_perturb + _impP(
                _ne_eqp, _ni_eqp, _ti_eqp, Z_imp)
        if p_fast is not None:
            _pf_eq = np.asarray(p_fast, dtype=float)
            pressure_total_perturb = pressure_total_perturb + (
                _to_eq(_pf_eq) if psi_N_kinetic is not None else _pf_eq)
        if p_diff is not None:
            pressure_total_perturb = pressure_total_perturb + np.asarray(
                p_diff, dtype=float)

        # Extract coil currents from TokaMaker
        coil_current_dict, _ = mygs.get_coil_currents()

        # ---- Build a perturbed p-file from the baseline p-file --------
        # Start from the baseline so that profiles we don't perturb
        # (beam density, rotation, kpol, etc.) are preserved as-is.
        # Replace ne, te, ni, ti with the perturbed values, then
        # recompute derived quantities (nz1, ptot, diamagnetic
        # rotations, ExB decomposition) self-consistently.
        # Only meaningful for a TEXT p-file source: a binary IDA .cdf cannot be
        # draw-perturbed, and storing the raw source per draw just duplicates
        # ~190 MB x n_draws (new IDA-database files). Binary sources store None
        # here (the raw cdf lives once in _baseline; readers fall back).
        from .schema import is_binary_profile_source as _is_binary_src
        _pfile_is_text = (pfile_bytes is not None
                          and not _is_binary_src(pfile_bytes))
        perturbed_pfile_bytes = pfile_bytes if _pfile_is_text else None
        if _pfile_is_text:
            try:
                from .io.pfile import PFile as _PFile
                from .io import GEQDSKEquilibrium as _GEQDSK
                from scipy.interpolate import interp1d

                pf = _PFile.from_bytes(pfile_bytes)

                # Interpolate perturbed profiles → p-file grid.
                # When psi_N_kinetic is provided, the perturbed
                # profiles already cover the SOL — no extrapolation
                # needed.  Otherwise, fall back to the equilibrium
                # grid with edge-value fill (no cubic extrapolation).
                _psi_src = (psi_N_kinetic if psi_N_kinetic is not None
                            else psi_N)
                for pf_key, arr_si, scale in [
                    ("ne", ne_perturb, 1e-20),   # m^-3 → 10^20/m^3
                    ("te", te_perturb, 1e-3),     # eV   → keV
                    ("ni", ni_perturb, 1e-20),
                    ("ti", ti_perturb, 1e-3),
                ]:
                    if pf_key in pf:
                        psi_grid = pf.psinorm_for(pf_key)
                        vals = interp1d(
                            _psi_src, arr_si * scale,
                            kind="cubic",
                            bounds_error=False,
                            fill_value=(arr_si[0] * scale,
                                        arr_si[-1] * scale),
                        )(psi_grid)
                        pf.set_profile(pf_key, psi_grid, vals)

                # Recompute the impurity density from THIS draw's (ne, ni)
                # via quasineutrality, so the p-file species block implies
                # exactly this draw's Zeff -- one Zeff per draw across the
                # archive, the aux datasets, and the solve. With the active
                # zeff channel, ni was derived from the drawn Zeff, so
                # nz1 >= 0 by construction; in independent-ni mode an
                # over-drawn ni can leave small negative nz1 values, which
                # are clipped to zero (the old behaviour FROZE the baseline
                # nz1 instead, which made the p-file imply a second,
                # uncorrelated Zeff realization).
                try:
                    import warnings as _qn_w
                    with _qn_w.catch_warnings():
                        _qn_w.simplefilter("ignore", UserWarning)
                        pf.compute_quasineutrality()
                    _nz_entry = pf["nz1"]
                    _nz_vals = np.asarray(_nz_entry["data"], dtype=float)
                    if np.any(_nz_vals < 0):
                        pf.set_profile("nz1", _nz_entry["psinorm"],
                                       np.clip(_nz_vals, 0.0, None))
                except Exception as _qn_exc:
                    print(f"  WARNING: per-draw nz1 recompute failed "
                          f"({_qn_exc}); baseline nz1 retained")

                # Recompute total pressure
                pf.compute_pressure()

                # Recompute diamagnetic rotations + decomposition
                # using the perturbed equilibrium just written to disk
                eq = _GEQDSK(full_path)
                psi_pf = pf.psinorm_for("ne")
                dpsi = eq.psi_boundary - eq.psi_axis
                psi_Wb = psi_pf * dpsi + eq.psi_axis

                pf.compute_diamagnetic_rotations(psi_Wb)

                # Outboard midplane R, Bp, Bt for ExB / Er / Hahm-Burrell
                mid = eq.midplane
                psi_eq = eq.psi_N
                R_mid = interp1d(
                    psi_eq, mid["R"],
                    fill_value="extrapolate")(psi_pf)
                Bp_mid = interp1d(
                    psi_eq, mid["Bp"],
                    fill_value="extrapolate")(psi_pf)
                Bt_mid = interp1d(
                    psi_eq, mid["Bt"],
                    fill_value="extrapolate")(psi_pf)

                pf.compute_rotation_decomposition(
                    R=R_mid, Bp=Bp_mid, Bt=Bt_mid, psi=psi_Wb)

                perturbed_pfile_bytes = pf.to_bytes()
            except Exception as exc:
                import traceback
                print(f"  WARNING: could not build perturbed p-file: {exc}")
                traceback.print_exc()

        # ---- Option-A archival (store_achieved_jphi): the stored per-draw
        # j_phi is the ACHIEVED FSA current of THIS draw's converged solve
        # (mygs is at the same state full_path's eqdsk was just saved from),
        # not the GPR/anchor target -- so the 1-D profile matches the group's
        # own eqdsk. j_inductive is recomputed as the residual against the
        # physical per-draw bootstrap (closure exact; sub-zero sliver floored
        # and absorbed into j_BS, same convention as the model split).
        _dr_jphi_store = jphi_perturb
        _dr_jBS_store = diagnostics["j_BS"]
        _dr_jind_store = diagnostics["j_inductive"]
        if store_achieved_jphi:
            try:
                _dr_jphi_store = _achieved_jphi_fsa(
                    mygs, psi_N, psi_pad, sign_ref=jphi_perturb)
                _fx = np.zeros_like(np.asarray(psi_N, dtype=float))
                if j_NBI is not None:
                    _fx = _fx + np.asarray(j_NBI, dtype=float)
                if j_RF is not None:
                    _fx = _fx + np.asarray(j_RF, dtype=float)
                _dr_jind_store = (_dr_jphi_store
                                  - np.asarray(diagnostics["j_BS"], dtype=float)
                                  - _fx)
                if np.any(_dr_jind_store < 0.0):
                    _dr_jind_store = np.maximum(_dr_jind_store, 0.0)
                    _dr_jBS_store = _dr_jphi_store - _dr_jind_store - _fx
            except Exception as _aexc:
                print(f"  WARN: achieved-jphi draw archival failed ({_aexc}); "
                      f"storing the target profile instead")
                _dr_jphi_store = jphi_perturb
                _dr_jBS_store = diagnostics["j_BS"]
                _dr_jind_store = diagnostics["j_inductive"]

        store_equilibrium(
            header, count, full_path,
            psi_N,
            _dr_jphi_store,
            _dr_jBS_store,
            _dr_jind_store,
            ne_perturb, te_perturb,
            ni_perturb, ti_perturb,
            w_ExB,
            li1, li3,
            scan_key=scan_key,
            pressure=pressure_total_perturb,
            pressure_thermal=pressure_perturb,
            j_BS_edge=diagnostics["j_BS_edge"],
            pfile_bytes=perturbed_pfile_bytes,
            Zeff=Zeff_profile,
            coil_currents=coil_current_dict,
            psi_N_kinetic=psi_N_kinetic,
            homotopy_pass=diagnostics.get('homotopy_pass'),
            homotopy_F_lim=diagnostics.get('homotopy_F_lim'),
            homotopy_VSC_lim=diagnostics.get('homotopy_VSC_lim'),
            max_F_drift_pct=diagnostics.get('max_F_drift_pct'),
            max_VSC_drift_pct=diagnostics.get('max_VSC_drift_pct'),
            in_spec=diagnostics.get('in_spec'),
            inspec_F_max=diagnostics.get('inspec_F_max'),
            inspec_VSC_max=diagnostics.get('inspec_VSC_max'),
            perturbed_lcfs_ref=perturbed_lcfs_ref,
            l_i_target_used=diagnostics.get('l_i_target_used'),
            l_i_uncertainty=diagnostics.get('l_i_uncertainty'),
            x_points=diagnostics.get('x_points'),
            diverted=diagnostics.get('diverted'),
            aux=diagnostics.get('aux'),
            eq_fsa=diagnostics.get('eq_fsa'),
        )

        # Clean up on-disk eqdsk after archiving
        try:
            os.remove(full_path)
            print(f"  Deleted temporary file: {full_path}")
        except OSError as exc:
            print(f"  WARNING: could not delete {full_path}: {exc}")

        all_diagnostics.append(diagnostics)

        # ---- Proxy-bias warmstart for next draw ----
        # Keep the most recent successful draw's observed bias factor
        # (proxy/real_l_i).  Empirically the bias is stable across
        # draws within a single bouquet run -- using the previous
        # draw's bias means the next draw's first outer iter already
        # lands at l_i_target, eliminating one ~30s outer iteration.
        _new_bias = diagnostics.get('proxy_bias_observed')
        if _new_bias is not None and np.isfinite(_new_bias) and _new_bias > 0:
            _proxy_bias_warmstart = float(_new_bias)

        # (Warmstart was captured ONCE at recon state before the loop --
        # do NOT re-capture per draw, that would let one draw's
        # converged state pollute all subsequent draws.)

    if pbar is not None:
        pbar.close()

    return all_diagnostics


# ====================================================================
#  Single-equilibrium reconstruction from geqdsk + kinetic profiles
# ====================================================================
def reconstruct_equilibrium(mygs, eqdsk, ne, te, ni, ti, Zeff,
                            isoflux_pts, weights, psi_pad,
                            guess_jinductive,n_k,psi_bridge,rescale_j_BS,
                            shelf_psi_N,initialize_psi=True,
                            isolate_edge_jBS=False,
                            p_fast=None, Z_imp=None,
                            l_i_tolerance=0.01):
    r"""Reconstruct a single Grad-Shafranov equilibrium from a geqdsk
    reference and kinetic profiles, matching the EFIT :math:`l_i(1)`.

    The workflow is:

    1. Set isoflux boundary targets from the geqdsk LCFS.
    2. Compute bootstrap current via ``solve_with_bootstrap``.
    3. Fit a smooth inductive current profile with
       :func:`fit_inductive_profile`.
    4. Iterate on the inductive scale factor (hybrid secant–bisection
       with step clamping and psi save/restore) until the TokaMaker
       :math:`l_i(1)` matches the geqdsk value.
    5. (Residual-Ip secant removed.)  The jphi-linterp Ip drift is now
       corrected natively by the OFT solver (cut-cell fix + Ip outer
       loop in the gs solve), so no post-li-match Python Ip-rescaling
       secant is needed.

    Parameters
    ----------
    mygs : TokaMaker
        Initialised TokaMaker GS solver (mesh, regions, coils already
        set up).
    eqdsk : GEQDSKEquilibrium
        Parsed geqdsk equilibrium object.
    ne : ndarray
        Electron density on ``eqdsk.psi_N`` [m\ :sup:`-3`].
    te : ndarray
        Electron temperature on ``eqdsk.psi_N`` [eV].
    ni : ndarray
        Ion density on ``eqdsk.psi_N`` [m\ :sup:`-3`].
    ti : ndarray
        Ion temperature on ``eqdsk.psi_N`` [eV].
    Zeff : ndarray
        Effective charge on ``eqdsk.psi_N``.
    isoflux_pts : ndarray, shape (N, 2)
        :math:`(R, Z)` coordinates of isoflux constraint points
        [m].  Passed to ``mygs.set_isoflux``.
    weights : ndarray, shape (N,)
        Weights for each isoflux constraint point.
    psi_pad : float
        Padding inside the LCFS for :math:`l_i` evaluation.
    guess_jinductive : ndarray
        Initial guess for the inductive current-density profile,
        passed to ``solve_with_bootstrap`` as the starting
        :math:`j_{\rm inductive}` shape.
    n_k : int
        Spline order for :func:`fit_inductive_profile` (``k``
        parameter).
    psi_bridge : float
        :math:`\hat{\psi}` above which edge data are replaced by a
        zero anchor in :func:`fit_inductive_profile`.
    rescale_j_BS : bool
        If ``True``, jointly optimise a bootstrap rescaling factor
        in :func:`fit_inductive_profile`.
    shelf_psi_N : float
        If > 0, apply a flat shelf to :math:`j_{\rm BS}` for
        :math:`\hat{\psi} <` *shelf_psi_N* in
        :func:`fit_inductive_profile`.  ``0`` disables the shelf.
    initialize_psi : bool
        If ``True`` (default), call ``mygs.init_psi`` using LCFS
        geometry estimated from the geqdsk boundary.  Set to ``False``
        to skip initialisation (e.g. when reusing a prior solution).
    p_fast : ndarray, optional
        Fixed fast-ion (beam) pressure [Pa] on ``eqdsk.psi_N`` -- i.e.
        already regridded onto the EQUILIBRIUM grid by the caller, the
        same array the draws solve with.  ``None`` (default) means zero,
        which reproduces the pre-fix thermal-only behaviour bitwise.
    Z_imp : float, optional
        Single effective impurity charge for the one-Zeff impurity
        pressure term.  ``None``/``0`` (default) disables it.
    l_i_tolerance : float, optional
        The per-draw :math:`l_i` acceptance band, as a FRACTION (default 0.01
        == 1 %; pass ``config.generation.l_i_tolerance``).  Used only to decide
        whether the step-7 corrective iteration moved :math:`l_i(3)` off the
        step-6 matched value by more than the draws are allowed to sit from it
        -- a REPORTED condition, never a raise.  See the ``[li post-corrective]``
        block and issue #25.

    Returns
    -------
    dict
        Result dictionary containing reconstructed profiles, fields,
        and comparison data keyed as documented inline.
    """
    # ---- 0. Validate the equilibrium-grid inputs (fail before the solve) ----
    # Checked ahead of the OpenFUSIONToolkit imports below so a caller-side
    # shape error surfaces immediately, without needing OFT present.
    # p_fast is contractually a full EQUILIBRIUM-grid profile.  Without this
    # check a scalar or a length-1 array is silently broadcast by `+` below,
    # producing a flat fast-pressure offset while the draw paths (which regrid
    # through pchip_interp) would have raised -- i.e. the recon and the draws
    # would again solve different pressures, the exact bug this plumbing fixes.
    if p_fast is not None:
        p_fast = np.asarray(p_fast, dtype=float)
        _n_eq = np.shape(eqdsk.psi_N)[0]
        if p_fast.shape != (_n_eq,):
            raise ValueError(
                "reconstruct_equilibrium: p_fast must be a 1-D array on the "
                f"equilibrium grid eqdsk.psi_N (expected shape ({_n_eq},), got "
                f"{p_fast.shape}). Regrid the kinetic-grid profile first, e.g. "
                "bouquet.utils.pchip_interp(psi_N_kinetic, p_fast_kin, "
                "eqdsk.psi_N) -- the same kin->eq map the draws use. Scalars "
                "are rejected deliberately: they would broadcast into a flat "
                "offset instead of a profile."
            )
    # Z_imp needs no analogous check: it is a single effective impurity charge
    # (a scalar by design, see the docstring), consumed only by
    # physics.impurity_pressure, which coerces it with `float(Z_imp)`.  There
    # is no grid for it to mismatch, and a non-scalar would already fail loudly
    # in that coercion rather than broadcasting silently.

    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
    from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap

    if initialize_psi:
        # Estimate shape parameters from geqdsk LCFS geometry
        geo = eqdsk.geometry
        R0 = geo['R'][-1]
        Z0 = geo['Z'][-1]
        a = geo['a'][-1]
        kappa = geo['kappa'][-1]
        delta = geo['delta'][-1]
        mygs.init_psi(R0, Z0, a, kappa, delta)

    eqdsk_jtor = abs(eqdsk.j_tor_averaged_direct)

    # ---- 2. Bootstrap current ----
    # isolate_edge_jBS=False keeps the FULL Sauter bootstrap (physical core hump
    # + edge spike) so the reconstructed j_BS/j_inductive split matches what the
    # draws recompute via solve_with_bootstrap. isolate_edge_jBS=True instead
    # isolates the edge spike and parks a flat shelf in the (g-file-degenerate)
    # core -- robust but non-physical, and 2x below the draws' Sauter hump, which
    # left the stored baseline ohmic inflated relative to every draw.
    results = solve_with_bootstrap(
        mygs, ne, te, ni, ti, Zeff,
        abs(eqdsk.Ip), guess_jinductive,
        scale_jBS=1.0,
        isolate_edge_jBS=isolate_edge_jBS,
        diagnostic_plots=False,
    )

    # Convert SWB's parallel-projected bootstrap to the toroidal convention
    # shared by eqdsk_jtor and the fitted inductive profile, so the
    # j_BS / j_inductive split is done in a single convention.
    # Smooth the fragile near-axis / shelf-transition zone IMMEDIATELY after
    # conversion (shared helper, also applied to every per-draw spike) so the
    # inductive fit below sees the artifact-free profile rather than the raw
    # collapsed axis point.
    j_BS_isolated_raw = _swb_jbs_to_toroidal(mygs, results['isolated_j_BS'],
                                             psi_pad)
    j_BS_isolated = smooth_jbs_transition(j_BS_isolated_raw)

    # ---- 2b. Classify the j_phi profile ----
    # Classification (and the shelf locator below) get the RAW profile:
    # classify_jphi_profile uses spike[0] as its shelf/height reference, a
    # convention calibrated on unsmoothed profiles. Feeding it the smoothed
    # profile lifts that reference by ~2x and, on low-current shots whose
    # edge spike is comparable to the core hump, silently flips the mode to
    # L_mode -- which ZEROES the j_BS split (caught by the sigma=0 guard on a
    # low-current case: smoothed shelf 0.347 vs edge max 0.313 MA/m2 -> L_mode;
    # raw shelf 0.153 -> Lmode_like_jphi, the correct historical result).
    jphi_mode, spike_metrics = classify_jphi_profile(
        eqdsk.psi_N, eqdsk_jtor, j_BS_isolated_raw
    )

    # Pre-compute shelf location (needed for mode-dependent iteration)
    _, _shelf_psi_recon = _shelf_blend_decompose(
        eqdsk.psi_N, eqdsk_jtor, j_BS_isolated_raw, eqdsk_jphi=eqdsk_jtor
    )  # just to get shelf_psi; j_ind result discarded

    # ---- 3. Fit inductive profile ----
    baseline_li_proxy = calc_cylindrical_li_proxy(mygs, eqdsk_jtor, psi_pad)

    fit_result = fit_inductive_profile(
        mygs, eqdsk_jtor, j_BS_isolated, eqdsk.psi_N, psi_pad,
        baseline_li_proxy,
        k=n_k, psi_bridge=psi_bridge,
        rescale_j_BS=rescale_j_BS,
        shelf_psi_N=shelf_psi_N,
    )

    j_inductive_fit_raw = fit_result['j_inductive_fit']
    scale_opt = fit_result['ind_scale']
    bs_scale_opt = fit_result['bs_scale']
    j_BS_isolated = fit_result['j_BS_used']

    print(f"[fit] ind_scale={scale_opt:.6f}  bs_scale={bs_scale_opt:.6f}  "
          f"li_proxy={fit_result['fit_li']:.6f}  (target={baseline_li_proxy:.6f})")

    # (The shelf->spike transition smoothing that used to live here is now
    # applied by smooth_jbs_transition right after the SWB conversion above
    # -- the same shared treatment every per-draw spike gets, keeping the
    # recon/draw split sigma=0-consistent -- so the inductive fit also saw
    # the artifact-free profile.)

    # Use the spline-fit j_inductive directly. The corrective iteration
    # (section 7) will drive TokaMaker's output to match the target
    # j_phi = j_inductive_fit + j_BS_isolated, compensating for any
    # geometry-coupling distortion at the edge.
    j_inductive_fit = j_inductive_fit_raw

    # DEBUG: check spline fit for divot before corrective iteration
    _d2_max = 0
    _d2_idx = 0
    for _di in range(1, len(eqdsk.psi_N) - 1):
        _s1 = (j_inductive_fit[_di] - j_inductive_fit[_di-1]) / (eqdsk.psi_N[_di] - eqdsk.psi_N[_di-1])
        _s2 = (j_inductive_fit[_di+1] - j_inductive_fit[_di]) / (eqdsk.psi_N[_di+1] - eqdsk.psi_N[_di])
        _d2 = abs(_s2 - _s1) / 1e6
        if _d2 > _d2_max:
            _d2_max = _d2
            _d2_idx = _di
    print(f"[spline_check] max |d²j_ind/dpsi²| = {_d2_max:.4f} MA/m²/psiN² "
          f"at index {_d2_idx} (psi_N={eqdsk.psi_N[_d2_idx]:.5f})")

    # ---- 4. Pressure and GS profiles ----
    # The GS pressure here MUST match what every consumer of this
    # reconstruction subsequently solves, or l_i_target is measured on a
    # different (lower-pressure) equilibrium than the draws it targets:
    # less pressure -> smaller Shafranov shift -> R_axis inboard ->
    # l_i(3) ~ 1/R_axis reads HIGH.  l_i_target is load-bearing (acceptance
    # band centre and the Newton proxy target), so that bias propagates.
    #
    # Term order and semantics below mirror, exactly:
    #   perturb_kinetic_equilibrium  (per-draw)   -- thermal, +p_fast, +impurity
    #   the state anchor `pressure_solve`         -- pressure + imp + fast + diff
    # Keep the three sites in step; if you change one, change all of them.
    pres_tmp = 1.6022e-19 * (ne * te + ni * ti)

    # Fixed fast-ion pressure -- constant across draws, never perturbed.
    # Supplied already on the equilibrium grid (eqdsk.psi_N) by the caller,
    # which applies the same kin->eq PCHIP the draws use.
    if p_fast is not None:
        pres_tmp = pres_tmp + np.asarray(p_fast, dtype=float)

    # Impurity (carbon) thermal pressure: one-Zeff single-impurity model on the
    # SAME (ne, ni, Z_imp) set that derived the main ion.  Single-ion
    # e*(ne*Te + ni*Ti) omits this.
    if Z_imp:
        from .physics import impurity_pressure
        pres_tmp = pres_tmp + impurity_pressure(ne, ni, ti, Z_imp)

    # NOTE: p_diff is deliberately NOT plumbed here.  It is defined as
    # (equilibrium.pressure - reconstructed baseline pressure), i.e. it is
    # computed FROM this reconstruction's output; feeding it back into the
    # reconstruction's input would be circular.  It is applied downstream, to
    # the baseline anchor and to every draw, where that definition holds.
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
    pprime_tmp = pchip_derivative(eqdsk.psi_N, pres_tmp) / psi_range
    pprime_tmp[-1] = 0.0

    pp_prof = {"type": "linterp", "y": pprime_tmp, "x": eqdsk.psi_N}
    ffp_prof = {
        "type": "jphi-linterp",
        "y": j_inductive_fit + j_BS_isolated,
        "x": eqdsk.psi_N,
    }

    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)
    mygs.solve()

    # ---- 5. Hybrid secant–bisection iteration to match eqdsk li(1) ----
    #
    # Guard-rails that prevent TokaMaker from being given profiles too
    # far from the last converged state:
    #   a) The secant step is clamped to ±max_step_frac of the current
    #      ind_factor so the GS solver always starts close to its
    #      previous solution.
    #   b) Once we have a bracket (one point above, one below target)
    #      bisection is used whenever the (clamped) secant would escape
    #      the bracket.
    #   c) The last converged psi is saved with get_psi / set_psi so
    #      that a non-converged solve does not poison subsequent
    #      iterations.
    # ---- l_i estimator: target and measure the SAME functional (issue #20) --
    #
    # bouquet's ``li(2)`` key (io/geqdsk.py:1562) is, despite its name,
    # numerically the ITER-normalized li(3):
    #
    #     li(2)_key = li_from_definition / circum^2 * 2 * vol / R_axis
    #               = 2 * int(Bp^2 dV) / ((mu0 * Ip)^2 * R_axis)
    #
    # i.e. every perimeter and volume factor cancels; only R_axis survives.
    # That is exactly the functional TokaMaker's ``li_normalization='iter'``
    # evaluates, and it is the ONLY estimator the two codes agree on:
    # measured 0.17% across the 16 DIII-D 169510 beta-scan g-files.
    # The key NAME is historical and misleading -- it is not Jackson's li(2).
    #
    # The previous target, ``li["li(1)_EFIT"]``, carries the g-file's
    # RBBBS/ZBBBS polygon perimeter and volume, while the convergence
    # measurement (`get_stats(li_normalization='std')`) carries TokaMaker's
    # own perimeter -- and TokaMaker's `gs_get_qprof` radially PROJECTS the
    # 1-lcfs_pad surface onto the true separatrix before summing arc length
    # (OFT grad_shaf.F90:4447-4456, unconditional) whereas `gs_comp_globals`
    # integrates to the separatrix with no pad.  The two li(1) numbers
    # therefore differ by +3.34% +/- 0.09% on the same equilibrium, so the
    # loop converged ~3.2% LOW in true l_i on every geqdsk-path
    # reconstruction.  See issue #20 for the full forensics.
    li_target = eqdsk.li["li(2)"]
    li_tol = 0.001
    max_li_iters = 20
    max_step_frac = 0.10  # cap secant steps at ±10 % of current value

    # -- save / restore helpers for the last known-good psi state -----
    _last_good_psi = mygs.get_psi(False).copy()

    def _save_psi():
        nonlocal _last_good_psi
        _last_good_psi = mygs.get_psi(False).copy()

    def _restore_psi():
        mygs.set_psi(_last_good_psi, update_bounds=True)

    def _solve_and_get_li(ind_factor):
        """Set profiles with scaled j_inductive, solve, return li(3)/'iter'.

        Saves psi on success; restores the previous good psi on
        TokaMaker solve failure so the next attempt starts clean.
        """
        ffp_tmp = {
            "type": "jphi-linterp",
            "y": ind_factor * j_inductive_fit + j_BS_isolated,
            "x": eqdsk.psi_N,
        }
        mygs.set_profiles(ffp_prof=ffp_tmp, pp_prof=pp_prof)
        try:
            mygs.solve()
        except ValueError:
            print(f"[li match]   solve failed for ind_factor={ind_factor:.6f}, "
                  "restoring last good psi")
            _restore_psi()
            return None  # signal failed solve
        _save_psi()
        # 'iter' == li(3) == the estimator li_target is on (issue #20).
        eq_stats = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
        return eq_stats['l_i']

    # -- bracket bookkeeping ------------------------------------------
    # bracket_lo: (ind, li) with li < li_target  (err < 0)
    # bracket_hi: (ind, li) with li > li_target  (err > 0)
    bracket_lo = bracket_hi = None

    def _update_bracket(ind, li):
        nonlocal bracket_lo, bracket_hi
        if li < li_target:
            if bracket_lo is None or abs(li - li_target) < abs(bracket_lo[1] - li_target):
                bracket_lo = (ind, li)
        else:
            if bracket_hi is None or abs(li - li_target) < abs(bracket_hi[1] - li_target):
                bracket_hi = (ind, li)

    # -- initial two evaluations --------------------------------------
    eq_stats_0 = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
    ind_0, li_0 = 1.0, eq_stats_0['l_i']
    _save_psi()
    _update_bracket(ind_0, li_0)

    ind_1 = 1.05
    li_1_sec = _solve_and_get_li(ind_1)
    if li_1_sec is not None:
        _update_bracket(ind_1, li_1_sec)

    print(f"[li match] target={li_target:.6f}  [estimator: li(3)/'iter']")
    print(f"[li match] iter 0: ind_factor={ind_0:.6f}  li={li_0:.6f}  err={li_0 - li_target:.6f}")
    print(f"[li match] iter 1: ind_factor={ind_1:.6f}  li={li_1_sec:.6f}  err={li_1_sec - li_target:.6f}")

    for li_iter in range(2, max_li_iters):
        err_0 = li_0 - li_target
        err_1 = li_1_sec - li_target

        if li_1_sec is not None and abs(err_1) < li_tol:
            print(f"[li match] converged at iter {li_iter}: "
                  f"ind_factor={ind_1:.6f}  li={li_1_sec:.6f}")
            break

        # -- propose next ind_factor ----------------------------------
        use_bisection = False

        if li_1_sec is None:
            # Previous solve failed — fall back to bisection if we have
            # a bracket, otherwise halve the step toward last good point
            use_bisection = True
        else:
            denom = err_1 - err_0
            if abs(denom) < 1e-14:
                use_bisection = True
            else:
                ind_secant = ind_1 - err_1 * (ind_1 - ind_0) / denom
                ind_secant = max(ind_secant, 0.0)

        if use_bisection and bracket_lo is not None and bracket_hi is not None:
            ind_new = 0.5 * (bracket_lo[0] + bracket_hi[0])
            print(f"[li match]   bisection -> {ind_new:.6f}")
        elif use_bisection:
            # No bracket yet — retreat halfway toward ind_0
            ind_new = 0.5 * (ind_0 + ind_1)
            print(f"[li match]   midpoint fallback -> {ind_new:.6f}")
        else:
            # Clamp secant step to ±max_step_frac of current value
            max_delta = max_step_frac * abs(ind_1)
            ind_clamped = np.clip(ind_secant,
                                  ind_1 - max_delta,
                                  ind_1 + max_delta)
            if ind_clamped != ind_secant:
                print(f"[li match]   clamped secant {ind_secant:.6f} "
                      f"-> {ind_clamped:.6f}")

            # If we have a bracket, ensure we stay inside it
            if bracket_lo is not None and bracket_hi is not None:
                blo, bhi = sorted([bracket_lo[0], bracket_hi[0]])
                if not (blo <= ind_clamped <= bhi):
                    ind_new = 0.5 * (bracket_lo[0] + bracket_hi[0])
                    print(f"[li match]   secant escaped bracket, "
                          f"bisection -> {ind_new:.6f}")
                else:
                    ind_new = ind_clamped
            else:
                ind_new = ind_clamped

        # -- evaluate ---------------------------------------------------
        ind_0, li_0 = ind_1, li_1_sec if li_1_sec is not None else li_0
        ind_1 = ind_new
        li_1_sec = _solve_and_get_li(ind_1)
        if li_1_sec is not None:
            _update_bracket(ind_1, li_1_sec)

        li_disp = f"{li_1_sec:.6f}" if li_1_sec is not None else "FAILED"
        err_disp = (f"{li_1_sec - li_target:.6f}"
                    if li_1_sec is not None else "N/A")
        print(f"[li match] iter {li_iter}: ind_factor={ind_1:.6f}  "
              f"li={li_disp}  err={err_disp}")
    else:
        print(f"[li match] WARNING: did not converge within "
              f"{max_li_iters} iterations")

    # Ensure the final state is from a converged solve
    if li_1_sec is None:
        _restore_psi()

    _eq_stats_final = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
    final_li = _eq_stats_final['l_i']
    Ip_tokamaker = _eq_stats_final['Ip']
    print(f"[li match] final li(3)={final_li:.6f}  target={li_target:.6f}  |err|={abs(final_li - li_target):.6f}")

    # ---- 6. li-matched inductive profile (Ip-correction secant removed) --
    # The jphi-linterp Ip drift is corrected natively by the OFT solver
    # (cut-cell fix + Ip outer loop in the gs solve), so the post-li-match
    # Python Ip-rescaling secant has been removed.  Retain the li-matched
    # inductive profile, which the corrective iteration (section 7) consumes.
    Ip_desired = abs(eqdsk.Ip)
    j_ind_li = ind_1 * j_inductive_fit  # li-matched inductive profile

    # -- Final stats (after li match) -------------------------------------
    _eq_stats_final = mygs.get_stats(li_normalization='iter', lcfs_pad=psi_pad)
    final_li = _eq_stats_final['l_i']
    Ip_tokamaker = _eq_stats_final['Ip']
    print(f"[final] li(3)={final_li:.6f}  Ip={Ip_tokamaker:.1f}  "
          f"Ip_err={100 * (Ip_tokamaker - Ip_desired) / Ip_desired:+.4f}%  "
          f"li_err={abs(final_li - li_target):.6f}")

    # ---- Cross-estimator drift report (issue #20 de-circularization) -----
    #
    # `li_err` above compares the two numbers the secant loop drives
    # together, so it reads ~0 BY CONSTRUCTION and can never expose an
    # estimator mismatch.  Report the *other* estimator as well, computed
    # both ways on the same converged equilibrium:
    #
    #   li3: g-file `li(2)` key    vs  TokaMaker 'iter'   <- the matched pair
    #   li1: g-file `li(1)_EFIT`   vs  TokaMaker 'std'    <- the free pair
    #
    # The li1 pair is NOT driven by anything, so its residual is a live
    # measurement of the geometry/convention drift between the two codes.
    # Historically ~+3.3% on DIII-D geqdsks; a change in that number is the
    # signal that a convention moved on one side or the other.
    _li1_final = float(mygs.get_stats(
        li_normalization='std', lcfs_pad=psi_pad)['l_i'])
    _li1_gfile = float(eqdsk.li.get("li(1)_EFIT", float('nan')))
    _li3_gfile = float(li_target)
    def _dpct(a, b):
        return 100.0 * (a - b) / b if (np.isfinite(b) and b != 0) else float('nan')
    li_cross = {
        # matched (targeted) pair -- small by construction
        "li3_tokamaker_iter": float(final_li),
        "li3_gfile_li2key": _li3_gfile,
        "li3_drift_pct": _dpct(final_li, _li3_gfile),
        # free (untargeted) pair -- the honest estimator-drift monitor
        "li1_tokamaker_std": _li1_final,
        "li1_gfile_efit": _li1_gfile,
        "li1_drift_pct": _dpct(_li1_final, _li1_gfile),
    }
    print(f"[li cross-estimator] MATCHED li(3): TokaMaker'iter'="
          f"{final_li:.6f} vs g-file li(2)key={_li3_gfile:.6f} "
          f"({li_cross['li3_drift_pct']:+.3f}%)")
    print(f"[li cross-estimator] FREE    li(1): TokaMaker'std' ="
          f"{_li1_final:.6f} vs g-file li(1)EFIT={_li1_gfile:.6f} "
          f"({li_cross['li1_drift_pct']:+.3f}%)  "
          f"<- untargeted; drift here is real, not circular")

    # ---- 7. Mode-dependent corrective iteration ----
    #
    # TokaMaker's jphi-linterp converts j_phi to FF' using flux-surface
    # geometry (<R>, <1/R>).  The output j_phi generally differs from
    # the input because the geometry changes after solving.  Iterate
    # the input to drive the output toward the geqdsk target.
    #
    # For Lmode_like_jphi: only correct in the core (psi_N < shelf),
    # preserving the Sauter edge spike that the geqdsk lacks.
    Ip_final_target = abs(eqdsk.Ip)

    if jphi_mode == 'L_mode':
        # TODO: validate with L-mode test data
        j_BS_isolated_corr = np.zeros_like(eqdsk.psi_N)
        corr_target = eqdsk_jtor.copy()
        print("[reconstruct] L_mode: zeroing j_BS_isolated, using geqdsk j_phi as j_inductive")
    else:
        # H_mode and Lmode_like_jphi: target = j_inductive_fit + j_BS_isolated.
        # Always trust the Sauter edge spike rather than the geqdsk edge.
        # The spline fit already matches the geqdsk in the core
        # (since it fits eqdsk - j_BS), so no blend with eqdsk_jtor
        # is needed — avoiding blend-induced kinks and preserving
        # the Sauter edge structure.
        corr_target = (j_ind_li + j_BS_isolated).copy()
        j_BS_isolated_corr = j_BS_isolated.copy()

    # Adaptive corrective iteration.
    #
    # protect_state=True: keep-best on the FULL-DOMAIN j_phi RMS, and land on
    # the best state seen instead of on whatever the last Newton step produced.
    # This is the same semantics the IMAS forward-solve path has used since the
    # corrector was reused there (run.py); the geqdsk path was the one call site
    # still trusting its last iterate.  It matters here for the same reason it
    # matters there -- the loop's own stopping rule is on the EDGE RMS
    # (psi_N > 0.9), so an iterate that improves the edge while degrading the
    # core satisfies it and is then kept.  Measured on the beta-scan family the
    # corrective iteration was WORSENING the core j_phi RMS by ~32% and the
    # on-axis value by ~73%; keep-best on the full-domain RMS makes the
    # trajectory monotone non-increasing by construction.  See issue #25.
    #
    # No knob values change: rtol=0.05, max_iters=8, min_iters=2, damping=1.0
    # (the IMAS site's rtol=0.02 / damping=0.5 are ITS tuning, not part of this
    # change -- only the state protection is mirrored).
    # Issue #29: the target's amplitude comes from the l_i secant and never
    # saw Ip -- measured +6.0 % of Ip on the golden, +6.2 % on a DIII-D
    # archive.  Renormalise (uniformly; l_i is shape-only) to the current the
    # solver will actually produce, so the corrector chases a reachable target.
    corr_target, _corr_ip_factor = _renormalize_target_to_Ip(
        mygs, eqdsk.psi_N, corr_target, Ip_final_target, psi_pad,
        label="jphi_corr/recon")
    j_phi_output_corr, _n_corr, _corr_hist = _corrective_jphi_iteration(
        mygs, eqdsk.psi_N, corr_target, pp_prof,
        Ip_final_target, pres_tmp[0], psi_pad,
        min_iters=2, max_iters=8, rtol=0.05, verbose=True,
        protect_state=True,
    )

    # ---- 7b. Did step 7 undo step 6?  (report, do not raise) -------------
    #
    # Step 6's secant matched l_i(3) to li_target to a fraction of a percent.
    # Step 7 then runs up to 8 further GS solves against a j_phi target, with a
    # stopping rule (rtol=0.05 on the EDGE RMS) that knows nothing about l_i.
    # Nothing re-checked l_i afterwards -- and there is no reason it should
    # land back on the matched value.  Issue #25 measured +0.50 / +0.75 / +0.76 %
    # drifts across the beta-scan family, i.e. up to 3/4 of a percent of pure
    # step-ordering error sitting under every draw.
    #
    # This is a REPORT, deliberately not a failure: making it fatal would be a
    # new acceptance criterion, which is not approved.  It prints loudly and is
    # archived (reconstruction_metrics), so a campaign carries the evidence
    # instead of the operator having to have been watching stdout.
    _li_post_corr = float(mygs.get_stats(
        li_normalization='iter', lcfs_pad=psi_pad)['l_i'])
    _li_corr_drift_pct = (100.0 * (_li_post_corr - final_li) / final_li
                          if final_li else float('nan'))
    _li_band_pct = 100.0 * float(l_i_tolerance)
    _li_corr_out_of_band = bool(np.isfinite(_li_corr_drift_pct)
                                and abs(_li_corr_drift_pct) > _li_band_pct)
    if _li_corr_out_of_band:
        print("  " + "!" * 68)
        print(f"  [li post-corrective] WARNING: the step-7 corrective "
              f"iteration moved l_i(3) OFF the step-6 matched value by "
              f"{_li_corr_drift_pct:+.3f}%, which is outside the per-draw l_i "
              f"band (+/-{_li_band_pct:.2f}%).")
        print(f"  [li post-corrective] matched (step 6) = {final_li:.6f}; "
              f"realized (post step 7) = {_li_post_corr:.6f}; "
              f"target = {li_target:.6f}.")
        print("  [li post-corrective] l_i_target is taken from the MATCHED "
              "value, so every draw is being banded around a number this "
              "equilibrium no longer carries.  See issue #25.")
        print("  " + "!" * 68)
    else:
        print(f"  [li post-corrective] l_i(3) matched={final_li:.6f} -> "
              f"realized={_li_post_corr:.6f} ({_li_corr_drift_pct:+.3f}%, "
              f"band +/-{_li_band_pct:.2f}%)")

    # ---- 8. Final profiles ----
    # The corrective iteration drove TokaMaker's output toward
    # corr_target (= Hermite-bridged j_inductive + j_BS in the edge,
    # geqdsk in the core).  The output j_phi_output_corr is smooth
    # and self-consistent.  Decompose by simple subtraction — no
    # re-running the Hermite bridge, which would create a different
    # optimisation and introduce kinks.
    j_phi_final = j_phi_output_corr.copy()
    j_BS_final = j_BS_isolated_corr.copy()

    if jphi_mode == 'L_mode':
        # TODO: validate with L-mode test data
        j_ind_final = j_phi_final.copy()
        j_BS_final = np.zeros_like(j_phi_final)
    else:
        j_ind_final = j_phi_final - j_BS_final
        j_ind_final = np.maximum(j_ind_final, 0.0)

    # ---- 9. Reconstruction quality metrics ----
    _edge_mask = eqdsk.psi_N > 0.9
    _core_mask = eqdsk.psi_N < 0.8

    # Boundary deviation: nearest-neighbor distance from geqdsk boundary
    # points to the TokaMaker LCFS contour (same method as plotting.py)
    from scipy.spatial import cKDTree as _cKDTree_q
    _psi_arr = mygs.get_psi(False)
    _psi_lcfs = float(mygs.psi_bounds[0])
    _fig_tmp, _ax_tmp = plt.subplots(1, 1)
    try:
        _cs = _ax_tmp.tricontour(
            mygs.r[:, 0], mygs.r[:, 1], mygs.lc, _psi_arr,
            levels=[_psi_lcfs])
        _segs = [v for seg in _cs.allsegs for v in seg if len(v) > 4]
    finally:
        plt.close(_fig_tmp)

    # Longest CLOSED segment.  On a diverted equilibrium the open separatrix
    # branch running to the divertor spans the full vessel height and can
    # carry MORE points than the closed LCFS, in which case a plain
    # max(..., key=len) measures the boundary against the wrong curve and the
    # verdict flips on a number that is off by two orders of magnitude
    # (issue #33).  Shared with plotting._lcfs_from_psi so the two cannot
    # drift apart.
    _lcfs_pts = select_closed_lcfs(_segs, context="reconstruction metrics")
    if _lcfs_pts is not None:
        _tree = _cKDTree_q(_lcfs_pts)
        _devs, _ = _tree.query(isoflux_pts)
        _bnd_rms_mm = float(np.sqrt(np.mean(_devs**2)) * 1e3)
        _bnd_max_mm = float(np.max(_devs) * 1e3)
    else:
        _bnd_rms_mm = float('nan')
        _bnd_max_mm = float('nan')

    quality = {
        'jphi_mode': jphi_mode,
        **spike_metrics,
        'jphi_core_rms': float(np.sqrt(np.mean(
            (j_phi_final[_core_mask] - eqdsk_jtor[_core_mask])**2))),
        'jphi_edge_rms': float(np.sqrt(np.mean(
            (j_phi_final[_edge_mask] - eqdsk_jtor[_edge_mask])**2))),
        'li_error': float(abs(final_li - li_target)),
        # l_i estimator scale that `li_error` / `final_li` are on, plus the
        # cross-estimator pair captured at the same state (issue #20).
        'li_scale': 'iter(li3)',
        # step-6 matched vs step-7 realized (issue #25).  Archived, not
        # enforced: a loud report, not an acceptance criterion.
        'li3_post_corrective': float(_li_post_corr),
        'li3_corrective_drift_pct': float(_li_corr_drift_pct),
        'li3_corrective_band_pct': float(_li_band_pct),
        'li3_corrective_out_of_band': bool(_li_corr_out_of_band),
        **{f'li_cross_{_k}': _v for _k, _v in li_cross.items()},
        'Ip_error_pct': float(100 * abs(Ip_tokamaker - Ip_desired) / Ip_desired),
        'boundary_rms_mm': _bnd_rms_mm,
        'boundary_max_dev_mm': _bnd_max_mm,
    }
    print(f"[quality] mode={jphi_mode}, core_rms={quality['jphi_core_rms']/1e6:.4f} MA/m², "
          f"edge_rms={quality['jphi_edge_rms']/1e6:.4f} MA/m², "
          f"li_err={quality['li_error']:.6f}, Ip_err={quality['Ip_error_pct']:.4f}%, "
          f"bnd_rms={_bnd_rms_mm:.2f} mm, bnd_max={_bnd_max_mm:.2f} mm")

    # FF' from the converged TokaMaker equilibrium
    _, F_prof, Fp_prof, _, _ = mygs.get_profiles(psi=eqdsk.psi_N)
    ffprime_tokamaker = F_prof * Fp_prof

    return {
        'ne': ne.copy(),
        'te': te.copy(),
        'ni': ni.copy(),
        'ti': ti.copy(),
        'Zeff': Zeff.copy(),
        'isoflux_pts': isoflux_pts.copy(),
        'weights': weights.copy(),
        'psi_lcfs_val': float(mygs.psi_bounds[0]),
        'j_inductive_fit': j_ind_final.copy(),
        'j_phi_fit': j_phi_final.copy(),
        'j_BS_used': j_BS_final.copy(),
        'psi': mygs.get_psi(False),
        'pprime': pprime_tmp.copy(),
        'ffprime': ffprime_tokamaker.copy(),
        'ind_factor_final': ind_1,
        'bs_factor_final': 1.0,
        'Ip_tokamaker': Ip_tokamaker,
        'eqdsk_jtor': eqdsk_jtor.copy(),
        'eqdsk_psi_N': eqdsk.psi_N.copy(),
        'eqdsk_pres': eqdsk.pres.copy(),
        'eqdsk_boundary_R': eqdsk.boundary_R.copy(),
        'eqdsk_boundary_Z': eqdsk.boundary_Z.copy(),
        'eqdsk_ffprim': eqdsk.ffprim.copy(),
        'eqdsk_li': dict(eqdsk.li),
        'eqdsk_Ip': eqdsk.Ip,
        'pres_tokamaker': pres_tmp.copy(),
        'psi_N_grid': eqdsk.psi_N.copy(),
        # `li_final` is the step-6 MATCHED l_i(3) -- the value the secant loop
        # actually drove onto li_target, and (issue #25) the one the ensemble's
        # l_i_target is now taken from.  The post-step-7 realized value is a
        # SEPARATE field, so provenance grows rather than shrinks.
        'li_final': final_li,
        'li_realized_post_corrective': float(_li_post_corr),
        'li_corrective_drift_pct': float(_li_corr_drift_pct),
        'quality': quality,
    }
