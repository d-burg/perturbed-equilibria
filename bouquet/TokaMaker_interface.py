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
  - ``generate_perturbed_equilibria`` – batch driver that archives
    perturbed equilibria to HDF5.
  - ``reconstruct_equilibrium`` – reconstruct a single equilibrium from
    a geqdsk reference and kinetic profiles, matching :math:`l_i(1)`
    via secant iteration.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from .sampling import (
    generate_perturbed_GPR,
    calc_cylindrical_li_proxy,
    get_li_proxy_geometry,
    calc_cylindrical_li_proxy_fast,
    _draw_monotonic_perturbation,
    EC,
    _MAX_PRESSURE_ITER,
    _MAX_LI_ITER,
)
from .utils import (
    Ip_flux_integral_vs_target,
    store_equilibrium,
    store_baseline_profiles,
)

# ---- Spline-based inductive profile fitting ----
def fit_inductive_profile(mygs, eqdsk_jtor, j_BS_isolated, psi_N, psi_pad,
                            baseline_li_proxy,
                            k=3, psi_bridge=0.99,
                            rescale_j_BS=False,
                            shelf_psi_N=0.0):
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

    spline = UnivariateSpline(psi_trusted, res_trusted, k=k,
                                s=len(psi_trusted) * np.var(res_trusted) * 0.01)
    j_inductive_basis = spline(psi_N)
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
        'spline': spline,
    }

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
    p_thresh=0.5,
    input_jinductive=None,
    l_i_tolerance=0.05,
    l_i_proxy_threshold=5.0,
    psi_pad=1e-3,
    constrain_sawteeth=True,
    recalculate_j_BS=True,
    isolate_edge_jBS=True,
    scale_jBS=1.0,
    diagnostic_plots=False,
    max_pressure_iter=_MAX_PRESSURE_ITER,
    max_li_iter=_MAX_LI_ITER,
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
        1-D baseline total pressure [Pa].
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
    Zeff : float
        Effective ion charge (scalar).
    npsi : int
        Normalised poloidal flux grid size.
    p_thresh : float
        Acceptable :math:`\langle P \rangle` mismatch [%].
    input_jinductive : ndarray or None
        Dimensionless inductive :math:`j_\phi` shape (required when
        ``recalculate_j_BS=True``).
    l_i_tolerance : float
        Absolute :math:`l_i` matching tolerance.
    l_i_proxy_threshold : float
        Proxy :math:`l_i` relative error threshold [%].
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
    diagnostic_plots : bool
        Show diagnostic matplotlib figures (including inside
        ``solve_with_bootstrap`` and ``find_optimal_scale``).
    max_pressure_iter : int
        Safety cap on pressure-matching loop.
    max_li_iter : int
        Safety cap on :math:`l_i`-matching loop.

    Returns
    -------
    tuple
        ``(ne_perturb, te_perturb, ni_perturb, ti_perturb,
        w_ExB, output_jphi, diagnostics)``
    """

    # ----------------------------------------------------------------
    #  1.  Lazy OFT imports (deferred so GPR-only use works without OFT)
    # ----------------------------------------------------------------
    from scipy.optimize import root_scalar
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS
    from OpenFUSIONToolkit.TokaMaker.bootstrap import (
        solve_with_bootstrap,
        find_optimal_scale,
    )

    # ----------------------------------------------------------------
    #  2.  Validate inputs
    # ----------------------------------------------------------------
    if recalculate_j_BS and input_jinductive is None:
        raise ValueError(
            "input_jinductive must be provided when recalculate_j_BS=True"
        )

    # ----------------------------------------------------------------
    #  3.  Perturb kinetic profiles to match <P>
    # ----------------------------------------------------------------
    inp_avg = mygs.flux_integral(psi_N, pressure)

    p_err = np.inf
    p_iter = 0
    print("Searching for pressure profile match...")

    while p_err > p_thresh:
        p_iter += 1
        if p_iter > max_pressure_iter:
            raise RuntimeError(
                f"Pressure match not found within {max_pressure_iter} iterations "
                f"(last error {p_err:.2f}% vs threshold {p_thresh}%)"
            )

        # Each profile gets its own σ, converted to normalised-profile units
        ne_perturb = _draw_monotonic_perturbation(
            psi_N, ne / ne[0], sigma_ne / ne[0], n_ls
        ) * ne[0]

        te_perturb = _draw_monotonic_perturbation(
            psi_N, te / te[0], sigma_te / te[0], t_ls
        ) * te[0]

        ni_perturb = _draw_monotonic_perturbation(
            psi_N, ni / ni[0], sigma_ni / ni[0], n_ls
        ) * ni[0]

        ti_perturb = _draw_monotonic_perturbation(
            psi_N, ti / ti[0], sigma_ti / ti[0], t_ls
        ) * ti[0]

        pres_tmp = EC * (
            ne_perturb * te_perturb + ni_perturb * ti_perturb
        )
        tmp_avg = mygs.flux_integral(psi_N, pres_tmp)
        p_err = np.mean(np.abs(inp_avg - tmp_avg) / inp_avg) * 100.0

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

    if recalculate_j_BS:
        # Always suppress solve_with_bootstrap's internal j_phi iteration
        # plots; the useful diagnostic is the "jphi-linterp | l_i iter"
        # figure produced later in the l_i loop.
        results = solve_with_bootstrap(
            mygs,
            ne_perturb, te_perturb, ni_perturb, ti_perturb,
            Zeff, Ip_target, input_jinductive,
            scale_jBS=scale_jBS,
            isolate_edge_jBS=isolate_edge_jBS,
            diagnostic_plots=False,
            verbose=False,
        )
        eq_stats = mygs.get_stats(lcfs_pad=psi_pad)

        new_jphi = results["total_j_phi"]
        full_j_BS = results["j_BS"]
        spike_profile = results["isolated_j_BS"]
        baseline_li_proxy = calc_cylindrical_li_proxy(mygs, new_jphi, psi_pad)

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
    final_scale_Ip = 1.0
    matched_j_inductive = (
        results["j_inductive"] if recalculate_j_BS else input_j_phi.copy()
    )

    # The proxy target starts at the baseline proxy value but is
    # adaptively corrected after each TokaMaker solve to account for
    # the systematic offset between the cylindrical proxy and the
    # actual equilibrium l_i.  This makes the proxy filter select
    # profiles that land near l_i_target in equilibrium space rather
    # than in proxy space.
    proxy_target = baseline_li_proxy

    for li_iter in range(1, max_li_iter + 1):
        if abs(l_i - l_i_target) <= l_i_tolerance:
            break

        t_phase = time.perf_counter()

        # ---- 5a. Draw j_phi perturbation matching l_i proxy --------
        step_j_phi = (
            results["j_inductive"] if recalculate_j_BS else input_j_phi
        )
        j_phi_0 = step_j_phi[0]

        # Pre-compute geometry once for the inner proxy loop (the
        # equilibrium state doesn't change between proxy evaluations).
        _geo = get_li_proxy_geometry(mygs, npsi, psi_pad)

        l_i_rel_err = np.inf
        proxy_draws = 0
        while l_i_rel_err > l_i_proxy_threshold:
            proxy_draws += 1
            jphi_perturb = generate_perturbed_GPR(
                psi_N,
                step_j_phi / j_phi_0,
                sigma_profile=sigma_jphi / j_phi_0,   # normalised σ
                length_scale=j_ls,
                n_samples=1,
                diag_plot=False,
            )
            jphi_perturb *= j_phi_0

            if np.any(jphi_perturb < 0.0):
                continue

            result_root = root_scalar(
                Ip_flux_integral_vs_target,
                args=(mygs, jphi_perturb, spike_profile, psi_N, Ip_target),
                bracket=[1.0e-10 * Ip_target, 1.0e1 * Ip_target],
                method="brentq",
                rtol=1e-6,
            )
            a_optimal = result_root.root
            matched_jphi_perturb = a_optimal * jphi_perturb + spike_profile

            # Fast proxy: uses cached geometry, no TokaMaker calls
            tmp_li_proxy = calc_cylindrical_li_proxy_fast(
                matched_jphi_perturb, _geo
            )
            l_i_rel_err = (
                100.0 * abs(tmp_li_proxy - proxy_target) / proxy_target
            )

        dt_proxy = time.perf_counter() - t_phase
        print(f"  [li_iter={li_iter}] Proxy matched in {proxy_draws} draws "
              f"({dt_proxy:.1f}s, err={l_i_rel_err:.3f}%)")

        # ---- 5b. Set up GS profiles --------------------------------
        psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
        pprime_tmp = np.gradient(pres_tmp) / (np.gradient(psi_N) * psi_range)
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
            spike_prof=spike_profile, find_j0=True,
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

        final_scale_Ip, _ = find_optimal_scale(
            mygs, psi_N, pres_tmp, ffp_prof, pp_prof,
            matched_j_inductive, Ip_target, psi_pad,
            spike_prof=spike_profile, find_j0=False,
            scale_j0=final_scale_j0, tolerance=0.001,
            diagnostic_plots=False, verbose=False,
        )
        dt_scale = time.perf_counter() - t_scale
        print(f"  [li_iter={li_iter}] find_optimal_scale: {dt_scale:.1f}s")

        # ---- 5d. Definitive sawtooth constraint (after Ip scaling) --
        if constrain_sawteeth:
            _, q, _, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
            if q[0] < 1.0:
                print("Skipping this equilibrium, q_0 < 1.0")
                l_i = np.inf
                continue

        j0_scales.append(final_scale_j0)
        Ip_scales.append(final_scale_Ip)

        # ---- 5e. Final GS solves (2 iterations for convergence) -----
        for _ in range(2):
            pprime_tmp = np.gradient(pres_tmp) / (
                np.gradient(psi_N) * psi_range
            )
            pprime_tmp[-1] = 0.0

            mygs.set_targets(Ip=Ip_target * final_scale_Ip, pax=pres_tmp[0])

            pp_prof = {"type": "linterp", "y": pprime_tmp, "x": psi_N}
            ffp_prof = {
                "type": "jphi-linterp",
                "y": matched_j_inductive * final_scale_j0 + spike_profile,
                "x": psi_N,
            }
            mygs.set_profiles(pp_prof=pp_prof, ffp_prof=ffp_prof)
            mygs.solve()

        # ---- 5f. Evaluate converged equilibrium ---------------------
        _, f, fp, p, pp = mygs.get_profiles(npsi=npsi, psi_pad=psi_pad)
        _, q, ravgs, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
        R_avg = ravgs[0]
        one_over_R_avg = ravgs[1]
        output_jphi = get_jphi_from_GS(f * fp, pp, R_avg, one_over_R_avg)

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

        eq_stats = mygs.get_stats(lcfs_pad=psi_pad)
        Ip = eq_stats["Ip"]
        l_i = eq_stats["l_i"]

        # Compute cylindrical proxy on the FINAL converged j_phi to
        # measure the proxy-vs-TokaMaker offset.  This diagnostic
        # helps calibrate the l_i_proxy_threshold parameter.
        final_li_proxy = calc_cylindrical_li_proxy_fast(output_jphi, _geo)
        proxy_vs_real = 100.0 * (final_li_proxy - l_i) / l_i if l_i != 0 else 0.0

        Ip_err = 100.0 * abs(Ip - Ip_target) / Ip_target

        # Adaptive proxy target correction: use the observed
        # proxy-to-equilibrium mapping to predict what proxy value
        # would produce l_i_target in the actual equilibrium.
        if l_i > 0 and np.isfinite(l_i):
            corrected_target = final_li_proxy * (l_i_target / l_i)
            # Blend: 70% new correction, 30% old target (smooths noise)
            proxy_target = 0.7 * corrected_target + 0.3 * proxy_target

        print(f"  l_i target (equil):   {l_i_target:.4f}")
        print(f"  proxy target:         {proxy_target:.4f}  (corrected)")
        print(f"  matched l_i (equil):  {l_i:.4f}")
        print(f"  matched l_i (proxy):  {final_li_proxy:.4f}")
        print(f"  Ip error vs target:   {Ip_err:.3f}%")
        print(f"  proxy vs real l_i:    {proxy_vs_real:+.2f}%")
        print(f"  |l_i - l_i_target|:   {abs(l_i - l_i_target):.4f}")

        iteration_l_is.append(l_i)
        iteration_Ips.append(Ip)
    else:
        # Fired only if the for-loop exhausted without break
        raise RuntimeError(
            f"l_i match not found after {max_li_iter} iterations "
            f"(last |l_i - target| = {abs(l_i - l_i_target):.4f}, "
            f"target={l_i_target:.4f}, best={l_i:.4f}).\n"
            f"Try reducing your kinetic profile uncertainties or "
            f"increasing l_i_tolerance."
        )

    # ----------------------------------------------------------------
    #  6.  Package outputs
    # ----------------------------------------------------------------
    # NOTE: w_ExB (E×B rotation) is not yet computed from the
    # perturbed equilibrium.  A zero placeholder is stored so the
    # output tuple and HDF5 schema remain forward-compatible.
    w_ExB = np.zeros_like(psi_N)

    diagnostics = {
        "j0_scales": j0_scales,
        "Ip_scales": Ip_scales,
        "iteration_l_is": iteration_l_is,
        "iteration_Ips": iteration_Ips,
        "j_inductive": matched_j_inductive * final_scale_j0,
        "j_BS": full_j_BS,
        "j_BS_edge": spike_profile,
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
    l_i_tolerance=0.03,
    l_i_proxy_threshold=5.0,
    psi_pad=1e-3,
    constrain_sawteeth=True,
    recalculate_j_BS=True,
    isolate_edge_jBS=True,
    jBS_scale_range=None,
    diagnostic_plots=True,
    scan_val=None,
    pfile_bytes=None,
    Zeff_profile=None,
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
    Zeff : float
        Effective ion charge.
    input_jinductive : ndarray or None
        Dimensionless inductive :math:`j_\phi` shape.
    l_i_tolerance : float
        Absolute :math:`l_i` tolerance.
    l_i_proxy_threshold : float
        Proxy :math:`l_i` relative-error threshold [%].
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
    diagnostic_plots : bool
        Show diagnostic matplotlib figures.
    scan_val : str, float, int, or None
        Optional scan-point label for nested HDF5 storage.
        ``None`` gives the flat layout.
    pfile_bytes : bytes or None
        Raw p-file content to store alongside each equilibrium.
    Zeff_profile : array-like or None
        1-D effective charge profile to store in HDF5.

    Returns
    -------
    list[dict]
        Diagnostics from each equilibrium.
    """
    all_diagnostics = []

    # self-consistent pressure for baseline <P>
    pressure = EC * (ne * te + ni * ti)
    npsi = len(pressure)

    # Pre-compute jBS scale factors for the whole batch (if requested).
    # Uses a uniform distribution within the specified range so that
    # extreme values (which can make l_i matching very difficult) are
    # strictly bounded.
    if jBS_scale_range is not None:
        lo, hi = jBS_scale_range
        jBS_scales = np.random.uniform(lo, hi, size=n_equils)
    else:
        jBS_scales = np.ones(n_equils)

    # Store baseline profiles and uncertainties so the .h5 file is
    # self-contained (the plotting GUI only needs the file path).
    store_baseline_profiles(
        header, psi_N,
        ne, te, ni, ti,
        pressure, input_j_phi,
        sigma_ne, sigma_te, sigma_ni, sigma_ti, sigma_jphi,
        initial_Ip_target, l_i_target,
        scan_val=scan_val,
    )

    t_batch_start = time.perf_counter()
    elapsed_times = []

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    pbar = (
        _tqdm(range(n_equils), desc="Bouquet", unit="eq")
        if _tqdm is not None
        else None
    )
    eq_iter = pbar if pbar is not None else range(n_equils)

    for count in eq_iter:
        scale_jBS = float(jBS_scales[count])
        eta_str = ""
        if elapsed_times:
            avg_s = np.mean(elapsed_times)
            remaining = avg_s * (n_equils - count)
            eta_min = remaining / 60.0
            eta_str = f"  ETA: {eta_min:.1f} min"
        print(f"\n{'='*60}")
        print(f"  Equilibrium {count+1}/{n_equils}  "
              f"(scale_jBS={scale_jBS:.4f}){eta_str}")
        print(f"{'='*60}")
        t_start = time.perf_counter()

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
                l_i_target,
                Zeff,
                npsi,
                input_jinductive=input_jinductive,
                l_i_tolerance=l_i_tolerance,
                l_i_proxy_threshold=l_i_proxy_threshold,
                psi_pad=psi_pad,
                constrain_sawteeth=constrain_sawteeth,
                recalculate_j_BS=recalculate_j_BS,
                isolate_edge_jBS=isolate_edge_jBS,
                scale_jBS=scale_jBS,
                diagnostic_plots=diagnostic_plots,
            )
        except RuntimeError as e:
            print(f"\n  STOPPED: {e}")
            print(f"  Skipping equilibrium {count+1}/{n_equils}.\n")
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

        # ---- save geqdsk to a temporary file, archive, delete -------
        eqdsk_filename = f"{header}_count={count}.geqdsk"
        full_path = os.path.abspath(eqdsk_filename)
        print(f"  Saving to: {full_path}")

        mygs.save_eqdsk(
            eqdsk_filename,
            nr=257, nz=257,
            truncate_eq=False,
            lcfs_pad=psi_pad,
        )

        eq_stats_std = mygs.get_stats(li_normalization="std", lcfs_pad=psi_pad)
        li1 = eq_stats_std["l_i"]
        eq_stats_iter = mygs.get_stats(li_normalization="iter", lcfs_pad=psi_pad)
        li3 = eq_stats_iter["l_i"]

        pressure_perturb = EC * (ne_perturb * te_perturb
                                  + ni_perturb * ti_perturb)

        # Extract coil currents from TokaMaker
        coil_current_dict, _ = mygs.get_coil_currents()

        store_equilibrium(
            header, count, full_path,
            psi_N,
            jphi_perturb,
            diagnostics["j_BS"],
            diagnostics["j_inductive"],
            ne_perturb, te_perturb,
            ni_perturb, ti_perturb,
            w_ExB,
            li1, li3,
            scan_val=scan_val,
            pressure=pressure_perturb,
            j_BS_edge=diagnostics["j_BS_edge"],
            pfile_bytes=pfile_bytes,
            Zeff=Zeff_profile,
            coil_currents=coil_current_dict,
        )

        # Clean up on-disk eqdsk after archiving
        try:
            os.remove(full_path)
            print(f"  Deleted temporary file: {full_path}")
        except OSError as exc:
            print(f"  WARNING: could not delete {full_path}: {exc}")

        all_diagnostics.append(diagnostics)

    if pbar is not None:
        pbar.close()

    return all_diagnostics


# ====================================================================
#  Single-equilibrium reconstruction from geqdsk + kinetic profiles
# ====================================================================
def reconstruct_equilibrium(mygs, eqdsk, ne, te, ni, ti, Zeff, 
                            isoflux_pts, weights, psi_pad,
                            guess_jinductive,n_k,psi_bridge,rescale_j_BS,
                            shelf_psi_N,initialize_psi=True):
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
    5. Correct residual Ip drift by iterating on the Ip target passed
       to ``mygs.set_targets(Ip=...)`` via a secant search.  The
       :math:`j_\phi` profile shape is preserved; TokaMaker internally
       enforces the Ip constraint.

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

    Returns
    -------
    dict
        Result dictionary containing reconstructed profiles, fields,
        and comparison data keyed as documented inline.
    """
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
    results = solve_with_bootstrap(
        mygs, ne, te, ni, ti, Zeff,
        abs(eqdsk.Ip), guess_jinductive,
        scale_jBS=1.0,
        isolate_edge_jBS=True,
        diagnostic_plots=False,
    )

    j_BS_isolated = results['isolated_j_BS']

    # ---- 3. Fit inductive profile ----
    baseline_li_proxy = calc_cylindrical_li_proxy(mygs, eqdsk_jtor, psi_pad)

    fit_result = fit_inductive_profile(
        mygs, eqdsk_jtor, j_BS_isolated, eqdsk.psi_N, psi_pad,
        baseline_li_proxy,
        k=n_k, psi_bridge=psi_bridge,
        rescale_j_BS=rescale_j_BS,
        shelf_psi_N=shelf_psi_N,
    )

    j_inductive_fit = fit_result['j_inductive_fit']
    scale_opt = fit_result['ind_scale']
    bs_scale_opt = fit_result['bs_scale']
    j_BS_isolated = fit_result['j_BS_used']

    print(f"[fit] ind_scale={scale_opt:.6f}  bs_scale={bs_scale_opt:.6f}  "
          f"li_proxy={fit_result['fit_li']:.6f}  (target={baseline_li_proxy:.6f})")

    # ---- 4. Pressure and GS profiles ----
    pres_tmp = 1.6022e-19 * (ne * te + ni * ti)
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
    pprime_tmp = np.gradient(pres_tmp) / (np.gradient(eqdsk.psi_N) * psi_range)
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
    li_target = eqdsk.li["li(1)_EFIT"]
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
        """Set profiles with scaled j_inductive, solve, return li(1).

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
        eq_stats = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
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
    eq_stats_0 = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
    ind_0, li_0 = 1.0, eq_stats_0['l_i']
    _save_psi()
    _update_bracket(ind_0, li_0)

    ind_1 = 1.05
    li_1_sec = _solve_and_get_li(ind_1)
    if li_1_sec is not None:
        _update_bracket(ind_1, li_1_sec)

    print(f"[li match] target={li_target:.6f}")
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

    _eq_stats_final = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
    final_li = _eq_stats_final['l_i']
    Ip_tokamaker = _eq_stats_final['Ip']
    print(f"[li match] final li(1)={final_li:.6f}  target={li_target:.6f}  |err|={abs(final_li - li_target):.6f}")

    # ---- 6. Ip-correction secant (set_targets Ip rescaling) ----------
    #
    # The jphi-linterp profile type has inherent Ip drift because the
    # actual integrated current depends on the equilibrium geometry.
    # We correct this by iterating on the Ip target passed to
    # mygs.set_targets() — TokaMaker internally rescales the current
    # density to enforce the constraint.  The j_phi profile *shape*
    # is unchanged, so li is minimally perturbed.
    Ip_desired = abs(eqdsk.Ip)
    Ip_tol = 0.0005  # relative tolerance on Ip
    max_Ip_iters = 8

    # Current j_phi components after li convergence (fixed throughout)
    j_ind_li = ind_1 * j_inductive_fit  # li-matched inductive profile
    j_phi_li = j_ind_li + j_BS_isolated

    def _solve_and_get_Ip(Ip_trial):
        """Set Ip target, solve, return (actual Ip, li)."""
        mygs.set_targets(Ip=Ip_trial,pax=pres_tmp[0])
        try:
            mygs.solve()
        except ValueError:
            print(f"[Ip match]   solve failed for Ip_target={Ip_trial:.1f}, "
                  "restoring last good psi")
            _restore_psi()
            return None, None
        _save_psi()
        stats = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
        return stats['Ip'], stats['l_i']

    Ip_err_rel = abs(Ip_tokamaker - Ip_desired) / Ip_desired
    if Ip_err_rel > Ip_tol:
        print(f"\n[Ip match] desired={Ip_desired:.1f}  current={Ip_tokamaker:.1f}  "
              f"err={100 * (Ip_tokamaker - Ip_desired) / Ip_desired:+.4f}%")

        # Two initial evaluations for secant:
        # pt 0: Ip_target = Ip_desired (the true target), result already known
        # pt 1: Ip_target adjusted by a first-order correction
        t0, Ip_0 = Ip_desired, Ip_tokamaker
        t1 = Ip_desired * (Ip_desired / Ip_tokamaker)  # first-order correction
        Ip_1, li_1_Ip = _solve_and_get_Ip(t1)

        if Ip_1 is not None:
            print(f"[Ip match] iter 0: Ip_target={t0:.1f}  Ip={Ip_0:.1f}")
            print(f"[Ip match] iter 1: Ip_target={t1:.1f}  Ip={Ip_1:.1f}  "
                  f"li={li_1_Ip:.6f}")

            for Ip_iter in range(2, max_Ip_iters):
                e0 = Ip_0 - Ip_desired
                e1 = Ip_1 - Ip_desired

                if abs(e1 / Ip_desired) < Ip_tol:
                    print(f"[Ip match] converged at iter {Ip_iter}: "
                          f"Ip_target={t1:.1f}  Ip={Ip_1:.1f}  li={li_1_Ip:.6f}")
                    break

                denom = e1 - e0
                if abs(denom) < 1.0:
                    print(f"[Ip match] secant denominator ~0, stopping")
                    break

                t_new = t1 - e1 * (t1 - t0) / denom
                t_new = max(t_new, 0.5 * Ip_desired)  # safety floor

                t0, Ip_0 = t1, Ip_1
                t1 = t_new
                Ip_1, li_1_Ip = _solve_and_get_Ip(t1)
                if Ip_1 is None:
                    print(f"[Ip match] solve failed, keeping previous result")
                    break

                print(f"[Ip match] iter {Ip_iter}: Ip_target={t1:.1f}  "
                      f"Ip={Ip_1:.1f}  li={li_1_Ip:.6f}")
            else:
                print(f"[Ip match] WARNING: did not converge within "
                      f"{max_Ip_iters} iterations")
    else:
        print(f"\n[Ip match] already within tolerance: "
              f"err={100 * (Ip_tokamaker - Ip_desired) / Ip_desired:+.4f}%")

    # -- Final stats after Ip correction --------------------------------
    _eq_stats_final = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
    final_li = _eq_stats_final['l_i']
    Ip_tokamaker = _eq_stats_final['Ip']
    print(f"[final] li(1)={final_li:.6f}  Ip={Ip_tokamaker:.1f}  "
          f"Ip_err={100 * (Ip_tokamaker - Ip_desired) / Ip_desired:+.4f}%  "
          f"li_err={abs(final_li - li_target):.6f}")

    # Final profiles (unchanged from li matching — Ip corrected via set_targets)
    j_ind_final = j_ind_li.copy()
    j_BS_final = j_BS_isolated.copy()
    j_phi_final = j_phi_li.copy()

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
        'li_final': final_li,
    }
