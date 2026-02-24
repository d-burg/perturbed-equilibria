"""
GPR profile perturber and perturbed Grad-Shafranov equilibrium workflow
=======================================================================

Provides:
  - ``GPRProfilePerturber`` – Gaussian-process-regression based profile
    perturbation class.
  - ``generate_perturbed_GPR`` – convenience one-call wrapper.
  - ``verify_gpr_statistics`` – Monte-Carlo validation of GPR sampling.
  - ``new_uncertainty_profiles`` – builds 1-D uncertainty envelopes.
  - ``perturb_kinetic_equilibrium`` – perturbs kinetic + current-density
    profiles and iterates to match Ip and l_i targets via TokaMaker.
  - ``generate_perturbed_equilibria`` – batch driver.
  - ``initialize_equilibrium_database`` / ``store_equilibrium`` /
    ``load_equilibrium`` – HDF5 archive helpers.
"""

import io
import os
import time
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import integrate
from scipy.stats import norm
from typing import Optional

import h5py

from .utils import (
    Ip_flux_integral_vs_target,
    initialize_equilibrium_database,
    store_equilibrium,
    store_baseline_profiles,
    load_equilibrium,
)

# OFT and scipy.optimize imports are deferred to perturb_kinetic_equilibrium
# so the GPR half of this module works without OpenFUSIONToolkit installed.


# ── physical constant used for pressure ────────────────────────────
EC = 1.6022e-19  # [J/eV]

# ── default iteration caps (safety valves) ─────────────────────────
_MAX_PRESSURE_ITER = int(1e5)
_MAX_LI_ITER = 20
_MAX_MONOTONIC_DRAWS = int(1e4)


# ====================================================================
#  GPR profile perturber
# ====================================================================
class GPRProfilePerturber:
    r"""Gaussian-process perturber for smooth 1-D MHD profiles.

    Generates correlated random perturbations whose pointwise
    standard deviation is set **directly** by the user-supplied
    :math:`\sigma(x)` (experimental uncertainty in profile units).

    Internally the kernel amplitude is fixed to unity so that

    .. math::

        \operatorname{Cov}\!\bigl[\delta f(x),\,\delta f(x')\bigr]
        = \sigma(x)\;\sigma(x')\; k_1\!\bigl(x,x'\bigr)

    where :math:`k_1` is the unit-variance base kernel and the
    marginal standard deviation at every grid point equals the
    input uncertainty exactly:

    .. math::

        \sigma_{\rm GP}(x) = \sigma(x)

    Parameters
    ----------
    kernel_func : str
        Kernel name: ``'rbf'`` (:math:`C^\infty`) or
        ``'matern52'`` (:math:`C^2`).
    length_scale : float
        Correlation length in :math:`\hat\psi` units.
        Controls *wiggliness* of the draws but does **not** affect
        the pointwise amplitude.
    """

    _ALLOWED_KERNELS = {"rbf", "matern52"}

    def __init__(
        self,
        kernel_func: str = "rbf",
        length_scale: float = 0.1,
    ):
        if kernel_func not in self._ALLOWED_KERNELS:
            raise ValueError(
                f"Kernel '{kernel_func}' not in {self._ALLOWED_KERNELS}.  "
                "Rougher kernels produce non-differentiable profiles "
                "unsuitable for MHD inputs."
            )
        self.kernel_func = kernel_func
        self.length_scale = float(length_scale)

        self._kernel = {
            "rbf": self._rbf_kernel,
            "matern52": self._matern52_kernel,
        }[self.kernel_func]

    # ---- unit-variance kernels --------------------------------------
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r"""Squared-exponential kernel with unit variance."""
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "euclidean")
        return np.exp(-0.5 * (d / self.length_scale) ** 2)

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r"""Matérn-5/2 kernel with unit variance."""
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "euclidean")
        s = np.sqrt(5.0) * d / self.length_scale
        return (1.0 + s + s**2 / 3.0) * np.exp(-s)

    # ---- core sampling method ----------------------------------------
    def generate_profiles(
        self,
        psi_N: np.ndarray,
        input_profile: np.ndarray,
        sigma_profile: np.ndarray,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""Draw perturbed profiles whose pointwise :math:`1\sigma`
        matches the supplied experimental uncertainty exactly.

        Parameters
        ----------
        psi_N : ndarray
            1-D normalised flux grid.
        input_profile : ndarray
            1-D baseline profile (GP mean).
        sigma_profile : ndarray
            1-D experimental uncertainty **in profile units** -- this
            becomes the GP's marginal standard deviation at every
            grid point.
        n_samples : int
            Number of independent draws.
        rng : numpy.random.Generator or None
            ``None`` creates a fresh unseeded generator.

        Returns
        -------
        ndarray, shape ``(n_samples, len(psi_N))``
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(psi_N)

        # 1. Unit-variance base kernel
        K = self._kernel(psi_N, psi_N)          # K(x,x) = 1

        # 2. Scale by σ(x):  C_ij = σ_i · σ_j · K_ij
        #    ⟹  C(x,x) = σ(x)²  ⟹  marginal std = σ(x)   ✓
        S = np.outer(sigma_profile, sigma_profile)
        K_scaled = K * S

        # 3. Eigen-decomposition (symmetric → eigh)
        vals, vecs = np.linalg.eigh(K_scaled)
        vals = np.maximum(vals, 0.0)

        # 4. Sample:  δf = V diag(√λ) z,   z ~ N(0, I)
        z = rng.standard_normal((n, n_samples))
        perturbations = vecs @ (np.sqrt(vals)[:, None] * z)

        # 5. Perturbed profiles  →  (n_samples, n_points)
        return input_profile[None, :] + perturbations.T


# ====================================================================
#  Convenience wrapper
# ====================================================================
def generate_perturbed_GPR(
    xdata: np.ndarray,
    profile: np.ndarray,
    sigma_profile: Optional[np.ndarray] = None,
    length_scale: float = 0.25,
    n_samples: int = 1,
    kernel_func: str = "rbf",
    rng: Optional[np.random.Generator] = None,
    diag_plot: bool = False,
) -> np.ndarray:
    r"""One-call wrapper: perturb a 1-D profile with a GPR draw.

    The ``sigma_profile`` input is the experimental
    :math:`1\sigma` uncertainty **in the same units as the profile**.
    It maps directly to the GP marginal standard deviation -- no
    separate ``variance`` parameter is needed.

    Parameters
    ----------
    xdata : ndarray
        1-D normalised flux grid :math:`\hat{\psi}`.
    profile : ndarray
        1-D baseline profile (GP mean).
    sigma_profile : ndarray or None
        1-D experimental :math:`1\sigma` uncertainty **in profile
        units**.  ``None`` gives zero (no perturbation).
    length_scale : float
        GPR correlation length (controls wiggliness).
    n_samples : int
        Number of independent draws.
    kernel_func : str
        ``'rbf'`` or ``'matern52'``.
    rng : numpy.random.Generator or None
        Optional random generator.
    diag_plot : bool
        Show a three-panel diagnostic figure.

    Returns
    -------
    ndarray
        If ``n_samples == 1``: 1-D array of length ``len(xdata)``.
        Otherwise: 2-D array ``(n_samples, len(xdata))``.
    """
    if sigma_profile is None:
        sigma_profile = np.zeros_like(xdata)

    perturber = GPRProfilePerturber(
        kernel_func=kernel_func,
        length_scale=length_scale,
    )

    perturbed = perturber.generate_profiles(
        psi_N=xdata,
        input_profile=profile,
        sigma_profile=sigma_profile,
        n_samples=n_samples,
        rng=rng,
    )

    # ---- diagnostic figure ------------------------------------------
    if diag_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Panel 1: profile ± σ_exp  (exact 1σ band)
        ax1.plot(xdata, profile, "k-", lw=2, label="Original profile")
        ax1.fill_between(
            xdata,
            profile - sigma_profile,
            profile + sigma_profile,
            alpha=0.3,
            label=r"$\pm\,1\sigma_{\rm exp}$ envelope",
        )
        ax1.set_xlabel(r"$\hat{\psi}$")
        ax1.set_ylabel("Profile value")
        ax1.set_title(r"Input profile with experimental $1\sigma$ uncertainty")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: sigma profile
        ax2.plot(xdata, sigma_profile, "r-", lw=2)
        ax2.set_xlabel(r"$\hat{\psi}$")
        ax2.set_ylabel(r"$\sigma_{\rm exp}(x)$ [profile units]")
        ax2.set_title("Experimental uncertainty profile")
        ax2.grid(True, alpha=0.3)

        # Panel 3: perturbed draws with band
        ax3.plot(xdata, profile, "k-", lw=3, label="Original")
        ax3.fill_between(
            xdata,
            profile - sigma_profile,
            profile + sigma_profile,
            alpha=0.15, color="gray",
            label=r"$\pm\,1\sigma_{\rm exp}$",
        )
        if n_samples == 1:
            ax3.plot(xdata, perturbed[0], "--", alpha=0.7, label="Perturbed")
        else:
            for i in range(perturbed.shape[0]):
                ax3.plot(
                    xdata, perturbed[i], "--", alpha=0.7,
                    label=f"Perturbed {i + 1}" if i < 10 else None,
                )
        ax3.set_xlabel(r"$\hat{\psi}$")
        ax3.set_ylabel("Profile value")
        ax3.set_title("Original and perturbed profiles")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    if n_samples == 1:
        return perturbed[0]
    return perturbed


# ====================================================================
#  Statistics verification
# ====================================================================
def verify_gpr_statistics(
    psi_N,
    profile,
    uncertainty_prof,
    length_scale=0.25,
    n_verification=5000,
    confidence_band=2.0,
):
    r"""Verify GPR sampling statistics against theoretical predictions.

    Draws a large number of samples and checks:

    1. Pointwise mean :math:`\approx` input profile  (bias check)
    2. Pointwise std  :math:`\approx` ``uncertainty_prof``  (variance check)
    3. Fraction of samples outside :math:`\pm k\sigma` :math:`\approx` theoretical  (tail check)

    Parameters
    ----------
    psi_N : ndarray
        Normalised flux grid.
    profile : ndarray
        Baseline profile (GP mean).
    uncertainty_prof : ndarray
        Experimental :math:`1\sigma` uncertainty envelope (same units
        as profile).
    length_scale : float
        GPR length-scale.
    n_verification : int
        Number of Monte-Carlo draws.
    confidence_band : float
        Number of :math:`\sigma` for the band (e.g. 2.0).

    Returns
    -------
    dict
        Verification diagnostics.
    """
    perturber = GPRProfilePerturber(
        kernel_func="rbf",
        length_scale=length_scale,
    )

    rng = np.random.default_rng(42)
    samples = perturber.generate_profiles(
        psi_N, profile, uncertainty_prof,
        n_samples=n_verification, rng=rng,
    )                                          # (n_verification, n_points)

    # ---- theoretical predictions ------------------------------------
    # Marginal std equals sigma_profile exactly by construction
    sigma_theory = uncertainty_prof

    # ---- empirical statistics ---------------------------------------
    empirical_mean = np.mean(samples, axis=0)
    empirical_std  = np.std(samples, axis=0)

    # ---- pointwise exceedance rate ----------------------------------
    residuals = samples - profile[None, :]     # (n_verification, n_points)
    outside = np.abs(residuals) > confidence_band * sigma_theory[None, :]
    # Fraction of samples outside the band at each point
    exceedance_per_point = np.mean(outside, axis=0)
    # Overall average exceedance rate
    avg_exceedance = np.mean(exceedance_per_point)

    # ---- theoretical exceedance for a Gaussian ----------------------
    theoretical_exceedance = 2.0 * norm.sf(confidence_band)  # two-tailed

    # ---- report -----------------------------------------------------
    print(f"Verification with {n_verification} samples")
    print(f"  Mean bias   (max |empirical - input|): "
          f"{np.max(np.abs(empirical_mean - profile)):.2e}")
    print(f"  Std ratio   (mean empirical/theory):   "
          f"{np.mean(empirical_std / np.maximum(sigma_theory, 1e-30)):.4f}"
          f"  (should be ≈ 1.0)")
    print(f"  Exceedance  (±{confidence_band:.1f}σ):              "
          f"{avg_exceedance:.4f}"
          f"  (theory: {theoretical_exceedance:.4f})")

    # ---- plot -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Empirical vs theoretical std
    axes[0, 0].plot(psi_N, sigma_theory, "k-", lw=2, label="Theory")
    axes[0, 0].plot(psi_N, empirical_std, "r--", lw=1.5, label="Empirical")
    axes[0, 0].set_ylabel(r"$\sigma(x)$")
    axes[0, 0].set_title("Pointwise std: theory vs empirical")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (b) Exceedance rate per point
    axes[0, 1].axhline(theoretical_exceedance, color="k", ls="--",
                        label=f"Theory ({theoretical_exceedance:.3f})")
    axes[0, 1].plot(psi_N, exceedance_per_point, "r-", alpha=0.7,
                     label="Empirical")
    axes[0, 1].set_ylabel(f"Fraction outside ±{confidence_band:.0f}σ")
    axes[0, 1].set_title("Exceedance rate per grid point")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (c) Histogram at the midpoint
    mid = len(psi_N) // 2
    axes[1, 0].hist(
        samples[:, mid], bins=60, density=True, alpha=0.6, label="Samples",
    )
    x_plot = np.linspace(
        profile[mid] - 4 * sigma_theory[mid],
        profile[mid] + 4 * sigma_theory[mid], 200,
    )
    axes[1, 0].plot(
        x_plot,
        norm.pdf(x_plot, loc=profile[mid], scale=sigma_theory[mid]),
        "k-", lw=2, label="Theory N(μ, σ²)",
    )
    axes[1, 0].set_title(
        f"Marginal distribution at ψ_N = {psi_N[mid]:.2f}"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Sample spaghetti with band
    axes[1, 1].plot(psi_N, profile, "k-", lw=2, label="Mean")
    axes[1, 1].fill_between(
        psi_N,
        profile - confidence_band * sigma_theory,
        profile + confidence_band * sigma_theory,
        alpha=0.25, label=f"±{confidence_band:.0f}σ band",
    )
    n_show = min(50, n_verification)
    for i in range(n_show):
        axes[1, 1].plot(psi_N, samples[i], "-", alpha=0.08, color="tab:blue")
    axes[1, 1].set_title(f"{n_show} sample draws")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$\hat{\psi}$")
    plt.tight_layout()
    plt.show()

    return {
        "empirical_mean": empirical_mean,
        "empirical_std": empirical_std,
        "sigma_theory": sigma_theory,
        "exceedance_per_point": exceedance_per_point,
        "avg_exceedance": avg_exceedance,
        "theoretical_exceedance": theoretical_exceedance,
    }


# ====================================================================
#  Internal inductance proxy
# ====================================================================
def calc_cylindrical_li_proxy(mygs, j_phi_profile, psi_pad):
    """
    Calculates a proxy for internal inductance l_i(3) using 1D profiles.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker Grad-Shafranov solver object (used for geometry queries).
    j_phi_profile : array-like
        The toroidal current density profile (perturbation target).
    psi_pad : float
        Padding inside the LCFS for profile queries.

    Returns
    -------
    li_proxy : float
        The estimated internal inductance.
    """
    n_psi = len(j_phi_profile)

    psi_N, f, fp, p, pp = mygs.get_profiles(npsi=n_psi, psi_pad=psi_pad)
    _, qvals, ravgs_q, dl, rbounds, zbounds = mygs.get_q(npsi=n_psi, psi_pad=psi_pad)
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]

    # 1. Unpack geometry from baseline
    R_avg = ravgs_q[0]
    dV_dPsi = ravgs_q[2]

    # 2. Calculate differentials
    grad_psi_N = np.gradient(psi_N)
    d_psi_real = grad_psi_N * psi_range
    dV = dV_dPsi * d_psi_real

    # 3. Geometry mappings
    V_enc = integrate.cumulative_trapezoid(dV, initial=0)
    V_tot = V_enc[-1]
    r_eff = np.sqrt(np.abs(V_enc) / (2 * np.pi**2 * R_avg))
    dA = dV / (2 * np.pi * R_avg)

    # 4. Enclosed current I(rho)
    dI = j_phi_profile * dA
    I_enc = integrate.cumulative_trapezoid(dI, initial=0)
    I_tot = I_enc[-1]

    # 5. Cylindrical poloidal field proxy B_p(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        B_p_proxy = I_enc / (2 * np.pi * r_eff)
    B_p_proxy[0] = 0.0  # core singularity: limit r→0

    # 6. Integrate magnetic energy
    B_p_sq = B_p_proxy**2
    W_pol_integral = integrate.trapezoid(B_p_sq * dV)

    # 7. Calculate l_i
    L_edge = 2 * np.pi * r_eff[-1]

    if I_tot == 0:
        return 0.0

    B_p_edge_avg = I_tot / L_edge
    li_proxy = W_pol_integral / (V_tot * B_p_edge_avg**2)

    return li_proxy

# ====================================================================
#  Helper: draw a monotonically-decreasing GPR perturbation
# ====================================================================
def _draw_monotonic_perturbation(
    psi_N,
    normalised_profile,
    sigma_profile,
    length_scale,
    max_draws=_MAX_MONOTONIC_DRAWS,
):
    r"""Repeatedly sample a GPR perturbation until the draw is
    monotonically decreasing.

    Parameters
    ----------
    psi_N : ndarray
        Normalised flux grid.
    normalised_profile : ndarray
        Profile divided by its on-axis value.
    sigma_profile : ndarray
        :math:`1\sigma` uncertainty in normalised-profile units
        (same grid).
    length_scale : float
        GPR correlation length.
    max_draws : int
        Safety cap on the number of attempts.

    Returns
    -------
    ndarray
        The accepted (still normalised) perturbation.

    Raises
    ------
    RuntimeError
        If no monotonic draw is found within *max_draws* attempts.
    """
    for _ in range(max_draws):
        sample = generate_perturbed_GPR(
            psi_N,
            normalised_profile,
            sigma_profile=sigma_profile,
            length_scale=length_scale,
            n_samples=1,
            diag_plot=False,
        )
        if np.all(np.diff(sample) <= 0.0):
            return sample

    raise RuntimeError(
        f"No monotonically-decreasing GPR draw found in {max_draws} attempts."
    )


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
    l_i_proxy_threshold=2.0,
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
    j_ls : float
        GPR length-scale for :math:`j_\phi`.
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
        results = solve_with_bootstrap(
            mygs,
            ne_perturb, te_perturb, ni_perturb, ti_perturb,
            Zeff, Ip_target, input_jinductive,
            scale_jBS=scale_jBS,
            isolate_edge_jBS=isolate_edge_jBS,
            diagnostic_plots=diagnostic_plots,
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

    for li_iter in range(1, max_li_iter + 1):
        if abs(l_i - l_i_target) <= l_i_tolerance:
            break

        # ---- 5a. Draw j_phi perturbation matching l_i proxy --------
        step_j_phi = (
            results["j_inductive"] if recalculate_j_BS else input_j_phi
        )
        j_phi_0 = step_j_phi[0]

        l_i_rel_err = np.inf
        while l_i_rel_err > l_i_proxy_threshold:
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

            tmp_li_proxy = calc_cylindrical_li_proxy(mygs, matched_jphi_perturb, psi_pad)
            l_i_rel_err = (
                100.0 * abs(tmp_li_proxy - baseline_li_proxy) / baseline_li_proxy
            )
        print("Found potential l_i match via proxy!\n")

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
        print(f"\n >>> Finding optimal j_phi scale factor")
        final_scale_j0, final_jphi = find_optimal_scale(
            mygs, psi_N, pres_tmp, ffp_prof, pp_prof,
            matched_j_inductive, Ip_target, psi_pad,
            spike_prof=spike_profile, find_j0=True,
            diagnostic_plots=diagnostic_plots,
        )

        print(f"\n >>> Finding optimal Ip scale factor")
        final_scale_Ip, _ = find_optimal_scale(
            mygs, psi_N, pres_tmp, ffp_prof, pp_prof,
            matched_j_inductive, Ip_target, psi_pad,
            spike_prof=spike_profile, find_j0=False,
            scale_j0=final_scale_j0, tolerance=0.001,
            diagnostic_plots=diagnostic_plots,
        )

        # ---- 5d. Optional sawtooth constraint -----------------------
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
            ax.plot(psi_N, output_jphi, label=r"Output $j_\phi$")
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

            ax2 = ax.twinx()
            ax2.set_ylabel(r"$\sigma_{\rm exp}$ [A/m$^2$]", color="red")
            ax2.plot(psi_N, sigma_jphi, color="red", ls="--", alpha=0.5)
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(0.0, None)

            ax.legend(loc="best")
            ax.set_xlabel(r"$\hat{\psi}$")
            ax.set_ylabel(r"$j_\phi$ [A/m$^2$]")
            ax.set_title(f"jphi-linterp  |  l_i iter {li_iter}")
            plt.tight_layout()
            plt.show()

        eq_stats = mygs.get_stats(lcfs_pad=psi_pad)
        Ip = eq_stats["Ip"]
        l_i = eq_stats["l_i"]

        Ip_err = 100.0 * abs(Ip - Ip_target) / Ip_target
        print(f"  matched l_i:          {l_i:.4f}")
        print(f"  Ip error vs target:   {Ip_err:.3f}%")
        print(f"  l_i proxy rel. err:   {l_i_rel_err:.3f}%")
        print(f"  |l_i - l_i_target|:   {abs(l_i - l_i_target):.4f}")

        iteration_l_is.append(l_i)
        iteration_Ips.append(Ip)
    else:
        # Fired only if the for-loop exhausted without break
        print(
            f"WARNING: l_i match not achieved in {max_li_iter} iterations "
            f"(last |l_i - target| = {abs(l_i - l_i_target):.4f})"
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
def generate_perturbed_equilibria(
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
    l_i_proxy_threshold=1.0,
    psi_pad=1e-3,
    constrain_sawteeth=True,
    recalculate_j_BS=True,
    isolate_edge_jBS=True,
    jBS_scale_range=None,
    diagnostic_plots=True,
    baseline=None,
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
    j_ls : float
        GPR length-scale for :math:`j_\phi`.
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
        :math:`\pm 1\sigma` bounds for a normally distributed
        multiplicative scale factor applied to :math:`j_{\rm BS}`.
        For example, ``[0.9, 1.1]`` draws from
        :math:`\mathcal{N}(\mu{=}1.0,\, \sigma{=}0.1)` each
        iteration, centred on 1.0 with a half-width of 0.1.
        When ``None``, no additional scaling is applied
        (``scale_jBS = 1.0`` for every sample).
    diagnostic_plots : bool
        Show diagnostic matplotlib figures.
    baseline : str, float, int, or None
        Optional scan-point label for nested HDF5 storage.
        ``None`` gives the flat layout.

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
    if jBS_scale_range is not None:
        lo, hi = jBS_scale_range
        jBS_mu = 0.5 * (lo + hi)
        jBS_sigma = 0.5 * (hi - lo)
        jBS_scales = np.random.normal(jBS_mu, jBS_sigma, size=n_equils)
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
        baseline=baseline,
    )

    for count in range(n_equils):
        scale_jBS = float(jBS_scales[count])
        print(f"Perturber count: {count}  (scale_jBS={scale_jBS:.4f})")
        t_start = time.perf_counter()

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

        elapsed = time.perf_counter() - t_start
        print(f"  Wall-clock time: {elapsed:.1f} s")

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
            baseline=baseline,
            pressure=pressure_perturb,
            j_BS_edge=diagnostics["j_BS_edge"],
        )

        # Clean up on-disk eqdsk after archiving
        try:
            os.remove(full_path)
            print(f"  Deleted temporary file: {full_path}")
        except OSError as exc:
            print(f"  WARNING: could not delete {full_path}: {exc}")

        all_diagnostics.append(diagnostics)

    return all_diagnostics
