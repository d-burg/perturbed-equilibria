"""
Gaussian-process-regression (GPR) profile perturber
====================================================

Provides a class for generating smooth, spatially-correlated
perturbations to 1-D MHD profiles (pressure, density, temperature,
current density) on a normalised-flux grid, and a convenience
wrapper for single-call usage.

The primary user-facing input is ``sigma_profile`` — the
experimental 1σ uncertainty **in the same units as the profile**.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Optional


# ====================================================================
#  GPR profile perturber
# ====================================================================
class GPRProfilePerturber:
    r'''! Gaussian-process perturber for smooth 1-D MHD profiles.

    Generates correlated random perturbations whose pointwise
    standard deviation is set **directly** by the user-supplied
    \f$ \sigma(x) \f$ (experimental uncertainty in profile units).

    Internally the kernel amplitude is fixed to unity so that

    \f[
        \operatorname{Cov}\!\bigl[\delta f(x),\,\delta f(x')\bigr]
        \;=\;
        \sigma(x)\;\sigma(x')\;
        k_1\!\bigl(x,x'\bigr)
    \f]

    where \f$ k_1 \f$ is the unit-variance base kernel and the
    marginal standard deviation at every grid point equals the
    input uncertainty exactly:

    \f[
        \sigma_{\rm GP}(x) \;=\; \sigma(x)
    \f]

    @param kernel_func   Kernel name: ``'rbf'`` (\f$ C^\infty \f$) or
                         ``'matern52'`` (\f$ C^2 \f$).
    @param length_scale  Correlation length in \f$ \hat\psi \f$ units.
                         Controls *wiggliness* of the draws but does
                         **not** affect the pointwise amplitude.
    '''

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
        r'''! Squared-exponential kernel with unit variance.'''
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "euclidean")
        return np.exp(-0.5 * (d / self.length_scale) ** 2)

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r'''! Matérn-5/2 kernel with unit variance.'''
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
        r'''! Draw perturbed profiles whose pointwise 1\f$\sigma\f$
        matches the supplied experimental uncertainty exactly.

        @param psi_N          1-D normalised flux grid
        @param input_profile  1-D baseline profile (GP mean)
        @param sigma_profile  1-D experimental uncertainty **in profile
                              units** — this becomes the GP's marginal
                              standard deviation at every grid point
        @param n_samples      Number of independent draws
        @param rng            ``numpy.random.Generator``; ``None``
                              creates a fresh unseeded generator
        @result 2-D array of shape ``(n_samples, len(psi_N))``
        '''
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
    r'''! One-call wrapper: perturb a 1-D profile with a GPR draw.

    The ``sigma_profile`` input is the experimental 1\f$\sigma\f$
    uncertainty **in the same units as the profile**.  It maps
    directly to the GP marginal standard deviation — no separate
    ``variance`` parameter is needed.

    @param xdata          1-D normalised flux grid \f$ \hat\psi \f$
    @param profile        1-D baseline profile (GP mean)
    @param sigma_profile  1-D experimental 1\f$\sigma\f$ uncertainty
                          **in profile units**.  ``None`` → zero
                          (no perturbation).
    @param length_scale   GPR correlation length (controls wiggliness)
    @param n_samples      Number of independent draws
    @param kernel_func    ``'rbf'`` or ``'matern52'``
    @param rng            Optional ``numpy.random.Generator``
    @param diag_plot      Show a three-panel diagnostic figure
    @result If ``n_samples == 1``: 1-D array of length ``len(xdata)``.
            Otherwise: 2-D array ``(n_samples, len(xdata))``.
    '''
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

def verify_gpr_statistics(
    psi_N,
    profile,
    uncertainty_prof,
    length_scale=0.25,
    variance=0.1,
    n_verification=5000,
    confidence_band=2.0,
):
    r'''! Verify GPR sampling statistics against theoretical predictions.

    Draws a large number of samples and checks:
    1. Pointwise mean ≈ input profile  (bias check)
    2. Pointwise std  ≈ u(x)·√variance (variance check)
    3. Fraction of samples outside ±kσ  ≈ theoretical  (tail check)

    @param psi_N             Normalised flux grid
    @param profile           Baseline profile (GP mean)
    @param uncertainty_prof  Uncertainty envelope u(x)
    @param length_scale      GPR length-scale
    @param variance          GPR kernel amplitude
    @param n_verification    Number of Monte-Carlo draws
    @param confidence_band   Number of σ for the band (e.g. 2.0)
    @result dict with verification diagnostics
    '''
    perturber = GPRProfilePerturber(
        kernel_func="rbf",
        length_scale=length_scale,
        variance=variance,
    )

    rng = np.random.default_rng(42)
    samples = perturber.generate_profiles(
        psi_N, profile, uncertainty_prof,
        n_samples=n_verification, rng=rng,
    )                                          # (n_verification, n_points)

    # ---- theoretical predictions ------------------------------------
    sigma_theory = perturber.marginal_std(uncertainty_prof)

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
    from scipy.stats import norm
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

"""
Perturbed Grad-Shafranov equilibrium workflow
==============================================

Functions for generating uncertainty envelopes, perturbing kinetic
and current-density profiles via GPR sampling, and iterating to
match target Ip and l_i within tolerance.
"""

import io
import os
import time
import tempfile

import numpy as np
import matplotlib.pyplot as plt


# ── physical constant used for pressure ────────────────────────────
EC = 1.6022e-19  # [J/eV]

# ── default iteration caps (safety valves) ─────────────────────────
_MAX_PRESSURE_ITER = int(1e5)
_MAX_LI_ITER = 20
_MAX_MONOTONIC_DRAWS = int(1e4)

import h5py
import os

from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS

def calc_cylindrical_li_proxy(j_phi_profile, psi_pad):
    """
    Calculates a proxy for internal inductance l_i(3) using 1D profiles.
    
    Parameters:
    -----------
    j_phi_profile : array-like
        The toroidal current density profile (perturbation target).
    psi_N : array-like
        Normalized poloidal flux (0 to 1).
    ravgs_q : list of arrays
        Result from get_q: [<R>, <1/R>, dV/dPsi].
    psi_range : float
        The scalar difference (psi_boundary - psi_axis) to un-normalize gradients.
        
    Returns:
    --------
    li_proxy : float
        The estimated internal inductance.
    """
    from scipy import integrate
    
    n_psi = len(j_phi_profile)

    psi_N,f,fp,p,pp = mygs.get_profiles(npsi=n_psi, psi_pad=psi_pad)
    _,qvals,ravgs_q,dl,rbounds,zbounds = mygs.get_q(npsi=n_psi, psi_pad=psi_pad)
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0] # psi bounds used to un-normalize psi_N
    
    # 1. Unpack Geometry from baseline
    R_avg = ravgs_q[0]  
    dV_dPsi = ravgs_q[2]
    
    # 2. Calculate differentials
    # d_psi_real corresponds to d_psi in the actual equilibrium
    grad_psi_N = np.gradient(psi_N)
    d_psi_real = grad_psi_N * psi_range
    
    # dV is the differential volume element
    dV = dV_dPsi * d_psi_real
    
    # 3. Calculate Geometry Mappings
    # FIX: cumtrapz -> cumulative_trapezoid
    V_enc = integrate.cumulative_trapezoid(dV, initial=0)
    V_tot = V_enc[-1]
    
    # Effective minor radius r_eff
    r_eff = np.sqrt(np.abs(V_enc) / (2 * np.pi**2 * R_avg))
    
    # Differential Poloidal Area dA
    dA = dV / (2 * np.pi * R_avg)
    
    # 4. Calculate Enclosed Current I(rho)
    dI = j_phi_profile * dA
    # FIX: cumtrapz -> cumulative_trapezoid
    I_enc = integrate.cumulative_trapezoid(dI, initial=0)
    I_tot = I_enc[-1]
    
    # 5. Calculate Cylindrical Poloidal Field Proxy B_p(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        B_p_proxy = I_enc / (2 * np.pi * r_eff)
    
    # Handle core singularity (limit r->0)
    # If J(0) is finite, B_p is linear in r, so B_p(0) = 0.
    B_p_proxy[0] = 0.0 
    
    # 6. Integrate Magnetic Energy
    B_p_sq = B_p_proxy**2
    
    # FIX: np.trapz is also deprecated in NumPy 2.0 / SciPy 1.14
    # Using scipy.integrate.trapezoid instead
    W_pol_integral = integrate.trapezoid(B_p_sq * dV)
    
    # 7. Calculate l_i
    L_edge = 2 * np.pi * r_eff[-1]
    
    # Avoid div/0 if total current is 0
    if I_tot == 0:
        return 0.0
        
    B_p_edge_avg = I_tot / L_edge
    
   # li_proxy = (2 * W_pol_integral) / (V_tot * B_p_edge_avg**2)
    li_proxy = W_pol_integral / (V_tot * B_p_edge_avg**2)

    return li_proxy

# ====================================================================
#  Uncertainty envelope builder. 
#  This needs to be multiplied against your profile of choice to give σ(ψ_N)
# ====================================================================
def new_uncertainty_profiles(
    psi_N,
    uncertainty,
    falloff_exp=None,
    edge_val=0.0,
    falloff_loc=0.8,
    tail_alpha=2.5,
):
    r'''! Build a 1-D uncertainty envelope over normalised flux.

    Two modes are supported:

    * **Power-law mode** (`falloff_exp` is not ``None``):
      \f$ u(\hat{\psi}) = U\,(1 - \hat{\psi})^{\mathrm{falloff\_exp}} \f$

    * **Flat + tail mode** (default, `falloff_exp` is ``None``):
      constant value \f$ U \f$ for \f$ \hat{\psi} \le \hat{\psi}_{\rm loc} \f$,
      then a cosine (or cosh) decay to `edge_val` at \f$ \hat{\psi}=1 \f$
      controlled by `tail_alpha`.

    @param psi_N        1-D array of normalised poloidal flux \f$ \hat{\psi} \f$
    @param uncertainty   Scalar amplitude \f$ U \f$ of the envelope
    @param falloff_exp   Exponent for the power-law branch (``None`` ⟹ flat + tail)
    @param edge_val      Envelope value at \f$ \hat{\psi}=1 \f$ (flat + tail mode)
    @param falloff_loc   \f$ \hat{\psi}_{\rm loc} \f$ where the tail begins
    @param tail_alpha    Sharpness exponent \f$ \alpha \f$ of the cosine/cosh tail
    @result 1-D ``ndarray`` of the same length as `psi_N`
    '''
    # ---- power-law branch -------------------------------------------
    if falloff_exp is not None:
        return uncertainty * (1.0 - psi_N) ** falloff_exp

    # ---- flat + cosine/cosh tail branch -----------------------------
    profile_left = uncertainty * np.ones_like(psi_N)
    stitch_height = uncertainty                        # value at falloff_loc

    dist_to_edge = max(1.0 - falloff_loc, 1e-5)       # avoid division by zero

    if edge_val < stitch_height:
        # --- cosine decay ---
        target_cos = np.clip(
            (edge_val / stitch_height) ** (1.0 / tail_alpha), -1.0, 1.0
        )
        omega = np.arccos(target_cos) / dist_to_edge

        base_cos = np.maximum(np.cos(omega * (psi_N - falloff_loc)), 0.0)
        profile_right = stitch_height * (base_cos ** tail_alpha)
    else:
        # --- cosh rise (rare) ---
        target_cosh = (edge_val / stitch_height) ** (1.0 / tail_alpha)
        omega = np.arccosh(target_cosh) / dist_to_edge
        profile_right = stitch_height * (
            np.cosh(omega * (psi_N - falloff_loc)) ** tail_alpha
        )

    profile = np.where(psi_N <= falloff_loc, profile_left, profile_right)

    if edge_val >= 0.0:
        profile = np.maximum(profile, 0.0)

    return profile


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
    r'''! Repeatedly sample a GPR perturbation until the draw is
    monotonically decreasing.

    @param psi_N              Normalised flux grid
    @param normalised_profile Profile divided by its on-axis value
    @param sigma_profile      1\f$\sigma\f$ uncertainty in normalised-
                              profile units (same grid)
    @param length_scale       GPR correlation length
    @param max_draws          Safety cap on the number of attempts
    @result 1-D ``ndarray`` – the accepted (still normalised) perturbation

    @warning Raises ``RuntimeError`` if no monotonic draw is found
             within `max_draws` attempts.
    '''
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
    diagnostic_plots=False,
    max_pressure_iter=_MAX_PRESSURE_ITER,
    max_li_iter=_MAX_LI_ITER,
):
    r'''! Perturb kinetic and current-density profiles and iterate to
    match \f$ I_p \f$ and \f$ l_i \f$ targets.

    @param mygs               TokaMaker Grad-Shafranov solver object
    @param psi_N              1-D normalised poloidal flux grid \f$ \hat{\psi} \f$
    @param pressure           1-D baseline total pressure [Pa]
    @param ne                 1-D electron density [m\f$^{-3}\f$]
    @param te                 1-D electron temperature [eV]
    @param ni                 1-D ion density [m\f$^{-3}\f$]
    @param ti                 1-D ion temperature [eV]
    @param input_j_phi        1-D toroidal current density [A/m\f$^2\f$];
                              must be the *inductive* component when
                              ``recalculate_j_BS=True``
    @param sigma_ne           1-D experimental 1\f$\sigma\f$ for \f$ n_e \f$ [m\f$^{-3}\f$]
    @param sigma_te           1-D experimental 1\f$\sigma\f$ for \f$ T_e \f$ [eV]
    @param sigma_ni           1-D experimental 1\f$\sigma\f$ for \f$ n_i \f$ [m\f$^{-3}\f$]
    @param sigma_ti           1-D experimental 1\f$\sigma\f$ for \f$ T_i \f$ [eV]
    @param sigma_jphi         1-D experimental 1\f$\sigma\f$ for \f$ j_\phi \f$ [A/m\f$^2\f$]
    @param n_ls               GPR length-scale for density profiles
    @param t_ls               GPR length-scale for temperature profiles
    @param j_ls               GPR length-scale for \f$ j_\phi \f$
    @param Ip_target          Target plasma current [A]
    @param l_i_target         Target internal inductance
    @param Zeff               Effective ion charge (scalar)
    @param npsi               1-D normalised poloidal flux grid size
    @param p_thresh           Acceptable \f$ \langle P \rangle \f$ mismatch [%]
    @param input_jinductive   Dimensionless inductive \f$ j_\phi \f$ shape
                              (required when ``recalculate_j_BS=True``)
    @param l_i_tolerance      Absolute \f$ l_i \f$ matching tolerance
    @param l_i_proxy_threshold Proxy \f$ l_i \f$ relative error threshold [%]
    @param psi_pad            Padding inside the LCFS for profile queries
    @param constrain_sawteeth Reject equilibria with \f$ q_0 < 1 \f$
    @param recalculate_j_BS   Recompute bootstrap current for perturbed profiles
    @param diagnostic_plots   Show diagnostic matplotlib figures
    @param max_pressure_iter  Safety cap on pressure-matching loop
    @param max_li_iter        Safety cap on \f$ l_i \f$-matching loop
    @result Tuple ``(ne_perturb, te_perturb, ni_perturb, ti_perturb,
            w_ExB, output_jphi, diagnostics)``
    '''

    # ----------------------------------------------------------------
    #  0.  Validate inputs
    # ----------------------------------------------------------------
    if recalculate_j_BS and input_jinductive is None:
        raise ValueError(
            "input_jinductive must be provided when recalculate_j_BS=True"
        )

    # ----------------------------------------------------------------
    #  1.  Perturb kinetic profiles to match <P>
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
    #  1b.  Optional diagnostic plots for kinetic profiles
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
    #  2.  Bootstrap-current recalculation (optional)
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
            scale_jBS=1.0,
            isolate_edge_jBS=True,
            diagnostic_plots=False,
        )
        eq_stats = mygs.get_stats(lcfs_pad=psi_pad)

        new_jphi = results["total_j_phi"]
        spike_profile = results["isolated_j_BS"]
        baseline_li_proxy = calc_cylindrical_li_proxy(new_jphi, psi_pad)

        j0_scales.append(results["scale_j0"])
        Ip_scales.append(results["scale_Ip"])
        iteration_l_is.append(eq_stats["l_i"])
        iteration_Ips.append(eq_stats["Ip"])
    else:
        # When bootstrap is not recalculated there is no edge spike
        spike_profile = np.zeros_like(psi_N)
        baseline_li_proxy = calc_cylindrical_li_proxy(input_j_phi, psi_pad)

    # ----------------------------------------------------------------
    #  3.  l_i matching loop
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
    
        # ---- 3a. Draw j_phi perturbation matching l_i proxy --------
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
                args=(jphi_perturb, spike_profile, psi_N, Ip_target),
                bracket=[1.0e-10 * Ip_target, 1.0e1 * Ip_target],
                method="brentq",
                rtol=1e-6,
            )
            a_optimal = result_root.root
            matched_jphi_perturb = a_optimal * jphi_perturb + spike_profile

            tmp_li_proxy = calc_cylindrical_li_proxy(matched_jphi_perturb, psi_pad)
            l_i_rel_err = (
                100.0 * abs(tmp_li_proxy - baseline_li_proxy) / baseline_li_proxy
            )
        print("Found potential l_i match via proxy!\n")

        # ---- 3b. Set up GS profiles --------------------------------
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

        # ---- 3c. Find optimal scale factors -------------------------
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

        # ---- 3d. Optional sawtooth constraint -----------------------
        if constrain_sawteeth:
            _, q, _, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
            if q[0] < 1.0:
                print("Skipping this equilibrium, q_0 < 1.0")
                l_i = np.inf
                continue

        j0_scales.append(final_scale_j0)
        Ip_scales.append(final_scale_Ip)

        # ---- 3e. Final GS solves (2 iterations for convergence) -----
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

        # ---- 3f. Evaluate converged equilibrium ---------------------
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
    #  4.  Package outputs
    # ----------------------------------------------------------------
    w_ExB = np.zeros_like(psi_N)  # placeholder – not yet computed

    diagnostics = {
        "j0_scales": j0_scales,
        "Ip_scales": Ip_scales,
        "iteration_l_is": iteration_l_is,
        "iteration_Ips": iteration_Ips,
        "j_inductive": matched_j_inductive * final_scale_j0,
        "j_BS": spike_profile,
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
    diagnostic_plots=True,
    baseline=None,
):
    r'''! Generate a batch of perturbed equilibria and archive to HDF5.

    @param mygs                TokaMaker GS solver object
    @param psi_N               1-D normalised flux grid \f$ \hat{\psi} \f$
    @param n_equils            Number of perturbed equilibria to generate
    @param header              Base name for the HDF5 database
    @param input_j_phi         1-D baseline toroidal current density [A/m\f$^2\f$]
    @param ne                  1-D electron density [m\f$^{-3}\f$]
    @param te                  1-D electron temperature [eV]
    @param ni                  1-D ion density [m\f$^{-3}\f$]
    @param ti                  1-D ion temperature [eV]
    @param sigma_ne            1-D experimental 1\f$\sigma\f$ for \f$ n_e \f$ [m\f$^{-3}\f$]
    @param sigma_te            1-D experimental 1\f$\sigma\f$ for \f$ T_e \f$ [eV]
    @param sigma_ni            1-D experimental 1\f$\sigma\f$ for \f$ n_i \f$ [m\f$^{-3}\f$]
    @param sigma_ti            1-D experimental 1\f$\sigma\f$ for \f$ T_i \f$ [eV]
    @param sigma_jphi          1-D experimental 1\f$\sigma\f$ for \f$ j_\phi \f$ [A/m\f$^2\f$]
    @param n_ls                GPR length-scale for density profiles
    @param t_ls                GPR length-scale for temperature profiles
    @param j_ls                GPR length-scale for \f$ j_\phi \f$
    @param initial_Ip_target   Target plasma current [A]
    @param l_i_target          Target internal inductance
    @param Zeff                Effective ion charge
    @param input_jinductive    Dimensionless inductive \f$ j_\phi \f$ shape
    @param l_i_tolerance       Absolute \f$ l_i \f$ tolerance
    @param l_i_proxy_threshold Proxy \f$ l_i \f$ relative-error threshold [%]
    @param psi_pad             LCFS padding for profile queries
    @param constrain_sawteeth  Reject equilibria with \f$ q_0 < 1 \f$
    @param recalculate_j_BS    Recompute bootstrap current each iteration
    @param diagnostic_plots    Show diagnostic matplotlib figures
    @param baseline            Optional scan-point index for nested HDF5 storage
    @result ``list[dict]`` – diagnostics from each equilibrium
    '''
    all_diagnostics = []
    
    # self-consistent pressure for baseline <P>
    pressure = EC * (ne * te + ni * ti)
    npsi = len(pressure)

    for count in range(n_equils):
        print(f"Perturber count: {count}")
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
        )

        # Clean up on-disk eqdsk after archiving
        try:
            os.remove(full_path)
            print(f"  Deleted temporary file: {full_path}")
        except OSError as exc:
            print(f"  WARNING: could not delete {full_path}: {exc}")

        all_diagnostics.append(diagnostics)

    return all_diagnostics

def initialize_equilibrium_database(header):
    """
    Create (or open) the top-level HDF5 database file on disk.

    Parameters
    ----------
    header : str
        Base name for the database.  File will be ``<header>.h5``.

    Returns
    -------
    db_path : str
        Absolute path to the HDF5 file.
    """
    db_path = os.path.abspath(f"{header}.h5")
    with h5py.File(db_path, "a"):
        pass
    return db_path


def _group_path(baseline, count):
    """Return the internal HDF5 group path for a given entry."""
    if baseline is not None:
        return f"baseline_{int(baseline):03d}/{int(count)}"
    return str(int(count))


def _eqdsk_dataset_name(header, baseline, count):
    """Return the dataset name used for the raw eqdsk bytes."""
    if baseline is not None:
        return f"{header}_{int(baseline):03d}_{int(count)}.eqdsk"
    return f"{header}_{int(count)}.eqdsk"


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
    baseline=None,
):
    """
    Write one perturbed equilibrium into the HDF5 database.

    Parameters
    ----------
    header : str
        Base name (same string passed to ``initialize_equilibrium_database``).
    count : int
        Perturbation index (typically 0 – 15).
    eqdsk_filepath : str
        Path to the ``.geqdsk`` / ``.eqdsk`` file.  Read as raw bytes so
        the Fortran-namelist formatting is preserved exactly.
    psi_N, j_phi, j_BS, j_inductive,
    n_e, T_e, n_i, T_i, w_ExB : array_like, 1-D
        Profile arrays (see docstring of previous version for units).
    li1 : float
        Internal inductance l_i(1).
    li3 : float
        Internal inductance l_i(3).
    baseline : int or None, optional
        Scan-point index.  When provided an extra ``baseline_XXX/``
        group layer is inserted so that many baselines coexist inside
        one HDF5 file.  ``None`` (default) gives the flat layout.
    """
    db_path = os.path.abspath(f"{header}.h5")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Database '{db_path}' not found.  "
            f"Call initialize_equilibrium_database('{header}') first."
        )

    with open(eqdsk_filepath, "rb") as fh:
        eqdsk_bytes = fh.read()

    grp_path = _group_path(baseline, count)
    ds_name  = _eqdsk_dataset_name(header, baseline, count)

    with h5py.File(db_path, "a") as hf:
        # clean slate if this entry already exists
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        # ---- raw eqdsk (opaque binary – bit-perfect) --------------------
        grp.create_dataset(ds_name, data=np.void(eqdsk_bytes))

        # ---- 1-D profiles -----------------------------------------------
        grp.create_dataset("psi_N",               data=np.asarray(psi_N,       dtype=np.float64))
        grp.create_dataset("j_phi [A/m^2]",       data=np.asarray(j_phi,       dtype=np.float64))
        grp.create_dataset("j_BS [A/m^2]",        data=np.asarray(j_BS,        dtype=np.float64))
        grp.create_dataset("j_inductive [A/m^2]", data=np.asarray(j_inductive, dtype=np.float64))
        grp.create_dataset("n_e [m^-3]",          data=np.asarray(n_e,         dtype=np.float64))
        grp.create_dataset("T_e [eV]",            data=np.asarray(T_e,         dtype=np.float64))
        grp.create_dataset("n_i [m^-3]",          data=np.asarray(n_i,         dtype=np.float64))
        grp.create_dataset("T_i [eV]",            data=np.asarray(T_i,         dtype=np.float64))
        grp.create_dataset("w_ExB [rad/s]",       data=np.asarray(w_ExB,       dtype=np.float64))

        # ---- scalars (group attributes) ----------------------------------
        grp.attrs["l_i(1)"]   = float(li1)
        grp.attrs["l_i(3)"]   = float(li3)
        # bookkeeping – handy when browsing with h5dump / HDFView
        if baseline is not None:
            grp.attrs["baseline"] = int(baseline)
        grp.attrs["count"] = int(count)


def load_equilibrium(header, count, baseline=None, eqdsk_out_dir=None):
    """
    Retrieve one equilibrium entry from the HDF5 database.

    Parameters
    ----------
    header : str
        Base name of the database.
    count : int
        Perturbation index.
    baseline : int or None, optional
        Scan-point index (must match what was used at write time).
    eqdsk_out_dir : str or None, optional
        If given, the raw eqdsk is written to a file in this directory.

    Returns
    -------
    result : dict
        Keys: ``"eqdsk_filepath"``, ``"eqdsk_bytes"``,
        the nine 1-D array names, ``"l_i(1)"``, ``"l_i(3)"``.
    """
    db_path  = os.path.abspath(f"{header}.h5")
    grp_path = _group_path(baseline, count)
    ds_name  = _eqdsk_dataset_name(header, baseline, count)

    array_keys = [
        "psi_N",
        "j_phi [A/m^2]",
        "j_BS [A/m^2]",
        "j_inductive [A/m^2]",
        "n_e [m^-3]",
        "T_e [eV]",
        "n_i [m^-3]",
        "T_i [eV]",
        "w_ExB [rad/s]",
    ]

    result = {}

    with h5py.File(db_path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {db_path}"
            )
        grp = hf[grp_path]

        # ---- eqdsk raw bytes -------------------------------------------
        eqdsk_bytes = bytes(grp[ds_name][()])
        result["eqdsk_bytes"] = eqdsk_bytes

        if eqdsk_out_dir is not None:
            os.makedirs(eqdsk_out_dir, exist_ok=True)
            out_path = os.path.join(eqdsk_out_dir, ds_name)
            with open(out_path, "wb") as fh:
                fh.write(eqdsk_bytes)
            result["eqdsk_filepath"] = os.path.abspath(out_path)
        else:
            result["eqdsk_filepath"] = None

        # ---- 1-D arrays ------------------------------------------------
        for key in array_keys:
            result[key] = np.array(grp[key])

        # ---- scalars ----------------------------------------------------
        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

    return result