"""GPR profile perturbation routines
===================================

Gaussian-process-regression (GPR) based sampling of smooth 1-D MHD
profiles.  These routines are pure-Python / NumPy / SciPy and do
**not** require OpenFUSIONToolkit or TokaMaker.

Provides:
  - ``GPRProfilePerturber`` – GPR-based profile perturbation class.
  - ``generate_perturbed_GPR`` – convenience one-call wrapper.
  - ``verify_gpr_statistics`` – Monte-Carlo validation of GPR sampling.
  - ``calc_cylindrical_li_proxy`` – cylindrical :math:`l_i` proxy from
    a 1-D :math:`j_\\phi` profile.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import integrate
from scipy.stats import norm
from typing import Optional


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
        length_scale=0.1,
    ):
        if kernel_func not in self._ALLOWED_KERNELS:
            raise ValueError(
                f"Kernel '{kernel_func}' not in {self._ALLOWED_KERNELS}.  "
                "Rougher kernels produce non-differentiable profiles "
                "unsuitable for MHD inputs."
            )
        self.kernel_func = kernel_func

        # length_scale: scalar or 1-D array (non-stationary Gibbs kernel)
        ls = np.asarray(length_scale, dtype=np.float64)
        self._ls_is_array = ls.ndim >= 1 and ls.size > 1
        self.length_scale = ls

        self._kernel = {
            "rbf": self._rbf_kernel,
            "matern52": self._matern52_kernel,
        }[self.kernel_func]

    # ---- unit-variance kernels --------------------------------------
    def _ell_matrices(self, n1: int, n2: int):
        r"""Build per-pair length-scale matrices for non-stationary kernels.

        Returns ``(ell_i, ell_j)`` each of shape ``(n1, n2)`` such that
        ``ell_i[a, b] = ell[a]`` and ``ell_j[a, b] = ell[b]``.
        For a scalar length scale, returns ``(scalar, scalar)`` unchanged.
        """
        if not self._ls_is_array:
            return self.length_scale, self.length_scale
        ell = self.length_scale
        return ell[:n1, None] * np.ones((1, n2)), np.ones((n1, 1)) * ell[:n2]

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r"""Squared-exponential (Gibbs) kernel with unit variance.

        For a scalar length scale this is the standard stationary RBF.
        For a spatially-varying :math:`\ell(x)` it uses the Gibbs
        non-stationary kernel:

        .. math::

            K(x_i, x_j) = \sqrt{\frac{2\,\ell_i\,\ell_j}
                                      {\ell_i^2 + \ell_j^2}}
                           \exp\!\Bigl(-\frac{d_{ij}^2}
                                             {\ell_i^2 + \ell_j^2}\Bigr)

        which preserves :math:`K(x, x) = 1`.
        """
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "euclidean")
        ell_i, ell_j = self._ell_matrices(len(X1), len(X2))

        if not self._ls_is_array:
            return np.exp(-0.5 * (d / ell_i) ** 2)

        ell2_sum = ell_i**2 + ell_j**2
        prefactor = np.sqrt(2.0 * ell_i * ell_j / ell2_sum)
        return prefactor * np.exp(-d**2 / ell2_sum)

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r"""Matérn-5/2 (Gibbs) kernel with unit variance.

        Non-stationary extension: replaces the scalar :math:`\ell` with
        the geometric mean :math:`\bar{\ell} = \sqrt{\ell_i \ell_j}` and
        applies the same Gibbs prefactor as the RBF kernel.
        """
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "euclidean")
        ell_i, ell_j = self._ell_matrices(len(X1), len(X2))

        if not self._ls_is_array:
            s = np.sqrt(5.0) * d / ell_i
            return (1.0 + s + s**2 / 3.0) * np.exp(-s)

        ell2_sum = ell_i**2 + ell_j**2
        prefactor = np.sqrt(2.0 * ell_i * ell_j / ell2_sum)
        ell_geom = np.sqrt(ell_i * ell_j)
        s = np.sqrt(5.0) * d / ell_geom
        return prefactor * (1.0 + s + s**2 / 3.0) * np.exp(-s)

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
#  Spatially-varying length scale helpers
# ====================================================================
def sigmoid_length_scale(
    psi_N: np.ndarray,
    ls_core: float = 0.3,
    ls_edge: float = 0.1,
    psi_transition: float = 0.7,
    steepness: float = 20.0,
) -> np.ndarray:
    r"""Build a sigmoid length-scale profile over normalised flux.

    .. math::

        \ell(\hat{\psi}) = \ell_{\rm core}
            - \frac{\ell_{\rm core} - \ell_{\rm edge}}
                   {1 + \exp\!\bigl[-k\,(\hat{\psi} - \hat{\psi}_t)\bigr]}

    Parameters
    ----------
    psi_N : ndarray
        1-D normalised flux grid.
    ls_core : float
        Correlation length in the core (:math:`\hat{\psi} \ll \hat{\psi}_t`).
    ls_edge : float
        Correlation length at the edge (:math:`\hat{\psi} \gg \hat{\psi}_t`).
    psi_transition : float
        Centre of the sigmoid transition.
    steepness : float
        Steepness *k* of the sigmoid (larger = sharper transition).

    Returns
    -------
    ndarray
        1-D array of length scales, same shape as ``psi_N``.
    """
    return ls_core - (ls_core - ls_edge) / (
        1.0 + np.exp(-steepness * (psi_N - psi_transition))
    )


# ====================================================================
#  Convenience wrapper
# ====================================================================
def generate_perturbed_GPR(
    xdata: np.ndarray,
    profile: np.ndarray,
    sigma_profile: Optional[np.ndarray] = None,
    length_scale=0.25,
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
    length_scale : float or ndarray
        GPR correlation length (controls wiggliness).  A scalar gives
        a stationary kernel; a 1-D array (same length as *xdata*)
        gives a non-stationary Gibbs kernel with spatially-varying
        correlation length.  See :func:`sigmoid_length_scale`.
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


def get_li_proxy_geometry(mygs, n_psi, psi_pad):
    """Pre-compute geometry arrays for :func:`calc_cylindrical_li_proxy_fast`.

    Call this **once** before a loop of proxy evaluations where the
    equilibrium state does not change between iterations (i.e.\\ no
    ``mygs.solve()`` calls in between).  The returned dict should be
    passed to :func:`calc_cylindrical_li_proxy_fast`.

    Returns
    -------
    geo : dict
        Geometry cache with keys ``psi_N``, ``R_avg``, ``dV``,
        ``V_enc``, ``V_tot``, ``r_eff``, ``dA``.
    """
    psi_N, f, fp, p, pp = mygs.get_profiles(npsi=n_psi, psi_pad=psi_pad)
    _, qvals, ravgs_q, dl, rbounds, zbounds = mygs.get_q(npsi=n_psi, psi_pad=psi_pad)
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]

    R_avg = ravgs_q[0]
    dV_dPsi = ravgs_q[2]
    grad_psi_N = np.gradient(psi_N)
    d_psi_real = grad_psi_N * psi_range
    dV = dV_dPsi * d_psi_real
    V_enc = integrate.cumulative_trapezoid(dV, initial=0)
    V_tot = V_enc[-1]
    r_eff = np.sqrt(np.abs(V_enc) / (2 * np.pi**2 * R_avg))
    dA = dV / (2 * np.pi * R_avg)

    return {
        "psi_N": psi_N, "R_avg": R_avg, "dV": dV,
        "V_enc": V_enc, "V_tot": V_tot, "r_eff": r_eff, "dA": dA,
    }


def calc_cylindrical_li_proxy_fast(j_phi_profile, geo):
    """Same calculation as :func:`calc_cylindrical_li_proxy` but using
    pre-computed geometry (no TokaMaker calls).

    Parameters
    ----------
    j_phi_profile : array-like
        Toroidal current density profile.
    geo : dict
        From :func:`get_li_proxy_geometry`.

    Returns
    -------
    li_proxy : float
    """
    dA = geo["dA"]
    r_eff = geo["r_eff"]
    dV = geo["dV"]
    V_tot = geo["V_tot"]

    dI = j_phi_profile * dA
    I_enc = integrate.cumulative_trapezoid(dI, initial=0)
    I_tot = I_enc[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        B_p_proxy = I_enc / (2 * np.pi * r_eff)
    B_p_proxy[0] = 0.0

    B_p_sq = B_p_proxy**2
    W_pol_integral = integrate.trapezoid(B_p_sq * dV)
    L_edge = 2 * np.pi * r_eff[-1]

    if I_tot == 0:
        return 0.0

    B_p_edge_avg = I_tot / L_edge
    return W_pol_integral / (V_tot * B_p_edge_avg**2)


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
    length_scale : float or ndarray
        GPR correlation length (scalar or spatially-varying).
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