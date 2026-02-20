
import numpy as np

# ====================================================================
#  Uncertainty envelope builder.
#  Multiply against your profile of choice to give σ(ψ_N).
# ====================================================================
def new_uncertainty_profiles(
    psi_N,
    uncertainty,
    falloff_exp=None,
    edge_val=0.0,
    falloff_loc=0.8,
    tail_alpha=2.5,
):
    r"""Build a 1-D uncertainty envelope over normalised flux.

    Two modes are supported:

    * **Power-law mode** (``falloff_exp`` is not ``None``):
      :math:`u(\hat{\psi}) = U\,(1 - \hat{\psi})^{\mathrm{falloff\_exp}}`

    * **Flat + tail mode** (default, ``falloff_exp`` is ``None``):
      constant value :math:`U` for
      :math:`\hat{\psi} \le \hat{\psi}_{\rm loc}`, then a cosine
      (or cosh) decay to ``edge_val`` at :math:`\hat{\psi}=1`
      controlled by ``tail_alpha``.

    Parameters
    ----------
    psi_N : ndarray
        1-D array of normalised poloidal flux :math:`\hat{\psi}`.
    uncertainty : float
        Scalar amplitude :math:`U` of the envelope.
    falloff_exp : float or None
        Exponent for the power-law branch (``None`` selects flat + tail).
    edge_val : float
        Envelope value at :math:`\hat{\psi}=1` (flat + tail mode).
    falloff_loc : float
        :math:`\hat{\psi}_{\rm loc}` where the tail begins.
    tail_alpha : float
        Sharpness exponent :math:`\alpha` of the cosine/cosh tail.

    Returns
    -------
    ndarray
        1-D array of the same length as ``psi_N``.
    """
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
