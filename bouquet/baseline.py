"""The :class:`Baseline` -- the common product every source resolves to.

``generate()`` consumes a :class:`Baseline` and never depends on reconstruction
directly. Both :class:`~bouquet.config.ReconstructionSource` and
:class:`~bouquet.config.ImasSource` resolve to this same structure, so adding a
new input pipeline means adding a resolver, not touching generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .config import BaselineSource, SolverConfig


@dataclass
class Baseline:
    """Separated baseline currents + targets + kinetic profiles.

    The currents are split into ohmic + bootstrap + driven (j_phi =
    j_inductive + j_BS + j_NBI + j_RF) regardless of provenance: the
    reconstruction source *produces* the split, the IMAS source *reads* it
    pre-separated.

    CURRENT CONVENTION: every current component here is a flux-surface-averaged
    *toroidal* current density j_phi [A/m^2]. IMAS/neoclassical inputs are
    parallel (<j.B>/B0) and are converted on read via
    :func:`bouquet.physics.parallel_to_toroidal`, so downstream code never mixes
    conventions.
    """

    # --- required fields (no defaults) ---------------------------------
    # current-density grid + separated currents
    psi_N: "np.ndarray"
    j_phi: "np.ndarray"            # total [A/m^2] = j_inductive + j_BS + j_NBI + j_RF
    j_inductive: "np.ndarray"     # ohmic part [A/m^2]   (perturbed via l_i matching)
    j_BS: "np.ndarray"            # bootstrap part [A/m^2] (recomputed per draw)

    # baseline kinetic profiles (SI: m^-3, eV) on `psi_N_kinetic`
    psi_N_kinetic: "np.ndarray"
    ne: "np.ndarray"
    te: "np.ndarray"
    ni: "np.ndarray"
    ti: "np.ndarray"
    Zeff: "np.ndarray"

    # targets held fixed across all perturbed draws
    Ip_target: float
    l_i_target: float

    provenance: str               # "reconstruction" | "imas"

    # --- optional / defaulted fields -----------------------------------
    # Which l_i estimator `l_i_target` (and every downstream l_i comparison --
    # the per-draw acceptance band, the archive `l_i_target` attr, plot_traces)
    # is expressed on.  "iter(li3)" == TokaMaker li_normalization='iter'
    # == 2*int(Bp^2 dV)/((mu0 Ip)^2 R_axis).  Carried explicitly so an archive
    # is self-describing and a future scale change cannot pass silently
    # (issue #20).  Single source of truth: bouquet.utils.LI_SCALE.
    l_i_scale: str = "iter(li3)"    # == utils.LI_SCALE
    # Fixed additive components -- summed into EVERY draw, never GPR-perturbed.
    # None is treated as zeros. See FixedComponentsConfig for the contract:
    #   j_phi_total = j_inductive + j_BS + j_NBI + j_RF
    #   p_total     = p_thermal(perturbed) + p_fast
    j_NBI: Optional["np.ndarray"] = None    # beam-driven current [A/m^2]
    j_RF: Optional["np.ndarray"] = None     # RF-driven current [A/m^2]
    p_fast: Optional["np.ndarray"] = None   # fast/beam pressure

    # bootstrap amplitude factor applied when the j_BS/j_inductive split is
    # rebuilt against SWB and calibrated to the measured l_i (IMAS path's
    # _forward_solve_imas_baseline, mirroring the g-file fit_inductive_profile).
    # 1.0 means no rebuild was done (raw source split kept).
    bs_scale: float = 1.0
    # 'ohmic' mode only: factor applied to FUSE j_inductive so the hybrid
    # (s*j_ohm + SWB_jBS + j_fixed) integrates to Ip_target. 1.0 otherwise.
    ohm_scale: float = 1.0
    # 'ohmic' mode bookkeeping: proxy-Ip of each component, the proxy's own
    # error on the FUSE total, and the jphi_diff anchor that was NOT applied.
    ip_closure: Optional[dict] = None

    # IMAS path only: the two slice-level facts closure_channel=
    # "sawtooth_bootstrap" gates on, read ONCE at load time because the reader
    # does not retain the (100s of MB) dd -- the source's sawtooth model
    # amplitude at this slice (core_sources source with identifier index 701)
    # and the dd's OWN on-axis q from equilibrium.profiles_1d.q.  The latter is
    # recorded for comparison only: the closure's reference is TokaMaker's q0
    # for the source total re-solved on the anchor, not the dd's own estimator
    # (issue #20 -- never compare two estimators of the same name).
    # Keys: source_index, present, j_par_max_abs, active, q0_dd.
    sawtooth: Optional[dict] = None

    # Case-B ("diff") fixed bootstrap correction profile [A/m^2] = FUSE_jBS - SWB,
    # added to the baseline AND every draw's j_phi so the total anchors to the
    # FUSE bootstrap while the SWB delta tracks per-draw kinetics. None => not in
    # diff mode (Case-A "rescale" or no SWB rebuild).
    jBS_diff: Optional["np.ndarray"] = None

    # IMAS pressure anchor ("diff" approach, mirroring jBS_diff). p_equilibrium
    # is the authoritative dd equilibrium.pressure on psi_N; p_diff =
    # p_equilibrium - reconstructed baseline pressure (e*(ne*Te + ni*Ti) +
    # impurity + p_fast). p_diff is added to the baseline AND every draw so the
    # solve pressure anchors to FUSE exactly while the reconstructed thermal
    # delta tracks per-draw kinetics. Z_imp is the single effective impurity
    # charge (one-Zeff single-impurity model) used to recover nz = (ne-ni)/Z_imp
    # for the impurity (carbon) pressure term. All None on the geqdsk path
    # (p-file pressure is already the total).
    p_equilibrium: Optional["np.ndarray"] = None
    p_diff: Optional["np.ndarray"] = None
    Z_imp: Optional[float] = None

    # IMAS total-current anchor: jphi_diff = equilibrium.profiles_1d.j_tor
    # (the GS-consistent current GPEC reads, with the pedestal current) minus the
    # reconstructed core_profiles.j_tor total. A FIXED offset added to the baseline
    # AND every draw so the total anchors to the equilibrium while the SWB bootstrap
    # + perturbed j_inductive ride underneath. None on the geqdsk path / when
    # GenerationConfig.anchor_jtor_to_equilibrium is False.
    jphi_diff: Optional["np.ndarray"] = None

    # raw bytes preserved for archival into the HDF5 (optional)
    eqdsk_bytes: Optional[bytes] = None
    pfile_bytes: Optional[bytes] = None

    # full reconstruction diagnostics, when provenance == "reconstruction"
    recon: Optional[dict] = None

    # l_i sanity metrics (IMAS path): IDS-reported li_1/li_3 vs the
    # TokaMaker-solved li_1/li_3 from the forward-solve. l_i_target is set to
    # the TokaMaker li_1; the IDS values are kept here for comparison/plots.
    li_metrics: Optional[dict] = None

    # auxiliary source-provided profiles available to the perturbation
    # switchboard -- rotation ('omega_tor', 'e_r'), transport ('chi_e',
    # 'chi_i') and impurity ('zeff') channels --
    # on psi_N_kinetic. Populated by the reader where the source has them; the
    # user may add/override via UncertaintyConfig.aux_baselines.
    aux: Optional[dict] = None

    # curated reconstruction quality metrics (reconstruction path only): a flat
    # dict consumed by Bouquet's reconstruction summary -- Ip/l_i target-vs-
    # achieved, boundary RMS/max, q0/q95, shape, and a pass/fail verdict.
    reconstruction_metrics: Optional[dict] = None
    # full captured solver chatter from the reconstruction (when verbose=False),
    # kept available for debugging without cluttering the notebook output.
    reconstruction_log: Optional[str] = None

    def __repr__(self):
        # concise summary -- the default dataclass repr dumps every numpy array,
        # which floods a notebook when `reconstruct()`/`prepare_baseline()` is the
        # last expression in a cell.
        import numpy as np
        ng = len(np.atleast_1d(self.psi_N))
        nk = len(np.atleast_1d(self.psi_N_kinetic))
        aux = list((self.aux or {}).keys())
        return (f"Baseline(provenance={self.provenance!r}, "
                f"Ip={self.Ip_target/1e6:.3f} MA, l_i={self.l_i_target:.3f}, "
                f"grids: {ng} eq / {nk} kinetic"
                + (f", aux={aux}" if aux else "") + ")")


def resolve_baseline(config: "BouquetConfig", mygs=None) -> Baseline:
    """Dispatch on ``config.source`` and return a populated :class:`Baseline`.

    * :class:`ImasSource` -> delegate to
      :func:`bouquet.io.imas.read_imas_baseline`: take j_ohmic / j_BS / Ip / l_i
      and core_profiles directly (no GS reconstruction), convert parallel->
      toroidal, isotropize p_fast. ``mygs`` is unused.
    * :class:`ReconstructionSource` -> read g-file + profiles (IDA .cdf or
      p-file), run :func:`reconstruct_equilibrium` on ``mygs``, package the
      fitted split currents (converted to toroidal) plus any user-supplied
      fixed components. (Implemented in phase 2 step 3.)

    Both paths return a :class:`Baseline` with every current as toroidal j_phi
    and the fixed additive components (j_NBI, j_RF, p_fast) attached.

    Implemented as a free function so sources stay declarative (plain config)
    and the resolution logic lives in one place.
    """
    from .config import ImasSource, ReconstructionSource

    source = config.source

    if isinstance(source, ImasSource):
        from .io.imas import read_imas_baseline
        return read_imas_baseline(
            source,
            fixed=config.fixed_components,
            p_fast_reduction=config.fixed_components.p_fast_reduction,
            allow_incomplete_pressure=config.generation.allow_incomplete_pressure,
            anchor_jtor_to_equilibrium=config.generation.anchor_jtor_to_equilibrium,
            kinetic_source=config.generation.kinetic_source,
            anchor_pressure_to_equilibrium=config.generation.anchor_pressure_to_equilibrium,
        )

    if isinstance(source, ReconstructionSource):
        return _resolve_reconstruction(source, config, mygs)

    raise TypeError(f"unknown baseline source type: {type(source).__name__}")


def resolve_uncertainty(config, baseline) -> dict:
    """Resolve the perturbation envelope for :func:`generate_bouquet`.

    Returns kinetic sigmas (on ``baseline.psi_N_kinetic``), ``sigma_jphi`` (a
    fractional envelope on ``|baseline.j_phi|``, on ``baseline.psi_N``), and the
    GPR correlation length scales.

    Kinetic sigma source precedence:
      1. ``UncertaintyConfig.ida_path`` if set;
      2. else the reconstruction source's own IDA ``.cdf`` (sigmas read once);
      3. else a per-channel flat fraction ``<chan>_scalar_sigma * |profile|``.
    An explicit ``UncertaintyConfig.sigma_profiles[chan]`` array overrides all.

    The resolved source is LOGGED, one line per channel (disable with
    ``UncertaintyConfig.log_sigma_sources=False``), and a ``UserWarning`` is
    raised when a ``<chan>_scalar_sigma`` was moved off its default but loses
    the precedence to an IDA file -- most importantly the case where a user
    zeroes the scalars intending a sigma=0 run and silently gets the full
    operational IDA envelope instead.  See :class:`UncertaintyConfig` for the
    full contract and the way to actually force zero.
    """
    import os
    import warnings

    import numpy as np
    from .config import ReconstructionSource, UncertaintyConfig

    # Read the defaults off the dataclass so "the user changed this" can never
    # drift from the declared defaults.
    _SCALAR_SIGMA_DEFAULTS = {
        _c: float(getattr(UncertaintyConfig, f"{_c}_scalar_sigma"))
        for _c in ("ne", "te", "ni", "ti")
    }

    unc = config.uncertainty
    src = config.source
    psi_kin = np.asarray(baseline.psi_N_kinetic, dtype=float)

    ida_path = unc.ida_path
    if ida_path is None and isinstance(src, ReconstructionSource) \
            and src.profiles_path.endswith(".cdf"):
        ida_path = src.profiles_path

    # IDA arrays (read once) available as a fallback below
    ida_sig = None
    if ida_path is not None:
        from .io.ida import read_ida
        ida = read_ida(
            ida_path, time=getattr(src, "time", None),
            sigma_mode=unc.sigma_mode, sigma_method=unc.sigma_method,
            sigma_ni_from_ne=unc.sigma_ni_from_ne,
        )

        def _to_kin(arr):
            return np.interp(psi_kin, ida.psi_N, np.asarray(arr, dtype=float))

        ida_sig = {"ne": _to_kin(ida.sigma_ne), "te": _to_kin(ida.sigma_te),
                   "ni": _to_kin(ida.sigma_ni), "ti": _to_kin(ida.sigma_ti)}

    # Per-channel resolution: explicit profile > IDA > flat scalar fraction.
    _profiles = unc.sigma_profiles or {}
    _scalars = {"ne": unc.ne_scalar_sigma, "te": unc.te_scalar_sigma,
                "ni": unc.ni_scalar_sigma, "ti": unc.ti_scalar_sigma}
    _baseprof = {"ne": baseline.ne, "te": baseline.te,
                 "ni": baseline.ni, "ti": baseline.ti}
    out = {}
    _won = {}       # channel -> human-readable winning source (for the log)
    _shadowed = []  # channels whose deliberately-set scalar lost to the IDA
    for _ch in ("ne", "te", "ni", "ti"):
        if _ch in _profiles and _profiles[_ch] is not None:
            _arr = np.asarray(_profiles[_ch], dtype=float)
            if _arr.shape != psi_kin.shape:
                raise ValueError(
                    f"sigma_profiles['{_ch}'] has shape {_arr.shape}; expected "
                    f"the kinetic grid {psi_kin.shape} (psi_N_kinetic)")
            out[f"sigma_{_ch}"] = _arr
            _won[_ch] = "explicit sigma_profiles"
        elif ida_sig is not None:
            out[f"sigma_{_ch}"] = ida_sig[_ch]
            _won[_ch] = f"IDA {os.path.basename(ida_path)}"
            # A scalar the user moved off its default is being SILENTLY
            # ignored. Zeroing them is the classic footgun: it reads like
            # "no perturbation" but leaves the full operational IDA envelope
            # in place, so a run intended as deterministic is a full-sigma
            # ensemble (the 2026-08 beta-scan case).
            if float(_scalars[_ch]) != _SCALAR_SIGMA_DEFAULTS[_ch]:
                _shadowed.append(
                    f"{_ch}_scalar_sigma={float(_scalars[_ch]):g}")
        else:
            out[f"sigma_{_ch}"] = float(_scalars[_ch]) * np.abs(
                np.asarray(_baseprof[_ch], dtype=float))
            _won[_ch] = f"scalar {float(_scalars[_ch]):g} x |baseline|"

    out["sigma_jphi"] = unc.jphi_scalar_sigma * np.abs(np.asarray(baseline.j_phi, dtype=float))
    _won["jphi"] = f"scalar {float(unc.jphi_scalar_sigma):g} x |j_phi|"
    out["n_ls"], out["t_ls"], out["j_ls"] = unc.n_ls, unc.t_ls, unc.j_ls

    # --- precedence audit trail ---------------------------------------------
    # One line per channel naming the winner and the resolved magnitude. The
    # precedence is silent by construction (a winning source simply shadows the
    # others), and a silent win can invert the meaning of a whole run, so it is
    # worth the four lines. Turn off with UncertaintyConfig.log_sigma_sources.
    if bool(getattr(unc, "log_sigma_sources", True)):
        for _ch in ("ne", "te", "ni", "ti", "jphi"):
            _s = np.asarray(out[f"sigma_{_ch}"], dtype=float)
            _pk = float(np.max(np.abs(_s))) if _s.size else 0.0
            print(f"  [sigma-source] sigma_{_ch:<4s} <- {_won[_ch]:<28s} "
                  f"(peak {_pk:.4g}{', ALL ZERO' if _pk == 0.0 else ''})")

    if _shadowed:
        warnings.warn(
            f"resolve_uncertainty: {', '.join(_shadowed)} was set but IGNORED "
            f"-- an IDA source ({os.path.basename(ida_path)}) is active and "
            f"wins the kinetic sigma precedence "
            f"(sigma_profiles > IDA .cdf > <chan>_scalar_sigma). The resolved "
            f"sigmas are the full IDA envelope, NOT your scalars. To force a "
            f"specific envelope (e.g. zero, for a deterministic run) pass "
            f"explicit arrays instead:  "
            f"unc.sigma_profiles = {{ch: np.zeros_like(baseline.psi_N_kinetic) "
            f"for ch in ('ne','te','ni','ti')}}  -- or clear "
            f"UncertaintyConfig.ida_path / use a non-.cdf profiles_path.",
            stacklevel=2,
        )

    # --- switchboard: resolve the auxiliary perturbed profiles ---------------
    # A sigma entry enables a profile. Baseline = manual (aux_baselines) over
    # source-provided (baseline.aux). Warn + skip if the baseline is absent or
    # all-zero (the user supplied a sigma for nothing).
    src_aux = dict(baseline.aux or {})
    man_base = dict(unc.aux_baselines or {})

    # Z_eff channel is enabled by default for EVERY source (the consistent
    # density scheme): unless the user set an explicit aux_sigmas['zeff'], a
    # flat fractional envelope zeff_scalar_sigma * Z_eff_baseline is injected.
    # The baseline Z_eff is source-provided (baseline.aux['zeff'] for IMAS,
    # else baseline.Zeff for the reconstruction path), on the kinetic grid.
    user_sigmas = dict(unc.aux_sigmas or {})
    if "zeff" not in user_sigmas and float(getattr(unc, "zeff_scalar_sigma", 0.0)) > 0:
        base_zeff = src_aux.get("zeff")
        if base_zeff is None:
            base_zeff = np.asarray(baseline.Zeff, dtype=float)
        if "zeff" not in man_base:
            man_base["zeff"] = np.asarray(base_zeff, dtype=float)
        user_sigmas["zeff"] = unc.zeff_scalar_sigma * np.abs(
            np.asarray(man_base["zeff"], dtype=float))

    resolved_sigma, resolved_base = {}, {}
    for name, sig in user_sigmas.items():
        base = man_base.get(name, src_aux.get(name))
        if base is None or not np.any(np.asarray(base, dtype=float)):
            warnings.warn(
                f"aux_sigmas['{name}'] supplied but its baseline is "
                f"{'absent' if base is None else 'all-zero'} in this source; "
                f"perturbation of '{name}' is skipped. Provide it via "
                f"UncertaintyConfig.aux_baselines."
            )
            continue
        resolved_sigma[name] = np.asarray(sig, dtype=float)
        resolved_base[name] = np.asarray(base, dtype=float)
    out["aux_sigmas"] = resolved_sigma
    out["aux_baselines"] = resolved_base
    out["aux_length_scales"] = dict(unc.aux_length_scales or {})
    return out


def floor_inductive_split(j_inductive, j_BS, psi_N=None, warn_frac=1e-4):
    """Enforce the ``j_inductive >= 0`` component convention on a (j_ind, j_BS)
    split, absorbing any negative sliver into ``j_BS`` so the pair still sums
    exactly to the same total.

    A strong pedestal can push the achieved total BELOW the full-Sauter
    bootstrap locally, leaving a small negative inductive residual. That is
    unphysical in this convention and -- fed to the GPR sampler as its mean --
    makes essentially every current draw go negative and be rejected. Returns
    ``(j_inductive_floored, j_BS_adjusted)`` (copies; inputs untouched) and
    prints a one-line note when the correction is non-trivial.
    """
    import numpy as np

    j_ind = np.asarray(j_inductive, dtype=float)
    jbs = np.asarray(j_BS, dtype=float)
    if not np.any(j_ind < 0.0):
        return j_ind, jbs
    j_ind_f = np.maximum(j_ind, 0.0)
    deficit = j_ind - j_ind_f                    # <= 0 where floored
    jbs_f = jbs + deficit                        # absorb -> sum unchanged
    scale = float(np.max(np.abs(j_ind))) or 1.0
    worst = float(-deficit.min())
    if worst / scale > warn_frac:
        n = int(np.sum(deficit < 0.0))
        where = ""
        if psi_N is not None:
            pn = np.asarray(psi_N, dtype=float)
            sel = pn[deficit < 0.0]
            where = f" over psi_N [{sel.min():.3f}, {sel.max():.3f}]"
        print(f"  [baseline] floored negative j_inductive ({n} pts{where}, "
              f"worst {worst/1e6:.4f} MA/m^2, {100*worst/scale:.2f}% of peak) "
              f"-- deficit absorbed into j_BS (split still sums to j_phi)")
    return j_ind_f, jbs_f


def _resolve_fixed(comp, src_psi, dst_psi):
    """Fixed additive component onto ``dst_psi`` (zeros if not supplied)."""
    import numpy as np

    if comp is None:
        return np.zeros_like(dst_psi)
    comp = np.asarray(comp, dtype=float)
    if src_psi is None:
        if comp.shape != dst_psi.shape:
            raise ValueError(
                "fixed-component array length does not match the target grid; "
                "provide FixedComponentsConfig.psi_N for resampling"
            )
        return comp
    return np.interp(dst_psi, np.asarray(src_psi, dtype=float), comp)


def _load_kinetic_profiles(source) -> dict:
    """Kinetic profiles (SI) from the reconstruction source's ``profiles_path``.

    Dispatches on file type: an IDA ``.cdf`` (ni = ne, quasi-neutrality) or an
    Osborne p-file (real ni via quasineutrality + Zeff from the ion mix). Returns
    a dict with ``psi_N`` (native kinetic grid), ``ne``/``te``/``ni``/``ti``
    (m^-3, eV), ``Zeff`` (clipped >= 1), and ``raw_bytes`` for archival.
    """
    import numpy as np

    path = source.profiles_path
    if path.endswith(".cdf"):
        from .io.ida import read_ida
        ida = read_ida(path, time=source.time, impurity_Z=source.impurity_Z)
        return dict(
            psi_N=np.asarray(ida.psi_N, dtype=float),
            ne=np.asarray(ida.ne, dtype=float),
            te=np.asarray(ida.te, dtype=float),
            ni=np.asarray(ida.ni, dtype=float),
            ti=np.asarray(ida.ti, dtype=float),
            Zeff=np.clip(np.asarray(ida.Zeff, dtype=float), 1.0, None),
            raw_bytes=ida.raw_bytes,
        )

    # Osborne p-file: ne/ni in 1e20 m^-3, Te/Ti in keV -> SI.
    from .io.pfile import read_pfile
    with open(path, "rb") as fh:
        raw = fh.read()
    pf = read_pfile(path)
    if pf.ion_species is None:
        raise ValueError(
            "p-file carries no ion species in its footer; cannot compute "
            "quasineutrality / Zeff"
        )
    pf.compute_quasineutrality()
    psi_pf, Zeff = pf.compute_zeff()
    return dict(
        psi_N=np.asarray(psi_pf, dtype=float),
        ne=np.asarray(pf.ne, dtype=float) * 1e20,
        te=np.asarray(pf.te, dtype=float) * 1e3,
        ni=np.asarray(pf.ni, dtype=float) * 1e20,
        ti=np.asarray(pf.ti, dtype=float) * 1e3,
        Zeff=np.clip(np.asarray(Zeff, dtype=float), 1.0, None),
        raw_bytes=raw,
    )


def _resolve_reconstruction(source, config, mygs) -> Baseline:
    """Reconstruction-source baseline: GS reconstruct on a live ``mygs``.

    Mirrors the operational notebook: read g-file + IDA profiles, interpolate
    onto the g-file psi_N grid, run :func:`reconstruct_equilibrium`, and package
    the (toroidal) fitted currents. The reconstructed total ``j_phi_fit`` already
    contains all driven current, so fixed components (j_NBI / j_RF) default to
    zero and only re-partition the inductive part if the user supplies them;
    ``p_fast`` (absent from thermal IDA profiles) likewise defaults to zero.
    """
    import numpy as np
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun

    from .io.geqdsk import read_geqdsk
    from .TokaMaker_interface import reconstruct_equilibrium

    if mygs is None:
        raise ValueError(
            "ReconstructionSource requires a live TokaMaker solver; call "
            "setup_solver() before prepare_baseline()"
        )
    if source.profile_overrides:
        raise NotImplementedError("profile_overrides is not yet applied")

    with open(source.geqdsk_path, "rb") as fh:
        eqdsk_bytes = fh.read()
    eqdsk = read_geqdsk(source.geqdsk_path, cocos=source.cocos)
    psi_N = np.asarray(eqdsk.psi_N, dtype=float)

    kin = _load_kinetic_profiles(source)
    psi_N_kin = kin["psi_N"]

    # kinetic profiles (native SI) regridded onto the equilibrium psi_N grid.
    # Shape-preserving PCHIP (single shared helper): a linear regrid leaves a
    # slope kink at every kinetic knot, which the Sauter bootstrap inherits
    # as a stepped j_BS (see utils.pchip_interp).
    from .utils import pchip_interp

    def to_eq(arr):
        return pchip_interp(psi_N_kin, arr, psi_N)

    ne_eq, te_eq, ni_eq, ti_eq = to_eq(kin["ne"]), to_eq(kin["te"]), to_eq(kin["ni"]), to_eq(kin["ti"])
    Zeff_eq = np.clip(to_eq(kin["Zeff"]), 1.0, None)

    # isoflux from the g-file boundary (matches the recon-stage notebook weight)
    iso_pts = np.column_stack([eqdsk.boundary_R, eqdsk.boundary_Z])
    iso_w = np.ones(len(iso_pts)) * 200.0
    mygs.set_isoflux(iso_pts, weights=iso_w)

    guess_jinductive = create_power_flux_fun(len(psi_N), 1.5, 1.5)["y"]

    # Fixed (non-perturbed) pressure components must be resolved BEFORE the
    # reconstruction, not after it: the reconstruction's GS pressure has to be
    # the same pressure every draw solves, otherwise l_i_target is measured on
    # a lower-pressure equilibrium than the draws it targets (see the pressure
    # block in reconstruct_equilibrium).
    #
    # Grid: p_fast is resolved onto the KINETIC grid first and then mapped to
    # the equilibrium grid with `to_eq` -- deliberately the same two-step path
    # the draws take (baseline resolves onto psi_N_kin, then
    # perturb_kinetic_equilibrium applies `_kin_to_eq`, which is the identical
    # pchip_interp).  Resolving fc.psi_N -> psi_N in one hop would be a
    # slightly different array and would reintroduce the very inconsistency
    # this is fixing.  `p_fast_kin` is also what the returned Baseline.p_fast
    # field carries (kinetic grid), which downstream depends on -- unchanged.
    #
    # When fc.p_fast is unset, pass None rather than the zeros _resolve_fixed
    # returns, so the default-off path does not even enter the new branch and
    # is provably a no-op (not merely "adds 0.0").
    fc = config.fixed_components
    p_fast_kin = _resolve_fixed(fc.p_fast, fc.psi_N, psi_N_kin)
    p_fast_eq = to_eq(p_fast_kin) if fc.p_fast is not None else None
    # Z_imp is plumbed for symmetry with the draw path, but is INERT here today:
    # FixedComponentsConfig (config.py) carries no Z_imp field at all -- Z_imp is
    # an *output* field of the Baseline dataclass, populated only on the IMAS
    # path. getattr keeps this a no-op now and makes it activate automatically
    # if the config ever gains the field, rather than silently diverging again.
    Z_imp_recon = getattr(fc, "Z_imp", None)

    # Capture the verbose solver chatter (DLSODE / gs_get_qprof / li-match) unless
    # the user asked for it; the curated summary is printed by Bouquet.reconstruct.
    from .utils import capture_native_output
    verbose = bool(getattr(config, "verbose", False))
    with capture_native_output(enabled=not verbose) as _cap:
        result = reconstruct_equilibrium(
            mygs, eqdsk,
            ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq,
            iso_pts, iso_w, source.psi_pad,
            guess_jinductive=guess_jinductive,
            n_k=source.n_k,
            psi_bridge=source.psi_bridge,
            rescale_j_BS=source.rescale_j_BS,
            shelf_psi_N=source.shelf_psi_N,
            initialize_psi=True,
            p_fast=p_fast_eq,
            Z_imp=Z_imp_recon,
            l_i_tolerance=float(config.generation.l_i_tolerance),
        )
        # get_stats traces the q-profile and can emit gs_get_qprof warnings, so
        # keep these inside the capture too.
        Ip_target = abs(float(eqdsk.Ip))
        # l_i scale: 'iter' == li(3) == 2*int(Bp^2 dV)/((mu0 Ip)^2 R_axis).
        # This is the estimator reconstruct_equilibrium targets (the g-file's
        # `li(2)` key, which is numerically the same functional) and the ONLY
        # one the two codes agree on (0.17%).  The whole downstream chain --
        # per-draw acceptance band, archive attrs, plots -- is on this scale;
        # see issue #20.  DO NOT mix with 'std'/li(1) numbers.
        #
        # WHICH l_i (issue #25).  This used to be a fresh get_stats read of
        # whatever state the reconstruction happened to END on -- i.e. AFTER
        # step 7's corrective iteration, which runs up to 8 further GS solves
        # with a stopping rule that knows nothing about l_i.  So the number the
        # whole ensemble is banded around was not the value step 6 matched; it
        # was that value plus however far step 7 drifted (measured +0.50 to
        # +0.76 % across the beta-scan family, against a 1 % per-draw band).
        #
        # `result['li_final']` IS the step-6 matched value: the one the secant
        # loop drove onto the g-file's li(2), and therefore the only one with a
        # defensible provenance as a target.  The post-corrective read is kept
        # as a separate archived diagnostic rather than dropped, so this change
        # ADDS provenance -- `l_i_realized_post_corrective` says where the
        # reconstruction actually finished, and its distance from the target is
        # reported loudly by reconstruct_equilibrium when it exceeds the band.
        #
        # `l_i_target` INTENTIONALLY DIFFERS FROM THE PRE-#27 ARCHIVED VALUE by
        # exactly the step-7 corrective drift -- that IS the change described
        # above, not a side effect of it.  Worked example, main @ 4ad4894 on
        # the synthetic golden: the old target (the post-corrective read) was
        # 0.656074, the new one (step-6 matched) is 0.653840, and
        # 0.653840 * 1.003416 = 0.656074 -- the +0.342 % recorded in
        # `li_corrective_drift_pct` and nothing else.  Still far in-band
        # (+/-5.00 %, `li_corrective_out_of_band` False).  This is also the
        # direct cause of the sigma=0 R2 l_i residual reading -0.083 % rather
        # than the pre-#27 -0.457 %: route R2 skips the corrective iteration,
        # so the old target was charging it for a drift it never applies (see
        # `_LI_REL` in tests/test_seeded_reproducibility.py, and issue #28).
        l_i_target = float(result["li_final"])
        # THE NOTE BELOW IS ABOUT THIS LINE ONLY -- a bit-neutral refactor, not
        # a claim about `l_i_target` (which moves by design, see above).
        # Consume the value reconstruct_equilibrium already measured at step 7b
        # rather than re-reading get_stats here.  The re-read was redundant --
        # same solver state, same lcfs_pad (source.psi_pad is exactly what was
        # passed in above), same li_normalization='iter' -- and it cost a
        # second q-profile trace plus its gs_get_qprof warnings.  It also had
        # the WEAKER provenance of the two: it reported whatever state the
        # solver happens to be in at THIS line, which is "post step 7" only for
        # as long as nothing in between touches the equilibrium.  Consuming the
        # returned field pins the number to the step it is named after.
        # Verified bit-identical to the old re-read on the synthetic golden
        # fixture (delta exactly 0.0), so no archived value moves.  RE-VERIFIED
        # on main @ 4ad4894 for issue #28, capturing both at this exact point
        # in the control flow: old re-read and consumed value are both
        # 0.6560742094407394, delta exactly 0.0.  Nothing between step 7b and
        # here mutates the equilibrium, so the "only for as long as" caveat
        # above still holds and is still worth keeping.
        l_i_realized_post_corrective = float(
            result["li_realized_post_corrective"])
        recon_metrics = _reconstruction_metrics(
            mygs, eqdsk, result, source, l_i_target,
            l_i_realized_post_corrective=l_i_realized_post_corrective)

    j_phi = np.asarray(result["j_phi_fit"], dtype=float)
    j_BS = np.asarray(result["j_BS_used"], dtype=float)

    j_NBI = _resolve_fixed(fc.j_NBI, fc.psi_N, psi_N)
    j_RF = _resolve_fixed(fc.j_RF, fc.psi_N, psi_N)
    j_inductive = j_phi - j_BS - j_NBI - j_RF   # == j_inductive_fit when NBI=RF=0
    # Physical component convention: the inductive current is >= 0. On shots
    # with a strong pedestal the achieved total can dip BELOW the full-Sauter
    # bootstrap there, leaving a small negative residual (~1% of the core) --
    # which, fed to the GPR sampler as its mean, makes essentially every draw
    # go negative and be rejected (observed on a strong-pedestal case:
    # 0/500 candidates survived).
    # Floor the inductive at zero and absorb the deficit into j_BS so the
    # split still sums exactly to j_phi.
    j_inductive, j_BS = floor_inductive_split(j_inductive, j_BS, psi_N)

    # Resolved above (before the reconstruction, which now consumes it).
    # Unchanged contract: the returned field is on the KINETIC grid.
    p_fast = p_fast_kin

    return Baseline(
        psi_N=psi_N,
        j_phi=j_phi,
        j_inductive=j_inductive,
        j_BS=j_BS,
        psi_N_kinetic=psi_N_kin,
        ne=kin["ne"],
        te=kin["te"],
        ni=kin["ni"],
        ti=kin["ti"],
        Zeff=kin["Zeff"],
        Ip_target=Ip_target,
        l_i_target=l_i_target,
        provenance="reconstruction",
        j_NBI=j_NBI,
        j_RF=j_RF,
        p_fast=p_fast,
        # SAME source as the value handed to the reconstruction above, so the
        # two paths activate together or not at all.  The draws read
        # `Baseline.Z_imp` (run.py hands it to generate_bouquet, and the
        # forward / sigma=0 solves read it directly); the recon reads
        # `Z_imp_recon`.  Leaving this at its None default while feeding
        # `Z_imp_recon` to the recon meant that the day FixedComponentsConfig
        # gains a Z_imp field, the reconstruction would start adding impurity
        # pressure while the draws still did not -- silently recreating the
        # exact recon-vs-draw pressure inconsistency this plumbing exists to
        # remove, through the very getattr that was meant to guard it.
        # Today both are None on this path, so this is a no-op.
        Z_imp=Z_imp_recon,
        eqdsk_bytes=eqdsk_bytes,
        pfile_bytes=kin["raw_bytes"],
        # expose the baseline Z_eff (kinetic grid) as a switchboard channel,
        # mirroring the IMAS path -- so the zeff channel resolves its baseline
        # for both the default auto-injection and an explicit aux_sigmas['zeff']
        aux={"zeff": np.asarray(kin["Zeff"], dtype=float)},
        recon=result,
        reconstruction_metrics=recon_metrics,
        reconstruction_log=_cap["text"] or None,
    )


def _reconstruction_metrics(mygs, eqdsk, result, source, l_i_achieved,
                            l_i_realized_post_corrective=None) -> dict:
    """Curate a TokaMaker-vs-EFIT reconstruction-fidelity dict for the summary.

    Each global scalar that isn't ~0 by construction (Ip, l_i, q0/q95, beta,
    shape, separatrix current, stored energy) is reported as a % error against
    the g-file's own EFIT value. Geometric residuals that *should* be ~0
    (boundary match, axis offset, j_phi profile RMS) stay absolute. A simple
    pass/fail verdict is applied to the reconstruction targets.
    """
    import numpy as np

    def pct(tok, ref):
        ref = float(ref)
        if not np.isfinite(ref) or abs(ref) < 1e-12:
            return float("nan")
        return float(100.0 * (float(tok) - ref) / abs(ref))

    q = dict(result.get("quality") or {})
    # Global scalars other than l_i are normalization-independent; take them
    # from the 'iter' call so there is exactly one get_stats scale in play.
    stats = mygs.get_stats(lcfs_pad=source.psi_pad, li_normalization="iter")
    # The FREE (untargeted) li(1) pair, for the cross-estimator report below.
    stats_std = mygs.get_stats(lcfs_pad=source.psi_pad, li_normalization="std")

    # --- TokaMaker (reconstructed) values ---
    Ip_tok = float(result.get("Ip_tokamaker", float("nan")))
    li_tok = float(l_i_achieved)
    q0_tok = float(stats.get("q_0", float("nan")))
    q95_tok = float(stats.get("q_95", float("nan")))
    betan_tok = float(stats.get("beta_n", float("nan")))
    betap_tok = float(stats.get("beta_pol", float("nan"))) / 100.0  # x100 -> standard
    kappa_tok = float(stats.get("kappa", float("nan")))
    delta_tok = float(stats.get("delta", float("nan")))
    W_tok = float(stats.get("W_MHD", float("nan"))) / 1e6           # MJ
    o_point = getattr(mygs, "o_point", [float("nan"), float("nan")])
    # separatrix current evaluated just inside the LCFS (psi_N=0.99), where the
    # edge current is better-defined than the near-singular psi_N=1 point.
    _psiN_eq = np.asarray(eqdsk.psi_N, dtype=float)
    _jsep_psiN = 0.99
    jphi = np.asarray(result.get("j_phi_fit"), dtype=float)
    jsep_tok = float(np.interp(_jsep_psiN, _psiN_eq, jphi)) / 1e6   # MA/m^2

    # --- EFIT (g-file) reference values; tolerate a tracing failure ---
    efit = {}
    try:
        efit["Ip"] = abs(float(eqdsk.Ip))
        # `li(2)` is the g-file-side li(3)/'iter' functional -- the estimator
        # the reconstruction actually targets (issue #20).  `li(1)` is kept
        # alongside it for the free cross-estimator pair.
        efit["li"] = float(eqdsk.li.get("li(2)", float("nan")))
        efit["li1"] = float(eqdsk.li.get("li(1)_EFIT", float("nan")))
        qpsi = np.asarray(eqdsk.qpsi, dtype=float)
        psiN = np.asarray(eqdsk.psi_N, dtype=float)
        efit["q0"] = float(qpsi[0])
        efit["q95"] = float(np.interp(0.95, psiN, qpsi))
        betas = eqdsk.betas
        efit["beta_n"] = float(betas.get("beta_n", float("nan")))
        efit["beta_p"] = float(betas.get("beta_p", float("nan")))
        geo = eqdsk.geometry
        efit["kappa"] = float(np.asarray(geo["kappa"])[-1])
        efit["delta"] = float(np.asarray(geo["delta"])[-1])
        efit["R_mag"] = float(eqdsk.R_mag)
        efit["Z_mag"] = float(eqdsk.Z_mag)
        efit["W_MHD"] = 1.5 * float(eqdsk.volume_integral(eqdsk.pres)[-1]) / 1e6
        jtor_efit = np.asarray(result.get("eqdsk_jtor"), dtype=float)
        efit["j_sep"] = float(np.interp(_jsep_psiN, _psiN_eq, jtor_efit)) / 1e6
    except Exception as exc:  # tracing/format issue -> % errors fall back to nan
        import warnings
        warnings.warn(f"EFIT reference metrics unavailable: {exc}")

    g = lambda k: float(efit.get(k, float("nan")))

    bnd_rms = float(q.get("boundary_rms_mm", float("nan")))
    bnd_max = float(q.get("boundary_max_dev_mm", float("nan")))
    axis_off_mm = float(np.hypot(o_point[0] - g("R_mag"),
                                 o_point[1] - g("Z_mag")) * 1e3)

    # Convergence of the final accepted GS state. reconstruct_equilibrium's
    # li-match only ever keeps successful solves (a failed solve restores the
    # last good psi and is retried), and an unrecoverable solver failure
    # raises before reaching here -- so the honest residual check is that the
    # final state produced finite global quantities.
    converged = bool(np.isfinite(Ip_tok) and np.isfinite(li_tok)
                     and np.isfinite(q95_tok))

    # verdict on the reconstruction *targets*: converged, Ip within 0.5%,
    # boundary RMS < 5 mm, l_i within 5%
    ip_err = pct(Ip_tok, g("Ip"))
    li_err = pct(li_tok, g("li"))
    li1_tok = float(stats_std.get("l_i", float("nan")))
    li1_cross_err = pct(li1_tok, g("li1"))
    ok = bool(
        converged
        and np.isfinite(ip_err) and abs(ip_err) < 0.5
        and np.isfinite(bnd_rms) and bnd_rms < 5.0
        and np.isfinite(li_err) and abs(li_err) < 5.0
    )
    return {
        "converged": converged,
        "verdict": "PASS" if ok else "CHECK",
        # fidelity: TokaMaker value, EFIT reference, % error
        "Ip_MA": Ip_tok / 1e6, "Ip_efit_MA": g("Ip") / 1e6, "Ip_err_pct": ip_err,
        # --- l_i, on the estimator the loop targets ('iter' == li(3)) -------
        # `li_err_pct` is the MATCHED pair: the secant loop drives these two
        # numbers together, so it is ~0 by construction and cannot detect an
        # estimator mismatch.  That is precisely how issue #20 stayed hidden.
        #
        # SINCE #25 IT IS VACUOUS OUTRIGHT, not merely weak.  `li` is now
        # `l_i_target` == `result['li_final']`, the step-6 MATCHED value; so
        # `li_err_pct` compares the secant's converged output against the very
        # target it converged onto, and reports nothing but that loop's own
        # tolerance.  Before #25 it was read post-step-7, so it at least
        # carried the corrective iteration's drift.  It is kept for continuity
        # of the summary table and because a NON-tiny value would mean the
        # secant did not converge -- but it is not a fidelity measure.
        # For "where did the reconstruction actually finish", read
        # `li_realized_post_corrective` / `li_corrective_drift_pct` below; for
        # a genuinely free estimator comparison, read the li(1) pair further
        # down, which nothing drives together.
        "li": li_tok, "li_efit": g("li"), "li_err_pct": li_err,
        "li_scale": "iter(li3)",
        # --- step-6 matched vs step-7 realized (issue #25) ------------------
        # `li` above (== l_i_target) is the MATCHED value.  These say where the
        # corrective iteration actually left the equilibrium, and whether that
        # is further from the target than a draw is allowed to be.  Archived,
        # never enforced: a visible condition, not an acceptance criterion.
        "li_realized_post_corrective": (
            float("nan") if l_i_realized_post_corrective is None
            else float(l_i_realized_post_corrective)),
        "li_corrective_drift_pct": float(
            q.get("li3_corrective_drift_pct", float("nan"))),
        "li_corrective_band_pct": float(
            q.get("li3_corrective_band_pct", float("nan"))),
        "li_corrective_out_of_band": bool(
            q.get("li3_corrective_out_of_band", False)),
        # --- cross-estimator drift monitor (de-circularization, issue #20) --
        # The FREE pair: TokaMaker's 'std'/li(1) vs the g-file's li(1)_EFIT.
        # Nothing drives these together, so a nonzero residual is a genuine
        # measurement of convention/geometry drift between the two codes
        # (historically ~+3.3% on DIII-D geqdsks, from TokaMaker's projected
        # separatrix perimeter).  If this ever collapses to ~0 or changes
        # sharply, an estimator moved -- investigate before trusting the run.
        "li1_cross": li1_tok, "li1_cross_efit": g("li1"),
        "li1_cross_err_pct": li1_cross_err,
        "q0": q0_tok, "q0_efit": g("q0"), "q0_err_pct": pct(q0_tok, g("q0")),
        "q95": q95_tok, "q95_efit": g("q95"), "q95_err_pct": pct(q95_tok, g("q95")),
        "beta_n": betan_tok, "beta_n_efit": g("beta_n"),
        "beta_n_err_pct": pct(betan_tok, g("beta_n")),
        "beta_p": betap_tok, "beta_p_efit": g("beta_p"),
        "beta_p_err_pct": pct(betap_tok, g("beta_p")),
        "kappa": kappa_tok, "kappa_efit": g("kappa"),
        "kappa_err_pct": pct(kappa_tok, g("kappa")),
        "delta": delta_tok, "delta_efit": g("delta"),
        "delta_err_pct": pct(delta_tok, g("delta")),
        "j_sep_MA": jsep_tok, "j_sep_efit_MA": g("j_sep"),
        "j_sep_err_pct": pct(jsep_tok, g("j_sep")),
        "W_MHD_MJ": W_tok, "W_MHD_efit_MJ": g("W_MHD"),
        "W_MHD_err_pct": pct(W_tok, g("W_MHD")),
        # zero-ideal residuals (absolute)
        "boundary_rms_mm": bnd_rms, "boundary_max_mm": bnd_max,
        "axis_offset_mm": axis_off_mm,
        "jphi_core_rms_MA": float(q.get("jphi_core_rms", float("nan"))) / 1e6,
        "jphi_edge_rms_MA": float(q.get("jphi_edge_rms", float("nan"))) / 1e6,
    }
