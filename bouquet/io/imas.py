"""Reader for FUSE IMAS/OMAS data-dictionary files (``dd_sim.json``).

FUSE writes the IMAS data dictionary as a plain JSON dump, so this reads with the
stdlib ``json`` module -- no IMAS/OMAS/OMFIT install required. It returns a
fully-separated baseline (no GS reconstruction needed): the IDS already carries
j_ohmic, j_bootstrap and the driven currents split apart.

Field mapping (verified against a D3D FUSE run)::

    equilibrium.time_slice[t].global_quantities.ip            -> Ip_target
    equilibrium.time_slice[t].global_quantities.li_3          -> l_i_target
    core_profiles.profiles_1d[t].grid.psi                     -> normalised -> psi_N
    core_profiles.profiles_1d[t].j_total                      -> total PARALLEL current
    core_profiles.profiles_1d[t].j_tor                        -> total TOROIDAL current
    core_profiles.profiles_1d[t].j_ohmic                      -> inductive (parallel)
    core_profiles.profiles_1d[t].j_bootstrap                  -> bootstrap (parallel)
    core_profiles.profiles_1d[t].electrons.{density_thermal,temperature}
    core_profiles.profiles_1d[t].ion[*].{density_thermal,temperature,element[].z_n}
    core_profiles.profiles_1d[t].{electrons,ion[*]}.pressure_fast_{perpendicular,parallel}
    core_sources.source[*].profiles_1d[t].j_parallel          -> beam-source j_NBI only

Currents are converted parallel->toroidal (see :func:`bouquet.physics.parallel_to_toroidal`)
via the per-surface factor c = j_tor/j_total, and fast pressure is isotropized
(see :func:`bouquet.physics.isotropize_fast_pressure`). The total j_phi is set to
the authoritative toroidal ``j_tor`` and the inductive component is taken as the
residual ``j_phi - j_BS - j_NBI - j_RF`` so the decomposition sums exactly and Ip
is preserved.

Note: ``j_BS`` read here is the FUSE bootstrap baseline, but it is *overridden*
when ``GenerationConfig.recalculate_j_BS`` is True -- bouquet then recomputes
bootstrap per draw via TokaMaker ``solve_with_bootstrap`` (whose output is also
parallel and must be converted to toroidal; see ``parallel_to_toroidal``).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from ..physics import (effective_impurity_charge, impurity_pressure,
                       isotropize_fast_pressure, main_ion_density_from_zeff,
                       parallel_to_toroidal)

# Elementary charge [C]: thermal pressure p = e * sum_s(n_s * T_s).
_EC = 1.602176634e-19

if TYPE_CHECKING:
    from ..config import ImasSource, FixedComponentsConfig
    from ..baseline import Baseline

# Core-source identifier index for neutral-beam current drive.
NBI_SOURCE_INDEX = 2          # neutral beam injection -> summed into j_NBI
# NOTE: j_RF is NOT computed internally (RF is the least-common input). It is
# left as zeros and accepted as a user-supplied array via
# FixedComponentsConfig.j_RF. See the "revisit RF" flag in the project notes
# if/when internal EC/IC/LH summation is wanted.


def _nearest_index(time_array, t: Optional[float], what: str) -> int:
    """Index of the slice nearest ``t`` (seconds) in ``time_array``."""
    ta = np.asarray(time_array, dtype=float)
    if ta.size == 1:
        return 0
    if t is None:
        raise ValueError(
            f"{what} has {ta.size} time slices; set ImasSource.time (seconds). "
            f"Range [s]: [{ta.min():.4f}, {ta.max():.4f}]"
        )
    return int(np.argmin(np.abs(ta - t)))


def _isotropic_fast_pressure(species: dict, method: str, n: int):
    """Isotropized fast pressure for one species, or zeros if it carries none."""
    if "pressure_fast_perpendicular" not in species:
        return np.zeros(n)
    p_perp = np.asarray(species["pressure_fast_perpendicular"], dtype=float)
    p_par = np.asarray(species.get("pressure_fast_parallel", p_perp), dtype=float)
    return isotropize_fast_pressure(p_perp, p_par, method=method)


def _override(arr, src_psi, dst_psi):
    """Resample a user-supplied fixed-component array onto the baseline grid."""
    arr = np.asarray(arr, dtype=float)
    if src_psi is None:
        if arr.shape != dst_psi.shape:
            raise ValueError(
                "fixed-component array length does not match the baseline grid; "
                "provide FixedComponentsConfig.psi_N for resampling"
            )
        return arr
    return np.interp(dst_psi, np.asarray(src_psi, dtype=float), arr)


def read_imas_geometry(source: "ImasSource"):
    """Return ``(F0, boundary_RZ)`` from a FUSE IDS for TokaMaker setup.

    ``F0 = |r0 * b0(t)|`` from ``equilibrium.vacuum_toroidal_field`` and the LCFS
    isoflux points from ``equilibrium.time_slice[t].boundary.outline``. Used by
    :meth:`Bouquet.setup_solver` when the source is an :class:`ImasSource`
    (replacing the g-file that the reconstruction path reads F0/boundary from).
    """
    import json

    with open(source.ids_path) as fh:
        dd = json.load(fh)
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], source.time, "equilibrium")
    vtf = eq["vacuum_toroidal_field"]
    r0 = float(vtf["r0"])
    b0 = vtf["b0"]
    b0v = float(b0[ie]) if isinstance(b0, list) else float(b0)
    F0 = abs(r0 * b0v)
    # Separatrix: an optional external g-file (if given) supplies the LCFS;
    # otherwise use the dd boundary outline. F0 stays from the dd vacuum field
    # either way (same shot).
    lcfs_g = getattr(source, "LCFS_geqdsk", None)
    if lcfs_g:
        from .geqdsk import read_geqdsk
        g = read_geqdsk(lcfs_g)
        boundary_RZ = np.column_stack([
            np.asarray(g.boundary_R, dtype=float),
            np.asarray(g.boundary_Z, dtype=float),
        ])
    else:
        out = eq["time_slice"][ie]["boundary"]["outline"]
        boundary_RZ = np.column_stack([
            np.asarray(out["r"], dtype=float), np.asarray(out["z"], dtype=float),
        ])
    return F0, boundary_RZ


def _validate_pressure_completeness(cp, ne, te, ni, ti, p_fast, p_imp,
                                    p_equilibrium, allow_incomplete):
    """Fail-fast when the IMAS source carries pressure the reconstruction omits.

    Hard fails (raise, or warn if ``allow_incomplete``) on DROPPED reconstructable
    components -- a thermal species or the fast channel:
      1. species  -- e*(ne*Te + ni*Ti) + impurity(nz*Ti) vs e*(ne*Te + Sum n_s*T_s)
      2. fast     -- any species carries pressure_fast_* but assembled p_fast ~ 0
    Plus a non-fatal backstop (warning only):
      3. reconstructed total vs equilibrium.pressure -- the equilibrium routinely
         carries more (anisotropic/beam + equilibrium-vs-transport gap) which the
         p_diff anchor absorbs; informational, not blocking.
    """
    problems = []
    # 1. species completeness (single-impurity model vs full Sum_s n_s T_s)
    full_thermal = _EC * ne * te
    for ion in cp["ion"]:
        full_thermal = full_thermal + _EC * (
            np.asarray(ion["density_thermal"], dtype=float)
            * np.asarray(ion["temperature"], dtype=float))
    recon_thermal = _EC * (ne * te + ni * ti) + p_imp
    denom = float(np.mean(np.abs(full_thermal))) or 1.0
    sp_gap = float(np.mean(np.abs(full_thermal - recon_thermal))) / denom
    if sp_gap > 0.02:
        problems.append(
            f"thermal species gap {sp_gap*100:.1f}% (>2%): single-impurity model "
            f"(Z_imp from Zeff, impurity at main-ion Ti) does not reproduce "
            f"Sum_s(n_s*T_s) over {len(cp['ion'])} ion species -- multiple "
            f"impurities or T_imp != Ti.")
    # 2. fast completeness
    avail_fast = 0.0
    for sp in [cp["electrons"]] + cp["ion"]:
        if "pressure_fast_perpendicular" in sp:
            avail_fast = max(avail_fast, float(np.max(np.abs(
                np.asarray(sp["pressure_fast_perpendicular"], dtype=float)))))
    if avail_fast > 0.0 and float(np.max(np.abs(p_fast))) <= 1e-3 * avail_fast:
        problems.append(
            f"fast pressure present in source (max {avail_fast/1e3:.1f} kPa) but "
            f"assembled p_fast ~ 0 -- fast-ion channel dropped.")
    # 3. backstop vs authoritative equilibrium pressure -- WARNING only, not a
    #    hard fail. The equilibrium.pressure routinely exceeds the core_profiles
    #    thermal+fast reconstruction (anisotropic/beam pressure that is large early
    #    in a beam-heavy discharge, plus the equilibrium-vs-transport inconsistency
    #    -- the pressure analogue of the j_tor gap). p_diff anchors the baseline to
    #    equilibrium.pressure EXACTLY regardless; this residual just rides fixed
    #    (it is fast/non-thermal, which the thermal perturbation should hold fixed
    #    anyway). So inform, don't block. Dropped *reconstructable* thermal/fast
    #    components are the hard fails (#1/#2 above).
    recon_total = recon_thermal + p_fast
    pe = float(np.mean(np.abs(p_equilibrium))) or 1.0
    bk = float(np.mean(np.abs(recon_total - p_equilibrium))) / pe
    if bk > 0.05:
        import warnings
        warnings.warn(
            f"reconstructed total pressure is {bk*100:.1f}% below dd "
            f"equilibrium.pressure; the p_diff anchor absorbs it (baseline still "
            f"matches FUSE), so this fraction rides fixed (mostly fast/anisotropic "
            f"pressure). Not perturbed in the UQ.")
    if problems:
        msg = ("IMAS pressure completeness check failed:\n  - "
               + "\n  - ".join(problems)
               + "\nThe diff anchor still matches the baseline to FUSE, but the "
               "per-draw thermal perturbation omits the flagged component(s). "
               "Set GenerationConfig.allow_incomplete_pressure=True to bypass "
               "(backend tests only).")
        if allow_incomplete:
            import warnings
            warnings.warn(msg)
        else:
            raise ValueError(msg)


def _read_ida_omega(path, time_s, psi_N):
    """IDA toroidal rotation (omega_tor_12C6) resampled onto psi_N; None if absent.
    Mirrors read_ida's nearest-time selection (IDA time is ms; ``time_s`` is s)."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "omega_tor_12C6" not in f:
                return None
            it = np.asarray(f["time"], dtype=float)            # ms
            tms = (time_s * 1e3) if time_s is not None else float(it[0])
            j = int(np.argmin(np.abs(it - tms)))
            ipsi = np.asarray(f["psi_n"], dtype=float)
            om = np.asarray(f["omega_tor_12C6"], dtype=float)[j]
            return np.interp(psi_N, ipsi, om)
    except Exception:
        return None


def _merge_ida_kinetics(psi_N, ne_fuse, ni_fuse, Zeff_fuse, ida_path, time, impurity_Z,
                         ni_source="all", ni_from_imas_Zeff=False, zeff_from_fuse=False):
    """IDA-hybrid kinetics: replace FUSE ne/ni/Te/Ti/Zeff (+omega) with IDA fits,
    resampled onto the FUSE ``psi_N`` grid (psi_N == psi_N_kinetic).

    Zeff and ni both default to IDA: Zeff measured directly (``zeff_from_fuse=True``
    keeps the FUSE Zeff instead); ni via ``ni_source`` ("Zeff"/"CER"/"all", with
    Jacobian-propagated sigma_ni -- see :func:`bouquet.io.ida.read_ida`). The
    "Zeff"/"all" dilution always uses IDA's own Zeff regardless of
    ``zeff_from_fuse``. ``ni_from_imas_Zeff`` forces ni from the FUSE ``Zeff_fuse`` 
    without updating sigma_ni.

    Returns ``(ne, te, ti, ni, zeff, omega_or_None, sigma_ne, sigma_te, sigma_ni,
    sigma_ti)`` on ``psi_N``.
    """
    from .ida import read_ida
    ida = read_ida(ida_path, time=time, impurity_Z=impurity_Z, ni_source=ni_source)
    _ipsi = np.asarray(ida.psi_N, dtype=float)
    g = lambda a: np.interp(psi_N, _ipsi, np.asarray(a, dtype=float))
    ne, ni, te, ti = g(ida.ne), g(ida.ni), g(ida.te), g(ida.ti)
    zeff = np.asarray(Zeff_fuse, dtype=float) if zeff_from_fuse else g(ida.Zeff)
    sigma_ne, sigma_te, sigma_ni, sigma_ti = (
        g(ida.sigma_ne), g(ida.sigma_te), g(ida.sigma_ni), g(ida.sigma_ti))
    if ni_from_imas_Zeff:
        # ni from FUSE Z_eff + IDA ne (single-impurity dilution; Z_imp = machine charge)
        ni = main_ion_density_from_zeff(ne, np.clip(Zeff_fuse, 1.0, impurity_Z), impurity_Z)
    omega = _read_ida_omega(ida_path, time, psi_N)
    return ne, te, ti, ni, zeff, omega, sigma_ne, sigma_te, sigma_ni, sigma_ti


def read_imas_baseline(
    source: "ImasSource",
    fixed: Optional["FixedComponentsConfig"] = None,
    p_fast_reduction: str = "trace",
    allow_incomplete_pressure: bool = False,
    anchor_jtor_to_equilibrium: bool = True,
    kinetic_source: str = "fuse",
    anchor_pressure_to_equilibrium: bool = False,
) -> "Baseline":
    """Read a FUSE ``dd_sim.json`` IDS and return a separated :class:`Baseline`.

    No Grad-Shafranov reconstruction is performed -- provenance is "imas".
    """
    import json
    from ..baseline import Baseline

    with open(source.ids_path, "rb") as fh:
        raw_bytes = fh.read()
    dd = json.loads(raw_bytes)
    T = source.time

    # --- targets from the equilibrium IDS ---
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], T, "equilibrium")
    gq = eq["time_slice"][ie]["global_quantities"]
    Ip_target = abs(float(gq["ip"]))
    ids_li_1 = float(gq["li_1"]) if "li_1" in gq else None
    ids_li_3 = float(gq["li_3"]) if "li_3" in gq else None
    # Provisional l_i_target = IDS li_3; the IMAS forward-solve (Bouquet) replaces
    # this with the TokaMaker-solved li_1 and records the IDS values for sanity.
    l_i_target = ids_li_3 if ids_li_3 is not None else 0.0

    # --- profiles + currents from core_profiles ---
    cp_ids = dd["core_profiles"]
    ic = _nearest_index(cp_ids["time"], T, "core_profiles")
    cp = cp_ids["profiles_1d"][ic]

    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psi_N = (psi - psi[0]) / (psi[-1] - psi[0])   # 0 (axis) -> 1 (boundary)
    n = psi_N.size

    j_total = np.asarray(cp["j_total"], dtype=float)   # total parallel
    j_tor = np.asarray(cp["j_tor"], dtype=float)       # total toroidal (authoritative)
    j_ohmic = np.asarray(cp["j_ohmic"], dtype=float)   # parallel (unused: inductive = residual)
    j_boot = np.asarray(cp["j_bootstrap"], dtype=float)  # parallel

    def to_toroidal(j_par):
        return parallel_to_toroidal(j_par, j_parallel_total=j_total, j_tor_total=j_tor)

    j_BS = to_toroidal(j_boot)

    # --- NBI: sum beam-source parallel currents, then convert ---
    src_ids = dd.get("core_sources", {})
    isrc = _nearest_index(src_ids["time"], T, "core_sources") if src_ids.get("time") else ic
    jnbi_par = np.zeros(n)
    for s in src_ids.get("source", []):
        if s.get("identifier", {}).get("index") == NBI_SOURCE_INDEX:
            pr = s.get("profiles_1d", [])
            if pr:
                idx = isrc if len(pr) > isrc else 0
                jnbi_par = jnbi_par + np.asarray(pr[idx]["j_parallel"], dtype=float)
    j_NBI = to_toroidal(jnbi_par)
    j_RF = np.zeros(n)   # never computed internally; user-supplied only

    # --- kinetic profiles + Zeff + fast pressure ---
    el = cp["electrons"]
    ne = np.asarray(el["density_thermal"], dtype=float)
    te = np.asarray(el["temperature"], dtype=float)   # eV
    p_fast = _isotropic_fast_pressure(el, p_fast_reduction, n)

    ni = None
    ti = None
    main_ion = None
    zeff_num = np.zeros(n)
    for ion in cp["ion"]:
        Z = float(ion["element"][0]["z_n"])
        n_s = np.asarray(ion["density_thermal"], dtype=float)
        zeff_num += n_s * Z * Z
        p_fast = p_fast + _isotropic_fast_pressure(ion, p_fast_reduction, n)
        if Z == 1.0 and ni is None:        # main (hydrogenic) ion
            ni = n_s
            ti = np.asarray(ion["temperature"], dtype=float)
            main_ion = ion
    if ni is None:
        raise ValueError("no hydrogenic (Z=1) main ion found in core_profiles.ion")
    Zeff = zeff_num / ne

    # --- auxiliary source-provided profiles for the switchboard ---------------
    # Read whatever this source carries (production FUSE files have rotation;
    # chi/E_r are typically absent and supplied via aux_baselines). All on the
    # core_profiles grid (== psi_N_kinetic for IMAS).
    aux = {"zeff": np.asarray(cp["zeff"], dtype=float) if "zeff" in cp else Zeff}
    if main_ion is not None and "rotation_frequency_tor" in main_ion:
        aux["omega_tor"] = np.asarray(main_ion["rotation_frequency_tor"], dtype=float)
    if "e_field" in cp and "radial" in cp["e_field"]:
        aux["e_r"] = np.asarray(cp["e_field"]["radial"], dtype=float)
    ctids = dd.get("core_transport")
    if ctids and ctids.get("model"):
        cpr = ctids["model"][0].get("profiles_1d", [])
        ict = (_nearest_index(ctids["time"], T, "core_transport")
               if ctids.get("time") else ic)
        if cpr:
            ctsl = cpr[ict if len(cpr) > ict else 0]
            ce = ctsl.get("electrons", {}).get("energy", {})
            if "d" in ce:
                aux["chi_e"] = np.asarray(ce["d"], dtype=float)
            if "d" in ctsl.get("total_ion_energy", {}):
                aux["chi_i"] = np.asarray(ctsl["total_ion_energy"]["d"], dtype=float)

    # --- IDA-hybrid: swap FUSE ne/Te/Ti/Zeff/omega for externally-fit IDA profiles ---
    # ni via source.ni_source; Zeff from IDA unless source.zeff_from_fuse. Done
    # before the pressure block so p_recon/Z_imp/p_imp use the IDA kinetics. IDA
    # sigmas land in aux as sigma_*_ida (informational -- resolve_uncertainty still
    # needs UncertaintyConfig.ida_path for the actual generation envelope).
    use_ida = bool(kinetic_source == "ida_hybrid" and getattr(source, "ida_path", None))
    if use_ida:
        (ne, te, ti, ni, Zeff, _omega,
         sigma_ne_ida, sigma_te_ida, sigma_ni_ida, sigma_ti_ida) = _merge_ida_kinetics(
            psi_N, ne, ni, Zeff, source.ida_path, T,
            getattr(source, "impurity_Z", 6.0),
            ni_source=getattr(source, "ni_source", "all"),
            zeff_from_fuse=getattr(source, "zeff_from_fuse", False))
        if _omega is not None:
            aux["omega_tor"] = _omega
        aux["zeff"] = Zeff   # keep the switchboard's zeff baseline consistent
        aux["sigma_ne_ida"] = sigma_ne_ida
        aux["sigma_te_ida"] = sigma_te_ida
        aux["sigma_ni_ida"] = sigma_ni_ida
        aux["sigma_ti_ida"] = sigma_ti_ida

    # --- user overrides for fixed additive components ---
    if fixed is not None:
        if fixed.p_fast is not None:
            p_fast = _override(fixed.p_fast, fixed.psi_N, psi_N)
        if fixed.j_NBI is not None:
            j_NBI = _override(fixed.j_NBI, fixed.psi_N, psi_N)
        if fixed.j_RF is not None:
            j_RF = _override(fixed.j_RF, fixed.psi_N, psi_N)

    # Authoritative toroidal total; inductive absorbs the residual so the
    # decomposition sums exactly and Ip is preserved.
    j_phi = j_tor.copy()
    j_inductive = j_phi - j_BS - j_NBI - j_RF

    # --- pressure anchor ("diff" approach) + completeness validation ----------
    # The authoritative dd equilibrium pressure (GS-consistent total, incl.
    # thermal impurity + isotropized fast) interpolated onto psi_N. The solve
    # builds e*(ne*Te + ni*Ti) + impurity(nz*Ti) + p_fast; p_diff anchors that to
    # FUSE exactly (mirrors jBS_diff) so the per-draw kinetics perturb the thermal
    # while the carbon/fast/GS residual rides as a fixed offset. Z_imp is the
    # single effective impurity charge (one-Zeff single-impurity model).
    eqp1 = eq["time_slice"][ie]["profiles_1d"]
    psi_eq = np.asarray(eqp1["psi"], dtype=float)
    psiN_eq = (psi_eq - psi_eq[0]) / (psi_eq[-1] - psi_eq[0])
    _o = np.argsort(psiN_eq)
    p_equilibrium = np.interp(psi_N, psiN_eq[_o],
                              np.asarray(eqp1["pressure"], dtype=float)[_o])
    # With IDA-hybrid kinetics, ni was built (read_ida / main_ion_density_from_zeff)
    # under single-impurity quasineutrality at charge source.impurity_Z, so that IS
    # the impurity charge.
    Z_imp = (float(getattr(source, "impurity_Z", 6.0)) if use_ida
             else effective_impurity_charge(ne, ni, Zeff))
    p_imp = impurity_pressure(ne, ni, ti, Z_imp)
    p_recon = _EC * (ne * te + ni * ti) + p_imp + p_fast
    # p_diff anchors the solve thermal pressure to the FUSE equilibrium.pressure.
    # Gated OFF by default: with IDA-hybrid kinetics we trust IDA's pressure and do
    # NOT force it back onto the FUSE total. When None, no anchor offset is added.
    p_diff = (p_equilibrium - p_recon) if anchor_pressure_to_equilibrium else None
    # The completeness guard compares the FUSE thermal/fast pressure against the FUSE
    # equilibrium.pressure; it is meaningless once ne/Te/Ti are swapped to IDA.
    if kinetic_source != "ida_hybrid":
        _validate_pressure_completeness(cp, ne, te, ni, ti, p_fast, p_imp,
                                        p_equilibrium, allow_incomplete_pressure)

    # --- total-current anchor: equilibrium.j_tor vs core_profiles.j_tor --------
    # core_profiles.j_tor (== j_phi here) is the transport parallel-current sum
    # (QED-diffused ohmic + Sauter bootstrap + NBI) converted to toroidal; it
    # differs from the GS-consistent equilibrium.j_tor (which GPEC reads) from
    # ~q=2 outward (the equilibrium carries more pedestal current). jphi_diff
    # anchors the total to the equilibrium as a FIXED offset (mirrors jBS_diff):
    # added to the baseline + every draw while the SWB bootstrap / perturbed
    # j_inductive ride underneath. jphi_diff integrates to ~0 (both totals carry
    # the same Ip), so it redistributes rather than adds net current.
    jphi_diff = None
    if anchor_jtor_to_equilibrium:
        eq_jtor = np.interp(psi_N, psiN_eq[_o],
                            np.asarray(eqp1["j_tor"], dtype=float)[_o])
        jphi_diff = eq_jtor - j_phi

    return Baseline(
        psi_N=psi_N,
        j_phi=j_phi,
        j_inductive=j_inductive,
        j_BS=j_BS,
        psi_N_kinetic=psi_N,
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        Ip_target=Ip_target,
        l_i_target=l_i_target,
        provenance="imas",
        j_NBI=j_NBI,
        j_RF=j_RF,
        p_fast=p_fast,
        p_equilibrium=p_equilibrium,
        p_diff=p_diff,
        Z_imp=Z_imp,
        jphi_diff=jphi_diff,
        eqdsk_bytes=None,
        # The OMAS JSON is NOT an Osborne p-file; don't pass it as pfile_bytes
        # (generate_bouquet would try to parse it). Per-draw p-files are built
        # from the kinetic profiles.
        pfile_bytes=None,
        li_metrics={"ids_li_1": ids_li_1, "ids_li_3": ids_li_3},
        aux=aux,
    )


# ===========================================================================
#  Perturbed-draw IMAS/OMAS write-back
#
#  Current-split fidelity: the parallel split (j_ohmic/j_bootstrap/j_total) is
#  now EXACT per draw (fidelity="exact"/"auto") -- it uses the draw's OWN
#  flux-surface geometry, captured from the live TokaMaker equilibrium at
#  generate time (physics.capture_equilibrium_fsa -> the eq_fsa archive block)
#  and applied via physics.toroidal_to_parallel. The legacy baseline-ratio
#  c(psi)=j_tor/j_total (fidelity="reconstruct") is kept as a fallback for
#  archives written without capture. j_tor is exact either way.
#
#  Remaining refinements (not blockers): (1) the EQUILIBRIUM IDS profiles_2d
#  still come from the archived 257^2 eqdsk (lossless to that grid,
#  machine-precision GS) rather than the live FE fields -- a direct OFT ODS
#  export would upgrade this; (2) exact <1/R^2> is computed by flux-surface
#  quadrature since TokaMaker does not yet expose it
#  (OpenFUSIONToolkit/OpenFUSIONToolkit#312) -- when it does, read it directly.
# ===========================================================================
def _imas_b0(out, ie, ic):
    """Reference vacuum field B0 for the IMAS <j.B>/B0 normalisation.

    Tries core_profiles then equilibrium ``vacuum_toroidal_field.b0`` (nearest
    time index); falls back to 1.0 (raw <j.B>) if neither is present.
    """
    for ids_name, it in (("core_profiles", ic), ("equilibrium", ie)):
        vtf = out.get(ids_name, {}).get("vacuum_toroidal_field")
        if vtf and vtf.get("b0") is not None:
            b0 = np.asarray(vtf["b0"], dtype=float)
            if b0.size:
                return abs(float(b0[min(it, b0.size - 1)]))
    return 1.0


def _eq_fsa_geom_on(eq_fsa, psiN_t, B0):
    """Interpolate a captured eq_fsa block onto the template psi grid -> geom
    dict for :func:`bouquet.physics.toroidal_to_parallel`. ``None`` if the
    block lacks the required FSA metrics (caller then reconstructs)."""
    try:
        src = np.asarray(eq_fsa["psi_N"], dtype=float)
        F = np.interp(psiN_t, src, np.asarray(eq_fsa["F"], dtype=float))
        avg_inv_R = np.interp(psiN_t, src, np.asarray(eq_fsa["avg_inv_R"], dtype=float))
        avg_B2 = np.interp(psiN_t, src, np.asarray(eq_fsa["avg_B2"], dtype=float))
    except (KeyError, TypeError):
        return None
    geom = {"F": F, "avg_inv_R": avg_inv_R, "avg_B2": avg_B2, "B0": float(B0)}
    if eq_fsa.get("avg_inv_R2") is not None:     # exact bracket when captured
        geom["avg_inv_R2"] = np.interp(
            psiN_t, src, np.asarray(eq_fsa["avg_inv_R2"], dtype=float))
    return geom


def write_imas_draw(h5path_or_header, draw_index, template_ids_path, out_path,
                    scan_key=None, time=None, fidelity="auto"):
    """Reconstruct a perturbed IMAS/OMAS IDS for one draw from the bouquet HDF5.

    Maps the draw's archived eqdsk to the ``equilibrium`` IDS
    (``profiles_1d`` / ``profiles_2d`` / ``global_quantities`` / ``boundary`` --
    lossless to the eqdsk grid, machine-precision GS) and the draw's ``.h5``
    kinetics/currents to ``core_profiles``. ``j_tor`` is exact.

    The parallel split (``j_total`` / ``j_ohmic`` / ``j_bootstrap`` =
    IMAS ``<j.B>/B0``) fidelity is set by ``fidelity``:

      * ``"exact"``       -- convert each toroidal component with the draw's OWN
        captured flux-surface geometry (``eq_fsa`` block, from
        ``capture_live_eq=True`` at generate time) via
        :func:`bouquet.physics.toroidal_to_parallel`. Raises if the block is
        absent.
      * ``"reconstruct"`` -- the interim baseline ratio ``c = j_tor/j_total``
        from the template (exact only when the draw's flux geometry matches the
        baseline's).
      * ``"auto"`` (default) -- exact when the ``eq_fsa`` block is present,
        else reconstruct.

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.
    draw_index : int
        Per-draw index within the scan group.
    template_ids_path : str
        The source IMAS/OMAS JSON, used as the structural template.
    out_path : str
        Where to write the perturbed IDS JSON.
    fidelity : {"auto", "exact", "reconstruct"}
        Parallel-current split fidelity (see above); default ``"auto"``.
    scan_key : str, float, or None
        Scan value selecting the ``scan/<scan_key>`` group.  The default
        ``None`` auto-resolves a single-scan archive (and is the flat layout
        for flat files); pass the explicit value only when the archive holds
        more than one scan.
    time : float, optional
        Time slice [s]; defaults to the template's nearest single slice.

    Returns
    -------
    str
        ``out_path``.
    """
    import json
    import copy
    import h5py
    from .geqdsk import read_geqdsk
    from ..utils import read_eqdsk_from_bytes, _group_path, _resolve_h5

    if fidelity not in ("auto", "exact", "reconstruct"):
        raise ValueError(
            f"fidelity must be 'auto'|'exact'|'reconstruct', got {fidelity!r}")

    with open(template_ids_path) as fh:
        out = json.load(fh)

    eq_ids = out["equilibrium"]
    ie = _nearest_index(eq_ids["time"], time, "equilibrium")
    cp_ids = out["core_profiles"]
    ic = _nearest_index(cp_ids["time"], time, "core_profiles")

    h5 = _resolve_h5(h5path_or_header)
    if scan_key is None:
        # Convenience: a single-scan archive resolves unambiguously, so the
        # caller need not echo the generation scan_key.  Flat-layout files
        # (discover_scan_keys -> None) keep scan_key=None.
        from ..utils import discover_scan_keys
        keys = discover_scan_keys(h5)
        if keys:
            if len(keys) != 1:
                raise ValueError(
                    f"{h5} holds {len(keys)} scans {keys}; pass an explicit "
                    "scan_key to write_imas_draw().")
            scan_key = keys[0]
    gp = _group_path(scan_key, draw_index)
    with h5py.File(h5, "r") as hf:
        if gp not in hf:
            raise KeyError(f"draw {draw_index} (scan {scan_key}) not in {h5}")
        g = hf[gp]
        ne = np.asarray(g["n_e"][()]); te = np.asarray(g["T_e"][()])
        ni = np.asarray(g["n_i"][()]); ti = np.asarray(g["T_i"][()])
        pkin = np.asarray(g["psi_N_kinetic"][()]); peq = np.asarray(g["psi_N"][()])
        j_tor = np.asarray(g["j_phi"][()])
        j_ind = np.asarray(g["j_inductive"][()])
        j_bs = np.asarray(g["j_BS"][()])
        zeff = np.asarray(g["aux_zeff"][()]) if "aux_zeff" in g else None
        li1 = float(g.attrs.get("l_i(1)", np.nan))
        li3 = float(g.attrs.get("l_i(3)", np.nan))
        from ..schema import find_bytes_dataset, EQ_FSA_GROUP
        eqk = find_bytes_dataset(g)
        if eqk is None:
            raise KeyError(f"draw {draw_index} has no archived eqdsk")
        geq = read_eqdsk_from_bytes(bytes(g[eqk][()]), read_geqdsk)
        # optional captured live-equilibrium FSA block (exact conversion)
        eq_fsa = None
        if EQ_FSA_GROUP in g:
            eq_fsa = {k: np.asarray(g[EQ_FSA_GROUP][k][()], dtype=float)
                      for k in g[EQ_FSA_GROUP]}

    # --- equilibrium IDS from the eqdsk (lossless to the eqdsk grid) ---------
    ts = eq_ids["time_slice"][ie]
    psi1d = geq.psi_axis + geq.psi_N * (geq.psi_boundary - geq.psi_axis)
    q95 = float(np.interp(0.95, geq.psi_N, geq.qpsi))
    ts["profiles_1d"] = {
        "psi": psi1d.tolist(),
        "q": geq.qpsi.tolist(),
        "pressure": geq.pres.tolist(),
        "f": geq.fpol.tolist(),
        "dpressure_dpsi": geq.pprime.tolist(),
        "f_df_dpsi": geq.ffprim.tolist(),
    }
    ts["profiles_2d"] = [{
        "grid_type": {"name": "rectangular", "index": 1},
        "grid": {"dim1": geq.R_grid.tolist(), "dim2": geq.Z_grid.tolist()},
        # psi_RZ is indexed [R][Z], matching IMAS dim1=R, dim2=Z
        "psi": geq.psi_RZ.tolist(),
    }]
    gq = dict(ts.get("global_quantities", {}))
    gq.update(
        ip=float(geq.Ip), psi_axis=float(geq.psi_axis),
        psi_boundary=float(geq.psi_boundary),
        magnetic_axis={"r": float(geq.R_mag), "z": float(geq.Z_mag)},
        q_axis=float(geq.qpsi[0]), q_95=q95,
        li_3=li3 if np.isfinite(li3) else geq.li.get("li(3)"),
        beta_normal=geq.betas.get("beta_n"), beta_pol=geq.betas.get("beta_p"),
        beta_tor=geq.betas.get("beta_t"),
    )
    if np.isfinite(li1):
        gq["li_1"] = li1
    ts["global_quantities"] = gq
    ts["boundary"] = {"outline": {"r": geq.boundary_R.tolist(),
                                  "z": geq.boundary_Z.tolist()}}

    # --- core_profiles from the draw (kinetics + currents) ------------------
    cp = cp_ids["profiles_1d"][ic]
    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])

    def to_t(arr, src):     # interp draw array (on src grid) -> template psi grid
        return np.interp(psiN_t, src, arr)

    cp["electrons"]["density_thermal"] = to_t(ne, pkin).tolist()
    cp["electrons"]["temperature"] = to_t(te, pkin).tolist()
    for ion in cp["ion"]:
        if float(ion["element"][0]["z_n"]) == 1.0:
            ion["density_thermal"] = to_t(ni, pkin).tolist()
            ion["temperature"] = to_t(ti, pkin).tolist()
            break
    if zeff is not None:
        cp["zeff"] = to_t(zeff, pkin).tolist()

    # j_tor is exact (bouquet stores toroidal current directly).
    jt_t = to_t(j_tor, peq)
    cp["j_tor"] = jt_t.tolist()

    # Parallel split (j_total / j_ohmic / j_bootstrap = IMAS <j.B>/B0). Two
    # fidelities (`fidelity` arg): EXACT uses the draw's own captured
    # flux-surface geometry (eq_fsa) via physics.toroidal_to_parallel;
    # RECONSTRUCT falls back to the interim baseline ratio c=j_tor/j_total from
    # the template (exact only when the draw's flux geometry matches baseline).
    if "j_total" in cp and "j_tor" in cp:
        use_exact = False
        if fidelity in ("auto", "exact") and eq_fsa is not None:
            geom = _eq_fsa_geom_on(eq_fsa, psiN_t, _imas_b0(out, ie, ic))
            if geom is not None:
                from ..physics import toroidal_to_parallel
                cp["j_total"] = toroidal_to_parallel(jt_t, geom=geom).tolist()
                cp["j_ohmic"] = toroidal_to_parallel(to_t(j_ind, peq), geom=geom).tolist()
                cp["j_bootstrap"] = toroidal_to_parallel(to_t(j_bs, peq), geom=geom).tolist()
                use_exact = True
        if fidelity == "exact" and not use_exact:
            raise ValueError(
                f"fidelity='exact' requested but draw {draw_index} has no "
                "captured eq_fsa block (generate with capture_live_eq=True). "
                "Use fidelity='auto' to fall back to the baseline-ratio "
                "reconstruction.")
        if not use_exact:                      # baseline-ratio reconstruction
            base_jtot = np.asarray(cp["j_total"], dtype=float)
            base_jtor = np.asarray(cp["j_tor"], dtype=float)
            eps = 1e-9 * np.nanmax(np.abs(base_jtot)) if base_jtot.size else 0.0
            good = np.abs(base_jtot) > eps
            c = np.ones_like(base_jtot)
            c[good] = base_jtor[good] / base_jtot[good]
            if not np.all(good):
                idx = np.arange(c.size)
                c[~good] = np.interp(idx[~good], idx[good], c[good])
            with np.errstate(divide="ignore", invalid="ignore"):
                cp["j_total"] = (jt_t / c).tolist()
                cp["j_ohmic"] = (to_t(j_ind, peq) / c).tolist()
                cp["j_bootstrap"] = (to_t(j_bs, peq) / c).tolist()

    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out_path


def export_imas_drawset(h5path_or_header, template_ids_path, out_dir,
                        scan_key=None, time=None, selection="selected",
                        fidelity="auto"):
    """Write one perturbed IMAS/OMAS IDS per draw (see :func:`write_imas_draw`).

    Files are ``{out_dir}/{header}_draw{idx}.json``. ``selection`` is
    ``"selected"`` (in-spec only) or ``"all"``. ``fidelity`` is forwarded to
    :func:`write_imas_draw` (``"auto"`` -> exact per-draw conversion when the
    archive carries the captured ``eq_fsa`` block, else baseline-ratio).

    Operates on a single scan.  Pass ``scan_key`` to select it; the default
    ``scan_key=None`` is the flat layout, and is only unambiguous when the
    file holds exactly one scan (otherwise raises -- pass an explicit key).

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.

    Returns
    -------
    list of str
        The written IDS paths.
    """
    import os
    from ..filtering import select_indices
    from ..utils import _resolve_h5

    os.makedirs(out_dir, exist_ok=True)
    h5 = _resolve_h5(h5path_or_header)
    base = os.path.basename(h5).replace(".h5", "")
    idxs = select_indices(h5, scan_key=scan_key, selection=selection)
    if isinstance(idxs, dict):
        # scan_key=None over a scan-layout file -> ambiguous unless single.
        if len(idxs) != 1:
            raise ValueError(
                f"{h5} holds {len(idxs)} scans {sorted(idxs)}; pass an "
                "explicit scan_key to export_imas_drawset().")
        scan_key, idxs = next(iter(idxs.items()))
    paths = []
    for i in idxs:
        out = os.path.join(out_dir, f"{base}_draw{i}.json")
        paths.append(write_imas_draw(h5, i, template_ids_path, out,
                                     scan_key=scan_key, time=time,
                                     fidelity=fidelity))
    return paths
