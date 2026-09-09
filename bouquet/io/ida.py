"""Reader for IDA integrated-data-analysis files (DIII-D, netCDF ``.cdf``).

The IDA ``.cdf`` is netCDF4 -- i.e. an HDF5 container -- so it is read with
**h5py alone** (already a bouquet dependency); no OMFIT / netCDF4 / OMFITnc
required. Datasets map directly: ``f['n_e'][:]``, ``f['n_e_err'][:]``, etc.

Returns the kinetic profiles together with their uncertainty (sigma) profiles,
on the IDA psi_N grid. Both the baseline reconstruction (profiles) and the
uncertainty envelope (sigmas) draw from this single read.

Operational DIII-D ``IDA_*.cdf`` layout (verified against a real IDA file):
    profiles are 2-D ``(n_time, n_radial)`` with companion ``*_err`` datasets
    (direct 1-sigma); the radial grid is ``psi_n`` (n_radial,), extending past
    the separatrix to ~1.2; ``time`` is in milliseconds. Units are already SI
    (n_e in m^-3; T_e, T_12C6 in eV).

There is no stored main-ion density; ``ni`` can come from ``Zeff``
reconstructed from visible bremsstrahlung data (``ni_source="Zeff"``), from
the measured carbon density ``n_12C6`` from charge exchange recombination
(``ni_source="CER"``), or from the mean of the two (``ni_source="all"``, default).
With both active, their disagreement beyond statistical error widens
``sigma_ni``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IDAProfiles:
    """Kinetic profiles + sigmas read from an IDA file, at one time slice.

    Values are returned in SI (ne/ni in m^-3, Te/Ti in eV) on ``psi_N``.
    ``psi_N`` typically extends past the separatrix (~1.2) into the SOL; pass it
    through to ``generate_bouquet`` as ``psi_N_kinetic`` rather than truncating.
    """

    psi_N: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    ni: np.ndarray
    ti: np.ndarray
    Zeff: np.ndarray

    sigma_ne: np.ndarray
    sigma_te: np.ndarray
    sigma_ni: np.ndarray
    sigma_ti: np.ndarray
    sigma_Zeff: Optional[np.ndarray]    # None only if the file has no Zeff_err

    time: float                     # selected slice [s]
    raw_bytes: Optional[bytes] = None   # original file bytes for archival
    # Per-radius tension between the two ni routes, in sigma; >1 is what
    # widens sigma_ni. None unless ni_source="all".
    ni_route_chi: Optional[np.ndarray] = None


@dataclass
class IDACERProfiles:
    """Impurity (carbon CER) channels for a radial-field / rotation analysis.

    Everything needed for the impurity radial force balance
    ``E_r = (dp_C/dR)/(Z_C e n_C) - v_pol B_phi + omega R B_pol`` (see
    :func:`bouquet.physics.radial_field_from_impurity_force_balance`), read at one time slice on
    ``psi_N``. SI units: ``n_carbon`` m^-3, ``t_carbon`` eV, ``omega_tor`` rad/s,
    ``v_pol`` m/s, ``Bpol`` T, ``Rmaj`` m, ``dpsiN_dR`` 1/m. The ``sigma_*``
    fields are the measured 1-sigma envelopes (for propagating E_r uncertainty).
    """

    psi_N: np.ndarray
    n_carbon: np.ndarray
    t_carbon: np.ndarray
    omega_tor: np.ndarray
    v_pol: np.ndarray
    Bpol: np.ndarray
    Rmaj: np.ndarray
    dpsiN_dR: np.ndarray

    sigma_n_carbon: np.ndarray
    sigma_t_carbon: np.ndarray
    sigma_omega_tor: np.ndarray
    sigma_v_pol: np.ndarray

    time: float


def _select_time_index(time_ms: np.ndarray, time_s: Optional[float]) -> int:
    """Return the index of the slice nearest ``time_s`` (seconds)."""
    if time_ms.size == 1:
        return 0
    if time_s is None:
        avail = ", ".join(f"{t/1e3:.4f}" for t in time_ms)
        raise ValueError(
            f"IDA file has {time_ms.size} time slices; pass `time` (seconds). "
            f"Available [s]: {avail}"
        )
    return int(np.argmin(np.abs(time_ms / 1e3 - time_s)))


def read_ida(
    path: str,
    time: Optional[float] = None,
    sigma_mode: str = "auto",
    sigma_method: str = "percentile",   # ensemble-layout band estimator
    ensemble_median: bool = False,      # ensemble-layout central estimator
    ni_source: str = "all",
    impurity_Z: float = 6.0,
) -> IDAProfiles:
    """Read an IDA ``.cdf`` and return profiles + sigmas at ``time``.

    Parameters
    ----------
    path : str
        Path to the IDA netCDF file.
    time : float, optional
        Time slice in seconds. Required when the file holds more than one slice;
        the nearest slice is selected.
    sigma_mode : {"auto", "direct", "ensemble"}
        ``"auto"`` (default) picks the layout from the array dimensionality.
        ``"direct"`` reads the ``*_err`` datasets (2-D operational layout);
        ``"ensemble"`` reduces a 3-D posterior-sample file (mean profile + a
        sample-spread sigma). Passing a mode that contradicts the file raises.
    sigma_method : {"percentile", "std"}
        Ensemble band estimator: ``"percentile"`` -> (p84-p16)/2 (robust),
        ``"std"`` -> sample standard deviation. Unused for the direct layout.
    ensemble_median : bool
        Ensemble central estimator: sample mean (default) or median.
    ni_source : {"Zeff", "CER", "all"}
        Which measurement the main-ion density comes from. ``"Zeff"`` 
        applies single-impurity quasineutrality to ``(ne, Zeff)``; ``"CER"``
        subtracts the measured carbon density ``n_12C6``; ``"all"`` takes the
        mean of the two (default). Each route needs its own ``*_err`` dataset 
        on the direct layout, and raises without it.
    impurity_Z : float
        Impurity charge Z (carbon Z=6).

    Notes
    -----
    Opens with ``h5py.File(path, "r")`` -- the file is netCDF4/HDF5, so no
    OMFIT or netCDF4 package is needed. Units are already SI; ``T_12C6`` maps to
    Ti.

    For ``ni_source="all"``, ``sigma_ni`` also carries what the two routes
    disagree on beyond their statistical errors. The term is one-sided -- it
    only widens ``sigma_ni`` -- and ``ni_route_chi`` reports the tension behind
    it.
    """
    import h5py

    if sigma_mode not in ("direct", "ensemble", "auto"):
        raise ValueError(
            f"unknown sigma_mode {sigma_mode!r}; expected 'direct', 'ensemble', or 'auto'")
    if sigma_method not in ("percentile", "std"):
        raise ValueError(
            f"unknown sigma_method {sigma_method!r}; expected 'percentile' or 'std'")
    if ni_source not in ("Zeff", "CER", "all"):
        raise ValueError(
            f"unknown ni_source {ni_source!r}; expected 'Zeff', 'CER', or 'all'")
    # Which of the two dilution measurements the requested route(s) need.
    use_zeff = ni_source in ("Zeff", "all")
    use_carbon = ni_source in ("CER", "all")
    if use_carbon and float(impurity_Z) != 6.0:
        raise ValueError(
            f"ni_source={ni_source!r} requires impurity_Z=6.0, got {impurity_Z!r}: "
            "the carbon route subtracts the 'n_12C6' density, which is carbon "
            "(Z=6). For another impurity use ni_source='Zeff'")

    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    from ..physics import main_ion_density_from_zeff

    with h5py.File(path, "r") as f:
        time_ms = np.asarray(f["time"][:], dtype=float).ravel()
        t_idx = _select_time_index(time_ms, time)
        t_sel = float(time_ms[t_idx] / 1e3)

        # Two field-validated layouts, distinguished by dimensionality:
        #   direct   : (n_time, n_radial) profiles + companion *_err datasets;
        #   ensemble : (n_time, n_samples, n_radial) posterior samples, no *_err
        #              -> profile = sample centre, sigma = sample spread.
        is_ensemble = (np.asarray(f["n_e"].shape).size == 3)
        if sigma_mode == "direct" and is_ensemble:
            raise ValueError("sigma_mode='direct' but the file is a 3-D posterior "
                             "(ensemble) IDA; use sigma_mode='auto' or 'ensemble'")
        if sigma_mode == "ensemble" and not is_ensemble:
            raise ValueError("sigma_mode='ensemble' but the file is a 2-D direct "
                             "IDA; use sigma_mode='auto' or 'direct'")
        if use_carbon and "n_12C6" not in f:
            raise KeyError(
                f"{path!r} has no 'n_12C6' dataset, so ni cannot be derived from "
                "the carbon density; pass ni_source='Zeff' to use the "
                "(ne, Zeff) quasineutrality route instead")
        if use_carbon and not is_ensemble and "n_12C6_err" not in f:
            raise KeyError(
                f"{path!r} has 'n_12C6' but no 'n_12C6_err', so the carbon term of "
                "sigma_ni cannot be propagated; pass ni_source='Zeff' to use "
                "the (ne, Zeff) quasineutrality route instead")
        if use_zeff and not is_ensemble and "Zeff_err" not in f:
            raise KeyError(
                f"{path!r} has no 'Zeff_err' dataset, so the Zeff term of sigma_ni "
                "cannot be propagated; pass ni_source='CER' to derive ni from "
                "the carbon density instead")

        if is_ensemble:
            def _samples(key):  # (n_samples, n_radial) at the selected slice
                return np.asarray(f[key][t_idx], dtype=float)

            def _band(a):       # symmetric 1-sigma-equivalent over the sample axis
                if sigma_method == "std":
                    return np.std(a, axis=0)
                lo, hi = np.percentile(a, [16.0, 84.0], axis=0)
                return 0.5 * (hi - lo)

            def _center(a):
                return np.median(a, axis=0) if ensemble_median else np.mean(a, axis=0)

            psi_N = np.asarray(f["psi_n"][t_idx], dtype=float)[0]   # shared radial grid
            ne_s, te_s = _samples("n_e"), _samples("T_e")
            ti_s, zf_s = _samples("T_12C6"), _samples("Zeff")
            ne, te, ti, Zeff = _center(ne_s), _center(te_s), _center(ti_s), _center(zf_s)
            sigma_ne, sigma_te, sigma_ti, sigma_Zeff = _band(ne_s), _band(te_s), _band(ti_s), _band(zf_s)
            if use_carbon:
                nc_s = _samples("n_12C6")
                n_carbon, sigma_n_carbon = _center(nc_s), _band(nc_s)
        else:
            def col(key):       # one radial profile at the selected time
                return np.asarray(f[key][t_idx], dtype=float)

            psi_N = np.asarray(f["psi_n"][:], dtype=float)
            ne, te = col("n_e"), col("T_e")          # m^-3, eV
            ti, Zeff = col("T_12C6"), col("Zeff")    # eV (carbon CER), dimensionless
            sigma_ne, sigma_te, sigma_ti = col("n_e_err"), col("T_e_err"), col("T_12C6_err")
            if use_carbon:
                n_carbon, sigma_n_carbon = col("n_12C6"), col("n_12C6_err")
            # Read whenever present: only use_zeff *requires* it (preflight check above),
            # it is also returned as the aux Z_eff envelope.
            sigma_Zeff = col("Zeff_err") if "Zeff_err" in f else None

        # ni is derived from (ne, Zeff, n_C); propagate via that function's
        # Jacobian. Both routes depend on ne, so dni/dne is summed across
        # active routes before squaring.
        # cov(Zeff, n_C) = 0: IDA stores no covariance, and the two come from
        # separate diagnostics (visible bremsstrahlung vs CER). Derivatives
        # are evaluated unclipped, which is conservative where a clip is active.
        w = 0.5 if ni_source == "all" else 1.0   # equal weights for "all"
        ni = np.zeros_like(ne)
        d_ne = np.zeros_like(ne)                 # dni/dne, summed over routes
        terms = []                               # |dni/dx| sigma_x for x != ne

        if use_zeff:
            # Single-impurity quasineutrality: ni = ne (Z_imp - Zeff)/(Z_imp - 1).
            # Zeff comes directly from IDA (visible bremsstrahlung), so dilution
            # is measured, not assumed. Zeff is clipped to [1, Z_imp] so
            # 0 <= ni <= ne.
            Zeff_c = np.clip(Zeff, 1.0, impurity_Z)
            ni_zeff = main_ion_density_from_zeff(ne, Zeff_c, impurity_Z)
            dne_zeff = (impurity_Z - Zeff_c) / (impurity_Z - 1.0)   # dni/dne
            sig_zeff = ne / (impurity_Z - 1.0) * sigma_Zeff         # |dni/dZeff| sigma
            ni += w * ni_zeff
            d_ne += w * dne_zeff
            terms.append(w * sig_zeff)

        if use_carbon:
            # Dilution straight from the CER carbon density: ni = ne - Z_imp n_C.
            ni_cer = np.maximum(ne - impurity_Z * n_carbon, 0.0)
            sig_cer = impurity_Z * sigma_n_carbon                   # |dni/dn_C| sigma
            ni += w * ni_cer
            d_ne += w                                               # dni/dne = 1
            terms.append(w * sig_cer)

        terms.append(d_ne * sigma_ne)
        var_ni = sum(t ** 2 for t in terms)

        ni_route_chi = None
        if use_zeff and use_carbon:
            delta = ni_zeff - ni_cer
            # ne is shared, so it reaches delta only through the difference of
            # the two derivatives, not as two independent terms.
            var_delta = (((dne_zeff - 1.0) * sigma_ne) ** 2
                         + sig_zeff ** 2 + sig_cer ** 2)

            # Both routes are GP fits, so delta is smooth in psi_N: a nonzero
            # value is a coherent offset, not point-to-point scatter. max(., 0)
            # keeps the term one-sided, and ni is the mean of the two routes, so
            # an offset delta displaces it by delta/2 -> variance excess /4.
            var_ni = var_ni + np.maximum(delta ** 2 - var_delta, 0.0) / 4.0
            ni_route_chi = np.sqrt(np.divide(
                delta ** 2, var_delta, out=np.full_like(delta, np.nan),
                where=var_delta > 0.0))

        sigma_ni = np.sqrt(var_ni)

    return IDAProfiles(
        psi_N=psi_N,
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        sigma_ne=sigma_ne, sigma_te=sigma_te, sigma_ni=sigma_ni, sigma_ti=sigma_ti,
        sigma_Zeff=sigma_Zeff,
        time=t_sel,
        raw_bytes=raw_bytes,
        ni_route_chi=ni_route_chi,
    )


def read_ida_cer(
    path: str,
    time: Optional[float] = None,
    sigma_method: str = "percentile",
    ensemble_median: bool = False,
) -> IDACERProfiles:
    """Read the carbon-CER channels needed for a radial-field (E_r) analysis.

    Returns the impurity density / temperature, toroidal + poloidal rotation, the
    midplane poloidal field and geometry (``Rmaj``, ``dpsiN_dR``), and their
    measured 1-sigma envelopes, at ``time`` on the IDA ``psi_N`` grid. Handles
    both file layouts like :func:`read_ida`: direct (2-D + ``*_err``) and ensemble
    (3-D posterior samples -> central profile + ``sigma_method`` band). Feed the
    result to :func:`bouquet.physics.radial_field_from_impurity_force_balance`.
    ``ensemble_median`` matches :func:`read_ida`; set both alike.
    """
    import h5py

    if sigma_method not in ("percentile", "std"):
        raise ValueError(f"unknown sigma_method {sigma_method!r}")

    with h5py.File(path, "r") as f:
        time_ms = np.asarray(f["time"][:], dtype=float).ravel()
        t_idx = _select_time_index(time_ms, time)
        t_sel = float(time_ms[t_idx] / 1e3)
        is_ensemble = (np.asarray(f["n_e"].shape).size == 3)

        def _band(a):
            if sigma_method == "std":
                return np.std(a, axis=0)
            lo, hi = np.percentile(a, [16.0, 84.0], axis=0)
            return 0.5 * (hi - lo)

        def _center(a):
            return np.median(a, axis=0) if ensemble_median else np.mean(a, axis=0)

        def read(key, err_key=None):
            """(value, sigma) for one channel across either layout."""
            if is_ensemble:
                s = np.asarray(f[key][t_idx], dtype=float)      # (n_samples, n_radial)
                return _center(s), _band(s)
            val = np.asarray(f[key][t_idx], dtype=float)
            sig = (np.asarray(f[err_key][t_idx], dtype=float)
                   if err_key and err_key in f else np.zeros_like(val))
            return val, sig

        if is_ensemble:
            psi_N = np.asarray(f["psi_n"][t_idx], dtype=float)[0]
        else:
            psi_N = np.asarray(f["psi_n"][:], dtype=float)

        n_c, s_nc = read("n_12C6", "n_12C6_err")
        t_c, s_tc = read("T_12C6", "T_12C6_err")
        omg, s_om = read("omega_tor_12C6", "omega_tor_12C6_err")
        vpol, s_vp = read("v_pol", "v_pol_err")
        bpol, _ = read("Bpol_midplane", "Bpol_midplane_err")
        rmaj, _ = read("Rmaj_midplane", "Rmaj_midplane_err")
        dpsidr, _ = read("dPsiN_dR_midplane", "dPsiN_dR_midplane_err")

    return IDACERProfiles(
        psi_N=psi_N, n_carbon=n_c, t_carbon=t_c, omega_tor=omg, v_pol=vpol,
        Bpol=bpol, Rmaj=rmaj, dpsiN_dR=dpsidr,
        sigma_n_carbon=s_nc, sigma_t_carbon=s_tc,
        sigma_omega_tor=s_om, sigma_v_pol=s_vp,
        time=t_sel,
    )
