"""
HDF5 archive helpers and eqdsk I/O utilities for perturbed equilibria.
"""

import os
import tempfile

import h5py
import numpy as np


# ====================================================================
#  Internal helpers
# ====================================================================

def Ip_flux_integral_vs_target(alpha, jtor_prof, spike_profile, psi_N, Ip_target):
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

def _baseline_key(baseline):
    """Convert a baseline label (float, int, or str) to an HDF5-safe string.

    Returns ``None`` when *baseline* is ``None`` (flat layout).
    """
    if baseline is None:
        return None
    return str(baseline)


def _group_path(baseline, count):
    """Return the internal HDF5 group path for a given entry."""
    bkey = _baseline_key(baseline)
    if bkey is not None:
        return f"scan/{bkey}/{int(count)}"
    return str(int(count))


def _eqdsk_dataset_name(header, baseline, count):
    """Return the dataset name used for the raw eqdsk bytes."""
    base = os.path.basename(header)
    bkey = _baseline_key(baseline)
    if bkey is not None:
        safe_key = bkey.replace("/", "_").replace(" ", "_")
        return f"{base}_{safe_key}_{int(count)}.eqdsk"
    return f"{base}_{int(count)}.eqdsk"


# ====================================================================
#  Database lifecycle
# ====================================================================
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


# ====================================================================
#  Per-equilibrium storage
# ====================================================================
_PROFILE_KEYS = [
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
    pressure=None,
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
    baseline : str, float, int, or None
        Scan-point label.  When provided, an extra ``scan/{label}/``
        group layer is inserted.  ``None`` gives the flat layout.
    pressure : array_like or None
        1-D total pressure [Pa].  Optional for backward compatibility
        with older workflows.
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

        # ---- raw eqdsk (opaque binary -- bit-perfect) --------------------
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

        if pressure is not None:
            grp.create_dataset("pressure [Pa]", data=np.asarray(pressure, dtype=np.float64))

        # ---- scalars (group attributes) ----------------------------------
        grp.attrs["l_i(1)"] = float(li1)
        grp.attrs["l_i(3)"] = float(li3)
        grp.attrs["count"]  = int(count)
        if baseline is not None:
            grp.attrs["baseline"] = baseline


def load_equilibrium(header, count, baseline=None, eqdsk_out_dir=None):
    """
    Retrieve one equilibrium entry from the HDF5 database.

    Parameters
    ----------
    header : str
        Base name of the database.
    count : int
        Perturbation index.
    baseline : str, float, int, or None
        Scan-point label (must match what was used at write time).
    eqdsk_out_dir : str or None, optional
        If given, the raw eqdsk is written to a file in this directory.

    Returns
    -------
    result : dict
        Keys: ``"eqdsk_filepath"``, ``"eqdsk_bytes"``,
        the 1-D array names, ``"l_i(1)"``, ``"l_i(3)"``,
        and optionally ``"pressure [Pa]"``.
    """
    db_path  = os.path.abspath(f"{header}.h5")
    grp_path = _group_path(baseline, count)
    ds_name  = _eqdsk_dataset_name(header, baseline, count)

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
        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure [Pa]" in grp:
            result["pressure [Pa]"] = np.array(grp["pressure [Pa]"])

        # ---- scalars ----------------------------------------------------
        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

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
    baseline=None,
):
    """
    Store the input (baseline) profiles and their uncertainties.

    For hierarchical layout (*baseline* is not ``None``), these are
    stored in ``scan/{label}/_baseline/``.  For flat layout they go
    in ``_baseline/``.

    This data is written once per scan-point and is required by the
    plotting GUI to be fully self-contained.
    """
    db_path = os.path.abspath(f"{header}.h5")
    bkey = _baseline_key(baseline)

    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    with h5py.File(db_path, "a") as hf:
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        grp.create_dataset("psi_N",              data=np.asarray(psi_N,      dtype=np.float64))
        grp.create_dataset("n_e [m^-3]",         data=np.asarray(ne,         dtype=np.float64))
        grp.create_dataset("T_e [eV]",           data=np.asarray(te,         dtype=np.float64))
        grp.create_dataset("n_i [m^-3]",         data=np.asarray(ni,         dtype=np.float64))
        grp.create_dataset("T_i [eV]",           data=np.asarray(ti,         dtype=np.float64))
        grp.create_dataset("pressure [Pa]",       data=np.asarray(pressure,   dtype=np.float64))
        grp.create_dataset("j_phi [A/m^2]",      data=np.asarray(j_phi,      dtype=np.float64))
        grp.create_dataset("sigma_ne [m^-3]",    data=np.asarray(sigma_ne,   dtype=np.float64))
        grp.create_dataset("sigma_te [eV]",      data=np.asarray(sigma_te,   dtype=np.float64))
        grp.create_dataset("sigma_ni [m^-3]",    data=np.asarray(sigma_ni,   dtype=np.float64))
        grp.create_dataset("sigma_ti [eV]",      data=np.asarray(sigma_ti,   dtype=np.float64))
        grp.create_dataset("sigma_jphi [A/m^2]", data=np.asarray(sigma_jphi, dtype=np.float64))

        grp.attrs["Ip_target"]  = float(Ip_target)
        grp.attrs["l_i_target"] = float(l_i_target)


# ====================================================================
#  Introspection helpers (used by GUI and notebook API)
# ====================================================================
def discover_scan_values(h5path):
    """
    Discover all scan values in an HDF5 equilibrium database.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.

    Returns
    -------
    scan_values : list[str] or None
        Sorted list of scan-value keys, or ``None`` if the file uses
        the flat layout (no ``scan/`` group).
    """
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


def count_equilibria(h5path, scan_value=None):
    """
    Count the number of perturbed equilibria stored for a scan value.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.
    scan_value : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    n : int
    """
    bkey = _baseline_key(scan_value)
    with h5py.File(h5path, "r") as hf:
        if bkey is not None:
            parent = hf[f"scan/{bkey}"]
        else:
            parent = hf
        # Count integer-named groups (skip _baseline and other metadata)
        return sum(1 for k in parent.keys()
                   if k not in ("_baseline", "scan"))


def load_baseline_profiles(h5path, scan_value=None):
    """
    Load the baseline profiles and uncertainties for a given scan value.

    Parameters
    ----------
    h5path : str
        Path to the ``.h5`` file.
    scan_value : str, float, or None
        ``None`` for flat-layout files.

    Returns
    -------
    result : dict
        All stored baseline arrays and scalar attributes.
    """
    bkey = _baseline_key(scan_value)
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


def load_equilibrium_by_path(h5path, count, scan_value=None):
    """
    Load one perturbed equilibrium from an HDF5 file by path.

    Like :func:`load_equilibrium` but takes a file path instead of a
    header string, and uses *scan_value* instead of *baseline*.  Does
    **not** extract the raw eqdsk bytes (use :func:`load_equilibrium`
    if you need those).
    """
    bkey = _baseline_key(scan_value)
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

        if "pressure [Pa]" in grp:
            result["pressure [Pa]"] = np.array(grp["pressure [Pa]"])

        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

    return result


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
