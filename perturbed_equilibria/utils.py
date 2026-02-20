import tempfile
import os


# ====================================================================
#  HDF5 archive helpers
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
        Perturbation index (typically 0 – N-1).
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
    baseline : int or None, optional
        Scan-point index.  When provided an extra ``baseline_XXX/``
        group layer is inserted.  ``None`` gives the flat layout.
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


def read_eqdsk_from_bytes(raw_bytes, reader_func):
    """
    Call an existing eqdsk reader that expects a filename,
    but feed it in-memory bytes instead of a file on disk.
    """
    # Write raw bytes to a temporary file, then pass its path
    # to the reader exactly as it expects.
    with tempfile.NamedTemporaryFile(
        mode="wb",          # write bytes
        suffix=".eqdsk",
        delete=False,       # keep alive until we've finished reading
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        result = reader_func(tmp_path)
    finally:
        os.remove(tmp_path)  # clean up no matter what

    return result