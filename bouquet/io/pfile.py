"""
Reader, writer, and interface for Osborne p-files (kinetic profile files).

The p-file format stores 1-D kinetic profiles (densities, temperatures,
rotation frequencies, etc.) on a normalised poloidal flux (``psinorm``)
grid.  Each profile block has a header line followed by rows of
``(psinorm, value, derivative)`` triples.  An optional ``N Z A of ION
SPECIES`` block records the atomic number, charge, and mass of each ion
species.

This module has **no** OMFIT or OMAS dependencies.
"""

import re
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from scipy import interpolate

# Unit conversion: n [10^20/m^3] * T [keV] -> p [kPa]
# = 1e20 * 1e3 * e / 1e3 = e * 1e20 = 16.0218 kPa per (10^20/m^3 * keV)
_NT_TO_KPA = 1.602176634e-19 * 1e20  # exactly 16.02176634

# ---------------------------------------------------------------------------
# Known profile metadata (adapted from OMFIT OMFITpFile)
# ---------------------------------------------------------------------------

DESCRIPTIONS = OrderedDict([
    ("ne", "Electron density"),
    ("te", "Electron temperature"),
    ("ni", "Ion density"),
    ("ti", "Ion temperature"),
    ("nb", "Fast ion density"),
    ("pb", "Fast ion pressure"),
    ("ptot", "Total pressure"),
    ("omeg", "Toroidal rotation: VTOR/R"),
    ("omegp", "Poloidal rotation: Bt * VPOL / (RBp)"),
    ("omgvb", "VxB rotation term in the ExB rotation frequency"),
    ("omgpp", "Diamagnetic term in the ExB rotation frequency"),
    ("omgeb", "ExB rotation frequency"),
    ("er", "Radial electric field from force balance"),
    ("ommvb", "Main ion VxB term of Er/RBp"),
    ("ommpp", "Main ion pressure term of Er/RBp"),
    ("omevb", "Electron VxB term of Er/RBp"),
    ("omepp", "Electron pressure term of Er/RBp"),
    ("kpol", "KPOL = VPOL/Bp"),
    ("omghb", "Hahm-Burrell ExB velocity shearing rate"),
    ("nz1", "Density of the 1st impurity species"),
    ("vtor1", "Toroidal velocity of the 1st impurity species"),
    ("vpol1", "Poloidal velocity of the 1st impurity species"),
    ("nz2", "Density of the 2nd impurity species"),
    ("vtor2", "Toroidal velocity of the 2nd impurity species"),
    ("vpol2", "Poloidal velocity of the 2nd impurity species"),
])

UNITS = OrderedDict([
    ("ne", "10^20/m^3"),
    ("te", "KeV"),
    ("ni", "10^20/m^3"),
    ("ti", "KeV"),
    ("nb", "10^20/m^3"),
    ("pb", "KPa"),
    ("ptot", "KPa"),
    ("omeg", "kRad/s"),
    ("omegp", "kRad/s"),
    ("omgvb", "kRad/s"),
    ("omgpp", "kRad/s"),
    ("omgeb", "kRad/s"),
    ("er", "kV/m"),
    ("ommvb", ""),
    ("ommpp", ""),
    ("omevb", ""),
    ("omepp", ""),
    ("kpol", "km/s/T"),
    ("omghb", ""),
    ("nz1", "10^20/m^3"),
    ("vtor1", "km/s"),
    ("vpol1", "km/s"),
    ("nz2", "10^20/m^3"),
    ("vtor2", "km/s"),
    ("vpol2", "km/s"),
])

# Header regex: "256 psinorm ne(10^20/m^3) dne/dpsiN"
_HEADER_RE = re.compile(
    r"^(\d+)\s+(\S+)\s+(\S+)\(([^)]*)\)\s+(.*?)\s*$"
)


# ---------------------------------------------------------------------------
# Low-level parser / writer
# ---------------------------------------------------------------------------

def _read_pfile(filename):
    """Parse an Osborne p-file into an OrderedDict.

    Parameters
    ----------
    filename : str or path-like
        Path to the p-file.

    Returns
    -------
    OrderedDict
        Keyed by profile name (``"ne"``, ``"te"``, ...).  Each value is a
        dict with keys ``"psinorm"``, ``"data"``, ``"derivative"``,
        ``"units"``, and ``"deriv_label"``.

        The special key ``"N Z A"`` (if present) maps to a dict with
        ``"N"``, ``"Z"``, ``"A"`` arrays.
    """
    with open(filename, "r") as f:
        lines = f.read().strip().splitlines()

    profiles = OrderedDict()
    idx = 0
    while idx < len(lines):
        header = lines[idx]
        tokens = header.split()
        if len(tokens) < 2:
            idx += 1
            continue

        count = int(tokens[0])

        # --- Special block: N Z A of ION SPECIES ---
        if "N Z A of ION SPECIES" in header:
            rows = []
            for i in range(idx + 1, idx + 1 + count):
                rows.append(list(map(float, lines[i].split())))
            cols = list(zip(*rows))
            profiles["N Z A"] = {
                "N": np.array(cols[0]),
                "Z": np.array(cols[1]),
                "A": np.array(cols[2]),
            }
            idx += 1 + count
            continue

        # --- Standard profile block ---
        m = _HEADER_RE.match(header)
        if m is None:
            idx += 1
            continue

        _xkey = m.group(2)      # e.g. "psinorm"
        key = m.group(3)        # e.g. "ne"
        units = m.group(4)      # e.g. "10^20/m^3"
        deriv_label = m.group(5)  # e.g. "dne/dpsiN"

        rows = []
        for i in range(idx + 1, idx + 1 + count):
            rows.append(list(map(float, lines[i].split())))
        cols = list(zip(*rows))

        profiles[key] = {
            "psinorm": np.array(cols[0]),
            "data": np.array(cols[1]),
            "derivative": np.array(cols[2]),
            "units": units,
            "deriv_label": deriv_label,
        }

        idx += 1 + count

    return profiles


def _write_pfile(profiles, filename):
    """Write an OrderedDict of profiles to an Osborne p-file.

    Parameters
    ----------
    profiles : OrderedDict
        Same structure as returned by :func:`_read_pfile`.
    filename : str or path-like
        Output path.
    """
    buf = []
    for key, val in profiles.items():
        if key == "N Z A":
            n = len(val["A"])
            buf.append(f"{n} N Z A of ION SPECIES\n")
            for i in range(n):
                buf.append(
                    f" {val['N'][i]:f}   {val['Z'][i]:f}   {val['A'][i]:f}\n"
                )
        else:
            n = len(val["data"])
            if n <= 1:
                continue
            units = val.get("units", "")
            deriv_label = val.get("deriv_label", f"d{key}/dpsiN")
            buf.append(f"{n} psinorm {key}({units}) {deriv_label}\n")
            for i in range(n):
                buf.append(
                    f" {val['psinorm'][i]:f}   {val['data'][i]:f}"
                    f"   {val['derivative'][i]:f}\n"
                )

    with open(filename, "w") as f:
        f.writelines(buf)


# ---------------------------------------------------------------------------
# PFile class
# ---------------------------------------------------------------------------

class PFile:
    """Interface for Osborne p-file kinetic profiles.

    Parameters
    ----------
    filename : str or path-like
        Path to the p-file.

    Examples
    --------
    >>> pf = PFile("p123456.01234")
    >>> pf.ne          # electron density array
    >>> pf.te          # electron temperature array
    >>> pf.psinorm_for("ne")  # psinorm grid for ne
    >>> "omgeb" in pf  # check if profile exists
    True
    """

    def __init__(self, filename):
        self._raw = _read_pfile(filename)

    @classmethod
    def from_bytes(cls, raw_bytes):
        """Construct from in-memory bytes.

        Parameters
        ----------
        raw_bytes : bytes
            Raw p-file content.
        """
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pfile", delete=False
        ) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        return cls(tmp_path)

    # --- Persistence ---

    def save(self, filename):
        """Write the profiles to *filename* in p-file format."""
        _write_pfile(self._raw, filename)

    # --- Dict-like access ---

    @property
    def keys(self):
        """Profile names in file order (list of str)."""
        return list(self._raw.keys())

    def __contains__(self, key):
        return key in self._raw

    def __getitem__(self, key):
        """Return the raw sub-dict for *key*."""
        return self._raw[key]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)

    # --- Per-profile accessors ---

    def psinorm_for(self, key):
        """Return the psinorm grid for profile *key*.

        Returns ``None`` if *key* is not present or is the ``"N Z A"``
        block.
        """
        entry = self._raw.get(key)
        if entry is None or key == "N Z A":
            return None
        return entry["psinorm"]

    def derivative_for(self, key):
        """Return the derivative array for profile *key*."""
        entry = self._raw.get(key)
        if entry is None or key == "N Z A":
            return None
        return entry["derivative"]

    def units_for(self, key):
        """Return the units string for profile *key*."""
        entry = self._raw.get(key)
        if entry is None or key == "N Z A":
            return None
        return entry.get("units", "")

    # --- Named properties for common profiles ---

    def _get_data(self, key):
        entry = self._raw.get(key)
        if entry is None:
            return None
        return entry["data"]

    @property
    def ne(self):
        """Electron density [10^20/m^3]."""
        return self._get_data("ne")

    @property
    def te(self):
        """Electron temperature [KeV]."""
        return self._get_data("te")

    @property
    def ni(self):
        """Ion density [10^20/m^3]."""
        return self._get_data("ni")

    @property
    def ti(self):
        """Ion temperature [KeV]."""
        return self._get_data("ti")

    @property
    def nb(self):
        """Fast ion density [10^20/m^3]."""
        return self._get_data("nb")

    @property
    def pb(self):
        """Fast ion pressure [KPa]."""
        return self._get_data("pb")

    @property
    def ptot(self):
        """Total pressure [KPa]."""
        return self._get_data("ptot")

    @property
    def omeg(self):
        """Toroidal rotation VTOR/R [kRad/s]."""
        return self._get_data("omeg")

    @property
    def omegp(self):
        """Poloidal rotation Bt*VPOL/(RBp) [kRad/s]."""
        return self._get_data("omegp")

    @property
    def omgvb(self):
        """VxB rotation term [kRad/s]."""
        return self._get_data("omgvb")

    @property
    def omgpp(self):
        """Diamagnetic rotation term [kRad/s]."""
        return self._get_data("omgpp")

    @property
    def omgeb(self):
        """ExB rotation frequency [kRad/s]."""
        return self._get_data("omgeb")

    @property
    def er(self):
        """Radial electric field [kV/m]."""
        return self._get_data("er")

    @property
    def kpol(self):
        """KPOL = VPOL/Bp [km/s/T]."""
        return self._get_data("kpol")

    @property
    def omghb(self):
        """Hahm-Burrell ExB shearing rate."""
        return self._get_data("omghb")

    @property
    def ion_species(self):
        """Ion species dict with 'N', 'Z', 'A' arrays, or None."""
        return self._raw.get("N Z A")

    # --- Construction helpers ---

    @classmethod
    def new(cls):
        """Create an empty PFile (no profiles loaded from disk).

        Returns
        -------
        PFile
        """
        obj = object.__new__(cls)
        obj._raw = OrderedDict()
        return obj

    def set_profile(self, key, psinorm, data, derivative=None, units=None):
        """Add or replace a profile.

        Parameters
        ----------
        key : str
            Profile name (e.g. ``"ne"``, ``"te"``).
        psinorm : array-like
            Normalised poloidal flux grid.
        data : array-like
            Profile values on *psinorm*.
        derivative : array-like or None
            Derivative d(data)/d(psinorm).  If ``None``, computed via
            ``np.gradient``.
        units : str or None
            Unit label.  If ``None``, looked up from :data:`UNITS`.
        """
        psinorm = np.asarray(psinorm, dtype=float)
        data = np.asarray(data, dtype=float)
        if derivative is None:
            derivative = np.gradient(data, psinorm)
        else:
            derivative = np.asarray(derivative, dtype=float)
        if units is None:
            units = UNITS.get(key, "")
        self._raw[key] = {
            "psinorm": psinorm,
            "data": data,
            "derivative": derivative,
            "units": units,
            "deriv_label": f"d{key}/dpsiN",
        }

    def set_ion_species(self, N, Z, A):
        """Set the ion species block.

        Parameters
        ----------
        N, Z, A : array-like
            Atomic number, charge state, and mass number for each species.
        """
        self._raw["N Z A"] = {
            "N": np.asarray(N, dtype=float),
            "Z": np.asarray(Z, dtype=float),
            "A": np.asarray(A, dtype=float),
        }

    def compute_derivatives(self):
        """Recompute d(data)/d(psinorm) for all profiles in place."""
        for key, val in self._raw.items():
            if key == "N Z A":
                continue
            val["derivative"] = np.gradient(val["data"], val["psinorm"])

    # --- Physics computations ---

    def compute_pressure(self):
        """Compute total pressure from density and temperature profiles.

        Uses the relation ``ptot = _NT_TO_KPA * (ne*Te + (ni + nz1)*Ti) + pb``
        where the constant converts from (10^20/m^3 * keV) to kPa.

        Requires ``ne``, ``te``, ``ni``, ``ti`` on the same psinorm grid.
        ``nz1`` and ``pb`` default to zero if absent.
        """
        psinorm = self._raw["ne"]["psinorm"]
        ne = self._raw["ne"]["data"]
        te = self._raw["te"]["data"]
        ni = self._raw["ni"]["data"]
        ti = self._raw["ti"]["data"]

        nz1 = self._get_data("nz1")
        if nz1 is None:
            nz1 = np.zeros_like(psinorm)
        pb = self._get_data("pb")
        if pb is None:
            pb = np.zeros_like(psinorm)

        ptot = _NT_TO_KPA * (ne * te + (ni + nz1) * ti) + pb
        self.set_profile("ptot", psinorm, ptot)

    def compute_quasineutrality(self):
        """Compute impurity density nz1 from quasi-neutrality.

        ``nz1 = (ne - ni - nb) / Z_impurity``

        Requires ``ne``, ``ni`` on the same grid and a ``"N Z A"`` block
        with at least one impurity species.  ``nb`` defaults to zero if
        absent.
        """
        nza = self._raw.get("N Z A")
        if nza is None:
            raise ValueError("Ion species (N Z A) block required")
        Z_imp = nza["Z"][0]

        psinorm = self._raw["ne"]["psinorm"]
        ne = self._raw["ne"]["data"]
        ni = self._raw["ni"]["data"]
        nb = self._get_data("nb")
        if nb is None:
            nb = np.zeros_like(psinorm)

        nz1 = (ne - ni - nb) / Z_imp
        self.set_profile("nz1", psinorm, nz1)

    def compute_zeff(self):
        """Compute the effective charge profile.

        .. math::

            Z_{\\mathrm{eff}}
            = \\frac{\\sum_s n_s Z_s^2}{n_e}
            = \\frac{n_i Z_{\\mathrm{main}}^2
                   + n_{z1} Z_{\\mathrm{imp}}^2
                   + n_b Z_{\\mathrm{beam}}^2}{n_e}

        Charge states are read from the ``"N Z A"`` block (OMFIT
        convention: impurities first, then main ion, beam ion last).

        Requires ``ne``, ``ni`` on the same grid and an ``"N Z A"`` block.
        ``nz1`` and ``nb`` default to zero if absent.

        Returns
        -------
        psinorm : numpy.ndarray
            Normalised poloidal flux grid.
        zeff : numpy.ndarray
            Effective charge profile (dimensionless).

        Notes
        -----
        This is intentionally **not** written into the p-file (Zeff is
        not a standard p-file key) to preserve OMFIT compatibility.
        """
        nza = self._raw.get("N Z A")
        if nza is None:
            raise ValueError("Ion species (N Z A) block required")
        Z_imp = nza["Z"][0]
        Z_main = nza["Z"][-2]
        Z_beam = nza["Z"][-1]

        psinorm = self._raw["ne"]["psinorm"]
        ne = self._raw["ne"]["data"]
        ni = self._raw["ni"]["data"]
        nz1 = self._get_data("nz1")
        if nz1 is None:
            nz1 = np.zeros_like(psinorm)
        nb = self._get_data("nb")
        if nb is None:
            nb = np.zeros_like(psinorm)

        with np.errstate(divide="ignore", invalid="ignore"):
            zeff = (ni * Z_main**2 + nz1 * Z_imp**2 + nb * Z_beam**2) / ne
        # At the boundary where ne -> 0, Zeff is undefined; default to 1
        np.nan_to_num(zeff, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
        return psinorm, zeff

    def compute_diamagnetic_rotations(self, psi, nI=None, TI=None):
        """Compute diamagnetic rotation frequencies from kinetic profiles.

        The diamagnetic frequency for species *s* is

        .. math::

            \\omega_{\\mathrm{dia},s}
            = \\frac{1}{n_s Z_s e} \\frac{\\mathrm{d}(n_s T_s)}{\\mathrm{d}\\psi}

        In p-file units (n in 10^20/m^3, T in keV, psi in Wb) this
        reduces to ``d(n*T)/dpsi / (n * Z)`` in kRad/s.

        Parameters
        ----------
        psi : array-like
            Poloidal flux in SI (Weber), same length as the profile grids.
            Typically from the g-file: ``psi = psiN * (psi_bdy - psi_axis)
            + psi_axis``.
        nI : array-like or None
            Impurity density in 10^20/m^3.  If ``None``, uses
            ``nz1`` from the pfile (must already be set).
        TI : array-like or None
            Impurity temperature in keV.  If ``None``, uses ``ti``
            (assumes all ions share the same temperature).

        Sets
        ----
        omgpp, ommpp, omepp : profiles in kRad/s
        """
        psi = np.asarray(psi, dtype=float)
        dpsi = np.gradient(psi)
        psinorm = self._raw["ne"]["psinorm"]

        ne = self._raw["ne"]["data"]
        te = self._raw["te"]["data"]
        ni = self._raw["ni"]["data"]
        ti = self._raw["ti"]["data"]

        if nI is None:
            nI = self._raw["nz1"]["data"]
        else:
            nI = np.asarray(nI, dtype=float)
        if TI is None:
            TI = ti
        else:
            TI = np.asarray(TI, dtype=float)

        nza = self._raw.get("N Z A")
        if nza is None:
            raise ValueError("Ion species (N Z A) block required")
        Z_imp = nza["Z"][0]

        # Impurity diamagnetic (counter-current, negative by convention)
        with np.errstate(divide="ignore", invalid="ignore"):
            omgpp = -np.abs(np.gradient(nI * TI) / dpsi / (nI * Z_imp))
            # Main ion diamagnetic (counter-current, negative by convention)
            ommpp = -np.abs(np.gradient(ni * ti) / dpsi / (ni * 1.0))
            # Electron diamagnetic (co-current, positive by convention)
            omepp = np.abs(np.gradient(ne * te) / dpsi / (ne * 1.0))
        # Replace NaN/inf at boundary where density -> 0
        np.nan_to_num(omgpp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(ommpp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(omepp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        self.set_profile("omgpp", psinorm, omgpp)
        self.set_profile("ommpp", psinorm, ommpp)
        self.set_profile("omepp", psinorm, omepp)

    def compute_rotation_decomposition(self, R=None, Bp=None, Bt=None,
                                       psi=None):
        """Compute ExB and VxB rotation frequencies and derived quantities.

        From the diamagnetic terms (``omgpp``, ``ommpp``, ``omepp``) and
        the impurity VxB rotation (``omgvb``, which must already be set or
        defaults to zero), computes:

        - ``omgeb = omgvb + omgpp``  (ExB rotation)
        - ``ommvb = omgeb - ommpp``  (main ion VxB)
        - ``omevb = omgeb - omepp``  (electron VxB)

        If equilibrium data (*R*, *Bp*, *Bt*, *psi*) are provided, also
        computes:

        - ``er = omgeb * R * Bp``  (radial electric field, kV/m)
        - ``omghb = (R*Bp)^2/Bt * d(omgeb)/dpsi``  (Hahm-Burrell rate)

        Parameters
        ----------
        R, Bp, Bt : array-like or None
            Midplane major radius [m], poloidal field [T], and toroidal
            field [T] on the profile psinorm grid.
        psi : array-like or None
            Poloidal flux in SI (Weber).
        """
        psinorm = self._raw["omgpp"]["psinorm"]

        omgvb = self._get_data("omgvb")
        if omgvb is None:
            omgvb = np.zeros_like(psinorm)

        omgpp = self._raw["omgpp"]["data"]
        ommpp = self._raw["ommpp"]["data"]
        omepp = self._raw["omepp"]["data"]

        omgeb = omgvb + omgpp
        ommvb = omgeb - ommpp
        omevb = omgeb - omepp

        self.set_profile("omgeb", psinorm, omgeb)
        self.set_profile("ommvb", psinorm, ommvb)
        self.set_profile("omevb", psinorm, omevb)

        if R is not None and Bp is not None:
            R = np.asarray(R, dtype=float)
            Bp = np.asarray(Bp, dtype=float)
            er = omgeb * R * Bp
            self.set_profile("er", psinorm, er)

            if Bt is not None and psi is not None:
                Bt = np.asarray(Bt, dtype=float)
                psi = np.asarray(psi, dtype=float)
                dpsi = np.gradient(psi)
                omghb = (R * Bp) ** 2 / Bt * np.gradient(omgeb) / dpsi
                self.set_profile("omghb", psinorm, omghb)

    # --- Remap ---

    def remap(self, psinorm=None, key="ne"):
        """Return a new :class:`PFile` with all profiles on a common grid.

        Parameters
        ----------
        psinorm : array-like, int, or None
            Target grid.  If ``None``, use the grid from *key*.
            If ``int``, use ``np.linspace(0, 1, psinorm)``.
        key : str
            Profile whose grid to use when *psinorm* is ``None``.

        Returns
        -------
        PFile
            New instance with interpolated profiles on the common grid.
        """
        if psinorm is None:
            if key not in self._raw:
                raise KeyError(f"Profile {key!r} not found for grid reference")
            target = self._raw[key]["psinorm"]
        elif isinstance(psinorm, (int, np.integer)):
            target = np.linspace(0, 1, int(psinorm))
        else:
            target = np.asarray(psinorm, dtype=float)

        new_raw = OrderedDict()
        for k, val in self._raw.items():
            if k == "N Z A":
                new_raw[k] = val.copy()
                continue

            f_data = interpolate.interp1d(
                val["psinorm"], val["data"],
                kind="linear", fill_value="extrapolate",
            )
            f_deriv = interpolate.interp1d(
                val["psinorm"], val["derivative"],
                kind="linear", fill_value="extrapolate",
            )
            new_raw[k] = {
                "psinorm": target.copy(),
                "data": f_data(target),
                "derivative": f_deriv(target),
                "units": val.get("units", ""),
                "deriv_label": val.get("deriv_label", f"d{k}/dpsiN"),
            }

        # Build a new PFile without re-parsing a file
        obj = object.__new__(PFile)
        obj._raw = new_raw
        return obj

    def __repr__(self):
        profile_keys = [k for k in self._raw if k != "N Z A"]
        return (
            f"PFile({len(profile_keys)} profiles: "
            f"{', '.join(profile_keys[:6])}"
            f"{'...' if len(profile_keys) > 6 else ''})"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def read_pfile(filename):
    """Read an Osborne p-file and return a :class:`PFile` object.

    Parameters
    ----------
    filename : str or path-like
        Path to the p-file.

    Returns
    -------
    PFile
    """
    return PFile(filename)
