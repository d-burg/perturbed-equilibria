"""Kinetic-sigma precedence tests for ``baseline.resolve_uncertainty``.

Each channel resolves independently as
``sigma_profiles > IDA .cdf > <chan>_scalar_sigma``, and a winning source
SHADOWS the ones below it.  That is silent by construction, and a silent win
inverts the meaning of a run: zeroing ``*_scalar_sigma`` against an IDA source
reads like "no perturbation" but leaves the full operational envelope in place,
so every supposedly-deterministic point is a full-sigma draw.

These tests pin the resolution itself, the audit log, and the warning that
fires in exactly that case.  The IDA ``.cdf`` is built synthetically with h5py
(the IDA file is netCDF4 = HDF5), as in ``test_ida.py`` -- no proprietary data.
"""
import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet.baseline import Baseline, resolve_uncertainty
from bouquet.config import (BouquetConfig, ReconstructionSource, SolverConfig,
                            UncertaintyConfig)

_CHANNELS = ("ne", "te", "ni", "ti")

# The IDA fractional envelopes written into the synthetic file.  They are the
# "full operational sigmas" the precedence hands back when the scalars lose.
_IDA_FRAC = {"ne": 0.05, "te": 0.04, "ti": 0.06}


# ---------------------------------------------------------------------------
#  fixtures / helpers
# ---------------------------------------------------------------------------
def _write_ida(path, nr=32):
    """Minimal direct-layout IDA .cdf (2-D profiles + companion *_err)."""
    psi = np.linspace(0.0, 1.0, nr)
    ne = 5e19 * (1 - 0.8 * psi ** 2)
    te = 3000.0 * (1 - 0.9 * psi ** 2) + 50.0
    ti = 0.9 * te
    zeff = 1.0 + 0.8 * psi
    nc = 5e18 * (1 - 0.7 * psi ** 2)          # ni_source needs the carbon channel
    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0])
        f["psi_n"] = psi
        for k, v in [("n_e", ne), ("T_e", te), ("T_12C6", ti),
                     ("Zeff", zeff), ("n_12C6", nc)]:
            f[k] = np.stack([v])
        for k, v in [("n_e_err", _IDA_FRAC["ne"] * ne),
                     ("T_e_err", _IDA_FRAC["te"] * te),
                     ("T_12C6_err", _IDA_FRAC["ti"] * ti),
                     ("Zeff_err", 0.10 * zeff), ("n_12C6_err", 0.05 * nc)]:
            f[k] = np.stack([v])
    return psi


def _baseline(psi_kin, n_eq=41):
    """A Baseline carrying only what resolve_uncertainty reads."""
    psi_N = np.linspace(0.0, 1.0, n_eq)
    ne = 5e19 * (1 - 0.8 * psi_kin ** 2)
    te = 3000.0 * (1 - 0.9 * psi_kin ** 2) + 50.0
    j_phi = 8.0e5 * (1.0 - psi_N ** 2)
    return Baseline(
        psi_N=psi_N, j_phi=j_phi,
        j_inductive=0.85 * j_phi, j_BS=0.15 * j_phi,
        psi_N_kinetic=psi_kin, ne=ne, te=te, ni=0.9 * ne, ti=0.9 * te,
        Zeff=np.full_like(psi_kin, 1.5),
        Ip_target=1.2e6, l_i_target=0.85, provenance="reconstruction",
    )


def _config(profiles_path, unc, tmp_path):
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=str(tmp_path / "g.geqdsk"),
                                    profiles_path=profiles_path, time=3.0),
        solver=SolverConfig(mesh_path=str(tmp_path / "mesh.h5")),
        output_header=str(tmp_path / "out"),
        uncertainty=unc,
    )


class _assert_no_shadow_warning:
    """Context manager asserting the scalar-precedence warning does NOT fire."""

    def __enter__(self):
        self._ctx = warnings.catch_warnings(record=True)
        self._rec = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        caught = [str(w.message) for w in self._rec
                  if "IGNORED" in str(w.message)]
        self._ctx.__exit__(*exc)
        assert not caught, caught
        return False


@pytest.fixture()
def ida_case(tmp_path):
    """``(config_factory, baseline)`` sharing one synthetic IDA ``.cdf``."""
    cdf = str(tmp_path / "ida_synth.cdf")
    bl = _baseline(_write_ida(cdf))
    return (lambda unc: _config(cdf, unc, tmp_path)), bl


@pytest.fixture()
def pfile_case(tmp_path):
    """The same, with a non-``.cdf`` profiles_path (no IDA branch)."""
    bl = _baseline(np.linspace(0.0, 1.0, 32))
    return (lambda unc: _config(str(tmp_path / "p.peqdsk"), unc, tmp_path)), bl


# ---------------------------------------------------------------------------
#  precedence
# ---------------------------------------------------------------------------
def test_ida_wins_over_scalars(ida_case):
    """With a .cdf profiles_path the IDA envelope resolves -- not the scalars."""
    make, bl = ida_case
    env = resolve_uncertainty(make(UncertaintyConfig()), bl)
    for ch in _CHANNELS:
        assert np.any(env[f"sigma_{ch}"] > 0.0), ch
    np.testing.assert_allclose(env["sigma_ne"], _IDA_FRAC["ne"] * bl.ne,
                               rtol=1e-6)
    np.testing.assert_allclose(env["sigma_te"], _IDA_FRAC["te"] * bl.te,
                               rtol=1e-6)


def test_explicit_zero_profiles_beat_an_active_ida(ida_case):
    """The documented way to force a deterministic run: zeros resolve to zero."""
    make, bl = ida_case
    n_kin = len(bl.psi_N_kinetic)
    unc = UncertaintyConfig(
        sigma_profiles={ch: np.zeros(n_kin) for ch in _CHANNELS})
    env = resolve_uncertainty(make(unc), bl)
    for ch in _CHANNELS:
        np.testing.assert_array_equal(env[f"sigma_{ch}"], np.zeros(n_kin))
        assert float(np.max(np.abs(env[f"sigma_{ch}"]))) == 0.0, ch


def test_partial_sigma_profiles_shadow_only_their_channel(ida_case):
    """Precedence is per channel: an explicit ne does not disturb te."""
    make, bl = ida_case
    n_kin = len(bl.psi_N_kinetic)
    unc = UncertaintyConfig(sigma_profiles={"ne": np.zeros(n_kin)})
    env = resolve_uncertainty(make(unc), bl)
    np.testing.assert_array_equal(env["sigma_ne"], np.zeros(n_kin))
    np.testing.assert_allclose(env["sigma_te"], _IDA_FRAC["te"] * bl.te,
                               rtol=1e-6)


def test_scalars_resolve_when_no_ida(pfile_case):
    """No .cdf anywhere -> the flat fractional fallback is what resolves."""
    make, bl = pfile_case
    env = resolve_uncertainty(
        make(UncertaintyConfig(ne_scalar_sigma=0.07, te_scalar_sigma=0.03,
                               ni_scalar_sigma=0.11, ti_scalar_sigma=0.13)), bl)
    np.testing.assert_allclose(env["sigma_ne"], 0.07 * np.abs(bl.ne))
    np.testing.assert_allclose(env["sigma_te"], 0.03 * np.abs(bl.te))
    np.testing.assert_allclose(env["sigma_ni"], 0.11 * np.abs(bl.ni))
    np.testing.assert_allclose(env["sigma_ti"], 0.13 * np.abs(bl.ti))


def test_scalars_all_zero_without_ida_really_are_zero(pfile_case):
    """The same zeroing DOES work when no IDA source is in play."""
    make, bl = pfile_case
    env = resolve_uncertainty(
        make(UncertaintyConfig(ne_scalar_sigma=0.0, te_scalar_sigma=0.0,
                               ni_scalar_sigma=0.0, ti_scalar_sigma=0.0)), bl)
    for ch in _CHANNELS:
        assert float(np.max(np.abs(env[f"sigma_{ch}"]))) == 0.0, ch


# ---------------------------------------------------------------------------
#  the footgun warning
# ---------------------------------------------------------------------------
def test_zeroed_scalars_under_active_ida_warn(ida_case):
    """The case that burned us: scalars zeroed, IDA silently wins."""
    make, bl = ida_case
    unc = UncertaintyConfig(ne_scalar_sigma=0.0, te_scalar_sigma=0.0,
                            ni_scalar_sigma=0.0, ti_scalar_sigma=0.0)
    with pytest.warns(UserWarning, match="IGNORED") as rec:
        env = resolve_uncertainty(make(unc), bl)
    msg = str(rec[0].message)
    for ch in _CHANNELS:
        assert f"{ch}_scalar_sigma=0" in msg, msg
    assert "sigma_profiles" in msg          # points at the fix that works
    assert "ida_synth.cdf" in msg           # names the source that won
    # and the sigmas really are the full IDA envelope, not zero
    assert float(np.max(env["sigma_ne"])) > 0.0


def test_nonzero_shadowed_scalar_warns(ida_case):
    """Also warns for a scalar deliberately moved to a NON-default nonzero."""
    make, bl = ida_case
    with pytest.warns(UserWarning, match="te_scalar_sigma=0.25"):
        resolve_uncertainty(make(UncertaintyConfig(te_scalar_sigma=0.25)), bl)


def test_default_scalars_under_ida_do_not_warn(ida_case):
    """Untouched defaults are not a mistake -- no noise for the common case."""
    make, bl = ida_case
    with _assert_no_shadow_warning():
        resolve_uncertainty(make(UncertaintyConfig()), bl)


def test_explicit_profile_suppresses_the_warning_for_that_channel(ida_case):
    """A channel resolved from sigma_profiles never consults its scalar."""
    make, bl = ida_case
    n_kin = len(bl.psi_N_kinetic)
    unc = UncertaintyConfig(
        ne_scalar_sigma=0.0, te_scalar_sigma=0.0,
        ni_scalar_sigma=0.0, ti_scalar_sigma=0.0,
        sigma_profiles={ch: np.zeros(n_kin) for ch in _CHANNELS})
    with _assert_no_shadow_warning():
        resolve_uncertainty(make(unc), bl)


# ---------------------------------------------------------------------------
#  audit log
# ---------------------------------------------------------------------------
def test_log_names_the_winning_source_per_channel(ida_case, capsys):
    make, bl = ida_case
    resolve_uncertainty(make(UncertaintyConfig()), bl)
    out = capsys.readouterr().out
    for ch in ("ne", "te", "ni", "ti", "jphi"):
        assert f"sigma_{ch}" in out, out
    assert "ida_synth.cdf" in out, out          # names the file that won
    assert "[sigma-source]" in out


def test_log_flags_an_all_zero_channel(ida_case, capsys):
    make, bl = ida_case
    n_kin = len(bl.psi_N_kinetic)
    resolve_uncertainty(
        make(UncertaintyConfig(
            sigma_profiles={ch: np.zeros(n_kin) for ch in _CHANNELS})), bl)
    out = capsys.readouterr().out
    assert out.count("ALL ZERO") == len(_CHANNELS), out
    assert "explicit sigma_profiles" in out


def test_log_can_be_switched_off(ida_case, capsys):
    make, bl = ida_case
    resolve_uncertainty(make(UncertaintyConfig(log_sigma_sources=False)), bl)
    assert "[sigma-source]" not in capsys.readouterr().out
