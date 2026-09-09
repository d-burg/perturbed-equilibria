"""Unit tests for bouquet.io.ida.read_ida -- both field-validated layouts.

Builds tiny synthetic IDA ``.cdf`` files with h5py (the IDA file is netCDF4 =
HDF5), so no proprietary data is needed:

  * direct   : (n_time, n_radial) profiles + companion ``*_err`` datasets;
  * ensemble : (n_time, n_samples, n_radial) posterior samples, no ``*_err``.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet.io.ida import read_ida, read_ida_cer

_CER = ["n_12C6", "T_12C6", "omega_tor_12C6", "v_pol",
        "Bpol_midplane", "Rmaj_midplane", "dPsiN_dR_midplane"]


def _cer_values(psi, ns=None):
    import numpy as np
    return {
        "n_12C6": 5e18 * (1 - 0.7 * (psi / 1.2) ** 2),
        "T_12C6": 2000.0 * (1 - 0.85 * (psi / 1.2) ** 2) + 40.0,
        "omega_tor_12C6": 1e5 * (1 - 0.8 * (psi / 1.2) ** 2),
        "v_pol": 2e3 * (psi / 1.2),
        "Bpol_midplane": -0.3 * (psi / 1.2),
        "Rmaj_midplane": 1.8 + 0.5 * (psi / 1.2),
        "dPsiN_dR_midplane": np.full_like(psi, 2.0),
    }


def _write_direct(path, nr=32):
    psi = np.linspace(0, 1.2, nr)
    ne = 5e19 * (1 - 0.8 * (psi / 1.2) ** 2)
    te = 3000.0 * (1 - 0.9 * (psi / 1.2) ** 2) + 50.0
    ti = 0.9 * te
    zeff = 1.0 + 0.8 * (psi / 1.2)
    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0, 3500.0])           # ms, 2 slices
        f["psi_n"] = psi
        for k, v in [("n_e", ne), ("T_e", te), ("T_12C6", ti), ("Zeff", zeff)]:
            f[k] = np.stack([v, v])                       # (n_time, n_radial)
        for k, v in [("n_e_err", 0.05 * ne), ("T_e_err", 0.04 * te),
                     ("T_12C6_err", 0.06 * ti), ("Zeff_err", 0.10 * zeff)]:
            f[k] = np.stack([v, v])
        cer = _cer_values(psi)
        for k, v in cer.items():
            if k in f:                       # T_12C6 already written above
                continue
            f[k] = np.stack([v, v])
            f[k + "_err"] = np.stack([0.05 * np.abs(v), 0.05 * np.abs(v)])


def _write_ensemble(path, nr=24, ns=256, seed=0):
    rng = np.random.default_rng(seed)
    psi = np.linspace(0, 1.2, nr)
    ne0 = 6e19 * (1 - 0.7 * (psi / 1.2) ** 2)
    te0 = 2500.0 * (1 - 0.85 * (psi / 1.2) ** 2) + 40.0
    ti0 = 0.9 * te0
    zf0 = 1.0 + 1.0 * (psi / 1.2)

    def samples(mu, frac):
        return mu[None, :] * (1.0 + frac * rng.standard_normal((ns, nr)))

    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0])                    # ms, single slice
        f["psi_n"] = np.stack([np.tile(psi, (ns, 1))])    # (1, ns, nr)
        f["n_e"] = np.stack([samples(ne0, 0.05)])         # (1, ns, nr)
        f["T_e"] = np.stack([samples(te0, 0.04)])
        f["T_12C6"] = np.stack([samples(ti0, 0.06)])
        f["Zeff"] = np.stack([samples(zf0, 0.03)])
        for k, v in _cer_values(psi).items():
            if k in f:                                    # T_12C6 already written above
                continue
            f[k] = np.stack([samples(v, 0.05)])           # (1, ns, nr), no *_err
    return dict(ne0=ne0, te0=te0, ti0=ti0, zf0=zf0, psi=psi)


class TestDirectLayout:
    def test_reads_and_derives_ni(self, tmp_path):
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        ida = read_ida(str(p), time=3.0, impurity_Z=6.0)
        assert ida.psi_N[-1] == pytest.approx(1.2)
        assert np.all(ida.ni <= ida.ne) and np.all(ida.ni > 0)
        # sigma comes straight from *_err
        assert np.allclose(ida.sigma_ne, 0.05 * ida.ne, rtol=1e-6)
        assert ida.time == pytest.approx(3.0)

    def test_time_selection(self, tmp_path):
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        ida = read_ida(str(p), time=3.5, impurity_Z=6.0)
        assert ida.time == pytest.approx(3.5)

    def test_direct_mode_rejects_ensemble(self, tmp_path):
        p = tmp_path / "ida_ens.cdf"
        _write_ensemble(str(p))
        with pytest.raises(ValueError, match="3-D posterior"):
            read_ida(str(p), sigma_mode="direct")


class TestEnsembleLayout:
    def test_auto_detect_mean_and_band(self, tmp_path):
        p = tmp_path / "ida_ens.cdf"
        truth = _write_ensemble(str(p))
        ida = read_ida(str(p), impurity_Z=6.0)              # auto-detect
        # sample mean recovers the underlying profile
        assert np.allclose(ida.ne, truth["ne0"], rtol=0.02)
        assert np.allclose(ida.te, truth["te0"], rtol=0.03)
        # percentile band is a positive ~5% fraction on ne
        frac = np.median(ida.sigma_ne[ida.ne > 0] / ida.ne[ida.ne > 0])
        assert 0.02 < frac < 0.09
        assert np.all(ida.ni <= ida.ne) and np.all(ida.ni > 0)

    def test_std_method(self, tmp_path):
        p = tmp_path / "ida_ens.cdf"
        _write_ensemble(str(p))
        ida = read_ida(str(p), sigma_method="std", impurity_Z=6.0)
        assert np.all(ida.sigma_te >= 0)

    def test_ensemble_mode_rejects_direct(self, tmp_path):
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        with pytest.raises(ValueError, match="2-D direct"):
            read_ida(str(p), time=3.0, sigma_mode="ensemble")


class TestReadCER:
    def test_direct_cer(self, tmp_path):
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        cer = read_ida_cer(str(p), time=3.0)
        assert cer.n_carbon.shape == cer.psi_N.shape
        assert np.all(cer.n_carbon > 0) and np.all(cer.Rmaj > 0)
        assert np.any(cer.sigma_omega_tor > 0)          # from *_err
        assert cer.time == pytest.approx(3.0)

    def test_ensemble_cer_auto(self, tmp_path):
        p = tmp_path / "ida_ens.cdf"
        _write_ensemble(str(p))
        cer = read_ida_cer(str(p))                       # single slice
        assert cer.omega_tor.shape == cer.psi_N.shape
        # sample-spread sigma is positive
        assert np.any(cer.sigma_v_pol > 0)

    def test_cer_feeds_radial_field(self, tmp_path):
        from bouquet.physics import radial_field_from_impurity_force_balance
        p = tmp_path / "ida_direct.cdf"
        _write_direct(str(p))
        cer = read_ida_cer(str(p), time=3.0)
        B_phi = np.full_like(cer.psi_N, -2.0)
        E_r, info = radial_field_from_impurity_force_balance(
            cer.psi_N, cer.n_carbon, cer.t_carbon, cer.omega_tor, cer.v_pol,
            cer.Bpol, cer.Rmaj, cer.dpsiN_dR, B_phi,
            sigma_omega_tor=cer.sigma_omega_tor, sigma_v_pol=cer.sigma_v_pol)
        assert E_r.shape == cer.psi_N.shape
        assert np.all(np.isfinite(E_r)) and np.any(info["sigma"] > 0)


def _write_consistent(path, nr=64, nc_scale=1.0, seed=0):
    """Direct-layout file whose two ni routes agree exactly at ``nc_scale=1``.

    Zeff is built from the carbon density by single-impurity quasineutrality,
    so any tension the reader reports is the injected ``nc_scale`` offset plus
    the statistical noise added here.
    """
    Z, rng = 6.0, np.random.default_rng(seed)
    psi = np.linspace(0, 1.2, nr)
    ne = 5e19 * (1 - 0.8 * (psi / 1.2) ** 2)
    nc = 0.01 * ne                                  # 1% carbon
    zeff = 1.0 + Z * (Z - 1.0) * nc / ne            # consistent with nc
    te = 3000.0 * (1 - 0.9 * (psi / 1.2) ** 2) + 50.0
    s_ne, s_zeff, s_nc = 0.05 * ne, 0.03 * zeff, 0.10 * nc
    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0])
        f["psi_n"] = psi
        for k, v in [("n_e", ne + s_ne * rng.standard_normal(nr)), ("T_e", te),
                     ("T_12C6", 0.9 * te),
                     ("Zeff", zeff + s_zeff * rng.standard_normal(nr)),
                     ("n_12C6", nc * nc_scale + s_nc * rng.standard_normal(nr))]:
            f[k] = np.asarray(v)[None, :]
        for k, v in [("n_e_err", s_ne), ("T_e_err", 0.04 * te),
                     ("T_12C6_err", 0.06 * te), ("Zeff_err", s_zeff),
                     ("n_12C6_err", s_nc)]:
            f[k] = np.asarray(v)[None, :]


def _plain_sigma_ni(path, Z=6.0):
    """sigma_ni from the Jacobian terms alone, with no discrepancy systematic.

    Recomputed from the file rather than from read_ida, so the tests below
    compare the reader against an independent propagation.
    """
    with h5py.File(path, "r") as f:
        ne, s_ne = f["n_e"][0], f["n_e_err"][0]
        zeff, s_zeff = f["Zeff"][0], f["Zeff_err"][0]
        s_nc = f["n_12C6_err"][0]
    w = 0.5                                        # ni_source="all"
    d_ne = w * (Z - np.clip(zeff, 1.0, Z)) / (Z - 1.0) + w
    return np.sqrt((w * ne / (Z - 1.0) * s_zeff) ** 2
                   + (w * Z * s_nc) ** 2 + (d_ne * s_ne) ** 2)


class TestRouteDiscrepancy:
    def test_agreeing_routes_are_not_inflated(self, tmp_path):
        p = tmp_path / "ok.cdf"
        _write_consistent(str(p))
        ida = read_ida(str(p), time=3.0)
        # chi ~ 1: the routes differ only by their own statistical errors
        assert 0.4 < np.median(ida.ni_route_chi) < 1.6
        assert np.allclose(ida.sigma_ni, _plain_sigma_ni(str(p)), rtol=0.05)

    def test_disagreeing_routes_inflate_sigma(self, tmp_path):
        p = tmp_path / "bad.cdf"
        _write_consistent(str(p), nc_scale=2.5)      # CER route 2.5x off
        ida = read_ida(str(p), time=3.0)
        plain = _plain_sigma_ni(str(p))
        assert np.median(ida.ni_route_chi) > 3.0
        assert np.all(ida.sigma_ni >= plain)
        assert np.median(ida.sigma_ni / plain) > 1.1

    def test_systematic_is_never_subtractive(self, tmp_path):
        # max(., 0) must hold: agreeing routes may not shrink sigma_ni below
        # the plain propagation.
        p = tmp_path / "ok.cdf"
        _write_consistent(str(p))
        ida = read_ida(str(p), time=3.0)
        assert np.all(ida.sigma_ni >= _plain_sigma_ni(str(p)) - 1e-9)

    def test_single_route_has_no_discrepancy_term(self, tmp_path):
        p = tmp_path / "bad.cdf"
        _write_consistent(str(p), nc_scale=2.5)
        for src in ("Zeff", "CER"):
            ida = read_ida(str(p), time=3.0, ni_source=src)
            assert ida.ni_route_chi is None
            assert np.all(np.isfinite(ida.sigma_ni)) and np.all(ida.sigma_ni > 0)
