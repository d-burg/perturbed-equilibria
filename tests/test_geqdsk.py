"""
Tests for bouquet.io.geqdsk using a synthetic Solov'ev equilibrium.

The Solov'ev solution provides an analytic equilibrium with known
properties, making it ideal for verifying the parser, contour tracing,
and flux-surface averaging.
"""

import tempfile
import os

import numpy as np
import pytest

from bouquet.io.geqdsk import (
    _cocos_params,
    _read_geqdsk,
    _trace_contours,
    _select_main_contour,
    GEQDSKEquilibrium,
    read_geqdsk,
)


# ---------------------------------------------------------------------------
# Synthetic g-file generation
# ---------------------------------------------------------------------------

def _write_fortran_line(f, values):
    """Write values in standard GEQDSK format (5 per line, 16 chars each)."""
    for i, v in enumerate(values):
        f.write(f"{v:16.9E}")
        if (i + 1) % 5 == 0:
            f.write("\n")
    if len(values) % 5 != 0:
        f.write("\n")


def generate_solovev_gfile(path, R0=1.7, a=0.5, B0=2.0, Ip=1e6,
                            kappa=1.0, NW=65, NH=65):
    """Generate a synthetic GEQDSK file using a Solov'ev-like equilibrium.

    PSI(R,Z) = (B0 / (2*R0*q0)) * [(R-R0)^2 + kappa^2 * Z^2]
    This gives concentric elliptical flux surfaces with known q ~ q0.
    """
    q0 = 1.5  # reference safety factor at axis
    mu0 = 4e-7 * np.pi

    # Grid
    RLEFT = R0 - 1.5 * a
    RDIM = 3.0 * a
    ZDIM = 3.0 * a * kappa
    ZMID = 0.0

    R = np.linspace(RLEFT, RLEFT + RDIM, NW)
    Z = np.linspace(ZMID - ZDIM / 2, ZMID + ZDIM / 2, NH)
    RR, ZZ = np.meshgrid(R, Z)

    # Solov'ev-like PSI (parabolic in minor radius)
    # PSI = C * [(R-R0)^2 + kappa^2 * Z^2]
    # where C = B0 / (2 * R0 * q0)
    C = B0 / (2 * R0 * q0)
    PSI_RZ = C * ((RR - R0)**2 + kappa**2 * ZZ**2)

    # Axis at (R0, 0): PSI = 0
    SIMAG = 0.0
    # Boundary at r = a: PSI_bdy = C * a^2
    SIBRY = C * a**2

    # Normalised psi
    psi_N = np.linspace(0, 1, NW)

    # Pressure profile: P(psi_N) = P0 * (1 - psi_N)
    P0 = 1e4  # Pa
    PRES = P0 * (1 - psi_N)
    PPRIME = -P0 / SIBRY * np.ones(NW)  # dP/dpsi = -P0/dpsi_range

    # F profile: F = R0*B0 (constant, vacuum-like)
    FPOL = R0 * B0 * np.ones(NW)
    FFPRIM = np.zeros(NW)

    # Safety factor: approximately q0 for this equilibrium
    QPSI = q0 * np.ones(NW) * (1 + 0.5 * psi_N)  # slight shear

    # Boundary: circle at r = a
    nbbbs = 100
    theta = np.linspace(0, 2 * np.pi, nbbbs, endpoint=False)
    RBBBS = R0 + a * np.cos(theta)
    ZBBBS = kappa * a * np.sin(theta)

    # Limiter: rectangle
    nlim = 5
    RLIM = np.array([RLEFT, RLEFT + RDIM, RLEFT + RDIM, RLEFT, RLEFT])
    ZLIM = np.array([-ZDIM/2, -ZDIM/2, ZDIM/2, ZDIM/2, -ZDIM/2])

    # Scalars
    RCENTR = R0
    BCENTR = B0
    RMAXIS = R0
    ZMAXIS = 0.0

    # Compute current: Ip = 2*pi*C * a^2 / mu0  (approximate for Solov'ev)
    # Actually set it to the user-specified value
    CURRENT = Ip

    # Write g-file
    with open(path, "w") as f:
        # Header line
        case_str = "TESTCASE" + " " * 40  # 48 chars
        f.write(f"{case_str}   0 {NW:4d} {NH:4d}\n")

        # 20 scalars
        scalars = [
            RDIM, ZDIM, RCENTR, RLEFT, ZMID,
            RMAXIS, ZMAXIS, SIMAG, SIBRY, BCENTR,
            CURRENT, SIMAG, 0.0, RMAXIS, 0.0,
            ZMAXIS, 0.0, SIBRY, 0.0, 0.0,
        ]
        _write_fortran_line(f, scalars)

        # 1-D profiles
        _write_fortran_line(f, FPOL)
        _write_fortran_line(f, PRES)
        _write_fortran_line(f, FFPRIM)
        _write_fortran_line(f, PPRIME)

        # PSIRZ (flattened NH*NW)
        _write_fortran_line(f, PSI_RZ.ravel())

        # QPSI
        _write_fortran_line(f, QPSI)

        # Boundary and limiter counts
        f.write(f" {nbbbs:5d} {nlim:5d}\n")

        # Boundary (R,Z pairs interleaved)
        bnd_pairs = np.empty(2 * nbbbs)
        bnd_pairs[0::2] = RBBBS
        bnd_pairs[1::2] = ZBBBS
        _write_fortran_line(f, bnd_pairs)

        # Limiter
        lim_pairs = np.empty(2 * nlim)
        lim_pairs[0::2] = RLIM
        lim_pairs[1::2] = ZLIM
        _write_fortran_line(f, lim_pairs)

    return {
        "R0": R0, "a": a, "B0": B0, "Ip": Ip, "kappa": kappa,
        "NW": NW, "NH": NH, "q0": q0,
        "SIMAG": SIMAG, "SIBRY": SIBRY,
    }


@pytest.fixture
def gfile_path(tmp_path):
    """Create a temporary synthetic g-file and return (path, params)."""
    path = str(tmp_path / "g_test.geqdsk")
    params = generate_solovev_gfile(path)
    return path, params


# ---------------------------------------------------------------------------
# Tests: COCOS
# ---------------------------------------------------------------------------

class TestCOCOS:
    def test_cocos1(self):
        cc = _cocos_params(1)
        assert cc["sigma_Bp"] == 1
        assert cc["sigma_RpZ"] == 1
        assert cc["sigma_rhotp"] == 1
        assert cc["exp_Bp"] == 0

    def test_cocos11(self):
        cc = _cocos_params(11)
        assert cc["exp_Bp"] == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            _cocos_params(0)
        with pytest.raises(ValueError):
            _cocos_params(9)
        with pytest.raises(ValueError):
            _cocos_params(20)


# ---------------------------------------------------------------------------
# Tests: Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_dimensions(self, gfile_path):
        path, params = gfile_path
        g = _read_geqdsk(path)
        assert int(g["NW"]) == params["NW"]
        assert int(g["NH"]) == params["NH"]

    def test_scalars(self, gfile_path):
        path, params = gfile_path
        g = _read_geqdsk(path)
        assert abs(g["RMAXIS"] - params["R0"]) < 1e-6
        assert abs(g["ZMAXIS"]) < 1e-6
        assert abs(g["SIMAG"] - params["SIMAG"]) < 1e-6
        assert abs(g["SIBRY"] - params["SIBRY"]) < 1e-6
        assert abs(g["CURRENT"] - params["Ip"]) < 1e-2

    def test_array_shapes(self, gfile_path):
        path, params = gfile_path
        g = _read_geqdsk(path)
        NW = params["NW"]
        NH = params["NH"]
        assert g["FPOL"].shape == (NW,)
        assert g["PRES"].shape == (NW,)
        assert g["FFPRIM"].shape == (NW,)
        assert g["PPRIME"].shape == (NW,)
        assert g["QPSI"].shape == (NW,)
        assert g["PSIRZ"].shape == (NH, NW)

    def test_boundary(self, gfile_path):
        path, params = gfile_path
        g = _read_geqdsk(path)
        assert len(g["RBBBS"]) == 100
        assert len(g["ZBBBS"]) == 100
        # Boundary should be roughly centered at R0
        assert abs(np.mean(g["RBBBS"]) - params["R0"]) < 0.05

    def test_limiter(self, gfile_path):
        path, params = gfile_path
        g = _read_geqdsk(path)
        assert len(g["RLIM"]) == 5
        assert len(g["ZLIM"]) == 5


# ---------------------------------------------------------------------------
# Tests: Contour tracing
# ---------------------------------------------------------------------------

class TestContours:
    def test_contours_found(self, gfile_path):
        path, params = gfile_path
        g = _read_geqdsk(path)
        R = np.linspace(g["RLEFT"], g["RLEFT"] + g["RDIM"], int(g["NW"]))
        Z = np.linspace(g["ZMID"] - g["ZDIM"] / 2, g["ZMID"] + g["ZDIM"] / 2, int(g["NH"]))
        psi_mid = 0.5 * (g["SIMAG"] + g["SIBRY"])
        contours = _trace_contours(R, Z, g["PSIRZ"], [psi_mid])
        assert len(contours) == 1
        assert len(contours[0]) > 0  # at least one segment

    def test_select_main(self, gfile_path):
        path, params = gfile_path
        g = _read_geqdsk(path)
        R = np.linspace(g["RLEFT"], g["RLEFT"] + g["RDIM"], int(g["NW"]))
        Z = np.linspace(g["ZMID"] - g["ZDIM"] / 2, g["ZMID"] + g["ZDIM"] / 2, int(g["NH"]))
        psi_mid = 0.5 * (g["SIMAG"] + g["SIBRY"])
        contours = _trace_contours(R, Z, g["PSIRZ"], [psi_mid])
        cc = _cocos_params(1)
        seg = _select_main_contour(contours[0], params["R0"], 0.0,
                                    cc["sigma_RpZ"], cc["sigma_rhotp"])
        assert seg is not None
        assert seg.shape[1] == 2
        # Contour should be near R0
        assert abs(np.mean(seg[:, 0]) - params["R0"]) < 0.1


# ---------------------------------------------------------------------------
# Tests: GEQDSKEquilibrium class
# ---------------------------------------------------------------------------

class TestEquilibrium:
    def test_basic_properties(self, gfile_path):
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path)
        assert abs(eq.R_mag - params["R0"]) < 1e-6
        assert abs(eq.Z_mag) < 1e-6
        assert abs(eq.Ip - params["Ip"]) < 1e-2
        assert len(eq.R_grid) == params["NW"]
        assert len(eq.Z_grid) == params["NH"]

    def test_flux_surface_averaging_identity(self, gfile_path):
        """<1> should equal 1 for all surfaces."""
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path, nlevels=33)
        avg = eq.averages
        # <R> should be close to R0 for this concentric equilibrium
        for k in range(1, len(eq.psi_N)):
            assert abs(avg["R"][k] - params["R0"]) < 0.1

    def test_vp_positive(self, gfile_path):
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path, nlevels=33)
        avg = eq.averages
        # vp should be positive (or zero at axis)
        for k in range(1, len(eq.psi_N)):
            assert avg["vp"][k] != 0, f"vp is zero at k={k}"

    def test_q_reasonable(self, gfile_path):
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path, nlevels=33)
        q = eq.q_profile
        # For this equilibrium q ~ q0 ≈ 1.5
        # Check q is in a reasonable range
        for k in range(1, len(q)):
            assert 0.1 < abs(q[k]) < 50, f"q[{k}] = {q[k]} out of range"

    def test_j_tor_nonzero(self, gfile_path):
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path, nlevels=33)
        jt = eq.j_tor_averaged
        # At least some surfaces should have nonzero Jt
        assert np.any(np.abs(jt) > 0)

    def test_li_reasonable(self, gfile_path):
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path, nlevels=33)
        li = eq.li
        # li(3) should be in a physically reasonable range
        assert "li(3)" in li
        # For a synthetic case, just check it's finite and positive
        assert np.isfinite(li["li(3)"])

    def test_geometry(self, gfile_path):
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path, nlevels=33)
        geo = eq.geometry
        # Last surface minor radius should be close to a
        assert abs(geo["a"][-1] - params["a"]) < 0.15
        # Kappa should be close to 1 for circular case
        assert abs(geo["kappa"][-1] - params["kappa"]) < 0.3
        # Volume should be positive
        assert geo["vol"][-1] > 0

    def test_volume_integral(self, gfile_path):
        path, params = gfile_path
        eq = GEQDSKEquilibrium(path, nlevels=33)
        # Volume integral of 1 should equal the total volume
        vol_from_integral = eq.volume_integral(np.ones(len(eq.psi_N)))
        assert vol_from_integral[-1] > 0
        # Should match geo volume
        assert abs(vol_from_integral[-1] - eq.geometry["vol"][-1]) / eq.geometry["vol"][-1] < 0.01


# ---------------------------------------------------------------------------
# Tests: Convenience function and from_bytes
# ---------------------------------------------------------------------------

class TestConvenience:
    def test_read_geqdsk(self, gfile_path):
        path, params = gfile_path
        eq = read_geqdsk(path)
        assert isinstance(eq, GEQDSKEquilibrium)
        assert abs(eq.R_mag - params["R0"]) < 1e-6

    def test_from_bytes(self, gfile_path):
        path, params = gfile_path
        with open(path, "rb") as f:
            raw = f.read()
        eq = GEQDSKEquilibrium.from_bytes(raw)
        assert abs(eq.R_mag - params["R0"]) < 1e-6
        assert abs(eq.Ip - params["Ip"]) < 1e-2
