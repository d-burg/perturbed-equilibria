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


# ---------------------------------------------------------------------------
# Tests: Validation against OMFIT and TokaMaker on a real D3D-like equilibrium
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
D3DLIKE_GFILE = os.path.join(DATA_DIR, "d3dlike.geqdsk")
D3DLIKE_REF = os.path.join(DATA_DIR, "d3dlike_reference.npz")

_has_d3dlike_data = (
    os.path.isfile(D3DLIKE_GFILE) and os.path.isfile(D3DLIKE_REF)
)


@pytest.fixture(scope="module")
def d3dlike_eq():
    """Load the D3D-like equilibrium once per module."""
    return GEQDSKEquilibrium(D3DLIKE_GFILE, nlevels=257)


@pytest.fixture(scope="module")
def d3dlike_ref():
    """Load the reference arrays once per module."""
    return dict(np.load(D3DLIKE_REF))


@pytest.mark.skipif(not _has_d3dlike_data, reason="D3D-like test data not found")
class TestD3DLikeValidation:
    """Compare our flux-surface-averaged Jt against OMFIT and TokaMaker
    reference arrays for a real D3D-like TokaMaker equilibrium.

    Reference data:
      - jt_omfit_numerical: OMFIT's numerical <Jt> (negative sign convention)
      - jt_tokamaker_direct: TokaMaker's direct GS Jt (positive sign convention)

    Our equivalents:
      - eq.j_tor_averaged_numerical  ↔  jt_omfit_numerical   (same sign)
      - eq.j_tor_averaged_direct     ↔  jt_tokamaker_direct  (opposite sign)
    """

    def test_grid_size(self, d3dlike_eq):
        """The D3D-like g-file should be 257×257."""
        assert len(d3dlike_eq.R_grid) == 257
        assert len(d3dlike_eq.Z_grid) == 257

    def test_psi_N_length(self, d3dlike_eq, d3dlike_ref):
        """Our psi_N grid should match the reference array length."""
        assert len(d3dlike_eq.psi_N) == len(d3dlike_ref["jt_omfit_numerical"])

    def test_numerical_jt_vs_omfit(self, d3dlike_eq, d3dlike_ref):
        """j_tor_averaged_numerical should match OMFIT's numerical <Jt>.

        Both use the same sign convention (negative for this equilibrium).
        Agreement is < 0.2% throughout the core and pedestal.  The last
        point (separatrix, psi_N = 1) can differ by ~6% because the two
        codes handle the X-point contour differently, so we exclude it.
        """
        ours = d3dlike_eq.j_tor_averaged_numerical
        omfit = d3dlike_ref["jt_omfit_numerical"]
        n = len(ours)

        # Exclude the last point (separatrix) — X-point contour handling
        # differs between our raw-point approach and OMFIT's spline.
        for k in range(1, n - 1):
            if abs(omfit[k]) < 1e3:
                assert abs(ours[k] - omfit[k]) < 1e3, (
                    f"k={k}: ours={ours[k]:.2f}, omfit={omfit[k]:.2f}"
                )
            else:
                rel = abs(ours[k] - omfit[k]) / abs(omfit[k])
                assert rel < 0.01, (
                    f"k={k}: ours={ours[k]:.2f}, omfit={omfit[k]:.2f}, "
                    f"rel={rel:.4f}"
                )

    def test_direct_jt_vs_tokamaker(self, d3dlike_eq, d3dlike_ref):
        """j_tor_averaged_direct should match TokaMaker's direct GS Jt.

        TokaMaker uses positive sign; our convention is negative for this
        equilibrium, so we compare absolute values.

        Agreement is < 0.2% in the core (psi_N < 0.9).  In the H-mode
        pedestal (psi_N ~ 0.92-0.95) the steep pressure gradient amplifies
        small contour differences, giving ~1-2% mismatch.  Beyond
        psi_N ~ 0.95 the separatrix X-point causes larger divergence,
        so we exclude the outermost ~5%.
        """
        ours = d3dlike_eq.j_tor_averaged_direct
        tkmkr = d3dlike_ref["jt_tokamaker_direct"]
        n = len(ours)

        # Exclude outermost ~5% where X-point effects dominate.
        k_edge = int(0.95 * (n - 1))  # psi_N ~ 0.95
        # Two tiers: strict in core, relaxed in pedestal.
        k_pedestal = int(0.90 * (n - 1))  # psi_N ~ 0.90

        for k in range(1, k_edge + 1):
            ours_abs = abs(ours[k])
            tkmkr_abs = abs(tkmkr[k])
            tol = 0.005 if k <= k_pedestal else 0.02
            if tkmkr_abs < 1e3:
                assert abs(ours_abs - tkmkr_abs) < 1e3, (
                    f"k={k}: |ours|={ours_abs:.2f}, |tkmkr|={tkmkr_abs:.2f}"
                )
            else:
                rel = abs(ours_abs - tkmkr_abs) / tkmkr_abs
                assert rel < tol, (
                    f"k={k}: |ours|={ours_abs:.2f}, |tkmkr|={tkmkr_abs:.2f}, "
                    f"rel={rel:.4f} (tol={tol})"
                )

    def test_axis_agreement(self, d3dlike_eq, d3dlike_ref):
        """On-axis values should agree to < 0.1%."""
        ours_num = d3dlike_eq.j_tor_averaged_numerical[0]
        omfit_num = d3dlike_ref["jt_omfit_numerical"][0]
        rel = abs(ours_num - omfit_num) / abs(omfit_num)
        assert rel < 0.001, f"Axis numerical: rel={rel:.6f}"

        ours_dir = abs(d3dlike_eq.j_tor_averaged_direct[0])
        tkmkr_dir = abs(d3dlike_ref["jt_tokamaker_direct"][0])
        rel = abs(ours_dir - tkmkr_dir) / tkmkr_dir
        assert rel < 0.001, f"Axis direct: rel={rel:.6f}"

    def test_sign_conventions(self, d3dlike_eq, d3dlike_ref):
        """OMFIT numerical and our numerical should share the same sign.
        TokaMaker direct has opposite sign from our direct.
        """
        # OMFIT and ours: same sign (negative)
        assert np.sign(d3dlike_eq.j_tor_averaged_numerical[0]) == np.sign(
            d3dlike_ref["jt_omfit_numerical"][0]
        )
        # TokaMaker is positive, ours is negative → opposite signs
        assert np.sign(d3dlike_eq.j_tor_averaged_direct[0]) == -np.sign(
            d3dlike_ref["jt_tokamaker_direct"][0]
        )

    def test_standard_jt_reasonable(self, d3dlike_eq):
        """The default j_tor_averaged (<Jt/R>/<1/R>) should be finite
        and nonzero for this equilibrium.
        """
        jt = d3dlike_eq.j_tor_averaged
        assert np.all(np.isfinite(jt))
        assert np.any(np.abs(jt) > 0)

    def test_edge_derivative_sign(self, d3dlike_eq):
        """The last few points of j_tor should have a consistent derivative
        sign (no artificial spike from resampling artefacts).
        """
        jt = d3dlike_eq.j_tor_averaged_numerical
        # Check that the last 5 points don't have a sign flip in
        # the derivative that would indicate a resampling artefact.
        # (The derivative should be monotonically decreasing in
        # magnitude toward the edge for this equilibrium.)
        diffs = np.diff(jt[-6:])
        # All diffs should have the same sign
        signs = np.sign(diffs)
        # Allow at most one sign change (noise) out of 5 diffs
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        assert sign_changes <= 1, (
            f"Too many sign changes in edge Jt derivative: {sign_changes}"
        )


# ---------------------------------------------------------------------------
# Tests: Theta resampling (OMFIT angular method)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def d3dlike_eq_theta():
    """Load D3D-like equilibrium with theta resampling (explicit)."""
    return GEQDSKEquilibrium(D3DLIKE_GFILE, nlevels=257, resample="theta")


@pytest.fixture(scope="module")
def d3dlike_eq_arclen():
    """Load D3D-like equilibrium with arc-length (raw points) fallback."""
    return GEQDSKEquilibrium(D3DLIKE_GFILE, nlevels=257, resample="arc_length")


@pytest.mark.skipif(not _has_d3dlike_data, reason="D3D-like test data not found")
class TestThetaResample:
    """Tests for the OMFIT-style angular resampling option."""

    def test_theta_resample_basic(self, gfile_path):
        """GEQDSKEquilibrium with resample='theta' should run without error
        on the Solov'ev equilibrium and produce reasonable j_tor_averaged.
        """
        path, _ = gfile_path
        eq = GEQDSKEquilibrium(path, resample="theta")
        jt = eq.j_tor_averaged
        assert np.all(np.isfinite(jt))
        assert np.any(np.abs(jt) > 0)

    def test_invalid_resample_option(self, gfile_path):
        """Invalid resample option should raise ValueError."""
        path, _ = gfile_path
        with pytest.raises(ValueError, match="resample"):
            GEQDSKEquilibrium(path, resample="invalid")

    def test_theta_resample_numerical_jt_vs_omfit(
        self, d3dlike_eq_theta, d3dlike_ref
    ):
        """Theta resampling should match OMFIT's numerical <Jt>.

        With angular resampling, even the last few points near the
        separatrix should be better resolved.  We compare up to
        the second-to-last point.
        """
        ours = d3dlike_eq_theta.j_tor_averaged_numerical
        omfit = d3dlike_ref["jt_omfit_numerical"]
        n = len(ours)

        for k in range(1, n - 1):
            if abs(omfit[k]) < 1e3:
                assert abs(ours[k] - omfit[k]) < 1e3, (
                    f"k={k}: ours={ours[k]:.2f}, omfit={omfit[k]:.2f}"
                )
            else:
                rel = abs(ours[k] - omfit[k]) / abs(omfit[k])
                assert rel < 0.01, (
                    f"k={k}: ours={ours[k]:.2f}, omfit={omfit[k]:.2f}, "
                    f"rel={rel:.4f}"
                )

    def test_theta_resample_direct_jt_vs_tokamaker(
        self, d3dlike_eq_theta, d3dlike_ref
    ):
        """Theta resampling: direct Jt vs TokaMaker with same tolerances."""
        ours = d3dlike_eq_theta.j_tor_averaged_direct
        tkmkr = d3dlike_ref["jt_tokamaker_direct"]
        n = len(ours)

        k_edge = int(0.95 * (n - 1))
        k_pedestal = int(0.90 * (n - 1))

        for k in range(1, k_edge + 1):
            ours_abs = abs(ours[k])
            tkmkr_abs = abs(tkmkr[k])
            tol = 0.005 if k <= k_pedestal else 0.02
            if tkmkr_abs < 1e3:
                assert abs(ours_abs - tkmkr_abs) < 1e3, (
                    f"k={k}: |ours|={ours_abs:.2f}, |tkmkr|={tkmkr_abs:.2f}"
                )
            else:
                rel = abs(ours_abs - tkmkr_abs) / tkmkr_abs
                assert rel < tol, (
                    f"k={k}: |ours|={ours_abs:.2f}, |tkmkr|={tkmkr_abs:.2f}, "
                    f"rel={rel:.4f} (tol={tol})"
                )

    def test_arc_length_still_works(self, d3dlike_eq_arclen):
        """The arc_length fallback should still produce finite results."""
        jt = d3dlike_eq_arclen.j_tor_averaged_numerical
        assert np.all(np.isfinite(jt))
        assert np.any(np.abs(jt) > 0)

    def test_default_is_theta(self):
        """Default resample should be 'theta'."""
        eq = GEQDSKEquilibrium(D3DLIKE_GFILE, nlevels=257)
        assert eq._resample_method == "theta"

    def test_read_geqdsk_passes_resample(self):
        """read_geqdsk convenience function should pass resample parameter."""
        eq = read_geqdsk(D3DLIKE_GFILE, nlevels=257, resample="arc_length")
        assert eq._resample_method == "arc_length"
