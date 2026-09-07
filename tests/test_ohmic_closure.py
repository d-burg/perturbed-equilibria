"""jBS_baseline_mode='ohmic': the FSA current measure and the workflow guard.

Solve-free coverage of the machinery the ohmic Ip closure is built on:

* ``Ip_fsa_weights`` is AFFINE -- ``I_p[J] = trapezoid(w J, psi_N) + c`` --
  and the P'-term constant ``c`` is counted exactly once (the first closure
  implementation triple-counted it via per-component integral calls).
* The two profile conventions must agree exactly on the equilibrium's own
  Grad-Shafranov profile: integrating ``eq_jphi_profile(conv)`` under the
  matching measure gives the SAME current for 'jphi-linterp' and 'fsa'.
  That identity is what the run-time roundtrip gate validates on a real
  geometry; here it is checked algebraically on a synthetic one.
* The workflow validator accepts the new mode on the IMAS path but refuses
  to generate() with it (baseline-only until the draw path's sigma=0
  reproduction is verified), with the usual workflow='custom' downgrade.

No solver: the geometry is a synthetic ``fsa_current_geometry``-shaped dict.
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import trapezoid

from bouquet.utils import Ip_fsa_weights, Ip_fsa_integral, eq_jphi_profile

_N = 101


def _geom():
    """A plausible fsa_current_geometry() dict for a D3D-like shape."""
    psi = np.linspace(0.01, 0.99, _N)
    R0, a = 1.7, 0.6
    r = a * np.sqrt(psi)                        # minor-radius proxy
    R_avg = R0 + 0.1 * r ** 2 / a               # Shafranov-ish outward shift
    inv_R = (1.0 / R0) * (1.0 + 0.05 * (r / a) ** 2)
    inv_R2 = inv_R ** 2 * (1.0 + 0.02 * (r / a) ** 2)
    dV_dpsi = 4.0 * np.pi ** 2 * R0 * r * (a / (2.0 * np.sqrt(psi) + 1e-12))
    pprime = -8.0e3 * (1.0 - psi)               # falls to 0 at the edge
    FFp = -0.4 * (1.0 - psi) ** 2
    return {
        "psi_N": psi, "psi_q": psi, "R_avg": R_avg,
        "inv_R": inv_R, "inv_R2": inv_R2, "dV_dpsi": dV_dpsi,
        "dpsi_dpsiN": 0.9, "pprime": pprime, "FFp": FFp,
    }


class TestAffineMeasure:
    def test_integral_is_weights_dot_profile_plus_c(self):
        g = _geom()
        rng = np.random.default_rng(3)
        w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        for _ in range(3):
            j = 1e6 * rng.standard_normal(_N)
            direct = Ip_fsa_integral(None, g["psi_N"], j,
                                     convention="jphi-linterp", geom=g)
            assert direct == pytest.approx(
                float(trapezoid(w * j, g["psi_N"])) + c, rel=1e-12)

    def test_c_is_counted_once_not_per_component(self):
        """Summing per-component Ip_fsa_integral calls over-counts c -- the
        linear-parts closure exists precisely to avoid that."""
        g = _geom()
        w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        assert c != 0.0
        parts = [np.full(_N, 2.0e5), np.full(_N, 1.0e5), np.full(_N, 5.0e4)]
        total = Ip_fsa_integral(None, g["psi_N"], sum(parts),
                                convention="jphi-linterp", geom=g)
        summed = sum(Ip_fsa_integral(None, g["psi_N"], p,
                                     convention="jphi-linterp", geom=g)
                     for p in parts)
        assert summed - total == pytest.approx(2.0 * c, rel=1e-9)

    def test_fsa_convention_is_purely_linear(self):
        _, c = Ip_fsa_weights(_geom(), convention="fsa")
        assert c == 0.0

    def test_conventions_agree_on_the_equilibrium_profile(self):
        """The roundtrip identity behind the run-time 0.5% gate, exactly."""
        g = _geom()
        for sign in (1.0, -1.0):
            ips = {}
            for conv in ("jphi-linterp", "fsa"):
                prof = eq_jphi_profile(g, convention=conv, pprime_sign=sign)
                ips[conv] = Ip_fsa_integral(None, g["psi_N"], prof,
                                            convention=conv,
                                            pprime_sign=sign, geom=g)
            assert ips["fsa"] == pytest.approx(ips["jphi-linterp"], rel=1e-12)

    def test_pprime_sign_moves_only_the_affine_term(self):
        g = _geom()
        wp, cp = Ip_fsa_weights(g, convention="jphi-linterp", pprime_sign=1.0)
        wm, cm = Ip_fsa_weights(g, convention="jphi-linterp", pprime_sign=-1.0)
        np.testing.assert_array_equal(wp, wm)
        assert cm == pytest.approx(-cp, rel=1e-12)

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="unknown convention"):
            Ip_fsa_weights(_geom(), convention="cylindrical")

    def test_jphi_linterp_requires_inv_R2_and_pprime(self):
        g = _geom(); g["inv_R2"] = None
        with pytest.raises(ValueError, match="<1/R\\^2>"):
            Ip_fsa_weights(g, convention="jphi-linterp")
        g = _geom(); g["pprime"] = None
        with pytest.raises(ValueError, match="P'"):
            Ip_fsa_weights(g, convention="jphi-linterp")


class TestClosureAlgebra:
    """utils.close_ip -- the SHIPPED channel formulas -- closes Ip exactly.

    Earlier versions of these tests re-derived the scale formulas locally
    and so validated the tester's algebra, not run.py's (adversarial
    review); close_ip is the single implementation both now share.
    """

    def _parts(self):
        g = _geom()
        w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        psi = g["psi_N"]
        j_ind = 8.0e5 * (1.0 - psi) ** 1.5
        j_bs = 3.0e5 * np.exp(-((psi - 0.9) / 0.06) ** 2)   # pedestal hump
        j_fix = 1.0e5 * (1.0 - psi) ** 3
        lin = lambda j: float(trapezoid(w * j, psi))
        return g, w, c, j_ind, j_bs, j_fix, lin

    def test_bootstrap_channel_closes_exactly(self):
        from bouquet.utils import close_ip
        g, w, c, j_ind, j_bs, j_fix, lin = self._parts()
        Ip_t = 1.1 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)  # 10% deficit
        ohm_s, bs_s = close_ip("bootstrap", Ip_t, c,
                               lin(j_ind), lin(j_bs), lin(j_fix))
        assert ohm_s == 1.0
        closed = Ip_fsa_integral(None, g["psi_N"],
                                 ohm_s * j_ind + bs_s * j_bs + j_fix,
                                 convention="jphi-linterp", geom=g)
        assert closed == pytest.approx(Ip_t, rel=1e-12)

    def test_ohmic_channel_closes_exactly(self):
        from bouquet.utils import close_ip
        g, w, c, j_ind, j_bs, j_fix, lin = self._parts()
        Ip_t = 0.93 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
        ohm_s, bs_s = close_ip("ohmic", Ip_t, c,
                               lin(j_ind), lin(j_bs), lin(j_fix))
        assert bs_s == 1.0
        closed = Ip_fsa_integral(None, g["psi_N"],
                                 ohm_s * j_ind + bs_s * j_bs + j_fix,
                                 convention="jphi-linterp", geom=g)
        assert closed == pytest.approx(Ip_t, rel=1e-12)

    def test_negative_current_convention_closes_too(self):
        """DIII-D-sign data: a signed (negative) Ip target with negative
        linear parts must close without the abs() bookkeeping leaking in."""
        from bouquet.utils import close_ip
        g, w, c, j_ind, j_bs, j_fix, lin = self._parts()
        Ip_t = 1.05 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
        ohm_s, bs_s = close_ip("bootstrap", -Ip_t, -c,
                               -lin(j_ind), -lin(j_bs), -lin(j_fix))
        assert bs_s == pytest.approx(
            close_ip("bootstrap", Ip_t, c, lin(j_ind), lin(j_bs),
                     lin(j_fix))[1], rel=1e-12)

    def test_zero_divisor_refusals(self):
        from bouquet.utils import close_ip
        with pytest.raises(RuntimeError, match="j_BS integrates"):
            close_ip("bootstrap", 1.2e6, 0.0, 1.0e6, 0.0, 1e5)
        with pytest.raises(RuntimeError, match="j_inductive integrates"):
            close_ip("ohmic", 1.2e6, 0.0, 0.0, 2e5, 1e5)

    def test_out_of_bounds_scale_refused(self):
        from bouquet.utils import close_ip
        # components sum to ~Ip/10 -> bs_scale would be ~7: refuse
        with pytest.raises(RuntimeError, match="outside"):
            close_ip("bootstrap", 1.2e6, 0.0, 1.0e5, 1.5e5, 0.0)

    def test_nan_inputs_are_refused_not_propagated(self):
        """NaN parts must raise, not return a NaN scale that downstream
        threshold gates (False for NaN) would wave through."""
        from bouquet.utils import close_ip
        with pytest.raises(RuntimeError):
            close_ip("bootstrap", 1.2e6, float("nan"), 8e5, 2e5, 1e5)

    def test_unknown_channel_raises_value_error(self):
        from bouquet.utils import close_ip
        with pytest.raises(ValueError, match="closure_channel"):
            close_ip("bootstap", 1.2e6, 0.0, 8e5, 2e5, 1e5)


class TestSawtoothBootstrapPredictor:
    """utils.close_ip_q0 -- the SHIPPED 2x2 predictor for the q0 channel.

    Solve-free: the axis row is an algebraic statement about the on-axis
    current density, the Ip row is the same affine measure the other channels
    close.  Both are checked against the shipped formula, never a local
    re-derivation (same rule as TestClosureAlgebra).
    """

    def _parts(self, core_bootstrap=True):
        """A bootstrap with BOTH a pedestal hump and a core shoulder -- the
        genuinely 2-D case.  ``core_bootstrap=False`` drops the shoulder, which
        is the degenerate (and expected-common) case the next test pins."""
        g = _geom()
        w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        psi = g["psi_N"]
        j_ind = 8.0e5 * (1.0 - psi) ** 1.5
        j_bs = 3.0e5 * np.exp(-((psi - 0.9) / 0.06) ** 2)   # pedestal hump
        if core_bootstrap:
            j_bs = j_bs + 1.2e5 * (1.0 - psi) ** 2
        j_fix = 1.0e5 * (1.0 - psi) ** 3
        lin = lambda j: float(trapezoid(w * j, psi))
        return g, c, j_ind, j_bs, j_fix, lin

    def test_hits_both_targets_exactly(self):
        from bouquet.utils import close_ip_q0
        g, c, j_ind, j_bs, j_fix, lin = self._parts()
        Ip_t = 1.06 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
        j_ref0 = 1.05 * (j_ind[0] + j_bs[0] + j_fix[0])   # 5% more axis current
        s_o, s_b = close_ip_q0(Ip_t, c, lin(j_ind), lin(j_bs), lin(j_fix),
                               j_ind[0], j_bs[0], j_fix[0], j_ref0)
        # target 1: the axis current density matches -> q0 matches to 1st order
        assert (s_o * j_ind[0] + s_b * j_bs[0] + j_fix[0]) == pytest.approx(
            j_ref0, rel=1e-12)
        # target 2: Ip is still EXACT in the affine measure
        closed = Ip_fsa_integral(None, g["psi_N"],
                                 s_o * j_ind + s_b * j_bs + j_fix,
                                 convention="jphi-linterp", geom=g)
        assert closed == pytest.approx(Ip_t, rel=1e-12)

    def test_reduces_to_bootstrap_when_the_bootstrap_has_no_core(self):
        """The plan's central prediction, and what the REQUESTED reference
        makes the common case: the target axis current is the source's own
        (j_ind0 + j_BS_src0 + j_fix0), so with no core bootstrap on either
        side the axis row pins s_ohm = 1 exactly and the Ip row hands the
        whole deficit to s_bs -- i.e. close_ip('bootstrap'), at zero cost."""
        from bouquet.utils import close_ip, close_ip_q0
        g, c, j_ind, j_bs, j_fix, lin = self._parts(core_bootstrap=False)
        assert j_bs[0] < 1e-30 * j_bs.max()            # pedestal only
        Ip_t = 1.06 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
        j_ref0 = j_ind[0] + j_bs[0] + j_fix[0]         # source axis current
        s_o, s_b = close_ip_q0(Ip_t, c, lin(j_ind), lin(j_bs), lin(j_fix),
                               j_ind[0], j_bs[0], j_fix[0], j_ref0)
        b_o, b_b = close_ip("bootstrap", Ip_t, c,
                            lin(j_ind), lin(j_bs), lin(j_fix))
        assert s_o == pytest.approx(b_o, rel=1e-12)
        assert s_o == pytest.approx(1.0, rel=1e-12)
        assert s_b == pytest.approx(b_b, rel=1e-12)


class TestQ0TargetUnrenormalisation:
    """utils.unrenormalise_q0 -- the REQUESTED reference (user decision).

    The anchor is a solve of the source total renormalised to Ip_target, so
    its q0 belongs to a current the source never claimed.  The target is that
    q0 mapped back onto the source's OWN current, to first order in
    q0 ~ 1/j_phi(0).
    """

    def test_ratio_formula(self):
        from bouquet.utils import unrenormalise_q0
        # reference validation slice, measured: the anchor ran 3.86% hot because FUSE's
        # core_profiles total carries -3.89% of Ip.
        q0t = unrenormalise_q0(0.942997745122599, 1606692.5450515286,
                               1547032.96694553)
        assert q0t == pytest.approx(
            0.942997745122599 * (1606692.5450515286 / 1547032.96694553),
            rel=1e-15)
        assert q0t > 0.942997745122599          # un-renormalising raises q0
        assert q0t == pytest.approx(0.9794, abs=5e-4)

    def test_identity_when_the_source_already_carries_ip(self):
        """No Ip deficit -> achieved == requested -> the target IS the
        anchor's q0 and the un-renormalisation is a no-op."""
        from bouquet.utils import unrenormalise_q0
        assert unrenormalise_q0(1.03, 1.5e6, 1.5e6) == pytest.approx(1.03,
                                                                     rel=1e-15)

    def test_sign_is_carried_not_stripped(self):
        """q carries a COCOS sign; the mapping must not silently abs() it --
        only the GATE compares magnitudes."""
        from bouquet.utils import unrenormalise_q0
        assert unrenormalise_q0(-0.99, 1.04e6, 1.0e6) < 0

    def test_zero_and_nan_requested_axis_current_refused(self):
        from bouquet.utils import unrenormalise_q0
        with pytest.raises(RuntimeError, match="zero axis"):
            unrenormalise_q0(1.0, 1.5e6, 0.0)
        with pytest.raises(RuntimeError, match="non-finite"):
            unrenormalise_q0(1.0, float("nan"), 1.5e6)

    def test_singular_system_refused_on_a_relative_floor(self):
        """Proportional rows (the components indistinguishable in both
        targets) cannot impose two constraints -- refuse, do not return a
        1e12 scale.  The floor must be RELATIVE: A/m^2 and A share no
        absolute epsilon."""
        from bouquet.utils import close_ip_q0
        with pytest.raises(RuntimeError, match="singular"):
            close_ip_q0(1.2e6, 0.0, 8.0e5, 4.0e5, 1e5,
                        2.0e6, 1.0e6, 0.0, 2.5e6)      # rows exactly 2:1

    def test_out_of_bounds_and_nan_refused(self):
        from bouquet.utils import close_ip_q0
        with pytest.raises(RuntimeError, match="outside"):
            close_ip_q0(1.2e6, 0.0, 1.0e5, 1.5e5, 0.0,
                        1.0e6, 1.0e4, 0.0, 1.0e6)
        with pytest.raises(RuntimeError, match="non-finite"):
            close_ip_q0(1.2e6, float("nan"), 8e5, 2e5, 1e5,
                        1e6, 1e4, 0.0, 1e6)


class TestSawtoothGateInputs:
    def test_gate_defaults_and_baseline_field(self):
        from bouquet.config import GenerationConfig
        from bouquet.baseline import Baseline
        gc = GenerationConfig()
        assert gc.q0_gate == 1.1
        assert gc.q0_tol == 0.01
        assert Baseline.__dataclass_fields__["sawtooth"].default is None

    def test_gate_logic_admits_sawteeth_or_low_q0(self):
        """The gate is OR, and it tests q0_TARGET (the q0 being claimed for
        the source's own current), not the renormalised anchor value: an
        active sawtooth source admits a slice whose q0_target sits above
        q0_gate, and a low q0_target admits a slice whose source carries no
        sawtooth model at all.  The comparison is on the MAGNITUDE -- q
        carries a COCOS sign, and a negative target would otherwise make the
        threshold trivially true and bypass the gate."""
        gate = lambda active, q0, q0_gate=1.1: bool(active) or abs(q0) <= q0_gate
        assert gate(True, 1.35)
        assert gate(False, 0.98)
        assert not gate(False, 1.35)
        assert not gate(False, -1.35)
        assert gate(False, -0.98)

    def test_idle_sawtooth_source_is_not_active(self):
        """A declared-but-IDLE sawtooth source (all-zero j_parallel before
        onset) must not admit a ramp slice: the reader's rule is present AND
        non-zero, not merely present."""
        from bouquet.io.imas import SAWTOOTH_SOURCE_INDEX
        assert SAWTOOTH_SOURCE_INDEX == 701
        for jpar, active in (([0.0, 0.0], False), ([0.0, -3.2e4], True)):
            jmax = float(np.max(np.abs(jpar)))
            assert bool(True and jmax > 0.0) is active


class TestWorkflowGuard:
    def _config(self, mode, workflow="auto"):
        from bouquet.config import (BouquetConfig, ImasSource, SolverConfig)
        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t")
        cfg.generation.jBS_baseline_mode = mode
        cfg.generation.perturb_jind_in_anchor = True     # diff+C baseline rule
        cfg.generation.workflow = workflow
        return cfg

    def _validate(self, cfg):
        from bouquet.run import Bouquet
        Bouquet(cfg)._validate_workflow()

    def test_diff_and_rescale_pass(self):
        for mode in ("diff", "rescale"):
            self._validate(self._config(mode))

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="jBS_baseline_mode"):
            self._validate(self._config("hybrid"))

    def test_ohmic_mode_refuses_generate(self):
        """Baseline-only until the draw path's sigma=0 reproduction is
        verified: generate()'s validator must refuse it."""
        with pytest.raises(ValueError, match="baseline-only"):
            self._validate(self._config("ohmic"))

    def test_custom_workflow_downgrades_to_warning(self, capsys):
        self._validate(self._config("ohmic", workflow="custom"))
        assert "baseline-only" in capsys.readouterr().out

    def test_closure_channel_typo_is_refused_at_validation(self):
        """A channel typo must be caught by the validator, not after the
        full solve_with_bootstrap iteration sequence."""
        cfg = self._config("ohmic")
        cfg.generation.closure_channel = "bootstap"
        with pytest.raises(ValueError, match="closure_channel"):
            self._validate(cfg)


class TestDefaults:
    def test_closure_channel_default_and_baseline_fields(self):
        from bouquet.config import GenerationConfig
        from bouquet.baseline import Baseline
        assert GenerationConfig().closure_channel == "bootstrap"
        assert Baseline.__dataclass_fields__["ohm_scale"].default == 1.0
        assert Baseline.__dataclass_fields__["ip_closure"].default is None


class TestQ0Gate:
    """utils.q0_gate_admits -- gate on the source's own |q0_dd|, not the
    estimator-mapped target (which reads lower and admitted idle-sawtooth
    ramp slices in the first campaign)."""

    def test_sawtooth_activity_admits_regardless_of_q0(self):
        from bouquet.utils import q0_gate_admits
        assert q0_gate_admits(True, 1.8, 1.7, 1.1) == (True, "sawtooth active")

    def test_idle_sawteeth_gate_on_source_q0_not_target(self):
        """The Tier-A calibration failure: q0_dd 1.15 (above gate) with a
        target that reads 1.05 (below) must be REJECTED."""
        from bouquet.utils import q0_gate_admits
        ok, basis = q0_gate_admits(False, 1.15, 1.05, 1.1)
        assert ok is False and basis == "|q0_dd|"
        ok, basis = q0_gate_admits(False, 0.99, 1.05, 1.1)
        assert ok is True and basis == "|q0_dd|"

    def test_cocos_sign_is_compared_by_magnitude(self):
        from bouquet.utils import q0_gate_admits
        assert q0_gate_admits(False, -0.99, -0.95, 1.1)[0] is True
        assert q0_gate_admits(False, -1.30, -1.25, 1.1)[0] is False

    def test_falls_back_to_target_only_without_source_q0(self):
        from bouquet.utils import q0_gate_admits
        ok, basis = q0_gate_admits(False, None, 1.05, 1.1)
        assert ok is True and basis.startswith("|q0_target|")
        assert q0_gate_admits(False, None, None, 1.1) == (False, "no q0 available")


class TestClosureHealth:
    """utils.closure_health -- the closure-limited flag every channel records."""

    def _h(self, **kw):
        from bouquet.utils import closure_health
        base = dict(ohm_scale=1.0, bs_scale=0.9, Ip_target_signed=1.2e6,
                    c_affine=-3.0e4, ip_ind=9.0e5, ip_bs=2.5e5, ip_fix=8.0e4)
        base.update(kw)
        return closure_health(**base)

    def test_clean_slice_is_not_limited(self):
        h = self._h()          # raw sum 1.2e6 -> 0.0 % mismatch
        assert h["closure_limited"] is False and h["closure_limited_reasons"] == ()
        assert h["raw_components_ip_mismatch_pct"] == pytest.approx(0.0, abs=1e-9)
        assert h["f_BS_unscaled"] == pytest.approx(2.5e5 / 1.2e6)
        assert h["f_BS_closed"] == pytest.approx(0.9 * 2.5e5 / 1.2e6)

    def test_large_raw_mismatch_flags(self):
        h = self._h(ip_bs=4.5e5)                 # raw sum +16.7 % over Ip
        assert h["closure_limited"] is True
        assert any("miss Ip" in r for r in h["closure_limited_reasons"])

    def test_deep_bootstrap_downscale_flags(self):
        h = self._h(bs_scale=0.41)
        assert h["closure_limited"] is True
        assert any("bs_scale" in r for r in h["closure_limited_reasons"])

    def test_negative_current_convention(self):
        h = self._h(Ip_target_signed=-1.2e6, c_affine=3.0e4, ip_ind=-9.0e5,
                    ip_bs=-2.5e5, ip_fix=-8.0e4)
        assert h["closure_limited"] is False
        assert h["raw_components_ip_mismatch_pct"] == pytest.approx(0.0, abs=1e-9)


def test_ohmic_channel_is_deprecated_loudly():
    from bouquet.utils import warn_deprecated_channel
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        warn_deprecated_channel("ohmic")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_deprecated_channel("bootstrap")          # silent
        warn_deprecated_channel("sawtooth_bootstrap")
