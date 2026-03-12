"""
Plotting utilities for perturbed equilibria.

Provides:
  - Core drawing functions (``draw_kinetic_profiles``,
    ``draw_pressure_profiles``, ``draw_jphi_profiles``) that operate
    on pre-loaded data arrays and matplotlib axes.
  - ``plot_family`` -- self-contained notebook-friendly API that loads
    everything from the ``.h5`` file and returns ``(fig, axes)``.
  - ``plot_tokamaker_comparison`` -- overview / single-shot comparison
    of TokaMaker reconstructions against source geqdsk files.
    Requires ``mygs`` (TokaMaker solver) and ``all_results`` (dict) to
    be available as module-level names in the calling namespace.
  - Legacy wrappers (``plot_kinetic_profiles``, ``plot_jphi_profiles``)
    for backward compatibility.
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    load_equilibrium,
    load_equilibrium_by_path,
    load_baseline_profiles,
    count_equilibria,
)
from .io import read_geqdsk

# =====================================================================
# Comparison plots: TokaMaker vs geqdsk / overview of all results
# =====================================================================
import matplotlib.cm as _cm
from scipy.spatial import cKDTree as _cKDTree
from matplotlib.collections import LineCollection as _LC
import matplotlib.colors as _mcolors_dev

_LW = 1.5   # universal line width

def _lcfs_from_psi(mygs, psi_arr, isoflux_fallback, psi_lcfs_val=None):
    r"""Extract the TokaMaker LCFS contour for a stored :math:`\psi` array.

    Uses ``tricontour`` on the TokaMaker mesh at the value
    ``psi_lcfs_val`` (defaults to ``mygs.psi_bounds[0]``) and returns
    the longest closed path as an ``(N, 2)`` array of ``[R, Z]`` points.
    Falls back to *isoflux_fallback* if no contour is found.

    .. note::
        Requires ``mygs`` to be set as a module-level (or notebook-level)
        name before calling.

    Parameters
    ----------
    psi_arr : ndarray
        Raw poloidal flux on the TokaMaker mesh.
    isoflux_fallback : ndarray, shape (N, 2)
        Isoflux target points used as a fallback when no contour path
        is found.
    psi_lcfs_val : float or None
        The :math:`\psi` value of the LCFS.  ``None`` uses
        ``mygs.psi_bounds[0]``.
    """
    if psi_lcfs_val is None:
        psi_lcfs_val = float(mygs.psi_bounds[0])
    _fig_tmp, _ax_tmp = plt.subplots(1, 1)
    try:
        _cs = _ax_tmp.tricontour(
            mygs.r[:, 0], mygs.r[:, 1], mygs.lc, psi_arr,
            levels=[psi_lcfs_val])
        _segs = [v for seg in _cs.allsegs for v in seg if len(v) > 4]
    finally:
        plt.close(_fig_tmp)
    if _segs:
        return max(_segs, key=len)
    return isoflux_fallback


def _core_contours(mygs, ax, psi_raw, nlevels=9):
    r"""Overplot core flux surface contours on *ax*.

    Normalises *psi_raw* by its own min/max and draws *nlevels* contours
    between :math:`\hat{\psi} = 0.1` and :math:`0.9`.

    .. note::
        Requires ``mygs`` to be set as a module-level (or notebook-level)
        name before calling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    psi_raw : ndarray
        Raw poloidal flux on the TokaMaker mesh.
    nlevels : int
        Number of contour levels.
    """
    _p_lo = psi_raw.min()
    _p_hi = psi_raw.max()
    if abs(_p_hi - _p_lo) < 1e-10:
        return
    _psi_n = (psi_raw - _p_lo) / (_p_hi - _p_lo)
    ax.tricontour(mygs.r[:, 0], mygs.r[:, 1], mygs.lc, _psi_n,
                  levels=np.linspace(0.1, 0.9, nlevels),
                  colors='steelblue', linewidths=0.5, alpha=0.4)


def _isoflux_deviation_plot(ax, fig, iso_pts, lcfs_pts, R_bnd, Z_bnd,
                             max_dev_mm=10.0, max_seg_len=0.1):
    """Colour-coded boundary deviation. Returns (devs, max_mm, rms_mm)."""
    tree = _cKDTree(lcfs_pts)
    devs, _ = tree.query(iso_pts)

    dev_cmap = _mcolors_dev.LinearSegmentedColormap.from_list(
        'dev_cmap', ['limegreen', 'yellow', 'red'])
    dev_norm = _mcolors_dev.Normalize(vmin=0.0, vmax=max_dev_mm * 1e-3)

    seg_list, col_list = [], []
    for _si in range(len(iso_pts) - 1):
        p0, p1 = iso_pts[_si], iso_pts[_si + 1]
        if np.linalg.norm(p1 - p0) <= max_seg_len:
            seg_list.append([p0, p1])
            col_list.append(devs[_si])

    if seg_list:
        lc_coll = _LC(np.array(seg_list), cmap=dev_cmap, norm=dev_norm,
                      linewidth=3, zorder=5)
        lc_coll.set_array(np.array(col_list))
        ax.add_collection(lc_coll)
        cax = ax.inset_axes([1.02, 0, 0.05, 1])
        cbar = fig.colorbar(lc_coll, cax=cax, label='Boundary deviation [mm]')
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x * 1e3:.1f}'))

    lcfs_R, lcfs_Z = [lcfs_pts[0, 0]], [lcfs_pts[0, 1]]
    for _bi in range(1, len(lcfs_pts)):
        if np.linalg.norm(lcfs_pts[_bi] - lcfs_pts[_bi - 1]) > max_seg_len:
            lcfs_R.append(np.nan); lcfs_Z.append(np.nan)
        lcfs_R.append(lcfs_pts[_bi, 0]); lcfs_Z.append(lcfs_pts[_bi, 1])
    ax.plot(lcfs_R, lcfs_Z, 'k-', lw=_LW, label='TokaMaker LCFS')

    ax.set_xlabel('$R$ [m]'); ax.set_ylabel('$Z$ [m]')
    ax.set_aspect('equal')
    ax.set_xlim(R_bnd.min() - 0.12, R_bnd.max() + 0.12)
    ax.set_ylim(Z_bnd.min() - 0.20, Z_bnd.max() + 0.20)
    ax.legend(fontsize=8, loc='lower right')

    max_mm = devs.max() * 1e3
    rms_mm = np.sqrt((devs**2).mean()) * 1e3
    return devs, max_mm, rms_mm


def plot_tokamaker_comparison(mygs, all_results, plot_idx=None):
    """Compare TokaMaker reconstructions against source geqdsk files.

    .. note::
        Requires ``mygs`` (TokaMaker solver object) and ``all_results``
        (dict mapping geqdsk file names to result dicts) to be available
        as module-level or notebook-level names.

    Parameters
    ----------
    plot_idx : int or None
        ``None`` → overview: overplot ALL results on each subplot.
        ``int``  → single-shot: compare that result against its geqdsk.

    Boundary panels (row 2):
      axes[2,0] — TokaMaker LCFS + geqdsk boundary + core flux surface
                  contours.
      axes[2,1] — Quantified boundary deviation: colour-coded segments
                  (green→red) showing nearest-neighbour distance from
                  each isoflux target point to the TokaMaker LCFS;
                  overview mode shows a max/RMS bar chart.
      axes[2,2] — :math:`l_i(1)` and :math:`I_p` % error bars
                  (TokaMaker − geqdsk) / |geqdsk|.
    """
    keys = list(all_results.keys())

    # colourblind-safe palette (Wong / Okabe-Ito)
    _C1 = '#0072B2'   # deep blue
    _C2 = '#D55E00'   # deep orange
    _C3 = '#009E73'   # green

    # -----------------------------------------------------------------------
    # OVERVIEW MODE
    # -----------------------------------------------------------------------
    if plot_idx is None:
        N = max(len(keys), 1)
        colors = _cm.tab10(np.linspace(0, 0.9, N))

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('TokaMaker overview — all reconstructions', fontsize=13, y=0.98)

        _li_efit_vals, _li_tkmkr_vals, _short_labels = [], [], []
        _li_pct_vals, _Ip_pct_vals = [], []
        _ov_max_dev_mm, _ov_rms_dev_mm = [], []
        _all_lcfs = []

        ax_te = axes[1, 2].twinx()

        for color, gkey in zip(colors, keys):
            r = all_results[gkey]
            psi_N = r['psi_N_grid']
            lbl = gkey.replace('.geqdsk', '')
            _short_labels.append(lbl)

            # (0,0) total j_phi — dashed=TokaMaker, dotted=geqdsk
            axes[0, 0].plot(psi_N, r['j_phi_fit'] / 1e6, color=color, lw=_LW, ls='--', label=lbl)
            axes[0, 0].plot(psi_N, r['eqdsk_jtor'] / 1e6, color=color, lw=_LW, ls=':', alpha=0.75)

            # (0,1) j_phi components (both TokaMaker — dashed=j_ind, dash-dot=j_BS)
            axes[0, 1].plot(psi_N, r['j_inductive_fit'] / 1e6, color=color, lw=_LW, ls='--',
                            label=f'{lbl} $j_{{\\rm ind}}$')
            axes[0, 1].plot(psi_N, r['j_BS_used'] / 1e6, color=color, lw=_LW, ls='-.',
                            label=f'{lbl} $j_{{BS}}$')

            # (0,2) residuals
            res = (r['j_phi_fit'] - r['eqdsk_jtor']) / 1e6
            rms = np.sqrt(np.mean(res**2))
            axes[0, 2].plot(psi_N, res, color=color, lw=_LW,
                            label=f'{lbl}  RMS={rms:.4f}')

            # (1,0) pressure — dashed=TokaMaker, dotted=geqdsk
            axes[1, 0].plot(psi_N, r['pres_tokamaker'] / 1e3, color=color, lw=_LW, ls='--', label=lbl)
            axes[1, 0].plot(psi_N, r['eqdsk_pres'] / 1e3, color=color, lw=_LW, ls=':', alpha=0.75)

            # (1,1) pprime (TokaMaker only)
            axes[1, 1].plot(psi_N, r['pprime'], color=color, lw=_LW, ls='--', label=lbl)

            # (1,2) ne dashed, te dash-dot
            axes[1, 2].plot(psi_N, r['ne'] / 1e19, color=color, lw=_LW, ls='--')
            ax_te.plot(psi_N, r['te'] / 1e3, color=color, lw=_LW, ls='-.')

            # (2,0) LCFS — dashed=TokaMaker, dotted=geqdsk
            _lcfs = _lcfs_from_psi(mygs, r['psi'], r['isoflux_pts'], r.get('psi_lcfs_val'))
            _all_lcfs.append(_lcfs)
            axes[2, 0].plot(_lcfs[:, 0], _lcfs[:, 1], color=color, lw=_LW, ls='--', label=lbl)
            axes[2, 0].plot(r['eqdsk_boundary_R'], r['eqdsk_boundary_Z'],
                            color=color, lw=_LW, ls=':', alpha=0.75)
            _p_lo_ov, _p_hi_ov = r['psi'].min(), r['psi'].max()
            if abs(_p_hi_ov - _p_lo_ov) > 1e-10:
                _psi_n_ov = (r['psi'] - _p_lo_ov) / (_p_hi_ov - _p_lo_ov)
                axes[2, 0].tricontour(mygs.r[:, 0], mygs.r[:, 1], mygs.lc, _psi_n_ov,
                                      levels=np.linspace(0.1, 0.9, 9),
                                      colors=[color], linewidths=0.5, alpha=0.35)

            # Boundary deviation stats
            _tree_ov = _cKDTree(_lcfs)
            _devs_ov, _ = _tree_ov.query(r['isoflux_pts'])
            _ov_max_dev_mm.append(_devs_ov.max() * 1e3)
            _ov_rms_dev_mm.append(np.sqrt((_devs_ov**2).mean()) * 1e3)

            # li and Ip % errors
            li_eqdsk = r['eqdsk_li']
            _li_efit = li_eqdsk.get('li(1)_EFIT', li_eqdsk.get('li(1)', 0.0))
            _li_efit_vals.append(_li_efit)
            _li_tkmkr_vals.append(r['li_final'])
            _li_pct_vals.append((r['li_final'] - _li_efit) / abs(_li_efit) * 100.0)
            _Ip_ref = abs(r['eqdsk_Ip'])
            _Ip_tkmkr_ov = r.get('Ip_tokamaker', float('nan'))
            _Ip_pct_vals.append((_Ip_tkmkr_ov - _Ip_ref) / _Ip_ref * 100.0)

        # ----- Axes labels / titles -----
        axes[0, 0].set_xlabel(r'$\psi_N$')
        axes[0, 0].set_ylabel(r'$j_\phi$ [MA m$^{-2}$]')
        axes[0, 0].set_title(r'Total $j_\phi$ (dashed=TokaMaker, dotted=geqdsk)')
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(ls=':')

        axes[0, 1].set_xlabel(r'$\psi_N$')
        axes[0, 1].set_ylabel(r'$j$ [MA m$^{-2}$]')
        axes[0, 1].set_title(r'$j_\phi$ components (dashed=$j_{\rm ind}$, dash-dot=$j_{BS}$)')
        axes[0, 1].legend(fontsize=6, ncol=2)
        axes[0, 1].grid(ls=':')

        axes[0, 2].axhline(0, color='k', ls=':', lw=_LW)
        axes[0, 2].set_xlabel(r'$\psi_N$')
        axes[0, 2].set_ylabel(r'$\Delta j_\phi$ [MA m$^{-2}$]')
        axes[0, 2].set_title(r'$j_\phi$ residuals (TokaMaker \u2212 geqdsk)')
        axes[0, 2].legend(fontsize=7)
        axes[0, 2].grid(ls=':')

        axes[1, 0].set_xlabel(r'$\psi_N$')
        axes[1, 0].set_ylabel(r'$p$ [kPa]')
        axes[1, 0].set_title('Pressure (dashed=TokaMaker, dotted=geqdsk)')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(ls=':')

        axes[1, 1].set_xlabel(r'$\psi_N$')
        axes[1, 1].set_ylabel(r"$p'$ [Pa Wb$^{-1}$]")
        axes[1, 1].set_title(r"$p'(\psi_N)$")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(ls=':')

        axes[1, 2].set_xlabel(r'$\psi_N$')
        axes[1, 2].set_ylabel(r'$n_e$ [$10^{19}$ m$^{-3}$] (dashed)')
        ax_te.set_ylabel(r'$T_e$ [keV] (dash-dot)')
        axes[1, 2].set_title('Kinetic profiles')
        axes[1, 2].grid(ls=':')
        from matplotlib.lines import Line2D as _L2D
        axes[1, 2].legend(
            [_L2D([0], [0], color='k', lw=_LW, ls='--'),
             _L2D([0], [0], color='k', lw=_LW, ls='-.')],
            [r'$n_e$', r'$T_e$'], fontsize=8, loc='upper right')

        # (2,0) LCFS panel — crop to union of all LCFS bounding boxes
        axes[2, 0].set_xlabel('$R$ [m]')
        axes[2, 0].set_ylabel('$Z$ [m]')
        axes[2, 0].set_aspect('equal')
        axes[2, 0].set_title('LCFS + core flux surfaces (dashed=TokaMaker, dotted=geqdsk)')
        axes[2, 0].legend(fontsize=7)
        _lcfs_pad = 0.06
        _all_R_ov = np.concatenate([_l[:, 0] for _l in _all_lcfs])
        _all_Z_ov = np.concatenate([_l[:, 1] for _l in _all_lcfs])
        axes[2, 0].set_xlim(_all_R_ov.min() - _lcfs_pad, _all_R_ov.max() + _lcfs_pad)
        axes[2, 0].set_ylim(_all_Z_ov.min() - _lcfs_pad, _all_Z_ov.max() + _lcfs_pad)

        # (2,1) boundary deviation bar chart
        _x_bars = np.arange(len(keys))
        _bar_w = 0.35
        axes[2, 1].bar(_x_bars - _bar_w/2, _ov_max_dev_mm, _bar_w,
                       label='Max dev.', color=_C2, edgecolor='k')
        axes[2, 1].bar(_x_bars + _bar_w/2, _ov_rms_dev_mm, _bar_w,
                       label='RMS dev.', color=_C1, edgecolor='k')
        axes[2, 1].set_xticks(_x_bars)
        axes[2, 1].set_xticklabels(_short_labels, rotation=30, ha='right', fontsize=8)
        axes[2, 1].set_ylabel('Boundary deviation [mm]')
        axes[2, 1].set_title('TokaMaker LCFS vs geqdsk boundary deviation')
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(axis='y', ls=':')

        # (2,2) % error bars for li(1) and Ip per shot
        _x_err = np.arange(len(keys))
        _w_err = 0.35
        axes[2, 2].bar(_x_err - _w_err/2, _li_pct_vals, _w_err,
                       label=r'$l_i(1)$', color=_C1, edgecolor='k')
        axes[2, 2].bar(_x_err + _w_err/2, _Ip_pct_vals, _w_err,
                       label=r'$I_p$', color=_C2, edgecolor='k')
        axes[2, 2].axhline(0, color='k', ls='--', lw=0.8)
        axes[2, 2].set_xticks(_x_err)
        axes[2, 2].set_xticklabels(_short_labels, rotation=30, ha='right', fontsize=8)
        axes[2, 2].set_ylabel('% error  (TokaMaker \u2212 geqdsk) / |geqdsk|')
        axes[2, 2].set_title(r'$l_i(1)$ and $I_p$ % error')
        axes[2, 2].legend(fontsize=8)
        axes[2, 2].grid(axis='y', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()

    # -----------------------------------------------------------------------
    # SINGLE-SHOT COMPARISON MODE
    # -----------------------------------------------------------------------
    else:
        geqdsk_key = keys[plot_idx]
        r = all_results[geqdsk_key]
        eqdsk_ref = read_geqdsk(geqdsk_key)

        psi_N = r['psi_N_grid']
        R_bnd = r['eqdsk_boundary_R']
        Z_bnd = r['eqdsk_boundary_Z']
        residual_jphi = r['j_phi_fit'] - r['eqdsk_jtor']
        rms_jphi = np.sqrt(np.mean(residual_jphi**2))

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f'TokaMaker vs geqdsk comparison:  {geqdsk_key}', fontsize=13, y=0.98)

        # (0,0) Total j_phi
        ax = axes[0, 0]
        ax.plot(psi_N, r['eqdsk_jtor'] / 1e6, 'k-', lw=_LW, label=r'geqdsk $j_\phi$')
        ax.plot(psi_N, r['j_phi_fit'] / 1e6, color=_C2, ls='--', lw=_LW,
                label=r'TokaMaker $j_\phi$')
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$j_\phi$ [MA m$^{-2}$]')
        ax.set_title(r'Total $j_\phi$'); ax.legend(fontsize=8); ax.grid(ls=':')

        # (0,1) j_phi components
        ax = axes[0, 1]
        ax.plot(psi_N, r['j_inductive_fit'] / 1e6, color=_C1, lw=_LW,
                label=r'$j_\mathrm{inductive}$ (fit)')
        ax.plot(psi_N, r['j_BS_used'] / 1e6, color=_C3, lw=_LW, ls='-.',
                label=rf'$j_{{BS}}$ (\u00d7{r["bs_factor_final"]:.3f})')
        ax.plot(psi_N, r['j_phi_fit'] / 1e6, color=_C2, ls='--', lw=_LW,
                label=r'$j_\mathrm{ind} + j_{BS}$')
        ax.plot(psi_N, r['eqdsk_jtor'] / 1e6, 'k-', lw=_LW,
                label=r'geqdsk $j_\phi$')
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$j$ [MA m$^{-2}$]')
        ax.set_title(r'$j_\phi$ components'); ax.legend(fontsize=7); ax.grid(ls=':')

        # (0,2) Residual
        ax = axes[0, 2]
        ax.plot(psi_N, residual_jphi / 1e6, color=_C2, lw=_LW)
        ax.axhline(0, color='k', ls=':', lw=_LW)
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$\Delta j_\phi$ [MA m$^{-2}$]')
        ax.set_title(rf'$j_\phi$ residual  (RMS = {rms_jphi/1e6:.4f} MA m$^{{-2}}$)')
        ax.grid(ls=':')

        # (1,0) Pressure
        ax = axes[1, 0]
        ax.plot(psi_N, eqdsk_ref.pres / 1e3, 'k-', lw=_LW, label='geqdsk $p$')
        ax.plot(psi_N, r['pres_tokamaker'] / 1e3, color=_C2, ls='--', lw=_LW,
                label='TokaMaker $p$ (kinetic)')
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$p$ [kPa]')
        ax.set_title('Pressure profile'); ax.legend(fontsize=8); ax.grid(ls=':')

        # (1,1) pprime
        ax = axes[1, 1]
        ax.plot(psi_N, eqdsk_ref.pprime, 'k-', lw=_LW, label="geqdsk $p'$")
        ax.plot(psi_N, r['pprime'], color=_C2, ls='--', lw=_LW, label="TokaMaker $p'$")
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r"$p'$ [Pa Wb$^{-1}$]")
        ax.set_title(r"$p'(\psi)$ comparison"); ax.legend(fontsize=8); ax.grid(ls=':')

        # (1,2) Kinetics
        ax = axes[1, 2]; ax2 = ax.twinx()
        ax.plot(psi_N, r['ne'] / 1e19, color=_C1, lw=_LW,
                label=r'$n_e$ [$10^{19}$ m$^{-3}$]')
        ax2.plot(psi_N, r['te'] / 1e3, color=_C2, lw=_LW, ls='--',
                 label=r'$T_e$ [keV]')
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel(r'$n_e$ [$10^{19}$ m$^{-3}$]', color=_C1)
        ax2.set_ylabel(r'$T_e$ [keV]', color=_C2)
        ax.tick_params(axis='y', labelcolor=_C1)
        ax2.tick_params(axis='y', labelcolor=_C2)
        h1_, l1_ = ax.get_legend_handles_labels()
        h2_, l2_ = ax2.get_legend_handles_labels()
        ax.legend(h1_ + h2_, l1_ + l2_, fontsize=7, loc='upper right')
        ax.set_title('Kinetic profiles'); ax.grid(ls=':')

        # --- Shared LCFS extraction ---
        _tk_lcfs = _lcfs_from_psi(mygs,r['psi'], r['isoflux_pts'], r.get('psi_lcfs_val'))

        # --- (2,0) LCFS + core flux surface contours ---
        ax_bndy = axes[2, 0]
        ax_bndy.plot(_tk_lcfs[:, 0], _tk_lcfs[:, 1],
                     color=_C1, lw=_LW, label='TokaMaker LCFS', zorder=5)
        ax_bndy.plot(R_bnd, Z_bnd, color=_C2, lw=_LW, ls='--',
                     label='geqdsk boundary', zorder=6)
        _core_contours(mygs,ax_bndy, r['psi'])
        _lcfs_pad = 0.06
        ax_bndy.set_xlim(_tk_lcfs[:, 0].min() - _lcfs_pad, _tk_lcfs[:, 0].max() + _lcfs_pad)
        ax_bndy.set_ylim(_tk_lcfs[:, 1].min() - _lcfs_pad, _tk_lcfs[:, 1].max() + _lcfs_pad)
        ax_bndy.set_xlabel('$R$ [m]'); ax_bndy.set_ylabel('$Z$ [m]')
        ax_bndy.set_aspect('equal')
        ax_bndy.legend(fontsize=8, loc='lower right')
        ax_bndy.set_title('LCFS + core flux surfaces')

        # --- (2,1) Quantified boundary deviation ---
        ax_dev = axes[2, 1]
        mygs.plot_machine(fig, ax_dev)
        _devs_ss, _dev_max_mm, _dev_rms_mm = _isoflux_deviation_plot(
            ax_dev, fig,
            iso_pts=r['isoflux_pts'],
            lcfs_pts=_tk_lcfs,
            R_bnd=R_bnd, Z_bnd=Z_bnd,
            max_dev_mm=10.0, max_seg_len=0.1)
        ax_dev.set_title(
            f'Boundary deviation  max={_dev_max_mm:.2f} mm  RMS={_dev_rms_mm:.2f} mm')

        # (2,2) % error bars for li(1) and Ip
        ax = axes[2, 2]
        li_eqdsk  = r['eqdsk_li']
        li_efit   = li_eqdsk.get('li(1)_EFIT', li_eqdsk.get('li(1)', 0.0))
        li_tkmkr  = r['li_final']
        li_pct    = (li_tkmkr - li_efit) / abs(li_efit) * 100.0
        Ip_eqdsk  = abs(r['eqdsk_Ip'])
        Ip_tkmkr  = r.get('Ip_tokamaker', float('nan'))
        Ip_pct    = (Ip_tkmkr - Ip_eqdsk) / Ip_eqdsk * 100.0
        _pct_vals = [li_pct, Ip_pct]
        _bar_cols = [_C1 if v >= 0 else _C2 for v in _pct_vals]
        bars = ax.bar([0, 1], _pct_vals, color=_bar_cols, edgecolor='k', width=0.5)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r'$l_i(1)$', r'$I_p$'], fontsize=11)
        ax.set_ylabel('% error  (TokaMaker \u2212 geqdsk) / |geqdsk|')
        ax.set_title(r'TokaMaker vs geqdsk: $l_i(1)$ and $I_p$ % error')
        ax.grid(axis='y', ls=':')
        for bar_, val_ in zip(bars, _pct_vals):
            _yoff = abs(val_) * 0.05 + 0.02
            _va   = 'bottom' if val_ >= 0 else 'top'
            _ytxt = val_ + (_yoff if val_ >= 0 else -_yoff)
            ax.text(bar_.get_x() + bar_.get_width() / 2, _ytxt,
                    f'{val_:+.3f}%', ha='center', va=_va, fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()

        print(f"\n--- Summary for {geqdsk_key} ---")
        print(f"  bs_factor_final : {r['bs_factor_final']:.6f}")
        print(f"  li(1) achieved  : {r['li_final']:.6f}")
        print(f"  j_phi RMS error : {rms_jphi:.2f} A/m^2  ({rms_jphi/1e6:.4f} MA/m^2)")
        print(f"  eqdsk Ip        : {r['eqdsk_Ip']:.0f} A")
        print(f"  TokaMaker Ip    : {Ip_tkmkr:.0f} A  ({Ip_pct:+.3f}%)")
        print(f"  Boundary dev.   : max={_dev_max_mm:.2f} mm  RMS={_dev_rms_mm:.2f} mm")


# ====================================================================
#  Core drawing functions (no I/O -- operate on axes + arrays)
# ====================================================================
def draw_kinetic_profiles(axes, psi_N, ne, ni, te, ti,
                          sigma_ne, sigma_ni, sigma_te, sigma_ti,
                          perturbed_data_list=None):
    r"""Draw kinetic profiles (ne, ni, Te, Ti) on a 2x2 axes array.

    Parameters
    ----------
    axes : ndarray of Axes, shape (2, 2)
        ``[0,0]`` = :math:`n_e`, ``[0,1]`` = :math:`n_i`,
        ``[1,0]`` = :math:`T_e`, ``[1,1]`` = :math:`T_i`.
    psi_N : 1-D array
        Normalised poloidal flux grid (baseline).
    ne, ni, te, ti : 1-D arrays
        Baseline kinetic profiles.
    sigma_ne, sigma_ni, sigma_te, sigma_ti : 1-D arrays
        1-:math:`\sigma` uncertainty envelopes.
    perturbed_data_list : list[dict] or None
        Each dict must have keys ``'n_e [m^-3]'``, ``'n_i [m^-3]'``,
        ``'T_e [eV]'``, ``'T_i [eV]'``.
    """
    _pairs = [
        #  axis       orig  scale  sigma      color        label      ylabel
        (axes[0, 0], ne, 1.0,  sigma_ne, "tab:red",    r"$n_e$", r"n [m$^{-3}$]"),
        (axes[0, 1], ni, 1.0,  sigma_ni, "tab:orange", r"$n_i$", None),
        (axes[1, 0], te, 1e-3, sigma_te, "tab:blue",   r"$T_e$", r"T [keV]"),
        (axes[1, 1], ti, 1e-3, sigma_ti, "tab:cyan",   r"$T_i$", None),
    ]
    _keys = ["n_e [m^-3]", "n_i [m^-3]", "T_e [eV]", "T_i [eV]"]

    # ---- baseline + sigma bands ------------------------------------------
    for a, orig, scale, sig, clr, lbl, ylabel in _pairs:
        a.cla()
        a.plot(psi_N, orig * scale, c="k", lw=2,
               label=f"input {lbl}", zorder=3)
        a.fill_between(
            psi_N,
            (orig - sig) * scale,
            (orig + sig) * scale,
            alpha=0.25, color=clr,
            label=r"$\pm\,1\sigma_{\rm exp}$", zorder=1,
        )
        a.plot(psi_N, (orig + 2 * sig) * scale, c="k", ls=":",
               lw=1.5, alpha=0.5, label=r"$\pm\,2\sigma_{\rm exp}$",
               zorder=2)
        a.plot(psi_N, (orig - 2 * sig) * scale, c="k", ls=":",
               lw=1.5, alpha=0.5, zorder=2)
        a.grid(ls=":")
        if ylabel:
            a.set_ylabel(ylabel)

    # ---- overlay perturbed profiles --------------------------------------
    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            for (a, orig, scale, sig, clr, lbl, ylabel), key in zip(
                _pairs, _keys
            ):
                a.plot(
                    psi_N, data[key] * scale,
                    c=clr, alpha=0.9, lw=1.5,
                    label=f"perturbed ({n_equils})" if i == 0 else None,
                    zorder=2,
                )

    # ---- legends and axis labels -----------------------------------------
    for a, *_ in _pairs:
        a.legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel(r"$\hat{\psi}$")
    axes[1, 1].set_xlabel(r"$\hat{\psi}$")


def draw_pressure_profiles(ax, psi_N, pressure, perturbed_data_list=None):
    """Draw total pressure overlay on a single axes.

    Parameters
    ----------
    ax : Axes
    psi_N : 1-D array
    pressure : 1-D array
        Baseline total pressure [Pa].
    perturbed_data_list : list[dict] or None
        Each dict must have ``'pressure [Pa]'``.
    """
    _kPa = 1e-3
    ax.cla()
    ax.plot(psi_N, pressure * _kPa, c="k", lw=2,
            label="input pressure", zorder=3)
    ax.grid(ls=":")
    ax.set_xlabel(r"$\hat{\psi}$")
    ax.set_ylabel("Pressure [kPa]")

    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            if "pressure [Pa]" in data:
                ax.plot(
                    psi_N, data["pressure [Pa]"] * _kPa,
                    c="tab:brown", alpha=0.9, lw=1.5,
                    label=f"perturbed ({n_equils})" if i == 0 else None,
                    zorder=2,
                )

    ax.legend(loc="best", fontsize=8)


def draw_jphi_total(ax, psi_N, j_phi, sigma_jphi,
                    perturbed_data_list=None):
    r"""Draw total :math:`j_\phi` with uncertainty bands on a single axes.

    Parameters
    ----------
    ax : Axes
    psi_N : 1-D array
    j_phi : 1-D array
        Baseline total :math:`j_\phi` [A m^-2].
    sigma_jphi : 1-D array
    perturbed_data_list : list[dict] or None
        Each dict must have ``'j_phi [A m^-2]'``.
    """
    _MA = 1e-6  # A → MA

    ax.cla()
    ax.plot(psi_N, j_phi * _MA, c="k", lw=2,
            label=r"input $j_\phi$", zorder=4)
    ax.fill_between(
        psi_N,
        (j_phi - sigma_jphi) * _MA,
        (j_phi + sigma_jphi) * _MA,
        alpha=0.25, color="tab:purple",
        label=r"$\pm\,1\sigma_{\rm exp}$", zorder=1,
    )
    ax.plot(psi_N, (j_phi + 2 * sigma_jphi) * _MA, c="k", ls=":", lw=1.5,
            alpha=0.5, label=r"$\pm\,2\sigma_{\rm exp}$", zorder=2)
    ax.plot(psi_N, (j_phi - 2 * sigma_jphi) * _MA, c="k", ls=":", lw=1.5,
            alpha=0.5, zorder=2)
    ax.set_ylabel(r"$j_\phi$ [MA/m$^2$]")
    ax.set_xlabel(r"$\hat{\psi}$")
    ax.grid(ls=":")

    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            ax.plot(psi_N, data["j_phi [A m^-2]"] * _MA, c="tab:purple",
                    lw=1.5, alpha=0.9,
                    label=f"perturbed ({n_equils})" if i == 0 else None,
                    zorder=3)

    ax.legend(loc="best", fontsize=8)


def draw_jphi_components(axes, psi_N, perturbed_data_list=None):
    r"""Draw :math:`j_\phi` component decomposition on a (2, 1) axes array.

    ``axes[0]`` = :math:`j_{\rm BS}` (solid) with :math:`j_{\rm BS,edge}`
    (dashed), ``axes[1]`` = :math:`j_{\rm inductive}`.  Total
    :math:`j_\phi` is shown as a black dashed reference on both panels.

    Parameters
    ----------
    axes : array-like of 2 Axes
    psi_N : 1-D array
    perturbed_data_list : list[dict] or None
        Each dict must have ``'j_phi [A m^-2]'``, ``'j_BS [A m^-2]'``,
        ``'j_inductive [A m^-2]'``.  ``'j_BS,edge [A m^-2]'`` is
        optional.
    """
    _MA = 1e-6  # A → MA

    _sub = [
        (axes[0], "j_BS [A m^-2]",        r"$j_{\rm BS}$",        "tab:green"),
        (axes[1], "j_inductive [A m^-2]",  r"$j_{\rm inductive}$", "tab:orange"),
    ]
    for ax, key, label, color in _sub:
        ax.cla()
        ax.set_ylabel(f"{label} " + r"[MA/m$^2$]")
        ax.grid(ls=":")

    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            lbl = f"perturbed ({n_equils})" if i == 0 else None

            # reference: total j_phi on both panels
            for ax, *_ in _sub:
                ax.plot(psi_N, data["j_phi [A m^-2]"] * _MA,
                        c="k", ls="--", lw=1.2, alpha=0.5,
                        label=r"$j_\phi$ (total)" if i == 0 else None,
                        zorder=2)

            # component curves
            for ax, key, label, color in _sub:
                if key in data:
                    ax.plot(psi_N, data[key] * _MA, c=color, lw=1.5,
                            alpha=0.9, label=lbl, zorder=3)

            # j_BS,edge overlay on the j_BS panel
            if "j_BS,edge [A m^-2]" in data:
                axes[0].plot(
                    psi_N, data["j_BS,edge [A m^-2]"] * _MA,
                    c="tab:red", ls="--", lw=1.5, alpha=0.8,
                    label=r"$j_{\rm BS,edge}$" if i == 0 else None,
                    zorder=4,
                )

    for ax in axes:
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(r"$\hat{\psi}$")


def draw_jphi_profiles(axes, psi_N, j_phi, sigma_jphi,
                       perturbed_data_list=None):
    r"""**Deprecated** -- use :func:`draw_jphi_total` and
    :func:`draw_jphi_components` instead.

    Draws on 3 vertically stacked axes for backward compatibility.
    """
    warnings.warn(
        "draw_jphi_profiles() is deprecated.  Use draw_jphi_total() and "
        "draw_jphi_components() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    draw_jphi_total(axes[0], psi_N, j_phi, sigma_jphi,
                    perturbed_data_list=perturbed_data_list)
    draw_jphi_components(axes[1:], psi_N,
                         perturbed_data_list=perturbed_data_list)


# ====================================================================
#  Data loading helper
# ====================================================================
def _load_all_perturbations(h5path, scan_value=None):
    """Load all perturbed equilibria for a scan value as a list of dicts."""
    n = count_equilibria(h5path, scan_value=scan_value)
    return [
        load_equilibrium_by_path(h5path, count=i, scan_value=scan_value)
        for i in range(n)
    ]


# ====================================================================
#  Notebook-friendly API
# ====================================================================
def plot_family(h5path_or_header, scan_value=None, mode="kinetic"):
    """Plot a family of perturbed equilibria from an HDF5 file.

    Parameters
    ----------
    h5path_or_header : str
        Path to the ``.h5`` file, or the header string (without
        extension).
    scan_value : str, float, or None
        Baseline scan-value label.  ``None`` for flat-layout files.
    mode : str
        ``'kinetic'``, ``'pressure'``, ``'j-phi'``, or ``'all'``.

    Returns
    -------
    fig : Figure  or  list[Figure]   (when *mode* = ``'all'``)
    axes : Axes   or  list[Axes]     (when *mode* = ``'all'``)
    """
    # ---- resolve path ----------------------------------------------------
    if not h5path_or_header.endswith(".h5"):
        h5path = os.path.abspath(f"{h5path_or_header}.h5")
    else:
        h5path = os.path.abspath(h5path_or_header)

    # ---- load data -------------------------------------------------------
    bl = load_baseline_profiles(h5path, scan_value=scan_value)
    psi_N = bl["psi_N"]
    perturbed = _load_all_perturbations(h5path, scan_value=scan_value)

    figs = []
    axes_list = []

    if mode in ("kinetic", "all"):
        fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
        draw_kinetic_profiles(
            ax, psi_N,
            bl["n_e [m^-3]"], bl["n_i [m^-3]"],
            bl["T_e [eV]"],   bl["T_i [eV]"],
            bl["sigma_ne [m^-3]"], bl["sigma_ni [m^-3]"],
            bl["sigma_te [eV]"],   bl["sigma_ti [eV]"],
            perturbed_data_list=perturbed,
        )
        fig.tight_layout()
        figs.append(fig)
        axes_list.append(ax)

    if mode in ("pressure", "all"):
        fig, ax = plt.subplots(figsize=(6, 4))
        draw_pressure_profiles(
            ax, psi_N, bl["pressure [Pa]"],
            perturbed_data_list=perturbed,
        )
        fig.tight_layout()
        figs.append(fig)
        axes_list.append(ax)

    if mode in ("j-phi", "all"):
        fig_jt, ax_jt = plt.subplots(figsize=(6, 4))
        draw_jphi_total(
            ax_jt, psi_N,
            bl["j_phi [A m^-2]"], bl["sigma_jphi [A m^-2]"],
            perturbed_data_list=perturbed,
        )
        fig_jt.tight_layout()
        figs.append(fig_jt)
        axes_list.append(ax_jt)

        fig_jc, ax_jc = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        draw_jphi_components(
            ax_jc, psi_N,
            perturbed_data_list=perturbed,
        )
        fig_jc.tight_layout()
        figs.append(fig_jc)
        axes_list.append(ax_jc)

    if mode == "all":
        return figs, axes_list
    if len(figs) == 1:
        return figs[0], axes_list[0]
    return figs, axes_list


# ====================================================================
#  Legacy wrappers (deprecated)
# ====================================================================
def plot_kinetic_profiles(header, n_equils, psi_N, ne, ni, te, ti,
                          sigma_ne, sigma_ni, sigma_te, sigma_ti):
    """**Deprecated** -- use :func:`plot_family` instead."""
    warnings.warn(
        "plot_kinetic_profiles() is deprecated.  Use plot_family() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    perturbed = [load_equilibrium(header, count=i) for i in range(n_equils)]
    draw_kinetic_profiles(
        axes, psi_N, ne, ni, te, ti,
        sigma_ne, sigma_ni, sigma_te, sigma_ti,
        perturbed_data_list=perturbed,
    )
    plt.tight_layout()
    plt.show()


def plot_jphi_profiles(psi_N, input_j_phi, sigma_jphi, header, n_equils):
    """**Deprecated** -- use :func:`plot_family` instead."""
    warnings.warn(
        "plot_jphi_profiles() is deprecated.  Use plot_family() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    perturbed = [load_equilibrium(header, count=i) for i in range(n_equils)]
    draw_jphi_profiles(
        axes, psi_N, input_j_phi, sigma_jphi,
        perturbed_data_list=perturbed,
    )
    plt.tight_layout()
    plt.show()
