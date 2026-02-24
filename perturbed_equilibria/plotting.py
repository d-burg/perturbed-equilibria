"""
Plotting utilities for perturbed equilibria.

Provides:
  - Core drawing functions (``draw_kinetic_profiles``,
    ``draw_pressure_profiles``, ``draw_jphi_profiles``) that operate
    on pre-loaded data arrays and matplotlib axes.
  - ``plot_family`` -- self-contained notebook-friendly API that loads
    everything from the ``.h5`` file and returns ``(fig, axes)``.
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
                    c="tab:green", alpha=0.9, lw=1.5,
                    label=f"perturbed ({n_equils})" if i == 0 else None,
                    zorder=2,
                )

    ax.legend(loc="best", fontsize=8)


def draw_jphi_profiles(axes, psi_N, j_phi, sigma_jphi,
                       perturbed_data_list=None):
    r"""Draw :math:`j_\phi` on 3 vertically stacked axes.

    ``axes[0]`` = total :math:`j_\phi`,
    ``axes[1]`` = :math:`j_{\rm BS}` (with :math:`j_{\rm BS,edge}` dashed),
    ``axes[2]`` = :math:`j_{\rm inductive}`.

    Parameters
    ----------
    axes : array-like of 3 Axes
    psi_N : 1-D array
    j_phi : 1-D array
        Baseline total :math:`j_\phi` [A m^-2].
    sigma_jphi : 1-D array
    perturbed_data_list : list[dict] or None
        Each dict must have ``'j_phi [A m^-2]'``, ``'j_BS [A m^-2]'``,
        ``'j_inductive [A m^-2]'``.  ``'j_BS,edge [A m^-2]'`` is
        optional; when present it is overplotted as a dashed line on
        the :math:`j_{\rm BS}` panel.
    """
    _MA = 1e-6  # A → MA

    # ---- panel 0: total j_phi with sigma bands ---------------------------
    ax0 = axes[0]
    ax0.cla()
    ax0.plot(psi_N, j_phi * _MA, c="k", lw=2,
             label=r"input $j_\phi$", zorder=4)
    ax0.fill_between(
        psi_N,
        (j_phi - sigma_jphi) * _MA,
        (j_phi + sigma_jphi) * _MA,
        alpha=0.25, color="tab:purple",
        label=r"$\pm\,1\sigma_{\rm exp}$", zorder=1,
    )
    ax0.plot(psi_N, (j_phi + 2 * sigma_jphi) * _MA, c="k", ls=":", lw=1.5,
             alpha=0.5, label=r"$\pm\,2\sigma_{\rm exp}$", zorder=2)
    ax0.plot(psi_N, (j_phi - 2 * sigma_jphi) * _MA, c="k", ls=":", lw=1.5,
             alpha=0.5, zorder=2)
    ax0.set_ylabel(r"$j_\phi$ [MA/m$^2$]")
    ax0.grid(ls=":")

    # ---- panels 1, 2: j_BS and j_inductive (no baseline sigma bands) -----
    _sub = [
        (axes[1], "j_BS [A m^-2]",        r"$j_{\rm BS}$",        "tab:green"),
        (axes[2], "j_inductive [A m^-2]",  r"$j_{\rm inductive}$", "tab:orange"),
    ]
    for ax, key, label, color in _sub:
        ax.cla()
        ax.set_ylabel(f"{label} " + r"[MA/m$^2$]")
        ax.grid(ls=":")

    # ---- overlay perturbed draws on all three panels ---------------------
    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            lbl = f"perturbed ({n_equils})" if i == 0 else None
            ax0.plot(psi_N, data["j_phi [A m^-2]"] * _MA, c="tab:purple",
                     lw=1.5, alpha=0.9, label=lbl, zorder=3)
            for ax, key, label, color in _sub:
                if key in data:
                    ax.plot(psi_N, data[key] * _MA, c=color, lw=1.5,
                            alpha=0.9, label=lbl, zorder=3)

            # overlay j_BS,edge as dashed on the j_BS panel
            if "j_BS,edge [A m^-2]" in data:
                axes[1].plot(
                    psi_N, data["j_BS,edge [A m^-2]"] * _MA,
                    c="tab:red", ls="--", lw=1.5, alpha=0.8,
                    label=r"$j_{\rm BS,edge}$" if i == 0 else None,
                    zorder=4,
                )

    for ax in axes:
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(r"$\hat{\psi}$")


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
        fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        draw_jphi_profiles(
            ax, psi_N,
            bl["j_phi [A m^-2]"], bl["sigma_jphi [A m^-2]"],
            perturbed_data_list=perturbed,
        )
        fig.tight_layout()
        figs.append(fig)
        axes_list.append(ax)

    if mode == "all":
        return figs, axes_list
    return figs[0], axes_list[0]


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
