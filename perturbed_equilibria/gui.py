"""
Interactive matplotlib GUI for browsing perturbed equilibrium families.

Launch from the terminal::

    plot-family equilibrium_output.h5

or via module invocation::

    python -m perturbed_equilibria equilibrium_output.h5
    python -m perturbed_equilibria.gui equilibrium_output.h5
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
# Switch to an interactive backend before pyplot is first imported.
# This only takes effect when gui.py itself is the entry point (CLI).
# In a Jupyter/IPython session the kernel would already have imported
# pyplot before this module is loaded, so the use() call is a no-op there.
_NON_INTERACTIVE = {"agg", "pdf", "ps", "svg", "pgf", "cairo"}
if matplotlib.get_backend().lower() in _NON_INTERACTIVE:
    for _backend in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg", "macosx"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from .utils import (
    discover_scan_values,
    load_baseline_profiles,
    count_equilibria,
)
from .plotting import (
    draw_kinetic_profiles,
    draw_pressure_profiles,
    draw_jphi_profiles,
    _load_all_perturbations,
)


class EquilibriumBrowser:
    """Interactive matplotlib window for browsing perturbed equilibria.

    Parameters
    ----------
    h5path : str
        Path to the HDF5 equilibrium database.
    """

    TABS = ("Kinetic", "Pressure", r"$j_\phi$")

    def __init__(self, h5path):
        self.h5path = os.path.abspath(h5path)
        self.scan_values = discover_scan_values(self.h5path)
        self.has_scan = self.scan_values is not None and len(self.scan_values) > 0

        # state
        self._current_tab = self.TABS[0]
        self._current_scan_idx = 0

        self._build_figure()
        self._update_plot()

    # ------------------------------------------------------------------
    #  Current scan value
    # ------------------------------------------------------------------
    def _current_scan_value(self):
        if not self.has_scan:
            return None
        return self.scan_values[self._current_scan_idx]

    # ------------------------------------------------------------------
    #  Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self):
        bottom = 0.12 if self.has_scan else 0.08

        self.fig = plt.figure(figsize=(10, 7.5))
        self.fig.canvas.manager.set_window_title(
            f"Perturbed Equilibria \u2014 {os.path.basename(self.h5path)}"
        )

        # ---- tab radio buttons (left sidebar) ----------------------------
        radio_ax = self.fig.add_axes([0.01, 0.40, 0.10, 0.20])
        self.radio = RadioButtons(radio_ax, self.TABS, activecolor="tab:blue")
        self.radio.on_clicked(self._on_tab_change)

        # ---- shared plot region ------------------------------------------
        left  = 0.18
        right = 0.96
        top   = 0.95
        bot   = bottom + 0.05
        w = right - left
        h = top - bot

        # Kinetic: 2x2
        gap = 0.06
        cw = (w - gap) / 2
        ch = (h - gap) / 2
        self.kinetic_axes = np.array([
            [self.fig.add_axes([left,            bot + ch + gap, cw, ch]),
             self.fig.add_axes([left + cw + gap, bot + ch + gap, cw, ch])],
            [self.fig.add_axes([left,            bot,            cw, ch]),
             self.fig.add_axes([left + cw + gap, bot,            cw, ch])],
        ])

        # Pressure: single axes
        self.pressure_ax = self.fig.add_axes([left, bot, w, h])

        # j_phi: 3 vertically stacked
        jgap = 0.04
        jh = (h - 2 * jgap) / 3
        self.jphi_axes = [
            self.fig.add_axes([left, bot + 2 * (jh + jgap), w, jh]),
            self.fig.add_axes([left, bot + (jh + jgap),     w, jh]),
            self.fig.add_axes([left, bot,                   w, jh]),
        ]

        # ---- scan slider (only for hierarchical files) -------------------
        self.slider = None
        self._scan_label = None
        if self.has_scan and len(self.scan_values) > 1:
            slider_ax = self.fig.add_axes([left, 0.02, 0.55, 0.03])
            self.slider = Slider(
                slider_ax, "Scan",
                valmin=0,
                valmax=len(self.scan_values) - 1,
                valinit=0,
                valstep=1,
                valfmt="%d",
            )
            self.slider.on_changed(self._on_slider_change)
            self._scan_label = self.fig.text(
                left + 0.58, 0.025,
                f"= {self.scan_values[0]}",
                fontsize=10, va="center",
            )

        # initially show kinetic tab
        self._set_tab_visibility(self.TABS[0])

    # ------------------------------------------------------------------
    #  Tab visibility
    # ------------------------------------------------------------------
    def _set_tab_visibility(self, tab):
        for ax in self.kinetic_axes.flat:
            ax.set_visible(tab == self.TABS[0])
        self.pressure_ax.set_visible(tab == self.TABS[1])
        for ax in self.jphi_axes:
            ax.set_visible(tab == self.TABS[2])

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------
    def _on_tab_change(self, label):
        self._current_tab = label
        self._set_tab_visibility(label)
        self._update_plot()
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        self._current_scan_idx = int(val)
        if self._scan_label is not None:
            self._scan_label.set_text(
                f"= {self.scan_values[self._current_scan_idx]}"
            )
        self._update_plot()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    #  Redraw the active tab
    # ------------------------------------------------------------------
    def _update_plot(self):
        sv = self._current_scan_value()
        bl = load_baseline_profiles(self.h5path, scan_value=sv)
        perturbed = _load_all_perturbations(self.h5path, scan_value=sv)
        psi_N = bl["psi_N"]

        tab = self._current_tab

        if tab == self.TABS[0]:  # Kinetic
            draw_kinetic_profiles(
                self.kinetic_axes, psi_N,
                bl["n_e [m^-3]"],  bl["n_i [m^-3]"],
                bl["T_e [eV]"],    bl["T_i [eV]"],
                bl["sigma_ne [m^-3]"], bl["sigma_ni [m^-3]"],
                bl["sigma_te [eV]"],   bl["sigma_ti [eV]"],
                perturbed_data_list=perturbed,
            )

        elif tab == self.TABS[1]:  # Pressure
            draw_pressure_profiles(
                self.pressure_ax, psi_N,
                bl["pressure [Pa]"],
                perturbed_data_list=perturbed,
            )

        elif tab == self.TABS[2]:  # j_phi
            draw_jphi_profiles(
                self.jphi_axes, psi_N,
                bl["j_phi [A m^-2]"], bl["sigma_jphi [A m^-2]"],
                perturbed_data_list=perturbed,
            )

    # ------------------------------------------------------------------
    #  Show
    # ------------------------------------------------------------------
    def show(self):
        """Enter the matplotlib event loop."""
        plt.show(block=True)


# ======================================================================
#  CLI entry point
# ======================================================================
def _validate_h5(path):
    """Check that *path* is a readable HDF5 equilibrium database."""
    import h5py

    if not os.path.isfile(path):
        raise SystemExit(f"Error: file not found: {path}")

    try:
        with h5py.File(path, "r") as hf:
            # Flat layout must have _baseline; hierarchical must have scan/
            if "_baseline" not in hf and "scan" not in hf:
                raise SystemExit(
                    f"Error: '{path}' does not look like a perturbed-equilibria "
                    f"HDF5 database (no '_baseline' or 'scan/' group found)."
                )
    except Exception as exc:
        if isinstance(exc, SystemExit):
            raise
        raise SystemExit(f"Error: cannot open '{path}' as HDF5: {exc}")


def main():
    """``plot-family`` -- browse perturbed equilibrium families interactively."""
    parser = argparse.ArgumentParser(
        prog="plot-family",
        description=(
            "Interactive matplotlib browser for perturbed equilibrium families "
            "stored in an HDF5 database."
        ),
    )
    parser.add_argument(
        "h5file",
        help="Path to the HDF5 equilibrium database (.h5).",
    )
    args = parser.parse_args()

    _validate_h5(args.h5file)

    browser = EquilibriumBrowser(args.h5file)
    browser.show()


if __name__ == "__main__":
    main()
