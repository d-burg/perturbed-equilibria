import matplotlib.pyplot as plt
from .utils import load_equilibrium

# plot kinetic profiles

def plot_kinetic_profiles(header, n_equils, psi_N, ne, ni, te, ti, sigma_ne, sigma_ni, sigma_te, sigma_ti):

    fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True)

    _pairs = [
        #  axis       orig  scale  σ_phys     color        label          ylabel
        (ax[0, 0], ne, 1.0,  sigma_ne, "tab:red",    r"$n_e$", r"n [m$^{-3}$]"),
        (ax[0, 1], ni, 1.0,  sigma_ni, "tab:orange", r"$n_i$", None),
        (ax[1, 0], te, 1e-3, sigma_te, "tab:blue",   r"$T_e$", r"T [keV]"),
        (ax[1, 1], ti, 1e-3, sigma_ti, "tab:cyan",   r"$T_i$", None),
    ]

    # ---- draw input profiles and ±1σ bands once ------------------------
    for a, orig, scale, sig, clr, lbl, ylabel in _pairs:
        a.plot(psi_N, orig * scale, c="k", lw=2, label=f"input {lbl}", zorder=3)
        a.fill_between(
            psi_N,
            (orig - sig) * scale,
            (orig + sig) * scale,
            alpha=0.25, color=clr,
            label=r"$\pm\,1\sigma_{\rm exp}$",
            zorder=1,
        )
        a.plot(psi_N, (orig + 2 * sig) * scale, c="k", ls=":", lw=1.5, alpha=0.5,
            label=r"$\pm\,2\sigma_{\rm exp}$" , zorder=2)#if a is ax[0, 0] else None, zorder=2)
        a.plot(psi_N, (orig - 2 * sig) * scale, c="k", ls=":", lw=1.5, alpha=0.5,
            zorder=2)
        a.grid(ls=":")
        if ylabel:
            a.set_ylabel(ylabel)

    # ---- overlay perturbed profiles from each equilibrium ---------------
    _keys = ["n_e [m^-3]", "n_i [m^-3]", "T_e [eV]", "T_i [eV]"]

    for i in range(n_equils):
        data = load_equilibrium(header, count=i)

        for (a, orig, scale, sig, clr, lbl, ylabel), key in zip(_pairs, _keys):
            a.plot(
                psi_N, data[key] * scale,
                c=clr, alpha=0.9, lw=1.5,# ls="--", 
                # Only label the first draw so the legend stays clean
                label=f"perturbed ({n_equils})" if i == 0 else None,
                zorder=2,
            )

    # ---- finalise -------------------------------------------------------
    for a, *_ in _pairs:
        a.legend(loc="best", fontsize=8)

    ax[1, 0].set_xlabel(r"$\hat{\psi}$")
    ax[1, 1].set_xlabel(r"$\hat{\psi}$")

    plt.tight_layout()
    plt.show()

def plot_jphi_profiles(psi_N, input_j_phi, sigma_jphi, header, n_equils):
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # ---- input profile and uncertainty bands (drawn once) ---------------
    ax.plot(psi_N, input_j_phi, c="k", lw=2, label=r"input $j_\phi$", zorder=4)

    ax.fill_between(
        psi_N,
        input_j_phi - sigma_jphi,
        input_j_phi + sigma_jphi,
        alpha=0.25, color="tab:purple",
        label=r"$\pm\,1\sigma_{\rm exp}$",
        zorder=1,
    )

    ax.plot(psi_N, input_j_phi + 2 * sigma_jphi, c="k", ls=":", lw=1.5, alpha=0.5,
            label=r"$\pm\,2\sigma_{\rm exp}$", zorder=2)
    ax.plot(psi_N, input_j_phi - 2 * sigma_jphi, c="k", ls=":", lw=1.5, alpha=0.5,
            zorder=2)

    # ---- overlay perturbed j_phi from each equilibrium ------------------
    for i in range(n_equils):
        data = load_equilibrium(header, count=i)
        ax.plot(
            psi_N, data["j_phi [A/m^2]"],
            c="tab:purple", lw=1.5, alpha=0.9, #ls="--",
            label=f"perturbed ({n_equils})" if i == 0 else None,
            zorder=3,
        )

    # ---- σ profile on twin axis ----------------------------------------
    # ax2 = ax.twinx()
    # ax2.set_ylabel(r"$\sigma_{\rm exp}$ [A/m$^2$]", color="red")
    # ax2.plot(psi_N, sigma_jphi, color="red", ls="--", alpha=0.5)
    # ax2.tick_params(axis="y", labelcolor="red")
    # ax2.set_ylim(0.0, None)

    # ---- finalise -------------------------------------------------------
    ax.set_ylim(
        0.0,
        max(input_j_phi[0], (input_j_phi + 2 * sigma_jphi)[0]),
    )
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(r"$\hat{\psi}$")
    ax.set_ylabel(r"$j_\phi$ [A/m$^2$]")
    #ax.set_title(r"Perturbed $j_\phi$ profiles")
    ax.grid(ls=":")
    plt.tight_layout()
    plt.show()