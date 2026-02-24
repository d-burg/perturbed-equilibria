# GUI Display Guide

`perturbed-equilibria` provides two ways to visualize equilibrium families:

| Method | Best for | Requires display? |
|---|---|---|
| `plot_family()` | Jupyter notebooks | No (inline rendering) |
| `plot-family` CLI | Terminal / desktop use | Yes |

This guide covers setup for the `plot-family` CLI across common HPC and
remote-access scenarios.

---

## Quick reference

| Environment | Command |
|---|---|
| Local desktop (Linux/Mac) | `plot-family file.h5` |
| SSH with X11 forwarding | `MPLBACKEND=TkAgg plot-family file.h5` |
| noVNC / VNC remote desktop | `plot-family file.h5` |
| Jupyter notebook | `plot_family("file.h5", mode="all")` |

---

## 1. SSH with X11 forwarding (recommended for HPC clusters)

Most HPC clusters and shared Linux workstations are accessed over SSH.
To display matplotlib windows on your local screen, use **X11 forwarding**
with the **TkAgg** backend.

### One-time setup (local machine)

**macOS** -- install [XQuartz](https://www.xquartz.org/):

```bash
brew install --cask xquartz
```

Then **log out and back in** (or reboot).  XQuartz must register itself as
the X11 display server before forwarding will work.

**Linux** -- X11 is usually available out of the box.  No extra setup needed.

**Windows** -- install an X server such as
[VcXsrv](https://sourceforge.net/projects/vcxsrv/) or
[MobaXterm](https://mobaxterm.mobatek.net/) (has a built-in X server).

### Connecting

```bash
ssh -X user@cluster          # standard forwarding
# or
ssh -Y user@cluster          # trusted forwarding (use if -X gives auth errors)
```

Verify the display is available:

```bash
echo $DISPLAY
# Should print something like  localhost:10.0
```

If `$DISPLAY` is empty, X11 forwarding did not activate.  Check that the
remote machine has `X11Forwarding yes` in `/etc/ssh/sshd_config`.

### Setting the matplotlib backend

Over X11, the **TkAgg** backend is strongly recommended.  It is lightweight,
does not require OpenGL, and renders faster than GTK or Qt over the network.

```bash
# One-off
MPLBACKEND=TkAgg plot-family equilibria.h5

# Permanent (add to ~/.bashrc on the remote machine)
echo 'export MPLBACKEND=TkAgg' >> ~/.bashrc
source ~/.bashrc
```

> **Why not the default backend?**  When `$DISPLAY` is set, matplotlib may
> auto-select GTK3Agg or Qt5Agg.  These toolkits attempt OpenGL-accelerated
> rendering, which typically fails over X11 forwarding with errors like:
>
> ```
> libGL error: No matching fbConfigs or visuals found
> libGL error: failed to load driver: swrast
> Gsk-Message: Failed to realize renderer of type 'GskGLRenderer'
> ```
>
> TkAgg avoids this entirely because it uses only 2D X11 drawing primitives.

### Verifying tkinter is available

TkAgg requires the `tkinter` Python module.  Test with:

```bash
python3 -c "import tkinter; print('tkinter OK')"
```

If this fails, install it:

```bash
# Debian / Ubuntu
sudo apt install python3-tk

# RHEL / CentOS / Fedora
sudo dnf install python3-tkinter

# Conda
conda install tk
```

---

## 2. noVNC / VNC remote desktop

If your cluster provides a browser-based remote desktop (noVNC, TurboVNC,
etc.), `plot-family` should work directly since a display server is already
running.

```bash
plot-family equilibria.h5
```

### Missing window decorations

If the plot window appears but has **no title bar, close button, or drag
handles**, the VNC session is running without a window manager.  This is
common on minimal HPC desktop provisions.

**Workaround (no install required):**  The `plot-family` GUI includes a
built-in **Close** button in the top-right corner of the figure.  Click it
to close the window cleanly.

**Proper fix (requires admin or sudo):**  Install a lightweight window
manager:

```bash
# Any one of these will work:
sudo apt install openbox      # very minimal
sudo apt install fluxbox      # slightly more featured
sudo apt install xfwm4        # XFCE's window manager
```

Then start it in the VNC session:

```bash
openbox &      # run in background
```

---

## 3. Local desktop (Linux / macOS)

On a machine where you are physically sitting at the screen, `plot-family`
should work out of the box:

```bash
plot-family equilibria.h5
```

matplotlib will auto-select an appropriate backend (usually TkAgg, Qt5Agg,
or the native macOS backend).

---

## 4. Jupyter notebooks

In a Jupyter notebook, use the Python API directly instead of the CLI:

```python
from perturbed_equilibria import plot_family

# Single plot type
fig, axes = plot_family("equilibria.h5", scan_value=1.0, mode="kinetic")

# All plot types at once (returns lists)
figs, axes_list = plot_family("equilibria.h5", scan_value=1.0, mode="all")
```

Available modes: `"kinetic"`, `"pressure"`, `"j-phi"`, `"all"`.

No backend configuration is needed -- the Jupyter kernel handles rendering
automatically.

---

## Troubleshooting

### No window appears and the command exits immediately

The matplotlib backend cannot open a display.  Check:

1. Is `$DISPLAY` set?  (`echo $DISPLAY`)
2. Are you over SSH without `-X`?  Reconnect with `ssh -X`.
3. Is an X server running on your local machine?

Diagnostic:

```bash
python3 -c "import matplotlib; print(matplotlib.get_backend())"
```

If this prints `agg`, no interactive backend is available.

### Window appears but is blank or garbled

Try forcing TkAgg:

```bash
MPLBACKEND=TkAgg plot-family file.h5
```

### "Unable to import Axes3D" warning

```
UserWarning: Unable to import Axes3D. This may be due to multiple versions
of Matplotlib being installed ...
```

This is harmless.  `perturbed-equilibria` uses only 2D plots.  The warning
comes from conflicting matplotlib installations (e.g. system package vs pip).
You can suppress it or unify your matplotlib install, but it does not affect
functionality.

### OpenGL errors over X11

```
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
```

Set `MPLBACKEND=TkAgg` (see [SSH section](#1-ssh-with-x11-forwarding-recommended-for-hpc-clusters) above).

If you must use GTK, you can also try:

```bash
GSK_RENDERER=cairo plot-family file.h5
```

This tells GTK to use its 2D Cairo renderer instead of OpenGL.
