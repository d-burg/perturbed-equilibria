# perturbed-equilibria
Perturbed equilibria. Plug-in that interfaces with OpenFUSIONToolkit/TokaMaker

## Usage

`perturbed_equilibria` is designed to be imported after OFT is already
set up in your script or notebook:

```python
# --- Standard TokaMaker setup (already in your notebook) ---
import os, sys
tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, 'python'))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
# ... rest of your OFT imports

# --- Then add perturbed_equilibria ---
from perturbed_equilibria import uncertainties, sampling, plotting
