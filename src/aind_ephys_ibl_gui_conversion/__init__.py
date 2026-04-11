"""Convert ephys data for use in IBL GUI"""

from importlib.metadata import PackageNotFoundError, version

from aind_ephys_ibl_gui_conversion.ephys import extract_continuous
from aind_ephys_ibl_gui_conversion.spikes import extract_spikes

try:
    __version__ = version("aind-ephys-ibl-gui-conversion")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["extract_continuous", "extract_spikes", "__version__"]
