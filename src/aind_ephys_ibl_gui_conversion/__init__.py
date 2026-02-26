"""Convert ephys data for use in IBL GUI"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aind-ephys-ibl-gui-conversion")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
