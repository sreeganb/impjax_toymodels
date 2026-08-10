"""impjax_toymodels public package API."""

from ._version import get_version
from .interface import InterfaceSpec, SamplingVariable
from .timing import TimingSnapshot, elapsed_timing, start_timing

__all__ = [
    "InterfaceSpec",
    "SamplingVariable",
    "TimingSnapshot",
    "elapsed_timing",
    "start_timing",
    "__version__",
]

__version__ = get_version()
