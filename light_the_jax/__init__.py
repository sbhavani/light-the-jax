"""JAX distribution installer with computation backend auto-detection."""

from ._patch import patch

__all__ = ["patch"]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "UNKNOWN"
