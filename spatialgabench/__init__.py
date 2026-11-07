"""SpatialGABench public package facade.

SpatialGABench provides reusable utilities and command-line entry points for
spatial ATAC gene activity construction benchmarks. The package intentionally
keeps data download and full benchmark execution outside import-time behavior.
"""

from ._version import __version__

__all__ = ["__version__"]
