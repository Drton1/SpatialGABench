"""Public wrapper namespace.

This facade keeps the public package layout stable while the project-specific
software wrappers remain under :mod:`spatialgabench.software_wrappers`.
"""

from spatialgabench.software_wrappers.common import peak_memory_mb

__all__ = ["peak_memory_mb"]
