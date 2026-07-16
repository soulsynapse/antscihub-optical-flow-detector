"""Canonical import surface for the structure-tensor explorer.

The implementation began under the narrower ``variance_explorer`` name.  Keep
that module importable for existing local launch commands while exposing the UI
under the name that matches the structure-tensor proof of concept.
"""

from gui.variance_explorer import StructureTensorExplorer

__all__ = ["StructureTensorExplorer"]
