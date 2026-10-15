"""Resolve editable and frozen application assets without machine-specific paths."""
from pathlib import Path


def application_root(entry, executable=None, frozen=False):
    return Path(executable if frozen else entry).resolve().parents[1]
