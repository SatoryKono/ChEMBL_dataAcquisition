"""Utilities for mapping ChEMBL identifiers to UniProt IDs.

This package exposes the :func:`map_chembl_to_uniprot` function that reads
an input CSV file, maps values from the ChEMBL database to UniProt IDs using
UniProt's ID mapping API and writes the result to a new CSV file.
"""

from .mapping import map_chembl_to_uniprot

__all__ = ["map_chembl_to_uniprot"]
