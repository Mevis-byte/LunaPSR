"""Error types for the OHRC product reader.

All reader failures raise a subclass of :class:`OHRCError` so that the UI can
present a useful message and log technical detail separately.  No raw
traceback is intended for normal users.
"""

from __future__ import annotations


class OHRCError(Exception):
    """Base class for every OHRC ingestion error."""


class ProductDiscoveryError(OHRCError):
    """The product directory could not be found or does not look like a product."""


class MetadataNotFoundError(OHRCError):
    """No parseable PDS4 metadata label was found for the scientific image."""


class ScientificImageNotFoundError(OHRCError):
    """The scientific image file (`.img`) could not be located."""


class UnsupportedDataTypeError(OHRCError):
    """The datatype declared in the label is not supported."""


class DimensionMismatchError(OHRCError):
    """The declared array dimensions do not match the file contents."""


class CorruptProductError(OHRCError):
    """The product fails an integrity check (size, offset, or parse)."""


class UnsupportedProductError(OHRCError):
    """The product structure is not one LunaPSR can ingest."""