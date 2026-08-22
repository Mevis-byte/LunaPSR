"""Chandrayaan-2 OHRC product ingestion.

Public API::

    from core.ohrc import OHRCProduct, OHRCError

    product = OHRCProduct.load("ch2_ohr_ncp_20260103T1005176450_d_img_d18")
    product.image          # OHRCImageData (scientific array, dims, dtype, units)
    product.metadata       # parsed PDS4 metadata
    product.geometry       # GeometryGrid (lat/lon tie-points) or None
    product.product_info   # UI summary dict

All loading is read-only.  The original product files are never modified.
"""

from core.ohrc.discovery import (
    DiscoveredProduct,
    DiscoveryScan,
    find_ohrc_products,
    select_product,
    validate_product,
)
from core.ohrc.exceptions import (
    CorruptProductError,
    DimensionMismatchError,
    MetadataNotFoundError,
    OHRCError,
    ProductDiscoveryError,
    ScientificImageNotFoundError,
    UnsupportedDataTypeError,
    UnsupportedProductError,
)
from core.ohrc.product import OHRCImageData, OHRCProduct

__all__ = [
    "OHRCProduct",
    "OHRCImageData",
    "OHRCError",
    "ProductDiscoveryError",
    "MetadataNotFoundError",
    "ScientificImageNotFoundError",
    "UnsupportedDataTypeError",
    "DimensionMismatchError",
    "CorruptProductError",
    "UnsupportedProductError",
    "find_ohrc_products",
    "validate_product",
    "select_product",
    "DiscoveryScan",
    "DiscoveredProduct",
]