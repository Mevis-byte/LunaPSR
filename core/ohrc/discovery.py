"""Automatic OHRC product discovery.

LunaPSR locates Chandrayaan-2 OHRC products directly from the project
directory (no upload required).  Discovery is *read-only*: it never copies,
moves or modifies a product.

A candidate is a directory that looks like an OHRC bundle (has a scientific
``.img`` plus a sibling ``.xml`` label).  Candidates are validated with the
Phase 2 reader before being presented:

* ``find_ohrc_products()`` scans ``PROJECT_ROOT`` (plus any explicitly given
  root) and returns one :class:`DiscoveredProduct` per candidate.
* A candidate whose science chain loads cleanly is marked ``VALID``;
  otherwise the reason is captured and shown in the UI, and the product is
  deliberately not loadable.
* ``select_product`` picks the single default automatically when exactly one
  valid product exists.

Everything here stays project-relative.  No Windows/WSL absolute path is
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from configs.defaults import PROJECT_ROOT
from core.ohrc.exceptions import OHRCError
from core.ohrc.reader import discover_product, parse_label_file

__all__ = [
    "DiscoveredProduct",
    "DiscoveryScan",
    "find_ohrc_products",
    "validate_product",
    "LOOKS_LIKE_OHRC_NAME",
]

# ISRO/ISDA name the science products ``<mission>_ohr_<name>_<level>_<fnum>``.
LOOKS_LIKE_OHRC_NAME = ("ohr", "ch2")


_SIGNATURE_SUBDIRS = {"data", "browse", "geometry"}


def _looks_like_product_dir(candidate: Path) -> bool:
    """True when the directory itself is an OHRC product bundle.

    A product directory (a) contains a scientific ``.img`` file directly
    beneath it, or (b) contains ``data`` / ``browse`` / ``geometry`` folders
    with an ``.img`` inside them (the real bundle layout).  A parent folder
    that merely *contains* a product must not match, because ``data`` and
    friends are common project folder names too.
    """
    try:
        children = list(candidate.iterdir())
    except OSError:
        return False
    for child in children:
        if child.is_file() and child.suffix.lower() == ".img":
            return True
    for child in children:
        if child.is_dir() and child.name in _SIGNATURE_SUBDIRS:
            try:
                for f in child.rglob("*.img"):
                    return True
            except OSError:
                continue
    return False


def _candidate_dirs(root: Path, max_depth: int = 3) -> List[Path]:
    """Collect plausible product directories under ``root``.

    Walks at most ``max_depth`` levels and stops descending once a directory
    has the OHRC science footprint, so a product living one folder deep (or in
    its own sub-directory) is identified correctly.  Products found directly in
    ``root`` are always preferred.  Still strictly project-relative.
    """
    root = Path(root)
    if not root.exists():
        return []
    found: List[Path] = []

    def visit(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        if _looks_like_product_dir(directory):
            found.append(directory)
            return
        for sub in sorted(directory.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                visit(sub, depth + 1)

    visit(root, 1)
    return found


@dataclass
class DiscoveredProduct:
    """One OHRC candidate and its validation verdict."""

    path: Path
    status: str = "UNKNOWN"          # VALID | INVALID | EMPTY | DETECTED
    error: Optional[str] = None
    product_name: str = ""
    lines: Optional[int] = None
    samples: Optional[int] = None
    datatype: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "VALID"

    @property
    def name(self) -> str:
        return self.product_name or self.path.name


@dataclass(init=False)
class DiscoveryScan:
    """Result of scanning one search root."""

    def __init__(self, root: Path, products: List[DiscoveredProduct]):
        self.root = Path(root)
        self.products = products

    @property
    def valid_products(self) -> List[DiscoveredProduct]:
        return [p for p in self.products if p.valid]

    def default(self) -> Optional[DiscoveredProduct]:
        """The automatic default: the single VALID product, if unambigous."""
        valids = self.valid_products
        if len(valids) == 1:
            return valids[0]
        return None


def validate_product(path: Path,
                     validator: Optional[Callable[[Path], object]] = None) -> DiscoveredProduct:
    """Validate a candidate with the Phase 2 reader.

    Lightweight validation parses the scientific label (no full memory-map);
    this is fast enough to run for every candidate during a scan.  The
    optional ``validator`` callable (default ``OHRCProduct.load``) performs a
    full reader-level load and can be used to double-check interactive use.
    """
    path = Path(path)
    entry = DiscoveredProduct(path=path, status="DETECTED", product_name=path.name)

    try:
        layout = discover_product(path)
    except OHRCError as exc:
        entry.status = "INVALID"
        entry.error = str(exc)
        return entry

    if layout.scientific_label is None:
        entry.status = "INVALID"
        entry.error = "Scientific image has no XML label."
        return entry

    try:
        metadata = parse_label_file(layout.scientific_label)
    except OHRCError as exc:
        entry.status = "INVALID"
        entry.error = str(exc)
        return entry

    entry.lines = metadata.array.lines
    entry.samples = metadata.array.samples
    entry.datatype = (
        metadata.array.data_type.pds_type if metadata.array.data_type else None
    )
    entry.product_name = metadata.product_name or path.name
    entry.extra = {
        "scientific_image": str(layout.scientific_image.name) if layout.scientific_image else None,
        "browse": layout.browse_image is not None,
        "geometry": layout.geometry_grid is not None,
        "complete_science_chain": layout.complete_science_chain,
    }

    if validator is not None:
        try:
            validator(path)
        except OHRCError as exc:
            entry.status = "INVALID"
            entry.error = str(exc)
            return entry

    entry.status = "VALID"
    return entry


def find_ohrc_products(root: Optional[Path] = None,
                       validator: Optional[Callable[[Path], object]] = None) -> DiscoveryScan:
    """Discover OHRC products under ``root`` (default: the project root)."""
    search_root = Path(root) if root is not None else PROJECT_ROOT
    entries = [validate_product(c, validator=validator)
               for c in _candidate_dirs(search_root)]
    return DiscoveryScan(search_root, entries)


def select_product(scan: DiscoveryScan,
                   name_or_path: Optional[str] = None) -> Optional[DiscoveredProduct]:
    """Return a chosen product.

    ``name_or_path`` may match the product name or resolve to a path (useful
    for manually added products).  Falls back to the automatic single-product
    default.
    """
    if name_or_path:
        query = str(name_or_path)
        for p in scan.products:
            if p.product_name == query or p.path == Path(query) or str(p.path) == query:
                return p
    return scan.default()