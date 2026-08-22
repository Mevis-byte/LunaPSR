"""PDS4 metadata label parser for Chandrayaan-2 OHRC products.

Parses the ISRO/ISDA ``Product_Observational`` XML label (the authoritative
description of a scientific image), without assuming a schema version.  The
parser is namespace-agnostic and only reads what is actually present;
anything that is missing is represented as ``None`` or an explicit unknown
state.  No values are fabricated.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.ohrc.exceptions import MetadataNotFoundError, UnsupportedDataTypeError

# PDS4 data_type keyword -> (numpy dtype, signed, byte order, bit depth)
# Byte order applies only to multi-byte elements.
_DATA_TYPE_MAP = {
    "SignedByte": ("i1", True, None, 8),
    "UnsignedByte": ("u1", False, None, 8),
    "SignedMSB2": (">i2", True, "big", 16),
    "SignedLSB2": ("<i2", True, "little", 16),
    "UnsignedMSB2": (">u2", False, "big", 16),
    "UnsignedLSB2": ("<u2", False, "little", 16),
    "SignedMSB4": (">i4", True, "big", 32),
    "SignedLSB4": ("<i4", True, "little", 32),
    "UnsignedMSB4": (">u4", False, "big", 32),
    "UnsignedLSB4": ("<u4", False, "little", 32),
    "SignedMSB8": (">i8", True, "big", 64),
    "SignedLSB8": ("<i8", True, "little", 64),
    "UnsignedMSB8": (">u8", False, "big", 64),
    "UnsignedLSB8": ("<u8", False, "little", 64),
    "IEEE754MSBSingle": (">f4", False, "big", 32),
    "IEEE754LSBSingle": ("<f4", False, "little", 32),
    "IEEE754MSBDouble": (">f8", False, "big", 64),
    "IEEE754LSBDouble": ("<f8", False, "little", 64),
}


@dataclass
class DataTypeInfo:
    """Interpreted elemental data type from the label."""

    pds_type: str
    numpy_dtype: str
    signed: bool
    byte_order: Optional[str]
    bit_depth: int

    @property
    def is_supported(self) -> bool:
        return self.pds_type in _DATA_TYPE_MAP

    def describe(self) -> str:
        return (
            f"{self.pds_type} ({self.bit_depth}-bit "
            f"{'signed' if self.signed else 'unsigned'}"
            f"{'' if self.byte_order is None else ', ' + self.byte_order + '-endian'})"
        )


@dataclass
class ArrayInfo:
    """Structural information about the stored binary array."""

    lines: Optional[int]
    samples: Optional[int]
    offset_bytes: int
    axes_order: Optional[str]
    data_type: Optional[DataTypeInfo]
    special_constants: Dict[str, float] = field(default_factory=dict)

    @property
    def nodata_values(self) -> tuple:
        """Values explicitly declared as missing/invalid by the label.

        PDS4 labels declare these under ``Special_Constants`` (keys such as
        ``missing_constant`` / ``invalid_constant`` / ``saturated_constant``).
        An empty tuple means the label declares **no** nodata value -- we must
        never guess one.
        """
        if not self.special_constants:
            return ()
        return tuple(float(v) for v in self.special_constants.values())

    @property
    def shape(self) -> Optional[tuple]:
        if self.lines is None or self.samples is None:
            return None
        return (self.lines, self.samples)

    @property
    def total_pixels(self) -> Optional[int]:
        if self.shape is None:
            return None
        return int(self.lines * self.samples)

    def expected_bytes(self) -> Optional[int]:
        shape = self.shape
        if shape is None or self.data_type is None:
            return None
        import numpy as np

        return int(shape[0] * shape[1] * np.dtype(self.data_type.numpy_dtype).itemsize)


@dataclass
class AcquisitionInfo:
    """Observation and timing context."""

    start_date_time: Optional[str] = None
    stop_date_time: Optional[str] = None
    imaging_orbit_number: Optional[int] = None
    dumping_orbit_number: Optional[int] = None
    spacecraft_altitude_km: Optional[float] = None
    target: Optional[str] = None
    area: Optional[str] = None
    line_exposure_ms: Optional[float] = None
    tdi_stages: Optional[str] = None
    bits_selection: Optional[str] = None
    reference_data_used: Optional[str] = None
    orbit_limb_direction: Optional[str] = None
    spacecraft_yaw_direction: Optional[str] = None


@dataclass
class CameraInfo:
    """OHRC instrument parameters."""

    name: Optional[str] = None
    description: Optional[str] = None
    band: Optional[str] = None
    focal_length_mm: Optional[float] = None
    detector_pixel_width_um: Optional[float] = None
    pixel_resolution_mp: Optional[float] = None


@dataclass
class CalibrationInfo:
    """What the 'Calibrated' level means for this product."""

    processing_level: Optional[str] = None
    level_description: Optional[str] = None
    units: Optional[str] = None
    units_comment: Optional[str] = None
    coefficients: Optional[Dict[str, float]] = None
    calibration_note: Optional[str] = None


@dataclass
class ObservationGeometry:
    """Pointing and illumination geometry declared in the label."""

    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    solar_incidence_deg: Optional[float] = None
    projection: Optional[str] = None
    corners: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)


@dataclass
class SourceFileInfo:
    """The physical file the label describes."""

    file_name: Optional[str] = None
    file_size: Optional[int] = None
    md5_checksum: Optional[str] = None
    creation_date_time: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class OHRCMetadata:
    """Everything parsed from a scientific-image PDS4 label."""

    product_name: Optional[str]
    logical_identifier: Optional[str]
    product_class: Optional[str]
    processing_level: Optional[str]
    acquisition: AcquisitionInfo
    camera: CameraInfo
    calibration: CalibrationInfo
    array: ArrayInfo
    source_file: SourceFileInfo
    geometry: ObservationGeometry
    label: Dict[str, object]

    def to_dict(self) -> dict:
        import dataclasses

        def _asdict(obj):
            return dataclasses.asdict(obj)

        return {
            "product_name": self.product_name,
            "logical_identifier": self.logical_identifier,
            "product_class": self.product_class,
            "processing_level": self.processing_level,
            "acquisition": _asdict(self.acquisition),
            "camera": _asdict(self.camera),
            "calibration": _asdict(self.calibration),
            "array": _asdict(self.array),
            "source_file": _asdict(self.source_file),
            "geometry": _asdict(self.geometry),
        }


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(elem: ET.Element) -> List[ET.Element]:
    return list(elem)


def _find(elem: ET.Element, tag: str) -> Optional[ET.Element]:
    for child in elem:
        if _local(child.tag) == tag:
            return child
    return None


def _find_all(elem: ET.Element, tag: str) -> List[ET.Element]:
    return [c for c in elem if _local(c.tag) == tag]


def _find_deep(elem: ET.Element, *path: str) -> Optional[ET.Element]:
    node: Optional[ET.Element] = elem
    for tag in path:
        if node is None:
            return None
        node = _find(node, tag)
    return node


def _text(elem: ET.Element, *path: str) -> Optional[str]:
    node = _find_deep(elem, *path)
    if node is None or node.text is None:
        return None
    value = " ".join(node.text.split())
    return value or None


def _float(elem: ET.Element, *path: str) -> Optional[float]:
    value = _text(elem, *path)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(elem: ET.Element, *path: str) -> Optional[int]:
    value = _text(elem, *path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def interpret_data_type(pds_type: str) -> DataTypeInfo:
    """Map a PDS4 ``data_type`` keyword to numpy + byte order information."""
    info = _DATA_TYPE_MAP.get(pds_type)
    if info is None:
        raise UnsupportedDataTypeError(
            f"Unsupported datatype: '{pds_type}'. Supported: "
            + ", ".join(sorted(_DATA_TYPE_MAP))
        )
    numpy_dtype, signed, byte_order, bit_depth = info
    return DataTypeInfo(
        pds_type=pds_type,
        numpy_dtype=numpy_dtype,
        signed=signed,
        byte_order=byte_order,
        bit_depth=bit_depth,
    )


# ---------------------------------------------------------------------------
# Label parser
# ---------------------------------------------------------------------------


def parse_scientific_label(xml_bytes: bytes) -> OHRCMetadata:
    """Parse a PDS4 label for an OHRC scientific image.

    Raises ``MetadataNotFoundError`` when the label cannot be parsed, and
    ``UnsupportedDataTypeError`` when the declared datatype is unsupported.
    """
    if not xml_bytes:
        raise MetadataNotFoundError("OHRC metadata file not found (empty label).")

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise MetadataNotFoundError(
            "OHRC metadata file could not be parsed (not a valid PDS4 XML label)."
        ) from exc

    ident = _find(root, "Identification_Area")
    observation = _find(root, "Observation_Area")
    file_area = _find(root, "File_Area_Observational")

    product_name = None
    logical_id = None
    product_class = None
    if ident is not None:
        logical_id = _text(ident, "logical_identifier")
        product_class = _text(ident, "product_class")
        if logical_id:
            product_name = logical_id.rsplit(":", 1)[-1]

    processing_level = None
    level_description = None
    acquisition = AcquisitionInfo()
    camera = CameraInfo()
    geometry = ObservationGeometry()
    calibration_units = None
    target = None

    if observation is not None:
        primary = _find(observation, "Primary_Result_Summary")
        if primary is not None:
            processing_level = _text(primary, "processing_level")
            level_description = _text(primary, "description")

        time_coords = _find(observation, "Time_Coordinates")
        if time_coords is not None:
            acquisition.start_date_time = _text(time_coords, "start_date_time")
            acquisition.stop_date_time = _text(time_coords, "stop_date_time")

        target_node = _find(observation, "Target_Identification")
        if target_node is not None:
            target = _text(target_node, "name")

        system = _find(observation, "Observing_System")
        if system is not None:
            for comp in _find_all(system, "Observing_System_Component"):
                comp_type = (_text(comp, "type") or "").lower()
                if comp_type == "instrument":
                    camera.name = _text(comp, "name")
                    camera.description = _text(comp, "description")
            if camera.description and "panchromatic" in camera.description.lower():
                camera.band = "visible panchromatic (500-800 nm)"
            else:
                camera.band = None

        mission = _find(observation, "Mission_Area")
        if mission is not None:
            pp = _find(mission, "Product_Parameters")
            if pp is not None:
                acquisition.imaging_orbit_number = _int(pp, "imaging_orbit_number")
                acquisition.dumping_orbit_number = _int(pp, "dumping_orbit_number")
                acquisition.line_exposure_ms = _float(pp, "line_exposure_duration")
                acquisition.tdi_stages = _text(pp, "tdi_stages")
                acquisition.bits_selection = _text(pp, "bits_selection")
                acquisition.reference_data_used = _text(pp, "reference_data_used")
                acquisition.orbit_limb_direction = _text(pp, "orbit_limb_direction")
                acquisition.spacecraft_yaw_direction = _text(pp, "spacecraft_yaw_direction")
                acquisition.spacecraft_altitude_km = _float(pp, "spacecraft_altitude")
                acquisition.area = _text(pp, "area")
                camera.focal_length_mm = _float(pp, "focal_length")
                camera.detector_pixel_width_um = _float(pp, "detector_pixel_width")
                camera.pixel_resolution_mp = _float(pp, "pixel_resolution")
                geometry.roll_deg = _float(pp, "roll")
                geometry.pitch_deg = _float(pp, "pitch")
                geometry.yaw_deg = _float(pp, "yaw")
                geometry.sun_azimuth_deg = _float(pp, "sun_azimuth")
                geometry.sun_elevation_deg = _float(pp, "sun_elevation")
                geometry.solar_incidence_deg = _float(pp, "solar_incidence")
                geometry.projection = _text(pp, "projection")

            gp = _find(mission, "Geometry_Parameters")
            if gp is not None:
                coords = _find(gp, "System_Level_Coordinates")
                if coords is not None:
                    corners = {
                        "upper_left": {
                            "lat": _float(coords, "upper_left_latitude"),
                            "lon": _float(coords, "upper_left_longitude"),
                        },
                        "upper_right": {
                            "lat": _float(coords, "upper_right_latitude"),
                            "lon": _float(coords, "upper_right_longitude"),
                        },
                        "lower_left": {
                            "lat": _float(coords, "lower_left_latitude"),
                            "lon": _float(coords, "lower_left_longitude"),
                        },
                        "lower_right": {
                            "lat": _float(coords, "lower_right_latitude"),
                            "lon": _float(coords, "lower_right_longitude"),
                        },
                    }
                    geometry.corners = {k: v for k, v in corners.items()}

    acquisition.target = target

    source_file = SourceFileInfo()
    array = ArrayInfo(lines=None, samples=None, offset_bytes=0, axes_order=None, data_type=None)

    if file_area is not None:
        file_node = _find(file_area, "File")
        if file_node is not None:
            source_file.file_name = _text(file_node, "file_name")
            source_file.file_size = _int(file_node, "file_size")
            source_file.md5_checksum = _text(file_node, "md5_checksum")
            source_file.creation_date_time = _text(file_node, "creation_date_time")
            source_file.comment = _text(file_node, "comment")

        array_node = _find(file_area, "Array_2D_Image")
        if array_node is not None:
            array.offset_bytes = _int(array_node, "offset") or 0
            array.axes_order = _text(array_node, "axis_index_order")

            element = _find(array_node, "Element_Array")
            pds_type = _text(element, "data_type") if element is not None else None
            if pds_type:
                array.data_type = interpret_data_type(pds_type)

            axes = _find_all(array_node, "Axis_Array")
            for axis in axes:
                name = (_text(axis, "axis_name") or "").lower()
                count = _int(axis, "elements")
                if name == "line":
                    array.lines = count
                elif name == "sample":
                    array.samples = count

            # Special_Constants declared by the label (missing / invalid /
            # saturated sentinel values).  Absent => nothing is presumed.
            constants: Dict[str, float] = {}
            for node in _find_all(array_node, "Special_Constants"):
                for child in node:
                    lname = _local(child.tag).lower()
                    if lname.endswith("_constant"):
                        value = _float(node, child.tag.split("}")[-1]) if child.text else None
                        if value is not None:
                            constants[lname] = value
            array.special_constants = constants

    if array.data_type is None or src_is_unsupported(array):
        raise UnsupportedDataTypeError(
            "Unsupported datatype in metadata label."
            if array.data_type is None
            else f"Unsupported datatype: '{array.data_type.pds_type}'."
        )

    calibration = CalibrationInfo(
        processing_level=processing_level,
        level_description=level_description,
        units=calibration_units,
        units_comment=(
            "No radiometric unit is declared in the label. The product is "
            "'count calibrated' DN data; values are counts, not radiance."
        ),
        coefficients=None,
        calibration_note=(
            "The label states the Calibrated level is a radiometric LUT applied "
            "to raw data. The LUT coefficients are not part of this product."
        ),
    )

    metadata = OHRCMetadata(
        product_name=product_name,
        logical_identifier=logical_id,
        product_class=product_class,
        processing_level=processing_level,
        acquisition=acquisition,
        camera=camera,
        calibration=calibration,
        array=array,
        source_file=source_file,
        geometry=geometry,
        label={"raw": xml_bytes.decode("utf-8", errors="replace")},
    )
    return metadata


def src_is_unsupported(array: ArrayInfo) -> bool:
    return array.data_type is not None and not array.data_type.is_supported