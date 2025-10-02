"""
PDF Measurement Processor
Handles distance, perimeter, and area measurements in PDF documents
with calibration support for real-world units.
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class MeasurementType(Enum):
    """Types of measurements supported"""
    DISTANCE = "distance"
    PERIMETER = "perimeter"
    AREA = "area"


class Unit(Enum):
    """Supported measurement units"""
    POINTS = "pt"
    INCHES = "in"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"
    FEET = "ft"
    METERS = "m"
    PIXELS = "px"


@dataclass
class ScaleCalibration:
    """Scale calibration for converting PDF units to real-world measurements"""
    pdf_distance: float = 72.0  # Distance in PDF points
    real_distance: float = 1.0   # Corresponding real-world distance
    unit: Unit = Unit.INCHES      # Unit of the real-world distance
    
    def get_conversion_factor(self) -> float:
        """Calculate the conversion factor from PDF points to real units"""
        return self.real_distance / self.pdf_distance
    
    def to_dict(self) -> dict:
        return {
            "pdf_distance": self.pdf_distance,
            "real_distance": self.real_distance,
            "unit": self.unit.value
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ScaleCalibration':
        return cls(
            pdf_distance=data["pdf_distance"],
            real_distance=data["real_distance"],
            unit=Unit(data["unit"])
        )


@dataclass
class MeasurementResult:
    """Result of a measurement operation"""
    measurement_type: MeasurementType
    value: float  # In PDF points
    real_value: float  # In calibrated units
    unit: Unit
    points: List[Tuple[float, float]]
    label: str = ""
    page_num: int = 0
    properties: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "type": self.measurement_type.value,
            "value_pdf_points": self.value,
            "value_real": self.real_value,
            "unit": self.unit.value,
            "label": self.label,
            "page_num": self.page_num,
            "points": self.points,
            "properties": self.properties
        }


class MeasurementProcessor:
    """
    Core processor for PDF measurements with calibration support
    """
    
    def __init__(self, page_calibrations: Optional[Dict[int, ScaleCalibration]] = None):
        """
        Initialize the measurement processor
        
        Args:
            page_calibrations: Dictionary mapping page numbers to their calibrations
        """
        self.page_calibrations = page_calibrations or {}
        self.default_calibration = ScaleCalibration()
        self.measurements: List[MeasurementResult] = []
    
    def set_calibration(self, page_num: int, calibration: ScaleCalibration):
        """Set calibration for a specific page"""
        self.page_calibrations[page_num] = calibration
    
    def get_calibration(self, page_num: int) -> ScaleCalibration:
        """Get calibration for a specific page"""
        return self.page_calibrations.get(page_num, self.default_calibration)
    
    def apply_calibration_to_all_pages(self, calibration: ScaleCalibration, num_pages: int):
        """Apply the same calibration to all pages"""
        for i in range(num_pages):
            self.page_calibrations[i] = calibration
    
    @staticmethod
    def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            Distance in PDF points
        """
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    @staticmethod
    def calculate_perimeter(points: List[Tuple[float, float]]) -> float:
        """
        Calculate perimeter of a polygon
        
        Args:
            points: List of points forming the polygon
            
        Returns:
            Perimeter in PDF points
        """
        if len(points) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            perimeter += MeasurementProcessor.calculate_distance(p1, p2)
        
        return perimeter
    
    @staticmethod
    def calculate_area(points: List[Tuple[float, float]]) -> float:
        """
        Calculate area of a polygon using the Shoelace formula
        
        Args:
            points: List of points forming the polygon (must be ordered)
            
        Returns:
            Area in square PDF points
        """
        if len(points) < 3:
            return 0.0
        
        # Shoelace formula
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def measure_distance(
        self, 
        p1: Tuple[float, float], 
        p2: Tuple[float, float],
        page_num: int = 0,
        label: str = ""
    ) -> MeasurementResult:
        """
        Measure distance between two points
        
        Args:
            p1: First point (x, y) in PDF coordinates
            p2: Second point (x, y) in PDF coordinates
            page_num: Page number
            label: Optional label for the measurement
            
        Returns:
            MeasurementResult with distance information
        """
        pdf_distance = self.calculate_distance(p1, p2)
        calibration = self.get_calibration(page_num)
        real_distance = pdf_distance * calibration.get_conversion_factor()
        
        result = MeasurementResult(
            measurement_type=MeasurementType.DISTANCE,
            value=pdf_distance,
            real_value=real_distance,
            unit=calibration.unit,
            points=[p1, p2],
            label=label,
            page_num=page_num,
            properties={"angle": self._calculate_angle(p1, p2)}
        )
        
        self.measurements.append(result)
        return result
    
    def measure_perimeter(
        self,
        points: List[Tuple[float, float]],
        page_num: int = 0,
        label: str = ""
    ) -> MeasurementResult:
        """
        Measure perimeter of a polygon
        
        Args:
            points: List of points forming the polygon
            page_num: Page number
            label: Optional label for the measurement
            
        Returns:
            MeasurementResult with perimeter information
        """
        pdf_perimeter = self.calculate_perimeter(points)
        calibration = self.get_calibration(page_num)
        real_perimeter = pdf_perimeter * calibration.get_conversion_factor()
        
        result = MeasurementResult(
            measurement_type=MeasurementType.PERIMETER,
            value=pdf_perimeter,
            real_value=real_perimeter,
            unit=calibration.unit,
            points=points,
            label=label,
            page_num=page_num,
            properties={"num_sides": len(points)}
        )
        
        self.measurements.append(result)
        return result
    
    def measure_area(
        self,
        points: List[Tuple[float, float]],
        page_num: int = 0,
        label: str = ""
    ) -> MeasurementResult:
        """
        Measure area of a polygon
        
        Args:
            points: List of points forming the polygon
            page_num: Page number
            label: Optional label for the measurement
            
        Returns:
            MeasurementResult with area information
        """
        pdf_area = self.calculate_area(points)
        calibration = self.get_calibration(page_num)
        conversion_factor = calibration.get_conversion_factor()
        real_area = pdf_area * (conversion_factor ** 2)  # Square the factor for area
        
        # Calculate perimeter as well
        pdf_perimeter = self.calculate_perimeter(points)
        real_perimeter = pdf_perimeter * conversion_factor
        
        result = MeasurementResult(
            measurement_type=MeasurementType.AREA,
            value=pdf_area,
            real_value=real_area,
            unit=calibration.unit,
            points=points,
            label=label,
            page_num=page_num,
            properties={
                "perimeter_pdf": pdf_perimeter,
                "perimeter_real": real_perimeter,
                "num_sides": len(points)
            }
        )
        
        self.measurements.append(result)
        return result
    
    @staticmethod
    def _calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate angle of line from p1 to p2 in degrees"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def get_measurements_for_page(self, page_num: int) -> List[MeasurementResult]:
        """Get all measurements for a specific page"""
        return [m for m in self.measurements if m.page_num == page_num]
    
    def clear_measurements(self, page_num: Optional[int] = None):
        """Clear measurements for a page or all measurements"""
        if page_num is not None:
            self.measurements = [m for m in self.measurements if m.page_num != page_num]
        else:
            self.measurements = []
    
    def export_to_csv(self) -> str:
        """
        Export all measurements to CSV format
        
        Returns:
            CSV string with all measurements
        """
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Page", "Type", "Label", "Value (PDF Points)", 
            f"Value (Real)", "Unit", "Points", "Properties"
        ])
        
        # Data rows
        for m in self.measurements:
            writer.writerow([
                m.page_num + 1,
                m.measurement_type.value,
                m.label,
                f"{m.value:.2f}",
                f"{m.real_value:.4f}",
                m.unit.value,
                str(m.points),
                json.dumps(m.properties)
            ])
        
        return output.getvalue()
    
    def export_to_json(self) -> str:
        """Export all measurements to JSON format"""
        data = {
            "measurements": [m.to_dict() for m in self.measurements],
            "calibrations": {
                page: cal.to_dict() 
                for page, cal in self.page_calibrations.items()
            }
        }
        return json.dumps(data, indent=2)
    
    def import_from_json(self, json_str: str):
        """Import measurements from JSON"""
        data = json.loads(json_str)
        
        # Import calibrations
        if "calibrations" in data:
            for page_str, cal_data in data["calibrations"].items():
                page = int(page_str)
                self.page_calibrations[page] = ScaleCalibration.from_dict(cal_data)
        
        # Import measurements
        if "measurements" in data:
            for m_data in data["measurements"]:
                measurement = MeasurementResult(
                    measurement_type=MeasurementType(m_data["type"]),
                    value=m_data["value_pdf_points"],
                    real_value=m_data["value_real"],
                    unit=Unit(m_data["unit"]),
                    points=[tuple(p) for p in m_data["points"]],
                    label=m_data.get("label", ""),
                    page_num=m_data.get("page_num", 0),
                    properties=m_data.get("properties", {})
                )
                self.measurements.append(measurement)


class SnapHelper:
    """
    Helper class for snapping to endpoints, midpoints, paths, and intersections
    """
    
    def __init__(self, snap_threshold: float = 10.0):
        """
        Initialize snap helper
        
        Args:
            snap_threshold: Distance threshold for snapping (in PDF points)
        """
        self.snap_threshold = snap_threshold
        self.endpoints: List[Tuple[float, float]] = []
        self.midpoints: List[Tuple[float, float]] = []
        self.paths: List[List[Tuple[float, float]]] = []
    
    def add_path(self, points: List[Tuple[float, float]]):
        """Add a path (line or polygon) to snap to"""
        if len(points) < 2:
            return
        
        self.paths.append(points)
        
        # Add endpoints
        self.endpoints.extend([points[0], points[-1]])
        
        # Add midpoints for each segment
        for i in range(len(points) - 1):
            midpoint = (
                (points[i][0] + points[i+1][0]) / 2,
                (points[i][1] + points[i+1][1]) / 2
            )
            self.midpoints.append(midpoint)
    
    def snap_to_nearest(
        self, 
        point: Tuple[float, float],
        snap_to_endpoints: bool = True,
        snap_to_midpoints: bool = True,
        snap_to_paths: bool = False
    ) -> Tuple[float, float]:
        """
        Snap a point to the nearest endpoint, midpoint, or path
        
        Args:
            point: The point to snap
            snap_to_endpoints: Whether to snap to endpoints
            snap_to_midpoints: Whether to snap to midpoints
            snap_to_paths: Whether to snap to paths (nearest point on line)
            
        Returns:
            Snapped point (or original point if no snap point found)
        """
        candidates = []
        
        # Check endpoints
        if snap_to_endpoints:
            candidates.extend(self.endpoints)
        
        # Check midpoints
        if snap_to_midpoints:
            candidates.extend(self.midpoints)
        
        # Find nearest candidate
        nearest_point = point
        min_distance = self.snap_threshold
        
        for candidate in candidates:
            dist = MeasurementProcessor.calculate_distance(point, candidate)
            if dist < min_distance:
                min_distance = dist
                nearest_point = candidate
        
        # Check paths if enabled
        if snap_to_paths:
            for path in self.paths:
                for i in range(len(path) - 1):
                    p1, p2 = path[i], path[i+1]
                    nearest_on_segment = self._nearest_point_on_segment(point, p1, p2)
                    dist = MeasurementProcessor.calculate_distance(point, nearest_on_segment)
                    if dist < min_distance:
                        min_distance = dist
                        nearest_point = nearest_on_segment
        
        return nearest_point
    
    @staticmethod
    def _nearest_point_on_segment(
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Find the nearest point on a line segment to a given point"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return seg_start
        
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp to segment
        
        return (x1 + t * dx, y1 + t * dy)
    
    def clear(self):
        """Clear all snap points and paths"""
        self.endpoints.clear()
        self.midpoints.clear()
        self.paths.clear()