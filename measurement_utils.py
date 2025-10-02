"""
Measurement utilities for converting between canvas coordinates and PDF coordinates
"""

from typing import Tuple, List, Dict
from PIL import Image, ImageDraw, ImageFont
import math


def canvas_to_pdf_coords(
    canvas_x: float,
    canvas_y: float,
    canvas_width: int,
    canvas_height: int,
    pdf_width: float,
    pdf_height: float,
    preview_dpi: int = 150
) -> Tuple[float, float]:
    """
    Convert canvas coordinates to PDF coordinates
    
    Args:
        canvas_x, canvas_y: Coordinates on the canvas
        canvas_width, canvas_height: Dimensions of the canvas
        pdf_width, pdf_height: Original PDF page dimensions in points
        preview_dpi: DPI used for preview rendering
        
    Returns:
        Tuple of (pdf_x, pdf_y) in PDF coordinate space
    """
    # Calculate the scale factor
    # Canvas is scaled to fit, and the image was rendered at preview_dpi
    dpi_scale = 72.0 / preview_dpi
    
    # Image dimensions at preview_dpi
    image_width = pdf_width * (preview_dpi / 72.0)
    image_height = pdf_height * (preview_dpi / 72.0)
    
    # Scale from canvas to image coordinates
    scale_x = image_width / canvas_width
    scale_y = image_height / canvas_height
    
    # Convert to image coordinates
    image_x = canvas_x * scale_x
    image_y = canvas_y * scale_y
    
    # Convert to PDF coordinates
    pdf_x = image_x * dpi_scale
    pdf_y = image_y * dpi_scale
    
    return (pdf_x, pdf_y)


def pdf_to_canvas_coords(
    pdf_x: float,
    pdf_y: float,
    canvas_width: int,
    canvas_height: int,
    pdf_width: float,
    pdf_height: float,
    preview_dpi: int = 150
) -> Tuple[float, float]:
    """
    Convert PDF coordinates to canvas coordinates
    
    Args:
        pdf_x, pdf_y: Coordinates in PDF space (points)
        canvas_width, canvas_height: Dimensions of the canvas
        pdf_width, pdf_height: Original PDF page dimensions in points
        preview_dpi: DPI used for preview rendering
        
    Returns:
        Tuple of (canvas_x, canvas_y)
    """
    # DPI scale factor
    dpi_scale = preview_dpi / 72.0
    
    # Convert PDF points to image pixels
    image_x = pdf_x * dpi_scale
    image_y = pdf_y * dpi_scale
    
    # Image dimensions at preview_dpi
    image_width = pdf_width * dpi_scale
    image_height = pdf_height * dpi_scale
    
    # Scale from image to canvas
    scale_x = canvas_width / image_width
    scale_y = canvas_height / image_height
    
    canvas_x = image_x * scale_x
    canvas_y = image_y * scale_y
    
    return (canvas_x, canvas_y)


def draw_measurement_on_image(
    image: Image.Image,
    measurement_type: str,
    points: List[Tuple[float, float]],
    label: str = "",
    value_text: str = "",
    color: str = "red",
    line_width: int = 2
) -> Image.Image:
    """
    Draw measurement annotations on an image
    
    Args:
        image: PIL Image to draw on
        measurement_type: Type of measurement ("distance", "perimeter", "area")
        points: List of points in image coordinates
        label: Optional label text
        value_text: Measurement value to display
        color: Color of the annotation
        line_width: Width of the lines
        
    Returns:
        Modified image
    """
    draw = ImageDraw.Draw(image)
    
    if measurement_type == "distance" and len(points) >= 2:
        # Draw line
        draw.line([points[0], points[1]], fill=color, width=line_width)
        
        # Draw endpoints
        for point in points:
            radius = 4
            draw.ellipse(
                [point[0] - radius, point[1] - radius, 
                 point[0] + radius, point[1] + radius],
                fill=color,
                outline="white",
                width=1
            )
        
        # Draw text at midpoint
        if value_text:
            mid_x = (points[0][0] + points[1][0]) / 2
            mid_y = (points[0][1] + points[1][1]) / 2
            
            # Background rectangle for text
            text_bbox = draw.textbbox((mid_x, mid_y), value_text)
            padding = 4
            draw.rectangle(
                [text_bbox[0] - padding, text_bbox[1] - padding,
                 text_bbox[2] + padding, text_bbox[3] + padding],
                fill="white",
                outline=color,
                width=1
            )
            draw.text((mid_x, mid_y), value_text, fill=color, anchor="mm")
    
    elif measurement_type in ["perimeter", "area"] and len(points) >= 3:
        # Draw polygon
        draw.polygon(points, outline=color, width=line_width)
        
        if measurement_type == "area":
            # Fill with semi-transparent color
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Convert color name to RGB with alpha
            color_map = {
                "red": (255, 0, 0, 50),
                "blue": (0, 0, 255, 50),
                "green": (0, 255, 0, 50),
                "yellow": (255, 255, 0, 50)
            }
            fill_color = color_map.get(color, (255, 0, 0, 50))
            overlay_draw.polygon(points, fill=fill_color)
            
            # Composite the overlay
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
        
        # Draw vertices
        for point in points:
            radius = 4
            draw.ellipse(
                [point[0] - radius, point[1] - radius,
                 point[0] + radius, point[1] + radius],
                fill=color,
                outline="white",
                width=1
            )
        
        # Draw text at centroid
        if value_text:
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Background for text
            text_bbox = draw.textbbox((centroid_x, centroid_y), value_text)
            padding = 4
            draw.rectangle(
                [text_bbox[0] - padding, text_bbox[1] - padding,
                 text_bbox[2] + padding, text_bbox[3] + padding],
                fill="white",
                outline=color,
                width=1
            )
            draw.text((centroid_x, centroid_y), value_text, fill=color, anchor="mm")
    
    return image


def format_measurement_value(value: float, unit: str, measurement_type: str) -> str:
    """
    Format measurement value for display
    
    Args:
        value: Numerical value
        unit: Unit of measurement
        measurement_type: Type of measurement
        
    Returns:
        Formatted string
    """
    if measurement_type == "area":
        # Area uses square units
        if unit in ["in", "inches"]:
            return f"{value:.3f} in²"
        elif unit in ["cm", "centimeters"]:
            return f"{value:.3f} cm²"
        elif unit in ["mm", "millimeters"]:
            return f"{value:.2f} mm²"
        elif unit in ["ft", "feet"]:
            return f"{value:.3f} ft²"
        elif unit in ["m", "meters"]:
            return f"{value:.3f} m²"
        else:
            return f"{value:.2f} {unit}²"
    else:
        # Linear measurements
        if unit in ["in", "inches"]:
            return f"{value:.3f} in"
        elif unit in ["cm", "centimeters"]:
            return f"{value:.3f} cm"
        elif unit in ["mm", "millimeters"]:
            return f"{value:.2f} mm"
        elif unit in ["ft", "feet"]:
            # Convert to feet and inches
            feet = int(value)
            inches = (value - feet) * 12
            if inches < 0.1:
                return f"{feet}'"
            return f"{feet}' {inches:.1f}\""
        elif unit in ["m", "meters"]:
            return f"{value:.3f} m"
        else:
            return f"{value:.2f} {unit}"


def extract_canvas_objects_as_points(
    canvas_objects: List[Dict],
    canvas_width: int,
    canvas_height: int,
    pdf_width: float,
    pdf_height: float,
    preview_dpi: int = 150
) -> List[Tuple[float, float]]:
    """
    Extract points from canvas objects (lines, polygons, etc.) and convert to PDF coords
    
    Args:
        canvas_objects: List of objects from streamlit-drawable-canvas
        canvas_width, canvas_height: Canvas dimensions
        pdf_width, pdf_height: PDF page dimensions
        preview_dpi: Preview DPI
        
    Returns:
        List of points in PDF coordinates
    """
    points = []
    
    for obj in canvas_objects:
        obj_type = obj.get("type")
        
        if obj_type == "line":
            # Line has x1, y1, x2, y2
            p1 = canvas_to_pdf_coords(
                obj.get("x1", 0), obj.get("y1", 0),
                canvas_width, canvas_height,
                pdf_width, pdf_height, preview_dpi
            )
            p2 = canvas_to_pdf_coords(
                obj.get("x2", 0), obj.get("y2", 0),
                canvas_width, canvas_height,
                pdf_width, pdf_height, preview_dpi
            )
            points.extend([p1, p2])
        
        elif obj_type == "path":
            # Path has a list of points
            path_points = obj.get("path", [])
            for path_point in path_points:
                if len(path_point) >= 2:
                    # Path points are typically ["L", x, y] or ["M", x, y]
                    if isinstance(path_point[0], str):
                        x, y = path_point[1], path_point[2]
                    else:
                        x, y = path_point[0], path_point[1]
                    
                    pdf_point = canvas_to_pdf_coords(
                        x, y, canvas_width, canvas_height,
                        pdf_width, pdf_height, preview_dpi
                    )
                    points.append(pdf_point)
        
        elif obj_type == "polygon":
            # Polygon has points array
            poly_points = obj.get("points", [])
            for poly_point in poly_points:
                pdf_point = canvas_to_pdf_coords(
                    poly_point.get("x", 0), poly_point.get("y", 0),
                    canvas_width, canvas_height,
                    pdf_width, pdf_height, preview_dpi
                )
                points.append(pdf_point)
        
        elif obj_type == "rect":
            # Rectangle - convert to 4 corner points
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            
            corners = [
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height)
            ]
            
            for corner in corners:
                pdf_point = canvas_to_pdf_coords(
                    corner[0], corner[1],
                    canvas_width, canvas_height,
                    pdf_width, pdf_height, preview_dpi
                )
                points.append(pdf_point)
    
    return points


def calculate_scale_from_known_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    known_distance: float,
    known_unit: str
) -> Dict:
    """
    Calculate scale calibration from a known distance
    
    Args:
        p1, p2: Two points in PDF coordinates
        known_distance: The real-world distance between these points
        known_unit: Unit of the known distance
        
    Returns:
        Dictionary with calibration information
    """
    from measurement_processor import MeasurementProcessor
    
    pdf_distance = MeasurementProcessor.calculate_distance(p1, p2)
    
    return {
        "pdf_distance": pdf_distance,
        "real_distance": known_distance,
        "unit": known_unit,
        "scale_ratio": f"1:{known_distance/pdf_distance:.4f}",
        "conversion_factor": known_distance / pdf_distance
    }