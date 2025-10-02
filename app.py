import os
import tempfile
import fitz  # PyMuPDF
from collections import defaultdict
from typing import List, Dict, Any
import time

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from redaction_logic import analyse_document_for_redactions
from pdf_processor import PDFProcessor
from utils import get_original_pdf_images

# Measurement imports
from measurement_processor import MeasurementProcessor, ScaleCalibration, Unit, MeasurementType
from measurement_utils import (
    canvas_to_pdf_coords,
    pdf_to_canvas_coords,
    extract_canvas_objects_as_points,
    format_measurement_value,
    draw_measurement_on_image
)

# ---------- Page / constants ----------
st.set_page_config(
    page_title="AI Document Redactor & Measurement Tool", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-powered document redaction and measurement tool for sensitive information protection and PDF analysis."
    }
)

PREVIEW_DPI = 150
CANVAS_DISPLAY_WIDTH = 800

# ---------- CSS Styling ----------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stats-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .suggestion-item {
        background: white;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .category-badge {
        background: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .shortcut-hint {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .measurement-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state initialisation ----------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("processed_file", None)
    ss.setdefault("original_pdf_images", [])
    ss.setdefault("display_images", [])
    ss.setdefault("active_page_index", 0)
    ss.setdefault("suggestions", [])
    ss.setdefault("manual_rects", defaultdict(list))
    ss.setdefault("final_pdf_path", None)
    ss.setdefault("last_promoted_ids", [])
    ss.setdefault("drawing_mode", "rect")
    ss.setdefault("user_context", "")
    ss.setdefault("analysis_timestamp", 0)
    ss.setdefault("processing_time", 0)
    ss.setdefault("file_info", {})
    ss.setdefault("suggestion_filter", "")
    ss.setdefault("category_filter", "All")
    
    # Measurement state
    ss.setdefault("measurement_mode", False)
    ss.setdefault("measurement_type", "distance")
    ss.setdefault("measurement_processor", None)
    ss.setdefault("show_measurement_overlay", True)
    ss.setdefault("pending_measurement_objects", [])

_init_state()

# ---------- Helper Functions ----------
def _ensure_dirs() -> Dict[str, str]:
    base_tmp = tempfile.gettempdir()
    temp_dir = os.path.join(base_tmp, "redactor_tmp")
    output_dir = os.path.join(os.getcwd(), "redacted_docs")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return {"temp_dir": temp_dir, "output_dir": output_dir}

def _save_upload_to_temp(upload, temp_dir: str) -> str:
    suffix = ".pdf"
    safe_name = os.path.basename(upload.name).replace(os.sep, "_")
    with tempfile.NamedTemporaryFile(prefix="upload_", suffix=suffix, dir=temp_dir, delete=False) as tf:
        tf.write(upload.getbuffer())
        saved_path = tf.name
    return saved_path

def _get_file_info(file_path: str) -> Dict:
    """Get file information for display"""
    try:
        file_size = os.path.getsize(file_path)
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        
        return {
            "size": f"{file_size / 1024 / 1024:.2f} MB",
            "pages": page_count,
            "name": os.path.basename(file_path)
        }
    except Exception as e:
        return {"error": str(e)}

def _init_measurement_processor():
    """Initialize measurement processor if not already created"""
    if st.session_state.measurement_processor is None:
        st.session_state.measurement_processor = MeasurementProcessor()

def _build_display_image(img: Image.Image, page_index: int) -> Image.Image:
    w, h = img.size
    scale = CANVAS_DISPLAY_WIDTH / float(w)
    display_height = int(h * scale)
    display_img = img.resize((CANVAS_DISPLAY_WIDTH, display_height))
    
    draw = ImageDraw.Draw(display_img)
    
    # Draw redaction suggestions (only in redaction mode)
    if not st.session_state.measurement_mode:
        approved_suggestions = [
            s for s in st.session_state.suggestions 
            if st.session_state.get(f"cb_{s.get('id')}", True) and s.get('page_num') == page_index
        ]
        
        dpi_to_display_scaling = (PREVIEW_DPI / 72.0) * scale
        
        for suggestion in approved_suggestions:
            for rect in suggestion.get('rects', []):
                scaled_rect = (
                    rect.x0 * dpi_to_display_scaling, 
                    rect.y0 * dpi_to_display_scaling,
                    rect.x1 * dpi_to_display_scaling, 
                    rect.y1 * dpi_to_display_scaling
                )
                draw.rectangle(scaled_rect, fill="black")
    
    # Draw measurements (only in measurement mode)
    if st.session_state.measurement_mode and st.session_state.measurement_processor:
        measurements = st.session_state.measurement_processor.get_measurements_for_page(page_index)
        
        # Get PDF dimensions for coordinate conversion
        if st.session_state.processed_file:
            doc = fitz.open(st.session_state.processed_file)
            page = doc[page_index]
            pdf_width, pdf_height = page.rect.width, page.rect.height
            doc.close()
            
            # Draw each measurement
            for measurement in measurements:
                points_canvas = []
                for pdf_point in measurement.points:
                    canvas_point = pdf_to_canvas_coords(
                        pdf_point[0], pdf_point[1],
                        CANVAS_DISPLAY_WIDTH, display_height,
                        pdf_width, pdf_height,
                        PREVIEW_DPI
                    )
                    points_canvas.append(canvas_point)
                
                # Choose color based on measurement type
                color_map = {
                    MeasurementType.DISTANCE: "red",
                    MeasurementType.PERIMETER: "blue",
                    MeasurementType.AREA: "green"
                }
                color = color_map.get(measurement.measurement_type, "red")
                
                # Format value text
                value_text = format_measurement_value(
                    measurement.real_value,
                    measurement.unit.value,
                    measurement.measurement_type.value
                )
                
                # Draw on image
                display_img = draw_measurement_on_image(
                    display_img,
                    measurement.measurement_type.value,
                    points_canvas,
                    measurement.label,
                    value_text,
                    color,
                    line_width=2
                )
    
    return display_img

def _page_count() -> int:
    return len(st.session_state.original_pdf_images)

def _goto_page(i: int) -> None:
    total = _page_count()
    if total == 0:
        st.session_state.active_page_index = 0
        return
    st.session_state.active_page_index = max(0, min(total - 1, i))

def _get_suggestion_stats() -> Dict:
    """Calculate statistics about suggestions"""
    if not st.session_state.suggestions:
        return {}
    
    total = len(st.session_state.suggestions)
    approved = sum(1 for s in st.session_state.suggestions 
                  if st.session_state.get(f"cb_{s.get('id')}", True))
    
    categories = {}
    for s in st.session_state.suggestions:
        cat = s.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total": total,
        "approved": approved,
        "rejected": total - approved,
        "categories": categories
    }

def _filter_suggestions(suggestions: List[Dict]) -> List[Dict]:
    """Filter suggestions based on current filters"""
    filtered = suggestions
    
    # Text filter
    if st.session_state.suggestion_filter:
        filter_text = st.session_state.suggestion_filter.lower()
        filtered = [s for s in filtered if filter_text in s.get('text', '').lower() 
                   or filter_text in s.get('category', '').lower()]
    
    # Category filter
    if st.session_state.category_filter != "All":
        filtered = [s for s in filtered if s.get('category') == st.session_state.category_filter]
    
    return filtered

# ---------- Main UI ----------
def main():
    # ---- Header ----
    st.markdown("""
    <div class="main-header">
        <h1>🔒 AI Document Redactor & 📏 Measurement Tool</h1>
        <p>Intelligent document redaction with AI-powered sensitive content detection and precision measurement tools</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📋 Document Info")
        
        if st.session_state.file_info:
            file_info = st.session_state.file_info
            if "error" not in file_info:
                st.info(f"""
                **File:** {file_info['name']}  
                **Size:** {file_info['size']}  
                **Pages:** {file_info['pages']}
                """)
                if st.session_state.processing_time > 0:
                    st.success(f"⏱️ Processed in {st.session_state.processing_time:.1f}s")
        
        st.divider()
        
        # Mode toggle
        if _page_count() > 0:
            st.header("🔧 Mode Selection")
            current_mode = "📏 Measurement" if st.session_state.measurement_mode else "🖍️ Redaction"
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🖍️ Redact", 
                            type="primary" if not st.session_state.measurement_mode else "secondary",
                            use_container_width=True):
                    st.session_state.measurement_mode = False
                    st.rerun()
            
            with col2:
                if st.button("📏 Measure",
                            type="primary" if st.session_state.measurement_mode else "secondary",
                            use_container_width=True):
                    st.session_state.measurement_mode = True
                    _init_measurement_processor()
                    st.rerun()
            
            st.caption(f"**Active:** {current_mode}")
            st.divider()
        
        # Statistics based on mode
        if st.session_state.measurement_mode and st.session_state.measurement_processor:
            st.header("📊 Measurement Stats")
            
            all_measurements = st.session_state.measurement_processor.measurements
            page_measurements = st.session_state.measurement_processor.get_measurements_for_page(
                st.session_state.active_page_index
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", len(all_measurements))
            with col2:
                st.metric("This Page", len(page_measurements))
            
            # Count by type
            if all_measurements:
                type_counts = {}
                for m in all_measurements:
                    t = m.measurement_type.value
                    type_counts[t] = type_counts.get(t, 0) + 1
                
                st.subheader("By Type")
                for mtype, count in type_counts.items():
                    st.write(f"• **{mtype.title()}:** {count}")
        
        elif st.session_state.suggestions:
            st.header("📊 Redaction Stats")
            stats = _get_suggestion_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Found", stats['total'])
                st.metric("Approved", stats['approved'], delta=None)
            with col2:
                st.metric("Rejected", stats['rejected'])
                coverage = (stats['approved'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.metric("Coverage", f"{coverage:.0f}%")
            
            # Category breakdown
            st.subheader("Categories")
            for cat, count in stats['categories'].items():
                st.write(f"• **{cat}:** {count}")
        
        st.divider()
        
        # Help section
        st.header("💡 Quick Help")
        
        if st.session_state.measurement_mode:
            st.markdown("""
            **Measurement Mode:**
            - Select measurement type (Distance/Perimeter/Area)
            - Set scale calibration for accuracy
            - Draw on canvas to measure
            - Export results to CSV/JSON
            
            **Calibration:**
            - Use "Known Distance" method for best results
            - Apply to current page or all pages
            """)
        else:
            st.markdown("""
            **Redaction Mode:**
            - Use arrow buttons to navigate pages
            - Click suggestions to jump to location
            - Draw manual boxes on canvas
            - Review all before exporting
            
            **Tips:**
            - Use specific instructions for better AI
            - Review all suggestions carefully
            """)

    # ---- Main content area ----
    paths = _ensure_dirs()
    
    # File uploader
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf", 
        key="pdf_uploader",
        help="Supported format: PDF files up to 200MB"
    )

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getbuffer()) / 1024 / 1024
        st.success(f"✅ **{uploaded_file.name}** uploaded ({file_size_mb:.2f} MB)")

    # User context (only in redaction mode)
    if not st.session_state.measurement_mode:
        st.subheader("🎯 Redaction Instructions")
        st.session_state.user_context = st.text_area(
            "Provide specific redaction instructions (optional):",
            value=st.session_state.user_context,
            placeholder="Examples:\n• 'Don't redact Oliver Hughes'\n• 'Also redact any mention of disciplinary actions'\n• 'Keep school name but redact teacher names'",
            help="The AI automatically finds common personal data. Add custom rules here for better control.",
            height=100
        )

    # Action buttons
    st.subheader("⚡ Actions")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analyse_clicked = st.button(
            "🔍 Analyze Document", 
            type="primary", 
            use_container_width=True,
            disabled=uploaded_file is None or st.session_state.measurement_mode
        )
    with col2:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)
    with col3:
        if st.session_state.suggestions and not st.session_state.measurement_mode:
            if st.button("💾 Quick Export", use_container_width=True):
                st.rerun()

    # Reset handling
    if reset_clicked:
        keys_to_delete = [k for k in st.session_state.keys() 
                         if k in ["processed_file", "original_pdf_images", "display_images", 
                                "active_page_index", "suggestions", "manual_rects", 
                                "final_pdf_path", "last_promoted_ids", "analysis_timestamp",
                                "processing_time", "file_info", "measurement_processor",
                                "pending_measurement_objects"] or k.startswith("cb_")]
        
        for k in keys_to_delete:
            if k in st.session_state:
                del st.session_state[k]
        _init_state()
        st.rerun()

    # Analysis handling
    if uploaded_file is not None and analyse_clicked:
        start_time = time.time()
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📁 Saving uploaded file...")
            progress_bar.progress(10)
            input_pdf_path = _save_upload_to_temp(uploaded_file, paths["temp_dir"])
            
            st.session_state.file_info = _get_file_info(input_pdf_path)
            
            st.session_state.processed_file = input_pdf_path
            st.session_state.final_pdf_path = None
            st.session_state.suggestions = []
            st.session_state.manual_rects = defaultdict(list)
            st.session_state.active_page_index = 0
            st.session_state.drawing_mode = "rect"
            st.session_state.analysis_timestamp = time.time()

            status_text.text("🖼️ Rendering document preview...")
            progress_bar.progress(30)
            orig_images: List[Image.Image] = get_original_pdf_images(input_pdf_path)
            st.session_state.original_pdf_images = orig_images

            status_text.text("🤖 Analyzing document with AI...")
            progress_bar.progress(50)
            suggestions = analyse_document_for_redactions(input_pdf_path, st.session_state.user_context)
            st.session_state.suggestions = suggestions or []
            
            for key in list(st.session_state.keys()):
                if key.startswith("cb_"):
                    del st.session_state[key]
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            st.session_state.processing_time = time.time() - start_time
            
            time.sleep(0.5)
            progress_container.empty()
        
        st.rerun()

    # Main content area
    if _page_count() > 0:
        st.divider()
        
        # Two-column layout
        col1, col2 = st.columns([1, 2], gap="large")
        
        # LEFT COLUMN - Context-dependent panel
        with col1:
            if st.session_state.measurement_mode:
                # MEASUREMENT CONTROLS
                _init_measurement_processor()
                
                st.header("📏 Measurement Tools")
                
                # Measurement type selection
                st.subheader("🎯 Measurement Type")
                measurement_icons = {
                    "distance": "📐",
                    "perimeter": "🔲",
                    "area": "⬛"
                }
                
                selected_type = st.radio(
                    "Select type:",
                    ["distance", "perimeter", "area"],
                    format_func=lambda x: f"{measurement_icons[x]} {x.title()}",
                    horizontal=True,
                    key="measurement_type_selector"
                )
                st.session_state.measurement_type = selected_type
                
                # Instructions
                if selected_type == "distance":
                    st.info("📐 **Distance**: Draw a line between two points")
                elif selected_type == "perimeter":
                    st.info("🔲 **Perimeter**: Draw a polygon, click 'Finalize' when done")
                else:
                    st.info("⬛ **Area**: Draw a closed shape, click 'Finalize' to calculate")
                
                st.divider()
                
                # Calibration settings
                st.subheader("⚙️ Scale Calibration")
                
                current_cal = st.session_state.measurement_processor.get_calibration(
                    st.session_state.active_page_index
                )
                
                st.markdown(f"""
                **Current Scale:** {current_cal.pdf_distance:.1f} pts = {current_cal.real_distance:.3f} {current_cal.unit.value}
                
                **Ratio:** 1 pt = {current_cal.get_conversion_factor():.6f} {current_cal.unit.value}
                """)
                
                with st.expander("🔧 Adjust Calibration", expanded=False):
                    col_pdf, col_real = st.columns(2)
                    
                    with col_pdf:
                        pdf_dist = st.number_input(
                            "PDF Distance (points)",
                            min_value=0.1,
                            value=float(current_cal.pdf_distance),
                            step=1.0,
                            help="72 points = 1 inch"
                        )
                    
                    with col_real:
                        real_dist = st.number_input(
                            "Real Distance",
                            min_value=0.001,
                            value=float(current_cal.real_distance),
                            step=0.1
                        )
                    
                    unit_options = ["inches", "cm", "mm", "feet", "meters"]
                    current_unit_name = current_cal.unit.name.lower()
                    if current_unit_name not in unit_options:
                        current_unit_name = "inches"
                    
                    unit_select = st.selectbox(
                        "Unit",
                        unit_options,
                        index=unit_options.index(current_unit_name)
                    )
                    
                    st.caption(f"**Scale:** 1 pt = {real_dist/pdf_dist:.6f} {unit_select}")
                    
                    apply_col1, apply_col2 = st.columns(2)
                    
                    with apply_col1:
                        if st.button("✅ Apply to Page", use_container_width=True):
                            unit_map = {
                                "inches": Unit.INCHES,
                                "cm": Unit.CENTIMETERS,
                                "mm": Unit.MILLIMETERS,
                                "feet": Unit.FEET,
                                "meters": Unit.METERS
                            }
                            unit_enum = unit_map.get(unit_select, Unit.INCHES)
                            new_cal = ScaleCalibration(pdf_dist, real_dist, unit_enum)
                            st.session_state.measurement_processor.set_calibration(
                                st.session_state.active_page_index, new_cal
                            )
                            st.success(f"✅ Applied to page {st.session_state.active_page_index + 1}")
                            st.rerun()
                    
                    with apply_col2:
                        if st.button("✅ Apply to All", use_container_width=True):
                            unit_map = {
                                "inches": Unit.INCHES,
                                "cm": Unit.CENTIMETERS,
                                "mm": Unit.MILLIMETERS,
                                "feet": Unit.FEET,
                                "meters": Unit.METERS
                            }
                            unit_enum = unit_map.get(unit_select, Unit.INCHES)
                            new_cal = ScaleCalibration(pdf_dist, real_dist, unit_enum)
                            st.session_state.measurement_processor.apply_calibration_to_all_pages(
                                new_cal, _page_count()
                            )
                            st.success(f"✅ Applied to all {_page_count()} pages")
                            st.rerun()
                
                st.divider()
                
                # Display measurements
                st.subheader(f"📋 Measurements (Page {st.session_state.active_page_index + 1})")
                
                page_measurements = st.session_state.measurement_processor.get_measurements_for_page(
                    st.session_state.active_page_index
                )
                
                if page_measurements:
                    for i, measurement in enumerate(page_measurements):
                        with st.container():
                            mtype_icon = measurement_icons.get(measurement.measurement_type.value, "📏")
                            
                            col_info, col_value, col_delete = st.columns([3, 2, 1])
                            
                            with col_info:
                                st.markdown(f"**{mtype_icon} {measurement.measurement_type.value.title()}**")
                            
                            with col_value:
                                formatted_value = format_measurement_value(
                                    measurement.real_value,
                                    measurement.unit.value,
                                    measurement.measurement_type.value
                                )
                                st.metric("", formatted_value)
                                st.caption(f"{measurement.value:.1f} pts")
                            
                            with col_delete:
                                if st.button("🗑️", key=f"del_m_{i}_{st.session_state.active_page_index}"):
                                    st.session_state.measurement_processor.measurements.remove(measurement)
                                    st.rerun()
                            
                            st.divider()
                else:
                    st.info("📭 No measurements on this page. Use the canvas to start measuring!")
                
                # Export section
                if st.session_state.measurement_processor.measurements:
                    st.divider()
                    st.subheader("📤 Export")
                    
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        csv_data = st.session_state.measurement_processor.export_to_csv()
                        st.download_button(
                            "📄 CSV",
                            data=csv_data,
                            file_name="measurements.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with export_col2:
                        json_data = st.session_state.measurement_processor.export_to_json()
                        st.download_button(
                            "📋 JSON",
                            data=json_data,
                            file_name="measurements.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    if st.button("🗑️ Clear All", use_container_width=True):
                        st.session_state.measurement_processor.clear_measurements()
                        st.success("Cleared all measurements")
                        st.rerun()
            
            else:
                # REDACTION SUGGESTIONS PANEL
                st.header("🎯 AI Suggestions")
                
                filter_col1, filter_col2 = st.columns([2, 1])
                with filter_col1:
                    st.session_state.suggestion_filter = st.text_input(
                        "🔍 Search suggestions", 
                        value=st.session_state.suggestion_filter,
                        placeholder="Filter by text or category..."
                    )
                
                with filter_col2:
                    categories = ["All"] + sorted(list(set(s.get('category', 'Unknown') 
                                                        for s in st.session_state.suggestions)))
                    st.session_state.category_filter = st.selectbox(
                        "Category", 
                        categories,
                        index=categories.index(st.session_state.category_filter) 
                              if st.session_state.category_filter in categories else 0
                    )
                
                bulk_col1, bulk_col2 = st.columns(2)
                with bulk_col1:
                    if st.button("✅ Approve All", use_container_width=True):
                        for suggestion in st.session_state.suggestions:
                            st.session_state[f"cb_{suggestion.get('id')}"] = True
                        st.rerun()
                with bulk_col2:
                    if st.button("❌ Reject All", use_container_width=True):
                        for suggestion in st.session_state.suggestions:
                            st.session_state[f"cb_{suggestion.get('id')}"] = False
                        st.rerun()
                
                if st.session_state.suggestions:
                    filtered_suggestions = _filter_suggestions(st.session_state.suggestions)
                    
                    if not filtered_suggestions:
                        st.warning("No suggestions match the current filters.")
                    else:
                        doc = fitz.open(st.session_state.processed_file)
                        try:
                            for suggestion in filtered_suggestions:
                                suggestion_id = suggestion.get('id')
                                category = suggestion.get('category', 'Unknown')
                                text = suggestion.get('text', 'No text')
                                page_num = suggestion.get('page_num', 0)
                                
                                try:
                                    context_snippet = f"Pg {page_num + 1}: {text}"
                                    if len(text) > 50:
                                        context_snippet = f"Pg {page_num + 1}: {text[:50]}..."
                                except:
                                    context_snippet = f"Pg {page_num + 1}: {text}"
                                
                                checkbox_key = f"cb_{suggestion_id}"
                                if checkbox_key not in st.session_state:
                                    st.session_state[checkbox_key] = True
                                
                                with st.container():
                                    col_check, col_goto = st.columns([4, 1])
                                    
                                    with col_check:
                                        st.checkbox(
                                            f"**{category}**: {context_snippet}",
                                            key=checkbox_key
                                        )
                                    
                                    with col_goto:
                                        if st.button("👁️", key=f"goto_{suggestion_id}", 
                                                   help="Jump to suggestion"):
                                            _goto_page(page_num)
                                            st.rerun()
                        finally:
                            doc.close()
                else:
                    st.info("🔍 No AI suggestions found for this document.")
        
        # RIGHT COLUMN - Canvas/Preview
        with col2:
            st.header("📄 Document Preview")
            
            total_pages = _page_count()
            page_index = st.session_state.active_page_index

            # Navigation
            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns([0.15, 0.15, 0.3, 0.2, 0.15, 0.15])
            
            with nav_col1:
                if st.button("⏮️", use_container_width=True, disabled=(page_index == 0)):
                    _goto_page(0)
                    st.rerun()
            
            with nav_col2:
                if st.button("◀️", use_container_width=True, disabled=(page_index == 0)):
                    _goto_page(page_index - 1)
                    st.rerun()
            
            with nav_col3:
                new_page = st.number_input(
                    "Page", 
                    min_value=1, 
                    max_value=total_pages, 
                    value=page_index + 1
                )
                if new_page - 1 != page_index:
                    _goto_page(new_page - 1)
                    st.rerun()
            
            with nav_col4:
                st.markdown(f"**of {total_pages}**")
            
            with nav_col5:
                if st.button("▶️", use_container_width=True, disabled=(page_index >= total_pages - 1)):
                    _goto_page(page_index + 1)
                    st.rerun()
            
            with nav_col6:
                if st.button("⏭️", use_container_width=True, disabled=(page_index >= total_pages - 1)):
                    _goto_page(total_pages - 1)
                    st.rerun()

            # Canvas based on mode
            if st.session_state.measurement_mode:
                # MEASUREMENT CANVAS
                st.subheader("📏 Measurement Canvas")
                
                if st.session_state.measurement_type == "distance":
                    canvas_drawing_mode = "line"
                    stroke_color = "#FF0000"
                else:
                    canvas_drawing_mode = "polygon"
                    stroke_color = "#0000FF" if st.session_state.measurement_type == "perimeter" else "#00FF00"
                
                base_display = _build_display_image(
                    st.session_state.original_pdf_images[page_index], 
                    page_index
                )
                display_height = base_display.size[1]
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.2)" if st.session_state.measurement_type == "area" else "rgba(0, 0, 0, 0)",
                    stroke_width=3,
                    stroke_color=stroke_color,
                    background_image=base_display,
                    update_streamlit=True,
                    height=display_height,
                    width=CANVAS_DISPLAY_WIDTH,
                    drawing_mode=canvas_drawing_mode,
                    key=f"measurement_canvas_{page_index}_{st.session_state.analysis_timestamp}",
                    display_toolbar=False
                )
                
                # Process measurement
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    
                    if objects:
                        if st.button("✅ Finalize Measurement", type="primary"):
                            doc = fitz.open(st.session_state.processed_file)
                            page = doc[page_index]
                            pdf_width, pdf_height = page.rect.width, page.rect.height
                            doc.close()
                            
                            points = extract_canvas_objects_as_points(
                                objects, CANVAS_DISPLAY_WIDTH, display_height,
                                pdf_width, pdf_height, PREVIEW_DPI
                            )
                            
                            if points:
                                processor = st.session_state.measurement_processor
                                measurement_type = st.session_state.measurement_type
                                
                                try:
                                    if measurement_type == "distance" and len(points) >= 2:
                                        result = processor.measure_distance(
                                            points[0], points[1], page_num=page_index
                                        )
                                        formatted = format_measurement_value(
                                            result.real_value, result.unit.value, "distance"
                                        )
                                        st.success(f"✅ Distance: {formatted}")
                                        
                                    elif measurement_type == "perimeter" and len(points) >= 2:
                                        result = processor.measure_perimeter(
                                            points, page_num=page_index
                                        )
                                        formatted = format_measurement_value(
                                            result.real_value, result.unit.value, "perimeter"
                                        )
                                        st.success(f"✅ Perimeter: {formatted}")
                                        
                                    elif measurement_type == "area" and len(points) >= 3:
                                        result = processor.measure_area(
                                            points, page_num=page_index
                                        )
                                        formatted = format_measurement_value(
                                            result.real_value, result.unit.value, "area"
                                        )
                                        st.success(f"✅ Area: {formatted}")
                                        
                                    else:
                                        st.warning(f"⚠️ Need more points for {measurement_type}")
                                    
                                    time.sleep(0.5)
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
            
            else:
                # REDACTION CANVAS
                st.subheader("🎨 Canvas Mode")
                mode = st.radio(
                    "",
                    ["✏️ Draw new redactions", "✂️ Edit/Delete existing"],
                    horizontal=True,
                    index=0 if st.session_state.drawing_mode == "rect" else 1,
                    key="canvas_mode",
                )
                st.session_state.drawing_mode = "rect" if mode.startswith("✏️") else "transform"

                base_display = _build_display_image(
                    st.session_state.original_pdf_images[page_index], 
                    page_index
                )
                display_height = base_display.size[1]

                if st.session_state.drawing_mode == "rect":
                    st.info("💡 **Draw mode**: Click and drag to add black redaction boxes.")
                else:
                    manual_boxes = st.session_state.manual_rects.get(page_index, [])
                    if not manual_boxes:
                        st.warning("⚠️ **Edit mode**: No manual boxes on this page.")
                    else:
                        st.info(f"💡 **Edit mode**: {len(manual_boxes)} manual boxes on this page.")

                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1.0)",
                    stroke_width=0,
                    background_image=base_display,
                    update_streamlit=True,
                    height=display_height,
                    width=CANVAS_DISPLAY_WIDTH,
                    drawing_mode=st.session_state.drawing_mode,
                    initial_drawing={"objects": st.session_state.manual_rects.get(page_index, [])},
                    display_toolbar=True,
                    key=f"canvas_{page_index}_{st.session_state.analysis_timestamp}",
                )

                if canvas_result.json_data is not None:
                    new_objs = canvas_result.json_data.get("objects", [])
                    if new_objs != st.session_state.manual_rects.get(page_index, []):
                        st.session_state.manual_rects[page_index] = new_objs

        # Export section (redaction mode only)
        if not st.session_state.measurement_mode:
            st.divider()
            st.header("📤 Export Redacted Document")
            
            if st.session_state.suggestions:
                stats = _get_suggestion_stats()
                total_redactions = stats['approved'] + sum(len(objs) for objs in st.session_state.manual_rects.values())
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("AI Redactions", stats['approved'])
                with summary_col2:
                    manual_count = sum(len(objs) for objs in st.session_state.manual_rects.values())
                    st.metric("Manual Redactions", manual_count)
                with summary_col3:
                    st.metric("Total Redactions", total_redactions)
            
            export_col1, export_col2 = st.columns([0.3, 0.7])
            
            with export_col1:
                export_clicked = st.button(
                    "🚀 Apply Redactions & Export", 
                    type="primary", 
                    use_container_width=True
                )
            
            with export_col2:
                if st.session_state.final_pdf_path:
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>Export Complete!</strong><br>
                        📁 File: {os.path.basename(st.session_state.final_pdf_path)}
                    </div>
                    """, unsafe_allow_html=True)

            if export_clicked:
                with st.spinner("🔄 Applying redactions..."):
                    all_redactions = defaultdict(list)
                    
                    # Add approved AI suggestions
                    for suggestion in st.session_state.suggestions:
                        suggestion_id = suggestion.get('id')
                        checkbox_key = f"cb_{suggestion_id}"
                        if st.session_state.get(checkbox_key, True):
                            page_num = suggestion.get('page_num', 0)
                            for rect in suggestion.get('rects', []):
                                all_redactions[page_num].append({
                                    "x": rect.x0, "y": rect.y0, 
                                    "w": rect.x1 - rect.x0, "h": rect.y1 - rect.y0
                                })
                    
                    # Add manual rectangles
                    for p_idx, objs in st.session_state.manual_rects.items():
                        if not objs:
                            continue
            
                        orig_w, orig_h = st.session_state.original_pdf_images[p_idx].size
                        display_w = CANVAS_DISPLAY_WIDTH
                        display_h = int(orig_h * CANVAS_DISPLAY_WIDTH / orig_w)
                        sx = orig_w / float(display_w)
                        sy = orig_h / float(display_h)
        
                        dpi_scale = 72.0 / PREVIEW_DPI

                        for o in objs:
                            if o.get("type") != "rect":
                                continue
                            left = o.get("left", 0) * sx * dpi_scale
                            top = o.get("top", 0) * sy * dpi_scale
                            width = o.get("width", 0) * sx * dpi_scale
                            height = o.get("height", 0) * sy * dpi_scale
                            all_redactions[p_idx].append({"x": left, "y": top, "w": width, "h": height})

                    if not any(all_redactions.values()):
                        st.error("❌ No redactions to apply.")
                    else:
                        processor = PDFProcessor()
                        input_path = st.session_state.processed_file
                        timestamp = int(time.time())
                        output_filename = f"redacted_{timestamp}_{os.path.basename(input_path)}"
                        output_path = os.path.join(paths["output_dir"], output_filename)
                        
                        try:
                            processor.apply_rect_redactions(input_path, dict(all_redactions), output_path)
                            st.session_state.final_pdf_path = output_path
                            
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    "📥 Download Redacted PDF", 
                                    data=file.read(),
                                    file_name=output_filename, 
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            
                            st.success(f"🎉 Redaction complete! File saved as `{output_filename}`")
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>🚀 Ready to Get Started?</h2>
            <p style="font-size: 1.2rem; color: #666;">Upload a PDF document above and click <strong>Analyze Document</strong> to begin intelligent redaction or switch to Measurement mode for precision measurements.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()