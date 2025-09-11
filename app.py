import streamlit as st
import fitz  # PyMuPDF
import os
from collections import defaultdict
from PIL import Image, ImageDraw
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
from redaction_logic import analyse_document_for_redactions
from pdf_processor import PDFProcessor

st.set_page_config(page_title="AI Document Redactor", layout="wide")

PREVIEW_DPI = 150
# Define a fixed display width for the canvas to prevent overflow
CANVAS_DISPLAY_WIDTH = 800

def get_original_pdf_images(pdf_path):
    """Extracts each page of a PDF as a Pillow Image object."""
    if not os.path.exists(pdf_path): return []
    try:
        doc = fitz.open(pdf_path)
        images = [Image.open(BytesIO(page.get_pixmap(dpi=PREVIEW_DPI).tobytes("png"))) for page in doc]
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error opening or rendering PDF: {e}")
        return []

def main():
    """The main function that runs the Streamlit application."""

    # Initialise all session state variables
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'final_pdf_path' not in st.session_state:
        st.session_state.final_pdf_path = None
    if 'approval_state' not in st.session_state:
        st.session_state.approval_state = {}
    if 'original_pdf_images' not in st.session_state:
        st.session_state.original_pdf_images = []
    if 'user_context' not in st.session_state:
        st.session_state.user_context = ""
    if 'manual_rects' not in st.session_state:
        st.session_state.manual_rects = defaultdict(list)
    if 'drawing_mode' not in st.session_state:
        st.session_state.drawing_mode = "rect"
    if 'active_page_index' not in st.session_state:
        st.session_state.active_page_index = 0

    # --- Main App UI ---
    st.title("AI-Powered Document Redaction Tool")
    st.write("Upload a PDF to redact. Use the options below to guide the AI and refine the results.")

    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

    st.text_area(
        "Provide specific redaction instructions for this document (optional):",
        placeholder=(
            "The AI redacts common personal data by default. Use this box to provide exceptions or new rules.\n\n"
        ),
        height=100,
        key='user_context'
    )

    if uploaded_file is not None:
        input_pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Analyse Document"):
            st.session_state.suggestions = []
            st.session_state.approval_state = {}
            st.session_state.final_pdf_path = None
            st.session_state.manual_rects = defaultdict(list)
            st.session_state.active_page_index = 0
            
            with st.spinner("Analysing document with your instructions..."):
                suggestions = analyse_document_for_redactions(input_pdf_path, st.session_state.user_context)
                st.session_state.suggestions = suggestions
                st.session_state.processed_file = input_pdf_path
                st.session_state.approval_state = {s['id']: True for s in suggestions}
                st.session_state.original_pdf_images = get_original_pdf_images(input_pdf_path)
            
            if suggestions:
                st.success(f"Analysis complete! Found {len(suggestions)} total instances to review.")
            else:
                st.warning("Analysis complete, but no sensitive information was found.")

    if st.session_state.suggestions:
        st.header("Review and Refine Redactions")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("AI Suggestions")
            st.write("Control all instances of a term or expand to manage each one individually.")
            grouped_suggestions = defaultdict(list)
            for s in st.session_state.suggestions:
                grouped_suggestions[s['text']].append(s)
            for text, instances in grouped_suggestions.items():
                all_ids = [inst['id'] for inst in instances]
                all_checked = all(st.session_state.approval_state.get(id, False) for id in all_ids)
                master_state = st.checkbox(
                    f"**{text}** ({instances[0]['category']}) - {len(instances)} instance(s)",
                    value=all_checked, key=f"master_{text}"
                )
                if master_state != all_checked:
                    for id in all_ids:
                        st.session_state.approval_state[id] = master_state
                    st.rerun()
                with st.expander("Show individual occurrences"):
                    for inst in instances:
                        context = inst['context']
                        start = max(0, context.find(text) - 30)
                        end = min(len(context), start + len(text) + 60)
                        label = f"Pg {inst['page_num'] + 1}: ...{context[start:end]}..."
                        st.session_state.approval_state[inst['id']] = st.checkbox(
                            label, value=st.session_state.approval_state.get(inst['id'], True), key=f"cb_{inst['id']}"
                        )

        with col2:
            st.subheader("Interactive Document Preview")
            if st.session_state.original_pdf_images:

                # Navigation Logic start
                total_pages = len(st.session_state.original_pdf_images)
                nav_cols = st.columns([1, 1, 6, 1, 1]) # Create columns for layout

                # Previous Page Button
                with nav_cols[0]:
                    if st.button("⬅️", use_container_width=True, disabled=(st.session_state.active_page_index == 0)):
                        st.session_state.active_page_index -= 1
                        st.rerun()

                with nav_cols[1]:
                    if st.button("Prev", use_container_width=True, disabled=(st.session_state.active_page_index == 0)):
                        st.session_state.active_page_index -= 1
                        st.rerun()
                
                # Page Counter Display
                with nav_cols[2]:
                    st.markdown(f"<p style='text-align: center; font-weight: bold;'>Page {st.session_state.active_page_index + 1} of {total_pages}</p>", unsafe_allow_html=True)
                
                # Next Page Button
                with nav_cols[3]:
                    if st.button("Next", use_container_width=True, disabled=(st.session_state.active_page_index >= total_pages - 1)):
                        st.session_state.active_page_index += 1
                        st.rerun()

                with nav_cols[4]:
                    if st.button("➡️", use_container_width=True, disabled=(st.session_state.active_page_index >= total_pages - 1)):
                        st.session_state.active_page_index += 1
                        st.rerun()

                page_index = st.session_state.active_page_index
                # --- END OF NAVIGATION LOGIC ---

                # Toggle between draw and edit mode
                mode_toggle = st.checkbox("Enable Edit/Delete Mode", value=(st.session_state.drawing_mode == "transform"))
                st.session_state.drawing_mode = "transform" if mode_toggle else "rect"

                # DYNAMIC BACKGROUND & SCALING LOGIC
                original_image = st.session_state.original_pdf_images[page_index]
                
                # Calculate scaling factor for display
                display_scaling_factor = CANVAS_DISPLAY_WIDTH / original_image.width
                display_height = int(original_image.height * display_scaling_factor)
                
                # Create the resized image that will be displayed
                display_image = original_image.resize((CANVAS_DISPLAY_WIDTH, display_height))
                
                # Draw existing redactions onto this display image
                draw = ImageDraw.Draw(display_image)

                # Draw approved AI suggestions, scaled to the display size
                approved_ai_suggestions = [
                    s for s in st.session_state.suggestions 
                    if st.session_state.approval_state.get(s['id']) and s['page_num'] == page_index
                ]
                dpi_to_display_scaling = (PREVIEW_DPI / 72.0) * display_scaling_factor
                for suggestion in approved_ai_suggestions:
                    for rect in suggestion.get('rects', []):
                        scaled_rect = (
                            rect.x0 * dpi_to_display_scaling, rect.y0 * dpi_to_display_scaling,
                            rect.x1 * dpi_to_display_scaling, rect.y1 * dpi_to_display_scaling
                        )
                        draw.rectangle(scaled_rect, fill="black")

                # Pre-draw existing manual redactions onto the background
                for obj in st.session_state.manual_rects.get(page_index, []):
                    # Manual rects are stored relative to the canvas, so we need to scale them for display
                    x1, y1 = obj["left"], obj["top"]
                    x2, y2 = x1 + obj["width"], y1 + obj["height"]
                    draw.rectangle((x1, y1, x2, y2), fill="black")
                
                st.info("In 'Draw' mode, create new redactions. In 'Edit/Delete' mode, you can move, resize, or double-click to delete shapes.")
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1.0)",
                    stroke_width=0,
                    background_image=display_image,
                    update_streamlit=True,
                    height=display_height,
                    width=CANVAS_DISPLAY_WIDTH,
                    drawing_mode=st.session_state.drawing_mode,
                    # Load initial drawing from state to persist edits
                    initial_drawing={"objects": st.session_state.manual_rects.get(page_index, [])},
                    key=f"canvas_{page_index}",
                )

                if canvas_result.json_data is not None:
                    # Only update state if the drawn objects have actually changed
                    if canvas_result.json_data.get("objects") != st.session_state.manual_rects.get(page_index, []):
                        st.session_state.manual_rects[page_index] = canvas_result.json_data.get("objects", [])
                        st.rerun() # Explicitly rerun to refresh the background with the new manual shape

        st.divider()
        st.header("Generate Final Document")
        if st.button("Generate Redacted PDF"):
            with st.spinner("Applying all redactions..."):
                approved_areas_by_page = defaultdict(list)

                # Add AI-approved redactions (these are already in PDF point coordinates)
                approved_suggestions = [s for s in st.session_state.suggestions if st.session_state.approval_state.get(s['id'])]
                for s in approved_suggestions:
                    approved_areas_by_page[s['page_num']].extend(s.get('rects', []))

                # Add manually drawn redactions, scaling them correctly back to PDF coordinates
                for page_num, canvas_objects in st.session_state.manual_rects.items():
                    original_img_width = st.session_state.original_pdf_images[page_num].width
                    # This factor converts from the displayed canvas coordinates back to PDF points
                    final_scaling_factor = (72.0 / PREVIEW_DPI) * (original_img_width / CANVAS_DISPLAY_WIDTH)
                    
                    for obj in canvas_objects:
                        x1, y1 = obj["left"], obj["top"]
                        x2, y2 = x1 + obj["width"], y1 + obj["height"]
                        pdf_rect = fitz.Rect(
                            x1 * final_scaling_factor, y1 * final_scaling_factor,
                            x2 * final_scaling_factor, y2 * final_scaling_factor
                        )
                        approved_areas_by_page[page_num].append(pdf_rect)
                
                if not approved_areas_by_page:
                    st.warning("No redactions were selected or drawn.")
                else:
                    final_redaction_areas = list(approved_areas_by_page.items())
                    output_filename = os.path.splitext(os.path.basename(st.session_state.processed_file))[0] + "_redacted.pdf"
                    output_pdf_path = os.path.join("temp_docs", output_filename)
                    
                    processor = PDFProcessor(st.session_state.processed_file)
                    processor.apply_redactions(final_redaction_areas, output_pdf_path)
                    
                    st.session_state.final_pdf_path = output_pdf_path
                    st.success(f"Successfully created redacted document: {output_filename}")

    if st.session_state.final_pdf_path and os.path.exists(st.session_state.final_pdf_path):
        with open(st.session_state.final_pdf_path, "rb") as f:
            st.download_button(
                "Download Redacted PDF", 
                f, 
                file_name=os.path.basename(st.session_state.final_pdf_path), 
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()  