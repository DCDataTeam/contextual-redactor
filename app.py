
import streamlit as st
import fitz
import os
import debugpy
import sys
from collections import defaultdict
from PIL import Image, ImageDraw
from io import BytesIO

from redaction_logic import analyse_document_for_redactions
from pdf_processor import PDFProcessor



# if not hasattr(sys, '_debugpy_attached'):
#     import debugpy
#     debugpy.listen(("127.0.0.1", 5678))
#     # Uncomment the next line to make Streamlit wait for the debugger to attach before continuing
#     debugpy.wait_for_client()
#     sys._debugpy_attached = True


# --- App Configuration and State Management  ---
st.set_page_config(page_title="AI Document Redactor", layout="wide")
if 'suggestions' not in st.session_state: st.session_state.suggestions = []
if 'processed_file' not in st.session_state: st.session_state.processed_file = None
if 'final_pdf_path' not in st.session_state: st.session_state.final_pdf_path = None
if 'approval_state' not in st.session_state: st.session_state.approval_state = {}
if 'original_pdf_images' not in st.session_state: st.session_state.original_pdf_images = []
if 'user_context' not in st.session_state: st.session_state.user_context = ""

# --- Helper Functions ---
PREVIEW_DPI = 150
def get_original_pdf_images(pdf_path):
    if not os.path.exists(pdf_path): return []
    doc = fitz.open(pdf_path)
    images = [page.get_pixmap(dpi=PREVIEW_DPI).tobytes("png") for page in doc]
    doc.close()
    return images

def generate_preview_images(original_images, all_suggestions, approval_state):
    preview_images = []
    approved_suggestions = [s for s in all_suggestions if approval_state.get(s['text'], False)]
    suggestions_by_page = defaultdict(list)
    for s in approved_suggestions: suggestions_by_page[s['page_num']].append(s)
    for i, img_bytes in enumerate(original_images):
        if i in suggestions_by_page:
            img = Image.open(BytesIO(img_bytes)).convert("RGBA")
            draw = ImageDraw.Draw(img)
            scaling_factor = PREVIEW_DPI / 72.0
            
            for suggestion in suggestions_by_page[i]:
                # The suggestion dictionary now contains a LIST of rectangles under the key 'rects'.
                # We need to loop through this list.
                if 'rects' in suggestion:
                    for rect in suggestion['rects']:
                        scaled_rect = (
                            rect.x0 * scaling_factor,
                            rect.y0 * scaling_factor,
                            rect.x1 * scaling_factor,
                            rect.y1 * scaling_factor
                        )
                        draw.rectangle(scaled_rect, fill="black")
            
            byte_io = BytesIO()
            img.save(byte_io, format='PNG')
            preview_images.append(byte_io.getvalue())
        else:
            preview_images.append(img_bytes)
    return preview_images

def main():
    st.title("AI-Powered Document Redaction Tool")
    st.write("Upload a PDF to identify personal data. Provide specific instructions to guide the AI for this document.")

    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    st.text_area(
        "Provide specific redaction instructions for this document (optional):",
        placeholder="Example: This report is about child A. Remove all mentions of their sibling, child B, including their name, school, and age.",
        height=100,
        key='user_context' # Bind the text area to session state
    )

    if uploaded_file is not None:
        input_pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_pdf_path, "wb") as f: f.write(uploaded_file.getbuffer())

        if st.button("Analyse Document"):
            st.session_state.suggestions, st.session_state.approval_state, st.session_state.final_pdf_path = [], {}, None
            
            with st.spinner("Processing document with your instructions..."):
                user_instructions = st.session_state.user_context
                suggestions = analyse_document_for_redactions(input_pdf_path, user_instructions)
                
                st.session_state.suggestions = suggestions
                st.session_state.processed_file = input_pdf_path
                unique_texts = {s['text'] for s in suggestions}
                st.session_state.approval_state = {text: True for text in unique_texts}
                st.session_state.original_pdf_images = get_original_pdf_images(input_pdf_path)
            
            if suggestions: st.success(f"Analysis complete! Found {len(st.session_state.approval_state)} unique items to review.")
            else: st.warning("Analysis complete, but no sensitive information was found.")

    if st.session_state.suggestions:
        st.header("Review Redaction Suggestions")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Redaction Checklist")
            st.write("Uncheck items to keep. This applies to all instances.")
            grouped_suggestions = defaultdict(list)
            for s in st.session_state.suggestions: grouped_suggestions[s['text']].append(s)
            for text, instances in grouped_suggestions.items():
                first_instance = instances[0]
                count = len(instances)
                label = f"**{text}** ({first_instance['category']}) - Found {count} time{'s' if count > 1 else ''}"
                st.session_state.approval_state[text] = st.checkbox(label, value=st.session_state.approval_state.get(text, True), key=f"cb_{text}")
                with st.expander("Show contexts"):
                    for i, instance in enumerate(instances):
                        context = instance['context']
                        start = max(0, context.find(text) - 40)
                        end = min(len(context), context.find(text) + len(text) + 40)
                        st.markdown(f"**Instance {i+1} on Page {instance['page_num'] + 1}:**")
                        st.info(f"...{context[start:end]}...")
        with col2:
            st.subheader("Document Preview")
            if st.session_state.original_pdf_images:
                preview_images = generate_preview_images(st.session_state.original_pdf_images, st.session_state.suggestions, st.session_state.approval_state)
                for i, img_bytes in enumerate(preview_images):
                    st.image(img_bytes, caption=f"Page {i+1}", width='stretch')
        st.divider()
        st.header("Generate Final Document")
        if st.button("Generate Redacted PDF"):
            approved_suggestions = [s for s in st.session_state.suggestions if st.session_state.approval_state.get(s['text'])]
            if not approved_suggestions: st.warning("No redactions were selected.")
            approved_areas_by_page = defaultdict(list)
            for s in approved_suggestions:
                if 'rects' in s:
                    approved_areas_by_page[s['page_num']].extend(s['rects'])
            final_redaction_areas = list(approved_areas_by_page.items())
            output_filename = os.path.splitext(os.path.basename(st.session_state.processed_file))[0] + "_redacted.pdf"
            output_pdf_path = os.path.join("temp_docs", output_filename)
            processor = PDFProcessor(st.session_state.processed_file)
            processor.apply_redactions(final_redaction_areas, output_pdf_path)
            st.session_state.final_pdf_path = output_pdf_path
            st.success(f"Successfully created redacted document: {output_filename}")
    if st.session_state.final_pdf_path and os.path.exists(st.session_state.final_pdf_path):
        with open(st.session_state.final_pdf_path, "rb") as f:
            st.download_button("Download Redacted PDF", f, file_name=os.path.basename(st.session_state.final_pdf_path), mime="application/pdf")


if __name__ == "__main__":
    # If running directly (e.g., with VS Code debugger), just call main()
    main()
 