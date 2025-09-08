from typing import List, Tuple
import fitz  # PyMuPDF

class PDFProcessor:
    """
    Handles PDF manipulation tasks like applying redactions.
    """
    def __init__(self, file_path: str):
        """
        Initializes the processor with the path to the PDF file.
        """
        self.file_path = file_path

    def apply_redactions(self, redaction_areas: List[Tuple[int, List[fitz.Rect]]], output_path: str):
        """
        Applies redactions to the PDF document and saves the output.

        Args:
            redaction_areas: A list of tuples, where each tuple contains a page number
                             and a list of fitz.Rect objects for that page.
            output_path: The path to save the redacted PDF.
        """
        print(f"Opening PDF: {self.file_path}")
        doc = fitz.open(self.file_path)

        if not redaction_areas:
            print("No redaction areas provided. Saving a copy of the original document.")
            doc.save(output_path)
            doc.close()
            return

        print("--- REDACTION DIAGNOSTICS ---")
        for page_num, rects in redaction_areas:
            if page_num < len(doc):
                page = doc[page_num]
                
                # --- DIAGNOSTIC PRINT ---
                # Print the dimensions of the page in points (width, height)
                print(f"\nProcessing Page {page_num + 1} (index {page_num})")
                print(f"  - Page Dimensions (w, h): ({page.rect.width}, {page.rect.height})")
                
                for i, rect in enumerate(rects):
                    # --- DIAGNOSTIC PRINT ---
                    # Print the coordinates of the rectangle we are about to redact
                    print(f"  - Applying Redaction #{i+1} at Rect: (x0={rect.x0:.2f}, y0={rect.y0:.2f}, x1={rect.x1:.2f}, y1={rect.y1:.2f})")

                    # Add the redaction annotation (the "sticker")
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                
                # Apply the redactions for THIS PAGE specifically.
                # The images=2 argument explicitly tells it to black out image content under the rect.
                page.apply_redactions(images=2) 
            else:
                # --- DIAGNOSTIC PRINT ---
                print(f"\n[WARNING] Attempted to redact on non-existent page index {page_num}.")

        print("\n--- END DIAGNOSTICS ---")

        doc.save(output_path)
        doc.close()
        print(f"Redacted PDF saved to: {output_path}")