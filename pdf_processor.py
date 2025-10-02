from typing import List, Tuple, Dict
import fitz  # PyMuPDF

class PDFProcessor:
    """
    Enhanced PDF processor with better error handling and multiple processing modes.
    """
    
    def __init__(self, file_path: str = None):
        """
        Initializes the processor with optional path to the PDF file.
        If no path provided, use apply_rect_redactions static method.
        """
        self.file_path = file_path

    def apply_redactions(self, redaction_areas: List[Tuple[int, List[fitz.Rect]]], output_path: str):
        """
        Applies redactions to the PDF document and saves the output.
        Legacy method for backward compatibility.

        Args:
            redaction_areas: A list of tuples, where each tuple contains a page number
                             and a list of fitz.Rect objects for that page.
            output_path: The path to save the redacted PDF.
        """
        if not self.file_path:
            raise ValueError("No file path provided to PDFProcessor constructor")
            
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
                
                print(f"\nProcessing Page {page_num + 1} (index {page_num})")
                print(f"  - Page Dimensions (w, h): ({page.rect.width}, {page.rect.height})")
                
                for i, rect in enumerate(rects):
                    print(f"  - Applying Redaction #{i+1} at Rect: (x0={rect.x0:.2f}, y0={rect.y0:.2f}, x1={rect.x1:.2f}, y1={rect.y1:.2f})")
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                
                page.apply_redactions(images=2) 
            else:
                print(f"\n[WARNING] Attempted to redact on non-existent page index {page_num}.")

        print("\n--- END DIAGNOSTICS ---")

        doc.save(
            output_path,
            garbage=4,
            deflate=True,
            clean=True
        )
        doc.close()
        print(f"Redacted PDF saved to: {output_path}")

    @staticmethod
    def apply_rect_redactions(input_path: str, redaction_dict: Dict[int, List[Dict]], output_path: str):
        """
        Enhanced static method for applying redactions from dictionary format.
        Used by the enhanced UI.

        Args:
            input_path: Path to the input PDF file
            redaction_dict: Dictionary mapping page numbers to lists of redaction rectangles
                           Each rectangle is a dict with keys: x, y, w, h
            output_path: Path to save the redacted PDF
        """
        print(f"🔓 Opening PDF: {input_path}")
        
        try:
            doc = fitz.open(input_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF file: {e}")

        if not redaction_dict:
            print("ℹ️ No redactions to apply. Saving a copy of the original document.")
            doc.save(output_path)
            doc.close()
            return

        total_redactions = sum(len(rects) for rects in redaction_dict.values())
        print(f"📊 Applying {total_redactions} redactions across {len(redaction_dict)} pages")
        print("=" * 60)

        processed_pages = 0
        
        for page_num in sorted(redaction_dict.keys()):
            redaction_rects = redaction_dict[page_num]
            
            if page_num >= len(doc):
                print(f"⚠️ WARNING: Page {page_num + 1} does not exist in document (max: {len(doc)})")
                continue
                
            if not redaction_rects:
                continue

            page = doc[page_num]
            processed_pages += 1
            
            print(f"\n📄 Processing Page {page_num + 1}")
            print(f"   📐 Page size: {page.rect.width:.1f} × {page.rect.height:.1f} pts")
            print(f"   🎯 Redactions: {len(redaction_rects)}")
            
            # Convert dictionary rectangles to fitz.Rect objects and apply
            applied_count = 0
            for i, rect_dict in enumerate(redaction_rects):
                try:
                    x = rect_dict.get('x', 0)
                    y = rect_dict.get('y', 0)
                    w = rect_dict.get('w', 0)
                    h = rect_dict.get('h', 0)
                    
                    # Create fitz.Rect from x, y, width, height
                    rect = fitz.Rect(x, y, x + w, y + h)
                    
                    # Validate rectangle bounds
                    if rect.is_empty or rect.is_infinite:
                        print(f"   ⚠️ Skipping invalid rectangle {i+1}: {rect}")
                        continue
                        
                    # Clip to page bounds to prevent errors
                    rect = rect & page.rect
                    
                    if not rect.is_empty:
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                        applied_count += 1
                        print(f"   ✅ Redaction {i+1}: ({x:.1f}, {y:.1f}) {w:.1f}×{h:.1f}")
                    else:
                        print(f"   ⚠️ Rectangle {i+1} is outside page bounds")
                        
                except Exception as e:
                    print(f"   ❌ Error processing rectangle {i+1}: {e}")
            
            # Apply all redactions for this page
            try:
                page.apply_redactions(images=2)
                print(f"   ✅ Applied {applied_count}/{len(redaction_rects)} redactions to page {page_num + 1}")
            except Exception as e:
                print(f"   ❌ Error applying redactions to page {page_num + 1}: {e}")

        print("=" * 60)
        print(f"📋 Summary: Processed {processed_pages} pages")

        # Save the document
        try:
            print(f"💾 Saving redacted document to: {output_path}")
            doc.save(
                output_path,
                garbage=4,       # Full garbage collection
                deflate=True,    # Compress
                clean=True       # Remove unused objects
            )
            print("✅ Document saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving document: {e}")
            raise
        finally:
            doc.close()

    @staticmethod
    def validate_pdf(file_path: str) -> Dict[str, any]:
        """
        Validates a PDF file and returns information about it.
        
        Returns:
            Dictionary with validation results and file info
        """
        try:
            doc = fitz.open(file_path)
            
            info = {
                'valid': True,
                'page_count': len(doc),
                'encrypted': doc.is_encrypted,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'creator': doc.metadata.get('creator', ''),
                'file_size': 0,
                'errors': []
            }
            
            # Check if pages can be accessed
            for i, page in enumerate(doc):
                try:
                    # Try to get page rect to verify page integrity
                    _ = page.rect
                except Exception as e:
                    info['errors'].append(f"Page {i+1}: {str(e)}")
            
            doc.close()
            
            # Get file size
            import os
            if os.path.exists(file_path):
                info['file_size'] = os.path.getsize(file_path)
            
            return info
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'page_count': 0,
                'encrypted': None,
                'file_size': 0,
                'errors': [str(e)]
            }

    @staticmethod
    def get_page_info(file_path: str, page_num: int) -> Dict[str, any]:
        """
        Gets detailed information about a specific page.
        
        Args:
            file_path: Path to PDF file
            page_num: 0-based page number
            
        Returns:
            Dictionary with page information
        """
        try:
            doc = fitz.open(file_path)
            
            if page_num >= len(doc) or page_num < 0:
                return {'valid': False, 'error': 'Page number out of range'}
            
            page = doc[page_num]
            
            info = {
                'valid': True,
                'page_number': page_num,
                'width': page.rect.width,
                'height': page.rect.height,
                'rotation': page.rotation,
                'word_count': len(page.get_text_words()),
                'image_count': len(page.get_images()),
                'annotation_count': len(page.annots()),
                'has_text': bool(page.get_text().strip())
            }
            
            doc.close()
            return info
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    @staticmethod
    def create_preview_image(file_path: str, page_num: int, dpi: int = 150) -> bytes:
        """
        Creates a preview image of a specific page.
        
        Args:
            file_path: Path to PDF file
            page_num: 0-based page number
            dpi: Resolution for the image
            
        Returns:
            PNG image data as bytes
        """
        try:
            doc = fitz.open(file_path)
            page = doc[page_num]
            
            # Create pixmap with specified DPI
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PNG bytes
            img_data = pix.tobytes("png")
            
            doc.close()
            return img_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to create preview image: {e}")