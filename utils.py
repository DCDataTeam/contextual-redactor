import logging
from typing import List, Dict
from collections import defaultdict, Counter
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentParagraph
from fuzzywuzzy import fuzz
import fitz

# --- Logger Setup ---
def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()

# --- Rectangle Merging Logic ---
def merge_consecutive_word_rects(word_rects: List[fitz.Rect]) -> List[fitz.Rect]:
    if not word_rects:
        return []
    lines = defaultdict(list)
    for rect in word_rects:
        lines[round(rect.y0)].append(rect)
    final_merged_rects = []
    for line_y in sorted(lines.keys()):
        sorted_rects = sorted(lines[line_y], key=lambda r: r.x0)
        if not sorted_rects:
            continue
        current_run_group = [sorted_rects[0]]
        for i in range(1, len(sorted_rects)):
            prev_rect = current_run_group[-1]
            current_rect = sorted_rects[i]
            max_gap = prev_rect.height * 0.75 
            actual_gap = current_rect.x0 - prev_rect.x1
            if actual_gap <= max_gap:
                current_run_group.append(current_rect)
            else:
                merged_run = fitz.Rect()
                for r in current_run_group:
                    merged_run |= r
                final_merged_rects.append(merged_run)
                current_run_group = [current_rect]
        if current_run_group:
            merged_run = fitz.Rect()
            for r in current_run_group:
                merged_run |= r
            final_merged_rects.append(merged_run)
    return final_merged_rects

# --- Mapping LLM Findings to Document Coordinates ---
def create_detailed_suggestions(
    analysis: AnalyzeResult, 
    all_findings_with_source: List[Dict]
) -> List[Dict]:
    """
    Maps sensitive text from an LLM to its specific coordinates, handling sources that are either
    a specific paragraph (for PII) or a whole page (for sensitive content).
    """
    detailed_suggestions = []
    scaling_factor = 72.0
    
    # --- Create a map of all words on each page, marked as "unused" ---
    words_by_page = defaultdict(list)
    for page in analysis.pages:
        for word in page.words:
            words_by_page[page.page_number - 1].append({'word_obj': word, 'used': False})

    suggestion_id_counter = 0

    # We sort all findings by length to prioritise longer matches across the board
    sorted_findings = sorted(all_findings_with_source, key=lambda x: len(x['llm_finding']['text']), reverse=True)
        
    for item in sorted_findings:
        llm_finding = item['llm_finding']
        text_to_find = llm_finding['text']
        source_paragraph = item.get('source_paragraph')
        source_page = item.get('source_page')
        words_to_search = []
        context = ""
        page_num = -1

        if source_paragraph:
            page_num = source_paragraph.bounding_regions[0].page_number - 1
                    
            para_span = source_paragraph.spans[0]
            words_to_search = [
                w_dict for w_dict in words_by_page.get(page_num, [])
                if w_dict['word_obj'].span.offset >= para_span.offset and
                   (w_dict['word_obj'].span.offset + w_dict['word_obj'].span.length) <= (para_span.offset + para_span.length)
            ]
            context = source_paragraph.content

        elif source_page:
            page_num = source_page.page_number - 1
            words_to_search = words_by_page.get(page_num, [])
            context = analysis.content[source_page.spans[0].offset : source_page.spans[0].offset + source_page.spans[0].length]

        else:
            continue  # Skip if neither source_paragraph nor source_page is available

        # Normalization for matching
        norm_text_to_find = text_to_find.lower().replace("’s", "").replace("'s", "").replace("'", "").replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        
        best_match_words_info = []
        best_match_score = 0

        for i in range(len(words_to_search)):
            for j in range(i, len(words_to_search)):
                candidate_word_info = words_to_search[i:j+1]
                
                # Skip if any word in this candidate sequence is already used
                if any(w['used'] for w in candidate_word_info):
                    continue

                candidate_words = [w['word_obj'] for w in candidate_word_info]
                reconstructed_text = "".join([w.content for w in candidate_words])
                norm_reconstructed = reconstructed_text.lower().replace("’s", "").replace("'s", "").replace("'", "").replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(" ", "").replace('"', '').replace('“', '').replace('”', '')
                
                score = fuzz.ratio(norm_reconstructed, norm_text_to_find)
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_words_info = candidate_word_info
                
                if best_match_score == 100: break
            if best_match_score == 100: break
        
        if best_match_score >= 90 and best_match_words_info:
            # Mark these words as "used" so they can't be matched again
            for w_info in best_match_words_info:
                w_info['used'] = True

            best_match_words = [w_info['word_obj'] for w_info in best_match_words_info]
            
            individual_word_rects = []
            for word_obj in best_match_words:
                if word_obj.polygon and len(word_obj.polygon) >= 8:
                    points = [
                        fitz.Point(word_obj.polygon[k] * scaling_factor, word_obj.polygon[k+1] * scaling_factor) 
                        for k in range(0, len(word_obj.polygon), 2)
                    ]
                    individual_word_rects.append(fitz.Quad(points).rect)
            
            if individual_word_rects:
                merged_line_rects = merge_consecutive_word_rects(individual_word_rects)
                detailed_suggestions.append({
                    'id': suggestion_id_counter, 'text': llm_finding['text'], 'category': llm_finding['category'],
                    'reasoning': llm_finding['reasoning'], 'context': context,
                    'page_num': page_num, 'rects': merged_line_rects
                })
                suggestion_id_counter += 1
    
    logger.info(f"Successfully created {len(detailed_suggestions)} detailed suggestions from {len(all_findings_with_source)} LLM findings.")
    if len(detailed_suggestions) != len(all_findings_with_source):
        logger.warning(f"Could not find a unique physical location for {len(all_findings_with_source) - len(detailed_suggestions)} LLM findings.")
        
        # Use Counter to correctly handle duplicate text entries
        llm_text_counts = Counter(item['llm_finding']['text'] for item in all_findings_with_source)
        mapped_text_counts = Counter(s['text'] for s in detailed_suggestions)
        
        # Subtract the mapped counts from the LLM counts to find the difference
        unmapped = llm_text_counts - mapped_text_counts
        
        if unmapped:
            logger.error("--- MISMATCH REPORT: The following LLM findings could not be located in the document ---")
            for text, count in unmapped.items():
                logger.error(f"  - Text: '{text}' (LLM found {count} unmapped instance(s))")
            logger.error("--- END OF REPORT ---")

    detailed_suggestions = sorted(
        detailed_suggestions, 
        key=lambda s: (s['page_num'], s['rects'][0].y0 if s['rects'] else 0)
    )
            
    return detailed_suggestions

map_llm_results_to_coordinates = create_detailed_suggestions

