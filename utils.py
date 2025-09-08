import logging
from typing import List, Dict
from collections import defaultdict
from azure.ai.documentintelligence.models import AnalyzeResult
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

# --- Main Suggestion Creation Logic ---
def create_detailed_suggestions(
    analysis: AnalyzeResult, 
    llm_results_with_source: List[Dict]
) -> List[Dict]:
    """
    Maps sensitive text from an LLM to its specific coordinates by searching
    ONLY within the source paragraph where the text was found.
    """
    detailed_suggestions = []
    scaling_factor = 72.0
    HORIZONTAL_OFFSET_ADJUSTMENT = 1.0 
    
    # Sort the list of findings from longest text to shortest text.
    llm_results_with_source.sort(key=lambda item: len(item['llm_finding']['text']), reverse=True)

    all_words = [word for page in analysis.pages for word in page.words]

    for item_index, item in enumerate(llm_results_with_source):
        llm_finding = item['llm_finding']
        source_paragraph = item['source_paragraph']
        text_to_find = llm_finding['text']

        para_span = source_paragraph.spans[0]
        words_in_paragraph = [
            word for word in all_words
            if word.span.offset >= para_span.offset and
               (word.span.offset + word.span.length) <= (para_span.offset + para_span.length)
        ]

        norm_text_to_find = text_to_find.lower().replace('.', '').replace(',', '').replace(' ', '').replace('(', '').replace(')', '')
        
        best_match_words = []
        best_match_score = 0

        for i in range(len(words_in_paragraph)):
            for j in range(i, len(words_in_paragraph)):
                candidate_words = words_in_paragraph[i:j+1]
                reconstructed_text = "".join([w.content for w in candidate_words])
                norm_reconstructed = reconstructed_text.lower().replace('.', '').replace(',', '').replace(' ', '').replace('(', '').replace(')', '')
                
                score = fuzz.ratio(norm_reconstructed, norm_text_to_find)
                if score > best_match_score:
                    best_match_score = score
                    best_match_words = candidate_words
                if best_match_score == 100: break
            if best_match_score == 100: break
        
        if best_match_score >= 90 and best_match_words:
            individual_word_rects = []
            for word_obj in best_match_words:
                if word_obj.polygon:
                    # The polygon is a flat list of floats [x0, y0, x1, y1, ...].
                    # We iterate through it in steps of 2 to create Point objects.
                    points = [
                        fitz.Point(
                            word_obj.polygon[k] * scaling_factor,      # The x coordinate
                            word_obj.polygon[k+1] * scaling_factor   # The y coordinate
                        ) for k in range(0, len(word_obj.polygon), 2)
                    ]
                    individual_word_rects.append(fitz.Quad(points).rect)
            
            if individual_word_rects:
                merged_line_rects = merge_consecutive_word_rects(individual_word_rects)

                offset_rects = [
                    fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 2, rect.y1 + 2)
                    for rect in merged_line_rects
                ]

                detailed_suggestions.append({
                    'id': item_index,
                    'text': llm_finding['text'],
                    'category': llm_finding['category'],
                    'reasoning': llm_finding['reasoning'],
                    'original_index': item['original_index'],
                    'context': source_paragraph.content,
                    'page_num': source_paragraph.bounding_regions[0].page_number - 1,
                    'rects': merged_line_rects
                })

    detailed_suggestions.sort(key=lambda item: item['original_index'], reverse=False)
        
    logger.info(f"Successfully created {len(detailed_suggestions)} detailed suggestions from {len(llm_results_with_source)} LLM findings.")
    return detailed_suggestions

map_llm_results_to_coordinates = create_detailed_suggestions