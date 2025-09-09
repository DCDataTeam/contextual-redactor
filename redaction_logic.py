
import os
from dotenv import load_dotenv
from azure_client import AzureAIClient
from utils import create_detailed_suggestions
from azure.ai.documentintelligence.models import DocumentParagraph

def merge_small_paragraphs(paragraphs: list[DocumentParagraph], min_length: int = 50) -> list[DocumentParagraph]:
    """
    Merges small paragraphs into the previous paragraph to create more
    semantically meaningful chunks for the LLM.
    """
    if not paragraphs:
        return []

    # Create a deep copy of the paragraphs to avoid modifying the original list from the analysis result
    from copy import deepcopy
    paragraphs_copy = deepcopy(paragraphs)

    merged_paragraphs = []
    for para in paragraphs_copy:
        if len(para.content) < min_length and merged_paragraphs:
            previous_para = merged_paragraphs[-1]
            new_content = previous_para.content + "\n" + para.content
            
            new_span_offset = previous_para.spans[0].offset
            new_span_length = (para.spans[0].offset + para.spans[0].length) - new_span_offset
            
            previous_para.content = new_content
            previous_para.spans[0].length = new_span_length
        else:
            merged_paragraphs.append(para)
            
    return merged_paragraphs


def analyse_document_for_redactions(input_pdf_path: str, user_context: str):
    """
    Orchestrates AI analysis using a 'context window' (previous, current, next)
    for each paragraph to balance reasoning context with matching accuracy.
    """
    load_dotenv()
    azure_client = AzureAIClient()
    
    print("Step 1: Analysing document with Azure AI Document Intelligence...")
    analysis_result = azure_client.analyze_document(input_pdf_path)

    if not analysis_result.paragraphs:
        print("No paragraphs found in the document.")
        return []
    
    print("Step 1.5: Merging small paragraphs for better context...")
    original_paragraphs = analysis_result.paragraphs
    paragraphs = merge_small_paragraphs(original_paragraphs)
    print(f"Merged {len(original_paragraphs)} paragraphs into {len(paragraphs)} more meaningful chunks.")

    all_llm_results_with_source = []
    num_paragraphs = len(paragraphs)
    print(f"Step 2: Processing {num_paragraphs} paragraphs with a context window...")

    for i, target_paragraph in enumerate(paragraphs):
        # --- BUILD THE CONTEXT WINDOW ---
        # Get the previous paragraph's content, if it exists
        prev_para_content = paragraphs[i-1].content if i > 0 else ""
        
        # Get the next paragraph's content, if it exists
        next_para_content = paragraphs[i+1].content if i < num_paragraphs - 1 else ""
        
        # Combine them into a single block for the LLM
        context_block = (
            f"PREVIOUS PARAGRAPH CONTEXT:\n{prev_para_content}\n\n"
            f"--- TARGET PARAGRAPH TO ANALYSE ---\n{target_paragraph.content}\n\n"
            f"NEXT PARAGRAPH CONTEXT:\n{next_para_content}"
        )
        
        print(f"  - Analysing paragraph {i + 1}/{num_paragraphs}...")
        llm_results_for_block = azure_client.get_sensitive_information(context_block, user_context)
        
        if llm_results_for_block:
            # Filter the results to only include findings that are ACTUALLY IN our target paragraph
            for result in llm_results_for_block:
                if result['text'] in target_paragraph.content:
                    all_llm_results_with_source.append({
                        'llm_finding': result,
                        'source_paragraph': target_paragraph, 
                        'original_index': i
                    })

    if not all_llm_results_with_source:
        print("No sensitive information found by the LLM.")
        return []
        
    print(f"Found a total of {len(all_llm_results_with_source)} potential redactions.")

    print("Step 3: Creating detailed suggestions with coordinate mapping...")
    detailed_suggestions = create_detailed_suggestions(
        analysis_result, 
        all_llm_results_with_source
    )
    
    return detailed_suggestions

