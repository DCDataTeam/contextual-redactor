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
    """Orchestrates the hybrid AI analysis with conditional entity linking and contextual DOB filtering."""
    load_dotenv()
    azure_client = AzureAIClient()
    
    # Parse user instructions 
    print("Step 1: Parsing user instructions...")
    parsed_instructions = azure_client.parse_user_instructions(user_context)
    pii_exceptions = [exc.lower() for exc in parsed_instructions.get("exceptions", [])]
    sensitive_content_rules = parsed_instructions.get("sensitive_content_rules")
    print(f"Found {len(pii_exceptions)} PII exceptions and a sensitive content rule: {'Yes' if sensitive_content_rules else 'No'}")

    # Analyse and merge 
    print("Step 2: Analysing document layout...")
    analysis_result = azure_client.analyse_document(input_pdf_path)
    if not analysis_result.paragraphs: return []
    print("Step 3: Merging small paragraphs...")
    paragraphs = merge_small_paragraphs(analysis_result.paragraphs)

    all_findings_with_source = []
    print(f"Step 4: Running hybrid analysis on {len(paragraphs)} text chunks...")

    # Keywords to identify a DateTime as a DateOfBirth
    DOB_KEYWORDS = ["dob", "d.o.b", "date of birth", "born"]

    for i, target_paragraph in enumerate(paragraphs):
        
        # Get all potential PII entities
        all_potential_entities = azure_client.get_pii(target_paragraph.content)

        if not all_potential_entities:
            continue

        # Pre-filtering of entities to identify DOBs and Schools
        validated_entities = []
        for entity in all_potential_entities:
            is_sensitive = False
            final_category = entity['category']

            if entity['category'] == 'DateTime':
                context_substring = target_paragraph.content[max(0, entity['offset'] - 20) : entity['offset']]
                if any(keyword in context_substring.lower() for keyword in DOB_KEYWORDS):
                    is_sensitive = True
                    final_category = 'DateOfBirth'
            elif entity['category'] == 'Organization':
                context_sentence = target_paragraph.content[max(0, entity['offset']-100):entity['offset']+entity['length']+100]
                if azure_client.is_school(entity['text'], context_sentence):
                    is_sensitive = True
                    final_category = 'School'
            elif entity['category'] == 'Age':
                is_sensitive = True
                final_category = 'Age' # Override the default "Quantity" with "Age"
            
            else: # Person, Address, Age, etc. are always considered sensitive 
                is_sensitive = True

            if is_sensitive:
                entity['final_category'] = final_category # Add the validated category to the dict
                validated_entities.append(entity)

        if not validated_entities:
            continue

        # Entity linking logic
        validated_categories = {ent['category'] for ent in validated_entities}
        has_person = "Person" in validated_categories
        has_other_pii = len(validated_categories - {"Person"}) > 0
        entity_linking_needed = has_person and has_other_pii and pii_exceptions

        entity_link_map = {}
        if entity_linking_needed:
            print(f"  - Chunk {i+1} is complex. Performing entity linking on validated entities...")
            prev_content = paragraphs[i-1].content if i > 0 else ""
            next_content = paragraphs[i+1].content if i < len(paragraphs) - 1 else ""
            context_block = f"{prev_content}\n\n---TARGET TEXT---\n{target_paragraph.content}\n\n---NEXT TEXT---\n{next_content}"
            entity_link_map = azure_client.link_entities_to_person(context_block, validated_entities)

        # Final filtering for user exceptions on the validated list
        for entity in validated_entities:   
            is_excepted = False
            owner_name = None

            if entity_linking_needed:
                owner_name = entity_link_map.get(entity['text'])
                if owner_name and owner_name.lower() in pii_exceptions:
                    is_excepted = True
            else:
                if entity['text'].lower() in pii_exceptions:
                    is_excepted = True

            if not is_excepted:
                all_findings_with_source.append({
                    'llm_finding': {'text': entity['text'], 'category': entity['final_category'], 'reasoning': f"Identified as sensitive PII ({entity['final_category']})."},
                    'source_paragraph': target_paragraph
                })

        # Nuanced LLM analysis for sensitive content
        if sensitive_content_rules:
            prev_content = paragraphs[i-1].content if i > 0 else ""
            next_content = paragraphs[i+1].content if i < len(paragraphs) - 1 else ""
            context_block = f"{prev_content}\n\n---TARGET TEXT---\n{target_paragraph.content}\n\n---NEXT TEXT---\n{next_content}"
            sensitive_findings = azure_client.get_sensitive_information(context_block, sensitive_content_rules)
            for finding in sensitive_findings:
                if finding['text'] in target_paragraph.content:
                    all_findings_with_source.append({
                        'llm_finding': finding, 'source_paragraph': target_paragraph
                    })

    if not all_findings_with_source: return []
    print(f"Found a total of {len(all_findings_with_source)} potential redactions.")

    # Map all combined findings to coordinates.
    print("Step 5: Creating detailed suggestions...")
    detailed_suggestions = create_detailed_suggestions(analysis_result, all_findings_with_source)
    
    return detailed_suggestions

