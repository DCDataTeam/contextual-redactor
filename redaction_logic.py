import os
from typing import List, Tuple
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
    Orchestrates the hybrid AI analysis with conditional entity linking and contextual DOB filtering.
    Enhanced with batch processing for improved performance.
    - Paragraph-by-paragraph for structured PII.
    - Page-by-page for subjective, context-aware content.
    """
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

    # Collect organization entities for batch processing
    organization_batch = []
    paragraph_entity_map = {}  # Track which entities belong to which paragraph

    for i, target_paragraph in enumerate(paragraphs):
        # Get all potential PII entities
        all_potential_entities = azure_client.get_pii(target_paragraph.content)

        if not all_potential_entities:   
            continue

        # Separate organizations for batch processing and handle other entities
        validated_entities = []
        organizations_for_batch = []
        
        for entity in all_potential_entities:
            is_sensitive = False
            final_category = entity['category']

            if entity['category'] == 'DateTime':
                context_substring = target_paragraph.content[max(0, entity['offset'] - 20) : entity['offset']]
                if any(keyword in context_substring.lower() for keyword in DOB_KEYWORDS):
                    is_sensitive = True
                    final_category = 'DateOfBirth'
            elif entity['category'] == 'Organization':
                # Add to batch processing list
                context_sentence = target_paragraph.content[max(0, entity['offset']-100):entity['offset']+entity['length']+100]
                organizations_for_batch.append((entity['text'], context_sentence))
                organization_batch.append((i, len(organizations_for_batch) - 1, entity, final_category))  # paragraph_index, batch_index, entity, category
                continue  # Skip individual processing for now
            elif entity['category'] == 'Age':
                is_sensitive = True
                final_category = 'Age' # Override the default "Quantity" with "Age"
            else: # Person, Address, Age, etc.
                # Pre-validated by Azure Language Service - are always considered sensitive 
                is_sensitive = True

            if is_sensitive:
                entity['final_category'] = final_category # Add the validated category to the dict
                validated_entities.append(entity)
        
        paragraph_entity_map[i] = validated_entities

    # Batch process all organizations at once
    print(f"  - Batch processing {len(organization_batch)} organizations...")
    if organization_batch:
        # Extract organization data for batch processing
        org_contexts = []
        for para_idx, batch_idx, entity, category in organization_batch:
            # Find the original organization and context from the batch we built
            target_paragraph = paragraphs[para_idx]
            context_sentence = target_paragraph.content[max(0, entity['offset']-100):entity['offset']+entity['length']+100]
            org_contexts.append((entity['text'], context_sentence))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_org_contexts = []
        for org_name, context in org_contexts:
            key = (org_name, context)
            if key not in seen:
                seen.add(key)
                unique_org_contexts.append((org_name, context))
        
        if unique_org_contexts:
            school_results = azure_client.classify_organizations_batch(unique_org_contexts)
            
            # Map results back to original entities
            result_map = {(org, ctx): result for (org, ctx), result in zip(unique_org_contexts, school_results)}
            
            for para_idx, batch_idx, entity, category in organization_batch:
                # Find the corresponding result
                target_paragraph = paragraphs[para_idx]
                context_sentence = target_paragraph.content[max(0, entity['offset']-100):entity['offset']+entity['length']+100]
                is_school = result_map.get((entity['text'], context_sentence), False)
                
                if is_school:
                    entity['final_category'] = 'School'
                    if para_idx not in paragraph_entity_map:
                        paragraph_entity_map[para_idx] = []
                    paragraph_entity_map[para_idx].append(entity)

    # Process each paragraph's validated entities
    for i, target_paragraph in enumerate(paragraphs):
        validated_entities = paragraph_entity_map.get(i, [])
        
        if validated_entities:
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
                        'llm_finding': {
                            'text': entity['text'], 
                            'category': entity['final_category'], 
                            'reasoning': f"Identified as sensitive PII ({entity['final_category']})."
                        },
                        'source_paragraph': target_paragraph
                    })

    # Nuanced LLM analysis for sensitive content (unchanged - this requires complex reasoning)
    if sensitive_content_rules:
        print("\nTask B: Processing sensitive content page-by-page...")
        for page in analysis_result.pages:
            page_content = analysis_result.content[page.spans[0].offset : page.spans[0].offset + page.spans[0].length]
            print(f"  - Analyzing Page {page.page_number} for sensitive content...")

            sensitive_findings = azure_client.get_sensitive_information(
                text_chunk=page_content,
                user_context=sensitive_content_rules
            )

            for finding in sensitive_findings:
                if finding['text'].lower() not in pii_exceptions:
                    all_findings_with_source.append({
                        'llm_finding': finding, 
                        'source_page': page
                    })

    if not all_findings_with_source: return []
    print(f"Found a total of {len(all_findings_with_source)} potential redactions.")

    # Map all combined findings to coordinates.
    print("Step 5: Creating detailed suggestions...")
    detailed_suggestions = create_detailed_suggestions(analysis_result, all_findings_with_source)
    
    return detailed_suggestions