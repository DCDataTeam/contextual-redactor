import os
import json
import re
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.textanalytics import TextAnalyticsClient, PiiEntityCategory
from openai import AzureOpenAI


class TaskComplexity(Enum):
    """Enum to define task complexity levels for model selection"""
    SIMPLE = "simple"
    COMPLEX = "complex"


class AzureAIClient:
    def __init__(self):
        try:
            doc_intel_endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
            doc_intel_key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
            openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            openai_key = os.environ["AZURE_OPENAI_KEY"]
            self.openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
            self.openai_fast_deployment = os.environ["AZURE_OPENAI_GPT35_DEPLOYMENT_NAME"]
            lang_endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
            lang_key = os.environ["AZURE_LANGUAGE_KEY"]

        except KeyError as e:
            raise RuntimeError(f"Environment variable not set: {e}") from e

        self.doc_intel_client = DocumentIntelligenceClient(
            endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key)
        )
        self.openai_client = AzureOpenAI(
            api_key=openai_key,
            api_version="2024-02-01",
            azure_endpoint=openai_endpoint
        )
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=lang_endpoint, credential=AzureKeyCredential(lang_key)
        )

        # Define complex tasks that require the advanced model
        self.complex_tasks = {
            "instruction_parsing",
            "entity_linking", 
            "sensitive_content",
            "document_classification",
            "relationship_analysis"
        }
        
    def get_appropriate_model(self, task_complexity: Union[str, TaskComplexity]) -> str:
        """Route to the most cost-effective model for the task"""
        if isinstance(task_complexity, str):
            task_complexity = TaskComplexity(task_complexity)
            
        if task_complexity == TaskComplexity.COMPLEX:
            return self.openai_deployment
        else:
            return self.openai_fast_deployment
    
    def is_complex_task(self, task_name: str) -> bool:
        """Check if a task requires the complex model"""
        return task_name in self.complex_tasks
        
    def analyse_document(self, file_path: str) -> AnalyzeResult:
        print("Analysing document with Azure AI Document Intelligence...")
        with open(file_path, "rb") as f:
            poller = self.doc_intel_client.begin_analyze_document(
                "prebuilt-layout", body=f, content_type="application/octet-stream"
            )
        result: AnalyzeResult = poller.result()
        print("Document analysis complete.")
        return result

    def parse_user_instructions(self, user_text: str) -> dict:
        """Uses an LLM to parse free-text instructions into a structured JSON object."""
        if not user_text or not user_text.strip():
            return {} # Return empty dict if there are no instructions

        system_prompt = """
        You are a configuration parser. Your task is to analyze the user's instructions for a document redaction tool and convert them into a structured JSON object.
        The JSON object should have two optional keys:
        1. "exceptions": A list of exact strings that the user wants to PREVENT from being redacted.
        2. "sensitive_content_rules": A single string describing any new, subjective content the user wants to find and redact.

        **CRITICAL RULE:** If you identify a multi-word person's name in the "exceptions" (e.g., "Oliver Hughes"), you MUST add BOTH the full name AND the first name to the exceptions list (e.g., ["Oliver Hughes", "Oliver"]). Do this only for names that look like people's names.

        If a category is not mentioned, omit its key from the JSON. Respond ONLY with the valid JSON object.

        --- EXAMPLES ---
        User Input: "keep sarah linton and oliver hughes, but also redact any mention of bullying"
        Your Output:
        {
        "exceptions": ["Sarah Linton", "Sarah", "Oliver Hughes", "Oliver"],
        "sensitive_content_rules": "Redact any mention of bullying."
        }
        ---
        User Input: "Don't remove the name Oliver Hughes"
        Your Output:
        {
        "exceptions": ["Oliver Hughes", "Oliver"]
        }
        ---
        User Input: "The company 'Hughes Construction' is fine to keep."
        Your Output:
        {
        "exceptions": ["Hughes Construction"]
        }
        ---
        User Input: "Find any quotes that are critical of the parents"
        Your Output:
        {
        "sensitive_content_rules": "Find any quotes that are critical of the parents"
        }
        """
        try:
            model = self.get_appropriate_model(TaskComplexity.COMPLEX)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error parsing user instructions: {e}")
            return {} # Return empty on failure

    def get_pii(self, text_chunk: str) -> list:
        """Extracts structured PII entities using Azure Language Studio."""
        comprehensive_pii_categories = [
            PiiEntityCategory.PERSON,
            PiiEntityCategory.PHONE_NUMBER,
            PiiEntityCategory.EMAIL,
            PiiEntityCategory.ADDRESS,
            PiiEntityCategory.DATE,
            PiiEntityCategory.AGE,
            PiiEntityCategory.UK_NATIONAL_INSURANCE_NUMBER,
            PiiEntityCategory.UK_NATIONAL_HEALTH_NUMBER,
            PiiEntityCategory.ORGANIZATION
        ]

        try:
            result = self.text_analytics_client.recognize_pii_entities(
                [text_chunk],
                categories_filter=comprehensive_pii_categories
            )
            entities = [
                {"text": ent.text, 
                 "category": ent.category,
                 "offset": ent.offset,
                 "length": ent.length                 
            }
                for doc in result if not doc.is_error for ent in doc.entities
            ]
            return entities
        except Exception as e:
            print(f"Error getting PII from Language Service: {e}")
            return []

    def is_school(self, organization_name: str, context_sentence: str, fallback_to_conservative: bool = True) -> bool:
        """
        Uses a cheap, fast LLM call to determine if an organization name is likely a school.
        Fixed: Now has proper error handling that defaults to conservative behavior.
        """
        system_prompt = """
        You are a simple boolean classifier. Your only task is to determine if the given organization name is an educational institution (like a school, college, or university) based on the name and the context sentence it appeared in.
        Respond with a single word: "true" if it is an educational institution, and "false" if it is not.
        """
        user_prompt = f"Organization Name: \"{organization_name}\"\nContext Sentence: \"{context_sentence}\""
        
        try:
            model = self.get_appropriate_model(TaskComplexity.SIMPLE)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1, # We only need one word
                temperature=0.0
            )
            return response.choices[0].message.content.lower() == "true"
        except Exception as e:
            print(f"Error in is_school check: {e}")
            # FIXED: Better error handling - if we can't determine, be conservative
            # and assume it could be a school (better to over-redact than under-redact)
            if fallback_to_conservative:
                print(f"Defaulting to conservative classification for '{organization_name}' due to error")
                return self._conservative_school_check(organization_name)
            return False

    def _conservative_school_check(self, organization_name: str) -> bool:
        """
        Fallback method that uses simple keyword matching when API is unavailable.
        Conservative approach - assumes it's a school if it contains school-related keywords.
        """
        school_keywords = [
            'school', 'college', 'university', 'academy', 'institute', 'education',
            'primary', 'secondary', 'high school', 'elementary', 'kindergarten',
            'nursery', 'preschool', 'campus', 'learning', 'student'
        ]
        organization_lower = organization_name.lower()
        return any(keyword in organization_lower for keyword in school_keywords)

    def classify_organizations_batch(self, organizations_with_context: List[Tuple[str, str]]) -> List[bool]:
        """
        Batch processing for organization classification - more efficient than individual calls.
        Takes list of (organization_name, context_sentence) tuples.
        """
        if not organizations_with_context:
            return []

        # Build batch prompt
        system_prompt = """
        You are a batch classifier for educational institutions. Below is a list of organization names with their context sentences.
        For each organization, determine if it is an educational institution (school, college, university, etc.).
        Respond with a JSON array of booleans in the same order as the input organizations.
        
        Format: [true, false, true, ...]
        """
        
        # Format organizations for the prompt
        org_list = []
        for i, (org_name, context) in enumerate(organizations_with_context):
            org_list.append(f"{i+1}. Organization: \"{org_name}\" | Context: \"{context}\"")
        
        user_prompt = "Organizations to classify:\n" + "\n".join(org_list)
        
        try:
            model = self.get_appropriate_model(TaskComplexity.SIMPLE)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            classifications = result.get("classifications", [])
            
            # Ensure we have the right number of results
            if len(classifications) != len(organizations_with_context):
                print(f"Warning: Batch classification returned {len(classifications)} results for {len(organizations_with_context)} organizations. Falling back to individual checks.")
                return [self.is_school(org, ctx) for org, ctx in organizations_with_context]
            
            return classifications
            
        except Exception as e:
            print(f"Error in batch organization classification: {e}")
            # Fallback to individual checks
            return [self.is_school(org, ctx) for org, ctx in organizations_with_context]

    def is_date_format(self, text: str) -> bool:
        """
        Use fast model for simple date format detection.
        Returns True if text appears to be a date.
        """
        # First try regex patterns for common date formats
        date_patterns = [
            r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}',
            r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}'  # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # If regex doesn't match, use LLM for edge cases
        return self._llm_date_check(text)

    def _llm_date_check(self, text: str) -> bool:
        """Use LLM for date format detection when regex fails"""
        system_prompt = """
        Determine if the given text represents a date. Respond with only "true" or "false".
        Consider various date formats including written dates, partial dates, etc.
        """
        
        try:
            model = self.get_appropriate_model(TaskComplexity.SIMPLE)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=1,
                temperature=0.0
            )
            return response.choices[0].message.content.lower() == "true"
        except Exception as e:
            print(f"Error in LLM date check: {e}")
            return False

    def is_phone_number_format(self, text: str) -> bool:
        """
        Use fast model for phone number validation.
        Returns True if text appears to be a phone number.
        """
        # First try regex patterns for common phone formats
        phone_patterns = [
            r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
            r'(\+\d{1,3}[-.\s]?)?\d{10,15}',  # International
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Basic format
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'  # International with country code
        ]
        
        for pattern in phone_patterns:
            if re.search(pattern, text):
                return True
        
        # If regex doesn't match, use LLM for edge cases
        return self._llm_phone_check(text)

    def _llm_phone_check(self, text: str) -> bool:
        """Use LLM for phone number detection when regex fails"""
        system_prompt = """
        Determine if the given text represents a phone number. Respond with only "true" or "false".
        Consider various phone number formats including international formats, extensions, etc.
        """
        
        try:
            model = self.get_appropriate_model(TaskComplexity.SIMPLE)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=1,
                temperature=0.0
            )
            return response.choices[0].message.content.lower() == "true"
        except Exception as e:
            print(f"Error in LLM phone check: {e}")
            return False

    def validate_pii_entities_batch(self, entities_with_context: List[Tuple[dict, str]]) -> List[dict]:
        """
        Batch validation of PII entities using simple model.
        Takes list of (entity_dict, context) tuples and returns validated entities.
        """
        if not entities_with_context:
            return []

        validated_entities = []
        phone_checks = []
        date_checks = []
        
        # Group entities by type for batch processing
        for entity, context in entities_with_context:
            if entity['category'] == 'PhoneNumber':
                phone_checks.append((entity, context))
            elif entity['category'] == 'DateTime':
                date_checks.append((entity, context))
            else:
                # For other types, add directly (they're already validated by Azure)
                validated_entities.append(entity)
        
        # Batch validate phone numbers
        if phone_checks:
            phone_results = self._batch_validate_phones([e[0]['text'] for e, _ in phone_checks])
            for (entity, context), is_valid in zip(phone_checks, phone_results):
                if is_valid:
                    validated_entities.append(entity)
        
        # Batch validate dates
        if date_checks:
            date_results = self._batch_validate_dates([e[0]['text'] for e, _ in date_checks])
            for (entity, context), is_valid in zip(date_checks, date_results):
                if is_valid:
                    validated_entities.append(entity)
        
        return validated_entities

    def _batch_validate_phones(self, phone_texts: List[str]) -> List[bool]:
        """Batch validate phone numbers"""
        if not phone_texts:
            return []
            
        system_prompt = """
        You are validating phone numbers. For each text provided, determine if it's a valid phone number.
        Respond with a JSON object: {"results": [true, false, true, ...]}
        The results array should be in the same order as the input texts.
        """
        
        user_prompt = f"Texts to validate: {json.dumps(phone_texts)}"
        
        try:
            model = self.get_appropriate_model(TaskComplexity.SIMPLE)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("results", [False] * len(phone_texts))
            
        except Exception as e:
            print(f"Error in batch phone validation: {e}")
            # Fallback to individual regex checks
            return [self.is_phone_number_format(text) for text in phone_texts]

    def _batch_validate_dates(self, date_texts: List[str]) -> List[bool]:
        """Batch validate dates"""
        if not date_texts:
            return []
            
        system_prompt = """
        You are validating dates. For each text provided, determine if it represents a date.
        Respond with a JSON object: {"results": [true, false, true, ...]}
        The results array should be in the same order as the input texts.
        """
        
        user_prompt = f"Texts to validate: {json.dumps(date_texts)}"
        
        try:
            model = self.get_appropriate_model(TaskComplexity.SIMPLE)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("results", [False] * len(date_texts))
            
        except Exception as e:
            print(f"Error in batch date validation: {e}")
            # Fallback to individual regex checks
            return [self.is_date_format(text) for text in date_texts]

    def link_entities_to_person(self, text_chunk: str, pii_entities: list) -> dict:
        """
        Uses an LLM to link PII entities to the primary person they belong to in the text.
        """
        if not pii_entities:
            return {}

        system_prompt = """
        You are an entity-linking specialist. Below is a block of text and a list of PII entities found within it.
        Your task is to link each PII entity to the primary person it belongs to in the text.
        Return a JSON object where the keys are the exact text of each PII entity, and the value is the name of the person it is associated with.
        - If an entity IS a person's name, the value should be the name itself.
        - If an entity clearly belongs to a person mentioned in the text, the value should be that person's name.
        - If an entity does not belong to any specific person, use the value 'None'.

        --- EXAMPLE ---
        Text: "Oliver (DOB: 14 March 2015) was quiet. He attends Bridgwater Primary School. Sarah Linton is the case worker."
        PII Entities: ["Oliver", "14 March 2015", "Bridgwater Primary School", "Sarah Linton"]

        Your Output:
        {
        "Oliver": "Oliver",
        "14 March 2015": "Oliver",
        "Bridgwater Primary School": "Oliver",
        "Sarah Linton": "Sarah Linton"
        }
        """
        # Format the PII entities for the user prompt
        entity_list_str = ", ".join([f'"{ent["text"]}"' for ent in pii_entities])
        user_prompt = f"Text: \"{text_chunk}\"\nPII Entities: [{entity_list_str}]"
        
        try:
            model = self.get_appropriate_model(TaskComplexity.COMPLEX)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error performing entity linking: {e}")
            return {}

    def get_sensitive_information(self, text_chunk: str, user_context: str) -> List[Dict]:
        """
        Uses an LLM for nuanced, context-aware redaction based on specific user rules.
        """

        system_prompt = f"""
        You are a highly advanced document analysis tool. Your task is to analyze a specific block of text based on a user's rule, using the surrounding text for context only.

        **USER'S SENSITIVE CONTENT RULE:** "{user_context}"

        --- YOUR THOUGHT PROCESS ---
        1. First, I will read the full text to understand the full context.
        2. Second, I will ONLY extract passages, sentences, or quotations from the "TARGET TEXT" that strictly match the user's rule. I will not extract anything from the context block.
        
        For each match, use the category `SensitiveContent`. In your reasoning, you MUST explain how the extracted text specifically relates to the user's rule.

        CRITICAL: Only extract text that directly matches the user's rule. Do not extract anything else.

        **Output Format:**
        Respond ONLY with a valid JSON object with a single key "redactions", which is an array of objects.
        Each object must have "text", "category", and "reasoning". If nothing is found, return an empty "redactions" array.
        """
        
        try:
            model = self.get_appropriate_model(TaskComplexity.COMPLEX)
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_chunk}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            if response_content:
                data = json.loads(response_content)
                return data.get("redactions", [])
            return []
        except Exception as e:
            print(f"An error occurred while calling Azure OpenAI: {e}")
            return []