import os
import json
from typing import List, Dict

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.textanalytics import TextAnalyticsClient, PiiEntityCategory
from openai import AzureOpenAI

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
            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
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
        
    def is_school(self, organization_name: str, context_sentence: str) -> bool:
        """
        Uses a cheap, fast LLM call to determine if an organization name is likely a school.
        """
        system_prompt = """
        You are a simple boolean classifier. Your only task is to determine if the given organization name is an educational institution (like a school, college, or university) based on the name and the context sentence it appeared in.
        Respond with a single word: "true" if it is an educational institution, and "false" if it is not.
        """
        user_prompt = f"Organization Name: \"{organization_name}\"\nContext Sentence: \"{context_sentence}\""
        
        try:
            # Note: For a task this simple, you could potentially use a much smaller, cheaper model deployment.
            response = self.openai_client.chat.completions.create(
                model=self.openai_fast_deployment,
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
            return False

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
                response = self.openai_client.chat.completions.create(
                    model=self.openai_deployment,
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
            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
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