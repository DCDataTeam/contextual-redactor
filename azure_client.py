import os
import json
from typing import List, Dict

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from openai import AzureOpenAI

class AzureAIClient:
    def __init__(self):
        # ... (credentials and client initializations remain the same)
        try:
            doc_intel_endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
            doc_intel_key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
            openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            openai_key = os.environ["AZURE_OPENAI_KEY"]
            self.openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
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
        
    def analyze_document(self, file_path: str) -> AnalyzeResult:
        print("Analyzing document with Azure AI Document Intelligence...")
        with open(file_path, "rb") as f:
            poller = self.doc_intel_client.begin_analyze_document(
                "prebuilt-layout", body=f, content_type="application/octet-stream"
            )
        result: AnalyzeResult = poller.result()
        print("Document analysis complete.")
        return result

    def get_sensitive_information(self, text_chunk: str, user_context: str) -> List[Dict]:
        """
        Uses a GPT-4 model to find sensitive information based on a base prompt and dynamic user instructions.
        """

        system_prompt = """
        You are a meticulous and forensic line-by-line document analyst specialising in data privacy and redaction. 
        Your task is to analyse the paraghraphs of text provided by the user and identify every instance of SPECIFIC, PERSONALLY IDENTIFYING INFORMATION (PII) for redaction, based on a strict two-level hierarchy of rules.

        --- CRITICAL EXCLUSIONS (What NOT to Redact) ---
        Before you begin, you MUST understand these global rules. Do NOT redact the following categories of text unless a user's instruction explicitly overrides it:
        - **Generic Roles or Job Titles:** Do NOT redact job titles or generic roles.
        - EXAMPLES TO IGNORE: "Family Case Worker", "school counselor", "Dr.", "Mrs.", "teacher".
        - **Somerset Council:** Do NOT redact any mention of "Somerset Council".
        - **Generic Nouns for People/Places:** Do NOT redact common nouns that refer to people or places in a non-specific way.
        - EXAMPLES TO IGNORE: "the children", "his father's home", "the school", "the hospital", "their teachers", "mum", "dad".
        - **General Dates and Timestamps:** Do NOT redact dates and times that are part of the document's metadata or general narrative (like email timestamps or appointment dates). Only redact dates if they are clearly a Date of Birth.
        - EXAMPLES TO IGNORE: "01 September 2025 15:10", "last Monday", "next Thursday".

        --- 1. BASE REDACTION RULES (Default Policy) ---
        By default, you MUST identify and extract every instance of the following general categories. This is your standing-order security policy.
            - **PersonName:** The full or partial proper names of specific private individuals.
                - EXAMPLE: Redact "Laura Bennett". Do NOT redact "the case worker".
            - **ContactDetails:** Personal email addresses and personal phone numbers.
            - **SpecificLocation:** The proper names of specific locations or full street addresses.
                - EXAMPLE: Redact "7 Brookside Lane, Taunton" or "Westfield Primary". Do NOT redact generic places such as "the school", "her father's house", or "a home visit".
            - **DateOfBirth:** A specific date identified as a person's date of birth.
            - **Ages:** A specific age that identifies a person.
            - **Medical_Information:** Specific named medical conditions or medications.
            - **Criminal_Record_Information:** Information on arrests, charges or convictions.
            - **Financial_Information:** Bank details, credit card numbers, salary information.
            - **Identification_Numbers:** National Insurance numbers, passport numbers, driver's license numbers
                

        --- 2. HOW TO APPLY USER INSTRUCTIONS (Amendments to the Policy) ---
        You may receive specific instructions for the current document from a user. These instructions are your HIGHEST PRIORITY and you MUST treat them as explicit amendments that override the base rules.

        Here is how you must reason about user instructions:

        - **For an EXCEPTION (Allow-Listing):**
        - If the user says: "Do not redact the name of Dr Helen Morris."
        - Your Logic: You will create NO redaction for the specific text "Dr Helen Morris", even though it matches the base 'PersonName' rule.

        - **For an ADDITION (Deny-Listing):**
        - If the user says: "Also redact all internal project codenames."
        - Your Logic: You will add "project codenames" to your search criteria. For these, use the 'UserSpecified' category.

        - **For a SPECIFIC OVERRIDE and the FALLBACK RULE:**
        - If a user rule is specific to a subset, like: "Redact the names of all external consultants."
        - Your Logic MUST be:
            1. "The user's rule for 'external consultants' is a specific override of the general 'PersonName' base rule."
            2. "Therefore, I will find and redact the names of anyone identified as an external consultant."
            3. "The user's rule was SILENT on the topic of internal employees."
            4. "Therefore, I MUST FALL BACK to the base 'PersonName' rule and redact the names of all internal employees as well."

        **CRITICAL DIRECTIVE: ALWAYS FALL BACK TO THE BASE RULES FOR ANY CATEGORY NOT EXPLICITLY MENTIONED IN THE USER'S INSTRUCTIONS.**
        **CRITICAL DIRECTIVE: Your primary goal is to find SPECIFIC IDENTIFIERS. If a term is generic or common, you should NOT redact it.**
        **CRITICAL DIRECTIVE: DO NOT SUMMARISE. DO NOT GROUP. EXTRACT ALL OCCURRENCES.**
            - If the name 'John Smith' appears 3 times in the text, you MUST return 3 separate JSON objects for 'John Smith'. Your task is to perform a complete and literal extraction.

        **CRITICAL RULES:**
        - Do NOT identify field labels (e.g., 'Name:'), only the actual data values.
        - You MUST return every instance of sensitive information, even if it appears multiple times. 
        - The 'text' field in your response must be the exact text from the document.        
        - You are thorough, dilligient and factual, never altering text or any information given to you.
        - You will use the Categories for Redaction to guide the type of redactions to be suggested to the user.
        - You will observe and follow the critical rules and output format given to you.
        - You will process emails, letters, reports and other documents and will interpret their formats and nuances accordingly.
        """

        # Append user context cleanly to the system prompt
        if user_context and user_context.strip():
            system_prompt += f"""
        --- ADDITIONAL USER INSTRUCTIONS FOR THIS DOCUMENT ---
        A user has provided the following specific instructions. You MUST treat these rules with the highest priority.

        USER INSTRUCTIONS: "{user_context}"
        --- END OF USER INSTRUCTIONS ---
        """

        # Add the output format instructions at the very end
        system_prompt += """
        **Output Format:**
        Respond ONLY with a valid JSON object containing a single key "redactions", which is an array of objects. Each object must contain "text", "category", and "reasoning". If nothing is found, return an empty "redactions" array.
        
        ---
        **EXAMPLE OF CORRECT ANALYSIS:**

        USER PROVIDES THIS TEXT CHUNK:
        "From: Laura Bennett
        To: Rachel Merton
        Subject: Update

        Dear Rachel,
        I spoke with the child today regarding the case. Here are my notes:
        Laura Bennett met with child and discussed the following...."

        YOUR CORRECT JSON OUTPUT SHOULD BE:
        {
        "redactions": [
            {
            "text": "Laura Bennett",
            "category": "PersonName",
            "reasoning": "Identified the sender's name in the 'From:' header."
            },
            {
            "text": "Rachel Merton",
            "category": "PersonName",
            "reasoning": "Identified the recipient's name in the 'To:' header."
            },
            {
            "text": "Rachel",
            "category": "PersonName",
            "reasoning": "Identified a name in the email's salutation."
            },
            {
            "text": "Laura Bennett",
            "category": "PersonName",
            "reasoning": "Identified a name mentioned in the main body of the text."
            }
        ]
        }
        ---
        """
        
        #print(user_prompt)  # Debug: Print the user prompt to verify its content
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