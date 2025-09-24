# Contextual Redactor

An intelligent, human-in-the-loop document redaction tool powered by a hybrid AI architecture. This application goes beyond simple keyword searching, using multiple AI models to understand context, follow nuanced user instructions, and perform forensically secure redactions on PDF documents.

## Overview

The Contextual Redactor can understand a subjective user request like "redact any quotations that are negative about the parents", and redact accordingly.

The Contextual Redactor uses the right AI toolfor the right job:

1. **Azure Language Service:** A fast, cost-effective, and highly accurate model for identifying structured Personally Identifying Information (PII) like names, addresses, ages, and organizations.
2. **Azure OpenAI (GPT-4o / GPT-4):** A powerful Large Language Model (LLM) for the most complex reasoning tasks:
   * Parsing free-text user instructions into structured commands.
   * Performing entity linking to understand which PII belongs to which person.
   * Analysing and redacting subjective, context-dependent content based on user rules.

This is all wrapped in an interactive Streamlit UI that allows a human reviewer to have the final say, correcting the AI's suggestions and adding their own manual redactions with a powerful drawing canvas.

## Features

- **Hybrid AI Analysis:** Combines the strengths of specialized NLP models (for PII) and large language models (for reasoning) for optimal speed, cost, and accuracy.
- **Nuanced Instruction Following:** Users can provide complex, natural language instructions (e.g., "don't redact any PII for Oliver Hughes") which the system intelligently parses and applies.
- **Context-Aware Redaction:** Capable of redacting subjective content, such as opinions or specific types of quotations, based on user-defined rules.
- **Interactive Review UI:** A clear two-column layout allows reviewers to see all AI suggestions, toggle them individually or in groups, and see a live preview of the final document.
- **Full Manual Control:** A powerful drawable canvas allows reviewers to:
  - Draw new redaction boxes directly on the document preview.
  - Edit, move, resize, or delete any redaction box (both AI-generated and manual).
- **"Redact All Occurrences" Power-User Feature:** A user can draw a box over a single word or phrase, and with one click, instruct the system to find and redact every other occurrence of that text in the document. Includes a one-step undo.
- **Forensically Secure Output:** Redactions are not just black boxes placed on top of text. The underlying text and image data is permanently removed from the PDF, and the final file is sanitized to remove metadata, ensuring information is unrecoverable.

## Demo

![A demonstration of the Contextual Redactor UI, showing the AI suggestions list and the interactive document preview.](./assets/demo-screenshot.png)

*The main user interface, showing the AI suggestion checklist on the left and the interactive document preview on the right.*

## Tech Stack

- **Backend & AI Logic:** Python 3.12+
- **Frontend:** Streamlit
- **PDF Processing:** PyMuPDF (`fitz`)
- **AI Services:**
  - **Azure Document Intelligence:** For layout analysis and OCR.
  - **Azure Language Service:** For fast, structured PII and Organization detection.
  - **Azure OpenAI Service:** For GPT-4o / GPT-35-Turbo models for advanced reasoning.
- **Dependency Management:** Poetry

## Setup and Installation

There are two ways to set up the project environment: using `pip` with `requirements.txt`, or using `Poetry`.

#### 1. Prerequisites

- Python 3.12 or higher.
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management. (Not essential)
- An active Microsoft Azure subscription with access to the required AI services.

#### 2. Clone the Repository

```bash
git clone https://github.com/YourUsername/contextual-redactor.git
cd contextual-redactor
```

#### 3. Install Dependencies

Poetry will create a virtual environment and install all necessary packages from the `pyproject.toml` and `poetry.lock` files.

```bash
poetry install
```



If you do not have Poetry installed you can use the standard Python approach:

- **Create a Virtual Environment:**

 ```bash
   python -m venv venv
   ```

- **Activate the Environment:**


 -On Windows: 

```bash
.\venv\Scripts\activate
```


 -On macOS/Linux: 

```bash
source venv/bin/activate
```

- **Install Dependencies:**

```bash
pip install -r requirements.txt
```

---

#### 4. Configure Environment Variables

The application requires credentials for three separate Azure services.

1. Create a file named `.env` in the root of the project.
2. Copy the contents of `.env.example` into it:

   ```env
   # .env.example

   # Azure Document Intelligence Credentials
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="<Your_Document_Intelligence_Endpoint>"
   AZURE_DOCUMENT_INTELLIGENCE_KEY="<Your_Document_Intelligence_Key>"

   # Azure OpenAI Credentials
   AZURE_OPENAI_ENDPOINT="<Your_OpenAI_Endpoint>"
   AZURE_OPENAI_KEY="<Your_OpenAI_Key>"
   AZURE_OPENAI_DEPLOYMENT_NAME="<Your_Deployment_Name_for_GPT-4o>"

   # Azure Language Service Credentials
   AZURE_LANGUAGE_ENDPOINT="<Your_Language_Service_Endpoint>"
   AZURE_LANGUAGE_KEY="<Your_Language_Service_Key>"
   ```
3. Log in to the [Azure Portal](https://portal.azure.com) and create the three required resources (Document Intelligence, OpenAI, Language Service).
4. Fill in the values in your `.env` file with the corresponding **Endpoint URLs** and **Keys** from your Azure resources.

## Usage

1. Activate the Poetry virtual environment (if using Poetry):

   ```bash
   poetry shell
   ```
2. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```
3. Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Project Structure

A brief overview of the key files in this project:

- `app.py`: The main Streamlit application. Manages the UI, state, and user interactions.
- `redaction_logic.py`: The central orchestrator. It manages the multi-step AI workflow, calling different services and processing their results.
- `azure_client.py`: A dedicated class that handles all communication with the various Azure AI service APIs.
- `utils.py`: Contains helper functions for coordinate mapping, text matching (`fuzzywuzzy`), and final suggestion list processing.
- `pdf_processor.py`: A class responsible for the final, secure redaction of the PDF file using PyMuPDF.
