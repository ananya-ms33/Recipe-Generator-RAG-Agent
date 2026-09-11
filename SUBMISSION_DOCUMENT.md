# Project Presentation Submission Document

---

## Slide 1: Title Slide

**Project Title –** Document Q&A RAG Agent for Recipe Generator  
**Domain of Project –** Generative AI, Retrieval-Augmented Generation (RAG) & Agentic AI  

| Student Name | Roll No | Email ID | Mobile number [WhatsApp] |
| :--- | :--- | :--- | :--- |
| **ANANYA M S** | **2024103625** | **ananyams003@gmail.com** | **6369779752** |

*Recommended font style and size:*
- Font Style: Arial
- Font Size (Heading): 28
- Font Size (Content): 20

---

## Slide 2: Problem Statement

**Problem Statement -**
Food enthusiasts and culinary professionals often rely on unstructured recipe documents, cookbooks, blogs, and user-submitted notes. Searching across these scattered resources is time-consuming, and it is difficult to adapt recipes based on dietary needs, available ingredients, or cultural preferences. The challenge is not just to retrieve recipes but to intelligently query, adapt, and generate personalized cooking instructions.

---

## Slide 3: Proposed Solution

**Proposed Solution -**
We built a Document Q&A RAG (Retrieval-Augmented Generation) Agent that acts as an intelligent recipe generator. The solution:
• **Ingests & Indexes Recipe Documents:** Parses cookbooks, PDFs, and uploaded recipe notes into a searchable vector knowledge base.
• **Enables Conversational Q&A:** Allows users to ask natural language questions (e.g., *"How can I make a sugar-free version of chocolate cake?"*) and receive precise, recipe-aware answers.
• **Creates a Personalized & Agentic Layer:** Adapts recipes dynamically based on dietary restrictions (Vegan, Keto, Gluten-Free, Sugar-Free), serving size scaling, and available pantry items.
• **Visualizes Cooking Guidance:** Presents structured step-by-step instructions, functional ingredient substitutions, estimated nutritional facts, and an automated shopping checklist.

---

## Slide 4: Technology Used

**Technology Used**
• **LangChain & Langflow:** Visual and programmatic orchestration of RAG pipelines, dynamic prompt templates, and agentic workflows.
• **Language Model:** Google Gemini (`gemini-flash-latest`) for culinary reasoning and recipe adaptation.
• **Vector Store & Knowledge Base:** ChromaDB with local ONNX embeddings (`all-MiniLM-L6-v2`) for quota-free semantic document retrieval.
• **User Interface:** Streamlit interactive web application with real-time Q&A, dietary filters, serving scaler, and document uploader.
• **Document Parsers:** PyPDF and TextLoader for parsing PDF and TXT cookbooks.

---

## Slide 5: Langflow Component Used

**Langflow component Used -**
1) **Chat Input** – Interface where users enter recipe questions, dietary preferences, or pantry ingredients.
2) **Knowledge Component (`CookBooks`)** – Ingests and performs semantic vector search over uploaded recipe files.
3) **Parser Component** – Extracts and formats retrieved document chunks (`Text: {content}`) into structured context.
4) **Prompt Component** – Combines dynamic `{context}` and `{question}` variables with chef instructions.
5) **Agent Component (`gemini-flash-latest`)** – Orchestrates culinary reasoning, recipe adaptation, ingredient scaling, and nutrition estimation.
6) **Chat Output** – Emits structured markdown responses (steps, substitutions, nutrition, shopping list).

---

## Slide 6: Langflow Workflow

**Langflow workflow -**
*(Paste the screenshot of your Langflow canvas here)*

**Workflow Description:**
• **Chat Input** connects to **Knowledge** (Search Query) and **Prompt** (`{question}`).
• **Knowledge Base** retrieves relevant recipe text and sends `Results` to **Parser**.
• **Parser** feeds structured `Parsed Text` as `{context}` into **Prompt**.
• **Prompt** forwards the enriched prompt to **Agent** (`gemini-flash-latest`).
• **Agent** processes the instructions and outputs response to **Chat Output**.

---

## Slide 7: Future Scope

**Future Scope**
• **Multimodal Ingredient Scanning:** Upload photos of your refrigerator/pantry to automatically recognize ingredients and suggest recipes.
• **Voice-Guided Cooking Mode:** Hands-free step-by-step audio instructions with interactive timers during active cooking.
• **Smart Grocery Delivery Integration:** One-click export of the generated shopping list to online grocery services (e.g., Instacart, Blinkit).
• **Nutritional & Fitness Sync:** Directly sync calculated recipe calories and macronutrients to health apps.

---

## Slide 8: Reference / GitHub Link

**Reference / GitHub Link**
• **GitHub Repository:**  
  `https://github.com/ananya-ms33/Recipe-Generator-RAG-Agent.git`

• **Tested & Working Features:**  
  - Complete Langflow Flow JSON & visual canvas setup  
  - Streamlit web frontend (`app.py`) with real-time Q&A and PDF upload  
  - Local ONNX Embeddings with zero quota limitations  
  - Built-in recipe knowledge base with dietary adaptation and pantry ingredient substitution  

---

## Slide 9: Thank You!

# Thank You!

**Thank you for your time and interest.**
