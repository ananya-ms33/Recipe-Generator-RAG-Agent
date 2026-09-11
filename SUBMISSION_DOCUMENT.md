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
We built an intelligent Document Q&A RAG (Retrieval-Augmented Generation) Agent integrated with Langflow and Streamlit. The solution delivers:
- **Ingest & Index Recipe Documents:** Parse cookbooks, PDFs, and uploaded recipe notes into a vector knowledge base.
- **Enable Conversational Q&A:** Allow users to ask natural language questions (e.g., *"How can I make a sugar-free version of chocolate cake?"*) and receive recipe-aware answers.
- **Create a Personalized & Agentic Layer:** Adapt recipes dynamically based on dietary restrictions (Vegan, Keto, Gluten-free, Sugar-free), serving size scaling, and available pantry items.
- **Visualize Cooking Guidance:** Present structured step-by-step instructions, functional ingredient substitutions, estimated nutritional facts, and an automated shopping checklist.

---

## Slide 4: Technology Used

**Technology Used:**
- **Langflow:** Visual drag-and-drop orchestration of RAG pipelines, prompts, and agent nodes.
- **LangChain:** Framework for document loading, text chunking, and LLM chaining.
- **LLM Model:** Google Gemini (`gemini-flash-latest` / `gemini-2.5-flash`) for real-time culinary reasoning and adaptation.
- **Vector Knowledge Base:** Ingests unstructured recipe corpora for semantic similarity search.
- **Streamlit:** Interactive web interface for cookbook uploads, preference filtering, and conversational recipe interaction.
- **PyPDF / Text Parsers:** Document parsing for PDF and TXT cookbooks.

---

## Slide 5: Langflow Component Used

**Langflow Component Used -**
1. **Chat Input** – Interface where users enter recipe questions, dietary preferences, or pantry ingredients.
2. **Knowledge Component (`CookBooks`)** – Ingests and performs semantic vector search over uploaded recipe files.
3. **Parser Component** – Extracts and standardizes retrieved document chunks (`Text: {content}`) into structured context.
4. **Prompt Component** – Combines dynamic `{context}` and `{question}` variables with chef instructions.
5. **Agent Component (`gemini-flash-latest`)** – Orchestrates culinary reasoning, recipe adaptation, ingredient scaling, and nutrition estimation.
6. **Chat Output** – Emits structured markdown responses (steps, substitutions, nutrition, shopping list).

---

## Slide 6: Langflow Workflow

**Langflow Workflow -**

```
[Chat Input] ───────────────┬───────────────────────────┐
                            │ (Query)                   │ (Question)
                            ▼                           ▼
                   [Knowledge: CookBooks]      [Prompt Template]
                            │                           ▲
                            ▼ (Results)                 │ (Context)
                     [Parser Node] ─────────────────────┘
                            │
                            ▼
                    [Agent Component]
                 (gemini-flash-latest)
                            │
                            ▼
                     [Chat Output]
```

---

## Slide 7: Future Scope

**Future Scope:**
- **Multimodal Ingredient Vision:** Upload a photo of the pantry/refrigerator to detect ingredients automatically and suggest matching recipes.
- **Voice-Guided Hands-Free Assistant:** Step-by-step voice guidance and timer alerts for hands-free cooking in the kitchen.
- **Automated Grocery Ordering:** One-click integration with grocery delivery services (Instacart, Amazon Fresh) to order missing ingredients.
- **Nutritional Tracking Integration:** Sync recipe macronutrients and calories with fitness apps (Apple Health, MyFitnessPal).

---

## Slide 8: Reference / GitHub Link

**Reference / GitHub Link:**
- Public GitHub Repository Format: `https://github.com/AnanyaMS/Recipe-Generator-RAG-Agent`
- Tested and verified with local Langflow server and Streamlit frontend.
- Includes sample cookbooks, prompt templates, API connectors, and CLI runner.

---

## Slide 9: Thank You!

**Thank You!**  
*Thank you for your time and interest.*
