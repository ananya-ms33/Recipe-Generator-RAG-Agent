# 🍳 Document Q&A RAG Agent for Recipe Generator

**Problem Statement No. 8:** Document Q&A RAG Agent for Recipe Generator  
**Student Name:** ANANYA M S  
**Roll No:** 2024103625  
**Email:** ananyams003@gmail.com  
**WhatsApp / Mobile:** 6369779752  

---

## 📌 Project Overview
Food enthusiasts and culinary professionals frequently struggle with unstructured recipe collections across PDFs, cookbooks, and notes. This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system integrated with **Langflow** and **Google Gemini** that:
1. **Ingests & Indexes Recipe Documents:** Parses cookbooks, text notes, and PDFs into vector embeddings.
2. **Enables Conversational Q&A:** Answers culinary questions with recipe context.
3. **Applies Agentic Personalization:** Adapts recipes to dietary restrictions (Sugar-free, Vegan, Gluten-free, Keto, Dairy-free), available pantry items, and serving sizes.
4. **Delivers Structured Cooking Guidance:** Generates step-by-step instructions, functional ingredient substitutions, nutritional estimates, and smart grocery checklists.

---

## 🏗️ Architecture & Langflow Flow

The pipeline is built visually in **Langflow** and connected to an interactive **Streamlit** user interface:

```
+------------------+         +----------------------------+
|  User Query &    | ------> |      Langflow Flow         |
|  Cookbook Upload |         |  (ID: 26d98697-ae15-...)   |
+------------------+         +----------------------------+
                                           |
                                           v
       +-------------------------------------------------------------+
       | [Chat Input]                                                |
       |      |                                                      |
       |      +-------------------> [Search Query: Knowledge]        |
       |      |                                | (Results)           |
       |      |                                v                     |
       |      |                         [Text Parser]                |
       |      |                                | (Parsed Context)    |
       |      v                                v                     |
       |  [question] -------------------> [Prompt Template]          |
       |                                       |                     |
       |                                       v                     |
       |                          [Agent: Gemini-Flash-Latest]       |
       |                                       |                     |
       |                                       v                     |
       |                                 [Chat Output]               |
       +-------------------------------------------------------------+
                                           |
                                           v
                             +---------------------------+
                             |   Streamlit Web UI / CLI  |
                             |  - Step-by-Step Guide     |
                             |  - Smart Substitutions    |
                             |  - Nutrition Breakdown    |
                             |  - Shopping Checklist     |
                             +---------------------------+
```

---

## 🧩 Langflow Components Used

1. **Chat Input:** Captures user queries, dietary constraints, and desired recipe adaptations.
2. **Knowledge Base / File Loader (`CookBooks`):** Ingests and performs semantic retrieval on cookbook files (`cookbook.txt`, PDFs).
3. **Parser Component:** Transforms raw vector store outputs (`Text: {content}`) into structured recipe context.
4. **Prompt Template:** Combines retrieved recipe context with user requirements and culinary instructions.
5. **Agent (Language Model: `gemini-flash-latest` / `gemini-2.5-flash`):** Expert chef agent that performs reasoning, ingredient scaling, substitution mapping, and nutrition calculation.
6. **Chat Output:** Emits the formatted recipe instructions and shopping guidance to the frontend.

---

## 🚀 Getting Started

### 1. Environment Configuration
Create a `.env` file with your credentials:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
LANGFLOW_URL=http://localhost:7860/api/v1/run/26d98697-ae15-4fad-ba76-5e1432bda889
LANGFLOW_API_TOKEN=sk-2Cy61yYiO3C6gaVM3776VgDWXon5X41s6p62YUJloSc
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### 4. Run the Terminal / CLI Agent
```bash
python cli_agent.py
```

---

## 🔮 Future Scope
- **Multimodal Ingredient Scanning:** Snap a picture of your refrigerator to automatically detect ingredients.
- **Voice-Guided Cooking Mode:** Hands-free step-by-step audio prompts while cooking.
- **Instant Grocery Delivery API:** Export generated shopping lists directly to grocery apps (Instacart/Blinkit).
