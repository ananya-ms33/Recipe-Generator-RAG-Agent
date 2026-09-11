# 🍳 Document Q&A RAG Agent for Recipe Generator

**Problem Statement No. 8:** Document Q&A RAG Agent for Recipe Generator  
**Student Name:** ANANYA M S  
**Roll No:** 2024103625  
**Email:** ananyams003@gmail.com  
**WhatsApp / Mobile:** 6369779752  
**GitHub Repository:** [https://github.com/ananya-ms33/Recipe-Generator-RAG-Agent](https://github.com/ananya-ms33/Recipe-Generator-RAG-Agent)

---

## 📌 Project Overview
Food enthusiasts and culinary professionals frequently struggle with unstructured recipe documents, cookbooks, blogs, and user notes. This project implements a **Retrieval-Augmented Generation (RAG)** system that:
1. **Ingests & Indexes Recipe Documents:** Parses cookbooks and documents (PDF/TXT) into local vector embeddings.
2. **Enables Conversational Q&A:** Answers culinary queries with precise citations.
3. **Applies Agentic Personalization:** Adapts recipes to dietary restrictions (Sugar-Free, Vegan, Gluten-Free, Keto, Dairy-Free), available pantry items, and serving sizes.
4. **Delivers Structured Cooking Guidance:** Generates step-by-step instructions, functional ingredient substitutions, nutritional estimates, and smart grocery checklists.

---

## 🏗️ Architecture & Workflow

```
[User Query & Document Upload]
              │
              ▼
    [Document Parser & Chunking]
              │
              ▼
 [ChromaDB Vector Store (Local ONNX Embeddings)]
              │ (Semantic Retrieval)
              ▼
 [Personalization & Agent Prompt Template]
 (Diet Filter, Serving Scaler, Pantry Constraints)
              │
              ▼
  [Google Gemini Model (gemini-flash-latest)]
              │
              ▼
 [Streamlit Web Interface]
  - Step-by-Step Cooking Guide
  - Smart Ingredient Substitutions
  - Macro & Nutrition Estimates
  - Smart Shopping List (Pantry vs Need to Buy)
```

---

## 🚀 Getting Started

### 1. Configure Environment
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🔮 Future Scope
- Multimodal ingredient scanning via image input.
- Real-time voice-guided hands-free cooking assistant.
- Direct grocery delivery integration (Instacart/Blinkit API).
