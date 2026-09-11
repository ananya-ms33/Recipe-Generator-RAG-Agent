# AI Recipe Generator & Document Q&A Agent

An intelligent culinary assistant that parses unstructured recipe documents (cookbooks, PDFs, text notes), performs semantic retrieval using Retrieval-Augmented Generation (RAG), and adapts recipes dynamically based on user dietary needs, serving sizes, and available pantry ingredients.

---

## 📌 Project Overview

Food enthusiasts and culinary professionals frequently deal with unstructured recipe collections scattered across cookbooks, blogs, and notes. This project builds an agentic system that allows users to query, adapt, and generate personalized recipes from any uploaded document.

### Key Features
- **Document Ingestion & Indexing:** Parses and chunks multi-page PDFs and text cookbooks into a local vector store using ChromaDB.
- **Conversational Q&A (RAG):** Answers natural language queries strictly using document context and provides exact source page citations.
- **Agentic Personalization Layer:**
  - **Dietary Adaptation:** Customizes recipes for Sugar-Free, Vegan, Vegetarian, Gluten-Free, Keto, Dairy-Free, and High-Protein diets.
  - **Pantry Ingredient Substitution:** Intelligently substitutes missing ingredients using items you currently have at home (e.g., using apples for eggs/sugar) and explains the culinary reasoning.
  - **Mathematical Serving Scaler:** Automatically scales ingredient portions from 1 to 12 servings.
- **Structured Cooking Guidance:** Generates step-by-step instructions, chef pro tips, estimated nutritional breakdowns, and a smart split grocery list (Items in Pantry vs. Items to Buy).

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **RAG & Agent Orchestration:** LangChain, Langflow
- **LLM Reasoning:** Google Gemini (`gemini-flash-latest`)
- **Vector Database & Embeddings:** ChromaDB with local ONNX embeddings (`all-MiniLM-L6-v2`)
- **Document Processing:** PyPDF, TextLoader

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ananya-ms33/Recipe-Generator-RAG-Agent.git
cd Recipe-Generator-RAG-Agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root folder:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 4. Run the Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. (Optional) Run via Terminal / CLI
```bash
python cli_agent.py
```

---

## 📁 Repository Structure

```
├── app.py               # Streamlit web application
├── rag_engine.py        # Core RAG retrieval & agentic reasoning pipeline
├── cli_agent.py         # Terminal-based interactive runner
├── cookbook_pdf.pdf     # Sample multi-page PDF cookbook (Food Hero)
├── cookbook.txt         # Sample structured TXT cookbook
├── requirements.txt     # Project dependencies
└── README.md            # Documentation
```
