# Recipe Generator using RAG

A Streamlit application that reads cookbook PDFs and text documents, answers recipe questions, and personalizes cooking instructions using Retrieval-Augmented Generation (RAG).

## What it does
- **Cookbook Q&A:** Upload or select any cookbook (PDF or TXT) and ask questions about recipes.
- **Dietary Preferences:** Adapt dishes for Vegan, Sugar-Free, Gluten-Free, Keto, and more.
- **Pantry Substitutions:** Enter the ingredients you have at home to get customized substitutions and recipes.
- **Cooking Guide & Shopping List:** Generates step-by-step instructions, scaled serving sizes, estimated nutrition facts, and a grocery checklist.

## Tech Stack
- Python, Streamlit
- LangChain, ChromaDB (Local Embeddings)
- Google Gemini API (`gemini-flash-latest`)
- PyPDF

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Key:**
   Add your Google Gemini API key in a `.env` file:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Start the app:**
   ```bash
   streamlit run app.py
   ```
