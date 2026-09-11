import os
import json
import re
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
import chromadb.utils.embedding_functions as ef

# ---------------------------------------------------------------------------
# Section A: 100% Local Embeddings (Zero Google Embedding API Quota Usage)
# ---------------------------------------------------------------------------
class LocalONNXEmbeddings(Embeddings):
    """Local ONNX-based embedding model (all-MiniLM-L6-v2). Zero API quota limits."""
    def __init__(self):
        self._fn = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.strip() if t.strip() else "recipe" for t in texts]
        return self._fn(cleaned)

    def embed_query(self, text: str) -> List[float]:
        cleaned = text.strip() if text.strip() else "recipe"
        return self._fn([cleaned])[0]


# ---------------------------------------------------------------------------
# Section B: Core Document Q&A RAG Agent (Problem Statement 8)
# ---------------------------------------------------------------------------
class RecipeRAGAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key not found. Please provide an API key.")
        
        # Local embeddings to prevent any 429 quota exhaustion
        self.embeddings = LocalONNXEmbeddings()
        
        # Gemini 2.5 Flash for fast, expert culinary intelligence
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.2
        )
        self.vector_store: Optional[Chroma] = None
        self.indexed_files: List[str] = []
        self.total_chunks: int = 0
        self.doc_name: str = ""

    def index_documents(self, file_paths: List[str]) -> int:
        """Loads and indexes any PDF or TXT document into local Chroma vector store."""
        all_docs = []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            try:
                if path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(path)
                else:
                    loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
                docs = loader.load()
                base_name = os.path.basename(path)
                for d in docs:
                    d.metadata["source_file"] = base_name
                all_docs.extend(docs)
                self.doc_name = base_name
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

        if not all_docs:
            raise ValueError("No text could be extracted from the document.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(all_docs)

        # In-memory Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        self.indexed_files = file_paths
        self.total_chunks = len(splits)
        return len(splits)

    def get_document_overview(self) -> str:
        """Extracts a quick overview of what recipes and topics are in the document."""
        if not self.vector_store:
            return "No document indexed."
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        sample_docs = retriever.invoke("recipes ingredients meals cooking food table of contents")
        sample_text = "\n\n".join([d.page_content for d in sample_docs])

        prompt = ChatPromptTemplate.from_template(
            "Summarize in 2 sentences what recipes/content this document contains and list 4-5 sample dishes:\n\n{text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"text": sample_text})
        except Exception:
            return "Document indexed successfully. Ready to answer recipe queries."

    def query(
        self,
        question: str,
        dietary_restriction: str = "None",
        available_ingredients: str = "",
        servings: int = 4,
        max_time_mins: Optional[int] = None,
        cuisine_preference: str = "Any"
    ) -> Dict[str, Any]:
        """
        Executes semantic search over unstructured document chunks and generates
        comprehensive, personalized cooking guidance fulfilling all Problem Statement 8 goals.
        """
        if not self.vector_store:
            raise ValueError("No document indexed. Please upload or select a document.")

        # 1. Retrieve top-k relevant chunks
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(question)
        
        context_parts = []
        for d in relevant_docs:
            source = d.metadata.get("source_file", "Document")
            page = d.metadata.get("page", None)
            page_info = f" (Page {page + 1})" if page is not None else ""
            context_parts.append(f"--- [Source: {source}{page_info}] ---\n{d.page_content}")
            
        context = "\n\n".join(context_parts)

        # 2. Structured Agent Prompt
        system_prompt = """You are an expert AI Culinary Chef and Document Q&A Recipe Generator Agent (Problem Statement 8).
Your mission is to answer user recipe queries by retrieving relevant text from the provided document context and adapting the recipes according to user constraints.

DOCUMENT CONTEXT (Extracted from unstructured cookbook / recipe file):
{context}

USER CONSTRAINTS:
- Dietary Restrictions: {dietary_restriction}
- Available Pantry Ingredients: {available_ingredients}
- Target Servings: {servings}
- Max Cooking Time: {max_time_mins}
- Cuisine / Flavor Preference: {cuisine_preference}

USER REQUEST:
{question}

Deliver your response in clean, beautiful GitHub Markdown with the following 5 structured sections:

### 1. 🍳 Recipe Title & Document Match
- **Recipe Name:** (Name of the recipe from or inspired by the document)
- **Document Source:** (Exact document file name and Page Number from the context)
- **Timing & Yield:** Prep Time: [X mins] | Cook Time: [Y mins] | Total Time: [Z mins] | Scaled for **{servings} servings**

### 2. 🥗 Ingredients & Smart Substitutions
- List all ingredients scaled precisely for **{servings} servings**.
- If dietary restrictions ({dietary_restriction}) or pantry constraints are specified, explicitly highlight which ingredients were substituted and explain **why the substitution works functionally** (e.g. sweetness, texture, binding, moisture).

### 3. ⏱️ Step-by-Step Cooking Instructions
- Provide numbered, chronological, easy-to-follow cooking steps based on the document.
- Include **💡 Pro Chef Tips** for heat control, texture, or flavor enhancement.

### 4. 📊 Estimated Nutritional Facts (Per Serving)
- **Calories:** ~[X] kcal
- **Protein:** ~[X] g | **Carbs:** ~[X] g | **Fats:** ~[X] g | **Dietary Fiber:** ~[X] g
- **Dietary Highlights:** (e.g. Sugar-Free, High-Protein, Low-Sodium, Heart-Healthy)

### 5. 🛒 Smart Shopping List
- A clean bulleted checklist of items needed to prepare this dish (accounting for available pantry items).

If the user query is an open-ended question (e.g. "What dishes are in this file?"), provide a rich, structured catalog of the recipes found in the document.
"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context if context else "No document excerpts found.",
            "dietary_restriction": dietary_restriction if dietary_restriction != "None" else "Standard / No restrictions",
            "available_ingredients": available_ingredients if available_ingredients else "Standard pantry ingredients",
            "servings": str(servings),
            "max_time_mins": f"{max_time_mins} minutes" if max_time_mins else "No strict limit",
            "cuisine_preference": cuisine_preference if cuisine_preference != "Any" else "Standard",
            "question": question
        })

        return {
            "answer": response,
            "sources": [
                {
                    "content": d.page_content[:250] + "...",
                    "file": d.metadata.get("source_file", "Document"),
                    "page": d.metadata.get("page", None)
                }
                for d in relevant_docs
            ]
        }
