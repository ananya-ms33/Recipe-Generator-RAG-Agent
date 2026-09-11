import os
import tempfile
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
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Section A: Quota-Free Local Embedding Wrapper
# ---------------------------------------------------------------------------
class LocalONNXEmbeddings(Embeddings):
    """Local ONNX-based embedding model (all-MiniLM-L6-v2). Zero API quota limits."""
    def __init__(self):
        self._fn = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.strip() if t.strip() else "empty" for t in texts]
        return self._fn(cleaned)

    def embed_query(self, text: str) -> List[float]:
        cleaned = text.strip() if text.strip() else "empty"
        return self._fn([cleaned])[0]


# ---------------------------------------------------------------------------
# Section B: General-Purpose Document Ingestion & RAG Agent
# ---------------------------------------------------------------------------
class RecipeRAGAgent:
    """
    General-purpose RAG agent designed to handle ANY unstructured recipe document
    (PDFs, TXT files, cookbooks, messy notes, blogs, scanned text).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key not found. Please provide an API key.")
        
        self.embeddings = LocalONNXEmbeddings()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.2
        )
        self.vector_store: Optional[Chroma] = None
        self.indexed_files: List[str] = []
        self.total_chunks: int = 0
        self.doc_summary: str = ""

    def index_documents(self, file_paths: List[str]) -> int:
        """
        Load, chunk, and index ANY unstructured recipe document into ChromaDB.
        Works with any PDF layout, arbitrary TXT, multi-page cookbooks, or single recipe sheets.
        """
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
            except Exception as e:
                print(f"Warning: Error loading {path}: {e}")

        if not all_docs:
            raise ValueError("Could not read any text from the provided file. Please verify the document.")

        # Robust text chunking that adapts to any document structure (paragraphs, bullet points, headers)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(all_docs)

        # In-memory vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        self.indexed_files = file_paths
        self.total_chunks = len(splits)
        return len(splits)

    def summarize_document(self) -> str:
        """Dynamically analyzes whatever document is loaded and summarizes its recipes and culinary contents."""
        if not self.vector_store:
            return "No document loaded."

        # Fetch sample chunks across the document
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        sample_docs = retriever.invoke("recipes ingredients cooking meals dishes food table of contents")
        sample_text = "\n\n".join([d.page_content for d in sample_docs[:6]])

        prompt = ChatPromptTemplate.from_template(
            "Based on the following excerpts from an uploaded document, provide a 2-3 sentence summary of what this cookbook or document contains and list 4-6 key dishes or topics mentioned:\n\n{text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            self.doc_summary = chain.invoke({"text": sample_text})
            return self.doc_summary
        except Exception:
            return "Custom recipe document loaded and indexed successfully."

    def query(
        self,
        question: str,
        dietary_restriction: str = "None",
        available_ingredients: str = "",
        servings: int = 4,
        max_time_mins: Optional[int] = None
    ) -> dict:
        """
        Answers natural language queries strictly using retrieved document context,
        applying agentic adaptations (dietary restrictions, pantry constraints, servings scaling).
        """
        if not self.vector_store:
            raise ValueError("No document indexed. Please upload or select a document first.")

        # 1. Semantic search across unstructured chunks
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(question)
        
        context_parts = []
        for d in relevant_docs:
            source = d.metadata.get("source_file", "Document")
            page = d.metadata.get("page", None)
            page_info = f" (Page {page + 1})" if page is not None else ""
            context_parts.append(f"--- [Source: {source}{page_info}] ---\n{d.page_content}")
            
        context = "\n\n".join(context_parts)

        # 2. Comprehensive Agentic Prompt
        system_prompt = """You are an expert AI Culinary Chef and Document Q&A Recipe Generator Agent.
Your role is to read the unstructured document excerpts provided below, extract relevant recipe information, and answer the user's request with culinary precision.

DOCUMENT CONTEXT (Extracted from uploaded unstructured file):
{context}

USER CONSTRAINTS:
- Dietary Restrictions: {dietary_restriction}
- Available Ingredients in Pantry: {available_ingredients}
- Target Servings: {servings}
- Max Cooking Time Limit: {max_time_mins}

USER REQUEST / QUESTION:
{question}

Please structure your response clearly using the following markdown sections:

### 1. 🍳 Recipe Title & Document Match
- Name of the recipe
- Exact source location (Mention the document file and page number from the context)
- Prep Time, Cook Time, Total Time, Servings (scaled for {servings})

### 2. 🥗 Ingredients & Smart Substitutions
- List of ingredients extracted from the document, accurately scaled for {servings} servings.
- If user specified dietary restrictions ({dietary_restriction}) or available pantry ingredients, highlight the substitutions made.
- Explain functionally why each substitution works (e.g., binds like eggs, sweetens like sugar, replaces dairy).

### 3. ⏱️ Step-by-Step Cooking Instructions
- Clear, numbered step-by-step cooking instructions based on the document.
- Include any chef tips, heat control advice, or techniques found in the document.

### 4. 📊 Estimated Nutritional Facts (Per Serving)
- Calories, Protein, Carbohydrates, Fats, Fiber (based on document or estimated for scaled portion).

### 5. 🛒 Smart Shopping List
- Clear bulleted checklist of ingredients needed.

If the user asks a general question (e.g. "What recipes are in this document?" or "Summarize the dishes"), provide a comprehensive, organized overview of what is in the document context.
"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context if context else "No document excerpts found.",
            "dietary_restriction": dietary_restriction if dietary_restriction != "None" else "Standard / No restrictions",
            "available_ingredients": available_ingredients if available_ingredients else "Standard pantry ingredients",
            "servings": str(servings),
            "max_time_mins": f"{max_time_mins} minutes" if max_time_mins else "No strict limit",
            "question": question
        })

        return {
            "answer": response,
            "sources": [
                {
                    "content": d.page_content[:300] + "...",
                    "file": d.metadata.get("source_file", "Document"),
                    "page": d.metadata.get("page", None)
                }
                for d in relevant_docs
            ]
        }
