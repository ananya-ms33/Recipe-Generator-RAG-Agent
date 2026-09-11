import os
import sys
from typing import List, Optional
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
# Quota-Free Local Embedding Wrapper (Avoids Google 429 RESOURCE_EXHAUSTED)
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


class RecipeRAGAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key not found. Please provide an API key.")
        
        self.embeddings = LocalONNXEmbeddings()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.3
        )
        self.vector_store: Optional[Chroma] = None
        self.indexed_files: List[str] = []
        self.total_chunks: int = 0

    def index_documents(self, file_paths: List[str]) -> int:
        """Load, chunk, and index recipe documents (PDF or TXT) into ChromaDB."""
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
                # Attach source metadata
                base_name = os.path.basename(path)
                for d in docs:
                    d.metadata["source_file"] = base_name
                all_docs.extend(docs)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

        if not all_docs:
            raise ValueError("No valid recipe documents could be loaded.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1400,
            chunk_overlap=200,
            separators=["\n## ", "\n# ", "\nRecipe", "\n---\n", "\n\n", "\n", " "]
        )
        splits = text_splitter.split_documents(all_docs)

        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        self.indexed_files = file_paths
        self.total_chunks = len(splits)
        return len(splits)

    def query(
        self,
        question: str,
        dietary_restriction: str = "None",
        available_ingredients: str = "",
        servings: int = 4,
        max_time_mins: Optional[int] = None
    ) -> dict:
        """Query the RAG system and generate structured, personalized recipe guidance."""
        if not self.vector_store:
            raise ValueError("No documents indexed. Please index recipes first.")

        # 1. Semantic search for top relevant chunks
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(question)
        
        context_parts = []
        for d in relevant_docs:
            source = d.metadata.get("source_file", "Cookbook")
            page = d.metadata.get("page", "")
            page_info = f" (Page {page + 1})" if isinstance(page, int) else ""
            context_parts.append(f"--- [Source: {source}{page_info}] ---\n{d.page_content}")
            
        context = "\n\n".join(context_parts)

        # 2. System prompt
        system_prompt = """You are an expert AI Culinary Chef and Recipe Generator Agent.
Your task is to answer user queries by retrieving recipes from the provided knowledge base context and personalizing them according to user constraints.

CONTEXT FROM RECIPE KNOWLEDGE BASE:
{context}

USER CONSTRAINTS:
- Dietary Restrictions: {dietary_restriction}
- Available Ingredients in Pantry: {available_ingredients}
- Target Servings: {servings}
- Max Cooking Time Limit: {max_time_mins}

USER QUERY / REQUEST:
{question}

Please structure your response clearly using the following markdown sections:

### 1. 🍳 Recipe Title & Overview
- Name of the recipe (adapted or retrieved)
- Source match / origin from the cookbook context (mention page/cookbook if available)
- Prep Time, Cook Time, Total Time, Servings (scaled for {servings})

### 2. 🥗 Adapted Ingredients & Smart Substitutions
- Precise list of ingredients scaled for {servings} servings.
- Highlight substitutions made to satisfy dietary restrictions ({dietary_restriction}) or available pantry ingredients.
- Mention why each substitution works functionally (e.g. applesauce/monk fruit for sugar, almond milk for dairy, almond flour for gluten, tofu/beans for meat).

### 3. ⏱️ Step-by-Step Cooking Instructions
- Clear, numbered step-by-step instructions.
- Pro tips for cooking technique, heat control, flavor enhancement, or texture.

### 4. 📊 Estimated Nutritional Facts (Per Serving)
- Calories, Protein, Carbohydrates, Fats, Dietary Fiber (estimated).
- Key dietary highlights (e.g. Sugar-free, High-protein, Low-carb, Heart-healthy).

### 5. 🛒 Smart Shopping List
- Bulleted grocery checklist of missing ingredients needed to prepare this dish.

If the requested recipe is not directly in the knowledge base, use the culinary principles from the context to generate a chef-grade recipe and state clearly that it was synthesized.
"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context if context else "No specific recipe found in documents.",
            "dietary_restriction": dietary_restriction if dietary_restriction != "None" else "Standard / No restrictions",
            "available_ingredients": available_ingredients if available_ingredients else "Any standard pantry ingredients",
            "servings": str(servings),
            "max_time_mins": f"{max_time_mins} minutes" if max_time_mins else "No strict limit",
            "question": question
        })

        return {
            "answer": response,
            "sources": [
                {
                    "content": d.page_content[:200] + "...",
                    "file": d.metadata.get("source_file", "Cookbook"),
                    "page": d.metadata.get("page", None)
                }
                for d in relevant_docs
            ]
        }
