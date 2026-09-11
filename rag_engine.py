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
        
        self.embeddings = LocalONNXEmbeddings()
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
        structured cooking guidance with strict pantry ingredient matching and clear layout.
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

        # 2. Strict Pantry & Agentic Prompt
        system_prompt = """You are an expert AI Culinary Chef and Document Q&A Recipe Generator Agent (Problem Statement 8).
Your mission is to answer user recipe queries by retrieving relevant text from the provided document context and adapting the recipes according to user constraints.

DOCUMENT CONTEXT:
{context}

USER CONSTRAINTS:
- Dietary Restrictions: {dietary_restriction}
- Available Pantry Ingredients: {available_ingredients}
- Target Servings: {servings}
- Max Cooking Time: {max_time_mins}
- Cuisine / Flavor Preference: {cuisine_preference}

USER REQUEST:
{question}

CRITICAL RULES FOR PANTRY INGREDIENTS & SUBSTITUTIONS:
1. If the user specified Available Pantry Ingredients ({available_ingredients}):
   - Actively use the ingredients in their pantry!
   - If the recipe requires an ingredient (like eggs, butter, milk, sugar) and the user has a substitute in their pantry (e.g., apples/applesauce for eggs/sugar, oats, olive oil, bananas), explicitly substitute it and explain the culinary chemistry.
   - If they lack an essential ingredient, state clearly what is missing and put it in the "Need to Buy" shopping list.
2. In the Shopping List, ALWAYS separate into:
   - **✅ In Your Pantry (Already have):** [List pantry ingredients that are used in this recipe]
   - **🛒 Need to Buy (Missing items):** [List only missing items]

Deliver your response using the following clean, beautifully formatted sections:

## 🍳 Recipe: [Recipe Title]
- **Document Source:** [Exact Document & Page Number]
- **Prep Time:** [X mins] | **Cook Time:** [Y mins] | **Total Time:** [Z mins]
- **Portion Size:** Scaled for **{servings} servings**

---

### 🥗 Ingredients & Smart Substitutions
(Scaled for **{servings} servings**)
- [List every ingredient with exact measurement]

**💡 Smart Substitutions & Pantry Adaptation:**
- **Adaptation:** [Explicitly explain any dietary or pantry substitutions made, e.g. how apples/bananas replace eggs or sugar for moisture/binding, almond milk for dairy, etc.]
- **Why it works:** [Culinary explanation of the chemistry and flavor profile]

---

### ⏱️ Step-by-Step Cooking Instructions
1. **[Step Name]:** [Detailed instruction]
2. **[Step Name]:** [Detailed instruction]
3. **[Step Name]:** [Detailed instruction]

> 💡 **Chef Pro Tip:** [Practical tip on heat control, texture, or flavor enhancement]

---

### 📊 Estimated Nutritional Facts (Per Serving)
| Nutrient | Amount Per Serving |
| :--- | :--- |
| **Calories** | ~[X] kcal |
| **Protein** | ~[X] g |
| **Total Carbohydrates** | ~[X] g |
| **Total Fats** | ~[X] g |
| **Dietary Fiber** | ~[X] g |

**Dietary Highlights:** [e.g. Sugar-Free, High-Fiber, Vegan, Heart-Healthy]

---

### 🛒 Smart Shopping List
**✅ In Your Pantry (Already Have):**
- [Item 1 from user pantry]
- [Item 2 from user pantry]

**🛒 Need to Buy (Missing Ingredients):**
- [ ] [Missing Item 1]
- [ ] [Missing Item 2]
"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context if context else "No document excerpts found.",
            "dietary_restriction": dietary_restriction if dietary_restriction != "None" else "Standard / No restrictions",
            "available_ingredients": available_ingredients if available_ingredients else "None specified (Standard pantry)",
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
