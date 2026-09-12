import os
import time
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


class LocalONNXEmbeddings(Embeddings):
    """Local embedding wrapper using all-MiniLM-L6-v2."""
    def __init__(self):
        self._fn = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.strip() if t.strip() else "recipe" for t in texts]
        return self._fn(cleaned)

    def embed_query(self, text: str) -> List[float]:
        cleaned = text.strip() if text.strip() else "recipe"
        return self._fn([cleaned])[0]


class RecipeRAGAgent:
    """Document Q&A RAG Agent for personalized recipe generation."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key not found.")
        
        self.embeddings = LocalONNXEmbeddings()
        # High quota models
        self.models_to_try = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemma-4-26b-a4b-it"]
        self.vector_store: Optional[Chroma] = None
        self.indexed_files: List[str] = []
        self.total_chunks: int = 0
        self.doc_name: str = ""

    def index_documents(self, file_paths: List[str]) -> int:
        """Parses and indexes PDF or TXT documents into vector store."""
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
                print(f"Error loading {path}: {e}")

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
        """Summarizes document content."""
        if not self.vector_store:
            return "No document indexed."
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        sample_docs = retriever.invoke("recipes ingredients meals cooking food")
        sample_text = "\n\n".join([d.page_content for d in sample_docs])

        prompt = ChatPromptTemplate.from_template(
            "Summarize what recipes and culinary topics this document covers in 2 concise sentences:\n\n{text}"
        )
        
        for model_name in self.models_to_try:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=0.2
                )
                chain = prompt | llm | StrOutputParser()
                return chain.invoke({"text": sample_text})
            except Exception:
                continue
        return "Document indexed successfully. Ready to answer recipe queries."

    def query(
        self,
        question: str,
        dietary_restriction: str = "None",
        available_ingredients: str = "",
        servings: int = 4,
        max_time_mins: Optional[int] = None,
        cuisine_preference: str = "Any",
        chat_history: Optional[List[Dict[str, str]]] = None,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Queries the vector store and generates structured recipe guidance or answers follow-up questions."""
        if not self.vector_store:
            raise ValueError("No document indexed. Please upload or select a document.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(question)
        
        context_parts = []
        for d in relevant_docs:
            source = d.metadata.get("source_file", "Document")
            page = d.metadata.get("page", None)
            page_info = f" (Page {page + 1})" if page is not None else ""
            context_parts.append(f"--- [Source: {source}{page_info}] ---\n{d.page_content}")
            
        context = "\n\n".join(context_parts)

        # Format conversation history
        history_text = "No previous conversation."
        if chat_history:
            recent_history = chat_history[-6:] # Keep last 6 messages
            formatted_turns = []
            for m in recent_history:
                role = "User" if m.get("role") == "user" else "Assistant"
                content = m.get("content", "")
                formatted_turns.append(f"{role}: {content}")
            history_text = "\n\n".join(formatted_turns)

        system_prompt = """You are an intelligent culinary assistant and recipe generator with multi-turn conversational memory.
Use the document context and conversation history below to answer the user's recipe query accurately.

Document Context:
{context}

Previous Conversation History:
{chat_history}

User Current Preferences:
- Dietary Restrictions: {dietary_restriction}
- Available Pantry Ingredients: {available_ingredients}
- Target Servings: {servings}
- Max Cooking Time: {max_time_mins}
- Cuisine Style: {cuisine_preference}

Current User Query:
{question}

Guidelines:
1. **If this is a follow-up question or clarification on the previously discussed recipe** (e.g. asking for substitutions like replacing tofu/eggs, adapting to an air fryer/microwave, storage/reheating tips, or macro questions):
   - Answer the question directly and conversationally using context from the previous recipe.
   - Explain the functional cooking technique, step adjustments, and estimated impact on nutrition/flavor.
   - Include a practical **💡 Chef Tip**.
   - Do NOT rigidly force an entire empty template if the user is asking a targeted question.

2. **If this is a new recipe generation request or major recipe search**:
   - Provide the complete, structured 5-part recipe guide formatted in Markdown with:
     ## 🍳 Recipe: [Recipe Name]
     - **Document Source:** [File Name and Page Number]
     - **Prep Time:** [X mins] | **Cook Time:** [Y mins] | **Total Time:** [Z mins]
     - **Portion Size:** Scaled for **{servings} servings**
     ---
     ### 🥗 Ingredients & Smart Substitutions
     (Scaled for **{servings} servings**)
     - [Ingredients with exact quantities]
     **💡 Smart Substitutions & Pantry Adaptation:**
     - **Adaptation:** [Explain dietary or pantry substitutions made]
     - **Why it works:** [Culinary explanation]
     ---
     ### ⏱️ Step-by-Step Cooking Instructions
     1. **[Step Name]:** [Instructions]
     2. **[Step Name]:** [Instructions]
     3. **[Step Name]:** [Instructions]
     > 💡 **Chef Tip:** [Practical tip for texture, flavor, or heat control]
     ---
     ### 📊 Estimated Nutritional Facts (Per Serving)
     | Nutrient | Amount Per Serving |
     | :--- | :--- |
     | **Calories** | ~[X] kcal |
     | **Protein** | ~[X] g |
     | **Total Carbohydrates** | ~[X] g |
     | **Total Fats** | ~[X] g |
     | **Dietary Fiber** | ~[X] g |
     **Dietary Highlights:** [e.g. Sugar-Free, High-Protein, Vegan, Heart-Healthy]
     ---
     ### 🛒 Smart Shopping List
     **✅ In Your Pantry (Already Have):**
     - [Pantry items used]
     **🛒 Need to Buy (Missing Ingredients):**
     - [ ] [Missing items]
"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        response_text = None
        last_err = None
        
        for model_name in self.models_to_try:
            try:
                active_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=0.3
                )
                chain = prompt | active_llm | StrOutputParser()
                response_text = chain.invoke({
                    "context": context if context else "No document excerpts found.",
                    "chat_history": history_text,
                    "dietary_restriction": dietary_restriction if dietary_restriction != "None" else "Standard / No restrictions",
                    "available_ingredients": available_ingredients if available_ingredients else "None specified (Standard pantry)",
                    "servings": str(servings),
                    "max_time_mins": f"{max_time_mins} minutes" if max_time_mins else "No strict limit",
                    "cuisine_preference": cuisine_preference if cuisine_preference != "Any" else "Standard",
                    "question": question
                })
                break
            except Exception as err:
                last_err = err
                continue

        if not response_text:
            raise last_err or RuntimeError("Failed to generate recipe.")

        return {
            "answer": response_text,
            "sources": [
                {
                    "content": d.page_content[:250] + "...",
                    "file": d.metadata.get("source_file", "Document"),
                    "page": d.metadata.get("page", None)
                }
                for d in relevant_docs
            ]
        }
