"""
=============================================================================
🍳 Document Q&A RAG Agent for Recipe Generator
Problem Statement No. 8 — Production Streamlit Application
- Fully Theme-Adaptive (Looks gorgeous in both Light and Dark mode)
- Dynamic Unstructured PDF / TXT Document Ingestion
- Real-time Pantry Ingredient Substitution & Mathematical Servings Scaler
- Zero Quota Limits (Local ONNX Embeddings + Gemini 2.5 Flash)
=============================================================================
"""

import os
import glob
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Section 1: Page Configuration & Theme-Adaptive UI
# ---------------------------------------------------------------------------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="AI Recipe Generator | Problem Statement 8",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-Adaptive CSS (Works seamlessly in Dark and Light themes)
st.markdown("""
<style>
    .title-text {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .subtitle-text {
        font-size: 15px;
        opacity: 0.85;
        margin-bottom: 15px;
    }
    .filter-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        background-color: rgba(255, 75, 75, 0.15);
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: API Key Configuration
# ---------------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Section 3: Sidebar Controls (Knowledge Base & Personalization Layer)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Setup")
    
    # API Key Input if missing
    if not api_key:
        input_key = st.text_input("🔑 Google Gemini API Key", type="password", help="Enter your Gemini API key")
        if input_key:
            api_key = input_key
        else:
            st.warning("⚠️ Google API Key required to run the agent.")

    st.divider()
    st.markdown("### 📚 1. Knowledge Base")
    st.caption("Select or upload **any PDF / TXT document**.")

    # Locate available documents in workspace
    local_pdfs = glob.glob(os.path.join(BASE_DIR, "*.pdf"))
    local_txts = glob.glob(os.path.join(BASE_DIR, "*.txt"))
    all_files = [os.path.basename(f) for f in (local_pdfs + local_txts)]
    
    default_idx = 0
    if "cookbook_pdf.pdf" in all_files:
        default_idx = all_files.index("cookbook_pdf.pdf")
    elif "cookbook.txt" in all_files:
        default_idx = all_files.index("cookbook.txt")

    selected_doc_name = st.selectbox(
        "Active Cookbook / Document:",
        all_files if all_files else ["None"],
        index=default_idx if all_files else 0
    )
    
    selected_doc_path = os.path.join(BASE_DIR, selected_doc_name) if all_files else None

    # Upload custom document
    uploaded_file = st.file_uploader("Upload Document (PDF / TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        save_path = os.path.join(BASE_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f" Uploaded `{uploaded_file.name}`!")
        selected_doc_path = save_path
        st.cache_resource.clear()

    st.divider()
    st.markdown("### 🎯 2. Agentic Personalization")
    st.caption("Configure recipe adaptation constraints:")
    
    diet = st.selectbox(
        "🥗 Dietary Restriction:",
        ["None", "Sugar-Free", "Vegan", "Vegetarian", "Gluten-Free", "Keto / Low-Carb", "Dairy-Free", "High-Protein", "Low-Sodium", "Nut-Free"]
    )
    servings = st.slider("👥 Target Servings:", min_value=1, max_value=12, value=4)
    pantry = st.text_input(
        "🥕 Your Available Pantry Items:",
        placeholder="Type items here, e.g. apples, oats, milk",
        help="Type ingredients you have at home. The agent will adapt recipes to use these and substitute missing items!"
    )
    cuisine = st.selectbox("🌶️ Cuisine / Flavor Preference:", ["Any", "Mexican", "Italian", "American Comfort", "Asian", "Mediterranean", "Kid-Friendly"])
    max_time = st.number_input("⏱️ Max Cooking Time (Mins, 0 = No limit):", min_value=0, max_value=240, value=0, step=5)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Section 4: Cached Local RAG Agent
# ---------------------------------------------------------------------------
from rag_engine import RecipeRAGAgent

@st.cache_resource(show_spinner=False)
def get_cached_rag_agent(doc_path: str, key: str):
    """Initializes and caches Chroma vector store with Local ONNX Embeddings."""
    if not key or not doc_path or not os.path.exists(doc_path):
        return None, "No document loaded."
    
    agent = RecipeRAGAgent(api_key=key)
    chunk_count = agent.index_documents([doc_path])
    summary = agent.get_document_overview()
    return agent, summary

# ---------------------------------------------------------------------------
# Section 5: Main Application View
# ---------------------------------------------------------------------------
st.markdown('<div class="title-text">🍳 Document Q&A RAG Agent for Recipe Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Problem Statement 8: Intelligent document retrieval, pantry-aware dietary adaptation, and step-by-step cooking guidance.</div>', unsafe_allow_html=True)

# Status Overview (Native Streamlit components for 100% theme compatibility)
rag_agent = None
if selected_doc_path and os.path.exists(selected_doc_path) and api_key:
    with st.spinner(f"📖 Indexing `{os.path.basename(selected_doc_path)}`..."):
        rag_agent, doc_summary = get_cached_rag_agent(selected_doc_path, api_key)
    
    # Active Settings Summary Bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📖 Document", os.path.basename(selected_doc_path), f"{rag_agent.total_chunks} Chunks")
    c2.metric("🥗 Diet Filter", diet)
    c3.metric("👥 Servings", f"{servings} Portions")
    c4.metric("🥕 Pantry Items", f"{len(pantry.split(',')) if pantry.strip() else 0} Specified")

    if pantry.strip():
        st.info(f"🥕 **Active Pantry Ingredients:** `{pantry.strip()}` — *The AI Chef will prioritize these items for substitutions!*")

    with st.expander("ℹ️ About this Document (AI Analysis)", expanded=False):
        st.markdown(doc_summary)
else:
    if not api_key:
        st.error("🔑 Please provide a Google Gemini API Key in the left sidebar.")
    else:
        st.warning("⚠️ No recipe document selected.")

# Initial Welcome Message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I am your AI Recipe & Document Q&A Assistant.**\n\nI can retrieve recipes from your uploaded document, intelligently substitute ingredients using items in your pantry, adapt to dietary constraints, calculate nutrition, and generate a shopping list.\n\n*Ask any cooking question or click one of the example queries below!*"
        }
    ]

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Document Sources"):
                for idx, src in enumerate(msg["sources"]):
                    page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Document Segment"
                    st.markdown(f"**Source {idx+1} ({src.get('file', 'Cookbook')} - {page_str}):**")
                    st.caption(src.get("content", ""))

# Quick Prompt Suggestions
st.markdown("---")
st.markdown("##### 💡 Example Questions:")
qc1, qc2, qc3, qc4 = st.columns(4)
quick_query = None

if qc1.button("📋 Summarize all recipes in book"):
    quick_query = "What recipes and dishes are in this cookbook? Summarize them."
if qc2.button("🍎 Adapt with Pantry Items"):
    quick_query = "How can I adapt a recipe from this book using the ingredients in my pantry?"
if qc3.button("🍪 Sugar-Free Dessert"):
    quick_query = "Find a dessert or snack recipe in this book and adapt it to be sugar-free."
if qc4.button("🍲 Healthy Soup / Main Dish"):
    quick_query = "Give me a healthy soup or skillet recipe from this book scaled for 4 servings with nutrition facts."

# Chat Input
user_query = st.chat_input("Ask a recipe question, requested substitutions, or cooking steps...") or quick_query

if user_query:
    if not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
    elif not rag_agent:
        st.error("Please select or upload a document first.")
    else:
        # User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant Message
        with st.chat_message("assistant"):
            with st.spinner("🤖 Consulting recipe document & personalizing cooking guidance..."):
                try:
                    time_limit = max_time if max_time > 0 else None
                    res = rag_agent.query(
                        question=user_query,
                        dietary_restriction=diet,
                        available_ingredients=pantry,
                        servings=servings,
                        max_time_mins=time_limit,
                        cuisine_preference=cuisine
                    )
                    
                    answer = res["answer"]
                    sources = res.get("sources", [])
                    
                    st.markdown(answer)
                    
                    # Download Recipe Button
                    st.download_button(
                        label="📥 Download Recipe (Text)",
                        data=answer,
                        file_name="personalized_recipe.txt",
                        mime="text/plain"
                    )

                    if sources:
                        with st.expander("🔍 View Retrieved Document Sources"):
                            for idx, src in enumerate(sources):
                                page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Document Segment"
                                st.markdown(f"**Source {idx+1} ({src.get('file', 'Cookbook')} - {page_str}):**")
                                st.caption(src.get("content", ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as ex:
                    st.error(f"Error querying recipe agent: {str(ex)}")
