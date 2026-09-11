"""
=============================================================================
🍳 Document Q&A RAG Agent for Recipe Generator
Problem Statement No. 8 — Full Production Streamlit Application
=============================================================================
"""

import os
import glob
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Section 1: Page Configuration & Custom UI Theme
# ---------------------------------------------------------------------------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="AI Recipe Generator | Problem Statement 8",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #E63946;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 2px;
    }
    .sub-header {
        color: #457B9D;
        font-size: 15px;
        margin-bottom: 12px;
    }
    .badge-card {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .recipe-section {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
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
# Section 3: Sidebar - Knowledge Base & Agent Personalization
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Setup")
    
    # API Key Input
    if not api_key:
        input_key = st.text_input("🔑 Google Gemini API Key", type="password", help="Enter your Gemini API key")
        if input_key:
            api_key = input_key
        else:
            st.warning("⚠️ Google API Key is required. Please paste your key above.")

    st.divider()
    st.markdown("### 📚 1. Recipe Document Knowledge Base")
    st.caption("Upload or select **any PDF / TXT document** (cookbooks, notes, recipes).")

    # Discover local cookbooks
    local_pdfs = glob.glob(os.path.join(BASE_DIR, "*.pdf"))
    local_txts = glob.glob(os.path.join(BASE_DIR, "*.txt"))
    all_files = [os.path.basename(f) for f in (local_pdfs + local_txts)]
    
    default_idx = 0
    if "cookbook_pdf.pdf" in all_files:
        default_idx = all_files.index("cookbook_pdf.pdf")
    elif "cookbook.txt" in all_files:
        default_idx = all_files.index("cookbook.txt")

    selected_doc_name = st.selectbox(
        "Select Active Cookbook:",
        all_files if all_files else ["None"],
        index=default_idx if all_files else 0
    )
    
    selected_doc_path = os.path.join(BASE_DIR, selected_doc_name) if all_files else None

    # Drag & Drop Uploader
    uploaded_file = st.file_uploader("Upload New Document (PDF / TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        save_path = os.path.join(BASE_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f" Uploaded `{uploaded_file.name}`!")
        selected_doc_path = save_path
        st.cache_resource.clear()

    st.divider()
    st.markdown("### 🎯 2. Agentic Personalization Layer")
    diet = st.selectbox(
        "🥗 Dietary Restriction:",
        ["None", "Sugar-Free", "Vegan", "Vegetarian", "Gluten-Free", "Keto / Low-Carb", "Dairy-Free", "High-Protein", "Low-Sodium", "Nut-Free"]
    )
    servings = st.slider("👥 Target Servings:", min_value=1, max_value=12, value=4)
    pantry = st.text_input("🥕 Available Pantry Ingredients (Optional):", placeholder="e.g., chicken, black beans, salsa")
    cuisine = st.selectbox("🌶️ Cuisine / Flavor Preference:", ["Any", "Mexican", "Italian", "American Comfort", "Asian", "Mediterranean", "Kid-Friendly"])
    max_time = st.number_input("⏱️ Max Cooking Time (Mins, Optional):", min_value=0, max_value=240, value=0, step=5)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Section 4: Cached Local RAG Pipeline Builder
# ---------------------------------------------------------------------------
from rag_engine import RecipeRAGAgent

@st.cache_resource(show_spinner=False)
def get_cached_rag_agent(doc_path: str, key: str):
    """Initializes Chroma with 100% local ONNX embeddings (Zero API quota usage)."""
    if not key or not doc_path or not os.path.exists(doc_path):
        return None, "No document loaded."
    
    agent = RecipeRAGAgent(api_key=key)
    chunk_count = agent.index_documents([doc_path])
    summary = agent.get_document_overview()
    return agent, summary

# ---------------------------------------------------------------------------
# Section 5: Main Application View
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🍳 Document Q&A RAG Agent for Recipe Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Problem Statement 8: Intelligent recipe document retrieval, personalized dietary adaptation, and smart cooking guidance.</div>', unsafe_allow_html=True)

# Status & Document Badge
rag_agent = None
if selected_doc_path and os.path.exists(selected_doc_path) and api_key:
    with st.spinner(f"📖 Indexing `{os.path.basename(selected_doc_path)}` into local vector store..."):
        rag_agent, doc_summary = get_cached_rag_agent(selected_doc_path, api_key)
    
    col_x, col_y, col_z = st.columns([2, 1, 1])
    with col_x:
        st.markdown(f"📄 **Active Document:** `{os.path.basename(selected_doc_path)}` ({rag_agent.total_chunks} indexed segments)")
    with col_y:
        st.markdown(f"🥗 **Diet Filter:** `{diet}`")
    with col_z:
        st.markdown(f"👥 **Servings:** `{servings}`")

    with st.expander("ℹ️ About this Document (AI Analysis)", expanded=False):
        st.info(doc_summary)
else:
    if not api_key:
        st.error("🔑 Please enter a valid Google Gemini API Key in the left sidebar.")
    else:
        st.warning("⚠️ No cookbook document selected. Please choose or upload a file in the sidebar.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I am your AI Recipe & Document Q&A Assistant.**\n\nI can retrieve recipes from your uploaded cookbook, adapt ingredients for your dietary preferences (Sugar-Free, Vegan, Gluten-Free, Keto), scale servings, calculate nutrition, and generate a shopping list.\n\n*Click one of the example queries below or ask any cooking question!*"
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
                    st.markdown(f"**Excerpt {idx+1} ({src.get('file', 'Cookbook')} - {page_str}):**")
                    st.caption(src.get("content", ""))

# Quick Prompt Suggestions
st.markdown("---")
st.markdown("##### 💡 Example Questions:")
c1, c2, c3, c4 = st.columns(4)
quick_query = None

if c1.button("📋 What recipes are in this document?"):
    quick_query = "What recipes and dishes are in this cookbook? Summarize them."
if c2.button("🍲 Find a high-protein soup"):
    quick_query = "Find a healthy soup or main dish in this book and scale it for 4 servings."
if c3.button("🍪 Sugar-Free Dessert / Snack"):
    quick_query = "Find a dessert or snack recipe in this book and adapt it to be sugar-free."
if c4.button("🛒 Generate shopping list"):
    quick_query = "Pick a popular recipe from this book and generate a complete shopping checklist with estimated nutrition."

# Chat Input
user_query = st.chat_input("Ask a recipe question, requested substitutions, or cooking steps...") or quick_query

if user_query:
    if not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
    elif not rag_agent:
        st.error("Please select or upload a document first.")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Agent Response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Consulting recipe knowledge base and synthesizing cooking guidance..."):
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
                    
                    if sources:
                        with st.expander("🔍 View Retrieved Document Sources"):
                            for idx, src in enumerate(sources):
                                page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Document Segment"
                                st.markdown(f"**Excerpt {idx+1} ({src.get('file', 'Cookbook')} - {page_str}):**")
                                st.caption(src.get("content", ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as ex:
                    st.error(f"Error querying recipe agent: {str(ex)}")
