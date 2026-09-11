"""
=============================================================================
🍳 Document Q&A RAG Agent for Recipe Generator (Problem Statement No. 8)
General-Purpose Unstructured Recipe Assistant
- Reads ANY PDF / TXT document regardless of layout or structure
- Dynamic semantic retrieval via Local ONNX Embeddings (Zero Quota Errors)
- Google Gemini 2.5 Flash for culinary reasoning & dietary adaptation
- Fully deployable to Streamlit Community Cloud
=============================================================================
"""

import os
import glob
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Section A: Page & Environment Setup
# ---------------------------------------------------------------------------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="AI Recipe Generator | Problem Statement 8",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Arial', sans-serif;
        color: #D90429;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #2B2D42;
        font-size: 15px;
        margin-bottom: 16px;
    }
    .summary-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #D90429;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section B: API Key Discovery (Cloud Secrets, .env, or User Input)
# ---------------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Section C: Sidebar - Document Ingestion & Personalization Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")
    
    # API Key Input
    if not api_key:
        user_key = st.text_input("🔑 Google API Key", type="password", help="Required for Gemini inference")
        if user_key:
            api_key = user_key
        else:
            st.warning("Please enter your Google API key to activate the chef agent.")

    st.divider()
    st.markdown("### 📚 Ingest Recipe Document")
    st.caption("Upload **ANY** PDF or TXT document (cookbooks, unstructured notes, recipes).")

    # Discover local documents in project folder
    local_pdfs = glob.glob(os.path.join(BASE_DIR, "*.pdf"))
    local_txts = glob.glob(os.path.join(BASE_DIR, "*.txt"))
    all_files = [os.path.basename(f) for f in (local_pdfs + local_txts)]
    
    default_idx = 0
    if "cookbook_pdf.pdf" in all_files:
        default_idx = all_files.index("cookbook_pdf.pdf")
    elif "cookbook.txt" in all_files:
        default_idx = all_files.index("cookbook.txt")

    selected_doc_name = st.selectbox(
        "Select Active Document:",
        all_files if all_files else ["No document found"],
        index=default_idx if all_files else 0
    )
    
    selected_doc_path = os.path.join(BASE_DIR, selected_doc_name) if all_files else None

    # File Uploader for any arbitrary format
    uploaded_file = st.file_uploader("Upload New Document (PDF / TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        save_path = os.path.join(BASE_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f" Uploaded `{uploaded_file.name}` successfully!")
        selected_doc_path = save_path
        st.cache_resource.clear()

    st.divider()
    st.markdown("### 🎯 Personalization Filters")
    diet = st.selectbox(
        "Dietary Restriction:",
        ["None", "Sugar-Free", "Vegan", "Vegetarian", "Gluten-Free", "Keto / Low-Carb", "Dairy-Free", "High-Protein", "Low-Sodium"]
    )
    servings = st.slider("Target Servings:", min_value=1, max_value=12, value=4)
    pantry = st.text_input("Available Pantry Ingredients (Optional):", placeholder="e.g., chicken, beans, garlic")
    max_time = st.number_input("Max Cooking Time (Minutes, Optional):", min_value=0, max_value=240, value=0, step=5)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Section D: Cached General RAG Agent Builder
# ---------------------------------------------------------------------------
from rag_engine import RecipeRAGAgent

@st.cache_resource(show_spinner=False)
def load_rag_pipeline(doc_path: str, key: str):
    """Loads and indexes any unstructured recipe document into the vector store."""
    if not key or not doc_path or not os.path.exists(doc_path):
        return None, "No document loaded"
    
    agent = RecipeRAGAgent(api_key=key)
    chunk_count = agent.index_documents([doc_path])
    summary = agent.summarize_document()
    return agent, summary

# ---------------------------------------------------------------------------
# Section E: Main Interface
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🍳 Document Q&A RAG Agent for Recipe Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Problem Statement 8: Reads any unstructured recipe document (PDF/TXT), retrieves cooking instructions, and adapts recipes dynamically.</div>', unsafe_allow_html=True)

# Document Status & Dynamic Summary
if selected_doc_path and os.path.exists(selected_doc_path) and api_key:
    with st.spinner("📄 Reading and indexing document contents..."):
        rag_agent, doc_summary = load_rag_pipeline(selected_doc_path, api_key)
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"📖 **Active Document:** `{os.path.basename(selected_doc_path)}` ({rag_agent.total_chunks} indexed segments)")
    with col_b:
        st.markdown(f"🎯 **Diet:** `{diet}` | 👥 **Servings:** `{servings}`")
        
    with st.expander("ℹ️ About this Document (AI Analysis)", expanded=False):
        st.info(doc_summary)
else:
    rag_agent = None
    st.warning("⚠️ Please provide a Google API Key in the sidebar to activate the AI Agent.")

# Initialize Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I am your AI Recipe & Document Q&A Agent.**\n\nI can read **any unstructured recipe document or cookbook PDF** you provide, extract dishes, adapt ingredients for your diet, scale servings, and generate shopping lists.\n\n*Ask any question about your document below!*"
        }
    ]

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Document Sources"):
                for idx, src in enumerate(message["sources"]):
                    page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Text Segment"
                    st.markdown(f"**Excerpt {idx+1} ({src.get('file', 'Document')} - {page_str}):**")
                    st.caption(src.get("content", ""))

# Suggested Natural Language Queries
st.markdown("---")
st.markdown("##### 💡 Example Questions you can ask:")
col1, col2, col3, col4 = st.columns(4)
quick_query = None

if col1.button("📋 What recipes are in this document?"):
    quick_query = "What recipes and dishes are contained in this document? Summarize them."
if col2.button("🍲 Find a main course recipe"):
    quick_query = "Find a healthy main dish recipe from this document, and give me the full ingredients and instructions."
if col3.button("🥗 Make a recipe Vegan/Sugar-Free"):
    quick_query = "Find a popular dish from this document and adapt it to be vegan and sugar-free."
if col4.button("🛒 Generate shopping list for a recipe"):
    quick_query = "Pick a recipe from this document and generate a complete shopping checklist for 4 servings."

user_query = st.chat_input("Ask any recipe or cooking question from your document...") or quick_query

if user_query:
    if not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
    elif not rag_agent:
        st.error("Please select or upload a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Scanning document and synthesizing cooking guidance..."):
                try:
                    time_limit = max_time if max_time > 0 else None
                    res = rag_agent.query(
                        question=user_query,
                        dietary_restriction=diet,
                        available_ingredients=pantry,
                        servings=servings,
                        max_time_mins=time_limit
                    )
                    
                    answer = res["answer"]
                    sources = res.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("🔍 View Retrieved Document Sources"):
                            for idx, src in enumerate(sources):
                                page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Text Segment"
                                st.markdown(f"**Excerpt {idx+1} ({src.get('file', 'Document')} - {page_str}):**")
                                st.caption(src.get("content", ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as ex:
                    st.error(f"Error querying document: {str(ex)}")
