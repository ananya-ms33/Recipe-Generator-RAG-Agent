import os
import glob
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="AI Recipe Assistant",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .title-text {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .subtitle-text {
        font-size: 15px;
        opacity: 0.85;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Retrieve API key
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

# Sidebar configuration
with st.sidebar:
    st.markdown("### Settings")
    
    if not api_key:
        input_key = st.text_input("Google API Key", type="password", help="Enter your Gemini API key")
        if input_key:
            api_key = input_key
        else:
            st.warning("Please enter your Google API key to continue.")

    st.divider()
    st.markdown("### Recipe Documents")
    
    # Locate available documents
    local_pdfs = glob.glob(os.path.join(BASE_DIR, "*.pdf"))
    local_txts = glob.glob(os.path.join(BASE_DIR, "*.txt"))
    all_files = [os.path.basename(f) for f in (local_pdfs + local_txts)]
    
    default_idx = 0
    if "cookbook_pdf.pdf" in all_files:
        default_idx = all_files.index("cookbook_pdf.pdf")
    elif "cookbook.txt" in all_files:
        default_idx = all_files.index("cookbook.txt")

    selected_doc_name = st.selectbox(
        "Select Document:",
        all_files if all_files else ["None"],
        index=default_idx if all_files else 0
    )
    
    selected_doc_path = os.path.join(BASE_DIR, selected_doc_name) if all_files else None

    uploaded_file = st.file_uploader("Upload New Document (PDF / TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        save_path = os.path.join(BASE_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name}")
        selected_doc_path = save_path
        st.cache_resource.clear()

    st.divider()
    st.markdown("### Personalization")
    diet = st.selectbox(
        "Dietary Preference:",
        ["None", "Sugar-Free", "Vegan", "Vegetarian", "Gluten-Free", "Keto / Low-Carb", "Dairy-Free", "High-Protein", "Low-Sodium", "Nut-Free"]
    )
    servings = st.slider("Servings:", min_value=1, max_value=12, value=4)
    pantry = st.text_input(
        "Available Pantry Items:",
        placeholder="e.g. apples, oats, milk, flour",
        help="Type ingredients you currently have at home."
    )
    cuisine = st.selectbox("Cuisine Style:", ["Any", "Mexican", "Italian", "American Comfort", "Asian", "Mediterranean", "Kid-Friendly"])
    max_time = st.number_input("Max Cooking Time (Minutes, 0 = no limit):", min_value=0, max_value=240, value=0, step=5)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# RAG Pipeline helper
from rag_engine import RecipeRAGAgent

@st.cache_resource(show_spinner=False)
def load_agent(doc_path: str, key: str):
    if not key or not doc_path or not os.path.exists(doc_path):
        return None, "No document loaded."
    agent = RecipeRAGAgent(api_key=key)
    agent.index_documents([doc_path])
    summary = agent.get_document_overview()
    return agent, summary

# Main application view
st.markdown('<div class="title-text">🍳 Recipe Generator & Cooking Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Ask questions, adapt recipes for dietary needs, and generate cooking instructions.</div>', unsafe_allow_html=True)

rag_agent = None
if selected_doc_path and os.path.exists(selected_doc_path) and api_key:
    with st.spinner("Loading document..."):
        rag_agent, doc_summary = load_agent(selected_doc_path, api_key)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Document", os.path.basename(selected_doc_path))
    col2.metric("Diet", diet)
    col3.metric("Servings", f"{servings}")
    col4.metric("Pantry Items", f"{len([x for x in pantry.split(',') if x.strip()]) if pantry.strip() else 'Standard'}")

    if pantry.strip():
        st.info(f"Using pantry items: `{pantry.strip()}`")

    with st.expander("Document Overview", expanded=False):
        st.markdown(doc_summary)
else:
    if not api_key:
        st.error("Please enter a Google API Key in the sidebar.")
    else:
        st.warning("No recipe document selected.")

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your recipe assistant. Ask me anything about the loaded cookbook, or request personalized recipes with your dietary preferences and available ingredients."
        }
    ]

# Render chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Document Sources"):
                for idx, src in enumerate(msg["sources"]):
                    page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Text Segment"
                    st.markdown(f"**Source {idx+1} ({src.get('file', 'Document')} - {page_str}):**")
                    st.caption(src.get("content", ""))

# Example queries
st.markdown("---")
st.markdown("##### Example Queries:")
c1, c2, c3, c4 = st.columns(4)
sample_query = None

if c1.button("Summarize Recipes"):
    sample_query = "What recipes and dishes are in this cookbook? Summarize them."
if c2.button("Adapt with Pantry Items"):
    sample_query = "How can I adapt a recipe from this book using the ingredients in my pantry?"
if c3.button("Sugar-Free Dessert"):
    sample_query = "Find a dessert or snack recipe in this book and adapt it to be sugar-free."
if c4.button("Healthy Soup"):
    sample_query = "Give me a healthy soup or skillet recipe from this book scaled for 4 servings with nutrition facts."

# Chat input
user_query = st.chat_input("Ask a recipe question...") or sample_query

if user_query:
    if not api_key:
        st.error("Please provide a Google API Key in the sidebar.")
    elif not rag_agent:
        st.error("Please select a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating recipe..."):
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
                    
                    st.download_button(
                        label="Download Recipe",
                        data=answer,
                        file_name="recipe.txt",
                        mime="text/plain"
                    )

                    if sources:
                        with st.expander("View Document Sources"):
                            for idx, src in enumerate(sources):
                                page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Text Segment"
                                st.markdown(f"**Source {idx+1} ({src.get('file', 'Document')} - {page_str}):**")
                                st.caption(src.get("content", ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as ex:
                    st.error(f"Error: {str(ex)}")
