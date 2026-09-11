"""
=============================================================================
🍳 Document Q&A RAG Agent for Recipe Generator (Problem Statement No. 8)
Fully Deployable Streamlit Web Application
- Supports Multi-page PDFs (cookbook_pdf.pdf) & TXT Cookbooks
- Zero Quota Errors (Local Fast Embeddings)
- Google Gemini 2.5 Flash for Chef Reasoning
- Cross-Platform & Streamlit Cloud Deployable
=============================================================================
"""

import os
import sys
import glob
import json
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

# Custom Styling for polished UI
st.markdown("""
<style>
    .main-title {
        font-family: 'Arial', sans-serif;
        color: #E63946;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #457B9D;
        font-size: 16px;
        margin-bottom: 18px;
    }
    .badge {
        background-color: #E9ECEF;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 13px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section B: Retrieve API Key (Cloud Secrets or .env or User Input)
# ---------------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Section C: Sidebar - Knowledge Base & Agent Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")
    
    # API Key check
    if not api_key:
        user_key = st.text_input("🔑 Enter Google API Key", type="password", help="Needed for Gemini model inference")
        if user_key:
            api_key = user_key
        else:
            st.warning("Please enter your Google API key to generate recipes.")

    st.divider()
    st.markdown("### 📚 Cookbook Knowledge Base")
    
    # Find existing cookbooks in directory
    existing_pdfs = glob.glob(os.path.join(BASE_DIR, "*.pdf"))
    existing_txts = glob.glob(os.path.join(BASE_DIR, "*.txt"))
    all_cookbooks = existing_pdfs + existing_txts
    
    cookbook_options = [os.path.basename(f) for f in all_cookbooks]
    default_idx = 0
    if "cookbook_pdf.pdf" in cookbook_options:
        default_idx = cookbook_options.index("cookbook_pdf.pdf")
    
    selected_cookbook_name = st.selectbox(
        "Select Active Cookbook:",
        cookbook_options if cookbook_options else ["None found"],
        index=default_idx if cookbook_options else 0,
        help="Choose a pre-loaded cookbook or upload your own below"
    )
    
    selected_cookbook_path = os.path.join(BASE_DIR, selected_cookbook_name) if cookbook_options else None

    # Upload custom cookbook
    uploaded_file = st.file_uploader("Upload New Cookbook (PDF / TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        save_path = os.path.join(BASE_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f" Uploaded `{uploaded_file.name}`!")
        selected_cookbook_path = save_path
        st.cache_resource.clear()

    st.divider()
    st.markdown("### 🎯 Personalization Filters")
    diet = st.selectbox(
        "Dietary Preference:",
        ["None", "Sugar-Free", "Vegan", "Vegetarian", "Gluten-Free", "Keto / Low-Carb", "Dairy-Free", "High-Protein", "Heart-Healthy"]
    )
    servings = st.slider("Target Servings:", min_value=1, max_value=12, value=4)
    pantry = st.text_input("Available Pantry Ingredients (Optional):", placeholder="e.g. chicken, black beans, salsa")
    max_time = st.number_input("Max Cooking Time (Mins, Optional):", min_value=0, max_value=240, value=0, step=5)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Section D: Cached RAG Agent Builder
# ---------------------------------------------------------------------------
from rag_engine import RecipeRAGAgent

@st.cache_resource(show_spinner=False)
def load_rag_pipeline(cookbook_path: str, key: str):
    """Initializes and caches the local vector store for the selected cookbook."""
    if not key:
        return None
    agent = RecipeRAGAgent(api_key=key)
    if cookbook_path and os.path.exists(cookbook_path):
        agent.index_documents([cookbook_path])
    return agent

# ---------------------------------------------------------------------------
# Section E: Main Interface & Chat Loop
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🍳 Document Q&A RAG Agent for Recipe Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Problem Statement 8: AI-powered culinary knowledge retrieval, personalized dietary adaptation, and smart cooking guidance.</div>', unsafe_allow_html=True)

# Status Badge
if selected_cookbook_path and os.path.exists(selected_cookbook_path):
    st.markdown(f"📖 **Active Cookbook:** `{os.path.basename(selected_cookbook_path)}` | 🎯 **Diet Filter:** `{diet}` | 👥 **Servings:** `{servings}`")
else:
    st.info("Please select or upload a cookbook to begin.")

# Initialize Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I am your AI Recipe & Cooking Assistant.**\n\nI can retrieve recipes from your cookbook, adapt them to dietary needs (Sugar-Free, Vegan, Gluten-Free, Keto), scale ingredients for your family, calculate nutrition, and generate shopping lists.\n\n*Click one of the example queries below or ask your own question!*"
        }
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Cookbook Citations"):
                for idx, src in enumerate(message["sources"]):
                    page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Text"
                    st.markdown(f"**Source {idx+1} ({src.get('file', 'Cookbook')} - {page_str}):**")
                    st.caption(src.get("content", ""))

# Quick Prompt Pills based on cookbook
st.markdown("---")
st.markdown("##### 💡 Example Questions for Active Cookbook:")
col1, col2, col3, col4 = st.columns(4)
quick_query = None

if "cookbook_pdf" in (selected_cookbook_name or ""):
    if col1.button("🍲 Chicken Enchilada Soup"):
        quick_query = "Give me the recipe for Chicken Enchilada Soup from the cookbook and scale it for 4."
    if col2.button("🥗 Vegan Cowboy Salad"):
        quick_query = "How can I make the Cowboy Salad from the cookbook vegan and high-fiber?"
    if col3.button("🍪 Healthy Oatmeal Cookies"):
        quick_query = "Show me the Oatmeal Cookies recipe adapted to be sugar-free with estimated nutrition."
    if col4.button("🥤 Dairy-Free Smoothie"):
        quick_query = "How can I make the Pumpkin Smoothie dairy-free, and what is the shopping list?"
else:
    if col1.button("🍫 Sugar-free Chocolate Cake"):
        quick_query = "How can I make a sugar-free version of the chocolate cake from the cookbook?"
    if col2.button("🍌 Vegan Banana Bread"):
        quick_query = "Give me step-by-step instructions and nutrition facts for Vegan Banana Bread."
    if col3.button("🍗 Tuscan Chicken"):
        quick_query = "How do I prepare the Creamy Tuscan Chicken and scale it for 2 people?"
    if col4.button("🛒 Shopping List & Substitutions"):
        quick_query = "Adapt the chocolate cake for a dairy-free diet and give me a complete shopping list."

user_query = st.chat_input("Ask for a recipe, substitutions, or cooking steps...") or quick_query

if user_query:
    if not api_key:
        st.error("Please enter a Google API Key in the sidebar to run the agent.")
    elif not selected_cookbook_path or not os.path.exists(selected_cookbook_path):
        st.error("No valid cookbook selected. Please select or upload a cookbook.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Searching cookbook knowledge base and personalizing recipe..."):
                try:
                    rag_agent = load_rag_pipeline(selected_cookbook_path, api_key)
                    if rag_agent is None:
                        st.error("Failed to initialize RAG agent. Please verify your API key.")
                    else:
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
                            with st.expander("🔍 View Retrieved Cookbook Citations"):
                                for idx, src in enumerate(sources):
                                    page_str = f"Page {src['page'] + 1}" if src.get('page') is not None else "Text"
                                    st.markdown(f"**Source {idx+1} ({src.get('file', 'Cookbook')} - {page_str}):**")
                                    st.caption(src.get("content", ""))

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                except Exception as ex:
                    st.error(f"Error generating recipe: {str(ex)}")
