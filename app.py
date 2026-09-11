"""
=============================================================================
Document Q&A RAG Agent for Recipe Generator (Problem Statement No. 8)
Supports Dual Mode:
  1) Direct Local AI RAG Engine (Gemini 2.5 Flash + ChromaDB + LangChain)
  2) Langflow Flow Integration (Flow ID: 26d98697-ae15-4fad-ba76-5e1432bda889)
With Auto-Fallback on Timeout
=============================================================================
"""

import os
import time
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Import the RAG Engine
from rag_engine import RecipeRAGAgent

# ---------------------------------------------------------------------------
# Section A: Configuration & Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recipe Generator RAG Agent",
    page_icon="🍳",
    layout="wide"
)

DEFAULT_LANGFLOW_URL = "http://localhost:7860/api/v1/run/26d98697-ae15-4fad-ba76-5e1432bda889"
DEFAULT_API_TOKEN = os.getenv("LANGFLOW_API_TOKEN", "sk-2Cy61yYiO3C6gaVM3776VgDWXon5X41s6p62YUJloSc")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Section B: Cached Local RAG Engine Initialization
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_rag_agent(cookbook_path: str):
    """Initializes and caches the Recipe RAG Agent with indexed documents."""
    agent = RecipeRAGAgent(api_key=GOOGLE_API_KEY)
    if os.path.exists(cookbook_path):
        agent.index_documents([cookbook_path])
    return agent

# ---------------------------------------------------------------------------
# Section C: Langflow API Helper with Safe Parsing & Timeout Handling
# ---------------------------------------------------------------------------
def extract_langflow_response(data: dict) -> str:
    """Extract message string from Langflow output structure."""
    if not isinstance(data, dict):
        return str(data)

    try:
        outputs = data.get("outputs", [])
        if outputs and isinstance(outputs, list):
            first_output = outputs[0]
            nested = first_output.get("outputs", [])
            if nested and isinstance(nested, list):
                results = nested[0].get("results", {})
                message = results.get("message", {})
                if isinstance(message, dict):
                    if "data" in message and isinstance(message["data"], dict) and "text" in message["data"]:
                        return message["data"]["text"]
                    if "text" in message:
                        return message["text"]
                artifacts = nested[0].get("artifacts", {})
                if isinstance(artifacts, dict) and "message" in artifacts:
                    return str(artifacts["message"])
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Response parsing error: {e}"


def query_langflow(prompt: str, api_url: str, api_token: str, timeout_sec: int = 120) -> dict:
    """Query the Langflow flow endpoint with configured timeout."""
    headers = {"Content-Type": "application/json"}
    if api_token and api_token.strip():
        headers["x-api-key"] = api_token.strip()

    payload = {
        "input_value": prompt,
        "output_type": "chat",
        "input_type": "chat",
        "tweaks": {}
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout_sec)
        if response.status_code == 200:
            return {"success": True, "answer": extract_langflow_response(response.json())}
        else:
            return {"success": False, "error": f"Langflow Error ({response.status_code}): {response.text}"}
    except requests.exceptions.Timeout:
        return {"success": False, "timeout": True, "error": f"Langflow request timed out after {timeout_sec}s."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot reach Langflow on port 7860. Make sure the server is active."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Section D: Sidebar Setup & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Agent Engine Mode")
    engine_mode = st.radio(
        "Select Execution Engine:",
        ["⚡ Direct AI RAG Engine (Fast & Reliable)", "🌐 Langflow Server Flow (Port 7860)"],
        index=0,
        help="Direct engine runs instantly with Gemini 2.5 Flash + ChromaDB. Langflow flow calls your visual canvas."
    )

    if "Langflow" in engine_mode:
        st.caption("Connected to Langflow Flow ID: `26d98697-ae15-4fad-ba76-5e1432bda889`")
        api_url = st.text_input("Langflow Endpoint", value=DEFAULT_LANGFLOW_URL)
        api_token = st.text_input("Application Token", value=DEFAULT_API_TOKEN, type="password")
        timeout_val = st.slider("Request Timeout (Seconds)", min_value=30, max_value=180, value=120)
        auto_fallback = st.checkbox("Auto-fallback to Direct RAG on timeout", value=True)
    else:
        api_url = DEFAULT_LANGFLOW_URL
        api_token = DEFAULT_API_TOKEN
        timeout_val = 120
        auto_fallback = True

    st.divider()
    st.header("📚 Recipe Knowledge Base")
    cookbook_file = "c:/Ananya/Recipe_Generator/cookbook.txt"
    uploaded_file = st.file_uploader("Upload Cookbook (PDF / TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        save_path = "c:/Ananya/Recipe_Generator/cookbook.txt" if uploaded_file.name.endswith('.txt') else "c:/Ananya/Recipe_Generator/cookbook.pdf"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f" Uploaded `{uploaded_file.name}`!")
        st.cache_resource.clear()
        cookbook_file = save_path

    st.divider()
    st.header("🎯 Personalization Filters")
    diet = st.selectbox("Dietary Preference", ["None", "Sugar-Free", "Vegan", "Gluten-Free", "Keto / Low-Carb", "Dairy-Free", "High-Protein"])
    servings = st.slider("Servings", min_value=1, max_value=12, value=4)
    pantry = st.text_input("Available Pantry Ingredients", placeholder="e.g. bananas, oats, cocoa powder")

# ---------------------------------------------------------------------------
# Section E: Main Interface
# ---------------------------------------------------------------------------
st.markdown("## 🍳 Document Q&A RAG Agent for Recipe Generator")
st.markdown("**Problem Statement No. 8:** Intelligent recipe search, dietary adaptation, step-by-step cooking instructions, nutritional estimation, and shopping lists.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I am your AI Recipe Agent.**\n\nI can retrieve recipes from your cookbook documents, adapt them for your dietary needs (Sugar-free, Vegan, Gluten-free, Keto), scale servings, calculate nutritional facts, and generate a shopping checklist.\n\n*Try an example below or ask any cooking question!*"
        }
    ]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick Prompt Buttons
st.markdown("---")
st.markdown("##### 💡 Example Queries:")
col1, col2, col3 = st.columns(3)
quick_query = None
if col1.button("🍫 Sugar-free Chocolate Cake"):
    quick_query = "How can I make a sugar-free version of the chocolate cake from the cookbook?"
if col2.button("🍌 Vegan Banana Bread Guide"):
    quick_query = "Give me step-by-step instructions and nutrition facts for Vegan Banana Bread."
if col3.button("🍗 Keto Herb Tuscan Chicken"):
    quick_query = "How do I make the Creamy Tuscan Chicken keto-friendly and what is its shopping list?"

user_query = st.chat_input("Ask a recipe question...") or quick_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        answer = None
        
        # Mode 1: Langflow Flow
        if "Langflow" in engine_mode:
            with st.spinner("🤖 Calling Langflow Flow (Port 7860)..."):
                lf_result = query_langflow(user_query, api_url, api_token, timeout_sec=timeout_val)
                if lf_result.get("success"):
                    answer = lf_result["answer"]
                else:
                    if lf_result.get("timeout") and auto_fallback:
                        st.warning("⚠️ Langflow server took longer than expected. Automatically generating response via Direct AI RAG Engine...")
                    else:
                        st.error(lf_result.get("error", "Unknown Langflow error"))
                        if auto_fallback:
                            st.info("🔄 Falling back to Direct AI RAG Engine...")

        # Mode 2: Direct Local RAG Engine (or fallback)
        if answer is None:
            with st.spinner("🤖 Consulting Recipe Knowledge Base & Gemini AI..."):
                try:
                    rag_agent = get_rag_agent(cookbook_file)
                    res = rag_agent.query(
                        question=user_query,
                        dietary_restriction=diet,
                        available_ingredients=pantry,
                        servings=servings
                    )
                    answer = res["answer"]
                except Exception as ex:
                    st.error(f"Error generating recipe: {ex}")
                    answer = f"Sorry, could not process the recipe query: {ex}"

        if answer:
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
