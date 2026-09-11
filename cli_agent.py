import os
import json
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Section A: Environment Setup & Constants
# ---------------------------------------------------------------------------
load_dotenv()

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://localhost:7860/api/v1/run/26d98697-ae15-4fad-ba76-5e1432bda889")
API_TOKEN = os.getenv("LANGFLOW_API_TOKEN", "sk-2Cy61yYiO3C6gaVM3776VgDWXon5X41s6p62YUJloSc")

# ---------------------------------------------------------------------------
# Section B: Safe Extraction Helper
# ---------------------------------------------------------------------------
def extract_response(data: dict) -> str:
    """Extract agent response from Langflow JSON payload."""
    if not isinstance(data, dict):
        return str(data)

    try:
        outputs = data.get("outputs", [])
        if outputs and isinstance(outputs, list):
            nested = outputs[0].get("outputs", [])
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
        return f"Error extracting response: {e}"


# ---------------------------------------------------------------------------
# Section C: Query Function
# ---------------------------------------------------------------------------
def generate_recipe_answer(prompt: str) -> str:
    """Call the Langflow Recipe RAG agent with user query."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_TOKEN
    }
    payload = {
        "input_value": prompt,
        "output_type": "chat",
        "input_type": "chat"
    }

    try:
        response = requests.post(LANGFLOW_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return extract_response(response.json())
        else:
            return f"❌ Langflow API Error ({response.status_code}): {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to Langflow. Please verify Langflow is running at http://localhost:7860."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ---------------------------------------------------------------------------
# Section D: Interactive Terminal Run Loop (Matching Reference Format)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("🍳 Document Q&A RAG Agent for Recipe Generator (Problem Statement 8)")
    print("=" * 65)
    print("Connected to Langflow flow: 26d98697-ae15-4fad-ba76-5e1432bda889")
    print("Type a recipe question (e.g., 'How to make sugar-free chocolate cake?')")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("\n👋 Goodbye! Happy cooking!")
                break
            if not user_input:
                continue

            print("\n🤖 Consulting Recipe Agent...")
            answer = generate_recipe_answer(user_input)
            print(f"\n🍳 Recipe Agent:\n{answer}\n")
            print("-" * 65)

        except KeyboardInterrupt:
            print("\n\n👋 Stopped. Goodbye!")
            break
