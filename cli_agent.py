import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://localhost:7860/api/v1/run/26d98697-ae15-4fad-ba76-5e1432bda889")
API_TOKEN = os.getenv("LANGFLOW_API_TOKEN", "sk-2Cy61yYiO3C6gaVM3776VgDWXon5X41s6p62YUJloSc")

def extract_response(data: dict) -> str:
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
        return f"Error: {e}"

def generate_recipe_answer(prompt: str) -> str:
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
            return f"Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Connection error: {str(e)}"

if __name__ == "__main__":
    print("Recipe Assistant ready. Type a recipe question, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            if not user_input:
                continue

            answer = generate_recipe_answer(user_input)
            print(f"\nAssistant:\n{answer}\n")

        except KeyboardInterrupt:
            print("\nStopped.")
            break
