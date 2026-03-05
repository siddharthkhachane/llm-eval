import requests


API_URL = "http://localhost:8000/generate"
SAMPLE_PROMPTS = [
    "In exactly eight words, describe the feeling of solving a hard bug.",
    "Give a one-sentence analogy for neural networks using a coffee shop.",
    "Write a two-line sci-fi micro-story featuring a lost satellite and hope.",
]


def main() -> None:
    for prompt in SAMPLE_PROMPTS:
        print(f"Prompt: {prompt}")
        try:
            response = requests.post(API_URL, json={"prompt": prompt}, timeout=120)
            response.raise_for_status()
            data = response.json()
            model_response = data.get("response", "<missing response field>")
            print(f"Response: {model_response}\n")
        except requests.exceptions.RequestException as exc:
            print(f"Response: Request failed: {exc}\n")
        except ValueError:
            print("Response: Invalid JSON returned by server.\n")


if __name__ == "__main__":
    main()
