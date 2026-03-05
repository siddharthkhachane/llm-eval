from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import uvicorn


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    response: str


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    payload = {
        "model": MODEL_NAME,
        "prompt": request.prompt,
        "stream": False,
    }

    try:
        ollama_response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        ollama_response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach Ollama at {OLLAMA_URL}: {exc}",
        ) from exc

    try:
        data = ollama_response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned non-JSON response.",
        ) from exc

    model_output = data.get("response")
    if not isinstance(model_output, str):
        raise HTTPException(
            status_code=502,
            detail="Ollama response did not include a valid 'response' field.",
        )

    return GenerateResponse(response=model_output.strip())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
