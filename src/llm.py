"""
llm.py — Interface to local Ollama LLM.

Requires Ollama running locally:
    ollama serve
    ollama pull llama3.2   (or mistral, gemma2, etc.)
"""

import requests
import json

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"   # change to "mistral" or "gemma2" if preferred

SYSTEM_PROMPT = """You are IndustrialCopilot, an expert AI assistant for manufacturing plant operations.
You analyze industrial sensor data and provide actionable insights for maintenance engineers.

Your responsibilities:
- Diagnose equipment anomalies from sensor readings (temperature, vibration, pressure, RPM, current)
- Identify root causes of machine failures
- Recommend preventive maintenance actions
- Flag critical safety concerns immediately
- Explain technical findings in clear, structured language

Always base your answers strictly on the provided sensor context. If the context doesn't contain 
enough information to answer confidently, say so clearly.

Format your responses with:
- A brief diagnosis summary
- Key evidence from sensor data
- Root cause analysis
- Recommended actions (prioritized)
"""


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    """Send a prompt to local Ollama and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": stream,
        "options": {
            "temperature": 0.2,   # low temp for factual industrial diagnostics
            "top_p": 0.9,
            "num_predict": 1024,
        }
    }

    try:
        if stream:
            return _stream_response(payload)
        else:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "❌ **Ollama not running.** Please start it with:\n"
            "```\nollama serve\nollama pull llama3.2\n```"
        )
    except Exception as e:
        return f"❌ LLM error: {str(e)}"


def _stream_response(payload: dict):
    """Generator that yields text chunks for streaming."""
    with requests.post(OLLAMA_URL, json={**payload, "stream": True}, stream=True, timeout=120) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break


def build_rag_prompt(question: str, context: str) -> str:
    """Combine retrieved context with the user question into a RAG prompt."""
    return f"""
### SENSOR DATA CONTEXT (retrieved from plant logs):
{context}

### ENGINEER'S QUESTION:
{question}

### YOUR ANALYSIS:
"""
