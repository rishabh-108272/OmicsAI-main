import os
from dotenv import load_dotenv # <--- ADD THIS LINE
load_dotenv() # <--- ADD THIS LINE: It will look for a .env file in the current or parent directory

import google.generativeai as genai

# 1) Configure Gemini API key
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)

genai.configure(api_key=GEMINI_API_KEY)

# 2) Choose Gemini model
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

print(f"[LLM] Using Gemini model: {MODEL_NAME}")


def run_llm(prompt: str) -> str:
    """
    Unified wrapper to call Gemini and return plaintext.
    Uses streaming and concatenates all chunk.text values.
    """

    # Safety: cap huge prompts
    MAX_INPUT_CHARS = 4000
    if len(prompt) > MAX_INPUT_CHARS:
        prompt = prompt[:MAX_INPUT_CHARS]

    try:
        # Use streaming so we can accumulate chunk.text even if
        # the final aggregated response is weird / has no parts.
        stream = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.6,
                "top_p": 0.9,
            },
            stream=True,
        )
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        return f"[LLM ERROR] {type(e).__name__}: {e}"

    collected = []

    try:
        for chunk in stream:
            # Newer SDKs expose chunk.text for each streamed delta
            text_piece = getattr(chunk, "text", None)
            if text_piece:
                collected.append(text_piece)
                continue

            # Fallback: try to access candidates / parts if .text is missing
            candidates = getattr(chunk, "candidates", None)
            if not candidates:
                continue

            cand = candidates[0]
            content = getattr(cand, "content", None)
            if not content:
                continue

            parts = getattr(content, "parts", None)
            if not parts:
                continue

            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    collected.append(t)
    except Exception as e:
        # Even if streaming breaks mid-way, return whatever we collected
        print(f"[LLM STREAM ERROR] {type(e).__name__}: {e}")

    if not collected:
        # Absolute last fallback if nothing came through
        return "[LLM ERROR] Gemini returned no text in streamed chunks."

    return "".join(collected).strip()
