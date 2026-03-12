"""
Multimodal AI - Gemini. Dynamic prompt, proper result, fast.
"""
import base64
import os


def ask_gemini(
    frames: list[dict],
    prompt: str,
    api_key: str = None,
    model_name: str = "gemini-1.5-flash",
) -> str:
    """
    Video frames + user prompt. AI prompt ke hisaab se accurate detect/answer.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install: pip install google-generativeai")

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY missing. Add in .env")

    genai.configure(api_key=key)

    parts = []
    for f in frames:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64decode(f["base64"]),
            }
        })

    # Clear, direct prompt - AI ko exactly pata kya karna hai
    system_prompt = f"""You are a precise video analyzer. These images are sequential frames from a video (chronological order).

USER'S REQUEST (follow exactly):
{prompt}

RULES:
- Answer ONLY based on what you see in the frames
- Be specific: colors, numbers, types, positions
- For counts: give exact numbers
- If not visible, say "Not visible in the video"
- Keep response concise and accurate
- No extra explanation unless asked"""

    parts.append({"text": system_prompt})
    model = genai.GenerativeModel(model_name)

    # Faster + consistent: low temp, limited tokens
    config = {"max_output_tokens": 1024, "temperature": 0.2}
    response = model.generate_content(parts, generation_config=config)
    return response.text if response.text else "No response generated."
