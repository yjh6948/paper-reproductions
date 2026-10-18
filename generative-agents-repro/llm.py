import json
import os
import time
import urllib.error
import urllib.request

MODEL = "gemini-2.5-flash"
API_KEY = os.environ.get("GEMINI_API_KEY")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"


def call_gemini(prompt: str, retries: int = 3) -> str:
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Gemini call failed after retries")


def score_importance(memory_content: str) -> float:
    prompt = (
        "On a scale of 1 to 10, where 1 is purely mundane "
        "(e.g., brushing teeth, making the bed) and 10 is extremely poignant "
        "(e.g., a breakup, a college acceptance), rate the likely poignancy "
        f"of the following memory. Respond with only a single number.\n\nMemory: {memory_content}"
    )
    text = call_gemini(prompt)
    digits = "".join(c for c in text if c.isdigit() or c == ".")
    try:
        return max(1.0, min(10.0, float(digits)))
    except ValueError:
        return 5.0


def generate_reflection_questions(recent_contents: list, n: int = 3) -> list:
    joined = "\n".join(f"- {c}" for c in recent_contents)
    prompt = (
        f"Given only the information below, what are the {n} most salient "
        "high-level questions we can answer about the subject in these statements? "
        "Respond as a numbered list, one question per line.\n\n" + joined
    )
    text = call_gemini(prompt)
    lines = [l.strip(" -0123456789.") for l in text.splitlines() if l.strip()]
    return lines[:n]


def answer_reflection_question(question: str, evidence_contents: list) -> str:
    joined = "\n".join(f"- {c}" for c in evidence_contents)
    prompt = (
        f"Question: {question}\n\nRelevant memories:\n{joined}\n\n"
        "In one or two sentences, answer the question with a higher-level insight "
        "grounded in the memories above."
    )
    return call_gemini(prompt)
