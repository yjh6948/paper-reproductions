"""Minimal Gemini 2.5 Flash REST client
guidance: CLI ignores -m so call REST directly; urllib only, 3s pacing,
429 backoff via retryDelay."""
import json
import os
import time
import urllib.request
import urllib.error

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

_last_call = 0.0


def generate(prompt: str, max_retries: int = 5) -> str:
    global _last_call
    if API_KEY is None:
        raise RuntimeError("GEMINI_API_KEY not set")

    elapsed = time.time() - _last_call
    if elapsed < 3.0:
        time.sleep(3.0 - elapsed)

    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            _last_call = time.time()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                body_txt = e.read().decode(errors="ignore")
                delay = 5.0
                try:
                    err = json.loads(body_txt)
                    for d in err.get("error", {}).get("details", []):
                        if "retryDelay" in d:
                            delay = float(d["retryDelay"].rstrip("s"))
                except Exception:
                    pass
                print(f"  [429] backing off {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Gemini API: exceeded retries")
