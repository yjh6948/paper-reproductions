import json
import math
import os
import time
import urllib.error
import urllib.request

API_KEY = os.environ["GEMINI_API_KEY"]
FLASH_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}"
EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={}"


def _request(url, body, retries=5):
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  429, backing off {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("exhausted retries")


def generate(prompt, json_mode=True):
    url = FLASH_URL.format(API_KEY)
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    data = _request(url, body)
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    time.sleep(7)
    return json.loads(text) if json_mode else text


def embed(text):
    url = EMBED_URL.format(API_KEY)
    body = {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}}
    data = _request(url, body)
    time.sleep(7)
    return data["embedding"]["values"]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)
