"""Score raw vs. humanized blog posts on 9 of SlopShape's (arXiv:2609.15369)
core structural features (Table 6), via real Gemini 2.5 Flash calls -- one
call per post, all 9 features together (documented simplification vs. the
paper's one-dimension-per-call scoring; see README).

Each feature has two forced-choice values, matching the paper's "LLM forced
to pick exactly one predefined option" design (Sec. 4.4). The 10th core
feature (Problem Placement, an ordinal position) is dropped -- its direction
isn't fully specified in the paper's Table 6, so it can't be turned into a
clean forced choice without guessing.
"""

import json
import os
import time
import urllib.request
import urllib.error

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

FEATURES = {
    "conclusion_behavior": {
        "question": "Does the post's final paragraph restate its main point or reframe/summarize the argument (rather than ending on a plain concrete fact or detail)?",
        "options": ["restates_or_reframes", "ends_on_plain_fact"],
        "ai_leaning": "restates_or_reframes",
    },
    "payoff_first_promise_location": {
        "question": "Is the post's main payoff or benefit promised directly in the title?",
        "options": ["title", "not_title"],
        "ai_leaning": "title",
    },
    "functional_stage_inventory": {
        "question": "Does the post include a distinct summary/synthesis stage (a paragraph whose main job is to recap or tie together the preceding points)?",
        "options": ["has_summary_stage", "no_summary_stage"],
        "ai_leaning": "has_summary_stage",
    },
    "external_participation_pathway": {
        "question": "Does the post include an explicit call to action or invitation for the reader to participate further (e.g. 'try this', 'let us know', a next step)?",
        "options": ["present", "absent"],
        "ai_leaning": "absent",
    },
    "legacy_vs_modern_contrast": {
        "question": "Does the post explicitly contrast an old/legacy way of doing things against a new/modern way?",
        "options": ["present", "absent"],
        "ai_leaning": "present",
    },
    "thesis_before_first_unit": {
        "question": "Is the post's main thesis or claim stated explicitly before the first substantive section/paragraph of content?",
        "options": ["yes", "no"],
        "ai_leaning": "yes",
    },
    "stakes_escalation": {
        "question": "Does the post escalate the stakes of the problem (emphasize how bad, urgent, or costly the problem is) before presenting the fix?",
        "options": ["present", "absent"],
        "ai_leaning": "present",
    },
    "primary_voice_identity": {
        "question": "Is the dominant voice of the post an editorial-explainer voice (an authority explaining/teaching a topic to the reader), as opposed to a first-person anecdotal or conversational voice?",
        "options": ["editorial_explainer", "other"],
        "ai_leaning": "editorial_explainer",
    },
    "article_length_class": {
        "question": "Is the post under 800 words?",
        "options": ["short_under_800", "800_or_more"],
        "ai_leaning": "800_or_more",
    },
}


def build_prompt(text):
    lines = [
        "You are scoring a blog post against a fixed set of structural features.",
        "For each feature, pick EXACTLY ONE of its predefined option values -- no other text.",
        "Answer only from the post text itself, not assumptions.",
        "",
        "POST:",
        text,
        "",
        "FEATURES:",
    ]
    for key, spec in FEATURES.items():
        lines.append(f"- {key}: {spec['question']} Options: {spec['options']}")
    lines.append("")
    lines.append(
        "Return strict JSON only, an object mapping each feature key to one of its option strings. "
        "No markdown, no explanation."
    )
    return "\n".join(lines)


def call_gemini(prompt, retries=5):
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(text)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  429, backing off {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("exhausted retries")


def main():
    with open("posts_raw.json", encoding="utf-8") as f:
        raw_posts = json.load(f)
    with open("posts_humanized.json", encoding="utf-8") as f:
        humanized_posts = json.load(f)

    results = {"raw": [], "humanized": []}
    for group_name, posts in (("raw", raw_posts), ("humanized", humanized_posts)):
        for post in posts:
            full_text = post["title"] + "\n\n" + post["text"]
            prompt = build_prompt(full_text)
            print(f"scoring {group_name} #{post['id']}: {post['title'][:40]}")
            answers = call_gemini(prompt)
            results[group_name].append({"id": post["id"], "title": post["title"], "answers": answers})
            time.sleep(4)

    with open("scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("done -> scores.json")


if __name__ == "__main__":
    main()
