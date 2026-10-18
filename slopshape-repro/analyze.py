import json

from score_features import FEATURES

with open("scores.json", encoding="utf-8") as f:
    data = json.load(f)


def ai_count(answers):
    return sum(1 for k, spec in FEATURES.items() if answers[k] == spec["ai_leaning"])


raw = {p["id"]: p["answers"] for p in data["raw"]}
hum = {p["id"]: p["answers"] for p in data["humanized"]}

print("id | raw_ai_count | humanized_ai_count | delta")
deltas = []
for i in sorted(raw):
    r, h = ai_count(raw[i]), ai_count(hum[i])
    deltas.append(h - r)
    print(f"{i:2d} | {r:2d}/9 | {h:2d}/9 | {h - r:+d}")

n = len(deltas)
mean_raw = sum(ai_count(raw[i]) for i in raw) / n
mean_hum = sum(ai_count(hum[i]) for i in hum) / n
decreased = sum(1 for d in deltas if d < 0)
increased = sum(1 for d in deltas if d > 0)
same = sum(1 for d in deltas if d == 0)
print(f"\nmean AI-lean count: raw={mean_raw:.2f}/9  humanized={mean_hum:.2f}/9")
print(f"posts where humanizing DECREASED AI-lean count: {decreased}/{n}")
print(f"posts where humanizing INCREASED AI-lean count: {increased}/{n}")
print(f"posts unchanged: {same}/{n}")

print("\nper-feature: how many of 10 posts flip from AI-leaning (raw) to non-AI-leaning (humanized)")
for key, spec in FEATURES.items():
    flips_to_human = 0
    flips_to_ai = 0
    raw_ai = 0
    hum_ai = 0
    for i in raw:
        r_is_ai = raw[i][key] == spec["ai_leaning"]
        h_is_ai = hum[i][key] == spec["ai_leaning"]
        raw_ai += r_is_ai
        hum_ai += h_is_ai
        if r_is_ai and not h_is_ai:
            flips_to_human += 1
        if not r_is_ai and h_is_ai:
            flips_to_ai += 1
    print(f"  {key:35s} raw_ai={raw_ai}/10 hum_ai={hum_ai}/10  flips_to_human={flips_to_human} flips_to_ai={flips_to_ai}")
