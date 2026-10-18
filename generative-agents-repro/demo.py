import time

from llm import answer_reflection_question, generate_reflection_questions, score_importance
from memory import MemoryObject
from reflection import should_reflect
from retrieval import retrieve_top_k

TOY_DAY = [
    "Alex woke up and made a cup of coffee.",
    "Alex checked email and replied to two routine messages.",
    "Alex's manager announced the team's biggest client is cancelling the contract next month.",
    "Alex brushed their teeth before the morning meeting.",
    "Alex spent an hour debugging a minor CSS alignment issue.",
    "Alex found out their apartment building is being sold and tenants may need to move out.",
    "Alex ate a sandwich for lunch at their desk.",
    "Alex's closest coworker, Sam, quietly mentioned they're planning to quit next week.",
    "Alex refilled the office printer with paper.",
    "Alex walked to the parking lot and noticed a flat tire on their car.",
    "Alex texted a friend to vent about the day.",
    "Alex went to bed feeling like everything happened at once.",
]


def build_memory_stream(contents):
    memories = []
    for i, content in enumerate(contents):
        t = float(i)
        importance = score_importance(content)
        print(f"  [importance] {importance:>4.1f}  {content}")
        time.sleep(3)
        memories.append(MemoryObject(content=content, created_at=t, last_accessed_at=t, importance=importance))
    return memories


def main():
    print("=== 1) memory stream 생성 + importance 채점 (실제 Gemini 호출) ===")
    memories = build_memory_stream(TOY_DAY)

    now = float(len(TOY_DAY))
    query = "Alex's day was stressful because of work and housing worries."

    print("\n=== 2) retrieval: 쿼리로 상위 3개 인출 ===")
    top = retrieve_top_k(memories, query, now, k=3)
    for mem, score in top:
        print(f"  score={score:.3f}  {mem.content}")

    print("\n=== 3) reflection 트리거 확인 ===")
    total = sum(m.importance or 0 for m in memories)
    if should_reflect(memories, threshold=40.0):
        print(f"  트리거됨 (importance 합계={total:.1f} >= 40) — 질문 생성 중...")
        contents = [m.content for m in memories]
        questions = generate_reflection_questions(contents, n=2)
        for q in questions:
            print(f"\n  Q: {q}")
            time.sleep(3)
            related = retrieve_top_k(memories, q, now, k=4)
            evidence = [m.content for m, _ in related]
            answer = answer_reflection_question(q, evidence)
            print(f"  A: {answer}")
    else:
        print(f"  트리거 안 됨 (importance 합계={total:.1f} < 40)")


if __name__ == "__main__":
    main()
