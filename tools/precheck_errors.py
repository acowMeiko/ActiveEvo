"""预检 100 道错题的 A/B/C/D 分布趋势（粗估，非人工标注精度）。
用法: python tools/precheck_errors.py error_analysis/samples.json
"""

import json, re, sys

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "to", "and", "or", "for",
    "on", "at", "with", "by", "be", "it", "that", "this", "as", "from",
    "has", "have", "was", "were", "not", "we", "can", "but", "if", "its",
    "each", "all", "no", "so", "than", "then", "also", "such", "how",
    "what", "which", "one", "two", "three", "four", "five", "you",
    "your", "find", "many", "let", "will", "would", "could", "should",
}


def keywords(text: str) -> set:
    return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2 and t not in STOPWORDS}


def kw_overlap(q_kw: set, text: str) -> int:
    """问题的关键词在文本中命中了多少个。"""
    return sum(1 for kw in q_kw if kw in text.lower())


def analyze(retrieved_str: str, mem_search_str: str, q_kw: set) -> dict:
    """给一条样本做粗分类。"""
    ret_overlap = kw_overlap(q_kw, retrieved_str) if retrieved_str else 0
    mem_overlap = kw_overlap(q_kw, mem_search_str) if mem_search_str else 0

    # 判断逻辑
    has_retrieval = bool(retrieved_str) and "未触发检索" not in retrieved_str
    has_memory = bool(mem_search_str) and "无匹配" not in mem_search_str

    if not has_retrieval:
        # 没触发检索 → 可能是 B（没东西可检）或检索阈值太高
        return {"guess": "B?", "reason": "未触发检索", "ret_hits": 0, "mem_hits": mem_overlap}

    # 计算检索到的 principle 的平均关键词命中
    ret_lines = [l for l in retrieved_str.split("\n") if l.strip()]
    ret_text = " ".join(ret_lines)

    if ret_overlap <= 2 and mem_overlap > ret_overlap + 2:
        return {"guess": "A_likely", "reason": f"检索弱相关(hits={ret_overlap})，经验库有更好的(hits={mem_overlap})",
                "ret_hits": ret_overlap, "mem_hits": mem_overlap}
    elif ret_overlap <= 2 and mem_overlap <= 2:
        return {"guess": "B_likely", "reason": f"检索和经验库都弱相关(ret={ret_overlap}, mem={mem_overlap})",
                "ret_hits": ret_overlap, "mem_hits": mem_overlap}
    elif ret_overlap > 2:
        return {"guess": "C/D_likely", "reason": f"检索相关(hits={ret_overlap})，但仍答错→可能执行错或题难",
                "ret_hits": ret_overlap, "mem_hits": mem_overlap}
    else:
        return {"guess": "?", "reason": f"边界情况", "ret_hits": ret_overlap, "mem_hits": mem_overlap}


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/precheck_errors.py error_analysis/samples.json")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        q_kw = keywords(item["question"])
        a = analyze(item.get("retrieved", ""), item.get("memory_search", ""), q_kw)
        a["idx"] = item["idx"]
        results.append(a)

    # 统计
    from collections import Counter
    counter = Counter(r["guess"] for r in results)

    print(f"=== 预检 {len(results)} 条样本 ===\n")
    print("注意: 这是关键词匹配的粗估，不是人工标注。供判断趋势用。\n")

    for label, count in counter.most_common():
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:20s}  {count:3d}  ({pct:5.1f}%)  {bar}")

    print(f"\n{'─'*50}")
    print("解读:")
    print("  A_likely   = 检索弱相关，但经验库有更好的 → 检索失败")
    print("  B_likely   = 检索和经验库都弱相关 → 经验缺失")
    print("  C/D_likely = 检索相关但仍答错 → 执行错误或题难")
    print("  B?         = 未触发检索 → 可能是检索阈值问题")
    print()

    # 打印 A_likely 的前几条例子
    a_samples = [r for r in results if "A" in r["guess"]]
    print(f"A_likely 样本 ({len(a_samples)} 条)，前 5 例:")
    for r in a_samples[:5]:
        item = next(d for d in data if d["idx"] == r["idx"])
        print(f"  idx={r['idx']}: ret_hits={r['ret_hits']}, mem_hits={r['mem_hits']}")
        print(f"    Q: {item['question'][:120]}...")
        print()

    b_samples = [r for r in results if "B" in r["guess"]]
    print(f"B_likely 样本 ({len(b_samples)} 条)，前 5 例:")
    for r in b_samples[:5]:
        item = next(d for d in data if d["idx"] == r["idx"])
        print(f"  idx={r['idx']}: ret_hits={r['ret_hits']}, mem_hits={r['mem_hits']}")
        print(f"    Q: {item['question'][:120]}...")
        print()


if __name__ == "__main__":
    main()
