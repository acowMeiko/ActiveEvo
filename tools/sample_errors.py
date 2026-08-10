"""从评测结果中随机采样错误样本，用于 Block 7 错误分析标注。

只输出核心字段: 题目、正确答案、模型回答、检索到的 principle、经验库搜索辅助。

用法:
  python tools/sample_errors.py \
    --results output/stage3_eval/llama_round4/eval_results.json \
    --memory-dir memory/llama \
    --n 100 \
    --output error_analysis/samples.json
"""

import argparse, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)


# ---------- 检索摘要 ----------

def _format_principles(retrieved: list) -> str:
    """SLP 实际检回的 principle（三维 + 正文摘要）。"""
    if not retrieved:
        return "（未触发检索）"
    texts = []
    for i, p in enumerate(retrieved, 1):
        inner = p.get("item", p)
        score = p.get("score", 0)
        skill = inner.get("core_skill", "")[:60]
        constraint = inner.get("context_constraints", "")[:60]
        obj = inner.get("task_objective", "")[:60]
        preview = inner.get("principle", "")[:100].replace("\n", " ")
        texts.append(
            f"[{i}] score={score:.2f}\n"
            f"    Skill: {skill}\n"
            f"    Constraints: {constraint}\n"
            f"    Objective: {obj}\n"
            f"    Principle: {preview}..."
        )
    return "\n".join(texts)


# ---------- 经验库关键词搜索 ----------

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "to", "and", "or", "for",
    "on", "at", "with", "by", "be", "it", "that", "this", "as", "from",
    "has", "have", "was", "were", "not", "we", "can", "but", "if", "its",
    "each", "all", "no", "so", "than", "then", "also", "such", "how",
    "what", "which", "one", "two", "three", "four", "five", "you",
    "your", "find", "many", "let", "will", "would", "could", "should",
}


def _keywords(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def _hit_score(principle: dict, keywords: set) -> int:
    combined = " ".join([
        principle.get("core_skill", ""),
        principle.get("context_constraints", ""),
        principle.get("task_objective", ""),
        principle.get("principle", ""),
    ]).lower()
    return sum(1 for kw in keywords if kw in combined)


def _search_memory(question: str, principles: list, top_k: int = 5) -> str:
    kws = _keywords(question)
    if not kws or not principles:
        return "（无关键词或无经验库）"
    scored = []
    for p in principles:
        s = _hit_score(p, kws)
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "（无匹配）"
    lines = []
    for i, (s, p) in enumerate(scored[:top_k], 1):
        lines.append(
            f"[{i}] hits={s}\n"
            f"    Skill: {p.get('core_skill','')[:60]}\n"
            f"    Objective: {p.get('task_objective','')[:60]}\n"
            f"    Principle: {p.get('principle','')[:100].replace(chr(10),' ')}..."
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Sample wrong answers for error analysis")
    p.add_argument("--results", required=True)
    p.add_argument("--memory-dir", default="memory/llama")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--output", default="error_analysis/samples.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results_path = _resolve(args.results)
    output_path = _resolve(args.output)
    memory_dir = _resolve(args.memory_dir)

    if not results_path.exists():
        print(f"ERROR: not found: {results_path}")
        sys.exit(1)

    # 加载经验库
    all_principles = []
    p_file = memory_dir / "principles" / "principles.json"
    if p_file.exists():
        all_principles = _load_json(p_file)
        print(f"Memory: {len(all_principles)} principles")

    # 加载结果
    data = _load_json(results_path)
    wrong = [r for r in data if not r.get("is_correct")]
    print(f"Total: {len(data)}  Wrong: {len(wrong)}  Acc: {(len(data)-len(wrong))/len(data)*100:.1f}%")

    random.seed(args.seed)
    sampled = random.sample(wrong, min(args.n, len(wrong)))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for item in sampled:
        q = item.get("question", "")
        rec = {
            "idx": item["idx"],
            "question": q,
            "expected": item.get("expected_answer", ""),
            "model_answer": item.get("model_answer", ""),
            "retrieved": _format_principles(item.get("retrieved_principles", [])),
            "memory_search": _search_memory(q, all_principles),
            "label": {
                "category": "",           # A/B/C/D/E
                "why": "",                # 为什么
                "in_bank": "",            # yes / no / unsure
                "notes": "",
            },
        }
        records.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} items -> {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
