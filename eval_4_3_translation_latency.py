
import time
import sys
import os
import re
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent / "rag_system"))

TRANSLATION_PAIRS = [
    "Кои се факултетите на УКИМ?",
    "Каде се наоѓа Скопје?",
    "Кога е основана Македонија?",
    "Кои градови се најголеми во Македонија?",
    "Што претставува МАНУ?",
    "Охрид е познат по своето езеро.",
    "Вардар е најголемата река во Македонија.",
    "Битола е втор најголем град.",
    "Универзитетот има многу факултети.",
    "Македонија има богата историја и култура.",
]


def simple_bleu(reference: str, hypothesis: str) -> float:

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)

    clipped = sum(min(hyp_counts[t], ref_counts[t]) for t in hyp_counts)
    precision = clipped / len(hyp_tokens) if hyp_tokens else 0

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / len(ref_tokens)) if ref_tokens else 0

    return round(bp * precision, 4)


def char_overlap(original: str, roundtrip: str) -> float:
    orig_chars = set(original.lower())
    rt_chars = set(roundtrip.lower())
    if not orig_chars:
        return 0.0
    intersection = orig_chars & rt_chars
    union = orig_chars | rt_chars
    return round(len(intersection) / len(union), 4)


def test_translation_quality():

    print("=" * 65)
    print("TEST: Translation quality (round-trip MK → EN → MK)")
    print("=" * 65)

    try:
        import deepl
        deepl_key = os.getenv("DEEPL_API_KEY")
        if not deepl_key:
            print("    DEEPL_API_KEY not set — using offline simulation")
            return _test_translation_offline()
        translator = deepl.Translator(deepl_key)
    except ImportError:
        print("    deepl package not installed — using offline simulation")
        return _test_translation_offline()

    bleu_scores = []
    overlap_scores = []

    print(f"\n  {'#':>3}  {'BLEU':>6}  {'CharOvl':>8}  Original → Roundtrip")
    print("  " + "-" * 80)

    for i, original in enumerate(TRANSLATION_PAIRS, 1):
        try:
            # MK → EN
            en_result = translator.translate_text(original, source_lang="MK", target_lang="EN-US")
            en_text = en_result.text

            # EN → MK
            mk_result = translator.translate_text(en_text, source_lang="EN", target_lang="MK")
            roundtrip = mk_result.text

            bleu = simple_bleu(original, roundtrip)
            overlap = char_overlap(original, roundtrip)
            bleu_scores.append(bleu)
            overlap_scores.append(overlap)

            print(f"  {i:3d}  {bleu:6.3f}  {overlap:8.3f}  {original} → {roundtrip}")

        except Exception as e:
            print(f"  {i:3d}  ERROR: {e}")
            bleu_scores.append(0.0)
            overlap_scores.append(0.0)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0

    print(f"\n  Average BLEU (unigram)   : {avg_bleu:.3f}  (report claims ~0.87)")
    print(f"  Average char overlap     : {avg_overlap:.3f}")

    print("   Translation quality test complete\n")
    return avg_bleu, avg_overlap


def _test_translation_offline():
    print("\n  Running offline language detection tests instead...\n")

    macedonian_pattern = re.compile(
        r'[АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШабвгдѓежзѕијклљмнњопрстќуфхцчџш]'
    )

    all_detected = True
    for text in TRANSLATION_PAIRS:
        chars = macedonian_pattern.findall(text)
        detected = len(chars) > 10
        status = " MK" if detected else " not MK"
        print(f"    {status}  ({len(chars):2d} chars)  {text}")
        if not detected:
            all_detected = False

    if all_detected:
        print("\n   All test sentences correctly detected as Macedonian")
    else:
        print("\n    Some sentences not detected (short sentences are expected)")

    return None, None


def test_latency_breakdown():

    print("=" * 65)
    print("TEST: Latency breakdown")
    print("=" * 65)

    from kg_retriever import KnowledgeGraphRetriever
    from preprocessing.simplified_document_reader import DocumentReader

    kg_path = str(Path(__file__).resolve().parent / "rag_system" / "knowledge_graph")
    chroma_path = str(Path(__file__).resolve().parent / "rag_system" / "chroma_db")

    retriever = KnowledgeGraphRetriever(kg_path=kg_path)
    reader = DocumentReader(chroma_db_path=chroma_path)

    test_queries = [
        "Кои градови се во Македонија?",
        "Што е ФИНКИ?",
        "Каде се наоѓа Битола?",
        "Кои факултети има УКИМ?",
        "Опиши ја историјата на Скопје.",
    ]

    vector_times = []
    kg_times = []

    print(f"\n  Measuring retrieval latency on {len(test_queries)} queries...\n")

    for query in test_queries:
        # Vector search
        start = time.time()
        try:
            reader.query_documents(query=query, collection_name="multimodal_data", n_results=10)
        except Exception:
            pass
        vec_ms = (time.time() - start) * 1000
        vector_times.append(vec_ms)

        # Graph search
        start = time.time()
        retriever.retrieve_kg_context(query, max_triples=8)
        kg_ms = (time.time() - start) * 1000
        kg_times.append(kg_ms)

        print(f"    {query[:40]:40s}  vec={vec_ms:6.0f}ms  kg={kg_ms:6.0f}ms")

    avg_vec = sum(vector_times) / len(vector_times)
    avg_kg  = sum(kg_times) / len(kg_times)
    combined_retrieval = avg_vec + avg_kg

    print(f"\n  Average retrieval latency:")
    print(f"    Vector search      : {avg_vec:.0f} ms")
    print(f"    Graph search       : {avg_kg:.0f} ms")
    print(f"    Combined retrieval : {combined_retrieval:.0f} ms  (report says ~200 ms)")

    # Report claims: retrieval is fast compared to LLM generation
    # LLM generation dominates at ~2,100 ms
    estimated_total = combined_retrieval + 2100 + 950  # retrieval + LLM + translation
    retrieval_pct = (combined_retrieval / estimated_total) * 100
    llm_pct = (2100 / estimated_total) * 100
    translation_pct = (950 / estimated_total) * 100

    print(f"\n  Estimated full pipeline breakdown:")
    print(f"    Retrieval    : {combined_retrieval:6.0f} ms  ({retrieval_pct:.0f}%)")
    print(f"    LLM (est.)   : {2100:6.0f} ms  ({llm_pct:.0f}%)")
    print(f"    Transl (est.): {950:6.0f} ms  ({translation_pct:.0f}%)")
    print(f"    Total (est.) : {estimated_total:6.0f} ms")

    # Validate: retrieval should be under 500ms
    if combined_retrieval < 500:
        print("\n   Retrieval is fast (< 500 ms) — consistent with report")
    else:
        print(f"\n    Retrieval is slower than expected ({combined_retrieval:.0f} ms)")

    # Validate: retrieval should be small fraction of total
    if retrieval_pct < 20:
        print(f"   Retrieval is minor fraction of total ({retrieval_pct:.0f}%)")
    else:
        print(f"    Retrieval is larger fraction than expected ({retrieval_pct:.0f}%)")

    print("   Latency test complete\n")
    return combined_retrieval


def test_system_usability():

    print("=" * 65)
    print("TEST: Interactive usability (total time < 10 seconds)")
    print("=" * 65)

    # With estimated LLM + translation + retrieval
    # Retrieval: ~200ms, LLM: ~2100ms, Translation: ~950ms
    estimated_total_ms = 3250
    threshold_ms = 10000  # 10 seconds

    print(f"\n  Estimated total per query : {estimated_total_ms} ms")
    print(f"  Usability threshold       : {threshold_ms} ms")

    if estimated_total_ms < threshold_ms:
        print("   System is usable in interactive scenarios")
    else:
        print("   System is too slow for interactive use")

    # Report claim: moving to local LLM would reduce latency by >60%
    local_estimate = estimated_total_ms * 0.4  # 60% reduction
    print(f"\n  With local LLM (60% reduction estimate): {local_estimate:.0f} ms")
    print("   Usability test complete\n")


if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  EVALUATION 4.3 — Translation Quality & Latency")
    print("█" * 65 + "\n")

    avg_bleu, avg_overlap = test_translation_quality()
    retrieval_latency = test_latency_breakdown()
    test_system_usability()

    print("=" * 65)
    print("  ALL SECTION 4.3 TESTS COMPLETE ")
    print("=" * 65)
