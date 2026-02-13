
import json
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "rag_system"))

TEST_QUESTIONS = [
    # Relational questions (KG-RAG should excel)
    "Кои градови се во Македонија?",
    "Кои факултети има на УКИМ?",
    "Каде се наоѓа Битола?",
    "Кои организации се поврзани со Скопје?",
    "Кои лица се поврзани со МАНУ?",
    "Кои институции се во Охрид?",
    "Кои универзитети постојат во Македонија?",
    "Каде работи Методиј?",
    "Кои се главните градови во регионот?",
    "Кои организации се наоѓаат во Скопје?",
    "Што е ФИНКИ?",
    "Зошто е значаен Охрид?",
    "Кога е основана Македонија?",
    "Опиши ја историјата на Скопје.",
    "Што претставува МАНУ?",
    "Какви се климатските услови во Македонија?",
    "Кои се традиционалните јадења во Македонија?",
    "Што е Вардар?",
    "Опиши го образовниот систем во Македонија.",
    "Кои се природните убавини на Македонија?",
]


def get_kg_context(retriever, query):
    start = time.time()
    ctx = retriever.retrieve_kg_context(query, max_triples=8)
    elapsed = (time.time() - start) * 1000
    return ctx, elapsed


def get_vector_context(reader, query):
    start = time.time()
    try:
        results = reader.query_documents(
            query=query,
            collection_name="multimodal_data",
            n_results=10,
        )
        ctx = "\n".join(results[:10]) if results else ""
    except Exception as e:
        print(f"    Vector search error: {e}")
        ctx = ""
    elapsed = (time.time() - start) * 1000
    return ctx, elapsed


def run_comparison():
    """Run test questions through both Vanilla RAG and KG-RAG engines."""
    
    from kg_retriever import KnowledgeGraphRetriever
    from preprocessing.simplified_document_reader import DocumentReader
    from macedonian_engine import MacedonianChatBot
    from vanilla_engine import VanillaChatEngine

    kg_path = Path(__file__).resolve().parent / "rag_system" / "knowledge_graph"
    chroma_path = str(Path(__file__).resolve().parent / "rag_system" / "chroma_db")

    print("Loading Knowledge Graph...")
    retriever = KnowledgeGraphRetriever(kg_path=str(kg_path))
    stats = retriever.get_stats()
    print(f"  ✓ KG loaded: {stats['total_triples']} triples, "
          f"{stats['unique_entities']} entities\n")

    print("Loading Vector DB...")
    reader = DocumentReader(chroma_db_path=chroma_path)
    print("  ✓ Vector DB loaded\n")

    print("Initializing RAG Engines...")
    try:
        print("  Initializing Vanilla RAG Engine (baseline, no KG)...")
        vanilla_engine = VanillaChatEngine(user_id="eval_vanilla")
        print("  ✓ Vanilla RAG Engine ready\n")
    except Exception as e:
        print(f"  ⚠ Vanilla RAG Engine failed to initialize: {e}")
        vanilla_engine = None

    try:
        print("  Initializing KG-RAG Engine (Macedonian with KG)...")
        kgrag_engine = MacedonianChatBot(user_id="eval_kgrag")
        print("  ✓ KG-RAG Engine ready\n")
    except Exception as e:
        print(f"  ⚠ KG-RAG Engine failed to initialize: {e}")
        kgrag_engine = None

    results = []

    print("=" * 90)
    print(f"  Running {len(TEST_QUESTIONS)} test questions on both engines")
    print("=" * 90)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─' * 90}")
        print(f"  Q{i}: {question}")
        print(f"{'─' * 90}")

        vec_ctx, vec_ms = get_vector_context(reader, question)
        vec_snippet = vec_ctx[:150].replace("\n", " ") if vec_ctx else "(empty)"

        kg_ctx, kg_ms = get_kg_context(retriever, question)
        kg_snippet = kg_ctx[:200].replace("\n", " | ") if kg_ctx else "(empty)"

        print(f"  📊 Retrieval:")
        print(f"    Vector DB ({vec_ms:.0f} ms): {vec_snippet}...")
        print(f"    KG search ({kg_ms:.0f} ms): {kg_snippet}")

        has_kg = bool(kg_ctx.strip())
        has_vec = bool(vec_ctx.strip())

        # Run through Vanilla RAG (baseline)
        vanilla_response = None
        vanilla_time_ms = 0
        if vanilla_engine:
            try:
                start = time.time()
                vanilla_response = vanilla_engine.process_query(question)
                vanilla_time_ms = (time.time() - start) * 1000
                # Extract just the response text
                if isinstance(vanilla_response, str):
                    try:
                        resp_json = json.loads(vanilla_response)
                        vanilla_text = resp_json.get("response", vanilla_response)[:200]
                    except:
                        vanilla_text = vanilla_response[:200]
                else:
                    vanilla_text = str(vanilla_response)[:200]
                print(f"  🔵 VANILLA RAG ({vanilla_time_ms:.0f} ms): {vanilla_text}...")
            except Exception as e:
                print(f"  🔵 VANILLA RAG Error: {e}")
                vanilla_response = f"Error: {e}"

        # Run through KG-RAG (Macedonian)
        kgrag_response = None
        kgrag_time_ms = 0
        if kgrag_engine:
            try:
                start = time.time()
                kgrag_response = kgrag_engine.process_query(question)
                kgrag_time_ms = (time.time() - start) * 1000
                # Extract just the response text
                if isinstance(kgrag_response, str):
                    try:
                        resp_json = json.loads(kgrag_response)
                        kgrag_text = resp_json.get("response", kgrag_response)[:200]
                    except:
                        kgrag_text = kgrag_response[:200]
                else:
                    kgrag_text = str(kgrag_response)[:200]
                print(f"  🟢 KG-RAG    ({kgrag_time_ms:.0f} ms): {kgrag_text}...")
            except Exception as e:
                print(f"  🟢 KG-RAG Error: {e}")
                kgrag_response = f"Error: {e}"

        results.append({
            "id": i,
            "question": question,
            "has_vector_context": has_vec,
            "has_kg_context": has_kg,
            "vector_time_ms": round(vec_ms, 1),
            "kg_time_ms": round(kg_ms, 1),
            "kg_context_preview": kg_ctx[:500] if kg_ctx else "",
            "vector_context_preview": vec_ctx[:500] if vec_ctx else "",
            # Engine responses (full)
            "vanilla_response": vanilla_response,
            "vanilla_response_time_ms": round(vanilla_time_ms, 1),
            "kgrag_response": kgrag_response,
            "kgrag_response_time_ms": round(kgrag_time_ms, 1),
            # Placeholders for manual scoring (1-3 scale)
            "vanilla_correctness": None,
            "kgrag_correctness": None,
            "vanilla_relevance": None,
            "kgrag_relevance": None,
            "vanilla_fluency": None,
            "kgrag_fluency": None,
        })

    return results


def analyze_results(results):
    """Analyze retrieval coverage, timing, and engine performance."""
    print("\n\n" + "=" * 90)
    print("  ANALYSIS — Retrieval & Engine Performance")
    print("=" * 90)

    total = len(results)
    kg_found = sum(1 for r in results if r["has_kg_context"])
    vec_found = sum(1 for r in results if r["has_vector_context"])

    print(f"\n  📊 Retrieval Coverage:")
    print(f"    Vector DB returned context : {vec_found}/{total} ({100*vec_found/total:.0f}%)")
    print(f"    KG returned context        : {kg_found}/{total} ({100*kg_found/total:.0f}%)")

    # Split into relational (first 10) vs comprehension (last 10)
    relational = results[:10]
    comprehension = results[10:]

    kg_rel = sum(1 for r in relational if r["has_kg_context"])
    kg_comp = sum(1 for r in comprehension if r["has_kg_context"])

    print(f"\n  📈 KG Context by Question Type:")
    print(f"    Relational questions (Q1-Q10)     : {kg_rel}/10 ({100*kg_rel/10:.0f}%)")
    print(f"    Comprehension questions (Q11-Q20) : {kg_comp}/10 ({100*kg_comp/10:.0f}%)")

    if kg_rel > kg_comp:
        print(f"     ✓ KG provides more context for relational questions (as expected)")
    else:
        print(f"     ⚠ KG did not provide more context for relational questions")

    avg_vec_ms = sum(r["vector_time_ms"] for r in results) / total
    avg_kg_ms  = sum(r["kg_time_ms"] for r in results) / total

    print(f"\n  ⏱️  Retrieval Latency:")
    print(f"    Vector search : {avg_vec_ms:.0f} ms")
    print(f"    Graph search  : {avg_kg_ms:.0f} ms")
    print(f"    Combined      : {avg_vec_ms + avg_kg_ms:.0f} ms")

    # Engine latency
    vanilla_times = [r.get("vanilla_response_time_ms", 0) for r in results if r.get("vanilla_response_time_ms")]
    kgrag_times = [r.get("kgrag_response_time_ms", 0) for r in results if r.get("kgrag_response_time_ms")]

    if vanilla_times:
        avg_vanilla_ms = sum(vanilla_times) / len(vanilla_times)
        print(f"\n  🔵 Vanilla RAG (avg response time): {avg_vanilla_ms:.0f} ms")
    else:
        print(f"\n  🔵 Vanilla RAG: No response times recorded")

    if kgrag_times:
        avg_kgrag_ms = sum(kgrag_times) / len(kgrag_times)
        print(f"  🟢 KG-RAG (avg response time)     : {avg_kgrag_ms:.0f} ms")
        if vanilla_times and avg_vanilla_ms > 0:
            overhead = ((avg_kgrag_ms - avg_vanilla_ms) / avg_vanilla_ms) * 100
            print(f"     KG-RAG overhead: {overhead:+.1f}%")
    else:
        print(f"  🟢 KG-RAG: No response times recorded")

    return {
        "total_questions": total,
        "vector_coverage": vec_found,
        "kg_coverage": kg_found,
        "kg_relational_coverage": kg_rel,
        "kg_comprehension_coverage": kg_comp,
        "avg_vector_ms": round(avg_vec_ms, 1),
        "avg_kg_ms": round(avg_kg_ms, 1),
        "avg_vanilla_response_ms": round(sum(vanilla_times) / len(vanilla_times), 1) if vanilla_times else None,
        "avg_kgrag_response_ms": round(sum(kgrag_times) / len(kgrag_times), 1) if kgrag_times else None,
    }


def save_scoring_template(results, analysis):
    
    output_path = Path(__file__).resolve().parent / "eval_4_2_scoring_template.json"

    output = {
        "description": (
            "Manual scoring template for Section 4.2 evaluation. "
            "Score each response on a 1-3 scale: "
            "1 = incorrect/irrelevant, 2 = partially correct, 3 = fully correct/relevant. "
            "Fill in the null fields after running both engines on each question."
        ),
        "analysis": analysis,
        "questions": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Scoring template saved to: {output_path}")
    print("  Fill in the null fields after running both engines.")


if __name__ == "__main__":
    print("\n" + "█" * 90)
    print("  EVALUATION 4.2 — KG-RAG vs Vanilla RAG (Baseline)")
    print("█" * 90 + "\n")

    results = run_comparison()
    analysis = analyze_results(results)
    save_scoring_template(results, analysis)

    print("\n" + "=" * 90)
    print("  EVALUATION 4.2 COMPLETE")
    print("  ✓ Test questions run on both Vanilla RAG (baseline) and KG-RAG engines")
    print("  ✓ Responses saved for manual scoring (correctness, relevance, fluency)")
    print("  ✓ Retrieval coverage and latency metrics computed")
    print("  Next: Fill in null scoring fields, then run eval_4_2_score_analysis.py")
    print("=" * 90 + "\n")
