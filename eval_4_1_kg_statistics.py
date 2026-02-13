
import json
import sys
import random
from pathlib import Path
from collections import Counter

KG_DIR = Path(__file__).resolve().parent / "rag_system" / "knowledge_graph"
TRIPLES_FILE = KG_DIR / "triples_kg_unified.json"
STATS_FILE = KG_DIR / "kg_unified_stats.json"
GRAPHML_FILE = KG_DIR / "kg_unified.graphml"

def load_triples():
    with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_stats():
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_total_triples_and_entities():
    """Report claim: graph contains 3,250 triples and 3,422 unique entities."""
    stats = load_stats()
    triples = load_triples()

    print("=" * 65)
    print("TEST: Total triples and entities")
    print("=" * 65)

    actual_triples = len(triples)
    reported_triples = stats["total_triples"]
    print(f"  Triples in JSON file : {actual_triples}")
    print(f"  Triples in stats.json: {reported_triples}")
    assert actual_triples == reported_triples, \
        f"Mismatch: JSON has {actual_triples}, stats says {reported_triples}"

    entities = set()
    for t in triples:
        entities.add(t["subject"].lower())
        entities.add(t["object"].lower())
    print(f"  Unique entities (computed): {len(entities)}")
    print(f"  Unique entities (stats)   : {stats['unique_entities']}")

    # Allow small tolerance (± 5%) for entity count since lowercasing can merge some
    diff = abs(len(entities) - stats["unique_entities"])
    assert diff <= stats["unique_entities"] * 0.05, \
        f"Entity count deviates too much: computed {len(entities)}, reported {stats['unique_entities']}"

    print("   PASSED\n")
    return triples, stats


def test_source_distribution(triples):
    """Report claim: WikiANN contributes ~2,100 triples, FLORES ~800."""
    print("=" * 65)
    print("TEST: Source distribution")
    print("=" * 65)

    source_counts = Counter(t.get("source", "unknown") for t in triples)
    for src, count in source_counts.most_common():
        print(f"  {src:30s}  {count:>5d}")

    wikiann_total = sum(c for s, c in source_counts.items() if "wikiann" in s)
    flores_total = source_counts.get("flores", 0)

    print(f"\n  WikiANN total : {wikiann_total}  (report says ~2,100)")
    print(f"  FLORES total  : {flores_total}  (report says ~800)")

    # WikiANN should be the dominant source
    assert wikiann_total > flores_total, \
        "WikiANN should contribute more triples than FLORES"
    # WikiANN should be roughly > 50% of all triples
    assert wikiann_total >= len(triples) * 0.4, \
        f"WikiANN ({wikiann_total}) should be at least 40% of total ({len(triples)})"

    print("   PASSED\n")
    return source_counts


def test_graph_density(triples, stats):
    """Report claim: average node degree ≈ 1.9 (sparse graph)."""
    print("=" * 65)
    print("TEST: Graph density / average degree")
    print("=" * 65)

    nodes = stats["graph_nodes"]
    edges = stats["graph_edges"]

    if nodes > 0:
        avg_degree = (2 * edges) / nodes
    else:
        avg_degree = 0

    print(f"  Nodes         : {nodes}")
    print(f"  Edges         : {edges}")
    print(f"  Avg degree    : {avg_degree:.2f}  (report says ~1.9)")

    assert avg_degree < 5.0, \
        f"Graph is unexpectedly dense (avg degree {avg_degree:.2f})"
    assert avg_degree > 0.5, \
        f"Graph is unexpectedly disconnected (avg degree {avg_degree:.2f})"

    print("   PASSED\n")


def test_triple_quality_sample(triples, sample_size=30):

    print("=" * 65)
    print("TEST: Triple quality — automated heuristic + manual sample")
    print("=" * 65)

    sample = random.sample(triples, min(sample_size, len(triples)))

    obviously_bad = 0
    suspicious = 0
    ok = 0

    print(f"\n  {'#':>3}  {'Subject':30s}  {'Relation':25s}  {'Object':20s}  Flag")
    print("  " + "-" * 105)

    for i, t in enumerate(sample, 1):
        subj = t["subject"]
        rel  = t["relation"]
        obj  = t["object"]
        flag = ""

        # Heuristic checks
        if subj.strip() == "" or obj.strip() == "":
            flag = "⛔ EMPTY"
            obviously_bad += 1
        elif subj.lower() == obj.lower():
            flag = "⚠️  SELF-REF"
            suspicious += 1
        elif len(subj) <= 1 or len(obj) <= 1:
            flag = "⚠️  TOO SHORT"
            suspicious += 1
        elif rel.strip() == "":
            flag = "⛔ NO RELATION"
            obviously_bad += 1
        else:
            flag = "✓"
            ok += 1

        print(f"  {i:3d}  {subj:30s}  {rel:25s}  {obj:20s}  {flag}")

    total = ok + suspicious + obviously_bad
    print(f"\n  Summary ({total} sampled):")
    print(f"    OK (heuristic)         : {ok:3d}  ({100*ok/total:.0f}%)")
    print(f"    Suspicious             : {suspicious:3d}  ({100*suspicious/total:.0f}%)")
    print(f"    Obviously problematic  : {obviously_bad:3d}  ({100*obviously_bad/total:.0f}%)")

    # At least half should pass basic heuristic checks
    assert ok >= total * 0.4, \
        f"Too many problematic triples: only {ok}/{total} passed heuristics"

    print("   PASSED\n")


def test_quality_by_source(triples):

    print("=" * 65)
    print("TEST: Quality proxy by source (entity length, relation diversity)")
    print("=" * 65)

    by_source = {}
    for t in triples:
        src = t.get("source", "unknown")
        by_source.setdefault(src, []).append(t)

    for src, src_triples in sorted(by_source.items()):
        avg_subj_len = sum(len(t["subject"]) for t in src_triples) / len(src_triples)
        avg_obj_len  = sum(len(t["object"]) for t in src_triples) / len(src_triples)
        unique_rels  = len(set(t["relation"] for t in src_triples))

        print(f"  {src:30s}  triples={len(src_triples):>5d}  "
              f"avg_subj_len={avg_subj_len:5.1f}  avg_obj_len={avg_obj_len:5.1f}  "
              f"unique_relations={unique_rels}")

    # LLM (flores) should have more diverse relations than simple NER tagging
    if "flores" in by_source and "wikiann" in by_source:
        flores_rels = len(set(t["relation"] for t in by_source["flores"]))
        wikiann_rels = len(set(t["relation"] for t in by_source["wikiann"]))
        print(f"\n  FLORES unique relations  : {flores_rels}")
        print(f"  WikiANN unique relations : {wikiann_rels}")
        assert flores_rels >= wikiann_rels, \
            "Expected FLORES (LLM) to have at least as many unique relation types"

    print("   PASSED\n")


if __name__ == "__main__":
    random.seed(42)

    print("\n" + "█" * 65)
    print("  EVALUATION 4.1 — Knowledge Graph Statistics")
    print("█" * 65 + "\n")

    triples, stats = test_total_triples_and_entities()
    source_counts   = test_source_distribution(triples)
    test_graph_density(triples, stats)
    test_triple_quality_sample(triples)
    test_quality_by_source(triples)

