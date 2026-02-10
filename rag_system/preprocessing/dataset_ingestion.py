import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from datasets import load_dataset
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import networkx as nx

load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not available. LLM-based triple extraction disabled.")

# WikiANN BIO tag scheme
_NER_TAG_MAP = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}

# Co-occurrence relation heuristics: (source_type, target_type) -> relation
_COOCCURRENCE_RELATIONS = {
    ("PER", "ORG"): "affiliated_with",
    ("PER", "LOC"): "located_in",
    ("ORG", "LOC"): "based_in",
}

_CLEAN_PREFIX = re.compile(r'^[\d\*\-]+\.?\s+')


class DatasetIngestionPipeline:

    def __init__(self,
                 chroma_db_path: str = "./chroma_db",
                 kg_output_path: str = "./knowledge_graph",
                 collection_name: str = "multimodal_data"):

        self.chroma_db_path = chroma_db_path
        self.kg_output_path = Path(kg_output_path)
        self.kg_output_path.mkdir(parents=True, exist_ok=True)

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        # Knowledge Graph
        self.graph = nx.MultiDiGraph()
        self.triples: List[Dict[str, str]] = []
        self._seen_triples: set = set()
        self.entities: set = set()

        # LLM client (optional)
        self.groq_client = None
        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                print("Groq client initialized for triple extraction")

        self.stats = {"documents_added": 0, "triples_added": 0, "duplicates_skipped": 0}


    def _upsert_documents(self, texts: List[str], metadatas: List[Dict],
                          ids: List[str], min_length: int = 20):
        rows = [(t, m, i) for t, m, i in zip(texts, metadatas, ids)
                if t and len(t.strip()) > min_length]
        if not rows:
            return
        batch_size = 100
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            try:
                self.collection.upsert(
                    documents=[r[0] for r in batch],
                    metadatas=[r[1] for r in batch],
                    ids=[r[2] for r in batch],
                )
                self.stats["documents_added"] += len(batch)
            except Exception as e:
                print(f"] Error upserting batch: {e}")


    def _add_triple(self, subject: str, relation: str, obj: str,
                    source: str = "unknown") -> bool:
        key = (subject.lower().strip(), relation.lower().strip(), obj.lower().strip())
        if key in self._seen_triples:
            self.stats["duplicates_skipped"] += 1
            return False
        self._seen_triples.add(key)
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(obj, type="entity")
        self.graph.add_edge(subject, obj, relation=relation, source=source)
        self.triples.append({"subject": subject, "relation": relation,
                             "object": obj, "source": source})
        self.entities.update([subject, obj])
        self.stats["triples_added"] += 1
        return True


    def _extract_triples_llm(self, text: str) -> List[Tuple[str, str, str]]:
        """Ask Groq to extract (subject, relation, object) triples from Macedonian text."""
        if not self.groq_client or not text or len(text.strip()) < 20:
            return []
        prompt = (
            "You are an expert in extracting knowledge from Macedonian text.\n"
            "Extract knowledge triples (subject, relation, object) from the following text.\n\n"
            f"Text: {text}\n\n"
            "Provide triples in the format:\nsubject | relation | object\n\n"
            "RULES: No numbering. Use specific entities from the text. "
            "Return ONLY triples, NO explanations.\n\nTriples:"
        )
        try:
            resp = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", max_tokens=500, temperature=0.3,
            )
            return self._parse_llm_triples(resp.choices[0].message.content)
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []

    @staticmethod
    def _parse_llm_triples(text: str) -> List[Tuple[str, str, str]]:
        triples = []
        for line in text.strip().splitlines():
            m = re.match(r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$', line.strip())
            if not m:
                continue
            subj = _CLEAN_PREFIX.sub('', m.group(1)).strip().strip('"\'')
            obj = _CLEAN_PREFIX.sub('', m.group(3)).strip().strip('"\'')
            rel = m.group(2).strip().strip('"\'').lower().replace(' ', '_')
            if subj and rel and obj:
                triples.append((subj, rel, obj))
        return triples

    # ---- NER helpers (BIO tags) -------------------------------------- #

    @staticmethod
    def _extract_bio_entities(tokens: list, tag_ids: list) -> List[Tuple[str, str]]:
        """Extract (entity_text, entity_type) from BIO-tagged token sequences."""
        entities, buf, cur_type = [], [], None
        for token, tid in zip(tokens, tag_ids):
            tag = _NER_TAG_MAP.get(tid, "O")
            if tag.startswith("B-"):
                if buf:
                    entities.append((" ".join(buf), cur_type))
                buf, cur_type = [token], tag[2:]
            elif tag.startswith("I-") and cur_type == tag[2:]:
                buf.append(token)
            else:
                if buf:
                    entities.append((" ".join(buf), cur_type))
                    buf, cur_type = [], None
        if buf:
            entities.append((" ".join(buf), cur_type))
        return entities
    

    def ingest_wikiann(self, language: str = "mk", max_samples: int = 5000):
        print(f"\n{'='*60}\nIngesting WikiANN ({language}) – max {max_samples}\n{'='*60}")

        try:
            dataset = load_dataset("wikiann", language, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load WikiANN: {e}")
            return

        texts, metadatas, ids = [], [], []
        entity_pairs: Dict[Tuple, List[Tuple]] = defaultdict(list)

        for split in ("train", "validation", "test"):
            if split not in dataset:
                continue
            split_data = dataset[split]
            n = min(max_samples // 3, len(split_data))

            for idx in range(n):
                item = split_data[idx]
                sentence = " ".join(item["tokens"])
                ents = self._extract_bio_entities(item["tokens"], item["ner_tags"])

                if sentence and len(sentence) > 20:
                    texts.append(sentence)
                    metadatas.append({"source": "wikiann", "language": language,
                                      "split": split, "entity_count": len(ents)})
                    ids.append(f"wikiann_{language}_{split}_{idx}")

                for ent_text, ent_type in ents:
                    self._add_triple(ent_text, f"is_{ent_type.lower()}", ent_type, source="wikiann")
                    for other_text, other_type in ents:
                        if ent_text != other_text:
                            entity_pairs[(ent_text, ent_type)].append((other_text, other_type))

        # Co-occurrence edges (≥ 2 co-occurrences)
        for (entity, etype), related in entity_pairs.items():
            for (rel_entity, rel_type), count in Counter(related).items():
                if count >= 2:
                    rel = _COOCCURRENCE_RELATIONS.get((etype, rel_type), "related_to")
                    self._add_triple(entity, rel, rel_entity, source="wikiann_cooccurrence")

        self._upsert_documents(texts, metadatas, ids)
        print(f"  WikiANN done – {len(texts)} docs, {len(self.entities)} entities")

    def ingest_flores(self, language_code: str = "mkd_Cyrl", max_samples: int = 1000):
        print(f"\n{'='*60}\nIngesting FLORES ({language_code}) – max {max_samples}\n{'='*60}")

        try:
            dataset = load_dataset("facebook/flores", language_code, split="dev", trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load FLORES: {e}")
            return

        texts, metadatas, ids = [], [], []
        n = min(max_samples, len(dataset))

        for idx in range(n):
            item = dataset[idx]
            sentence = item.get("sentence") or item.get("text", "")
            if not sentence or len(sentence.strip()) < 20:
                continue

            texts.append(sentence)
            metadatas.append({"source": "flores", "language": language_code,
                              "sentence_id": item.get("id", idx)})
            ids.append(f"flores_{language_code}_{idx}")

            if self.groq_client and idx % 3 == 0:
                for subj, rel, obj in self._extract_triples_llm(sentence):
                    self._add_triple(subj, rel, obj, source="flores")

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{n} sentences...")

        self._upsert_documents(texts, metadatas, ids)
        print(f"  FLORES done – {len(texts)} docs")

    def ingest_tydiqa(self, max_samples: int = 2000):
        """Ingest TyDiQA passages into the vector DB (no KG triples)."""
        print(f"\n{'='*60}\nIngesting TyDiQA – max {max_samples}\n{'='*60}")

        dataset = None
        for name, config in [("tydiqa", "secondary_task"),
                             ("copenlu/answerable_tydiqa", None),
                             ("khalidalt/tydiqa-goldp", None)]:
            try:
                dataset = load_dataset(name, config, trust_remote_code=True) if config \
                    else load_dataset(name, trust_remote_code=True)
                print(f"  Loaded from: {name}")
                break
            except Exception:
                continue
        if dataset is None:
            print("  Could not load TyDiQA. Skipping.")
            return

        texts, metadatas, ids = [], [], []
        seen = set()

        for split in ("train", "validation"):
            if split not in dataset:
                continue
            split_data = dataset[split]
            n = min(max_samples // 2, len(split_data))

            for idx in range(n):
                item = split_data[idx]
                context = item.get("context", "")
                if not context or len(context.strip()) < 50:
                    continue
                key = context[:200]
                if key in seen:
                    continue
                seen.add(key)

                question = item.get("question", "")
                lang = item.get("id", "").split("-")[0] if "-" in item.get("id", "") else "unknown"
                texts.append(context)
                metadatas.append({"source": "tydiqa", "language": lang,
                                  "split": split, "question": question[:200]})
                ids.append(f"tydiqa_{split}_{idx}")

        self._upsert_documents(texts, metadatas, ids, min_length=50)
        print(f"  TyDiQA done – {len(texts)} docs")

    def ingest_tatoeba(self, language_pair: str = "mkd-eng", max_samples: int = 2000):
        print(f"\n{'='*60}\nIngesting Tatoeba ({language_pair}) – max {max_samples}\n{'='*60}")

        dataset = None
        for pair in (language_pair, language_pair.replace("-", "_")):
            try:
                dataset = load_dataset("mteb/tatoeba-bitext-mining", pair, trust_remote_code=True)
                break
            except Exception:
                continue
        if dataset is None:
            print("  Could not load Tatoeba. Skipping.")
            return

        texts, metadatas, ids = [], [], []
        for split in dataset.keys():
            split_data = dataset[split]
            n = min(max_samples, len(split_data))
            for idx in range(n):
                item = split_data[idx]
                for si, sent in enumerate([item.get("sentence1", ""), item.get("sentence2", "")]):
                    if sent and len(sent.strip()) > 10:
                        texts.append(sent)
                        metadatas.append({"source": "tatoeba",
                                          "language_pair": language_pair, "pair_id": idx})
                        ids.append(f"tatoeba_{language_pair}_{idx}_{si}")

        self._upsert_documents(texts, metadatas, ids, min_length=10)
        print(f"  Tatoeba done – {len(texts)} docs")


    def save_knowledge_graph(self, prefix: str = "kg_unified"):
        print(f"\n{'='*60}\nSaving Knowledge Graph\n{'='*60}")

        graphml_path = self.kg_output_path / f"{prefix}.graphml"
        nx.write_graphml(self.graph, graphml_path)

        triples_path = self.kg_output_path / f"triples_{prefix}.json"
        with open(triples_path, 'w', encoding='utf-8') as f:
            json.dump(self.triples, f, indent=2, ensure_ascii=False)

        stats_path = self.kg_output_path / f"{prefix}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_triples": len(self.triples),
                "unique_entities": len(self.entities),
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "sources": list({t.get("source", "unknown") for t in self.triples}),
            }, f, indent=2)

        print(f"  GraphML:  {graphml_path}")
        print(f"  Triples:  {triples_path}")
        print(f"  Stats:    {stats_path}")

    def run_full_pipeline(self, *,
                          include_wikiann: bool = True,
                          include_flores: bool = True,
                          include_tydiqa: bool = True,
                          include_tatoeba: bool = True,
                          max_samples_per_dataset: int = 2000):

        print("\n" + "=" * 70)
        print(" UNIFIED DATASET INGESTION PIPELINE")
        print("=" * 70)

        if include_wikiann:
            self.ingest_wikiann(max_samples=max_samples_per_dataset)
        if include_flores:
            self.ingest_flores(max_samples=max_samples_per_dataset)
        if include_tydiqa:
            self.ingest_tydiqa(max_samples=max_samples_per_dataset)
        if include_tatoeba:
            self.ingest_tatoeba(max_samples=max_samples_per_dataset)

        self.save_knowledge_graph("kg_unified")

        print("\n" + "=" * 70)
        print(" FINAL STATISTICS")
        print("=" * 70)
        print(f"  Documents added:    {self.stats['documents_added']}")
        print(f"  KG triples:         {self.stats['triples_added']}")
        print(f"  Unique entities:    {len(self.entities)}")
        print(f"  Duplicates skipped: {self.stats['duplicates_skipped']}")
        print(f"  Graph nodes:        {self.graph.number_of_nodes()}")
        print(f"  Graph edges:        {self.graph.number_of_edges()}")
        print("=" * 70)
        return self.stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Dataset Ingestion Pipeline")
    parser.add_argument("--chroma-path", default="./chroma_db")
    parser.add_argument("--kg-path", default="./knowledge_graph")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--no-wikiann", action="store_true")
    parser.add_argument("--no-flores", action="store_true")
    parser.add_argument("--no-tydiqa", action="store_true")
    parser.add_argument("--no-tatoeba", action="store_true")
    args = parser.parse_args()

    pipeline = DatasetIngestionPipeline(
        chroma_db_path=args.chroma_path,
        kg_output_path=args.kg_path,
    )
    pipeline.run_full_pipeline(
        include_wikiann=not args.no_wikiann,
        include_flores=not args.no_flores,
        include_tydiqa=not args.no_tydiqa,
        include_tatoeba=not args.no_tatoeba,
        max_samples_per_dataset=args.max_samples,
    )
