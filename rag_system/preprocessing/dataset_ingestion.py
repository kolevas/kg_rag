"""
Unified Dataset Ingestion Pipeline
Ingests unstructured data into both Vector Database (ChromaDB) and Knowledge Graph.

Supported Datasets:
- WikiANN (Macedonian NER) - Pre-labeled entities for KG
- FLORES (Macedonian) - Sentences for vector DB + triple extraction
- TyDiQA - QA passages for vector DB
- Tatoeba - Bilingual sentence pairs
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from datasets import load_dataset
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import networkx as nx

load_dotenv()

# Optional: Import Groq for triple extraction
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not available. LLM-based triple extraction disabled.")


class DatasetIngestionPipeline:
    """
    Unified pipeline for ingesting datasets into both Vector DB and Knowledge Graph.
    """
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 kg_output_path: str = "./knowledge_graph",
                 collection_name: str = "multimodal_data"):
        
        self.chroma_db_path = chroma_db_path
        self.kg_output_path = Path(kg_output_path)
        self.kg_output_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # Initialize Knowledge Graph
        self.graph = nx.MultiDiGraph()
        self.triples: List[Dict[str, str]] = []
        self.seen_triples: set = set()
        self.entities: set = set()
        
        # Initialize Groq client for triple extraction (if available)
        self.groq_client = None
        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                print("✅ Groq client initialized for triple extraction")
        
        # Statistics
        self.stats = {
            "documents_added": 0,
            "triples_added": 0,
            "entities_added": 0,
            "duplicates_skipped": 0
        }
    
    # ==================== VECTOR DB METHODS ====================
    
    def add_to_vector_db(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to ChromaDB vector database."""
        if not texts:
            return
        
        # Filter out empty texts
        valid_data = [
            (t, m, i) for t, m, i in zip(texts, metadatas, ids) 
            if t and len(t.strip()) > 20
        ]
        
        if not valid_data:
            return
        
        texts, metadatas, ids = zip(*valid_data)
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i+batch_size])
            batch_metadatas = list(metadatas[i:i+batch_size])
            batch_ids = list(ids[i:i+batch_size])
            
            try:
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                self.stats["documents_added"] += len(batch_texts)
            except Exception as e:
                print(f"⚠️ Error adding batch to vector DB: {e}")
    
    # ==================== KNOWLEDGE GRAPH METHODS ====================
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity for deduplication."""
        return entity.lower().strip()
    
    def _add_triple(self, subject: str, relation: str, obj: str, source: str = "unknown") -> bool:
        """Add a triple to the knowledge graph."""
        # Normalize for deduplication
        normalized = (
            self._normalize_entity(subject),
            relation.lower().strip(),
            self._normalize_entity(obj)
        )
        
        if normalized in self.seen_triples:
            self.stats["duplicates_skipped"] += 1
            return False
        
        self.seen_triples.add(normalized)
        
        # Add to graph
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(obj, type="entity")
        self.graph.add_edge(subject, obj, relation=relation, source=source)
        
        # Add to triples list
        self.triples.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "source": source
        })
        
        # Track entities
        self.entities.add(subject)
        self.entities.add(obj)
        
        self.stats["triples_added"] += 1
        return True
    
    def extract_triples_with_llm(self, text: str, source: str = "llm") -> List[Tuple[str, str, str]]:
        """Extract triples from text using Groq LLM."""
        if not self.groq_client or not text or len(text.strip()) < 20:
            return []
        
        prompt = f"""You are an expert in extracting knowledge from Macedonian text.

Extract knowledge triples (subject, relation, object) from the following text.

Text: {text}

Provide triples in the format:
subject | relation | object

IMPORTANT RULES:
1. DO NOT number entities - just the entity name
2. Use specific entities from the text
3. Extract 3-10 relevant triples
4. Return ONLY triples, NO explanations

Triples:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=500,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            return self._parse_llm_triples(result)
        
        except Exception as e:
            print(f"⚠️ LLM extraction error: {e}")
            return []
    
    def _parse_llm_triples(self, llm_response: str) -> List[Tuple[str, str, str]]:
        """Parse LLM response into triples."""
        triples = []
        
        for line in llm_response.strip().split('\n'):
            match = re.match(r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$', line.strip())
            if match:
                subject, relation, obj = match.groups()
                subject = self._clean_entity(subject)
                obj = self._clean_entity(obj)
                relation = relation.strip().strip('"\'').lower().replace(' ', '_')
                
                if subject and relation and obj:
                    triples.append((subject, relation, obj))
        
        return triples
    
    def _clean_entity(self, entity: str) -> str:
        """Clean entity string."""
        entity = entity.strip().strip('\"\'')
        entity = re.sub(r'^[\d\*\-]+\.?\s+', '', entity)
        return entity.strip()
    
    # ==================== WIKIANN INGESTION ====================
    
    def ingest_wikiann(self, language: str = "mk", max_samples: int = 5000):
        """
        Ingest WikiANN dataset (NER data with pre-labeled entities).
        
        WikiANN provides:
        - Tokens with NER tags (PER, ORG, LOC)
        - Pre-labeled entities that can be directly converted to KG nodes
        """
        print(f"\n{'='*60}")
        print(f"📥 Ingesting WikiANN ({language}) - max {max_samples} samples")
        print(f"{'='*60}")
        
        try:
            dataset = load_dataset("wikiann", language, trust_remote_code=True)
        except Exception as e:
            print(f"❌ Failed to load WikiANN: {e}")
            return
        
        # NER tag mapping
        tag_map = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}
        
        texts = []
        metadatas = []
        ids = []
        
        entity_pairs = defaultdict(list)  # Track entities appearing together
        
        for split in ["train", "validation", "test"]:
            if split not in dataset:
                continue
            
            split_data = dataset[split]
            sample_size = min(max_samples // 3, len(split_data))
            
            for idx in range(sample_size):
                item = split_data[idx]
                tokens = item["tokens"]
                ner_tags = item["ner_tags"]
                
                # Reconstruct sentence
                sentence = " ".join(tokens)
                
                # Extract named entities
                current_entity = []
                current_type = None
                entities_in_sentence = []
                
                for token, tag_id in zip(tokens, ner_tags):
                    tag = tag_map.get(tag_id, "O")
                    
                    if tag.startswith("B-"):
                        # Save previous entity
                        if current_entity:
                            entity_text = " ".join(current_entity)
                            entities_in_sentence.append((entity_text, current_type))
                        # Start new entity
                        current_entity = [token]
                        current_type = tag[2:]  # PER, ORG, or LOC
                    elif tag.startswith("I-") and current_type == tag[2:]:
                        current_entity.append(token)
                    else:
                        if current_entity:
                            entity_text = " ".join(current_entity)
                            entities_in_sentence.append((entity_text, current_type))
                            current_entity = []
                            current_type = None
                
                # Don't forget last entity
                if current_entity:
                    entity_text = " ".join(current_entity)
                    entities_in_sentence.append((entity_text, current_type))
                
                # Add to vector DB
                if sentence and len(sentence) > 20:
                    doc_id = f"wikiann_{language}_{split}_{idx}"
                    texts.append(sentence)
                    metadatas.append({
                        "source": "wikiann",
                        "language": language,
                        "split": split,
                        "has_entities": len(entities_in_sentence) > 0,
                        "entity_count": len(entities_in_sentence)
                    })
                    ids.append(doc_id)
                
                # Create KG triples from entities
                for entity_text, entity_type in entities_in_sentence:
                    # Add entity type relation
                    type_relation = f"is_{entity_type.lower()}"
                    self._add_triple(entity_text, type_relation, entity_type, source="wikiann")
                    
                    # Track co-occurring entities for relationship extraction
                    for other_entity, other_type in entities_in_sentence:
                        if entity_text != other_entity:
                            entity_pairs[(entity_text, entity_type)].append((other_entity, other_type))
        
        # Add co-occurrence relationships to KG
        print("🔗 Creating co-occurrence relationships...")
        for (entity, etype), related in entity_pairs.items():
            # Count co-occurrences
            related_counts = defaultdict(int)
            for rel_entity, rel_type in related:
                related_counts[(rel_entity, rel_type)] += 1
            
            # Add relations for frequently co-occurring entities
            for (rel_entity, rel_type), count in related_counts.items():
                if count >= 2:  # At least 2 co-occurrences
                    # Determine relation based on entity types
                    if etype == "PER" and rel_type == "ORG":
                        relation = "affiliated_with"
                    elif etype == "PER" and rel_type == "LOC":
                        relation = "located_in"
                    elif etype == "ORG" and rel_type == "LOC":
                        relation = "based_in"
                    else:
                        relation = "related_to"
                    
                    self._add_triple(entity, relation, rel_entity, source="wikiann_cooccurrence")
        
        # Batch add to vector DB
        self.add_to_vector_db(texts, metadatas, ids)
        
        print(f"✅ WikiANN ingestion complete")
        print(f"   - Documents: {len(texts)}")
        print(f"   - Entities extracted: {len(self.entities)}")
    
    # ==================== FLORES INGESTION ====================
    
    def ingest_flores(self, language_code: str = "mkd_Cyrl", max_samples: int = 1000):
        """
        Ingest FLORES dataset (parallel sentences).
        Uses LLM for triple extraction.
        """
        print(f"\n{'='*60}")
        print(f"📥 Ingesting FLORES ({language_code}) - max {max_samples} samples")
        print(f"{'='*60}")
        
        try:
            dataset = load_dataset("facebook/flores", language_code, split="dev", trust_remote_code=True)
        except Exception as e:
            print(f"❌ Failed to load FLORES: {e}")
            return
        
        texts = []
        metadatas = []
        ids = []
        
        sample_size = min(max_samples, len(dataset))
        
        for idx in range(sample_size):
            item = dataset[idx]
            sentence = item["sentence"]
            
            if not sentence or len(sentence.strip()) < 20:
                continue
            
            # Add to vector DB
            doc_id = f"flores_{language_code}_{idx}"
            texts.append(sentence)
            metadatas.append({
                "source": "flores",
                "language": language_code,
                "sentence_id": item.get("id", idx)
            })
            ids.append(doc_id)
            
            # Extract triples with LLM (every 3rd sentence to save API calls)
            if self.groq_client and idx % 3 == 0:
                triples = self.extract_triples_with_llm(sentence, source="flores")
                for subj, rel, obj in triples:
                    self._add_triple(subj, rel, obj, source="flores")
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{sample_size} sentences...")
        
        # Batch add to vector DB
        self.add_to_vector_db(texts, metadatas, ids)
        
        print(f"✅ FLORES ingestion complete")
        print(f"   - Documents: {len(texts)}")
    
    # ==================== TYDIQA INGESTION ====================
    
    def ingest_tydiqa(self, max_samples: int = 2000):
        """
        Ingest TyDiQA dataset (multilingual QA with passages).
        Great for RAG context documents.
        """
        print(f"\n{'='*60}")
        print(f"📥 Ingesting TyDiQA - max {max_samples} samples")
        print(f"{'='*60}")
        
        try:
            # Try different dataset names
            dataset = None
            dataset_names = [
                ("tydiqa", "secondary_task"),
                ("copenlu/answerable_tydiqa", None),
                ("khalidalt/tydiqa-goldp", None),
            ]
            
            for name, config in dataset_names:
                try:
                    if config:
                        dataset = load_dataset(name, config, trust_remote_code=True)
                    else:
                        dataset = load_dataset(name, trust_remote_code=True)
                    print(f"✅ Loaded TyDiQA from: {name}")
                    break
                except Exception:
                    continue
            
            if dataset is None:
                print("⚠️ Could not load TyDiQA from any source. Skipping...")
                return
                
        except Exception as e:
            print(f"❌ Failed to load TyDiQA: {e}")
            return
        
        texts = []
        metadatas = []
        ids = []
        
        seen_contexts = set()  # Deduplicate contexts
        
        for split in ["train", "validation"]:
            if split not in dataset:
                continue
            
            split_data = dataset[split]
            sample_size = min(max_samples // 2, len(split_data))
            
            for idx in range(sample_size):
                item = split_data[idx]
                
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", {})
                language = item.get("id", "").split("-")[0] if "-" in item.get("id", "") else "unknown"
                
                # Skip if already seen this context
                context_hash = hash(context[:200])
                if context_hash in seen_contexts:
                    continue
                seen_contexts.add(context_hash)
                
                if not context or len(context.strip()) < 50:
                    continue
                
                # Add context to vector DB
                doc_id = f"tydiqa_{split}_{idx}"
                texts.append(context)
                metadatas.append({
                    "source": "tydiqa",
                    "language": language,
                    "split": split,
                    "has_qa": True,
                    "question": question[:200] if question else ""
                })
                ids.append(doc_id)
                
                # Create simple QA-based triples
                if question and answers.get("text"):
                    answer_text = answers["text"][0] if answers["text"] else ""
                    if answer_text:
                        # Extract a simple relation
                        self._add_triple(
                            question[:100],
                            "has_answer",
                            answer_text[:100],
                            source="tydiqa"
                        )
        
        # Batch add to vector DB
        self.add_to_vector_db(texts, metadatas, ids)
        
        print(f"✅ TyDiQA ingestion complete")
        print(f"   - Documents: {len(texts)}")
    
    # ==================== TATOEBA INGESTION ====================
    
    def ingest_tatoeba(self, language_pair: str = "mkd-eng", max_samples: int = 2000):
        """
        Ingest Tatoeba bilingual sentence pairs.
        Useful for Macedonian-English parallel data.
        """
        print(f"\n{'='*60}")
        print(f"📥 Ingesting Tatoeba ({language_pair}) - max {max_samples} samples")
        print(f"{'='*60}")
        
        try:
            dataset = load_dataset("mteb/tatoeba-bitext-mining", language_pair, trust_remote_code=True)
        except Exception as e:
            print(f"❌ Failed to load Tatoeba: {e}")
            # Try alternative name format
            try:
                alt_pair = language_pair.replace("-", "_")
                dataset = load_dataset("mteb/tatoeba-bitext-mining", alt_pair, trust_remote_code=True)
            except:
                print(f"❌ Could not load Tatoeba with either format")
                return
        
        texts = []
        metadatas = []
        ids = []
        
        for split in dataset.keys():
            split_data = dataset[split]
            sample_size = min(max_samples, len(split_data))
            
            for idx in range(sample_size):
                item = split_data[idx]
                
                # Tatoeba has sentence1 and sentence2
                sent1 = item.get("sentence1", "")
                sent2 = item.get("sentence2", "")
                
                # Add both sentences
                for sent_idx, sentence in enumerate([sent1, sent2]):
                    if sentence and len(sentence.strip()) > 10:
                        doc_id = f"tatoeba_{language_pair}_{idx}_{sent_idx}"
                        texts.append(sentence)
                        metadatas.append({
                            "source": "tatoeba",
                            "language_pair": language_pair,
                            "is_translation": True,
                            "pair_id": idx
                        })
                        ids.append(doc_id)
        
        # Batch add to vector DB
        self.add_to_vector_db(texts, metadatas, ids)
        
        print(f"✅ Tatoeba ingestion complete")
        print(f"   - Documents: {len(texts)}")
    
    # ==================== SAVE KNOWLEDGE GRAPH ====================
    
    def save_knowledge_graph(self, filename_prefix: str = "kg_unified"):
        """Save the knowledge graph to files."""
        print(f"\n{'='*60}")
        print(f"💾 Saving Knowledge Graph")
        print(f"{'='*60}")
        
        # Save GraphML
        graphml_path = self.kg_output_path / f"{filename_prefix}.graphml"
        nx.write_graphml(self.graph, graphml_path)
        print(f"   - GraphML: {graphml_path}")
        
        # Save triples JSON
        triples_path = self.kg_output_path / f"triples_{filename_prefix}.json"
        with open(triples_path, 'w', encoding='utf-8') as f:
            json.dump(self.triples, f, indent=2, ensure_ascii=False)
        print(f"   - Triples JSON: {triples_path}")
        
        # Save statistics
        stats_path = self.kg_output_path / f"{filename_prefix}_stats.json"
        kg_stats = {
            "total_triples": len(self.triples),
            "unique_entities": len(self.entities),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "sources": list(set(t.get("source", "unknown") for t in self.triples))
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(kg_stats, f, indent=2)
        print(f"   - Stats: {stats_path}")
        
        print(f"\n✅ Knowledge Graph saved successfully!")
    
    # ==================== RUN FULL PIPELINE ====================
    
    def run_full_pipeline(self, 
                          include_wikiann: bool = True,
                          include_flores: bool = True,
                          include_tydiqa: bool = True,
                          include_tatoeba: bool = True,
                          max_samples_per_dataset: int = 2000):
        """
        Run the full ingestion pipeline for all datasets.
        """
        print("\n" + "="*70)
        print("🚀 UNIFIED DATASET INGESTION PIPELINE")
        print("="*70)
        print(f"Target: Vector DB ({self.chroma_db_path}) + Knowledge Graph ({self.kg_output_path})")
        print("="*70)
        
        if include_wikiann:
            self.ingest_wikiann(language="mk", max_samples=max_samples_per_dataset)
        
        if include_flores:
            self.ingest_flores(language_code="mkd_Cyrl", max_samples=max_samples_per_dataset)
        
        if include_tydiqa:
            self.ingest_tydiqa(max_samples=max_samples_per_dataset)
        
        if include_tatoeba:
            self.ingest_tatoeba(language_pair="mkd-eng", max_samples=max_samples_per_dataset)
        
        # Save knowledge graph
        self.save_knowledge_graph("kg_unified")
        
        # Print final statistics
        print("\n" + "="*70)
        print("📊 FINAL STATISTICS")
        print("="*70)
        print(f"Vector Database:")
        print(f"   - Total documents added: {self.stats['documents_added']}")
        print(f"   - Collection: {self.collection_name}")
        print(f"\nKnowledge Graph:")
        print(f"   - Total triples: {self.stats['triples_added']}")
        print(f"   - Unique entities: {len(self.entities)}")
        print(f"   - Duplicates skipped: {self.stats['duplicates_skipped']}")
        print(f"   - Graph nodes: {self.graph.number_of_nodes()}")
        print(f"   - Graph edges: {self.graph.number_of_edges()}")
        print("="*70)
        
        return self.stats


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Dataset Ingestion Pipeline")
    parser.add_argument("--chroma-path", default="./chroma_db", help="ChromaDB path")
    parser.add_argument("--kg-path", default="./knowledge_graph", help="Knowledge graph output path")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max samples per dataset")
    parser.add_argument("--no-wikiann", action="store_true", help="Skip WikiANN")
    parser.add_argument("--no-flores", action="store_true", help="Skip FLORES")
    parser.add_argument("--no-tydiqa", action="store_true", help="Skip TyDiQA")
    parser.add_argument("--no-tatoeba", action="store_true", help="Skip Tatoeba")
    
    args = parser.parse_args()
    
    pipeline = DatasetIngestionPipeline(
        chroma_db_path=args.chroma_path,
        kg_output_path=args.kg_path,
        collection_name="multimodal_data"
    )
    
    pipeline.run_full_pipeline(
        include_wikiann=not args.no_wikiann,
        include_flores=not args.no_flores,
        include_tydiqa=not args.no_tydiqa,
        include_tatoeba=not args.no_tatoeba,
        max_samples_per_dataset=args.max_samples
    )
