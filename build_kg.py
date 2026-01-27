import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
from collections import Counter
from groq import Groq
from dotenv import load_dotenv
from datasets import load_dataset
load_dotenv()

class TripleExtractor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("api key not found in environment")
        self.client = Groq(api_key=api_key)
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        if not text or len(text.strip()) < 20:
            return []
        
        prompt = f"""You are an expert in extracting knowledge from Macedonian text.

Extract knowledge triples (subject, relation, object) from the following MACEDONIAN text.

Macedonian Text: {text}

Provide triples in the format:
subject | relation | object

IMPORTANT RULES:
1. DO NOT number entities (no "1.", "2.", etc.) - just the entity name
2. Use specific entities from the text, not generic pronouns or categories
3. Triple direction: CONTAINER/INSTITUTION comes FIRST as subject
   ✓ CORRECT: Универзитетот | има_факултет | медицинскиот факултет
   ✗ WRONG: медицинскиот факултет | е_дел_од | Универзитетот
   ✓ CORRECT: Универзитетот Стенфорд | има_вработени | научниците
   ✗ WRONG: научниците | работат_во | Универзитетот Стенфорд
4. Relations in Macedonian: има_вработени, има_дел, содржи, се_наоѓа_во, припаѓа_на
5. Entities are concise nouns, NOT full phrases. DO NOT include prepositional phrases:
   ✓ CORRECT: стандардни инк-цет печатачи
   ✗ WRONG: со помош на стандардни инк-цет печатачи
6. Prepositional phrases like "со помош на", "во врска со" belong in RELATIONS, not entities
7. Extract 5-15 relevant triples
8. Return ONLY triples, NO explanations

Triples:"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=800,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            return self._parse_triples(result)
        
        except Exception as e:
            print(f"groq extraction error: {e}")
            return []
    
    def _parse_triples(self, llm_response: str) -> List[Tuple[str, str, str]]:
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
        entity = entity.strip().strip('\"\'')
        entity = re.sub(r'^[\d\*\-]+\.?\s+', '', entity)

        phrases = r'^(со\s+помош\s+на|во\s+врска\s+со|во\s+текот\s+на|од\s+страна\s+на|за\s+време\s+на|во\s+однос\s+на)\s+'
        entity = re.sub(phrases, '', entity, flags=re.IGNORECASE)
        
        return entity.strip()
    
class KnowledgeGraphBuilder:
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.triple_extractor = TripleExtractor()
        self.all_triples = []
        self.seen_triples = set() 
    
    def _normalize_triple(self, subject: str, relation: str, obj: str) -> Tuple[str, str, str]:
        return (subject.lower().strip(), relation.lower().strip(), obj.lower().strip())
    
    def _add_triple(self, subject: str, relation: str, obj: str) -> bool:
        normalized = self._normalize_triple(subject, relation, obj)
        
        if normalized in self.seen_triples:
            return False
        
        self.seen_triples.add(normalized)
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(obj, type="entity")
        self.graph.add_edge(subject, obj, relation=relation)
        self.all_triples.append((subject, relation, obj))
        return True
    
    def process_dataset(self, max_sentences: int = 10):
        dataset = load_dataset("facebook/flores", "mkd_Cyrl", split="dev", trust_remote_code=True)
        
        for idx in range(min(max_sentences, len(dataset))):
            item = dataset[idx]
            print(f"\nProcessing sentence {idx + 1}")
            triples = self.triple_extractor.extract_triples(item['sentence'])
            print(f"extracted {len(triples)} triples")
            if triples:
                print(f"sample: {triples[0]}")
            
            added = sum(1 for s, r, o in triples if self._add_triple(s, r, o))
            duplicates = len(triples) - added
            if duplicates > 0:
                print(f"added {added} new, skipped {duplicates} duplicates")
        
        self._save_graph()
    
    def _save_graph(self):
        output_dir = Path("rag_system/knowledge_graph")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        graph_file = output_dir / "kg_flores.graphml"
        nx.write_graphml(self.graph, graph_file)
        
        triples_file = output_dir / "triples_flores.json"
        with open(triples_file, 'w', encoding='utf-8') as f:
            json.dump([
                {"subject": s, "relation": r, "object": o}
                for s, r, o in self.all_triples
            ], f, indent=2, ensure_ascii=False)
  
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    builder.process_dataset(max_sentences=300)
    
    print("\ncomplete")
