"""Re-implement retrieval methods to better leverage the structure of the knowledge graph and improve relevance of retrieved triples."""
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict
from collections import deque

class KnowledgeGraphRetriever:
    
    def __init__(self, kg_path: str = None):
        if kg_path is None:
            kg_path = Path(__file__).parent / "knowledge_graph"
        
        self.kg_path = Path(kg_path)
        self.graph = None
        self.triples = []
        self.entities = set()
        self.relations = set()
        
        self._load_kg()
    
    def _load_kg(self):
        try:
            graphml_files = [self.kg_path / "kg_unified.graphml", self.kg_path / "kg_flores.graphml"]

            for graphml_file in graphml_files:
                if graphml_file.exists():
                    self.graph = nx.read_graphml(graphml_file)
                    print(f"Loaded KG from {graphml_file}")
                    break

            triples_files = [
                self.kg_path / "triples_kg_unified.json",
                self.kg_path / "triples_flores.json",
                self.kg_path / "triples_wikipedia.json",
            ]

            for triples_file in triples_files:
                if triples_file.exists():
                    with open(triples_file, 'r', encoding='utf-8') as f:
                        triples_data = json.load(f)
                        if not triples_data:
                            continue
                        self.triples.extend(triples_data)

                        for triple in triples_data:
                            self.entities.add(triple['subject'].lower())
                            self.entities.add(triple['object'].lower())
                            self.relations.add(triple['relation'].lower())

                    print(f"Loaded {len(triples_data)} triples from {triples_file}")

            print(f"Total: {len(self.triples)} triples, {len(self.entities)} entities, {len(self.relations)} relations")

            if self.graph is None:
                print("GraphML file not found, creating graph from triples")
                self._build_graph_from_triples()
        
        except Exception as e:
            print(f"Error loading KG: {e}")
            self.graph = nx.MultiDiGraph()
            self.triples = []
            self.entities = set()
            self.relations = set()
    
    def _build_graph_from_triples(self):
        self.graph = nx.MultiDiGraph()
        
        for triple in self.triples:
            subject = triple['subject']
            relation = triple['relation']
            obj = triple['object']
            
            self.graph.add_node(subject, type="entity")
            self.graph.add_node(obj, type="entity")
            self.graph.add_edge(subject, obj, relation=relation)
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        matched = []
        sorted_entities = sorted(self.entities, key=len, reverse=True)

        for entity in sorted_entities:
            if entity in query_lower:
                matched.append(entity)
                continue

            entity_words = {w for w in entity.split() if len(w) >= 3}
            if not entity_words:
                continue
            overlap = entity_words & query_words
            if len(overlap) >= max(1, len(entity_words) // 2):
                matched.append(entity)

        return matched
    
    def _extract_relations_from_query(self, query: str) -> List[str]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        matched = []

        for relation in self.relations:
            relation_text = relation.replace('_', ' ')
            if relation_text in query_lower:
                matched.append(relation)
                continue
            rel_words = relation_text.split()
            if len(rel_words) == 1 and rel_words[0] in query_words:
                matched.append(relation)

        return matched
    
    def _get_entity_neighbors(self, entity: str, depth: int = 2) -> List[Dict]:
        #BFS
        entity_lower = entity.lower()
        triples = []

        if self.graph is None or entity_lower not in self.graph:
            return triples

        visited = set()
        queue = deque([(entity_lower, 0)])
        seen_edges = set()

        while queue:
            current, dist = queue.popleft()
            if current in visited or dist >= depth:
                continue
            visited.add(current)

            # Outgoing edges
            for target in self.graph.successors(current):
                for _, edge_data in self.graph.get_edge_data(current, target).items():
                    rel = edge_data.get('relation', 'related_to')
                    key = (current, rel, target)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        triples.append({'subject': current, 'relation': rel, 'object': target})
                if dist + 1 < depth:
                    queue.append((target, dist + 1))

            # Incoming edges
            for source in self.graph.predecessors(current):
                for _, edge_data in self.graph.get_edge_data(source, current).items():
                    rel = edge_data.get('relation', 'related_to')
                    key = (source, rel, current)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        triples.append({'subject': source, 'relation': rel, 'object': current})
                if dist + 1 < depth:
                    queue.append((source, dist + 1))

        return triples
    
    def _score_triple(self, triple: Dict, query_words: set) -> float:
        triple_text = f"{triple['subject']} {triple['relation']} {triple['object']}".lower()
        triple_words = set(triple_text.split())
        overlap = query_words & {w for w in triple_words if len(w) >= 3}
        return len(overlap)

    def retrieve_kg_context(self, query: str, max_triples: int = 10) -> str:
        if not self.triples:
            return ""

        query_lower = query.lower()
        query_words = {w for w in query_lower.split() if len(w) >= 3}

        if not query_words:
            return ""

        matched_entities = self._extract_entities_from_query(query)
        seen = set()
        scored: List[tuple] = []  

        for entity in matched_entities:
            for t in self._get_entity_neighbors(entity, depth=2):
                key = (t['subject'].lower(), t['relation'].lower(), t['object'].lower())
                if key not in seen:
                    seen.add(key)
                    score = self._score_triple(t, query_words) + 1  # +1 bonus for graph proximity
                    scored.append((score, t))

        # 2. Score remaining triples by word overlap
        for t in self.triples:
            key = (t['subject'].lower(), t['relation'].lower(), t['object'].lower())
            if key in seen:
                continue
            score = self._score_triple(t, query_words)
            if score > 0:
                seen.add(key)
                scored.append((score, t))

        if not scored:
            return ""

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in scored[:max_triples]]

        context_lines = ["## Knowledge Graph Context:"]
        for i, triple in enumerate(top, 1):
            context_lines.append(
                f"{i}. {triple['subject']} --[{triple['relation']}]--> {triple['object']}"
            )
        return "\n".join(context_lines)
    
    def query_triples(self, subject: str = None, relation: str = None, 
                     object_: str = None) -> List[Dict]:
        
        results = []
        
        for triple in self.triples:
            subject_match = (
                subject is None or 
                triple['subject'].lower() == subject.lower()
            )
            relation_match = (
                relation is None or 
                triple['relation'].lower() == relation.lower()
            )
            object_match = (
                object_ is None or 
                triple['object'].lower() == object_.lower()
            )
            
            if subject_match and relation_match and object_match:
                results.append(triple)
        
        return results

    def get_stats(self) -> Dict:
        return {
            'total_triples': len(self.triples),
            'unique_entities': len(self.entities),
            'unique_relations': len(self.relations),
            'graph_nodes': self.graph.number_of_nodes() if self.graph else 0,
            'graph_edges': self.graph.number_of_edges() if self.graph else 0
        }


if __name__ == "__main__":
    retriever = KnowledgeGraphRetriever()
    
    print("\n=== KG Stats ===")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    test_queries = [
        "Универзитетот Стенфорд",
        "медицинскиот факултет",
        "научниците",
        "Детроит",
    ]
    
    print("\n=== Test Queries ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        context = retriever.retrieve_kg_context(query, max_triples=5)
        print(context)
