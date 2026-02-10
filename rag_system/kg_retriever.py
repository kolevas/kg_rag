"""Re-implement retrieval methods to better leverage the structure of the knowledge graph and improve relevance of retrieved triples."""
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict
from collections import deque

class KnowledgeGraphRetriever:
    
    def __init__(self, kg_path: str = None):
        if kg_path is None:
            kg_path = Path(__file__).parent
        
        self.kg_path = Path(kg_path)
        self.graph = None
        self.triples = []
        self.entities = set()
        self.relations = set()
        
        self._load_kg()
    
    def _load_kg(self):
        try:
            graphml_files = [ self.kg_path / "kg_unified.graphml",self.kg_path / "kg_flores.graphml"]
            
            for graphml_file in graphml_files:
                if graphml_file.exists():
                    self.graph = nx.read_graphml(graphml_file)
                    print(f"Loaded KG from {graphml_file}")
                    break
            
            triples_files = [self.kg_path / "triples_kg_unified.json",self.kg_path / "triples_flores.json"]
            
            for triples_file in triples_files:
                if triples_file.exists():
                    with open(triples_file, 'r', encoding='utf-8') as f:
                        triples_data = json.load(f)
                        self.triples = triples_data
                        
                        for triple in triples_data:
                            self.entities.add(triple['subject'].lower())
                            self.entities.add(triple['object'].lower())
                            self.relations.add(triple['relation'].lower())
                    
                    print(f"Loaded {len(self.triples)} triples from {triples_file}")
                    break
            
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
        matched_entities = []
        
        # Sort by length descending to match longer phrases first
        sorted_entities = sorted(self.entities, key=len, reverse=True)
        
        for entity in sorted_entities:
            # Try exact match
            if entity in query_lower:
                matched_entities.append(entity)
            # Try keyword matching (more lenient)
            else:
                entity_words = set(entity.split())
                query_words = set(query_lower.split())
                if entity_words & query_words:  # if there's overlap
                    matched_entities.append(entity)
        
        return matched_entities
    
    def _extract_relations_from_query(self, query: str) -> List[str]:
        query_lower = query.lower()
        matched_relations = []
        
        for relation in self.relations:
            # Replace underscores with spaces for matching
            relation_text = relation.replace('_', ' ')
            if relation_text in query_lower:
                matched_relations.append(relation)
        
        return matched_relations
    
    def _get_entity_neighbors(self, entity: str, depth: int = 2) -> Dict[str, List[Dict]]:

        entity_lower = entity.lower()
        result = {
            'outgoing': [],  # entity -> others
            'incoming': []   # others -> entity
        }
        
        if self.graph is None or entity_lower not in self.graph:
            return result
        
        # BFS for outgoing edges
        visited = set()
        queue = deque([(entity_lower, 0)])
        
        while queue:
            current, dist = queue.popleft()
            if current in visited or dist >= depth:
                continue
            visited.add(current)
            
            # Get outgoing edges
            for target in self.graph.successors(current):
                edges = self.graph.get_edge_data(current, target)
                for _, edge_data in edges.items():
                    relation = edge_data.get('relation', 'related_to')
                    result['outgoing'].append({
                        'subject': current,
                        'relation': relation,
                        'object': target
                    })
                
                if dist < depth - 1:
                    queue.append((target, dist + 1))
        
        # Get incoming edges
        visited = set()
        queue = deque([(entity_lower, 0)])
        
        while queue:
            current, dist = queue.popleft()
            if current in visited or dist >= depth:
                continue
            visited.add(current)
            
            for source in self.graph.predecessors(current):
                edges = self.graph.get_edge_data(source, current)
                for _, edge_data in edges.items():
                    relation = edge_data.get('relation', 'related_to')
                    result['incoming'].append({
                        'subject': source,
                        'relation': relation,
                        'object': current
                    })
                
                if dist < depth - 1:
                    queue.append((source, dist + 1))
        
        return result
    
    def retrieve_kg_context(self, query: str, max_triples: int = 10) -> str:

        if not self.triples:
            return ""
        
        # Extract entities and relations from query
        matched_entities = self._extract_entities_from_query(query)
        matched_relations = self._extract_relations_from_query(query)
        
        # Collect relevant triples
        relevant_triples = []
        
        # 1. Get triples for matched entities
        for entity in matched_entities:
            neighbors = self._get_entity_neighbors(entity, depth=2)
            relevant_triples.extend(neighbors['outgoing'])
            relevant_triples.extend(neighbors['incoming'])
        
        # 2. Score all triples by keyword/substring overlap
        query_lower = query.lower()
        scored_triples = []
        
        for triple in self.triples:
            score = 0
            
            # Check for substring matches (important, e.g., "stanford" in "стенфорд")
            triple_text = f"{triple['subject']} {triple['relation']} {triple['object']}".lower()
            
            # Word-level matching
            query_words = set(query_lower.split())
            triple_words = set(triple_text.split())
            word_overlap = len(query_words & triple_words)
            score += word_overlap * 10  # Weight word overlap heavily
            
            # Character sequence matching (helps with transliterations)
            for query_part in query_lower.split():
                if len(query_part) > 3:  # Only for longer words
                    if query_part in triple_text or any(
                        query_part[i:i+4] in triple_text 
                        for i in range(len(query_part)-3)
                    ):
                        score += 5
            
            # Common knowledge entities bonus
            if any(keyword in triple_text for keyword in 
                   ['university', 'stanford', 'scientist', 'printer', 'chip', 'media',
                    'universitet', 'naučn', 'científ', 'печата', 'научн']):
                score += 3
            
            if score > 0:
                scored_triples.append({
                    'subject': triple['subject'],
                    'relation': triple['relation'],
                    'object': triple['object'],
                    'score': score
                })
        
        # Sort by score and remove duplicates
        scored_triples = sorted(scored_triples, key=lambda x: x['score'], reverse=True)
        
        seen = set()
        unique_triples = []
        for triple in scored_triples:
            key = (triple['subject'].lower(), triple['relation'].lower(), triple['object'].lower())
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        unique_triples = unique_triples[:max_triples]
        
        # If still no results, add some from matched entities with zero score
        if not unique_triples and matched_entities:
            for entity in matched_entities[:2]:
                neighbors = self._get_entity_neighbors(entity, depth=1)
                for triple in (neighbors['outgoing'] + neighbors['incoming'])[:3]:
                    key = (triple['subject'].lower(), triple['relation'].lower(), triple['object'].lower())
                    if key not in seen:
                        seen.add(key)
                        unique_triples.append(triple)
        
        # Last resort: just return some triples from entities that exist
        if not unique_triples:
            import random
            sample_size = min(max_triples, max(5, len(self.triples) // 100))
            unique_triples = [
                {
                    'subject': t['subject'],
                    'relation': t['relation'],
                    'object': t['object']
                }
                for t in random.sample(self.triples, sample_size)
            ]
        
        if not unique_triples:
            return ""
        
        # Format as context
        context_lines = ["## Knowledge Graph Context:"]
        for i, triple in enumerate(unique_triples[:max_triples], 1):
            context_lines.append(
                f"{i}. {triple['subject']} --[{triple['relation']}]--> {triple['object']}"
            )
        
        return "\n".join(context_lines)
    
    def query_triples(self, subject: str = None, relation: str = None, 
                     object_: str = None) -> List[Dict]:
        """
        Query triples by subject, relation, and/or object.
        
        Args:
            subject: Filter by subject (case-insensitive)
            relation: Filter by relation (case-insensitive)
            object_: Filter by object (case-insensitive)
            
        Returns:
            List of matching triples
        """
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
    
    def get_entity_info(self, entity: str) -> Dict:
        """Get all information about an entity."""
        entity_lower = entity.lower()
        
        return {
            'entity': entity,
            'outgoing_relations': [
                t for t in self.triples 
                if t['subject'].lower() == entity_lower
            ],
            'incoming_relations': [
                t for t in self.triples 
                if t['object'].lower() == entity_lower
            ]
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded KG."""
        return {
            'total_triples': len(self.triples),
            'unique_entities': len(self.entities),
            'unique_relations': len(self.relations),
            'graph_nodes': self.graph.number_of_nodes() if self.graph else 0,
            'graph_edges': self.graph.number_of_edges() if self.graph else 0
        }


# Example usage
if __name__ == "__main__":
    retriever = KnowledgeGraphRetriever()
    
    print("\n=== KG Stats ===")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test queries
    test_queries = [
        "Tell me about Stanford University",
        "What do we know about the medical faculty?",
        "Who are the scientists?"
    ]
    
    print("\n=== Test Queries ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        context = retriever.retrieve_kg_context(query, max_triples=5)
        print(context)
