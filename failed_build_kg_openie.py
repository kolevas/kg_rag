"""
Knowledge Graph Builder using Stanford OpenIE
Extracts entities and relationships from Macedonian text using OpenIE
"""
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
import networkx as nx
from datetime import datetime
from collections import defaultdict
from openie import StanfordOpenIE


class OpenIEEntityExtractor:
    """Entity and relationship extractor using Stanford OpenIE."""
    
    def __init__(self):
        print("🔧 Initializing Stanford OpenIE...")
        self.client = StanfordOpenIE()
        print("✅ Stanford OpenIE initialized!")
        
        # Core entity patterns (same as rule-based)
        self.org_indicators = [
            'министерство', 'влада', 'собрание', 'универзитет', 'факултет',
            'институт', 'агенција', 'фонд', 'здружение', 'центар', 'компанија',
            'претпријатие', 'организација', 'комисија', 'управа', 'дирекција',
            'служба', 'одбор', 'совет', 'фондација', 'коалиција'
        ]
        
        self.loc_indicators = [
            'скопје', 'битола', 'прилеп', 'охрид', 'тетово', 'куманово', 'велес',
            'штип', 'гостивар', 'струмица', 'кавадарци', 'македонија', 'србија',
            'бугарија', 'грција', 'албанија', 'косово', 'град', 'село', 'општина',
            'регион', 'држава', 'република'
        ]
        
        # Money/currency patterns
        self.money_pattern = re.compile(
            r'(\d+(?:[.,]\d+)*)\s*(денари|евра|евро|долари|долар|милион|милиони|милијарда|милијарди)',
            re.IGNORECASE
        )
        
        # Date patterns
        self.date_pattern = re.compile(
            r'(\d{4})\s*година|'
            r'(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})|'
            r'(јануари|февруари|март|април|мај|јуни|јули|август|септември|октомври|ноември|декември)\s+(\d{4})',
            re.IGNORECASE
        )
        
        # Metric patterns
        self.metric_pattern = re.compile(
            r'(\d+(?:[.,]\d+)?)\s*(%|процент|проценти|км|метри|km|m)',
            re.IGNORECASE
        )
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities from text using patterns."""
        entities = {
            'ORG': [],
            'LOC': [],
            'DATE': [],
            'MONEY': [],
            'PROJECT': [],
            'METRIC': []
        }
        
        # Extract MONEY entities
        for match in self.money_pattern.finditer(text):
            entities['MONEY'].append({
                'text': match.group(0),
                'value': match.group(1),
                'currency': match.group(2),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract DATE entities
        for match in self.date_pattern.finditer(text):
            entities['DATE'].append({
                'text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract METRIC entities
        for match in self.metric_pattern.finditer(text):
            entities['METRIC'].append({
                'text': match.group(0),
                'value': match.group(1),
                'unit': match.group(2),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract ORG and LOC using sentence-based approach
        sentences = self._split_sentences(text)
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            # Extract ORG
            for indicator in self.org_indicators:
                if indicator in sent_lower:
                    org_name = self._extract_entity_with_indicator(sent, indicator)
                    if org_name:
                        entities['ORG'].append({
                            'text': org_name,
                            'indicator': indicator,
                            'sentence': sent
                        })
            
            # Extract LOC
            for indicator in self.loc_indicators:
                if indicator in sent_lower:
                    loc_name = self._extract_location(sent, indicator)
                    if loc_name:
                        entities['LOC'].append({
                            'text': loc_name,
                            'indicator': indicator,
                            'sentence': sent
                        })
        
        # Extract PROJECT (quoted phrases)
        project_pattern = re.compile(r'проект\s+"([^"]+)"|"([А-ЏАБВГДЕЖЗИЈКЛМНОПРСТУФХЦЧШ][^"]{10,})"', re.IGNORECASE)
        for match in project_pattern.finditer(text):
            project_name = match.group(1) or match.group(2)
            if project_name:
                entities['PROJECT'].append({
                    'text': project_name,
                    'sentence': self._get_sentence_containing(text, match.start())
                })
        
        return entities
    
    def extract_triples_openie(self, text: str) -> List[Dict]:
        """Extract subject-predicate-object triples using Stanford OpenIE."""
        try:
            # Split into sentences for better processing
            sentences = self._split_sentences(text)
            all_triples = []
            
            for sent in sentences:
                # Skip very short sentences
                if len(sent.split()) < 4:
                    continue
                
                # Extract triples using OpenIE
                try:
                    triples = self.client.annotate(sent)
                    
                    for triple in triples:
                        all_triples.append({
                            'subject': triple.get('subject', ''),
                            'predicate': triple.get('relation', ''),
                            'object': triple.get('object', ''),
                            'sentence': sent,
                            'confidence': triple.get('confidence', 0.0)
                        })
                except Exception as e:
                    # Skip problematic sentences
                    continue
            
            return all_triples
            
        except Exception as e:
            print(f"  ⚠️ OpenIE extraction error: {e}")
            return []
    
    def merge_entities_with_triples(self, entities: Dict[str, List[Dict]], 
                                     triples: List[Dict]) -> List[Dict]:
        """Merge extracted entities with OpenIE triples to create enriched relationships."""
        relationships = []
        
        # Create a quick lookup for entities
        entity_texts = set()
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_texts.add(entity['text'].lower())
        
        # Process each triple
        for triple in triples:
            subject = triple['subject'].strip()
            predicate = triple['predicate'].strip()
            obj = triple['object'].strip()
            
            # Skip if any field is empty
            if not subject or not predicate or not obj:
                continue
            
            # Try to identify entity types for subject and object
            subject_type = self._identify_entity_type(subject, entities)
            object_type = self._identify_entity_type(obj, entities)
            
            # Only keep triples where at least one side is a recognized entity
            if subject_type or object_type:
                relationships.append({
                    'subject': subject,
                    'subject_type': subject_type or 'UNKNOWN',
                    'predicate': predicate,
                    'object': obj,
                    'object_type': object_type or 'UNKNOWN',
                    'sentence': triple['sentence'],
                    'confidence': triple.get('confidence', 0.0),
                    'source': 'openie'
                })
        
        return relationships
    
    def _identify_entity_type(self, text: str, entities: Dict[str, List[Dict]]) -> str:
        """Identify the entity type of a text span."""
        text_lower = text.lower()
        
        # Check against extracted entities
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity['text'].lower() in text_lower or text_lower in entity['text'].lower():
                    return entity_type
        
        # Pattern-based type identification
        if re.search(r'\d+.*(?:денари|евра|долари|милион|милијарда)', text_lower):
            return 'MONEY'
        if re.search(r'\d+.*(?:%|процент|км|метри)', text_lower):
            return 'METRIC'
        if re.search(r'\d{4}|јануари|февруари|март|април|мај', text_lower):
            return 'DATE'
        
        # Check for organization indicators
        for indicator in self.org_indicators:
            if indicator in text_lower:
                return 'ORG'
        
        # Check for location indicators
        for indicator in self.loc_indicators:
            if indicator in text_lower:
                return 'LOC'
        
        return ''
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _extract_entity_with_indicator(self, sentence: str, indicator: str) -> str:
        """Extract entity containing the indicator."""
        words = sentence.split()
        words_lower = [w.lower() for w in words]
        
        # Find indicator position
        indicator_idx = -1
        for i, w in enumerate(words_lower):
            if indicator in w:
                indicator_idx = i
                break
        
        if indicator_idx == -1:
            return ''
        
        # Collect capitalized words around indicator
        start_idx = max(0, indicator_idx - 2)
        end_idx = min(len(words), indicator_idx + 3)
        
        entity_words = []
        for i in range(start_idx, end_idx):
            word = words[i]
            if (word and word[0].isupper()) or words_lower[i] == indicator:
                entity_words.append(word)
            elif entity_words:
                break
        
        if entity_words:
            entity_text = ' '.join(entity_words)
            entity_text = re.sub(r'^[^\wА-Яа-я]+', '', entity_text)
            entity_text = re.sub(r'[^\wА-Яа-я\s-]+$', '', entity_text)
            words_list = entity_text.split()
            if len(words_list) > 6:
                entity_text = ' '.join(words_list[:6])
            return entity_text.strip()
        return ''
    
    def _extract_location(self, sentence: str, indicator: str) -> str:
        """Extract location name."""
        indicator_cap = indicator.capitalize()
        if indicator_cap in sentence:
            return indicator_cap
        
        words = sentence.split()
        for word in words:
            word_clean = re.sub(r'[,.:;!?]', '', word)
            if word_clean.lower() == indicator and word_clean and word_clean[0].isupper():
                return word_clean
        
        return ''
    
    def _get_sentence_containing(self, text: str, position: int) -> str:
        """Get sentence containing a position."""
        sentences = self._split_sentences(text)
        cumulative_pos = 0
        for sent in sentences:
            cumulative_pos += len(sent) + 2
            if cumulative_pos >= position:
                return sent
        return sentences[-1] if sentences else ''


class KnowledgeGraphBuilderOpenIE:
    """Build knowledge graph using OpenIE extraction."""
    
    def __init__(self, output_dir: str = "./rag_system/knowledge_graph_openie"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph = nx.MultiDiGraph()
        self.extractor = OpenIEEntityExtractor()
        self.entity_counts = defaultdict(int)
        self.relationship_count = 0
        
    def process_documents(self, data_dir: str, max_docs: int = None):
        """Process Macedonian documents and build KG."""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return
        
        txt_files = list(data_path.glob("*.txt"))
        if max_docs:
            txt_files = txt_files[:max_docs]
        
        print(f"📚 Found {len(txt_files)} documents to process")
        
        all_entities = []
        all_relationships = []
        
        for i, file_path in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}] Processing: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Skip very short documents
                if len(text) < 100:
                    print("  ⚠️ Skipping - too short")
                    continue
                
                # Extract entities
                print("  - Extracting entities...")
                entities = self.extractor.extract_entities(text)
                
                # Count entities
                for entity_type, entity_list in entities.items():
                    self.entity_counts[entity_type] += len(entity_list)
                    if len(entity_list) > 0:
                        print(f"    • {entity_type}: {len(entity_list)} found")
                
                # Extract triples using OpenIE
                print("  - Extracting relationships with OpenIE...")
                triples = self.extractor.extract_triples_openie(text)
                print(f"    • Raw triples: {len(triples)} found")
                
                # Merge entities with triples
                relationships = self.extractor.merge_entities_with_triples(entities, triples)
                print(f"    • Filtered relationships: {len(relationships)} found")
                
                # Add to graph
                self._add_to_graph(entities, relationships, file_path.name)
                
                all_entities.append({
                    'file': file_path.name,
                    'entities': entities
                })
                all_relationships.extend(relationships)
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results
        self._save_graph()
        self._save_json(all_entities, all_relationships)
        self._print_statistics()
        
    def _add_to_graph(self, entities: Dict[str, List[Dict]], 
                       relationships: List[Dict], source_file: str):
        """Add entities and relationships to graph."""
        
        # Add entity nodes
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                node_id = f"{entity_type}:{entity['text']}"
                self.graph.add_node(
                    node_id,
                    label=entity['text'],
                    type=entity_type,
                    source=source_file
                )
        
        # Add relationship edges
        for rel in relationships:
            subject_id = f"{rel['subject_type']}:{rel['subject']}"
            object_id = f"{rel['object_type']}:{rel['object']}"
            
            # Add nodes if they don't exist
            if subject_id not in self.graph:
                self.graph.add_node(subject_id, label=rel['subject'], type=rel['subject_type'])
            if object_id not in self.graph:
                self.graph.add_node(object_id, label=rel['object'], type=rel['object_type'])
            
            self.graph.add_edge(
                subject_id,
                object_id,
                relation=rel['predicate'],
                sentence=rel['sentence'][:200],  # Truncate long sentences
                confidence=rel.get('confidence', 0.0),
                source=source_file
            )
            self.relationship_count += 1
    
    def _save_graph(self):
        """Save graph in multiple formats."""
        graphml_path = self.output_dir / "knowledge_graph_openie.graphml"
        nx.write_graphml(self.graph, str(graphml_path))
        print(f"\n✅ Graph saved as GraphML: {graphml_path}")
        
        gml_path = self.output_dir / "knowledge_graph_openie.gml"
        nx.write_gml(self.graph, str(gml_path))
        print(f"✅ Graph saved as GML: {gml_path}")
        
        edgelist_path = self.output_dir / "knowledge_graph_openie_edges.txt"
        nx.write_edgelist(self.graph, str(edgelist_path), data=True)
        print(f"✅ Edge list saved: {edgelist_path}")
        
    def _save_json(self, entities: List[Dict], relationships: List[Dict]):
        """Save extracted data as JSON."""
        json_path = self.output_dir / "extracted_data_openie.json"
        data = {
            'extraction_date': datetime.now().isoformat(),
            'total_documents': len(entities),
            'entity_counts': dict(self.entity_counts),
            'total_relationships': len(relationships),
            'entities_by_document': entities,
            'all_relationships': relationships
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON data saved: {json_path}")
    
    def _print_statistics(self):
        """Print statistics."""
        print("\n" + "="*60)
        print("📊 KNOWLEDGE GRAPH STATISTICS (OpenIE)")
        print("="*60)
        print(f"Total Nodes: {self.graph.number_of_nodes()}")
        print(f"Total Edges: {self.graph.number_of_edges()}")
        print(f"Total Relationships Extracted: {self.relationship_count}")
        print("\nEntity Counts:")
        for entity_type, count in sorted(self.entity_counts.items()):
            print(f"  - {entity_type}: {count}")
        print("="*60)


def main():
    """Main execution."""
    print("🚀 Knowledge Graph Builder with Stanford OpenIE")
    print("="*60)
    
    # Initialize builder
    builder = KnowledgeGraphBuilderOpenIE()
    
    # Process documents (start with first 10 for testing)
    data_dir = "./rag_system/macedonian_data"
    print("\n⚠️  Starting with first 10 documents for testing...")
    print("    (OpenIE is slower - remove max_docs to process all)\n")
    
    builder.process_documents(data_dir, max_docs=10)
    
    print("\n✅ Knowledge Graph building complete!")
    print(f"📁 Output directory: {builder.output_dir}")
    print("\n💡 To process all documents, remove 'max_docs' parameter")


if __name__ == "__main__":
    main()
