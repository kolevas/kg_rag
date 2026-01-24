"""
Knowledge Graph Builder using Groq
Strategy: Direct Macedonian Extraction → Build Graph (no translation needed)
"""
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
from datetime import datetime
from collections import defaultdict, Counter
from groq import Groq
from dotenv import load_dotenv
import pdfplumber

load_dotenv()


class GroqTripleExtractor:
    """Use Groq (Llama 3) to extract triples from Macedonian text."""
    
    def __init__(self):
        """Initialize Groq client."""
        print(f"🔄 Initializing Groq (Llama 3)...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        print(f"✅ Groq initialized: {self.model}")
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract knowledge triples from Macedonian text using Groq."""
        if not text or len(text.strip()) < 20:
            return []
        
        prompt = f"""You are an expert in extracting knowledge from educational content about computer science, architecture, and engineering. The text is in Macedonian.

Extract knowledge triples (subject, relation, object) from the following MACEDONIAN text. Focus on:
- Educational concepts and their relationships
- Technical entities (courses, topics, technologies, components)
- Teaching relationships
- Hierarchical structures
- Definitional relationships

Macedonian Text: {text}

Provide triples in the format:
subject | relation | object

Relations should be simple and standardized (e.g., "teaches", "is_a", "contains", "uses", "belongs_to", "has_component", "requires", "implements").
Keep entity names in their original form (Macedonian or English technical terms).
Extract as many relevant triples as possible (aim for 5-15 per chunk).

Triples:"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=800,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            return self._parse_triples(result)
        
        except Exception as e:
            print(f"    ⚠️  Groq extraction error: {e}")
            return []
    
    def _parse_triples(self, llm_response: str) -> List[Tuple[str, str, str]]:
        """Parse triples from LLM response."""
        triples = []
        
        for line in llm_response.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to parse: subject | relation | object
            match = re.match(r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)$', line)
            if match:
                subject, relation, obj = match.groups()
                subject = subject.strip().strip('\"\'')
                relation = relation.strip().strip('\"\'').lower().replace(' ', '_')
                obj = obj.strip().strip('\"\'')
                
                if subject and relation and obj:
                    triples.append((subject, relation, obj))
        
        return triples


class KnowledgeGraphBuilderGroq:
    """Build knowledge graph using Groq extraction."""
    
    def __init__(self):
        """Initialize components."""
        self.graph = nx.MultiDiGraph()
        self.groq_extractor = GroqTripleExtractor()
        self.all_triples = []
    
    def _add_triple(self, subject: str, relation: str, obj: str):
        """Add a triple to the graph."""
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(obj, type="entity")
        self.graph.add_edge(subject, obj, relation=relation)
        self.all_triples.append((subject, relation, obj))
    
    def process_documents(self, data_dir: str, max_docs: int = 15, max_chunks: int = 3, target_file: str = None):
        """Process FINKI course PDFs."""
        data_path = Path(data_dir)
        finki_courses_path = data_path / "finki_courses"
        
        # Find all PDFs
        pdf_files = []
        for course_dir in finki_courses_path.iterdir():
            if course_dir.is_dir():
                pdf_files.extend(list(course_dir.glob("*.pdf")))
        
        # Filter to target file if specified
        if target_file:
            pdf_files = [f for f in pdf_files if f.name == target_file]
            if pdf_files:
                print(f"✅ Found target file: {pdf_files[0]}")
            else:
                print(f"❌ Target file not found: {target_file}")
                return
        
        pdf_files = sorted(pdf_files)[:max_docs]
        
        print(f"📚 Found {len(pdf_files)} FINKI course PDF files")
        print(f"📖 Processing courses from: {finki_courses_path}")
        print(f"🎯 Processing first {max_docs} documents\n")
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            course_name = pdf_path.parent.name
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
            print(f"  📘 Course: {course_name}...")
            
            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                print("  ⚠️  No text extracted")
                continue
            
            # Split into chunks
            chunks = self._split_text(text, chunk_size=1000)
            if not chunks:
                print("  ⚠️  No valid chunks")
                continue
            
            # Process each chunk
            for i, chunk in enumerate(chunks[:max_chunks], 1):
                print(f"\n  Chunk {i}: {chunk[:60]}...")
                
                # Extract triples directly from Macedonian text
                triples = self.groq_extractor.extract_triples(chunk)
                print(f"    → Extracted {len(triples)} triples")
                
                if triples:
                    print(f"    → Sample: {triples[0]}")
                
                # Add to graph
                for subject, relation, obj in triples:
                    self._add_triple(subject, relation, obj)
        
        # Save outputs
        self._save_graph()
        self._print_statistics()
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages[:10]:  # First 10 pages
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n".join(text_parts)
        except Exception as e:
            print(f"    ⚠️  PDF extraction error: {e}")
            return ""
    
    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentences/paragraphs
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _save_graph(self):
        """Save graph and triples."""
        output_dir = Path("rag_system/knowledge_graph_groq")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save GraphML
        graph_file = output_dir / "kg_groq.graphml"
        nx.write_graphml(self.graph, graph_file)
        print(f"\n✅ Graph saved: {graph_file}")
        
        # Save triples JSON
        triples_file = output_dir / "triples_groq.json"
        with open(triples_file, 'w', encoding='utf-8') as f:
            json.dump([
                {"subject": s, "relation": r, "object": o}
                for s, r, o in self.all_triples
            ], f, indent=2, ensure_ascii=False)
        print(f"✅ Triples saved: {triples_file}")
    
    def _print_statistics(self):
        """Print graph statistics."""
        print("\n" + "="*60)
        print("📊 KNOWLEDGE GRAPH STATISTICS (Groq Llama 3)")
        print("="*60)
        print(f"Total Nodes: {self.graph.number_of_nodes()}")
        print(f"Total Edges: {self.graph.number_of_edges()}")
        print(f"Total Triples: {len(self.all_triples)}")
        
        # Relation distribution
        relations = Counter([r for _, r, _ in self.all_triples])
        print(f"Unique Relations: {len(relations)}")
        print(f"\nTop 15 Relations:")
        for rel, count in relations.most_common(15):
            print(f"  {rel}: {count}")
        print("="*60)


def main():
    """Main execution."""
    print("🚀 Knowledge Graph Builder with Groq (Llama 3)")
    print("="*60)
    print("Strategy: Direct Macedonian Extraction → Build Graph\n")
    
    builder = KnowledgeGraphBuilderGroq()
    builder.process_documents("./rag_system/macedonian_data", max_docs=15, max_chunks=3)
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()
