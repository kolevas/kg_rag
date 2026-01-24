import json
import torch
import pdfplumber
import networkx as nx
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer


def load_models():
    """Load translation and REBEL models."""
    # Translation model
    trans_model_name = "Helsinki-NLP/opus-mt-mk-en"
    trans_tokenizer = MarianTokenizer.from_pretrained(trans_model_name)
    trans_model = MarianMTModel.from_pretrained(trans_model_name)
    
    # REBEL model for relation extraction
    rebel_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    rebel_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    
    # Use MPS (Mac GPU) if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    rebel_model = rebel_model.to(device)
    
    return trans_model, trans_tokenizer, rebel_model, rebel_tokenizer, device


# Load once at startup
TRANSLATION_MODEL, TRANSLATION_TOKENIZER, REBEL_MODEL, REBEL_TOKENIZER, DEVICE = load_models()


def extract_pdf_text(pdf_path):
    """Extract text from PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page_text := page.extract_text():
                text += page_text + " "
    return text.strip()


def translate_text(text):
    """Translate Macedonian to English using Opus-MT."""
    try:
        # Tokenize
        inputs = TRANSLATION_TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Translate
        translated = TRANSLATION_MODEL.generate(**inputs)
        
        # Decode
        english_text = TRANSLATION_TOKENIZER.decode(translated[0], skip_special_tokens=True)
        
        return english_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original if translation fails


def clean_entity(text):
    """Clean and validate entity text."""
    text = text.strip()
    
    # Remove if purely numeric
    if text.replace('.', '').replace(',', '').replace('/', '').isdigit():
        return None
    
    # Remove if too short
    if len(text) < 5:
        return None
    
    # Remove if contains too many numbers
    digit_count = sum(c.isdigit() for c in text)
    if digit_count / len(text) > 0.5:
        return None
    
    # Remove common noise patterns
    noise_patterns = ['chapter', 'page', 'figure', 'table', 'section']
    if any(pattern in text.lower() for pattern in noise_patterns):
        return None
    
    return text


def extract_triplets(text):
    """Parse REBEL output with special tokens."""
    triplets = []
    subject, relation, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({
                    'subject': subject.strip(),
                    'relation': relation.strip(),
                    'object': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({
                    'subject': subject.strip(),
                    'relation': relation.strip(),
                    'object': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
                
    if subject != '' and relation != '' and object_ != '':
        # Clean entities
        clean_subj = clean_entity(subject)
        clean_obj = clean_entity(object_)
        
        if clean_subj and clean_obj:
            triplets.append({
                'subject': clean_subj,
                'relation': relation.strip(),
                'object': clean_obj
            })
    return triplets


def extract_triples_llm(text):
    """Extract triples using Ollama LLM from English text with chunking."""
    import requests
    
    words = text.split()
    chunk_size = 400  # Larger chunks for better context
    overlap = 100
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    print(f"  Processing {len(chunks)} chunks...")
    all_triples = []
    
    for idx, chunk in enumerate(chunks):
        # Translate to English first
        english_text = translate_text(chunk)
        
        # Extract triples using LLM
        prompt = f"""Extract knowledge graph triples from this computer science text about courses and topics.

TEXT: {english_text}

Return ONLY a JSON array of triples. Each triple has subject, relation, and object.
Valid relations: teaches, prerequisite, part_of, related_to, uses, has_property, is_a

Example: [{{"subject":"Computer Architecture","relation":"teaches","object":"CPU design"}}]

JSON:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
                timeout=120
            )
            
            if response.status_code == 200:
                text_resp = response.json()["response"]
                # Extract JSON array
                start = text_resp.find('[')
                end = text_resp.rfind(']')
                
                if start != -1 and end != -1:
                    triples = json.loads(text_resp[start:end+1])
                    
                    # Filter with clean_entity
                    for triple in triples:
                        subj = clean_entity(str(triple.get("subject", "")))
                        obj = clean_entity(str(triple.get("object", "")))
                        
                        if subj and obj:
                            all_triples.append({
                                "subject": subj,
                                "relation": triple.get("relation", "related_to"),
                                "object": obj
                            })
        except Exception as e:
            print(f"  Error chunk {idx}: {e}")
            continue
    
    return all_triples


def build_graph(all_triples):
    """Build NetworkX graph from triples."""
    G = nx.MultiDiGraph()
    
    for triple in all_triples:
        # Handle both string and list values
        subj = triple.get("subject", "")
        obj = triple.get("object", "")
        
        # Convert to string if it's a list
        if isinstance(subj, list):
            subj = subj[0] if subj else ""
        if isinstance(obj, list):
            obj = obj[0] if obj else ""
        
        subj = str(subj).strip()
        obj = str(obj).strip()
        rel = triple.get("relation", "related_to")
        
        # Convert relation to string if it's a list
        if isinstance(rel, list):
            rel = rel[0] if rel else "related_to"
        rel = str(rel)
        
        if len(subj) >= 3 and len(obj) >= 3:
            # Add nodes
            if not G.has_node(subj):
                G.add_node(subj, freq=0)
            if not G.has_node(obj):
                G.add_node(obj, freq=0)
            
            # Increment frequency
            G.nodes[subj]["freq"] += 1
            G.nodes[obj]["freq"] += 1
            
            # Add edge
            G.add_edge(subj, obj, relation=rel)
    
    return G


def export_graph(G, output_file="output/kg_simple.json"):
    """Export graph to JSON."""
    Path(output_file).parent.mkdir(exist_ok=True)
    
    data = {
        "nodes": [{"id": n, "frequency": G.nodes[n]["freq"]} for n in G.nodes()],
        "edges": [{"source": u, "target": v, "relation": str(d["relation"])} 
                  for u, v, d in G.edges(data=True)]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Save GraphML with cleaned attributes
    G_clean = G.copy()
    for u, v, k, d in G_clean.edges(keys=True, data=True):
        # Ensure all edge attributes are strings
        G_clean[u][v][k]["relation"] = str(d["relation"])
    
    nx.write_graphml(G_clean, output_file.replace(".json", ".graphml"))
    print(f"Saved: {output_file}")


def main(pdf_dir="rag_system/macedonian_data/finki_courses", max_files=5):
    """Main processing function."""
    print("Simple LLM Knowledge Graph Builder")
    print("=" * 50)
    
    # Get PDFs
    pdfs = list(Path(pdf_dir).glob("**/*.pdf"))[:max_files]
    print(f"Processing {len(pdfs)} PDFs...\n")
    
    all_triples = []
    
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        
        # Extract text
        text = extract_pdf_text(pdf)
        if not text:
            continue
        
        # Get triples from LLM
        triples = extract_triples_llm(text)
        print(f"  → {len(triples)} triples\n")
        all_triples.extend(triples)
    
    # Build and export graph
    print(f"\nTotal triples: {len(all_triples)}")
    G = build_graph(all_triples)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    export_graph(G)
    
    # Show top entities
    top = sorted(G.nodes(data=True), key=lambda x: x[1]["freq"], reverse=True)[:10]
    print(f"\nTop entities:")
    for node, data in top:
        print(f"  {node}: {data['freq']}")


if __name__ == "__main__":
    main(max_files=2)
