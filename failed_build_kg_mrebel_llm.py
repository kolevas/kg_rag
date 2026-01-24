"""
Hybrid Knowledge Graph Builder: mREBEL + LLM Post-processing
Uses mREBEL for fast triple generation, then LLM to clean and validate.
"""

import json
import os
from pathlib import Path
import pdfplumber
import networkx as nx
import re
from typing import Dict, List
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0400-\u04FF.,;:?!()\-]', '', text)
    return text.strip()


def load_mrebel_model():
    """Load mREBEL model with proper Macedonian support."""
    print("Loading mREBEL model...")
    
    # Detect device (MPS for Mac GPU, CUDA for Nvidia, CPU fallback)
    if torch.backends.mps.is_available():
        device = "mps"
        device_name = "MPS (Mac GPU)"
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = "CUDA (GPU)"
    else:
        device = "cpu"
        device_name = "CPU"
    
    print(f"Using device: {device_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('Babelscape/mrebel-large')
        model = AutoModelForSeq2SeqLM.from_pretrained('Babelscape/mrebel-large')
        
        if device != "cpu":
            model = model.to(device)
            if device == "mps" or device == "cuda":
                model = model.half()  # Use float16 for speed
        
        print("✓ mREBEL model loaded")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading mREBEL: {e}")
        return None, None, None


def extract_triples_mrebel(text: str, model, tokenizer, device, max_length: int = 512) -> str:
    """Extract raw triples using mREBEL with Macedonian language support."""
    
    # Set source language to Macedonian
    tokenizer.src_lang = "mk_MK"
    
    # Split into chunks
    words = text.split()
    chunk_size = 200  # words per chunk
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    all_outputs = []
    for idx, chunk in enumerate(chunks[:10]):  # Process first 10 chunks
        try:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
            
            # Move inputs to device
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with proper Macedonian settings and repetition prevention
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_start_token_id=tokenizer.convert_tokens_to_ids("<triplet>"),
                max_length=256,
                num_beams=5,
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                repetition_penalty=2.0,   # Strongly penalize repetition
                early_stopping=True,
                forced_bos_token_id=None
            )
            
            decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0]
            
            if decoded_text and len(decoded_text) > 10:
                all_outputs.append(decoded_text)
                print(f"  Chunk {idx+1}: {len(decoded_text)} chars")
        except Exception as e:
            print(f"  mREBEL error on chunk {idx+1}: {e}")
            continue
    
    combined_output = "\n".join(all_outputs)
    print(f"\n  === RAW mREBEL OUTPUT ({len(combined_output)} chars) ===")
    print(combined_output)  # Show full output
    print("  ...\n")
    return combined_output


def call_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    """Call Ollama API locally."""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2000
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Ollama API error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""


def extract_triples_with_llm(mrebel_output: str) -> List[Dict]:
    """Use LLM to extract and parse triples from raw mREBEL output."""
    
    prompt = f"""Parse mREBEL output to JSON. mREBEL format: <triplet> subject <subj> relation <obj> object <triplet>

INPUT:
{mrebel_output}

OUTPUT FORMAT (JSON only, no code, no text):
[{{"subject":"Дискретна математика","relation":"teaches","object":"Граф"}}]

RULES:
- Relations: teaches, prerequisite, part_of, related_to, example_of, uses, is_a, affects, has_property
- Skip: ако, нека, решение, задача, како, дали, пример, според, тогаш, значи, колку, сите, вкупно
- Min 3 chars

JSON:"""

    response = call_ollama(prompt)
    
    if response:
        print(f"\n  === LLM RESPONSE ({len(response)} chars) ===")
        print(response[:500])  # Show first 500 chars
        print("  ...\n")
        
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```(?:json|python)?\s*', '', response)
            response = re.sub(r'```', '', response)
            
            # Find JSON array
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                triples = json.loads(json_match.group())
                print(f"  ✓ LLM extracted: {len(triples)} triples")
                if triples:
                    print(f"  Sample: {triples[0]}")
                return triples
            else:
                print(f"  ✗ No JSON array found")
                print(f"  Response: {response[:300]}")
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parsing error: {e}")
            print(f"  Response: {response[:300]}")
    
    return []

    response = call_ollama(prompt)
    
    if response:
        try:
            # Find JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                triples = json.loads(json_match.group())
                print(f"  LLM cleaned: {len(triples)} valid triples")
                return triples
            else:
                print(f"  No JSON found in LLM response")
                print(f"  Response preview: {response[:200]}")
        except json.JSONDecodeError as e:
            print(f"  JSON parsing error: {e}")
            print(f"  Response preview: {response[:200]}")
    
    return []


def derive_entities_from_triples(triples: List[Dict]) -> List[Dict]:
    """Extract unique entities from triples."""
    entities = {}
    
    for triple in triples:
        subject = triple.get("subject", "").strip()
        object_val = triple.get("object", "").strip()
        
        # Add subject
        if subject and len(subject) >= 3:
            if subject not in entities:
                entities[subject] = {"name": subject, "type": "CONCEPT", "frequency": 1}
            else:
                entities[subject]["frequency"] += 1
        
        # Add object
        if object_val and len(object_val) >= 3:
            if object_val not in entities:
                entities[object_val] = {"name": object_val, "type": "CONCEPT", "frequency": 1}
            else:
                entities[object_val]["frequency"] += 1
    
    return list(entities.values())


def build_graph(entities: List[Dict], relationships: List[Dict]) -> nx.MultiDiGraph:
    """Build NetworkX graph from entities and relationships."""
    G = nx.MultiDiGraph()
    
    # Add nodes
    for entity in entities:
        G.add_node(
            entity["name"],
            label=entity["name"],
            type=entity["type"],
            frequency=entity.get("frequency", 1)
        )
    
    # Add edges
    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        relation = rel.get("relation", "related_to")
        
        if source and target and G.has_node(source) and G.has_node(target):
            G.add_edge(source, target, relation=relation)
    
    return G


def export_graph(G: nx.MultiDiGraph, output_dir: str):
    """Export graph to JSON and GraphML formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to JSON
    graph_data = {
        "nodes": [
            {
                "id": node,
                "label": G.nodes[node].get("label", node),
                "type": G.nodes[node].get("type", "CONCEPT"),
                "frequency": G.nodes[node].get("frequency", 1)
            }
            for node in G.nodes()
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "relation": data.get("relation", "related_to")
            }
            for u, v, data in G.edges(data=True)
        ]
    }
    
    json_path = os.path.join(output_dir, "kg_mrebel_llm.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved JSON to: {json_path}")
    
    # Export to GraphML
    graphml_path = os.path.join(output_dir, "kg_mrebel_llm.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"Saved GraphML to: {graphml_path}")


def process_pdfs(input_dir: str, output_dir: str, max_files: int = 5):
    """Process PDFs using mREBEL + LLM hybrid approach."""
    
    print("=" * 60)
    print("HYBRID KNOWLEDGE GRAPH BUILDER (mREBEL + LLM)")
    print("=" * 60)
    
    # Load mREBEL model
    model, tokenizer, device = load_mrebel_model()
    if not model or not tokenizer:
        print("Cannot load mREBEL model. Exiting.")
        return
    
    # Get all PDF files recursively
    pdf_files = list(Path(input_dir).glob("**/*.pdf"))
    pdf_files = pdf_files[:max_files]
    
    print(f"\nFound {len(pdf_files)} PDF files to process")
    
    all_entities = {}
    all_relationships = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        # Extract text
        text = extract_text_from_pdf(str(pdf_path))
        if not text:
            print("  Skipping: No text extracted")
            continue
        
        text = clean_text(text)
        print(f"  Extracted {len(text)} characters")
        
        # Step 1: Generate raw triples with mREBEL
        print("  Generating triples with mREBEL...")
        mrebel_output = extract_triples_mrebel(text, model, tokenizer, device)
        print(f"  mREBEL output: {len(mrebel_output)} characters")
        
        if not mrebel_output:
            print("  No mREBEL output, skipping")
            continue
        
        # Step 2: Extract triples using LLM
        print("  Extracting triples with LLM...")
        triples = extract_triples_with_llm(mrebel_output)
        print(f"  Final: {len(triples)} clean triples")
        
        # Derive entities from triples
        entities = derive_entities_from_triples(triples)
        
        # Merge entities
        for entity in entities:
            name = entity["name"]
            if name in all_entities:
                all_entities[name]["frequency"] += entity.get("frequency", 1)
            else:
                all_entities[name] = entity
        
        # Store relationships
        for triple in triples:
            all_relationships.append({
                "source": triple.get("subject"),
                "target": triple.get("object"),
                "relation": triple.get("relation", "related_to")
            })
    
    # Build graph
    print(f"\n{'=' * 60}")
    print("BUILDING KNOWLEDGE GRAPH")
    print(f"{'=' * 60}")
    
    entities_list = list(all_entities.values())
    print(f"Total unique entities: {len(entities_list)}")
    print(f"Total relationships: {len(all_relationships)}")
    
    G = build_graph(entities_list, all_relationships)
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Export graph
    export_graph(G, output_dir)
    
    # Show top entities
    print(f"\nTop 10 Entities by Frequency:")
    sorted_entities = sorted(entities_list, key=lambda x: x.get("frequency", 1), reverse=True)
    for entity in sorted_entities[:10]:
        print(f"  {entity['name']}: {entity.get('frequency', 1)}")
    
    print(f"\n{'=' * 60}")
    print("✅ COMPLETE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print("\nHYBRID APPROACH: mREBEL (fast) + LLM (quality)")
    print("=" * 60)
    
    input_dir = "rag_system/macedonian_data/finki_courses"
    output_dir = "output"
    
    process_pdfs(input_dir, output_dir, max_files=3)
