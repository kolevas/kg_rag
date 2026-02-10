"""
Unified Knowledge Graph Visualization Tool

Supports visualizing different knowledge graphs:
  - FLORES-200 Macedonian dataset
  - Groq extraction
  - Wikipedia
  - Any custom .graphml file

Usage:
    python visualize_kg.py                          # Default: FLORES graph
    python visualize_kg.py --graph flores           # FLORES graph
    python visualize_kg.py --graph wikipedia        # Wikipedia graph
    python visualize_kg.py --graph unified          # Unified graph
    python visualize_kg.py --file path/to/kg.graphml --output custom_viz.html
"""
import argparse
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from collections import Counter

# Default color palette for relation types
DEFAULT_RELATION_COLORS = {
    'has_property': '#ff6b6b',
    'is_a': '#4ecdc4',
    'located_in': '#45b7d1',
    'part_of': '#96ceb4',
    'announced': '#ffeaa7',
    'works_at': '#dfe6e9',
    'има': '#fd79a8',
    'е': '#fdcb6e',
    'has': '#74b9ff',
    'is_part_of': '#a29bfe',
    'се_однесува_на': '#ff7675',
    'has_part': '#00b894',
    'е_дел_од': '#00cec9',
    'relation': '#6c5ce7',
    'participated_in': '#fab1a0',
    'contains': '#4ecdc4',
    'has_component': '#45b7d1',
    'uses': '#96ceb4',
    'belongs_to': '#dfe6e9',
    'requires': '#fd79a8',
    'teaches': '#fdcb6e',
    'содржи': '#74b9ff',
    'дефинира': '#ff7675',
    'не_е': '#fd79a8',
    'defines': '#00b894',
    'implements': '#00cec9',
    'implies': '#6c5ce7',
}

# Predefined graph configurations
GRAPH_PRESETS = {
    'flores': {
        'file': 'rag_system/knowledge_graph/kg_flores.graphml',
        'output': 'kg_flores_visualization.html',
        'title': 'Knowledge Graph (FLORES-200)',
        'description': 'Macedonian news articles',
    },
    'wikipedia': {
        'file': 'rag_system/knowledge_graph/kg_wikipedia.graphml',
        'output': 'kg_wikipedia_visualization.html',
        'title': 'Knowledge Graph (Wikipedia)',
        'description': 'Macedonian Wikipedia articles',
    },
    'unified': {
        'file': 'rag_system/knowledge_graph/kg_unified.graphml',
        'output': 'kg_unified_visualization.html',
        'title': 'Knowledge Graph (Unified)',
        'description': 'Combined knowledge graph',
    },
}


def visualize_knowledge_graph(
    graph_file: str,
    output_file: str,
    title: str = "Knowledge Graph",
    description: str = "",
    relation_colors: dict = None,
):
    """
    Visualize a knowledge graph from a .graphml file.

    Args:
        graph_file: Path to the .graphml file
        output_file: Path for the output HTML file
        title: Title shown in the legend
        description: Description shown in the legend
        relation_colors: Dict mapping relation names to hex colors
    """
    if relation_colors is None:
        relation_colors = DEFAULT_RELATION_COLORS

    # Load the graph
    graph_path = Path(graph_file)
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_file}")
        return

    G = nx.read_graphml(graph_path)
    print(f"📊 Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create Pyvis network
    net = Network(height="900px", width="100%", bgcolor="#1a1a1a", font_color="white")

    # Count relations
    all_relations = []
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', 'unknown')
        all_relations.append(rel)
    relation_counts = Counter(all_relations)

    # Add nodes
    for node in G.nodes():
        net.add_node(
            node,
            label=str(node)[:50],
            title=str(node),
            color='#95a5a6',
            size=20,
        )

    # Add edges with colors
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'unknown')
        color = relation_colors.get(relation, '#636e72')
        net.add_edge(
            u, v,
            title=relation,
            color=color,
            arrows='to',
            width=2,
        )

    # Physics options
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)

    # Save the graph
    net.save_graph(output_file)

    # Add custom legend
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    desc_line = f'<p><strong>Dataset:</strong> {description}</p>' if description else ''
    legend_html = f"""
    <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; color: white; font-family: Arial; max-height: 85vh; overflow-y: auto;">
        <h3 style="margin-top: 0;">📊 {title}</h3>
        {desc_line}
        <p><strong>Nodes:</strong> {G.number_of_nodes()}</p>
        <p><strong>Edges:</strong> {G.number_of_edges()}</p>
        <p><strong>Relations:</strong> {len(relation_counts)}</p>
        <hr style="border-color: #555;">
        <h4>Top Relations:</h4>
    """

    for rel, count in relation_counts.most_common(15):
        color = relation_colors.get(rel, '#636e72')
        legend_html += f'<div style="margin: 5px 0;"><span style="display: inline-block; width: 12px; height: 12px; background: {color}; margin-right: 8px; border-radius: 2px;"></span>{rel}: {count}</div>'

    legend_html += "</div>"
    html_content = html_content.replace('<body>', f'<body>{legend_html}')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Visualization saved: {output_file}")
    print(f"\n📊 Top 10 Relations:")
    for rel, count in relation_counts.most_common(10):
        print(f"  {rel}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a Knowledge Graph")
    parser.add_argument(
        '--graph', 
        choices=list(GRAPH_PRESETS.keys()),
        default='flores',
        help="Predefined graph to visualize (default: flores)"
    )
    parser.add_argument(
        '--file',
        type=str,
        help="Path to a custom .graphml file (overrides --graph)"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Output HTML filename (default: based on graph name)"
    )
    parser.add_argument(
        '--title',
        type=str,
        help="Title for the visualization legend"
    )

    args = parser.parse_args()

    if args.file:
        # Custom file mode
        graph_file = args.file
        output_file = args.output or Path(args.file).stem + '_visualization.html'
        title = args.title or "Knowledge Graph"
        description = ""
    else:
        # Preset mode
        preset = GRAPH_PRESETS[args.graph]
        graph_file = preset['file']
        output_file = args.output or preset['output']
        title = args.title or preset['title']
        description = preset['description']

    visualize_knowledge_graph(
        graph_file=graph_file,
        output_file=output_file,
        title=title,
        description=description,
    )


if __name__ == '__main__':
    main()
