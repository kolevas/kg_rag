"""
Visualize Knowledge Graph from Groq extraction
"""
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from collections import Counter

# Load the graph
graph_file = Path("rag_system/knowledge_graph_groq/kg_groq.graphml")
G = nx.read_graphml(graph_file)

print(f"📊 Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Create Pyvis network
net = Network(height="900px", width="100%", bgcolor="#1a1a1a", font_color="white")

# Relation colors
relation_colors = {
    'is_a': '#ff6b6b',
    'contains': '#4ecdc4',
    'has_component': '#45b7d1',
    'uses': '#96ceb4',
    'е': '#ffeaa7',
    'belongs_to': '#dfe6e9',
    'requires': '#fd79a8',
    'teaches': '#fdcb6e',
    'содржи': '#74b9ff',
    'has_property': '#a29bfe',
    'дефинира': '#ff7675',
    'не_е': '#fd79a8',
    'defines': '#00b894',
    'implements': '#00cec9',
    'implies': '#6c5ce7'
}

# Get all relations for counting
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
        size=20
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
        width=2
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

# Save
output_file = "kg_groq_visualization.html"
net.save_graph(output_file)

# Add custom legend
with open(output_file, 'r', encoding='utf-8') as f:
    html_content = f.read()

legend_html = f"""
<div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; color: white; font-family: Arial; max-height: 85vh; overflow-y: auto;">
    <h3 style="margin-top: 0;">📊 Knowledge Graph (Groq)</h3>
    <p><strong>Nodes:</strong> {G.number_of_nodes()}</p>
    <p><strong>Edges:</strong> {G.number_of_edges()}</p>
    <p><strong>Relations:</strong> {len(relation_counts)}</p>
    <hr style="border-color: #555;">
    <h4>Top Relations:</h4>
"""

for rel, count in relation_counts.most_common(15):
    color = relation_colors.get(rel, '#636e72')
    legend_html += f'<div style="margin: 5px 0;"><span style="display: inline-block; width: 12px; height: 12px; background: {color}; margin-right: 8px; border-radius: 2px;"></span>{rel}: {count}</div>'

legend_html += """
</div>
"""

html_content = html_content.replace('<body>', f'<body>{legend_html}')

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Visualization saved: {output_file}")
print(f"\n📊 Top 10 Relations:")
for rel, count in relation_counts.most_common(10):
    print(f"  {rel}: {count}")
