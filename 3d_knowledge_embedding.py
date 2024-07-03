import spacy
import networkx as nx
import pytesseract
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import community
from scipy.spatial import ConvexHull
import plotly.figure_factory as ff

nlp = spacy.load("en_core_web_sm")

def extract_concepts_and_relations(text):
    doc = nlp(text)
    concepts = []
    relations = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ["NOUN", "PROPN"]:
                concepts.append(token.text)
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                head = token.head.text
                if head not in concepts:
                    concepts.append(head)
                relations.append((token.text, token.dep_, head))
    return list(set(concepts)), relations

def create_knowledge_graph(concepts, relations):
    G = nx.Graph()
    for concept in concepts:
        G.add_node(concept)
    for subject, relation, object in relations:
        if subject in G.nodes() and object in G.nodes():
            G.add_edge(subject, object, relation=relation)
    return G

def hyperbolic_layout(G):
    def h_dist(u, v):
        return np.arccosh(1 + 2 * (np.linalg.norm(u - v)**2) / ((1 - np.linalg.norm(u)**2) * (1 - np.linalg.norm(v)**2)))
    
    pos = nx.spring_layout(G, dim=3)
    for _ in range(50):  # Refine the layout
        for node in G.nodes():
            nbrs = list(G.neighbors(node))
            if not nbrs:
                continue
            center = np.mean([pos[nbr] for nbr in nbrs], axis=0)
            dist = h_dist(pos[node], center)
            if dist > 0:
                pos[node] = (pos[node] - center) * (np.tanh(dist) / dist) + center
    return pos

def visualize_geometric_knowledge(G, pos):
    # Perform community detection
    communities = community.best_partition(G)
    
    node_x, node_y, node_z = zip(*[pos[node] for node in G.nodes()])
    
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    # Create edges trace
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                              line=dict(width=1, color='#888'),
                              hoverinfo='none', mode='lines')

    # Create nodes trace
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z,
                              mode='markers+text',
                              text=list(G.nodes()),
                              hoverinfo='text',
                              marker=dict(size=5, 
                                          color=[communities[node] for node in G.nodes()],
                                          colorscale='Viridis',
                                          opacity=0.8))

    # Create convex hulls for each community
    hulls = []
    for community_id in set(communities.values()):
        community_nodes = [node for node in G.nodes() if communities[node] == community_id]
        if len(community_nodes) > 3:
            points = np.array([pos[node] for node in community_nodes])
            hull = ConvexHull(points)
            x, y, z = points[hull.simplices].T
            hull_trace = go.Mesh3d(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                                   opacity=0.2, color=f'rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})')
            hulls.append(hull_trace)

    # Create layout
    layout = go.Layout(
        title='Geometric Knowledge Representation',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        hovermode='closest'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace] + hulls, layout=layout)

    # Add a hyperbolic manifold representation
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    hyperbolic_surface = go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='Greys')
    fig.add_trace(hyperbolic_surface)

    fig.show()

def process_mindmap(image_path):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    print("Text extracted from image.")

    print("Extracting concepts and relations...")
    concepts, relations = extract_concepts_and_relations(text)

    print("Creating knowledge graph...")
    G = create_knowledge_graph(concepts, relations)

    if G.number_of_nodes() == 0:
        print("Warning: No concepts were extracted. The graph is empty.")
        return G, concepts, relations

    print("Generating hyperbolic layout...")
    pos = hyperbolic_layout(G)

    print("Visualizing geometric knowledge representation...")
    visualize_geometric_knowledge(G, pos)

    return G, concepts, relations

# Main execution
if __name__ == "__main__":
    image_path = '/Users/nesy/KG/Screenshot 2024-07-03 at 11.36.53.png'  # Update this path
    G, concepts, relations = process_mindmap(image_path)

    print("\nExtracted Concepts:")
    print(", ".join(concepts))

    print("\nExtracted Relations:")
    for relation in relations:
        print(f"{relation[0]} --{relation[1]}--> {relation[2]}")

    print("\nGraph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        central_concepts = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nMost central concepts:")
        for concept, centrality in central_concepts:
            print(f"{concept}: {centrality:.4f}")
    else:
        print("The graph is empty. No further analysis can be performed.")