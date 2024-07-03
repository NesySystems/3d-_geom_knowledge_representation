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
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import os

nlp = spacy.load("en_core_web_sm")

def spherical_projection(pos):
    coords = np.array(list(pos.values()))
    r = np.linalg.norm(coords, axis=1)
    theta = np.arccos(coords[:, 2] / r)
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    return {node: np.array([r[i], theta[i], phi[i]]) for i, node in enumerate(pos.keys())}

def toroidal_projection(pos):
    coords = np.array(list(pos.values()))
    R = 1
    r = 0.3
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    phi = np.arctan2(coords[:, 2], np.sqrt(coords[:, 0]**2 + coords[:, 1]**2) - R)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return {node: np.array([x[i], y[i], z[i]]) for i, node in enumerate(pos.keys())}

def hyperbolic_projection(pos):
    coords = np.array(list(pos.values()))
    norm = np.linalg.norm(coords, axis=1, keepdims=True)
    return {node: coords[i] / (1 + np.sqrt(1 + norm[i]**2)) for i, node in enumerate(pos.keys())}

def visualize_geometric_knowledge(G, pos, threshold=0.5, manifold_type='euclidean'):
    if manifold_type == 'spherical':
        pos = spherical_projection(pos)
    elif manifold_type == 'toroidal':
        pos = toroidal_projection(pos)
    elif manifold_type == 'hyperbolic':
        pos = hyperbolic_projection(pos)

    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                              line=dict(width=1, color='#888'),
                              hoverinfo='none', mode='lines')

    node_x, node_y, node_z = zip(*[pos[node] for node in G.nodes()])
    communities = community.best_partition(G, resolution=threshold)
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z,
                              mode='markers',
                              text=list(G.nodes()),
                              hoverinfo='text',
                              marker=dict(size=5, 
                                          color=[communities[node] for node in G.nodes()],
                                          colorscale='Viridis',
                                          opacity=0.8))

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

    fig = go.Figure(data=[edge_trace, node_trace] + hulls, layout=layout)

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    hyperbolic_surface = go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='Greys')
    fig.add_trace(hyperbolic_surface)

    fig.update_layout(clickmode='event+select')

    return fig

def create_atlas(pos, num_charts=5, overlap=0.2):
    all_coords = np.array(list(pos.values()))
    min_coords = np.min(all_coords, axis=0)
    max_coords = np.max(all_coords, axis=0)
    chart_size = (max_coords - min_coords) / (num_charts - overlap)
    atlas = []
    for i in range(num_charts):
        for j in range(num_charts):
            for k in range(num_charts):
                chart_min = min_coords + np.array([i, j, k]) * chart_size * (1 - overlap)
                chart_max = chart_min + chart_size
                chart_nodes = [node for node, coords in pos.items() 
                               if np.all(coords >= chart_min) and np.all(coords <= chart_max)]
                if chart_nodes:
                    atlas.append({
                        'bounds': (chart_min, chart_max),
                        'nodes': chart_nodes
                    })
    return atlas

def visualize_atlas(fig, atlas):
    for chart in atlas:
        min_bound, max_bound = chart['bounds']
        x, y, z = zip(min_bound, max_bound)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, opacity=0.2, color='red'))
    return fig

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
    for _ in range(50):
        for node in G.nodes():
            nbrs = list(G.neighbors(node))
            if not nbrs:
                continue
            center = np.mean([pos[nbr] for nbr in nbrs], axis=0)
            dist = h_dist(pos[node], center)
            if dist > 0:
                pos[node] = (pos[node] - center) * (np.tanh(dist) / dist) + center
    return pos

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
        return G, concepts, relations, {}

    print("Generating hyperbolic layout...")
    pos = hyperbolic_layout(G)

    return G, concepts, relations, pos

def visualize_different_geometries(G, pos, manifold_type='euclidean'):
    fig = visualize_geometric_knowledge(G, pos, manifold_type=manifold_type)
    return fig

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='3d-scatter-plot'),
        dcc.Dropdown(
            id='manifold-type',
            options=[
                {'label': 'Euclidean', 'value': 'euclidean'},
                {'label': 'Spherical', 'value': 'spherical'},
                {'label': 'Toroidal', 'value': 'toroidal'},
                {'label': 'Hyperbolic', 'value': 'hyperbolic'}
            ],
            value='euclidean',
            style={'width': '48%', 'display': 'inline-block'}
        ),
        dcc.Input(
            id='image-path-input',
            type='text',
            value='/Users/nesy/KG/Screenshot 2024-07-03 at 11.36.53.png',
            style={'width': '48%', 'padding': '10px', 'display': 'inline-block', 'marginTop': '20px'}  # Changed to camelCase
        )
    ]),
    html.Div(id='node-info', style={'whiteSpace': 'pre-line', 'marginTop': '20px'})  # Changed to camelCase
])

@app.callback(
    Output('3d-scatter-plot', 'figure'),
    Output('node-info', 'children'),
    Input('manifold-type', 'value'),
    Input('image-path-input', 'value'),
    Input('3d-scatter-plot', 'clickData')
)
def update_graph(manifold_type, image_path, clickData):
    if not os.path.exists(image_path):
        return go.Figure(data=[], layout=go.Layout(title='Invalid image path')), "Invalid image path"

    G, concepts, relations, pos = process_mindmap(image_path)
    fig = visualize_different_geometries(G, pos, manifold_type=manifold_type)
    
    centrality = nx.degree_centrality(G)
    communities = community.best_partition(G)

    node_info = "Click on a node to see details"
    
    if clickData:
        print("clickData:", clickData)  # Debug information
        if 'points' in clickData and clickData['points']:
            point_data = clickData['points'][0]
            if 'text' in point_data:
                node = point_data['text']
                if node in G.nodes():
                    neighbors = list(G.neighbors(node))
                    node_info = (
                        f"Node: {node}\n"
                        f"Neighbors: {', '.join(neighbors)}\n"
                        f"Centrality: {centrality[node]:.4f}\n"
                        f"Community: {communities[node]}"
                    )
                else:
                    node_info = "Node not found in the graph."
            else:
                node_info = "No 'text' key in clickData['points'][0]"
        else:
            node_info = "No points found in clickData"
    else:
        node_info = "Click data is not valid or no point was clicked."

    return fig, node_info

if __name__ == '__main__':
    app.run_server(debug=True)
