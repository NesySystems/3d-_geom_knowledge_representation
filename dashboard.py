import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import importlib

knowledge_module = importlib.import_module("3d_knowledge_embedding")
app = knowledge_module.app

if __name__ == "__main__":
    app.run_server(debug=True)


# Import your custom module
knowledge_module = importlib.import_module("3d_knowledge_embedding")
process_mindmap = knowledge_module.process_mindmap
visualize_geometric_knowledge = knowledge_module.visualize_geometric_knowledge

# Rest of your dashboard code...

# Load your mind map and process it
image_path = '/Users/nesy/KG/Screenshot 2024-07-03 at 11.36.53.png'  # Update this path
G, concepts, relations = process_mindmap(image_path)

# Create the layout
app.layout = html.Div([
    html.H1("Advanced Geometric Knowledge Representation Dashboard"),
    
    html.Div([
        dcc.Graph(id='knowledge-graph-3d', style={'height': '80vh', 'width': '70%'}),
        html.Div([
            html.H3("Node Coordinates"),
            html.Div(id='node-coordinates'),
            html.H3("Node Vectors"),
            html.Div(id='node-vectors')
        ], style={'width': '30%', 'float': 'right'})
    ], style={'display': 'flex'}),
    
    # ... (rest of the layout remains the same)
])

@app.callback(
    Output('knowledge-graph-3d', 'figure'),
    Output('central-concepts-list', 'children'),
    Output('community-info', 'children'),
    Output('node-coordinates', 'children'),
    Output('node-vectors', 'children'),
    Input('community-threshold', 'value')
)
def update_graph(threshold):
    pos = nx.spring_layout(G, dim=3)
    fig = visualize_geometric_knowledge(G, pos, threshold)
    
    # ... (rest of the function remains the same)
    
    # Prepare node coordinates and vectors
    coordinates = []
    vectors = []
    for node, (x, y, z) in pos.items():
        coordinates.append(html.P(f"{node}: ({x:.2f}, {y:.2f}, {z:.2f})"))
        vector = np.array([x, y, z])
        vectors.append(html.P(f"{node}: {vector}"))
    
    return fig, central_concepts_list, community_info, coordinates[:10], vectors[:10]  # Limit to first 10 for brevity

if __name__ == '__main__':
    app.run_server(debug=True)