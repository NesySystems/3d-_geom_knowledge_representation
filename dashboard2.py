import importlib
import os
import sys
import networkx as nx
from dash import html, dcc, Input, Output
import dash

# Ensure the custom module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module
knowledge_module = importlib.import_module("3d_knowledge_embedding_with_intelligence")

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
            style={'width': '48%', 'padding': '10px', 'display': 'inline-block', 'marginTop': '20px'}
        )
    ]),
    html.Div(id='node-info', style={'whiteSpace': 'pre-line', 'marginTop': '20px'})
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

    G, concepts, relations, pos = knowledge_module.process_mindmap(image_path)
    fig = knowledge_module.visualize_different_geometries(G, pos, manifold_type=manifold_type)
    
    centrality = nx.degree_centrality(G)
    communities = knowledge_module.community.best_partition(G)

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
