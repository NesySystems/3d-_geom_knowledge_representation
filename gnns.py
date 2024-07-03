import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from knowledge_extraction import extract_knowledge_from_mindmap
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def visualize_graph(G):
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=1000, arrows=True)
    
    # Add labels to nodes
    labels = nx.get_node_attributes(G, 'content')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")
    
    # Add edge labels
    edge_labels = {(u, v): f"{u}->{v}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title("Knowledge Graph Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def prepare_data_for_gnn(G):
    print("Preparing data for GNN...")
    # Convert node content to numerical features (e.g., word count)
    node_features = [[len(G.nodes[n]['content'].split())] for n in G.nodes()]
    x = torch.tensor(node_features, dtype=torch.float)
    print(f"Node features shape: {x.shape}")
    
    # Prepare edge index
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    print(f"Edge index shape: {edge_index.shape}")
    
    return Data(x=x, edge_index=edge_index)

def main():
    try:
        print("Starting knowledge extraction...")
        
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()

        # Open file dialog to select the mind map image
        mindmap_path = filedialog.askopenfilename(
            title="Select your mind map image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )

        if not mindmap_path:
            print("No file selected. Exiting.")
            return

        print(f"Selected file: {mindmap_path}")
        
        knowledge_graph = extract_knowledge_from_mindmap(mindmap_path)
        
        print("Visualizing extracted graph...")
        visualize_graph(knowledge_graph)
        
        print("\nPreparing data for GNN...")
        data = prepare_data_for_gnn(knowledge_graph)
        
        print("\nInitializing and training GNN model...")
        model = GNN(num_features=data.num_features, hidden_channels=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            # Use a simple reconstruction loss
            loss = F.mse_loss(out, data.x)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
        
        print("\nTraining complete. You can now query the model.")
        
        while True:
            query = input("\nEnter a node index to find related nodes (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            try:
                node_idx = int(query)
                if node_idx not in knowledge_graph.nodes():
                    print(f"Node {node_idx} not found in the graph.")
                    continue
                
                # Get embeddings for all nodes
                model.eval()
                with torch.no_grad():
                    embeddings = model(data)
                
                # Compute similarities with all other nodes
                query_embedding = embeddings[node_idx]
                similarities = F.cosine_similarity(query_embedding, embeddings)
                
                # Get top 5 most similar nodes
                top_similar = similarities.argsort(descending=True)[1:6]  # Exclude the node itself
                
                print(f"Nodes most related to node {node_idx} ({knowledge_graph.nodes[node_idx]['content']}):")
                for idx in top_similar:
                    print(f"Node {idx.item()}: {knowledge_graph.nodes[idx.item()]['content']}")
            
            except ValueError:
                print("Please enter a valid node index.")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()