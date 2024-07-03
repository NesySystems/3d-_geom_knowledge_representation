import spacy
import networkx as nx
import pytesseract
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load SpaCy model for NLP tasks
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

    return list(set(concepts)), relations  # Remove duplicate concepts

def create_knowledge_graph(concepts, relations):
    G = nx.DiGraph()
    for concept in concepts:
        G.add_node(concept)
    for subject, relation, object in relations:
        if subject in G.nodes() and object in G.nodes():
            G.add_edge(subject, object, relation=relation)
    return G

def embed_graph_in_3d(G):
    # Create an adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    # Use t-SNE for 3D embedding
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_3d = tsne.fit_transform(adj_matrix)

    return embeddings_3d

def visualize_3d_embeddings(embeddings, labels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = embeddings.T
    ax.scatter(xs, ys, zs)

    for i, label in enumerate(labels):
        ax.text(xs[i], ys[i], zs[i], label, fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Knowledge Embedding')
    plt.show()

def process_mindmap(image_path):
    print(f"Processing image: {image_path}")
    # Extract text from image
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    print("Text extracted from image.")

    # Extract concepts and relations
    print("Extracting concepts and relations...")
    concepts, relations = extract_concepts_and_relations(text)

    # Create knowledge graph
    print("Creating knowledge graph...")
    G = create_knowledge_graph(concepts, relations)

    # Embed graph in 3D
    print("Embedding graph in 3D...")
    embeddings = embed_graph_in_3d(G)

    # Visualize 3D embeddings
    print("Visualizing 3D embeddings...")
    visualize_3d_embeddings(embeddings, concepts)

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

    # You can now query the graph or perform further analysis
    # For example, finding the most central concepts:
    central_concepts = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nMost central concepts:")
    for concept, centrality in central_concepts:
        print(f"{concept}: {centrality:.4f}")