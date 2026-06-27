import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

SEED = 42
np.random.seed(SEED)

# Generate the network: 1000 nodes, each new node connects to 3 existing ones
N = 1000
M = 3
G = nx.barabasi_albert_graph(n=N, m=M, seed=SEED)

print(f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
# Degree statistics
degrees = [d for n, d in G.degree()]
avg_degree = np.mean(degrees)
max_degree = np.max(degrees)

# Centrality measures (you'll need these in Phase 3 for seeding strategies)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, k=200, seed=SEED)  # k=200 samples for speed on 1000 nodes

# Clustering coefficient: how "clique-y" the network is
avg_clustering = nx.average_clustering(G)

# Spectral radius: determines the theoretical epidemic threshold
adjacency_matrix = nx.to_numpy_array(G)
eigenvalues = np.linalg.eigvals(adjacency_matrix)
spectral_radius = max(eigenvalues.real)

print(f"Average degree: {avg_degree:.2f}")
print(f"Max degree: {max_degree}")
print(f"Average clustering coefficient: {avg_clustering:.4f}")
print(f"Spectral radius: {spectral_radius:.4f}")

gamma = 1.0  # we'll fix recovery rate at 1.0 for simplicity; beta is what we vary
theoretical_beta_threshold = gamma / spectral_radius

print(f"Theoretical epidemic threshold (beta_c): {theoretical_beta_threshold:.4f}")
print(f"  -> beta below this: outbreak should fizzle out")
print(f"  -> beta above this: outbreak should spread widely")

plt.figure(figsize=(9, 6))

# Color nodes by degree so hubs are visually obvious
node_colors = [degrees[i] for i in range(len(degrees))]
pos = nx.spring_layout(G, seed=SEED, k=0.3)  # k controls spacing; tweak if too cluttered

nx.draw(
    G, pos,
    node_size=30,
    node_color=node_colors,
    cmap=plt.cm.viridis,
    edge_color="gray",
    alpha=0.6,
    with_labels=False
)
plt.title("Synthetic Social Network (Barabási–Albert, n=1000)")
plt.savefig("outputs/network_structure.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot to outputs/network_structure.png")

# Save the graph object itself
with open("data/network.pkl", "wb") as f:
    pickle.dump(G, f)

# Save the diagnostics as a simple text/JSON summary too, for quick reference
import json
diagnostics = {
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "avg_degree": float(avg_degree),
    "max_degree": int(max_degree),
    "avg_clustering": float(avg_clustering),
    "spectral_radius": float(spectral_radius),
    "theoretical_beta_threshold": float(theoretical_beta_threshold),
}
with open("data/network_diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)

# Save centrality dictionaries too — Phase 3 needs these for seeding strategies
with open("data/centrality.pkl", "wb") as f:
    pickle.dump({
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality
    }, f)

print("Saved network, diagnostics, and centrality data to /data")