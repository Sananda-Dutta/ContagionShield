import EoN
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("data/network.pkl", "rb") as f:
    G = pickle.load(f)

with open("data/centrality.pkl", "rb") as f:
    centrality = pickle.load(f)

with open("data/baseline_results.pkl", "rb") as f:
    baseline = pickle.load(f)

degree_centrality = centrality["degree_centrality"]
betweenness_centrality = centrality["betweenness_centrality"]
beta = baseline["operating_beta"]
gamma = baseline["gamma"]
N = G.number_of_nodes()

print(f"Loaded network ({N} nodes), operating beta = {beta:.4f}")
print(f"Baseline (no inoculation) mean attack rate: {np.mean(baseline['attack_rates_above']):.3f}")
SEED = 42

def select_random(G, n_select, rng):
    return list(rng.choice(list(G.nodes()), size=n_select, replace=False))

def select_top_degree(degree_centrality, n_select):
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_nodes[:n_select]]

def select_top_betweenness(betweenness_centrality, n_select):
    sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_nodes[:n_select]]

def run_strategy(G, beta, gamma, inoculated_nodes, n_initial_infected, rng, tmax=50):
    susceptible_pool = [n for n in G.nodes() if n not in set(inoculated_nodes)]
    initial_infecteds = list(rng.choice(susceptible_pool, size=n_initial_infected, replace=False))
    
    t, S, I, R = EoN.fast_SIR(
        G, beta, gamma,
        initial_infecteds=initial_infecteds,
        initial_recovereds=inoculated_nodes,
        tmax=tmax
    )
    
    # R[-1] counts only nodes that recovered THROUGH the simulated infection process,
    # not the pre-set inoculated nodes — so we do NOT subtract len(inoculated_nodes) here.
    final_attack_rate = R[-1] / (N - len(inoculated_nodes))
    return final_attack_rate

# Test at 10% inoculation first
inoculation_pct = 0.10
n_select = int(N * inoculation_pct)
print(f"Inoculating {n_select} nodes ({inoculation_pct*100:.0f}% of network)")

rng = np.random.default_rng(SEED)

random_nodes = select_random(G, n_select, rng)
degree_nodes = select_top_degree(degree_centrality, n_select)
betweenness_nodes = select_top_betweenness(betweenness_centrality, n_select)

rate_random = run_strategy(G, beta, gamma, random_nodes, n_initial_infected=5, rng=np.random.default_rng(SEED))
rate_degree = run_strategy(G, beta, gamma, degree_nodes, n_initial_infected=5, rng=np.random.default_rng(SEED))
rate_betweenness = run_strategy(G, beta, gamma, betweenness_nodes, n_initial_infected=5, rng=np.random.default_rng(SEED))

print(f"Random seeding attack rate:       {rate_random:.3f}")
print(f"Degree-centrality attack rate:    {rate_degree:.3f}")
print(f"Betweenness-centrality attack rate: {rate_betweenness:.3f}")

def run_many_replicates_strategy(G, beta, gamma, inoculated_nodes, n_replicates=100, n_initial=5, base_seed=SEED):
    rates = []
    for i in range(n_replicates):
        rng_i = np.random.default_rng(base_seed + i)  # different seed each replicate, but reproducible
        rate = run_strategy(G, beta, gamma, inoculated_nodes, n_initial, rng_i)
        rates.append(rate)
    return rates

print("\nRunning Monte Carlo for each strategy (this will take a few minutes)...")

results = {}
for strategy_name, nodes in [("random", random_nodes), ("degree", degree_nodes), ("betweenness", betweenness_nodes)]:
    print(f"  Running strategy: {strategy_name}")
    results[strategy_name] = run_many_replicates_strategy(G, beta, gamma, nodes, n_replicates=100)

for strategy_name, rates in results.items():
    print(f"{strategy_name:12s} — mean attack rate: {np.mean(rates):.3f}, std: {np.std(rates):.3f}")

plt.figure(figsize=(9, 6))
plt.boxplot(
    [results["random"], results["degree"], results["betweenness"]],
    tick_labels=["Random", "Degree Centrality", "Betweenness Centrality"]
)
plt.axhline(np.mean(baseline["attack_rates_above"]), color="red", linestyle="--", label="No inoculation (baseline)")
plt.ylabel("Final attack rate")
plt.title(f"Inoculation Strategy Comparison at {inoculation_pct*100:.0f}% Coverage")
plt.legend()
plt.savefig("outputs/strategy_comparison_10pct.png", dpi=150)
plt.show()
print("Saved outputs/strategy_comparison_10pct.png")

strategy_results = {
    "inoculation_pct": inoculation_pct,
    "n_select": n_select,
    "results": results,
    "random_nodes": random_nodes,
    "degree_nodes": degree_nodes,
    "betweenness_nodes": betweenness_nodes,
}

with open("data/strategy_results_10pct.pkl", "wb") as f:
    pickle.dump(strategy_results, f)

print("Saved strategy comparison results to data/strategy_results_10pct.pkl")
