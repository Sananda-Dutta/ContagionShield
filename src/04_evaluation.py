import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import mannwhitneyu
from EoN import fast_SIR

with open("data/strategy_results_10pct.pkl", "rb") as f:
    strategy_results = pickle.load(f)

with open("data/baseline_results.pkl", "rb") as f:
    baseline = pickle.load(f)

results_10pct = strategy_results["results"]

# Mann-Whitney U test: are degree/betweenness genuinely different from random,
# or could this just be noise? (non-parametric, doesn't assume normal distributions —
# appropriate here since attack rates aren't necessarily normally distributed)
stat_degree, p_degree = mannwhitneyu(results_10pct["random"], results_10pct["degree"], alternative="greater")
stat_betweenness, p_betweenness = mannwhitneyu(results_10pct["random"], results_10pct["betweenness"], alternative="greater")

print("=== Statistical Significance (at 10% inoculation) ===")
print(f"Random vs Degree:      p = {p_degree:.2e}")
print(f"Random vs Betweenness: p = {p_betweenness:.2e}")

baseline_mean = np.mean(baseline["attack_rates_above"])
random_mean = np.mean(results_10pct["random"])
degree_mean = np.mean(results_10pct["degree"])
betweenness_mean = np.mean(results_10pct["betweenness"])

pct_reduction_degree_vs_random = (random_mean - degree_mean) / random_mean * 100
pct_reduction_betweenness_vs_random = (random_mean - betweenness_mean) / random_mean * 100
pct_reduction_degree_vs_baseline = (baseline_mean - degree_mean) / baseline_mean * 100

print("\n=== Effect Sizes (at 10% inoculation) ===")
print(f"Degree strategy reduces attack rate by {pct_reduction_degree_vs_random:.1f}% vs random seeding")
print(f"Betweenness strategy reduces attack rate by {pct_reduction_betweenness_vs_random:.1f}% vs random seeding")
print(f"Degree strategy reduces attack rate by {pct_reduction_degree_vs_baseline:.1f}% vs no inoculation at all")


def select_top_degree(degree_centrality, n_select):
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_nodes[:n_select]]

def select_top_betweenness(betweenness_centrality, n_select):
    sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_nodes[:n_select]]

def run_strategy(G, beta, gamma, inoculated_nodes, n_initial_infected, rng, tmax=50):
    susceptible_pool = [n for n in G.nodes() if n not in set(inoculated_nodes)]
    initial_infecteds = list(rng.choice(susceptible_pool, size=n_initial_infected, replace=False))
    t, S, I, R = fast_SIR(G, beta, gamma, initial_infecteds=initial_infecteds,
                          initial_recovereds=inoculated_nodes, tmax=tmax)
    return R[-1] / (len(G.nodes()) - len(inoculated_nodes))

with open("data/network.pkl", "rb") as f:
    G = pickle.load(f)
with open("data/centrality.pkl", "rb") as f:
    centrality = pickle.load(f)

degree_centrality = centrality["degree_centrality"]
betweenness_centrality = centrality["betweenness_centrality"]
beta = baseline["operating_beta"]
gamma = baseline["gamma"]
N = G.number_of_nodes()

percentages = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
n_replicates_sweep = 30
SEED = 42

sweep_results = {"random": [], "degree": [], "betweenness": []}

print("\n=== Running Sensitivity Sweep ===")
for pct in percentages:
    n_select = int(N * pct)
    print(f"Coverage: {pct*100:.0f}% ({n_select} nodes)")
    
    rng_select = np.random.default_rng(SEED)
    random_nodes = list(rng_select.choice(list(G.nodes()), size=n_select, replace=False))
    degree_nodes = select_top_degree(degree_centrality, n_select)
    betweenness_nodes = select_top_betweenness(betweenness_centrality, n_select)
    
    for strategy_name, nodes in [("random", random_nodes), ("degree", degree_nodes), ("betweenness", betweenness_nodes)]:
        rates = []
        for i in range(n_replicates_sweep):
            rng_i = np.random.default_rng(SEED + i)
            rate = run_strategy(G, beta, gamma, nodes, n_initial_infected=5, rng=rng_i)
            rates.append(rate)
        sweep_results[strategy_name].append(np.mean(rates))
        print(f"  {strategy_name:12s}: mean attack rate = {np.mean(rates):.3f}")
        
plt.figure(figsize=(9, 6))
pct_labels = [p * 100 for p in percentages]

plt.plot(pct_labels, sweep_results["random"], marker="o", label="Random", color="gray")
plt.plot(pct_labels, sweep_results["degree"], marker="o", label="Degree Centrality", color="darkorange")
plt.plot(pct_labels, sweep_results["betweenness"], marker="o", label="Betweenness Centrality", color="steelblue")
plt.axhline(baseline_mean, color="red", linestyle="--", alpha=0.6, label="No inoculation (baseline)")

plt.xlabel("% of Network Inoculated")
plt.ylabel("Mean Final Attack Rate")
plt.title("Inoculation Strategy Effectiveness vs Coverage")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("outputs/sensitivity_sweep.png", dpi=150)
plt.show()
print("Saved outputs/sensitivity_sweep.png")

summary_df = pd.DataFrame({
    "coverage_pct": pct_labels,
    "random_attack_rate": sweep_results["random"],
    "degree_attack_rate": sweep_results["degree"],
    "betweenness_attack_rate": sweep_results["betweenness"],
})
summary_df.to_csv("outputs/sensitivity_summary.csv", index=False)
print("\nSaved summary table to outputs/sensitivity_summary.csv")
print(summary_df)

# Save final headline stats for the README
final_stats = {
    "p_value_degree": p_degree,
    "p_value_betweenness": p_betweenness,
    "pct_reduction_degree_vs_random_10pct": pct_reduction_degree_vs_random,
    "pct_reduction_betweenness_vs_random_10pct": pct_reduction_betweenness_vs_random,
    "baseline_mean_attack_rate": baseline_mean,
}
with open("data/final_stats.pkl", "wb") as f:
    pickle.dump(final_stats, f)