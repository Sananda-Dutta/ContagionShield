import EoN #Epidemics on Networks. It is a highly specialized math library that handles the heavy processing loops required to simulate a virus spreading across a network.
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

with open("data/network.pkl", "rb") as f:
    G = pickle.load(f)

with open("data/network_diagnostics.json") as f:
    diagnostics = json.load(f)

beta_threshold = diagnostics["theoretical_beta_threshold"]
N = diagnostics["n_nodes"]
print(f"Loaded network with {N} nodes. Threshold beta_c = {beta_threshold:.4f}")

gamma = 1.0  # recovery rate, fixed throughout the project

beta_below = beta_threshold * 0.5   # should fizzle out
beta_above = beta_threshold * 3.0   # should spread widely — this becomes our "operating" beta for the rest of the project

print(f"Testing beta_below = {beta_below:.4f}")
print(f"Testing beta_above = {beta_above:.4f}")

SEED = 42
rng = np.random.default_rng(SEED)

# Pick 5 random "patient zero" nodes — same starting point for both runs, for a fair comparison
initial_infecteds = list(rng.choice(list(G.nodes()), size=5, replace=False))

def run_sir(G, beta, gamma, initial_infecteds, tmax=50):
    t, S, I, R = EoN.fast_SIR(G, beta, gamma, initial_infecteds=initial_infecteds, tmax=tmax)
    return t, S, I, R

t_below, S_below, I_below, R_below = run_sir(G, beta_below, gamma, initial_infecteds)
t_above, S_above, I_above, R_above = run_sir(G, beta_above, gamma, initial_infecteds)

attack_rate_below = R_below[-1] / N
attack_rate_above = R_above[-1] / N

print(f"Final attack rate (below threshold): {attack_rate_below:.3f}")
print(f"Final attack rate (above threshold): {attack_rate_above:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(t_below, I_below, color="orange", label="Infected")
axes[0].plot(t_below, R_below, color="green", label="Recovered")
axes[0].set_title(f"Below Threshold (β={beta_below:.3f})")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Number of nodes")
axes[0].legend()

axes[1].plot(t_above, I_above, color="orange", label="Infected")
axes[1].plot(t_above, R_above, color="green", label="Recovered")
axes[1].set_title(f"Above Threshold (β={beta_above:.3f})")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Number of nodes")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/threshold_validation.png", dpi=150)
plt.show()
print("Saved outputs/threshold_validation.png")

def run_many_replicates(G, beta, gamma, n_replicates=50, n_initial=5, tmax=50):
    final_attack_rates = []
    rng_local = np.random.default_rng(SEED)
    for i in range(n_replicates):
        seeds = list(rng_local.choice(list(G.nodes()), size=n_initial, replace=False))
        t, S, I, R = EoN.fast_SIR(G, beta, gamma, initial_infecteds=seeds, tmax=tmax)
        final_attack_rates.append(R[-1] / N)
    return final_attack_rates

print("Running Monte Carlo for below-threshold beta...")
attack_rates_below = run_many_replicates(G, beta_below, gamma, n_replicates=50)

print("Running Monte Carlo for above-threshold beta...")
attack_rates_above = run_many_replicates(G, beta_above, gamma, n_replicates=50)

print(f"Below threshold — mean attack rate: {np.mean(attack_rates_below):.3f}, std: {np.std(attack_rates_below):.3f}")
print(f"Above threshold — mean attack rate: {np.mean(attack_rates_above):.3f}, std: {np.std(attack_rates_above):.3f}")

plt.figure(figsize=(8, 5))
plt.boxplot([attack_rates_below, attack_rates_above], labels=["Below threshold", "Above threshold"])
plt.ylabel("Final attack rate (fraction of network infected)")
plt.title("Baseline Epidemic Outcomes: Below vs Above Threshold")
plt.savefig("outputs/baseline_threshold_boxplot.png", dpi=150)
plt.show()
print("Saved outputs/baseline_threshold_boxplot.png")

baseline_results = {
    "beta_below": beta_below,
    "beta_above": beta_above,
    "gamma": gamma,
    "attack_rates_below": attack_rates_below,
    "attack_rates_above": attack_rates_above,
    "operating_beta": beta_above,   # this is the beta we'll use for the rest of the project
    "operating_initial_infected_count": 5,
}

with open("data/baseline_results.pkl", "wb") as f:
    pickle.dump(baseline_results, f)

print("Saved baseline results to data/baseline_results.pkl")
print(f"\nLOCKED IN: operating beta for rest of project = {beta_above:.4f}")
print(f"This represents our 'misinformation epidemic' with no inoculation.")
print(f"Baseline mean attack rate with no intervention: {np.mean(attack_rates_above):.3f}")
