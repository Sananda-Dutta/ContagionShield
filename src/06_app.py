import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
from EoN import fast_SIR
import pandas as pd

# 1. THE ABSOLUTE FIRST STREAMLIT CALL: Set up config once and only once
st.set_page_config(page_title="ContagionShield", layout="wide")

@st.cache_resource
def load_data():
    with open("data/network.pkl", "rb") as f:
        G = pickle.load(f)
    with open("data/centrality.pkl", "rb") as f:
        centrality = pickle.load(f)
    with open("data/baseline_results.pkl", "rb") as f:
        baseline = pickle.load(f)
    return G, centrality, baseline

# Initialize data foundations
G, centrality, baseline = load_data()
N = G.number_of_nodes()
beta = baseline["operating_beta"]
gamma = baseline["gamma"]

# Sidebar Layout Controls
st.sidebar.header("Simulation Controls")

strategy = st.sidebar.selectbox(
    "Inoculation Strategy",
    ["Random", "Degree Centrality", "Betweenness Centrality"]
)

inoculation_pct = st.sidebar.slider(
    "% of Network Inoculated",
    min_value=0, max_value=30, value=10, step=1
)

run_button = st.sidebar.button("Run Simulation", type="primary")

def select_nodes(strategy, pct, G, centrality, seed=42):
    n_select = int(N * pct / 100)
    if n_select == 0:
        return []
    if strategy == "Random":
        rng = np.random.default_rng(seed)
        return list(rng.choice(list(G.nodes()), size=n_select, replace=False))
    elif strategy == "Degree Centrality":
        sorted_nodes = sorted(centrality["degree_centrality"].items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:n_select]]
    else:
        sorted_nodes = sorted(centrality["betweenness_centrality"].items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:n_select]]

# Core Simulation Execution Engine
if run_button:
    inoculated_nodes = select_nodes(strategy, inoculation_pct, G, centrality)
    
    n_demo_replicates = 15  
    attack_rates = []
    curves = []  
    
    with st.spinner(f"Running {n_demo_replicates} simulations..."):
        for i in range(n_demo_replicates):
            rng_i = np.random.default_rng(42 + i)
            susceptible_pool = [n for n in G.nodes() if n not in set(inoculated_nodes)]
            initial_infecteds = list(rng_i.choice(susceptible_pool, size=min(5, len(susceptible_pool)), replace=False))
            
            t, S, I, R = fast_SIR(
                G, beta, gamma,
                initial_infecteds=initial_infecteds,
                initial_recovereds=inoculated_nodes,
                tmax=50
            )
            rate = R[-1] / (N - len(inoculated_nodes)) if len(inoculated_nodes) < N else 0
            attack_rates.append(rate)
            curves.append((t, I, R))
    
    final_attack_rate = np.mean(attack_rates)
    attack_rate_std = np.std(attack_rates)
    baseline_rate = np.mean(baseline["attack_rates_above"])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Attack Rate (avg of 15 runs)", f"{final_attack_rate:.1%}", f"±{attack_rate_std:.1%} std")
    col2.metric("No-Intervention Baseline", f"{baseline_rate:.1%}")
    reduction = (baseline_rate - final_attack_rate) / baseline_rate * 100 if baseline_rate > 0 else 0
    col3.metric("Reduction vs. Baseline", f"{reduction:.1f}%")
    
    # Plot the FIRST replicate as a representative curve using explicit 3:2 aspect ratio canvas
    t, I, R = curves[0]
    fig, ax = plt.subplots(figsize=(9, 6))  # Explicit 3:2 Dimension Matrix Lock
    ax.plot(t, I, label="Infected", color="orange", linewidth=2)
    ax.plot(t, R, label="Recovered/Inoculated-effective", color="green", linewidth=2)
    ax.axhline(baseline_rate * N, color="red", linestyle="--", alpha=0.5, label="Baseline final size")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Nodes")
    ax.set_title(f"Epidemic Curve (1 of {n_demo_replicates} runs) — {strategy} @ {inoculation_pct}% Coverage")
    ax.legend()
    st.pyplot(fig, bbox_inches="tight")
    
    st.caption(f"Metrics above are averaged across {n_demo_replicates} simulation runs to account for natural stochastic variance (especially high for random seeding). Curve shown is one representative run.")
    
st.markdown("---")
st.header("📊 Sensitivity Analysis (Pre-computed)")
st.image("outputs/sensitivity_sweep.png", caption="Strategy effectiveness across inoculation coverage levels")

st.markdown("---")
st.header("🔍 What Are We Inoculating Against?")
st.markdown("Zero-shot classification of manipulation techniques in example text:")

nlp_df = pd.read_csv("outputs/nlp_classification_results.csv")
st.dataframe(nlp_df[["text", "true_label", "predicted_label", "confidence"]], use_container_width=True)