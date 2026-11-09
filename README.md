# Theoretical Guarantees for Graph Signal Sampling under Topology Mismatch

This repository contains the MATLAB simulation code accompanying the paper:

**L. Dabush and T. Routtenberg, "Theoretical Guarantees for Graph Signal Sampling under Topology Mismatch," 2026. Manuscript submitted for publication.**

---

## Overview

Graph signal processing (GSP) methods for sampling and reconstruction rely on a known graph topology. In practice, the assumed topology may differ from the true one — a problem known as **topology mismatch**. This paper provides the first theoretical guarantees on the estimation risk under such mismatch.

The code validates the paper's main results on the **IEEE 118-bus power system** graph, using a Bayesian signal model with the DC power-flow graph filter. All theoretical risk curves are compared against Monte Carlo simulations.

---

## Theoretical Background

Consider a graph signal `x ~ N(0, (μ h_R(L))†)` observed under additive noise and a stochastic topology perturbation `Δh`:

```
y = h_M(L) x + e − Δh x,     e ~ N(0, Σ)
```

Under the **proportional mismatch model** `C_δ = αΣ`, the risk of the graph-filter-based regularised least-squares (GFR-ML) estimator decomposes exactly as:

```
R_α(d) = BMSE(d) + α · bCRB(d)
```

where `d` is the sampling design (a subset of nodes), `BMSE(d) = tr(K⁻¹)` and `bCRB(d) = tr(K⁻²M)`, with `K = M + F_r`, `M = H_rᵀ D Σ⁻¹ D H_r`.

### Key Results

| Result | Statement |
|---|---|
| **Theorem 1** | Exact risk identity `R_α(d) = BMSE(d) + α·bCRB(d)` for all α ≥ 0 |
| **Theorem 2** | The bCRB-optimal design `d_C` achieves regret `Reg_α(d_C) ≤ Δ_B` for all α, where `Δ_B = BMSE(d_C) − BMSE(d_B)` |
| **Theorem 3** | The BMSE-optimal design `d_B` suffers regret that grows linearly with α |
| **Corollary 1** | The minimax-optimal design for the uncertainty set `[α/κ, κα]` is `d_{κα}`, regardless of the true α |
| **Corollary 2** | Explicit worst-case regret bound for the minimax design over all C_δ in the uncertainty set |
| **Corollary 3** | Crossover α₀ = Δ_B / Δ_C: for α < α₀, `d_B` is preferred; for α > α₀, `d_C` is preferred |

---

## Requirements

- MATLAB R2020b or later  
- No additional toolboxes are required (all linear algebra uses built-in MATLAB functions)

---

## Repository Structure

```
.
├── main_model_matched_mismatch.m       # Main simulation script (entry point)
│
├── GreedySampling/
│   ├── GreedySampling.m                # Core greedy sampling algorithm
│   ├── greedy_sampling_switch_method.m # Dispatches greedy runs by cost function
│   └── efficient_update_function/
│       ├── Calc_BMSE_efficient.m       # Rank-1 (Sherman–Morrison) BMSE update
│       └── Calc_BCRB_efficient.m       # Rank-1 (Sherman–Morrison) bCRB update
│
├── cost_functions/
│   ├── Calc_BMSE.m                     # Full BMSE = tr(K⁻¹)
│   └── Calc_BCRB.m                     # Full bCRB = tr(K⁻²M)
│
├── sampling_methods/
│   ├── A_optimal_design.m              # A-optimal (minimum trace) design
│   ├── E_optimal_design.m              # E-optimal (maximum min-eigenvalue) design
│   ├── D_optimal_design.m              # D-optimal (maximum determinant) design
│   └── T_optimal_design.m              # T-optimal design
│
├── auxiliary_functions/
│   ├── pseudo_inverse.m                # Numerically stable pseudoinverse
│   ├── samplingSet2vector.m            # Converts a node index set to a 0/1 vector
│   ├── ensure_incresing_order.m        # Sorts eigenvectors by eigenvalue
│   ├── getPlottingConstants.m          # Line styles, colours, method metadata
│   └── load_cdf.m                      # Parses IEEE Common Data Format (.cdf) files
│
├── data_source_files/
│   └── ieee118cdf.txt                  # IEEE 118-bus power system (see citation below)
│
├── PGD_Sampling/                       # Projected gradient descent sampling (auxiliary)
│
└── Figures/
    └── ieee118/          # Output figures and workspace (.mat)
```

---

## Running the Simulation

Open MATLAB, navigate to the repository root, and run:

```matlab
main_model_matched_mismatch
```

The script will:

1. Load the IEEE 118-bus admittance matrix and construct the graph Laplacian.
2. Set up the Bayesian signal model (prior, noise, mismatch).
3. Compute greedy sampling designs: `d_B`, `d_C`, oracle `d_α` (for each α), and minimax `d_mm` (for each α).
4. Run a Monte Carlo simulation with T = 20,000 realizations.
5. Compute theoretical risk curves and empirical MSE.
6. Evaluate regret, nominal gaps Δ_B and Δ_C, the crossover α₀, and worst-case minimax regret.
7. Save figures to `./Figures/model_matched/ieee118/` and the workspace to `workspace.mat`.


---

## Key Functions

### `GreedySampling.m`
Sequential greedy sampling that minimises a given cost function over node subsets. For the first selected node the full cost function is evaluated; for subsequent nodes a **Sherman–Morrison rank-1 update** is used to avoid re-inverting the full matrix at each step, reducing the per-step cost from O(n³) to O(n²).

### `Calc_BMSE.m` / `Calc_BMSE_efficient.m`
Compute `BMSE(d) = tr(K⁻¹)`. The efficient version uses the Sherman–Morrison formula for rank-1 updates inside the greedy loop.

### `Calc_BCRB.m` / `Calc_BCRB_efficient.m`
Compute `bCRB(d) = tr(K⁻²M)`. Used as the cost function for the bCRB-optimal design `d_C` and as the mismatch-risk term in composite objectives.

### `main_model_matched_mismatch.m` — local functions

| Function | Purpose |
|---|---|
| `design_costs(S, ...)` | Returns BMSE, bCRB, K⁻¹, M, D for a given sampling set S |
| `eval_set(S, ...)` | Evaluates theoretical risk and Monte Carlo MSE across all α values |
| `bcrb_node_swap(S, ...)` | 2-opt node-swap local search minimising bCRB (post-processes `d_C`) |
| `composite_node_swap(S, ..., α)` | 2-opt node-swap minimising BMSE + α·bCRB (used for oracle/minimax refinement) |

---

## Validated Results

Running `main_model_matched_mismatch.m` reproduces:

- **Panel (a) — NMSE vs α:** Theory lines (`BMSE + α·bCRB`) match Monte Carlo markers for all designs (Theorem 1).
- **Panel (b) — Normalised regret vs α:** `d_C` regret stays at or below the Δ_B line for all α (Theorem 2); `d_B` regret grows linearly (Theorem 3); minimax regret is bounded (Corollary 2).
- **Console output:** Worst-case risk and regret table for the minimax policy at each α_max, together with the Corollary 2 bound (all entries verified numerically).
- **NMSE table:** Performance of all designs at α = 10 for sensor budgets q ∈ {30%, …, 90%}, saved as a CSV.

---

## Data Source

The IEEE 118-bus power system test case is provided in IEEE Common Data Format (CDF):

> *IEEE 118 Bus Test Case*, Power Systems Test Case Archive, University of Washington.  
> Available: https://labs.ece.uw.edu/pstca/pf118/pg_tca118bus.htm

The admittance matrix is parsed by `load_cdf.m`. The imaginary part of `Y` gives the susceptance matrix; diagonal corrections enforce `L·1 = 0` so that the resulting Laplacian is positive semi-definite with a single zero eigenvalue (the slack bus, node 111).

---

## Contact

For questions or inquiries, please contact:

- Lital Dabush: litaldab@post.bgu.ac.il
- Tirza Routtenberg: tirzar@bgu.ac.il
<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@misc{Dabush2026Routtenberg,
  author        = {L. Dabush and T. Routtenberg},
  title         = {Theoretical Guarantees for Graph Signal Sampling under Topology Mismatch},
  year          = {2026},
  eprint        = {2XXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SP}
}
``` -->

