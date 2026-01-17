# Last-Mile Delivery Optimizer - Project Report

**Authors:** [Your Names]  
**Course:** Optimisation Combinatoire  
**Date:** January 2026  
**Institution:** ENSI

---

## 1. Introduction

### 1.1 Context and Motivation

Urban grocery delivery is a rapidly growing market with significant operational challenges. Last-mile delivery—the final step of getting products from distribution centers to customers—accounts for up to 50% of total logistics costs. This project addresses key challenges in modern delivery operations:

- **Uncertain customer demands**: Order sizes are unknown until arrival
- **Soft time windows**: Customers prefer specific delivery times but violations can be tolerated with penalties
- **Variable travel times**: Traffic conditions, time of day, and other factors affect travel duration
- **Limited resources**: Fixed fleet size and vehicle capacity constraints

### 1.2 Problem Statement

Develop a decision-support tool that generates robust daily delivery routes by:
1. Predicting uncertain parameters (demand, travel times) from historical data
2. Optimizing routes that minimize total cost including penalties for time window violations

### 1.3 Objectives

- Build a **predict-then-optimize pipeline** combining machine learning and optimization
- Compare performance against baseline approaches
- Analyze trade-offs between prediction accuracy and optimization quality
- Provide actionable insights for urban delivery operations

---

## 2. Problem Formulation

### 2.1 Mathematical Formulation

**Decision Variables:**
- $x_{ij}^k \in \{0,1\}$: Binary variable = 1 if vehicle $k$ travels from node $i$ to node $j$
- $t_i \geq 0$: Arrival time at customer $i$
- $l_i^k \geq 0$: Load of vehicle $k$ when arriving at customer $i$

**Parameters:**
- $N$: Set of customers
- $K$: Set of vehicles
- $d_{ij}$: Distance from node $i$ to node $j$
- $\tau_{ij}$: Travel time from node $i$ to node $j$
- $q_i$: Demand of customer $i$ (uncertain)
- $[e_i, l_i]$: Time window for customer $i$
- $Q$: Vehicle capacity
- $T_{max}$: Maximum route duration

**Objective Function:**

Minimize:
$$Z = \sum_{k \in K} \sum_{i,j \in N} c \cdot d_{ij} \cdot x_{ij}^k + \sum_{k \in K} f \cdot y_k + \sum_{i \in N} (p_e \cdot \max(0, e_i - t_i) + p_l \cdot \max(0, t_i - l_i))$$

Where:
- $c$: Cost per km
- $f$: Fixed vehicle cost
- $p_e$: Early delivery penalty per hour
- $p_l$: Late delivery penalty per hour
- $y_k \in \{0,1\}$: Binary variable = 1 if vehicle $k$ is used

**Constraints:**

1. **Visit each customer exactly once:**
   $$\sum_{k \in K} \sum_{j \in N} x_{ij}^k = 1, \quad \forall i \in N$$

2. **Flow conservation:**
   $$\sum_{j \in N} x_{ij}^k = \sum_{j \in N} x_{ji}^k, \quad \forall i \in N, k \in K$$

3. **Vehicle capacity:**
   $$\sum_{i \in N} q_i \cdot \sum_{j \in N} x_{ij}^k \leq Q, \quad \forall k \in K$$

4. **Time window (soft):**
   $$e_i \leq t_i \leq l_i + slack_i, \quad \forall i \in N$$

5. **Route duration:**
   $$t_{depot}^{return} - t_{depot}^{start} \leq T_{max}, \quad \forall k \in K$$

### 2.2 Handling Uncertainty

Since demands $q_i$ and travel times $\tau_{ij}$ are uncertain, we use:

- **Point predictions** from ML models: $\hat{q}_i, \hat{\tau}_{ij}$
- **Prediction intervals** for robust planning: $[\hat{q}_i^{lower}, \hat{q}_i^{upper}]$
- **Conservative approach**: Use upper bound of demand for capacity planning

---

## 3. Literature Review and Background

### 3.1 Vehicle Routing Problem (VRP)

The VRP is a classic combinatorial optimization problem (Dantzig & Ramser, 1959). Our variant combines:

- **Capacitated VRP (CVRP)**: Vehicle capacity constraints
- **VRP with Time Windows (VRPTW)**: Customer time windows (Desrochers et al., 1988)
- **VRP with Soft Time Windows**: Violations allowed with penalties (Balakrishnan, 1993)

**Complexity:** VRP is NP-hard; exact methods struggle with >50 customers. Modern heuristics and metaheuristics provide good solutions efficiently.

### 3.2 Predict-then-Optimize Framework

Traditional approach: Predict → Optimize separately  
Modern approach: End-to-end learning (Elmachtoub & Grigas, 2022)

Our implementation:
1. **Predict:** Use Random Forest for demand and travel time
2. **Optimize:** Use OR-Tools CP-SAT solver for route optimization

**Key challenge:** Prediction errors propagate to optimization quality

### 3.3 Related Work

- **Stochastic VRP:** Model uncertainty explicitly (Gendreau et al., 1996)
- **Robust Optimization:** Worst-case guarantees (Bertsimas & Sim, 2004)
- **Real-time routing:** Dynamic adjustments (Psaraftis, 1988)

---

## 4. Methodology

### 4.1 Overall Architecture

```
[Historical Data] → [Prediction Models] → [Optimizer] → [Routes]
                         ↓                      ↓
                   [Demand, Travel Time]  [VRP Solver]
                         ↓                      ↓
                   [Uncertainty Bounds]   [Cost Evaluation]
```

### 4.2 Data Generation

**Synthetic data generation** (100 days historical + 10 test scenarios):

- **Customer locations:** Uniform distribution in 20km × 20km grid
- **Demands:** Customer-type dependent (small/medium/large) + day-of-week effect
- **Time windows:** 2-4 hour windows around preferred times
- **Travel times:** Distance-based + traffic multipliers (rush hour, weekends)

**Rationale:** Allows controlled experiments and reproducible results

### 4.3 Prediction Models

#### 4.3.1 Demand Prediction

**Model:** Random Forest Regressor (100 trees)

**Features:**
- Customer location (x, y)
- Day of week
- Distance from depot
- Historical average demand

**Output:** 
- Point estimate: $\hat{q}_i$
- 80% prediction interval: $[\hat{q}_i^{0.1}, \hat{q}_i^{0.9}]$ using quantile regression forest

**Evaluation metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

#### 4.3.2 Travel Time Prediction

**Model:** Random Forest Regressor (100 trees)

**Features:**
- Distance
- Hour of day
- Day of week
- Is rush hour (binary)
- Is weekend (binary)

**Output:** Predicted travel time $\hat{\tau}_{ij}$

### 4.4 Optimization Model

**Solver:** Google OR-Tools CP-SAT

**Implementation details:**
1. **Distance callback:** Euclidean distance between locations
2. **Time callback:** Travel time + service time
3. **Capacity dimension:** Track vehicle load
4. **Time dimension:** Track arrival times with soft windows
5. **Search strategy:** Guided Local Search (GLS)

**Parameters:**
- Time limit: 30 seconds
- First solution: Path Cheapest Arc
- Local search: GLS metaheuristic

### 4.5 Evaluation Methodology

**Three approaches compared:**

1. **Predict-then-Optimize (Proposed)**
   - Use ML predictions for optimization
   - Evaluate with actual values

2. **Oracle (Upper Bound)**
   - Optimize with perfect information
   - Best possible performance

3. **Baseline (Naive)**
   - Use historical averages
   - No sophisticated prediction

**Metrics:**
- Total cost (travel + vehicle + penalties)
- Prediction accuracy (MAE, RMSE)
- Time window compliance
- Capacity utilization
- Optimality gap from oracle

---

## 5. Experimental Results

### 5.1 Data Summary

**Historical Data (100 days):**
- Total deliveries: [X] records
- Average customers per day: [Y]
- Demand range: [min - max] kg
- Travel time range: [min - max] hours

**Test Scenarios (10 scenarios):**
- Average customers per scenario: [Z]
- Total test deliveries: [W]

### 5.2 Prediction Performance

#### Demand Prediction
| Metric | Value |
|--------|-------|
| MAE    | X.XX kg |
| RMSE   | X.XX kg |
| R²     | 0.XXX |

**Analysis:** [Interpret results - is accuracy sufficient? Where does model struggle?]

#### Travel Time Prediction
| Metric | Value |
|--------|-------|
| MAE    | X.XX hours (XX min) |
| RMSE   | X.XX hours |
| R²     | 0.XXX |

**Analysis:** [Interpret results]

### 5.3 Optimization Results (Scenario 0)

#### Route Statistics
- Number of vehicles used: X
- Total distance: XX.X km
- Total time: XX.X hours
- Average stops per vehicle: X.X
- Capacity utilization: XX%

#### Cost Breakdown
| Cost Component | Amount (€) |
|----------------|-----------|
| Travel Cost    | XXX.XX |
| Vehicle Cost   | XXX.XX |
| Penalty Cost   | XXX.XX |
| **Total**      | **XXX.XX** |

### 5.4 Comparative Analysis

| Approach | Total Cost (€) | Gap from Oracle |
|----------|----------------|-----------------|
| Oracle (Perfect Info) | XXX.XX | 0% |
| Predict-Optimize | XXX.XX | X.X% |
| Baseline (Averages) | XXX.XX | X.X% |

**Key Findings:**

1. **Predict-then-Optimize vs Oracle:** [X]% gap shows prediction impact
2. **Predict-then-Optimize vs Baseline:** [Y]% improvement demonstrates ML value
3. **Time window violations:** [Analysis of early/late deliveries]
4. **Capacity violations:** [Were there any? Impact?]

### 5.5 Sensitivity Analysis

[Optional: Test different parameters]

- Effect of prediction accuracy on total cost
- Impact of time window size
- Sensitivity to penalty costs
- Vehicle capacity variations

---

## 6. Discussion and Analysis

### 6.1 Why Did the Method Work (or Fail)?

**Successes:**
- [What worked well?]
- [Where did predictions help?]
- [Quality of routes?]

**Limitations:**
- [Where did model struggle?]
- [Impact of prediction errors?]
- [Optimization challenges?]

### 6.2 Critical Analysis

**Prediction Quality:**
- Random Forest captured [what patterns?]
- Uncertainty quantification via quantiles was [effective/limited because...]
- Main error sources: [identify]

**Optimization Quality:**
- OR-Tools found good solutions within time limit
- Soft time windows allowed feasibility but [trade-off...]
- Could improve by: [suggestions]

**Integration Challenges:**
- Prediction errors compound in optimization
- Conservative approach (upper bounds) led to [over/under] estimation
- Alternative: stochastic optimization could [improve/not help because...]

### 6.3 Trade-offs

**Accuracy vs Robustness:**
- Using upper bounds for demand → more conservative, higher vehicle costs
- Using point estimates → risk capacity violations

**Computation vs Quality:**
- 30s time limit: good balance for daily planning
- Longer optimization might reduce costs by [X]%

**Simplicity vs Realism:**
- Homogeneous fleet simplifies but limits flexibility
- Static depot vs multi-depot
- No real-time updates

### 6.4 Comparison to Literature

- Our gap from oracle ([X]%) compared to literature [Y]%
- Prediction approach similar to [reference] but differs in [aspect]
- Could adopt [technique from paper Z] to improve [metric]

---

## 7. Conclusions

### 7.1 Key Achievements

1. Successfully implemented predict-then-optimize pipeline
2. Demonstrated [X]% cost reduction vs naive baseline
3. Identified key trade-offs between prediction and optimization
4. Created reusable framework for delivery optimization

### 7.2 Main Insights

- **Insight 1:** [Key learning about prediction impact]
- **Insight 2:** [Key learning about optimization]
- **Insight 3:** [Key learning about integration]

### 7.3 Limitations

1. **Data:** Synthetic data may not capture all real-world complexities
2. **Model:** Assumes independent demands and travel times
3. **Optimization:** Heuristic solution, not guaranteed optimal
4. **Scope:** Single depot, homogeneous fleet, no dynamic updates

### 7.4 Future Work

**Short-term improvements:**
- Incorporate real traffic data (APIs)
- Test on real historical delivery data
- Implement online learning for continuous improvement

**Long-term extensions:**
- Deep learning for demand prediction (LSTM for sequences)
- Stochastic optimization (scenario-based)
- Multi-objective optimization (cost vs service quality)
- Real-time dynamic routing
- Integration with warehouse operations

**Research directions:**
- End-to-end learning (optimize prediction for decision quality, not accuracy)
- Robust optimization under distributional uncertainty
- Multi-agent coordination for large-scale systems

---

## 8. References

1. Dantzig, G. B., & Ramser, J. H. (1959). The truck dispatching problem. *Management Science*, 6(1), 80-91.

2. Desrochers, M., Desrosiers, J., & Solomon, M. (1992). A new optimization algorithm for the vehicle routing problem with time windows. *Operations Research*, 40(2), 342-354.

3. Elmachtoub, A. N., & Grigas, P. (2022). Smart "predict, then optimize". *Management Science*, 68(1), 9-26.

4. Gendreau, M., Laporte, G., & Séguin, R. (1996). Stochastic vehicle routing. *European Journal of Operational Research*, 88(1), 3-12.

5. Bertsimas, D., & Sim, M. (2004). The price of robustness. *Operations Research*, 52(1), 35-53.

6. Toth, P., & Vigo, D. (2014). *Vehicle routing: Problems, methods, and applications*. SIAM.

7. Google OR-Tools Documentation. https://developers.google.com/optimization

8. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

---

## Appendices

### Appendix A: Configuration Parameters

[Include key parameters from config.py]

### Appendix B: Additional Results

[Extra tables, figures, or detailed results]

### Appendix C: Code Structure

[Brief overview of code organization]

---

**Note:** Replace placeholders [X], [Y], etc. with your actual results. Add specific analyses based on your experiments.
