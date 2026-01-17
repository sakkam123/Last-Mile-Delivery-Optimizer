# Last-Mile Delivery Optimizer - Presentation Outline
## 15-Minute Presentation Guide

---

## Slide 1: Title Slide (30 seconds)
**Title:** Last-Mile Delivery Optimizer: A Predict-Then-Optimize Approach

**Content:**
- Project title
- Your names
- Course: Optimisation Combinatoire - ENSI
- Date: January 2026

**Notes:** Introduce yourselves and the project topic briefly.

---

## Slide 2: Problem Context (1 minute)
**Title:** The Last-Mile Delivery Challenge

**Content:**
- Urban grocery delivery is complex and costly
- **Key Challenges:**
  - 📦 Uncertain customer demands
  - ⏰ Soft time window constraints (with penalties)
  - 🚗 Variable travel times
  - 💰 Limited resources (vehicles, capacity)
- Last-mile delivery = 50% of total logistics costs

**Visual:** Image of delivery truck in urban environment

**Notes:** Set the stage - why this problem matters. Emphasize the real-world relevance.

---

## Slide 3: Problem Statement & Objectives (1 minute)
**Title:** Our Approach

**Problem:**
Generate robust daily delivery routes when demands and travel times are uncertain

**Solution:**
**Predict-Then-Optimize Pipeline**

1. **Predict:** Use ML to forecast demands and travel times
2. **Optimize:** Use OR-Tools to generate cost-optimal routes

**Objectives:**
- Minimize total cost (travel + penalties + vehicles)
- Handle uncertainty robustly
- Compare against baselines

**Visual:** Simple flowchart: Historical Data → ML Models → Predictions → Optimizer → Routes

---

## Slide 4: Mathematical Formulation (1.5 minutes)
**Title:** Vehicle Routing Problem with Soft Time Windows

**Content:**

**Decision Variables:**
- Route assignments: $x_{ij}^k$ (vehicle k goes from i to j)
- Arrival times: $t_i$

**Objective:**
$$\min: \text{Travel Cost} + \text{Vehicle Cost} + \text{Penalty Cost}$$

**Constraints:**
- Vehicle capacity: $\sum q_i \leq Q$
- Visit each customer once
- Time windows (soft): penalties for early/late delivery
- Route duration limits

**Key Innovation:** Soft time windows allow flexibility with economic penalties

**Visual:** Simple diagram showing depot, customers, routes, time windows

**Notes:** Don't go deep into math - focus on intuition. Highlight the soft time window concept.

---

## Slide 5: Methodology - Overall Pipeline (1 minute)
**Title:** System Architecture

**Content:**
```
Historical Data (100 days)
         ↓
   [ML Prediction Models]
    • Random Forest for demand
    • Random Forest for travel time
    • Uncertainty quantification
         ↓
   [Predictions + Bounds]
         ↓
   [OR-Tools Optimizer]
    • CP-SAT Solver
    • Guided Local Search
    • 30-second time limit
         ↓
   [Optimized Routes]
         ↓
   [Evaluation with Actual Values]
```

**Notes:** Walk through the pipeline. Emphasize the two-stage approach.

---

## Slide 6: Data & Features (45 seconds)
**Title:** Data Generation & Features

**Historical Data:**
- 100 days of delivery records
- 15-25 customers per day
- Synthetic but realistic patterns

**Prediction Features:**

| Demand | Travel Time |
|--------|-------------|
| • Location | • Distance |
| • Day of week | • Hour of day |
| • Historical avg | • Rush hour flag |
| • Distance | • Weekend flag |

**Uncertainty Handling:**
- Predict 80% confidence intervals
- Use upper bound for robust planning

---

## Slide 7: Results - Prediction Performance (1.5 minutes)
**Title:** Machine Learning Prediction Results

**Content:**

**Demand Prediction:**
- MAE: [X.XX] kg
- R²: [0.XXX]
- ✓ Captures customer type and day-of-week patterns

**Travel Time Prediction:**
- MAE: [XX] minutes
- R²: [0.XXX]
- ✓ Captures rush hour and distance effects

**Visual:** 
- Two scatter plots: Predicted vs Actual (for demand and travel time)
- Show prediction accuracy visually

**Notes:** Briefly comment on prediction quality. Are the models accurate enough?

---

## Slide 8: Results - Route Optimization (1.5 minutes)
**Title:** Optimized Route Example (Scenario 0)

**Content:**

**Route Statistics:**
- Customers: [XX]
- Vehicles used: [X/3]
- Total distance: [XX.X] km
- Capacity utilization: [XX]%

**Visual:** 
- **Main visual:** Map showing optimized routes with different colors per vehicle
- Depot (red square) and customers (colored circles)
- Route lines connecting them

**Notes:** Walk through one example route. Point out how solver balanced time windows and capacity.

---

## Slide 9: Results - Cost Analysis (1.5 minutes)
**Title:** Cost Breakdown & Performance

**Content:**

**Cost Components (Scenario 0):**
| Component | Amount (€) | % of Total |
|-----------|-----------|------------|
| Travel Cost | [XXX.XX] | [XX]% |
| Vehicle Cost | [XXX.XX] | [XX]% |
| Penalty Cost | [XXX.XX] | [XX]% |
| **Total** | **[XXX.XX]** | **100%** |

**Time Window Performance:**
- On-time deliveries: [XX]%
- Early arrivals: [XX]%
- Late arrivals: [XX]%

**Visual:** Stacked bar chart or pie chart of cost components

**Notes:** Explain what drives costs. Discuss time window compliance.

---

## Slide 10: Comparative Analysis (2 minutes)
**Title:** Performance Comparison: Three Approaches

**Content:**

| Approach | Description | Total Cost | Gap from Oracle |
|----------|-------------|------------|-----------------|
| **Oracle** | Perfect information | €[XXX] | 0% |
| **Predict-Optimize** | ML predictions | €[XXX] | **[X.X]%** |
| **Baseline** | Historical averages | €[XXX] | [XX]% |

**Key Findings:**
- ✅ Predict-optimize **[X]% better** than baseline
- ✅ Only **[Y]% gap** from perfect information
- ✅ Demonstrates value of ML predictions

**Visual:** Bar chart comparing the three approaches

**Notes:** This is a key slide! Emphasize the improvement over baseline and small gap from oracle.

---

## Slide 11: Critical Analysis - Why It Works (1.5 minutes)
**Title:** Analysis: Strengths & Trade-offs

**What Worked Well:**
- ✓ Random Forest captured demand patterns effectively
- ✓ Uncertainty quantification prevented capacity violations
- ✓ OR-Tools found near-optimal solutions quickly
- ✓ Soft time windows provided practical flexibility

**Trade-offs Identified:**
- **Accuracy vs Robustness:** Using upper bounds → conservative but safer
- **Computation vs Quality:** 30s limit → good balance for daily planning
- **Prediction vs Optimization:** Errors compound, but impact is manageable

**Limitations:**
- Synthetic data (not real-world tested)
- Homogeneous fleet (single vehicle type)
- No dynamic re-routing
- Independent predictions (demands could be correlated)

**Notes:** Show critical thinking! Discuss why the approach works and where it could fail.

---

## Slide 12: Visualizations (1 minute)
**Title:** Visual Insights

**Content:**
Show 2-3 impactful visualizations:

1. **Route Map:** Optimized routes on city grid
2. **Time Window Chart:** Arrival times vs windows (color-coded: early/on-time/late)
3. **Prediction Accuracy:** Scatter plots showing prediction quality

**Notes:** Let the visuals speak. Point out interesting patterns or insights.

---

## Slide 13: Key Insights & Lessons Learned (1 minute)
**Title:** Main Takeaways

**Key Insights:**

1. **Prediction Matters:** ML reduces costs by [X]% vs naive approach
   - But optimization is robust to moderate prediction errors

2. **Soft Constraints Are Powerful:** Time window flexibility enables feasibility
   - Economic penalties naturally balance service vs cost

3. **Uncertainty Management:** Prediction intervals crucial for robust planning
   - Conservative approach prevents failures but increases vehicle costs

4. **Integration Challenges:** Prediction and optimization must work together
   - Could improve with end-to-end learning

**Notes:** Synthesize the main learnings. What would you tell a practitioner?

---

## Slide 14: Future Work & Extensions (45 seconds)
**Title:** Future Improvements

**Short-term:**
- 🌐 Integrate real traffic APIs
- 📊 Test on real delivery data
- 🔄 Online learning for continuous improvement

**Long-term:**
- 🧠 Deep learning (LSTM for time-series demand)
- 🎯 Stochastic optimization (scenario-based)
- ⚡ Real-time dynamic routing
- 🏢 Multi-depot, heterogeneous fleet
- 🎓 End-to-end learning (optimize predictions for decisions, not accuracy)

**Research Directions:**
- Robust optimization under distributional uncertainty
- Multi-objective optimization (cost vs. sustainability vs. service)

**Notes:** Show you're thinking beyond the project. What's the next level?

---

## Slide 15: Conclusion (30 seconds)
**Title:** Conclusion

**Summary:**
- ✅ Built complete predict-then-optimize pipeline for urban delivery
- ✅ Achieved [X]% cost improvement over baseline
- ✅ Demonstrated value of ML + optimization integration
- ✅ Identified key trade-offs and future research directions

**Impact:**
Decision-support tool ready for real-world testing in logistics startups

**Thank You!**

Questions?

**Notes:** End strong. Recap main achievement. Be ready for questions!

---

## Presentation Tips

### Timing Breakdown:
- Introduction & Context: 2.5 min
- Methodology: 3 min
- Results: 5 min
- Analysis & Discussion: 3 min
- Conclusion: 1 min
- **Total: 14.5 min** (30s buffer)

### Delivery Tips:
1. **Practice:** Rehearse to stay within time
2. **Visuals:** Let charts and diagrams do the talking
3. **Storytelling:** Frame as a journey from problem → solution → insights
4. **Engagement:** Make eye contact, vary pace
5. **Backup slides:** Prepare extra slides for Q&A (technical details, more results)

### Common Questions to Prepare For:
- "Why Random Forest instead of [other model]?"
- "How would this work with real data?"
- "What if predictions are very wrong?"
- "Computational complexity - can it scale?"
- "How does this compare to industry solutions?"

### Visual Guidelines:
- Use consistent color scheme
- Large fonts (≥24pt for body text)
- Minimize text, maximize visuals
- Highlight key numbers in **bold**
- Use animations sparingly

---

**Good luck with your presentation!**
