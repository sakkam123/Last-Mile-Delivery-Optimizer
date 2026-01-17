# Quick Start Guide - Last-Mile Delivery Optimizer

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```powershell
cd "c:\Users\MSI\OneDrive\Bureau\ENSI\Optimisation Combinatoire\last_mile_delivery"
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline
```powershell
python main.py
```

This will:
1. ✅ Generate synthetic historical data (100 days)
2. ✅ Train ML prediction models
3. ✅ Run optimization on test scenarios
4. ✅ Generate visualizations
5. ✅ Compare different approaches

**Expected runtime:** 2-5 minutes depending on your computer

---

## 📁 Project Structure

```
last_mile_delivery/
│
├── 📄 config.py                    # Configuration parameters
├── 📄 data_generator.py            # Generate historical data
├── 📄 predictor.py                 # ML prediction models
├── 📄 optimizer.py                 # OR-Tools route optimization
├── 📄 pipeline.py                  # Predict-then-optimize integration
├── 📄 visualize.py                 # Visualization tools
├── 📄 main.py                      # Main execution script
│
├── 📄 README.md                    # Comprehensive documentation
├── 📄 REPORT_TEMPLATE.md           # Report writing guide
├── 📄 PRESENTATION_OUTLINE.md      # Presentation guide
├── 📄 QUICK_START.md              # This file
├── 📄 requirements.txt             # Python dependencies
│
├── 📂 data/                        # Generated data (auto-created)
│   ├── historical_data.csv
│   ├── test_scenarios.csv
│   └── models/                     # Trained ML models
│       ├── demand_predictor.pkl
│       └── travel_time_predictor.pkl
│
└── 📂 results/                     # Outputs (auto-created)
    ├── solution.json
    ├── pipeline_results.json
    └── visualizations/             # PNG plots
        ├── routes.png
        ├── time_windows.png
        ├── prediction_accuracy.png
        ├── cost_comparison.png
        └── vehicle_utilization.png
```

---

## 🎯 Running Individual Components

### Generate Data Only
```powershell
python data_generator.py
```
Creates `data/historical_data.csv` and `data/test_scenarios.csv`

### Train Models Only
```powershell
python predictor.py
```
Trains and saves models to `data/models/`

### Run Optimizer Only
```powershell
python optimizer.py
```
Optimizes routes for scenario 0, saves to `results/solution.json`

### Run Full Pipeline
```powershell
python pipeline.py
```
Complete predict-then-optimize with comparative analysis

### Generate Visualizations Only
```powershell
python visualize.py
```
Creates plots in `results/visualizations/`

---

## ⚙️ Customization

### Change Parameters
Edit [config.py](config.py):

```python
# Vehicle configuration
NUM_VEHICLES = 3              # Increase for more vehicles
VEHICLE_CAPACITY = 100        # kg per vehicle

# Cost parameters
LATE_PENALTY = 20            # €/hour for late delivery
EARLY_PENALTY = 10           # €/hour for early delivery

# Data generation
NUM_HISTORICAL_DAYS = 100     # More data = better predictions
NUM_CUSTOMERS_MIN = 15        # Min customers per day
NUM_CUSTOMERS_MAX = 25        # Max customers per day
```

### Test Different Scenarios
```python
# In pipeline.py or main.py, change:
results = pipeline.compare_approaches(test_scenarios, scenario_id=0)
# to
results = pipeline.compare_approaches(test_scenarios, scenario_id=5)
```

---

## 📊 Understanding the Results

### Key Output Files

1. **data/historical_data.csv**
   - Training data with actual demands and travel times
   - Used to train ML models

2. **data/test_scenarios.csv**
   - 10 test scenarios with ground truth
   - Used to evaluate performance

3. **results/solution.json**
   - Optimized routes with stop-by-stop details
   - Distance, time, load for each vehicle

4. **results/visualizations/*.png**
   - Route maps showing optimized paths
   - Time window compliance charts
   - Prediction accuracy plots
   - Cost comparisons

### Interpreting Costs

**Total Cost = Travel Cost + Vehicle Cost + Penalty Cost**

- **Travel Cost:** Distance × €0.50/km
- **Vehicle Cost:** €50 per vehicle used
- **Penalty Cost:** Hours early/late × penalty rate

**Lower is better!** Compare your approach vs oracle and baseline.

---

## 🔍 What to Analyze for Your Report

### 1. Prediction Quality
- How accurate are demand predictions?
- How accurate are travel time predictions?
- Does accuracy vary by customer type or time of day?

### 2. Optimization Quality
- How close to oracle (perfect info) are you?
- Is the gap acceptable?
- What causes the gap?

### 3. Trade-offs
- Using upper bounds (conservative) vs point estimates (risky)
- Short optimization time vs solution quality
- Service quality (time windows) vs cost

### 4. Sensitivity
Try changing:
- Penalty costs (what if late penalty is €50 instead of €20?)
- Vehicle capacity (what if capacity is 80kg instead of 100kg?)
- Number of vehicles (what if only 2 vehicles available?)

### 5. Comparative Analysis
- Why is predict-optimize better than baseline?
- What would it take to close gap to oracle?
- Is ML prediction worth the complexity?

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'ortools'"
**Solution:** Install dependencies
```powershell
pip install -r requirements.txt
```

### Problem: "No solution found!" in optimizer
**Possible causes:**
1. Too few vehicles for the demand
2. Time windows too tight
3. Vehicle capacity too small

**Solution:** Adjust parameters in config.py

### Problem: Visualizations don't show
**Solution:** Make sure matplotlib backend is configured:
```powershell
python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt"
```

### Problem: Slow execution
- Reduce `NUM_HISTORICAL_DAYS` in config.py
- Reduce time limit in optimizer (trade-off: solution quality)
- Use fewer test scenarios

---

## 📝 Next Steps for Your Project

### 1. Run the Code ✅
```powershell
python main.py
```

### 2. Analyze Results 📊
- Review visualizations in `results/visualizations/`
- Study cost comparisons
- Understand prediction accuracy

### 3. Write Report 📄
- Use `REPORT_TEMPLATE.md` as guide
- Fill in your actual results
- Add critical analysis

### 4. Prepare Presentation 🎤
- Follow `PRESENTATION_OUTLINE.md`
- Create slides (15 minutes)
- Practice delivery

### 5. Experiment 🧪
- Try different parameters
- Test sensitivity
- Compare variations

---

## 📚 Additional Resources

### Understanding the Algorithms

**Random Forest:**
- Ensemble of decision trees
- Robust to outliers
- Provides uncertainty estimates

**OR-Tools CP-SAT:**
- Constraint Programming solver
- Good for routing problems
- Guided Local Search metaheuristic

**Predict-Then-Optimize:**
- Two-stage approach: predict → optimize
- Alternative: end-to-end learning
- Trade-off: simplicity vs optimality

### Relevant Papers
1. Elmachtoub & Grigas (2022) - Smart Predict-Then-Optimize
2. Toth & Vigo (2014) - Vehicle Routing
3. Bertsimas & Sim (2004) - Price of Robustness

### Documentation
- OR-Tools: https://developers.google.com/optimization
- scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/

---

## 💡 Pro Tips

1. **Run multiple scenarios:** Don't just analyze scenario 0
2. **Visualize everything:** Pictures tell better stories than tables
3. **Compare approaches:** Show why your method is better
4. **Be critical:** Discuss limitations honestly
5. **Think practical:** What would a real company want?

---

## ❓ FAQ

**Q: Can I use real data instead of synthetic?**
A: Yes! Modify `data_generator.py` or load your CSV with required columns.

**Q: How do I add more vehicles?**
A: Change `NUM_VEHICLES` in `config.py`

**Q: Can I use different ML models?**
A: Yes! Edit `predictor.py` - try GradientBoosting, XGBoost, or neural networks

**Q: How do I make time windows stricter?**
A: Increase `LATE_PENALTY` and `EARLY_PENALTY` in `config.py`

**Q: Can this scale to 100+ customers?**
A: Yes, but increase optimizer time limit and consider metaheuristics

---

## 🎓 Learning Objectives Covered

✅ **Combinatorial Optimization:** VRP with constraints  
✅ **Machine Learning:** Prediction with uncertainty  
✅ **Integration:** Connecting prediction and optimization  
✅ **Evaluation:** Comparing approaches scientifically  
✅ **Critical Thinking:** Analyzing trade-offs and limitations  

---

**Good luck with your project! 🚀**

For questions or issues, review the main README.md or consult course materials.
