import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from optimizer import DeliveryOptimizer  # ton script principal avec l'optimizer
import os

st.set_page_config(page_title="🚚 Last-Mile Delivery Optimizer", layout="wide")
st.title("🚚 Last-Mile Delivery Optimizer")
st.write("Optimisation des livraisons last-mile avec prédictions et OR-Tools.")

# -------------------------------
# SIDEBAR - paramètres utilisateurs
# -------------------------------
st.sidebar.header("🔧 Paramètres utilisateurs")

NUM_VEHICLES = st.sidebar.number_input("Nombre de véhicules", min_value=1, max_value=10, value=3)
VEHICLE_CAPACITY = st.sidebar.number_input("Capacité véhicule (kg)", min_value=10, max_value=500, value=100)
WORK_DAY_START = st.sidebar.slider("Début journée (heure)", 0, 24, 8)
WORK_DAY_END = st.sidebar.slider("Fin journée (heure)", 0, 24, 18)
SERVICE_TIME = st.sidebar.number_input("Temps service (h)", min_value=0.05, max_value=1.0, value=0.1, step=0.05)
COST_PER_KM = st.sidebar.number_input("Coût par km (€)", min_value=0.0, max_value=5.0, value=0.5)
EARLY_PENALTY = st.sidebar.number_input("Pénalité arrivée tôt (€)", min_value=0, max_value=100, value=10)
LATE_PENALTY = st.sidebar.number_input("Pénalité retard (€)", min_value=0, max_value=100, value=20)

start_button = st.sidebar.button("Lancer Predict-Then-Optimize")

# -------------------------------
# Charger scénario test
# -------------------------------
@st.cache_data
def load_scenario(scenario_id=0):
    df = pd.read_csv("data/test_scenarios.csv")
    return df[df['scenario_id'] == scenario_id].copy()

scenario_df = load_scenario()

st.subheader("📋 Aperçu du scénario")
st.dataframe(scenario_df.head(10))

# -------------------------------
# LANCEMENT DU PIPELINE
# -------------------------------
if start_button:
    st.info("🔄 Lancement du Predict-Then-Optimize...")

    # Mise à jour des paramètres dans le config global
    import config
    config.NUM_VEHICLES = NUM_VEHICLES
    config.VEHICLE_CAPACITY = VEHICLE_CAPACITY
    config.WORK_DAY_START = WORK_DAY_START
    config.WORK_DAY_END = WORK_DAY_END
    config.SERVICE_TIME = SERVICE_TIME
    config.COST_PER_KM = COST_PER_KM
    config.EARLY_PENALTY = EARLY_PENALTY
    config.LATE_PENALTY = LATE_PENALTY

    optimizer = DeliveryOptimizer(use_predictions=True)
    routes = optimizer.solve(scenario_df, time_limit=300)

    if routes:
        st.success("✅ Solution trouvée !")
        costs = optimizer.calculate_actual_costs(routes, scenario_df)

        # -------------------------------
        # Affichage résumé coûts
        # -------------------------------
        st.subheader("💰 Résumé des coûts")
        st.write(f"**Distance totale:** {routes['total_distance']:.2f} km")
        st.write(f"**Temps total:** {routes['total_time']:.2f} h")
        st.write(f"**Charge totale:** {routes['total_load']:.2f} kg")
        st.write(f"**Coût de déplacement:** €{costs['travel_cost']:.2f}")
        st.write(f"**Coût véhicules:** €{costs['vehicle_cost']:.2f}")
        st.write(f"**Pénalités:** €{costs['penalty_cost']:.2f}")
        st.write(f"**Coût total:** €{costs['total_cost']:.2f}")

        # -------------------------------
        # Affichage des routes
        # -------------------------------
        st.subheader("🛣️ Routes par véhicule")
        for route in routes['routes']:
            customer_ids = [stop.get('customer_id', 'DEPOT') for stop in route['stops'] if stop['node'] != 0]
            st.write(f"**Véhicule {route['vehicle_id']}:** DEPOT → " + " → ".join(map(str, customer_ids)) + " → DEPOT")
            st.write(f"Distance: {route['total_distance']:.2f} km | Charge: {route['total_load']:.2f} kg")

        # -------------------------------
        # Graphique comparatif des coûts
        # -------------------------------
        st.subheader("📊 Visualisation des coûts")
        fig, ax = plt.subplots()
        ax.bar(['Travel', 'Vehicle', 'Penalty'], 
               [costs['travel_cost'], costs['vehicle_cost'], costs['penalty_cost']], color=['blue', 'green', 'red'])
        ax.set_ylabel("Coût (€)")
        ax.set_title("Répartition des coûts")
        st.pyplot(fig)

        # -------------------------------
        # Sauvegarde solution
        # -------------------------------
        save_path = "results/dashboard_solution.json"
        optimizer.save_solution(filepath=save_path)
        st.info(f"💾 Solution sauvegardée dans `{save_path}`")
