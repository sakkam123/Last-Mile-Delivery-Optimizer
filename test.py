from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math

# =====================================================
# 1️⃣ DONNÉES DU PROBLÈME
# =====================================================
print("\n1️⃣ Création des données du problème")

data = {}
data["locations"] = [
    (0, 0),  # Dépôt
    (2, 3),  # Client 1
    (5, 1),  # Client 2
    (6, 4),  # Client 3
]

data["demands"] = [0, 10, 15, 20]          # kg
data["vehicle_capacities"] = [40, 40]      # 2 véhicules
data["num_vehicles"] = 2
data["depot"] = 0

# Fenêtres horaires (en "temps abstrait")
data["time_windows"] = [
    (0, 1000),  # Dépôt
    (0, 1000),  # Client 1
    (0, 1000),  # Client 2
    (0, 1000),  # Client 3
]

print("✔ Locations :", data["locations"])
print("✔ Demands   :", data["demands"])
print("✔ Capacités :", data["vehicle_capacities"])
print("✔ Time windows :", data["time_windows"])

# =====================================================
# 2️⃣ DISTANCE / TEMPS
# =====================================================
print("\n2️⃣ Calcul des distances")

def distance(i, j):
    x1, y1 = data["locations"][i]
    x2, y2 = data["locations"][j]
    return math.hypot(x2 - x1, y2 - y1)

# =====================================================
# 3️⃣ CRÉATION DU ROUTING MODEL
# =====================================================
print("\n3️⃣ Initialisation du Routing Model")

manager = pywrapcp.RoutingIndexManager(
    len(data["locations"]),
    data["num_vehicles"],
    data["depot"]
)

routing = pywrapcp.RoutingModel(manager)

# =====================================================
# 4️⃣ COÛT = DISTANCE
# =====================================================
print("\n4️⃣ Définition du coût (distance)")

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(distance(from_node, to_node) * 100)  # entier obligatoire

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# =====================================================
# 5️⃣ CONTRAINTE DE CAPACITÉ
# =====================================================
print("\n5️⃣ Ajout de la contrainte de capacité")

def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return data["demands"][from_node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  # pas de slack
    data["vehicle_capacities"],
    True,  # start cumul = 0
    "Capacity"
)

# =====================================================
# 6️⃣ CONTRAINTE DE TEMPS (CORRIGÉE ✅)
# =====================================================
print("\n6️⃣ Ajout des fenêtres horaires")

routing.AddDimension(
    transit_callback_index,
    100,    # attente max
    1000,   # temps max
    True,   # start cumul = 0  ⚠️ IMPORTANT
    "Time"
)

time_dimension = routing.GetDimensionOrDie("Time")

# Clients
for node in range(1, len(data["locations"])):
    index = manager.NodeToIndex(node)
    time_dimension.CumulVar(index).SetRange(
        data["time_windows"][node][0],
        data["time_windows"][node][1]
    )

# START / END de chaque véhicule (OBLIGATOIRE)
for vehicle_id in range(data["num_vehicles"]):
    time_dimension.CumulVar(routing.Start(vehicle_id)).SetRange(0, 1000)
    time_dimension.CumulVar(routing.End(vehicle_id)).SetRange(0, 1000)

# =====================================================
# 7️⃣ PARAMÈTRES DU SOLVEUR
# =====================================================
print("\n7️⃣ Paramètres du solveur")

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.time_limit.seconds = 10

# =====================================================
# 8️⃣ RÉSOLUTION
# =====================================================
print("\n8️⃣ Résolution du problème...")
solution = routing.SolveWithParameters(search_parameters)

# =====================================================
# 9️⃣ AFFICHAGE DES ROUTES
# =====================================================
print("\n9️⃣ Résultat final")

if solution:
    print("✅ Solution trouvée\n")

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = f"🚚 Véhicule {vehicle_id} : "
        load = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            load += data["demands"][node]
            route += f"{node} → "
            index = solution.Value(routing.NextVar(index))

        route += "DEPOT"
        print(route)
        print(f"   Charge transportée : {load} kg\n")
else:
    print("❌ Aucune solution trouvée")
