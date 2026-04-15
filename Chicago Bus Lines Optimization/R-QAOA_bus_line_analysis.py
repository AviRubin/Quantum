import pandas as pd
import numpy as np
import re
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize

# 1. SETTINGS
# Hardcoded cordinates for Region 12
WEST, EAST = -87.647208, -87.62308
SOUTH, NORTH = 41.88886, 41.911401

def run_high_accuracy_city_optimizer(filename):
    # --- WHAT-IF SCENARIO SETTINGS ---
    WHAT_IF_ENABLED = True      # 'False' runs the normal baseline, 'True" freezes a target_route
    TARGET_ROUTE = '146'        # The route to "freeze" in the recursive part of R-QAOA
    IMPROVEMENT_FACTOR = 0.7    # 0.7 means a 30% reduction in congestion weight
    # ---------------------------------

    # Load data
    df = pd.read_csv(filename)
    route_data = []
    coord_pattern = re.compile(r'(-?\d+\.\d+)\s+(\d+\.\d+)')
    
    for _, row in df.iterrows():
        coords = coord_pattern.findall(str(row['the_geom']))
        points = sum(1 for lon, lat in coords if WEST <= float(lon) <= EAST and SOUTH <= float(lat) <= NORTH)
        
        if points > 0:
            # Start with the raw weight (points)
            final_weight = points
            
            # If What-If is active and this is our target route, apply the reduction
            if WHAT_IF_ENABLED and str(row['ROUTE']) == TARGET_ROUTE:
                final_weight = points * IMPROVEMENT_FACTOR
                print(f"--- [WHAT-IF ACTIVE] --- Reducing Route {TARGET_ROUTE} weight to {final_weight:.2f}")

            route_data.append({
                'ROUTE': row['ROUTE'], 
                'NAME': row['NAME'], 
                'WEIGHT': final_weight
            })
    
    # Process the top 8 candidates based on the (potentially modified) weights
    candidates = pd.DataFrame(route_data).sort_values('WEIGHT', ascending=False).head(8)
    
    active_routes = list(candidates['ROUTE'].values)
    active_names = list(candidates['NAME'].values)
    active_weights = list(candidates['WEIGHT'].values)
    
    final_selections = []
    k_target = 2 # Number of bus routes to identify
    
    # ACCURACY LEVERS
    REPS = 4           # Number of layers in circuit (the Ansatz). Increase depth for better resolution
    MAX_ITER = 150     # Number of optimizer guesses per trial, must be high enough to converge
    N_RESTARTS = 5     # Number of times to run the total process. Multiple starts of random guesses helps avoid local minima
    
    print(f"--- STARTING BUS ROUTE AUDIT ---")

    while len(final_selections) < k_target:
        num_qubits = len(active_routes)
        print(f"\n[Step {len(final_selections)+1}/{k_target}]: Solving for best remaining bottleneck...")
        
        # Define cost
        def get_cost(x, w, k_needed):
            # WEIGHT SHARPENING: square the weights to make the 'gap' between routes larger
            selection_sum = sum((w[i]**2) * x[i] for i in range(len(x)))
            penalty = 1000 * (sum(x) - k_needed)**2
            return -(selection_sum - penalty)

        sampler = StatevectorSampler()
        cost_op = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
        ansatz = QAOAAnsatz(cost_op, reps=REPS)
        ansatz.measure_all()

        best_overall_res = None
        best_overall_cost = float('inf')

        # MULTI-START LOOP
        for start_node in range(N_RESTARTS):
            print(f"  > Trial {start_node+1}/{N_RESTARTS}...")
            
            def objective(params):
                pub = (ansatz, [params])
                counts = sampler.run([pub]).result()[0].data.meas.get_counts()
                shots = sum(counts.values())
                k_needed = k_target - len(final_selections)
                return sum(get_cost([int(bit) for bit in bstr[::-1]], active_weights, k_target-len(final_selections)) * (count/shots) for bstr, count in counts.items())

            res = minimize(objective, np.random.rand(ansatz.num_parameters), method='COBYLA', options={'maxiter': MAX_ITER})
            
            if res.fun < best_overall_cost:
                best_overall_cost = res.fun
                best_overall_res = res

        # EXTRACT BIAS FROM THE BEST TRIAL
        final_counts = sampler.run([(ansatz, [best_overall_res.x])]).result()[0].data.meas.get_counts()
        total_shots = sum(final_counts.values())
        biases = np.zeros(num_qubits)
        for bstr, count in final_counts.items():
            for i, bit in enumerate(bstr[::-1]):
                if bit == '1': biases[i] += (count / total_shots)

        best_idx = np.argmax(biases)
        
        # CITY COMPLIANCE CHECK
        confidence = biases[best_idx]

        final_selections.append({'ROUTE': active_routes[best_idx], 'NAME': active_names[best_idx], 'CONFIDENCE': confidence})
        
        # Remove winner for the next recursion
        active_routes.pop(best_idx)
        active_names.pop(best_idx)
        active_weights.pop(best_idx)

    print(f"\n--- AUDIT COMPLETE ---")
    for r in final_selections:
        print(f"Priority Route: {r['ROUTE']} | Confidence: {r['CONFIDENCE']:.2%}")

if __name__ == "__main__":
    run_high_accuracy_city_optimizer('QAOA_bus_routes.csv')