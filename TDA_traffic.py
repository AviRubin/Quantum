# ----- Goal is to identify regions whose speed underperforms other regions with similar statistics -----
# .csv data from below link. City of Chicago traffic congestion from April 1, 2017 - April 15, 2017
# https://data.cityofchicago.org/Transportation/Chicago-Traffic-Tracker-Historical-Congestion-Esti/emtn-qqdi/explore/

import pandas as pd
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Load and Clean Data
filename = 'TDA_traffic_data.csv' 
try:
    df_raw = pd.read_csv(filename)
    df_raw.columns = df_raw.columns.str.strip().str.upper()
    
    # Identify columns
    bus_col = [c for c in df_raw.columns if 'BUS' in c][0]
    reads_col = [c for c in df_raw.columns if 'READ' in c][0]
    speed_col = [c for c in df_raw.columns if 'SPEED' in c][0]
    region_col = [c for c in df_raw.columns if 'REGION' in c][0]

    # Clean numeric formatting
    for col in [bus_col, reads_col, speed_col]:
        df_raw[col] = df_raw[col].astype(str).str.replace(',', '')
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    df = df_raw.dropna(subset=[bus_col, reads_col, speed_col]).reset_index(drop=True)
    total_rows = len(df)
    print(f"Dataset cleaned. Total rows: {total_rows}")

except Exception as e:
    print(f"Initialization Error: {e}")
    exit()

# Global Scaling
features = df[[bus_col, reads_col, speed_col]].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# QTDA Loop due to local memory limitations
chunk_size = 1024 
topological_scores = np.zeros(total_rows)

print(f"Starting analysis in chunks of {chunk_size}...")
start_time = time.time()

for i in range(0, total_rows, chunk_size):
    # Define chunk boundaries
    chunk_end = min(i + chunk_size, total_rows)
    chunk_features = features_scaled[i:chunk_end]
    n_nodes = len(chunk_features) # This replaces the 'actual_chunk_size' variable
    
    # Calculate Topology (Laplacian)
    dist_matrix = np.linalg.norm(chunk_features[:, None] - chunk_features, axis=2)
    epsilon = np.percentile(dist_matrix, 5)
    adj = (dist_matrix < epsilon).astype(float)
    np.fill_diagonal(adj, 0)
    L = np.diag(adj.sum(axis=1)) - adj
    
    # Determine Hilbert Space Size (power of 2)
    num_qubits = int(np.ceil(np.log2(n_nodes)))
    if num_qubits == 0: num_qubits = 1
    dim = 2**num_qubits
    
    # Normalize & Pad Matrix
    max_eig = np.max(np.linalg.eigvalsh(L))
    L_norm = L / (max_eig + 1e-6)
    L_padded = np.zeros((dim, dim))
    L_padded[:n_nodes, :n_nodes] = L_norm
    
    # Quantum Evolution simulates how a quantum wave propagates through the traffic data 'shape'
    # U = exp(-i * L * t)
    U_matrix = expm(-1j * L_padded * 5.0)
    
    # Initial state |00...0>
    initial_state = Statevector.from_int(0, dim)
    
    # Evolve and extract probabilities
    final_state = initial_state.evolve(U_matrix).data
    probs = np.abs(final_state[:n_nodes])**2
    topological_scores[i:chunk_end] = probs
    
    if (i // chunk_size) % 10 == 0:
        print(f"Progress: {(chunk_end/total_rows)*100:.1f}%")

# Results and Visualization
df['TOPOLOGICAL_SCORE'] = topological_scores
bottlenecks = df.sort_values(by='TOPOLOGICAL_SCORE').head(15)

print(f"\nAnalysis complete in {time.time() - start_time:.2f}s")
print("\n--- PRIORITY REGIONS FOR SPEED IMPROVEMENT ---")
print(bottlenecks[[region_col, speed_col, bus_col, 'TOPOLOGICAL_SCORE']])

plt.figure(figsize=(12, 6))
plt.scatter(df[bus_col], df[speed_col], c=df['TOPOLOGICAL_SCORE'], cmap='plasma_r', alpha=0.1, s=5)
plt.scatter(bottlenecks[bus_col], bottlenecks[speed_col], color='cyan', marker='x', s=100, label='Top Bottlenecks')
plt.colorbar(label='Topological Score (Lower = More Isolated)')
plt.xlabel('Bus Volume')
plt.ylabel('Speed')
plt.title('City-Wide Traffic Topology Analysis')
plt.legend()
plt.show()