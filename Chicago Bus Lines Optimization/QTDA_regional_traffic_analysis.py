import pandas as pd
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyEigensolver
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components

def identify_isolated_outliers(filename):
    print(f"--- Running Isolation & Outlier Analysis: {filename} ---")
    
    # 1. DATA CLEANING
    df = pd.read_csv(filename)
    for col in ['NUMBER OF READS', 'BUS COUNT', 'SPEED']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df = df.dropna(subset=['NUMBER OF READS', 'BUS COUNT', 'SPEED'])

    # 2. FEATURE-SPACE AGGREGATION
    regional_profiles = df.groupby('REGION_ID').agg({
        'SPEED': 'mean',
        'BUS COUNT': 'mean',
        'NUMBER OF READS': 'mean'
    }).reset_index()

    # 3. CONGESTION PENALTY
    regional_profiles['CONGESTION_PENALTY'] = (
        regional_profiles['NUMBER OF READS'] / (regional_profiles['BUS COUNT'] + 1)
    )

    # 4. CONSTRUCT SIMPLICIAL COMPLEX
    scaler = StandardScaler()
    features = scaler.fit_transform(regional_profiles[['SPEED', 'BUS COUNT', 'NUMBER OF READS']])
    
    num_regions = len(features)
    # LEVER: Lower Epsilon (0.70) makes it harder to connect, highlighting the true outliers
    epsilon = 0.95 
    
    adj_matrix = np.zeros((num_regions, num_regions))
    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            dist = np.linalg.norm(features[i] - features[j])
            if dist < epsilon:
                adj_matrix[i, j] = 1.0

    # 5. ISOLATION LOGIC
    # Find which regions belong to which island (Betti-0 Component Mapping)
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    regional_profiles['BETTI_0_COMPONENT'] = labels
    
    # Calculate how many regions are in each component
    component_counts = regional_profiles['BETTI_0_COMPONENT'].value_counts().to_dict()
    regional_profiles['COMPONENT_SIZE'] = regional_profiles['BETTI_0_COMPONENT'].map(component_counts)

    # 6. QUANTUM VERIFICATION (Calculating the Global Betti-0)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix
    dim = 2**int(np.ceil(np.log2(num_regions)))
    padded_laplacian = np.eye(dim)
    padded_laplacian[:num_regions, :num_regions] = laplacian
    qubit_op = SparsePauliOp.from_operator(padded_laplacian)
    solver = NumPyEigensolver(k=num_regions) 
    result = solver.compute_eigenvalues(qubit_op)
    quantum_betti_0 = sum(1 for ev in result.eigenvalues if abs(ev) < 1e-5)

    # 7. FINAL RANKING
    # Rank first by isolation, second by slowest speed
    results = regional_profiles.sort_values(by=['COMPONENT_SIZE', 'SPEED'], ascending=[True, True])
    
    # Clean output
    print(f"\nTotal Unique Traffic Profiles (Betti-0): {quantum_betti_0}")
    print("\n--- THE MOST ISOLATED OUTLIERS (Ranked by Connectivity Failure & Speed) ---")
    print(results[['REGION_ID', 'COMPONENT_SIZE', 'CONGESTION_PENALTY', 'SPEED', 'BUS COUNT', 'BETTI_0_COMPONENT']].head(10))
    
    return results

if __name__ == "__main__":
    final_analysis = identify_isolated_outliers('QTDA_traffic_data.csv')