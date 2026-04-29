import pandas as pd
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyEigensolver
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_quantum_betti(laplacian):
    if laplacian is None or laplacian.size == 0:
        return 0
    if not np.any(laplacian):
        return laplacian.shape[0]

    num_nodes = laplacian.shape[0]
    dim = 2**int(np.ceil(np.log2(num_nodes)))
    padded = np.eye(dim)
    padded[:num_nodes, :num_nodes] = laplacian
    
    try:
        qubit_op = SparsePauliOp.from_operator(padded)
        if len(qubit_op.coeffs) == 0:
            return num_nodes
            
        solver = NumPyEigensolver(k=num_nodes)
        result = solver.compute_eigenvalues(qubit_op)
        return sum(1 for ev in result.eigenvalues if abs(ev) < 1e-5)
    except Exception:
        return num_nodes - np.linalg.matrix_rank(laplacian)

def build_boundary_operators(num_regions, adj_matrix):
    edges = [tuple(e) for e in np.argwhere(np.triu(adj_matrix) > 0)]
    num_edges = len(edges)
    if num_edges == 0: 
        return None, None, 0

    triangles = []
    for combo in combinations(range(num_regions), 3):
        if adj_matrix[combo[0], combo[1]] and adj_matrix[combo[1], combo[2]] and adj_matrix[combo[0], combo[2]]:
            triangles.append(combo)
    
    d1 = np.zeros((num_regions, num_edges))
    for j, (u, v) in enumerate(edges):
        d1[u, j], d1[v, j] = -1, 1
    
    num_triangles = len(triangles)
    d2 = np.zeros((num_edges, num_triangles)) if num_triangles > 0 else np.zeros((num_edges, 1))
    for j, (u, v, w) in enumerate(triangles):
        try:
            e1, e2, e3 = edges.index((u, v)), edges.index((v, w)), edges.index((u, w))
            d2[e1, j], d2[e2, j], d2[e3, j] = 1, 1, -1
        except ValueError: continue
    return d1, d2, num_edges

def analyze_traffic_topology(filename, window_minutes=15):
    print(f"--- Loading and Cleaning Data: {filename} ---")
    df = pd.read_csv(filename)
    df.columns = [c.strip() for c in df.columns]

    # Add any region IDs here to ignore. Currenlty ignoring Region 13 because it is downtown and overshadowing any other results
    regions_to_exclude = [13]
    df = df[~df['REGION_ID'].isin(regions_to_exclude)]
    print(f"Excluded Regions: {regions_to_exclude}")

    # Numeric conversion and Velocity Floor
    for col in ['SPEED', 'BUS COUNT', 'NUMBER OF READS']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    
    # Add speed threshold to filter (avoid zeroes / null data)
    df = df[df['SPEED'] >= 0.5]
    df = df.dropna(subset=['SPEED', 'BUS COUNT', 'NUMBER OF READS'])

    # Timestamp Detection
    time_col = next((c for c in ['TIMESTAMP', 'TIME', 'DATETIME', 'date', 'Date', 'Time', 'ts'] if c in df.columns), None)
    if not time_col: return None
    
    df['TS_CLEAN'] = pd.to_datetime(df[time_col])
    df['TIME_WINDOW'] = df['TS_CLEAN'].dt.floor(f'{window_minutes}min')

    # Aggregation
    windowed_data = df.groupby(['TIME_WINDOW', 'REGION_ID']).agg({
        'SPEED': ['mean', 'std'],
        'BUS COUNT': 'mean',
        'NUMBER OF READS': 'mean'
    })
    windowed_data.columns = ['_'.join(col).strip().upper() for col in windowed_data.columns.values]
    windowed_data = windowed_data.reset_index().rename(columns={'BUS COUNT_MEAN': 'BUS_COUNT', 'NUMBER OF READS_MEAN': 'READS'})
    windowed_data['SPEED_STD'] = windowed_data['SPEED_STD'].fillna(0)

    all_results = []

    for window, group in windowed_data.groupby('TIME_WINDOW'):
        if len(group) < 3: continue 
        
        scaler = StandardScaler()
        features = scaler.fit_transform(group[['SPEED_MEAN', 'SPEED_STD', 'BUS_COUNT']])
        num_regions = len(features)
        
        adj_matrix = np.zeros((num_regions, num_regions))
        for i, j in combinations(range(num_regions), 2):
            if np.linalg.norm(features[i] - features[j]) < 0.95:
                adj_matrix[i, j] = adj_matrix[j, i] = 1

        l0 = np.diag(np.sum(adj_matrix, axis=1)) - adj_matrix
        beta_0 = get_quantum_betti(l0)
        
        d1, d2, _ = build_boundary_operators(num_regions, adj_matrix)
        beta_1 = 0
        if d1 is not None:
            l1 = d1.T @ d1 + d2 @ d2.T
            beta_1 = get_quantum_betti(l1)

        for _, row in group.iterrows():
            all_results.append({
                'TIME': window, 'REGION': row['REGION_ID'], 'SPEED': row['SPEED_MEAN'],
                'B1_LOOPS': beta_1, 'CONGESTION': row['READS'] / (row['BUS_COUNT'] + 1)
            })

    report_df = pd.DataFrame(all_results)
    print("\n" + "="*60 + "\nQUBIT-SCALE TOPOLOGICAL TRAFFIC REPORT\n" + "="*60)

    gridlocks = report_df[report_df['B1_LOOPS'] > 0].sort_values(by='SPEED')
    if not gridlocks.empty:
        print(gridlocks[['TIME', 'REGION', 'SPEED', 'B1_LOOPS']].head(15).to_string(index=False))
    else:
        print("No topological loops detected in active traffic (excluding Region 13).")

    return report_df

if __name__ == "__main__":
    final_report = analyze_traffic_topology('QTDA_traffic_data.csv')