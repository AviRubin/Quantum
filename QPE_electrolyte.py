# Find the binding energy required for Li to strip away its EC solvent shell

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider 
import os

# AUTHENTICATION & SETUP
token = os.getenv('IONQ_API_KEY')
provider = IonQProvider(token)
backend = provider.get_backend("simulator")
print(f"Backend Name: {backend.name}")
print(f"Max Qubits: {backend.num_qubits}")

# Define qubit counts
n_precision = 12  # bits of precision (~10 is scientific standard)
n_system = 1      # Simplified 'Active Space' for the Li-Oxygen bond
total_qubits = n_precision + n_system

qc = QuantumCircuit(total_qubits, n_precision)

# STATE PREPARATION: The "Bonded" State
# Initialize the system qubit to |1>, representing the Li+
# being chemically bonded to the Oxygen in the solvent
qc.x(n_precision) 

# INITIAL SUPERPOSITION
# Put all precision qubits into superposition to start the "clock"
qc.h(range(n_precision))

# CONTROLLED UNITARY (Desolvation Physics)
# 'target_phase' is the mathematical representation of the binding energy.
# In a real research scenario, this angle would be calculated using a
# molecular Hamiltonian. Here, 0.314 represents a standard Li-EC binding energy profile.
target_phase = 0.314

for i in range(n_precision):
    # Number of times to apply the interaction for this bit of precision
    repetitions = 2**i
    angle = 2 * np.pi * target_phase * repetitions
    
    for _ in range(repetitions):
        # Decomposed CP gate within scope of free plan
        qc.rz(angle/2, n_precision)
        qc.cx(i, n_precision)
        qc.rz(-angle/2, n_precision)
        qc.cx(i, n_precision)

qc.barrier()

# INVERSE QFT (The Decoding Lens)
# Decomposed CP gate within scope of free plan
def native_inverse_qft(circuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            lam = -np.pi / float(2**(j - m))
            # Decomposed CP gate for the QFT
            circuit.rz(lam/2, j)
            circuit.cx(m, j)
            circuit.rz(-lam/2, j)
            circuit.cx(m, j)
        circuit.h(j)

native_inverse_qft(qc, n_precision)

# MEASUREMENT
qc.measure(range(n_precision), range(n_precision))

# EXECUTION ON IONQ CLOUD
# Transpile specifically for IonQ native gate set
transpiled_qc = transpile(qc, backend, optimization_level=1)
job = backend.run(transpiled_qc, shots=1024)

print(f"Job submitted to IonQ Cloud. Job ID: {job.job_id()}")

# Retrieve results:
result = job.result()
counts = result.get_counts()
# print(counts)

# RESULT PRESENTATION
print("\n" + "="*60)
print("       CONSOLIDATED ELECTROLYTE ANALYSIS")
print("="*60)

total_shots = sum(counts.values())
phase_map = {}

# Map every result to its "Primary" Phase (0.0 to 0.5)
# If a phase is > 0.5, treat it as its symmetrical counterpart
for bitstring, count in counts.items():
    measured_int = int(bitstring, 2)
    phase = measured_int / (2**n_precision)

# Consolidate symmetry: if phase is 0.89, it maps back to 0.39 (0.89 - 0.5)
    normalized_phase = round(phase % 0.5, 4) 
    
    if normalized_phase not in phase_map:
        phase_map[normalized_phase] = 0
    phase_map[normalized_phase] += count

# Sort consolidated eesults
sorted_phases = sorted(phase_map.items(), key=lambda item: item[1], reverse=True)

# Print table
print(f"{'Normalized Phase':<20} | {'Total Counts':<12} | {'Combined Confidence'}")
print("-" * 60)

for i in range(min(5, len(sorted_phases))):
    phase, count = sorted_phases[i]
    confidence = (count / total_shots) * 100
    print(f"{phase:<20.4f} | {count:<12} | {confidence:>18.2f}%")

# 5. Final conclusion
top_phase = sorted_phases[0][0]
print("-" * 60)
print(f"CONSOLIDATED CHEMICAL SIGNATURE: {top_phase:.4f}")
print(f"TOTAL SYSTEM CONFIDENCE:        { (sorted_phases[0][1]/total_shots)*100 :.2f}%")
print(f"ESTIMATED BINDING ENERGY:       {top_phase * 2.0:.4f} eV")
print("="*60)