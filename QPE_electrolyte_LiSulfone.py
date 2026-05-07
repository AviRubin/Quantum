# Find the binding energy required for Li to strip away a theoretical SO solvent shell

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
import os

# AUTHENTICATION & SETUP
token = os.getenv('IONQ_API_KEY')
provider = IonQProvider(token)
backend = provider.get_backend("simulator")
print(f"Backend Name: {backend.name}")
print(f"Max Qubits: {backend.num_qubits}")

# Define qubit counts
n_precision = 6  # bits of precision
n_system = 4     # active orbitals for Sulfone
total_qubits = n_precision + n_system

qc = QuantumCircuit(total_qubits, n_precision)

# Define explicit qubit lists for mapping
precision_qubits = list(range(n_precision))
target_qubits = list(range(n_precision, total_qubits))

# STATE PREPARATION: The "Bonded" State
# Initialize the system qubit to |1>, representing the Li+ chemically bonded
qc.x(target_qubits[0])

# INITIAL SUPERPOSITION
qc.h(precision_qubits)

# HAMILTONIAN FOR SULFONE BLEND
# Active terms currently using the number of qubits declared above, running on a simulator:
active_terms = [
    ("ZZII", -0.50),  # Primary electrostatic pull of Oxygen 1 on Li+
    ("IIZZ", -0.50),  # Primary electrostatic pull of Oxygen 2 on Li+
    ("IZIZ",  0.12),  # Weak polarization effect of the Sulfur d-orbital
]

# --- Additional block uncomment when scaling to a large physical QPU ---
# These terms model the complex "electron soup" of a real Sulfone molecule.
#showcase_terms = [
    # ("XXII",  0.05),  # Electron hopping between polar Oxygens
    # ("YYII",  0.05),  # Spin-flip exchange term
    # ("ZIZI", -0.15),  # Charge-dipole interaction with Sulfur core
    # ("IZZI",  0.08),  # Neighboring carbon ring induction
    # ("IXIX",  0.02),  # Weak dispersion/van der Waals forces
    # ("XIXI",  0.02),  # Secondary orbital hybridization
    # ("ZZZZ", -0.05),  # Four-body electron correlation term
    # ("YYYY",  0.01),  # Higher-order exchange corrections
#]

# Combine lists
full_terms = active_terms # + showcase_terms
sulfone_hamiltonian = SparsePauliOp.from_list(full_terms)

# CONTROLLED TIME EVOLUTION
for i in range(n_precision):
    repetitions = 2**i
    evolution_gate = PauliEvolutionGate(sulfone_hamiltonian, time=repetitions)
# Apply controlled evolution: Precision qubit 'i' controls the target system qubits
    qc.append(evolution_gate.control(1), [precision_qubits[i], *target_qubits])

qc.barrier()

# INVERSE QFT (The Decoding Lens)
# Decomposed CP gate (within scope of free plan)
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
qc.measure(precision_qubits, range(n_precision))

# EXECUTION ON IONQ CLOUD
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
    normalized_phase = round(phase % 0.5, 4) 
    
    if normalized_phase not in phase_map:
        phase_map[normalized_phase] = 0
    phase_map[normalized_phase] += count

# Sort consolidated results
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