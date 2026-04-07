import numpy as np
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# 1. SETUP
backend = AerSimulator()

# 2. THE CHALLENGE
secret_angle = random.uniform(0, 2 * np.pi)
print(f"--- SECRET PHASE GENERATED: {np.degrees(secret_angle):.2f}° ---")

# --- CHANGE 1: Increase Register Size ---
n_obs = 4       # 4 sensing qubits
rotor_idx = 4   # The 5th qubit is the rotor (index 4)
qc = QuantumCircuit(5, 4) # 5 total qubits, 4 classical bits

# Step A: Prepare the Rotor
qc.x(rotor_idx)
qc.p(secret_angle, rotor_idx)
qc.barrier()

# Step B: Phase Estimation (Encoding)
# --- CHANGE 2: Loop now covers 4 qubits ---
for q in range(n_obs):
    qc.h(q)
    # The kickback scales up to 2^3 (8) for the 4th qubit
    qc.cp(secret_angle * (2**q), q, rotor_idx)
qc.barrier()

# Step C: Inverse QFT (The 4-Qubit Lens)
# --- CHANGE 3: Expanded IQFT Unrolling ---
# 1. Swap the outer pairs to handle Qiskit endianness
qc.swap(0, 3)
qc.swap(1, 2)

# 2. First block (q0)
qc.h(0)
qc.cp(-np.pi/2, 0, 1)
qc.cp(-np.pi/4, 0, 2)
qc.cp(-np.pi/8, 0, 3)

# 3. Second block (q1)
qc.h(1)
qc.cp(-np.pi/2, 1, 2)
qc.cp(-np.pi/4, 1, 3)

# 4. Third block (q2)
qc.h(2)
qc.cp(-np.pi/2, 2, 3)

# 5. Fourth block (q3)
qc.h(3)

# Step D: Measurement
# --- CHANGE 4: Measure all 4 sensing qubits ---
qc.measure(range(n_obs), range(n_obs))

# VISUALIZE THE CIRCUIT
print("\nCircuit Diagram:")
print(qc.draw(output='text')) # Use output='mpl' if running in a Jupyter Notebook

# 4. EXECUTION
optimized_qc = transpile(qc, backend=backend)
job = backend.run(optimized_qc, shots=2048) # Increased shots for higher resolution
result = job.result()
counts = result.get_counts()

# 5. RESULTS & CIRCULAR ACCURACY
highest_prob_bitstring = max(counts, key=counts.get)
measured_int = int(highest_prob_bitstring, 2)
# --- CHANGE 5: Division is now by 2^4 (16) ---
detected_angle = (measured_int / (2**n_obs)) * 360

diff = abs(np.degrees(secret_angle) - detected_angle)
if diff > 180:
    diff = 360 - diff
accuracy = 100 - (diff / 180 * 100)

print("\n--- 4-QUBIT SENSOR RESULTS ---")
print(f"Most frequent bitstring: {highest_prob_bitstring}")
print(f"Detected Rotation: {detected_angle:.2f}° (Resolution: 22.5°)")
print(f"Original Secret: {np.degrees(secret_angle):.2f}°")
print(f"Circular Distance Error: {diff:.2f}°")
print(f"Accuracy: {accuracy:.2f}%")
