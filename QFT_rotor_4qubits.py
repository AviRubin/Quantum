# --- The challenge is to pick a secret rotation for one qubit, then use the other 3 qubits to "detect" it with QFT ---

import numpy as np
import random
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Authetication and Service
# Option A: local simulator
backend = AerSimulator()
print("Using Local AerSimulator")

# Option B: IBM hardware (likely need to vet when not limited to run time)
# service = QiskitRuntimeService(channel="ibm_quantum_platform")
# backend = service.least_busy(operational=True, simulator=False, min_qubits=4)
# print(f"Connecting to IBM Hardware: {backend.name}")

# THE CHALLENGE: Generate a Secret Rotation
secret_angle = random.uniform(0, 2 * np.pi)
print(f"--- SECRET PHASE GENERATED: {np.degrees(secret_angle):.2f}° ---")

# Circuit Construction (3 Observation Qubits + 1 Rotor)
n_obs = 3
rotor_idx = 3
qc = QuantumCircuit(4, 3)

# Step A: Prepare the Rotor in state |1> and apply the secret phase
qc.x(rotor_idx)
qc.p(secret_angle, rotor_idx)
qc.barrier()

# Step B: Phase Estimation (Encoding the rotation into the register)
for q in range(n_obs):
    qc.h(q)
    # The phase kickback: 2^q rotations
    # This is the "Sensing" part of the radar
    qc.cp(secret_angle * (2**q), q, rotor_idx)
qc.barrier()

# Step C: Inverse QFT (The "Lens" that focuses the data)
# Manually unrolling the IQFT for 3 qubits
qc.swap(0, 2)
qc.h(0)
qc.cp(-np.pi/2, 0, 1)
qc.h(1)
qc.cp(-np.pi/4, 0, 2)
qc.cp(-np.pi/2, 1, 2)
qc.h(2)

# Step D: Measurement
qc.measure([0, 1, 2], [0, 1, 2])

# Visualize the Circuit
print("\nCircuit Diagram:")
print(qc.draw(output='text')) # Use output='mpl' if running in a Jupyter Notebook

# Execution
# Transpile for the chosen backend (Aer or IBM)
optimized_qc = transpile(qc, backend=backend)

# Logic check: Use 'backend.run' for Aer, but 'Sampler' for IBM Runtime
if isinstance(backend, AerSimulator):
    # LOCAL RUN
    job = backend.run(optimized_qc, shots=1024)
    result = job.result()
    counts = result.get_counts()
else:
    # IBM RUN (Uncomment 'Sampler' import at top if using this)
    # backend = service.least_busy(operational=True, simulator=False, min_qubits=4)
    # print(f"\nConnecting to: {backend.name}...")
    # sampler = Sampler(mode=backend)
    # job = sampler.run([optimized_qc])
    # print(f"Job submitted. ID: {job.job_id()}")
    # print("Waiting for results (this may take a few minutes in the queue)...")
    # result = job.result()
    # counts = result[0].data.meas.get_counts()
    pass

# Logic to find the 'Winner' (the peak in the data)
highest_prob_bitstring = max(counts, key=counts.get)
measured_int = int(highest_prob_bitstring, 2)
detected_angle = (measured_int / (2**n_obs)) * 360

# Calculate the modular difference (distance on a circle)
diff = abs(np.degrees(secret_angle) - detected_angle)
if diff > 180:
    diff = 360 - diff
accuracy = 100 - (diff / 180 * 100) # Percentage based on how "far" it was

print("\n--- RESULTS ---")
print(f"Most frequent bitstring: {highest_prob_bitstring}")
print(f"Detected Rotation: {detected_angle:.2f}°")
print(f"Original Secret: {np.degrees(secret_angle):.2f}°")
# print(f"Accuracy: {100 - abs(np.degrees(secret_angle) - detected_angle):.2f}%")
print(f"Circular Distance Error: {diff:.2f}°")
print(f"Accuracy: {accuracy:.2f}%")
