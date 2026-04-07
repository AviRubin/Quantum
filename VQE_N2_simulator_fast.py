import numpy as np
from qiskit_aer import AerSimulator
# In Qiskit 2.x, the 'Estimator' at the top level is a V2 primitive.
# We must use the StatevectorEstimator for VQE to work smoothly.
from qiskit.primitives import StatevectorEstimator as Estimator

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper

# Define Geometry and Run Driver
# Nitrogen (N2) at its equilibrium bond length
geometry = "N 0.0 0.0 0.0; N 0.0 0.0 1.1"

# Initialize PySCF driver
driver = PySCFDriver(atom=geometry, charge=0, spin=0, basis="sto3g")
# PySCF might throw a warning, but it will run
problem = driver.run()

# Active Space Selection
transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
problem = transformer.transform(problem)

# Map to qubits
mapper = ParityMapper()
hamiltonian = problem.hamiltonian.second_q_op()
qubit_hamiltonian = mapper.map(hamiltonian)

# Build Ansatz
ansatz = TwoLocal(
    num_qubits=qubit_hamiltonian.num_qubits,
    rotation_blocks="ry",
    entanglement="full",
    reps=1,
)

# Estimator
# StatevectorEstimator is the standard in 2.x for exact simulation.
estimator = Estimator()

# VQE
vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100)
)

# Run VQE algo
result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)

# Final Results
# Add nuclear repulsion back to the electronic energy for the total energy
total_energy = result.eigenvalue.real + problem.nuclear_repulsion_energy

print("\n" + "="*45)
print(f"Nitrogen Ground State Energy: {total_energy:.6f} Hartree")
print("="*45)
