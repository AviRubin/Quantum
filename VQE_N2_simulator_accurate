import numpy as np
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import StatevectorEstimator as Estimator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# Define Geometry and Run Driver
# Nitrogen (N2) at its equilibrium bond length
geometry = "N 0.0 0.0 0.0; N 0.0 0.0 1.1"
driver = PySCFDriver(atom=geometry, charge=0, spin=0, basis="sto3g")
problem = driver.run()

# Active Space Selection
transformer = ActiveSpaceTransformer(num_electrons=6, num_spatial_orbitals=6)
problem = transformer.transform(problem)

# Extract and Map the Hamiltonian
# extract the second_q_op from the problem first
hamiltonian = problem.hamiltonian.second_q_op() 
mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(hamiltonian)

# Build the Chemistry-Aware Ansatz
# Hartree-Fock provides the optimal starting bitstring (e.g., |111000...>)
initial_state = HartreeFock(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
)

# UCCSD generates a circuit based on possible electron excitations
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=initial_state,
)

# Execution Logic
optimizer = SLSQP(maxiter=100)
estimator = Estimator()

def callback(eval_count, parameters, mean, std):
    print(f"Iteration {eval_count}: Energy = {mean:.8f}")

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback
)

# Run VQE algo
result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)

# Final Results
# Add nuclear repulsion back to the electronic energy for the total energy
total_energy = result.eigenvalue.real + problem.nuclear_repulsion_energy

print("\n" + "="*50)
print(f"Calculated Total Energy: {total_energy:.8f} Hartree")
print(f"Number of Qubits: {qubit_hamiltonian.num_qubits}")
print(f"Number of Parameters: {ansatz.num_parameters}")
print("="*50)
