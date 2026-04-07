import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import efficient_su2
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_algorithms import VarQITE, TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

# 1. CHEMISTRY SETUP (Same as before)
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper

# adjust value in PySCFDriver for distance between proton centers, unique for each molecule. Below is for H2
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
problem = driver.run()
mapper = ParityMapper(num_particles=problem.num_particles)
h2_hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

# 2. VARQITE COMPONENTS
ansatz = efficient_su2(num_qubits=h2_hamiltonian.num_qubits, reps=1)
# Start with some non-zero initial parameters
init_params = np.full(ansatz.num_parameters, 0.01) 

# The "Principle" defines how we map imaginary time to parameter changes
var_principle = ImaginaryMcLachlanPrinciple()
estimator = Estimator()

# 3. DEFINE THE EVOLUTION PROBLEM
# 'time' here is 'imaginary time' (tau). As tau increases, we reach the ground state.
total_imaginary_time = 50.0 
evolution_problem = TimeEvolutionProblem(h2_hamiltonian, time=total_imaginary_time, aux_operators=[h2_hamiltonian])

# 4. RUN THE EVOLUTION
var_qite = VarQITE(ansatz, init_params, var_principle, estimator)
evolution_result = var_qite.evolve(evolution_problem)

# 5. EXTRACT AND PLOT RESULTS
# We can see the energy 'decaying' into the ground state
energies = [obs[0][0] + problem.nuclear_repulsion_energy for obs in evolution_result.observables]
times = np.linspace(0, total_imaginary_time, len(energies))

print(f"\nFinal QITE Energy: {energies[-1]:.6f} Hartree")
print(f"Target Ground State: ~ -1.137 Hartree")

plt.plot(times, energies)
plt.xlabel('Imaginary Time (tau)')
plt.ylabel('Energy (Hartree)')
plt.title('H2 Cooling via VarQITE')
plt.axhline(y=-1.137, color='r', linestyle='--', label='Target')
plt.legend()
plt.show()
