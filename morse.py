import numpy as np
from sys import argv

from simulation import Grid, Solution, Simulation
from helper.magnus4 import Magnus_CF4
from helper.potentials import morse_threefold
from helper.propagators import Physical_Flow

output_folder = str(argv[1])

# Time ==================================
tspan = [0., 4*np.pi]
steps = 2**8
split = 'SS'

# Spacial and Momentum Grid =============
g = Grid(2, 2**8, 8*np.pi)

# Magnetic field 2-form =================
B0 = 0.5*np.array([[0,-1],[1,0]])
B = lambda t: B0
minus_B2 = lambda t: -np.dot(B(t), B(t))

# Initial data ==========================
var = 0.5
psi0 = lambda X : 1./((2*np.pi*var)**(g.d/4.)) * np.exp( -( (X[0]-2.)**2 + (X[1]-2.)**2) / (4.*var) ) * np.exp( 1j * 2 * X[0])

# Potential =============================
V = lambda X : morse_threefold(X, V0=16.0, sigma=0.5)

# Simulation ============================
sim = Simulation(g, psi0, Magnus_CF4, B, Physical_Flow, args=(V, minus_B2))
sim.print()

# Plot energies
E0 = sim.initial_energies(tspan[0])
E, N, t = sim.run_dynamics(tspan, steps, split)
print(np.max(np.abs(E[0,-1]+E[1,-1]+E[2,-1]-E0[0]-E0[1]-E0[2])))
Simulation.plot_energies(E, N, t, output_folder, ('energies.pdf', 'norm.pdf', 'angular_mom.pdf'), bbox_inches='tight')

# Plot wavefunction
sim.sol.phaseplot(output_folder + 'solution.pdf', title="Solution at Time $t=4\pi$", bbox_inches='tight')
X = g.mesh()
gauss = Solution(psi0(X), g, tspan, steps)
gauss.phaseplot(output_folder + 'initial_data.pdf', title="Initial Data $\psi_0(x)$", bbox_inches='tight')

# Plot potential
sim.plot_potential(output_folder + 'potential.pdf', 50, zorder=-9, cmap='Greys_r')
