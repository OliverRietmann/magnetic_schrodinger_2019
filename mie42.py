import numpy as np
from sys import argv
from time import time

from simulation import Grid, Solution, Simulation
from helper.magnus4 import Magnus_CF4
from helper.propagators import Physical_Flow

output_folder = str(argv[1])

# Time ==================================
tspan = [0., 2.0 * np.pi]

# Spacial and Momentum Grid =============
g = Grid(4, 2**6, 8 * np.pi)

# Magnetic field 2-form =================
skew = np.array([[0.0, -1.0],[1.0, 0.0]])
zero = np.zeros((2, 2))
B0 = 0.25 * np.block([[skew, zero], [zero, skew]])
B = lambda t : B0
minus_B2 = lambda t : -np.dot(B(t), B(t))

# Initial data ==========================
var = 0.5
psi0 = lambda X : 1. / ((2 * np.pi * var)**(g.d / 4.)) * np.exp( -( (X[0] - 6.0)**2 + (X[1] - 4.0)**2 + (X[2] + 4.0)**2 + (X[3] + 4.0)**2 ) / (4. * var) ) * np.exp( 1j * 1 * X[0])

# Potential =============================
mie42_r2 = lambda r : 32.0 * (3.0**4 / (r**2 + 0.1) - 3.0**2 / (r + 0.1)) + 8.0
radius2 = lambda X : (X[0] - X[2])**2 + (X[1] - X[3])**2
V = lambda X : mie42_r2(radius2(X))

# Simulation ============================
start = time()
sim = Simulation(g, psi0, Magnus_CF4, B, Physical_Flow, args=(V, minus_B2))
sim.print()

E0 = sim.initial_energies(tspan[0])
E, N, t = sim.run_dynamics(tspan, 2**5, 'SS', 8)
print(np.max(np.abs(E[0,-1] + E[1,-1] + E[2,-1] - E0[0] - E0[1] - E0[2])))
Simulation.plot_energies(E, N, t, output_folder, ('energies.pdf', 'norm.pdf', 'H_B_energy.pdf'), bbox_inches='tight')
end = time()

print('Elapsed time in minutes: ', (end - start) / 60.0)
print('done')
