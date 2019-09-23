import numpy as np
from sys import argv

from simulation import Grid, Solution, Simulation
from helper.magnus4 import Magnus_CF4
from helper.propagators import Physical_Flow

output_folder = str(argv[1])

# Time ==================================
tspan = [0., 2 * np.pi]
color = ['g','r', 'm']
# Use commented lists to get the plots for the lower order splittings
steps = [ 2**i for i in range(5,9) ] # [ 2**i for i in range(4,8) ]
split = ['SS', 'PRKS6', 'Y61'] # ['SS', 'PRKS6', 'Y61']

# Spacial and Momentum Grid =============
g = Grid(3, 2**7, 8 * np.pi)

# Magnetic field 2-form =================
Omega = lambda b: np.array([[0,-b[2],b[1]],[b[2],0,-b[0]],[-b[1],b[0],0]])
I = np.eye(3)
e = np.array([Omega(i) for i in I])
one = lambda t : 1
B = lambda t : 1. / np.sqrt(3.) * (np.cos(t) * e[0] + np.sin(t) * e[1] + one(t) * e[2])
minus_B2 = lambda t : -np.dot(B(t), B(t))

# Initial data ==========================
var = 0.5
psi0 = lambda X : 1./((2 * np.pi*var)**(g.d / 4)) * np.exp( -sum((X - 1)**2) / (4.0 * var) ) * np.exp( 1j * X[0] * 2)

# Potential =============================
V = lambda X : 1 / 32. * (X[0]**2 + X[1]**2 + X[2]**2)**2 - (X[0]**2 + 1.5 * X[1]**2 + 2 * X[2]**2)
Flow = lambda tspan, psi, X, V: Physical_Flow(tspan, psi, X, V, minus_B2)

# Simulation ============================
sim = Simulation(g, psi0, Magnus_CF4, B, Flow, args=(V))
sim.print()

# Reference solution
reference = sim.run(tspan, 2**8, 'KL8')

# Plot convergence
err, h = sim.run_convergence(reference, tspan, steps, split)
Simulation.plot_convergence(h, err, split, color, output_folder + 'convergence.pdf', bbox_inches='tight')
