import numpy as np
from sys import argv

from simulation import Grid, Solution, Simulation
from helper.magnus4 import Magnus_CF4

output_folder = str(argv[1])

# Time ==================================
tspan = [0., 2*np.pi]
color = ['g','r','m']
# Use commented lists to get the plots for the lower order splittings
steps = [ 2**i for i in range(4,8) ] # [ 2**i for i in range(5,9) ]
split = ['BM42', 'KL6', 'KL8']       # ['SS', 'PRKS6', 'Y61']

# Spacial and Momentum Grid =============
g = Grid(3, 2**7, 8*np.pi)

# Magnetic field 2-form =================
Omega = lambda b: np.array([[0,-b[2],b[1]],[b[2],0,-b[0]],[-b[1],b[0],0]])
B0 = Omega( np.array([1,1,1])/np.sqrt(3.) )
B = lambda t: np.cos(t)*B0

# Initial data ==========================
var = 0.5
psi0 = lambda X : 1./((2*np.pi*var)**(g.d/4)) * np.exp( -sum((X-1)**2) / (4.0*var) ) * np.exp( 1j * X[0] * 2)

# Potential =============================
V = lambda X : sum(X**2)
Flow = lambda tspan, psi, X, V: np.exp( -1j*(tspan[1]-tspan[0])*V(X) ) * psi

# Simulation ============================
sim = Simulation(g, psi0, Magnus_CF4, B, Flow, args=(V))
sim.print()

# By periodicity, initial data serves as reference.
reference = Solution(sim.psi0(sim.X), tspan, g, steps[-1])
err, h = sim.run_convergence(reference, tspan, steps, split)
Simulation.plot_convergence(h, err, split, color, output_folder + "convergence.pdf", bbox_inches='tight')
