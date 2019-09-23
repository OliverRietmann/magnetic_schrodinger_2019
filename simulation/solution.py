from numpy import pi
import matplotlib.pyplot as plt
import pickle

from .grid import Grid
from .phaseplot import plotcf2d

class Solution:
	def __init__(self, psi, g, tspan, steps):
		self.psi = psi
		self.g = g
		self.tspan = tspan
		self.steps = steps
		
	def norm(self):
		return self.g.norm(self.psi)
	
	def difference(self, other):
		return self.g.norm(self.psi - other.psi)
		
	def print(self):
		print("Time interval: [" + str(self.tspan[0]) + ", " + str(self.tspan[1]) + "]")
		print("Number of time steps: " + str(self.steps))
		self.g.print()

	def phaseplot(self, filename, darken=0.05, title='Solution', bbox_inches=None):
		x = self.g.mesh_1D()
		fig = plt.figure()
		fig.add_subplot(111)
		plt.title(title)
		cs = plotcf2d(x, x, self.psi, darken=darken)
		cbar = plt.colorbar(cs, ticks=[0, pi/2, pi, 3*pi/2, 2*pi])
		cbar.ax.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
		fig.savefig(filename, bbox_inches=bbox_inches)
		plt.close(fig)

	def save(self, filename):
		with open(filename, "wb") as f:
			pickle.dump(self, f)
	
	@staticmethod	
	def load(filename):
		with open(filename, "rb") as f:
			return pickle.load(f)
