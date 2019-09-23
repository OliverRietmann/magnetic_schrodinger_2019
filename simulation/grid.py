from numpy import array, linspace, meshgrid, sqrt, pi, exp
from numpy.linalg import norm
from numpy.fft import fftshift

class Grid:
	def __init__(self, d, n, L):
		self.d = d
		self.n = n
		self.L = L
	def print(self):
		print("Dimension: " + str(self.d))
		print("Number of grid points along a single axis: " + str(self.n))
		print("Edge length of torus: " + str(self.L))
	def mesh_1D(self):
		return linspace(-self.L/2, self.L/2, self.n, endpoint=False)
	def mesh(self):
		return array(meshgrid(*[self.mesh_1D()]*self.d))
	def dual_mesh(self):
		return array(meshgrid(*[fftshift(linspace(-pi*self.n/self.L, pi*self.n/self.L, self.n, endpoint=False))]*self.d))
	def norm(self, psi):
		return norm(psi) * sqrt(self.L / self.n)**self.d

if __name__ == '__main__':

	g = Grid(3, 2**8, 8*pi)
	X = g.mesh()

	var = 0.5
	psi0 = lambda X : 1./((2*pi*var)**(g.d/4)) * exp( -sum([(x-1)**2 for x in X]) / (4.0*var) ) * exp( 1j * X[0] * 2)


	print( g.norm( psi0(X) ) )
	g.print()
