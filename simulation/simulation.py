from .grid import Grid
from .solution import Solution
from .SplittingParametersRotation import SplittingParameters

from numpy import array, tensordot, dot, vdot, transpose, exp, real, sqrt, zeros, linspace, real, imag
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import makedirs
from os.path import isdir

class Simulation:

	S = SplittingParameters()

	def __init__(self, g, psi0, solver, B, Phi, args=()):
		self.g = g
		self.psi0 = psi0
		self.solver = solver
		self.B = B
		self.Phi = Phi

		self.X = g.mesh()
		if not type(args) is tuple: args = (args,)
		self.args = (self.X,) + args
		self.dual_X = self.g.dual_mesh()
		self.K = sum(self.dual_X**2)

	Phi_2 = staticmethod( lambda tspan, psi, K: ifftn( exp( -1j*(tspan[1]-tspan[0])*K ) * fftn(psi) ) )
	Phi_3 = staticmethod( lambda R, X: tensordot( transpose(R), X, 1) )
	@staticmethod
	def plot_energies(E,N,t, output_path, filenames, bbox_inches=None):
		try:
			makedirs(output_path)
		except OSError:
			if not isdir(output_path):
				raise;

		plt.figure()
		plt.title('Energies along the Solution')
		plt.plot(t,E[0,:], label="$E_{\mathrm{kin}}(t)$")
		plt.plot(t,E[1,:], label="$E_{\mathrm{pot}}(t)$")
		plt.plot(t,E[2,:], label="$E_{\mathrm{mag}}(t)$")
		plt.plot(t,E[0,:]+E[1,:]+E[2,:], label="$E_{\mathrm{tot}}(t)$")
		plt.xlabel('time t')
		plt.ylabel('energy')
		plt.legend(loc='lower left')
		plt.grid(True)
		plt.savefig(output_path+filenames[0], bbox_inches=bbox_inches)
		plt.close()

		plt.figure()
		plt.title('Norm of the Solution')
		plt.plot(t,N, label="$||\psi||_{L^2}(t)$")
		plt.xlabel('time t')
		plt.ylabel('norm')
		plt.legend(loc='lower left')
		plt.grid(True)
		plt.savefig(output_path+filenames[1], box_inches=bbox_inches)
		plt.close()

		plt.figure()
		plt.title('Angular Momentum')
		plt.plot(t,(-1)*E[3,:], label='angular momentum')
		plt.xlabel('time')
		plt.ylabel('angular momentum')
		plt.legend(loc='lower left')
		plt.grid(True)
		plt.savefig(output_path+filenames[2], bbox_inches=bbox_inches)
		plt.close()

	@staticmethod
	def plot_convergence(h, err, split, color, filename, bbox_inches=None):
		order = [ SplittingParameters.order(s) for s in split ]
		plt.title('Order of Convergence')
		for i in range(len(split)):
			plt.loglog(h, err[i,:], 'o-', color=color[i], label=split[i], basex=2)
			plt.loglog(h, h**order[i], '--', color=color[i], label="$x^{"+str(order[i])+"}$", basex=2)
		plt.xlabel('step size')
		plt.ylabel('error')
		plt.legend(loc='lower right')
		plt.savefig(filename, bbox_inches=bbox_inches)
		plt.close()

	def run(self, tspan, steps, split):
		a, b = Simulation.S.build(split)
		self.sol = Solution( Simulation.S.intsplit(self.Phi, Simulation.Phi_2, Simulation.Phi_3, a, b, tspan, steps, self.psi0, args1=self.args, args2=(self.K), args3=(self.solver, self.B)) , self.g, tspan, steps)
		return self.sol
	
	def norm(self):
		return self.sol.norm()
		
	def diff_to_inital(self):
		return self.g.norm(self.sol.psi - self.psi0(self.X))

	def print(self):
		self.g.print()
		
	def print_sol(self):
		self.sol.print()

	def plot(self, filename, colors=10, bbox_inches=None):
		initial = self.psi0(self.X)
		fig = plt.figure()

		ax1 = fig.add_subplot(221)
		cp = ax1.contourf(self.X[0], self.X[1], real(self.sol.psi), colors)
		ax1.set_title("$\mathrm{Re}(\psi(x,y,t))$")
		ax1.set_xticks([])
		ax1.set_aspect('equal')

		ax2 = fig.add_subplot(222)
		ax2.contourf(self.X[0], self.X[1], imag(self.sol.psi), colors)
		ax2.set_title("$\mathrm{Im}(\psi(x,y,t))$")
		ax2.set_xticks([])
		ax2.set_yticks([])
		ax2.set_aspect('equal')

		ax3 = fig.add_subplot(223)
		ax3.contourf(self.X[0], self.X[1], real(initial), colors)
		ax3.set_title("$\mathrm{Re}(\psi(x,y,0))$")
		ax3.set_xticks([])
		ax3.set_aspect('equal')

		ax4 = fig.add_subplot(224)
		ax4.contourf(self.X[0], self.X[1], imag(initial), colors)
		ax4.set_title("$\mathrm{Im}(\psi(x,y,0))$")
		ax4.set_xticks([])
		ax4.set_yticks([])
		ax4.set_aspect('equal')

		cbar = fig.add_axes([0.47, 0.1, 0.05, 0.80])
		fig.colorbar(cp, cax=cbar)

		fig.savefig(filename, bbox_inches=bbox_inches)
		plt.close(fig)

	def plot_potential(self, filename, *args, **kwargs):
		potential = self.args[1](self.X)
		fig = plt.figure()
		CS = plt.contourf(self.X[0], self.X[1], potential, *args, **kwargs)
		plt.colorbar(CS)
		plt.title("Potential $V(x)$")
		plt.gca().set_rasterization_zorder(-1)
		fig.savefig(filename, bbox_inches='tight')
		plt.close(fig)

	def energies(self):
		nd = self.g.n**self.g.d
		scale = self.g.L**2 / nd

		psi_hat = fftn(self.sol.psi) #, norm='ortho'
		E_kin = real(vdot(psi_hat, self.K * psi_hat)) / nd

		E_pot = real(vdot(self.sol.psi, self.args[1](self.X)*self.sol.psi))

		Bt = self.B(self.sol.tspan[1])
		angular_mom = 0
		for i in range(self.g.d):
			angular_mom += real( vdot(fftn(self.X[i]*self.sol.psi), sum([Bt[i,j]*self.dual_X[j] for j in range(i+1,self.g.d)])*psi_hat) - vdot(fftn(sum([Bt[i,j]*self.X[j] for j in range(i+1,self.g.d)])*self.sol.psi), self.dual_X[i]*psi_hat) ) / nd

		E_mag = real(vdot(self.sol.psi, sum(Simulation.Phi_3(transpose(Bt), self.X)**2) * self.sol.psi)) - angular_mom

		return array([E_kin, E_pot, E_mag, angular_mom]) * scale

	def initial_energies(self, t0):
		nd = self.g.n**self.g.d
		scale = self.g.L**2 / nd
		psi = self.psi0(self.X)

		psi_hat = fftn(psi)
		E_kin = real(vdot(psi_hat, self.K * psi_hat)) / nd

		E_pot = real(vdot(psi, self.args[1](self.X)*psi))

		Bt = self.B(t0)
		angular_mom = 0
		for i in range(self.g.d):
			angular_mom += real( vdot(fftn(self.X[i]*psi), sum([Bt[i,j]*self.dual_X[j] for j in range(i+1,self.g.d)])*psi_hat) - vdot(fftn(sum([Bt[i,j]*self.X[j] for j in range(i+1,self.g.d)])*psi), self.dual_X[i]*psi_hat) ) / nd

		E_mag = real(vdot(psi, sum([ x**2 for x in Simulation.Phi_3(transpose(Bt), self.X) ]) * psi)) - angular_mom

		return array([E_kin, E_pot, E_mag, angular_mom]) * scale

	def run_dynamics(self, tspan, steps, split, substeps=1):
		E = zeros((4,steps+1))
		N = zeros(steps+1)
		t = linspace(tspan[0], tspan[1], steps+1, endpoint=True)
		print("Do up to " + str(steps) + " steps.")
		for i in range(1, steps+1):
			print("Now doing " + str(i) + " steps.")
			self.run([t[0],t[i]], i * substeps, split)
			E[:,i] = self.energies()
			N[i] = self.norm()
		E[:,0] = self.initial_energies(tspan[0])
		N[0] = self.g.norm(self.psi0(self.X))
		return E, N, t

	def run_convergence(self, reference, tspan, steps, split):
		h = tspan[1] / array(steps)
		err = zeros((len(split),len(h)))
		for i in range(0, len(split)):
			print("Splitting: " + str(split[i]))
			for j in range(0,len(steps)):
				print("Doing " + str(steps[j]) + " steps.")
				self.run(tspan, steps[j], split[i])
				err[i,j] = self.sol.difference(reference)
		return err, h
