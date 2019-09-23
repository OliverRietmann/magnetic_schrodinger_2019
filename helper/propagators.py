import numpy as np
from scipy.integrate import fixed_quad, quad

# Evaluate the quadratic form associated with matrix M on meshgrid X.
def quad_form(M, X):
	s = M.shape
	result = np.zeros(X[0].shape)
	for i in range(s[0]):
		for j in range(s[1]):
			result += M[i,j] * X[i] * X[j]
	return result

# Quadrature for matrix-valued function M (e.g. for -B^2(t))
def quad_mat(M, tspan, quad, *args):
	s = M(tspan[0]).shape
	I = np.zeros(s)
	for i in range(s[0]):
		for j in range(s[1]):
			I[i,j] = quad(lambda t: M(t)[i,j], tspan[0], tspan[1], *args)[0]
	return I

# V: potential const in time, Q: -B^2(t)
Physical_Flow = lambda tspan, psi, X, V, Q: np.exp( -1j*( (tspan[1]-tspan[0])*V(X) + quad_form(quad_mat(Q, tspan, quad), X) ) ) * psi
#0.25*(tspan[1]-tspan[0])*sum([ x**2 for x in X ])       
Special_Flow = lambda tspan, psi, X, V, M: np.exp( -1j*( (tspan[1]-tspan[0])*V(X) + quad_form(M(tspan[1]) - M(tspan[0]), X) ) ) * psi