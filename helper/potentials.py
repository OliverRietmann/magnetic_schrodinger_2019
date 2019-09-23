from numpy import sqrt, exp, cos, arctan2
# harmonic
V = lambda X : sum([ x**2 for x in X ])

# morse
def morse(X, V0=8.0, beta=0.3, r0=1):
    R = sqrt(sum([ x**2 for x in X ])) - r0;
    return V0*(exp(-2.0*beta*R) - 2.0*exp(-beta*R)) + V0;

# A threefold morse potential
# This formula is for 2D real inputs only due to atan2
def morse_threefold(X, V0 = 8.0, sigma=0.05):
    return V0*(1 - exp((X[0]**2+X[1]**2) * (-sigma*(1-cos(3*arctan2(X[1],X[0])))**2/16.0)))**2;
