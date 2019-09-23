"""
Example 1 from S. Blanes, P.C. Moan / Applied Numerical Mathematics 56 (2006) 1519-1537 commutator free Magnus integrator of order 4
"""

from numpy import polyfit, zeros, eye, exp, log, sin, cos, array, pi, dot, arange, sqrt
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt

# Magnus constants
c1 = 0.5*(1. - 0.5773502691896258)
c2 = 0.5*(1. + 0.5773502691896258)
a1 = 0.5*(0.5 - 0.5773502691896258)
a2 = 0.5*(0.5 + 0.5773502691896258)

def Magnus_CF4(tspan, A, N, *args):
        u = 1.*eye( len( A(1.*tspan[0], *args) ) )
        h = (tspan[1]-tspan[0]) / (1.*N)
        for k in range(N):
            t0 = k*h + tspan[0]
            t1 = t0 + c1*h
            t2 = t0 + c2*h
            A1 = A(t1, *args)
            A2 = A(t2, *args)
            u = dot(expm(a1*h*A1+a2*h*A2), dot(expm(a2*h*A1+a1*h*A2), u))
            
        return u

if __name__ == "__main__":
    # General parameters and Functions
    tspan = [0.,20*pi]
    A0 = array([[0,1],[-1,0]])
    dt = 5;
    #A = lambda t: sqrt(t+dt)*A0
    
    Omega = lambda b: array([[0,-b[2],b[1]],[b[2],0,-b[0]],[-b[1],b[0],0]])
    I = eye(3)
    e = array([Omega(i) for i in I])

    one = lambda t : 1
    #b_vec = lambda t : -array([np.cos(t), sin(t), one(t)]) / sqrt(2.)
    A = lambda t : 1. / sqrt(3) * (cos(t) * e[0] + sin(t) * e[1] + one(t) * e[2])


    print('Compute the almost exact solution by brute force')
    N = 2**14
    print(('N=%d'%N))
    res = Magnus_CF4(tspan, A, N)


	##print('Compute the almost exact solution by analytical computations')
	#F = lambda t: 2*sqrt(t+dt)**3/3.
	#res = expm((F(tspan[1])-F(tspan[0]))*A0)

    print('go for iterations')
    err = [];
    Ns = 2**arange(4, 14)
    for N in Ns:
        print(('N=%d'%N))
        err_m = norm(Magnus_CF4(tspan, A, N) - res)
        print(err_m)
        err += [err_m]

    ts = (tspan[1] - tspan[0]) / Ns
    err = array(err);
    p = polyfit(log(ts[2:]),log(err[2:]),1)
    print(p[0])

    plt.title('A(t)=sqrt(t+5)*A0')
    plt.loglog(Ns, err, 'o-', label='Magnus')
    plt.loglog(Ns, 1./Ns**4, label='$x^{-4}$')
    plt.legend()
    plt.show()
