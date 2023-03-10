# -*- coding: utf-8 -*-
"""

Authors: V. Gradinaru, R. Bourquin

https://github.com/WaveBlocks/WaveBlocksND/blob/master/WaveBlocksND/SplittingParameters.py

"""

from numpy import zeros, flipud, double, array, dot, transpose, ceil

__all__ = ["SplittingParameters", "PrepSplittingParameters"]


class SplittingParameters(object):


    def build(self, method):
        """
        :param method: A string specifying the method for time integration.
        :return: Two arrays :math:`a` and :math:`b`.

        ====== ===== =========== =========
        Method Order Authors     Reference
        ====== ===== =========== =========
        LT     1     Lie/Trotter [1]_, [3]_ page 42, equation 5.2
        S2     2     Strang      [2]_, [3]_ page 42, equation 5.3
        SS     2     Strang      [2]_, [3]_ page 42, equation 5.3
        PRKS6  4     Blanes/Moan [4]_ page 318, table 2, 'S6'
        BM42   4     Blanes/Moan [4]_ page 318, table 3, 'SRKNb6'
        Y4     4     Yoshida     [5]_, [3]_ page 40, equation 4.4
        Y61    6     Yoshida     [5]_, [3]_ page 144, equation 3.11
        BM63   6     Blanes/Moan [4]_ page 318, table 3, 'SRKNa14'
        KL6    6     Kahan/Li    [6]_, [3]_ page 144, equation 3.12
        KL8    8     Kahan/Li    [6]_, [3]_ page 145, equation 3.14
        L42    (4,2) McLachlan   [7]_ page 6
        L84    (8,4) McLachlan   [7]_ page 8
        ====== ===== =========== =========

        .. [1] H.F. Trotter, "On the product of semi-groups of operators",
               Proc. Am. Math. Soc.1O (1959) 545-551.

        .. [2] G. Strang, "On the construction and comparison of difference schemes",
               SIAM J. Numer. Anal. 5 (1968) 506-517.

        .. [3] E. Hairer, C. Lubich, and G. Wanner, "Geometric Numerical Integration -
               Structure-Preserving Algorithms for Ordinary Differential Equations",
               Springer-Verlag, New York, 2002 (Corrected second printing 2004).

        .. [4] S. Blanes and P.C. Moan, "Practical Symplectic Partitioned
               Runge-Kutta and Runge-Kutta-Nystrom Methods", J. Computational and
               Applied Mathematics, Volume 142, Issue 2, (2002) 313-330.

        .. [5] H. Yoshida, "Construction of higher order symplectic integrators",
               Phys. Lett. A 150 (1990) 262-268.

        .. [6] W. Kahan and  R.-c. Li, "Composition constants for raising the orders
               of unconventional schemes for ordinary differential equations",
               Math. Comput. 66 (1997) 1089-1099.

        .. [7] R.I. McLachlan, "Composition methods in the presence of small parameters",
               BIT Numerical Mathematics, Volume 35, Issue 2, (1995) 258-268.
        """
        if method == "LT":
            s = 1
            a = zeros(s)
            b = zeros(s)
            a[0] = 1.0
            b[0] = 1.0
        elif method == "S2":
            s = 2
            a = zeros(s)
            b = zeros(s)
            a[1] = 1.0
            b[0] = 0.5
            b[1] = 0.5
        elif method == "SS":
            s = 2
            a = zeros(s)
            b = zeros(s)
            a[0] = 0.5
            a[1] = 0.5
            b[0] = 1.0
        elif method == "L42":
            # Pattern ABA and m = s = 2
            s = 3
            a = zeros(s)
            b = zeros(s)
            a[1] = 0.5773502691896258
            a[0] = 0.5*(1.-a[1])
            a[2] = a[0]
            b[0] = 0.5
            b[1] = 0.5
        elif method == "BM42":
            s = 7
            a = zeros(s)
            b = zeros(s)
            a[1] = 0.245298957184271
            a[2] = 0.604872665711080
            a[3] = 0.5 - a[:3].sum()
            a[4:] = flipud(a[1:4])
            b[0] = 0.0829844064174052
            b[1] = 0.396309801498368
            b[2] = -0.0390563049223486
            b[3] = 1.0 - 2.0*b[:3].sum()
            b[4:] = flipud(b[:3])
        elif method == "Y4":
            s = 4
            a = zeros(s)
            b = zeros(s)
            pp = 3.0
            theta = 1.0/(2.0-2**(1.0/pp))
            vi = -2**(1.0/pp)*theta
            a[0] = 0.0
            a[1] = theta
            a[2] = vi
            a[3] = theta
            b[0] = 0.5*theta
            b[1] = 0.5*(vi+theta)
            b[2:] = flipud(b[:2])
        elif method == "Y61":
            s = 8
            a = zeros(s)
            b = zeros(s)
            a[1] = 0.78451361047755726381949763
            a[2] = 0.23557321335935813368479318
            a[3] = -1.17767998417887100694641568
            a[4] = 1.0 - 2.0*a[1:4].sum()
            a[5:] = flipud(a[1:4])
            b[0] = 0.5*a[1]
            b[1] = 0.5*a[1:3].sum()
            b[2] = 0.5*a[2:4].sum()
            b[3] = 0.5*(1-4*b[1]-a[3])
            b[4:] = flipud(b[0:4])
        elif method == "BM63":
            s = 15
            a = zeros(s)
            b = zeros(s)
            a[1] = 0.09171915262446165
            a[2] = 0.183983170005006
            a[3] = -0.05653436583288827
            a[4] = 0.004914688774712854
            a[5] = 0.143761127168358
            a[6] = 0.328567693746804
            a[7] = 0.5 - a[:7].sum()
            a[8:] = flipud(a[1:8])
            b[0] = 0.0378593198406116
            b[1] = 0.102635633102435
            b[2] = -0.0258678882665587
            b[3] = 0.314241403071447
            b[4] = -0.130144459517415
            b[5] = 0.106417700369543
            b[6] = -0.00879424312851058
            b[7] = 1.0 - 2.0*b[:7].sum()
            b[8:] = flipud(b[:7])
        elif method =="PRKS6":
            s = 7
            a = zeros(s)
            b = zeros(s)
            a[1] = 0.209515106613362
            a[2] = -0.143851773179818
            a[3] = 0.5 - a[:3].sum()
            a[4:] = flipud(a[1:4])
            b[0] = 0.0792036964311957
            b[1] = 0.353172906049774
            b[2] = -0.0420650803577195
            b[3] = 1 - 2*b[:3].sum()
            b[4:] = flipud(b[:3])
        elif method == "KL6":
            s = 10
            a = zeros(s)
            b = zeros(s)
            a[1] = 0.39216144400731413927925056
            a[2] = 0.33259913678935943859974864
            a[3] = -0.70624617255763935980996482
            a[4] = 0.08221359629355080023149045
            a[5] = 1.0 - 2.0*a[1:5].sum()
            a[6:] = flipud(a[1:5])
            b[0] = 0.5*a[1]
            b[1] = 0.5*a[1:3].sum()
            b[2] = 0.5*a[2:4].sum()
            b[3] = 0.5*a[3:5].sum()
            b[4] = 0.5*(1-2*a[1:4].sum()-a[4])
            b[5:] = flipud(b[0:5])
        elif method == "KL8":
            s = 18
            a = zeros(s)
            b = zeros(s)
            a[0] = 0.0
            a[1] = 0.13020248308889008087881763
            a[2] = 0.56116298177510838456196441
            a[3] = -0.38947496264484728640807860
            a[4] = 0.15884190655515560089621075
            a[5] = -0.39590389413323757733623154
            a[6] = 0.18453964097831570709183254
            a[7] = 0.25837438768632204729397911
            a[8] = 0.29501172360931029887096624
            a[9] = -0.60550853383003451169892108
            a[10:] = flipud(a[1:9])
            b[0:-1] = 0.5*(a[:-1]+a[1:])
            b[-1] = 1.*b[0]
        elif method == "L84":
            # Pattern ABA
            s = 6
            a = zeros(s)
            b = zeros(s)
            a[0] = 0.07534696026989288842
            a[1] = 0.51791685468825678230
            a[2] = -0.09326381495814967072
            b[0] = 0.19022593937367661925
            b[1] = 0.84652407044352625706
            b[2] = -1.07350001963440575260
            a[3:] = flipud(a[0:3])
            b[3:-1] = flipud(b[0:2])
        else:
            raise NotImplementedError("Unknown method: " + method)

        return a, b

    @staticmethod
    def order(method):
        r"""
        :param method: A string specifying the method for time integration.
        :return: The order of this method.
        """
        return {
            "LT":    1,
            "S2":    2,
            "SS":    2,
            "PRKS6": 4,
            "BM42":  4,
            "Y4":    4,
            "Y61":   6,
            "BM63":  6,
            "KL6":   6,
            "KL8":   8,
            "KL10": 10}[method]
			
    def intsplit(self, Phi_1, Phi_2, Phi_3, a, b, tspan, N, psi0, args1=(), args2=(), args3=()):
        r"""
        Compute a single, full propagation step by operator splitting.

        :param psi1: First evolution operator :math:`\Psi_a`
        :param psi2: Second evolution operator :math:`\Psi_b`
        :param a: Parameters for evolution with :math:`\Psi_a`
        :param b: Parameters for evolution with :math:`\Psi_b`
        :param tspan: Timespan :math:`t` of a single, full splitting step
        :param N: Number of substeps to perform
        :param y0: Current value of the solution :math:`y(t)`
        :param getall: Return the full (internal) time evolution of the solution
        :param args1: Additional optional arguments of :math:`\Psi_a`
        :param args2: Additional optional arguments of :math:`\Psi_b`
        """
        if not type(args1) is tuple: args1 = (args1,)
        if not type(args2) is tuple: args2 = (args2,)
        if not type(args3) is tuple: args3 = (args3,)
        
        h = (tspan[1] - tspan[0]) / (1.0*N)
        ta = tspan[0]
        tb = tspan[0]

        X = args1[0]
        U = args3[0]
        B = args3[1]
        
        get_steps = lambda interval : ceil(1000 * abs(interval) + 1).astype('int')
        U_steps = get_steps(tspan[1] - tspan[0])
        R = U(tspan, B, U_steps)
        y = psi0(Phi_3(R, X))
        
        s = a.shape[0]
        for k in range(N):
            for j in range(s):             
                y = Phi_1([ta, ta + a[j]*h], y, Phi_3(R, X), *(args1[1:]))
                
                I = [tb, tb + b[j]*h]
                U_steps = get_steps(I[1] - I[0])
                r = U(I, B, U_steps)
                R = dot(R, transpose(r))
                
                y = Phi_2(I, y, *args2)
                
                ta = ta + a[j]*h
                tb = tb + b[j]*h    
        return y

class PrepSplittingParameters(object):

    def build(self, method):
        if method == "BCR764": # Blanes, Casas, Ros 2000, table IV
            # kernel ABA
            # exchanged a and b from paper in order to have consistency with previous code
            a = zeros(4) # for Beps
            a[1] = 1.5171479707207228
            a[2] = 1.-2.*a[1]
            a[3] = 1.* a[1]
            #
            b = zeros(4) # for Abig
            b[0] = 0.5600879810924619
            b[1] = 0.5-b[0]
            b[2] = 1.*b[1]
            b[3] = 1.*b[0]
            # processor
            z = zeros(6) # for Abig
            z[0] = -0.3346222298730
            z[1] = 1.097567990732164
            z[2] = -1.038088746096783
            z[3] = 0.6234776317921379
            z[4] = -1.102753206303191
            z[5] = -0.0141183222088869
            #
            y = zeros(6) # for Beps
            y[0] = -1.621810118086801
            y[1] = 0.0061709468110142
            y[2] = 0.8348493592472594
            y[3] = -0.0511253369989315
            y[4] = 0.5633782670698199
            y[5] = -0.5
        else:
            raise NotImplementedError("Unknown method: " + method)

        forBeps = a
        forAbig = b
        y4Beps = y
        z4Abig = z
        #return a, b, y, z
        return forBeps, forAbig, y4Beps, z4Abig


    def intprepsplit(self, Beps, Abig, forBeps, forAbig, y, z, tspan, N, u0, getall=False):#, args1=(), args2=()):
        r"""
        """
        #if type(args1) != type(()): args1 = (args1,)
        #if type(args2) != type(()): args2 = (args2,)

        s = forBeps.shape[0]
        p = y.shape[0]
        h = (tspan[1]-tspan[0])/(1.0*N)
        u = 1.*u0
        if getall:
            uu = [u0]
            t = [tspan[0]]

        #print h

        for j in range(p): # preprocessor
            u = Abig(-z[j]*h, 1.*u)#, *args1)
            u = Beps(-y[j]*h, 1.*u)#, *args2)

        #if getall:
        #    uu += [u]
        #    t+= [t[-1]+h]

        for k in range(N): # kernel
            for j in range(s):
                u = Beps(forBeps[j]*h, 1.*u)#, *args1)
                u = Abig(forAbig[j]*h, 1.*u)#, *args2)
            if getall:
                v = 1.* u
                for j in range(p-1,-1,-1): # postprocessor
                    v = Abig(y[j]*h, 1.*v)#, *args1)
                    v = Beps(z[j]*h, 1.*v)#, *args2)

                uu += [v]
                t+= [t[-1]+h]

        for j in range(p-1,-1,-1): # postprocessor
            u = Beps(y[j]*h, 1.*u)#, *args1)
            u = Abig(z[j]*h, 1.*u)#, *args2)

        #if getall:
        #    uu += [u]
        #    t+= [t[-1]+h]

        if getall: return array(t), array(uu)
        else: return u
