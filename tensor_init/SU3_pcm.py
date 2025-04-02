import gc
import numpy as np
import cupy as cp
import opt_einsum as oe

from utility.randomized_svd import rsvd
from time import time

class SU3_pcm_initialize():
    """
    Ktheta: number of θs
    Kphi: number of φs
    Dcut: bond dimension
    dim: dimensional of space-time
    beta: coupling constant
    mu1: chemical potential
    mu2: chemical potential
    mu3: chemical potential

    discretize:
    >>> "part": partition
    >>> "mp": midpoint
    >>> "glq": gauss_legendre_quadrature
    >>> "glq+p": gauss_legendre_quadrature_and_partition
    """
    def __init__(self, Ktheta:int, Kphi:int, dim:int, Dcut:int, 
                 beta:float, h:float, 
                 mu1:float, mu2:float, mu3:float,
                 discretize="glq+p"):
        self.Kt   = Ktheta
        self.Kp   = Kphi
        self.Dcut = Dcut
        self.dim  = dim
        self.beta = beta
        self.h    = h
        self.mu1  = mu1
        self.mu2  = mu2
        self.mu3  = mu3
        self.discretize = discretize

    def gauss_legendre_quadrature(self):
        Kt1 = self.Kt
        Kt2 = self.Kt
        Kt3 = self.Kt
        Kf1 = self.Kp
        Kf2 = self.Kp
        Kf3 = self.Kp
        Kf4 = self.Kp
        Kf5 = self.Kp

        from scipy.special import roots_legendre
        t1, wt1 = roots_legendre(Kt1)
        t2, wt2 = roots_legendre(Kt2)
        t3, wt3 = roots_legendre(Kt3)
        f1, wf1 = roots_legendre(Kf1)
        f2, wf2 = roots_legendre(Kf2)
        f3, wf3 = roots_legendre(Kf3)
        f4, wf4 = roots_legendre(Kf4)
        f5, wf5 = roots_legendre(Kf5)

        return t1, t2, t3, f1, f2, f3, f4, f5, wt1 ,wt2 ,wt3 ,wf1 ,wf2 ,wf3 ,wf4 ,wf5
    
    def partition(self):
        Kt1 = self.Kt
        Kt2 = self.Kt
        Kt3 = self.Kt
        Kf1 = self.Kp
        Kf2 = self.Kp
        Kf3 = self.Kp
        Kf4 = self.Kp
        Kf5 = self.Kp

        t1 = np.linspace(start=-1,stop=1,num=Kt1, endpoint=False)
        t2 = np.linspace(start=-1,stop=1,num=Kt2, endpoint=False)
        t3 = np.linspace(start=-1,stop=1,num=Kt3, endpoint=False)
        f1 = np.linspace(start=-1,stop=1,num=Kf1, endpoint=False)
        f2 = np.linspace(start=-1,stop=1,num=Kf2, endpoint=False)
        f3 = np.linspace(start=-1,stop=1,num=Kf3, endpoint=False)
        f4 = np.linspace(start=-1,stop=1,num=Kf4, endpoint=False)
        f5 = np.linspace(start=-1,stop=1,num=Kf5, endpoint=False)
        wt1 = 2 * np.ones(Kt1) / Kt1
        wt2 = 2 * np.ones(Kt2) / Kt2
        wt3 = 2 * np.ones(Kt3) / Kt3
        wf1 = 2 * np.ones(Kf1) / Kf1
        wf2 = 2 * np.ones(Kf2) / Kf2
        wf3 = 2 * np.ones(Kf3) / Kf3
        wf4 = 2 * np.ones(Kf4) / Kf4
        wf5 = 2 * np.ones(Kf5) / Kf5

        return t1, t2, t3, f1, f2, f3, f4, f5, wt1 ,wt2 ,wt3 ,wf1 ,wf2 ,wf3 ,wf4 ,wf5
    
    def midpoint(self):
        Kt1 = self.Kt
        Kt2 = self.Kt
        Kt3 = self.Kt
        Kf1 = self.Kp
        Kf2 = self.Kp
        Kf3 = self.Kp
        Kf4 = self.Kp
        Kf5 = self.Kp

        t1 = np.linspace(start=-1,stop=1,num=Kt1, endpoint=False) + 1/Kt1
        t2 = np.linspace(start=-1,stop=1,num=Kt2, endpoint=False) + 1/Kt2
        t3 = np.linspace(start=-1,stop=1,num=Kt3, endpoint=False) + 1/Kt3
        f1 = np.linspace(start=-1,stop=1,num=Kf1, endpoint=False) + 1/Kf1
        f2 = np.linspace(start=-1,stop=1,num=Kf2, endpoint=False) + 1/Kf2
        f3 = np.linspace(start=-1,stop=1,num=Kf3, endpoint=False) + 1/Kf3
        f4 = np.linspace(start=-1,stop=1,num=Kf4, endpoint=False) + 1/Kf4
        f5 = np.linspace(start=-1,stop=1,num=Kf5, endpoint=False) + 1/Kf5
        wt1 = 2 * np.ones(Kt1) / Kt1
        wt2 = 2 * np.ones(Kt2) / Kt2
        wt3 = 2 * np.ones(Kt3) / Kt3
        wf1 = 2 * np.ones(Kf1) / Kf1
        wf2 = 2 * np.ones(Kf2) / Kf2
        wf3 = 2 * np.ones(Kf3) / Kf3
        wf4 = 2 * np.ones(Kf4) / Kf4
        wf5 = 2 * np.ones(Kf5) / Kf5

        return t1, t2, t3, f1, f2, f3, f4, f5, wt1 ,wt2 ,wt3 ,wf1 ,wf2 ,wf3 ,wf4 ,wf5

    def gauss_legendre_quadrature_and_partition(self):
        Kt1 = self.Kt
        Kt2 = self.Kt
        Kt3 = self.Kt
        Kf1 = self.Kp
        Kf2 = self.Kp
        Kf3 = self.Kp
        Kf4 = self.Kp
        Kf5 = self.Kp

        from scipy.special import roots_legendre
        t1, wt1 = roots_legendre(Kt1)
        t2, wt2 = roots_legendre(Kt2)
        t3, wt3 = roots_legendre(Kt3)

        f1 = np.linspace(start=-1,stop=1,num=Kf1, endpoint=False)
        f2 = np.linspace(start=-1,stop=1,num=Kf2, endpoint=False)
        f3 = np.linspace(start=-1,stop=1,num=Kf3, endpoint=False)
        f4 = np.linspace(start=-1,stop=1,num=Kf4, endpoint=False)
        f5 = np.linspace(start=-1,stop=1,num=Kf5, endpoint=False)

        wf1 = 2 * np.ones(Kf1) / Kf1
        wf2 = 2 * np.ones(Kf2) / Kf2
        wf3 = 2 * np.ones(Kf3) / Kf3
        wf4 = 2 * np.ones(Kf4) / Kf4
        wf5 = 2 * np.ones(Kf5) / Kf5

        return t1, t2, t3, f1, f2, f3, f4, f5, wt1 ,wt2 ,wt3 ,wf1 ,wf2 ,wf3 ,wf4 ,wf5
    
    def tanh_sinh_quadrature_and_partition(self):
        Kt1 = self.Kt
        Kt2 = self.Kt
        Kt3 = self.Kt
        Kf1 = self.Kp
        Kf2 = self.Kp
        Kf3 = self.Kp
        Kf4 = self.Kp
        Kf5 = self.Kp

        from scipy import optimize
        def tanh_sinh_transform(N):
            def tanhsinh(t):
                return np.tanh((np.pi/2)*np.sinh(t))

            def residue(t, x):
                return tanhsinh(t) - x

            def tanhsinhprime(t, x):
                return (np.pi/2) * np.cosh(t) / (np.cosh(np.pi/2 * np.sinh(t))**2)

            x = np.linspace(start=-1, stop=1, num=N, endpoint=False) + 1/N
            t = np.ones_like(x)
            for i,xi in enumerate(x):
                t[i] = optimize.fsolve(residue, x0=0.0, args=xi, 
                                       fprime=tanhsinhprime)[0]
        
            return t, tanhsinhprime(t, 0), x
        

        a1, wat1, t1 = tanh_sinh_transform(Kt1)
        a2, wat2, t2 = tanh_sinh_transform(Kt2)
        a3, wat3, t3 = tanh_sinh_transform(Kt3)
        wt1 = 2 * np.ones(Kt1) * wat1 / Kt1
        wt2 = 2 * np.ones(Kt2) * wat2 / Kt2
        wt3 = 2 * np.ones(Kt3) * wat3 / Kt3

        f1 = np.linspace(start=-1,stop=1,num=Kf1, endpoint=False)
        f2 = np.linspace(start=-1,stop=1,num=Kf2, endpoint=False)
        f3 = np.linspace(start=-1,stop=1,num=Kf3, endpoint=False)
        f4 = np.linspace(start=-1,stop=1,num=Kf4, endpoint=False)
        f5 = np.linspace(start=-1,stop=1,num=Kf5, endpoint=False)

        wf1 = 2 * np.ones(Kf1) / Kf1
        wf2 = 2 * np.ones(Kf2) / Kf2
        wf3 = 2 * np.ones(Kf3) / Kf3
        wf4 = 2 * np.ones(Kf4) / Kf4
        wf5 = 2 * np.ones(Kf5) / Kf5

        return t1, t2, t3, f1, f2, f3, f4, f5, wt1 ,wt2 ,wt3 ,wf1 ,wf2 ,wf3 ,wf4 ,wf5

    def SU3_matrix(self):
        Kt1 = self.Kt
        Kt2 = self.Kt
        Kt3 = self.Kt
        Kf1 = self.Kp
        Kf2 = self.Kp
        Kf3 = self.Kp
        Kf4 = self.Kp
        Kf5 = self.Kp

        print(f"K_θ1={Kt1}")
        print(f"K_θ2={Kt2}")
        print(f"K_θ3={Kt3}")
        print(f"K_φ1={Kf1}")
        print(f"K_φ2={Kf2}")
        print(f"K_φ3={Kf3}")
        print(f"K_φ4={Kf4}")
        print(f"K_φ5={Kf5}")
        print()

        if self.discretize == "glq+p":
            t1 , t2 , t3 , f1 , f2 , f3 , f4 , f5 , \
            wt1, wt2, wt3, wf1, wf2, wf3, wf4, wf5 = self.gauss_legendre_quadrature_and_partition()
        elif self.discretize == "part":
            t1 , t2 , t3 , f1 , f2 , f3 , f4 , f5 , \
            wt1, wt2, wt3, wf1, wf2, wf3, wf4, wf5 = self.partition()
        elif self.discretize == "mp":
            t1 , t2 , t3 , f1 , f2 , f3 , f4 , f5 , \
            wt1, wt2, wt3, wf1, wf2, wf3, wf4, wf5 = self.midpoint()
        elif self.discretize == "glq":
            t1 , t2 , t3 , f1 , f2 , f3 , f4 , f5 , \
            wt1, wt2, wt3, wf1, wf2, wf3, wf4, wf5 = self.gauss_legendre_quadrature()


        ct1  = np.cos(np.pi * (t1+1) / 4) #cos(θ1)
        ct2  = np.cos(np.pi * (t2+1) / 4) #cos(θ2)
        ct3  = np.cos(np.pi * (t3+1) / 4) #cos(θ3)
        st1  = np.sin(np.pi * (t1+1) / 4) #sin(θ1)
        st2  = np.sin(np.pi * (t2+1) / 4) #sin(θ2)
        st3  = np.sin(np.pi * (t3+1) / 4) #sin(θ3)
        epif1 = np.exp( 1j * np.pi * (f1+1)) #exp( iπφ_1)
        epif2 = np.exp( 1j * np.pi * (f2+1)) #exp( iπφ_2)
        epif3 = np.exp( 1j * np.pi * (f3+1)) #exp( iπφ_3)
        epif4 = np.exp( 1j * np.pi * (f4+1)) #exp( iπφ_4)
        epif5 = np.exp( 1j * np.pi * (f5+1)) #exp( iπφ_5)
        emif1 = np.exp(-1j * np.pi * (f1+1)) #exp(-iπφ_1)
        emif2 = np.exp(-1j * np.pi * (f2+1)) #exp(-iπφ_2)
        emif3 = np.exp(-1j * np.pi * (f3+1)) #exp(-iπφ_3)
        emif4 = np.exp(-1j * np.pi * (f4+1)) #exp(-iπφ_4)
        emif5 = np.exp(-1j * np.pi * (f5+1)) #exp(-iπφ_5)

        It1 = np.ones(shape=Kt1, dtype=complex)
        It2 = np.ones(shape=Kt2, dtype=complex)
        It3 = np.ones(shape=Kt3, dtype=complex)
        If1 = np.ones(shape=Kf1, dtype=complex)
        If2 = np.ones(shape=Kf2, dtype=complex)
        If3 = np.ones(shape=Kf3, dtype=complex)
        If4 = np.ones(shape=Kf4, dtype=complex)
        If5 = np.ones(shape=Kf5, dtype=complex)

        #Uij = Uij(θ1, θ2, θ3, φ1, φ2, φ3, φ4, φ5)
        U = np.zeros(shape=(3, 3, Kt1, Kt2, Kt3, Kf1, Kf2, Kf3, Kf4, Kf5), dtype=complex)
        path = "a,b,c,d,e,f,g,h->abcdefgh"
        U[0,0] =   oe.contract(path, ct1, ct2, It3, epif1, If2,   If3,   If4,   If5)
        U[0,1] =   oe.contract(path, st1, It2, It3, If1,   If2,   epif3, If4,   If5)
        U[0,2] =   oe.contract(path, ct1, st2, It3, If1,   If2,   If3,   epif4, If5)

        U[1,0] =   oe.contract(path, It1, st2, st3, If1,   If2,   If3,   emif4, emif5)\
                 - oe.contract(path, st1, ct2, ct3, epif1, epif2, emif3, If4,   If5)
        U[1,1] =   oe.contract(path, ct1, It2, ct3, If1,   epif2, If3,   If4,   If5)
        U[1,2] = - oe.contract(path, It1, ct2, st3, emif1, If2,   If3,   If4,   emif5)\
                 - oe.contract(path, st1, st2, ct3, If1,   epif2, emif3, epif4, If5)

        U[2,0] = - oe.contract(path, st1, ct2, st3, epif1, If2,   emif3, If4,   epif5)\
                 - oe.contract(path, It1, st2, ct3, If1,   emif2, If3,   emif4, If5)
        U[2,1] =   oe.contract(path, ct1, It2, st3, If1,   If2,   If3,   If4,   epif5)
        U[2,2] =   oe.contract(path, It1, ct2, ct3, emif1, emif2, If3,   If4,   If5)\
                 - oe.contract(path, st1, st2, st3, If1,   If2,   emif3, epif4, epif5)
        

        I = np.zeros_like(U)
        I[0,0] = oe.contract(path, It1, It2, It3, If1, If2, If3, If4, If5)
        I[1,1] = oe.contract(path, It1, It2, It3, If1, If2, If3, If4, If5)
        I[2,2] = oe.contract(path, It1, It2, It3, If1, If2, If3, If4, If5)

        #from itertools import product 
        #iter = product(range(Kt1), range(Kt2), range(Kt3), 
        #               range(Kf1), range(Kf2), range(Kf3), range(Kf4), range(Kf5))
        #for i in iter:
        #    a,b,c,d,e,f,g,h = i
        #    u = U[:,:,a,b,c,d,e,f,g,h]
        #    I = u @ np.conj(u.T)
        #    detI = np.linalg.det(I)
        #    if np.abs(detI-1) > 1e-13:
        #        print(i)
        
        #Jacobian = 1/(2π^5) sin(θ1)cos^3(θ1) sin(θ2)cos(θ2) sin(θ3)cos(θ3)
        Jt1 = st1 * (ct1**3)
        Jt2 = st2 * ct2
        Jt3 = st3 * ct3
        Jf1 = If1
        Jf2 = If2
        Jf3 = If3
        Jf4 = If4
        Jf5 = If5
        J = oe.contract(path, Jt1, Jt2, Jt3, Jf1, Jf2, Jf3, Jf4, Jf5) * (cp.pi**3 / 128)
        #J = oe.contract(path, Jt1, Jt2, Jt3, Jf1, Jf2, Jf3, Jf4, Jf5) * (cp.pi**3 / 4)

        #weight
        w = oe.contract(path, wt1, wt2, wt3, wf1, wf2, wf3, wf4, wf5)

        U = np.reshape(U, newshape=(3, 3, int((self.Kt**3) * (self.Kp**5))))
        I = np.reshape(I, newshape=(3, 3, int((self.Kt**3) * (self.Kp**5))))
        w = np.reshape(w, newshape=int((self.Kt**3) * (self.Kp**5)))
        J = np.reshape(J, newshape=int((self.Kt**3) * (self.Kp**5)))

        #U = cp.asarray(U)
        #I = cp.asarray(I)
        #w = cp.asarray(w)
        #J = cp.asarray(J)

        return U, w, J, I
    
    def cal_Boltzmann_weight(self, U, I, direction="spatial"):
        """
        direction: spatial or temporal 
        return: Boltzmann weight matrix M
        """

        print(f"β={self.beta}, h={self.h:e}, μ1={self.mu1}, μ2={self.mu2}, μ3={self.mu3}")

        #if direction == "spatial":
        #    M = oe.contract("ija,ijb->ab", U, np.conj(U))
        #    M = np.exp(self.beta * 3 * 2 * M.real)
        #elif direction == "temporal":
        #    M = oe.contract("ija,ijb->ab", U, np.conj(U))
        #    M = np.exp(self.beta * 3 * 2 * M.real)

        if direction == "spatial":
            M =   3 * oe.contract("ija,ijb->ab", U, np.conj(U)) \
                + 3 * oe.contract("ija,ijb->ab", np.conj(U), U) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", U, I) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", I, U)
            M = np.exp(self.beta * M)
        elif direction == "temporal":
            M =   3 * oe.contract("ija,ijb->ab", U, np.conj(U)) \
                + 3 * oe.contract("ija,ijb->ab", np.conj(U), U) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", U, I) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", I, U)
            M = np.exp(self.beta * M)

        #err = np.linalg.norm(M-M0) / np.linalg.norm(M0)
        #print(f"err(M-M0)= {err:.12e}")
        #
        #err = np.linalg.norm(M0-np.conj(M0).T) / np.linalg.norm(M0)
        #print(f"err(M0-M0H)= {err:.12e}")
        #
        #err = np.linalg.norm(M-np.conj(M).T) / np.linalg.norm(M)
        #print(f"err(M-MH)= {err:.12e}")
        #
        #M = cp.asarray(M0)
        #print(f"Size of Boltzmann weight is: {M.shape[0]} x {M.shape[1]}")

        TrM = cp.trace(M)
        print(f"TrM={TrM:.12e}")

        return M 
    
    def svd_Boltzmann_weight(self, M, k, del_M=True, split=False):

        #from utility.randomized_svd import rsvd2
        #u, s, vh = rsvd2(A=M, k=k, n_oversamples=4*k, seed=1234, del_A=del_M)

        #s, u = cp.linalg.eigh(M)
        #s = s[::-1]
        #u = u[:,::-1]
        #vh = cp.conj(u.T)

        u, s , vh = cp.linalg.svd(M)
        
        s = s[:k]
        u = u[:,:k]
        vh = vh[:k,:]

        print("Singular values of Boltzmann weight is:")
        with cp.printoptions(formatter={'float_kind':'{:.6e}'.format}):
            print(s)
            print(f"s_Dcut/s_1 = {s[k-1]/s[0]:.6e}")
        print()

        if split:
            A = oe.contract("ia,a->ia", u, cp.sqrt(s))
            B = oe.contract("a,aj->aj", cp.sqrt(s), vh)
            return A, B
        else:
            return u, s, vh
    
    def cal_init_tensor(self, Dinit, legs="xXyY"):
        U, w, J, I = self.SU3_matrix()
        #w = cp.reshape(w, newshape=int((self.Kt**3) * (self.Kp**5)))
        #J = cp.reshape(J, newshape=int((self.Kt**3) * (self.Kp**5)))

        if (self.mu1 + self.mu2 + self.mu3) <= 1e-12:
            Ms = self.cal_Boltzmann_weight(U, I, direction="spatial")
            As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
            At, Bt = As, Bs
        else:
            raise KeyError(f"Finite density is not supported.")

        if self.dim==2 and len(legs)==4:
            path = f"U,U,xU,UX,yU,UY->{legs}"
            T = oe.contract(path, J, w, Bs, As, Bt, At)

        elif self.dim==3 and len(legs)==6:
            path = f"U,U,xU,UX,yU,UY,tU,UT->{legs}"
            T = oe.contract(path, J, w, Bs, As, Bs, As, Bt, At)
        
        elif self.dim==4 and len(legs)==8:
            path = f"U,U,xU,UX,yU,UY,zU,UZ,tU,UT->{legs}"
            T = oe.contract(path, J, w, Bs, As, Bs, As, Bs, As, Bt, At)

        else:
            raise KeyError(f"{self.dim}-d dose not match the legs {legs}")

        #T *= (cp.pi**3 / 4)

        del As, Bs, At, Bt

        return T
    
    def cal_ATRG_init_tensor(self, Dinit:int, k:int, p=0, q=0, seed=1234):
        """
        J : Jacobian determinant of integration. 
        w : weights of quadrature rules.
        As: Spacial legs of positive direction As_{iX}, As_{iY}, As_{iZ}.
        Bs: Temporal legs of negative direction Bs_{xi}, As_{yi}, As_{zi}.
        At: Spacial leg of positive direction As_{iT}.
        Bt: Temporal leg of negative direction Bs_{ti}.
        k : Svd rank or bond dimensions of internal degrees of freedom.
        p : Over samples.
        q : Times of iteration.
        seed: Seed for random standard normal compression matrix.
        """

        U, w, J, I = self.SU3_matrix()

        #w = cp.reshape(w, newshape=int((self.Kt**3) * (self.Kp**5)))
        #J = cp.reshape(J, newshape=int((self.Kt**3) * (self.Kp**5)))

        if (self.mu1 + self.mu2 + self.mu3) <= 1e-12:
            Ms = self.cal_Boltzmann_weight(U, I, direction="spatial")
            As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
            del Ms
            At, Bt = As, Bs
        else:
            raise KeyError(f"Finite density is not supported.")

        #ichunks = self.para_chunks if ichunks is None else ichunks
        from tensor_init.ATRG_init import initial_tensor_for_ATRG as init
        U, s, VH = init(dim=self.dim, J=J, w=w, 
                        As=As, Bs=Bs, 
                        At=At, Bt=Bt,
                        k=k, 
                        p=p, 
                        q=q, 
                        seed=seed)

        from tensor_class.tensor_class import Tensor
        T = Tensor(U, s, VH)
        
        del As, Bs, At, Bt

        return T