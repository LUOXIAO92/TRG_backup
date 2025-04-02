import gc
import numpy as np
import cupy as cp
import opt_einsum as oe

from utility.randomized_svd import rsvd
from time import time

from tensor_class.tensor_class import ATRG_Tensor as Tensor

class SU3_spin_model_initialize():
    def __init__(self, K:int, dim:int, Dcut:int, beta:float, h=0.0, mu=0.0):
        self.K = K
        self.Dcut = Dcut
        self.dim  = dim
        self.beta = beta
        self.h    = h
        self.mu   = mu

    def Trace_of_SU3_matrix(self):
        Kp1 = self.K
        Kp2 = self.K

        from scipy.special import roots_legendre
        p1, w1 = roots_legendre(Kp1)
        p2, w2 = roots_legendre(Kp2)

        p1 = np.asarray(np.pi * p1)
        p2 = np.asarray(np.pi * p2)

        Ip1 = np.ones(shape=Kp1, dtype=complex)
        Ip2 = np.ones(shape=Kp2, dtype=complex)

        TrU = oe.contract("i,j->ij", np.exp(1j*p1), Ip2) + oe.contract("i,j->ij", Ip1, np.exp(1j*p2)) \
            + oe.contract("i,j->ij", np.exp(-1j*p1), np.exp(-1j*p2))
        I = np.ones_like(TrU)
        
        #Jacobian, normalized factor=(8/(3π^2))
        J1 = oe.contract("i,j->ij", p1, Ip2) - oe.contract("i,j->ij", Ip1, p2)
        J1 = np.sin(J1/2)**2

        J2 = oe.contract("i,j->ij", 2*p1, Ip2) + oe.contract("i,j->ij", Ip1, p2)
        J2 = np.sin(J2/2)**2

        J3 = oe.contract("i,j->ij", p1, Ip2) + oe.contract("i,j->ij", Ip1, 2*p2)
        J3 = np.sin(J3/2)**2

        J = oe.contract("ij,ij,ij->ij", J1, J2, J3) * 8 / (3 * (np.pi**3))

        #weight
        w = oe.contract("i,j->ij", w1, w2)

        TrU = cp.reshape(TrU, newshape=Kp1*Kp2)
        I = cp.reshape(I, newshape=Kp1*Kp2)
        w = cp.reshape(w, newshape=Kp1*Kp2)
        J = cp.reshape(J, newshape=Kp1*Kp2)

        TrU = cp.asarray(TrU)
        I = cp.asarray(I)
        w = cp.asarray(w)
        J = cp.asarray(J)

        return TrU, w, J, I
    
    def cal_Boltzmann_weight(self, TrU, I):
        """
        direction: spatial or temporal 
        return: Boltzmann weight matrix M_{n,n+ν}
        """

        #M = self.beta * oe.contract("a,b->ab", TrU, cp.conj(TrU)) \
        #  + self.beta * oe.contract("a,b->ab", cp.conj(TrU), TrU) \
        #  + (self.h / (2*self.dim)) * np.exp( self.mu) * (oe.contract("a,b->ab", TrU, I) + oe.contract("a,b->ab", I, TrU)) \
        #  + (self.h / (2*self.dim)) * np.exp(-self.mu) * (oe.contract("a,b->ab", cp.conj(TrU), I) + oe.contract("a,b->ab", I, cp.conj(TrU)))
        #M = cp.exp(M)

        M = self.beta * oe.contract("a,b->ab", TrU, cp.conj(TrU)) \
          + self.beta * oe.contract("a,b->ab", cp.conj(TrU), TrU) 
        #  + (self.h / (2*self.dim)) * np.exp( self.mu) * (oe.contract("a,b->ab", TrU, I) + oe.contract("a,b->ab", I, TrU)) \
        #  + (self.h / (2*self.dim)) * np.exp(-self.mu) * (oe.contract("a,b->ab", cp.conj(TrU), I) + oe.contract("a,b->ab", I, cp.conj(TrU)))
        M = cp.exp(M)

        site = self.h * (cp.exp(self.mu) * TrU + cp.exp(-self.mu) * cp.conj(TrU))
        site = cp.exp(site)

        return M, site
    
    def svd_Boltzmann_weight(self, M, k, del_M=True, split=False):

        #from utility.randomized_svd import rsvd2
        #u, s, vh = rsvd2(A=M, k=k, seed=1234, del_A=del_M)
        u, s, vh = cp.linalg.svd(M)
        u = u[:,:k]
        s = s[:k]
        vh = vh[:k,:]
        if del_M:
            del M

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

    def form_hotrg_tensor(self, J, w, s, Bs, As, Bt, At):

        if self.dim==2:
            subscript = "U,U,U,xU,UX,yU,UY->xXyY"
            T = oe.contract(subscript, J, w, s, Bs, As, Bt, At)

        elif self.dim==3:
            UxX = oe.contract("U,U,U,xU,UX->UxX", J, w, s, Bs, As)
            UyY = oe.contract("yU,UY->UyY", Bs, As)
            UtT = oe.contract("tU,UT->UtT", Bt, At)

            path = [(0, 1, 2)]
            slicing = 100
            n = len(J) // slicing
            T = cp.zeros(shape=(Bs.shape[0], As.shape[1], Bs.shape[0], As.shape[1], Bt.shape[0], At.shape[1]), dtype=complex)
            for i in range(0, len(J), n):
                T += oe.contract("UxX,UyY,UzZ->xXyYzZ", UxX[i:i+n,:,:], UyY[i:i+n,:,:], UtT[i:i+n,:,:], optimize=path)

        elif self.dim==4:
            subscript = "U,U,U,xU,UX,yU,UY,zU,UZ,tU,UT->xXyYzZtT"
            T = oe.contract(subscript, J, w, s, Bs, As, Bs, As, Bs, As, Bt, At)

        return T
    
    def cal_hotrg_init_tensor(self, Dinit, impure=False):
        TrU, w, J, I = self.Trace_of_SU3_matrix()

        Ms, site = self.cal_Boltzmann_weight(TrU, I)
        w = w * site
        As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
        At, Bt = As, Bs

        I = cp.ones_like(w)
        T = self.form_hotrg_tensor(J, w, I, Bs, As, Bt, At)
        if impure:
            return T, J, w, I, Bs, As, Bt, At, TrU
        else:
            return T
    
    def form_ATRG_tensor(self, J, w, O, Bs, As, Bt, At, k:int, is_impure=False, p=0, q=0, seed=1234)->Tensor:
        """
        J: Jacobian determinant of integration. \\
        w: weights of quadrature rules.\\
        O: Operator of impure tensor. All one if making a pure tensor.\\
        Bs, As, Bt, At: Components of spatial and temporal direction.\\
        k: Svd rank or bond dimensions of internal degrees of freedom.\\
        p: Over samples.\\
        q: Times of iteration.\\
        seed: Seed for random standard normal compression matrix.\\
        """

        #ichunks = self.para_chunks if ichunks is None else ichunks
        from tensor_init.ATRG_init import initial_tensor_for_ATRG as init
        U, s, VH = init(dim=self.dim, J=J, w=w*O, 
                        As=As, Bs=Bs, 
                        At=At, Bt=Bt,
                        k=k, 
                        p=p, 
                        q=q, 
                        seed=seed)
        T = Tensor(U, s, VH, Dcut=self.Dcut, dim=self.dim, is_impure=is_impure, loc={})
        return T
    
    def form_ATRG_impureTensor(self, T:Tensor, J, w, O, Bs, As, Bt, At):
        """
        J: Jacobian determinant of integration. \\
        w: weights of quadrature rules.\\
        O: Operator of impure tensor. All one if making a pure tensor.\\
        Bs, As, Bt, At: Components of spatial and temporal direction.\\
        """

        slicing = 100
        n = len(J) // slicing
        simp = cp.zeros(shape=(len(T.s), len(T.s)))
    

    def cal_ATRG_init_tensor(self, Dinit:int, k:int, impure=False, p=0, q=0, seed=1234):
        """
        J: Jacobian determinant of integration. \\
        w: weights of quadrature rules.\\
        k: Svd rank or bond dimensions of internal degrees of freedom.\\
        p: Over samples.\\
        q: Times of iteration.\\
        seed: Seed for random standard normal compression matrix.\\
        impure:
        >>> False: Only return pure tensor.
        >>> True: Return all components
        """

        TrU, w, J, I = self.Trace_of_SU3_matrix()
        Ms, site = self.cal_Boltzmann_weight(TrU, I)
        w = w * site

        As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
        del Ms
        At, Bt = As, Bs

        #ichunks = self.para_chunks if ichunks is None else ichunks
        #from tensor_init.ATRG_init import initial_tensor_for_ATRG as init
        #U, s, VH = init(dim=self.dim, J=J, w=w, 
        #                As=As, Bs=Bs, 
        #                At=At, Bt=Bt,
        #                k=k, 
        #                p=p, 
        #                q=q, 
        #                seed=seed)

        I = cp.ones_like(w)
        T = self.form_ATRG_tensor(J, w, I, Bs, As, Bt, At, k, False, p, q, seed)

        
        #T = Tensor(U, s, VH, dim=self.dim, is_impure=False, loc={})
        if impure:
            return T, J, w, TrU, I, Bs, As, Bt, At
        else:
            return T
        