import gc
import sys
import time 

import cupy as cp
import numpy as np
import opt_einsum as oe


class infinite_density:
    def __init__(self, dim, Dcut, q, k, h, mu):
        self.q = q
        self.k = k
        self.h = h
        self.mu = mu
        self.Dcut = Dcut
        self.dim = dim

    def potts_spin(self):
        i = cp.arange(self.q)
        Z = cp.exp(1j*2*cp.pi*i/self.q)
        return Z
    
    def cal_Q(self):
        def krondelta(a:int,b:int):
            return 1 if a==b else 0

        s = cp.arange(self.q)
        t = cp.arange(self.q)

        Q = cp.zeros(shape=(self.q, self.q), dtype=complex)
        for si in range(self.q):
            for tj in range(self.q):
                Q[si,tj] = cp.exp(1j * 2*cp.pi * t[tj] * s[si] / self.q) \
                         * cp.sqrt( (cp.exp(self.k) - 1 + self.q * krondelta(t[tj], 0)) / self.q )
        return Q

    def tensor_component_3d(self):
        I = cp.ones(self.q)
        Z = self.potts_spin()
        M = self.k*cp.eye(self.q) + (self.h/(2*self.dim))*cp.exp( self.mu)*cp.einsum("i,j->ij", Z, I) \
                                  + (self.h/(2*self.dim))*cp.exp(-self.mu)*cp.einsum("i,j->ij", I, cp.conj(Z))
        print(M)
        M = cp.exp(M)


        u, s, vh = cp.linalg.svd(M)

        #u  = cp.einsum("ij,j->ij", u, cp.sqrt(s))
        #vh = cp.einsum("i,ij->ij", cp.sqrt(s), vh)

        #w  = cp.exp((self.h/6)*Z)
        #w  = cp.ones(self.q)
        #w  = cp.exp(self.h*w/6).astype(cp.complex128)
        u  = cp.einsum("ij,j->ij", u, cp.sqrt(s))
        vh = cp.einsum("i,ij->ij", cp.sqrt(s), vh)

        return u, vh, Z
    
    def cal_HOTRG_init_tensor(self, impure=False):
        us, svh, Z = self.tensor_component_3d()

        if self.dim == 2:
            T = oe.contract("xa,aX,ya,aY->xXyY", svh, us, svh, us)
        elif self.dim == 3:
            T = oe.contract("xa,aX,ya,aY,ta,aT->xXyYtT", svh, us, svh, us, svh, us)
        elif self.dim == 4:
            T = oe.contract("xa,aX,ya,aY,za,aZ,ta,aT->xXyYzZtT", svh, us, svh, us, svh, us, svh, us)

        #Q = self.cal_Q()
        #Qs = cp.conj(Q)
        #T = oe.contract("sx,sX,sy,sY,st,sT->xXyYtT", Q, Qs, Q, Qs, Q, Qs)

        if impure:
            return T, us, svh, Z
        else:
            return T
    
    def cal_ATRG_init_tensor(self, impure=False):
        us, svh, Z = self.tensor_component_3d()
        T = oe.contract("xa,aX,ya,aY,ta,aT->xXyYtT", svh, us, svh, us, svh, us)
        T = cp.transpose(T, (5,1,3,4,0,2))
        #T = oe.contract("aT,aX,aY,ta,xa,ya->TXYtxy", us, us, us, svh, svh, svh)
        T = T.reshape((self.q**3, self.q**3))
        U, S, VH = cp.linalg.svd(T)
        U  =  U.reshape((self.q, self.q, self.q, len(S)))
        VH = VH.reshape((len(S), self.q, self.q, self.q))
        del T

        from tensor_class.tensor_class import ATRG_Tensor as Tensor
        T = Tensor(U, S, VH, self.Dcut, self.dim, False, {})

        if impure:
            return T, us, svh, Z
        else:
            return T
    
    def tensor_component_2d(self):
        I = cp.ones(self.q)
        Z = self.potts_spin()
        M = self.k*cp.eye(self.q) + (self.h/4)*cp.einsum("i,j->ij", Z, I) + (self.h/4)*cp.einsum("i,j->ij", I, Z)
        M = cp.exp(M)


        u, s, vh = cp.linalg.svd(M)

        #u  = cp.einsum("ij,j->ij", u, cp.sqrt(s))
        #vh = cp.einsum("i,ij->ij", cp.sqrt(s), vh)

        #w  = cp.exp((self.h/6)*Z)
        #w  = cp.ones(self.q)
        #w  = cp.exp(self.h*w/6).astype(cp.complex128)
        u  = cp.einsum("ij,j->ij", u, cp.sqrt(s))
        vh = cp.einsum("i,ij->ij", cp.sqrt(s), vh)

        return u, vh, Z
    
    def tensor_component_3d_classical(self):
        I = cp.ones(self.q)
        Z = self.potts_spin()
        delta_Z = cp.array([1, 0, 0])
        M = self.k*cp.eye(self.q) + (self.h/6)*cp.einsum("i,j->ij", delta_Z, I) + (self.h/6)*cp.einsum("i,j->ij", I, delta_Z)
        M = cp.exp(M)

        u, s, vh = cp.linalg.svd(M)

        u  = cp.einsum("ij,j->ij", u, cp.sqrt(s)).astype(cp.complex128)
        vh = cp.einsum("i,ij->ij", cp.sqrt(s), vh).astype(cp.complex128)

        return u, vh, Z
    
    def tensor_component_2d_classical(self):
        I = cp.ones(self.q)
        Z = self.potts_spin()
        delta_Z = cp.array([1, 0, 0])
        M = self.k*cp.eye(self.q) + (self.h/4)*cp.einsum("i,j->ij", delta_Z, I) + (self.h/4)*cp.einsum("i,j->ij", I, delta_Z)
        M = cp.exp(M)

        u, s, vh = cp.linalg.svd(M)

        u  = cp.einsum("ij,j->ij", u, cp.sqrt(s)).astype(cp.complex128)
        vh = cp.einsum("i,ij->ij", cp.sqrt(s), vh).astype(cp.complex128)

        return u, vh, Z