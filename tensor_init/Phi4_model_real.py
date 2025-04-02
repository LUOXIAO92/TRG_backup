import gc
import sys
import time 

import cupy as cp
import numpy as np
import opt_einsum as oe

class gauss_hermite_quadrature:
    def __init__(self, nth, mu02, lam, h, Dcut):
        self.nth  = nth
        self.mu02 = mu02
        self.lam  = lam
        self.h    = h
        self.Dcut = Dcut

        self.SAMPLE_NUM = int(nth**2)

    def svd(self, M, Dcut):
        U, s, VH = cp.linalg.svd(M)
        U = U[:,:Dcut]
        s = s[:Dcut]
        VH = VH[:Dcut,:]
        
        print("s=", s)

        U  = oe.contract("ai,i->ai", U , cp.sqrt(s))
        VH = oe.contract("ia,i->ia", VH, cp.sqrt(s))
        del s

        return U, VH

    def phi4_init_tensor_component(self):
        rs = cp.random.RandomState(seed=5678)
        print("μ0^2=",self.mu02, "λ=",self.lam, "h={:e}".format(self.h))

        from scipy.special import roots_hermite
        phi, w = roots_hermite(self.nth)
        phi = cp.asarray(phi)
        #print("φ",phi)
        #print("w",w)
        
        term1 = cp.asarray([[ (phi1-phi2)**2 for phi2 in phi] for phi1 in phi])
        term1 = -0.5 * term1

        I = cp.ones_like(phi)
        term2 = oe.contract("a,b->ab", phi**2, I) + oe.contract("a,b->ab", I, phi**2)
        term2 = -(self.mu02 / 8) * term2

        term3 = oe.contract("a,b->ab", phi**4, I) + oe.contract("a,b->ab", I, phi**4)
        term3 = -(self.lam / 16) * term3

        term4 = oe.contract("a,b->ab", phi, I) + oe.contract("a,b->ab", I, phi)
        term4 = (self.h / 4) * term4

        M = term1 + term2 + term3 + term4
        M = cp.exp(M).astype(cp.complex128)
        print("M hermit err", cp.linalg.norm(M-cp.conj(M.T)))
        #u, s1, vh = rsvd(M, k=Dcut, n_oversamples=10, seed=rs)
        u, vh = self.svd(M, self.Dcut)
        
        del M

        w   = cp.asarray(w)
        lnw = cp.log(w)
        #exp: factor 
        #exp = lnw + phi**2
        #exp = cp.exp(exp)
        exp = w*cp.exp(phi**2)

        U  = [u,u]
        VH = [vh,vh]
        return U, VH, exp, phi