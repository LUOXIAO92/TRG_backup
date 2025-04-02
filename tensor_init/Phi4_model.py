import configparser
import gc
import sys
from time import time

import cupy as cp
import numpy as np
from mpi4py import MPI
from opt_einsum import contract
from utility.randomized_svd import rsvd

#simul_confi = configparser.ConfigParser()
#simul_confi.read(sys.argv[1])

comm = MPI.COMM_WORLD 
myrank = comm.Get_rank() 
nproc = comm.Get_size() 
name = MPI.Get_processor_name() 
cuda = cp.cuda.Device(myrank)
cuda.use()

class gauss_hermite_quadrature():
    SAMPLE_NUM = 64

    def svd(self, M, Dcut):
        U1, s1, VH1 = cp.linalg.svd(M)
        U1 = U1[:,:Dcut]
        s1 = s1[:Dcut]
        VH1 = VH1[:Dcut,:]
        return U1, s1, VH1

    def phi4_init_tensor_component(self, m, mu, lam, Dcut):
        rs = cp.random.RandomState(seed=5678)

        from scipy.special import roots_hermite
        a, wa = roots_hermite(self.SAMPLE_NUM)
        b, wb = roots_hermite(self.SAMPLE_NUM)
        a  = cp.asarray(a,  dtype=cp.float64)
        wa = cp.asarray(wa, dtype=cp.float64)
        b  = cp.asarray(b,  dtype=cp.float64)
        wb = cp.asarray(wb, dtype=cp.float64)

        def cal_phi(a,b):
            return (a+1j*b)/cp.sqrt(2)
        def cal_phi2(a,b):
            return (a**2+b**2)/2
        def cal_phi4(a,b):
            return ((a**2+b**2)/2)**2

        phi  = cp.asarray([ [ cal_phi(aa,bb)  for bb in a] for aa in a], dtype=cp.complex128)
        phi2 = cp.asarray([ [ cal_phi2(aa,bb) for bb in a] for aa in a], dtype=cp.complex128)
        phi4 = cp.asarray([ [ cal_phi4(aa,bb) for bb in a] for aa in a], dtype=cp.complex128)

        m2 = m**2
        I = cp.ones(self.SAMPLE_NUM, dtype=cp.complex128)
        term1 = contract("ab,c,d->abcd", phi2, I, I) + contract("a,b,cd->abcd", I, I, phi2)
        term1 = -(1+m2/4.0) * term1

        term2 = contract("ab,c,d->abcd", phi4, I, I) + contract("a,b,cd->abcd", I, I, phi4)
        term2 = -(lam/4.0) * term2

        term3 = contract("ab,cd->abcd", cp.conj(phi), phi)
        term4 = contract("ab,cd->abcd", phi, cp.conj(phi))

        M1 = term1 + term2 + term3 + term4
        M1 = cp.exp(M1)
        M1 = cp.reshape(M1, (self.SAMPLE_NUM*self.SAMPLE_NUM, self.SAMPLE_NUM*self.SAMPLE_NUM))
        #print(cp.linalg.norm(M1-cp.conj(M1.T)))
        #U1, s1, VH1 = rsvd(M1, k=Dcut, n_oversamples=10, seed=rs)
        U1, s1, VH1 = self.svd(M1, Dcut)
        U1  = contract("ai,i->ai",  U1, cp.sqrt(s1))
        VH1 = contract("ia,i->ia", VH1, cp.sqrt(s1))
        U1  = cp.reshape(U1 , (self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH1 = cp.reshape(VH1, (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM))
        del M1

        term3 = cp.exp(mu)*contract("ab,cd->abcd", cp.conj(phi), phi)
        term4 = cp.exp(-mu)*contract("ab,cd->abcd", phi, cp.conj(phi))

        M2 = term1 + term2 + term3 + term4
        M2 = cp.exp(M2)
        M2 = cp.reshape(M2, (self.SAMPLE_NUM*self.SAMPLE_NUM, self.SAMPLE_NUM*self.SAMPLE_NUM))
        #print(cp.linalg.norm(M2-cp.conj(M2.T)))
        #U2, s2, VH2= rsvd(M2, k=Dcut, n_oversamples=10, seed=rs)
        U2, s2, VH2 = self.svd(M2, Dcut)
        U2  = contract("ai,i->ai",  U2, cp.sqrt(s2))
        VH2 = contract("ia,i->ia", VH2, cp.sqrt(s2))
        U2  = cp.reshape(U2 , (self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH2 = cp.reshape(VH2, (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM))
        del M2

        lnwa = cp.log(wa)
        #exp: factor 
        exp = lnwa + a**2
        exp = contract("a,b->ab", exp, I) + contract("a,b->ab", I, exp)
        exp = cp.exp(exp)

        U  = [U1,U2]
        VH = [VH1,VH2]
        #T = contract("ab,iab,abj,kab,abl->ijkl", exp, VH1, U1, VH2, U2)
        #T_imp0 = contract("ab,ab,iab,abj,iab,abj", gw, cp.conj(phi), VH1, U1, VH2, U2)
        return U, VH, exp, phi