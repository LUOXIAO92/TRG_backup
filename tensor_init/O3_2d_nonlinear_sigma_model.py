import gc
import numpy as np
import cupy as cp

from utility.randomized_svd import rsvd
from opt_einsum import contract
from time import time
from mpi4py import MPI

import configparser
import sys

#simul_confi = configparser.ConfigParser()
#simul_confi.read(sys.argv[1])

#comm = MPI.COMM_WORLD 
#myrank = comm.Get_rank() 
#nproc = comm.Get_size() 
#name = MPI.Get_processor_name() 
#cuda = cp.cuda.Device(myrank)
#cuda.use()

class gauss_legendre_quadrature():
    #if simul_confi["SIMULATIONSETTING"]["nth_Lgd"] != "":
    #    SAMPLE_NUM = int(simul_confi["SIMULATIONSETTING"]["nth_Lgd"])

    SAMPLE_NUM = 100

    from scipy.special import eval_legendre
    def zeropoint_cal(self, x0, n):
        err = 10e-12
        diff_x1x0 = 1.0
        while diff_x1x0 > err:
            Pnx0 = self.eval_legendre(n, x0)
            Pnm1x0 = self.eval_legendre(n - 1, x0)
            derivative_Pnx0 = n * (Pnm1x0 - x0 * Pnx0) / (1 - x0**2)
            x1 = x0 - Pnx0 / derivative_Pnx0
            diff_x1x0 = np.abs(x1 - x0) 
            x0 = x1

        return x0

    def gl_method_samples_weights(self, SAMPLE_NUM):
        x = np.zeros(SAMPLE_NUM, dtype = np.float64)
        w = np.zeros(SAMPLE_NUM, dtype = np.float64)
        for i in range(SAMPLE_NUM):
            x0 = np.cos(np.pi * (i + 0.75) / (SAMPLE_NUM + 0.5))
            x[i] = self.zeropoint_cal(x0, SAMPLE_NUM)
            w[i] = 2 * (1 - x[i]**2) / ((SAMPLE_NUM * self.eval_legendre(SAMPLE_NUM-1, x[i]))**2)

        x_id = np.argsort(x)
        x = x[x_id]
        w = w[x_id]
        return x, w

#generic operations:
    def delta_function(self, x, y):
        if np.abs(x - y) < 10e-12:
            return 1.0
        else:
            return 0.0

    def svd(self, M, Dcut):
        U, s, VH = cp.linalg.svd(M)
        return U[:,:Dcut], s[:Dcut], VH[:Dcut,:]

    def __init_tensor_component_parts_finit_density__(self, beta, mu, Dcut):
        rs = cp.random.RandomState(seed=3456)
        time_start=time()

        from scipy.special import roots_legendre
        a, wtheta = roots_legendre(self.SAMPLE_NUM)
        b, wphi = roots_legendre(self.SAMPLE_NUM)
        a = cp.asarray(a, dtype = cp.float64)
        b = cp.asarray(b, dtype = cp.float64)
        wtheta = cp.asarray(wtheta, dtype = cp.float64)
        wphi = cp.asarray(wphi, dtype = cp.float64)
        
        I = cp.ones(self.SAMPLE_NUM, dtype = cp.float64)
        sin_theta = cp.sin(cp.pi*(a+1)/2)
        cos_theta = cp.cos(cp.pi*(a+1)/2)
        phi = cp.pi * (1+b)
        #dphi = cp.asarray([[phi[i]-phi[j] for j in range(self.SAMPLE_NUM)] for i in range(self.SAMPLE_NUM)], dtype = cp.float64)
        dphi = contract("a,b->ab", phi, I) - contract("a,b", I, phi)
        
        cos = cp.cos(dphi)
        M1 = contract("a,c,b,d->abcd", cos_theta, cos_theta, I, I) + contract("a,c,bd->abcd",sin_theta, sin_theta, cos)
        M1 = cp.exp(beta * M1)
        M1 = cp.reshape(M1, (M1.shape[0] * M1.shape[1], M1.shape[2] * M1.shape[3]))
        Dcut = min(Dcut, int(self.SAMPLE_NUM**2))
        #U1, s1, VH1 = rsvd(M1, k=Dcut, n_oversamples=100, n_power_iter=0, seed=rs)
        U1, s1, VH1 = self.svd(M1, Dcut)
        del M1

        U1  = cp.reshape( U1, (self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH1 = cp.reshape(VH1, (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U1  = contract("abi,i->abi",  U1, cp.sqrt(s1))
        VH1 = contract("iab,i->iab", VH1, cp.sqrt(s1))

        cos = cp.cos(dphi-1j*mu)
        M2 = contract("a,c,b,d->abcd", cos_theta, cos_theta, I, I) + contract("a,c,bd->abcd",sin_theta, sin_theta, cos)
        M2 = cp.exp(beta * M2)
        M2 = cp.reshape(M2, (M2.shape[0] * M2.shape[1], M2.shape[2] * M2.shape[3]))
        #U2, s2, VH2 = rsvd(M2, k=Dcut, n_oversamples=100, n_power_iter=0, seed=rs)
        U2, s2, VH2 = self.svd(M2, Dcut)
        del M2

        U2  = cp.reshape( U2, (self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH2 = cp.reshape(VH2, (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U2  = contract("abi,i->abi",  U2, cp.sqrt(s2))
        VH2 = contract("iab,i->iab", VH2, cp.sqrt(s2))

        print("s1",s1)
        print("sum(s1)",cp.sum(s1))
        print("s2",s2)
        print("sum(s2)",cp.sum(s2))

        time_end=time()
        print("tensor initialization finished, time:{:.6f}s".format(time_end-time_start))

        gc.collect()

        return U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wtheta, wphi

    