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

class gauss_legendre_quadrature():
    #if simul_confi["SIMULATIONSETTING"]["nth_Lgd"] != "":
    #    SAMPLE_NUM = int(simul_confi["SIMULATIONSETTING"]["nth_Lgd"])

    SAMPLE_NUM = 26

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

    def __init_tensor_component_parts_finit_density__(self, beta, mu, Dcut):
        time_start=time()

        from scipy.special import roots_legendre
        #b, wphi = self.gl_method_samples_weights(self.SAMPLE_NUM)
        b, wphi = roots_legendre(self.SAMPLE_NUM)
        b = cp.asarray(b, dtype = cp.complex128)
        wphi = cp.asarray(wphi, dtype = cp.complex128)

        #b = cp.arange(-1, 1, 2/self.SAMPLE_NUM, complex)
        #wphi = cp.ones_like(b) * (2/self.SAMPLE_NUM)

        phi = cp.pi * (1+b)
        I = cp.ones_like(phi)
        dphi = cp.einsum("a,b->ab",phi,I) - cp.einsum("a,b->ab",I,phi)
        #dphi = cp.asarray([[phi[i]-phi[j] for j in range(self.SAMPLE_NUM)] for i in range(self.SAMPLE_NUM)], dtype = cp.complex128)
        #dphi = cp.einsum()
        
        cos = cp.cos(dphi+0j*mu)
        M1 = cos
        M1 = cp.exp(beta * M1)
        Dcut = min(Dcut, int(self.SAMPLE_NUM**2))
        #U1, s1, VH1 = rsvd(M1, k=Dcut, n_oversamples=Dcut, n_power_iter=Dcut)
        U1, s1, VH1 = cp.linalg.svd(M1)
        del M1

        U1  = contract("ai,i->ai",  U1, cp.sqrt(s1))
        VH1 = contract("ia,i->ia", VH1, cp.sqrt(s1))

        cos = cp.cos(dphi-1j*mu)
        M2 = cos
        M2 = cp.exp(beta * M2)
        #U2, s2, VH2 = rsvd(M2, k=Dcut, n_oversamples=Dcut, n_power_iter=Dcut)
        U2, s2, VH2 = cp.linalg.svd(M2)
        del M2

        U2  = contract("ai,i->ai",  U2, cp.sqrt(s2))
        VH2 = contract("ia,i->ia", VH2, cp.sqrt(s2))

        time_end=time()
        print("tensor initialization finished, time:{:.6f}s".format(time_end-time_start))

        gc.collect()

        print("s1:",s1[:Dcut])
        print("s2:",s2[:Dcut])

        return U1[:,:Dcut], VH1[:Dcut,:], U2[:,:Dcut], VH2[:Dcut,:], phi, wphi


    #def __init_tensor_component_parts_zero_density__(self, beta, Dcut):
    #    time_start=time()
    #
    #    from scipy.special import roots_legendre
    #    #b, wphi = self.gl_method_samples_weights(self.SAMPLE_NUM)
    #    b, wphi = roots_legendre(self.SAMPLE_NUM)
    #    b = cp.asarray(b, dtype = cp.float64)
    #    wphi = cp.asarray(wphi, dtype = cp.float64)
    #
    #    phi = cp.pi * (1+b)
    #    dphi = cp.asarray([[phi[i]-phi[j] for j in range(self.SAMPLE_NUM)] for i in range(self.SAMPLE_NUM)], dtype = cp.float64)
    #    
    #    cos = cp.cos(dphi)
    #    M1 = cos
    #    M1 = cp.exp(beta * M1)
    #    Dcut = min(Dcut, int(self.SAMPLE_NUM**2))
    #    #U1, s1, VH1 = rsvd(M1, k=Dcut, n_oversamples=10, n_power_iter=2)
    #    U1, s1, VH1 = cp.linalg.svd(M1)
    #    del M1
    #
    #    U1  = contract("ai,i->ai",  U1, cp.sqrt(s1))
    #    VH1 = contract("ia,i->ia", VH1, cp.sqrt(s1))
    #
    #    cos = cp.cos(dphi)
    #    M2 = cos
    #    M2 = cp.exp(beta * M2)
    #    #U2, s2, VH2 = rsvd(M2, k=Dcut, n_oversamples=10, n_power_iter=2)
    #    U2, s2, VH2 = cp.linalg.svd(M2)
    #    del M2
    #
    #    U2  = contract("ai,i->ai",  U2, cp.sqrt(s2))
    #    VH2 = contract("ia,i->ia", VH2, cp.sqrt(s2))
    #
    #    time_end=time()
    #    print("tensor initialization finished, time:{:.6f}s".format(time_end-time_start))
    #
    #    gc.collect()
    #
    #    return U1[:,:Dcut], VH1[:Dcut,:], U2[:,:Dcut], VH2[:Dcut,:], phi, wphi

    
class character_expansion():
    def delta_func(self, x, y):
                if np.abs(x-y) < 1e-12:
                    return 1.0
                else:
                    return 0.0

    def __init_tensor_component_parts_finit_density__(self, beta, mu, Dcut):
        def delta_tensor_index(x1,t1,x2,t2):
            I = np.ones(len(x1))
            delta = np.einsum("a,b,c,d->abcd",x1,I,I,I) - np.einsum("a,b,c,d->abcd",I,x2,I,I)\
                    + np.einsum("a,b,c,d->abcd",I,I,t1,I) - np.einsum("a,b,c,d->abcd",I,I,I,t2)
            return delta
        
        from scipy.special import iv
        n = np.arange(-int(Dcut/2), int(Dcut/2+1))
        I_beta = iv(n, beta)
        delta_index = delta_tensor_index(n, n, n, n)
        exp_mut = np.exp(mu * n)

        return I_beta, exp_mut, delta_index

    def init_impure_tensor_particle_number_2d(self, beta, mu, Dcut, part:int):
        """
        part 1: exp{iθn-iθn+ν}....
        part 2: exp{-iθn+iθn+ν}....
        """
        time_start=time()
        I_beta, exp_mut, delta_index = self.__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

        udelta_func = np.frompyfunc(self.delta_func, 2, 1)

        if part == 1:
            delta_timp0 = udelta_func(delta_index,  1)
            delta_timp1 = udelta_func(delta_index, -1)
        elif part == 2:
            delta_timp0 = udelta_func(delta_index, -1)
            delta_timp1 = udelta_func(delta_index,  1)

        I_beta = cp.asarray(I_beta, dtype = cp.float64)
        exp_mut = cp.asarray(exp_mut, dtype = cp.float64)

        delta_t = udelta_func(delta_index, 0)
        delta_t = cp.asarray(delta_t, dtype = cp.float64)
        delta_timp0 = cp.asarray(delta_timp0, dtype = cp.float64)
        delta_timp1 = cp.asarray(delta_timp1, dtype = cp.float64)

        T = cp.einsum("a,b,c,c,d,d->abcd", I_beta, I_beta, I_beta, exp_mut, I_beta, exp_mut)
        T = cp.sqrt(T)

        T_imp0 = cp.einsum("abcd,abcd->abcd", T, delta_timp0)
        T_imp1 = cp.einsum("abcd,abcd->abcd", T, delta_timp1)
        T = cp.einsum("abcd,abcd->abcd", T, delta_t)

        time_finish=time()
        print("tensor initialization finished, time:{:.6f}s".format(time_finish-time_start))

        return T, T_imp0, T_imp1


    def init_impure_tensor_internal_energy_2d(self, beta, mu, Dcut):
            """
            part 1: exp{iθn-iθn+ν}....
            part 2: exp{-iθn+iθn+ν}....
            """
            time_start=time()
            I_beta, exp_mut, delta_index = self.__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

            udelta_func = np.frompyfunc(self.delta_func, 2, 1)

            delta_timp_p = udelta_func(delta_index,  1)
            delta_timp_m = udelta_func(delta_index, -1)
            

            I_beta = cp.asarray(I_beta, dtype = cp.float64)
            exp_mut = cp.asarray(exp_mut, dtype = cp.float64)

            delta_t = udelta_func(delta_index, 0)
            delta_t = cp.asarray(delta_t, dtype = cp.float64)
            delta_timp_p = cp.asarray(delta_timp_p, dtype = cp.float64)
            delta_timp_m = cp.asarray(delta_timp_m, dtype = cp.float64)

            T = cp.einsum("a,b,c,c,d,d->abcd", I_beta, I_beta, I_beta, exp_mut, I_beta, exp_mut)
            T = cp.sqrt(T)

            T_impp = cp.einsum("abcd,abcd->abcd", T, delta_timp_p)
            T_impm = cp.einsum("abcd,abcd->abcd", T, delta_timp_m)
            T = cp.einsum("abcd,abcd->abcd", T, delta_t)

            time_finish=time()
            print("tensor initialization finished, time:{:.6f}s".format(time_finish-time_start))

            return T, T_impp, T_impm
