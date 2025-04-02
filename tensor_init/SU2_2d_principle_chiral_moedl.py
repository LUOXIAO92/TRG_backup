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

comm = MPI.COMM_WORLD 
myrank = comm.Get_rank() 
nproc = comm.Get_size() 
name = MPI.Get_processor_name() 

cuda = cp.cuda.Device(myrank)
cuda.use()
njobs = 4


class fibonacci_lattice():

    d_USPHMNF = 3
    #if simul_confi["SIMULATIONSETTING"]["nth_Fib"] != "":
    #    nth_Fib = int(simul_confi["SIMULATIONSETTING"]["nth_Fib"])
    SAMPLE_NUM = 10609
    nth_Fib = 18

#generic operations:
    def gnrl_Fib(self):
        Fib_s = np.zeros(self.d_USPHMNF, dtype = np.int)
        Fib_s[self.d_USPHMNF-1] = 1
        Fib_M = np.array([ [ self.delta_function(i, j - 1) for j in range(self.d_USPHMNF)] 
                                                            for i in range(self.d_USPHMNF)], dtype = np.int)
        Fib_M[len(Fib_M) - 1] = 1
        Fib_M = np.linalg.matrix_power(Fib_M, self.nth_Fib)
        Fib_s = np.matmul(Fib_M, Fib_s)
        return Fib_s

    def delta_function(self, x, y):
        if np.abs(x - y) < 10e-12:
            return 1.0
        else:
            return 0.0

    def __cos_dphi_cal__(self, phii, phij):
        return np.cos(phii-phij)

    def __psi_cal__(self, t):
        psi = 0.6 * np.pi
        err = 10e-12
        dpsi = 1.0
        while np.abs(dpsi) > err:
            f_psi = psi - 0.5 * np.sin(2 * psi) - np.pi * t
            f_psi_prime = 1 - np.cos(2 * psi)
            dpsi = - f_psi / f_psi_prime
            psi += dpsi
        return psi

#operations for zero density:
    def __init_tensor_component_parts_zero_density__(self, beta, Dcut):
        #d=4
        #m=int(d/2)
        #n=5333
        #N=2*n
        #self.SAMPLE_NUM = N
        #from modulo import mod
        #k = np.zeros(2, dtype=np.int64)
        #k[0] = mod(2, 0*int((n-1)/m), n)
        #k[1] = mod(2, 1*int((n-1)/m), n)
        #
        #F = np.array([[ j*i for j in range(n)] for i in k], dtype=np.float64)
        #F = np.exp(2*np.pi*1j*F/n)
        #
        #a = np.zeros(shape=(d, N), dtype=np.float64)
        #a[ :m,  :n] = np.real(F)
        #a[ :m, n:N] = -np.imag(F)
        #a[m:d,  :n] = np.imag(F)
        #a[m:d, n:N] = np.real(F)
        #del F
        #a /= np.sqrt(m)
        #a = cp.asarray(a, dtype=cp.float64)
        #print("||a||_i= ",cp.sum(cp.einsum("ia,ia->a",a,a))/N)

        Fib_s = self.gnrl_Fib()
        self.SAMPLE_NUM = Fib_s[0]
        if myrank == 0:
            print("Generalizations of Fibonacci numbers:",Fib_s)
        
        t1 = cp.asarray([  i / self.SAMPLE_NUM          for i in range(self.SAMPLE_NUM)], dtype = cp.float64)
        t2 = cp.asarray([ (i * Fib_s[1] / Fib_s[0]) % 1 for i in range(self.SAMPLE_NUM)], dtype = cp.float64)
        t3 = cp.asarray([ (i * Fib_s[2] / Fib_s[0]) % 1 for i in range(self.SAMPLE_NUM)], dtype = cp.float64)

        #from modulo import mod
        #self.SAMPLE_NUM = 10597
        #print("sample number:",self.SAMPLE_NUM)
        #z0 = mod(5, int(0*(self.SAMPLE_NUM-1)/6), self.SAMPLE_NUM)
        #z1 = mod(5, int(1*(self.SAMPLE_NUM-1)/6), self.SAMPLE_NUM)
        #z2 = mod(5, int(2*(self.SAMPLE_NUM-1)/6), self.SAMPLE_NUM)
        #print("generating vector:",z0,z1,z2)
        #t1 = cp.asarray([ ((i * z0) % self.SAMPLE_NUM) / self.SAMPLE_NUM for i in range(self.SAMPLE_NUM)], dtype = cp.float64)
        #t2 = cp.asarray([ ((i * z1) % self.SAMPLE_NUM) / self.SAMPLE_NUM for i in range(self.SAMPLE_NUM)], dtype = cp.float64)
        #t3 = cp.asarray([ ((i * z2) % self.SAMPLE_NUM) / self.SAMPLE_NUM for i in range(self.SAMPLE_NUM)], dtype = cp.float64)
        
        psi   = cp.asarray([ self.__psi_cal__(t1_ele) for t1_ele in t1 ], dtype = cp.float64)
        theta = cp.arccos(1 - 2*t2)
        phi   = 2 * cp.pi * t3
        
        cos_psi   = cp.cos(psi)
        sin_psi   = cp.sin(psi)
        cos_theta = cp.cos(theta)
        sin_theta = cp.sin(theta)
        cos_phi   = cp.cos(phi)
        sin_phi   = cp.sin(phi)
        del t1, t2, t3, psi, theta, phi
        
        a = cp.zeros((4, self.SAMPLE_NUM), dtype = cp.float64)
        a[0,:] = cos_psi
        a[1,:] = sin_psi * cos_theta
        a[2,:] = sin_psi * sin_theta * cos_phi
        a[3,:] = sin_psi * sin_theta * sin_phi
        del sin_psi, cos_theta, sin_theta, cos_phi, sin_phi, cos_psi
        print("||a||_i= ",cp.sum(cp.einsum("ia,ia->a",a,a))/self.SAMPLE_NUM)

        M = contract("a,b->ab", a[0,:], a[0,:])\
          + contract("a,b->ab", a[1,:], a[1,:])\
          + contract("a,b->ab", a[2,:], a[2,:])\
          + contract("a,b->ab", a[3,:], a[3,:])

        M = cp.exp(beta * 8 * M)
        U, s, VH = rsvd(M, k=Dcut, iterator="QR")
        U  = contract("ai,i->ai",  U, cp.sqrt(s))
        VH = contract("ia,i->ia", VH, cp.sqrt(s))

        return U, VH, a

    def __impure_tensor_contract_zero_density__(self, a, U, VH):
        T_init_impure = contract("a,ia,aj,ak,la->ijkl", a, VH, U, U, VH) / self.SAMPLE_NUM
        return T_init_impure

    
#operations for finit density:
    def __eigvals_of_chemical_potential_matrix_term__(self, mu1, mu2, direction):
        """returns eigen values and unitary matrix"""
        D = cp.zeros((4,4), dtype = cp.complex128)

        for i in range(4):
            if i == 1 or i == 2:
                D[i,i] = cp.cosh(mu2 * self.delta_function(direction, 1))
            elif i == 0 or i == 3:
                D[i,i] = cp.cosh(mu1 * self.delta_function(direction, 1))
        D[0,3] = - 1j * cp.sinh(mu1 * self.delta_function(direction, 1))
        D[1,2] =   1j * cp.sinh(mu2 * self.delta_function(direction, 1))
        D[2,1] = - 1j * cp.sinh(mu2 * self.delta_function(direction, 1))
        D[3,0] =   1j * cp.sinh(mu1 * self.delta_function(direction, 1))

        #print("nu={:d}".format(direction+1))
        #for i in range(D.shape[0]):
        #    for j in range(D.shape[1]):
        #        print("{:19.12e}".format(D[i,j]), end=" ")
        #    print()

        eigvals, U = cp.linalg.eigh(D)
        return eigvals, U

    def __eigvals_of_impure_chemical_potential_matrix_term_mu__(self, mu):
        """returns eigen values and unitary matrix"""
        U = cp.zeros((4,4), dtype = cp.complex128)
        U[0,0] = -1  / cp.sqrt(2)
        U[3,0] =  1j / cp.sqrt(2)
        U[1,1] =  1j / cp.sqrt(2)
        U[2,1] = -1  / cp.sqrt(2)
        U[1,2] =  1j / cp.sqrt(2)
        U[2,2] =  1  / cp.sqrt(2)
        U[0,3] = -1  / cp.sqrt(2)
        U[3,3] = -1j / cp.sqrt(2)

        eigv = cp.asarray([-cp.exp(-mu), -cp.exp(-mu), cp.exp(mu), cp.exp(mu)], dtype = cp.complex128)
        return eigv, U

    def __eigvals_of_impure_chemical_potential_matrix_term_mu1__(self, mu1):
        """returns eigen values and unitary matrix"""
        U = cp.zeros((4,4), dtype = cp.complex128)
        U[0,0] = -1  / cp.sqrt(2)
        U[3,0] =  1j / cp.sqrt(2)
        U[2,1] =  1
        U[1,2] =  1j
        U[0,3] = -1  / cp.sqrt(2)
        U[3,3] = -1j / cp.sqrt(2)

        eigv = cp.asarray([-cp.exp(-mu1), 0, 0, cp.exp(mu1)], dtype = cp.complex128)
        return eigv, U

    def __eigvals_of_impure_chemical_potential_matrix_term_mu2__(self, mu2):
        """returns eigen values and unitary matrix"""
        U = cp.zeros((4,4), dtype = cp.complex128)
        U[1,0] = -1  / cp.sqrt(2)
        U[2,0] = -1j / cp.sqrt(2)
        U[0,1] =  1
        U[3,2] =  1
        U[1,3] = -1  / cp.sqrt(2)
        U[2,3] =  1j / cp.sqrt(2)

        eigv = cp.asarray([-cp.exp(-mu2), 0, 0, cp.exp(mu2)], dtype = cp.complex128)
        return eigv, U

    def __chemical_potential_matrix_term__(self, mu1, mu2, direction):
        D = cp.zeros((4,4), dtype = cp.complex128)
        for i in range(4):
            if i == 1 or i == 2:
                D[i,i] = cp.cosh(mu2 * self.delta_function(direction, 1))
            elif i == 0 or i == 3:
                D[i,i] = cp.cosh(mu1 * self.delta_function(direction, 1))
            D[0,3] = - 1j * cp.sinh(mu1 * self.delta_function(direction, 1))
            D[1,2] =   1j * cp.sinh(mu2 * self.delta_function(direction, 1))
            D[2,1] = - 1j * cp.sinh(mu2 * self.delta_function(direction, 1))
            D[3,0] =   1j * cp.sinh(mu1 * self.delta_function(direction, 1))
        
        return D

    def __init_tensor_component_parts_finit_density__(self, beta, mu1, mu2, Dcut:int):
        """
        return U:list, VH:list, a:cp.array, b:list\\
        index in the list:the 0th is spacial, and the 1rd is temporal direction\\
        M=U*s*VH in which M=exp{2betaN^2...}, O(4) spin a(n), normalized O(4) spin b(n)=sqrt(eigval)*UH*a(n) in which D=U*eigval*UH
        """
        
        Fib_s = self.gnrl_Fib()
        self.SAMPLE_NUM = Fib_s[0]
        if myrank == 0:
            print("Generalizations of Fibonacci numbers:",Fib_s)

        t1 = np.array([  i / self.SAMPLE_NUM          for i in range(self.SAMPLE_NUM)], dtype = np.float64)
        t2 = np.array([ (i * Fib_s[1] / Fib_s[0]) % 1 for i in range(self.SAMPLE_NUM)], dtype = np.float64)
        t3 = np.array([ (i * Fib_s[2] / Fib_s[0]) % 1 for i in range(self.SAMPLE_NUM)], dtype = np.float64)
        psi   = np.asarray([ self.__psi_cal__(t1_ele) for t1_ele in t1 ], dtype = np.float64)
        theta = np.arccos(1 - 2*t2)
        phi   = 2 * np.pi * t3
        psi   = cp.asarray(psi  , dtype = cp.complex128)
        theta = cp.asarray(theta, dtype = cp.complex128)
        phi   = cp.asarray(phi  , dtype = cp.complex128)

        cos_psi   = cp.cos(psi)
        sin_psi   = cp.sin(psi)
        cos_theta = cp.cos(theta)
        sin_theta = cp.sin(theta)
        cos_phi   = cp.cos(phi)
        sin_phi   = cp.sin(phi)
        del t1, t2, t3, psi, theta, phi

        #eigvals0, eigvacts0 = self.__eigvals_of_chemical_potential_matrix_term__(mu1, mu2, 0)
        #eigvals1, eigvacts1 = self.__eigvals_of_chemical_potential_matrix_term__(mu1, mu2, 1)
        D0 = self.__chemical_potential_matrix_term__(mu1, mu2, 0)
        D1 = self.__chemical_potential_matrix_term__(mu1, mu2, 1)

        a = cp.zeros((4, self.SAMPLE_NUM), dtype = cp.complex128)
        a[0,:] = cos_psi
        a[1,:] = sin_psi * cos_theta
        a[2,:] = sin_psi * sin_theta * cos_phi
        a[3,:] = sin_psi * sin_theta * sin_phi
        del sin_psi, cos_theta, sin_theta, cos_phi, sin_phi, cos_psi

        b = [cp.ndarray] * 2
        b[0] = contract("ij,ja->ia", D0, a)
        b[1] = contract("ij,ja->ia", D1, a)
        #b[0] = contract("i,ji,jn->in", cp.sqrt(eigvals0), cp.conj(eigvacts0), a)
        #b[1] = contract("i,ji,jn->in", cp.sqrt(eigvals1), cp.conj(eigvacts1), a)

        U  = [cp.ndarray] * 2
        VH = [cp.ndarray] * 2
        
        M0 = contract("ia,ib->ab", a, b[0])
        M0 = cp.exp(8 * beta * M0)
        U[0], s0, VH[0] = rsvd(M0, k=Dcut, iterator="QR")
        U[0]  = contract("ai,i->ai",  U[0], cp.sqrt(s0))
        VH[0] = contract("ia,i->ia", VH[0], cp.sqrt(s0))
        del M0, s0

        M1 = contract("ia,ib->ab", a, b[1])
        M1 = cp.exp(8 * beta * M1)
        U[1], s1, VH[1] = rsvd(M1, k=Dcut, iterator="QR")
        U[1]  = contract("ai,i->ai",  U[1], cp.sqrt(s1))
        VH[1] = contract("ia,i->ia", VH[1], cp.sqrt(s1))
        del M1, s1

        return U, VH, a, b

    def __init_impure_tensor__(self, a, U0, VH0, U1, VH1)->cp.ndarray:
        """0:spacial direction; 1:temporal direction"""
        T_init_impure = contract("a,ia,aj,ak,la->ijkl", a, VH0, U0, U1, VH1) / self.SAMPLE_NUM
        return T_init_impure


#tensor for zero density:
    def init_pure_tensor_zero_density(self, beta, Dcut):
        time_start=time()
        U, VH, _ = self.__init_tensor_component_parts_zero_density__(beta, Dcut)
        T_init = contract("ia,aj,ak,la->ijkl", VH, U, U, VH) / self.SAMPLE_NUM
        time_finish=time()
        print("rank{:d}:tensor initialization finished, total time:{:.2f}s".format(myrank, time_finish-time_start))        
        return T_init

    def init_impure_tensor_zero_density(self, beta, Dcut):
        time_start=time()
        
        U, VH, a = self.__init_tensor_component_parts_zero_density__(beta, Dcut)

        T_init_impure = [cp.ndarray] * 4
        for i in range(4):
            T_init_impure[i] = self.__impure_tensor_contract_zero_density__(a[i,:], U, VH)
        T_init_pure = contract("ia,aj,ak,la->ijkl", VH, U, U, VH) / self.SAMPLE_NUM

        time_finish=time()
        print("rank{:d}:tensor initialization finished, total time:{:.2f}s".format(myrank, time_finish-time_start))

        del U, VH, a

        return T_init_pure, T_init_impure

#tensor for finit density: 
    def init_pure_tensor_finit_density(self, beta, mu1, mu2, Dcut):
        time_start=time()

        U, VH, _, _ = self.__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
        T_init_pure = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / self.SAMPLE_NUM
        
        time_finish=time()
        print("rank{:d}:tensor initialization finished, total time:{:.2f}s".format(myrank, time_finish-time_start))
       
        return T_init_pure

    def init_impure_tensor_finit_density(self, beta, mu1, mu2, Dcut:int):
        time_start = time()

        U, VH, a, b = self.__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
        T_init_impure1 = [[cp.ndarray for i in range(4)] for d in range(2)]
        for i in range(4):
            T_init_impure1[0][i] = self.__init_impure_tensor__(a[i,:], U[0], VH[0], U[1], VH[1])
        for i in range(4):
            T_init_impure1[1][i] = self.__init_impure_tensor__(b[0][i,:], U[0], VH[0], U[1], VH[1])

        T_init_impure2 = [[cp.ndarray for i in range(4)] for d in range(2)]
        for i in range(4):
            T_init_impure2[0][i] = self.__init_impure_tensor__(a[i,:], U[0], VH[0], U[1], VH[1])
        for i in range(4):
            T_init_impure2[1][i] = self.__init_impure_tensor__(b[1][i,:], U[0], VH[0], U[1], VH[1])
                
        T_init_pure = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / self.SAMPLE_NUM
        
        time_finish=time()
        print("rank{:d}:tensor initialization finished, total time:{:.2f}s".format(myrank, time_finish-time_start))
        
        return T_init_pure, T_init_impure1, T_init_impure2

    def __chemical_potential_matrix_term_mu__(self, mu):
        D = cp.zeros((4,4), dtype = cp.complex128)
        for i in range(4):
            D[i,i] = cp.sinh(mu)
        D[0,3] = - 1j * cp.cosh(mu)
        D[1,2] =   1j * cp.cosh(mu)
        D[2,1] = - 1j * cp.cosh(mu)
        D[3,0] =   1j * cp.cosh(mu)

        return D
        
    def __chemical_potential_matrix_term_mu1__(self, mu1):
        D = cp.zeros((4,4), dtype = cp.complex128)
        D[0,0] =       cp.sinh(mu1)
        D[0,3] = -1j * cp.cosh(mu1)
        D[3,0] =  1j * cp.cosh(mu1)
        D[3,3] =       cp.sinh(mu1)
        
        return D

    def __chemical_potential_matrix_term_mu2__(self, mu2):
        D = cp.zeros((4,4), dtype = cp.complex128)
        D[1,1] =       cp.sinh(mu2)
        D[1,2] =  1j * cp.cosh(mu2)
        D[2,1] = -1j * cp.cosh(mu2)
        D[2,2] =       cp.sinh(mu2)
        
        return D

    def init_impure_tensor_partical_num(self, beta, mu1, mu2, Dcut:int, mu_or_mu1_or_mu2:str):
        
        #print("tensor initialization begin")
        time_start=time()
        
        U, VH, a, _ = self.__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
        
        if mu_or_mu1_or_mu2 == "mu":
            D = self.__chemical_potential_matrix_term_mu__(mu1)
        elif mu_or_mu1_or_mu2 == "mu1":
            D = self.__chemical_potential_matrix_term_mu1__(mu1)
        elif mu_or_mu1_or_mu2 == "mu2":
            D = self.__chemical_potential_matrix_term_mu2__(mu2)
        else:
            import sys
            print("no such mu")
            sys.exit(1)


        b = contract("ij,ja->ia", D, a)
        
        if mu_or_mu1_or_mu2 == "mu":
            T_init_impure = [ [cp.ndarray for i in range(4)] for d in range(2)]
            for i in range(4):
                T_init_impure[0][i] = self.__init_impure_tensor__(a[i,:], U[0], VH[0], U[1], VH[1]) 
            for i in range(4):
                T_init_impure[1][i] = self.__init_impure_tensor__(b[i,:], U[0], VH[0], U[1], VH[1]) 

        elif mu_or_mu1_or_mu2 == "mu1":
            T_init_impure = [ [cp.ndarray for i in range(2)] for d in range(2)]
            T_init_impure[0][0] = self.__init_impure_tensor__(a[0,:], U[0], VH[0], U[1], VH[1]) 
            T_init_impure[0][1] = self.__init_impure_tensor__(a[3,:], U[0], VH[0], U[1], VH[1]) 
            T_init_impure[1][0] = self.__init_impure_tensor__(b[0,:], U[0], VH[0], U[1], VH[1]) 
            T_init_impure[1][1] = self.__init_impure_tensor__(b[3,:], U[0], VH[0], U[1], VH[1]) 

        elif mu_or_mu1_or_mu2 == "mu2":
            T_init_impure = [ [cp.ndarray for i in range(2)] for d in range(2)]
            T_init_impure[0][0] = self.__init_impure_tensor__(a[1,:], U[0], VH[0], U[1], VH[1]) 
            T_init_impure[0][1] = self.__init_impure_tensor__(a[2,:], U[0], VH[0], U[1], VH[1]) 
            T_init_impure[1][0] = self.__init_impure_tensor__(b[1,:], U[0], VH[0], U[1], VH[1]) 
            T_init_impure[1][1] = self.__init_impure_tensor__(b[2,:], U[0], VH[0], U[1], VH[1]) 
        
        T_init_pure = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / self.SAMPLE_NUM

        time_finish=time()
        print("rank{:d}:tensor initialization finished, total time:{:.2f}s".format(myrank, time_finish-time_start))

        return T_init_pure, T_init_impure


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

    def pure_tensor(self, wa, wb, wc, VH1, U1, VH2, U2):
        T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,abck,labc->ijkl", wa, wb, wc, VH1, U1, VH2, U2)
        return T

    def impure_tensor(self, s, wa, wb, wc, VH1, U1, VH2, U2):
        T = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,abck,labc->ijkl", s, wa, wb, wc, VH1, U1, VH2, U2)
        return T

#operations for finit density:
    def __eigvals_of_chemical_potential_matrix_term__(self, mu1, mu2, direction):
        """returns eigen values and unitary matrix"""

        D = cp.zeros((4,4), dtype = cp.complex128)
        for i in range(4):
            if i == 1 or i == 2:
                D[i,i] = cp.cosh(mu2 * self.delta_function(direction, 1))
            elif i == 0 or i == 3:
                D[i,i] = cp.cosh(mu1 * self.delta_function(direction, 1))
            D[0,3] = - 1j * cp.sinh(mu1 * self.delta_function(direction, 1))
            D[1,2] =   1j * cp.sinh(mu2 * self.delta_function(direction, 1))
            D[2,1] = - 1j * cp.sinh(mu2 * self.delta_function(direction, 1))
            D[3,0] =   1j * cp.sinh(mu1 * self.delta_function(direction, 1))
        eigvals, U = cp.linalg.eigh(D) 

        return eigvals, U

    def __eigvals_of_impure_chemical_potential_matrix_term_mu__(self, mu):
        """returns eigen values and unitary matrix"""
        U = cp.zeros((4,4), dtype = cp.complex128)
        U[0,0] = -1  / cp.sqrt(2)
        U[3,0] =  1j / cp.sqrt(2)
        U[1,1] =  1j / cp.sqrt(2)
        U[2,1] = -1  / cp.sqrt(2)
        U[1,2] =  1j / cp.sqrt(2)
        U[2,2] =  1  / cp.sqrt(2)
        U[0,3] = -1  / cp.sqrt(2)
        U[3,3] = -1j / cp.sqrt(2)

        eigv = cp.asarray([-cp.exp(-mu), -cp.exp(-mu), cp.exp(mu), cp.exp(mu)], dtype = cp.complex128)
        return eigv, U

    def __eigvals_of_impure_chemical_potential_matrix_term_mu1__(self, mu1):
        """returns eigen values and unitary matrix"""
        U = cp.zeros((4,4), dtype = cp.complex128)
        U[0,0] = -1  / cp.sqrt(2)
        U[3,0] =  1j / cp.sqrt(2)
        U[2,1] =  1
        U[1,2] =  1j
        U[0,3] = -1  / cp.sqrt(2)
        U[3,3] = -1j / cp.sqrt(2)

        eigv = cp.asarray([-cp.exp(-mu1), 0, 0, cp.exp(mu1)], dtype = cp.complex128)
        return eigv, U

    def __eigvals_of_impure_chemical_potential_matrix_term_mu2__(self, mu2):
        """returns eigen values and unitary matrix"""
        U = cp.zeros((4,4), dtype = cp.complex128)
        U[1,0] = -1  / cp.sqrt(2)
        U[2,0] = -1j / cp.sqrt(2)
        U[0,1] =  1
        U[3,2] =  1
        U[1,3] = -1  / cp.sqrt(2)
        U[2,3] =  1j / cp.sqrt(2)

        eigv = cp.asarray([-cp.exp(-mu2), 0, 0, cp.exp(mu2)], dtype = cp.complex128)
        return eigv, U

    def __chemical_potential_matrix_term__(self, mu1, mu2, direction):
        D = cp.zeros((4,4), dtype = cp.complex128)
        for i in range(4):
            if i == 1 or i == 2:
                D[i,i] = cp.cosh(mu2 * self.delta_function(direction, 1))
            elif i == 0 or i == 3:
                D[i,i] = cp.cosh(mu1 * self.delta_function(direction, 1))
            D[0,3] = - 1j * cp.sinh(mu1 * self.delta_function(direction, 1))
            D[1,2] =   1j * cp.sinh(mu2 * self.delta_function(direction, 1))
            D[2,1] = - 1j * cp.sinh(mu2 * self.delta_function(direction, 1))
            D[3,0] =   1j * cp.sinh(mu1 * self.delta_function(direction, 1))
        
        return D

    def __svd__(self, M, Dcut):
        U, s, VH = cp.linalg.svd(M)
        return U[:,:Dcut], s[:Dcut], VH[:Dcut,:]

    def __init_tensor_component_parts_finit_density__(self, beta, mu1, mu2, Dcut:int):
        """
        return U:list, VH:list, a:cp.array, b:list, w:list\\
        index in the list:the 0th is spacial, and the 1rd is temporal direction\\
        U, VH, M=U*s*VH in which M=exp{2betaN^2...}\\
        a(n), O(4) spin, normalized O(4) spin b(n)=sqrt(eigval)*UH*a(n) in which D=U*eigval*UH\\
        w, the weight of Gauss Lengendre quadrature
        """

        t0 = time()
        #print("tensor initialization start")
        t = [np.ndarray] * 3
        w = [np.ndarray] * 3
        t[0], w[0] = self.gl_method_samples_weights(self.SAMPLE_NUM)
        t[1], w[1] = self.gl_method_samples_weights(self.SAMPLE_NUM)
        t[2], w[2] = self.gl_method_samples_weights(self.SAMPLE_NUM)
        psi   = np.pi * (t[0] + 1) / 2
        theta = np.pi * (t[1] + 1) / 2
        phi   = np.pi * (t[2] + 1)
        cos_psi   = np.cos(psi)
        sin_psi   = np.sin(psi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi   = np.cos(phi)
        sin_phi   = np.sin(phi)
    
        cos_psi   = cp.asarray(cos_psi,   dtype = cp.complex128)
        sin_psi   = cp.asarray(sin_psi,   dtype = cp.complex128)
        cos_theta = cp.asarray(cos_theta, dtype = cp.complex128)
        sin_theta = cp.asarray(sin_theta, dtype = cp.complex128)
        cos_phi   = cp.asarray(cos_phi,   dtype = cp.complex128)
        sin_phi   = cp.asarray(sin_phi,   dtype = cp.complex128)        

        D0 = self.__chemical_potential_matrix_term__(mu1, mu2, 0)
        D1 = self.__chemical_potential_matrix_term__(mu1, mu2, 1)

        I = cp.ones(self.SAMPLE_NUM, dtype = cp.complex128)
        a = cp.zeros((4, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM), dtype = cp.complex128)
        b = [cp.ndarray] * 2
        a[0,:,:,:] = contract("a,b,c->abc", cos_psi, I, I)
        a[1,:,:,:] = contract("a,b,c->abc", sin_psi, cos_theta, I)
        a[2,:,:,:] = contract("a,b,c->abc", sin_psi, sin_theta, cos_phi)
        a[3,:,:,:] = contract("a,b,c->abc", sin_psi, sin_theta, sin_phi)
        b[0] = contract("ij,jabc->iabc", D0, a)
        b[1] = contract("ij,jabc->iabc", D1, a)

        U  = [cp.ndarray] * 2
        VH = [cp.ndarray] * 2

        M0 = contract("iabc,idef->abcdef", a, b[0])
        M0 = cp.exp(beta * 8 * M0)
        M0 = cp.reshape(M0, (cp.shape(M0)[0] * cp.shape(M0)[1] * cp.shape(M0)[2], cp.shape(M0)[3] * cp.shape(M0)[4] * cp.shape(M0)[5]))
        U[0], s0, VH[0] = rsvd(M0, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR")
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M0
        U[0]  = cp.reshape( U[0], (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[0] = cp.reshape(VH[0], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[0]  = contract("abci,i->abci",  U[0], cp.sqrt(s0))
        VH[0] = contract("iabc,i->iabc", VH[0], cp.sqrt(s0))
        del s0

        #M1 = contract("iabc,idef->abcdef", cp.conj(b[1]), b[1])
        M1 = contract("iabc,idef->abcdef", a, b[1])
        M1 = cp.exp(beta * 8 * M1)
        M1 = cp.reshape(M1, (cp.shape(M1)[0] * cp.shape(M1)[1] * cp.shape(M1)[2], cp.shape(M1)[3] * cp.shape(M1)[4] * cp.shape(M1)[5]))
        U[1], s1, VH[1] = rsvd(M1, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR")
        #U[1], s1, VH[1] = self.__svd__(M1, Dcut)
        del M1
        U[1]  = cp.reshape( U[1], (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[1] = cp.reshape(VH[1], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[1]  = contract("abci,i->abci",  U[1], cp.sqrt(s1))
        VH[1] = contract("iabc,i->iabc", VH[1], cp.sqrt(s1))
        del s1

        w[0] = contract("a,a->a", w[0], cp.square(sin_psi))
        w[1] = contract("b,b->b", w[1], sin_theta)
        w[0] = cp.asarray(w[0], dtype = cp.complex128)
        w[1] = cp.asarray(w[1], dtype = cp.complex128)
        w[2] = cp.asarray(w[2], dtype = cp.complex128)

        t1 = time()
        print("Tensor initialization finished. Time= {:.6f} s".format(t1-t0))

        return U, VH, a, b, w

    def __init_tensor_component_parts_finit_density_4TRG__(self, beta:float, mu1:float, mu2:float, Dcut:int):
        t0 = time()
        
        from scipy.special import roots_legendre

        t = [np.ndarray] * 3
        w = [np.ndarray] * 3
        t[0], w[0] = roots_legendre(self.SAMPLE_NUM)
        t[1], w[1] = roots_legendre(self.SAMPLE_NUM)
        t[2], w[2] = roots_legendre(self.SAMPLE_NUM)

        theta = cp.asarray(np.pi * (t[0] + 1) / 4)
        a     = cp.asarray(np.pi * (t[1] + 1))
        b     = cp.asarray(np.pi * (t[2] + 1))

        exp_pitheta = cp.exp(1j*theta)
        exp_pia = cp.exp(1j*a)
        exp_pib = cp.exp(1j*b)
        exp_mitheta = cp.exp(-1j*theta)
        exp_mia = cp.exp(-1j*a)
        exp_mib = cp.exp(-1j*b)
        I = cp.ones(self.SAMPLE_NUM)

        D = cp.asarray([[1/2,1/2,0,0],[0,0,1/2,1/2]], dtype=cp.complex128)
        #s_{ij}=θ×α×β
        s = cp.zeros((4,2,self.SAMPLE_NUM,self.SAMPLE_NUM,self.SAMPLE_NUM), dtype=cp.complex128)
        s[0,0] =     contract("t,a,b->tab", exp_pitheta, exp_pia, I)
        s[1,0] =     contract("t,a,b->tab", exp_mitheta, exp_pia, I)
        s[0,1] = -1j*contract("t,a,b->tab", exp_pitheta, I, exp_pib)
        s[1,1] =  1j*contract("t,a,b->tab", exp_mitheta, I, exp_pib)
        s[2,0] =  1j*contract("t,a,b->tab", exp_pitheta, I, exp_mib)
        s[3,0] = -1j*contract("t,a,b->tab", exp_mitheta, I, exp_mib)
        s[2,1] =     contract("t,a,b->tab", exp_pitheta, exp_mia, I)
        s[3,1] =     contract("t,a,b->tab", exp_mitheta, exp_mia, I)
        
        u = cp.zeros((2,2,self.SAMPLE_NUM,self.SAMPLE_NUM,self.SAMPLE_NUM), dtype=cp.complex128)
        u = contract("ij,jkabc->ikabc", D, s)

        lam = (mu1-mu2) / 2
        gam = (mu1+mu2) / 2

        e1_lam = cp.zeros((2,2), dtype=cp.complex128)
        e1_gam = cp.zeros((2,2), dtype=cp.complex128)
        e2_lam = cp.zeros((2,2), dtype=cp.complex128)
        e2_gam = cp.zeros((2,2), dtype=cp.complex128)

        e1_lam[0,0], e1_lam[1,1] = cp.exp(lam), cp.exp(-lam)
        e1_gam[0,0], e1_gam[1,1] = cp.exp(gam), cp.exp(-gam)
        e2_lam[0,0], e2_lam[1,1] = cp.exp(-lam), cp.exp(lam)
        e2_gam[0,0], e2_gam[1,1] = cp.exp(-gam), cp.exp(gam)
        e_zero = cp.asarray([[1.0,0.0],[0.0,1.0]], dtype=cp.complex128)

        f = [cp.ndarray] * 2
        g = [cp.ndarray] * 2
        f[0] = contract("ci,ij,ja,bd->abcd", cp.conj(D.T), e_zero, D, e_zero)
        f[1] = contract("ci,ij,ja,bd->abcd", cp.conj(D.T), e1_gam, D, e1_lam)
        g[0] = contract("ai,ij,jc,db->abcd", cp.conj(D.T), e_zero, D, e_zero)
        g[1] = contract("ai,ij,jc,db->abcd", cp.conj(D.T), e2_gam, D, e2_lam)

        U  = [cp.ndarray] * 2
        VH = [cp.ndarray] * 2

        M0 = contract("ij,jkabc,kl,ildef->abcdef", e_zero, u, e_zero, cp.conj(u)) \
           + contract("ij,kjabc,kl,lidef->abcdef", e_zero, cp.conj(u), e_zero, u)
        M0 = cp.exp(beta * 2 * M0)
        M0 = cp.reshape(M0, (cp.shape(M0)[0] * cp.shape(M0)[1] * cp.shape(M0)[2], cp.shape(M0)[3] * cp.shape(M0)[4] * cp.shape(M0)[5]))
        U[0], s0, VH[0] = rsvd(M0, k=Dcut, n_oversamples=12*Dcut, n_power_iter=2, iterator="QR",seed=cp.random.RandomState(1234))
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M0
        U[0]  = cp.reshape( U[0], (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[0] = cp.reshape(VH[0], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[0]  = contract("abci,i->abci",  U[0], cp.sqrt(s0))
        VH[0] = contract("iabc,i->iabc", VH[0], cp.sqrt(s0))
        del s0

        M1 = contract("ij,jkabc,kl,ildef->abcdef", e1_gam, u, e1_lam, cp.conj(u)) \
           + contract("ij,kjabc,kl,lidef->abcdef", e2_lam, cp.conj(u), e2_gam, u)
        M1 = cp.exp(beta * 2 * M1)
        M1 = cp.reshape(M1, (cp.shape(M1)[0] * cp.shape(M1)[1] * cp.shape(M1)[2], cp.shape(M1)[3] * cp.shape(M1)[4] * cp.shape(M1)[5]))
        U[1], s1, VH[1] = rsvd(M1, k=Dcut, n_oversamples=12*Dcut, n_power_iter=2, iterator="QR",seed=cp.random.RandomState(1234))
        #U[1], s1, VH[1] = self.__svd__(M1, Dcut)
        del M1
        U[1]  = cp.reshape( U[1], (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[1] = cp.reshape(VH[1], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[1]  = contract("abci,i->abci",  U[1], cp.sqrt(s1))
        VH[1] = contract("iabc,i->iabc", VH[1], cp.sqrt(s1))
        del s1

        w[0] = contract("a,a,a->a", cp.sin(theta), cp.cos(theta), w[0])
        w[0] = cp.asarray(w[0], dtype = cp.complex128)
        w[1] = cp.asarray(w[1], dtype = cp.complex128)
        w[2] = cp.asarray(w[2], dtype = cp.complex128)

        t1 = time()
        print("Tensor initialization finished. Time= {:.6f} s".format(t1-t0))

        return U, VH, s, w, f, g

    def __init_impure_tensor__(self, a, w, U0, VH0, U1, VH1):
        T_init_impure = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a, w[0], w[1], w[2], VH0, U0, VH1, U1)
        return T_init_impure

    def __init_pure_tensor__(self, w, U0, VH0, U1, VH1):
        T_init_pure = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH0, U0, VH1, U1)
        return T_init_pure
    
    
#tensor for zero density:
    def __chemical_potential_matrix_term_mu__(self, mu):
        D = cp.zeros((4,4), dtype = cp.complex128)
        for i in range(4):
            D[i,i] = cp.sinh(mu)
        D[0,3] = - 1j * cp.cosh(mu)
        D[1,2] =   1j * cp.cosh(mu)
        D[2,1] = - 1j * cp.cosh(mu)
        D[3,0] =   1j * cp.cosh(mu)

        return D
        
    def __chemical_potential_matrix_term_mu1__(self, mu1):
        D = cp.zeros((4,4), dtype = cp.complex128)
        D[0,0] =       cp.sinh(mu1)
        D[0,3] = -1j * cp.cosh(mu1)
        D[3,0] =  1j * cp.cosh(mu1)
        D[3,3] =       cp.sinh(mu1)
        
        return D

    def __chemical_potential_matrix_term_mu2__(self, mu2):
        D = cp.zeros((4,4), dtype = cp.complex128)
        D[1,1] =       cp.sinh(mu2)
        D[1,2] =  1j * cp.cosh(mu2)
        D[2,1] = -1j * cp.cosh(mu2)
        D[2,2] =       cp.sinh(mu2)
        
        return D



    def __init_tensor_component_parts_finit_density_test__(self, beta, mu1, mu2, Dcut:int):
        """
        return U:list, VH:list, a:cp.array, b:list, w:list\\
        index in the list:the 0th is spacial, and the 1rd is temporal direction\\
        U, VH, M=U*s*VH in which M=exp{2betaN^2...}\\
        a(n), O(4) spin, normalized O(4) spin b(n)=sqrt(eigval)*UH*a(n) in which D=U*eigval*UH\\
        w, the weight of Gauss Lengendre quadrature
        """

        t0 = time()
        #print("tensor initialization start")
        t = [np.ndarray] * 3
        w = [np.ndarray] * 3
        t[0], w[0] = self.gl_method_samples_weights(self.SAMPLE_NUM)
        t[1], w[1] = self.gl_method_samples_weights(self.SAMPLE_NUM)
        t[2], w[2] = self.gl_method_samples_weights(self.SAMPLE_NUM)
        psi   = np.pi * (t[0] + 1) / 2
        theta = np.pi * (t[1] + 1) / 2
        phi   = np.pi * (t[2] + 1)
        cos_psi   = np.cos(psi)
        sin_psi   = np.sin(psi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi   = np.cos(phi)
        sin_phi   = np.sin(phi)
    
        cos_psi   = cp.asarray(cos_psi,   dtype = cp.complex128)
        sin_psi   = cp.asarray(sin_psi,   dtype = cp.complex128)
        cos_theta = cp.asarray(cos_theta, dtype = cp.complex128)
        sin_theta = cp.asarray(sin_theta, dtype = cp.complex128)
        cos_phi   = cp.asarray(cos_phi,   dtype = cp.complex128)
        sin_phi   = cp.asarray(sin_phi,   dtype = cp.complex128)        

        D0 = self.__chemical_potential_matrix_term__(mu1, mu2, 0)
        D1 = self.__chemical_potential_matrix_term__(mu1, mu2, 1)

        I = cp.ones(self.SAMPLE_NUM, dtype = cp.complex128)
        a = cp.zeros((4, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM), dtype = cp.complex128)
        b = [cp.ndarray] * 2
        a[0,:,:,:] = contract("a,b,c->abc", cos_psi, I, I)
        a[1,:,:,:] = contract("a,b,c->abc", sin_psi, cos_theta, I)
        a[2,:,:,:] = contract("a,b,c->abc", sin_psi, sin_theta, cos_phi)
        a[3,:,:,:] = contract("a,b,c->abc", sin_psi, sin_theta, sin_phi)
        b[0] = contract("ij,jabc->iabc", D0, a)
        b[1] = contract("ij,jabc->iabc", D1, a)

        U  = [cp.ndarray] * 2
        VH = [cp.ndarray] * 2

        M0 = contract("iabc,idef->abcdef", a, b[0])
        M0 = cp.exp(beta * 8 * M0)
        M0 = cp.reshape(M0, (cp.shape(M0)[0] * cp.shape(M0)[1] * cp.shape(M0)[2], cp.shape(M0)[3] * cp.shape(M0)[4] * cp.shape(M0)[5]))
        U[0], s0, VH[0] = rsvd(M0, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR",seed=cp.random.RandomState(1234))
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M0
        U[0]  = cp.reshape( U[0], (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[0] = cp.reshape(VH[0], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[0]  = contract("abci,i->abci",  U[0], cp.sqrt(s0))
        VH[0] = contract("iabc,i->iabc", VH[0], cp.sqrt(s0))
        del s0

        M1 = contract("iabc,idef->abcdef", cp.conj(b[1]), b[1])
        M1 = contract("iabc,idef->abcdef", a, b[1])
        M1 = cp.exp(beta * 8 * M1)
        M1 = cp.reshape(M1, (cp.shape(M1)[0] * cp.shape(M1)[1] * cp.shape(M1)[2], cp.shape(M1)[3] * cp.shape(M1)[4] * cp.shape(M1)[5]))
        U[1], s1, VH[1] = rsvd(M1, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR",seed=cp.random.RandomState(1234))
        #U[1], s1, VH[1] = self.__svd__(M1, Dcut)
        del M1
        U[1]  = cp.reshape( U[1], (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[1] = cp.reshape(VH[1], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[1]  = contract("abci,i->abci",  U[1], cp.sqrt(s1))
        VH[1] = contract("iabc,i->iabc", VH[1], cp.sqrt(s1))
        del s1

        w[0] = contract("a,a->a", w[0], cp.square(sin_psi))
        w[1] = contract("b,b->b", w[1], sin_theta)
        w[0] = cp.asarray(w[0], dtype = cp.complex128)
        w[1] = cp.asarray(w[1], dtype = cp.complex128)
        w[2] = cp.asarray(w[2], dtype = cp.complex128)

        t1 = time()
        print("Tensor initialization finished. Time= {:.6f} s".format(t1-t0))

        return U, VH, a, b, w


    def __init_tensor_component_parts_zero_density_4TRG__(self, beta, Dcut:int):
        t0 = time()
        
        from scipy.special import roots_legendre

        t = [np.ndarray] * 3
        w = [np.ndarray] * 3
        t[0], w[0] = roots_legendre(self.SAMPLE_NUM)
        t[1], w[1] = roots_legendre(self.SAMPLE_NUM)
        t[2], w[2] = roots_legendre(self.SAMPLE_NUM)

        theta = cp.asarray(np.pi * (t[0] + 1) / 4)
        a     = cp.asarray(np.pi * (t[1] + 1))
        b     = cp.asarray(np.pi * (t[2] + 1))

        exp_pia = cp.exp(1j*a)
        exp_pib = cp.exp(1j*b)
        exp_mia = cp.exp(-1j*a)
        exp_mib = cp.exp(-1j*b)
        sin_theta = cp.sin(theta)
        cos_theta = cp.cos(theta)
        I = cp.ones(self.SAMPLE_NUM)

        S = cp.zeros((4,2,self.SAMPLE_NUM,self.SAMPLE_NUM,self.SAMPLE_NUM), dtype=cp.complex128)
        D = cp.asarray([[1/2,1/2,0,0],[0,0,1/2,1/2]], dtype=cp.float64)
        DdagD = D.T @ D
        exp_pitheta = cp.exp(1j*theta)
        exp_mitheta = cp.exp(-1j*theta)
        S[0,0] =     contract("t,a,b->tab", exp_pitheta, exp_pia, I)
        S[1,0] =     contract("t,a,b->tab", exp_mitheta, exp_pia, I)
        S[0,1] = -1j*contract("t,a,b->tab", exp_pitheta, I, exp_pib)
        S[1,1] =  1j*contract("t,a,b->tab", exp_mitheta, I, exp_pib)
        S[2,0] =  1j*contract("t,a,b->tab", exp_pitheta, I, exp_mib)
        S[3,0] = -1j*contract("t,a,b->tab", exp_mitheta, I, exp_mib)
        S[2,1] =     contract("t,a,b->tab", exp_pitheta, exp_mia, I)
        S[3,1] =     contract("t,a,b->tab", exp_mitheta, exp_mia, I)
        
        #u1 = cp.zeros((2,2,self.SAMPLE_NUM,self.SAMPLE_NUM,self.SAMPLE_NUM), dtype=cp.complex128)
        #u1[0,0] =  contract("t,a,b->tab", cos_theta, exp_pia, I)
        #u1[0,1] =  contract("t,a,b->tab", sin_theta, I, exp_pib)
        #u1[1,0] = -contract("t,a,b->tab", sin_theta, I, exp_mib)
        #u1[1,1] =  contract("t,a,b->tab", cos_theta, exp_mia, I)

        u = cp.zeros((2,2,self.SAMPLE_NUM,self.SAMPLE_NUM,self.SAMPLE_NUM), dtype=cp.complex128)
        u = contract("ij,jkabc->ikabc", D, S)

        e_zero = cp.asarray([[1.0,0.0],[0.0,1.0]], dtype=cp.complex128)
        M0 = contract("ij,jkabc,kl,ildef->abcdef", e_zero, u, e_zero, cp.conj(u)) \
           + contract("ij,kjabc,kl,lidef->abcdef", e_zero, cp.conj(u), e_zero, u)

        #M0 = contract("ij,ikabc,jkdef->abcdef", DdagD, S, cp.conj(S))\
        #  + contract("ij,ikabc,jkdef->abcdef", DdagD, cp.conj(S), S)

        #M0 = contract("ijabc,ijdef->abcdef", u, cp.conj(u)) \
        #  + contract("ijabc,ijdef->abcdef", cp.conj(u), u)
        U = [cp.ndarray] * 2
        VH = [cp.ndarray] * 2

        M0 = cp.exp(beta * 2 * M0)
        M0 = cp.reshape(M0, (cp.shape(M0)[0] * cp.shape(M0)[1] * cp.shape(M0)[2], cp.shape(M0)[3] * cp.shape(M0)[4] * cp.shape(M0)[5]))
        U[0], s0, VH[0] = rsvd(M0, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR")
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M0
        U[0]  = cp.reshape(U[0] , (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[0] = cp.reshape(VH[0], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[0]  = contract("abci,i->abci",  U[0], cp.sqrt(s0))
        VH[0] = contract("iabc,i->iabc", VH[0], cp.sqrt(s0))
        del s0

        U[1]  = U[0]
        VH[1] = VH[0]

        w[0] = contract("a,a,a->a", cp.sin(theta), cp.cos(theta), w[0])
        w[0] = cp.asarray(w[0], dtype = cp.complex128)
        w[1] = cp.asarray(w[1], dtype = cp.complex128)
        w[2] = cp.asarray(w[2], dtype = cp.complex128)

        t1 = time()
        print("Tensor initialization finished. Time= {:.6f} s".format(t1-t0))

        return U, VH, u, w, S
    

    def __init_tensor_component_parts_SU2__(self, beta, mu1, mu2, Dcut:int):
        t0 = time()
        
        from scipy.special import roots_legendre

        t = [np.ndarray] * 3
        w = [np.ndarray] * 3
        t[0], w[0] = roots_legendre(self.SAMPLE_NUM)
        t[1], w[1] = roots_legendre(self.SAMPLE_NUM)
        t[2], w[2] = roots_legendre(self.SAMPLE_NUM)

        theta = cp.asarray(np.pi * (t[0] + 1) / 4)
        a     = cp.asarray(np.pi * (t[1] + 1))
        b     = cp.asarray(np.pi * (t[2] + 1))

        exp_pia = cp.exp(1j*a)
        exp_pib = cp.exp(1j*b)
        exp_mia = cp.exp(-1j*a)
        exp_mib = cp.exp(-1j*b)
        sin_theta = cp.sin(theta)
        cos_theta = cp.cos(theta)
        I = cp.ones(self.SAMPLE_NUM)
        
        u = cp.zeros((2,2,self.SAMPLE_NUM,self.SAMPLE_NUM,self.SAMPLE_NUM), dtype=cp.complex128)
        u[0,0] =  contract("t,a,b->tab", cos_theta, exp_pia, I)
        u[0,1] =  contract("t,a,b->tab", sin_theta, I, exp_pib)
        u[1,0] = -contract("t,a,b->tab", sin_theta, I, exp_mib)
        u[1,1] =  contract("t,a,b->tab", cos_theta, exp_mia, I)

        e11 = cp.diag(cp.asarray([np.exp( (mu1+mu2)/2), np.exp(-(mu1+mu2)/2)]))
        e12 = cp.diag(cp.asarray([np.exp( (mu1-mu2)/2), np.exp(-(mu1-mu2)/2)]))
        e21 = cp.diag(cp.asarray([np.exp(-(mu1-mu2)/2), np.exp( (mu1-mu2)/2)]))
        e22 = cp.diag(cp.asarray([np.exp(-(mu1+mu2)/2), np.exp( (mu1+mu2)/2)]))
        e00 = cp.diag([1.0,1.0])
        
        U = [cp.ndarray] * 2
        VH = [cp.ndarray] * 2

        M0 = contract("ij,jkabc,kl,ildef->abcdef", e00, u, e00, cp.conj(u)) \
           + contract("ij,kjabc,kl,lidef->abcdef", e00, cp.conj(u), e00, u)
        M0 = cp.exp(beta * 2 * M0)
        M0 = cp.reshape(M0, (cp.shape(M0)[0] * cp.shape(M0)[1] * cp.shape(M0)[2], cp.shape(M0)[3] * cp.shape(M0)[4] * cp.shape(M0)[5]))
        U[0], s0, VH[0] = rsvd(M0, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR")
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M0
        U[0]  = cp.reshape(U[0] , (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[0] = cp.reshape(VH[0], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[0]  = contract("abci,i->abci",  U[0], cp.sqrt(s0))
        VH[0] = contract("iabc,i->iabc", VH[0], cp.sqrt(s0))
        del s0

        M1 = contract("ij,jkabc,kl,ildef->abcdef", e11, u, e12, cp.conj(u)) \
           + contract("ij,kjabc,kl,lidef->abcdef", e21, cp.conj(u), e22, u)
        M1 = cp.exp(beta * 2 * M1)
        M1 = cp.reshape(M1, (cp.shape(M1)[0] * cp.shape(M1)[1] * cp.shape(M1)[2], cp.shape(M1)[3] * cp.shape(M1)[4] * cp.shape(M1)[5]))
        U[1], s1, VH[1] = rsvd(M1, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR")
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M1
        U[1]  = cp.reshape(U[1] , (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[1] = cp.reshape(VH[1], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[1]  = contract("abci,i->abci",  U[1], cp.sqrt(s1))
        VH[1] = contract("iabc,i->iabc", VH[1], cp.sqrt(s1))
        del s1

        w[0] = contract("a,a,a->a", cp.sin(theta), cp.cos(theta), w[0])
        w[0] = cp.asarray(w[0], dtype = cp.complex128)
        w[1] = cp.asarray(w[1], dtype = cp.complex128)
        w[2] = cp.asarray(w[2], dtype = cp.complex128)

        t1 = time()
        print("Tensor initialization finished. Time= {:.6f} s".format(t1-t0))

        return U, VH, u, w
    
    def __init_tensor_component_parts_SU2_2__(self, beta, mu1, mu2, Dcut:int):
        t0 = time()
        
        from scipy.special import roots_legendre

        t = [np.ndarray] * 3
        w = [np.ndarray] * 3
        t[0], w[0] = roots_legendre(self.SAMPLE_NUM)
        t[1], w[1] = roots_legendre(self.SAMPLE_NUM)
        t[2], w[2] = roots_legendre(self.SAMPLE_NUM)

        theta = cp.asarray(np.pi * (t[0] + 1) / 4)
        a     = cp.asarray(np.pi * (t[1] + 1))
        b     = cp.asarray(np.pi * (t[2] + 1))

        exp_pia = cp.exp(1j*a)
        exp_pib = cp.exp(1j*b)
        exp_mia = cp.exp(-1j*a)
        exp_mib = cp.exp(-1j*b)
        sin_theta = cp.sin(theta)
        cos_theta = cp.cos(theta)
        I = cp.ones(self.SAMPLE_NUM)
        
        u = cp.zeros((2,2,self.SAMPLE_NUM,self.SAMPLE_NUM,self.SAMPLE_NUM), dtype=cp.complex128)
        u[0,0] =  contract("t,a,b->tab", cos_theta, exp_pia, I)
        u[0,1] =  contract("t,a,b->tab", sin_theta, I, exp_pib)
        u[1,0] = -contract("t,a,b->tab", sin_theta, I, exp_mib)
        u[1,1] =  contract("t,a,b->tab", cos_theta, exp_mia, I)

        W0 = cp.ones((2,2), dtype=cp.float64)
        W1 = cp.zeros((2,2), dtype=cp.float64)
        W1[0,0] = cp.exp(mu1)
        W1[0,1] = cp.exp(mu2)
        W1[1,0] = cp.exp(-mu2)
        W1[1,1] = cp.exp(-mu1)
        
        U = [cp.ndarray] * 2
        VH = [cp.ndarray] * 2

        M0 = contract("ij,ijabc,ijdef->abcdef", W0, u, cp.conj(u))
        M0 = cp.exp(beta * 4 * M0)
        M0 = cp.reshape(M0, (cp.shape(M0)[0] * cp.shape(M0)[1] * cp.shape(M0)[2], cp.shape(M0)[3] * cp.shape(M0)[4] * cp.shape(M0)[5]))
        U[0], s0, VH[0] = rsvd(M0, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR")
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M0
        U[0]  = cp.reshape(U[0] , (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[0] = cp.reshape(VH[0], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[0]  = contract("abci,i->abci",  U[0], cp.sqrt(s0))
        VH[0] = contract("iabc,i->iabc", VH[0], cp.sqrt(s0))
        del s0

        M1 = contract("ij,ijabc,ijdef->abcdef", W1, u, cp.conj(u))
        M1 = cp.exp(beta * 4 * M1)
        M1 = cp.reshape(M1, (cp.shape(M1)[0] * cp.shape(M1)[1] * cp.shape(M1)[2], cp.shape(M1)[3] * cp.shape(M1)[4] * cp.shape(M1)[5]))
        U[1], s1, VH[1] = rsvd(M1, k=Dcut, n_oversamples=12*Dcut, n_power_iter=0, iterator="QR")
        #U[0], s0, VH[0] = self.__svd__(M0, Dcut)
        del M1
        U[1]  = cp.reshape(U[1] , (self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM, Dcut))
        VH[1] = cp.reshape(VH[1], (Dcut, self.SAMPLE_NUM, self.SAMPLE_NUM, self.SAMPLE_NUM))
        U[1]  = contract("abci,i->abci",  U[1], cp.sqrt(s1))
        VH[1] = contract("iabc,i->iabc", VH[1], cp.sqrt(s1))
        del s1

        w[0] = contract("a,a,a->a", cp.sin(theta), cp.cos(theta), w[0])
        w[0] = cp.asarray(w[0], dtype = cp.complex128)
        w[1] = cp.asarray(w[1], dtype = cp.complex128)
        w[2] = cp.asarray(w[2], dtype = cp.complex128)

        t1 = time()
        print("Tensor initialization finished. Time= {:.6f} s".format(t1-t0))

        return U, VH, u, w
    

class Randomized_init():
    def __init__(self, K, beta, mu1, mu2, Dcut, seed):
        self.K = K
        self.beta = beta
        self.mu1 = mu1
        self.mu2 = mu2
        self.Dcut = Dcut
        self.seed = seed

    def chemical_term(self, a:float):
        sigma3 = cp.asarray([[1,0], [0,-1]])
        Identity = cp.asarray([[1,0], [0,1]])
        D = cp.cosh(a)*Identity + cp.sinh(a)*sigma3
        return D
    
    def random_SU2_matrix(self):
        rs = cp.random.RandomState(self.seed)
        alphas = rs.rand(self.K) * 2 * cp.pi
        betas  = rs.rand(self.K) * 2 * cp.pi
        thetas = rs.rand(self.K) * 2 * cp.pi / 4
        u00s =  cp.exp( 1j*alphas) * cp.cos(thetas)
        u01s =  cp.exp( 1j*betas ) * cp.sin(thetas)
        u10s = -cp.exp(-1j*betas ) * cp.sin(thetas)
        u11s =  cp.exp(-1j*alphas) * cp.cos(thetas)
        U = []
        I = []
        for u00,u01,u10,u11 in zip(u00s,u01s,u10s,u11s):
            u_ = cp.asarray([[u00,u01],[u10,u11]])
            U.append(u_)
            i_ = cp.asarray([[1,0],[0,1]])
            I.append(i_)
        U = cp.asarray(U)
        return U, I
    
    def cal_Eij(self):
        U, I = self.random_SU2_matrix()
        Eij = - contract("iab,jab->ij", U, cp.conj(U)) - contract("iab,jab->ij", cp.conj(U), U)
        return Eij
    
    def cal_Eijkl(self):
        U, I = self.random_SU2_matrix(self.seed)
        Eij = - contract("iab,jcb,kcd,lda->ijkl", U, cp.conj(U), I, I) - contract("iba,jbc,kcd,lda->ijkl", cp.conj(U), U, I, I)
        Ejk = - contract("iab,jbc,kdc,lda->ijkl", I, U, cp.conj(U), I) - contract("iab,jcb,kcd,lda->ijkl", I, cp.conj(U), U, I)
        Ekl = - contract("iab,jbc,kcd,lad->ijkl", I, I, U, cp.conj(U)) - contract("iab,jbc,kdc,lda->ijkl", I, I, cp.conj(U), U)
        Eli = - contract("iba,jbc,kcd,lda->ijkl", cp.conj(U), I, I, U) - contract("iab,jbc,kcd,lad->ijkl", U, I, I, cp.conj(U))
        Eijkl = Eij + Ejk + Ekl + Eli
        Eijkl /= 2
        return Eijkl
    
    def cal_Boltzmann_weight(self):
        Eij = self.cal_Eij()
        Mij = cp.exp(-self.beta * 2 * Eij)
        return Mij
    
    #def svd_Boltzmann_weight(self):
    #    Mij = self.cal_Boltzmann_weight()
    #    u, s, vh = np.linalg.svd(Mij)
    #    return u, s, vh
    
    def initial_tensor(self):
        Eijkl = self.cal_Eijkl()

        Mij = cp.exp(-self.beta * 2 * Eij / 2)
        Mij = Mij.reshape(-1)
        T = contract("a,b,c,d->abcd", Mij, Mij, Mij, Mij)
        return T

    #def init_tensor(self):
    #    from tensor_init.tensor_initialization import calinit
    #
    #    D1 = self.chemical_term( (self.mu1 + self.mu2)/2)
    #    D2 = self.chemical_term( (self.mu1 - self.mu2)/2)
    #    D3 = self.chemical_term(-(self.mu1 - self.mu2)/2)
    #    D4 = self.chemical_term(-(self.mu1 + self.mu2)/2)
    #
    #    #  j   k
    #    #   \ /
    #    #    A
    #    #   / \
    #    #  i   l
    #
    #    U, I = self.random_SU2_matrix()
    #    term1 = contract("iab,jcd,kda,lad->ijkl", U, I, I, cp.conj(U))
    #    term2 = contract("iab,jcb,kcd,lda->ijkl", I, cp.conj(U), U, I)
    #
    #    term3 = contract("iab,jbc,cd,kde,df,lfa->ijkl", I, I, D2, cp.conj(U), D1, U)
    #    term4 = contract("ae,ibe,bf,jfc,kcd,lda->ijkl", D3, cp.conj(U), D4, U, I, I)
    #
    #    A = term1 + term2 + term3 + term4
    #    A = cp.exp(self.beta * 2 * A) / self.K
    #
    #    delta = cp.diag(cp.ones(shape=self.K))
    #    B = contract("ij,jk,kl,li->ijkl", delta, delta, delta, delta)
    #
    #    #T = calinit()