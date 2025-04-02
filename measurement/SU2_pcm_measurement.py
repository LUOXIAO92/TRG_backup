import gc
import time
import numpy as np
import cupy as cp
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')
simul_confi = configparser.ConfigParser()
simul_confi.read(sys.argv[1])
if simul_confi["SIMULATIONSETTING"]["method"] == "fl":
    print("using fibonacci lattice")
    from tensor_init.SU2_2d_principle_chiral_moedl import fibonacci_lattice as ti
elif simul_confi["SIMULATIONSETTING"]["method"] == "gl":
    print("using gauss legendre quadrature")
    from tensor_init.SU2_2d_principle_chiral_moedl import gauss_legendre_quadrature as ti

import trg.HOTRG_2d as hotrg

def ln_Z_over_V(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int):
    #T = ti.init_pure_tensor_finit_density(beta, mu1, mu2, Dcut)
    U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    #U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, mu1, mu2, Dcut)
    #U, VH, _, w = ti().__init_tensor_component_parts_SU2__(beta, mu1, mu2, Dcut)
    #U, VH, _, w = ti().__init_tensor_component_parts_SU2_2__(beta, mu1, mu2, Dcut)
    
    T = ti().__init_pure_tensor__(w, U[0], VH[0], U[1], VH[1])
    T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V
    print(beta,mu1,mu2)
    return ln_ZoverV

def ln_Z_over_V2(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int):
    #T = ti.init_pure_tensor_finit_density(beta, mu1, mu2, Dcut)
    #U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, mu1, mu2, Dcut)
    #U, VH, _, w = ti().__init_tensor_component_parts_SU2__(beta, mu1, mu2, Dcut)
    #U, VH, _, w = ti().__init_tensor_component_parts_SU2_2__(beta, mu1, mu2, Dcut)
    
    T = ti().__init_pure_tensor__(w, U[0], VH[0], U[1], VH[1])
    T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V
    print(beta,mu1,mu2)
    return ln_ZoverV

def internal_energy_zero_density_HOTRG(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.HOTRG_2d as hotrg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    
    n2pf_sum = 0.0
    for i in range(4):
        if scheme == "fl":
            T = contract("ia,aj,ka,al->ijkl", VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp0 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp1 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
        elif scheme == "gl": 
            T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])

        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
        
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        n2pf_sum += normfact_imp*trace_imp/trace
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum 
    
    return lnZoV, internal_energy

def internal_energy_zero_density_TRG(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    
    n2pf_sum = 0.0
    for i in range(4):
        if scheme == "fl":
            T = contract("ia,aj,ka,al->ijkl", VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp0 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp1 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
        elif scheme == "gl": 
            T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            
        T     = cp.transpose(T    , axes=(0,2,1,3))
        Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
        Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

        T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
        
        trace = contract("abab", T)
        trace_imp = contract("abab", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        n2pf_sum += normfact_imp*trace_imp/trace
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum 
    
    return lnZoV, internal_energy

def internal_energy_finit_density_TRG(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]
    if scheme == "fl":
        U, VH, a, b = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    elif scheme == "gl":
        U, VH, a, b, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    
    n2pf_sum = 0.0
    for d in range(2):
        for i in range(4):
            if scheme == "fl":
                T = contract("ia,aj,ka,al->ijkl", VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                Timp0 = contract("ia,aj,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                Timp1 = contract("ia,aj,",b[d][i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            elif scheme == "gl": 
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[d][i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            
    
            T     = cp.transpose(T    , axes=(0,2,1,3))
            Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
            Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

            if d == 0:
                T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, T, T, Timp1, Dcut, XLOOPS, YLOOPS)
            elif d == 1:
                T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)

            trace = contract("abab", T)
            trace_imp = contract("abab", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum / 2
    
    return lnZoV, internal_energy


def partical_number_HOTRG(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int, mu_mu1_mu2:str):
    scheme = simul_confi["SIMULATIONSETTING"]["method"]

    time_start=time.time()
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, mu1, mu2, Dcut)
    time_finish=time.time()
    print("tensor initialization finished, total time:{:.2f}s".format(time_finish-time_start))

    if mu_mu1_mu2 == "mu":
        D = ti().__chemical_potential_matrix_term_mu__(mu1)
    elif mu_mu1_mu2 == "mu1":
        D = ti().__chemical_potential_matrix_term_mu1__(mu1)
    elif mu_mu1_mu2 == "mu2":
        D = ti().__chemical_potential_matrix_term_mu2__(mu2)
    else:
        import sys
        print("no such mu")
        sys.exit(1)
    
    if scheme == "fl":
        b = contract("ij,ja->ia", D, a)
    elif scheme == "gl":
        b = contract("ij,jabc->iabc", D, a)

    if mu_mu1_mu2 == "mu":
        n2pf_sum = 0.0
        for i in range(4):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[i,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[i,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
            trace = contract("aabb", T)
            trace_imp = contract("aabb", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    elif mu_mu1_mu2 == "mu1":
        n2pf_sum = 0.0
        for i in range(2):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[-i,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[-i,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[-i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[-i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
            trace = contract("aabb", T)
            trace_imp = contract("aabb", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    elif mu_mu1_mu2 == "mu2":
        n2pf_sum = 0.0
        for i in range(2):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[i+1,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[i+1,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i+1,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[i+1,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
            trace = contract("aabb", T)
            trace_imp = contract("aabb", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    partical_number = 2 * 4 * beta * n2pf_sum
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    return lnZoV, partical_number

def partical_number_HOTRG_test(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int, mu_mu1_mu2:str):
    scheme = simul_confi["SIMULATIONSETTING"]["method"]

    time_start=time.time()
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, mu1, mu2, Dcut)
    time_finish=time.time()
    print("tensor initialization finished, total time:{:.2f}s".format(time_finish-time_start))

    if mu_mu1_mu2 == "mu":
        D = ti().__chemical_potential_matrix_term_mu__(mu1)
    elif mu_mu1_mu2 == "mu1":
        D = ti().__chemical_potential_matrix_term_mu1__(mu1)
    elif mu_mu1_mu2 == "mu2":
        D = ti().__chemical_potential_matrix_term_mu2__(mu2)
    else:
        import sys
        print("no such mu")
        sys.exit(1)

    n2pf_sum = 0.0
    for i in range(4):
        for j in range(4):
            if np.abs(D[i,i]) > 1e-12:
                print(f"D[{i},{j}]")
                if scheme == "fl":
                    T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                    Timp0 = ti().__impure_tensor_calculation_finit_density__(a[i,:], U[0], VH[0], U[1], VH[1])
                    Timp1 = ti().__impure_tensor_calculation_finit_density__(a[j,:], U[0], VH[0], U[1], VH[1])
                elif scheme == "gl":
                    T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                    Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                    Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[j,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
                trace = contract("aabb", T)
                trace_imp = contract("aabb", Timp0)
                normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
                n2pf_sum += D[i,i]*normfact_imp*trace_imp/trace

    partical_number = 2 * 4 * beta * n2pf_sum
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    return lnZoV, partical_number


def partical_number_TRG(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int, mu_mu1_mu2:str):
    import trg.TRG_2d as trg
    
    scheme = simul_confi["SIMULATIONSETTING"]["method"]

    time_start=time.time()
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    time_finish=time.time()
    print("tensor initialization finished, total time:{:.2f}s".format(time_finish-time_start))

    if mu_mu1_mu2 == "mu":
        D = ti().__chemical_potential_matrix_term_mu__(mu1)
    elif mu_mu1_mu2 == "mu1":
        D = ti().__chemical_potential_matrix_term_mu1__(mu1)
    elif mu_mu1_mu2 == "mu2":
        D = ti().__chemical_potential_matrix_term_mu2__(mu2)
    else:
        import sys
        print("no such mu")
        sys.exit(1)
    
    if scheme == "fl":
        b = contract("ij,ja->ia", D, a)
    elif scheme == "gl":
        b = contract("ij,jabc->iabc", D, a)

    if mu_mu1_mu2 == "mu":
        n2pf_sum = 0.0
        for i in range(4):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[i,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[i,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            
            T     = cp.transpose(T    , axes=(0,2,1,3))
            Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
            Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

            T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
            trace = contract("abab", T)
            trace_imp = contract("abab", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    elif mu_mu1_mu2 == "mu1":
        n2pf_sum = 0.0
        for i in range(2):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[-i,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[-i,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[-i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[-i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            
            T     = cp.transpose(T    , axes=(0,2,1,3))
            Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
            Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

            T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
            trace = contract("abab", T)
            trace_imp = contract("abab", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    elif mu_mu1_mu2 == "mu2":
        n2pf_sum = 0.0
        for i in range(2):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[i+1,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[i+1,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i+1,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[i+1,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            
            T     = cp.transpose(T    , axes=(0,2,1,3))
            Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
            Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

            T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
            trace = contract("abab", T)
            trace_imp = contract("abab", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    partical_number = 2 * 4 * beta * n2pf_sum
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    return lnZoV, partical_number

def ln_Z_over_V_test(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int):
    
    U, VH, _, w, _, _ = ti().__init_tensor_component_parts_finit_density_4TRG__(beta, mu1, mu2, Dcut)
    T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])

    #U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, mu1, mu2, Dcut)
    #T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
    
    T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def internal_energy_zero_density_TRG_test(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    elif scheme == "gl":
        U, VH, s, w, _, _ = ti().__init_tensor_component_parts_finit_density_4TRG__(beta, 0, 0, Dcut)

    D = cp.asarray([[1/2,1/2,0,0],[0,0,1/2,1/2]])
    D = D.T @ D
    
    n2pf_sum = 0.0
    for i in range(4):
        for j in range(4):
            if cp.abs(D[i,j]) > 1e-12:
                for k in range(2):
                    if scheme == "fl":
                        T = contract("ia,aj,ka,al->ijkl", VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                        Timp0 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                        Timp1 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                    elif scheme == "gl": 
                        T     = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                        Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", s[i,k,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                        Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", cp.conj(s[j,k,:,:,:]), w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])

                    T     = cp.transpose(T    , axes=(0,2,1,3))
                    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
                    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
                    t0 = time.time()
                    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
                    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
                    t1 = time.time()
                    print("time= {:.2f} s".format(t1-t0))

                    trace = contract("abab", T)
                    trace_imp = contract("abab", Timp0)
                    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
                    n2pf_sum += D[i,j]*2*(normfact_imp*trace_imp/trace).real / 4
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum 
    
    return lnZoV, internal_energy


def partical_number_HOTRG_QR(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int, mu_mu1_mu2:str):
    import trg.HOTRG_2d_QR as hotrg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]

    time_start=time.time()
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, mu1, mu2, Dcut)
    time_finish=time.time()
    print("tensor initialization finished, total time:{:.2f}s".format(time_finish-time_start))

    if mu_mu1_mu2 == "mu":
        D = ti().__chemical_potential_matrix_term_mu__(mu1)
    elif mu_mu1_mu2 == "mu1":
        D = ti().__chemical_potential_matrix_term_mu1__(mu1)
    elif mu_mu1_mu2 == "mu2":
        D = ti().__chemical_potential_matrix_term_mu2__(mu2)
    else:
        import sys
        print("no such mu")
        sys.exit(1)
    
    if scheme == "fl":
        b = contract("ij,ja->ia", D, a)
    elif scheme == "gl":
        b = contract("ij,jabc->iabc", D, a)

    n1 = 0.0
    n2 = 0.0
    if mu_mu1_mu2 == "mu":
        n2pf_sum = 0.0
        for i in range(4):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[i,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[i,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
            trace = contract("aabb", T)
            trace_imp = contract("aabb", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

            if i == 0 or i==3:
                n1 += normfact_imp*trace_imp/trace
            elif i==1 or i==2:
                n2 += normfact_imp*trace_imp/trace

    elif mu_mu1_mu2 == "mu1":
        n2pf_sum = 0.0
        for i in range(2):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[-i,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[-i,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[-i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[-i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
            trace = contract("aabb", T)
            trace_imp = contract("aabb", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    elif mu_mu1_mu2 == "mu2":
        n2pf_sum = 0.0
        for i in range(2):
            if scheme == "fl":
                T = contract("ia,aj,ak,la->ijkl", VH[0], U[0], U[1], VH[1]) / ti().SAMPLE_NUM
                Timp0 = ti().__impure_tensor_calculation_finit_density__(a[i+1,:], U[0], VH[0], U[1], VH[1])
                Timp1 = ti().__impure_tensor_calculation_finit_density__(b[i+1,:], U[0], VH[0], U[1], VH[1])
            elif scheme == "gl":
                T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i+1,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", b[i+1,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
            trace = contract("aabb", T)
            trace_imp = contract("aabb", Timp0)
            normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
            n2pf_sum += normfact_imp*trace_imp/trace

    partical_number = 2 * 4 * beta * n2pf_sum
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    return lnZoV, partical_number, n1, n2

def internal_energy_zero_density_HOTRG_QR(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.HOTRG_2d_QR as hotrg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, 0, 0, Dcut)
    
    n2pf_sum = 0.0
    for i in range(4):
        if scheme == "fl":
            T = contract("ia,aj,ka,al->ijkl", VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp0 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp1 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
        elif scheme == "gl": 
            T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])

        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
        
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        n2pf_sum += normfact_imp*trace_imp/trace

        del Timp0, Timp1, T
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum 
    
    return lnZoV, internal_energy


def internal_energy_zero_density_HOTRG_su2(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.HOTRG_2d as hotrg
    #U, VH, u, w, s = ti().__init_tensor_component_parts_zero_density_4TRG__(beta, Dcut)
    U, VH, s, w, _, _ = ti().__init_tensor_component_parts_finit_density_4TRG__(beta, 0.0, 0.0, Dcut)
    
    #n2pf_sum = 0.0
    #for i in range(2):
    #    for j in range(2):
    #        T     = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH, U, VH, U)
    #        Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", u[i,j], w[0], w[1], w[2], VH, U, VH, U)
    #        Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", cp.conj(u[i,j]), w[0], w[1], w[2], VH, U, VH, U)
    #
    #        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    #
    #        trace = contract("aabb", T)
    #        trace_imp = contract("aabb", Timp0)
    #        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    #        n2pf_sum += (normfact_imp*trace_imp/trace).real / 2
    #
    #        del Timp0, Timp1, T

    D = cp.asarray([[1/2,1/2,0,0],[0,0,1/2,1/2]])
    D = D.T @ D
    n2pf_sum = 0.0
    for a in range(4):
        for b in range(4):
            if D[a,b] > 1e-12:
                for i in range(2):
                    T     = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                    Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", s[a,i], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
                    Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", cp.conj(s[b,i]), w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
    
                    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    
                    trace = contract("aabb", T)
                    trace_imp = contract("aabb", Timp0)
                    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
                    n2pf_sum += D[a,b]*(normfact_imp*trace_imp/trace).real / 2
    
                    del Timp0, Timp1, T
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum 
    
    return lnZoV, internal_energy



#ATRG
from tensor_class.tensor_class import Tensor
def to_tensor_gl(U, VH, w, Dcut:int, s=cp.full(1, fill_value=False, dtype=bool))->Tensor:
    if (s == False).all():
        T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
    else:
        T = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", s[:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
    
    T = cp.transpose(T, (3,0,2,1))
    T = cp.reshape(T, (Dcut*Dcut, Dcut*Dcut))
    U, s, VH = cp.linalg.svd(T)
    del T

    U  = U[:,:Dcut]
    s  = s[:Dcut]
    VH = VH[:Dcut,:]

    U  = cp.reshape(U , (Dcut,Dcut,Dcut))
    VH = cp.reshape(VH, (Dcut,Dcut,Dcut))
    T = Tensor(U, s, VH)
    del U, s, VH

    return T

def ln_Z_over_V_ATRG(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.ATRG_2d as atrg 
    U, VH, _, w, _, _ = ti().__init_tensor_component_parts_finit_density_4TRG__(beta, mu1, mu2, Dcut)
    T = to_tensor_gl(U, VH, w, Dcut)

    T, ln_normfact = atrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)

    trace = contract("ija,a,aij", T.U, T.s, T.VH)
    del T

    V = 2**(XLOOPS+YLOOPS)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def internal_energy_zero_density_ATRG(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.ATRG_2d as atrg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    elif scheme == "gl":
        U, VH, s, w, _, _ = ti().__init_tensor_component_parts_finit_density_4TRG__(beta, 0.0, 0.0, Dcut)
        #T_pure = to_tensor_gl(U, VH, w, Dcut)

    D = cp.asarray([[1/2,1/2,0,0],[0,0,1/2,1/2]])
    D = D.T @ D
    n2pf_sum = 0.0
    for i in range(4):
        for j in range(4):
            if cp.abs(D[i,j]) > 1e-12:
                for k in range(2):
                    if scheme == "fl":
                        T = contract("ia,aj,ka,al->ijkl", VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                        Timp0 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                        Timp1 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
                    elif scheme == "gl": 
                        T     = to_tensor_gl(U, VH, w, Dcut)
                        Timp0 = to_tensor_gl(U, VH, w, Dcut, s[i,k,:,:,:])
                        Timp1 = to_tensor_gl(U, VH, w, Dcut, cp.conj(s[j,k,:,:,:]))

                    t0 = time.time()
                    T, Timp0, ln_normfact, ln_normfact_imp = atrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
                    t1 = time.time()
                    print("time= {:.2f} s".format(t1-t0))

                    trace = contract("ija,a,aij", T.U, T.s, T.VH)
                    trace_imp = contract("ija,a,aij", Timp0.U, Timp0.s, Timp0.VH)
                    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
                    n2pf_sum += D[i,j]*2*(normfact_imp*trace_imp/trace).real / 4
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum 
    
    return lnZoV, internal_energy


def internal_energy_zero_density_ATRG_sincos(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.ATRG_2d as atrg
    scheme = simul_confi["SIMULATIONSETTING"]["method"]
    if scheme == "fl":
        U, VH, a, _ = ti().__init_tensor_component_parts_finit_density__(beta, 0, 0, Dcut)
    elif scheme == "gl":
        U, VH, a, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, 0, 0, Dcut)
    
    n2pf_sum = 0.0
    for i in range(4):
        if scheme == "fl":
            T = contract("ia,aj,ka,al->ijkl", VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp0 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
            Timp1 = contract("ia,aj,ka,al->ijkl,",a[i,:], VH[0], U[0], VH[1], U[1]) / ti().SAMPLE_NUM
        elif scheme == "gl": 
            #T = (cp.pi / 8) * contract("a,b,c,iabc,abcj,kabc,abcl->ijkl", w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            #Timp0 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            #Timp1 = (cp.pi / 8) * contract("abc,a,b,c,iabc,abcj,kabc,abcl->ijkl", a[i,:,:,:], w[0], w[1], w[2], VH[0], U[0], VH[1], U[1])
            T     = to_tensor_gl(U, VH, w, Dcut)
            Timp0 = to_tensor_gl(U, VH, w, Dcut, a[i,:,:,:])
            Timp1 = to_tensor_gl(U, VH, w, Dcut, a[i,:,:,:])

        T, Timp0, ln_normfact, ln_normfact_imp = atrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
        
        trace = contract("ija,a,aij", T.U, T.s, T.VH)
        trace_imp = contract("ija,a,aij", Timp0.U, Timp0.s, Timp0.VH)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        n2pf_sum += normfact_imp*trace_imp/trace

        del Timp0, Timp1, T
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    internal_energy = 1 - n2pf_sum 
    
    return lnZoV, internal_energy