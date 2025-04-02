import gc
import time
import numpy as np
import cupy as cp
from cuquantum import einsum
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')
from tensor_init.XY_2d_nonlinear_sigma_model import gauss_legendre_quadrature as ti

import trg.HOTRG_3d as hotrg

from mpi4py import MPI
comm = MPI.COMM_WORLD 
myrank = comm.Get_rank() 
nproc = comm.Get_size() 
name = MPI.Get_processor_name() 
cuda = cp.cuda.Device(myrank)
cuda.use()

dim = 3

def ln_Z_over_V(beta, mu, Dcut:int, *LOOPS):
    """
    beta, mu, Dcut
    LOOPS:
    2d: XLOOPS, YLOOPS
    3d: XLOOPS, YLOOPS, ZLOOPS
    """
    #T = ti.init_pure_tensor_finit_density(beta, mu1, mu2, Dcut)
    #U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    if len(LOOPS) == 2:
        T = 0.5 * einsum("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, LOOPS[0], LOOPS[1])
        trace = contract("aabb", T)
    elif len(LOOPS) == 3:
        import trg.HOTRG_3d as hotrg
        
        T = 0.5 * einsum("a,ia,aj,ka,al,ma,an->ijklmn", wphi, VH1, U1, VH1, U1, VH2, U2)

        trace = contract("aabbcc", T/cp.max(cp.abs(T)))
        print(trace)
        T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, LOOPS[0], LOOPS[1], LOOPS[2])
        trace = contract("aabbcc", T)

    del T

    V = 2**(sum(LOOPS))
    print(V)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def particle_number(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    T = 0.5 * einsum("a,ia,aj,ka,al,ma,an->ijklmn", wphi, VH1, U1, VH1, U1, VH2, U2)
    Timp0 = 0.5 * einsum("a,a,ia,aj,ka,al,ma,an->ijklmn", wphi, exp_piphi, VH1, U1, VH1, U1, VH2, U2)
    Timp1 = 0.5 * einsum("a,a,ia,aj,ka,al,ma,an->ijklmn", wphi, exp_miphi, VH1, U1, VH1, U1, VH2, U2)
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    trace1 = contract("aabbcc", T)
    trace_imp1 = contract("aabbcc", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T = 0.5 * einsum("a,ia,aj,ka,al,ma,an->ijklmn", wphi, VH1, U1, VH1, U1, VH2, U2)
    Timp0 = 0.5 * einsum("a,a,ia,aj,ka,al,ma,an->ijklmn", wphi, exp_miphi, VH1, U1, VH1, U1, VH2, U2)
    Timp1 = 0.5 * einsum("a,a,ia,aj,ka,al,ma,an->ijklmn", wphi, exp_piphi, VH1, U1, VH1, U1, VH2, U2)
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    trace2 = contract("aabbcc", T)
    trace_imp2 = contract("aabbcc", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)

    return lnZoV, n

def internal_energy_0(beta, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    s0[0] = cp.exp(1j*phi)
    s1[0] = cp.exp(-1j*phi)
    s0[1] = cp.exp(-1j*phi)
    s1[1] = cp.exp(1j*phi)
    two_point_y = 0.0
    for i in range(2):
        T = 0.5 * contract("a,ia,aj,ka,al,ma,an->ijklmn", wphi, VH1, U1, VH1, U1, VH2, U2)
        Timp0 = 0.5 * contract("a,a,ia,aj,ka,al,ma,an->ijklmn", s0[i], wphi, VH1, U1, VH1, U1, VH2, U2)
        Timp1 = 0.5 * contract("a,a,ia,aj,ka,al,ma,an->ijklmn", s1[i], wphi, VH1, U1, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS, ZLOOPS)
        print(ln_normfact_imp)
        trace = contract("aabbcc", T)
        trace_imp = contract("aabbcc", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        two_point_y += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = -(3/2)*two_point_y
    return lnZoV, energy


def internal_energy(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wphi = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(exp_piphi), dtype=cp.float64)
    s0[0] = einsum("a,b->ab", cos_theta, I)
    s1[0] = einsum("a,b->ab", cos_theta, I)
    
    s0[1] = cp.exp(mu/2)*einsum("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = cp.exp(mu/2)*einsum("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = cp.exp(-mu/2)*einsum("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = cp.exp(-mu/2)*einsum("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    two_point_y = 0.0
    for i in range(3):
        T = 0.5 * einsum("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp0 = 0.5 * einsum("a,a,ab,b,ia,aj,ka,al->ijkl", wa, sin_theta, s0[i], wphi, VH1, U1, VH2, U2)
        Timp1 = 0.5 * einsum("a,a,ab,b,ia,aj,ka,al->ijkl", wa, sin_theta, s1[i], wphi, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    
        trace = einsum("aabb", T)
        trace_imp = einsum("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_y += normfact_imp * trace_imp / trace

    s0[1] = einsum("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = einsum("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = einsum("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = einsum("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    two_point_x = 0.0
    for i in range(3):
        T = 0.5 * einsum("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp0 = 0.5 * einsum("a,a,ab,b,ia,aj,ka,al->ijkl", wa, sin_theta, s0[i], wphi, VH1, U1, VH2, U2)
        Timp1 = 0.5 * einsum("a,a,ab,b,ia,aj,ka,al->ijkl", wa, sin_theta, s1[i], wphi, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 1, 0, Dcut, XLOOPS, YLOOPS)
    
        trace = einsum("aabb", T)
        trace_imp = einsum("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_x += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = -two_point_y-two_point_x

    return lnZoV, energy