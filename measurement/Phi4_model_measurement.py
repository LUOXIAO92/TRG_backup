import gc
import time
import numpy as np
import cupy as cp
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')
from tensor_init.Phi4_model import gauss_hermite_quadrature as ti

import trg.HOTRG_2d_QR as hotrg

from mpi4py import MPI
comm = MPI.COMM_WORLD 
myrank = comm.Get_rank() 
nproc = comm.Get_size() 
name = MPI.Get_processor_name() 
cuda = cp.cuda.Device(myrank)
cuda.use()


def ln_Z_over_V(m, lam, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U, VH, exp, _ = ti().phi4_init_tensor_component(m, mu, lam, Dcut)
    T = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def particle_number(m, lam, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U, VH, exp, phi = ti().phi4_init_tensor_component(m, mu, lam, Dcut)

    T = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    Timp0 = contract("ab,ab,iab,abj,kab,abl->ijkl", cp.conj(phi), exp, VH[0], U[0], VH[1], U[1])
    Timp1 = contract("ab,ab,iab,abj,kab,abl->ijkl", phi, exp, VH[0], U[0], VH[1], U[1])
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("aabb", T)
    trace_imp1 = contract("aabb", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    Timp0 = contract("ab,ab,iab,abj,kab,abl->ijkl", phi, exp, VH[0], U[0], VH[1], U[1])
    Timp1 = contract("ab,ab,iab,abj,kab,abl->ijkl", cp.conj(phi), exp, VH[0], U[0], VH[1], U[1])
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("aabb", T)
    trace_imp2 = contract("aabb", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2
    #print(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2)
    return lnZoV, n, normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2

def field_strength(m, lam, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U, VH, exp, phi = ti().phi4_init_tensor_component(m, mu, lam, Dcut)

    phi2 = phi * cp.conj(phi)
    T = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    Timp0 = contract("ab,ab,iab,abj,kab,abl->ijkl", phi2, exp, VH[0], U[0], VH[1], U[1])
    Timp1 = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace = contract("aabb", T)
    trace_imp = contract("aabb", Timp0)
    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0


    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    phi2 = normfact_imp*trace_imp/trace
    #print(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2)
    return lnZoV, phi2




def particle_number_TRG(m, lam, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    U, VH, exp, phi = ti().phi4_init_tensor_component(m, mu, lam, Dcut)

    T     = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    Timp0 = contract("ab,ab,iab,abj,kab,abl->ijkl", cp.conj(phi), exp, VH[0], U[0], VH[1], U[1])
    Timp1 = contract("ab,ab,iab,abj,kab,abl->ijkl", phi, exp, VH[0], U[0], VH[1], U[1])

    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("abab", T)
    trace_imp1 = contract("abab", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T     = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    Timp0 = contract("ab,ab,iab,abj,kab,abl->ijkl", phi, exp, VH[0], U[0], VH[1], U[1])
    Timp1 = contract("ab,ab,iab,abj,kab,abl->ijkl", cp.conj(phi), exp, VH[0], U[0], VH[1], U[1])

    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("abab", T)
    trace_imp2 = contract("abab", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2

    return lnZoV, n, normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2

def field_strength_TRG(m, lam, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    U, VH, exp, phi = ti().phi4_init_tensor_component(m, mu, lam, Dcut)

    phi2 = phi * cp.conj(phi)
    T     = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])
    Timp0 = contract("ab,ab,iab,abj,kab,abl->ijkl", phi2, exp, VH[0], U[0], VH[1], U[1])
    Timp1 = contract("ab,iab,abj,kab,abl->ijkl", exp, VH[0], U[0], VH[1], U[1])

    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace = contract("abab", T)
    trace_imp = contract("abab", Timp0)
    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    phi2 = normfact_imp*trace_imp/trace
    #print(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2)
    return lnZoV, phi2