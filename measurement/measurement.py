import os
import numpy as np
import cupy as cp
import opt_einsum as oe

from operator import add
from functools import reduce

def save_entropy(T, Dcut, rgscheme, rgstep, outdir):
    if rgscheme != "trg":
        rgstepx = rgstep[0]
        rgstepy = rgstep[1]
        rgsteps = rgstepx + rgstepy
    else:
        rgsteps = rgstep

    TrTBTA = oe.contract("ijaa,jibb", T, T)
    print("TrTBTA=",TrTBTA)
    Dens_Mat = oe.contract("ijab,jicd->acbd", T, T) / TrTBTA
    chi_1, chi_2, chi_3, chi_4 = Dens_Mat.shape[0], Dens_Mat.shape[1], Dens_Mat.shape[2], Dens_Mat.shape[3]
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1*chi_2, chi_3*chi_4))
    _, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))
    if rgsteps == 0:
        with open("{:}/STE.dat".format(outdir), "w") as ste_out:
            ste_out.write("{:.15e}\n".format(STE))
    else:
        with open("{:}/STE.dat".format(outdir), "a") as ste_out:
            ste_out.write("{:.15e}\n".format(STE))

    if not os.path.exists("{:}/densitymatrix".format(outdir)):
        os.mkdir("{:}/densitymatrix".format(outdir))

    file_name = f"/densitymatrix_lx{rgstepx}_lt{rgstepy}.dat" if rgscheme != "trg" else f"densitymatrix_l{rgsteps}.dat"
    with open(f"{outdir}/densitymatrix"+file_name, "w") as svout:
        emax = cp.max(e_Dens_Mat)
        e_dens_mat = e_Dens_Mat / emax
        if len(e_dens_mat) < Dcut*Dcut:
            e_dens_mat = cp.pad(e_dens_mat, pad_width=(0,Dcut*Dcut-len(e_dens_mat)), mode='constant', constant_values=0.0)
        svout.write("#ρmax={:.12e}\n".format(emax))
        for ee in e_dens_mat:
           svout.write("{:.12e}\n".format(ee))

    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1, chi_2, chi_3, chi_4))
    rho_A = oe.contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    SEE = -cp.sum(e_A * cp.log(e_A))
    if rgstep == 0:
        with open(f"{outdir}/SEE.dat", "w") as see_out:
            see_out.write("{:.15e}\n".format(SEE))
    else:
        with open(f"{outdir}/SEE.dat", "a") as see_out:
            see_out.write("{:.15e}\n".format(SEE))

    if not os.path.exists(f"{outdir}/densitymatrix_A"):
        os.mkdir(f"{outdir}/densitymatrix_A".format(outdir))

    file_name = f"/densitymatrix_A_lx{rgstepx}_lt{rgstepy}.dat" if rgscheme != "trg" else f"densitymatrix_l{rgsteps}.dat"
    with open(f"{outdir}/densitymatrix_A", "w") as svout:
        emax = cp.max(e_A)
        e_a = e_A / emax
        if len(e_a) < Dcut:
            e_a = cp.pad(e_a, pad_width=(0,Dcut-len(e_a)), mode='constant', constant_values=0.0)
        svout.write("#ρAmax={:.12e}\n".format(emax))
        for ee in e_a:
           svout.write("{:.12e}\n".format(ee))

def save_ground_state_energy(norm_factors, max_eigv, rgstep, outdir):
    L2 = 2**rgstep
    E0 = cp.sum(norm_factors) + cp.log(max_eigv) / L2
    
    if rgstep == 0:
        with open(f"{outdir}/E0_over_L2.dat", "w") as output:
            output.write(f"{E0:.12e}\n")
    elif rgstep > 0:
        with open(f"{outdir}/E0_over_L2.dat", "a") as output:
            output.write(f"{E0:.12e}\n")

def save_entropy_transfermatrix_method(T, ln_normfact, Dcut, rgscheme, ls, ly, outdir):
    if rgscheme == "trg":
        raise ValueError("Only support hotrg-like!")
    
    TA, TB = T, T
    ln_normfact_A, ln_normfact_B = ln_normfact, ln_normfact

    chiA_x, chiA_X, chiA_y, chiA_Y = TA.shape
    chiB_x, chiB_X, chiB_y, chiB_Y = TB.shape
    
    Trans_Mat = oe.contract("ijab,jicd->acbd", TB, TA)
    Trans_Mat = cp.reshape(Trans_Mat, (chiB_Y*chiA_Y, chiB_y*chiA_y))
    print("Trans_Mat hermit err",cp.linalg.norm(Trans_Mat-cp.conj(Trans_Mat.T))/cp.linalg.norm(Trans_Mat))

    u, e, _ = cp.linalg.svd(Trans_Mat)
    e_max = cp.max(cp.abs(e))
    e = e / e_max

    Nx_A = 2**ls
    Nx_B = 2**ls

    result_outdir = outdir + "/results"
    if not os.path.exists(result_outdir):
        os.mkdir(result_outdir)
    result_filename = result_outdir + f"/ls{ls}.dat"
    result_output = open(result_filename, "w")
    result_output.close()

    for lt in range(ly+1):
        Nt = 2**lt

        #lnZ/V
        TrTBTA = cp.sum(e**Nt)
        VA = Nx_A*Nt
        VB = Nx_B*Nt
        ln_ZoverV = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_B) + cp.log(TrTBTA) + Nt*cp.log(e_max))/(VA+VB)
        ln_ZoverV = 0.0 if cp.abs(ln_ZoverV) < 1e-99 else ln_ZoverV

        #SEE
        Dens_Mat = oe.contract("ij,j,kj->ik", u, e**Nt, cp.conj(u)) / cp.sum(e**Nt)
        Dens_Mat = cp.reshape(Dens_Mat, (chiB_Y, chiA_Y, chiB_y, chiA_y))
        rho_A = oe.contract("aiaj->ij", Dens_Mat)
        _, e_A, _ = cp.linalg.svd(rho_A)
        SEE = -cp.sum(e_A * cp.log(e_A))
        SEE = 0.0 if cp.abs(SEE) < 1e-99 else SEE

        #STE
        Z = cp.sum(e**Nt) 
        STE = - Nt * cp.sum( (e**Nt) * cp.log(e))/Z + np.log(Z)
        STE = 0.0 if cp.abs(STE) < 1e-99 else STE

        #renyi
        N = 11
        Sn = cp.zeros(N, dtype=cp.complex128)
        n  = cp.array([i for i in range(1,N+1)], dtype=cp.float64)
        n[0] = 1/2
        for i in range(len(n)):
            Sn[i] = cp.sum(e_A**(n[i]))
        Sn = cp.log(Sn)
        Sn = Sn / (1-n)
        Sn = Sn.real
        Sn[cp.abs(Sn) < 1e-99] = 0.0
        to_str = lambda sn: f" {sn:.12e}"
        sn = map(to_str, Sn)
        sn = reduce(add, sn)

        output = f"{ln_ZoverV.real:.12e} {STE:.12e} {SEE:.12e}" + sn
        with open(result_filename, "a") as result_output:
            result_output.write(output + "\n")

def save_free_energy(T, ln_normfact, rgscheme, rgstep, outdir):
    if rgscheme != "trg":
        rgstepx = rgstep[0]
        rgstepy = rgstep[1]
        rgsteps = rgstepx + rgstepy
    else:
        rgsteps = rgstep

    trace = oe.contract("aabb", T)
    V = 2**(rgsteps)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V
    if rgstep == 0:
        with open("{:}/free_energy.dat".format(outdir), "w") as fe:
            fe.write("{:.15e}\n".format(ln_ZoverV))
    else:
        with open("{:}/free_energy.dat".format(outdir), "a") as fe:
            fe.write("{:.15e}\n".format(ln_ZoverV))
