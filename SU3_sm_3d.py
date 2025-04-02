import sys
import os


import time

if __name__ == "__main__":
    
    xloop = int(sys.argv[1])
    yloop = int(sys.argv[2])
    tloop = int(sys.argv[3])
    Dcut  = int(sys.argv[4])
    K     = int(sys.argv[5])

    beta  = float(sys.argv[6])
    h     = float(sys.argv[7])
    mu    = float(sys.argv[8])

    from tensor_init.SU3_spin_model import SU3_spin_model_initialize as SU3sm
    su3sm = SU3sm(K=K, dim=3, Dcut=Dcut, beta=beta, h=h, mu=mu)

    gilt_eps = float(os.environ['GILT_EPS'])
    output_dir = os.environ['OUTPUT_DIR']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outputname = output_dir + '/result.dat'

    
    import measurement.SU3_sm_3d_measurement as SU3sm_measurement

    #time_start = time.time()
    #lnZ_OVER_V, e = SU3sm_measurement.internal_energy(su3sm, xloop, yloop, tloop)
    #time_finish = time.time()
    #
    #output = open(outputname, "w")
    #print("β={:}, h={:}, μ={:}, lnZ/V={:.12e}, e={:12e}, time={:.2f}s".format(sys.argv[6], sys.argv[7], sys.argv[8], 
    #                                                                lnZ_OVER_V, e, time_finish-time_start))
    #output.write("{:} {:} {:} {:19.12e} {:19.12e}\n".format(beta, h, mu, lnZ_OVER_V.real, e.real))
    #output.close()


    time_start = time.time()
    lnZ_OVER_V, P, Pdag = SU3sm_measurement.field_expected_value(su3sm, xloop, yloop, tloop)
    time_finish = time.time()
    
    output = open(outputname, "w")
    print("β={:}, h={:}, μ={:}, lnZ/V={:.12e}, <P>={:12e}, <P*>={:12e}, time={:.2f}s".format(sys.argv[6], sys.argv[7], sys.argv[8], 
                                                                    lnZ_OVER_V, P, Pdag, time_finish-time_start))
    output.write("{:} {:} {:} {:19.12e} {:19.12e} {:19.12e}\n".format(beta, h, mu, lnZ_OVER_V.real, P.real, Pdag.real))
    output.close()
    