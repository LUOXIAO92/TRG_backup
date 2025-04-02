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
    mu1   = float(sys.argv[8])
    mu2   = float(sys.argv[9])

    from tensor_init.SU2_pcm import SU2_pcm_initialize as SU2pcm
    su2pcm = SU2pcm(K=K, dim=3, Dcut=Dcut, beta=beta, h=h, mu1=mu1, mu2=mu2)

    gilt_eps = float(os.environ['GILT_EPS'])
    output_dir = os.environ['OUTPUT_DIR']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outputname = output_dir + '/result.dat'

    
    import measurement.SU2_pcm_3d_measurement as SU2pcm_measurement

    time_start = time.time()
    lnZ_OVER_V = SU2pcm_measurement.ln_Z_over_V(su2pcm, xloop, yloop, tloop)
    time_finish = time.time()

    output = open(outputname, "w")
    print("β={:}, h={:}, μ1={:}, μ2={:}, lnZ/V={:.12e}, time={:.2f}s".format(sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], 
                                                                    lnZ_OVER_V, time_finish-time_start))
    output.write("{:} {:} {:} {:} {:19.12e}\n".format(beta, h, mu1, mu2, lnZ_OVER_V.real))

    #time_start = time.time()
    #lnZ_OVER_V, e = SU2pcm_measurement.internal_energy(su2pcm, xloop, yloop, tloop)
    #time_finish = time.time()
    #
    #output = open(outputname, "w")
    #print("β={:}, h={:}, μ1={:}, μ2={:}, lnZ/V={:.12e}, e={:12e}, time={:.2f}s".format(sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], 
    #                                                                lnZ_OVER_V, e, time_finish-time_start))
    #output.write("{:} {:} {:} {:} {:19.12e} {:19.12e}\n".format(beta, h, mu1, mu2, lnZ_OVER_V.real, e.real))

    
    output.close()
    