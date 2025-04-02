import sys
import os


import time

if __name__ == "__main__":
    
    xloop = int(sys.argv[1])
    yloop = int(sys.argv[2])
    tloop = int(sys.argv[3])
    Dcut  = int(sys.argv[4])

    q  = int(sys.argv[5])
    k  = float(sys.argv[6])
    h  = float(sys.argv[7])
    mu = float(sys.argv[8])


    gilt_eps = float(os.environ['GILT_EPS'])
    output_dir = os.environ['OUTPUT_DIR']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outputname = output_dir + '/result.dat'

    
    import measurement.Potts3_model_3d as Pottsq

    time_start = time.time()
    lnZ_OVER_V = Pottsq.ln_Z_over_V(q, k, h, mu, Dcut, xloop, yloop, tloop, gilt_eps)
    time_finish = time.time()

    output = open(outputname, "w")
    print("κ={:}, h={:}, μ={:}, lnZ/V={:.12e}, time={:.2f}s".format(sys.argv[6], sys.argv[7], sys.argv[8], 
                                                                    lnZ_OVER_V, time_finish-time_start))
    output.write("{:} {:} {:} {:19.12e}\n".format(k, h, mu, lnZ_OVER_V.real))

    #time_start = time.time()
    #lnZ_OVER_V, e = SU3sm_measurement.internal_energy(su3sm, xloop, yloop, tloop)
    #time_finish = time.time()
    #
    #output = open(outputname, "w")
    #print("β={:}, h={:}, μ={:}, lnZ/V={:.12e}, e={:12e}, time={:.2f}s".format(sys.argv[6], sys.argv[7], sys.argv[8], 
    #                                                                lnZ_OVER_V, e, time_finish-time_start))
    #output.write("{:} {:} {:} {:19.12e} {:19.12e}\n".format(beta, h, mu, lnZ_OVER_V.real, e.real))
    
    output.close()
    