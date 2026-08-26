# [depends] %LIB%/VAc_datagen.py
# [depends] %LIB%/utils.py

import sys, os
sys.path.append('lib/')
import numpy as np
#import mpld3
from VAc_datagen import Data_generator
from utils import PickleTool
import matplotlib.pyplot as plt

case_type = ['test']
data_meas = {}

os.makedirs('plots', exist_ok=True)
np.random.seed(123)

for case in case_type:
    Plant = Data_generator()
    # EOS settings
    Plant.constQ = 1 # if 0 neglect the change in volumetric flow rate
    Plant.rho_var = 1 # use eos
    Plant.const_pressure = 1 # use constant pressure mode

    Plant.factfor2 = 1 # turn on the undesired reaction
    Plant.n = 2 # number of CSTRs + 1
    Plant.U = 0
    Plant.delcp = 0.
    Plant.heatfact = 0
    Plant.bounds_free = 0 # keep bounds on control inputs as large as possible
    Plant.prod_shut = 0. # keep nominal production rate constraint equal to zero
    Plant.t_start = 0
    Plant.train_fraction = 0.7 # fraction of data used for training
    Plant.random_perturb = 1
    Plant.alpha = 0.
    Plant.beta = 0.
    Plant.ta = 1.0
    Plant.mean_noise = 0 # no bias
    
    Plant.sampling = 2 # i.e. sample every 2 seconds
    Plant.t_end_spec = 560 # 560 is the original one
    Plant.time_for_steady = 500
   
    Plant.det_ss = 0 # i.e. do not estimate the time required to reach steady state
    Plant.solve_transient_using_exp = 0 # don't use exponential transformation for solving the transient
    
    Plant.random_jumps = 200 # number of random jumps in the control inputs

    if case == 'test':
        Plant.param = 'test'
        Plant.arr_rxn2 = 2.5
        Plant.arr_rxn1 = 2.7
        Plant.cat_decay = 3.73e-3
        Plant.theta1 = np.array([0.7, 15, 1.1, 1.1, 1])
        Plant.theta2 = np.array([5.1, 21, 1, 1])
        Plant.keep_Q_const_trans = 0
        Plant.keep_T_const = 0
        Plant.n = 2
        Plant.std_noise = 0.
        Plant.t_end_spec = 560*2
        Plant.sampling = 4
        perturb = {}
        perturb['alpha'] = 0.3
        perturb['beta'] = 0.3
        perturb['no'] = 0.3
        perturb['ne'] = 0.3
        perturb['na'] = 0.3
        perturb['ta'] = 0.1
        bounds = {}
        bounds['alpha'] = [(0., 0.99)]   
        bounds['beta'] = [(0.6, 0.6)]    # clamp the N_A-beta flat direction into range
        bounds['NO'] = [(1e-1, 100)] 
        bounds['NEe'] = [(1e-1, 100)]
        bounds['NA'] = [(1e-1, 100)]
        bounds['T2'] = [(0.5, 5)]
        Plant.bounds = bounds
        Plant.perturb = perturb 
        
        Plant.penalty = [1. , 0.5, 0.5, 0.33, 0.5, 0.0, 0., 3, 0.3, 0., 0, 0e-3, 0]
        Plant.penalty = [i * (1 + np.random.normal(0, 0.0)) * 100 for i in Plant.penalty]
        Plant.p_spec = [0.7, 0.3, 1. , 1, 100, 100, 100]
        Plant.uR_spec = [0.5, 0.5, 1.2 , 1, 1.5, 1.5, 0.8]
        Plant.uR_spec = [i/i for i in Plant.uR_spec]
        Plant.u_spec = [0.68, 0.41, 1.07998 , 1, 1.7723, 2.0348, 1.08629]

    else:
        raise ValueError(f"Unknown case type: {case}")
    
    # save and visualize the data
    data = Plant.save_data('dummy', save=False)
    plt = Plant.visualize(save=True, show=True, plot_opt = True)
    data_meas[case] = data
    plt.savefig(f'plots/VAc_meas_{case}.pdf') # to inspect the data visually
    # fig = plt.gcf()                 # current figure
    # mpld3.save_html(fig, f"VAc_meas_{case}.html")
    plt.close()
    #plt.ioff()
    #plt.show()

PickleTool.save(data_meas, 'VAc_meas.pickle')