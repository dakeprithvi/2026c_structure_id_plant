# [depends] %LIB%/VAc_flowsheet.py

import sys
sys.path.append('lib/')
import numpy as np
from VAc_flowsheet import Simulator
from utils import PickleTool
import casadi

np.random.seed(0) 

data_meas = PickleTool.load('VAc_meas.pickle', 'read')['test']
data_param = PickleTool.load('VAc_param_fit_paresto.pickle', 'read')

Plant = Simulator()
Plant.constQ = 1
Plant.const_pressure = 1  # Use constant pressure operation
Plant.rho_var = 1  # Use EOS
Plant.alpha = 0
Plant.beta = 0.
# to keep T in [140,170] C and N10[4] ~ 13.16, alpha/beta in [0.5,0.9].
# Config (n=2, 1 tank): arr=(2.7,2.5), cat_decay=3.73e-3, penalty[8]=0.3
#   -> T=153.9 C, N10~13.2, alpha=0.569 (strong), beta pinned 0.6, all eig > 0.
Plant.n = 2
Plant.factfor2 = 1
Plant.arr_rxn2 = 2.5
Plant.arr_rxn1 = 2.7
Plant.delcp = 0
Plant.U = 0
Plant.cat_decay = 3.73e-3
Plant.heatfact = 0
Plant.ta = 1.0
Plant.print_level = 0
Plant.param = 'test'

Plant.theta1 = np.array([0.7, 15, 1.1, 1.1, 1])
Plant.theta2 = np.array([5.1, 21, 1, 1])

Plant.Stream_table()

# very loose bounds
bounds = {}
bounds['alpha'] = [(0., 0.99)]   
bounds['beta'] = [(0.6, 0.6)]   
bounds['NO'] = [(1e-1, 100)] 
bounds['NEe'] = [(1e-1, 100)]
bounds['NA'] = [(1e-1, 100)]
bounds['T2'] = [(0.5, 5)]
bounds['P2'] = [(1, 1)]
bounds['x'] = [(0, casadi.inf)] * Plant.vars

Plant.penalty = [1. , 0.5, 0.5, 0.33, 0.5, 0.0, 0., 3, 0.3, 0., 0, 0e-3, 0]
Plant.penalty = [i * (1 + np.random.normal(0, 0.0)) * 100 for i in Plant.penalty]

Plant.p_spec = [0.7, 0.3, 1. , 1, 100, 100, 100]
Plant.uR_spec = [0.5, 0.5, 1.2 , 1, 1.5, 1.5, 0.8]
Plant.uR_spec = [i/i for i in Plant.uR_spec]
Plant.u_spec = [0.68, 0.41, 1.07998 , 1, 1.7723, 2.0348, 1.08629]

# random initial guess 
Plant.p_spec = [i * (1 + np.random.normal(0, 1)) for i in Plant.p_spec]
result = Plant.result * (1 + np.random.normal(0, 0.1, size=Plant.result.shape))

Plant.optimize(result, bounds)

# extract the solution and apply RTO to the plant
Plant.alpha = Plant.sol_alpha
Plant.beta = Plant.sol_beta
Plant.no = Plant.sol_no
Plant.ne = Plant.sol_ne
Plant.na = Plant.sol_na
Plant.ta = Plant.sol_ta
Plant.pa = Plant.sol_pa

# sanity check on steady-state plant (not part of casadi optimization)
Plant.Stream_table(Plant.sol_x)
Plant.Economics()

print('Obj: ', Plant.total_econ)
print('Pen: ', Plant.penalty_term)

# Analyze reduced Hessian and its eigenvalues for convexity and conditioning insights
hess = Plant.L_schur
e, v = np.linalg.eig(hess)
u_s_true = np.array([Plant.sol_ne * Plant.N1_clean[0], Plant.sol_na * Plant.N1_clean[2], Plant.sol_no * Plant.N1_clean[5], Plant.sol_alpha, Plant.sol_ta * Plant.T20])
l_s_true = Plant.total_econ 
eigcheck_hess_true = Plant.eigcheck_hess
profit = Plant.total_econ.copy()
revenue = Plant.revenue.copy()
opex = Plant.opex.copy()
per_revenue = profit / revenue * 100
per_opex = profit / opex * 100

Q = Plant.Qflow[-1].copy()
Qcheck = Plant.Qcheck.copy()


u_s = {}
l_s_model = {}
l_s_plant = {}
eigcheck_hess_model = {}
for i in [1, 3, 5]:
    Plant.theta1 = data_param['fits'][i]['theta1']
    Plant.theta2 = data_param['fits'][i]['theta2']

    Plant.optimize(result, bounds)
    Plant.alpha = Plant.sol_alpha
    Plant.beta = Plant.sol_beta
    Plant.no = Plant.sol_no
    Plant.ne = Plant.sol_ne
    Plant.na = Plant.sol_na
    Plant.ta = Plant.sol_ta
    Plant.pa = Plant.sol_pa
    Plant.Stream_table(Plant.sol_x)
    Plant.Economics()
    u_s[i] = np.array([Plant.sol_ne * Plant.N1_clean[0], Plant.sol_na * Plant.N1_clean[2], Plant.sol_no * Plant.N1_clean[5], Plant.sol_alpha, Plant.sol_ta * Plant.T20])
    l_s_model[i] = Plant.total_econ
    eigcheck_hess_model[i] = Plant.eigcheck_hess
    bounds_model = bounds.copy()
    bounds_model['alpha'] = [(Plant.sol_alpha, Plant.sol_alpha)]
    bounds_model['beta'] = [(Plant.sol_beta, Plant.sol_beta)]
    bounds_model['NO'] = [(Plant.sol_no, Plant.sol_no)]
    bounds_model['NEe'] = [(Plant.sol_ne, Plant.sol_ne)]
    bounds_model['NA'] = [(Plant.sol_na, Plant.sol_na)]
    bounds_model['T2'] = [(Plant.sol_ta, Plant.sol_ta)]
    bounds_model['P2'] = [(Plant.sol_pa, Plant.sol_pa)]
    Plant.theta1 = np.array([0.7, 15, 1.1, 1.1, 1])
    Plant.theta2 = np.array([5.1, 21, 1, 1])
    Plant.optimize(result, bounds_model)
    Plant.alpha = Plant.sol_alpha
    Plant.beta = Plant.sol_beta
    Plant.no = Plant.sol_no
    Plant.ne = Plant.sol_ne
    Plant.na = Plant.sol_na
    Plant.ta = Plant.sol_ta
    Plant.pa = Plant.sol_pa
    Plant.Stream_table(Plant.sol_x)
    Plant.Economics()
    l_s_plant[i] = Plant.total_econ

print('True plant solution:')
print('\nu_s:', u_s_true)
print('l_s:', l_s_true)

print('\n')
print('Eigenvalues:\n', np.sort(e))
check_hess_pd = np.all(e > 0)
print('Hessian PD:', check_hess_pd)
print('Condition number:', np.linalg.cond(hess))
print('Eigenvalue check (should be True):', eigcheck_hess_true)

print('\npercentage of profit to revenue: ', per_revenue, '%')
print('percentage of profit to OPEX: ', per_opex, '%')

print('\nEstimated plant solutions:')
noise_levels = [0, 1e-3, 1e-2]
for i in [1, 3, 5]:
    print(f'\nFit {i} (noise level {noise_levels[i//2]}):')
    print('u_s:', u_s[i])
    print('l_s_model:', l_s_model[i])
    print('l_s_plant:', l_s_plant[i])
    print('Eigenvalue check (should be True):', eigcheck_hess_model[i])
    print('loss', (l_s_plant[i]-data_meas['opti_conc']['profit'])/data_meas['opti_conc']['profit']*100, '%')

