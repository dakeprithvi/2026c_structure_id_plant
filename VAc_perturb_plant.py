# [depends] VAc_meas.pickle
# [depends] VAc_collect_param_fit_nn.pickle
# [depends] VAc_param_fit_paresto_reduced.pickle
# [depends] %LIB%/VAc_flowsheet.py

import sys, os
sys.path.append('lib/')
import numpy as np
from VAc_flowsheet import Simulator
import casadi
from utils import PickleTool

plants = ['test', 'cmeas', 'cmeas_noise', 'rmeas', 'cmeas_rinit', 'param']
# pickle/VAc_meas.pickle only stores 'test' measurement sets;
# map every plant onto an available key for the optimisation bounds.
meas_key = {'test': 'test', 'cmeas': 'test', 'cmeas_noise': 'test', 'rmeas': 'test','cmeas_rinit': 'test', 'param': 'test'}
mc_count = 3
data_plant = {}

file_base = os.path.basename(__file__).replace('.py', '')

# NN identifications now come from the single collected pickle written by
# VAc_collect_param_fit_nn.py (nested merged[base][case][mc][cheat_tag][kind]),
# the same pickle VAc_opti.py and VAc_plot_train_val_param_nn.py read.
def load_pickle_optional(filename):
    """Forgiving pickle load: return the data, or None (with a warning) when the
    file is absent so the dependent plant case can simply be skipped."""
    if not os.path.exists(filename):
        print(f'  WARNING: {filename} not found — skipping its plant case(s)')
        return None
    return PickleTool.load(filename, 'read')


merged = load_pickle_optional('VAc_collect_param_fit_nn.pickle') or {}
data_param = load_pickle_optional('VAc_param_fit_paresto.pickle')
data_base = load_pickle_optional('VAc_meas.pickle')['test']
BASE_NN = 'VAc_param_fit_nn_param_noise'
ID_CASE = 'test'

# friendly plant name -> (base, cheat_tag, noise_tag, kind) coordinates in the
# collected pickle, whose nested layout is
#   merged[base][case][mc_iter][cheat_tag][noise_tag][kind].
nn_source = {
    'cmeas':       (BASE_NN, 'nocheat', 'nonoise', 'main'),
    'cmeas_noise': (BASE_NN, 'nocheat', 'noise',   'main'),
    'rmeas':       (BASE_NN, 'cheat',   'nonoise', 'direct'),
    'cmeas_rinit':  (BASE_NN, 'cheat',   'nonoise', 'main'),
}

def nn_mc_iters(plant):
    """Every mc_iter present for `plant`'s base in the collected pickle (the
    'pickle types' we loop over); empty when the base/case was never produced."""
    base = nn_source[plant][0]
    try:
        return sorted(merged[base][ID_CASE])
    except (KeyError, TypeError):
        return []

def nn_data(plant, mc_iter):
    """Return the identification data for `plant` at `mc_iter`, or None if any
    level of the collected pickle is missing (e.g. the noisy ablation was never
    produced)."""
    base, cheat_tag, noise_tag, kind = nn_source[plant]
    try:
        return merged[base][ID_CASE][mc_iter][cheat_tag][noise_tag][kind]
    except (KeyError, TypeError):
        return None

# Seeding now happens per outer-MC seed inside the loop below; the True_plant
# setup uses only deterministic (zero-variance) draws, so it needs no seed here.
True_plant = Simulator()
True_plant.constQ = 1
True_plant.const_pressure = 1  # Use constant pressure operation
True_plant.rho_var = 1  # Use EOS
True_plant.alpha = 0
True_plant.beta = 0.
True_plant.n = 2
True_plant.factfor2 = 1
True_plant.theta1 = np.array([0.7, 15, 1.1, 1.1, 1])
True_plant.theta2 = np.array([5.1, 21, 1, 1])
True_plant.delcp = 0
True_plant.U = 0
True_plant.cat_decay = 3.73e-3
True_plant.param = 'test'
True_plant.arr_rxn2 = 2.5
True_plant.arr_rxn1 = 2.7
True_plant.cat_decay = 3.73e-3
True_plant.heatfact = 0
True_plant.ta = 1.0
True_plant.print_level = 0
True_plant.Stream_table()

True_plant.penalty = [1. , 0.5, 0.5, 0.33, 0.5, 0.0, 0., 3., 0.3, 0., 0, 0e-3, 0]
True_plant.penalty = [i * (1 + np.random.normal(0, 0.0)) * 100 for i in True_plant.penalty]
True_plant.p_spec = [0.7, 0.3, 1. , 1, 100, 100, 100]
True_plant.uR_spec = [0.5, 0.5, 1.2 , 1, 1.5, 1.5, 0.8]
True_plant.uR_spec = [i/i for i in True_plant.uR_spec]
True_plant.u_spec = [0.68, 0.41, 1.07998 , 1, 1.7723, 2.0348, 1.08629]

mid_alpha = (data_base['opti_conc']['bounds']['alpha'][0] + data_base['opti_conc']['bounds']['alpha'][1]) / 2
mid_beta = (data_base['opti_conc']['bounds']['beta'][0] + data_base['opti_conc']['bounds']['beta'][1]) / 2
mid_no = (data_base['opti_conc']['bounds']['no'][0] + data_base['opti_conc']['bounds']['no'][1]) / 2
mid_ne = (data_base['opti_conc']['bounds']['ne'][0] + data_base['opti_conc']['bounds']['ne'][1]) / 2
mid_na = (data_base['opti_conc']['bounds']['na'][0] + data_base['opti_conc']['bounds']['na'][1]) / 2
mid_ta = (data_base['opti_conc']['bounds']['ta'][0] + data_base['opti_conc']['bounds']['ta'][1]) / 2

True_plant.alpha = mid_alpha
True_plant.beta = mid_beta
True_plant.no = mid_no
True_plant.ne = mid_ne
True_plant.na = mid_na
True_plant.ta = mid_ta

u_initial_guess = [mid_alpha, mid_beta, mid_ta , 1, mid_ne, mid_na, mid_no]
True_plant.Stream_table()
c_initial_guess = True_plant.result.copy()

# Generate mc_count perturbed versions of the isolated base guesses, one per
# outer-MC index: u_initial_guess -> Plant.p_spec, c_initial_guess -> result.
# Each version is seeded so the multistart is reproducible; the outer loop below
# just iterates over these pre-generated guesses.
seeds = [1 + i for i in range(mc_count)]
u_initial_guesses = []
c_initial_guesses = []
guess_std = 0.2
for s in seeds:
    np.random.seed(s)
    u_initial_guesses.append(
        np.array(u_initial_guess) * (1 + np.random.normal(0, guess_std, size=len(u_initial_guess))))
    c_initial_guesses.append(
        np.array(c_initial_guess) * (1 + guess_std * np.random.normal(size=np.shape(c_initial_guess))))


# ── Monte-Carlo demarcation ─────────────────────────────────────────────────
# OUTER MC (mc_count guesses): each pre-generated (p_spec, result) pair is one
#   RTO multistart point. For a given guess index every case is handed the SAME
#   guess (common random numbers), so optimum differences trace to the model,
#   not the RNG.
# INNER MC (per case): each case is then run over its OWN initialization MC —
#   the NN identification iterations (id_mc) stored in the collected pickle; the
#   analytic cases (test / param) have a single trivial id_mc = None.

# Resolve each surviving case's data container and its own initialization-MC
# list once, up front, so the seed loop just iterates the cases that have data.
plant_id_mcs = {}
for plant in plants:
    if plant in nn_source:
        id_mcs = [m for m in nn_mc_iters(plant) if nn_data(plant, m) is not None]
        if not id_mcs:
            print(f'  WARNING: no identification data for plant {plant!r} — skipping')
            continue
    else:
        id_mcs = [None]
    if plant == 'param' and data_param is None:
        print(f'  WARNING: no paresto data for plant {plant!r} — skipping')
        continue
    plant_id_mcs[plant] = id_mcs
    # 'seed' / 'id_mc' tag each sample with its outer-MC seed and inner-MC
    # identification iteration, so plots can group results by seed.
    data_plant[plant] = {'na_model': [], 'ne_model': [], 'no_model': [], 'alpha_model': [], 'ta_model': [],
                         'profit_model': [], 'p_model': [], 'hess_cond': [], 'rate_1': [], 'rate_2': [],
                         'profit_plant': [], 'profit_loss': [], 'seed': [], 'id_mc': [], 'eigcheck_hess': [],
                         'revenue': [], 'opex': []}

# every case maps onto the same measurement/bounds key, loaded once per case
data_meas_all = {plant: PickleTool.load('VAc_meas.pickle', 'read')[meas_key[plant]]
                 for plant in plant_id_mcs}

for mc_idx, seed in enumerate(seeds):                 # ── OUTER MC: guess index
    # the pre-generated guess shared by every case at this index (common random
    # numbers), so optimum differences trace to the model, not the RNG.
    u_guess = u_initial_guesses[mc_idx]
    c_guess = c_initial_guesses[mc_idx]
    for plant, id_mcs in plant_id_mcs.items():        # ── each case
        data_meas = data_meas_all[plant]
        for id_mc in id_mcs:                          # ── case's own init MC
            mid_alpha = (data_meas['opti_conc']['bounds']['alpha'][0] + data_meas['opti_conc']['bounds']['alpha'][1]) / 2
            mid_beta = (data_meas['opti_conc']['bounds']['beta'][0] + data_meas['opti_conc']['bounds']['beta'][1]) / 2
            mid_no = (data_meas['opti_conc']['bounds']['no'][0] + data_meas['opti_conc']['bounds']['no'][1]) / 2
            mid_ne = (data_meas['opti_conc']['bounds']['ne'][0] + data_meas['opti_conc']['bounds']['ne'][1]) / 2
            mid_na = (data_meas['opti_conc']['bounds']['na'][0] + data_meas['opti_conc']['bounds']['na'][1]) / 2
            mid_ta = (data_meas['opti_conc']['bounds']['ta'][0] + data_meas['opti_conc']['bounds']['ta'][1]) / 2
            Plant = Simulator()
            Plant.constQ = 1
            Plant.const_pressure = 1  # Use constant pressure operation
            Plant.rho_var = 1  # Use EOS
            # get plant solution at midpoint of bounds, to use as initial guess for RTO solve
            Plant.alpha = mid_alpha
            Plant.beta = mid_beta
            Plant.no = mid_no
            Plant.ne = mid_ne
            Plant.na = mid_na
            Plant.ta = mid_ta
            Plant.prod_shut = 0. # keep nominal production rate equal to zero
            Plant.n = 2
            Plant.factfor2 = 1
            #Plant.U = 100
            Plant.theta1 = np.array([0.7, 15, 1.1, 1.1, 1])
            Plant.theta2 = np.array([5.1, 21, 1, 1])
            if plant == 'test':
                param = 'test'
                Plant.arr_rxn2 = 2.5
                Plant.arr_rxn1 = 2.7
                Plant.cat_decay = 3.73e-3
            elif plant == 'cmeas':
                data = nn_data(plant, id_mc)
                Plant.mean = data['mean']
                Plant.std = data['std']
                Plant.nn_rec1 = data['nvec'][0]
                Plant.nn_rec2 = data['nvec'][1]
                param = 'stoic_base'
                Plant.mult_r = data['nvec'][2]
            elif plant == 'cmeas_noise':
                data = nn_data(plant, id_mc)
                Plant.mean = data['mean']
                Plant.std = data['std']
                Plant.nn_rec1 = data['nvec'][0]
                Plant.nn_rec2 = data['nvec'][1]
                param = 'stoic_base'
                Plant.mult_r = data['nvec'][2]
            elif plant == 'rmeas':
                data = nn_data(plant, id_mc)
                Plant.mean = data['mean']
                Plant.std = data['std']
                Plant.nn_rec1 = data['nvec'][0]
                Plant.nn_rec2 = data['nvec'][1]
                param = 'stoic_base'
                Plant.mult_r = data['nvec'][2]
            elif plant == 'cmeas_rinit':
                data = nn_data(plant, id_mc)
                Plant.mean = data['mean']
                Plant.std = data['std']
                Plant.nn_rec1 = data['nvec'][0]
                Plant.nn_rec2 = data['nvec'][1]
                param = 'stoic_base'
                Plant.mult_r = data['nvec'][2]
            elif plant == 'param':
                param = 'test'
                Plant.arr_rxn2 = 2.5
                Plant.arr_rxn1 = 2.7
                Plant.cat_decay = 3.73e-3
                Plant.theta1 = data_param['fits'][1]['theta1']
                Plant.theta2 = data_param['fits'][1]['theta2']
            Plant.delcp = 0
            Plant.U = 0

            Plant.heatfact = 0
            Plant.ta = 1.0
            Plant.print_level = 0

            # solve always the plant here
            Plant.param = 'test'
            Plant.Stream_table()
            Plant.param = param 

            # one RTO solve per (guess, case, id_mc); use the pre-generated
            # initial guess for this outer-MC index.
            Plant.penalty = True_plant.penalty.copy()
            Plant.p_spec = np.array(u_guess).copy()
            result = np.array(c_guess).copy()
            Plant.uR_spec = [0.5, 0.5, 1.2 , 1, 1.5, 1.5, 0.8]
            Plant.uR_spec = [i/i for i in Plant.uR_spec]
            Plant.u_spec = [0.68, 0.41, 1.07998 , 1, 1.7723, 2.0348, 1.08629]

            bounds = {}
            bounds['alpha'] = [(data_meas['opti_conc']['bounds']['alpha'][0],data_meas['opti_conc']['bounds']['alpha'][1])]
            # since beta is fixed use the optimal solution itself.
            bounds['beta'] = [(data_meas['opti_conc']['beta'],data_meas['opti_conc']['beta'])]  
            bounds['NO'] = [(data_meas['opti_conc']['bounds']['no'][0], data_meas['opti_conc']['bounds']['no'][1])]
            bounds['NEe'] = [(data_meas['opti_conc']['bounds']['ne'][0], data_meas['opti_conc']['bounds']['ne'][1])]
            bounds['NA'] = [(data_meas['opti_conc']['bounds']['na'][0], data_meas['opti_conc']['bounds']['na'][1])]
            bounds['T2'] = [(data_meas['opti_conc']['bounds']['ta'][0], data_meas['opti_conc']['bounds']['ta'][1])]
            bounds['P2'] = [(1, 1)]
            bounds['x'] = [(0, casadi.inf)] * Plant.vars

            bounds_true = bounds.copy()
            Plant.optimize(result, bounds)
            bounds_true['alpha'] = [(Plant.sol_alpha, Plant.sol_alpha)]
            bounds_true['beta'] = [(Plant.sol_beta, Plant.sol_beta)]
            bounds_true['NO'] = [(Plant.sol_no, Plant.sol_no)]
            bounds_true['NEe'] = [(Plant.sol_ne, Plant.sol_ne)]
            bounds_true['NA'] = [(Plant.sol_na, Plant.sol_na)]
            bounds_true['T2'] = [(Plant.sol_ta, Plant.sol_ta)]
            True_plant.optimize(result, bounds_true)
            True_plant.alpha = Plant.sol_alpha
            True_plant.beta = Plant.sol_beta
            True_plant.no = Plant.sol_no
            True_plant.ne = Plant.sol_ne
            True_plant.na = Plant.sol_na
            True_plant.ta = Plant.sol_ta
            True_plant.Stream_table(True_plant.sol_x)
            True_plant.Economics()
            if Plant.solver.stats()['return_status'] == 'Solve_Succeeded' and True_plant.solver.stats()['return_status'] == 'Solve_Succeeded':
                data_plant[plant]['na_model'].append(Plant.sol_na)
                data_plant[plant]['ne_model'].append(Plant.sol_ne)
                data_plant[plant]['no_model'].append(Plant.sol_no)
                data_plant[plant]['alpha_model'].append(Plant.sol_alpha)
                data_plant[plant]['ta_model'].append(Plant.sol_ta)
                data_plant[plant]['profit_model'].append(Plant.obj)
                data_plant[plant]['profit_plant'].append(True_plant.total_econ)
                data_plant[plant]['revenue'].append(True_plant.revenue)
                data_plant[plant]['opex'].append(True_plant.opex)
                
                # evaluate plant at model solution and then compute loss wrt to plant's optimal solution
                loss_profit = (True_plant.total_econ - data_meas['opti_conc']['profit']) / data_meas['opti_conc']['profit']

                data_plant[plant]['profit_loss'].append(loss_profit)
                data_plant[plant]['hess_cond'].append(Plant.cond_hess)
                data_plant[plant]['eigcheck_hess'].append(Plant.eigcheck_hess)
                data_plant[plant]['rate_1'].append(Plant.rates[0][0].toarray()[0])
                data_plant[plant]['rate_2'].append(Plant.rates[1][0].toarray()[0])
                data_plant[plant]['seed'].append(seed)
                data_plant[plant]['id_mc'].append(id_mc)


PickleTool.save(data_plant, f'{file_base}.pickle')
