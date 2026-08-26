# [depends] VAc_meas.pickle
# [depends] VAc_collect_param_fit_nn.pickle
# [depends] VAc_collect_param_fit_bbox_nn.pickle
# [depends] VAc_param_fit_paresto_reduced.pickle
# [depends] %LIB%/VAc_flowsheet.py
#
# Profit landscape over two control inputs.
#
# Same plant/model setup as VAc_perturb_plant.py / VAc_perturb_bbox_plant.py, but
# instead of letting the RTO pick the operating point, we SWEEP two chosen control
# inputs on a grid (over their training bounds) while pinning the other four at
# their optimum.  At every grid point we record each case's OWN objective
# (Plant.obj) — this is the optimisation landscape of that model.  The 'test' case
# (analytic kinetics) supplies the true-plant landscape.  We do NOT evaluate the
# true plant on the model inputs here (that is the perturb-study's profit-loss, a
# different question).  The result is one N x N surface per case per identification
# MC iteration (a single ground-truth surface for 'test'), saved for a contour plot.

import sys, os
sys.path.append('lib/')
import numpy as np
from VAc_flowsheet import Simulator
import casadi
from utils import PickleTool
import time

# ── user inputs ──────────────────────────────────────────────────────────────
# Two of: alpha, beta, no, ne, na, ta.  These are swept; the rest are fixed at
# their optimum.
AXIS1 = 'ta'
AXIS2 = 'alpha'
N = 20                      # grid points per axis (coarse to start)

# Every identification MC iteration is swept — we no longer cherry-pick a single
# (e.g. worst) fit per case.  For each model case we compute one landscape per
# mc_iter available in that case's collected pickle and store them all keyed by mc.
# The analytic 'test' case has no identification and yields one ground-truth surface.

# ── static maps ──────────────────────────────────────────────────────────────
# control name -> key used in the optimiser `bounds` dict
BND_KEY = {'alpha': 'alpha', 'beta': 'beta', 'no': 'NO',
           'ne': 'NEe', 'na': 'NA', 'ta': 'T2'}
CONTROLS = ['alpha', 'beta', 'no', 'ne', 'na', 'ta']

file_base = os.path.basename(__file__).replace('.py', '')

BASE_STRUCT = 'VAc_param_fit_nn_param_noise'        # structured (greybox) NN base
BASE_BBOX = 'VAc_param_fit_bbox_param_noise'        # black-box NN base
ID_CASE = 'test'

# Case table.  'analytic' => True-plant kinetics (ground truth); 'structured' =>
# recurrent stoichiometric NN via Plant.optimize; 'bbox' => black-box NN via
# Plant.optimize_cbox.  Each model case is swept over all its available id MC iters.
CASES = [
    {'name': 'test',             'kind': 'analytic'},
    {'name': 'cmeas',            'kind': 'structured', 'src': ('nocheat', 'nonoise', 'main')},
    {'name': 'cmeas_noise',      'kind': 'structured', 'src': ('nocheat', 'noise',   'main')},
    {'name': 'rmeas',            'kind': 'structured', 'src': ('cheat',   'nonoise', 'direct')},
    {'name': 'cmeas_rinit',       'kind': 'structured', 'src': ('cheat',   'nonoise', 'main')},
    {'name': 'cmeas_bbox',       'kind': 'bbox',       'src': ('nonoise',)},
    {'name': 'cmeas_noise_bbox', 'kind': 'bbox',       'src': ('noise',)},
]


def load_pickle_optional(filename):
    """Forgiving pickle load: return the data, or None (with a warning) when the
    file is absent so the dependent case can simply be skipped."""
    if not os.path.exists(filename):
        print(f'  WARNING: {filename} not found — skipping its case(s)')
        return None
    return PickleTool.load(filename, 'read')


merged_struct = load_pickle_optional('VAc_collect_param_fit_nn.pickle') or {}
merged_bbox = load_pickle_optional('VAc_collect_param_fit_bbox_nn.pickle') or {}
data_base = load_pickle_optional('VAc_meas.pickle')['test']
oc = data_base['opti_conc']     # every case maps onto the 'test' bounds/optimum


def struct_nn_data(mc, cheat_tag, noise_tag, kind_field):
    try:
        return merged_struct[BASE_STRUCT][ID_CASE][mc][cheat_tag][noise_tag][kind_field]
    except (KeyError, TypeError):
        return None


def bbox_nn_data(mc, noise_tag):
    try:
        return merged_bbox[BASE_BBOX][ID_CASE][mc][noise_tag]
    except (KeyError, TypeError):
        return None


def available_mcs(case):
    """Sorted list of identification MC iterations available for a model case
    (empty when the case's collected pickle is missing)."""
    if case['kind'] == 'analytic':
        return []
    try:
        if case['kind'] == 'structured':
            return sorted(merged_struct[BASE_STRUCT][ID_CASE].keys())
        return sorted(merged_bbox[BASE_BBOX][ID_CASE].keys())
    except (KeyError, TypeError):
        return []


def nn_data_for(case, mc):
    """Resolve the identification data for a model case at MC iteration `mc`, or
    None when unavailable so that (case, mc) can be skipped."""
    if case['kind'] == 'analytic':
        return None
    if case['kind'] == 'structured':
        cheat, noise, kind_field = case['src']
        return struct_nn_data(mc, cheat, noise, kind_field)
    return bbox_nn_data(mc, case['src'][0])


# ── True plant, built once, only to generate the deterministic warm-start guess ─
True_plant = Simulator()
True_plant.constQ = 1
True_plant.const_pressure = 1
True_plant.rho_var = 1
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

True_plant.penalty = [1., 0.5, 0.5, 0.33, 0.5, 0.0, 0., 3, 0.3, 0., 0, 0e-3, 0]
True_plant.penalty = [i * (1 + np.random.normal(0, 0.0)) * 100 for i in True_plant.penalty]
True_plant.p_spec = [0.7, 0.3, 1., 1, 100, 100, 100]
True_plant.uR_spec = [0.5, 0.5, 1.2, 1, 1.5, 1.5, 0.8]
True_plant.uR_spec = [i / i for i in True_plant.uR_spec]
True_plant.u_spec = [0.68, 0.41, 1.07998, 1, 1.7723, 2.0348, 1.08629]

# midpoint-of-bounds operating point, used as the deterministic initial guess for
# every fixed-point solve (all controls are pinned, so no multistart is needed).
mid = {u: (oc['bounds'][u][0] + oc['bounds'][u][1]) / 2 for u in CONTROLS}
True_plant.alpha = mid['alpha']
True_plant.beta = mid['beta']
True_plant.no = mid['no']
True_plant.ne = mid['ne']
True_plant.na = mid['na']
True_plant.ta = mid['ta']
u_initial_guess = [mid['alpha'], mid['beta'], mid['ta'], 1, mid['ne'], mid['na'], mid['no']]
True_plant.Stream_table()
c_initial_guess = True_plant.result.copy()


def control_bounds(sim, fixed):
    """Bounds dict pinning every control at the scalar given in `fixed`."""
    b = {BND_KEY[u]: [(fixed[u], fixed[u])] for u in CONTROLS}
    b['P2'] = [(1, 1)]
    b['x'] = [(0, casadi.inf)] * sim.vars
    return b


def solve_point(case, nn_dat, fixed):
    """Solve this case's model at one pinned operating point and return its own
    objective Plant.obj (the landscape value); np.nan on a failed solve."""
    Plant = Simulator()
    Plant.constQ = 1
    Plant.const_pressure = 1
    Plant.rho_var = 1
    # midpoint start for the free internal states
    Plant.alpha = mid['alpha']
    Plant.beta = mid['beta']
    Plant.no = mid['no']
    Plant.ne = mid['ne']
    Plant.na = mid['na']
    Plant.ta = mid['ta']
    Plant.prod_shut = 0.
    Plant.n = 2
    Plant.factfor2 = 1
    Plant.theta1 = np.array([0.7, 15, 1.1, 1.1, 1])
    Plant.theta2 = np.array([5.1, 21, 1, 1])

    if case['kind'] == 'analytic':
        param = 'test'
        Plant.arr_rxn2 = 2.5
        Plant.arr_rxn1 = 2.7
        Plant.cat_decay = 3.73e-3
    elif case['kind'] == 'structured':
        Plant.mean = nn_dat['mean']
        Plant.std = nn_dat['std']
        Plant.nn_rec1 = nn_dat['nvec'][0]
        Plant.nn_rec2 = nn_dat['nvec'][1]
        Plant.mult_r = nn_dat['nvec'][2]
        param = 'stoic_base'
    else:  # bbox
        Plant.mean = nn_dat['mean_y']
        Plant.std = nn_dat['std_y']
        Plant.mean_u = nn_dat['mean_u']
        Plant.std_u = nn_dat['std_u']
        Plant.nn_bbox = nn_dat['nvec'][0]
        Plant.mean_dcdt = nn_dat['mean_dcdt']
        Plant.std_dcdt = nn_dat['std_dcdt']
        param = 'bbox'
    Plant.delcp = 0
    Plant.U = 0
    Plant.heatfact = 0
    Plant.ta = 1.0
    Plant.print_level = 0

    Plant.param = 'test'
    Plant.Stream_table()
    Plant.param = param

    Plant.penalty = [1., 0.5, 0.5, 0.33, 0.5, 0.0, 0., 3, 0.3, 0., 0, 0e-3, 0]
    Plant.penalty = [i * 100 for i in Plant.penalty]
    Plant.p_spec = np.array(u_initial_guess).copy()
    result = np.array(c_initial_guess).copy()
    Plant.uR_spec = [0.5, 0.5, 1.2, 1, 1.5, 1.5, 0.8]
    Plant.uR_spec = [i / i for i in Plant.uR_spec]
    Plant.u_spec = [0.68, 0.41, 1.07998, 1, 1.7723, 2.0348, 1.08629]

    bounds = control_bounds(Plant, fixed)
    if case['kind'] == 'bbox':
        Plant.optimize_cbox(result, bounds)
    else:
        Plant.optimize(result, bounds)

    # each case's OWN objective at the pinned point IS its landscape value;
    # 'test' (analytic kinetics) is the true-plant landscape.
    if Plant.solver.stats()['return_status'] == 'Solve_Succeeded':
        return Plant.obj
    return np.nan


# ── grid over the two chosen axes (their training bounds) ────────────────────
V1 = np.linspace(oc['bounds'][AXIS1][0], oc['bounds'][AXIS1][1], N)
V2 = np.linspace(oc['bounds'][AXIS2][0], oc['bounds'][AXIS2][1], N)
A1, A2 = np.meshgrid(V1, V2, indexing='ij')

def compute_surface(case, nn_dat, label):
    """Sweep the two chosen axes and return this case's N x N objective surface."""
    profit_model = np.full((N, N), np.nan)
    for i, v1 in enumerate(V1):
        for j, v2 in enumerate(V2):
            fixed = {u: oc[u] for u in CONTROLS}   # all four others at optimum
            fixed[AXIS1] = v1
            fixed[AXIS2] = v2
            profit_model[i, j] = solve_point(case, nn_dat, fixed)
        print(f'  {label}: row {i + 1}/{N} done')
    return profit_model


def surface_entry(profit_model, mc):
    return {
        'axis1': AXIS1, 'axis2': AXIS2,
        'axis1_grid': A1, 'axis2_grid': A2,
        'profit_model': profit_model,
        'id_mc': mc, 'profit_opt': oc['profit'],
    }


# For 'test' (analytic) data_hist[name] is a single surface entry; for every model
# case it is a dict {mc: surface entry} spanning all available id MC iterations.
data_hist = {}
tic = time.time()
for case in CASES:
    name = case['name']

    if case['kind'] == 'analytic':
        profit_model = compute_surface(case, None, name)
        data_hist[name] = surface_entry(profit_model, None)
        print(f'{name}: done (analytic ground truth)')
        continue

    mcs = available_mcs(case)
    if not mcs:
        print(f'  WARNING: no identification data for {name!r} — skipping')
        continue

    per_mc = {}
    for mc in mcs:
        nn_dat = nn_data_for(case, mc)
        if nn_dat is None:
            print(f'  WARNING: no identification data for {name!r} (mc={mc}) — skipping')
            continue
        profit_model = compute_surface(case, nn_dat, f'{name}/mc{mc}')
        per_mc[mc] = surface_entry(profit_model, mc)
        print(f'{name}: mc {mc} done')
    if per_mc:
        data_hist[name] = per_mc
        print(f'{name}: done ({len(per_mc)} MC iterations)')
    
toc = time.time()
runtime = (toc - tic) / 60
print(f'All cases done in {runtime:.1f} minutes')

data_hist['runtime'] = runtime

PickleTool.save(data_hist, f'{file_base}.pickle')
