# [depends] %LIB%/utils.py
# [depends] VAc_meas.pickle

import sys
sys.path.append('lib/')
import os
from paresto import Paresto
import casadi as casadi
import numpy as np
from utils import PickleTool

np.random.seed(123) # for reproducibility

# use VAc_meas.pickle to load the data
# Usage: python VAc_param_fit_nn_param_noise.py <case_type> <mc_iter>
if len(sys.argv) > 2:
    case_type = sys.argv[1]
    mc_iter   = int(sys.argv[2])
elif len(sys.argv) > 1:
    case_type = sys.argv[1]
    mc_iter   = 0
else:
    case_type = 'test'
    mc_iter   = 0

# set up log file in the log directory; results are pickled for the plot script
os.makedirs('log', exist_ok=True)
log_file_base = os.path.basename(__file__).replace('.py', '')
log_handle = open(f'log/{log_file_base}_{case_type}.txt', 'w', buffering=1)
sys.stdout = log_handle

# pick the right case for training
data_meas = PickleTool.load('VAc_meas.pickle', 'read')
data = data_meas[case_type]

# data preparation
# use val_index to control the time range
# stride subsamples the trajectory so the dataset stays small enough for paresto
val_index = 280 * 8
stride = 15
meas_t  = data['time'][:val_index:stride]
print(meas_t.shape)
print(data['time'].shape)
meas_c0 = data['meas'][0, :]
meas    = data['meas'][:val_index:stride]
val_t   = data['time'][val_index:]
val_c0  = data['meas'][val_index, :]
val     = data['meas'][val_index:]
num_of_batch = int(data['train_jumps'])
clean_meas = data['clean_meas'][:val_index:stride]
clean_val  = data['clean_meas'][val_index:]
Nfee = data['NEe'][:val_index:stride]
Nfa = data['NA'][:val_index:stride]
Nfw = data['NW'][:val_index:stride]
Nfv = data['NV'][:val_index:stride]
Nfo = data['NO'][:val_index:stride]
Nfc = data['NC'][:val_index:stride]
T = data['T2'][:val_index:stride]
time = data['time'][:val_index:stride]
alpha = data['alpha'][:val_index:stride]
beta = data['beta'][:val_index:stride]

Tmean = np.mean(T)

pe = Paresto()
pe.x = ['cEe', 'cEa', 'cA', 'cW', 'cV', 'cO', 'cC']
pe.p = ['lk1', 'lE1', 'a1', 'a2', 'a3', 'lk2', 'lE2', 'a4', 'a5', 'P', 'R', 'Vr', 'nt', 'e', 'l', 'pcat', 'mu', 'At', 'constQ', 'cEe_std', 'cEa_std', 'cA_std', 'cW_std', 'cV_std', 'cO_std', 'cC_std', 'cEe_mean', 'cEa_mean', 'cA_mean', 'cW_mean', 'cV_mean', 'cO_mean', 'cC_mean', 'Tmean']
pe.d = ['cEem', 'cEam', 'cAm', 'cWm', 'cVm', 'cOm', 'cCm', 'Nfee', 'Nfa', 'Nfw', 'Nfv', 'Nfo', 'Nfc', 'T', 'alpha', 'beta']


def ode(t, y, p):
    recycled = y.alpha * (y.cEe + y.cEa + y.cO + (1 - p.mu) * y.cC) + y.beta * y.cA

    NFee = y.Nfee / p.nt
    NFea = 0.001 * NFee
    NFa = y.Nfa / p.nt
    NFw = y.Nfw / p.nt
    NFv = y.Nfv / p.nt
    NFo = y.Nfo / p.nt
    NFc = y.Nfc / p.nt

    # define rate here
    r1, r2 = rate(t, y, p)
       
    ree = -r1 - r2
    rea = 0 * r1 + 0 * r2
    ra  = -r1
    rw  = r1 + 2.0 * r2
    rv  = r1
    ro  = -0.5 * r1 - 3.0 * r2
    rc  = 2.0 * r2

    sum_Rj = ree + rea + ra + rw + rv + ro + rc

    N1_sum = NFee + NFea + NFa + NFw + NFv + NFo + NFc
    num = N1_sum + p.constQ * p.pcat * p.R * p.At * p.l * recycled * (y.T * sum_Rj / p.P)

    denom = p.P/p.R/y.T - recycled
    Qf = num / denom
    Q = Qf + p.pcat * p.R * p.At * p.l * y.T * sum_Rj / p.P

    N2ee = NFee + Q * y.alpha * y.cEe
    N2ea = NFea + Q * y.alpha * y.cEa
    N2a = NFa + Q * y.beta * y.cA
    N2o = NFo + Q * y.alpha * y.cO
    N2w = NFw
    N2v = NFv
    N2c = NFc + Q * y.alpha * (1 - p.mu) * y.cC

    c2ee = N2ee / Qf
    c2ea = N2ea / Qf
    c2a = N2a / Qf
    c2o = N2o / Qf
    c2w = N2w / Qf
    c2v = N2v / Qf
    c2c = N2c / Qf

    dcee = (1/p.Vr * (Qf * c2ee - Q * y.cEe) + p.pcat * ree) / p.e
    dcea = (1/p.Vr * (Qf * c2ea - Q * y.cEa) + p.pcat * rea) / p.e
    dca = (1/p.Vr * (Qf * c2a - Q * y.cA) + p.pcat * ra) / p.e
    dcw = (1/p.Vr * (Qf * c2w - Q * y.cW) + p.pcat * rw) / p.e
    dcv = (1/p.Vr * (Qf * c2v - Q * y.cV) + p.pcat * rv) / p.e
    dco = (1/p.Vr * (Qf * c2o - Q * y.cO) + p.pcat * ro) / p.e
    dcc = (1/p.Vr * (Qf * c2c - Q * y.cC) + p.pcat * rc) / p.e

    return [dcee, dcea, dca, dcw, dcv, dco, dcc]
    
def lsq(t, y, p):
    return [(y.cEe - y.cEem) / p.cEe_std, (y.cEa - y.cEam) / p.cEa_std, (y.cA - y.cAm) / p.cA_std, (y.cW - y.cWm) / p.cW_std, (y.cV - y.cVm) / p.cV_std, (y.cO - y.cOm) / p.cO_std, (y.cC - y.cCm) / p.cC_std]


def rate(t, y, p):
    k1 = np.exp(p.lk1)
    k2 = np.exp(p.lk2)
    E1 = np.exp(p.lE1)
    E2 = np.exp(p.lE2)
    cEe_n = casadi.fmax(y.cEe/p.cEe_mean, 1e-8)
    cO_n = casadi.fmax(y.cO/p.cO_mean, 1e-8)
    cA_n = casadi.fmax(y.cA/p.cA_mean, 1e-8)
    r1 =  k1  * np.exp(-E1 / p.R * (1.0 / y.T - 1.0 / p.Tmean)) * cEe_n**p.a1 * cO_n**p.a2 * cA_n**p.a3
    r2 =  k2  * np.exp(-E2 / p.R * (1.0 / y.T - 1.0 / p.Tmean)) * cEe_n**p.a4 * cO_n**p.a5
    return r1, r2


def eval_rate(theta, cEe, cO, cA, T):
    # numpy mirror of rate() for plotting: same algebra, casadi.fmax -> np.maximum.
    # theta is a plain dict of parameter values (lk1, lE1, a1, a2, a3, lk2, lE2).
    k1 = np.exp(theta['lk1'])
    k2 = np.exp(theta['lk2'])
    E1 = np.exp(theta['lE1'])
    E2 = np.exp(theta['lE2'])
    cEe_n = np.maximum(cEe / pe.ic.cEe_mean, 1e-8)
    cO_n = np.maximum(cO / pe.ic.cO_mean, 1e-8)
    cA_n = np.maximum(cA / pe.ic.cA_mean, 1e-8)
    r1 =  k1  * np.exp(-E1 / pe.ic.R * (1.0 / T - 1.0 / pe.ic.Tmean)) * cEe_n**theta['a1'] * cO_n**theta['a2'] * cA_n**theta['a3']
    r2 =  k2  * np.exp(-E2 / pe.ic.R * (1.0 / T - 1.0 / pe.ic.Tmean)) * cEe_n**theta['a4'] * cO_n**theta['a5']
    return r1, r2

pe.ode = ode
pe.lsq = lsq
pe.tout = time

comb_meas = np.concatenate([meas, Nfee[:, None], Nfa[:, None], Nfw[:, None], Nfv[:, None], Nfo[:, None], Nfc[:, None], T[:, None], alpha[:, None], beta[:, None]], axis=1)

# settings
pe.transcription = 'sundials'
pe.print_level = 0

pe.initialize()

pe.ic.P = 882529.0
pe.ic.R = 8.314
pe.ic.Vr = 0.010752
pe.ic.At = 0.25 * np.pi * (3.7e-2) ** 2
pe.ic.l = 10.0
pe.ic.nt = 622.0
pe.ic.e = 0.8
pe.ic.mu = 0.03
pe.ic.constQ = 1.0
pe.ic.pcat = 385.0
pe.ic.cEe_std = np.std(meas, axis=0)[0]
pe.ic.cEa_std = np.std(meas, axis=0)[1]
pe.ic.cA_std = np.std(meas, axis=0)[2]
pe.ic.cW_std = np.std(meas, axis=0)[3]
pe.ic.cV_std = np.std(meas, axis=0)[4]
pe.ic.cO_std = np.std(meas, axis=0)[5]
pe.ic.cC_std = np.std(meas, axis=0)[6]
pe.ic.cEe_mean = np.mean(meas, axis=0)[0]
pe.ic.cEa_mean = np.mean(meas, axis=0)[1]
pe.ic.cA_mean = np.mean(meas, axis=0)[2]
pe.ic.cW_mean = np.mean(meas, axis=0)[3]
pe.ic.cV_mean = np.mean(meas, axis=0)[4]
pe.ic.cO_mean = np.mean(meas, axis=0)[5]
pe.ic.cC_mean = np.mean(meas, axis=0)[6]
pe.ic.Tmean = Tmean

print(f'mean {np.mean(meas, axis=0)}')
print(f'std {np.std(meas, axis=0)}')


mee, mo, ma = pe.ic.cEe_mean, pe.ic.cO_mean, pe.ic.cA_mean
k1_true = 0.7 * mee**1.1 * mo**1.1 * ma * np.exp(2.7 * 15e3 / pe.ic.R * (1/450 - 1/Tmean)) * 3.73 * 1e-7
k2_true = 5.1 * mee * mo * np.exp(2.5 * 21e3 / pe.ic.R * (1/450 - 1/Tmean)) * 3.73 * 1e-7
pe.ic.lk1 = np.log(k1_true)
pe.ic.lE1 = np.log(40500)
pe.ic.a1 = 1.1
pe.ic.a2 = 1.1  
pe.ic.a3 = 1.0
pe.ic.lk2 = np.log(k2_true)
pe.ic.lE2 = np.log(52500)
pe.ic.a4 = 1.0
pe.ic.a5 = 1.0

# do not forget the initial conditions.
pe.ic.cEe = meas[0, 0]
pe.ic.cEa = meas[0, 1]
pe.ic.cA = meas[0, 2]
pe.ic.cW = meas[0, 3]
pe.ic.cV = meas[0, 4]
pe.ic.cO = meas[0, 5]
pe.ic.cC = meas[0, 6]


pe.lb_ub()
# with this, lb and ub both are fixed at ic

# CASE 1: perfect initial guess and fixed parameters (lb=ub=ic) should produce zero error
est, y, p = pe.optimize(comb_meas.T[:,:,None])
# smoke test fixed param should produce the same measurements
pred = [y.cEe.flatten(), y.cEa.flatten(), y.cA.flatten(), y.cW.flatten(), y.cV.flatten(), y.cO.flatten(), y.cC.flatten()]
pred_concat = np.column_stack(pred)

species = ['cEe', 'cEa', 'cA', 'cW', 'cV', 'cO', 'cC']

# capture the base-case (correct params, perfect IC) fit for plotting later
base_case = {'meas': np.array(meas), 'pred': pred_concat}

mean_meas = np.mean(meas, axis=0)
rmse_meas = np.sqrt(np.mean(np.square((pred_concat - meas) / mean_meas))) 

print(f'\nRMSE meas: {rmse_meas}')

# SWAP:
meas = pred_concat
clean_meas = pred_concat

# NOISE SWEEP: slack the bounds, then sweep noise level for both
# perfect initial guess and random initial guess. Collect bbox + theta
# per parameter so we can plot bbox/|theta| vs noise level.

# slack the bounds; only k1, k2 are in log space, everything else unchanged
pe.lb.lk1, pe.ub.lk1 = np.log(1e-6), np.log(1e1)   # k1 in (1, 7e5]
pe.lb.lE1, pe.ub.lE1 = pe.ic.lE1 + np.log(0.7), pe.ic.lE1 + np.log(1.3)   # ±30% on physical E
pe.lb.a1, pe.ub.a1 = 1.1 * 0.1, 1.1 * 2
pe.lb.a2, pe.ub.a2 = 1.1 * 0.1, 1.1 * 2
pe.lb.a3, pe.ub.a3 = 1.0 * 0.1, 1.0 * 2
pe.lb.a4, pe.ub.a4 = 1.0 * 0.1, 1.0 * 2
pe.lb.a5, pe.ub.a5 = 1.0 * 0.1, 1.0 * 2
pe.lb.lk2, pe.ub.lk2 = np.log(1e-6), np.log(1e1)   # k2 in (1, 5.1e5]
pe.lb.lE2, pe.ub.lE2 = pe.ic.lE2 + np.log(0.7), pe.ic.lE2 + np.log(1.3)   # ±30% on physical E


# true parameter values (base-case exponents used to generate the data)
# identify p_true based on which param do not have lb = ub
p_base = {'lk1': pe.ic.lk1, 'lE1': pe.ic.lE1, 'a1': pe.ic.a1, 'a2': pe.ic.a2, 'a3': pe.ic.a3, 'lk2': pe.ic.lk2, 'lE2': pe.ic.lE2, 'a4': pe.ic.a4, 'a5': pe.ic.a5}
p_true = {k: v for k, v in p_base.items() if float(getattr(pe.lb, k)) != float(getattr(pe.ub, k))}
p_perfect_ic = dict(p_true)

noise_levels = [0., 1e-3, 1e-2] # no noise and slight noise

results = {
    'perfect': {k: {'theta': [], 'bbox': []} for k in p_true},
    'random':  {k: {'theta': [], 'bbox': []} for k in p_true},
}

# per-fit records (noisy meas, prediction, rate-parity arrays) for the plot script
fits = []

def set_ic(values):
    for k, v in values.items():
        setattr(pe.ic, k, v)

for nl in noise_levels:
    print(f'\n===== Noise level {nl} =====')

    # regenerate noisy measurements from the clean signal — no accumulation across noise levels.
    # Fixed seed per noise level so the perfect-IC and random-IC fits see the same data.
    rng = np.random.default_rng(123)
    meas_n = clean_meas + clean_meas * nl * rng.standard_normal(size=clean_meas.shape)
    # ensure meas_n is not negative use clip
    meas_n = np.clip(meas_n, a_min=1e-8, a_max=None)
    # refresh per-species std from the noisy measurements so lsq weighting tracks the current noise level.
    # pe.lb_ub() earlier pinned lb=ub=ic for all fixed parameters, so we must update lb/ub too — not just ic.
    stds = np.std(meas_n, axis=0)
    for name, s in zip(['cEe_std', 'cEa_std', 'cA_std', 'cW_std', 'cV_std', 'cO_std', 'cC_std'], stds):
        setattr(pe.ic, name, s)
        setattr(pe.lb, name, s)
        setattr(pe.ub, name, s)
    comb_meas = np.concatenate([meas_n, Nfee[:, None], Nfa[:, None], Nfw[:, None], Nfv[:, None], Nfo[:, None], Nfc[:, None], T[:, None], alpha[:, None], beta[:, None]], axis=1)

    # do the analysis for both perfect and random initial conditions
    for ic_kind in ('perfect', 'random'):
        if ic_kind == 'perfect':
            set_ic(p_perfect_ic)
        else:
            rng_ic = np.random.default_rng(0)
            # genuine random restart: sample each param uniformly across its full [lb, ub] box
            set_ic({k: rng_ic.uniform(float(getattr(pe.lb, k)), float(getattr(pe.ub, k))) for k in p_true})

        est, y, p = pe.optimize(comb_meas.T[:, :, None])
        conf = pe.confidence(est, 0.95)
        c1 = pe.collinearity(est)


        print(f'\n--- {ic_kind} IC ---')
        print('theta: ' + ', '.join(f'{k}={float(est.theta[k]):.4g}' for k in p_true))
        print('bbox:  ' + ', '.join(f'{k}={float(conf.bbox[k]):.4g}' for k in p_true))

        # recover physical k1, k2 from the log estimates; other params are direct
        mee, mo, ma = pe.ic.cEe_mean, pe.ic.cO_mean, pe.ic.cA_mean
        k1 = float(np.exp(est.theta['lk1']))
        k2 = float(np.exp(est.theta['lk2']))
        E1 = float(np.exp(est.theta['lE1']))
        E2 = float(np.exp(est.theta['lE2']))
        ind = ['a1', 'a2', 'a3', 'a4', 'a5']
        for index in ind:
            if index not in est.theta:
                est.theta[index] = float(getattr(pe.ic, index))
        a1, a2, a3, a4, a5 = float(est.theta['a1']), float(est.theta['a2']), float(est.theta['a3']), float(est.theta['a4']), float(est.theta['a5'])
        print(f'physical: k1={k1:.4g}, E1={E1:.4g}, a1={a1:.4g}, a2={a2:.4g}, a3={a3:.4g}, a4={a4:.4g}, a5={a5:.4g}, k2={k2:.4g}, E2={E2:.4g}')
        # map to cheat form: absolute conc + 450 K ref (means and Tmean shift folded into k).
        # E1, E2 are already the physical activation energies (= exp(theta)), so the
        # reference shift from Tmean to 450 K uses them directly.
        k1c = k1 * mee**-a1 * mo**-a2 * ma**-a3 * np.exp(-E1 / pe.ic.R * (1/450 - 1/Tmean))
        k2c = k2 * mee**-a4 * mo**-a5 * np.exp(-E2 / pe.ic.R * (1/450 - 1/Tmean))
        print(f'cheat r1: {k1c:.4g} * exp(-{E1:.4g}/R*(1/T-1/450)) * cEe^{a1:.4g}*cO^{a2:.4g}*cA^{a3:.4g}')
        print(f'cheat r2: {k2c:.4g} * exp(-{E2:.4g}/R*(1/T-1/450)) * cEe^{a4:.4g}*cO^{a5:.4g}')
        # good starting point to identify collinear parameters
        print('parameters     :', c1['names'])
        print('collinear pairs:', c1['collinear_pairs'])
        print('clusters       :', c1['clusters'])

        for k in p_true:
            results[ic_kind][k]['theta'].append(float(est.theta[k]))
            results[ic_kind][k]['bbox'].append(float(conf.bbox[k]))

        e, v = np.linalg.eig(conf.H)
        # ['k1', 'E1', 'a1', 'a2', 'a3', 'k2', 'E2', 'a4', 'a5']
        print('eigenvalues of Hessian:', e)
        print('parameters:\n ', conf.Hnames)
        print(
                    'eigenvectors of Hessian (columns):\n',
                    np.array2string(v, formatter={'float_kind': lambda x: f'{x: .4f}'})
                )
        print('condition number of Hessian:', np.linalg.cond(conf.H))

        # per-fit measurement vs prediction (collected for the plot script)
        pred = [y.cEe.flatten(), y.cEa.flatten(), y.cA.flatten(), y.cW.flatten(), y.cV.flatten(), y.cO.flatten(), y.cC.flatten()]
        pred_concat = np.column_stack(pred)

        # print the rmse
        mean_meas = np.mean(meas_n, axis=0)
        rmse_meas = np.sqrt(np.mean(np.square((pred_concat - meas_n) / mean_meas)))
        print(f'RMSE meas: {rmse_meas}')

        # parity data: rate from fitted params (r_fit) vs rate from true params
        # (r_true), both evaluated on the same clean concentration trajectory so
        # only the rate-law parameters differ between the two curves.
        fit_theta = dict(p_base)
        for k in p_true:
            fit_theta[k] = float(est.theta[k])
        cc = clean_meas  # columns: cEe, cEa, cA, cW, cV, cO, cC
        r1_true, r2_true = eval_rate(p_base, cc[:, 0], cc[:, 5], cc[:, 2], T)
        r1_fit, r2_fit = eval_rate(fit_theta, cc[:, 0], cc[:, 5], cc[:, 2], T)

        # store everything this fit needs for plotting
        fits.append({
            'nl': nl,
            'ic_kind': ic_kind,
            'meas_n': np.array(meas_n),
            'pred': pred_concat,
            'r1_true': np.asarray(r1_true), 'r2_true': np.asarray(r2_true),
            'r1_fit': np.asarray(r1_fit),  'r2_fit': np.asarray(r2_fit),
            'theta1': np.array([k1c/(3.73*1e-7), E1/(2.7*1e3), a1, a2, a3,]),
            'theta2': np.array([k2c/(3.73*1e-7), E2/(2.5*1e3), a4, a5])
        })

param_keys = list(p_true)

# tabular summary at the end of the log
print('\n===== Noise sweep summary =====')
header = 'noise_level  ic_kind  ' + '  '.join(f'{k}(theta/bbox)' for k in param_keys)
print(header)
for i, nl in enumerate(noise_levels):
    for ic_kind in ('perfect', 'random'):
        cols = []
        for k in param_keys:
            t = results[ic_kind][k]['theta'][i]
            b = results[ic_kind][k]['bbox'][i]
            cols.append(f'{t:.4g}/{b:.4g}')
        print(f'{nl:>10g}  {ic_kind:>7s}  ' + '  '.join(cols))

# save everything the plot script needs to a pickle alongside this file
plot_data = {
    'case_type':    case_type,
    'species':      species,
    'time':         np.asarray(time),
    'pcat':         float(pe.ic.pcat),
    'noise_levels': noise_levels,
    'param_keys':   param_keys,
    'p_true':       p_true,
    'results':      results,
    'base_case':    base_case,
    'fits':         fits,
}
PickleTool.save(plot_data, f'{log_file_base}.pickle')

# restore stdout
sys.stdout = sys.__stdout__
log_handle.close()
