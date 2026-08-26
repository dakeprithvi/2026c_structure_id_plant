# [depends] %LIB%/VAc_struct_lib.py
# [depends] VAc_meas.pickle
# [depends] %LIB%/utils.py

import sys, os, jax, optax, time
sys.path.append('lib/')
import numpy as np
from scipy.stats.qmc import LatinHypercube
import VAc_struct_lib as _bbox_mod
from VAc_struct_lib import (learning_dict, rate_1, rate_2, dum_input_1, dum_input_2)
from utils import PickleTool
import jax.numpy as jnp
# ensure some high-level settings for JAX. The NN modules set
# param_dtype=float64 at their Dense layers, so nvec is float64 from init and
# stays float64 throughout (no later promotion needed for the LBFGS phases).
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config
config.update("jax_enable_x64", True)

safe_run = None
epoch_budget = 500
safe_hess = None
hess_calc = False
cheat = True
noise = False

# --- Seeding ---
init_noise = True   # vary nvec initialization across MC iterations
alg_noise  = False  # fix Adam batching across MC iterations
dat_noise  = False  # fix measurement noise across MC iterations

# --- MC configuration ---
# Usage: python ..._cheat.py <case_type> <mc_iter> [cheat]
# cheat defaults to the module flag above; pass 0/1 as the 3rd arg to override.
case_type = sys.argv[1] if len(sys.argv) > 1 else 'test'
mc_iter   = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if len(sys.argv) > 3:
    cheat = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    noise = bool(int(sys.argv[4]))
cheat_tag = 'cheat' if cheat else 'nocheat'
noise_tag = 'noise' if noise else 'nonoise'

os.makedirs('log', exist_ok=True)
# each (case, mc_iter, cheat) run writes its own pickle here so parallel jobs
# never collide; VAc_collect_param_fit_nn.py merges them into one nested pickle.
os.makedirs('pickle', exist_ok=True)

# log file per iteration
log_file_base = os.path.basename(__file__).replace('.py', '')
log_handle = open(f'log/{log_file_base}_{case_type}_{mc_iter}_{cheat_tag}_{noise_tag}.txt', 'w')
sys.stdout = log_handle

print(f'MC iteration: {mc_iter}, case_type: {case_type}', flush=True)

init_key = np.random.RandomState(mc_iter if init_noise else 0)
alg_key  = np.random.RandomState(mc_iter if alg_noise  else 0)
dat_key  = np.random.RandomState(mc_iter if dat_noise  else 0)

# print the experiment configuration
print(f'init_noise: {init_noise}, alg_noise: {alg_noise}, dat_noise: {dat_noise}', flush=True)
print(f'init_key check: {np.random.RandomState(mc_iter if init_noise else 0).normal(size=3)}', flush=True)
print(f'alg_key check:  {np.random.RandomState(mc_iter if alg_noise  else 0).permutation(5)}', flush=True)
print(f'dat_key check:  {np.random.RandomState(mc_iter if dat_noise  else 0).normal(size=3)}', flush=True)

# pick the right case for training
data_meas = PickleTool.load('VAc_meas.pickle', 'read')
data = data_meas[case_type]

# data preparation
val_index = data['val_index']
meas_t  = data['time'][:val_index]
meas_c0 = data['meas'][0, :]
meas    = data['meas'][:val_index]
meas    = meas + meas * 0. * dat_key.normal(size=meas.shape)
val_t   = data['time'][val_index:]
val_c0  = data['meas'][val_index, :]
val     = data['meas'][val_index:]
num_of_batch = int(data['train_jumps'])
clean_meas = data['clean_meas'][:val_index]
clean_val  = data['clean_meas'][val_index:]

# batch creation
meas_batch = jnp.array([i for i in jnp.split(meas, num_of_batch, axis=0)])
eps = 1e-5
meas_batch = jnp.clip(meas_batch, min=0.0, max=jnp.inf) + eps
t_batch = jnp.array([i for i in jnp.split(meas_t, num_of_batch, axis=0)])
meas_c0_batch = meas_batch[:, 0, :]

# params dict
params = {}
params['case']    = 0 if case_type in ['new', 'test'] else 1
params['dt0']     = 0.1
params['P']       = 882529.0
params['R']       = 8.314
params['Vr']      = 0.010752
params['e']       = 0.8
params['pcat']    = 385.0
params['mu']      = 0.03
params['extrap']  = 0.0
params['nt']      = 622.0
params['At']      = 0.25 * np.pi * (3.7e-2) ** 2
params['l']       = 10.0
params['weights'] = jnp.array([1, 1, 1, 1, 1, 1, 1])
params['time']    = data['time']
params['NEe']     = data['NEe']
params['NA']      = data['NA']
params['NO']      = data['NO']
params['NW']      = data['NW']
params['NV']      = data['NV']
params['NC']      = data['NC']
params['alpha']   = data['alpha']
params['beta']    = data['beta']
params['constQ']  = 1.0
params['T']       = data['T2']
params['Qmeas']   = data['Q'][:val_index] / params['nt']
params['Qmeas_batch'] = jnp.array([i for i in jnp.split(params['Qmeas'], num_of_batch, axis=0)])
params['weight_q'] = 0

mean_T = np.mean(data['T2'][:val_index])
std_T  = np.std(data['T2'][:val_index]) + 1e-2
params['mean'] = np.mean(data['meas'][:val_index], axis=0)
params['std']  = np.std(data['meas'][:val_index], axis=0)
params['mean'] = np.concatenate((params['mean'], np.array([mean_T])))
params['std']  = np.concatenate((params['std'],  np.array([std_T])))

# get functions from the bbox library
solve_ode     = learning_dict['solve_ode']
training_step = learning_dict['training_step']
lbfgs_training_step = learning_dict['lbfgs_training_step']
lbfgs_optimizer     = learning_dict['lbfgs_optimizer']
batch_solve   = learning_dict['batch_solve']
rate_true     = learning_dict['rate_true']
save_Q        = learning_dict['save_Q']
save_rate     = learning_dict['save_rate']
save_rxn      = learning_dict['save_rxn']
hessian_fn    = learning_dict['hessian_fn']
c_matrix      = learning_dict['c_matrix']

rate_stats = {'r1_mean': data['r1_tilde'].mean(0), 'r1_std': data['r1_tilde'].std(0),
              'r2_mean': data['r2_tilde'].mean(0), 'r2_std': data['r2_tilde'].std(0)}


# nvec3[:, 0] = log10(rate_mean), nvec3[:, 1] = log10(rate_scale)
# centers[0,1] → r1_mean, r2_mean; centers[2,3] → r1_std, r2_std (all log10)
centers = np.array([np.log10(rate_stats['r1_mean']), np.log10(rate_stats['r2_mean']),
                    np.log10(rate_stats['r1_std']),  np.log10(rate_stats['r2_std']),])

# print some useful stuff about the rates
print('r_tilde is tank average it is equal to r_end for a single CSTR')
r1 = np.asarray(data['r1_tilde'])
r2 = np.asarray(data['r2_tilde'])
print(f'--- {case_type} ---')
print(f'  r1: min={r1.min():.4g}  max={r1.max():.4g}  mean={r1.mean():.4g}  std={r1.std():.4g}')
print(f'  r2: min={r2.min():.4g}  max={r2.max():.4g}  mean={r2.mean():.4g}  std={r2.std():.4g}')
print(f'  ratio r1 max/ r1 min: {r1.max()/r1.min():.4g}')
print(f'  ratio r2 max/ r2 min: {r2.max()/r2.min():.4g}')
print(f'  ratio r1/r2: mean={r1.mean()/r2.mean():.4g}  (min={ (r1/r2).min():.4g}  max={(r1/r2).max():.4g})')


# LHS initialization for nvec3: ±1.0 window in log10 space across 50 MC iterations
n_mc = 50
lhs_sampler = LatinHypercube(d=4, seed=42)
lhs_samples = lhs_sampler.random(n=n_mc)           # shape (50, 4) in [0, 1]
nvec3_flat = centers if mc_iter == 0 else centers + 0 * (lhs_samples[mc_iter] * 2.0 - 1.0)
nvec3 = jnp.array([[nvec3_flat[0], nvec3_flat[2]],
                    [nvec3_flat[1], nvec3_flat[3]]])

# nvec: fully re-initialize NN weights from scratch per mc_iter using init_key-derived JAX PRNGKeys
init_seed = int(init_key.randint(0, 2**31))
nvec1 = rate_1.init(jax.random.PRNGKey(init_seed),     dum_input_1)
nvec2 = rate_2.init(jax.random.PRNGKey(init_seed + 1), dum_input_2)
nvec  = (nvec1, nvec2, nvec3)
print(f'init_seed: {init_seed}', flush=True)
print(f'nvec3 centers (log10): {centers}', flush=True)
print(f'nvec3 init:\n{nvec3}', flush=True)

# total number of trainable parameters, by unpacking nvec into its leaves
nn_param_count = sum(int(np.size(leaf)) for leaf in jax.tree_util.tree_leaves(nvec))
print(f'nn_param_count: {nn_param_count} '
      f'(rate_1: {sum(int(np.size(l)) for l in jax.tree_util.tree_leaves(nvec1))}, '
      f'rate_2: {sum(int(np.size(l)) for l in jax.tree_util.tree_leaves(nvec2))}, '
      f'nvec3: {int(np.size(nvec3))})', flush=True)

# --- Self-consistency: regenerate the measurements from the SAME ODE solver ---
params['rate_select'] = jnp.array([0.0])
sim = solve_ode(nvec, meas_c0, data['time'], params)
sim_ys = sim.ys

clean_meas = sim_ys[:val_index]
clean_val  = sim_ys[val_index:]
if noise == False:
    meas    = clean_meas + clean_meas * 0. * dat_key.normal(size=clean_meas.shape)
else:
    meas    = clean_meas + clean_meas * 1e-2 * dat_key.normal(size=clean_meas.shape)
val     = clean_val
meas_c0 = sim_ys[0]
val_c0  = sim_ys[val_index]

# rebuild the training batches from the regenerated measurements
meas_batch = jnp.array([i for i in jnp.split(meas, num_of_batch, axis=0)])
meas_batch = jnp.clip(meas_batch, min=0.0, max=jnp.inf) + eps
meas_c0_batch = meas_batch[:, 0, :]

print(f'self-consistent meas regenerated from true-rate ODE solve; '
      f'meas: {meas.shape}, val: {val.shape}', flush=True)

# Optional sanity plot (data vs the same solver's trajectory):
# import matplotlib.pyplot as plt
# plt.figure()
# for i in range(sim_ys.shape[1]):
#     plt.subplot(2, 4, i+1)
#     plt.plot(meas_t, clean_meas[:, i], 'b', mfc='none', markersize=1)
#     plt.plot(meas_t, meas[:, i], 'ro', mfc='none', markersize=1)
# plt.tight_layout()
# plt.savefig(f'plots/VAc_param_fit_{case_type}.pdf')
# plt.close()

# restore the NN rate model for the fit
params['rate_select'] = jnp.array([2.0])


if cheat:
    # ------------------------------------------------------------------
    # Stage 0 -- direct reaction-rate pretraining (the "cheat").
    # Fit the NN rate heads to the TRUE rates (rate_true) evaluated at the
    # measured states, BEFORE the expensive ODE-fit, so the main loop starts
    # in a good basin and only has to fine-tune. nvec3 (the affine denorm
    # mean/scale) is seeded from the true-rate stats and FROZEN since we know 
    # the rate measurements; only the
    # NN weights (nvec1, nvec2) move during pretraining.
    # ------------------------------------------------------------------
    temp_meas = params['T'][:val_index]
    temp_val  = params['T'][val_index:]
    keys = ['rtee', 'rtea', 'rta', 'rtw', 'rtv', 'rto', 'rtc',
            'rnee', 'rnea', 'rna', 'rnw', 'rnv', 'rno', 'rnc']

    epoch_rxn_adamw = 5000   # Stage 0a: AdamW steps on the rxn-rate loss
    epoch_rxn_lbfgs = 5000    # Stage 0b: LBFGS fine-tune steps on the same loss
    lr_rxn          = 1e-2

    # true rates over the measurement trajectory are the (fixed) regression
    # targets -- rate_true ignores the NN weights, so r*_mean/std are constants.
    r1, r2, _, _ = save_rxn(nvec, meas, temp_meas, params)
    r1_mean, r1_std = r1.mean(), r1.std()
    r2_mean, r2_std = r2.mean(), r2.std()

    r1ind_mean, r1ind_std = jnp.log10(r1_mean), jnp.log10(r1_std)
    r2ind_mean, r2ind_std = jnp.log10(r2_mean), jnp.log10(r2_std)

    nvec_mean_std = jnp.array([[r1ind_mean, r1ind_std], 
                        [r2ind_mean, r2ind_std]])

    # should match the centers we seeded nvec3 with above, since those were computed from the same rates
    print(f'nvec3 seed from true-rate stats (log10):\n{nvec_mean_std}', flush=True)
    nvec = (nvec[0], nvec[1], nvec_mean_std)

    optimizer = optax.adamw(learning_rate=lr_rxn)
    opt_state = optimizer.init(nvec)

    @jax.jit
    def loss_rxn(nvec, meas, temp_meas, params):
        rt1, rt2, rn1, rn2 = save_rxn(nvec, meas, temp_meas, params)
        # normalize
        rtn1, rnn1 = (rt1 - r1_mean) / r1_std, (rn1 - r1_mean) / r1_std
        rtn2, rnn2 = (rt2 - r2_mean) / r2_std, (rn2 - r2_mean) / r2_std
        # get mean difference
        err = jnp.mean(jnp.square(rtn1 - rnn1)) + jnp.mean(jnp.square(rtn2 - rnn2))
        return err

    @jax.jit
    def training_rxn_step(nvec, meas, temp_meas, params, opt_state):
        grad_fn = jax.value_and_grad(loss_rxn)
        loss, grad = grad_fn(nvec, meas, temp_meas, params)
        g1, g2, g3 = grad
        # no update to nvec2 required
        g3 = jnp.zeros_like(g3)
        grad = (g1, g2, g3)
        updates, opt_state = optimizer.update(grad, opt_state, nvec)
        u1, u2, u3 = updates
        updates = (u1, u2, jnp.zeros_like(u3))
        nvec = optax.apply_updates(nvec, updates)
        return nvec, opt_state, loss

    # --- Stage 0a: AdamW on the rxn-rate loss ---
    loss_per_epoch = []
    tic = time.time()
    for step in range(epoch_rxn_adamw):
        nvec, opt_state, loss = training_rxn_step(nvec, meas, temp_meas, params, opt_state)
        loss_per_epoch.append(loss)
        if step % 100 == 0:
            print(f'[rxn-adam]  Epoch: {step}, Loss: {loss:.6f}', flush=True)

    # --- Stage 0b: LBFGS fine-tune on the SAME rxn-rate loss ---
    # loss_rxn is a pointwise rate regression (no ODE solve) with FIXED targets,
    # so LBFGS here is cheap and well-conditioned. nvec is float64 throughout
    # (param_dtype=float64 at init), matching the float64 grads the zoom line
    # search requires. nvec3 stays frozen, matching Stage 0a.
    rxn_lbfgs_opt = optax.lbfgs()
    lbfgs_state = rxn_lbfgs_opt.init(nvec)

    @jax.jit
    def lbfgs_rxn_step(nvec, meas, temp_meas, params, opt_state):
        def value_fn(nv):
            return loss_rxn(nv, meas, temp_meas, params)
        value_and_grad = optax.value_and_grad_from_state(value_fn)
        loss, grad = value_and_grad(nvec, state=opt_state)
        g1, g2, g3 = grad
        grad = (g1, g2, jnp.zeros_like(g3))   # freeze nvec3
        updates, opt_state = rxn_lbfgs_opt.update(
            grad, opt_state, nvec, value=loss, grad=grad, value_fn=value_fn)
        u1, u2, u3 = updates
        updates = (u1, u2, jnp.zeros_like(u3))
        nvec = optax.apply_updates(nvec, updates)
        return nvec, opt_state, loss

    for step in range(epoch_rxn_lbfgs):
        nvec, lbfgs_state, loss = lbfgs_rxn_step(nvec, meas, temp_meas, params, lbfgs_state)
        loss_per_epoch.append(loss)
        if step % 20 == 0:
            print(f'[rxn-lbfgs] Epoch: {step}, Loss: {loss:.6f}', flush=True)
    toc = time.time()
    rxn_training_time = (toc - tic) / 60  # minutes

    # --- save the direct-pretraining result to its own pickle ---
    direct_data = {}
    direct_data['mc_iter']  = mc_iter
    direct_data['nvec']     = nvec
    direct_data['loss_rxn'] = loss_per_epoch
    direct_data['mean']     = params['mean'].reshape(-1, 1)
    direct_data['std']      = params['std'].reshape(-1, 1)
    meas_pred = solve_ode(nvec, meas_c0, meas_t[:safe_run], params)
    val_pred  = solve_ode(nvec, val_c0,  val_t[:safe_run],  params)
    direct_data['meas_pred'] = meas_pred.ys
    direct_data['val_pred']  = val_pred.ys
    direct_data['rmse_meas'] = jnp.sqrt(jnp.mean(jnp.square((meas[:safe_run, :] - meas_pred.ys) / params['mean'][:7])))
    direct_data['rmse_val']  = jnp.sqrt(jnp.mean(jnp.square((val[:safe_run, :]  - val_pred.ys)  / params['mean'][:7])))
    direct_data['training_time'] = rxn_training_time
    for k, value in zip(keys, save_rate(nvec, clean_meas, temp_meas, params)):
        direct_data[k + '_meas'] = value
    for k, value in zip(keys, save_rate(nvec, clean_val, temp_val, params)):
        direct_data[k + '_val'] = value
    PickleTool.save(direct_data, f'pickle/{log_file_base}_{case_type}_{mc_iter}_{cheat_tag}_{noise_tag}_direct.pickle')

    print(f'\n[rxn] pretraining time (min): {rxn_training_time}', flush=True)
    print(f'[rxn] RMSE meas: {direct_data["rmse_meas"]}', flush=True)
    print(f'[rxn] RMSE val:  {direct_data["rmse_val"]}', flush=True)
    print(f'[rxn] nvec3 (recovered mean/std): {10 ** nvec[2]}', flush=True)

    # nvec now carries the pretrained weights (float64) into the main ODE-fit
    # loop below, which re-inits its optimizer/opt_state from this nvec.
else:
    print('cheat=False: skipping direct rate pretraining; '
          'main loop trains from the LHS-seeded nvec.', flush=True)

# start training
tic = time.time()
loss_container = []
lossv_container = []
best_loss = jnp.inf
patience = 8000 # we use fixed epoch budget approach
counter = 0
strategy = [5, 5] # number of mini-batches per epoch.
steps_strategy = epoch_budget // 2
batch_size = strategy[0]
lr = 5e-2
epochs = epoch_budget
optimizer = optax.adamw(learning_rate=lr)
_bbox_mod.optimizer = optimizer   # propagate lr into training_step's closure
opt_state = optimizer.init(nvec)
all_bind = jnp.arange(num_of_batch)   # full-batch index set for the LBFGS phase
lbfgs_active = False
patience_exceed = []
best_nvec = []
patience_violations = 0
max_violations = 3

for step in range(epochs):
    if step > steps_strategy - 1:
        batch_size = strategy[1]
        if not lbfgs_active:   # switch AdamW -> LBFGS at the halfway point
            # nvec is float64 throughout (param_dtype=float64 at init), so it
            # already matches the float64 grads LBFGS's line search requires.
            opt_state = lbfgs_optimizer.init(nvec)
            lbfgs_active = True
            print(f'Switching to LBFGS optimizer at epoch {step}', flush=True)
    if lbfgs_active:
        nvec, opt_state, loss = lbfgs_training_step(nvec, meas_c0_batch, t_batch,
                                                    meas_batch, opt_state, params, all_bind)
        loss_container.append(loss)
    else:
        bind = jnp.array(alg_key.permutation(num_of_batch)).reshape(-1, num_of_batch // batch_size)
        loss_per_epoch = []
        for i in range(bind.shape[0]):
            nvec, opt_state, loss = training_step(nvec, meas_c0_batch, t_batch, meas_batch,
                                                  opt_state, params, bind[i, :])
            loss_per_epoch.append(loss)
        loss_container.append(jnp.mean(jnp.array(loss_per_epoch)))
    print(f'Epoch: {step}, Loss: {loss_container[-1]:.6f}', flush=True)
    val_pred = solve_ode(nvec, val_c0, val_t[:100], params)
    mse_val = float(jnp.mean(jnp.square(val[:100, :] - val_pred.ys)))
    lossv_container.append(mse_val)
    if mse_val < best_loss:
        best_loss = mse_val
        best_nvec.append(nvec)
        counter = 0
    else:
        counter += 1
    print(f'best_val_loss: {best_loss:.6f}')
    if counter > patience:
        print('Ran out of patience \n logging and continuing training', flush=True)
        patience_exceed.append(step)
        counter = 0
        patience_violations += 1
        if patience_violations > max_violations:
            print('Too many patience violations', flush=True)
            break

toc = time.time()
best_save_nvec = best_nvec[-1]
training_time = (toc - tic) / 60

# test fit on meas and val
meas_pred = solve_ode(nvec, meas_c0, meas_t[:safe_run], params)
val_pred  = solve_ode(nvec, val_c0,  val_t[:safe_run],  params)

rmse_meas = jnp.sqrt(jnp.mean(jnp.square((meas[:safe_run, :] - meas_pred.ys) / params['mean'][:7])))
rmse_val  = jnp.sqrt(jnp.mean(jnp.square((val[:safe_run, :]  - val_pred.ys)  / params['mean'][:7])))

if hess_calc:
    hess_nvec, sigma2 = hessian_fn(nvec, meas[:safe_hess, :], meas_c0, meas_t[:safe_hess], params)
else:
    hess_nvec, sigma2 = None, None

keys = ['rtee', 'rtea', 'rta', 'rtw', 'rtv', 'rto', 'rtc',
        'rnee', 'rnea', 'rna', 'rnw', 'rnv', 'rno', 'rnc']
temp_meas = params['T'][:val_index]
temp_val  = params['T'][val_index:]
eps = 1e-5
clean_meas = jnp.clip(clean_meas, min=0.0, max=jnp.inf) + eps
clean_val  = jnp.clip(clean_val,  min=0.0, max=jnp.inf) + eps
meas_tuple = save_rate(nvec, clean_meas, temp_meas, params)
val_tuple  = save_rate(nvec, clean_val,  temp_val,  params)

# C = E_{c~rho}[ (dr/dc)^T (dr/dc) ] averaged over the trajectory states (7x7)
C_meas = c_matrix(nvec, clean_meas, temp_meas, params)
C_val  = c_matrix(nvec, clean_val,  temp_val,  params)

Qpred = save_Q(meas_t[:safe_run], meas_pred.ys[:safe_run], params, nvec)
qmeas = params['Qmeas'] * params['nt']

data = {}
data['mc_iter']       = mc_iter
data['meas']          = meas
data['val']           = val
data['meas_pred']     = meas_pred.ys
data['val_pred']      = val_pred.ys
data['mean']          = params['mean'].reshape(-1, 1)
data['std']           = params['std'].reshape(-1, 1)
data['training_time'] = training_time
data['rmse_meas']     = rmse_meas
data['rmse_val']      = rmse_val
data['loss']          = loss_container
data['loss_v']        = lossv_container
data['nvec']          = nvec
data['q_pred']        = Qpred
data['q_meas']        = qmeas
data['hess_nvec']     = hess_nvec
data['sigma2']        = sigma2
data['C_meas']        = C_meas
data['C_val']         = C_val

for k, value in zip(keys, meas_tuple):
    data[k + '_meas'] = value
for k, value in zip(keys, val_tuple):
    data[k + '_val'] = value

data['best_nvec'] = best_save_nvec

PickleTool.save(data, f'pickle/{log_file_base}_{case_type}_{mc_iter}_{cheat_tag}_{noise_tag}.pickle')

print(f'\n Training time (hrs): {training_time / 60}', flush=True)
print(f'\n RMSE meas: {rmse_meas}', flush=True)
print(f'\n RMSE val: {rmse_val}', flush=True)
print(f'\n nvec3: {10 ** nvec[2]}', flush=True)

sys.stdout = sys.__stdout__
log_handle.close()
