# [depends] %LIB%/VAc_bbox_lib.py
# [depends] VAc_meas.pickle
# [depends] %LIB%/utils.py

import sys, os, jax, optax, time
sys.path.append('lib/')
import numpy as np
import VAc_bbox_lib as _bbox_mod
from VAc_bbox_lib import learning_dict, cbox_nn, qbox_nn, dum_input
from utils import PickleTool
import jax.numpy as jnp
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config
config.update("jax_enable_x64", True)

# ---- control variables ----
safe_run     = None
epoch_budget = 500
safe_hess    = None
hess_calc    = False
learn_q      = False
log_nn       = False
noise        = False

# --- Seeding ---
init_noise   = True   # vary nvec initialization across MC iterations
alg_noise    = False  # fix Adam batching across MC iterations
dat_noise    = False  # fix measurement noise across MC iterations

# --- MC configuration ---
# Usage: python VAc_param_fit_bbox_param_noise.py <case_type> <mc_iter> [noise]
# noise defaults to the module flag above; pass 0/1 as the 3rd arg to override.
case_type = sys.argv[1] if len(sys.argv) > 1 else 'test'
mc_iter   = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if len(sys.argv) > 3:
    noise = bool(int(sys.argv[3]))
noise_tag = 'noise' if noise else 'nonoise'

os.makedirs('log', exist_ok=True)
os.makedirs('pickle', exist_ok=True)

# log file per iteration
log_file_base = os.path.basename(__file__).replace('.py', '')
log_handle = open(f'log/{log_file_base}_{case_type}_{mc_iter}_{noise_tag}.txt', 'w')
sys.stdout = log_handle

print(f'MC iteration: {mc_iter}, case_type: {case_type}', flush=True)

init_key = np.random.RandomState(mc_iter if init_noise else 0)
alg_key  = np.random.RandomState(mc_iter if alg_noise  else 0)
dat_key  = np.random.RandomState(mc_iter if dat_noise  else 0)

print(f'init_noise: {init_noise}, alg_noise: {alg_noise}, dat_noise: {dat_noise}', flush=True)
print(f'init_key check: {np.random.RandomState(mc_iter if init_noise else 0).normal(size=3)}', flush=True)
print(f'alg_key check:  {np.random.RandomState(mc_iter if alg_noise  else 0).permutation(5)}', flush=True)
print(f'dat_key check:  {np.random.RandomState(mc_iter if dat_noise  else 0).normal(size=3)}', flush=True)

# ---- load and prepare data ----
data_meas = PickleTool.load('VAc_meas.pickle', 'read')
data      = data_meas[case_type]

val_index    = data['val_index']
meas_t       = data['time'][:val_index]
val_t        = data['time'][val_index:]
num_of_batch = int(data['train_jumps'])

clean_meas   = data['clean_meas'][:val_index]
clean_val    = data['clean_meas'][val_index:]
if noise == False:
    meas = clean_meas + clean_meas * 0.   * dat_key.normal(size=clean_meas.shape)
else:
    meas = clean_meas + clean_meas * 1e-2 * dat_key.normal(size=clean_meas.shape)
val          = clean_val
meas_c0      = clean_meas[0, :]
val_c0       = clean_val[0, :]

# batch creation
meas_batch    = jnp.array([i for i in jnp.split(meas, num_of_batch, axis=0)])
eps           = 1e-5
meas_batch    = jnp.clip(meas_batch, min=0.0, max=jnp.inf) + eps
t_batch       = jnp.array([i for i in jnp.split(meas_t, num_of_batch, axis=0)])
meas_c0_batch = meas_batch[:, 0, :]

# ---- control normalization ----
control_u = jnp.array([
    data['NEe'][:val_index], data['NA'][:val_index],  data['NO'][:val_index],
    data['NW'][:val_index],  data['NV'][:val_index],  data['NC'][:val_index],
    data['alpha'][:val_index], data['beta'][:val_index], data['T2'][:val_index],
]).T
mean_u = jnp.mean(control_u, axis=0)
std_u  = jnp.std(control_u,  axis=0) + 1e-2

# ---- params dict ----
params = {}
params['extrap']      = 0.0
params['dt0']         = 1e-1
params['nt']          = 622.0
params['P']           = 882529.0
params['R']           = 8.314
params['Vr']          = 0.010752
params['e']           = 0.8
params['pcat']        = 385.0
params['mu']          = 0.03
params['At']          = 0.25 * np.pi * (3.7e-2) ** 2
params['l']           = 10.0
params['constQ']      = 1.0
params['weights']     = jnp.array([1, 1, 1, 1, 1, 1, 1], dtype=jnp.float64)
params['time']        = data['time']
params['NEe']         = data['NEe']
params['NA']          = data['NA']
params['NO']          = data['NO']
params['NW']          = data['NW']
params['NV']          = data['NV']
params['NC']          = data['NC']
params['alpha']       = data['alpha']
params['beta']        = data['beta']
params['T']           = data['T2']
params['mean_u']      = mean_u
params['std_u']       = std_u
params['Qmeas']       = data['Q'][:len(meas_t)] / params['nt']
params['Qmeas_batch'] = jnp.array([i for i in jnp.split(params['Qmeas'], num_of_batch, axis=0)])
params['learn_Q']     = 0.0
params['mean_y']      = np.mean(data['meas'][:val_index], axis=0)
params['std_y']       = np.std(data['meas'][:val_index],  axis=0)

# ---- physical dc/dt statistics for output scaling ----
dcdt_phys           = np.asarray(data['dcdt'])[:, 0, :].T[:val_index]   # (val_index, 7) physical dc/dt
params['mean_dcdt'] = jnp.asarray(dcdt_phys.mean(axis=0))              # (7,)
params['std_dcdt']  = jnp.asarray(dcdt_phys.std(axis=0) + 1e-12)       # (7,)

# ---- unpack library functions ----
batch_solve        = learning_dict['batch_solve']
solve_ode          = learning_dict['solve_ode']
training_step_cbox = learning_dict['training_step_cbox']
lbfgs_training_step_cbox = learning_dict['lbfgs_training_step_cbox']
lbfgs_optimizer    = learning_dict['lbfgs_optimizer']
save_nn_output     = learning_dict['save_nn_output']
save_q_nn_output   = learning_dict['save_q_nn_output']
train_q            = learning_dict['train_q']
hessian_fn_bbox    = learning_dict['hessian_fn_bbox']
nn_output_stats    = learning_dict['nn_output_stats']
save_prod_rates    = learning_dict['save_prod_rates']

# ---- initialize nvec ----
init_seed = int(init_key.randint(0, 2**31))
nvec1 = cbox_nn.init(jax.random.PRNGKey(init_seed),     dum_input)
nvec2 = qbox_nn.init(jax.random.PRNGKey(init_seed + 1), dum_input)
# nvec3 = learnable dc/dt standardization, initialized AT THE TRUTH (like the
# structured fit VAc_param_fit_nn_param_noise.py seeds mult_r from the true rate
# stats): column 0 = offset = mean_dcdt (linear), column 1 = log10(std_dcdt)
# (the scale is applied as 10**col1 in nn_bbox).
nvec3 = jnp.stack([params['mean_dcdt'], jnp.log10(params['std_dcdt'])], axis=1)  # (7, 2)
nvec  = (nvec1, nvec2, nvec3)
print(f'init_seed: {init_seed}', flush=True)
# total number of trainable parameters, by unpacking nvec into its leaves
nn_param_count = sum(int(np.size(leaf)) for leaf in jax.tree_util.tree_leaves(nvec))
print(f'nn_param_count: {nn_param_count} '
      f'(cbox: {sum(int(np.size(l)) for l in jax.tree_util.tree_leaves(nvec1))}, '
      f'qbox: {sum(int(np.size(l)) for l in jax.tree_util.tree_leaves(nvec2))}, '
      f'stats: {int(np.size(nvec3))})', flush=True)

# ---- training ----
tic = time.time()
loss_container  = []
lossv_container = []
best_loss           = jnp.inf
patience            = 8000
counter             = 0
strategy            = [5, 5] # number of mini-batches per epoch
steps_strategy      = epoch_budget // 2
batch_size          = strategy[0]
lr                  = 1e-2
epochs              = epoch_budget
optimizer           = optax.adamw(learning_rate=lr)
_bbox_mod.optimizer = optimizer
opt_state           = optimizer.init(nvec)
all_bind            = jnp.arange(num_of_batch)   # full-batch index set for the LBFGS phase
lbfgs_active        = False
patience_exceed     = []
best_nvec           = []
patience_violations = 0
max_violations      = 3

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
        nvec, opt_state, loss = lbfgs_training_step_cbox(
            nvec, params, t_batch, meas_c0_batch, meas_batch, opt_state, all_bind)
        loss_container.append(loss)
    else:
        bind = jnp.array(alg_key.permutation(num_of_batch)).reshape(-1, num_of_batch // batch_size)
        loss_per_epoch = []
        for i in range(bind.shape[0]):
            nvec, opt_state, loss = training_step_cbox(
                nvec, params, t_batch, meas_c0_batch, meas_batch, opt_state, bind[i, :])
            loss_per_epoch.append(loss)
        loss_container.append(jnp.mean(jnp.array(loss_per_epoch)))

    # validation loss (short horizon to save time)
    val_c0_norm   = (val_c0 - params['mean_y']) / params['std_y']
    val_pred_norm = solve_ode(nvec, val_c0_norm, val_t[:100], params).ys
    val_pred_100  = val_pred_norm * params['std_y'] + params['mean_y']
    mse_val = float(jnp.mean(jnp.square(val[:100, :] - val_pred_100)))
    lossv_container.append(mse_val)

    print(f'Epoch: {step}, Loss: {loss_container[-1]:.6f}, val_loss: {mse_val:.6f}', flush=True)
    if log_nn:
        rhs_mean, rhs_std = nn_output_stats(nvec, params, meas, meas_t)
        print(f'  rhs_mean: {np.array(rhs_mean).round(4)},  rhs_std: {np.array(rhs_std).round(4)}', flush=True)

    if mse_val < best_loss:
        best_loss = mse_val
        best_nvec.append(nvec)
        counter = 0
    else:
        counter += 1
    print(f'best_val_loss: {best_loss:.6f}', flush=True)

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
training_time  = (toc - tic) / 60

# ---- final evaluation ----
meas_c0_norm  = (meas_c0 - params['mean_y']) / params['std_y']
meas_pred_norm = solve_ode(nvec, meas_c0_norm, meas_t[:safe_run], params).ys
meas_pred      = meas_pred_norm * params['std_y'] + params['mean_y']

val_c0_norm  = (val_c0 - params['mean_y']) / params['std_y']
val_pred_norm = solve_ode(nvec, val_c0_norm, val_t[:safe_run], params).ys
val_pred      = val_pred_norm * params['std_y'] + params['mean_y']

rmse_meas = jnp.sqrt(jnp.mean(jnp.square((meas[:safe_run, :] - meas_pred) / params['mean_y'])))
rmse_val  = jnp.sqrt(jnp.mean(jnp.square((val[:safe_run, :]  - val_pred)  / params['mean_y'])))

rhs = save_nn_output(nvec, params, meas_t[:safe_run], meas[:safe_run])

# ---- hessian analysis ----
if hess_calc:
    hess_nvec, sigma2 = hessian_fn_bbox(
        nvec, meas[:safe_hess, :], meas_c0, meas_t[:safe_hess], params)
else:
    hess_nvec, sigma2 = None, None

# ---- production rate extraction via stoichiometric pseudo-inverse ----
nu = np.array([
    [-1.,   -1.],
    [ 0.,    0.],
    [-1.,    0.],
    [ 1.,    2.],
    [ 1.,    0.],
    [-0.5,  -3.],
    [ 0.,    2.],
])
nu_pinv    = np.linalg.pinv(nu)                               # shape (2, 7)
R_mat      = np.array(save_prod_rates(meas_t[:safe_run], meas_pred, nvec, params))  # (meas, 7)
rxn_rates  = (nu_pinv @ R_mat.T).T                           # (meas, 2): [r1, r2]
r1_mean, r1_std = float(np.mean(rxn_rates[:, 0])), float(np.std(rxn_rates[:, 0]))
r2_mean, r2_std = float(np.mean(rxn_rates[:, 1])), float(np.std(rxn_rates[:, 1]))
print(f'\n Estimate of reaction rates (r1, r2) via stoichiometric pseudo-inverse:', flush=True)
print(f'\n r1: mean={r1_mean:.6e}, std={r1_std:.6e}', flush=True)
print(f'\n r2: mean={r2_mean:.6e}, std={r2_std:.6e}', flush=True)

# ---- save results ----
out_data = {}
out_data['mc_iter']       = mc_iter
out_data['meas']          = meas
out_data['val']           = val
out_data['meas_pred']     = meas_pred
out_data['val_pred']      = val_pred
out_data['mean_y']        = params['mean_y']
out_data['std_y']         = params['std_y']
out_data['mean_u']        = params['mean_u']
out_data['std_u']         = params['std_u']
out_data['plant_mean_dcdt']     = params['mean_dcdt']          
out_data['plant_std_dcdt']      = params['std_dcdt']          
out_data['mean_dcdt'] = nvec[2][:, 0]
out_data['std_dcdt']  = 10.0 ** nvec[2][:, 1]
out_data['training_time'] = training_time
out_data['rmse_meas']     = rmse_meas
out_data['rmse_val']      = rmse_val
out_data['loss']          = loss_container
out_data['loss_v']        = lossv_container
out_data['nvec']          = nvec
out_data['best_nvec']     = best_save_nvec
out_data['rhs']           = rhs
out_data['hess_nvec']     = hess_nvec
out_data['sigma2']        = sigma2
out_data['R_mat']         = R_mat
out_data['rxn_rates']     = rxn_rates
out_data['nu']            = nu

PickleTool.save(out_data, f'pickle/{log_file_base}_{case_type}_{mc_iter}_{noise_tag}.pickle')

print(f'\n Training time (hrs): {training_time / 60}', flush=True)
print(f'\n RMSE meas: {rmse_meas}', flush=True)
print(f'\n RMSE val:  {rmse_val}', flush=True)

sys.stdout = sys.__stdout__
log_handle.close()
