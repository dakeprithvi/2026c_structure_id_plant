import flax.linen as nn
import jax, interpax, optax, diffrax
import jax.flatten_util          # ravel_pytree for hessian_fn_bbox
import jax.numpy as jnp
import numpy as np
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Kvaerno3
from functools import partial
jax.config.update("jax_enable_x64", True)
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

SOLVER = Kvaerno3()
MAX_STEPS = int(1e6)
STEPSIZE_CONTROLLER = PIDController(rtol=1e-5, atol=1e-5)
# dummy optimizer : redefine in main script
optimizer = optax.adamw(learning_rate=0.5e-2)

# This py file is meant to be a library of functions for the complete black-box
# (bbox) formulation. Both the RHS (7 outputs) and Q (1 output) are predicted
# by neural networks. The ODE is solved in normalized state space:
#   y_norm = (c - mean_y) / std_y
# All functions that accept y expect it in this normalized form unless the
# name starts with "save_" (those accept raw concentrations and normalize internally).

# -------- interpolator --------

def get_controls(t, params):
    extrap = params['extrap']
    time   = params['time']
    nEe    = params['NEe']
    nA     = params['NA']
    nO     = params['NO']
    nW     = params['NW']
    nV     = params['NV']
    nC     = params['NC']
    alpha  = params['alpha']
    beta   = params['beta']
    T      = params['T']
    nt     = params['nt']
    method = 'nearest'
    nEei  = interpax.interp1d(t, time, nEe,   method=method, extrap=extrap)
    nAi   = interpax.interp1d(t, time, nA,    method=method, extrap=extrap)
    nOi   = interpax.interp1d(t, time, nO,    method=method, extrap=extrap)
    nWi   = interpax.interp1d(t, time, nW,    method=method, extrap=extrap)
    nVi   = interpax.interp1d(t, time, nV,    method=method, extrap=extrap)
    nCi   = interpax.interp1d(t, time, nC,    method=method, extrap=extrap)
    alphai = interpax.interp1d(t, time, alpha, method=method, extrap=extrap)
    betai  = interpax.interp1d(t, time, beta,  method=method, extrap=extrap)
    Ti     = interpax.interp1d(t, time, T,     method=method, extrap=extrap)
    return nEei/nt, nAi/nt, nOi/nt, nWi/nt, nVi/nt, nCi/nt, alphai, betai, Ti


def _get_u(t, params):
    """9-element control vector [Nfee,...,Nfc, alpha, beta, T] at time t."""
    nfee, nfa, nfo, nfw, nfv, nfc, alpha, beta, T = get_controls(t, params)
    nt = params['nt']
    return jnp.array([nfee*nt, nfa*nt, nfo*nt, nfw*nt, nfv*nt, nfc*nt,
                      alpha, beta, T])


# -------- neural network modules --------

class bbox(nn.Module):
    # Emits 7 standardized dc/dt outputs (~zero-mean, unit-std); nn_bbox rescales
    # them with LEARNABLE dc/dt statistics (nvec[2], see below) and std_y.
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128, param_dtype=jnp.float64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=7, param_dtype=jnp.float64)(x)
        return x


class qbox(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64, param_dtype=jnp.float64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1, param_dtype=jnp.float64)(x)
        return x


cbox_nn = bbox()
qbox_nn = qbox()

dum_input = jnp.ones(16)
nvec1 = cbox_nn.init(jax.random.PRNGKey(0), dum_input)
nvec2 = qbox_nn.init(jax.random.PRNGKey(1), dum_input)
nvec3 = jnp.stack([jnp.full((7,), 0.0,    dtype=jnp.float64),
                   jnp.full((7,), -1.0, dtype=jnp.float64)], axis=1)  # (7, 2)
nvec  = (nvec1, nvec2, nvec3)


def nn_bbox(y_norm, u, nvec, params):
    u_norm = (u - params['mean_u']) / params['std_u']
    bigy   = jnp.concatenate([y_norm, u_norm], axis=-1)
    z      = cbox_nn.apply(nvec[0], bigy).squeeze()      # (7,) standardized
    mean_dcdt, log_std_dcdt = nvec[2][:, 0], nvec[2][:, 1]     # learnable stats
    dc_dt  = (10.0 ** log_std_dcdt) * z + mean_dcdt            # (7,) physical dc/dt
    return dc_dt / params['std_y']                       # (7,) dy_norm/dt


def nn_qbox(y_norm, u, nvec, params):
    """Q black box. y_norm must already be in normalized state space."""
    u_norm = (u - params['mean_u']) / params['std_u']
    bigy   = jnp.concatenate([y_norm, u_norm], axis=-1)
    return qbox_nn.apply(nvec[1], bigy).squeeze()


# -------- ODE and Q --------
# ode returns dy_norm/dt = nn_bbox directly (nn_bbox rescales the standardized NN
# output to physical dc/dt, then divides by std_y); get_prod_rates multiplies by
# std_y to recover dc/dt.

@jax.jit
def ode(t, y, args):
    nvec, params = args
    u = _get_u(t, params)
    return nn_bbox(y, u, nvec, params) 


@jax.jit
def get_Q(t, y, args):
    """y must be in normalized state space (same convention as ode)."""
    nvec, params = args
    u = _get_u(t, params)
    return nn_qbox(y, u, nvec, params) 


# -------- ODE solvers --------

@jax.jit
def solve_ode(nvec, y0_norm, t, params):
    dt0    = params['dt0']
    saveat = SaveAt(ts=t)
    sol = diffeqsolve(ODETerm(ode), solver=SOLVER, t0=t[0], t1=t[-1],
                      y0=y0_norm, dt0=dt0, saveat=saveat,
                      stepsize_controller=STEPSIZE_CONTROLLER,
                      max_steps=MAX_STEPS, args=(nvec, params))
    return sol


def solve_ode_hess(nvec, y0_norm, t, params):
    """solve_ode with DirectAdjoint for forward-mode Jacobians (hessian analysis)."""
    dt0    = params['dt0']
    saveat = SaveAt(ts=t)
    sol = diffeqsolve(ODETerm(ode), solver=SOLVER, t0=t[0], t1=t[-1],
                      y0=y0_norm, dt0=dt0, saveat=saveat,
                      stepsize_controller=STEPSIZE_CONTROLLER,
                      max_steps=MAX_STEPS, args=(nvec, params),
                      adjoint=diffrax.DirectAdjoint())
    return sol


# -------- batched solve --------

@jax.jit
def batch_solve(nvec, params, t_batch, y0_mat, meas_batch):
    y0_mat_norm = (y0_mat - params['mean_y']) / params['std_y']
    y_batch     = jax.vmap(solve_ode, in_axes=(None, 0, 0, None))(
                      nvec, y0_mat_norm, t_batch, params)

    measurement = jnp.reshape(meas_batch, (-1, meas_batch.shape[-1]))
    t_sol       = jnp.reshape(t_batch, (-1,))
    y_sol_norm  = jnp.reshape(y_batch.ys, (-1, y_batch.ys.shape[-1]))
    y_sol       = y_sol_norm * params['std_y'] + params['mean_y']

    q_pred_batch = jax.vmap(get_Q, in_axes=(0, 0, None))(
                       t_sol, y_sol_norm, (nvec, params))
    q_pred = jnp.reshape(q_pred_batch, (-1,))
    return y_sol, measurement, q_pred


# -------- loss and training --------

@jax.jit
def loss_fn_bbox(nvec, params, t_batch, y0_mat, meas_batch, bind):
    y0_mat_sel    = y0_mat[bind]
    meas_batch_sel = meas_batch[bind]
    t_batch_sel   = t_batch[bind]
    y_sol, meas_sol, Q_sol = batch_solve(
        nvec, params, t_batch_sel, y0_mat_sel, meas_batch_sel)
    y_sol_norm    = (y_sol   - params['mean_y']) / params['std_y']
    meas_sol_norm = (meas_sol - params['mean_y']) / params['std_y']
    Q_meas_batch  = params['Qmeas_batch'][bind, :]
    Q_meas        = jnp.reshape(Q_meas_batch, (-1,)) * params['nt']
    error_Q = (Q_meas - Q_sol) / (Q_meas.std() + 1e-5)
    err     = (meas_sol_norm - y_sol_norm) * params['weights']
    return jnp.mean(jnp.square(err)) + params['learn_Q'] * jnp.mean(jnp.square(error_Q))


@jax.jit
def training_step_cbox(nvec, params, t_batch, y0_mat, meas_batch, opt_state, bind):
    loss, grad = jax.value_and_grad(loss_fn_bbox)(nvec, params, t_batch, y0_mat, meas_batch, bind)
    g1, g2, g3 = grad
    clip = optax.clip_by_global_norm(1.0)
    (clipped, _) = clip.update((g1, g2), clip.init((g1, g2)))
    g1c, g2c = clipped
    grad = (g1c, g2c, g3)
    updates, opt_state = optimizer.update(grad, opt_state, nvec)
    nvec = optax.apply_updates(nvec, updates)
    return nvec, opt_state, loss


# LBFGS phase: same loss_fn_bbox, full-batch (pass bind = all batch indices).
# nvec is float64 (param_dtype=float64 at init), matching the float64 grads the
# zoom line search requires.
lbfgs_optimizer = optax.lbfgs()


@jax.jit
def lbfgs_training_step_cbox(nvec, params, t_batch, y0_mat, meas_batch, opt_state, bind):
    def value_fn(nv):
        return loss_fn_bbox(nv, params, t_batch, y0_mat, meas_batch, bind)
    value_and_grad = optax.value_and_grad_from_state(value_fn)
    loss, grad = value_and_grad(nvec, state=opt_state)
    updates, opt_state = lbfgs_optimizer.update(
        grad, opt_state, nvec, value=loss, grad=grad, value_fn=value_fn)
    nvec = optax.apply_updates(nvec, updates)
    return nvec, opt_state, loss


# -------- Q-only training --------

@jax.jit
def save_q_nn_output(t, y, args):
    # Evaluate Q for a raw (un-normalized) trajectory.
    nvec, params = args
    def single_eval(ti, yi):
        yi_norm = (yi - params['mean_y']) / params['std_y']
        return get_Q(ti, yi_norm, (nvec, params))
    return jax.vmap(single_eval)(t, y)


@jax.jit
def loss_qbox(nvec, params, t, y, qmeas):
    qpred   = save_q_nn_output(t, y, (nvec, params))
    error_Q = (qmeas - qpred) / (qmeas.std() + 1e-5)
    return jnp.mean(jnp.square(error_Q))


@jax.jit
def train_q(nvec, params, t, y, qmeas, opt_state):
    loss, grads    = jax.value_and_grad(loss_qbox)(nvec, params, t, y, qmeas)
    updates, opt_state = optimizer.update(grads, opt_state, nvec)
    nvec = optax.apply_updates(nvec, updates)
    return nvec, opt_state, loss


# -------- diagnostic / save functions --------

@jax.jit
def save_nn_output(nvec, params, t, y):
    """Evaluate bbox RHS for a raw (un-normalized) trajectory."""
    def single_eval(ti, yi):
        u      = _get_u(ti, params)
        yi_norm = (yi - params['mean_y']) / params['std_y']
        return nn_bbox(yi_norm, u, nvec, params)
    return jax.vmap(single_eval)(t, y)


@jax.jit
def nn_output_stats(nvec, params, meas_raw, meas_t):
    """Mean and std of the physical NN RHS outputs across a trajectory (monitoring).
    Goes through nn_bbox so the stats are over the 7 EOS-consistent dc/dt
    components, not the 6 latent outputs."""
    def _raw(yi, ti):
        u      = _get_u(ti, params)
        yi_norm = (yi - params['mean_y']) / params['std_y']
        return nn_bbox(yi_norm, u, nvec, params)
    rhs_all = jax.vmap(_raw)(meas_raw, meas_t)   # (N, 7)
    return jnp.mean(rhs_all, axis=0), jnp.std(rhs_all, axis=0)


# -------- hessian / uncertainty analysis --------

def straight_loss_fn_bbox(nvec, meas, meas_c0_raw, meas_t, params):
    """Residual vector for a single trajectory (raw initial condition).
    Used as the building block for the Gauss-Newton Hessian."""
    c0_norm  = (meas_c0_raw - params['mean_y']) / params['std_y']
    y_norm   = solve_ode_hess(nvec, c0_norm, meas_t, params).ys
    meas_norm = (meas - params['mean_y']) / params['std_y']
    error    = (meas_norm - y_norm) * params['weights']
    return error.ravel()


def hessian_fn_bbox(nvec, meas, meas_c0_raw, meas_t, params):
    """Gauss-Newton Hessian H ≈ 2 J^T J and residual variance sigma2.

    Returns
    -------
    H      : (n_params, n_params) array — Gauss-Newton Hessian approximation
    sigma2 : scalar — mean squared residual (noise variance estimate)

    Covariance of parameters: cov ≈ sigma2 * inv(H / 2)
    """
    nvec_flat, nvec_unravel = jax.flatten_util.ravel_pytree(nvec)
    def _loss_flat(nv):
        return straight_loss_fn_bbox(nvec_unravel(nv), meas, meas_c0_raw, meas_t, params)
    J      = jax.jacfwd(_loss_flat)(nvec_flat)
    err    = straight_loss_fn_bbox(nvec, meas, meas_c0_raw, meas_t, params)
    sigma2 = jnp.mean(err ** 2)
    return 2 * J.T @ J, sigma2

# -------- production-rate extraction --------
# The bbox gives dc/dt in normalized space. Denormalize and invert the CSTR
# balance to recover the 7 production rates.
#
# The EOS constraint sum_j c_j = P/(RT) makes the linear equation for
# S = sum_j R_j degenerate (coefficient identically zero). Instead, Q is
# determined from mass conservation sum_j M_j R_j = 0, which gives a direct
# closed-form formula — no root-finding, no qbox needed.
#
# Params dict must contain: Vr, e, pcat, mu (plus normalization and control keys).

@jax.jit
def get_prod_rates_by_root_finding(t, y_raw, nvec, params):
    """Extract (ree, rea, ra, rw, rv, ro, rc) at a single time point.

    Q is determined directly from mass conservation sum_j M_j R_j = 0 using
    only the bbox output — no qbox, no root-finding.
    Returns a length-7 array ordered [ee, ea, a, w, v, o, c].
    """
    Vr = params['Vr'];  e = params['e'];  pcat = params['pcat'];  mu = params['mu']

    Nfee, Nfa, Nfo, Nfw, Nfv, Nfc, alpha, beta, _ = get_controls(t, params)
    NFee = Nfee;  NFea = 0.001 * NFee
    NFa  = Nfa;   NFo = Nfo;  NFw = Nfw;  NFv = Nfv;  NFc = Nfc

    N1 = jnp.array([NFee, NFea, NFa, NFw, NFv, NFo, NFc])

    # (1 - zeta_j) for each component [ee, ea, a, w, v, o, c]
    one_minus_zeta = jnp.array([
        1.0 - alpha,
        1.0 - alpha,
        1.0 - beta,
        1.0,
        1.0,
        1.0 - alpha,
        1.0 - alpha * (1.0 - mu),
    ])

    # denormalized dc/dt from bbox -- must match what solve_ode integrates:
    # ode returns dy_norm/dt = nn_bbox (no extra scaling), so dc/dt = std_y * nn_bbox.
    y_norm = (y_raw - params['mean_y']) / params['std_y']
    u      = _get_u(t, params)
    dc_dt  = nn_bbox(y_norm, u, nvec, params) * params['std_y']

    # molecular weights [g/mol]: [Ee, Ea, A, W, V, O, C]
    M = jnp.array([28., 30., 60., 18., 86., 32., 44.])

    # Q from mass conservation: sum_j M_j R_j = 0
    omzeta_c = one_minus_zeta * y_raw
    Q = (jnp.dot(M, N1) - e * Vr * jnp.dot(M, dc_dt)) / jnp.dot(M, omzeta_c)

    # R_j from CSTR inversion: epsilon*dc_j/dt = (N1_j - Q*(1-zeta_j)*c_j)/Vr + pcat*R_j
    return (e * dc_dt - (N1 - Q * omzeta_c) / Vr) / pcat


@jax.jit
def save_prod_rates(t_traj, y_traj_raw, nvec, params):
    """Evaluate production rates over a full trajectory (raw concentrations)."""
    return jax.vmap(
        lambda ti, yi: get_prod_rates_by_root_finding(ti, yi, nvec, params)
    )(t_traj, y_traj_raw)


# -------- learning dict --------

learning_dict = {}
learning_dict['batch_solve']           = batch_solve
learning_dict['solve_ode']             = solve_ode
learning_dict['solve_ode_hess']        = solve_ode_hess
learning_dict['ode']                   = ode
learning_dict['get_Q']                 = get_Q
learning_dict['training_step_cbox']    = training_step_cbox
learning_dict['lbfgs_training_step_cbox'] = lbfgs_training_step_cbox
learning_dict['lbfgs_optimizer']       = lbfgs_optimizer
learning_dict['nvec']                  = nvec
learning_dict['save_nn_output']        = save_nn_output
learning_dict['save_q_nn_output']      = save_q_nn_output
learning_dict['loss_qbox']             = loss_qbox
learning_dict['train_q']               = train_q
learning_dict['straight_loss_fn_bbox']        = straight_loss_fn_bbox
learning_dict['hessian_fn_bbox']              = hessian_fn_bbox
learning_dict['nn_output_stats']              = nn_output_stats
learning_dict['get_prod_rates_by_root_finding'] = get_prod_rates_by_root_finding
learning_dict['save_prod_rates']              = save_prod_rates
