import flax.linen as nn
import jax, interpax, optax, diffrax
import jax.numpy as jnp
import numpy as np
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, ImplicitEuler, Tsit5, Kvaerno3
from functools import partial
from scipy.linalg import null_space
jax.config.update("jax_enable_x64", True)
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

SOLVER = Kvaerno3()
MAX_STEPS = int(1e6)
STEPSIZE_CONTROLLER = PIDController(rtol=1e-5, atol=1e-5)
# dummy optimizer : redefine in main script
optimizer = optax.adamw(learning_rate = 0.5e-2)

# null space of mass conservation
L = jnp.array([28.0, 30.0, 60.0, 18.0, 86.0, 32.0, 44.0])
L = L.reshape(-1,1).T
Z = null_space(L)

# This py file is meant to be a library of functions to be used for 
# different structuring problems. Since JAX uses functional programming,
# we define all variations of function here and call them in main script.

# Interpolator to get the control inputs
def get_controls(t, params):
    extrap = params['extrap']
    time = params['time']
    nEe = params['NEe']
    nA = params['NA']
    nO = params['NO']
    nW = params['NW']
    nV = params['NV']
    nC = params['NC']
    alpha = params['alpha']
    beta = params['beta']
    T = params['T']
    nt = params['nt']
    method = 'nearest'
    nEei = interpax.interp1d(t, time, nEe, method=method, extrap=extrap)
    nAi = interpax.interp1d(t, time, nA, method=method, extrap=extrap)
    nOi = interpax.interp1d(t, time, nO, method=method, extrap=extrap)
    nWi = interpax.interp1d(t, time, nW, method=method, extrap=extrap)
    nVi = interpax.interp1d(t, time, nV, method=method, extrap=extrap)
    nCi = interpax.interp1d(t, time, nC, method=method, extrap=extrap)
    alphai = interpax.interp1d(t, time, alpha, method=method, extrap=extrap)
    betai = interpax.interp1d(t, time, beta, method=method, extrap=extrap)
    Ti = interpax.interp1d(t, time, T, method=method, extrap=extrap)
    return nEei/nt, nAi/nt, nOi/nt, nWi/nt, nVi/nt, nCi/nt, alphai, betai, Ti

# true rate laws
def rate_true(cee, cea, ca, cw, cv, co, cc, R, T, params, nvec):
    operands = (cee, cea, ca, cw, cv, co, cc, R, T)

    def current_branch(op):
        cee, cea, ca, cw, cv, co, cc, R, T = op
        r1 = 3.73 * 0.7e-7 * jnp.exp(-2.7 * 15e3 / R * (1.0 / T - 1.0 / 450.0)) * cee**1.1 * co**1.1 * ca
        r2 = 3.73 * 5.1e-7 * jnp.exp(-2.5 * 21e3 / R * (1.0 / T - 1.0 / 450.0)) * cee * co
        return r1, r2

    def aiche_branch(op):
        cee, cea, ca, cw, cv, co, cc, R, T = op
        pee = R * T * cee / 1000.0
        pa  = R * T * ca  / 1000.0
        po  = R * T * co  / 1000.0
        r1 = 0.7e-4 * jnp.exp(-15e3 / R / T) * pee**0.35 * po**0.20 * pa
        r2 = 5.1e-3 * jnp.exp(-21e3 / R / T) * (jnp.exp(-0.004 * pee) - jnp.exp(-10.0 * pee)) * po**0.85
        return r1, r2

    r1, r2 = jax.lax.switch(params["case"], [current_branch, aiche_branch], operands)

    ree = -r1 - r2
    rea = 0 * r1 + 0 * r2
    ra  = -r1
    rw  = r1 + 2.0 * r2
    rv  = r1
    ro  = -0.5 * r1 - 3.0 * r2
    rc  = 2.0 * r2
    return ree, rea, ra, rw, rv, ro, rc

class r1(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64, param_dtype=jnp.float64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1, param_dtype=jnp.float64)(x)
        return x

class r2(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64, param_dtype=jnp.float64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1, param_dtype=jnp.float64)(x)
        return x
    

rate_1 = r1()
rate_2 = r2()
dum_input_1 = jnp.ones(4)
dum_input_2 = jnp.ones(3)

nvec1 = rate_1.init(jax.random.PRNGKey(123), dum_input_1)
nvec2 = rate_2.init(jax.random.PRNGKey(10), dum_input_2)
nvec3 = jnp.ones((2,2)) * -3.2
nvec = (nvec1, nvec2, nvec3)


# black boxed rate law
def rate_nn(cee, cea, ca, cw, cv, co, cc, R, T, params, nvec):
    mean = params['mean']
    std = params['std']
    nvec1 = nvec[0]
    nvec2 = nvec[1]
    nvec3 = nvec[2]
    pee = (cee - mean.at[0].get()) / std.at[0].get()
    pea = (cea - mean.at[1].get()) / std.at[1].get()
    pa = (ca - mean.at[2].get()) / std.at[2].get()
    pw = (cw - mean.at[3].get()) / std.at[3].get()
    pv = (cv - mean.at[4].get()) / std.at[4].get()
    po = (co - mean.at[5].get()) / std.at[5].get()
    pc = (cc - mean.at[6].get()) / std.at[6].get()
    temp = (T - mean.at[7].get()) / std.at[7].get()
    x1 = jnp.array([pee, pa, po, temp])
    x2 = jnp.array([pee, po, temp])
    r1 = rate_1.apply(nvec1, x1).squeeze()
    r2 = rate_2.apply(nvec2, x2).squeeze()
    r = jnp.array([r1, r2])
    #jax.debug.print('max {}', jnp.max(x))
    #jax.debug.print('min {}', jnp.min(x))
    vT = jnp.array([[-1, -1],
                [0, 0],
                [-1, 0], 
                [1, 2],
                [1, 0],
                [-0.5, -3],
                [0, 2]], dtype=jnp.float64)
    mean_r = nvec3[:, 0]
    std_r = nvec3[:, 1]
    #check = rate_1.apply(nvec1, x).squeeze()
    #r = 0.5e-2 * (rate_1.apply(nvec1, x).squeeze())
    r = (10 ** std_r) * r + (10 ** mean_r)
    r = vT @ r
    #r = r.at[1].set(0.0)  # set inert rate to zero
    #jax.debug.print('check {}', check)
    #r = r - (W_inv @ mol) * (jnp.dot(r, mol) / jnp.dot(mol, W_inv @mol))
    # get only the constraint satisfying rates
    #dotprod = jnp.dot(r, mol)
    #CCTinv = jnp.linalg.inv(constraint.T @ constraint)
    #r = r - constraint @ (CCTinv @ constraint.T @ r)
    #deltar = constraint @ (CCTinv @ constraint.T @ r)
    #jax.debug.print('r {}', r)
    #jax.debug.print('deltar {}', deltar)
    ree = r[0]
    rea = r[1]
    ra = r[2]
    rw = r[3]
    rv = r[4]
    ro = r[5]
    rc = r[6]
    return ree, rea, ra, rw, rv, ro, rc


# bare NN reaction rates [r1, r2] as a function of the 7-dim concentration
# state c and temperature T (i.e. before the vT stoichiometry mapping in
# rate_nn that turns them into the 7 species rates). Used for the C matrix.
def reaction_rate_nn(c, T, params, nvec):
    mean = params['mean']
    std = params['std']
    nvec1, nvec2, nvec3 = nvec
    cee, cea, ca, cw, cv, co, cc = c
    pee = (cee - mean.at[0].get()) / std.at[0].get()
    pa = (ca - mean.at[2].get()) / std.at[2].get()
    po = (co - mean.at[5].get()) / std.at[5].get()
    temp = (T - mean.at[7].get()) / std.at[7].get()
    x1 = jnp.array([pee, pa, po, temp])
    x2 = jnp.array([pee, po, temp])
    r1 = rate_1.apply(nvec1, x1).squeeze()
    r2 = rate_2.apply(nvec2, x2).squeeze()
    r = jnp.array([r1, r2])
    mean_r = nvec3[:, 0]
    std_r = nvec3[:, 1]
    r = (10 ** std_r) * r + (10 ** mean_r)
    return r


# Outer-product of the rate Jacobian averaged over a trajectory:
#     C = E_{c ~ rho}[ (dr/dc)^T (dr/dc) ]  in R^{7 x 7}
# where r = [r1, r2] are the two NN reaction rates (reaction_rate_nn) and
# c is the 7-dim concentration state. At each trajectory point dr/dc is the
# (2, 7) Jacobian; C averages J^T J over all points (rho = empirical
# distribution of states along the trajectory).
@jax.jit
def c_matrix(nvec, meas, temp, params):
    def jac_single(c, T):
        return jax.jacrev(reaction_rate_nn, argnums=0)(c, T, params, nvec)  # (2, 7)
    J = jax.vmap(jac_single)(meas, temp)                 # (N, 2, 7)
    C = jnp.mean(jnp.einsum('nij,nik->njk', J, J), axis=0)  # (7, 7)
    return C


# process model
@jax.jit
def ode(t, y, args):
    params, nvec, = args
    Nfee, Nfa, Nfo, Nfw, Nfv, Nfc, alpha, beta, T = get_controls(t, params)
    P = params['P']
    R = params['R']
    Vr = params['Vr']
    e = params['e']
    pcat = params['pcat']
    mu = params['mu']
    At = params['At']
    l = params['l']
    constQ = params['constQ']
    cee = y.at[0].get()
    cea = y.at[1].get()
    ca = y.at[2].get()
    cw = y.at[3].get()
    cv = y.at[4].get()
    co = y.at[5].get()
    cc = y.at[6].get()
    

    recycled = alpha * (cee + cea + co + (1 - mu) * cc) + beta * ca

    NFee = Nfee
    NFea = 0.001 * NFee
    NFa = Nfa
    NFo = Nfo
    NFw = Nfw
    NFv = Nfv
    NFc = Nfc

    cs = (cee, cea, ca, cw, cv, co, cc, R, T, params, nvec)

    r = jax.lax.cond(
        params['rate_select'][0] == 0.0, 
        lambda cs: rate_true(*cs),  
        lambda cs: rate_nn(*cs), cs)

    ree = r[0]
    rea = r[1]
    ra = r[2]
    rw = r[3]
    rv = r[4]
    ro = r[5]
    rc = r[6]

    sum_Rj = (ree + rea + ra + ro + rw + rv + rc)

    N1_sum = (NFee + NFea + NFa + NFo + NFw + NFv + NFc)
    num = N1_sum + constQ * pcat * R * (At * l) * recycled * (T * sum_Rj / P)
    denom = P/R/T - recycled
    Qf = num / denom
    #Qf = ((NFee + NFea + NFa + NFo + NFw + NFv + NFc) / (P / (R * T) - recycled))
    Q = Qf + pcat * R * (At * l) * (T * sum_Rj / P)

    N2ee = NFee + Q * alpha * cee
    N2ea = NFea + Q * alpha * cea
    N2a = NFa + Q * beta * ca
    N2o = NFo + Q * alpha * co
    N2w = NFw
    N2v = NFv
    N2c = NFc + Q * alpha * (1 - mu) * cc

    c2ee = N2ee / Qf
    c2ea = N2ea / Qf
    c2a = N2a / Qf
    c2o = N2o / Qf
    c2w = N2w / Qf
    c2v = N2v / Qf
    c2c = N2c / Qf


    dcee = (1 / Vr * (Qf * c2ee - Q * cee) + pcat * ree) / e
    dcea = (1 / Vr * (Qf * c2ea - Q * cea) + pcat * rea) / e
    dca = (1 / Vr * (Qf * c2a - Q * ca) + pcat * ra) / e
    dcw = (1 / Vr * (Qf * c2w - Q * cw) + pcat * rw) / e
    dcv = (1 / Vr * (Qf * c2v - Q * cv) + pcat * rv) / e
    dco = (1 / Vr * (Qf * c2o - Q * co) + pcat * ro) / e
    dcc = (1 / Vr * (Qf * c2c - Q * cc) + pcat * rc) / e

    return jnp.array([dcee, dcea, dca, dcw, dcv, dco, dcc])


@jax.jit
def get_Q(t, y, args):
    params, nvec, = args
    Nfee, Nfa, Nfo, Nfw, Nfv, Nfc, alpha, beta, T = get_controls(t, params)
    P = params['P']
    R = params['R']
    Vr = params['Vr']
    e = params['e']
    pcat = params['pcat']
    mu = params['mu']
    At = params['At']
    l = params['l']
    constQ = params['constQ']
    cee = y.at[0].get()
    cea = y.at[1].get()
    ca = y.at[2].get()
    cw = y.at[3].get()
    cv = y.at[4].get()
    co = y.at[5].get()
    cc = y.at[6].get()
    

    recycled = alpha * (cee + cea + co + (1 - mu) * cc) + beta * ca

    NFee = Nfee
    NFea = 0.001 * NFee
    NFa = Nfa
    NFo = Nfo
    NFw = Nfw
    NFv = Nfv
    NFc = Nfc

    cs = (cee, cea, ca, cw, cv, co, cc, R, T, params, nvec)

    r = jax.lax.cond(
        params['rate_select'][0] == 0.0, 
        lambda cs: rate_true(*cs),  
        lambda cs: rate_nn(*cs), cs)

    ree = r[0]
    rea = r[1]
    ra = r[2]
    rw = r[3]
    rv = r[4]
    ro = r[5]
    rc = r[6]

    sum_Rj = (ree + rea + ra + ro + rw + rv + rc)

    N1_sum = (NFee + NFea + NFa + NFo + NFw + NFv + NFc)
    num = N1_sum + constQ * pcat * R * (At * l) * recycled * (T * sum_Rj / P)
    denom = P/R/T - recycled
    Qf = num / denom
    #Qf = ((NFee + NFea + NFa + NFo + NFw + NFv + NFc) / (P / (R * T) - recycled))
    Q = Qf + pcat * R * (At * l) * (T * sum_Rj / P)

    return Q

# ode solver 
@jax.jit
def solve_ode(nvec, y0, tplot, params):
    dt0 = params['dt0']
    # NOTE: Take care to use log-transformed initial conditions
    #y0_log = jnp.log(y0)
    t_start = tplot[0]
    t_end = tplot[-1]
    solver = SOLVER
    stepsize_controller = STEPSIZE_CONTROLLER
    saveat = SaveAt(ts=tplot)

    sol = diffeqsolve(ODETerm(ode), solver=solver, t0=t_start, t1=t_end, y0=y0, dt0=dt0,
                    saveat=saveat, stepsize_controller=stepsize_controller,
                    max_steps=MAX_STEPS, args=(params, nvec))
    return sol


@jax.jit
def solve_ode_hess(nvec, y0, tplot, params):
    dt0 = params['dt0']
    t_start = tplot[0]
    t_end = tplot[-1]
    saveat = SaveAt(ts=tplot)
    sol = diffeqsolve(ODETerm(ode), solver=Tsit5(), t0=t_start, t1=t_end, y0=y0, dt0=dt0,
                    saveat=saveat, stepsize_controller=STEPSIZE_CONTROLLER,
                    max_steps=MAX_STEPS, args=(params, nvec), adjoint=diffrax.DirectAdjoint())
    return sol


# solves ode for a batch of initial conditions
@jax.jit
def batch_solve(y0_mat, t_batch, nvec, params):
    def solve_single_ode(y0, t_plot):
        return solve_ode(nvec, y0, t_plot, params)
    return jax.vmap(solve_single_ode)(y0_mat, t_batch)



# randomly solve a batch of initial conditions
@jax.jit
def random_batch_solve(y0_mat, t_batch, meas_batch, nvec, params, bind):
    y0_mat_sel = y0_mat[bind]
    t_batch_sel = t_batch[bind]
    meas_batch_sel = meas_batch[bind, :, :]
    measure = jnp.reshape(meas_batch_sel, (-1, meas_batch_sel.shape[-1]))
    sol_batch = batch_solve(y0_mat_sel, t_batch_sel, nvec, params)
    sol_shaped = jnp.reshape(sol_batch.ys, (-1, sol_batch.ys.shape[-1]))
    tmeas = jnp.reshape(t_batch_sel, (-1,))
    return sol_shaped, measure, tmeas


# loss function 
@jax.jit
def loss_fn(nvec, y0_mat, t_batch, meas_batch, params, bind):
    sol_shaped, measure, tmeas = random_batch_solve(y0_mat, t_batch, meas_batch, nvec,
                                              params, bind)
    meas_norm = (measure - params['mean'][:7]) / params['std'][:7]
    sol_shaped_norm = (sol_shaped - params['mean'][:7]) / params['std'][:7]
    error = (meas_norm - sol_shaped_norm) * params['weights']
    Q_pred_sel = jax.vmap(get_Q, in_axes=(0,0,None))(tmeas, sol_shaped, (params, nvec))
    Q_pred = Q_pred_sel * params['nt']
    Q_meas_batch = params['Qmeas_batch'][bind, :]
    Q_meas = jnp.reshape(Q_meas_batch, (-1,)) * params['nt']
    error_Q = (Q_meas - Q_pred) / (Q_meas.std()+1e-5)
    weight_q = params['weight_q']
    loss = jnp.mean(jnp.square(error)) + weight_q * jnp.mean(jnp.square(error_Q))
    return loss




# training step
@jax.jit
def training_step(nvec, y0_mat, t_batch, meas_batch, opt_state, params, bind):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(nvec, y0_mat, t_batch, meas_batch, params, bind)
    g1, g2, g3 = grad
    clip = optax.clip_by_global_norm(1.0)
    (clipped, _) = clip.update((g1, g2), clip.init((g1, g2)))
    g1c, g2c = clipped
    grad = (g1c, g2c, g3)
    updates, opt_state = optimizer.update(grad, opt_state, nvec)
    nvec = optax.apply_updates(nvec, updates)
    return nvec, opt_state, loss


# LBFGS training step (full-batch quasi-Newton). Additive: leaves training_step
# and the `optimizer` global untouched so existing callers are unaffected. The
# zoom line search sets the step size, so no gradient clipping is applied here.
lbfgs_optimizer = optax.lbfgs()


@jax.jit
def lbfgs_training_step(nvec, y0_mat, t_batch, meas_batch, opt_state, params, bind):
    def value_fn(nv):
        return loss_fn(nv, y0_mat, t_batch, meas_batch, params, bind)
    value_and_grad = optax.value_and_grad_from_state(value_fn)
    loss, grad = value_and_grad(nvec, state=opt_state)
    updates, opt_state = lbfgs_optimizer.update(
        grad, opt_state, nvec, value=loss, grad=grad, value_fn=value_fn)
    nvec = optax.apply_updates(nvec, updates)
    return nvec, opt_state, loss


def straight_loss_fn(nvec, meas, meas_c0, meas_t, params):
    meas_pred = solve_ode_hess(nvec, meas_c0, meas_t, params).ys
    meas_pred_norm = (meas_pred - params['mean'][:7]) / params['std'][:7]
    meas_norm = (meas - params['mean'][:7]) / params['std'][:7]
    error = (meas_norm - meas_pred_norm) * params['weights']
    return error.ravel()


def hessian_fn(nvec, meas, meas_c0, meas_t, params):
    nvec_flat, nvec_unravel = jax.flatten_util.ravel_pytree(nvec)
    def _loss_flat(nvec_flat):
        return straight_loss_fn(nvec_unravel(nvec_flat), meas, meas_c0, meas_t, params)
    J = jax.jacfwd(_loss_flat)(nvec_flat)
    err = straight_loss_fn(nvec, meas, meas_c0, meas_t, params)
    sigma2 = jnp.mean(err**2)
    return 2 * J.T @ J, sigma2


@jax.jit
def save_rate(nvec, meas, temp, params):
    # Unpack arguments
    R = params['R']
    T = temp
    def compute_rates(cee, cea, ca, cw, cv, co, cc, T):
        rtee, rtea, rta, rtw, rtv, rto, rtc = rate_true(cee, cea, ca, cw, cv, co, cc, R, T, params, nvec)
        rnee, rnea, rna, rnw, rnv, rno, rnc = rate_nn(cee, cea, ca, cw, cv, co, cc, R, T, params, nvec)
        return rtee, rtea, rta, rtw, rtv, rto, rtc, rnee, rnea, rna, rnw, rnv, rno, rnc
    compute_rates_vmap = jax.vmap(compute_rates)
    cee = meas[:, 0]
    cea = meas[:, 1]
    ca = meas[:, 2]
    cw = meas[:, 3]
    cv = meas[:, 4]
    co = meas[:, 5]
    cc = meas[:, 6]
    rtee, rtea, rta, rtw, rtv, rto, rtc, rnee, rnea, rna, rnw, rnv, rno, rnc = compute_rates_vmap(cee, cea, ca, cw, cv, co, cc, T)
    return rtee, rtea, rta, rtw, rtv, rto, rtc, rnee, rnea, rna, rnw, rnv, rno, rnc



def save_Q(meas_t, meas, params, nvec):
    Q_vmap = jax.vmap(get_Q, in_axes=(0,0,None))(meas_t, meas, (params, nvec))
    Q_saved = Q_vmap * params['nt']
    return Q_saved


@jax.jit
def save_rxn(nvec, meas, temp, params):
    # Unpack arguments
    R = params['R']
    T = temp
    def compute_rates(cee, cea, ca, cw, cv, co, cc, T):
        rtee, rtea, rta, rtw, rtv, rto, rtc = rate_true(cee, cea, ca, cw, cv, co, cc, R, T, params, nvec)
        rnee, rnea, rna, rnw, rnv, rno, rnc = rate_nn(cee, cea, ca, cw, cv, co, cc, R, T, params, nvec)
        return rtee, rtea, rta, rtw, rtv, rto, rtc, rnee, rnea, rna, rnw, rnv, rno, rnc
    compute_rates_vmap = jax.vmap(compute_rates)
    cee = meas[:, 0]
    cea = meas[:, 1]
    ca = meas[:, 2]
    cw = meas[:, 3]
    cv = meas[:, 4]
    co = meas[:, 5]
    cc = meas[:, 6]
    rtee, rtea, rta, rtw, rtv, rto, rtc, rnee, rnea, rna, rnw, rnv, rno, rnc = compute_rates_vmap(cee, cea, ca, cw, cv, co, cc, T)
    rt1, rt2, rn1, rn2 = rtv, rtc / 2, rnv, rnc / 2
    return rt1, rt2, rn1, rn2


####### data processing and saving ########

# We next define dictionaries that record the functions we need to call for
# different problems.

learning_dict = {}
learning_dict['solve_ode'] = solve_ode
learning_dict['training_step'] = training_step
learning_dict['lbfgs_training_step'] = lbfgs_training_step
learning_dict['lbfgs_optimizer'] = lbfgs_optimizer
learning_dict['nvec'] = nvec
learning_dict['rate_true'] = rate_true
learning_dict['rate_nn'] = rate_nn
learning_dict['batch_solve'] = batch_solve
learning_dict['save_rate'] = save_rate
learning_dict['save_rxn'] = save_rxn
learning_dict['get_Q'] = get_Q
learning_dict['save_Q'] = save_Q
learning_dict['hessian_fn'] = hessian_fn
learning_dict['reaction_rate_nn'] = reaction_rate_nn
learning_dict['c_matrix'] = c_matrix
