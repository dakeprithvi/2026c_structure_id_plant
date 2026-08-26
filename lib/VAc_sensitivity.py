# Convenience class to analyze sensitivities

import numpy as np
import casadi
from VAc_flowsheet import Simulator

class SensitivityAnalyzer(Simulator):
    def __init__(self, config=None):
        super().__init__(config)
        if config is not None:
            config = {}

    def reduced_hessian(self, initial, bounds):
        # NOTE: equality constraints are always active. 
        # step 1 identify freezed u (freezed by me + freezed by bounds)
        # step 2 identify freezed x (freezed by bounds - leads to degeneracy - raise flag)
        # step 3 identify active constraint set = model + step 1 
        # step 4 identify the final x (includes freezed u + freezed by bounds) and u

        self.initialize()
        x = casadi.SX.sym('x', 7 + 9 * (self.n - 1))

        if self.param in self.rate_param_list:
            p = casadi.SX.sym('p', np.hstack([self.theta1, self.theta2]).shape[0])
            theta1 = p[np.arange(self.theta1.shape[0])]
            theta2 = p[np.arange(self.theta2.shape[0])+self.theta1.shape[0]]
            theta = (theta1, theta2)

        alpha = x[0]
        beta = x[1]
        T2 = x[2] * self.T20
        P2 = x[3] * self.P20
        N1f = casadi.SX(self.N1f)
        N1f[0] = x[4] * self.N1_clean[0]
        N1f[1] = x[4] * self.N1_clean[1]
        N1f[2] = x[5] * self.N1_clean[2]
        N1f[5] = x[6] * self.N1_clean[5]

        N10 = N1f / self.nt
        
        x_mat = casadi.reshape(x[7:], self.n - 1, 9)
        cEe, cEa, cA, cW, cV, cO, cC, T, P = casadi.vertsplit(x_mat.T)

        # Weights for the concentration variables
        w = self.weights
        cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]

        # Weights for temp and press are initial values themselves
        T = casadi.vertcat(T2, T.T * w[7])
        P = casadi.vertcat(P2, P.T * w[8])

        T_shaped = casadi.reshape(T[1:], -1, 1)
        P_shaped = casadi.reshape(P[1:], -1, 1)
        c = (cEe.T, cEa.T, cA.T, cW.T, cV.T, cO.T, cC.T, T_shaped)
        if self.param in self.rate_param_list:
            rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate_cas(c, alpha, beta, N1f, theta)
        else:
            rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate_cas(c, alpha, beta, N1f, None)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC

        # rec_factor calculates the fraction of all species that go back to the mixer
        rec_factor = (self.phi * (1 - self.mu) * alpha +
                   (1 - self.phi) * self.gamma * beta)
        
        T_inlet = T2 if self.rho_var else self.T_const_inlet
        rho_in = P2 / self.R / T_inlet

        t_factor = T[-1]*P[0]/ P[-1]/T[0] if self.const_pressure else 1
        t_fac_array = [1.0]
        for i in range(self.n - 2):
            t_fac_array.append(t_fac_array[i] * T[-1 - i] * P[-1 - (i  + 1)] / T[-1 - (i + 1)] / P[-1 - i])
        t_fac_array = casadi.vertcat(*t_fac_array)
        t_fac_array = t_fac_array[::-1] if self.const_pressure else np.ones(self.n - 1)

        # We calculate the conc and volumetric flow rate in last tank to avoid DAEs.
        C_in_last_cstr = casadi.vertcat(cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1])
        Qf = (casadi.sum1(N10) + self.constQ * self.pcat * self.R* (self.At * self.l / (self.n-1)) * casadi.sum1(rec_factor * C_in_last_cstr) * 
              casadi.sum1(T_shaped * sum_Rj * t_fac_array / P_shaped)) / (rho_in - t_factor * casadi.sum1(rec_factor * C_in_last_cstr)) 
        Q_in_last_cstr = Qf * t_factor + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * casadi.sum1(T_shaped / P_shaped * sum_Rj * t_fac_array)  
        # Construct the actual molar flow entering the reactor
        N2 = N10 + Q_in_last_cstr.T * C_in_last_cstr * rec_factor

        # Construct the initial concentrations 
        # (Note: This trick allows not to calculate the initial concentrations in the reactor which we don't know due to recycle)
        Q_for_first_cstr = casadi.sum1(N2) / rho_in
        cEe_0, cEa_0, cA_0, cW_0, cV_0, cO_0, cC_0 = casadi.vertsplit(N2 / Q_for_first_cstr, 1)

        # Now we have all concentrations in all the tanks
        cEe = casadi.vertcat(cEe_0, cEe.T)
        cEa = casadi.vertcat(cEa_0, cEa.T)
        cA = casadi.vertcat(cA_0, cA.T)
        cW = casadi.vertcat(cW_0, cW.T)
        cV = casadi.vertcat(cV_0, cV.T)
        cO = casadi.vertcat(cO_0, cO.T)
        cC = casadi.vertcat(cC_0, cC.T)

        cTotal = cEe + cEa + cA + cW + cV + cO + cC
        C = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC)

        c = (cEe, cEa, cA, cW, cV, cO, cC, T)
        if self.param in self.rate_param_list:
            rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate_cas(c, alpha, beta, N1f, theta)
        else:
            rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate_cas(c, alpha, beta, N1f, None)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC

        Mav = casadi.mtimes(self.mol.reshape(1,-1), C.T).T / cTotal
        rhoav = P * Mav / self.R / T * 10 ** -3

        Q = [Qf] 
        if self.const_pressure:
            for i in range(self.n - 1):
                Qi_next = Q[i] * T[i+1] * P[i]/T[i]/P[i+1] + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * T[i+1] / P[i+1] * sum_Rj[i+1]
                Q.append(Qi_next)
        else:
            for i in range(self.n - 1):
                Qi_next = Q[i] + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * T[i+1] / P[i+1] * sum_Rj[i+1] 
                Q.append(Qi_next)
        Q = casadi.vertcat(*Q)

        # Algebraic equations for ODEs using tanks-in-series
        resEe = ((self.n-1) / self.At / self.l) * (Q[:-1] * cEe[:-1] - Q[1:] * cEe[1:]) + self.pcat * rEe[1:]
        resEa = ((self.n-1) / self.At / self.l) * (Q[:-1] * cEa[:-1] - Q[1:] * cEa[1:]) + self.pcat * rEa[1:]
        resA = ((self.n-1) / self.At / self.l) * (Q[:-1] * cA[:-1] - Q[1:] * cA[1:]) + self.pcat * rA[1:]
        resW = ((self.n-1) / self.At / self.l) * (Q[:-1] * cW[:-1] - Q[1:] * cW[1:]) + self.pcat * rW[1:]
        resV = ((self.n-1) / self.At / self.l) * (Q[:-1] * cV[:-1] - Q[1:] * cV[1:]) + self.pcat * rV[1:]
        resO = ((self.n-1) / self.At / self.l) * (Q[:-1] * cO[:-1] - Q[1:] * cO[1:]) + self.pcat * rO[1:]
        resC = ((self.n-1) / self.At / self.l) * (Q[:-1] * cC[:-1] - Q[1:] * cC[1:]) + self.pcat * rC[1:]

        resP = P[:-1] - P[1:] - self.l / (self.n-1) * self.f * rhoav[1:] * Q[1:] ** 2
        # scale pressure drop with inlet pressure to keep the terms in similar range
        resP_scaled = resP / self.P20
        #resP = P[1:] - cTotal[1:] * self.R * T[1:]

        MrEe = self.mol[0] * rEe[1:]
        MrEa = self.mol[1] * rEa[1:]
        MrA = self.mol[2] * rA[1:]
        MrW = self.mol[3] * rW[1:]
        MrV = self.mol[4] * rV[1:]
        MrO = self.mol[5] * rO[1:]
        MrC = self.mol[6] * rC[1:]
        sumMr = MrEe + MrEa + MrA + MrW + MrV + MrO + MrC
        sumMr_mean = casadi.sum1(MrEe**2 + MrEa**2 + MrA**2 + MrW**2 + MrV**2 + MrO**2 + MrC**2) / (self.n - 1)
        sumMr_denom = casadi.sqrt(sumMr_mean + 1e-10)
        sumMr = sumMr / sumMr_denom
        sumMr = casadi.sum1(sumMr)


        delta_T_square = (T - 273.15) ** 2 - (self.T0 - 273.15) ** 2
        delta_T = T - self.T0

        spec = (casadi.mtimes(casadi.reshape(delta_T_square, self.n, 1), casadi.reshape((self.spec_b / 2), 1, 7)) +
                casadi.mtimes(casadi.reshape(delta_T, self.n, 1), casadi.reshape(self.spec_a, 1, 7)))

        speb = (casadi.mtimes(casadi.reshape(((T[:-1] - 273.15) ** 2 - (T[1:] - 273.15) ** 2), self.n - 1, 1),
                casadi.reshape((self.spec_b / 2),1, 7)) +
                casadi.mtimes(casadi.reshape((T[:-1] - T[1:]),self.n - 1, 1), casadi.reshape( self.spec_a,1, 7)))
        
        delH1n = -(-self.delH1 + self.delcp *(spec[:, 4] + spec[:, 3] - spec[:, 2] - spec[:, 0] - 0.5 * spec[:, 5]))
        delH2n = -(-self.delH2 + self.delcp *(2 * spec[:, 3] + 2 * spec[:, 6] - spec[:, 0] - 3 * spec[:, 5]))

        Hfn_Ee = self.Hf_Ee + spec[:, 0]
        Hfn_Ea = self.Hf_Ea + spec[:, 1]
        Hfn_A = self.Hf_A + spec[:, 2]
        Hfn_W = self.Hf_W + spec[:, 3]
        Hfn_V = self.Hf_V + spec[:, 4]
        Hfn_O = self.Hf_O + spec[:, 5]
        Hfn_C = self.Hf_C + spec[:, 6]
        sum_RjHj = -(rEe[1:] * Hfn_Ee[1:] + rEa[1:] * Hfn_Ea[1:] + rA[1:] * Hfn_A[1:] + rW[1:] * Hfn_W[1:] + rV[1:] * Hfn_V[1:] + rO[1:] * Hfn_O[1:] + rC[1:] * Hfn_C[1:])

        if self.delcp != 0:
            resT =  (((self.n-1) / self.At / self.l) *
                        ((speb[:, 0] * Q[:-1] * cEe[:-1]) +
                        (speb[:, 1] * Q[:-1]  * cEa[:-1]) +
                        (speb[:, 2] * Q[:-1]  * cA[:-1]) +
                        (speb[:, 3] * Q[:-1]  * cW[:-1]) +
                        (speb[:, 4] * Q[:-1]  * cV[:-1]) +
                        (speb[:, 5] * Q[:-1]  * cO[:-1]) +
                        (speb[:, 6] * Q[:-1]  * cC[:-1])) +
                        #self.pcat * (r1[1:] * delH1n[1:] + r2[1:] * delH2n[1:] + rheat[1:] * self.delH2) -
                        self.pcat * self.delcp * sum_RjHj -
                        self.U  * (T[1:] - self.Tc) * np.pi * self.dt / self.At
                    )
        else:
            resT = (T[:-1] - T[1:]) / self.T20

        res = casadi.vertcat(resEe, resEa, resA, resW, resV, resO, resC, resT, resP_scaled)

        Qflow = Q
        N1 = N1f
        N2 = N2 * self.nt
        NcE = Q * cEe * self.nt
        NcEt = Q * cEa * self.nt
        NcA = Q * cA * self.nt
        NcW = Q * cW * self.nt
        NcV = Q * cV * self.nt
        NcO = Q * cO * self.nt
        NcC = Q * cC * self.nt
        N3 = Q_in_last_cstr * C_in_last_cstr * self.nt
        N4 = (1 - self.phi) * N3
        N5 = self.phi * N3
        N6 = (1 - self.mu) * N5
        N7 = alpha * N6
        N8 = self.gamma * N4
        N9 = (1 - alpha) * N6
        N10 = (1 - self.gamma) * N4
        N11 = self.mu * N5
        N12 = (1 - beta) * N8
        N13 = beta * N8
        Nvec = [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10,
                     N11, N12, N13]
        initial_conc = [cEe[0], cEa[0], cA[0], cW[0], cV[0], cO[0], cC[0]]
        o2_con = N2[5] / casadi.sum1(N2) * 100
        Tmax_cons = (200 + 273.15) - T
        Tout_cons = -(130 + 273.15) + T[-1]
        Trec_cons = T[0] - T[-1]
        feed_econ = self.penalty[0] * N1[0] + self.penalty[1] * N1[2] + self.penalty[2] * N1[5]
        rec_econ = self.penalty[3] * casadi.sum1(N7) + self.penalty[5] * casadi.sum1(N7)**2 + self.penalty[4] * casadi.sum1(N13) + self.penalty[6] * casadi.sum1(N13)**2
        #rec_econ = self.penalty[3] * (casadi.sum1(N7)**2) + self.penalty[4] * (casadi.sum1(N13) + casadi.sum1(N13)**2)
        prod_econ = self.penalty[7] * N10[4] 
        purge_gas_econ = self.penalty[8] * casadi.sum1(N9)
        purge_liquid_econ = self.penalty[9] * casadi.sum1(N12)
        #purge_co2_econ = self.penalty[10] * casadi.sum1(N11)
        purge_co2_econ = self.penalty[10] * (N3[6] - N2[6])
        heat_removed = casadi.sum1(self.U * (T[1:] - self.Tc) * np.pi * self.dt * self.l / (self.n-1))
        total_eco = prod_econ - rec_econ - feed_econ - purge_gas_econ - purge_liquid_econ - purge_co2_econ - self.penalty[11] * heat_removed 
        
        input_to_nlp = [x, p] if self.param in self.rate_param_list else [x]
        residue_model = casadi.Function('residue_model', input_to_nlp, [res])
        get_all_conc = casadi.Function('get_all_conc', input_to_nlp, [cEe, cEa, cA, cW, cV, cO, cC, T, P, Qflow])
        get_rates_one_two = casadi.Function('get_rates_one_two', input_to_nlp, [r1, r2])
        check_sum_MR = casadi.Function('check_sum_MR', input_to_nlp, [sumMr])


        bound_keys = ['alpha', 'beta', 'T2', 'P2', 'NEe', 'NA', 'NO', 'x']
        bounds_list = []
        for key in bound_keys:
            bounds_list += bounds[key]
        ub, lb = [i[-1] for i in bounds_list], [i[0] for i in bounds_list]
        p_spec = casadi.DM(self.p_spec)

        # construct the IFT machinery
        # divide x into u (control inputs) and x (using the IFT map x = x(u))
        ub_ref = np.array(ub)[np.arange(len(self.p_spec))]
        lb_ref = np.array(lb)[np.arange(len(self.p_spec))]
        # get indices of ub-lb < tol
        # gives the fixed control inputs u 
        tol = 1e-5
        idx = np.where(ub_ref - lb_ref < tol)[0]

        u_freezed_by_me = idx
        u_free = np.delete(np.arange(len(self.p_spec)), u_freezed_by_me)
        
        scales = np.array([1, 1, self.T20, self.P20, self.N1_clean[0], self.N1_clean[2], self.N1_clean[5]])
        scales_use = scales[u_free]
        # form the penalty matrix on the control inputs
        formR = np.array(self.uR_spec)
        R = np.diag(1/formR[u_free])
        R_scaled = np.diag(1/(formR[u_free]*scales_use))
        self.R_matrix = R
        u = x[u_free]
        u0 = np.array(self.u_spec)[u_free]
        scaled_u = u 
        scaled_u0 = u0
        delta_u = scaled_u - scaled_u0
        penalty_term = casadi.mtimes(casadi.mtimes(delta_u.T, R), delta_u)
        save_penalty = casadi.Function('save_penalty', [x], [penalty_term])
        total_econ = total_eco - self.penalty[12] * penalty_term
       

        if self.param in self.rate_param_list:
            p0 = np.hstack([self.theta1, self.theta2])
            p0 = casadi.DM(p0)

        initial = casadi.DM(initial)
        initial = casadi.vertcat(p_spec, initial)
        prod = N10[4]*60/1000
        constraint = casadi.vertcat(res, prod, o2_con, Tmax_cons, Tout_cons, Trec_cons, sumMr)
        
        lbg = casadi.vertcat(casadi.DM.zeros(9 * (self.n - 1)), self.prod_shut * 0.79, 0, -self.Tmax_cons * 1e10 * np.ones(self.n), -self.Tout_cons * 1e10, 0 - self.Trec_cons, 0 - self.Mrsum)
        ubg = casadi.vertcat(casadi.DM.zeros(9 * (self.n - 1)), casadi.inf, self.o2_reg * 8.00, casadi.inf * np.ones(self.n), casadi.inf, 1e-3 + self.Trec_cons, 0 + self.Mrsum)
        
        if self.param in self.rate_param_list:
            nlp = {'x': x, 'f': -total_econ, 'g': constraint, 'p': p}
        else:
            nlp = {'x': x, 'f': -total_econ, 'g': constraint}
        solver = casadi.nlpsol('solver', 'ipopt', nlp, {'ipopt': {'print_level': self.print_level, 'tol': 1e-8, 'max_iter': 3000, 
                                                                  'nlp_scaling_method': 'none', 'mu_init':1e-10},})
        if self.param in self.rate_param_list:
            self.sol = solver(x0=initial, lbx=lb, ubx=ub, lbg=lbg, ubg=ubg, p=p0)
        else:
            self.sol = solver(x0=initial, lbx=lb, ubx=ub, lbg=lbg, ubg=ubg)
    
        self.sol_x = self.sol['x'].toarray().flatten()[7:]
        self.sol_alpha = self.sol['x'].toarray().flatten()[0]
        self.sol_beta = self.sol['x'].toarray().flatten()[1]
        self.sol_ta = self.sol['x'].toarray().flatten()[2]
        self.sol_pa = self.sol['x'].toarray().flatten()[3]
        self.sol_ne = self.sol['x'].toarray().flatten()[4]
        self.sol_na = self.sol['x'].toarray().flatten()[5]
        self.sol_no = self.sol['x'].toarray().flatten()[6]
        self.sol_cEe = self.sol['x'].toarray().flatten()[7 + (self.n - 2)] * self.weights[0]
        self.sol_cEa = self.sol['x'].toarray().flatten()[8 + 2 * (self.n - 2)] / self.weights[1]
        self.sol_cA = self.sol['x'].toarray().flatten()[9 + 3 * (self.n - 2)] * self.weights[2]
        self.sol_cW = self.sol['x'].toarray().flatten()[10 + 4 * (self.n - 2)] * self.weights[3]
        self.sol_cV = self.sol['x'].toarray().flatten()[11 + 5 * (self.n - 2)] * self.weights[4]
        self.sol_cO = self.sol['x'].toarray().flatten()[12 + 6 * (self.n - 2)] * self.weights[5]
        self.sol_cC = self.sol['x'].toarray().flatten()[13 + 7 * (self.n - 2)] * self.weights[6]
        self.solver = solver
        self.obj = -self.sol['f'].toarray().flatten()[0]
        self.save_penalty_val = save_penalty(self.sol['x']).toarray().flatten()[0]
        print("\nOptimization success: ", self.solver.stats()['return_status'])
        self.perturb_dict = {
        'alpha': self.sol_alpha,
        'beta': self.sol_beta,
        'ta': self.sol_ta,
        'pa': self.sol_pa,
        'ne': self.sol_ne,
        'na': self.sol_na,
        'no': self.sol_no,
        'sol': self.sol_x,
        'profit': self.obj,
        'penalty': self.save_penalty_val * self.penalty[12]
        }

        lb = np.array(lb)
        ub = np.array(ub)
        input_to_nlp_num = [self.sol['x'], p0] if self.param in self.rate_param_list else [self.sol['x']]

        self.residue_model = residue_model(*input_to_nlp_num).toarray().flatten()
        self.get_all_conc = get_all_conc(*input_to_nlp_num)
        self.rates = get_rates_one_two(*input_to_nlp_num)
        self.check_sum_MR = check_sum_MR(*input_to_nlp_num).toarray().flatten()[0]

        # for active constraint set h(x, u) = 0
        active_g = np.arange(0, 9 * (self.n - 1))   
        active_x = u_freezed_by_me

        active_g_cons = constraint[active_g]
        active_x_cons = x[active_x] - lb[active_x]

        # get u_freezed_by_opt which are the control inputs freezed by the optimization (i.e. at their bounds)
        check_lams = np.where(~np.isclose(self.sol['lam_x'][u_free], 0, rtol=1e-5, atol=1e-5))
        for i in check_lams[0]:
            if i not in active_x:
                active_x = np.hstack((active_x, i))
                if self.sol['lam_x'][i] > 0:
                    active_x_cons = casadi.vertcat(active_x_cons, x[i] - lb[i])
                else:
                    active_x_cons = casadi.vertcat(active_x_cons, x[i] - ub[i])
                
        u_freezed_by_opt = np.delete(np.sort(active_x), u_freezed_by_me)

        # finally we get the u and x.
        # finally we get h(x, u) = 0

        u_free = np.delete(np.arange(len(self.p_spec)), active_x)
        xu_bound = np.delete(np.arange(int(x.numel())), u_free)
        u_bound = np.union1d(u_freezed_by_me, u_freezed_by_opt)
        x_bound = np.delete(xu_bound, u_bound)

        self.u_free = u_free
        self.x_bound = x_bound
        self.u_bound = u_bound
        self.xu_bound = xu_bound
        self.u_freezed_by_opt = u_freezed_by_opt
        self.u_freezed_by_me = u_freezed_by_me

        h_act = casadi.vertcat(active_g_cons, active_x_cons)
        self.active_cons = h_act

        x_calc = x[xu_bound]
        u_calc = x[u_free]

        h_x = casadi.jacobian(h_act, x_calc)
        h_x_func = casadi.Function('h_x_func', input_to_nlp, [h_x])
        h_x_num = h_x_func(*input_to_nlp_num).toarray()
        # check shape using x_calc shape and u_calc shape
        assert h_x_num.shape[1] == x_calc.numel(), f"h_x shape is {h_x_num.shape}, expected ({h_act.numel()}, {x_calc.numel()})"

        # check whether we have row rank
        rank = np.linalg.matrix_rank(h_x_num)
        if rank < h_x_num.shape[0]:
            print("Warning: h_x is singular, IFT fails")
            self.log_licq = False
        # check whether any of the active constraints are at their bounds
        # this should not happen, if it does, we have redundant constraints in the active set
        # since ub is infty check only lb
        
        if np.isclose(self.sol_x, lb[x_bound], rtol=1e-5, atol=1e-5).any():
            print("Warning: LB on conc is active which is not included in h," \
                  "IFT fails")
            self.log_licq = False
        else:
            self.log_licq = True

        # we require luu, lux, lxu, lxx, lx, hu, huu, hux, hxu, hxx

        l = -total_econ
        l_x = casadi.jacobian(l, x_calc)
        l_x_func = casadi.Function('l_x_func', input_to_nlp, [l_x])
        l_x_num = l_x_func(*input_to_nlp_num).toarray().flatten()
        # check shape
        assert l_x_num.shape[0] == x_calc.numel(), f"l_x shape is {l_x_num.shape}, expected ({x_calc.numel()},)"

        l_xx = casadi.hessian(l, x_calc)[0]
        l_xx_func = casadi.Function('l_xx_func', input_to_nlp, [l_xx])
        l_xx_num = l_xx_func(*input_to_nlp_num).toarray()
        # check shape
        assert l_xx_num.shape[0] == x_calc.numel(), f"l_xx shape is {l_xx_num.shape}, expected ({x_calc.numel()},)"

        l_uu = casadi.hessian(l, u_calc)[0]
        l_uu_func = casadi.Function('l_uu_func', input_to_nlp, [l_uu])
        l_uu_num = l_uu_func(*input_to_nlp_num).toarray()
        # check shape
        assert l_uu_num.shape[0] == u_calc.numel(), f"l_uu shape is {l_uu_num.shape}, expected ({u_calc.numel()},)"

        l_xu = casadi.jacobian(casadi.jacobian(l, x_calc), u_calc)
        l_xu_func = casadi.Function('l_xu_func', input_to_nlp, [l_xu])
        l_xu_num = l_xu_func(*input_to_nlp_num).toarray()
        # check shape
        assert l_xu_num.shape == (x_calc.numel(), u_calc.numel()), f"l_xu shape is {l_xu_num.shape}, expected ({x_calc.numel()}, {u_calc.numel()})"

        l_ux = casadi.jacobian(casadi.jacobian(l, u_calc), x_calc)
        l_ux_func = casadi.Function('l_ux_func', input_to_nlp, [l_ux])
        l_ux_num = l_ux_func(*input_to_nlp_num).toarray()
        # check shape
        assert l_ux_num.shape == (u_calc.numel(), x_calc.numel()), f"l_ux shape is {l_ux_num.shape}, expected ({u_calc.numel()}, {x_calc.numel()})"

        h_u = casadi.jacobian(h_act, u_calc)
        h_u_func = casadi.Function('h_u_func', input_to_nlp, [h_u])
        h_u_num = h_u_func(*input_to_nlp_num).toarray()
        # check shape
        assert h_u_num.shape[1] == u_calc.numel(), f"h_u shape is {h_u_num.shape}, expected ({h_act.numel()}, {u_calc.numel()})"    


        n_x = x_calc.numel()
        n_u = u_calc.numel()
        n_h = h_act.numel()

        # x_u = -h_x^{-1} h_u  (n_x, n_u)
        x_u = -np.linalg.solve(h_x_num, h_u_num)

        # compute B row by row — B is (n_h, n_u, n_u)
        B = np.zeros((n_h, n_u, n_u))
        for i in range(n_h):
            h_uu_i = casadi.hessian(h_act[i], u_calc)[0]
            h_ux_i = casadi.jacobian(casadi.jacobian(h_act[i], u_calc), x_calc)
            h_xu_i = casadi.jacobian(casadi.jacobian(h_act[i], x_calc), u_calc)
            h_xx_i = casadi.hessian(h_act[i], x_calc)[0]

            f_i = casadi.Function(f'f_{i}', input_to_nlp, [h_uu_i, h_ux_i, h_xu_i, h_xx_i])
            h_uu_i_num, h_ux_i_num, h_xu_i_num, h_xx_i_num = [a.toarray() for a in f_i(*input_to_nlp_num)]

            B[i] = h_uu_i_num + h_ux_i_num @ x_u + x_u.T @ h_xu_i_num + x_u.T @ h_xx_i_num @ x_u

        # x_uu is (n_x, n_u, n_u) — solve slice by slice
        x_uu = np.zeros((n_x, n_u, n_u))
        for j in range(n_u):
            for k in range(n_u):
                x_uu[:, j, k] = np.linalg.solve(h_x_num, -B[:, j, k])

        # l_x @ x_uu: (n_x,) contracted with (n_x, n_u, n_u) -> (n_u, n_u)
        l_x_xuu = np.einsum('i,ijk->jk', l_x_num, x_uu)

        d2l_du2 = l_uu_num + l_ux_num @ x_u + x_u.T @ l_xu_num + x_u.T @ l_xx_num @ x_u + l_x_xuu
        self.d2l_du2 = d2l_du2

        scales_use = scales[u_free]
        R_scaled = np.diag(1/(formR[u_free]*scales_use))
        self.R_matrix = R_scaled    
        R_scaled_inv = np.linalg.inv(R_scaled)

        # scale the hessian to get back into the absolute scale of u
        self.d2l_du2_scaled = R_scaled_inv @ d2l_du2 @ R_scaled_inv

        # store condition number of hessian
        try:
            self.cond_hess = np.linalg.cond(self.d2l_du2)
            self.cond_hess_scaled = np.linalg.cond(self.d2l_du2_scaled)
        except np.linalg.LinAlgError:
            self.cond_hess = np.inf
            self.cond_hess_scaled = np.inf

        if self.param in self.rate_param_list:
            h_p = casadi.jacobian(h_act, p)
            h_p_func = casadi.Function('h_p_func', input_to_nlp, [h_p])
            h_p_num = h_p_func(*input_to_nlp_num).toarray()
            # check shape
            assert h_p_num.shape[1] == p.numel(), f"h_p shape is {h_p_num.shape}, expected ({h_act.numel()}, {p.numel()})"  

            # compute x_p = -h_x^{-1} h_p
            x_p = -np.linalg.solve(h_x_num, h_p_num)

            l_p = casadi.jacobian(l, p)
            l_p_func = casadi.Function('l_p_func', input_to_nlp, [l_p])
            l_p_num = l_p_func(*input_to_nlp_num).toarray().flatten()
            # check shape
            assert l_p_num.shape[0] == p.numel(), f"l_p shape is {l_p_num.shape}, expected ({p.numel()},)"

            l_xp = casadi.jacobian(casadi.jacobian(l, x_calc), p)
            l_xp_func = casadi.Function('l_xp_func', input_to_nlp, [l_xp])
            l_xp_num = l_xp_func(*input_to_nlp_num).toarray()
            # check shape
            assert l_xp_num.shape == (x_calc.numel(), p.numel()), f"l_xp shape is {l_xp_num.shape}, expected ({x_calc.numel()}, {p.numel()})"

            l_up = casadi.jacobian(casadi.jacobian(l, u_calc), p)
            l_up_func = casadi.Function('l_up_func', input_to_nlp, [l_up])
            l_up_num = l_up_func(*input_to_nlp_num).toarray()
            # check shape
            assert l_up_num.shape == (u_calc.numel(), p.numel()), f"l_up shape is {l_up_num.shape}, expected ({u_calc.numel()}, {p.numel()})"   

            n_p = p.numel()
            # compute C row by row — C is (n_h, n_u, n_p)
            C = np.zeros((n_h, n_u, n_p))
            for i in range(n_h):
                h_up_i = casadi.jacobian(casadi.jacobian(h_act[i], u_calc), p)
                h_ux_i = casadi.jacobian(casadi.jacobian(h_act[i], u_calc), x_calc)
                h_xp_i = casadi.jacobian(casadi.jacobian(h_act[i], x_calc), p)
                h_xx_i = casadi.hessian(h_act[i], x_calc)[0]
                f_i = casadi.Function(f'f_{i}', input_to_nlp, [h_up_i, h_ux_i, h_xp_i, h_xx_i])
                h_up_i_num, h_ux_i_num, h_xp_i_num, h_xx_i_num = [a.toarray() for a in f_i(*input_to_nlp_num)]
                C[i] = (h_up_i_num
                        + h_ux_i_num @ x_p
                        + (h_xp_i_num.T @ x_u).T
                        + (x_p.T @ h_xx_i_num @ x_u).T)

            # x_up is (n_x, n_u, n_p) — solve slice by slice
            x_up = np.zeros((n_x, n_u, n_p))
            for j in range(n_u):
                for k in range(n_p):
                    x_up[:, j, k] = np.linalg.solve(h_x_num, -C[:, j, k])

            # l_x @ x_up: (n_x,) contracted with (n_x, n_u, n_p) -> (n_u, n_p)
            l_x_xup = np.einsum('i,ijk->jk', l_x_num, x_up)

            u_p = -np.linalg.solve(d2l_du2, (x_p.T @ l_xx_num @ x_u).T + (l_xp_num.T @ x_u).T + l_x_xup + l_ux_num @ x_p + l_up_num)

            l_p = l_x_num @ x_p + l_p_num
            u_num = self.sol['x'][u_free].toarray().flatten()
            p_num = p0.toarray().flatten()
            u_p_relative = (1/u_num[:, None]) * u_p * p_num[None, :]

            self.u_p = u_p
            self.l_p = l_p
