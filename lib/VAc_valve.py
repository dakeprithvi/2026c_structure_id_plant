# Implementation for lagged control inputs. We use first-order filter for 
# fresh feeds and second-order filter for recycle

import numpy as np
from scipy.integrate import solve_ivp
from VAc_flowsheet import Simulator

class Lag_plant(Simulator):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        self.feed_lag = config.get('feed_lag', 200)
        self.alpha_lag = config.get('alpha_lag', 500)
        self.beta_lag = config.get('beta_lag', 500)
        self.ta_lag = config.get('ta_lag', 200)
        self.decay_ref = config.get('decay_ref', 0.5)
        self.simulate_temp = config.get('simulate_temp', 0)
        # for now we assume that the decay is very very small
        self.cat_tau = config.get('cat_tau', 1e10)

    def Transient_lag(self, t, x, *args):
        '''
        Prepares the RHS of the dynamic model of the flowsheet. 
        Same strategy as steady-state solver to simplify DAE to ODE
        if switch = 0,
        Returns RHS of the ODEs
        if switch != 0
        Returns inlet concentrations for each time step.
        '''
        switch,  = args

        decay, N1Ee, N1A, N1O, alp1, alp2, beta1, beta2, T2 = x[:9]
        N1f_lag = np.array([N1Ee, N1Ee * self.N1f[1] / self.N1f[0], N1A, self.N1f[3], self.N1f[4], N1O, self.N1f[6]])
        N10 = N1f_lag / self.nt
        x_mat = np.reshape(x[9:], (8, self.n - 1))

        resdecay = - (decay - self.decay_ref) / self.cat_tau
        # form the filter odes
        resNEe = (self.N1f[0] - N1Ee) / self.feed_lag
        resNA = (self.N1f[2] - N1A) / self.feed_lag
        resNO = (self.N1f[5] - N1O) / self.feed_lag
        resalp1 = (self.alpha - alp1) / self.alpha_lag
        resalp2 = (alp1 - alp2) / self.alpha_lag
        resbeta1 = (self.beta - beta1) / self.beta_lag
        resbeta2 = (beta1 - beta2) / self.beta_lag
        resT2 = (self.T2 - T2) / self.ta_lag

        cEe, cEa, cA, cW, cV, cO, cC, T = [row for row in x_mat]

        T = np.concatenate([np.array([T2]), T])
        P = np.concatenate([np.array([self.P2]), np.ones(self.n-1) * self.P2])

        c = (cEe, cEa, cA, cW, cV, cO, cC, T[1:])
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate(c)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC  
        

        self.rec_factor = self.phi * (1 - self.mu) * alp2 + (1 - self.phi) * self.gamma * beta2

        T_inlet = T2 if self.rho_var else self.T_const_inlet
        rho_in = self.P2 / self.R / T_inlet

        t_factor =T[-1]*P[0]/ P[-1]/T[0] if self.const_pressure else 1
        t_fac_array = np.ones(self.n - 1)
        for i in range(self.n - 2):
            t_fac_array[i + 1] = t_fac_array[i] * T[-1 - i] * P[-1 - (i  + 1)] / T[-1 - (i + 1)] / P[-1 - i]
        t_fac_array = np.flip(t_fac_array) if self.const_pressure else np.ones(self.n - 1)


        C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
        Qf = (np.sum(N10) + self.constQ * self.pcat  * self.R* (self.At * self.l / (self.n-1)) * np.sum(self.rec_factor * C_in_last_cstr) * 
              np.sum(T[1:] * sum_Rj * t_fac_array / P[1:])) / (rho_in - t_factor * np.sum(self.rec_factor * C_in_last_cstr)) 
        Q_in_last_cstr = Qf * t_factor + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * np.sum(T[1:] / P[1:] * sum_Rj * t_fac_array)
        N2 = N10 + Q_in_last_cstr * C_in_last_cstr * self.rec_factor

        Q_in_first_cstr = np.sum(N2) / rho_in
        cEe_0, cEa_0, cA_0, cW_0, cV_0, cO_0, cC_0 = N2 / Q_in_first_cstr

        cEe = np.concatenate([np.array([cEe_0]), cEe])
        cEa = np.concatenate([np.array([cEa_0]), cEa])
        cA = np.concatenate([np.array([cA_0]), cA])
        cW = np.concatenate([np.array([cW_0]), cW])
        cV = np.concatenate([np.array([cV_0]), cV])
        cO = np.concatenate([np.array([cO_0]), cO])
        cC = np.concatenate([np.array([cC_0]), cC])

        cTotal = cEe + cEa + cA + cW + cV + cO + cC
        C = np.vstack((cEe, cEa, cA, cW, cV, cO, cC))
        c = (cEe, cEa, cA, cW, cV, cO, cC, T)
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate(c)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC

        Mav = np.dot(self.mol, C) / cTotal
        rhoav = P * Mav / self.R / T * 10 ** -3
        Q = np.zeros(self.n)
        Q[0] = Qf
        if self.const_pressure:
            for i in range(len(Q) - 1):
                Q[i + 1] = Q[i]* T[i+1] * P[i]/T[i]/P[i+1] + self.constQ * self.pcat  * self.R * (self.At * self.l / (self.n-1)) * T[i+1] / P[i+1] * sum_Rj[i+1]
        else:
            for i in range(len(Q) - 1):
                Q[i + 1] = Q[i]  + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * T[i+1] / P[i+1] * sum_Rj[i+1]

        # ODEs RHS using tanks-in-series
        resEe = (((self.n-1) / self.At / self.l) * (Q[:-1] * cEe[:-1] - Q[1:] * cEe[1:]) + self.pcat * rEe[1:]) / self.e
        resEa = (((self.n-1) / self.At / self.l) * (Q[:-1] * cEa[:-1] - Q[1:] * cEa[1:]) + self.pcat * rEa[1:]) / self.e
        resA = (((self.n-1) / self.At / self.l) * (Q[:-1] * cA[:-1] -  Q[1:] * cA[1:]) + self.pcat * rA[1:]) / self.e
        resW = (((self.n-1) / self.At / self.l) * (Q[:-1] * cW[:-1] - Q[1:] * cW[1:]) + self.pcat * rW[1:]) / self.e
        resV = (((self.n-1) / self.At / self.l) * (Q[:-1] * cV[:-1] - Q[1:] * cV[1:]) + self.pcat * rV[1:]) / self.e
        resO = (((self.n-1) / self.At / self.l) * (Q[:-1] * cO[:-1] - Q[1:] * cO[1:]) + self.pcat * rO[1:]) / self.e
        resC = (((self.n-1) / self.At / self.l) * (Q[:-1] * cC[:-1] - Q[1:] * cC[1:]) + self.pcat * rC[1:]) / self.e

        spec = (np.dot(np.reshape((T - 273.15) ** 2 - (self.T0 - 273.15) ** 2, (self.n, 1)),
                      np.reshape(self.spec_b / 2, (1, 7))) +
                np.dot(np.reshape((T - self.T0), (self.n, 1)), np.reshape(self.spec_a, (1, 7))))

        spe = (np.dot(np.reshape((T - 273.15), (self.n, 1)), np.reshape(self.spec_b , (1, 7))) +
               np.dot(np.ones((self.n, 1)), np.reshape(self.spec_a, (1, 7))))

        speb = np.dot(np.reshape((T[:-1] - 273.15) ** 2 - (T[1:] - 273.15) ** 2, (self.n - 1, 1)),
                      np.reshape(self.spec_b / 2, (1, 7))) + np.dot(np.reshape((T[:-1] - T[1:]), (self.n - 1, 1)),
                                                                    np.reshape(self.spec_a, (1, 7)))
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

        if self.simulate_temp == 1:
            resT = (((((self.n-1) / self.At / self.l) *
                    ((speb[:, 0] * Q[:-1]  * cEe[:-1]) +
                    (speb[:, 1] * Q[:-1]  * cEa[:-1]) +
                    (speb[:, 2] * Q[:-1]  * cA[:-1]) +
                    (speb[:, 3] * Q[:-1]  * cW[:-1]) +
                    (speb[:, 4] * Q[:-1]  * cV[:-1]) +
                    (speb[:, 5] * Q[:-1]  * cO[:-1]) +
                    (speb[:, 6] * Q[:-1]  * cC[:-1])) +
                    #self.pcat * (r1[1:] * delH1n[1:] + r2[1:] * delH2n[1:] + rheat[1:] * self.delH2) -
                    self.pcat * self.delcp * sum_RjHj -
                    self.U  * (T[1:] - self.Tc) * np.pi * self.dt / self.At))
                    / (self.pcat * self.ccat + self.e * (spe[1:,0] * cEe[1:] + spe[1:,1] * cEa[1:] +
                                                        spe[1:,2] * cA[1:] + spe[1:,3] * cW[1:] +
                                                        spe[1:,4] * cV[1:] + spe[1:,5] * cO[1:] +
                                                        spe[1:,6] * cC[1:])))
        else:
            resT = (T[:-1] - T[1:]) / 1e-5
    
        
        scalar_res = np.array([resdecay, resNEe, resNA, resNO, resalp1, resalp2, resbeta1, resbeta2, resT2])
        self.resTotal = np.concatenate((scalar_res, resEe, resEa, resA, resW, resV, resO, resC, resT))
        if switch == 0:
            return self.resTotal

        else:
            return cEe[0], cEa[0], cA[0], cW[0], cV[0], cO[0], cC[0], Q

    def solve_t_lag(self, tupl):
        '''
        A convenience function that solves the transient model and return a 
        nice formatted set of variables that can be used further. 
        '''
        self.initialize()
        C_at_first_cstr_t0, result = tupl
        decay, N1Ee, N1A, N1O, alp1, alp2, beta1, beta2, T2 = result[:9]
        x_mat = np.reshape(result[9:], (9, self.n - 1))
        cEe, cEa, cA, cW, cV, cO, cC, T, P = [row for row in x_mat]

        # Weights for the concentration variables
        w = self.weights
        cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]
        scalar = np.array([decay, N1Ee, N1A, N1O, alp1, alp2, beta1, beta2, T2])
        self.initial = np.concatenate([scalar, cEe, cEa, cA, cW, cV, cO, cC, w[7] * T])
        self.ode_sol = solve_ivp(self.Transient_lag, self.time, self.initial, method = "Radau", t_eval = self.teval, args = (0, ), rtol = 1e-8, atol = 1e-8)
        #if self.ode_sol.status != 0:
        print("\nODE solver status: ", self.ode_sol.message)

        self.cEe0t = np.zeros((1, len(self.teval)))
        self.cEa0t = np.zeros((1, len(self.teval)))
        self.cA0t = np.zeros((1, len(self.teval)))
        self.cW0t = np.zeros((1, len(self.teval)))
        self.cV0t = np.zeros((1, len(self.teval)))
        self.cO0t = np.zeros((1, len(self.teval)))
        self.cC0t = np.zeros((1, len(self.teval)))
        self.Qa = np.zeros((self.n, len(self.teval)))

        for i in range(len(self.teval)):
            cEe_0, cEa_0, cA_0, cW_0, cV_0, cO_0, cC_0, Q = self.Transient_lag(self.teval[i], self.ode_sol.y.T[i], (1,))
            self.cEe0t[:,i] = cEe_0
            self.cEa0t[:,i] = cEa_0
            self.cA0t[:,i] = cA_0
            self.cW0t[:,i] = cW_0
            self.cV0t[:,i] = cV_0
            self.cO0t[:,i] = cO_0
            self.cC0t[:,i] = cC_0
            self.Qa[:,i] = Q
            
        # We need to replace the intial values for the first CSTR. This is because, we may start the transient model
        # with different rec_factor, thus, the calculated initial values are not correct. 

        # This is not a bug, but a compromise to avoid solving DAEs in the transient model.
        self.cEe0t[:,0] = C_at_first_cstr_t0[0]
        self.cEa0t[:,0] = C_at_first_cstr_t0[1]
        self.cA0t[:,0] = C_at_first_cstr_t0[2]
        self.cW0t[:,0] = C_at_first_cstr_t0[3]
        self.cV0t[:,0] = C_at_first_cstr_t0[4]
        self.cO0t[:,0] = C_at_first_cstr_t0[5]
        self.cC0t[:,0] = C_at_first_cstr_t0[6]
        
        # Extract calculated concentrations for each component
        self.decay = self.ode_sol.y[0, :]
        self.N1Ee = self.ode_sol.y[1, :]
        self.N1A = self.ode_sol.y[2, :]
        self.N1O = self.ode_sol.y[3, :]
        self.alp1 = self.ode_sol.y[4, :]
        self.alp2 = self.ode_sol.y[5, :]
        self.beta1 = self.ode_sol.y[6, :]
        self.beta2 = self.ode_sol.y[7, :]
        self.T2_lag = self.ode_sol.y[8, :]
        COL = [self.ode_sol.y[9 + i * (self.n - 1):9 + (i + 1) * (self.n - 1), :] for i in range(8)]

        self.cEe_t = COL[0]
        self.cEa_t = COL[1]
        self.cA_t = COL[2]
        self.cW_t = COL[3]
        self.cV_t = COL[4]
        self.cO_t = COL[5]
        self.cC_t = COL[6]
        self.T_t = COL[7]

        # Finally prepare and store the variables for further use. 
        self.cEe_tall = np.vstack((self.cEe0t, self.cEe_t))
        self.cEa_tall = np.vstack((self.cEa0t, self.cEa_t))
        self.cA_tall = np.vstack((self.cA0t, self.cA_t))
        self.cW_tall = np.vstack((self.cW0t, self.cW_t))
        self.cV_tall = np.vstack((self.cV0t, self.cV_t))
        self.cO_tall = np.vstack((self.cO0t, self.cO_t))
        self.cC_tall = np.vstack((self.cC0t, self.cC_t))
        self.T_tall = np.vstack((self.T2_lag.reshape((1, len(self.teval))), self.T_t))
    
    def Trans_Econ_lag(self):
        '''
        One more convenience function to calculate the molar flowrates for end time step. 
        It can also be use to calculate the end profit of the transient model.
        Return the profit.
        '''
        cEe = self.cEe_tall[:,-1]
        cEa = self.cEa_tall[:,-1]
        cA = self.cA_tall[:,-1]
        cW = self.cW_tall[:,-1]
        cV = self.cV_tall[:,-1]
        cO = self.cO_tall[:,-1]
        cC = self.cC_tall[:,-1]
        T = self.T_tall[:,-1]
        P = self.P2 * np.ones((1, len(T)))

        C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
        rec_factor = self.phi * (1 - self.mu) * self.alp2[-1] + (1 - self.phi) * self.gamma * self.beta2[-1]
        N1f_lag = np.array([self.N1Ee[-1], self.N1Ee[-1] * self.N1f[1] / self.N1f[0], self.N1A[-1], self.N1f[3], self.N1f[4], self.N1O[-1], self.N1f[6]])
        N10 = N1f_lag / self.nt
        N2 = N10 + self.Qa[-1,-1] * C_in_last_cstr * rec_factor

        self.N1_t = N1f_lag
        self.N2_t = N2 * self.nt
        self.NcE_t = self.Qa[:,-1] * self.cEe_tall[:,-1] * self.nt
        self.NcEt_t = self.Qa[:,-1] * self.cEa_tall[:,-1] * self.nt
        self.NcA_t = self.Qa[:,-1] * self.cA_tall[:,-1] * self.nt
        self.NcW_t = self.Qa[:,-1] * self.cW_tall[:,-1] * self.nt
        self.NcV_t = self.Qa[:,-1] * self.cV_tall[:,-1] * self.nt
        self.NcO_t = self.Qa[:,-1] * self.cO_tall[:,-1] * self.nt
        self.NcC_t = self.Qa[:,-1] * self.cC_tall[:,-1] * self.nt
        self.N3_t = self.Qa[-1,-1] * C_in_last_cstr * self.nt
        self.N4_t = (1 - self.phi) * self.N3_t
        self.N5_t = self.phi * self.N3_t
        self.N6_t = (1 - self.mu) * self.N5_t
        self.N7_t = self.alp2[-1] * self.N6_t
        self.N8_t = self.gamma * self.N4_t
        self.N9_t = (1 - self.alp2[-1]) * self.N6_t
        self.N10_t = (1 - self.gamma) * self.N4_t
        self.N11_t = self.mu * self.N5_t
        self.N12_t = (1 - self.beta2[-1]) * self.N8_t
        self.N13_t = self.beta2[-1] * self.N8_t

        o2_con = self.N2_t[5] / np.sum(self.N2_t)
        feed_econ = self.penalty[0] * self.N1_t[0] + self.penalty[1] * self.N1_t[2] + self.penalty[2] * self.N1_t[5]
        rec_econ = self.penalty[3] * np.sum(self.N7_t) + self.penalty[4] * np.sum(self.N13_t)
        prod_econ = self.penalty[5] * self.N10_t[4]
        purge_gas_econ = self.penalty[6] * np.sum(self.N9_t)
        purge_liquid_econ = self.penalty[7] * np.sum(self.N12_t)
        purge_co2_econ = self.penalty[8] * np.sum(self.N11_t)
        total_econ = prod_econ - rec_econ - feed_econ - purge_gas_econ - purge_liquid_econ - purge_co2_econ
        
        
        w = self.weights
        T, P = T[1:]/w[7], np.ones(len(T[1:]))
        initial_conc = [cEe[0], cEa[0], cA[0], cW[0], cV[0], cO[0], cC[0]]
        cEe, cEa, cA, cW, cV, cO, cC = cEe[1:] / w[0], cEa[1:] * w[1], cA[1:] / w[2], cW[1:] / w[3], cV[1:] / w[4], cO[1:] / w[5], cC[1:] / w[6]
        scalars = np.array([self.decay[-1], self.N1Ee[-1], self.N1A[-1], self.N1O[-1], self.alp1[-1], self.alp2[-1], self.beta1[-1], self.beta2[-1], self.T2_lag[-1]])
        self.impresult = np.concatenate([scalars, cEe, cEa, cA, cW, cV, cO, cC, T, P])
        self.init_for_lag_dynamics = (initial_conc, self.impresult)
        self.cons_vio = -(0.79 * self.prod_shut * 1000/60 - self.N10_t[4])
        return total_econ
    
    def get_dynamic_lag(self):
        N_Ee_fs = []
        N_Ea_fs = []
        N_A_fs = []
        N_W_fs = []
        N_V_fs = []
        N_O_fs = []
        N_C_fs = []
        N1inlet = []
        Qcheck = []
        T_end = []
        cEe_tanks = np.zeros((self.n, len(self.teval)))
        cEa_tanks = np.zeros((self.n, len(self.teval)))
        cA_tanks = np.zeros((self.n, len(self.teval)))
        cW_tanks = np.zeros((self.n, len(self.teval)))
        cV_tanks = np.zeros((self.n, len(self.teval)))
        cO_tanks = np.zeros((self.n, len(self.teval)))
        cC_tanks = np.zeros((self.n, len(self.teval)))
        for i in range(len(self.teval)):
            cEe = self.cEe_tall[-1, i]
            cEa = self.cEa_tall[-1, i]
            cA = self.cA_tall[-1, i]
            cW = self.cW_tall[-1, i]
            cV = self.cV_tall[-1, i]
            cO = self.cO_tall[-1, i]
            cC = self.cC_tall[-1, i]
            T = self.T_tall[-1, i]
            P = self.P2 


            C_in_last_cstr = np.array([cEe, cEa, cA, cW, cV, cO, cC])
            rec_factor = self.phi * (1 - self.mu) * self.alp2[i] + (1 - self.phi) * self.gamma * self.beta2[i]
            N1f_lag = np.array([self.N1Ee[i], self.N1Ee[i] * self.N1f[1] / self.N1f[0], self.N1A[i], self.N1f[3], self.N1f[4], self.N1O[i], self.N1f[6]])
            N10 = N1f_lag / self.nt
            N2 = N10 + self.Qa[-1, i] * C_in_last_cstr * rec_factor

            N1_t = self.N1f
            N2_t = N2 * self.nt
            NcE_t = self.Qa[-1, i] * self.cEe_tall[-1 , i] * self.nt
            NcEt_t = self.Qa[-1, i] * self.cEa_tall[-1, i] * self.nt
            NcA_t = self.Qa[-1, i] * self.cA_tall[-1, i] * self.nt
            NcW_t = self.Qa[-1, i] * self.cW_tall[-1, i] * self.nt
            NcV_t = self.Qa[-1, i] * self.cV_tall[-1, i] * self.nt
            NcO_t = self.Qa[-1, i] * self.cO_tall[-1, i] * self.nt
            NcC_t = self.Qa[-1, i] * self.cC_tall[-1, i] * self.nt
            N3_t = self.Qa[-1, i] * C_in_last_cstr * self.nt
            N4_t = (1 - self.phi) * N3_t
            N5_t = self.phi * N3_t
            N6_t = (1 - self.mu) * N5_t
            N7_t = self.alp2[i] * N6_t
            N8_t = self.gamma * N4_t
            N9_t = (1 - self.alp2[i]) * N6_t
            N10_t = (1 - self.gamma) * N4_t
            N11_t = self.mu * N5_t
            N12_t = (1 - self.beta2[i]) * N8_t
            N13_t = self.beta2[i] * N8_t
            N_Ee_fs.append(self.cEe_tall[-1, i])
            N_Ea_fs.append(self.cEa_tall[-1, i])
            N_A_fs.append(self.cA_tall[-1, i])
            N_W_fs.append(self.cW_tall[-1, i])
            N_V_fs.append(self.cV_tall[-1, i])
            N_O_fs.append(self.cO_tall[-1, i])
            N_C_fs.append(self.cC_tall[-1, i])
            T_end.append(self.T_tall[-1, i])
            N1inlet.append(N1_t)
            Qcheck.append(self.Qa[-1, i] * self.nt)
            cEe_tanks[:, i] = self.cEe_tall[:, i]
            cEa_tanks[:, i] = self.cEa_tall[:, i]
            cA_tanks[:, i] = self.cA_tall[:, i]
            cW_tanks[:, i] = self.cW_tall[:, i]
            cV_tanks[:, i] = self.cV_tall[:, i]
            cO_tanks[:, i] = self.cO_tall[:, i]
            cC_tanks[:, i] = self.cC_tall[:, i]
        dyn_dict = {}
        dyn_dict['cEe'] = N_Ee_fs
        dyn_dict['cEa'] = N_Ea_fs
        dyn_dict['cA'] = N_A_fs
        dyn_dict['cW'] = N_W_fs
        dyn_dict['cV'] = N_V_fs
        dyn_dict['cO'] = N_O_fs
        dyn_dict['cC'] = N_C_fs
        dyn_dict['cEe_tanks'] = cEe_tanks
        dyn_dict['cEa_tanks'] = cEa_tanks
        dyn_dict['cA_tanks'] = cA_tanks
        dyn_dict['cW_tanks'] = cW_tanks
        dyn_dict['cV_tanks'] = cV_tanks
        dyn_dict['cO_tanks'] = cO_tanks
        dyn_dict['cC_tanks'] = cC_tanks
        dyn_dict['alpha'] = self.alpha * np.ones(len(self.teval))
        dyn_dict['beta'] = self.beta * np.ones(len(self.teval))
        dyn_dict['NEe'] = self.ne * np.ones(len(self.teval)) * self.N1_clean[0]
        dyn_dict['NA'] = self.na * np.ones(len(self.teval)) * self.N1_clean[2]
        dyn_dict['NO'] = self.no * np.ones(len(self.teval)) * self.N1_clean[5]
        dyn_dict['alpha_2'] = self.alp2
        dyn_dict['beta_2'] = self.beta2
        dyn_dict['alpha_1'] = self.alp1
        dyn_dict['beta_1'] = self.beta1
        dyn_dict['decay'] = self.decay
        dyn_dict['T2']  = self.ta * np.ones(len(self.teval)) * self.T20
        dyn_dict['P2'] = self.pa * np.ones(len(self.teval)) * self.P20
        dyn_dict['NEe_1'] = self.N1Ee
        dyn_dict['NA_1'] = self.N1A
        dyn_dict['NO_1'] = self.N1O
        dyn_dict['T_1'] = self.T2_lag
        dyn_dict['time'] = self.teval
        dyn_dict['N1f'] = N1inlet
        dyn_dict['T_end'] = T_end
        dyn_dict['Q'] = Qcheck
        return dyn_dict
    
# test


# Plant = Lag_plant()
# Plant.constQ = 0
# Plant.heatfact = 0
# Plant.delcp = 0
# Plant.U = 0
# Plant.n = 2
# Plant.Stream_table()
# lag = 100
# Plant.alpha_lag = lag
# Plant.beta_lag = lag
# Plant.feed_lag = lag
# Plant.ne = 1.5
# Plant.na = 1.1
# Plant.no = 0.5
# Plant.alpha = 0.5
# Plant.beta = 0.5
# Plant.time = [0, 2000]
# Plant.t_points = 200

# Plant.solve_t_lag(Plant.init_for_lag_t0)
# num_vars = 5
# import matplotlib.pyplot as plt
# plt.figure()
# plt.subplot(num_vars, 1, 1)
# plt.plot(Plant.teval, Plant.cA_tall[-1,:], 'r', label='cEe')
# plt.plot(Plant.teval, Plant.cA_tall[0,:], 'b', label='cEa_0')
# plt.subplot(num_vars, 1, 2)
# plt.plot(Plant.teval, Plant.alp1, label='alp1')
# plt.subplot(num_vars, 1, 3)
# plt.plot(Plant.teval, Plant.alp2, label='alp2')
# plt.subplot(num_vars, 1, 4)
# plt.plot(Plant.teval, Plant.Qa[-1,:], label='Q')
# plt.subplot(num_vars, 1, 5)
# plt.plot(Plant.teval, Plant.beta1, label='N1Ee')

# plt.legend()
# plt.tight_layout()
# plt.savefig('lag_plant.pdf')


# # test
# Plant = Simulator()
# Plant.constQ = 0
# Plant.heatfact = 0
# Plant.delcp = 0
# Plant.U = 0
# Plant.n = 2
# Plant.Stream_table()
# lag = 1e-5
# Plant.alpha_lag = lag
# Plant.beta_lag = lag
# Plant.feed_lag = lag
# Plant.ne = 1.5
# Plant.na = 1.1
# Plant.no = 0.5
# Plant.alpha = 0.5
# Plant.beta = 0.5
# Plant.time = [0, 2000]
# Plant.t_points = 200

# Plant.solve_t(Plant.init_for_t0)
# num_vars = 5

# plt.figure()
# plt.subplot(num_vars, 1, 1)
# plt.plot(Plant.teval, Plant.cA_tall[-1,:], 'r', label='cEe')
# plt.plot(Plant.teval, Plant.cA_tall[0,:], 'b', label='cEa_0')
# plt.subplot(num_vars, 1, 2)
# plt.plot(Plant.teval, [Plant.alpha] * len(Plant.teval), label='alp1')
# plt.subplot(num_vars, 1, 3)
# plt.plot(Plant.teval, [Plant.alpha] * len(Plant.teval), label='alp2')
# plt.subplot(num_vars, 1, 4)
# plt.plot(Plant.teval, Plant.Qa[-1,:], label='Q')
# plt.subplot(num_vars, 1, 5)
# plt.plot(Plant.teval, [Plant.beta] * len(Plant.teval), label='N1Ee')

# plt.legend()
# plt.tight_layout()
# plt.savefig('plant.pdf')