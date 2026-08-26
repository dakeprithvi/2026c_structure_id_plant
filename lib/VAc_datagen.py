# Convenience class to generate training data set
# Right now, you can generate data for lagged plant inputs and multiple cstr

import numpy as np
import matplotlib.pyplot as plt
import casadi
from config import labels, get_label
from VAc_valve import Lag_plant
from utils import PickleTool
import time
import pandas as pd
from itertools import product

class Data_generator(Lag_plant):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}

        self.constQ = config.get('constQ', 0)
        self.n = config.get('n', 2)
        self.penalty = config.get('penalty', [0.1, 0.5, 0.5, 0.1, 0.5, 4, 0, 0, 0])
        #self.delcp = config.get('delcp', 0)
        #self.heatfact = config.get('heatfact', 0)
        self.factfor2 = config.get('factfor2', 1)
        #self.U = config.get('U', 0)
        self.prod_shut = config.get('prod_shut', 0.)
        self.bias = config.get('bias', None)
        # define variable for the data generator
        self.sampling = config.get('sampling', 20)
        self.t_start = config.get('t_start', 0)
        self.t_end_spec = config.get('t_end_spec', 2000)
        self.random_jumps = config.get('random_jumps', 10)
        self.std_noise = config.get('std_noise', 0)
        self.mean_noise = config.get('mean_noise', 0.0)
        self.train_fraction = config.get('train_fraction', 0.7)
        self.perturb = config.get('perturb', 0.4)
        self.bounds_free = config.get('bounds_free', 1)
        self.valve_lag = config.get('valve_lag', 0)
        self.det_ss = config.get('det_ss', 1)
        self.simulate_custom_traj = config.get('simulate_custom_traj', 0)
        self.random_perturb = config.get('random_perturb', 1)
        self.keep_T_const = config.get('keep_T_const', 0)
        self.keep_Q_const_trans = config.get('keep_Q_const_trans', 0)
        self.digital_stepping_mag = config.get('digital_stepping_mag', 0.02)
        self.digital_stepping = config.get('digital_stepping', 0)
        self.great_guess = config.get('great_guess', None)

    def keep_bounds_free(self):
        if self.bounds_free == 1:
            bounds = {}
            bounds['alpha'] = [(0., 0.99)]
            bounds['beta'] = [(0., 0.99)]
            bounds['NO'] = [(1e-1, 100)]
            bounds['NEe'] = [(1e-1, 100)]
            bounds['NA'] = [(1e-1, 100)]
            if self.keep_T_const == 1:
                bounds['T2'] = [(1, 1)]
            else:
                bounds['T2'] = [(0.9, 5)]
            self.bounds = bounds
        else:
            # check if bounds are already provided
            if not hasattr(self, 'bounds'):
                ValueError("Bounds not provided!")
        self.bounds['P2'] = [(1, 1)]


    def dummy_optimize(self):
        if self.great_guess is not None:
            self.Stream_table(self.great_guess)
        else:
            self.Stream_table()
        self.keep_bounds_free()
        if self.valve_lag == 0:
            self.initial_for_datagen = self.init_for_t0
        else:
            self.initial_for_datagen = self.init_for_lag_t0 
        self.bounds['x'] = [(1e-6, casadi.inf)] * self.vars
        self.optimize(self.result, self.bounds)
        self.opti_dict = {}
        self.opti_dict['alpha'] = self.sol_alpha
        self.opti_dict['beta'] = self.sol_beta
        self.opti_dict['NO'] = self.sol_no * self.N1_clean[5]
        self.opti_dict['NEe'] = self.sol_ne * self.N1_clean[0]
        self.opti_dict['NA'] = self.sol_na * self.N1_clean[2]
        self.opti_dict['T2'] = self.sol_ta * self.T20
        self.opti_dict['ta'] = self.sol_ta
        self.opti_dict['na'] = self.sol_na
        self.opti_dict['ne'] = self.sol_ne
        self.opti_dict['no'] = self.sol_no
        self.opti_dict['cEe'] = self.sol_cEe
        self.opti_dict['cEa'] = self.sol_cEa
        self.opti_dict['cA'] = self.sol_cA
        self.opti_dict['cW'] = self.sol_cW
        self.opti_dict['cV'] = self.sol_cV
        self.opti_dict['cO'] = self.sol_cO
        self.opti_dict['cC'] = self.sol_cC
        self.opti_dict['profit'] = self.obj
        
       
        # warm solve to get the time (tau) to reach steady state
        # such that y(t+h) - y(t) < 0.05 for Ethylene concentration
        # take the sampling time as tau/50
        # and t_end_spec as 4 * tau without temp and 4.5 with temp
        if self.det_ss == 1:
            self.alpha = 0.96
            self.beta = 0.96
            self.time = [0, 5e5]
            self.teval = np.linspace(0, 1e5, 5000)
            if self.valve_lag == 0 and self.delcp == 0:
                self.solve_t(self.initial_for_datagen)
                norm = abs(self.ode_sol.y[0,:-1] - self.ode_sol.y[0,1:])
                threshold = 5e-2
                mask = norm < threshold
                consecutive = mask[:-2] & mask[1:-1] & mask[2:]
                idx = np.where(consecutive)[0][0]
                self.time_for_steady = self.ode_sol.t[idx]
                self.sampling = int(self.time_for_steady / 50)
                self.t_end_spec = int(4 * self.time_for_steady)
            elif self.valve_lag == 1 and self.delcp == 0:
                self.solve_t_lag(self.initial_for_datagen)
                norm = abs(self.ode_sol.y[8,:-1] - self.ode_sol.y[8,1:])
                threshold = 0.7e-2
                mask = norm < threshold
                consecutive = mask[:-2] & mask[1:-1] & mask[2:]
                idx = np.where(consecutive)[0][0]
                self.time_for_steady = self.ode_sol.t[idx]
                self.sampling = int(self.time_for_steady / 50)
                self.t_end_spec = int(4 * self.time_for_steady)
            elif self.valve_lag == 0 and self.delcp != 0:
                self.solve_t(self.initial_for_datagen)
                norm = abs(self.ode_sol.y[-1,:-1] - self.ode_sol.y[-1,1:])
                threshold = 5e-3
                mask = norm < threshold
                consecutive = mask[:-2] & mask[1:-1] & mask[2:]
                idx = np.where(consecutive)[0][0]
                self.time_for_steady = self.ode_sol.t[idx]
                self.sampling = int(self.time_for_steady / 50)
                self.t_end_spec = int(4 * self.time_for_steady)
            elif self.valve_lag == 1 and self.delcp != 0:
                self.solve_t_lag(self.initial_for_datagen)
                norm = abs(self.ode_sol.y[-1,:-1] - self.ode_sol.y[-1,1:])
                threshold = 5e-3
                mask = norm < threshold
                consecutive = mask[:-2] & mask[1:-1] & mask[2:]
                idx = np.where(consecutive)[0][0]
                self.time_for_steady = self.ode_sol.t[idx]
                self.sampling = int(self.time_for_steady / 50)
                self.t_end_spec = int(4 * self.time_for_steady)
            else:
                raise ValueError("valve_lag must be 0 or 1, delcp must be 0 or 1")
            
            print('\nData generation will take some time')
            print(f"tau = {self.time_for_steady}, sampling = {self.sampling}, t_end_spec = {self.t_end_spec}")
        else:
            print('\nCopying specified values from user')
            print('\nData generation will take some time')
            print(f"tau = {self.time_for_steady}, sampling = {self.sampling}, t_end_spec = {self.t_end_spec}")

    def lb_ub_bounds(self):
        self.dummy_optimize()
        lb_ub = {}
        if self.bias is None:
            for key in ['alpha', 'beta', 'no', 'ne', 'na', 'ta']:
                if key in ['alpha', 'beta']:
                    lower = max(0, self.opti_dict[key] - self.perturb[key] * self.opti_dict[key])
                    upper = min(0.99, self.opti_dict[key] + self.perturb[key] * self.opti_dict[key])
                    lb_ub[key] = [lower, upper]
                else:
                    lb_ub[key] = [self.opti_dict[key] - self.perturb[key] * self.opti_dict[key],
                            self.opti_dict[key] + self.perturb[key] * self.opti_dict[key]]
        else:
            print('Using biased values for generating data')
            for key in ['alpha', 'beta', 'no', 'ne', 'na', 'ta']:
                if key in ['alpha', 'beta']:
                    lower = max(0, self.bias[key] - self.perturb[key] * self.bias[key])
                    upper = min(0.99, self.bias[key] + self.perturb[key] * self.bias[key])
                    lb_ub[key] = [lower, upper]
                else:
                    lb_ub[key] = [self.bias[key] - self.perturb[key] * self.bias[key],
                            self.bias[key] + self.perturb[key] * self.bias[key]]
        #lb_ub['ta'] = [self.bounds['T2'][0][0], self.bounds['T2'][0][1]]
        lb_ub['pa'] = [1, 1]
        self.lb_ub = lb_ub
        self.opti_dict['bounds'] = self.lb_ub
        self.opti_dict['physical_bounds'] = {}
        scale_dict = {'alpha': 1, 'beta': 1, 'no': self.N1_clean[5], 'ne': self.N1_clean[0], 'na': self.N1_clean[2], 'ta': self.T20}
        self.opti_dict['physical_bounds'] = {key: [self.lb_ub[key][0] * scale_dict[key],
                                            self.lb_ub[key][1] * scale_dict[key]] for key in self.lb_ub.keys() if key in scale_dict.keys()}
        doe_dict = {}
        for key in ['alpha', 'beta', 'no', 'ne', 'na', 'ta']:
            doe_dict[key] = [self.lb_ub[key][0], self.opti_dict[key], self.lb_ub[key][1]]
        self.doe_dict = doe_dict
        self.safe_initial = self.initial_for_datagen
        
    def digital_input(self, rand_step_idx):
        dict_step_size_change = {}
        change_time = self.change_time
        maintain_ss = self.t_end_spec
        num_of_ss = maintain_ss // change_time
        num_steps = {}
        prev = {}
        rand_val = {}
        for key in self.lb_ub.keys():
            dict_step_size_change[key] = (self.lb_ub[key][1] - self.lb_ub[key][0]) * self.digital_stepping_mag[key] # % of the range
            rand_val[key] = np.random.uniform(self.lb_ub[key][0], self.lb_ub[key][1])
            # get level of changes
            prev[key] = getattr(self, key)
            total_change = prev[key] - rand_val[key]
            # total step inputs
            if dict_step_size_change[key] == 0:
                ns = 0
            else:
                ns = int(abs(total_change) / dict_step_size_change[key])
            num_steps[key] = ns
            # create a list of inputs
        
        self.check_text = num_steps
        max_num_steps = max(num_steps.values())
        step_inputs_for_all_keys = {}
        for key in self.lb_ub.keys():
            if num_steps[key] < max_num_steps:
                num_of_ss_key = num_of_ss + (max_num_steps - num_steps[key])
            else:
                num_of_ss_key = num_of_ss
            step_inputs = np.linspace(prev[key], rand_val[key], num_steps[key])
            step_inputs = np.hstack((step_inputs, np.ones(num_of_ss_key) * rand_val[key]))
            if rand_step_idx == 0:
                step_inputs = np.ones(num_of_ss) * rand_val[key]
                step_inputs_for_all_keys[key] = step_inputs
            else:
                step_inputs_for_all_keys[key] = step_inputs
        self.step_inputs_for_all_keys = step_inputs_for_all_keys

    def prbs_seq(self):
        self.lb_ub_bounds()
        if self.keep_Q_const_trans == 1:
            self.constQ = 0
        # set seed for reproducibility
        np.random.seed(123)
        keys_lag = ['alpha_1', 'alpha_2', 'beta_1', 'beta_2', 'NEe_1', 'NO_1', 'NA_1', 'T_1']
        keys_tank = ['cEe_tanks', 'cEa_tanks', 'cA_tanks', 'cW_tanks', 'cV_tanks', 'cO_tanks', 'cC_tanks']
        keys_avg = ['cEe_avg', 'cEa_avg', 'cA_avg', 'cW_avg', 'cV_avg', 'cO_avg', 'cC_avg']
        keys_rates = ['r1_end', 'r2_end', 'r1_tilde', 'r2_tilde', 'dcdt']
        keys_flow = ['Nv7']
        other_keys = labels['keys'] + keys_tank + keys_rates + keys_avg + keys_flow
        key_list = other_keys if self.valve_lag == 0 else keys_lag + other_keys

        # create the doe df
        df = pd.DataFrame(product(*self.doe_dict.values()), columns=self.doe_dict.keys())
        df['pa'] = 1
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        self.df_doe = df

        data = {key: [] for key in key_list}
        self.validation_start = self.random_jumps - int(self.random_jumps * (1 - self.train_fraction))
        if self.simulate_custom_traj == 1:
            self.random_jumps = len(self.custom_traj['alpha'])
        if self.digital_stepping == 0:
            for i in range(self.random_jumps):
                print('Iter: ', i+1, '\n')
                if i == self.validation_start:
                    data['val_index'] = len(data['time']) 
                attributes = ['alpha', 'beta', 'no', 'ne', 'na', 'ta', 'pa']
                if self.simulate_custom_traj == 0:
                    if self.random_perturb == 1:
                        for attr in attributes:
                            #setattr(self, attr, np.random.uniform(self.lb_ub[attr][0], self.lb_ub[attr][1]))
                            rand_val = self.lb_ub[attr][0] + np.random.beta(a=1, b=1) * (self.lb_ub[attr][1] - self.lb_ub[attr][0])
                            setattr(self, attr, rand_val)
                    else:
                        for attr in attributes:
                            setattr(self, attr, df.iloc[i][attr])
                elif self.simulate_custom_traj == 1:
                    for attr in attributes:
                        if len(self.custom_traj[attr]) != self.random_jumps:
                            raise ValueError(f"Length of trajectory for {attr} must be equal to random_jumps") 
                        setattr(self, attr, self.custom_traj[attr][i])
                else:
                    raise ValueError("simulate_custom_traj must be 0 or 1")

                t_end = (round(np.random.uniform(self.t_end_spec, self.t_end_spec) / self.sampling)
                        * self.sampling + self.t_start)
                
                
                self.t_points = int((t_end - self.t_start) // self.sampling)
                if i == 0:
                    self.time = [self.t_start, t_end]
                    self.teval = np.linspace(self.t_start, self.time[-1] - self.sampling, self.t_points)
                else:
                    self.time = [self.t_start - self.sampling, t_end]
                    self.teval = np.linspace(self.t_start , self.time[-1] - self.sampling, self.t_points)

                if self.valve_lag == 0:
                    if self.delcp == 0:
                        self.prev_ta = self.initial_for_datagen[1][7 * (self.n-1): 7 * (self.n-1) + (self.n-1)].copy()
                        self.initial_for_datagen[1][7 * (self.n-1): 7 * (self.n-1) + (self.n-1)] = self.ta
                        self.initial_for_datagen = list(self.initial_for_datagen)
                        self.initial_for_datagen[1] = np.clip(self.initial_for_datagen[1], a_min=1e-5, a_max=None)

                        # BUG: Q becomes negative since we try to enforce constant pressure mode even in transient operation.
                        # In other words, since both the volume and pressure are fixed, the flowrate becomes negative.
                        # We cannot send in a denser recycle with less dense feed, it is asking for more mass inflow per unit volume 
                        # than possible, unless we increase the pressure or decrease the temperature.
                        # NOTE: This is limitation of our model. The implemented fix changes the temperature to ensure positive flowrate. 

                        _, res = self.initial_for_datagen
                        res = np.reshape(res, (9, self.n - 1))
                        cEe, cEa, cA, cW, cV, cO, cC, T, P = [row for row in res]
                        w = self.weights
                        cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]
                        C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
                        T_inlet = self.T20 * self.ta
                        rho_in = self.P2 / self.R / T_inlet
                        rec = self.phi * (1 - self.mu) * self.alpha + (1 - self.phi) * self.gamma * self.beta
                        diff = rho_in - np.sum(rec * C_in_last_cstr)
                        self.diff_check = diff
                        eps_to_add = 1e-2
                        if diff < 0:
                            print('Resolving with better control input for inlet temperature')
                            self.ta = (self.P2 / self.R / (eps_to_add + np.sum(rec * C_in_last_cstr))) / self.T20
                            # reassign the new temperature to the initial condition to simulate isothermal conditions.
                            self.initial_for_datagen[1][7 * (self.n-1): 7 * (self.n-1) + (self.n-1)] = self.ta
                        self.solve_t(self.initial_for_datagen)
                        self.Trans_Econ()
                        dyn_dict = self.get_dynamic()
                    else:
                        print('Implementation details for maintaining positive Q needs to be studied for non-isothermal case')
                        raise NotImplementedError
                else:
                    print('Warning: Lagged temperature input cannot ensure isothermal operation')
                    print('Implementation details for maintaining positive Q needs to be studied for non-isothermal case')
                    raise NotImplementedError
                    _, res = self.initial_for_datagen
                    res = np.reshape(res[9:], (9, self.n - 1))
                    cEe, cEa, cA, cW, cV, cO, cC, T, P = [row for row in res]
                    w = self.weights
                    cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]
                    T = np.concatenate([np.array([self.T2]), T * w[7]])
                    P = np.concatenate([np.array([self.P2]), P * w[8]])
                    C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
                    T_inlet = self.T20 * self.ta if self.rho_var else self.T_const_inlet
                    t_factor = T[-1]*P[0]/ P[-1]/T[0] if self.const_pressure else 1
                    rho_in = self.P2 / self.R / T_inlet
                    rec = self.phi * (1 - self.mu) * self.alpha + (1 - self.phi) * self.gamma * self.beta
                    diff = rho_in - t_factor * np.sum(rec * C_in_last_cstr)
                    eps_to_add = 1e-2
                    if diff < 0:
                        print('Resolving with better control input for inlet temperature')
                        self.ta = (self.P2 / self.R / (eps_to_add + np.sum(rec * C_in_last_cstr))) / self.T20
                        # reassign the new temperature to the initial condition to simulate isothermal conditions.
                        self.initial_for_datagen[1][9 + 7 * (self.n-1): 9 + 7 * (self.n-1) + (self.n-1)] = self.ta
                    self.solve_t_lag(self.initial_for_datagen)
                    self.Trans_Econ_lag()
                    dyn_dict = self.get_dynamic_lag()

                
                for key in key_list:
                    if key == 'N1f' or key in keys_tank:
                        if np.array(data[key]).size == 0:
                            data[key] = np.stack(dyn_dict[key])
                        elif key == 'N1f':
                            data[key] = np.concatenate((data[key], np.stack(dyn_dict[key])), axis=0)
                        elif key in keys_tank:
                            data[key] = np.concatenate((data[key], np.stack(dyn_dict[key])), axis=1)
                        else:
                            ValueError(f"Key {key} not recognized for data storage")
                    elif key == 'dcdt':
                        # dcdt is 3-D (7, n-1, T_seg); concatenate along the time axis
                        # to preserve its shape instead of flattening it.
                        if np.array(data[key]).size == 0:
                            data[key] = np.asarray(dyn_dict[key])
                        else:
                            data[key] = np.concatenate((data[key], np.asarray(dyn_dict[key])), axis=2)
                    else:
                        data[key].extend(np.ravel(dyn_dict[key]))

                if self.valve_lag == 0:
                    self.initial_for_datagen = self.init_for_dynamics
                else:
                    self.initial_for_datagen = self.init_for_lag_dynamics

                self.t_start = t_end

        elif self.digital_stepping == 1:
            for j in range(self.random_jumps):
                print('Iter: ', j+1, '\n')
                if j == self.validation_start:
                    data['val_index'] = len(data['time']) 
                attributes = ['alpha', 'beta', 'no', 'ne', 'na', 'ta', 'pa']
                self.digital_input(j)
                for i in range(len(self.step_inputs_for_all_keys['alpha'])):
                    for attr in attributes:
                        setattr(self, attr, self.step_inputs_for_all_keys[attr][i])

                    t_end = (round(np.random.uniform(self.change_time, self.change_time) / self.sampling)
                            * self.sampling + self.t_start)
                    
                    
                    self.t_points = int((t_end - self.t_start) // self.sampling)
                    if i == 0:
                        self.time = [self.t_start, t_end]
                        self.teval = np.linspace(self.t_start, self.time[-1] - self.sampling, self.t_points)
                    else:
                        self.time = [self.t_start - self.sampling, t_end]
                        self.teval = np.linspace(self.t_start , self.time[-1] - self.sampling, self.t_points)

                    if self.valve_lag == 0:
                        if self.delcp == 0:
                            self.prev_ta = self.initial_for_datagen[1][7 * (self.n-1): 7 * (self.n-1) + (self.n-1)].copy()
                            self.initial_for_datagen[1][7 * (self.n-1): 7 * (self.n-1) + (self.n-1)] = self.ta
                            self.initial_for_datagen = list(self.initial_for_datagen)
                            self.initial_for_datagen[1] = np.clip(self.initial_for_datagen[1], a_min=1e-5, a_max=None)

                            # BUG: Q becomes negative since we try to enforce constant pressure mode even in transient operation.
                            # In other words, since both the volume and pressure are fixed, the flowrate becomes negative.
                            # We cannot send in a denser recycle with less dense feed, it is asking for more mass inflow per unit volume 
                            # than possible, unless we increase the pressure or decrease the temperature.
                            # NOTE: This is limitation of our model. The implemented fix changes the temperature to ensure positive flowrate. 

                            _, res = self.initial_for_datagen
                            res = np.reshape(res, (9, self.n - 1))
                            cEe, cEa, cA, cW, cV, cO, cC, T, P = [row for row in res]
                            w = self.weights
                            cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]
                            C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
                            T_inlet = self.T20 * self.ta
                            rho_in = self.P2 / self.R / T_inlet
                            rec = self.phi * (1 - self.mu) * self.alpha + (1 - self.phi) * self.gamma * self.beta
                            diff = rho_in - np.sum(rec * C_in_last_cstr)
                            self.diff_check = diff
                            eps_to_add = 1e-2
                            if diff < 0:
                                print('Resolving with better control input for inlet temperature')
                                self.ta = (self.P2 / self.R / (eps_to_add + np.sum(rec * C_in_last_cstr))) / self.T20
                                # reassign the new temperature to the initial condition to simulate isothermal conditions.
                                self.initial_for_datagen[1][7 * (self.n-1): 7 * (self.n-1) + (self.n-1)] = self.ta
                            self.solve_t(self.initial_for_datagen)
                            self.Trans_Econ()
                            dyn_dict = self.get_dynamic()
                        else:
                            print('Implementation details for maintaining positive Q needs to be studied for non-isothermal case')
                            raise NotImplementedError
                    else:
                        print('Warning: Lagged temperature input cannot ensure isothermal operation')
                        print('Implementation details for maintaining positive Q needs to be studied for non-isothermal case')
                        raise NotImplementedError
                        _, res = self.initial_for_datagen
                        res = np.reshape(res[9:], (9, self.n - 1))
                        cEe, cEa, cA, cW, cV, cO, cC, T, P = [row for row in res]
                        w = self.weights
                        cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]
                        T = np.concatenate([np.array([self.T2]), T * w[7]])
                        P = np.concatenate([np.array([self.P2]), P * w[8]])
                        C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
                        T_inlet = self.T20 * self.ta if self.rho_var else self.T_const_inlet
                        t_factor = T[-1]*P[0]/ P[-1]/T[0] if self.const_pressure else 1
                        rho_in = self.P2 / self.R / T_inlet
                        rec = self.phi * (1 - self.mu) * self.alpha + (1 - self.phi) * self.gamma * self.beta
                        diff = rho_in - t_factor * np.sum(rec * C_in_last_cstr)
                        eps_to_add = 1e-2
                        if diff < 0:
                            print('Resolving with better control input for inlet temperature')
                            self.ta = (self.P2 / self.R / (eps_to_add + np.sum(rec * C_in_last_cstr))) / self.T20
                            # reassign the new temperature to the initial condition to simulate isothermal conditions.
                            self.initial_for_datagen[1][9 + 7 * (self.n-1): 9 + 7 * (self.n-1) + (self.n-1)] = self.ta
                        self.solve_t_lag(self.initial_for_datagen)
                        self.Trans_Econ_lag()
                        dyn_dict = self.get_dynamic_lag()

                    
                    for key in key_list:
                        if key == 'N1f' or key in keys_tank:
                            if np.array(data[key]).size == 0:
                                data[key] = np.stack(dyn_dict[key])
                            elif key == 'N1f':
                                data[key] = np.concatenate((data[key], np.stack(dyn_dict[key])), axis=0)
                            elif key in keys_tank:
                                data[key] = np.concatenate((data[key], np.stack(dyn_dict[key])), axis=1)
                            else:
                                ValueError(f"Key {key} not recognized for data storage")
                        elif key == 'dcdt':
                            # dcdt is 3-D (7, n-1, T_seg); concatenate along the time axis
                            # to preserve its shape instead of flattening it.
                            if np.array(data[key]).size == 0:
                                data[key] = np.asarray(dyn_dict[key])
                            else:
                                data[key] = np.concatenate((data[key], np.asarray(dyn_dict[key])), axis=2)
                        else:
                            data[key].extend(np.ravel(dyn_dict[key]))

                    if self.valve_lag == 0:
                        self.initial_for_datagen = self.init_for_dynamics
                    else:
                        self.initial_for_datagen = self.init_for_lag_dynamics

                    self.t_start = t_end
        else:
            raise ValueError("digital_stepping must be 0 or 1")
        
        for key, value in data.items():
            data[key] = np.array(value)
        self.store_data = data
        return self.store_data


    def save_data(self, filename, save=True):
        print(f"\nGenerating data for {self.random_jumps} jumps with following parameters:")
        tic = time.time()
        self.prbs_seq()
        toc = time.time()
        print(f"\nData generation took {(toc - tic)/60:.2f} min \n")
        eps = 1e-5
        if self.delcp == 0:
            self.store_data['clean_meas'] = np.vstack((self.store_data['cEe'], self.store_data['cEa'],
                                                    self.store_data['cA'], self.store_data['cW'],
                                                    self.store_data['cV'], self.store_data['cO'],
                                                    self.store_data['cC'])).T
            self.store_data['T_end_clean'] = self.store_data['T_end']
            self.store_data['meas'] = self.store_data['clean_meas'] + (self.store_data['clean_meas'].std(0)
                        * np.random.normal(self.mean_noise, self.std_noise, self.store_data['clean_meas'].shape))
        
        if self.delcp != 0:
            self.store_data['clean_meas'] = np.vstack((self.store_data['cEe'], self.store_data['cEa'],
                                                    self.store_data['cA'], self.store_data['cW'],
                                                    self.store_data['cV'], self.store_data['cO'],
                                                    self.store_data['cC'], self.store_data['T_end'])).T
            self.store_data['T_end_clean'] = self.store_data['T_end']
            

            self.store_data['meas'] = self.store_data['clean_meas'] + (self.store_data['clean_meas'].std(0)
                        * np.random.normal(self.mean_noise, self.std_noise, self.store_data['clean_meas'].shape))
            self.store_data['T_end'] = self.store_data['meas'][:, 7]
        # ensure we pass no negative values
        self.store_data['meas'] = np.clip(self.store_data['meas'], a_min=eps, a_max=None)
        self.store_data['cEe'] = self.store_data['meas'][:, 0]
        self.store_data['cEa'] = self.store_data['meas'][:, 1]
        self.store_data['cA'] = self.store_data['meas'][:, 2]
        self.store_data['cW'] = self.store_data['meas'][:, 3]
        self.store_data['cV'] = self.store_data['meas'][:, 4]
        self.store_data['cO'] = self.store_data['meas'][:, 5]
        self.store_data['cC'] = self.store_data['meas'][:, 6]
        self.store_data['NEa'] = self.store_data['N1f'][:, 1] 
        self.store_data['NW'] = self.store_data['N1f'][:, 3] 
        self.store_data['NV'] = self.store_data['N1f'][:, 4]
        self.store_data['NC'] = self.store_data['N1f'][:, 6]
        self.store_data['sampling'] = self.sampling
        self.store_data['t_end'] = self.store_data['time'][-1]
        self.store_data['train_jumps'] = self.validation_start
        self.store_data['opti_conc'] = self.opti_dict
        self.store_data['time_for_steady'] = self.time_for_steady
        self.store_data['sampling'] = self.sampling
        self.store_data['t_end_spec'] = self.t_end_spec
        get_max_min = ['cA', 'cEe', 'cEa', 'cW', 'cV', 'cO', 'cC', 'T_end']
        self.store_data['max_min'] = {key: (np.max(self.store_data[key]), np.min(self.store_data[key])) for key in get_max_min}

        if save:
            PickleTool.save(self.store_data, filename)
            print(f"\nData saved to {filename}")
        return self.store_data

    def visualize(self, name='dummy', save=False, show=True, plot_opt=True):
        plt.ion()
        plt.figure(figsize=(15, 4))
        lag_dict = {'alpha': 'alpha_2', 'beta': 'beta_2', 'NEe': 'NEe_1', 'NO': 'NO_1', 'NA': 'NA_1', 'T2': 'T_1'}
        for i, key in enumerate(labels['keys'][:-6]):
            plt.subplot(2, 8, i + 1)
            if key in ['alpha', 'beta', 'NO', 'NEe', 'NA', 'T2']:
                plt.plot(self.store_data['time'], self.store_data[key],'b', label='Meas')
            else:
                plt.plot(self.store_data['time'], self.store_data[key],'b', ms = 1, label='Meas')
            if key in lag_dict and self.valve_lag == 1:
                plt.plot(self.store_data['time'], self.store_data[lag_dict[key]], 'g', label='Lagged Meas')

            # plot clean measurements as well (say if you add bias and want to compare)
            if i in [0, 1, 2, 3, 4, 5, 6, 7]:
                if i < 7:
                    plt.plot(self.store_data['time'], self.store_data['clean_meas'][:, i],'r--', label='Corr')
                else:
                    plt.plot(self.store_data['time'], self.store_data['T_end_clean'],'r--', label='Corr')
            plt.ylabel(get_label(key, labels))

            # add legend of meas and corr distinction
            if i == 6:
                plt.legend()

            if i >= 7:
                plt.xlabel(r'$t$')
            
            # add horizontal line for optimal concentration and control inputs
            if plot_opt:
                if key in ['alpha', 'beta', 'NO', 'NEe', 'NA', 'cEe', 'cEa', 'cA', 'cW', 'cV', 'cO', 'cC', 'T2']:
                    plt.axhline(y=self.store_data['opti_conc'][key], color='k', linestyle='--')
                    plt.axhline(y=np.mean(self.store_data[key]), color='g', linestyle='--')
        plt.tight_layout()
        if save:
            plt.savefig(f'{name}.pdf')
        if show:
            plt.show()
        return plt
        
# test use               

# Datagen = Data_generator()
# Datagen.constQ = 0
# Datagen.valve_lag = 0
# Datagen.feed_lag = 100
# Datagen.alpha_lag = 200
# Datagen.beta_lag = 200
# Datagen.n = 2
# #Datagen.penalty = [0.1, 0.5, 0.5, 0.1, 0.5, 4, 0, 0, 0]
# Datagen.penalty = [1, 1.2, 1, 0.8, 1.20, 8.5, 0, 0, 0]
# bounds = {}
# bounds['alpha'] = [(0., 0.99)]
# bounds['beta'] = [(0., 0.99)]
# bounds['NO'] = [(0.1, 1.5)]
# bounds['NEe'] = [(0.1, 1.5)]
# bounds['NA'] = [(0.1, 1.5)]
# bounds['T2'] = [(1, 1)]
# bounds['P2'] = [(1, 1)]
# Datagen.bounds = bounds
# #Datagen.bias = {'alpha': 0.82, 'beta': 0.82, 'no': 0.35, 'ne': 1, 'na': 0.85}

# Datagen.delcp = 0
# Datagen.heatfact = 0
# Datagen.factfor2 = 1
# Datagen.U = 0
# Datagen.prod_shut = 0.
# Datagen.sampling = 1
# Datagen.t_start = 0
# Datagen.t_end_spec = 5000
# Datagen.random_jumps = 2
# Datagen.std_noise = 0
# Datagen.mean_noise = 0.0
# Datagen.train_fraction = 0.7
# Datagen.perturb = 0.20

# store_data_no_lag = Datagen.save_data('dummy', save=False)

# Datagen = Data_generator()
# Datagen.constQ = 0
# Datagen.valve_lag = 1
# Datagen.feed_lag = 600
# Datagen.alpha_lag = 600
# Datagen.beta_lag = 600
# Datagen.n = 2
# #Datagen.penalty = [0.1, 0.5, 0.5, 0.1, 0.5, 4, 0, 0, 0]
# Datagen.penalty = [1, 1.2, 1, 0.8, 1.20, 8.5, 0, 0, 0]
# bounds = {}
# bounds['alpha'] = [(0., 0.99)]
# bounds['beta'] = [(0., 0.99)]
# bounds['NO'] = [(0.1, 1.5)]
# bounds['NEe'] = [(0.1, 1.5)]
# bounds['NA'] = [(0.1, 1.5)]
# bounds['T2'] = [(1, 1)]
# bounds['P2'] = [(1, 1)]
# Datagen.bounds = bounds
# #Datagen.bias = {'alpha': 0.82, 'beta': 0.82, 'no': 0.35, 'ne': 1, 'na': 0.85}

# Datagen.delcp = 0
# Datagen.heatfact = 0
# Datagen.factfor2 = 1
# Datagen.cat_tau = 2e10
# Datagen.U = 0
# Datagen.prod_shut = 0.
# Datagen.sampling = 1
# Datagen.t_start = 0
# Datagen.t_end_spec = 5000
# Datagen.random_jumps = 2
# Datagen.std_noise = 0
# Datagen.mean_noise = 0.0
# Datagen.train_fraction = 0.7
# Datagen.perturb = 0.20

# store_data_lag = Datagen.save_data('dummy', save=False)

# key_dict = {}
# key_dict['alpha'] = 'alpha_2'
# key_dict['beta'] = 'beta_2'
# key_dict['NO'] = 'NO_1'
# key_dict['NEe'] = 'NEe_1'
# key_dict['NA'] = 'NA_1'


# plt.ion()
# plt.figure(figsize=(15, 4))
# for i, key in enumerate(labels['keys'][:-4]):
#     plt.subplot(2, int(len(labels['keys'][:-3]) / 2), i + 1)
#     if key in ['cEe', 'cEa', 'cA', 'cW', 'cV', 'cO', 'cC']:
#         plt.plot(store_data_lag['time'], store_data_lag[key],'r', label=get_label(key, labels))
#         plt.plot(store_data_no_lag['time'], store_data_no_lag[key],'g--', label=get_label(key, labels))
#     else:
#         if key in ['alpha','beta','NEe', 'NO', 'NA']:
#             plt.plot(store_data_lag['time'], store_data_lag[key_dict[key]],'r--', label=get_label(key, labels))
#             plt.plot(store_data_no_lag['time'], store_data_no_lag[key],'g', label=get_label(key, labels))
#         else:
#             plt.plot(store_data_lag['time'], store_data_lag[key],'r--', label=get_label(key, labels))
#             plt.plot(store_data_no_lag['time'], store_data_no_lag[key],'g', label=get_label(key, labels))
#     plt.ylabel(get_label(key, labels))
#     if i >= 7:
#         plt.xlabel(r'$t$')
#     #if key in ['alpha', 'beta', 'NO', 'NEe', 'NA', 'cEe', 'cEa', 'cA', 'cW', 'cV', 'cO', 'cC']:
#         #    plt.axhline(y=store_data_lag['opti_conc'][key], color='k', linestyle='--')
# plt.tight_layout()
# plt.savefig('VAc_datagen_lag_no_lag.pdf')
# #plt.show()