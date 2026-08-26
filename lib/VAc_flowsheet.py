# Vinyl acetate monomer (VAM) reactor model
# Author: Prithvi Dake
# Helps to solve steady-state, dynamics and optimization problem for VAM

import numpy as np
from tabulate import tabulate
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import null_space
import casadi
import sys
import os
from jax import config
config.update("jax_enable_x64", True)

class Simulator:
    def __init__(self, config=None):
        '''
        Configuration dictionary for reactor model parameters, dynamic and 
        optimization simulation settings.

        Multiplication/control factors:
        - heatfact: A number between 0 and 1 used to control the heat of reaction
          (default 1 corresponds to the nominal heat of reaction).
        - no: A multiple that controls the oxygen feed rate (no = 1 corresponds
          to 831 mol/min).
        - ne: A multiple that controls the ethylene feed rate (ne = 1 corresponds
          to 785 mol/min).
        - na: A multiple that controls the acetic acid feed rate (na = 1 
          corresponds to 831 mol/min).
        - ta: A multiple that controls the inlet temperature of the reactor 
          (ta = 1 corresponds to 150°C).
        - pa: A multiple that controls the inlet pressure of the reactor 
          (pa = 1 corresponds to 882529 Pa).
        - delcp: A control switch to enable/disable the use of Kirchhoff's law 
          (Delcp = 1 corresponds to using Kirchhoff's law).
        - param: A control switch to select between two sets of rate equations 
          (param = 'true' corresponds to the first set of rate equations).
        - constQ: A control switch to enable/disable the use of constant flow 
          rates (constQ = 1 disables constant flowrate).
        - factfor2: A control switch to enable/disable the second reaction 
          (factfor2 = 1 corresponds to enabling the second reaction).
        - penalty: A vector containing the penalties on the control variables 
          (default [1,1,1,0.7,0.7,5,0,0,0])
        - weights: A vector containing convenience scaling factor for the 
          concentration variables. 
          These are completely arbitrary and found to make the problem easier to
          solve using fsolve. (default [10, 10, 10, 10, 10, 10, 10])
        - prod_shut: A control switch to disable production constraint during
          optimization. (default kept on)
        - rate_param_rec1: A hyperparameter that scales output of NN rate law
          (default 0.5e-3).
        - rate_param_rec2: A hyperparameter that scales output of NN rate law
          (default 0.5e-3).
        - cat_decay: A scalar parameter that notes how efficient the catalyst is.
          (default 1), applies only to first reaction (thus, to create an impression
          that the desired reaction worsens over time).
        
        Flowsheet variables:
        - N1f: A numpy array of length 7 containing the nominal fresh feed 
          rates of the 7 species in the reactor.
        - P2: The nominal inlet pressure of the reactor (default 882.529 kPa).
        - T2: The nominal inlet temperature of the reactor (default 150°C).
        - n: The total number of tanks plus 1 in series in the reactor 
          (default 10, i.e., 9 tanks in series).
        - phi: Gas-liquid separation factor (default [1, 1, 0, 0, 0, 1, 1]).
        - gamma: Distillation separation factor (default [0, 0, 1, 0, 0, 0, 0]).
        - beta: Fraction of the distillate that is recycled (default 0).
        - mu: Fraction of CO2 removed (default [0, 0, 0, 0, 0, 0, 0.03]).
        - alpha: Fraction of the gas that is recycled (default 0).
        - guess: vector of initial guesses for the steay-state solver.
        - vars: number of variables to be solved (default 9 * (n - 1)).
        - p_spec: initial guess for the optimization problem.

        Reactor property variables:
        - R: The ideal gas constant (default 8.314 J/mol/K).
        - mol: A numpy array of length 7 containing the molecular weights 
          of the 7 species in the reactor.
        - Dl: Deprecated provision for dispersion coefficient (default 0 m^2/s).
        - f: Constant for the correlation used in the pressure drop calculation
          (default 0 Pa·s^2/kg/m^2).
          (fitted values : 24821136 Pa·s^2/kg/m^2).
        - e: Catalyst porosity (default 0.8).
        - ccat: Catalyst heat capacity (default 962.32 J/kg/K).
        - pcat: The nominal catalyst density (default 385 kg/m^3).

        Specific heat capacity for gas cp = a + bT:
        - spec_a: A numpy array of length 7 containing the coefficients 'a' 
          for the specific heat capacity of the 7 species.
        - spec_b: A numpy array of length 7 containing the coefficients 'b' 
          for the specific heat capacity of the 7 species.

        Reactor specifications:
        - delH1: Heat of reaction for the first reaction (default 176.2 kJ/mol).
        - delH2: Heat of reaction for the second reaction (default 1323 kJ/mol).
        - T0: Reference temperature (default 25°C).
        - Tc: BFW temperature (default 133°C).
        - U: Heat transfer coefficient (default 174.25 W/m^2/K).
        - dt: Diameter of the reactor (default 3.7 cm).
        - nt: Total number of tubes in the reactor (default 622 tubes).
        - At: Calculated cross-sectional area of the reactor.
        - l: Length of the reactor (default 10 m).
        - length: Calculated convenience vector for the length of the reactor 
          (default corresponds to the number of tanks in series).

        Dynamic models parameters:
        - time: A list containing the start and end time of the simulation 
          (default [0, 1]).
        - t_points: Number of time points in the simulation (default 10).
        - t_eval: Time points at which the solution is evaluated 
          (default np.linspace(0, 1, 10)).

        Rate laws:
        - 1: Denotes the true rate laws.
        - 'expfit': Denotes the rate laws fitted to an LH rate law.
        - 'nnfit': Denotes the rate laws fitted to a neural network rate law.
           (Need to pass function object for this option)
        - nn_rec1 : A list containing the weights and biases of the NN for rec 1.
        - nn_rec2 : A list containing the weights and biases of the NN for rec 2
        - mean: Mean of the training data.
        - std: Standard deviation of the training data.
        '''
        if config is None:
            config = {}
        # Multiplication/control factors
        self.heatfact = config.get('heatfact', 1)
        self.no = config.get('no', 0.6)
        self.ne = config.get('ne', 1)
        self.na = config.get('na', 1)
        self.ta = config.get('ta', 1)
        self.pa = config.get('pa', 1)
        self.param = config.get('param', 'true')
        self.constQ = config.get('constQ', 1)
        self.delcp = config.get('delcp', 1)
        self.factfor2 = config.get('factfor2', 1)
        self.penalty = config.get('penalty', [0.03 , 0.17, 0.17, 0.01, 0.1, 0.0, 0.2, 3, 0., 0., 0, 3e-3, 0])
        self.weights = config.get('weights', [10, 10, 10, 10, 10, 10, 10, 150 
                                              + 273.15, 882529])
        self.prod_shut = config.get('prod_shut', 0)
        self.o2_reg = config.get('o2_reg', 100)
        self.Tmax_cons = config.get('Tmax_cons', 1)
        self.Tout_cons = config.get('Tout_cons', 1)
        self.Trec_cons = config.get('Trec_cons', casadi.inf)
        self.Mrsum = config.get('Mrsum', casadi.inf)
        self.rate_param_rec1 = config.get('rate_param', 0.5e-3)
        self.rate_param_rec2 = config.get('rate_param', 0.5e-3)
        self.cat_decay = config.get('cat_decay', 1)
        self.arr_rxn1 = config.get('arr_rxn1', 1)
        self.arr_rxn2 = config.get('arr_rxn2', 1)
        self.rho_var = config.get('rho_var', 1)
        self.const_pressure = config.get('const_pressure', 0)
        self.heat_control = config.get('heat_control', 0)
        self.solve_transient_using_exp = config.get('solve_transient_using_exp', 0)

        # Flowsheet variables
        self.N1_clean = config.get('N10', np.array(
            [831 / 60, 0.001 * 831 / 60, 785 / 60, 0.0 / 60, 0.0 / 60, 831 / 60,
             0.0 / 60]))
        self.T_const_inlet = config.get('T_const_inlet', 150 + 273.15)
        self.P20 = config.get('P2',  882529)
        self.T20 = config.get('T2',  (150 + 273.15))
        self.n = config.get('n', 2)
        self.phi = config.get('phi', np.array([1., 1., 0., 0., 0., 1., 1.]))
        self.gamma = config.get('gamma', np.array([0., 0., 1., 0., 0., 0., 0.]))
        self.beta = config.get('beta', 0.5)
        self.mu = config.get('mu', np.array([0., 0., 0., 0., 0., 0., 0.03]))
        self.alpha = config.get('alpha', 0.5)
        self.nrec = config.get('nrec', 1)
        self.guess = config.get('guess', np.ones(9 * (self.n - 1)))
        self.vars = 9 * (self.n - 1)
        self.p_spec = config.get('p_spec', [0, 0, 1, 1, 1.5, 1.5, 1])
        self.lagrangian = config.get('lagrangian', 0)
        self.bias_r2 = config.get('bias_r2', 0)

        # Reactor property variables
        self.R = config.get('R', 8.314)
        self.mol = config.get('mol', np.array([28.0, 30.0, 60.0, 18.0, 
                                               86.0, 32.0, 44.0]))
        self.Dl = config.get('Dl', 0)
        #self.f = config.get('f', 24821136)
        self.f = config.get('f', 0)
        self.e = config.get('e', 0.8)
        self.ccat = config.get('ccat', 962.32)
        self.pcat = config.get('pcat', 385.0)

        # Specific heat capacity for gas cp = a + bT
        self.spec_a = config.get('spec_a', np.array([43.424, 46.520, 130.649,
                                             42.223, 104.458, 29.187, 42.352]))
        self.spec_b = config.get('spec_b', np.array([0.08215284, 0.08801044,
                         0.17587444, 0.12063309, 0.21612034, 0.0133888, 0]))

        # Reactor specifications
        self.delH10 = config.get('delH1', 176.2 * 10 ** 3) # J/mol
        self.delH20 = config.get('delH2', 1323 * 10 ** 3) # J/mol
        self.T0 = config.get('T0', 25 + 273.15)
        self.Tc = config.get('Tc', 133 + 273.15)
        self.U = config.get('U', 174.25)
        self.dt = config.get('dt', 3.7e-2)
        self.nt = config.get('nt', 622.0)
        self.l = config.get('l', 10.0)

        # Heat of formations at standard conditions (J/mol)
        self.Hf_Ee = config.get('Hf_Ee', 52.47 * 10 ** 3)
        self.Hf_Ea = config.get('Hf_Ea', -84 * 10 ** 3)
        self.Hf_A = config.get('Hf_A', -433 * 10 ** 3)
        self.Hf_W = config.get('Hf_W', -241.83 * 10 ** 3)
        self.Hf_V = config.get('Hf_V', -313.6 * 10 ** 3)
        self.Hf_O = config.get('Hf_O', 0)
        self.Hf_C = config.get('Hf_C', -393.51 * 10 ** 3)
        
        # Dynamic models parameters
        self.time = config.get('t_int', [0, 1])
        self.t_points = config.get('t_points', 10)

        # Rate laws 
        self.nn_rec1 = config.get('nn_rec1', None)
        self.nn_rec2 = config.get('nn_rec2', None)
        self.mean = config.get('mean', None)
        self.std = config.get('std', None)    
        self.mean_ceeco = config.get('mean_ceeco', None)
        self.std_ceeco = config.get('std_ceeco', None)

        # Cbox
        self.nn_bbox = config.get('nn_bbox', None)
        self.mean_u =  config.get('mean_u', None)
        self.std_u = config.get('std_u', None)
        self.solve_by_opt = config.get('solve_by_opt', 0)
        # high-level control
        self.print_level = config.get('print_level', 0)

        rate_param_list = ['true', 'fake', 'expfit', 'expfit_ver2', 'test']
        self.rate_param_list = config.get('rate_param_list', rate_param_list)

    def initialize(self):
        """
        Recalcuates variables based on the configuration parameters. 
        Called internally in most of the functions.

        Attributes initialized:
        - N1f: Adjusted feed rates of species in the reactor based on 
          `self.N1_clean` and scaling factors (`ne`, `na`, `no`).
        - T2: Inlet temperature of the reactor, scaled by the temperature 
          factor (`ta`).
        - P2: Inlet pressure of the reactor, scaled by the pressure factor (`pa`).
        - delH1, delH2: Heats of reaction, scaled by `heatfact`.
        - At: Cross-sectional area of the reactor, calculated from diameter (`dt`).
        - length: Vector of length positions along the reactor for discretization.
        - teval: Evaluation time points for the simulation based on the total 
          simulation time (`time`) and points (`t_points`).
        - guess: Initial guess vector for solving the steady-state equations.
        - vars: Number of variables to solve in the steady-state model, 
          dependent on the number of reactor stages (`n`).
        """
        self.N1f = self.N1_clean.copy()
        self.N1f[0] = self.N1_clean[0] * self.ne
        self.N1f[1] = self.N1_clean[1] * self.ne
        self.N1f[2] = self.N1_clean[2] * self.na
        self.N1f[5] = self.N1_clean[5] * self.no
        self.T2 = self.T20 * self.ta
        self.P2 = self.P20 * self.pa
        self.delH1 = self.delH10 * self.heatfact
        self.delH2 = self.delH20 * self.heatfact
        self.At = 0.25 * np.pi * self.dt ** 2
        self.length = np.linspace(0, self.l, self.n)
        self.guess = np.ones(9 * (self.n - 1))
        self.vars = 9 * (self.n - 1)
        self.null_mass = null_space(self.mol.reshape(-1,1).T)
        # parameters for the different rate laws
        if self.theta1 is None or self.theta2 is None:
            if self.param == 'true':
                self.theta1 = np.array([0.7, 15, 0.35, 0.20, 1])
                self.theta2 = np.array([5.1, 21, 4, 10, 0.85])
            elif self.param == 'fake':
                self.theta1 = np.array([0.7, 15, 0.35, 0.38, 1])
                self.theta2 = np.array([5.1, 21, 0.33, 0.35])
            elif self.param == 'expfit':
                self.theta1 = np.array([0.7, 15, 0.35, 0.38, 1])
                self.theta2 = np.array([5.1, 21, 0.33, 0.35])
            elif self.param == 'expfit_ver2':
                self.theta1 = np.array([0.7, 15, 0.35, 0.20, 1])
                self.theta2 = np.array([5.1, 21, 4, 10, 0.85])

    def jaxnumpy(self, params, input):
        """
        params: dict, Flax parameters, assumed structure:
            {'params': {'Dense_0': {'kernel': ..., 'bias': ...}, ...}}
        input: np.ndarray, shape (features,) or (batch, features)
        Returns:
            y: np.ndarray, network output
        """
        layers = params['params']
        num_layers = len(layers)
        dense_keys = sorted(layers.keys(), key=lambda s: int(s.split('_')[-1]))  # Ensure correct order

        y = input
        for i, k in enumerate(dense_keys):
            # NOTE: Ensure weights and biases are float64 in the numpy version. 
            # Otherwise, leads to unreasonable failures.
            W = layers[k]['kernel'].astype(np.float64)
            b = layers[k]['bias'].astype(np.float64)
            y = ((y.T @ W) + b).T
            if i < num_layers - 1:
                y = np.tanh(y)
        return y

    def jaxadi(self, params, input):
        """
        params: dict, Flax parameters, assumed structure:
            {'params': {'Dense_0': {'kernel': ..., 'bias': ...}, ...}}
        x: CasADi SX or MX symbolic variable (input vector)
        Returns:
            y: CasADi symbolic expression representing NN output
        """
        layers = params['params']
        num_layers = len(layers)
        dense_keys = sorted(layers.keys(), key=lambda s: int(s.split('_')[-1]))  # Ensure correct order

        y = input
        for i, k in enumerate(dense_keys):
            W = np.array(layers[k]['kernel'])
            b = np.array(layers[k]['bias'])
            y = casadi.mtimes(y.T, W).T + b
            # Add tanh activation after every layer except the last
            if i < num_layers - 1:
                y = casadi.tanh(y)
        return y

    def Rate(self, c):
        """
        Calculates reaction rates r1 and r2.
        The rate law used is determined by the `self.param` attribute, 
        which can specify one of three models:
        - `param = 1` for the base rate law.
        - `param = 'expfit'` for an empirically fitted rate law.
        - `param = 'nnfit'` for a neural network fitted rate law.

        Parameters
        ----------
        c : array-like
            Concentration array, where `c[:-1]` represents the concentrations 
            of the species, 
            and `c[-1]` is the temperature in Kelvin.

        Returns
        -------
        tuple of floats
            The reaction rates `(r1, r2)` for the two reactions, calculated 
            based on the selected model.

        Notes
        -----
        - For `param = 1`, uses true rate law (ref: Goodman. 2005).
        - For `param = 'expfit'`,
          uses Langmuir-Hinshelwood rate law fitted to the true rate law.
        - For `param = 'nnfit'`, `r1` is calculated using a neural network model
          (`Rate_NN`), scaled by `self.rate_param`. In this case, `r2` is set to
          zero.

        Additional Details
        -------------------
        - The parameter `self.factfor2` Ons/Offs `r2` in all cases.
        - When `param = 'nnfit'`, the input concentrations are normalized using 
          the `self.mean` and `self.std` attributes.
        - To complex numbers, a lower bound of `1e-10` is applied.

        """
        T = c[-1]
        if self.rho_var:
            T_to_be_used = T
        else:
            T_to_be_used = self.T_const_inlet
        pressures = [self.R * T_to_be_used * i / 1000 for i in c[:-1]]
        pEe, pEa, pA, pW, pV, pO, pC = pressures
        cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
        
        if self.param == 'true':
            theta1 = self.theta1
            theta2 = self.theta2
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * np.exp(- self.arr_rxn1 * theta1[1] * 1e3 / (self.R * T))
                    * np.maximum(pEe, 1e-10) ** theta1[2]
                    * np.maximum(pO, 1e-10) ** theta1[3]
                    * np.maximum(pA, 1e-10) ** theta1[4]
                )
            # negative power index term from Goodman et al.
            # NOTE: See there was an impending danger of r2 blowing up due to 
            # small pEe (and negative index). So it only makes sense to set it 
            # to a small constant value (1) when it really tries to blow up i.e.
            # ethylene amount is really low, still not the best solution
            # but beware if you see negative concentrations.
            term1 = np.clip(np.maximum(pEe, 1e-10) ** -0.31, 0, 1)
            # NOTE: Discarded the Goodman term and using following approximation
            subterm = np.maximum(pEe, 1e-10)
            term2 = np.exp(-0.001 * theta2[2] * subterm) - np.exp(-theta2[3] * subterm)
            term3 = np.maximum(pEe, 1e-10) ** 0.95
            r2 = (
                    self.factfor2 * theta2[0] * 1e-3
                    * np.exp(-self.arr_rxn2 * theta2[1] * 1e3 / (self.R * T))
                    * term2
                    * np.maximum(pO, 1e-10) ** theta2[4]
                )
            r2 = r2 + self.bias_r2
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1
            self.check_Ee = pEe
            self.arr_term1 = np.exp(- self.arr_rxn1 * 15e3 / (self.R * T))
            self.arr_term2 = np.exp(- self.arr_rxn2 * 21e3 / (self.R * T))
            self.k2 = 5.1e-3 * np.exp(- self.arr_rxn2 * 21e3 / (self.R * T)) 
            self.k1 = 0.7e-4 * np.exp(- self.arr_rxn1 * 15e3 / (self.R * T))

        elif self.param == 'fake':
            theta1 = self.theta1
            theta2 = self.theta2
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * np.exp(- self.arr_rxn1 * theta1[1] * 1e3 / (self.R * T))
                    * np.maximum(pEe, 1e-10) ** theta1[2]
                    * np.maximum(pO, 1e-10) ** theta1[3]
                    * np.maximum(pA, 1e-10) ** theta1[4]
                )
            # negative power index term from Goodman et al.
            # NOTE: See there was an impending danger of r2 blowing up due to 
            # small pEe (and negative index). So it only makes sense to set it 
            # to a small constant value (1) when it really tries to blow up i.e.
            # ethylene amount is really low, still not the best solution
            # but beware if you see negative concentrations.
            term1 = np.clip(np.maximum(pEe, 1e-10) ** -0.31, 0, 1)
            # NOTE: Discarded the Goodman term and using following approximation
            subterm = np.maximum(pEe, 1e-10)
            term2 = np.exp(-0.004 * subterm) - np.exp(-10 * subterm)
            term3 = np.maximum(pEe, 1e-10) ** theta2[2]
            r2 = (
                    self.factfor2 * theta2[0] * 1e-3
                    * np.exp(-self.arr_rxn2 * theta2[1] * 1e3 / (self.R * T))
                    * term3
                    * np.maximum(pO, 1e-10) ** theta2[3]
                )
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1
            self.rate1 = r1
            self.rate2 = r2
            self.pEe = pEe


        elif self.param == 'test':
            theta1 = self.theta1
            theta2 = self.theta2
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * np.exp(- self.arr_rxn1 * theta1[1] * 1e3 / self.R * (1/T - 1/450))
                    * np.maximum(cEe, 1e-10) ** theta1[2]
                    * np.maximum(cO, 1e-10) ** theta1[3]
                    * np.maximum(cA, 1e-10) ** theta1[4]
                )
            # negative power index term from Goodman et al.
            # NOTE: See there was an impending danger of r2 blowing up due to 
            # small pEe (and negative index). So it only makes sense to set it 
            # to a small constant value (1) when it really tries to blow up i.e.
            # ethylene amount is really low, still not the best solution
            # but beware if you see negative concentrations.
            term1 = np.clip(np.maximum(cEe, 1e-10) ** -0.31, 0, 1)
            # NOTE: Discarded the Goodman term and using following approximation
            subterm = np.maximum(cEe, 1e-10)
            term2 = np.exp(-0.004 * subterm) - np.exp(-10 * subterm)
            term3 = np.maximum(cEe, 1e-10) ** theta2[2]
            k2 = self.factfor2 * self.cat_decay * theta2[0] * 1e-4  
            r2 = (
                    k2
                    * np.exp(-self.arr_rxn2 * theta2[1] * 1e3 / self.R * (1/T - 1/450))
                    * term3
                    * np.maximum(cO, 1e-10) ** theta2[3]
                )
            r2 = r2 + self.bias_r2
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1 
            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'lh':
            theta1 = self.theta1
            theta2 = self.theta2
            lnk1 = self.theta1[0]
            lnk2 = self.theta2[0]
            E1 = self.theta1[1]
            E2 = self.theta2[1]
            lnKO = self.theta1[2]
            KE = self.theta1[3]
            Tm = self.theta1[4]

        elif self.param == 'expfit':
            theta1 = self.theta1
            theta2 = self.theta2
            #theta1 = [0.943, 13.274, 0.313, 0.334, 0.916]
            #theta2 = [5.118, 21.02, 0.33, 0.35]
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * np.exp(- self.arr_rxn1 * 1e3 * theta1[1] / (self.R * T))
                    * np.maximum(pEe, 1e-10) ** theta1[2]
                    * np.maximum(pO, 1e-10) ** theta1[3]
                    * np.maximum(pA, 1e-10) ** theta1[4]
                )
            subterm = np.maximum(pEe, 1e-10)
            term2 = np.exp(-4.137e-3* subterm) - np.exp(-0.458 * subterm)
            term3 = np.maximum(pEe, 1e-10) ** theta2[2]
            r2 = (
                    self.factfor2 * 1e-3 * theta2[0]
                    * np.exp(-self.arr_rxn2 * 1e3 * theta2[1] / (self.R * T))
                    * term3
                    * np.maximum(pO, 1e-10) ** theta2[3]
                )
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1
            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'expfit_ver2':
            theta1 = self.theta1
            theta2 = self.theta2
            #theta1 = [0.943, 13.274, 0.313, 0.334, 0.916]
            #theta2 = [5.118, 21.02, 0.33, 0.35]
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * np.exp(- self.arr_rxn1 * 1e3 * theta1[1] / (self.R * T))
                    * np.maximum(pEe, 1e-10) ** theta1[2]
                    * np.maximum(pO, 1e-10) ** theta1[3]
                    * np.maximum(pA, 1e-10) ** theta1[4]
                )
            subterm = np.maximum(pEe, 1e-10)
            term2 = np.exp(-theta2[2] * 1e-3 * subterm) - np.exp(-theta2[3] * subterm)
            term3 = np.maximum(pEe, 1e-10) ** theta2[2]
            r2 = (
                    self.factfor2 * 1e-3 * theta2[0]
                    * np.exp(-self.arr_rxn2 * 1e3 * theta2[1] / (self.R * T))
                    * term2
                    * np.maximum(pO, 1e-10) ** theta2[4]
                )
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1
            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'stoic_base': 
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            # input1 = np.array([cEe, cA, cO, T])
           
            # input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]] 
            # input2 = np.array([cEe, cO, T])      
            # input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            # r1 = (10 ** self.mult_r[0]) * self.jaxnumpy(self.nn_rec1, input1).squeeze(axis=0) + (10 ** self.mult_r[2])
            # r2 = (10 ** self.mult_r[1]) * self.factfor2 * self.jaxnumpy(self.nn_rec2, input2).squeeze(axis=0) + (10 ** self.mult_r[3])

            input1 = np.array([cEe, cA, cO, T])
            input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]]
            input2 = np.array([cEe, cO, T])
            input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            mean_r = self.mult_r[:, 0].reshape(-1, 1)
            std_r = self.mult_r[:, 1].flatten()
            mean_r = 10 ** mean_r
            std_r = np.diag(10 ** std_r)
            r1 = self.jaxnumpy(self.nn_rec1, input1)
            r2 = self.jaxnumpy(self.nn_rec2, input2)
            r = np.vstack([r1, r2])
            #print(r.shape)
            # denormalizing z * std + mean
            r =   std_r @ r + mean_r
            vT = np.array([[-1, -1],
                [0, 0],
                [-1, 0], 
                [1, 2],
                [1, 0],
                [-0.5, -3],
                [0, 2]], dtype=np.float64)
            
            r = vT @ r

            rEe = r[0]
            rEa = r[1]
            rA = r[2]
            rW = r[3]
            rV = r[4]
            rO = r[5]
            rC = r[6]
            rheat = 0 * r
            r1 = rV
            r2 = rC / 2

            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'stoic_red_base': 
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            # input1 = np.array([cEe, cA, cO, T])
           
            # input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]] 
            # input2 = np.array([cEe, cO, T])      
            # input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            # r1 = (10 ** self.mult_r[0]) * self.jaxnumpy(self.nn_rec1, input1).squeeze(axis=0) + (10 ** self.mult_r[2])
            # r2 = (10 ** self.mult_r[1]) * self.factfor2 * self.jaxnumpy(self.nn_rec2, input2).squeeze(axis=0) + (10 ** self.mult_r[3])

            input1 = np.array([cEe * cO, cA, T])
            mean_ceeco = np.array(self.mean_ceeco).reshape(-1)
            std_ceeco = np.array(self.std_ceeco).reshape(-1)
            input1 = (input1 - np.array([mean_ceeco, self.mean[2], self.mean[7]])) \
                            / np.array([std_ceeco,  self.std[2],  self.std[7]])
            input2 = np.array([cEe * cO, T])
            input2 = (input2 - np.array([mean_ceeco, self.mean[7]])) \
                            / np.array([std_ceeco,  self.std[7]])
            mean_r = self.mult_r[:, 0].reshape(-1, 1)
            std_r = self.mult_r[:, 1].flatten()
            mean_r = 10 ** mean_r
            std_r = np.diag(10 ** std_r)
            r1 = self.jaxnumpy(self.nn_rec1, input1)
            r2 = self.jaxnumpy(self.nn_rec2, input2)
            r = np.vstack([r1, r2])
            #print(r.shape)
            # denormalizing z * std + mean
            r =   std_r @ r + mean_r
            vT = np.array([[-1, -1],
                [0, 0],
                [-1, 0], 
                [1, 2],
                [1, 0],
                [-0.5, -3],
                [0, 2]], dtype=np.float64)
            
            r = vT @ r

            rEe = r[0]
            rEa = r[1]
            rA = r[2]
            rW = r[3]
            rV = r[4]
            rO = r[5]
            rC = r[6]
            rheat = 0 * r
            r1 = rV
            r2 = rC / 2

            self.rate1 = r1
            self.rate2 = r2
            
        elif self.param == 'stoic': 
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            # input1 = np.array([cEe, cA, cO, T])
           
            # input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]] 
            # input2 = np.array([cEe, cO, T])      
            # input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            # r1 = (10 ** self.mult_r[0]) * self.jaxnumpy(self.nn_rec1, input1).squeeze(axis=0) + (10 ** self.mult_r[2])
            # r2 = (10 ** self.mult_r[1]) * self.factfor2 * self.jaxnumpy(self.nn_rec2, input2).squeeze(axis=0) + (10 ** self.mult_r[3])

            input1 = np.array([cEe, cEa, cA, cW, cV, cO, cC, T])
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0].reshape(-1, 1)
            std_r = self.mult_r[:, 1].flatten()
            mean_r = 10 ** mean_r
            std_r = np.diag(10 ** std_r)
            # denormalizing z * std + mean
            r =   std_r @ self.jaxnumpy(self.nn_net, input1)  + mean_r
            vT = np.array([[-1, -1],
                [0, 0],
                [-1, 0], 
                [1, 2],
                [1, 0],
                [-0.5, -3],
                [0, 2]], dtype=np.float64)
            
            r = vT @ r

            rEe = r[0]
            rEa = r[1]
            rA = r[2]
            rW = r[3]
            rV = r[4]
            rO = r[5]
            rC = r[6]
            rheat = 0 * r
            r1 = rV
            r2 = rC / 2

            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'prod':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = np.array([cEe, cEa, cA, cW, cV, cO, cC, T])
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0].reshape(-1, 1)
            std_r = self.mult_r[:, 1].flatten()
            mean_r = 10 ** mean_r
            std_r = np.diag(10 ** std_r)
            r1 =   std_r @ self.jaxnumpy(self.nn_net, input1)  + mean_r
            rEe = r1[0]
            rEa = r1[1]
            rA = r1[2]
            rW = r1[3]
            rV = r1[4]
            rO = r1[5]
            rC = r1[6]
            rheat = 0 * r1
            r1 = rV
            r2 = rC / 2
            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'prod_null':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = np.array([cEe, cEa, cA, cW, cV, cO, cC, T])
            input1 = (input1 - self.mean) / self.std 
            r1 = self.mult_r * (self.jaxnumpy(self.nn_net, input1) * self.std[:7] + self.mean[:7])
            W = np.diag(1/self.mult_r/self.std[:7].flatten())
            W_inv = np.linalg.inv(W)
            A = (W_inv @ self.mol).reshape(-1, 1)
            B = np.dot(self.mol, W_inv @ self.mol)
            C = np.dot(self.mol, r1)
            D = (C / B).reshape(1, -1)
            delta_r = A @ D

            r1 = r1- delta_r
            rEe = r1[0]
            rEa = r1[1]
            rA = r1[2]
            rW = r1[3]
            rV = r1[4]
            rO = r1[5]
            rC = r1[6]
            rheat = 0 * r1
            r1 = rV
            r2 = rC / 2
            self.rate1 = r1
            self.rate2 = r2
        
        elif self.param == 'prod_e':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = np.array([cEe, cEa, cA, cW, cV, cO, cC, T])
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0].reshape(-1, 1)
            std_r = self.mult_r[:, 1].flatten()
            mean_r = 10 ** mean_r
            std_r = np.diag(10 ** std_r)
            r1 =   std_r @ self.jaxnumpy(self.nn_net, input1)  + mean_r
            vT = np.array([[1.5, -2.0, 1.0, -0.5],
                   [-1., 1., .0, 1.0],
                   [-0.5, -1., -1., -1.],
                   [1.0, 0., 0., 0.],
                   [0.0, 1.0, 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
            r1 = vT @ r1
            rEe = r1[0]
            rEa = r1[1]
            rA = r1[2]
            rW = r1[3]
            rV = r1[4]
            rO = r1[5]
            rC = r1[6]
            rheat = 0 * r1
            r1 = rV
            r2 = rC / 2
            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'prod_m':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = np.array([cEe, cEa, cA, cW, cV, cO, cC, T])
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0].reshape(-1, 1)
            std_r = self.mult_r[:, 1].flatten()
            mean_r = 10 ** mean_r
            std_r = np.diag(10 ** std_r)
            r1 =   std_r @ self.jaxnumpy(self.nn_net, input1)  + mean_r
            vT = self.null_mass
            r1 = vT @ r1
            rEe = r1[0]
            rEa = r1[1]
            rA = r1[2]
            rW = r1[3]
            rV = r1[4]
            rO = r1[5]
            rC = r1[6]
            rheat = 0 * r1
            r1 = rV
            r2 = rC / 2
            self.rate1 = r1
            self.rate2 = r2

        elif self.param == 'stoic_notT': 
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = np.array([cEe, cA, cO])
           
            input1 = (input1 - self.mean[[0,2,5]]) / self.std[[0,2,5]] 
            input2 = np.array([cEe, cO])      
            input2 = (input2 - self.mean[[0,5]]) / self.std[[0,5]]
            r1 = 0.5e-3 * self.jaxnumpy(self.nn_rec1, input1).squeeze(axis=0)
            r2 = 0.5e-3 * self.factfor2 * self.jaxnumpy(self.nn_rec2, input2).squeeze(axis=0)
                       
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'prod_notT':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = np.array([cEe, cEa, cA, cW, cV, cO, cC])
            input1 = (input1 - self.mean) / self.std 
            r1 = 0.5e-4 * (self.jaxnumpy(self.nn_net, input1) * self.std + self.mean)
            rEe = r1[0]
            rEa = r1[1]
            rA = r1[2]
            rW = r1[3]
            rV = r1[4]
            rO = r1[5]
            rC = r1[6]
            rheat = 0 * r1[0]

        elif self.param == 'stoic_delay':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
            # get the shape of the conc array  
            shape_for_copy = cEe.shape[-1]

            # form the conc array
            c = np.array([cEe, cEa, cA, cW, cV, cO, cC])
            c_norm = ((c - self.mean[:,None]) / self.std[:, None]).T
            
            # form the u array
            u = np.array([self.N1f[0], self.N1f[2], self.N1f[5], self.alpha, self.beta, self.ta * self.T20])
            u_norm = (u - self.mean_u)/self.std_u
            u_norm = np.tile(u_norm[:, np.newaxis], (1, shape_for_copy)).T
            
            # form the delay embedding (at steady-state just copy of current c's and u's)
            c_z_norm = np.tile(c_norm, reps=self.n_past) 
            u_z_norm = np.tile(u_norm, reps=self.n_past) 

            # form the current conc, u and delay embedding (already normalized)
            c_u_z = np.hstack((c_norm, u_norm, c_z_norm, u_z_norm))
            std_r = np.array([self.std_r1, self.std_r2])
            mean_r = np.array([self.mean_r1, self.mean_r2])
            rate = self.jaxnumpy(self.nn_net, c_u_z.T)
            r1, r2 = rate * std_r[:,None] + mean_r[:,None]
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'stoic_exp':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = np.array([cEe, cA, cO, T])
           
            input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]] 
            input2 = np.array([cEe, cO, T])      
            input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            r1 = self.std_r1 * self.jaxnumpy(self.nn_rec1, input1).squeeze(axis=0)
            r2 = self.std_r2 * self.factfor2 * self.jaxnumpy(self.nn_rec2, input2).squeeze(axis=0)

            r1 = r1 + self.mean_r1
            r2 = r2 + self.mean_r2
            r1 = np.exp(- self.arr_rxn1 * 15e3 / (self.R * T)) * r1
            r2 = np.exp(- self.arr_rxn2 * 21e3 / (self.R * T)) * r2
                       
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        else:
            raise ValueError(f"Unknown param value: {self.param}")

        return rEe, rEa, rA, rW, rV, rO, rC, rheat

    def fsolve_steady(self, x):
        """
        Prepares residues for the steady-state solver to compute the 
        solution of algebraic equations.

        Parameters
        ----------
        x : array-like
            Array of values representing the initial guess for the steady-state
            variables.

        Returns
        -------
        numpy.ndarray
            Array of residuals for each equation in the system. 

        Notes
        -----
        - Concentrations are initially weighted and adjusted for the 'n' tanks. 
        - The recycle factor, `self.rec_factor`, determines the fraction of 
          species returned to the mixer, influencing feed concentrations.
        - Flow rates `Q` are computed iteratively across the tanks to ensure
          mass conservation.

        Residuals
        ---------
        - `resEe`, `resEa`, `resA`, `resW`, `resV`, `resO`, `resC`: concentrations
        - `resP`: pressure
        - `resT`: temperature
        """
        N10 = self.N1f / self.nt
        x_mat = np.reshape(x, (9, self.n - 1))
        cEe, cEa, cA, cW, cV, cO, cC, T, P = x_mat

        # Weights for the concentration variables
        w = self.weights
        cEe, cEa, cA, cW, cV, cO, cC = (cEe * w[0], cEa / w[1], cA * w[2], 
                                 cW * w[3], cV * w[4], cO * w[5], cC * w[6])

        # Weights for temp and press are initial values themselves
        T = np.concatenate([np.array([self.T2]), T * w[7]])
        P = np.concatenate([np.array([self.P2]), P * w[8]])
        
        c = (cEe, cEa, cA, cW, cV, cO, cC, T[1:])
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate(c)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC
      
        # rec_factor calculates the fraction of all species that go back to the mixer
        self.rec_factor = (self.phi * (1 - self.mu) * self.alpha +
                   (1 - self.phi) * self.gamma * self.beta)
        
        T_inlet = self.T2 if self.rho_var else self.T_const_inlet
        rho_in = self.P2 / self.R / T_inlet

        t_factor = T[-1]*P[0]/ P[-1]/T[0] if self.const_pressure else 1
        t_fac_array = np.ones(self.n - 1)

        self.t_factor = t_factor
        self.t_fac_array = t_fac_array
        for i in range(self.n - 2):
            t_fac_array[i + 1] = t_fac_array[i] * T[-1 - i] * P[-1 - (i  + 1)] / T[-1 - (i + 1)] / P[-1 - i]
        t_fac_array = np.flip(t_fac_array) if self.const_pressure else np.ones(self.n - 1)

        # We calculate the conc and volumetric flow rate in last tank to avoid DAEs.
        C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
        Qf = (np.sum(N10) + self.constQ * self.pcat * self.R* (self.At * self.l / (self.n-1)) * np.sum(self.rec_factor * C_in_last_cstr) * 
              np.sum(T[1:] * sum_Rj * t_fac_array / P[1:])) / (rho_in - t_factor * np.sum(self.rec_factor * C_in_last_cstr)) 
        Q_in_last_cstr = Qf * t_factor + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * np.sum(T[1:] / P[1:] * sum_Rj * t_fac_array)  
        # Construct the actual molar flow entering the reactor
        N2 = N10 + Q_in_last_cstr * C_in_last_cstr * self.rec_factor

        # Construct the initial concentrations 
        # (Note: This trick allows not to calculate the initial concentrations 
        # in the reactor which we don't know due to recycle)
        Q_for_first_cstr = np.sum(N2) / rho_in
        cEe_0, cEa_0, cA_0, cW_0, cV_0, cO_0, cC_0 = N2 / Q_for_first_cstr

        # Now we have all concentrations in all the tanks
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
                Q[i + 1] = Q[i] * T[i+1] * P[i]/T[i]/P[i+1] + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * T[i+1] / P[i+1] * sum_Rj[i+1]
        else:
            for i in range(len(Q) - 1):
                Q[i + 1] = Q[i] + self.constQ * self.pcat  * self.R * (self.At * self.l / (self.n-1)) * T[i+1] / P[i+1] * sum_Rj[i+1]
        
        # Algebraic equations for ODEs using tanks-in-series
        resEe = ((self.n-1) / self.At / self.l) * (Q[:-1] * cEe[:-1] - Q[1:] * cEe[1:]) + self.pcat * rEe[1:]
        resEa = ((self.n-1) / self.At / self.l) * (Q[:-1] * cEa[:-1] - Q[1:] * cEa[1:]) + self.pcat * rEa[1:]
        resA = ((self.n-1) / self.At / self.l) * (Q[:-1] * cA[:-1] - Q[1:] * cA[1:]) + self.pcat * rA[1:]
        resW = ((self.n-1) / self.At / self.l) * (Q[:-1] * cW[:-1] - Q[1:] * cW[1:]) + self.pcat * rW[1:]
        resV = ((self.n-1) / self.At / self.l) * (Q[:-1] * cV[:-1] - Q[1:] * cV[1:]) + self.pcat * rV[1:]
        resO = ((self.n-1) / self.At / self.l) * (Q[:-1] * cO[:-1] - Q[1:] * cO[1:]) + self.pcat * rO[1:]
        resC = ((self.n-1) / self.At / self.l) * (Q[:-1] * cC[:-1] - Q[1:] * cC[1:]) + self.pcat * rC[1:]

        resP = P[:-1] - P[1:] - self.l / (self.n-1) * self.f * rhoav[1:] * Q[1:] ** 2
        #resP = P[1:] - cTotal[1:] * self.R * T[1:]

        spec = np.dot(np.reshape((T - 273.15) ** 2 - (self.T0 - 273.15) ** 2, (self.n, 1)),
                      np.reshape(self.spec_b / 2, (1, 7))) + np.dot(np.reshape((T - self.T0), (self.n, 1)),
                                                                    np.reshape(self.spec_a, (1, 7)))

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
        self.sum_RjHj = sum_RjHj

        # NOTE: In case you want to compare with the old version
        self.sum_RjHj_ver = (r1[1:] * delH1n[1:] + r2[1:] * delH2n[1:] + rheat[1:] * self.delH2) 

        self.delH1n = delH1n
        self.delH2n = delH2n
        self.heat_added = self.pcat * (r1[1:] * delH1n[1:] + r2[1:] * delH2n[1:] + rheat[1:] * self.delH2)
        self.heat_removed = self.U  * (T[1:] - self.Tc) * np.pi * self.dt / self.At
        self.enthalpy = (((self.n-1) / self.At / self.l) *
                    ((speb[:, 0] * Q[:-1] * cEe[:-1]) +
                    (speb[:, 1] * Q[:-1]  * cEa[:-1]) +
                    (speb[:, 2] * Q[:-1]  * cA[:-1]) +
                    (speb[:, 3] * Q[:-1]  * cW[:-1]) +
                    (speb[:, 4] * Q[:-1]  * cV[:-1]) +
                    (speb[:, 5] * Q[:-1]  * cO[:-1]) +
                    (speb[:, 6] * Q[:-1]  * cC[:-1])))

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
                    self.U  * (T[1:] - self.Tc) * np.pi * self.dt / self.At)
      
        self.res = np.concatenate((resEe, resEa, resA, resW, resV, resO, resC, resT, resP))
        return self.res

    def Plant_steady(self, guess):
        '''
        Solves the steady-state model for the plant using the `fsolve` function.

        Parameters
        ----------
        guess : array-like
            Initial guess for the steady-state variables. 

        Returns
        -------
        result : numpy.ndarray
            Array containing the steady-state solution values for each variable.
        
        ier : int
            Integer flag indicating the convergence status. 
            A value of 1 signifies successful convergence.
        '''

        #result, info, ier, msg = fsolve(self.fsolve_steady, guess, full_output=True, xtol=1e-8)
        sol = root(fun=self.fsolve_steady, x0=guess, method='hybr', options={'xtol':1e-8})
        result = sol.x
        ier = sol.success
        msg = sol.message
        if self.solve_by_opt:
            if ier == False or np.any(result < -1e-3):
                if ier == False:
                    print(f'Solution by fsolve failed with {msg}, trying optimization')
                else:
                    print('fsolve found negative concentrations, trying optimization')
                eps = 1e-3
                bounds={}
                bounds['alpha'] = [(self.alpha - eps, self.alpha + eps)]
                bounds['beta'] = [(self.beta - eps, self.beta + eps)]
                bounds['NO'] = [(self.ne - eps, self.ne + eps)]
                bounds['NA'] = [(self.na - eps, self.na + eps)]
                bounds['NEe'] = [(self.no - eps, self.no + eps)]
                bounds['T2'] = [(self.ta - eps, self.ta + eps)]
                bounds['P2'] = [(self.pa - eps, self.pa + eps)]
                bounds['x'] = [(1e-6, casadi.inf)] * self.vars
                self.optimize(guess, bounds)
                result = self.sol_x
                print('Solution found by optimization succeeded: ', self.solver.stats()['return_status'])
                if self.solver.stats()['return_status'] == 'Solve_Succeeded':
                    ier = 1
        else:
            print("Convergence message : ", msg, "\n")
        return result, ier

    def get_output_grad(self, guess):
        '''
        Gradient of the reactor-outlet (last-tank) species concentrations with
        respect to the control inputs, evaluated at steady state.

        Controls (u, length 11, physical units):
            u = [alpha, beta, T2 (K), P2 (Pa), N1f(7) (mol/min)]
        Outputs (length 7, physical units):
            outlet concentrations [cEe, cEa, cA, cW, cV, cO, cC] in the last tank.

        The control values are read off the current object state
        (self.alpha, self.beta, self.T2, self.P2, self.N1f), so set those
        (or the ne/na/no/ta/pa multipliers) before calling. The steady state is
        first solved with the usual scipy path (Plant_steady) using `guess`; that
        solution is handed to a CasADi rootfinder as x0, and the sensitivity is
        obtained by differentiating through the rootfinder (CasADi supplies the
        implicit-function-theorem Jacobian). The internal state is solved in the
        scaled (well-conditioned) coordinates, but c_out and u are formed in
        physical units, so the returned Jacobian is in original scales.

        Parameters
        ----------
        guess : array-like
            Initial guess for the steady-state solve (same layout as
            fsolve_steady / Plant_steady).

        Returns
        -------
        numpy.ndarray, shape (7, 11)
            self.output_grad -- Jacobian d(outlet conc)/d(u). Row i is the
            gradient of outlet species i; column j the derivative w.r.t.
            control j. Also returned directly.

        Also sets
        ---------
        self.output_grad_all : numpy.ndarray, shape (7*n, 11)
            d(all tank concs)/d(u), species-major: rows i*n .. i*n+n-1 are the
            n tanks of species i (tank 0 = reactor inlet, tank n-1 = outlet).
        self.output_grad_u : numpy.ndarray, shape (11,)
            The physical control values u at which the gradient was evaluated.
        '''
        self.initialize()
        n = self.n
        nx = 9 * (n - 1)

        # symbolic state (same layout/convention as optimize's x[7:]) and controls
        st = casadi.SX.sym('st', nx)
        u = casadi.SX.sym('u', 11)
        alpha = u[0]
        beta = u[1]
        T2 = u[2]
        P2 = u[3]
        N1f = u[4:11]
        theta = (self.theta1, self.theta2) if self.param in self.rate_param_list else None

        N10 = N1f / self.nt
        x_mat = casadi.reshape(st, n - 1, 9)
        cEe, cEa, cA, cW, cV, cO, cC, T, P = casadi.vertsplit(x_mat.T)

        # Weights for the concentration variables (scaled -> physical)
        w = self.weights
        cEe, cEa, cA, cW, cV, cO, cC = (cEe * w[0], cEa / w[1], cA * w[2],
                                        cW * w[3], cV * w[4], cO * w[5], cC * w[6])

        # Weights for temp and press are initial values themselves
        T = casadi.vertcat(T2, T.T * w[7])
        P = casadi.vertcat(P2, P.T * w[8])

        T_shaped = casadi.reshape(T[1:], -1, 1)
        P_shaped = casadi.reshape(P[1:], -1, 1)
        c = (cEe.T, cEa.T, cA.T, cW.T, cV.T, cO.T, cC.T, T_shaped)
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate_cas(c, alpha, beta, N1f, theta)
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC

        rec_factor = (self.phi * (1 - self.mu) * alpha +
                      (1 - self.phi) * self.gamma * beta)

        T_inlet = T2 if self.rho_var else self.T_const_inlet
        rho_in = P2 / self.R / T_inlet

        t_factor = T[-1] * P[0] / P[-1] / T[0] if self.const_pressure else 1
        t_fac_array = [1.0]
        for i in range(n - 2):
            t_fac_array.append(t_fac_array[i] * T[-1 - i] * P[-1 - (i + 1)] / T[-1 - (i + 1)] / P[-1 - i])
        t_fac_array = casadi.vertcat(*t_fac_array)
        t_fac_array = t_fac_array[::-1] if self.const_pressure else np.ones(n - 1)

        C_in_last_cstr = casadi.vertcat(cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1])
        Qf = (casadi.sum1(N10) + self.constQ * self.pcat * self.R * (self.At * self.l / (n - 1)) * casadi.sum1(rec_factor * C_in_last_cstr) *
              casadi.sum1(T_shaped * sum_Rj * t_fac_array / P_shaped)) / (rho_in - t_factor * casadi.sum1(rec_factor * C_in_last_cstr))
        Q_in_last_cstr = Qf * t_factor + self.constQ * self.pcat * self.R * (self.At * self.l / (n - 1)) * casadi.sum1(T_shaped / P_shaped * sum_Rj * t_fac_array)
        N2 = N10 + Q_in_last_cstr.T * C_in_last_cstr * rec_factor

        Q_for_first_cstr = casadi.sum1(N2) / rho_in
        cEe_0, cEa_0, cA_0, cW_0, cV_0, cO_0, cC_0 = casadi.vertsplit(N2 / Q_for_first_cstr, 1)

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
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate_cas(c, alpha, beta, N1f, theta)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC

        Mav = casadi.mtimes(self.mol.reshape(1, -1), C.T).T / cTotal
        rhoav = P * Mav / self.R / T * 10 ** -3

        Q = [Qf]
        if self.const_pressure:
            for i in range(n - 1):
                Qi_next = Q[i] * T[i + 1] * P[i] / T[i] / P[i + 1] + self.constQ * self.pcat * self.R * (self.At * self.l / (n - 1)) * T[i + 1] / P[i + 1] * sum_Rj[i + 1]
                Q.append(Qi_next)
        else:
            for i in range(n - 1):
                Qi_next = Q[i] + self.constQ * self.pcat * self.R * (self.At * self.l / (n - 1)) * T[i + 1] / P[i + 1] * sum_Rj[i + 1]
                Q.append(Qi_next)
        Q = casadi.vertcat(*Q)

        resEe = ((n - 1) / self.At / self.l) * (Q[:-1] * cEe[:-1] - Q[1:] * cEe[1:]) + self.pcat * rEe[1:]
        resEa = ((n - 1) / self.At / self.l) * (Q[:-1] * cEa[:-1] - Q[1:] * cEa[1:]) + self.pcat * rEa[1:]
        resA = ((n - 1) / self.At / self.l) * (Q[:-1] * cA[:-1] - Q[1:] * cA[1:]) + self.pcat * rA[1:]
        resW = ((n - 1) / self.At / self.l) * (Q[:-1] * cW[:-1] - Q[1:] * cW[1:]) + self.pcat * rW[1:]
        resV = ((n - 1) / self.At / self.l) * (Q[:-1] * cV[:-1] - Q[1:] * cV[1:]) + self.pcat * rV[1:]
        resO = ((n - 1) / self.At / self.l) * (Q[:-1] * cO[:-1] - Q[1:] * cO[1:]) + self.pcat * rO[1:]
        resC = ((n - 1) / self.At / self.l) * (Q[:-1] * cC[:-1] - Q[1:] * cC[1:]) + self.pcat * rC[1:]

        resP = P[:-1] - P[1:] - self.l / (n - 1) * self.f * rhoav[1:] * Q[1:] ** 2
        resP_scaled = resP / self.P20

        delta_T_square = (T - 273.15) ** 2 - (self.T0 - 273.15) ** 2
        delta_T = T - self.T0
        spec = (casadi.mtimes(casadi.reshape(delta_T_square, n, 1), casadi.reshape((self.spec_b / 2), 1, 7)) +
                casadi.mtimes(casadi.reshape(delta_T, n, 1), casadi.reshape(self.spec_a, 1, 7)))
        speb = (casadi.mtimes(casadi.reshape(((T[:-1] - 273.15) ** 2 - (T[1:] - 273.15) ** 2), n - 1, 1),
                casadi.reshape((self.spec_b / 2), 1, 7)) +
                casadi.mtimes(casadi.reshape((T[:-1] - T[1:]), n - 1, 1), casadi.reshape(self.spec_a, 1, 7)))

        Hfn_Ee = self.Hf_Ee + spec[:, 0]
        Hfn_Ea = self.Hf_Ea + spec[:, 1]
        Hfn_A = self.Hf_A + spec[:, 2]
        Hfn_W = self.Hf_W + spec[:, 3]
        Hfn_V = self.Hf_V + spec[:, 4]
        Hfn_O = self.Hf_O + spec[:, 5]
        Hfn_C = self.Hf_C + spec[:, 6]
        sum_RjHj = -(rEe[1:] * Hfn_Ee[1:] + rEa[1:] * Hfn_Ea[1:] + rA[1:] * Hfn_A[1:] + rW[1:] * Hfn_W[1:] + rV[1:] * Hfn_V[1:] + rO[1:] * Hfn_O[1:] + rC[1:] * Hfn_C[1:])

        if self.delcp != 0:
            resT = (((n - 1) / self.At / self.l) *
                    ((speb[:, 0] * Q[:-1] * cEe[:-1]) +
                     (speb[:, 1] * Q[:-1] * cEa[:-1]) +
                     (speb[:, 2] * Q[:-1] * cA[:-1]) +
                     (speb[:, 3] * Q[:-1] * cW[:-1]) +
                     (speb[:, 4] * Q[:-1] * cV[:-1]) +
                     (speb[:, 5] * Q[:-1] * cO[:-1]) +
                     (speb[:, 6] * Q[:-1] * cC[:-1])) +
                    self.pcat * self.delcp * sum_RjHj -
                    self.U * (T[1:] - self.Tc) * np.pi * self.dt / self.At)
        else:
            resT = (T[:-1] - T[1:]) / self.T20

        res = casadi.vertcat(resEe, resEa, resA, resW, resV, resO, resC, resT, resP_scaled)
        # outputs: species concentrations in every tank (physical units), stacked
        # species-major -> [cEe(0..n-1), cEa(0..n-1), ..., cC(0..n-1)], length 7*n.
        # Tank index 0 is the reactor inlet (post-recycle mix); index -1 the outlet.
        c_all = casadi.vertcat(cEe, cEa, cA, cW, cV, cO, cC)

        # robust steady-state solve via the usual scipy path (reads controls from self)
        result, ier = self.Plant_steady(guess)
        u_val = casadi.DM(np.concatenate([[self.alpha, self.beta, self.T2, self.P2], self.N1f]))

        # differentiate through the solve: hand the scipy solution to a CasADi
        # rootfinder as x0 and let CasADi supply the IFT sensitivity.
        rf = casadi.rootfinder('get_output_grad_rf', 'newton', {'x': st, 'p': u, 'g': res})
        Call = casadi.Function('Call', [st, u], [c_all])
        # outer graph must be MX (rootfinder has no eval_sx)
        usym = casadi.MX.sym('u', 11)
        st_sol = rf(x0=casadi.DM(result), p=usym)['x']
        Jfun = casadi.Function('Jfun', [usym], [casadi.jacobian(Call(st_sol, usym), usym)])

        # d(all tank concs)/d(u), shape (7*n, 11), species-major (see c_all layout)
        J_all = np.array(Jfun(u_val))
        # outlet rows: species i lives at rows i*n .. i*n+n-1; outlet is i*n+(n-1)
        outlet_rows = [i * n + (n - 1) for i in range(7)]

        self.output_grad_all = J_all                       # (7*n, 11) all tanks
        self.output_grad = J_all[outlet_rows, :]           # (7, 11)   outlet only
        self.output_grad_u = np.array(u_val).ravel()
        return self.output_grad

    def Stream_table(self, guess=None, just_use_this=False):
        '''
        A convenience function to print the stream table.
        Returns stream table and stores some important variables in class 
        variables for further use.
        '''
        self.initialize()
        if guess is None:
            guess = self.guess
        if just_use_this:
            self.result = guess
        else:
            self.result, self.ier = self.Plant_steady(guess)
        N10 = self.N1f / self.nt
        x_mat = np.reshape(self.result, (9, self.n - 1))
        cEe, cEa, cA, cW, cV, cO, cC, T, P = x_mat

        # Weights for the concentration variables
        w = self.weights
        cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]

        # Weights for temp and press are initial values themselves
        self.T = np.concatenate([np.array([self.T2]), T * w[7]])
        self.P = np.concatenate([np.array([self.P2]), P * w[8]])
      
        c = (cEe, cEa, cA, cW, cV, cO, cC, self.T[1:])
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate(c)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC

        T_inlet = self.T2 if self.rho_var else self.T_const_inlet
        rho_in = self.P2 / self.R / T_inlet

        t_factor = self.T[-1] * self.P[0] / self.T[0] / self.P[-1] if self.const_pressure else 1
        t_fac_array = np.ones(self.n - 1)
        for i in range(self.n - 2):
            t_fac_array[i + 1] = t_fac_array[i] * self.T[-1 - i]* self.P[-1 - (i  + 1)] / self.T[-1 - (i + 1)]/ self.P[-1 - i]
        t_fac_array = np.flip(t_fac_array) if self.const_pressure else np.ones(self.n - 1)
        self.t_fac_array = t_fac_array 

        # We calculate the conc and volumetric flow rate in last tank to avoid DAEs.
        C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
        Qf = (np.sum(N10) + self.constQ * self.pcat * self.R* (self.At * self.l / (self.n-1)) * np.sum(self.rec_factor * C_in_last_cstr) * 
              np.sum(self.T[1:] * sum_Rj * t_fac_array / self.P[1:])) / (rho_in - t_factor * np.sum(self.rec_factor * C_in_last_cstr)) 
        Q_in_last_cstr = Qf * t_factor + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * np.sum(self.T[1:] / self.P[1:] * sum_Rj * t_fac_array)  
        self.Q_in_last_cstr = Q_in_last_cstr
        
        # Construct the actual molar flow entering the reactor
        N2 = N10 + Q_in_last_cstr * C_in_last_cstr * self.rec_factor

        # Construct the initial concentrations 
        # (Note: This trick allows not to calculate the initial concentrations in the reactor which we don't know due to recycle)
        Q_for_first_cstr = np.sum(N2) / rho_in
        self.Q_in_first_cstr = Q_for_first_cstr
        cEe_0, cEa_0, cA_0, cW_0, cV_0, cO_0, cC_0 = N2 / Q_for_first_cstr

        # Now we have all concentrations in all the tanks
        self.cEe = np.concatenate([np.array([cEe_0]), cEe])
        self.cEa = np.concatenate([np.array([cEa_0]), cEa])
        self.cA = np.concatenate([np.array([cA_0]), cA])
        self.cW = np.concatenate([np.array([cW_0]), cW])
        self.cV = np.concatenate([np.array([cV_0]), cV])
        self.cO = np.concatenate([np.array([cO_0]), cO])
        self.cC = np.concatenate([np.array([cC_0]), cC])

        c = (self.cEe, self.cEa, self.cA, self.cW, self.cV, self.cO, self.cC, self.T)
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate(c)
        r1 = rV
        r2 = rC / 2
        self.ratio_of_rates = r2 / r1
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC
        
        Q = np.zeros(self.n)
        Q[0] = Qf
        if self.const_pressure:
            for i in range(len(Q) - 1):
                Q[i + 1] = Q[i] * self.T[i+1]* self.P[i]/self.T[i]/self.P[i+1] + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * self.T[i+1] / self.P[i+1] * sum_Rj[i+1]
        else:
            for i in range(len(Q) - 1):
                Q[i + 1] = Q[i] + self.constQ * self.pcat * self.R * (self.At * self.l / (self.n-1)) * self.T[i+1] / self.P[i+1] * sum_Rj[i+1]

        # Save important variables in class variables
        self.tau = 1 / (Q * ((self.n - 1) / self.l / self.At))
        self.damkohler = -np.sum(r1[1:] * self.pcat * self.tau[1:]) / np.sum(self.cV[:-1] - self.cV[1:])
        self.Qflow = Q
        self.rEe = rEe
        self.rEa = rEa
        self.rA = rA
        self.rW = rW
        self.rV = rV
        self.rO = rO
        self.rC = rC
        

        self.N1 = self.N1f
        self.N2 = N2 * self.nt
        self.NcE = Q * self.cEe * self.nt
        self.NcEt = Q * self.cEa * self.nt
        self.NcA = Q * self.cA * self.nt
        self.NcW = Q * self.cW * self.nt
        self.NcV = Q * self.cV * self.nt
        self.NcO = Q * self.cO * self.nt
        self.NcC = Q * self.cC * self.nt
        self.N3 = Q_in_last_cstr * C_in_last_cstr * self.nt
        self.N4 = (1 - self.phi) * self.N3
        self.N5 = self.phi * self.N3
        self.N6 = (1 - self.mu) * self.N5
        self.N7 = self.alpha * self.N6
        self.N8 = self.gamma * self.N4
        self.N9 = (1 - self.alpha) * self.N6
        self.N10 = (1 - self.gamma) * self.N4
        self.N11 = self.mu * self.N5
        self.N12 = (1 - self.beta) * self.N8
        self.N13 = self.beta * self.N8
        self.Nvec = [self.N1, self.N2, self.N3, self.N4, self.N5, self.N6, self.N7, self.N8, self.N9, self.N10,
                     self.N11, self.N12, self.N13]
        initial_conc = [self.cEe[0], self.cEa[0], self.cA[0], self.cW[0], self.cV[0], self.cO[0], self.cC[0]]
        fsolve_result = self.result
        self.init_for_t0 = (initial_conc, fsolve_result)
        scalars = np.array([self.cat_decay, self.N1f[0], self.N1f[2], self.N1f[5], self.alpha, self.alpha, self.beta, self.beta, self.T2])
        lag_result = np.concatenate([scalars, fsolve_result])
        self.init_for_lag_t0 = (initial_conc, lag_result)

        self.Qcheck = np.dot(self.mol, self.N1f) / np.dot(self.mol, C_in_last_cstr * (1-self.rec_factor)) / self.nt
        # must match Qflow[-1] exactly
        # Mass recycle ratio
        N1calc = self.N1 @ self.mol
        N7calc = self.N7 @ self.mol
        N13calc = self.N13 @ self.mol

        MrEe = self.mol[0] * rEe[1:]
        MrEa = self.mol[1] * rEa[1:]
        MrA = self.mol[2] * rA[1:]
        MrW = self.mol[3] * rW[1:]
        MrV = self.mol[4] * rV[1:]
        MrO = self.mol[5] * rO[1:]
        MrC = self.mol[6] * rC[1:]

        denom = Q[:-1] * (self.mol @ np.array([self.cEe[:-1], self.cEa[:-1], self.cA[:-1], self.cW[:-1], self.cV[:-1], self.cO[:-1], self.cC[:-1]]))
        num = self.pcat * self.At * self.l / (self.n - 1) * (MrEe + MrEa + MrA + MrW + MrV + MrO + MrC)

        self.mR_sum = num / denom * 100
        # O2 con in reactor feed
        self.o2_con = self.N2[5] / np.sum(self.N2)
        # Heat removed from each CSTR summed over
        self.removed = np.sum(self.U * (self.T[1:] - self.Tc) * np.pi * self.dt * self.l / (self.n-1))
        
        component = ['Ee', 'Ea', 'A', 'W', 'V', 'O', 'C']
        np.set_printoptions(suppress=True)

        table_data = []
        table_data = [[f"N{i+1}"] + np.round(self.Nvec[i], 3).tolist() for i in range(len(self.Nvec))]

        print(tabulate(table_data, headers=["Component"] + component, tablefmt="grid"))

        #print("Gas recycle : ", np.sum(N7calc) / np.sum(N1calc))
        #print("Liquid recycle : ", np.sum(N13calc) / np.sum(N1calc))
        print('Gas recycle : ', self.alpha)
        print('Liquid recycle : ', self.beta)
        print("Per-pass conversion : ", (self.N2[0] - self.N3[0]) / self.N2[0] * 100)
        print("Selectivity : ",
              (self.N2[0] - self.N3[0] - 0.5 * self.N3[-1] + 0.5 * self.N2[-1]) / (self.N2[0] - self.N3[0]) * 100)
        print("Pressure drop % : ", (self.P[0] - self.P[-1]) / self.P[0] * 100)
        print("O2 vol % : ", self.o2_con * 100)
        print("Peak rise in temperature : ", np.max(self.T) - self.T[0])
        print("Inlet temperature : ", self.T[0])
        print("Outlet temperature : ", self.T[-1])
        print('Model :', self.param, '\n')

    def Economics(self):
        '''
        Function used to calculate the economics of the flowsheet.
        Returns the profit.
        '''
        feed_econ = self.penalty[0] * self.N1[0] + self.penalty[1] * self.N1[2] + self.penalty[2] * self.N1[5]
        rec_econ = self.penalty[3] * np.sum(self.N7) + self.penalty[5] * np.sum(self.N7)**2 + self.penalty[4] * np.sum(self.N13) + self.penalty[6] * np.sum(self.N13)**2
        prod_econ = self.penalty[7] * self.N10[4]
        purge_gas_econ = self.penalty[8] * np.sum(self.N9)
        purge_liquid_econ = self.penalty[9] * np.sum(self.N12)
        purge_co2_econ = self.penalty[10] * (self.N3[6] - self.N2[6])
        total_econ = prod_econ - rec_econ - feed_econ - purge_gas_econ - purge_liquid_econ - purge_co2_econ - self.penalty[11] * self.removed
        self.feed_econ = feed_econ
        self.rec_econ = rec_econ
        self.rec_econ_a = self.penalty[3] * np.sum(self.N7) + self.penalty[5] * np.sum(self.N7)**2
        self.rec_econ_b = self.penalty[4] * np.sum(self.N13) + self.penalty[6] * np.sum(self.N13)**2
        self.prod_econ = prod_econ
        self.purge_gas_econ = purge_gas_econ
        self.purge_liquid_econ = purge_liquid_econ
        self.purge_co2_econ = purge_co2_econ
        self.cons_vio = -(0.79 * self.prod_shut * 1000/60 - self.N10[4])
        # construct \delu R \delu
        scales = np.array([1, 1, self.T20, self.P20, self.N1_clean[0], self.N1_clean[2], self.N1_clean[5]])
        scales_use = scales[self.thetaind]
        formR = np.array(self.uR_spec) 
        R = np.diag(1/formR[self.thetaind])
        u = np.array([self.alpha, self.beta, self.ta, self.pa, self.ne, self.na, self.no])
        u = u[self.thetaind] 
        u0 = np.array(self.u_spec)[self.thetaind]
        delta_u = u - u0
        penalty_term = delta_u.T @ R @ delta_u
        self.total_econ = total_econ
        total_econ = total_econ - self.penalty[12] * penalty_term
        self.penalty_term = self.penalty[12] * penalty_term
        self.delta_u = penalty_term ** 0.5
        return total_econ

    def Transient(self, t, x, *args):
        '''
        Prepares the RHS of the dynamic model of the flowsheet. 
        Same strategy as steady-state solver to simplify DAE to ODE
        if switch = 0,
        Returns RHS of the ODEs
        if switch != 0
        Returns inlet concentrations for each time step.
        '''
        switch,  = args

        N10 = self.N1f / self.nt
        
        x_mat = np.reshape(x, (8, self.n - 1))

        cEe, cEa, cA, cW, cV, cO, cC, T = [row for row in x_mat]
        # NOTE: solve as log-transformed variables, helps to deal with really small concentrations
        if self.solve_transient_using_exp:
            cEe, cEa, cA, cW, cV, cO, cC, T = np.exp(cEe), np.exp(cEa), np.exp(cA), np.exp(cW), np.exp(cV), np.exp(cO), np.exp(cC), np.exp(T)
            
        else:
            cEe, cEa, cA, cW, cV, cO, cC, T = cEe, cEa, cA, cW, cV, cO, cC, T

        T = np.concatenate([np.array([self.T2]), T])
        P = np.concatenate([np.array([self.P2]), np.ones(self.n-1) * self.P2])

        c = (cEe, cEa, cA, cW, cV, cO, cC, T[1:])
        rEe, rEa, rA, rW, rV, rO, rC, rheat = self.Rate(c)
        r1 = rV
        r2 = rC / 2
        sum_Rj = rEe + rEa + rA + rW + rV + rO + rC

        self.rec_factor = self.phi * (1 - self.mu) * self.alpha + (1 - self.phi) * self.gamma * self.beta
        
        T_inlet = self.T2 if self.rho_var else self.T_const_inlet
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
        if self.solve_transient_using_exp:
            resEe = (((self.n-1) / self.At / self.l) * (Q[:-1] * cEe[:-1] - Q[1:] * cEe[1:]) + self.pcat * rEe[1:]) / self.e / (cEe[1:])
            resEa = (((self.n-1) / self.At / self.l) * (Q[:-1] * cEa[:-1] - Q[1:] * cEa[1:]) + self.pcat * rEa[1:]) / self.e / (cEa[1:])
            resA = (((self.n-1) / self.At / self.l) * (Q[:-1] * cA[:-1] -  Q[1:] * cA[1:]) + self.pcat * rA[1:]) / self.e / (cA[1:])
            resW = (((self.n-1) / self.At / self.l) * (Q[:-1] * cW[:-1] - Q[1:] * cW[1:]) + self.pcat * rW[1:]) / self.e / (cW[1:])
            resV = (((self.n-1) / self.At / self.l) * (Q[:-1] * cV[:-1] - Q[1:] * cV[1:]) + self.pcat * rV[1:]) / self.e / (cV[1:])
            resO = (((self.n-1) / self.At / self.l) * (Q[:-1] * cO[:-1] - Q[1:] * cO[1:]) + self.pcat * rO[1:]) / self.e / (cO[1:])
            resC = (((self.n-1) / self.At / self.l) * (Q[:-1] * cC[:-1] - Q[1:] * cC[1:]) + self.pcat * rC[1:]) / self.e / (cC[1:])
        else:
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

        if self.solve_transient_using_exp:
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
                                                        spe[1:,6] * cC[1:]))) / T[1:]
        else:
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

        self.resTotal = np.concatenate((resEe, resEa, resA, resW, resV, resO, resC, resT))
        resTotal = np.concatenate((resEe, resEa, resA, resW, resV, resO, resC, resT))
        if switch == 0:
            return resTotal

        else:
            return cEe[0], cEa[0], cA[0], cW[0], cV[0], cO[0], cC[0], Q, r1, r2, resTotal
        

    def solve_t(self, tupl):
        '''
        A convenience function that solves the transient model and return a 
        nice formatted set of variables that can be used further. 
        '''
        self.initialize()
        C_at_first_cstr_t0, result = tupl
        x_mat = np.reshape(result, (9, self.n - 1))
        cEe, cEa, cA, cW, cV, cO, cC, T, P = [row for row in x_mat]

        # Weights for the concentration variables
        w = self.weights
        cEe, cEa, cA, cW, cV, cO, cC = cEe * w[0], cEa / w[1], cA * w[2], cW * w[3], cV * w[4], cO * w[5], cC * w[6]

        # NOTE: solve as log-transformed variables, helps to deal with really small concentrations
        if self.solve_transient_using_exp:
            self.initial = np.log(np.concatenate([cEe, cEa, cA, cW, cV, cO, cC, w[7] * T]))
        else:
            self.initial = np.concatenate([cEe, cEa, cA, cW, cV, cO, cC, w[7] * T])
        
        self.ode_sol = solve_ivp(self.Transient, self.time, self.initial, method = "BDF", t_eval = self.teval, args = (0, ), rtol = 1e-8, atol = 1e-8)
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
        self.r1 = np.zeros((self.n, len(self.teval)))
        self.r2 = np.zeros((self.n, len(self.teval)))
        self.resTotal_t = np.zeros((8 * (self.n - 1), len(self.teval)))

        for i in range(len(self.teval)):
            cEe_0, cEa_0, cA_0, cW_0, cV_0, cO_0, cC_0, Q, r1, r2, resTotal = self.Transient(self.teval[i], self.ode_sol.y.T[i], (1,))
            self.cEe0t[:,i] = cEe_0
            self.cEa0t[:,i] = cEa_0
            self.cA0t[:,i] = cA_0
            self.cW0t[:,i] = cW_0
            self.cV0t[:,i] = cV_0
            self.cO0t[:,i] = cO_0
            self.cC0t[:,i] = cC_0
            self.Qa[:,i] = Q
            self.r1[:,i] = r1
            self.r2[:,i] = r2
            self.resTotal_t[:,i] = resTotal

        # Split the total RHS into concentration (first 7 components) and temperature (last) parts
        self.resTotal_c = self.resTotal_t[:7 * (self.n - 1), :]
        self.resTotal_T = self.resTotal_t[7 * (self.n - 1):, :]

        rEe = -self.r1 - self.r2
        rEa = 0 * self.r1 + 0 * self.r2
        rA = -self.r1
        rW = self.r1 + 2 * self.r2
        rV = self.r1
        rO = -0.5 * self.r1 - 3 * self.r2
        rC = 2 * self.r2

        MrEe = self.mol[0] * rEe[1:, :]
        MrEa = self.mol[1] * rEa[1:, :]
        MrA = self.mol[2] * rA[1:, :]
        MrW = self.mol[3] * rW[1:, :]
        MrV = self.mol[4] * rV[1:, :]
        MrO = self.mol[5] * rO[1:, :]
        MrC = self.mol[6] * rC[1:, :]

        self.mR_denom = np.sqrt(np.mean(MrEe**2 + MrEa**2 + MrA**2 + MrW**2 + MrV**2 + MrO**2 + MrC**2, axis=0))
        self.mR_num = (MrEe + MrEa + MrA + MrW + MrV + MrO + MrC)
        self.mR_sum = self.mR_num / np.clip(self.mR_denom, 1e-5, None)

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
        # NOTE: solve as log-transformed variables, helps to deal with really small concentrations
        if self.solve_transient_using_exp:
            COL = [np.exp(self.ode_sol.y[i * (self.n - 1):(i + 1) * (self.n - 1), :]) for i in range(8)]
        else:
            COL = [self.ode_sol.y[i * (self.n - 1):(i + 1) * (self.n - 1), :] for i in range(8)]

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
        self.T_tall = np.vstack((self.T2 * np.ones((1, len(self.teval))), self.T_t))
    
    def Trans_Econ(self):
        '''
        One more convenience function to calculate the molar flowrates for end time step. 
        It can also be use to calculate the end profit of the transient model.
        Return the profit.
        '''
        total_econ_tall = np.zeros(len(self.teval))
        for i in range(len(self.teval)):
            cEe = self.cEe_tall[:,i]
            cEa = self.cEa_tall[:,i]
            cA = self.cA_tall[:,i]
            cW = self.cW_tall[:,i]
            cV = self.cV_tall[:,i]
            cO = self.cO_tall[:,i]
            cC = self.cC_tall[:,i]
            T = self.T_tall[:,i]
            P = self.P2 * np.ones((1, len(T)))

            C_in_last_cstr = np.array([cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1]])
            N2 = self.N1f / self.nt + self.Qa[-1,i] * C_in_last_cstr * self.rec_factor

            self.N1_t = self.N1f
            self.N2_t = N2 * self.nt
            self.NcE_t = self.Qa[:,i] * self.cEe_tall[:,i] * self.nt
            self.NcEt_t = self.Qa[:,i] * self.cEa_tall[:,i] * self.nt
            self.NcA_t = self.Qa[:,i] * self.cA_tall[:,i] * self.nt
            self.NcW_t = self.Qa[:,i] * self.cW_tall[:,i] * self.nt
            self.NcV_t = self.Qa[:,i] * self.cV_tall[:,i] * self.nt
            self.NcO_t = self.Qa[:,i] * self.cO_tall[:,i] * self.nt
            self.NcC_t = self.Qa[:,i] * self.cC_tall[:,i] * self.nt
            self.N3_t = self.Qa[-1,i] * C_in_last_cstr * self.nt
            self.N4_t = (1 - self.phi) * self.N3_t
            self.N5_t = self.phi * self.N3_t
            self.N6_t = (1 - self.mu) * self.N5_t
            self.N7_t = self.alpha * self.N6_t
            self.N8_t = self.gamma * self.N4_t
            self.N9_t = (1 - self.alpha) * self.N6_t
            self.N10_t = (1 - self.gamma) * self.N4_t
            self.N11_t = self.mu * self.N5_t
            self.N12_t = (1 - self.beta) * self.N8_t
            self.N13_t = self.beta * self.N8_t

            o2_con = self.N2_t[5] / np.sum(self.N2_t)
            self.removed_t = np.sum(self.U * (self.T[1:] - self.Tc) * np.pi * self.dt * self.l / (self.n-1))
            feed_econ = self.penalty[0] * self.N1_t[0] + self.penalty[1] * self.N1_t[2] + self.penalty[2] * self.N1_t[5]
            
            rec_econ = self.penalty[3] * np.sum(self.N7_t) + self.penalty[5] * np.sum(self.N7_t)**2 + self.penalty[4] * np.sum(self.N13_t) + self.penalty[6] * np.sum(self.N13_t)**2
            prod_econ = self.penalty[7] * self.N10_t[4]
            purge_gas_econ = self.penalty[8] * np.sum(self.N9_t)
            purge_liquid_econ = self.penalty[9] * np.sum(self.N12_t)
            purge_co2_econ = self.penalty[10] * np.sum(self.N3_t[6] - self.N2_t[6])
            total_econ = prod_econ - rec_econ - feed_econ - purge_gas_econ - purge_liquid_econ - purge_co2_econ - self.penalty[11] * self.removed_t
            
            
            w = self.weights
            T, P = T[1:]/w[7], np.ones(len(T[1:]))
            initial_conc = [cEe[0], cEa[0], cA[0], cW[0], cV[0], cO[0], cC[0]]
            cEe, cEa, cA, cW, cV, cO, cC = cEe[1:] / w[0], cEa[1:] * w[1], cA[1:] / w[2], cW[1:] / w[3], cV[1:] / w[4], cO[1:] / w[5], cC[1:] / w[6]
            self.impresult = np.concatenate([cEe, cEa, cA, cW, cV, cO, cC, T, P])
            self.init_for_dynamics = (initial_conc, self.impresult)
            self.cons_vio = -(0.79 * self.prod_shut * 1000/60 - self.N10_t[4])
            scales = np.array([1, 1, self.T20, self.P20, self.N1_clean[0], self.N1_clean[2], self.N1_clean[5]])
            scales_use = scales[self.thetaind]
            formR = np.array(self.uR_spec) 
            R = np.diag(1/formR[self.thetaind])
            u = np.array([self.alpha, self.beta, self.ta, self.pa, self.ne, self.na, self.no])
            u = u[self.thetaind] 
            u0 = np.array(self.u_spec)[self.thetaind]
            delta_u = u - u0
            penalty_term = delta_u.T @ R @ delta_u
            self.total_econ = total_econ
            total_econ = total_econ - self.penalty[12] * penalty_term
            self.penalty_term = self.penalty[12] * penalty_term
            self.delta_u = penalty_term ** 0.5
            total_econ_tall[i] = total_econ

        return total_econ_tall
    
    def get_dynamic(self):
        total_econ_tall = self.Trans_Econ()
        c_Ee_ft = []
        c_Ea_ft = []
        c_A_ft = []
        c_W_ft = []
        c_V_ft = []
        c_O_ft = []
        c_C_ft = []
        N1inlet = []
        Qcheck = []
        T_end = []
        r1_tilde = []
        r2_tilde = []
        r1_end = []
        r2_end = []

        cEe_tanks = np.zeros((self.n, len(self.teval)))
        cEa_tanks = np.zeros((self.n, len(self.teval)))
        cA_tanks = np.zeros((self.n, len(self.teval)))
        cW_tanks = np.zeros((self.n, len(self.teval)))
        cV_tanks = np.zeros((self.n, len(self.teval)))
        cO_tanks = np.zeros((self.n, len(self.teval)))
        cC_tanks = np.zeros((self.n, len(self.teval)))
        dcdt = np.zeros((7, self.n - 1, len(self.teval)))
        cEe_avg = np.zeros(len(self.teval))
        cEa_avg = np.zeros(len(self.teval))
        cA_avg = np.zeros(len(self.teval))
        cW_avg = np.zeros(len(self.teval))
        cV_avg = np.zeros(len(self.teval))
        cO_avg = np.zeros(len(self.teval))
        cC_avg = np.zeros(len(self.teval))
        Nv7 = np.zeros(len(self.teval))
        total_econ_t = np.zeros(len(self.teval))
        penalty_term_t = np.zeros(len(self.teval))
        obj_t = np.zeros(len(self.teval))
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
            N2 = self.N1f / self.nt + self.Qa[-1, i] * C_in_last_cstr * self.rec_factor

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
            N7_t = self.alpha * N6_t
            N8_t = self.gamma * N4_t
            N9_t = (1 - self.alpha) * N6_t
            N10_t = (1 - self.gamma) * N4_t
            N11_t = self.mu * N5_t
            N12_t = (1 - self.beta) * N8_t
            N13_t = self.beta * N8_t
            Nv7[i] = (N10_t[4])
            c_Ee_ft.append(self.cEe_tall[-1, i])
            c_Ea_ft.append(self.cEa_tall[-1, i])
            c_A_ft.append(self.cA_tall[-1, i])
            c_W_ft.append(self.cW_tall[-1, i])
            c_V_ft.append(self.cV_tall[-1, i])
            c_O_ft.append(self.cO_tall[-1, i])
            c_C_ft.append(self.cC_tall[-1, i])
            T_end.append(self.T_tall[-1, i])
            # NOTE: You had a bug here, you were calculating mean over the complete
            # tanks (that includes the inlet rate which makes no sense). We only need 
            # mean over the tanks, thus we need to exclude the inlet rate (first row).
            r1_tilde.append(np.mean(self.r1[1:, i]))
            r2_tilde.append(np.mean(self.r2[1:, i]))
            r1_end.append(self.r1[-1, i])
            r2_end.append(self.r2[-1, i])
            dcdt[:, :, i] = self.resTotal_c[:, i].reshape((7, self.n - 1))
            cEe_tanks[:, i] = self.cEe_tall[:, i]
            cEa_tanks[:, i] = self.cEa_tall[:, i]
            cA_tanks[:, i] = self.cA_tall[:, i]
            cW_tanks[:, i] = self.cW_tall[:, i]
            cV_tanks[:, i] = self.cV_tall[:, i]
            cO_tanks[:, i] = self.cO_tall[:, i]
            cC_tanks[:, i] = self.cC_tall[:, i]
            cEe_avg[i] = np.mean(self.cEe_tall[:, i])
            cEa_avg[i] = np.mean(self.cEa_tall[:, i])
            cA_avg[i] = np.mean(self.cA_tall[:, i])
            cW_avg[i] = np.mean(self.cW_tall[:, i])
            cV_avg[i] = np.mean(self.cV_tall[:, i])
            cO_avg[i] = np.mean(self.cO_tall[:, i])
            cC_avg[i] = np.mean(self.cC_tall[:, i])
            total_econ_t[i] = self.total_econ
            penalty_term_t[i] = self.penalty_term
            obj_t[i] = total_econ_t[i] - penalty_term_t[i]
            N1inlet.append(N1_t)
            Qcheck.append(self.Qa[-1, i] * self.nt)
        dyn_dict = {}
        dyn_dict['cEe'] = c_Ee_ft
        dyn_dict['cEa'] = c_Ea_ft
        dyn_dict['cA'] = c_A_ft
        dyn_dict['cW'] = c_W_ft
        dyn_dict['cV'] = c_V_ft
        dyn_dict['cO'] = c_O_ft
        dyn_dict['cC'] = c_C_ft
        dyn_dict['cEe_tanks'] = cEe_tanks
        dyn_dict['cEa_tanks'] = cEa_tanks
        dyn_dict['cA_tanks'] = cA_tanks
        dyn_dict['cW_tanks'] = cW_tanks
        dyn_dict['cV_tanks'] = cV_tanks
        dyn_dict['cO_tanks'] = cO_tanks
        dyn_dict['cC_tanks'] = cC_tanks
        dyn_dict['cEe_avg'] = cEe_avg
        dyn_dict['cEa_avg'] = cEa_avg
        dyn_dict['cA_avg'] = cA_avg
        dyn_dict['cW_avg'] = cW_avg
        dyn_dict['cV_avg'] = cV_avg
        dyn_dict['cO_avg'] = cO_avg
        dyn_dict['cC_avg'] = cC_avg
        dyn_dict['Nv7'] = Nv7
        dyn_dict['total_econ'] = total_econ_tall
        dyn_dict['alpha'] = self.alpha * np.ones(len(self.teval))
        dyn_dict['beta'] = self.beta * np.ones(len(self.teval))
        dyn_dict['T2']  = self.ta * np.ones(len(self.teval)) * self.T20
        dyn_dict['P2'] = self.pa * np.ones(len(self.teval)) * self.P20
        dyn_dict['NEe'] = self.ne * np.ones(len(self.teval)) * self.N1_clean[0]
        dyn_dict['NA'] = self.na * np.ones(len(self.teval)) * self.N1_clean[2]
        dyn_dict['NO'] = self.no * np.ones(len(self.teval)) * self.N1_clean[5]
        dyn_dict['time'] = self.teval
        dyn_dict['N1f'] = N1inlet
        dyn_dict['T_end'] = T_end
        dyn_dict['Q'] = Qcheck
        #dyn_dict['total_econ'] = total_econ_t
        dyn_dict['penalty_term'] = penalty_term_t
        dyn_dict['objective'] = obj_t
        dyn_dict['r1_tilde'] = r1_tilde
        dyn_dict['r2_tilde'] = r2_tilde
        dyn_dict['r1_end'] = r1_end
        dyn_dict['r2_end'] = r2_end
        dyn_dict['dcdt'] = dcdt
        return dyn_dict
    
    def Rate_cas(self, c, alpha, beta, N1f, theta):
        '''
        Function to calculate reaction rates.  
        Use 'if' condition with self.param to add and use different rate equations.
        '''
        T = c[-1]
        if self.rho_var:
            T_to_be_used = T
        else:
            T_to_be_used = self.T_const_inlet
        pressures = [self.R * T_to_be_used * i / 1000 for i in c[:-1]]
        pEe, pEa, pA, pW, pV, pO, pC = pressures
        cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
        if self.param == 'true':
            theta1, theta2 = theta
            
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * casadi.exp(-self.arr_rxn1 * theta1[1] * 1e3 / (self.R * T))
                    * casadi.fmax(pEe, 1e-10) ** theta1[2]
                    * casadi.fmax(pO, 1e-10) ** theta1[3]
                    * casadi.fmax(pA, 1e-10) ** theta1[4]
                )
            # NOTE: Refer the notes in the numpy rate equation
            # for term1 and term2
            term1 = casadi.fmax(pEe, 1e-10) ** -0.31
            subterm = casadi.fmax(pEe, 1e-10)
            term2 = casadi.exp(-0.001 * theta2[2] * subterm) - casadi.exp(-theta2[3] * subterm)
            term3 = casadi.fmax(pEe, 1e-10) ** 0.95
            r2 = (
                    self.factfor2 * theta2[0] * 1e-3
                    * casadi.exp(- self.arr_rxn2  * theta2[1] * 1e3 / (self.R * T))
                    * term2
                    * casadi.fmax(pO, 1e-10) ** theta2[4]
                )
            r2 = r2 + self.bias_r2
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'fake':
            theta1, theta2 = theta
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * casadi.exp(-self.arr_rxn1 * theta1[1] * 1e3 / (self.R * T))
                    * casadi.fmax(pEe, 1e-10) ** theta1[2]
                    * casadi.fmax(pO, 1e-10) ** theta1[3]
                    * casadi.fmax(pA, 1e-10) ** theta1[4]
                )
            # NOTE: Refer the notes in the numpy rate equation
            # for term1 and term2
            term1 = casadi.fmax(pEe, 1e-10) ** -0.31
            subterm = casadi.fmax(pEe, 1e-10)
            term2 = casadi.exp(-0.004 * subterm) - casadi.exp(-10 * subterm)
            term3 = casadi.fmax(pEe, 1e-10) ** theta2[2]
            r2 = (
                    self.factfor2 * theta2[0] * 1e-3
                    * casadi.exp(- self.arr_rxn2  * theta2[1] * 1e3 / (self.R * T))
                    * term3
                    * casadi.fmax(pO, 1e-10) ** theta2[3]
                )
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'test':
            theta1, theta2 = theta
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * casadi.exp(-self.arr_rxn1 * theta1[1] * 1e3 / self.R * (1/T - 1/450))
                    * casadi.fmax(cEe, 1e-10) ** theta1[2]
                    * casadi.fmax(cO, 1e-10) ** theta1[3]
                    * casadi.fmax(cA, 1e-10) ** theta1[4]
                )
            # NOTE: Refer the notes in the numpy rate equation
            # for term1 and term2
            term1 = casadi.fmax(cEe, 1e-10) ** -0.31
            subterm = casadi.fmax(cEe, 1e-10)
            term2 = casadi.exp(-0.004 * subterm) - casadi.exp(-10 * subterm)
            term3 = casadi.fmax(cEe, 1e-10) ** theta2[2]

            k2 = self.factfor2 * self.cat_decay * theta2[0] * 1e-4 

            r2 = (
                    k2
                    * casadi.exp(- self.arr_rxn2  * theta2[1] * 1e3 / self.R * (1/T - 1/450))
                    * term3
                    * casadi.fmax(cO, 1e-10) ** theta2[3]
                ) 
            r2 = r2 + self.bias_r2
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'expfit':
            theta1, theta2 = theta
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * casadi.exp(-self.arr_rxn1 * 1e3 * theta1[1] / (self.R * T))
                    * casadi.fmax(pEe, 1e-10) ** theta1[2]
                    * casadi.fmax(pO, 1e-10) ** theta1[3]
                    * casadi.fmax(pA, 1e-10) ** theta1[4]
                )
            subterm = casadi.fmax(pEe, 1e-10)
            term2 = casadi.exp(-4.137e-3 * subterm) - casadi.exp(-0.458 * subterm)
            term3 = casadi.fmax(pEe, 1e-10) ** theta2[2]
            r2 = (
                    self.factfor2 * 1e-3 * theta2[0]
                    * casadi.exp(- self.arr_rxn2  * 1e3 * theta2[1] / (self.R * T))
                    * term3
                    * casadi.fmax(pO, 1e-10) ** theta2[3]
                )
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'expfit_ver2':
            theta1, theta2 = theta
            r1 = (
                    self.cat_decay * theta1[0] * 1e-4
                    * casadi.exp(-self.arr_rxn1 * 1e3 * theta1[1] / (self.R * T))
                    * casadi.fmax(pEe, 1e-10) ** theta1[2]
                    * casadi.fmax(pO, 1e-10) ** theta1[3]
                    * casadi.fmax(pA, 1e-10) ** theta1[4]
                )
            subterm = casadi.fmax(pEe, 1e-10)
            term2 = casadi.exp(-theta2[2] * 1e-3 * subterm) - casadi.exp(-theta2[3] * subterm)
            term3 = casadi.fmax(pEe, 1e-10) ** theta2[2]
            r2 = (
                    self.factfor2 * 1e-3 * theta2[0]
                    * casadi.exp(- self.arr_rxn2  * 1e3 * theta2[1] / (self.R * T))
                    * term2
                    * casadi.fmax(pO, 1e-10) ** theta2[4]
                )
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'stoic_base':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
            # input1 = casadi.horzcat(cEe, cA, cO, T)
            # input1 = input1.T
            # input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]]
            # input2 = casadi.horzcat(cEe, cO, T)
            # input2 = input2.T
            # input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            # r1_badshape = self.jaxadi(self.nn_rec1, input1)
            # r1_norm = casadi.reshape(r1_badshape, -1, 1)
            # r1 = (10 ** self.mult_r[0]) * r1_norm + (10 ** self.mult_r[2])
            # r2_badshape = self.jaxadi(self.nn_rec2, input2)
            # r2_norm = casadi.reshape(r2_badshape, -1, 1)
            # r2 = (10 ** self.mult_r[1]) * self.factfor2 * r2_norm + (10 ** self.mult_r[3])

            input1 = casadi.horzcat(cEe, cA, cO, T).T
            input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]] 
            input2 = casadi.horzcat(cEe, cO, T).T
            input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]] 
            r1 = self.jaxadi(self.nn_rec1, input1)
            r2 = self.jaxadi(self.nn_rec2, input2)
            r = casadi.vertcat(r1, r2)
            mean_r = self.mult_r[:, 0]
            std_r = self.mult_r[:, 1]
            r = (10 ** std_r) * r  + (10 ** mean_r)
            vT = np.array([[-1, -1],
                [0, 0],
                [-1, 0], 
                [1, 2],
                [1, 0],
                [-0.5, -3],
                [0, 2]], dtype=np.float64)
            r1 = vT @ r
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 *  r1[0, :]

        elif self.param == 'stoic_red_base':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
            # input1 = casadi.horzcat(cEe, cA, cO, T)
            # input1 = input1.T
            # input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]]
            # input2 = casadi.horzcat(cEe, cO, T)
            # input2 = input2.T
            # input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            # r1_badshape = self.jaxadi(self.nn_rec1, input1)
            # r1_norm = casadi.reshape(r1_badshape, -1, 1)
            # r1 = (10 ** self.mult_r[0]) * r1_norm + (10 ** self.mult_r[2])
            # r2_badshape = self.jaxadi(self.nn_rec2, input2)
            # r2_norm = casadi.reshape(r2_badshape, -1, 1)
            # r2 = (10 ** self.mult_r[1]) * self.factfor2 * r2_norm + (10 ** self.mult_r[3])

            input1 = casadi.horzcat(cA, T).T
            input1 = (input1 - self.mean[[2,7]]) / self.std[[2,7]]
            input2 = casadi.horzcat(T).T
            input2 = (input2 - self.mean[[7]]) / self.std[[7]]
            red_input = (cEe * cO - self.mean_ceeco) / self.std_ceeco
            input1 = casadi.horzcat(red_input, input1.T).T
            input2 = casadi.horzcat(red_input, input2.T).T
            r1 = self.jaxadi(self.nn_rec1, input1)
            r2 = self.jaxadi(self.nn_rec2, input2)
            r = casadi.vertcat(r1, r2)
            mean_r = self.mult_r[:, 0]
            std_r = self.mult_r[:, 1]
            r = (10 ** std_r) * r  + (10 ** mean_r)
            vT = np.array([[-1, -1],
                [0, 0],
                [-1, 0], 
                [1, 2],
                [1, 0],
                [-0.5, -3],
                [0, 2]], dtype=np.float64)
            r1 = vT @ r
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 *  r1[0, :]

        elif self.param == 'stoic':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
            # input1 = casadi.horzcat(cEe, cA, cO, T)
            # input1 = input1.T
            # input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]]
            # input2 = casadi.horzcat(cEe, cO, T)
            # input2 = input2.T
            # input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            # r1_badshape = self.jaxadi(self.nn_rec1, input1)
            # r1_norm = casadi.reshape(r1_badshape, -1, 1)
            # r1 = (10 ** self.mult_r[0]) * r1_norm + (10 ** self.mult_r[2])
            # r2_badshape = self.jaxadi(self.nn_rec2, input2)
            # r2_norm = casadi.reshape(r2_badshape, -1, 1)
            # r2 = (10 ** self.mult_r[1]) * self.factfor2 * r2_norm + (10 ** self.mult_r[3])

            input1 = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC, T).T
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0]
            std_r = self.mult_r[:, 1]
            r = (10 ** std_r) * self.jaxadi(self.nn_net, input1)  + (10 ** mean_r)
            vT = np.array([[-1, -1],
                [0, 0],
                [-1, 0], 
                [1, 2],
                [1, 0],
                [-0.5, -3],
                [0, 2]], dtype=np.float64)
            r1 = vT @ r
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 *  r1[0, :]

        elif self.param == 'prod':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC, T).T
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0]
            std_r = self.mult_r[:, 1]
            r1 = (10 ** std_r) * self.jaxadi(self.nn_net, input1)  + (10 ** mean_r)
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 * r1[0, :]

        elif self.param == 'prod_null':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC, T).T
            input1 = (input1 - self.mean) / self.std 
            r1 = self.mult_r * (self.jaxadi(self.nn_net, input1) * self.std[:7] + self.mean[:7])
            W = np.diag(1/self.mult_r/self.std[:7].flatten())
            W_inv = np.linalg.inv(W)
            A = (W_inv @ self.mol).reshape(-1, 1)
            B = np.dot(self.mol, W_inv @ self.mol)
            print(r1.shape)
            C = self.mol.reshape(1, -1) @ r1
            print(r1.shape)
            r1 = r1- A @ (C / B)
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 * r1[0, :]
        
        elif self.param == 'prod_e':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC, T).T
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0]
            std_r = self.mult_r[:, 1]
            r1 = (10 ** std_r) * self.jaxadi(self.nn_net, input1)  + (10 ** mean_r)
            vT = np.array([[1.5, -2.0, 1.0, -0.5],
                            [-1., 1., .0, 1.0],
                            [-0.5, -1., -1., -1.],
                            [1.0, 0., 0., 0.],
                            [0.0, 1.0, 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
            r1 = vT @ r1
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 * r1[0, :]

        elif self.param == 'prod_m':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC, T).T
            input1 = (input1 - self.mean) / self.std 
            mean_r = self.mult_r[:, 0]
            std_r = self.mult_r[:, 1]
            r1 = (10 ** std_r) * self.jaxadi(self.nn_net, input1)  + (10 ** mean_r)
            vT = self.null_mass
            r1 = vT @ r1
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 * r1[0, :]

        elif self.param == 'stoic_notT':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
            input1 = casadi.horzcat(cEe, cA, cO)
            input1 = input1.T
            input1 = (input1 - self.mean[[0,2,5]]) / self.std[[0,2,5]]
            input2 = casadi.horzcat(cEe, cO)
            input2 = input2.T
            input2 = (input2 - self.mean[[0,5]]) / self.std[[0,5]]
            r1_badshape = self.jaxadi(self.nn_rec1, input1)
            r1_norm = casadi.reshape(r1_badshape, -1, 1)
            r1 = 0.5e-3 * r1_norm
            r2_badshape = self.jaxadi(self.nn_rec2, input2)
            r2_norm = casadi.reshape(r2_badshape, -1, 1)
            r2 = 0.5e-3 * self.factfor2 * r2_norm
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        elif self.param == 'prod_notT':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]  
            input1 = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC).T
            input1 = (input1 - self.mean) / self.std 
            r1 = 0.5e-4 * (self.jaxadi(self.nn_net, input1) * self.std + self.mean)
            rEe = r1[0, :]
            rEa = r1[1, :]
            rA = r1[2, :]
            rW = r1[3, :]
            rV = r1[4, :]
            rO = r1[5, :]
            rC = r1[6, :]
            rheat = 0 * r1[0, :]

        elif self.param == 'stoic_delay':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
            # get the shape of the conc array  
            shape_for_copy = cEe.shape[0]

            # form the conc array
            c = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC).T
            c_norm = ((c - self.mean) / self.std).T
            
            # form the u array
            u = casadi.vertcat(N1f[0], N1f[2], N1f[5], alpha, beta, T[0])
            u_norm = (u - self.mean_u)/self.std_u
            u_norm_expanded = casadi.horzcat(*[u_norm for _ in range(shape_for_copy)]).T
            
            # form the delay embedding (at steady-state just copy of current c's and u's)
            c_z_norm = casadi.horzcat(*[c_norm for _ in range(self.n_past)])
            u_z_norm = casadi.horzcat(*[u_norm_expanded for _ in range(self.n_past)])

            # form the current conc, u and delay embedding (already normalized)
            c_u_z = casadi.horzcat(c_norm, u_norm_expanded, c_z_norm, u_z_norm)
            std_r = np.array([self.std_r1, self.std_r2])
            mean_r = np.array([self.mean_r1, self.mean_r2])
            rate = self.jaxadi(self.nn_net, c_u_z.T)
            r = rate * std_r + mean_r
            r1 = r[0, :]
            r2 = r[1, :]
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * rEe

        elif self.param == 'stoic_exp':
            cEe, cEa, cA, cW, cV, cO, cC = c[:-1]
            input1 = casadi.horzcat(cEe, cA, cO, T)
            input1 = input1.T
            input1 = (input1 - self.mean[[0,2,5,7]]) / self.std[[0,2,5,7]]
            input2 = casadi.horzcat(cEe, cO, T)
            input2 = input2.T
            input2 = (input2 - self.mean[[0,5,7]]) / self.std[[0,5,7]]
            r1_badshape = self.jaxadi(self.nn_rec1, input1)
            r1_norm = casadi.reshape(r1_badshape, -1, 1)
            r1 = r1_norm
            r2_badshape = self.jaxadi(self.nn_rec2, input2)
            r2_norm = casadi.reshape(r2_badshape, -1, 1)
            r2 = self.factfor2 * r2_norm
            r1 = self.std_r1 * r1 + self.mean_r1
            r2 = self.std_r2 * r2 + self.mean_r2
            r1 = casadi.exp(-self.arr_rxn1 * 15e3 / (self.R * T)) * r1
            r2 = casadi.exp(- self.arr_rxn2  * 21e3 / (self.R * T)) * r2
            rEe = -r1 - r2
            rEa = 0 * r1 + 0 * r2
            rA = -r1
            rW = r1 + 2 * r2
            rV = r1
            rO = -0.5 * r1 - 3 * r2
            rC = 2 * r2
            rheat = 0 * r1

        else:
            raise ValueError(f"Unknown param value: {self.param}")
        
        return rEe, rEa, rA, rW, rV, rO, rC, rheat

    def optimize(self, initial, bounds):
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
        revenue = prod_econ
        opex = rec_econ + feed_econ + purge_gas_econ + purge_liquid_econ + purge_co2_econ + self.penalty[11] * heat_removed
        total_eco = revenue - opex
        #total_eco = prod_econ - rec_econ - feed_econ - purge_gas_econ - purge_liquid_econ - purge_co2_econ - self.penalty[11] * heat_removed 
        
        input_to_casadi_func = [x, p] if self.param in self.rate_param_list else [x]
        resfun = casadi.Function('resfun', input_to_casadi_func, [res])
        getcs = casadi.Function('getcs', input_to_casadi_func, [cEe, cEa, cA, cW, cV, cO, cC, T, P, Qflow])
        r = casadi.Function('r', input_to_casadi_func, [r1, r2])

        revenue_func = casadi.Function('revenue_func', input_to_casadi_func, [revenue])
        opex_func = casadi.Function('opex_func', input_to_casadi_func, [opex])


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

        thetaind = np.delete(np.arange(len(self.p_spec)), idx)
        zind = np.delete(np.arange(int(x.numel())), thetaind)
        procind = np.delete(np.arange(int(x.numel())), np.arange(len(self.p_spec)))
        self.thetaind = thetaind

        scales = np.array([1, 1, self.T20, self.P20, self.N1_clean[0], self.N1_clean[2], self.N1_clean[5]])
        scales_use = scales[thetaind]
        # form the penalty matrix on the control inputs
        formR = np.array(self.uR_spec)
        R = np.diag(1/formR[thetaind])
        R_scaled = np.diag(1/(formR[thetaind]*scales_use))
        self.R_matrix = R
        u = x[thetaind]
        u0 = np.array(self.u_spec)[thetaind]
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
        input_to_casadi_func_num = [self.sol['x'], p0] if self.param in self.rate_param_list else [self.sol['x']]

        resfunnum = resfun(*input_to_casadi_func_num).toarray().flatten()
        self.res_ipopt = resfunnum

        self.conc = getcs(*input_to_casadi_func_num)
        self.rates = r(*input_to_casadi_func_num)

        self.revenue = revenue_func(*input_to_casadi_func_num).toarray().flatten()[0]
        self.opex = opex_func(*input_to_casadi_func_num).toarray().flatten()[0]

        # we need to form the active set jacobian
        active_g = np.arange(0, 9 * (self.n - 1))   
        active_x = np.delete(np.arange(len(self.p_spec)), thetaind)

        active_g_cons = constraint[active_g]
        active_x_cons = x[active_x] - lb[active_x]

        check_lams = np.where(~np.isclose(self.sol['lam_x'][thetaind], 0, rtol=1e-5, atol=1e-5))
        for i in check_lams[0]:
            if i not in active_x:
                active_x = np.hstack((active_x, i))
                if self.sol['lam_x'][i] > 0:
                    active_x_cons = casadi.vertcat(active_x_cons, x[i] - lb[i])
                else:
                    active_x_cons = casadi.vertcat(active_x_cons, x[i] - ub[i])

        # get the new thetaind
        thetaind = np.delete(np.arange(len(self.p_spec)), np.sort(active_x))
        self.thetaind = thetaind
        zind = np.delete(np.arange(int(x.numel())), thetaind)
        self.zind = zind
        # also get new R_scaled_inv
        scales_use = scales[np.delete(np.arange(len(self.p_spec)), np.sort(active_x))]
        R_scaled = np.diag(1/(formR[np.delete(np.arange(len(self.p_spec)), np.sort(active_x))]*scales_use))
        self.R_matrix = R_scaled    
        R_scaled_inv = np.linalg.inv(R_scaled)


        active_cons = casadi.vertcat(active_g_cons, active_x_cons)
        self.active_cons = active_cons

        # # test old code construct jacobian of active g
        # self.Jt = casadi.jacobian(active_g_cons, x)
        # self.Jt_fun = casadi.Function('Jt_fun', [x], [self.Jt])
        # self.Jt_num = self.Jt_fun(self.sol['x']).toarray()

        self.active_jac = casadi.jacobian(active_cons, x)
        self.active_jac_fun = casadi.Function('active_jac_fun', input_to_casadi_func, [self.active_jac])
        self.active_jac_num = self.active_jac_fun(*input_to_casadi_func_num).toarray()

        # check whether we have row rank
        rank = np.linalg.matrix_rank(self.active_jac_num)
        if rank < self.active_jac_num.shape[0]:
            print("Warning: Active constraint Jacobian is rank deficient")
            self.log_licq = False
        # check whether any of the active constraints are at their bounds
        # this should not happen, if it does, we have redundant constraints in the active set
        # since ub is infty check only lb
        
        if np.isclose(self.sol_x, lb[procind], rtol=1e-5, atol=1e-5).any():
            print("Warning: Redundant constraints in active set, hessian should be positive semidefinite")
            self.log_licq = False
        else:
            self.log_licq = True

        # collect the relevant multipliers
        lam_g_active = np.array(self.sol['lam_g']).ravel()[active_g]
        lam_x_active = np.array(self.sol['lam_x']).ravel()[active_x]
        self.lam_active = np.hstack((lam_g_active, lam_x_active))

        self.Mrcheck_fun = casadi.Function('Mrcheck_fun', input_to_casadi_func, [sumMr])
        self.Mrcheck = self.Mrcheck_fun(*input_to_casadi_func_num).toarray().flatten()[0]

        # form the lagrangian
        self.lagrangian = -total_econ + casadi.dot(self.lam_active, self.active_cons)

        # form the various derivatives
        y = x[thetaind]
        z = x[zind]
        # L_yy
        self.L_x = casadi.jacobian(total_econ, x)
        self.L_x_fun = casadi.Function('L_x_fun', input_to_casadi_func, [self.L_x])
        self.L_x_num = self.L_x_fun(*input_to_casadi_func_num).toarray().flatten()
        self.L_yy = casadi.hessian(self.lagrangian, y)[0]
        self.L_yy_fun = casadi.Function('L_yy_fun', input_to_casadi_func, [self.L_yy])
        self.L_yy_num = self.L_yy_fun(*input_to_casadi_func_num).toarray()

        # L_zz
        self.L_zz = casadi.hessian(self.lagrangian, z)[0]
        self.L_zz_fun = casadi.Function('L_zz_fun', input_to_casadi_func, [self.L_zz])
        self.L_zz_num = self.L_zz_fun(*input_to_casadi_func_num).toarray()

        # L_yz (i.e. first differentiate wrt y, then wrt z)
        self.L_yz = casadi.jacobian(casadi.jacobian(self.lagrangian, y), z)
        self.L_yz_fun = casadi.Function('L_yz_fun', input_to_casadi_func, [self.L_yz])
        self.L_yz_num = self.L_yz_fun(*input_to_casadi_func_num).toarray()

        # L_zy (i.e. first differentiate wrt z, then wrt y)
        self.L_zy = casadi.jacobian(casadi.jacobian(self.lagrangian, z), y)
        self.L_zy_fun = casadi.Function('L_zy_fun', input_to_casadi_func, [self.L_zy])
        self.L_zy_num = self.L_zy_fun(*input_to_casadi_func_num).toarray()

        # J_y
        self.J = self.active_jac_num
        self.J_y = self.J[:, thetaind]
        self.J_z = self.J[:, zind]

        # if self.param in self.rate_param_list:
        #     self.E_p = casadi.jacobian(total_econ, p)
        #     self.E_p_fun = casadi.Function('E_p_fun', input_to_casadi_func, [self.E_p])
        #     self.E_p_num = self.E_p_fun(*input_to_casadi_func_num).toarray().flatten()

        #     self.E_x = casadi.jacobian(total_econ, z[idx.shape[0]:])
        #     self.E_x_fun = casadi.Function('E_x_fun', input_to_casadi_func, [self.E_x])
        #     self.E_x_num = self.E_x_fun(*input_to_casadi_func_num).toarray().flatten()

        #     self.J_t = casadi.jacobian(active_g_cons, p)
        #     self.J_t_fun = casadi.Function('J_t_fun', input_to_casadi_func, [self.J_t])
        #     self.J_t_num = self.J_t_fun(*input_to_casadi_func_num).toarray()

        #     req_k = 5.2e12
        #     _, S, _ = np.linalg.svd(self.J_z[idx.shape[0]:,idx.shape[0]:])
        #     sigma_max = S[0]
        #     sigma_min = S[-1]
        #     req_jitter = (sigma_max ** 2 - req_k * sigma_min ** 2) / (req_k - 1)
        #     self.req_jitter = req_jitter 
        #     self.current_k = sigma_max ** 2 / (sigma_min + 1e-5) ** 2

        #     self.DJ_DT = -self.E_x_num @ (np.linalg.inv(self.J_z[idx.shape[0]:,idx.shape[0]:] + self.req_jitter * np.eye(self.J_z[idx.shape[0]:,idx.shape[0]:].shape[0])) @ self.J_t_num) + self.E_p_num
        #     self.DJ_DT_scaled = self.DJ_DT * (np.hstack([self.theta1, self.theta2]) / self.obj)
        # # form inverse of J_z
        self.J_z_inv = np.linalg.inv(self.J_z)

        # form the schur complement
        self.L_schur = self.L_yy_num - self.L_yz_num @ self.J_z_inv @ self.J_y - self.J_y.T @ self.J_z_inv.T @ self.L_zy_num + self.J_y.T @ self.J_z_inv.T @ self.L_zz_num @ self.J_z_inv @ self.J_y
        # make sure it's symmetric
        self.L_schur = 0.5 * (self.L_schur + self.L_schur.T)
        R_scaled_inv = np.linalg.inv(R_scaled)
        self.L_schur_scaled = R_scaled_inv.T @ self.L_schur @ R_scaled_inv


        # compare with H_red (null space method)
        # form null space of active jacobian
        self.Z = null_space(self.active_jac_num)
        L_xx = casadi.hessian(self.lagrangian, x)[0]
        L_xx_fun = casadi.Function('L_xx_fun', input_to_casadi_func, [L_xx])
        L_xx_num = L_xx_fun(*input_to_casadi_func_num).toarray()
        self.L_red = self.Z.T @ L_xx_num @ self.Z

        # make sure it's symmetric
        self.L_red = 0.5 * (self.L_red + self.L_red.T)

        # convert to y coordinates
        z_y = - self.J_z_inv @ self.J_y
        # build S_y and S_z
        n=self.active_jac_num.shape[1]
        p=len(thetaind)
        q=n-p
        Sy = np.zeros((n, p));  Sy[thetaind, np.arange(p)] = 1.0
        Sz = np.zeros((n, q));  Sz[zind, np.arange(q)] = 1.0
        self.S = Sy + Sz @ z_y
        self.L_red_true = self.S.T @ L_xx_num @ self.S
        self.L_red_true = 0.5 * (self.L_red_true + self.L_red_true.T)

        # store condition number of hessian
        try:
            self.cond_hess = np.linalg.cond(self.L_schur)
        except np.linalg.LinAlgError:
            self.cond_hess = np.inf

        # classify definiteness of reduced hessian from its eigenvalues
        # eigvalsh returns real eigenvalues in ascending order (symmetric solver)
        try:
            w = np.linalg.eigvalsh(self.L_schur)
            if w.size == 0:
                self.eigcheck_hess = 'empty'         # no free variables
            else:
                tol = np.finfo(float).eps * np.abs(w).max() * self.L_schur.shape[0]
                if w[0] > tol:
                    self.eigcheck_hess = 'pos_def'
                elif w[-1] < -tol:
                    self.eigcheck_hess = 'neg_def'
                elif w[0] >= -tol:
                    self.eigcheck_hess = 'pos_semidef'   # all >= 0, at least one ~0
                elif w[-1] <= tol:
                    self.eigcheck_hess = 'neg_semidef'   # all <= 0, at least one ~0
                else:
                    self.eigcheck_hess = 'indefinite'    # eigenvalues of both signs
        except np.linalg.LinAlgError:
            self.eigcheck_hess = 'undefined'

        # NOTE: equality constraints are always active.
        # step 1 identify freezed u (freezed by me + freezed by bounds)
        # step 2 identify freezed x (freezed by bounds - leads to degeneracy - raise flag)
        # step 3 identify active constraint set = model + step 1 
        # step 4 identify the final x (includes freezed u + freezed by bounds) and u

    @staticmethod
    def perturbation(Plant, u, dict, perturb=0.1, steps=20, cbox=False):
        traj = np.linspace(dict[u] + perturb * dict[u],
                           dict[u] - perturb * dict[u],
                           steps)
        if u in ['alpha', 'beta']:
            traj_clipped = np.clip(traj, 0, 0.99)
            if not np.array_equal(traj, traj_clipped):
                traj_truncated = np.linspace(np.min(traj_clipped), np.max(traj_clipped), steps)
                traj = traj_truncated
            else:
                traj = traj


        Collect_econ = []
        Collect_con_vio = []
        mask_neg = []
        if cbox == True:
            Collect_econ_cbox = []
        keys = ['alpha', 'beta', 'ta', 'pa', 'ne', 'na', 'no']
        #if u == 'ta':
        #    traj = np.linspace(0.9,1.2, steps)
        for key in keys:
            setattr(Plant, key, dict[key])
        #guess = dict['sol']
        success = 0
        initial = Plant.init_for_t0
        for i in traj:
            setattr(Plant, u, i)
            guess = dict['sol']
            Plant.Stream_table(guess)
            if Plant.ier == 1 and np.all(Plant.result>0):
                Collect_econ.append(Plant.Economics() - Plant.penalty_term)
                Collect_con_vio.append(Plant.cons_vio)
                guess = Plant.result
                mask_neg.append(0)
                if cbox == True:
                    Plant.cbox_solve(guess)
                    Collect_econ_cbox.append(Plant.total_econ)
                guess = Plant.result
            else:
                Plant.time = [0, 1000]
                Plant.teval = np.linspace(Plant.time[0], Plant.time[1], Plant.t_points)
                Plant.solve_t(initial)
                Collect_econ.append(Plant.Trans_Econ() - Plant.penalty_term)
                Collect_con_vio.append(Plant.cons_vio)
                mask_neg.append(1)
                if cbox == True:
                    Plant.cbox_solve(Plant.impresult)
                    Collect_econ_cbox.append(Plant.total_econ)

        scale_dict = {}
        scale_dict['ne'] = 0
        scale_dict['na'] = 2
        scale_dict['no'] = 5
    

        if u in ['alpha', 'beta']:
            traj = traj 
            pair = (dict['profit'] - dict['penalty'], dict[u])
        elif u == 'ta':
            traj = traj * Plant.T20
            pair = (dict['profit'] - dict['penalty'], dict[u] * Plant.T20)
        else:
            traj = traj * Plant.N1_clean[scale_dict[u]]
            pair = (dict['profit'] - dict['penalty'], dict[u] * Plant.N1_clean[scale_dict[u]])
        if cbox == False:
            return Collect_econ, traj, mask_neg, pair, success
        else:
            return Collect_econ, traj, mask_neg, pair, success, Collect_econ_cbox

    def optimize_cbox(self, initial, bounds):
        self.initialize()
        x = casadi.SX.sym('x', 7 + 9 * (self.n - 1))

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

        C = casadi.horzcat(cEe, cEa, cA, cW, cV, cO, cC)
        U = casadi.horzcat(*(N1f[0], N1f[2], N1f[5], N1f[3], N1f[4], N1f[6], alpha, beta, T2))
        Cnorm = ((C.T - self.mean) / self.std).T
        Unorm = ((U.T - self.mean_u) / self.std_u).T
        input_to_nn = casadi.horzcat(Cnorm, Unorm).T
        res = (self.jaxadi(self.nn_bbox, input_to_nn) * self.std_dcdt + self.mean_dcdt) / self.std
        resP = P[:-1] - P[1:]
        resP_scaled = resP / self.P20
        resT = (T[:-1] - T[1:]) / self.T20

        res = casadi.vertcat(res, resT, resP_scaled)
        rec_factor = (self.phi * (1 - self.mu) * alpha +
                   (1 - self.phi) * self.gamma * beta)
        
        # We calculate the conc and volumetric flow rate in last tank to avoid DAEs.
        C_in_last_cstr = casadi.vertcat(cEe[-1], cEa[-1], cA[-1], cW[-1], cV[-1], cO[-1], cC[-1])

        Q_in_last_cstr = casadi.dot(self.mol, N1f) / casadi.dot(self.mol, C_in_last_cstr * (1 - rec_factor)) / self.nt
        N2 = N10 + Q_in_last_cstr * C_in_last_cstr * rec_factor
        # Construct the actual molar flow
        N1 = N1f
        N2 = N2 * self.nt
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
        feed_econ = self.penalty[0] * N1[0] + self.penalty[1] * N1[2] + self.penalty[2] * N1[5]
        rec_econ = self.penalty[3] * casadi.sum1(N7) + self.penalty[5] * casadi.sum1(N7)**2 + self.penalty[4] * casadi.sum1(N13) + self.penalty[6] * casadi.sum1(N13)**2
        prod_econ = self.penalty[7] * N10[4]
        purge_gas_econ = self.penalty[8] * casadi.sum1(N9)
        purge_liquid_econ = self.penalty[9] * casadi.sum1(N12)
        #purge_co2_econ = self.penalty[10] * casadi.sum1(N11)
        purge_co2_econ = self.penalty[10] * (N3[6] - N2[6])
        heat_removed = casadi.sum1(self.U * (T[1:] - self.Tc) * np.pi * self.dt * self.l / (self.n-1))
        total_eco = prod_econ - rec_econ - feed_econ - purge_gas_econ - purge_liquid_econ - purge_co2_econ - self.penalty[11] * heat_removed 


        bound_keys = ['alpha', 'beta', 'T2', 'P2', 'NEe', 'NA', 'NO', 'x']
        bounds_list = []
        for key in bound_keys:
            bounds_list += bounds[key]
        ub, lb = [i[-1] for i in bounds_list], [i[0] for i in bounds_list]
        p_spec = casadi.DM(self.p_spec)

        # construct the IFT machinery
        # divide x into y (thetas) and z (remaining)
        ub_ref = np.array(ub)[np.arange(len(self.p_spec))]
        lb_ref = np.array(lb)[np.arange(len(self.p_spec))]
        # get indices of ub-lb < tol
        tol = 1e-5
        idx = np.where(ub_ref - lb_ref < tol)[0]

        thetaind = np.delete(np.arange(len(self.p_spec)), idx)
        zind = np.delete(np.arange(int(x.numel())), thetaind)
        procind = np.delete(np.arange(int(x.numel())), np.arange(len(self.p_spec)))
        self.thetaind = thetaind
        scales = np.array([1, 1, self.T20, self.P20, self.N1_clean[0], self.N1_clean[2], self.N1_clean[5]])
        scales_use = scales[thetaind]
        # form the penalty matrix on the control inputs
        formR = np.array(self.uR_spec)
        R = np.diag(1/formR[thetaind])
        R_scaled = np.diag(1/(formR[thetaind]*scales_use))
        self.R_matrix = R
        u = x[thetaind]
        u0 = np.array(self.u_spec)[thetaind]
        scaled_u = u 
        scaled_u0 = u0
        delta_u = scaled_u - scaled_u0
        penalty_term = casadi.mtimes(casadi.mtimes(delta_u.T, R), delta_u)
        save_penalty = casadi.Function('save_penalty', [x], [penalty_term])
        total_econ = total_eco - self.penalty[12] * penalty_term

        initial = casadi.DM(initial)
        initial = casadi.vertcat(p_spec, initial)
        prod = N10[4]*60/1000
        constraint = casadi.vertcat(res, prod)
        lbg = casadi.vertcat(casadi.DM.zeros(9 * (self.n - 1)), self.prod_shut * 0.79)
        ubg = casadi.vertcat(casadi.DM.zeros(9 * (self.n - 1)), casadi.inf)
        nlp = {'x': x, 'f': -total_econ, 'g': constraint}
        solver = casadi.nlpsol('solver', 'ipopt', nlp, {'ipopt': {'print_level': self.print_level, 'tol': 1e-8, 'max_iter': 3000, 
                                                                  'nlp_scaling_method': 'none', 'mu_init':1e-10},})
        self.sol = solver(x0 = initial, lbx = lb, ubx = ub, lbg = lbg, ubg = ubg)
        self.rootfinder_problem = casadi.Function('rootfinder_problem', [x], [res, alpha, total_econ])

        self.sol_x = self.sol['x'].toarray().flatten()[7:]
        self.sol_alpha = self.sol['x'].toarray().flatten()[0]
        self.sol_beta = self.sol['x'].toarray().flatten()[1]
        self.sol_ta = self.sol['x'].toarray().flatten()[2]
        self.sol_pa = self.sol['x'].toarray().flatten()[3]
        self.sol_ne = self.sol['x'].toarray().flatten()[4]
        self.sol_na = self.sol['x'].toarray().flatten()[5]
        self.sol_no = self.sol['x'].toarray().flatten()[6]
        self.sol_cEe = self.sol['x'].toarray().flatten()[0] * self.weights[0]
        self.sol_cEa = self.sol['x'].toarray().flatten()[1] / self.weights[1]
        self.sol_cA = self.sol['x'].toarray().flatten()[2] * self.weights[2]
        self.sol_cW = self.sol['x'].toarray().flatten()[3] * self.weights[3]
        self.sol_cV = self.sol['x'].toarray().flatten()[4] * self.weights[4]
        self.sol_cO = self.sol['x'].toarray().flatten()[5] * self.weights[5]
        self.sol_cC = self.sol['x'].toarray().flatten()[6] * self.weights[6]
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
        'profit': self.obj + self.save_penalty_val * self.penalty[12],
        'penalty': self.save_penalty_val * self.penalty[12]
        }
        lb = np.array(lb)
        input_to_casadi_func = [x]
        input_to_casadi_func_num = [self.sol['x']]
        active_g = np.arange(0, 9 * (self.n - 1))   
        active_x = np.delete(np.arange(len(self.p_spec)), thetaind)

        active_g_cons = constraint[active_g]
        active_x_cons = x[active_x] - lb[active_x]

        check_lams = np.where(~np.isclose(self.sol['lam_x'][thetaind], 0, rtol=1e-5, atol=1e-5))
        for i in check_lams[0]:
            if i not in active_x:
                active_x = np.hstack((active_x, i))
                if self.sol['lam_x'][i] > 0:
                    active_x_cons = casadi.vertcat(active_x_cons, x[i] - lb[i])
                else:
                    active_x_cons = casadi.vertcat(active_x_cons, x[i] - ub[i])

        # get the new thetaind
        thetaind = np.delete(np.arange(len(self.p_spec)), np.sort(active_x))
        self.thetaind = thetaind
        zind = np.delete(np.arange(int(x.numel())), thetaind)
        self.zind = zind
        # also get new R_scaled_inv
        scales_use = scales[np.delete(np.arange(len(self.p_spec)), np.sort(active_x))]
        R_scaled = np.diag(1/(formR[np.delete(np.arange(len(self.p_spec)), np.sort(active_x))]*scales_use))
        self.R_matrix = R_scaled    
        R_scaled_inv = np.linalg.inv(R_scaled)


        active_cons = casadi.vertcat(active_g_cons, active_x_cons)
        self.active_cons = active_cons

        # # test old code construct jacobian of active g
        # self.Jt = casadi.jacobian(active_g_cons, x)
        # self.Jt_fun = casadi.Function('Jt_fun', [x], [self.Jt])
        # self.Jt_num = self.Jt_fun(self.sol['x']).toarray()

        self.active_jac = casadi.jacobian(active_cons, x)
        self.active_jac_fun = casadi.Function('active_jac_fun', input_to_casadi_func, [self.active_jac])
        self.active_jac_num = self.active_jac_fun(*input_to_casadi_func_num).toarray()

        # check whether we have row rank
        rank = np.linalg.matrix_rank(self.active_jac_num)
        if rank < self.active_jac_num.shape[0]:
            print("Warning: Active constraint Jacobian is rank deficient")
            self.log_licq = False
        # check whether any of the active constraints are at their bounds
        # this should not happen, if it does, we have redundant constraints in the active set
        # since ub is infty check only lb
        
        if np.isclose(self.sol_x, lb[procind], rtol=1e-5, atol=1e-5).any():
            print("Warning: Redundant constraints in active set, hessian should be positive semidefinite")
            self.log_licq = False
        else:
            self.log_licq = True

        # collect the relevant multipliers
        lam_g_active = np.array(self.sol['lam_g']).ravel()[active_g]
        lam_x_active = np.array(self.sol['lam_x']).ravel()[active_x]
        self.lam_active = np.hstack((lam_g_active, lam_x_active))

        # form the lagrangian
        self.lagrangian = -total_econ + casadi.dot(self.lam_active, self.active_cons)

        # form the various derivatives
        y = x[thetaind]
        z = x[zind]
        # L_yy
        self.L_x = casadi.jacobian(total_econ, x)
        self.L_x_fun = casadi.Function('L_x_fun', input_to_casadi_func, [self.L_x])
        self.L_x_num = self.L_x_fun(*input_to_casadi_func_num).toarray().flatten()
        self.L_yy = casadi.hessian(self.lagrangian, y)[0]
        self.L_yy_fun = casadi.Function('L_yy_fun', input_to_casadi_func, [self.L_yy])
        self.L_yy_num = self.L_yy_fun(*input_to_casadi_func_num).toarray()

        # L_zz
        self.L_zz = casadi.hessian(self.lagrangian, z)[0]
        self.L_zz_fun = casadi.Function('L_zz_fun', input_to_casadi_func, [self.L_zz])
        self.L_zz_num = self.L_zz_fun(*input_to_casadi_func_num).toarray()

        # L_yz (i.e. first differentiate wrt y, then wrt z)
        self.L_yz = casadi.jacobian(casadi.jacobian(self.lagrangian, y), z)
        self.L_yz_fun = casadi.Function('L_yz_fun', input_to_casadi_func, [self.L_yz])
        self.L_yz_num = self.L_yz_fun(*input_to_casadi_func_num).toarray()

        # L_zy (i.e. first differentiate wrt z, then wrt y)
        self.L_zy = casadi.jacobian(casadi.jacobian(self.lagrangian, z), y)
        self.L_zy_fun = casadi.Function('L_zy_fun', input_to_casadi_func, [self.L_zy])
        self.L_zy_num = self.L_zy_fun(*input_to_casadi_func_num).toarray()

        # J_y
        self.J = self.active_jac_num
        self.J_y = self.J[:, thetaind]
        self.J_z = self.J[:, zind]

        # form inverse of J_z
        try:
            self.J_z_inv = np.linalg.inv(self.J_z)
        except np.linalg.LinAlgError:
            print("Warning: J_z is singular, cannot compute its inverse.")
            self.J_z_inv = np.linalg.pinv(self.J_z)
        # form the schur complement
        self.L_schur = self.L_yy_num - self.L_yz_num @ self.J_z_inv @ self.J_y - self.J_y.T @ self.J_z_inv.T @ self.L_zy_num + self.J_y.T @ self.J_z_inv.T @ self.L_zz_num @ self.J_z_inv @ self.J_y
        # make sure it's symmetric
        self.L_schur = 0.5 * (self.L_schur + self.L_schur.T)
        R_scaled_inv = np.linalg.inv(R_scaled)
        self.L_schur_scaled = R_scaled_inv.T @ self.L_schur @ R_scaled_inv


        # compare with H_red (null space method)
        # form null space of active jacobian
        self.Z = null_space(self.active_jac_num)
        L_xx = casadi.hessian(self.lagrangian, x)[0]
        L_xx_fun = casadi.Function('L_xx_fun', input_to_casadi_func, [L_xx])
        L_xx_num = L_xx_fun(*input_to_casadi_func_num).toarray()
        self.L_red = self.Z.T @ L_xx_num @ self.Z

        # make sure it's symmetric
        self.L_red = 0.5 * (self.L_red + self.L_red.T)

        # convert to y coordinates
        z_y = - self.J_z_inv @ self.J_y
        # build S_y and S_z
        n=self.active_jac_num.shape[1]
        p=len(thetaind)
        q=n-p
        Sy = np.zeros((n, p));  Sy[thetaind, np.arange(p)] = 1.0
        Sz = np.zeros((n, q));  Sz[zind, np.arange(q)] = 1.0
        self.S = Sy + Sz @ z_y
        self.L_red_true = self.S.T @ L_xx_num @ self.S
        self.L_red_true = 0.5 * (self.L_red_true + self.L_red_true.T)

        # store condition number of hessian
        try:
            self.cond_hess = np.linalg.cond(self.L_schur)
        except np.linalg.LinAlgError:
            self.cond_hess = np.inf

        # classify definiteness of reduced hessian from its eigenvalues
        # eigvalsh returns real eigenvalues in ascending order (symmetric solver)
        try:
            w = np.linalg.eigvalsh(self.L_schur)
            if w.size == 0:
                self.eigcheck_hess = 'empty'         # no free variables
            else:
                tol = np.finfo(float).eps * np.abs(w).max() * self.L_schur.shape[0]
                if w[0] > tol:
                    self.eigcheck_hess = 'pos_def'
                elif w[-1] < -tol:
                    self.eigcheck_hess = 'neg_def'
                elif w[0] >= -tol:
                    self.eigcheck_hess = 'pos_semidef'   # all >= 0, at least one ~0
                elif w[-1] <= tol:
                    self.eigcheck_hess = 'neg_semidef'   # all <= 0, at least one ~0
                else:
                    self.eigcheck_hess = 'indefinite'    # eigenvalues of both signs
        except np.linalg.LinAlgError:
            self.eigcheck_hess = 'undefined'


    def cbox_solve(self, guess=None):
        self.initialize()
        if guess is None:
            guess = self.guess
        x = casadi.SX.sym('x', 9 * (self.n - 1))
        p = casadi.SX.sym('p', 7)
        res, alpha, total_econ = self.rootfinder_problem(casadi.vertcat(p, x))
        rootfinder = casadi.rootfinder('solver', 'newton', {'x': x, 'p': p, 'g': res})
        p_value = casadi.vertcat(self.alpha, self.beta, self.ta, self.pa, self.ne, self.na, self.no) 
        self.root = rootfinder(guess, p_value).toarray().flatten()
        self.total_econ = self.rootfinder_problem(casadi.vertcat(p_value, self.root))[2].toarray().flatten()

    _suppress_output = False
    _original_stdout = sys.stdout
    @staticmethod
    def no_print(enable):
        '''
        A nice static method to suppress output during the simulation,
        as needed. 
        '''
        if enable:
            if not Simulator._suppress_output:
                # Redirect stdout to null device
                Simulator._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                Simulator._suppress_output = True
        else:
            if Simulator._suppress_output:
                # Restore original stdout
                sys.stdout = Simulator._original_stdout
                Simulator._suppress_output = False
