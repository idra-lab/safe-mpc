import numpy as np
import casadi as cs
from copy import deepcopy
from .safe_set import NetSafeSet, AnalyticSafeSet


class NaiveOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model):
        self.params = model.params
        self.model = model
        self.nq = model.nq

        N = self.params.N
        opti = cs.Opti()
        x_init = opti.parameter(model.nx)
        cost = 0

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]
        for k in range(N):
            X += [opti.variable(model.nx)]
            opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
            U += [opti.variable(model.nu)]

        opti.subject_to(X[0] == x_init)
        
        Q = self.params.Q_weight * np.eye(model.ee_ref.shape[0])
        R = self.params.R_weight * np.eye(self.model.nu)
        ee_ref = model.ee_ref
        dist_b = []

        for k in range(N + 1):
                
            ee_pos = model.ee_fun(X[k])
            cost += (ee_pos - ee_ref).T @ Q @ (ee_pos - ee_ref)

            if k < N:
                cost += U[k].T @ R @ U[k]
                # Dynamics constraint
                opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
                # Torque constraints
                opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))

            for constr in self.model.collisions_constr_fun:
                opti.subject_to(opti.bounded(constr[1], constr[0](X[k]), constr[2]))

        opti.minimize(cost)
        self.opti = opti
        self.X = X
        self.U = U
        self.x_init = x_init    
        self.cost = cost
        self.dist_b = dist_b
        self.additionalSetting()
        self.reset_controller()

    def additionalSetting(self):
        self.net_name = ''

    def reset_controller(self):
        self.model.params.use_net = None
        
    def instantiateProblem(self):
        opti = self.opti
        opti.solver('ipopt', self.model.params.ipopt_opts)
        return opti


class TerminalZeroVelOCP(NaiveOCP):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        self.opti.subject_to(self.X[-1][self.nq:] == 0.)
        self.net_name = ''

class AccBoundsOCP(NaiveOCP):
    def __init__(self, model):
        super().__init__(model)
        
    def additionalSetting(self):
        nq = self.model.nq
        ddq_max = np.ones(self.model.nv) * 10.
        dq_min = - self.X[-1][nq:] ** 2 / ddq_max + self.X[-1][:nq]
        dq_max = self.X[-1][nq:] ** 2 / ddq_max + self.X[-1][:nq]
        self.opti.subject_to(dq_min >= self.model.x_min[:nq])        
        self.opti.subject_to(dq_max <= self.model.x_max[:nq])
        self.net_name = ''

class HardTerminalOCP(NaiveOCP):
    def __init__(self, model):
        super().__init__(model)

    def create_safe_set(self):
        # Analytic or network set
        if self.model.params.use_net == True:
            self.safe_set = NetSafeSet(self.model,cs.MX.sym('p',5))
            self.net_name = '_net'
        elif self.model.params.use_net == False: 
            self.safe_set = AnalyticSafeSet(self.model,cs.MX.sym('p',5))
            self.net_name = 'analytic_set'

    def additionalSetting(self):
        self.create_safe_set()
        safe_set_funs = self.safe_set.get_constraints_fun()
        safe_set_bounds = self.safe_set.get_bounds()
        for i,func in enumerate(safe_set_funs):
            self.opti.subject_to(self.opti.bounded(safe_set_bounds[i][0],func(self.X[-1]),safe_set_bounds[i][1]))
    
    def reset_controller(self):
        pass


class SoftTerminalOCP(HardTerminalOCP):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        self.create_safe_set()
        safe_set_funs = self.safe_set.get_constraints_fun()
        safe_set_bounds = self.safe_set.get_bounds()

                
        slack_vars = [] 
        for i,func in enumerate(safe_set_funs):
            slack_vars.append(self.opti.variable(1))
            self.opti.subject_to(self.opti.bounded(safe_set_bounds[i][0],func(self.X[-1])+slack_vars[-1],safe_set_bounds[i][1]))     
            self.opti.subject_to(self.opti.bounded(-1e6, slack_vars[-1], 1e6)) 
    
            self.cost += self.params.ws_t * slack_vars[-1]**2

    def reset_controller(self):
        pass

class SafeAbortOCP(NaiveOCP):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        cost = 0
        Q = 1e-4 * np.eye(self.model.nx)
        # Penalize only the velocity
        Q[self.model.nq:, self.model.nq:] = 1e4 * np.eye(self.model.nq)
        R = 5e-3 * np.eye(self.model.nu)
        for k in range(self.params.N):
            cost += self.X[k].T @ Q @ self.X[k] 
            cost += self.U[k].T @ R @ self.U[k]
        cost += self.X[-1].T @ Q @ self.X[-1]
        self.opti.minimize(cost)
        self.opti.subject_to(self.X[-1][self.nq:] == 0.)
        self.cost = cost

class InverseKinematicsOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, ee_pos,obstacles=None):
        self.params = model.params
        self.model = model
        self.nq = model.nq
        self.obstacles = obstacles

        opti = cs.Opti()
        cost = 0

        # Define decision variables
        X = []
        X += [opti.variable(model.nx)]

        # Constraints
        opti.subject_to(opti.bounded(model.x_min, X[0], model.x_max))
        opti.subject_to(self.model.ee_fun(X[0])==ee_pos)
        opti.subject_to(X[0][self.nq:]==0)

        for constr in self.model.collisions_constr_fun:
                opti.subject_to(opti.bounded(constr[1], constr[0](X[0]), constr[2]))

        opti.minimize(cost)
        self.opti = opti
        self.X = X    
        self.cost = cost

    def instantiateProblem(self):
        opti = self.opti
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 10e-6,
            'ipopt.constr_viol_tol': 1e-6,
            'ipopt.compl_inf_tol': 1e-6,
            'ipopt.hessian_approximation': 'limited-memory',
            # 'detect_simple_bounds': 'yes',
            'ipopt.max_iter': self.params.nlp_max_iter,
            #'ipopt.linear_solver': 'ma57',
            'ipopt.sb': 'yes'
        }

        opti.solver('ipopt', opts)
        return opti

