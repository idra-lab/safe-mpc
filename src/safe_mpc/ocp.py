import numpy as np
import casadi as cs
from copy import deepcopy


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
        
        Q = self.params.Q
        R = self.params.R * np.eye(self.model.nu)
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

            for pair in self.params.collisions_pairs:
                if pair['type'] == 0:
                    dist = self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk_fun'](X[k]),*pair['elements'][1]['end_points_fk_fun'](X[k]))
                    lb = (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2
                    ub = 1e6
                    opti.subject_to(opti.bounded(lb, dist, ub))
                elif pair['type'] == 1:
                    dist = self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk_fun'](X[k]),pair['elements'][0]['length'],pair['elements'][1]['position'])
                    lb = (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2
                    ub = 1e6
                    opti.subject_to(opti.bounded(lb, dist, ub))
                elif pair['type'] == 2:
                    for point in pair['elements'][0]['end_points_fk_fun'](X[k]):
                        lb = pair['elements'][1]['bounds'][0]
                        ub = pair['elements'][1]['bounds'][1]
                        opti.subject_to(opti.bounded(lb, point[2], ub))

        opti.minimize(cost)
        self.opti = opti
        self.X = X
        self.U = U
        self.x_init = x_init    
        self.cost = cost
        self.dist_b = dist_b
        self.additionalSetting()

    def additionalSetting(self):
        pass

    def instantiateProblem(self):
        opti = self.opti
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
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


class TerminalZeroVelOCP(NaiveOCP):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        self.opti.subject_to(self.X[-1][self.nq:] == 0.)

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


class HardTerminalOCP(NaiveOCP):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        safe_set_funs = self.model.safe_set.get_constraints_fun()
        safe_set_bounds = self.model.safe_set.get_bounds()
        for i,func in enumerate(safe_set_funs):
            self.opti.subject_to(self.opti.bounded(safe_set_bounds[i][0],func(self.X[-1]),safe_set_bounds[i][1]))


class SoftTerminalOCP(NaiveOCP):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        safe_set_funs = self.model.safe_set.get_constraints_fun()
        safe_set_bounds = self.model.safe_set.get_bounds()
        slack_vars =[] 
        for i,func in enumerate(safe_set_funs):
            slack_vars.append(self.opti.variable(1))
            self.opti.subject_to(self.opti.bounded(safe_set_bounds[i][0],func(self.X[-1])+slack_vars[-1],safe_set_bounds[i][1]))        
            self.opti.subject_to(
                self.opti.bounded(0., slack_vars[-1], 1e6)
            )
            self.cost += self.params.ws_t * slack_vars[-1]

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

        opti.minimize(cost)
        self.opti = opti
        self.X = X    
        self.cost = cost

    def checkCollision(self, x):
        if self.obstacles is not None and self.params.obs_flag:
            t_glob = self.model.jointToEE(x) 
            for obs in self.obstacles:
                if obs['name'] == 'floor':
                    if t_glob[2] + self.params.tol_obs < obs['bounds'][0]:
                        return False
                elif obs['name'] == 'ball':
                    dist_b = np.sum((t_glob.flatten() - obs['position']) ** 2)
                    if dist_b + self.params.tol_obs < obs['bounds'][0]:
                        return False
        return True

    def instantiateProblem(self):
        opti = self.opti
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
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

