import numpy as np
import casadi as cs


class NaiveOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, obstacles=None):
        self.params = model.params
        self.model = model
        self.nq = model.nq
        self.obstacles = obstacles

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
        
        if self.model.amodel.name == 'triple_pendulum': 
            Q = 1e-4 * np.eye(self.model.nx)
            Q[0, 0] = 5e2
            R = 1e-4 * np.eye(self.model.nu)
            x_ref = np.zeros(self.model.nx)
            x_ref[:self.model.nq] = np.pi
            x_ref[0] = self.params.q_max - 0.05
        else:
            Q = 1e2 * np.eye(self.model.np)
            R = 5e-3 * np.eye(self.model.nu)
            ee_ref = model.ee_ref
        dist_b = []
        for k in range(N + 1):
                
            if self.model.amodel.name == 'triple_pendulum':
                cost += (X[k] - x_ref).T @ Q @ (X[k] - x_ref)
            else:
                ee_pos = model.ee_fun(X[k])
                cost += (ee_pos - ee_ref).T @ Q @ (ee_pos - ee_ref)

            if k < N:
                cost += U[k].T @ R @ U[k]
                # Dynamics constraint
                opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
                # Torque constraints
                opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))

            if obstacles is not None and self.params.obs_flag:
                # Collision avoidance
                for obs in obstacles:
                    ee_pos = model.ee_fun(X[k])
                    if obs['name'] == 'floor':
                        lb = obs['bounds'][0]
                        ub = obs['bounds'][1]
                        opti.subject_to(opti.bounded(lb, ee_pos[2], ub))
                    elif obs['name'] == 'ball':
                        lb = obs['bounds'][0]
                        ub = obs['bounds'][1]
                        dist_b += [(ee_pos - obs['position']).T @ (ee_pos - obs['position'])]
                        opti.subject_to(opti.bounded(lb, dist_b[-1], ub))

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
            'ipopt.linear_solver': 'ma57',
            'ipopt.sb': 'yes'
        }

        opti.solver('ipopt', opts)
        return opti


class TerminalZeroVelOCP(NaiveOCP):
    def __init__(self, model, obstacles=None):
        super().__init__(model, obstacles)

    def additionalSetting(self):
        self.opti.subject_to(self.X[-1][self.nq:] == 0.)


class HardTerminalOCP(NaiveOCP):
    def __init__(self, model, obstacles=None):
        super().__init__(model, obstacles)

    def additionalSetting(self):
        self.model.setNNmodel()
        self.opti.subject_to(self.model.nn_func(self.X[-1]) >= 0.)


class SoftTerminalOCP(NaiveOCP):
    def __init__(self, model, obstacles=None):
        super().__init__(model, obstacles)

    def additionalSetting(self):
        s_N = self.opti.variable(1)
        self.model.setNNmodel()
        self.opti.subject_to(self.model.nn_func(self.X[-1]) + s_N >= 0.)
        self.opti.subject_to(
            self.opti.bounded(0., s_N, 1e6)
        )
        self.cost += self.params.ws_t * s_N


class SafeAbortOCP(NaiveOCP):
    def __init__(self, model, obstacles=None):
        super().__init__(model, obstacles)

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