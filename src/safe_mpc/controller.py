import numpy as np
import scipy.linalg as lin
from .abstract import AbstractController
from casadi import Function


class NaiveController(AbstractController):
    def __init__(self, model, obstacles=None):
        super().__init__(model, obstacles)

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.model.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               np.all([self.checkCollision(x) for x in self.x_temp])

    def initialize(self, x0, u0=None):
        # Trivial guess
        self.x_guess = np.full((self.N + 1, self.model.nx), x0)
        if u0 is None:
            u0 = np.zeros(self.model.nu)
        self.u_guess = np.full((self.N, self.model.nu), u0)
        # Solve the OCP
        status = self.solve(x0)
        if status == 0 and self.checkGuess():
            self.x_guess = np.copy(self.x_temp)
            self.u_guess = np.copy(self.u_temp)
            return 1
        return 0

    def step(self, x):
        status = self.solve(x)
        if status == 0:
            self.fails = 0
        else:
            self.fails += 1
        return self.provideControl()
    
    
class TerminalZeroVelocity(NaiveController):
    """ Naive MPC with zero terminal velocity as constraint """
    def __init__(self, model, obstacles=None):
        super().__init__(model, obstacles)

    def additionalSetting(self):
        x_min_e = np.hstack((self.model.x_min[:self.model.nq], np.zeros(self.model.nv)))
        x_max_e = np.hstack((self.model.x_max[:self.model.nq], np.zeros(self.model.nv)))

        self.ocp.constraints.lbx_e = x_min_e
        self.ocp.constraints.ubx_e = x_max_e
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)


class RecedingAccBounds(NaiveController):
    """ MPC with acceleration bounds """
    def __init__(self, model, obstacles=None):
        super().__init__(model, obstacles)  
        self.abort_flag = self.params.abort_flag

    def additionalSetting(self):
        num_nl_e = np.sum([c.shape[0] for c in self.nl_con_e])
        num_nl = np.sum([c.shape[0] for c in self.nl_con])
        num_nl_0 = np.sum([c.shape[0] for c in self.nl_con_0])
        # Acc bounds constraints
        nq = self.model.nq
        self.ddq_max = np.ones(self.model.nv) * 10. 
        self.dq_min = - self.model.x[nq:] ** 2 / self.ddq_max + self.model.x[:nq]       # >= q_min
        self.dq_max = self.model.x[nq:] ** 2 / self.ddq_max + self.model.x[:nq]         # <= q_max

        self.min_vel = Function('min_vel', [self.model.x], [self.dq_min])
        self.max_vel = Function('max_vel', [self.model.x], [self.dq_max])

        # Soft terminal constraint
        self.nl_con_e.append(self.dq_min)
        self.nl_lb_e.append(self.model.x_min[:nq])
        self.nl_ub_e.append(np.ones(nq) * 1e6)

        self.nl_con_e.append(self.dq_max)
        self.nl_lb_e.append(-np.ones(nq) * 1e6)
        self.nl_ub_e.append(self.model.x_max[:nq])

        self.ocp.constraints.idxsh_e = np.arange(nq * 2) + num_nl_e
        self.ocp.cost.zl_e = np.ones(nq * 2) * self.params.ws_t
        self.ocp.cost.zu_e = np.zeros(nq * 2)
        self.ocp.cost.Zl_e = np.zeros(nq * 2)
        self.ocp.cost.Zu_e = np.zeros(nq * 2)

        # Hard receding constraints
        self.nl_con_0.append(self.dq_min)
        self.nl_lb_0.append(-np.ones(nq) * 1e6)
        self.nl_ub_0.append(np.ones(nq) * 1e6)
        self.nl_con_0.append(self.dq_max)
        self.nl_lb_0.append(-np.ones(nq) * 1e6)
        self.nl_ub_0.append(np.ones(nq) * 1e6) 

        self.idx_h_0 = np.arange(nq * 2) + num_nl_0
        self.nl_con.append(self.dq_min)
        self.nl_lb.append(-np.ones(nq) * 1e6)
        self.nl_ub.append(np.ones(nq) * 1e6)
        self.nl_con.append(self.dq_max)
        self.nl_lb.append(-np.ones(nq) * 1e6)
        self.nl_ub.append(np.ones(nq) * 1e6)

        self.idx_h = np.arange(nq * 2) + num_nl
        self.idx_h_e = np.arange(nq * 2) + num_nl_e
        # dq_min lower bounded, dq_max upper bounded
        self.lh_rec = np.hstack([self.model.x_min[:nq], -np.ones(nq) * 1e6])
        self.uh_rec = np.hstack([np.ones(nq) * 1e6, self.model.x_max[:nq]])

    def checkVelocityViability(self, x):
        q_min = self.model.x_min[:self.model.nq]
        q_max = self.model.x_max[:self.model.nq]
        return np.all(self.min_vel(x) >= q_min) and np.all(self.max_vel(x) <= q_max)

    def step(self, x):
        # Terminal constraint --> already set, Receding constraint
        lh, uh = self.ocp.constraints.lh, self.ocp.constraints.uh
        if self.r == self.N:
            lh_r, uh_r = np.copy(self.ocp.constraints.lh_e), np.copy(self.ocp.constraints.uh_e)
            lh_r[self.idx_h_e], uh_r[self.idx_h_e] = self.lh_rec, self.uh_rec
        else:
            lh_r, uh_r = np.copy(lh), np.copy(uh)  
            lh_r[self.idx_h], uh_r[self.idx_h] = self.lh_rec, self.uh_rec  
        self.ocp_solver.constraints_set(self.r, "lh", lh_r)
        self.ocp_solver.constraints_set(self.r, "uh", uh_r)
        for i in range(self.N):
            if i != self.r:
                # No constraints on other running states
                self.ocp_solver.constraints_set(i, "lh", lh)
                self.ocp_solver.constraints_set(i, "uh", uh)
        # Solve the OCP
        status = self.solve(x)

        if self.abort_flag:
            self.r -= 1          
        else:
            if self.r > 0:
                self.r -= 1
        for i in range(self.r + 2, self.N + 1):
            if self.checkVelocityViability(self.x_temp[i]):
                self.r = i - 1

        if self.r == 0 and self.abort_flag:
            self.x_viable = np.copy(self.x_guess[1])
            return self.u_guess[0], True

        if status == 0 and self.model.checkStateConstraints(self.x_temp)  \
                and np.all([self.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
        else:
            self.fails += 1
            
        return self.provideControl()
    

class STController(NaiveController):
    def __init__(self, model, obstacles):
        super().__init__(model, obstacles)

    def additionalSetting(self):
        self.terminalConstraint()


class STWAController(STController):
    def __init__(self, model, obstacles):
        super().__init__(model, obstacles)
        self.x_viable = None

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.model.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               self.model.checkSafeConstraints(self.x_temp[-1]) and \
               np.all([self.checkCollision(x) for x in self.x_temp])

    def step(self, x):
        status = self.solve(x)
        if status == 0 and self.model.checkStateConstraints(self.x_temp) and \
                np.all([self.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
        else:
            if self.fails == 0:
                self.x_viable = np.copy(self.x_guess[-2])       
            if self.fails >= self.N:
                return self.u_guess[0], True
            self.fails += 1
        return self.provideControl()

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess
        self.x_viable = x_guess[-1]


class HTWAController(STWAController):
    def __init__(self, model, obstacles):
        super().__init__(model, obstacles)

    def additionalSetting(self):
        self.terminalConstraint(soft=False)


class RecedingController(STWAController):
    def __init__(self, model, obstacles):
        super().__init__(model, obstacles)
        self.r = self.N
        self.r_last = self.N
        self.abort_flag = self.params.abort_flag

    def additionalSetting(self):
        # Terminal constraint before, since it construct the nn model
        self.terminalConstraint()
        self.runningConstraint()

    def step(self, x):
        # Terminal constraint
        self.ocp_solver.cost_set(self.N, "zl", self.params.ws_t * np.ones((1,)))
        # Receding constraint
        self.ocp_solver.cost_set(self.r, "zl", self.params.ws_r * np.ones((1,)))
        # self.ocp_solver.constraints_set(self.r, "lh", lh_rec)
        for i in range(self.N):
            if i != self.r:
                # No constraints on other running states
                self.ocp_solver.cost_set(i, "zl", np.zeros((1,)))
                # self.ocp_solver.constraints_set(i, "lh", lh)
        # Solve the OCP
        status = self.solve(x)

        if self.abort_flag:
            self.r -= 1          
        else:
            if self.r > 0:
                self.r -= 1
        for i in range(self.r + 2, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                self.r = i - 1

        if self.r == 0 and self.abort_flag:
            self.x_viable = np.copy(self.x_guess[1])
            return self.u_guess[0], True

        if status == 0 and self.model.checkStateConstraints(self.x_temp)  \
                and np.all([self.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
        else:
            self.fails += 1
            
        return self.provideControl()

class SafeBackupController(AbstractController):
    def __init__(self, model, obstacles):
        super().__init__(model, obstacles)

    def additionalSetting(self):
        # Linear LS cost
        self.ocp.cost.cost_type = 'LINEAR_LS'        
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.Q = np.zeros((self.model.nx, self.model.nx))
        # Penalize only the velocity
        self.Q[self.model.nq:, self.model.nq:] = np.eye(self.model.nv) * self.params.q_dot_gain

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        self.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        self.ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
        self.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        self.ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)
        self.ocp.cost.Vx_e = np.eye(self.model.nx)

        self.ocp.cost.yref = np.zeros(self.model.ny)
        self.ocp.cost.yref_e = np.zeros(self.model.nx)

        # Terminal constraint --> zero velocity
        q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(self.model.nv)])
        q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(self.model.nv)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)
