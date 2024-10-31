import numpy as np
import scipy.linalg as lin
from .abstract import AbstractController


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
