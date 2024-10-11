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
        # self.adjustGuess(self.x_temp, self.u_temp)
        # TODO: compatible with the new version (see RecedingController)
        if status == 0 and self.model.checkSafeConstraints(self.x_temp[-1]) and \
                self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
                np.all([self.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
        else:
            if self.fails == 0:
                self.x_viable = np.copy(self.x_guess[-2])       
            if self.fails >= self.N:
                return None
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

    def additionalSetting(self):
        # Terminal constraint before, since it construct the nn model
        self.terminalConstraint()
        self.runningConstraint()

    def step(self, x):
        # Terminal constraint
        self.ocp_solver.cost_set(self.N, "zl", self.params.ws_t * np.ones((1,)))
        # Receding constraint
        self.ocp_solver.cost_set(self.r, "zl", self.params.ws_r * np.ones((1,)))
        for i in range(1, self.N):
            if i != self.r:
                # No constraints on other running states
                self.ocp_solver.cost_set(i, "zl", np.zeros((1,)))
        # Solve the OCP
        status = self.solve(x)

        self.r -= 1          
        for i in range(self.r + 2, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                self.r = i - 1

        if self.r == 1:
            self.x_viable = np.copy(self.x_guess[self.r])
            # self.x_viable = self.safeGuess(self.x_temp[0], self.u_temp, 1)[1]
            return None

        if status == 0 and self.model.checkStateConstraints(self.x_temp)  \
                and np.all([self.checkCollision(x) for x in self.x_temp]):
            self.fails = 0
        else:
            self.fails += 1
            
        return self.provideControl()


class SafeBackupController(AbstractController):
    def __init__(self, model):
        super().__init__(model)

    def additionalSetting(self):
        self.Q = np.zeros((self.model.nx, self.model.nx))
        self.Q[self.model.nq:, self.model.nq:] = np.eye(self.model.nv) * self.params.q_dot_gain

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        # TODO: must be introduced again
        # q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(self.model.nv)])
        # q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(self.model.nv)])

        # self.ocp.constraints.lbx_e = q_fin_lb
        # self.ocp.constraints.ubx_e = q_fin_ub
        # self.ocp.constraints.idxbx_e = np.arange(self.model.nx)
