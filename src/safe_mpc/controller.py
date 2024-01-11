import numpy as np
from .abstract import AbstractController


class NaiveController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp)

    def initialize(self, x0, u0=None):
        flag = 0
        # Trivial guess
        self.x_guess = np.full((self.N + 1, self.model.nx), x0)
        if u0 is None:
            u0 = np.zeros(self.model.nu)
        self.u_guess = np.full((self.N, self.model.nu), u0)
        # Solve the OCP
        status = self.solve(x0)
        if (status == 0 or status == 2) and self.checkGuess():
            self.x_guess = np.copy(self.x_temp)
            self.u_guess = np.copy(self.u_temp)
            return 1
        return 0

    def step(self, x):
        status = self.solve(x)
        if status == 0 and self.model.checkControlConstraints(self.u_temp[0]) and \
           self.simulator.checkDynamicsConstraints(self.x_temp[:2], np.array([self.u_temp[0]])):
            self.fails = 0
        else:
            if self.fails >= self.N:
                return None
            self.fails += 1
        return self.provideControl()


class STController(NaiveController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.terminalConstraint()


class STWAController(STController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.x_viable = None

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               self.model.checkSafeConstraints(self.x_temp[-1])

    def step(self, x):
        status = self.solve(x)
        if status == 0 and self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
                self.model.checkSafeConstraints(self.x_temp[-1]):
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

    def getLastViableState(self):
        return np.copy(self.x_viable)


class HTWAController(STWAController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.terminalConstraint(soft=False)


class RecedingController(STWAController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.r = self.N

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

        r_new = -1
        for i in range(1, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                r_new = i - 1

        if status == 0 and self.model.checkRunningConstraints(self.x_temp, self.u_temp) and r_new > 0:
            self.fails = 0
            self.r = r_new
        else:
            if self.r == 1:
                self.x_viable = np.copy(self.x_guess[self.r])
                return None
            self.fails += 1
            self.r -= 1
        return self.provideControl()
