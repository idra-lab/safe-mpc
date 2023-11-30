import numpy as np
from .abstract import AbstractController


class NaiveController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def checkGuess(self, x, u):
        return self.model.checkRunningConstraints(x, u)

    def initialize(self, x0):
        # Trivial guess
        self.x_guess = np.full((self.N + 1, self.model.nx), x0)
        self.u_guess = np.zeros((self.N, self.model.nu))
        # Solve the OCP
        status = self.solve(x0)
        # self.ocp_solver.print_statistics()
        # print('status = ', status)
        if status == 0 or status == 2:
            for i in range(self.N):
                self.x_guess[i] = self.ocp_solver.get(i, "x")
                self.u_guess[i] = self.ocp_solver.get(i, "u")
            self.x_guess[self.N] = self.ocp_solver.get(self.N, "x")

            if self.checkGuess(self.x_guess, self.u_guess): #and \
               # self.simulator.checkDynamicsConstraints(self.x_guess, self.u_guess):
                self.success += 1
            else:
                self.x_guess *= np.nan
                self.u_guess *= np.nan

    def step(self, x):
        status = self.solve(x)
        u = self.ocp_solver.get(0, "u")
        if status == 0 and self.model.checkControlConstraints(u):
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
        self.x_viable = None                # TODO: to be initialized

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


class HTWAController(STWAController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.terminalConstraint(soft=False)

    def checkGuess(self, x, u):
        # Check also the terminal constraint
        return self.model.checkRunningConstraints(x, u) and self.model.checkSafeConstraints(x[-1])


class RecedingController(NaiveController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.x_viable = None                # TODO: to be initialized
        self.r = self.N

    def additionalSetting(self):
        # Terminal constraint before, since it construct the nn model
        self.terminalConstraint()
        self.runningConstraint()

    def step(self, x):
        # Terminal constraint
        self.ocp.ocp_solver.cost_set(self.N, "Zl", 1e5 * np.ones((1,)))
        # Receding constraint
        self.ocp.ocp_solver.cost_set(self.r, "Zl", 1e8 * np.ones((1,)))
        for i in range(1, self.N):
            if i != self.r:
                # No constraints on other running states
                self.ocp.ocp_solver.cost_set(i, "Zl", np.zeros((1,)))
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
