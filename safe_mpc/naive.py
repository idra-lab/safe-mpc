import numpy as np
from safe_mpc.controller import AbstractController


class NaiveController(AbstractController):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.success = 0

    def initialize(self, x0):
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))

        status = self.solve(x0)

        if status == 0 or status == 2:
            for i in range(self.N):
                self.x_guess[i] = self.ocp_solver.get(i, "x")
                self.u_guess[i] = self.ocp_solver.get(i, "u")
            self.x_guess[self.N] = self.ocp_solver.get(self.N, "x")

            if self.model.checkStateConstraints(self.x_guess) and \
               self.model.checkControlConstraints(self.u_guess):
                self.success += 1
            else:
                self.x_guess *= 0
                self.u_guess *= 0

    def step(self, x):
        status = self.solve(x)
        u = self.ocp_solver.get(0, "u")
        if status == 0 and self.model.checkControlConstraints(u):
            self.fails = 0
            for i in range(self.N-1):
                self.x_guess[i] = self.ocp_solver.get(i+1, "x")
                self.u_guess[i] = self.ocp_solver.get(i+1, "u")
            self.x_guess[self.N] = self.ocp_solver.get(self.N, "x")
        else:
            if self.fails >= self.N:
                return None
            u = self.u_guess[0]
            self.fails += 1
            self.x_guess = np.roll(self.x_guess, -1, axis=0)
            self.u_guess = np.roll(self.u_guess, -1, axis=0)
        
        self.x_guess[-1] = np.copy(self.x_guess[-2])
        self.u_guess[-1] = np.copy(self.u_guess[-2])
        return u