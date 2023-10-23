import numpy as np
import scipy.linalg as lin
from acados_template import AcadosOcp, AcadosOcpSolver


class AbstractController:
    def __init__(self, params, model):

        self.model = model
        self.N = params.N
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = params.T
        self.ocp.dims.N = params.N

        # Cost
        self.Q = 1e-4 * np.eye(model.nx)
        self.Q[0, 0] = 5e2
        self.R = 1e-4 * np.eye(model.nu)

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((model.ny, model.nx))
        self.ocp.cost.Vx[:model.nx, :model.nx] = np.eye(model.nx)
        self.ocp.cost.Vu = np.zeros((model.ny, model.nu))
        self.ocp.cost.Vu[model.nx:, :model.nu] = np.eye(model.nu)
        self.ocp.cost.Vx_e = np.eye(model.nx)

        self.ocp.cost.yref = np.zeros(model.ny)
        self.ocp.cost.yref_e = np.zeros(model.nx)
        self.ocp.parameter_values = np.array([0.])

        # Constraints
        self.ocp.constraints.lbx_0 = model.x_min
        self.ocp.constraints.ubx_0 = model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(model.nx)

        self.ocp.constraints.lbu = model.u_min
        self.ocp.constraints.ubu = model.u_max
        self.ocp.constraints.idxbu = np.arange(model.nu)
        self.ocp.constraints.lbx = model.x_min
        self.ocp.constraints.ubx = model.x_max
        self.ocp.constraints.idxbx = np.arange(model.nx)

        self.ocp.constraints.lbx_e = model.x_min
        self.ocp.constraints.ubx_e = model.x_max
        self.ocp.constraints.idxbx_e = np.arange(model.nx)

        # Solver options
        self.ocp.solver_options.nlp_solver_type = params.solver_type
        self.ocp.solver_options.qp_solver_iter_max = params.qp_max_iter
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        # Addiotional settings, in general is an empty method
        self.additionalSetting()

        self.ocp.model = model.model
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=params.regenerate)

        self.fails = 0
        self.time = 0
        self.x_ref = np.zeros(model.nx)

    def additionalSetting(self):
        pass

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state (check if we can use directly property x0)
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        y_ref = np.zeros(self.model.ny)
        y_ref[:self.model.nx] = self.x_ref
        W = lin.block_diag(self.Q, self.R)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.cost_set(i, 'yref', y_ref, api='new')
            self.ocp_solver.cost_set(i, 'W', W, api='new')

        self.ocp_solver.set(self.N, 'x', self.x_guess[self.N])
        self.ocp_solver.cost_set(self.N, 'yref', y_ref[:self.model.nx], api='new')
        self.ocp_solver.cost_set(self.N, 'W', self.Q, api='new')

        # Solve the OCP
        status = self.ocp_solver.solve()
        self.time = self.ocp_solver.get_stats('time_tot')
        return status
    
    def step(self, x0):
        pass
    
    def setQRWeights(self, Q, R):
        self.Q = Q
        self.R = R

    def setReference(self, x_ref):
        self.x_ref = x_ref

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getTime(self):
        return self.time
    
    def getGuess(self):
        return self.x_guess, self.u_guess