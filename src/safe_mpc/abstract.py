import numpy as np
import re
import scipy.linalg as lin
from casadi import SX, vertcat, repmat, fmax, norm_2
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver, AcadosOcp, AcadosOcpSolver


class AbstractModel:
    def __init__(self, params):
        self.amodel = AcadosModel()
        # Dummy dynamics (double integrator)
        self.amodel.name = "double_integrator"
        self.x = SX.sym("x")
        self.x_dot = SX.sym("x_dot")
        self.u = SX.sym("u")
        self.f_expl = self.u
        self.p = SX.sym("p")
        self.addDynamicsModel(params)
        self.amodel.f_expl_expr = self.f_expl
        self.amodel.x = self.x
        self.amodel.xdot = self.x_dot
        self.amodel.u = self.u
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = int(self.nx / 2)
        self.nv = self.nx - self.nq

        # Joint limits
        self.u_min = -params.u_max * np.ones(self.nu)
        self.u_max = params.u_max * np.ones(self.nu)

        self.x_min = np.hstack([params.q_min * np.ones(self.nq), -params.dq_max * np.ones(self.nq)])
        self.x_max = np.hstack([params.q_max * np.ones(self.nq), params.dq_max * np.ones(self.nq)])

        # NN model (viability constraint)
        self.nn_model = None

    def addDynamicsModel(self, params):
        pass

    def checkStateConstraints(self, x):
        return np.all((x >= self.x_min) & (x <= self.x_max))

    def checkControlConstraints(self, u):
        return np.all((u >= self.u_min) & (u <= self.u_max))

    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) and self.checkControlConstraints(u)

    def checkDynamicsConstraints(self, x, u):
        pass

    def checkSafeConstraints(self, x):
        return True if self.nn_model(x) >= 0. else False

    def getNNmodel(self, params, mean, std, safety_margin, x):
        vel_norm = fmax(norm_2(x[self.nq:]), 1e-3)
        mean_vec = vertcat([mean] * self.nq, [0] * self.nq)
        std_vec = vertcat([std] * self.nq, repmat(vel_norm, self.nq, 1))
        out = (x - mean_vec) / std_vec

        i = 0
        for param in params:
            param = SX(param.tolist())
            if i % 2 == 0:
                out = param @ out
            else:
                out = param + out
                if i == 1 or i == 3:
                    out = fmax(0., out)
            i += 1
        self.nn_model = out * (100 - safety_margin) / 100 - vel_norm


class SimDynamics:
    def __init__(self, params, model):
        self.model = model
        sim = AcadosSim()
        sim.model = model.model
        sim.solver_options.T = params.dt
        sim.solver_options.num_stages = 4
        sim.parameter_values = np.array([0.])
        self.integrator = AcadosSimSolver(sim, build=params.regenerate)

    def simulate(self, x, u):
        self.integrator.set("x", x)
        self.integrator.set("u", u)
        self.integrator.solve()
        x_next = self.integrator.get("x")
        return x_next


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

        # Additional settings, in general is an empty method
        self.additionalSetting(params)

        self.ocp.model = model.amodel
        clas_name = self.__class__.__name__
        gen_name = params.GEN_DIR + '/ocp_' + "".join(re.findall('[A-Z][a-z]*', clas_name)[:-1]).lower()
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=params.regenerate)
        # TODO: check if the RE and folder/json generation are correct
        # TODO: add GEN_DIR in params (actually is in definitions)

        # Initialize guess
        self.success = 0
        self.fails = 0
        self.time = 0
        self.x_ref = np.zeros(model.nx)

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

    def additionalSetting(self, params):
        pass

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
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

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.cost_set(self.N, 'yref', y_ref[:self.model.nx], api='new')
        self.ocp_solver.cost_set(self.N, 'W', self.Q, api='new')

        # Solve the OCP
        status = self.ocp_solver.solve()
        self.time = self.ocp_solver.get_stats('time_tot')

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.ocp_solver.get(i, "x")
            self.u_temp[i] = self.ocp_solver.get(i, "u")
        self.x_temp[-1] = self.ocp_solver.get(self.N, "x")

        return status

    def provideControl(self):
        """ Save the guess for the next MPC step and return u_opt[0] """
        if self.fails > 0:
            u = self.u_guess[0]
            # Rollback the previous guess
            self.x_guess = np.roll(self.x_guess, -1, axis=0)
            self.u_guess = np.roll(self.u_guess, -1, axis=0)
        else:
            u = self.u_temp[0]
            # Save the current temporary solution
            self.x_guess = np.roll(self.x_temp, -1, axis=0)
            self.u_guess = np.roll(self.u_temp, -1, axis=0)
        # Copy the last values
        self.x_guess[-1] = np.copy(self.x_guess[-2])
        self.u_guess[-1] = np.copy(self.u_guess[-2])
        return u

    def step(self, x0):
        pass

    def setQRWeights(self, Q, R):
        self.Q = Q
        self.R = R

    def setReference(self, x_ref):
        self.x_ref = x_ref

    def getTime(self):
        return self.time

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return self.x_guess, self.u_guess
