import numpy as np
import re
from copy import deepcopy
import scipy.linalg as lin
from casadi import MX, vertcat, norm_2, fmax, Function
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver, AcadosOcp, AcadosOcpSolver
import torch
import torch.nn as nn
import l4casadi as l4c


class NeuralNetDIR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetDIR, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


class AbstractModel:
    def __init__(self, params):
        self.params = params
        self.amodel = AcadosModel()
        # Dummy dynamics (double integrator)
        self.amodel.name = "double_integrator"
        self.x = MX.sym("x")
        self.x_dot = MX.sym("x_dot")
        self.u = MX.sym("u")
        self.f_expl = self.u
        self.p = MX.sym("p")
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
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

    def addDynamicsModel(self, params):
        pass

    def checkStateConstraints(self, x):
        return np.all((x >= self.x_min) & (x <= self.x_max))

    def checkControlConstraints(self, u):
        return np.all((u >= self.u_min) & (u <= self.u_max))

    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) and self.checkControlConstraints(u)

    def checkSafeConstraints(self, x):
        return True if self.nn_func(x, self.params.alpha) >= 0. else False

    def setNNmodel(self):
        device = torch.device('cuda')
        model = NeuralNetDIR(self.nx, (self.nx - 1) * 100, 1).to(device)
        model.load_state_dict(torch.load(self.params.NN_DIR + 'model_3dof_vboc', map_location=device))
        mean = torch.load(self.params.NN_DIR + 'mean_3dof_vboc')
        std = torch.load(self.params.NN_DIR + 'std_3dof_vboc')

        x_cp = deepcopy(self.x)
        x_cp[self.nq] += 1e-6
        vel_norm = norm_2(x_cp[self.nq:])
        pos = (x_cp[:self.nq] - mean) / std
        vel_dir = x_cp[self.nq:] / vel_norm
        state = vertcat(pos, vel_dir)

        # i = 0
        # out = state
        # weights = list(model.parameters())
        # for weight in weights:
        #     weight = MX(np.array(weight.tolist()))
        #     if i % 2 == 0:
        #         out = weight @ out
        #     else:
        #         out = weight + out
        #         if i == 1 or i == 3:
        #             out = fmax(0., out)
        #     i += 1
        # self.nn_model = out * (100 - self.p) / 100 - vel_norm

        self.l4c_model = l4c.L4CasADi(model,
                                      device='cuda',
                                      name=self.amodel.name + '_model',
                                      build_dir=self.params.GEN_DIR + 'nn_' + self.amodel.name)
        self.nn_model = self.l4c_model(state) * (100 - self.p) / 100 - vel_norm
        self.nn_func = Function('nn_func', [self.x, self.p], [self.nn_model])


class SimDynamics:
    def __init__(self, model):
        self.model = model
        self.params = model.params
        sim = AcadosSim()
        sim.model = model.amodel
        sim.solver_options.T = self.params.dt_s
        sim.solver_options.num_stages = self.params.integrator_type
        sim.parameter_values = np.array([0.])
        gen_name = self.params.GEN_DIR + '/sim_' + sim.model.name
        sim.code_export_directory = gen_name
        self.integrator = AcadosSimSolver(sim, build=self.params.regenerate, json_file=gen_name + '.json')

    def simulate(self, x, u):
        self.integrator.set("x", x)
        self.integrator.set("u", u)
        self.integrator.solve()
        x_next = self.integrator.get("x")
        return x_next

    def checkDynamicsConstraints(self, x, u):
        # Rollout the control sequence
        n = np.shape(u)[0]
        x_sim = np.zeros((n + 1, self.model.nx))
        x_sim[0] = np.copy(x[0])
        for i in range(n):
            x_sim[i + 1] = self.simulate(x_sim[i], u[i])
        # Check if the rollout state trajectory is almost equal to the optimal one
        return True if np.linalg.norm(x - x_sim) < 1e-5 * np.sqrt(n+1) else False


class AbstractController:
    def __init__(self, simulator):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.simulator = simulator
        self.params = simulator.params
        self.model = simulator.model

        self.N = self.params.N
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.T
        self.ocp.dims.N = self.params.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.Q = 1e-4 * np.eye(self.model.nx)
        self.Q[0, 0] = 5e2
        self.R = 1e-4 * np.eye(self.model.nu)

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        self.ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
        self.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        self.ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)
        self.ocp.cost.Vx_e = np.eye(self.model.nx)

        self.ocp.cost.yref = np.zeros(self.model.ny)
        self.ocp.cost.yref_e = np.zeros(self.model.nx)
        # Set alpha to zero as default
        self.ocp.parameter_values = np.array([0.])

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)
        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Solver options
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        # self.ocp.solver_options.levenberg_marquardt = 1e2

        # Additional settings, in general is an empty method
        self.additionalSetting()

        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.regenerate)

        # Initialize guess
        self.success = 0
        self.fails = 0
        self.time = 0
        self.x_ref = np.zeros(self.model.nx)

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

    def additionalSetting(self):
        pass

    def terminalConstraint(self, soft=True):
        self.model.setNNmodel()
        self.model.amodel.con_h_expr_e = self.model.nn_model

        self.ocp.solver_options.model_external_shared_lib_dir = self.model.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.model.l4c_model.name

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        if soft:
            self.ocp.constraints.idxsh_e = np.array([0])

            self.ocp.cost.zl_e = np.ones((1,)) * self.params.ws_t
            self.ocp.cost.zu_e = np.zeros((1,))
            self.ocp.cost.Zl_e = np.zeros((1,))
            self.ocp.cost.Zu_e = np.zeros((1,))

    def runningConstraint(self, soft=True):
        # Suppose that the NN model is already set (same for external model shared lib)
        self.model.amodel.con_h_expr = self.model.nn_model

        self.ocp.constraints.lh = np.array([0.])
        self.ocp.constraints.uh = np.array([1e6])

        if soft:
            self.ocp.constraints.idxsh = np.array([0])

            # Set zl initially to zero, then apply receding constraint in the step method
            self.ocp.cost.zl = np.zeros((1,))
            self.ocp.cost.zu = np.zeros((1,))
            self.ocp.cost.Zl = np.zeros((1,))
            self.ocp.cost.Zu = np.zeros((1,))

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
        fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
                  'time_glob', 'time_reg', 'time_tot']
        return np.array([self.ocp_solver.get_stats(field) for field in fields])
        # return np.copy(self.time)

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)
