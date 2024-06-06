import re
import numpy as np
from copy import deepcopy
import scipy.linalg as lin
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
from casadi import MX, vertcat, norm_2, Function
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import torch
import torch.nn as nn
import l4casadi as l4c


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU()):
        super().__init__()
        # TODO: name of stack must match the one in the training script
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )

    def forward(self, x):
        out = self.linear_stack(x)
        return out


class AdamModel:
    def __init__(self, params, n_dofs=False):
        self.params = params
        self.amodel = AcadosModel()
        # Robot dynamics with Adam (IIT)
        robot = URDF.from_xml_file(params.robot_urdf)
        try:
            n_dofs = n_dofs if n_dofs else len(robot.joints)
            if n_dofs > len(robot.joints) or n_dofs < 1:
                raise ValueError
        except ValueError:
            print(f'\nInvalid number of degrees of freedom! Must be > 1 and <= {len(robot.joints)}\n')
            exit()
        robot_joints = robot.joints[1:n_dofs+1] if params.urdf_name == 'z1' else robot.joints[:n_dofs]
        joint_names = [joint.name for joint in robot_joints]
        kin_dyn = KinDynComputations(params.robot_urdf, joint_names, robot.get_root())        
        kin_dyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
        self.mass = kin_dyn.mass_matrix_fun()                           # Mass matrix
        self.bias = kin_dyn.bias_force_fun()                            # Nonlinear effects  
        self.gravity = kin_dyn.gravity_term_fun()                       # Gravity vector
        self.fk = kin_dyn.forward_kinematics_fun(params.frame_name)     # Forward kinematics
        nq = len(joint_names)

        self.amodel.name = params.urdf_name
        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nq)
        self.p = MX.sym("p", 1)
        self.f_disc = vertcat(
            self.x[:nq] + params.dt * self.x[nq:] + 0.5 * params.dt**2 * self.u,
            self.x[nq:] + params.dt * self.u
        ) 
            
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.disc_dyn_expr = self.f_disc
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints]) 
        joint_effort = np.array([joint.limit.effort for joint in robot_joints]) 

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])
        self.eps = params.state_tol

        # Target
        self.x_ref = np.zeros(self.nx)
        self.x_ref[:self.nq] = np.pi
        self.x_ref[params.joint_target] = joint_upper[params.joint_target] - params.ubound_gap

        # NN model (viability constraint)
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

        # Cartesian constraint
        self.t_loc = np.array([0., 0., 0.2])
        self.z_bounds = np.array([-0.25, 1e6])

    def checkStateConstraints(self, x):
        return np.all(np.logical_and(x >= self.x_min + self.eps, x <= self.x_max - self.eps))

    def checkControlConstraints(self, u):
        return np.all(np.logical_and(u >= self.tau_min, u <= self.tau_max))

    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) #and self.checkControlConstraints(u)

    def checkSafeConstraints(self, x):
        return self.nn_func(x, self.params.alpha) >= 0. 
    
    def integrate(self, x, u):
        x_next = np.zeros(self.nx)
        x_next[:self.nq] = x[:self.nq] + self.params.dt * x[self.nq:] + 0.5 * self.params.dt**2 * u
        x_next[self.nq:] = x[self.nq:] + self.params.dt * u
        return x_next

    def checkDynamicsConstraints(self, x, u):
        # Rollout the control sequence
        n = np.shape(u)[0]
        x_sim = np.zeros((n + 1, self.nx))
        x_sim[0] = np.copy(x[0])
        for i in range(n):
            x_sim[i + 1] = self.integrate(x_sim[i], u[i])
        # Check if the rollout state trajectory is almost equal to the optimal one
        return np.linalg.norm(x - x_sim) < self.eps * np.sqrt(n+1) 
    
    def setNNmodel(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeuralNetwork(self.nx, (self.nx - 1) * 100, 1).to(device)
        nn_data = torch.load(f'{self.params.NN_DIR}model_{self.nq}dof.pt')
        model.load_state_dict(nn_data['model'])

        x_cp = deepcopy(self.x)
        x_cp[self.nq] += self.eps
        vel_norm = norm_2(x_cp[self.nq:])
        pos = (x_cp[:self.nq] - nn_data['mean']) / nn_data['std']
        vel_dir = x_cp[self.nq:] / vel_norm
        state = vertcat(pos, vel_dir)

        self.l4c_model = l4c.L4CasADi(model,
                                      device='cuda',
                                      name=f'{self.amodel.name}_model',
                                      build_dir=f'{self.params.GEN_DIR}nn_{self.amodel.name}')
        # TODO: try to simplify the expression, take only the function
        self.nn_model = self.l4c_model(state) * (100 - self.p) / 100 - vel_norm
        self.nn_func = Function('nn_func', [self.x, self.p], [self.nn_model])


class AbstractController:
    def __init__(self, model):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model

        self.N = int(self.params.T / self.params.dt)
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.T
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.Q = 1e-4 * np.eye(self.model.nx)
        self.R = 1e-4 * np.eye(self.model.nu)
        self.Q[0, 0] = 5e2
        self.Q[2, 2] = 1e2
        self.Q[1, 1] = 1e2

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
        # Minimize effort
        H_b = np.eye(4)
        tau = self.model.mass(H_b, self.model.x[:self.model.nq])[6:, 6:] @ self.model.u + \
              self.model.bias(H_b, self.model.x[:self.model.nq], np.zeros(6), self.model.x[self.model.nq:])[6:]
        # self.Q = 5e-2 * np.eye(self.model.nx)
        # self.R = 1e0 * np.eye(self.model.nu)

        # self.ocp.cost.cost_type_0 = "EXTERNAL"
        # self.ocp.cost.cost_type = "EXTERNAL"

        # min_effort = self.model.x.T @ self.Q @ self.model.x + tau.T @ self.R @ tau 
        # self.ocp.model.cost_expr_ext_cost_0 = min_effort
        # self.ocp.model.cost_expr_ext_cost = min_effort

        # Set alpha to zero as default
        self.ocp.parameter_values = np.array([0.])

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)
        
        # Nonlinear constraint
        self.nl_con, self.nl_con_e = [], []
        self.nl_lb, self.nl_lb_e = [], []
        self.nl_ub, self.nl_ub_e = [], []
        
        # --> dynamics (only on running nodes)
        
        self.nl_con.append(tau)
        self.nl_lb.append(self.model.tau_min)
        self.nl_ub.append(self.model.tau_max)
        
        # --> collision (both on running and terminal nodes)
        if self.params.obs_flag:
            T_ee = self.model.fk(H_b, self.model.x[:self.model.nq]) 
            T_ee[:3, 3] += T_ee[:3, :3] @ self.model.t_loc
            self.nl_con.append(T_ee[2, 3])
            self.nl_con_e.append(T_ee[2, 3])

            self.nl_lb.append(self.model.z_bounds[0])
            self.nl_ub.append(self.model.z_bounds[1])
            self.nl_lb_e.append(self.model.z_bounds[0])
            self.nl_ub_e.append(self.model.z_bounds[1])

        # Additional settings, in general is an empty method
        self.additionalSetting()

        self.model.amodel.con_h_expr_0 = vertcat(*self.nl_con)   
        self.model.amodel.con_h_expr = vertcat(*self.nl_con)
        self.model.amodel.con_h_expr_e = vertcat(*self.nl_con_e)
        
        self.ocp.constraints.lh_0 = np.hstack(self.nl_lb)
        self.ocp.constraints.uh_0 = np.hstack(self.nl_ub)
        self.ocp.constraints.lh = np.hstack(self.nl_lb)
        self.ocp.constraints.uh = np.hstack(self.nl_ub)
        self.ocp.constraints.lh_e = np.hstack(self.nl_lb_e)
        self.ocp.constraints.uh_e = np.hstack(self.nl_ub_e)

        # Solver options
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        # self.ocp.solver_options.qp_solver_tol_stat = self.params.qp_tol_stat
        # self.ocp.solver_options.nlp_solver_tol_stat = self.params.nlp_tol_stat

        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.regenerate)

        # Reference and fails counter
        self.fails = 0
        self.x_ref = np.zeros(self.model.nx)

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

        # Viable state (None for Naive and ST controllers)
        self.x_viable = None

        # Time stats
        self.time_fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
                            'time_glob', 'time_reg', 'time_tot']

    def additionalSetting(self):
        pass

    def terminalConstraint(self, soft=True):
        # Get the actual number of nl_constraints --> will be the index for the soft constraint
        num_nl_e = len(self.nl_con_e)

        self.model.setNNmodel()
        self.nl_con_e.append(self.model.nn_model)

        self.ocp.solver_options.model_external_shared_lib_dir = self.model.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.model.l4c_model.name

        self.nl_lb_e.append(np.array([0.]))
        self.nl_ub_e.append(np.array([1e6]))

        if soft:
            self.ocp.constraints.idxsh_e = np.array([num_nl_e])

            self.ocp.cost.zl_e = np.array([self.params.ws_t])
            self.ocp.cost.zu_e = np.array([0.])
            self.ocp.cost.Zl_e = np.array([0.])
            self.ocp.cost.Zu_e = np.array([0.])

    def runningConstraint(self, soft=True):
        # Suppose that the NN model is already set (same for external model shared lib)
        num_nl = len(self.nl_con)
        self.nl_con.append(self.model.nn_model)

        self.nl_lb.append(np.array([0.]))
        self.nl_ub.append(np.array([1e6]))      

        if soft:
            self.ocp.constraints.idxsh = np.array([num_nl])

            # Set zl initially to zero, then apply receding constraint in the step method
            self.ocp.cost.zl = np.array([0.])
            self.ocp.cost.zu = np.array([0.])
            self.ocp.cost.Zl = np.array([0.])
            self.ocp.cost.Zu = np.array([0.])

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        y_ref = np.zeros((self.N + 1, self.model.ny))
        # Minimize effort --> no x_ref
        y_ref[:, :self.model.nx] = self.x_ref
        W = lin.block_diag(self.Q, self.R)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.cost_set(i, 'yref', y_ref[i], api='new')
            self.ocp_solver.cost_set(i, 'W', W, api='new')

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.cost_set(self.N, 'yref', y_ref[-1,:self.model.nx], api='new')
        self.ocp_solver.cost_set(self.N, 'W', self.Q, api='new')

        # Solve the OCP
        status = self.ocp_solver.solve()

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
        return np.array([self.ocp_solver.get_stats(field) for field in self.time_fields])

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def getLastViableState(self):
        return np.copy(self.x_viable)
    
    def resetHorizon(self, N):
        self.N = N
        if self.ocp_name == 'receding':
            self.r = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)

        self.x_temp = np.zeros((self.N + 1, self.model.nx))
        self.u_temp = np.zeros((self.N, self.model.nu))
