import re
import numpy as np
from copy import deepcopy
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
from casadi import MX, vertcat, norm_2, Function, cos, sin
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import torch
import scipy.linalg as lin
import torch.nn as nn
import l4casadi as l4c


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU(), ub=None):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )
        self.ub = ub if ub is not None else 1.0

    def forward(self, x):
        out = self.linear_stack(x) * self.ub
        return out


class OldNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
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
        self.p = MX.sym("p", 3)     # Cartesian EE position
        # Double integrator
        self.f_disc = vertcat(
            self.x[:nq] + params.dt * self.x[nq:] + 0.5 * params.dt**2 * self.u,
            self.x[nq:] + params.dt * self.u
        ) 
        self.f_fun = Function('f', [self.x, self.u], [self.f_disc])
            
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.disc_dyn_expr = self.f_disc
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq
        self.np = self.amodel.p.size()[0]

        # Real dynamics
        H_b = np.eye(4)
        self.tau = self.mass(H_b, self.x[:nq])[6:, 6:] @ self.u + \
                   self.bias(H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:]
        self.tau_fun = Function('tau', [self.x, self.u], [self.tau])

        # EE position (global frame)
        T_ee = self.fk(np.eye(4), self.x[:nq])
        self.t_loc = np.array([0.035, 0., 0.])
        self.t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ self.t_loc
        self.ee_fun = Function('ee_fun', [self.x], [self.t_glob])

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints]) 
        # joint_effort = np.array([joint.limit.effort for joint in robot_joints]) 
        joint_effort = np.array([2., 23., 10., 4.])

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])

        # EE target
        self.ee_ref = self.jointToEE(np.zeros(self.nx))

        # NN model (viability constraint)
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

        # Cartesian constraints
        self.obs_string = '_obs' if params.obs_flag else ''

    def jointToEE(self, x):
        return np.array(self.ee_fun(x))

    def checkStateConstraints(self, x):
        return np.all(np.logical_and(x >= self.x_min - self.params.tol_x, 
                                     x <= self.x_max + self.params.tol_x))

    def checkTorqueConstraints(self, tau):
        # for i in range(len(tau)):
        #     print(f' Iter {i} : {self.tau_max - np.abs(tau[i].flatten())}')
        return np.all(np.logical_and(tau >= self.tau_min - self.params.tol_tau, 
                                     tau <= self.tau_max + self.params.tol_tau))

    def checkRunningConstraints(self, x, u):
        tau = np.array([self.tau_fun(x[i], u[i]).T for i in range(len(u))])
        return self.checkStateConstraints(x) and self.checkTorqueConstraints(tau)

    def checkSafeConstraints(self, x):
        return self.nn_func(x) >= - self.params.tol_nn 
    
    def integrate(self, x, u):
        x_next = np.zeros(self.nx)
        tau = np.array(self.tau_fun(x, u).T)
        if not self.checkTorqueConstraints(tau):
            # Cannot exceed the torque limits --> sat and compute forward dynamics on real system 
            H_b = np.eye(4)
            tau_sat = np.clip(tau, self.tau_min, self.tau_max)
            M = np.array(self.mass(H_b, x[:self.nq])[6:, 6:])
            h = np.array(self.bias(H_b, x[:self.nq], np.zeros(6), x[self.nq:])[6:])
            u = np.linalg.solve(M, (tau_sat.T - h)).T
        x_next[:self.nq] = x[:self.nq] + self.params.dt * x[self.nq:] + 0.5 * self.params.dt**2 * u
        x_next[self.nq:] = x[self.nq:] + self.params.dt * u
        return x_next, u

    def checkDynamicsConstraints(self, x, u):
        # Rollout the control sequence
        n = np.shape(u)[0]
        x_sim = np.zeros((n + 1, self.nx))
        x_sim[0] = np.copy(x[0])
        for i in range(n):
            x_sim[i + 1], _ = self.integrate(x_sim[i], u[i])
        # Check if the rollout state trajectory is almost equal to the optimal one
        return np.linalg.norm(x - x_sim) < self.params.tol_dyn * np.sqrt(n+1) 
    
    def setNNmodel(self):
        nls = {
            'relu': torch.nn.ReLU(),
            'elu': torch.nn.ELU(),
            'tanh': torch.nn.Tanh(),
            'gelu': torch.nn.GELU(approximate='tanh'),
            'silu': torch.nn.SiLU()
        }
        act = self.params.act
        act_fun = nls[act]

        if act in ['tanh']: #, 'sine']:
            ub = max(self.x_max[self.nq:]) * np.sqrt(self.nq)
        else:
            ub = 1

        model = NeuralNetwork(self.nx, 256, 1, act_fun, ub)
        # print(model)
        # print(f'{self.params.NN_DIR}{self.nq}dof_{act}{self.obs_string}.pt')
        nn_data = torch.load(f'{self.params.NN_DIR}{self.nq}dof_{act}{self.obs_string}.pt',
                             map_location=torch.device('cpu'))
        model.load_state_dict(nn_data['model'])

        x_cp = deepcopy(self.x)
        x_cp[self.nq] += self.params.eps
        vel_norm = norm_2(x_cp[self.nq:])
        pos = (x_cp[:self.nq] - nn_data['mean']) / nn_data['std']
        vel_dir = x_cp[self.nq:] / vel_norm
        state = vertcat(pos, vel_dir)

        self.l4c_model = l4c.L4CasADi(model,
                                      device='cpu',
                                      name=f'{self.amodel.name}_model',
                                      build_dir=f'{self.params.GEN_DIR}nn_{self.amodel.name}')
        self.nn_model = self.l4c_model(state) * (100 - self.params.alpha) / 100 - vel_norm
        self.nn_func = Function('nn_func', [self.x], [self.nn_model])

class TriplePendulumModel(AdamModel):
    ''' Triple pendulum model from the Lagrangian formulation '''
    def __init__(self, params, n_dofs=3):
        self.params = params
        self.amodel = AcadosModel()

        nq = n_dofs
        self.amodel.name = 'triple_pendulum'
        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nq)
        # Double integrator
        self.f_disc = vertcat(
            self.x[:nq] + params.dt * self.x[nq:] + 0.5 * params.dt**2 * self.u,
            self.x[nq:] + params.dt * self.u
        ) 
        self.f_fun = Function('f', [self.x, self.u], [self.f_disc])
            
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.disc_dyn_expr = self.f_disc

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq

        # Real dynamics
        self.tau = vertcat(
            params.l1
            * (
                self.u[0] * params.l1 * params.m1
                + self.u[0] * params.l1 * params.m2
                + self.u[0] * params.l1 * params.m3
                + self.u[1]
                * params.l2
                * (params.m2 + params.m3)
                * cos(self.x[0] - self.x[1])
                + self.u[2] * params.l2 * params.m3 * cos(self.x[0] - self.x[2])
                + params.g * params.m1 * sin(self.x[0])
                + params.g * params.m2 * sin(self.x[0])
                + params.g * params.m3 * sin(self.x[0])
                + params.l2 * params.m2 * self.x[4] ** 2 * sin(self.x[0] - self.x[1])
                + params.l2 * params.m3 * self.x[4] ** 2 * sin(self.x[0] - self.x[1])
                + params.l2 * params.m3 * self.x[5] ** 2 * sin(self.x[0] - self.x[2])
            ),
            params.l2
            * (
                self.u[1] * params.l2 * params.m2
                + self.u[1] * params.l2 * params.m3
                + self.u[0]
                * params.l1
                * (params.m2 + params.m3)
                * cos(self.x[0] - self.x[1])
                + self.u[2] * params.l2 * params.m3 * cos(self.x[1] - self.x[2])
                - params.l1 * params.m2 * self.x[3] ** 2 * sin(self.x[0] - self.x[1])
                - params.l1 * params.m3 * self.x[3] ** 2 * sin(self.x[0] - self.x[1])
                + params.g * params.m2 * sin(self.x[1])
                + params.g * params.m3 * sin(self.x[1])
                + params.l2 * params.m3 * self.x[5] ** 2 * sin(self.x[1] - self.x[2])
            ),
            params.l2
            * params.m3
            * (
                self.u[2] * params.l2
                + self.u[0] * params.l1 * cos(self.x[0] - self.x[2])
                + self.u[1] * params.l2 * cos(self.x[1] - self.x[2])
                - params.l1 * self.x[3] ** 2 * sin(self.x[0] - self.x[2])
                - params.l2 * self.x[4] ** 2 * sin(self.x[1] - self.x[2])
                + params.g * sin(self.x[2])
            ),
        )
        self.tau_fun = Function('tau', [self.x, self.u], [self.tau])

        # Acceleration with forward dynamics
        f_expl = vertcat(
            (
                -params.g
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * sin(-2 * self.x[2] + 2 * self.x[1] + self.x[0])
                - params.g
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * sin(2 * self.x[2] - 2 * self.x[1] + self.x[0])
                + 2
                * self.u[0]
                * params.l2
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + 2 * self.x[1])
                + 2
                * self.x[3] ** 2
                * params.l1 ** 2
                * params.l2
                * params.l2
                * params.m2
                * (params.m2 + params.m3)
                * sin(-2 * self.x[1] + 2 * self.x[0])
                - 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + self.x[0] + self.x[2])
                - 2
                * self.u[1]
                * params.l1
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + self.x[1] + self.x[0])
                + 2
                * params.l1
                * params.l2
                * params.l2 ** 2
                * params.m2
                * params.m3
                * self.x[5] ** 2
                * sin(-2 * self.x[1] + self.x[0] + self.x[2])
                + 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(self.x[0] - self.x[2])
                + 2
                * (
                    self.u[1]
                    * params.l1
                    * (params.m3 + 2 * params.m2)
                    * cos(-self.x[1] + self.x[0])
                    + (
                        params.g
                        * params.l1
                        * params.m2
                        * (params.m2 + params.m3)
                        * sin(-2 * self.x[1] + self.x[0])
                        + 2
                        * self.x[4] ** 2
                        * params.l1
                        * params.l2
                        * params.m2
                        * (params.m2 + params.m3)
                        * sin(-self.x[1] + self.x[0])
                        + params.m3
                        * self.x[5] ** 2
                        * sin(self.x[0] - self.x[2])
                        * params.l1
                        * params.l2
                        * params.m2
                        + params.g
                        * params.l1
                        * (
                            params.m2 ** 2
                            + (params.m3 + 2 * params.m1) * params.m2
                            + params.m1 * params.m3
                        )
                        * sin(self.x[0])
                        - self.u[0] * (params.m3 + 2 * params.m2)
                    )
                    * params.l2
                )
                * params.l2
            )
            / params.l1 ** 2
            / params.l2
            / (
                params.m2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + 2 * self.x[0])
                + params.m1 * params.m3 * cos(-2 * self.x[2] + 2 * self.x[1])
                - params.m2 ** 2
                + (-params.m3 - 2 * params.m1) * params.m2
                - params.m1 * params.m3
            )
            / params.l2
            / 2,
            (
                -2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(2 * self.x[0] - self.x[2] - self.x[1])
                - 2
                * params.l1
                * params.l2
                * params.l2 ** 2
                * params.m2
                * params.m3
                * self.x[5] ** 2
                * sin(2 * self.x[0] - self.x[2] - self.x[1])
                + params.g
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * sin(self.x[1] + 2 * self.x[0] - 2 * self.x[2])
                - params.g
                * params.l1
                * params.l2
                * (
                    (params.m1 + 2 * params.m2) * params.m3
                    + 2 * params.m2 * (params.m1 + params.m2)
                )
                * params.l2
                * sin(-self.x[1] + 2 * self.x[0])
                - 2
                * self.x[4] ** 2
                * params.l1
                * params.l2 ** 2
                * params.l2
                * params.m2
                * (params.m2 + params.m3)
                * sin(-2 * self.x[1] + 2 * self.x[0])
                + 2
                * self.u[1]
                * params.l1
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + 2 * self.x[0])
                + 2
                * params.l1
                * params.l2 ** 2
                * params.l2
                * params.m1
                * params.m3
                * self.x[4] ** 2
                * sin(-2 * self.x[2] + 2 * self.x[1])
                - 2
                * self.u[0]
                * params.l2
                * params.l2
                * params.m3
                * cos(-2 * self.x[2] + self.x[1] + self.x[0])
                + 2
                * params.l1 ** 2
                * params.l2
                * params.l2
                * params.m1
                * params.m3
                * self.x[3] ** 2
                * sin(-2 * self.x[2] + self.x[1] + self.x[0])
                - 2
                * params.l1 ** 2
                * params.l2
                * self.x[3] ** 2
                * (
                    (params.m1 + 2 * params.m2) * params.m3
                    + 2 * params.m2 * (params.m1 + params.m2)
                )
                * params.l2
                * sin(-self.x[1] + self.x[0])
                + 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m3 + 2 * params.m1 + params.m2)
                * cos(-self.x[2] + self.x[1])
                + (
                    2
                    * self.u[0]
                    * params.l2
                    * (params.m3 + 2 * params.m2)
                    * cos(-self.x[1] + self.x[0])
                    + params.l1
                    * (
                        4
                        * self.x[5] ** 2
                        * params.m3
                        * params.l2
                        * (params.m1 + params.m2 / 2)
                        * params.l2
                        * sin(-self.x[2] + self.x[1])
                        + params.g
                        * params.m3
                        * params.l2
                        * params.m1
                        * sin(-2 * self.x[2] + self.x[1])
                        + params.g
                        * (
                            (params.m1 + 2 * params.m2) * params.m3
                            + 2 * params.m2 * (params.m1 + params.m2)
                        )
                        * params.l2
                        * sin(self.x[1])
                        - 2 * self.u[1] * (params.m3 + 2 * params.m1 + 2 * params.m2)
                    )
                )
                * params.l2
            )
            / (
                params.m2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + 2 * self.x[0])
                + params.m1 * params.m3 * cos(-2 * self.x[2] + 2 * self.x[1])
                + (-params.m1 - params.m2) * params.m3
                - 2 * params.m1 * params.m2
                - params.m2 ** 2
            )
            / params.l1
            / params.l2
            / params.l2 ** 2
            / 2,
            (
                -2
                * params.m3
                * self.u[1]
                * params.l1
                * params.l2
                * (params.m2 + params.m3)
                * cos(2 * self.x[0] - self.x[2] - self.x[1])
                + params.g
                * params.m3
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(2 * self.x[0] + self.x[2] - 2 * self.x[1])
                + 2
                * self.u[2]
                * params.l1
                * params.l2
                * (params.m2 + params.m3) ** 2
                * cos(-2 * self.x[1] + 2 * self.x[0])
                - params.g
                * params.m3
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(2 * self.x[0] - self.x[2])
                - params.g
                * params.m3
                * params.l1
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(-self.x[2] + 2 * self.x[1])
                - 2
                * params.l1
                * params.l2
                * params.l2 ** 2
                * params.m1
                * params.m3 ** 2
                * self.x[5] ** 2
                * sin(-2 * self.x[2] + 2 * self.x[1])
                - 2
                * self.u[0]
                * params.l2
                * params.l2
                * params.m3
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + self.x[0] + self.x[2])
                + 2
                * params.m3
                * self.x[3] ** 2
                * params.l1 ** 2
                * params.l2
                * params.l2
                * params.m1
                * (params.m2 + params.m3)
                * sin(-2 * self.x[1] + self.x[0] + self.x[2])
                + 2
                * params.m3
                * self.u[1]
                * params.l1
                * params.l2
                * (params.m3 + 2 * params.m1 + params.m2)
                * cos(-self.x[2] + self.x[1])
                + (params.m2 + params.m3)
                * (
                    2 * self.u[0] * params.l2 * params.m3 * cos(self.x[0] - self.x[2])
                    + params.l1
                    * (
                        -2
                        * params.m3
                        * self.x[3] ** 2
                        * params.l1
                        * params.l2
                        * params.m1
                        * sin(self.x[0] - self.x[2])
                        - 4
                        * params.m3
                        * self.x[4] ** 2
                        * sin(-self.x[2] + self.x[1])
                        * params.l2
                        * params.l2
                        * params.m1
                        + params.g * params.m3 * sin(self.x[2]) * params.l2 * params.m1
                        - 2 * self.u[2] * (params.m3 + 2 * params.m1 + params.m2)
                    )
                )
                * params.l2
            )
            / params.m3
            / (
                params.m2
                * (params.m2 + params.m3)
                * cos(-2 * self.x[1] + 2 * self.x[0])
                + params.m1 * params.m3 * cos(-2 * self.x[2] + 2 * self.x[1])
                + (-params.m1 - params.m2) * params.m3
                - 2 * params.m1 * params.m2
                - params.m2 ** 2
            )
            / params.l1
            / params.l2 ** 2
            / params.l2
            / 2,
        )
        # IMPORTANT: in the self.u now the torque should be inserted
        self.acc_fun = Function('acc', [self.x, self.u], [f_expl])

        # Joint limits
        joint_lower = np.ones(nq) * params.q_min
        joint_upper = np.ones(nq) * params.q_max
        joint_velocity = np.ones(nq) * params.dq_lim 
        joint_effort = np.ones(nq) * params.tau_lim 

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])

        # NN model (viability constraint)
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

    def integrate(self, x, u):
        x_next = np.zeros(self.nx)
        tau = np.array(self.tau_fun(x, u).T)
        if not self.checkTorqueConstraints(tau):
            # Cannot exceed the torque limits --> sat and compute forward dynamics on real system 
            u = np.array(self.acc_fun(x, tau).T)
        x_next[:self.nq] = x[:self.nq] + self.params.dt * x[self.nq:] + 0.5 * self.params.dt**2 * u
        x_next[self.nq:] = x[self.nq:] + self.params.dt * u
        return x_next, u
    
    def setNNmodel(self):
        device = torch.device('cpu')
        model = OldNeuralNetwork(self.nx, (self.nx - 1) * 100, 1)
        model.load_state_dict(torch.load(self.params.NN_DIR + 'model.zip', map_location=device))
        mean = torch.load(self.params.NN_DIR + 'mean.zip')
        std = torch.load(self.params.NN_DIR + 'std.zip')

        x_cp = deepcopy(self.x)
        x_cp[self.nq] += self.params.eps
        vel_norm = norm_2(x_cp[self.nq:])
        pos = (x_cp[:self.nq] - mean) / std
        vel_dir = x_cp[self.nq:] / vel_norm
        state = vertcat(pos, vel_dir)

        self.l4c_model = l4c.L4CasADi(model,
                                      device='cpu',
                                      name=f'{self.amodel.name}_model',
                                      build_dir=f'{self.params.GEN_DIR}nn_{self.amodel.name}')
        self.nn_model = self.l4c_model(state) * (100 - self.params.alpha) / 100 - vel_norm
        self.nn_func = Function('nn_func', [self.x], [self.nn_model])


class AbstractController:
    def __init__(self, model, obstacles=None):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model
        self.obstacles = obstacles

        self.N = self.params.N
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.dt * self.N
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        if self.model.amodel.name == 'triple_pendulum':
            self.Q = 1e-4 * np.eye(self.model.nx)
            self.Q[0, 0] = 5e2
            self.R = 1e-4 * np.eye(self.model.nu)

            self.ocp.cost.W = lin.block_diag(self.Q, self.R)
            self.ocp.cost.W_e = self.Q
            
            self.ocp.cost.cost_type = 'LINEAR_LS'
            self.ocp.cost.cost_type_e = 'LINEAR_LS'

            self.x_ref = np.zeros(self.model.nx)
            self.x_ref[:self.model.nq] = np.pi
            self.x_ref[0] = self.params.q_max - 0.05

            self.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
            self.ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
            self.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
            self.ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)
            self.ocp.cost.Vx_e = np.eye(self.model.nx)

            self.ocp.cost.yref = np.zeros(self.model.ny)
            self.ocp.cost.yref[:self.model.nx] = self.x_ref 
            self.ocp.cost.yref_e = self.x_ref

        else:
            self.Q = 1e2 * np.eye(self.model.np)
            self.R = 5e-3 * np.eye(self.model.nu) 

            self.ocp.cost.cost_type = 'EXTERNAL'
            self.ocp.cost.cost_type_e = 'EXTERNAL'

            t_glob = self.model.t_glob
            delta = t_glob - self.model.p
            track_ee = delta.T @ self.Q @ delta 
            self.ocp.model.cost_expr_ext_cost = track_ee + self.model.u.T @ self.R @ self.model.u
            self.ocp.model.cost_expr_ext_cost_e = track_ee
            self.ocp.parameter_values = np.zeros(self.model.np)

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
        self.nl_con_0, self.nl_lb_0, self.nl_ub_0 = [], [], []
        self.nl_con, self.nl_lb, self.nl_ub = [], [], []
        self.nl_con_e, self.nl_lb_e, self.nl_ub_e = [], [], []
        
        # --> dynamics (only on running nodes)
        self.nl_con_0.append(self.model.tau)
        self.nl_lb_0.append(self.model.tau_min)
        self.nl_ub_0.append(self.model.tau_max)
        
        self.nl_con.append(self.model.tau)
        self.nl_lb.append(self.model.tau_min)
        self.nl_ub.append(self.model.tau_max)
        
        # --> collision (both on running and terminal nodes)
        if obstacles is not None and self.params.obs_flag:
            # Collision avoidance with two obstacles
            for obs in self.obstacles:
                if obs['name'] == 'floor':
                    self.nl_con_0.append(t_glob[2])
                    self.nl_con.append(t_glob[2])
                    self.nl_con_e.append(t_glob[2])

                    self.nl_lb_0.append(obs['bounds'][0])
                    self.nl_ub_0.append(obs['bounds'][1])
                    self.nl_lb.append(obs['bounds'][0])
                    self.nl_ub.append(obs['bounds'][1])
                    self.nl_lb_e.append(obs['bounds'][0])
                    self.nl_ub_e.append(obs['bounds'][1])
                elif obs['name'] == 'ball':
                    dist_b = (t_glob - obs['position']).T @ (t_glob - obs['position'])
                    self.nl_con_0.append(dist_b)
                    self.nl_con.append(dist_b)
                    self.nl_con_e.append(dist_b)

                    self.nl_lb_0.append(obs['bounds'][0])
                    self.nl_ub_0.append(obs['bounds'][1])
                    self.nl_lb.append(obs['bounds'][0])
                    self.nl_ub.append(obs['bounds'][1])
                    self.nl_lb_e.append(obs['bounds'][0])
                    self.nl_ub_e.append(obs['bounds'][1])

        # Additional settings, in general is an empty method
        self.additionalSetting()

        self.model.amodel.con_h_expr_0 = vertcat(*self.nl_con_0)   
        self.model.amodel.con_h_expr = vertcat(*self.nl_con)
        
        self.ocp.constraints.lh_0 = np.hstack(self.nl_lb_0)
        self.ocp.constraints.uh_0 = np.hstack(self.nl_ub_0)
        self.ocp.constraints.lh = np.hstack(self.nl_lb)
        self.ocp.constraints.uh = np.hstack(self.nl_ub)
        if len(self.nl_con_e) > 0:
            self.model.amodel.con_h_expr_e = vertcat(*self.nl_con_e)
            self.ocp.constraints.lh_e = np.hstack(self.nl_lb_e)
            self.ocp.constraints.uh_e = np.hstack(self.nl_ub_e)

        # Solver options
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_dyn = 0   
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        # self.ocp.solver_options.alpha_reduction = self.params.alpha_reduction
        # self.ocp.solver_options.alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt
        # self.ocp.solver_options.tol = 1e-4
        # self.ocp.solver_options.qp_tol = 1e-4
        # self.ocp.solver_options.regularize_method = 'PROJECT'   # Maybe is a good idea if exact hessian is not used
        # self.ocp.solver_options.ext_fun_compile_flags = '-O2'

        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)

        # Reference and fails counter
        self.fails = 0

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

        # Viable state (None for Naive and ST controllers)
        self.x_viable = None

        # Time stats
        self.time_fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
                            'time_glob', 'time_reg', 'time_tot']
        self.last_status = 4

    def additionalSetting(self):
        pass

    def terminalConstraint(self, soft=True):
        # Get the actual number of nl_constraints --> will be the index for the soft constraint
        num_nl_e = np.sum([c.shape[0] for c in self.nl_con_e])  # verify if it works

        self.model.setNNmodel()
        self.nl_con_e.append(self.model.nn_model)

        self.ocp.solver_options.model_external_shared_lib_dir = self.model.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.model.l4c_model.name

        self.nl_lb_e.append(np.array([0.]))
        self.nl_ub_e.append(np.array([1e6]))

        if soft:
            self.ocp.constraints.idxsh_e = np.array([num_nl_e])

            self.ocp.cost.zl_e = np.array([self.params.ws_r])   # FIXME
            self.ocp.cost.zu_e = np.array([0.])
            self.ocp.cost.Zl_e = np.array([0.])
            self.ocp.cost.Zu_e = np.array([0.])

    def runningConstraint(self, soft=True):
        # Suppose that the NN model is already set (same for external model shared lib)
        num_nl = np.sum([c.shape[0] for c in self.nl_con])
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

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            if not self.model.amodel.name == 'triple_pendulum':
                self.ocp_solver.set(i, 'p', self.model.ee_ref)

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        if not self.model.amodel.name == 'triple_pendulum':
            self.ocp_solver.set(self.N, 'p', self.model.ee_ref)

        # Solve the OCP
        status = self.ocp_solver.solve()
        # self.ocp_solver.store_iterate(overwrite=True)
        # if status != 0:
        #     self.ocp_solver.print_statistics()
        #     self.ocp_solver.dump_last_qp_to_json("qp.json", overwrite=True)
            # self.ocp_solver.reset()         # Reset the solver, to remove all the NaNs

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.ocp_solver.get(i, "x")
            self.u_temp[i] = self.ocp_solver.get(i, "u")
        self.x_temp[-1] = self.ocp_solver.get(self.N, "x")

        self.last_status = status
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

    def setReference(self, ee_ref):
        self.model.ee_ref = ee_ref

    def getTime(self):
        return np.array([self.ocp_solver.get_stats(field) for field in self.time_fields])

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def getLastViableState(self):
        return np.copy(self.x_viable)
    
    def checkCollision(self, x):
        if self.obstacles is not None and self.params.obs_flag:
            t_glob = self.model.jointToEE(x) 
            for obs in self.obstacles:
                if obs['name'] == 'floor':
                    if t_glob[2] + self.params.tol_obs < obs['bounds'][0]:
                        return False
                elif obs['name'] == 'ball':
                    dist_b = np.sum((t_glob.flatten() - obs['position']) ** 2)
                    if dist_b + self.params.tol_obs < obs['bounds'][0]:
                        return False
        return True
    
    def resetHorizon(self, N):
        self.N = N
        if self.ocp_name == 'receding':
            self.r = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)

        self.x_temp = np.zeros((self.N + 1, self.model.nx))
        self.u_temp = np.zeros((self.N, self.model.nu))

    def safeGuess(self, x, u, n_safe):
        for i in range(n_safe):
            x, _ = self.model.integrate(x, u[i])
            if not self.model.checkStateConstraints(x) or not self.checkCollision(x):
                return False, None
        return self.model.checkSafeConstraints(x), x
    
    def guessCorrection(self):
        x_sim = np.copy(self.x_guess)
        u_sim = np.copy(self.u_guess)
        for i in range(self.N):
            x_sim[i + 1], u_sim[i] = self.model.integrate(x_sim[i], self.u_guess[i])
        if self.model.checkStateConstraints(x_sim) and np.all([self.checkCollision(x) for x in x_sim]):
            self.x_guess = np.copy(x_sim)
            self.u_guess = np.copy(u_sim)
            return True
        return False