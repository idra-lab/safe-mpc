import re
import numpy as np
from copy import deepcopy
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
import casadi as cs
from casadi import MX, vertcat, norm_2, Function, if_else
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
    def __init__(self, params, n_dofs):
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
        self.kin_dyn = KinDynComputations(params.robot_urdf, joint_names, robot.get_root())        
        self.kin_dyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
        self.mass = self.kin_dyn.mass_matrix_fun()                           # Mass matrix
        self.bias = self.kin_dyn.bias_force_fun()                            # Nonlinear effects  
        self.gravity = self.kin_dyn.gravity_term_fun()                       # Gravity vector
        self.fk = self.kin_dyn.forward_kinematics_fun(params.frame_name)     # Forward kinematics
        nq = len(joint_names)

        self.amodel.name = params.urdf_name
        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nq)
        self.p = MX.sym("p", 5)     # Safety margin for the NN model, EE, logic variable
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
        self.ee_rot = Function('ee_rot', [self.x], [T_ee[:3, :3]])

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints]) 
        # joint_effort = np.array([joint.limit.effort for joint in robot_joints]) 
        joint_effort = np.array([2., 23., 10., 4., 4., 4.])
        joint_effort = joint_effort[:nq]

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])

        # EE target
        self.ee_ref = self.jointToEE(np.zeros(self.nx))
        self.R_ref = np.eye(3)

        # NN model (viability constraint)
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

        # Cartesian constraints
        self.obs_string = '_obs' if params.obs_flag else ''
        self.joint_names = joint_names

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
        return self.nn_func(x, self.params.alpha) >= - self.params.tol_nn 
    
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
        nn_dofs = self.params.nn_dofs

        if act in ['tanh']: #, 'sine']:
            ub = max(self.x_max[self.nq:]) * np.sqrt(self.nq)
        else:
            ub = 1

        model = NeuralNetwork(self.params.nn_dofs * 2, 256, 1, act_fun, ub)
        nn_data = torch.load(f'{self.params.NN_DIR}{nn_dofs}dof_{act}{self.obs_string}.pt',
                             map_location=torch.device('cpu'))
        model.load_state_dict(nn_data['model'])

        x_cp = deepcopy(self.x)
        x_cp[self.nq] += self.params.eps
        vel_norm = norm_2(x_cp[self.nq:self.nq + nn_dofs])
        pos = (x_cp[:nn_dofs] - nn_data['mean']) / nn_data['std']
        vel_dir = x_cp[self.nq:self.nq + nn_dofs] / vel_norm
        state = vertcat(pos, vel_dir)

        self.l4c_model = l4c.L4CasADi(model,
                                      device='cpu',
                                      name=f'{self.amodel.name}_model',
                                      build_dir=f'{self.params.GEN_DIR}nn_{self.amodel.name}')
        self.nn_model = if_else(self.p[4] > 0, 
                                self.l4c_model(state) * (100 - self.p[3]) / 100 - vel_norm, 
                                1., True)
        self.nn_func = Function('nn_func', [self.x, self.p], [self.nn_model])

    def casadi_segment_dist(self,A_s,B_s,C_s,D_s):
        # ab_a = cs.MX.sym('ab_a',3,1)
        # ab_b = cs.MX.sym('ab_b',3,1)
        # cd_c = cs.MX.sym('cd_a',3,1)
        # cd_d = cs.MX.sym('cd_b',3,1)

        R = cs.sum1((B_s-A_s)*(D_s-C_s))
        S1 = cs.sum1((B_s-A_s)*(C_s-A_s))
        D1 = cs.sum1((B_s-A_s)**2)
        S2 = cs.sum1((D_s-C_s)*(C_s-A_s))
        D2 = cs.sum1((D_s-C_s)**2)

        t = (S1*D2 - S2*R)/(D1*D2 - (R**2+1e-5))
        t = cs.fmax(cs.fmin(t,1),0)
        #u = -(S2*D1 - S1*R)/(D1*D2 - R**2)
        u = (t*R - S2)/D2
        u = cs.fmax(cs.fmin(u,1),0)

        t = (u*R + S1) / D1
        t = cs.fmax(cs.fmin(t,1),0)

        constr_expr = cs.sum1(((B_s-A_s)*t - (D_s-C_s)*u - (C_s-A_s))**2)

        return constr_expr
    
    def np_segment_dist(self,A_s,B_s,C_s,D_s):
        R = np.sum((B_s-A_s)*(D_s-C_s),axis=1)
        S1 = np.sum((B_s-A_s)*(C_s-A_s),axis=1)
        D1 = np.sum(((B_s-A_s)**2),axis=1)
        S2 = np.sum((D_s-C_s)*(C_s-A_s),axis=1)
        D2 = np.sum(((D_s-C_s)**2),axis=1)

        t = (np.multiply(S1,D2) - np.multiply(S2,R))/(np.multiply(D1,D2) - (R**2+1e-5))
        t = np.fmax(np.fmin(t,1),0)
        #u = -(S2*D1 - S1*R)/(D1*D2 - R**2)
        u = (t*R - S2)/D2
        u = np.fmax(np.fmin(u,1),0)

        t = (u*R + S1) / D1
        t = np.fmax(np.fmin(t,1),0)

        return np.sum(((B_s-A_s)*t[:,np.newaxis] - (D_s-C_s)*u[:,np.newaxis] - (C_s-A_s))**2,axis=1)
    
    def ball_segment_dist(self,A_s,B_s,capsule_length,obs_pos):
        obst_pos = cs.MX(obs_pos)
        t = cs.fmin(cs.fmax(cs.dot((obst_pos-A_s),(B_s-A_s)) / (capsule_length**2),0),1)
        d = cs.sum1((obst_pos-(A_s+(B_s-A_s)*t))**2) 
        return d
    
    def np_ball_segment_dist(self,A_s,B_s,capsule_length,obs_pos):
        t = np.fmin(np.fmax(np.dot((obs_pos-A_s),(B_s-A_s)) / (capsule_length**2),0),1)
        d = np.sum((obs_pos-(A_s+(B_s-A_s)*t))**2) 
        return d


class AbstractController:
    def __init__(self, model, obstacles=None, capsules=None, capsule_pairs=None):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model
        self.obstacles = obstacles
        self.capsules = capsules
        self.capsule_pairs = capsule_pairs

        self.N = self.params.N
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.dt * self.N
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.Q_trasl = self.params.Q_trasl
        self.Q_rot = self.params.Q_rot
        self.R = self.params.R * np.eye(self.model.nu) 

        self.ocp.cost.cost_type = 'EXTERNAL'
        self.ocp.cost.cost_type_e = 'EXTERNAL'

        # Tras
        t_glob = self.model.t_glob
        delta = t_glob - self.model.p[:3]
        track_ee = delta.T @ self.Q_trasl @ delta 
        # Rot
        T_ee = self.model.fk(np.eye(4), self.model.x[:self.model.nq])
        rot_ee = cs.trace((np.eye(3) - model.R_ref.T @ T_ee[:3, :3]) @ self.Q_rot)

        self.ocp.model.cost_expr_ext_cost = track_ee + self.model.u.T @ self.R @ self.model.u + rot_ee
        self.ocp.model.cost_expr_ext_cost_e = track_ee + rot_ee
        self.ocp.parameter_values = np.hstack([self.model.ee_ref, [self.params.alpha, 1.]])

        # Capsules end-points forward kinematics
        n_cap=0
        for capsule in self.capsules:
            capsule['index']=n_cap
            if capsule['type'] == 'moving':
                rot_mat=np.eye(4)
                if capsule['rotation_offset'] != None:
                    th_off=capsule['rotation_offset']
                    rot_mat_x = np.array([[1,0,0,0],
                                          [0,np.cos(th_off[0]),-np.sin(th_off[0]),0],
                                          [0,np.sin(th_off[0]),np.cos(th_off[0]),0],
                                          [0,0,0,1]])
                    
                    rot_mat_y = np.array([[np.cos(th_off[1]),0,np.sin(th_off[1]),0],
                                          [0,1,0,0],
                                          [-np.sin(th_off[1]),0,np.cos(th_off[1]),0],
                                          [0,0,0,1]])
                    
                    rot_mat_z = np.array([[np.cos(th_off[2]),-np.sin(th_off[2]),0,0],
                                          [np.sin(th_off[2]), np.cos(th_off[2]),0,0],
                                          [0,0,1,0],
                                          [0,0,0,1]])
                    rot_mat = rot_mat_x@rot_mat_y@rot_mat_z
                if capsule['spatial_offset'] != None:
                    prism_mat = np.array([[1,0,0,capsule['spatial_offset'][0]],
                                            [0,1,0,capsule['spatial_offset'][1]],
                                            [0,0,1,capsule['spatial_offset'][2]],
                                            [0,0,0,1]])
                    rot_mat = prism_mat@rot_mat  
                fk_capsule_points = self.model.kin_dyn.forward_kinematics_fun(capsule['link_name'])   
                T_capsule_points = fk_capsule_points(np.eye(4), self.model.x[:self.model.nq])@rot_mat
                capsule['end_points_fk'] = deepcopy([(T_capsule_points @ capsule['end_points'][0])[:3],
                                                     (T_capsule_points @ capsule['end_points'][1])[:3]])
                capsule['end_points_T_fun'] = deepcopy(cs.Function(f'fun_T_{n_cap}',[self.model.x],[T_capsule_points]))
                capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.model.x],[(T_capsule_points @ capsule['end_points'][0])[:3],
                                                                                                      (T_capsule_points @ capsule['end_points'][1])[:3]]))
            n_cap += 1

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

        for pair in self.capsule_pairs:
            if pair['type'] == 0:
                self.nl_con_0.append(self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']))
                self.nl_con.append(self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']))
                self.nl_con_e.append(self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']))

                self.nl_lb_0.append((pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2)
                self.nl_ub_0.append(1e6)
                self.nl_lb.append((pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2)
                self.nl_ub.append(1e6)
                self.nl_lb_e.append((pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2)
                self.nl_ub_e.append(1e6)

            elif pair['type'] == 1:
                self.nl_con_0.append(self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']))
                self.nl_con.append(self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']))
                self.nl_con_e.append(self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']))

                self.nl_lb_0.append((pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2)
                self.nl_ub_0.append(1e6)
                self.nl_lb.append((pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2)
                self.nl_ub.append(1e6)
                self.nl_lb_e.append((pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2)
                self.nl_ub_e.append(1e6)     

            elif pair['type'] == 2:
                for point in pair['elements'][0]['end_points_fk']:
                    self.nl_con_0.append(point[2])
                    self.nl_con.append(point[2])
                    self.nl_con_e.append(point[2])

                    self.nl_lb_0.append(pair['elements'][1]['bounds'][0])
                    self.nl_ub_0.append(pair['elements'][1]['bounds'][1])
                    self.nl_lb.append(pair['elements'][1]['bounds'][0])
                    self.nl_ub.append(pair['elements'][1]['bounds'][1])
                    self.nl_lb_e.append(pair['elements'][1]['bounds'][0])
                    self.nl_ub_e.append(pair['elements'][1]['bounds'][1])
        
        # # --> collision (both on running and terminal nodes)
        # if obstacles is not None and self.params.obs_flag:
        #     # Collision avoidance with two obstacles
        #     for obs in self.obstacles:
        #         if obs['name'] == 'floor':
        #             self.nl_con_0.append(t_glob[2])
        #             self.nl_con.append(t_glob[2])
        #             self.nl_con_e.append(t_glob[2])

        #             self.nl_lb_0.append(obs['bounds'][0])
        #             self.nl_ub_0.append(obs['bounds'][1])
        #             self.nl_lb.append(obs['bounds'][0])
        #             self.nl_ub.append(obs['bounds'][1])
        #             self.nl_lb_e.append(obs['bounds'][0])
        #             self.nl_ub_e.append(obs['bounds'][1])
        #         elif obs['name'] == 'ball':
        #             dist_b = (t_glob - obs['position']).T @ (t_glob - obs['position'])
        #             self.nl_con_0.append(dist_b)
        #             self.nl_con.append(dist_b)
        #             self.nl_con_e.append(dist_b)

        #             self.nl_lb_0.append(obs['bounds'][0])
        #             self.nl_ub_0.append(obs['bounds'][1])
        #             self.nl_lb.append(obs['bounds'][0])
        #             self.nl_ub.append(obs['bounds'][1])
        #             self.nl_lb_e.append(obs['bounds'][0])
        #             self.nl_ub_e.append(obs['bounds'][1])

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
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt
        self.ocp.solver_options.ext_fun_compile_flags = self.params.ext_flag
        # self.ocp.solver_options.tol = 1e-4
        # self.ocp.solver_options.qp_tol = 1e-4
        # self.ocp.solver_options.regularize_method = 'PROJECT'   # Maybe is a good idea if exact hessian is not used

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

        nq = self.model.nq
        nn_dofs = self.params.nn_dofs
        # if nq > nn_dofs:
        #     # Terminal constraint -> middle joint position, zero velocity
        #     # x_middle = (self.model.x_min[nn_dofs:nq] \
        #     #             + self.model.x_max[nn_dofs:nq]) / 2
        #     # self.ocp.constraints.lbx_e[nn_dofs:nq] = x_middle
        #     # self.ocp.constraints.ubx_e[nn_dofs:nq] = x_middle

        #     self.ocp.constraints.lbx_e[nq + nn_dofs:] = np.zeros(nq - nn_dofs)
        #     self.ocp.constraints.ubx_e[nq + nn_dofs:] = np.zeros(nq - nn_dofs)
        #     # TODO: may add these also to the soft constraint?

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

        num_nl_0 = np.sum([c.shape[0] for c in self.nl_con_0])
        self.nl_con_0.append(self.model.nn_model)

        self.nl_lb_0.append(np.array([0.]))
        self.nl_ub_0.append(np.array([1e6]))    

        nq = self.model.nq
        nn_dofs = self.params.nn_dofs
        if nq > nn_dofs:
            # Terminal constraint -> middle joint position, zero velocity
            # self.x_middle = (self.model.x_min[nn_dofs:nq] \
            #                 + self.model.x_max[nn_dofs:nq]) / 2
            self.x_zerovel = np.zeros(nq - nn_dofs)

        if soft:
            self.ocp.constraints.idxsh = np.array([num_nl])

            # Set zl initially to zero, then apply receding constraint in the step method
            self.ocp.cost.zl = np.array([0.])
            self.ocp.cost.zu = np.array([0.])
            self.ocp.cost.Zl = np.array([0.])
            self.ocp.cost.Zu = np.array([0.])

            self.ocp.constraints.idxsh_0 = np.array([num_nl_0])

            self.ocp.cost.zl_0 = np.array([0.])
            self.ocp.cost.zu_0 = np.array([0.])
            self.ocp.cost.Zl_0 = np.array([0.])
            self.ocp.cost.Zu_0 = np.array([0.])

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            if self.ocp_name != 'receding':
                self.ocp_solver.set(i, 'p', np.hstack([self.model.ee_ref, [self.params.alpha, 1.]]))

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        if self.ocp_name != 'receding':
            self.ocp_solver.set(self.N, 'p', np.hstack([self.model.ee_ref, [self.params.alpha, 1.]]))

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
        return u, False

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
        if self.capsule_pairs is None:
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
        else:
            capsules_pos = []
            for capsule in self.capsules:
                if capsule['type'] == 'moving':
                    capsules_pos.append(np.array([capsule['end_points_fk_fun'](x[i] if len(x.shape)>1 else x ) for i in range(x.shape[0] if len(x.shape)>1 else 1)]))
                elif capsule['type'] == 'fixed':
                    capsules_pos.append(np.array(capsule['end_points']).reshape(1,2,3,1))
            for pair in self.capsule_pairs:
                if pair['type'] == 0:
                    # A_s=capsules_pos[pair['elements'][0]['index']][:,0]
                    # B_s =capsules_pos[pair['elements'][0]['index']][:,1]
                    # C_s=capsules_pos[pair['elements'][1]['index']][:,0]
                    # D_s =capsules_pos[pair['elements'][1]['index']][:,1]
                    # dists = np.array([self.model.casadi_segment_dist(A_s[i],B_s[i],C_s[i],D_s[i]) for i in range(A_s.shape[0] if len(A_s.shape)>1 else 1)]) 
                    # if not(dists >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                    #     return False
                    if not(self.model.np_segment_dist(capsules_pos[pair['elements'][0]['index']][:,0],capsules_pos[pair['elements'][0]['index']][:,1],
                        capsules_pos[pair['elements'][1]['index']][:,0],capsules_pos[pair['elements'][1]['index']][:,1]) >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                        return False
                elif pair['type'] == 1:
                    A_s = capsules_pos[pair['elements'][0]['index']][:,0]
                    B_s = capsules_pos[pair['elements'][0]['index']][:,1]
                    dists = np.array([self.model.np_ball_segment_dist(A_s[i].flatten(),B_s[i].flatten(),pair['elements'][0]['length'],pair['elements'][1]['position']) for i in range(A_s.shape[0] if len(A_s.shape)>1 else 1)]) 
                    if not(dists >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                        return False
                elif pair['type'] == 2:
                    if not(capsules_pos[pair['elements'][0]['index']][:,0,2] >=  pair['elements'][1]['bounds'][0]).all(): return False
                    if not(capsules_pos[pair['elements'][0]['index']][:,0,2] <=  pair['elements'][1]['bounds'][1]).all(): return False
                    if not(capsules_pos[pair['elements'][0]['index']][:,1,2] >=  pair['elements'][1]['bounds'][0]).all(): return False
                    if not(capsules_pos[pair['elements'][0]['index']][:,1,2] <=  pair['elements'][1]['bounds'][1]).all(): return False
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