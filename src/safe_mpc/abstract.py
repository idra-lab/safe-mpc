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
from .safe_set import NetSafeSet, AnalyticSafeSet

class AdamModel:
    def __init__(self, params, n_dofs):
        self.params = params
        self.amodel = AcadosModel()
        # Robot dynamics with Adam (IIT)
        #                # URDF.from_xml_file(params.robot_urdf)
        try:
            n_dofs = n_dofs if n_dofs else len(self.params.robot_descr.joints)
            if n_dofs > len(self.params.robot_descr.joints) or n_dofs < 1:
                raise ValueError
        except ValueError:
            print(f'\nInvalid number of degrees of freedom! Must be > 1 and <= {len(self.params.robot_descr.joints)}\n')
            exit()
        
        robot_joints = []
        jj=0
        while jj < n_dofs:
            for jointt in self.params.robot_descr.joints:
                if jointt.type != 'fixed':
                    robot_joints.append(jointt)
                    jj +=1
                    if jj == n_dofs:
                        break

        joint_names = [joint.name for joint in robot_joints]
        self.kin_dyn = KinDynComputations(params.robot_urdf, joint_names, self.params.robot_descr.get_root())        
        self.kin_dyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
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

        # EE jacobian
        self.jac = self.kin_dyn.jacobian_fun(params.frame_name)

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints]) 
        joint_effort = np.array([joint.limit.effort for joint in robot_joints]) 
        joint_effort = joint_effort[:nq]

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])

        # EE target
        self.ee_ref = self.params.ee_ref

        # Cartesian constraints
        self.obs_string = self.params.obs_string
        self.joint_names = joint_names

        # Analytic or network set
        if self.params.use_net == True:
            self.safe_set = NetSafeSet(self.x,self.p,self.nq,self.params)
            self.net_name = '_net'
        elif self.params.use_net == False: 
            self.safe_set = AnalyticSafeSet(self.x,self.x_min,self.x_max,self.nq,self.params,self.t_glob,self.jac)
            self.net_name = 'analytic_set'
        else:
            self.safe_set=None
            self.net_name = ''
        # Capsules end-points forward kinematics
        n_cap=0
        for capsule in self.params.robot_capsules:
            capsule['index']=n_cap
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
            fk_capsule_points = self.kin_dyn.forward_kinematics_fun(capsule['link_name'])   
            T_capsule_points = fk_capsule_points(np.eye(4), self.x[:self.nq])@rot_mat
            capsule['end_points_fk'] = deepcopy([(T_capsule_points @ capsule['end_points'][0])[:3],
                                                    (T_capsule_points @ capsule['end_points'][1])[:3]])
            capsule['end_points_T_fun'] = deepcopy(cs.Function(f'fun_T_{n_cap}',[self.x],[T_capsule_points]))
            capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.x],[(T_capsule_points @ capsule['end_points'][0])[:3],
                                                                                                    (T_capsule_points @ capsule['end_points'][1])[:3]]))
            n_cap += 1
        for capsule in self.params.obst_capsules:
            capsule['index']=n_cap
            capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.x],[capsule['end_points'][0], capsule['end_points'][1]]))
            n_cap += 1

        self.collisions_constr_fun = self.gen_collisions_constr_fun()

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
        return self.safe_set.check_constraint(x) 
    
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
    
    def checkCollision(self, x):
        for pair in self.collisions_constr_fun:
            if not(pair[1]<=pair[0](x)<=pair[2]):
                return False
        return True

            # capsules_pos = []
            # for capsule in self.params.robot_capsules:
            #     capsules_pos.append(capsule['end_points_fk_fun'](x))
            # for capsule in self.params.obst_capsules:
            #     capsules_pos.append(capsule['end_points'])
            # for pair in self.params.collisions_pairs:
            #     if pair['type'] == 0:
            #         if not(self.casadi_segment_dist(*capsules_pos[pair['elements'][0]['index']],*capsules_pos[pair['elements'][1]['index']])): 
            #             return False
            #     elif pair['type'] == 1:
            #         if not(self.ball_segment_dist(*capsules_pos[pair['elements'][0]['index']],pair['elements'][0]['length'],pair['elements'][1]['position'])>=(pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2): 
            #             return False
            #     elif pair['type'] == 2:
            #         if not(capsules_pos[pair['elements'][0]['index']][0][2] >=  pair['elements'][1]['bounds'][0]): return False
            #         if not(capsules_pos[pair['elements'][0]['index']][0][2] <=  pair['elements'][1]['bounds'][1]): return False
            #         if not(capsules_pos[pair['elements'][0]['index']][1][2] >=  pair['elements'][1]['bounds'][0]): return False
            #         if not(capsules_pos[pair['elements'][0]['index']][1][2] <=  pair['elements'][1]['bounds'][1]): return False
            # return True

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
    
    def ball_segment_dist(self,A_s,B_s,capsule_length,obs_pos):
        t = cs.fmin(cs.fmax(cs.dot((obs_pos-A_s),(B_s-A_s)) / (capsule_length**2),0),1)
        d = cs.sum1((obs_pos-(A_s+(B_s-A_s)*t))**2) 
        return d
    
    def generate_NLconstraints_list(self):
        constraint_list_0 = []
        constraint_list_1_N_minus_1 = []
        constraint_list_N = []

        # dynamics always present
        constraint_list_0.append([self.tau,self.tau_min,self.tau_max])
        constraint_list_1_N_minus_1.append([self.tau,self.tau_min,self.tau_max])

        # collisions
        for pair in self.params.collisions_pairs:
            if pair['type'] == 0:
                constraint_list_0.append([self.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']), \
                                          (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2,1e6])
                constraint_list_1_N_minus_1.append([self.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']), \
                                          (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2,1e6])
                constraint_list_N.append([self.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']), \
                                          (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2,1e6])
            elif pair['type'] == 1:
                constraint_list_0.append([self.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']), \
                                          (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2,1e6])
                constraint_list_1_N_minus_1.append([self.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']), \
                                          (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2,1e6])
                constraint_list_N.append([self.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']), \
                                          (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2,1e6])
            elif pair['type'] == 2:
                for point in pair['elements'][0]['end_points_fk']:
                    constraint_list_0.append([point[2],pair['elements'][1]['bounds'][0],pair['elements'][1]['bounds'][1]])
                    constraint_list_1_N_minus_1.append([point[2],pair['elements'][1]['bounds'][0],pair['elements'][1]['bounds'][1]])
                    constraint_list_N.append([point[2],pair['elements'][1]['bounds'][0],pair['elements'][1]['bounds'][1]])
        return constraint_list_0,constraint_list_1_N_minus_1,constraint_list_N

    def gen_collisions_constr_fun(self):
        fun_list = []
        for pair in self.params.collisions_pairs:
            if pair['type'] == 0:
                fun_list.append([cs.Function(f"{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[self.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk'])]), \
                                          (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2,1e6])
            elif pair['type'] == 1:
                fun_list.append([cs.Function(f"{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[self.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position'])]), \
                                          (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2,1e6])
            elif pair['type'] == 2:
                for i in range(len(pair['elements'][0]['end_points'])):
                    fun_list.append([cs.Function(f"{pair['elements'][0]['name']}_{pair['elements'][1]['name']}_{i}",[self.x],[pair['elements'][0]['end_points_fk'][i][2]])\
                                     ,pair['elements'][1]['bounds'][0],pair['elements'][1]['bounds'][1]])
        return fun_list
