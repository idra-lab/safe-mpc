import re
import numpy as np
from copy import deepcopy
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
import casadi as cs
from casadi import MX, vertcat, Function
from acados_template import AcadosModel
import scipy.linalg as lin
import torch.nn as nn
import l4casadi as l4c
from .safe_set import NetSafeSet, AnalyticSafeSet
import xml.etree.ElementTree as ET
from .utils import rot_mat_x,rot_mat_y,rot_mat_z, casadi_segment_dist,ball_segment_dist,sphere_sphere_dist,plane_sphere_dist, randomize_model

class AdamModel:
    def __init__(self, params):
        self.params = params
        self.amodel = AcadosModel()
        # Robot dynamics with Adam (IIT)                
        robot_joints = []
        jj=0
        for jointt in self.params.robot_descr.joints:
            if jointt.type != 'fixed':
                robot_joints.append(jointt)
                jj +=1
                if jj == self.params.nq:
                    break

        joint_names = [joint.name for joint in robot_joints]

        randomize_model(params.robot_urdf, noise_mass_percentage = self.params.noise_mass , noise_inertia_percentage = self.params.noise_inertia, noise_cm_position_percentage = self.params.noise_cm)

        # Formal assumed model, used by controller
        self.kin_dyn = KinDynComputations(params.robot_urdf, joint_names, self.params.robot_descr.get_root())        
        self.kin_dyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
        self.mass = self.kin_dyn.mass_matrix_fun()                           # Mass matrix
        self.bias = self.kin_dyn.bias_force_fun()                            # Nonlinear effects  
        self.gravity = self.kin_dyn.gravity_term_fun()                       # Gravity vector
        self.fk = self.kin_dyn.forward_kinematics_fun(params.frame_name)     # Forward kinematics

        # Model with noise
        self.kin_dyn_noisy = KinDynComputations(params.robot_urdf[:-5] + '_randomized.urdf', joint_names, self.params.robot_descr.get_root())        
        self.kin_dyn_noisy.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
        self.mass_noisy = self.kin_dyn_noisy.mass_matrix_fun()                           # Mass matrix
        self.bias_noisy = self.kin_dyn_noisy.bias_force_fun()                            # Nonlinear effects  
        self.gravity_noisy = self.kin_dyn_noisy.gravity_term_fun()                       # Gravity vector
        self.fk_noisy = self.kin_dyn_noisy.forward_kinematics_fun(params.frame_name)     # Forward kinematics

        nq = len(joint_names)

        self.amodel.name = params.urdf_name
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
        self.nq = self.params.nq
        self.nv = nq

        # Inverse dynamics, torque computation
        H_b = np.eye(4)
        self.tau = self.mass(H_b, self.x[:nq])[6:, 6:] @ self.u + \
                   self.bias(H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:]
        self.tau_fun = Function('tau', [self.x, self.u], [self.tau])

        # Noisy dynamics
        H_b = np.eye(4)
        self.tau_noisy = self.mass_noisy(H_b, self.x[:nq])[6:, 6:] @ self.u + \
                   self.bias_noisy(H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:]
        self.tau_noisy_fun = Function('tau', [self.x, self.u], [self.tau_noisy])

        # EE position (global frame)
        T_ee = self.fk(np.eye(4), self.x[:nq])
        self.t_loc = self.params.ee_pos
        self.t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ self.t_loc
        self.ee_fun = Function('ee_fun', [self.x], [self.t_glob])

        # Noisy EE position (global frame)
        T_ee_noisy = self.fk_noisy(np.eye(4), self.x[:nq])
        self.t_loc_noisy = self.params.ee_pos
        self.t_glob_noisy = T_ee_noisy[:3, 3] + T_ee_noisy[:3, :3] @ self.t_loc_noisy
        self.ee_fun_noisy = Function('ee_fun', [self.x], [self.t_glob_noisy])

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
        
        # Capsules end-points forward kinematics
        n_cap=0
        for capsule in self.params.robot_capsules:
            capsule['index']=n_cap
            rot_mat=np.eye(4)
            if capsule['rotation_offset'] != None:
                th_off=capsule['rotation_offset']
                rot_mat = rot_mat_x(th_off[0])@rot_mat_y(th_off[1])@rot_mat_z(th_off[2])
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
            capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.x],[capsule['end_points_fk'][0][:3],
                                                                                            capsule['end_points_fk'][1][:3]]))
            n_cap += 1
        for capsule in self.params.obst_capsules:
            capsule['index']=n_cap
            capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.x],[capsule['end_points'][0], capsule['end_points'][1]]))
            n_cap += 1
        n_cap = 0
        for sphere in self.params.spheres_robot:
            fk_sphere = self.kin_dyn.forward_kinematics_fun(sphere['link_name'])
            T_sphere = fk_sphere(np.eye(4), self.x[:nq])
            sphere['fk'] = T_sphere[:3,3] +T_sphere[:3, :3]@ sphere['spatial_offset']      
            sphere['fk_fun'] = Function(f'sphere_fk_{n_cap}', [self.x], [sphere['fk']])
            sphere['index']=n_cap
            n_cap +=1

        self.NL_external = self.generate_NLconstraints_list()

    def jointToEE(self, x):
        return np.array(self.ee_fun(x))

    def checkStateConstraints(self, x):
        return np.all(np.logical_and(x >= self.x_min - self.params.tol_x, 
                                     x <= self.x_max + self.params.tol_x)) and \
                                     self.checkCollision(x)

    def checkStateBounds(self, x):
        return np.all(np.logical_and(x >= self.x_min - self.params.tol_x, 
                                     x <= self.x_max + self.params.tol_x))

    def checkTorqueConstraints(self, x,u):
        tau = np.array([self.tau_fun(x[i], u[i]).T for i in range(len(u))])
        return np.all(np.logical_and(tau >= self.tau_min - self.params.tol_tau, 
                                     tau <= self.tau_max + self.params.tol_tau))
    
    def checkTorqueBounds(self,tau):
        return np.all(np.logical_and(tau >= self.tau_min - self.params.tol_tau, 
                                     tau <= self.tau_max + self.params.tol_tau))

    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) and self.checkTorqueBounds(x,u)

    
    def integrate(self, x, u):
        x_next = np.zeros(self.nx)
        tau = np.array(self.tau_noisy_fun(x, u).T)
        # Cannot exceed the torque limits --> sat and compute forward dynamics on real system 
        H_b = np.eye(4)
        tau_sat = np.clip(tau, self.tau_min, self.tau_max)
        M = np.array(self.mass_noisy(H_b, x[:self.nq])[6:, 6:])
        h = np.array(self.bias_noisy(H_b, x[:self.nq], np.zeros(6), x[self.nq:])[6:])
        u = np.linalg.solve(M, (tau_sat.T - h)).T
        # x_next[:self.nq] = x[:self.nq] + self.params.dt * x[self.nq:] + 0.5 * self.params.dt**2 * u
        # x_next[self.nq:] = x[self.nq:] + self.params.dt * u
        x_next = np.array(self.f_fun(x,u)).squeeze()
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
        x_tmp = np.atleast_2d(deepcopy(x))
        for i in range(len(x_tmp)):
            for pair in self.collisions_constr_fun:
                if not(pair[1]<=pair[0](x_tmp[i])<=pair[2]):
                    return False
            return True

    
    def generate_NLconstraints_list(self):
        """
        Generate list of nonlinear constraints with bounds, for nodes 0, 1 - N-1, and N, as well as the list of casadi function of the collision constraints
        """
        constraint_list_0 = []
        constraint_list_1_N_minus_1 = []
        constraint_list_N = []

        # generate also list of collision function for collision checks
        self.collisions_constr_fun = []

        # torque limits constraints present
        constraint_list_0.append([self.tau,self.tau_min,self.tau_max])
        constraint_list_1_N_minus_1.append([self.tau,self.tau_min,self.tau_max])

        # collisions
        # if self.params.use_capsules:
        for i,pair in enumerate(self.params.collisions_pairs):
            if pair['type'] == 'capsule-capsule':
                constraint_list_0.append([casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']), \
                                        (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2,1e6])
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}",[self.x],[constraint_list_N[-1][0]]), \
                                        constraint_list_N[-1][1]-self.params.tol_obs,constraint_list_N[-1][2]+self.params.tol_obs])
            elif pair['type'] == 'capsule-sphere':
                constraint_list_0.append([ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']), \
                                        (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2,1e6])
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}",[self.x],[constraint_list_N[-1][0]]), \
                                        constraint_list_N[-1][1]-self.params.tol_obs,constraint_list_N[-1][2]+self.params.tol_obs])
            elif pair['type'] == 'capsule-plane':
                for point in pair['elements'][0]['end_points_fk']:
                    constraint_list_0.append([point[pair['elements'][1]['perpendicular_axis']],pair['elements'][1]['bounds'][0]+pair['elements'][0]['radius'],pair['elements'][1]['bounds'][1]-pair['elements'][0]['radius']])
                    constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                    constraint_list_N.append(constraint_list_0[-1])
                    self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}",[self.x],[constraint_list_N[-1][0]]), \
                                        constraint_list_N[-1][1]-self.params.tol_obs,constraint_list_N[-1][2]+self.params.tol_obs])
            elif pair['type'] == 'sphere-sphere':
                constr_expr = sphere_sphere_dist(pair['elements'][1],pair['elements'][0]['fk'])
                constraint_list_0.append([constr_expr, (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2,1e6])
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}",[self.x],[constraint_list_N[-1][0]]), \
                                        constraint_list_N[-1][1]-self.params.tol_obs,constraint_list_N[-1][2]+self.params.tol_obs])
            if pair['type'] == 'sphere-plane':
                constr_expr = plane_sphere_dist(pair['elements'][1],pair['elements'][0]['fk'])
                constraint_list_0.append([constr_expr,pair['elements'][1]['bounds'][0] + pair['elements'][0]['radius'],pair['elements'][1]['bounds'][1] - pair['elements'][0]['radius']])
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}",[self.x],[constraint_list_N[-1][0]]), \
                                        constraint_list_N[-1][1]-self.params.tol_obs,constraint_list_N[-1][2]+self.params.tol_obs])
        # else:
        #     for i,obs in enumerate(self.params.obstacles):
        #         if obs['type'] == 'sphere':
        #             constr_expr = sphere_sphere_dist(obs,self.t_glob)
        #             constraint_list_0.append([constr_expr, (self.params.ee_radius+obs['radius'])**2,1e6])
        #             constraint_list_1_N_minus_1.append(constraint_list_0[-1])
        #             constraint_list_N.append(constraint_list_0[-1])
        #             self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}",[self.x],[constraint_list_N[-1][0]]), \
        #                                   constraint_list_N[-1][1]-self.params.tol_obs,constraint_list_N[-1][2]+self.params.tol_obs])
        #         if obs['type'] == 'plane':
        #             constr_expr = plane_sphere_dist(obs,self.t_glob)
        #             constraint_list_0.append([constr_expr,obs['bounds'][0] + self.params.ee_radius,obs['bounds'][1] - self.params.ee_radius])
        #             constraint_list_1_N_minus_1.append(constraint_list_0[-1])
        #             constraint_list_N.append(constraint_list_0[-1])
        #             self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}",[self.x],[constraint_list_N[-1][0]]), \
        #                                   constraint_list_N[-1][1]-self.params.tol_obs,constraint_list_N[-1][2]+self.params.tol_obs])

        return constraint_list_0,constraint_list_1_N_minus_1,constraint_list_N
