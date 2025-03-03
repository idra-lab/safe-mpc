import numpy as np
import casadi as cs
from copy import deepcopy


class NaiveOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, obstacles=None, capsules=None, capsule_pairs=None):
        self.params = model.params
        self.model = model
        self.nq = model.nq
        self.obstacles = obstacles
        self.capsules = capsules
        self.capsule_pairs = capsule_pairs

        N = self.params.N
        opti = cs.Opti()
        x_init = opti.parameter(model.nx)
        cost = 0

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]
        for k in range(N):
            X += [opti.variable(model.nx)]
            opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
            U += [opti.variable(model.nu)]

        opti.subject_to(X[0] == x_init)
        
        Q = self.params.Q
        R = self.params.R * np.eye(self.model.nu)
        ee_ref = model.ee_ref
        dist_b = []

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
            elif capsule['type'] == 'fixed':
                capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.model.x],[capsule['end_points'][0], capsule['end_points'][1]]))
            n_cap += 1

        for k in range(N + 1):
                
            ee_pos = model.ee_fun(X[k])
            ee_rot = model.ee_rot(X[k])
            cost += (ee_pos - ee_ref).T @ Q @ (ee_pos - ee_ref)
            # cost += cs.trace((np.eye(3) - ee_rot) @ Q)

            if k < N:
                cost += U[k].T @ R @ U[k]
                # Dynamics constraint
                opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
                # Torque constraints
                opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))

            # if obstacles is not None and self.params.obs_flag:
            #     # Collision avoidance
            #     for obs in obstacles:
            #         ee_pos = model.ee_fun(X[k])
            #         if obs['name'] == 'floor':
            #             lb = obs['bounds'][0]
            #             ub = obs['bounds'][1]
            #             opti.subject_to(opti.bounded(lb, ee_pos[2], ub))
            #         elif obs['name'] == 'ball':
            #             lb = obs['bounds'][0]
            #             ub = obs['bounds'][1]
            #             dist_b += [(ee_pos - obs['position']).T @ (ee_pos - obs['position'])]
            #             opti.subject_to(opti.bounded(lb, dist_b[-1], ub))

            for pair in self.capsule_pairs:
                if pair['type'] == 0:
                    dist = self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk_fun'](X[k]),*pair['elements'][1]['end_points_fk_fun'](X[k]))
                    lb = (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2
                    ub = 1e6
                    opti.subject_to(opti.bounded(lb, dist, ub))
                elif pair['type'] == 1:
                    dist = self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk_fun'](X[k]),pair['elements'][0]['length'],pair['elements'][1]['position'])
                    lb = (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2
                    ub = 1e6
                    opti.subject_to(opti.bounded(lb, dist, ub))
                elif pair['type'] == 2:
                    for point in pair['elements'][0]['end_points_fk_fun'](X[k]):
                        lb = pair['elements'][1]['bounds'][0]
                        ub = pair['elements'][1]['bounds'][1]
                        opti.subject_to(opti.bounded(lb, point[2], ub))

        opti.minimize(cost)
        self.opti = opti
        self.X = X
        self.U = U
        self.x_init = x_init    
        self.cost = cost
        self.dist_b = dist_b
        self.additionalSetting()

    def additionalSetting(self):
        pass

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

    def instantiateProblem(self):
        opti = self.opti
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.constr_viol_tol': 1e-6,
            'ipopt.compl_inf_tol': 1e-6,
            'ipopt.hessian_approximation': 'limited-memory',
            # 'detect_simple_bounds': 'yes',
            'ipopt.max_iter': self.params.nlp_max_iter,
            'ipopt.linear_solver': 'ma57',
            'ipopt.sb': 'yes'
        }

        opti.solver('ipopt', opts)
        return opti


class TerminalZeroVelOCP(NaiveOCP):
    def __init__(self, model, obstacles=None, capsules=None, capsule_pairs=None):
        super().__init__(model, obstacles, capsules, capsule_pairs)

    def additionalSetting(self):
        self.opti.subject_to(self.X[-1][self.nq:] == 0.)


class HardTerminalOCP(NaiveOCP):
    def __init__(self, model, obstacles=None, capsules=None, capsule_pairs=None):
        super().__init__(model, obstacles, capsules, capsule_pairs)

    def additionalSetting(self):
        self.model.setNNmodel()
        self.opti.subject_to(self.model.nn_func(self.X[-1], self.params.alpha) >= 0.)
        nq = self.model.nq
        nn_dofs = self.params.nn_dofs
        if nq > nn_dofs:
            x_middle = (self.model.x_min[nn_dofs:nq] \
                        + self.model.x_max[nn_dofs:nq]) / 2
            self.opti.subject_to(self.X[-1][nn_dofs:nq] == x_middle)
            self.opti.subject_to(self.X[-1][nq + nn_dofs:] == 0.)


class SoftTerminalOCP(NaiveOCP):
    def __init__(self, model, obstacles=None, capsules=None, capsule_pairs=None):
        super().__init__(model, obstacles, capsules, capsule_pairs)

    def additionalSetting(self):
        s_N = self.opti.variable(1)
        self.model.setNNmodel()
        self.opti.subject_to(self.model.nn_func(self.X[-1], self.params.alpha) + s_N >= 0.)
        self.opti.subject_to(
            self.opti.bounded(0., s_N, 1e6)
        )
        self.cost += self.params.ws_t * s_N
        nq = self.model.nq
        nn_dofs = self.params.nn_dofs
        if nq > nn_dofs:
            s_bound = self.opti.variable(nq - nn_dofs)
            x_middle = (self.model.x_min[nn_dofs:nq] \
                        + self.model.x_max[nn_dofs:nq]) / 2
            self.opti.subject_to(self.X[-1][nn_dofs:nq] + s_bound == x_middle)
            self.opti.subject_to(
                self.opti.bounded(-1 * np.ones(nq - nn_dofs), s_bound, 1 * np.ones(nq - nn_dofs))
            )
            self.opti.subject_to(self.X[-1][nq + nn_dofs:] == 0.)


class SafeAbortOCP(NaiveOCP):
    def __init__(self, model, obstacles=None, capsules=None, capsule_pairs=None):
        super().__init__(model, obstacles, capsules, capsule_pairs)

    def additionalSetting(self):
        cost = 0
        Q = 1e-4 * np.eye(self.model.nx)
        # Penalize only the velocity
        Q[self.model.nq:, self.model.nq:] = 1e4 * np.eye(self.model.nq)
        R = 5e-3 * np.eye(self.model.nu)
        for k in range(self.params.N):
            cost += self.X[k].T @ Q @ self.X[k] 
            cost += self.U[k].T @ R @ self.U[k]
        cost += self.X[-1].T @ Q @ self.X[-1]
        self.opti.minimize(cost)
        self.opti.subject_to(self.X[-1][self.nq:] == 0.)
        self.cost = cost