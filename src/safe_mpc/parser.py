import os
import yaml
import numpy as np
import argparse
from urdf_parser_py.urdf import URDF
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='z1',
                        help='Systems to test. Available: pendulum, double_pendulum, ur5, z1')
    parser.add_argument('-d', '--dofs', type=int, default=4,
                        help='Number of desired degrees of freedom of the system')
    parser.add_argument('-c', '--controller', type=str, default='naive',
                        help='Controllers to test. Available: naive, st, stwa, htwa, receding')
    parser.add_argument('-b', '--build', action='store_true',
                        help='Build the code of the embedded controller')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Alpha parameter for the NN safety factor')
    parser.add_argument('--horizon', type=int, default=45,
                        help='Horizon of the optimal control problem')
    parser.add_argument('-a', '--activation', type=str, default='gelu',
                        help='Activation function for the neural network')
    parser.add_argument('--back_hor',type=int,default=45,help='Horizon of the backup controller')
    return vars(parser.parse_args())

def align_vectors(a, b):
    """
    Function used only for obstacle capsule visualization purposes. Its purpose is to find the transformation matrix to get vector b when applied to a.
    The matrix is saved in as the transformation for fixed capsule, and a used there is always [0,1,0] because it is the default cylinder's 
    axis direction (a capsule is rendered using 3 solids: 2 sphere and 1 cylinder).
    """
    b = b / np.linalg.norm(b) # normalize a
    a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)

    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        return -np.eye(3, dtype=np.float64)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                     [v3, 0, -v1],
                     [-v2,v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R

class Parameters:
    def __init__(self, urdf_name, rti=True, filename=None):
        self.urdf_name = urdf_name
        # Define all the useful paths
        self.PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = self.PKG_DIR.split('/src/safe_mpc')[0]
        self.CONF_DIR = os.path.join(self.ROOT_DIR, 'config/')
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data/')
        self.GEN_DIR = os.path.join(self.ROOT_DIR, 'generated/')
        self.NN_DIR = os.path.join(self.ROOT_DIR, 'nn_models/' + urdf_name + '/')
        self.ROBOTS_DIR = os.path.join(self.ROOT_DIR, 'robots/')
            
        self.robot_urdf = f'{self.ROBOTS_DIR}/{urdf_name}_description/urdf/{urdf_name}.urdf'

        self.robot_descr = URDF.from_xml_file(self.robot_urdf)
        self.links = [self.robot_descr.links[i].name for i in range(len(self.robot_descr.links))]
        self.joints = [self.robot_descr.joints[i] for i in range(len(self.robot_descr.joints))]

        if filename is None:
            parameters = yaml.load(open(self.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)
        else:
            parameters = yaml.load(open(filename), Loader=yaml.FullLoader)

        self.test_num = int(parameters['test_num'])
        self.n_steps = int(parameters['n_steps'])
        self.cpu_num = int(parameters['cpu_num'])
        self.build = False
        
        self.N = int(parameters['N'])
        self.back_hor = int(parameters['back_hor'])
        self.dt = float(parameters['dt'])
        
        self.alpha = float(parameters['alpha'])
        self.act = str(parameters['act_fun'])
        nls = {
            'relu': torch.nn.ReLU(),
            'elu': torch.nn.ELU(),
            'tanh': torch.nn.Tanh(),
            'gelu': torch.nn.GELU(approximate='tanh'),
            'silu': torch.nn.SiLU()
        }
        self.act_fun = nls[self.act]
        self.net_size = parameters['network_size']
        self.use_net = bool(parameters['use_net'])
        self.n_dof_safe_set = int(parameters['n_dof_safe_set'])
        self.net_path = str(parameters['network_path'])

        self.reg_term_analytic_constr = float(parameters['reg_term'])

        self.nq = int(parameters['n_dofs'])
        self.ee_ref = np.array(parameters['ee_ref'])
        self.ee_pos = np.array(parameters['ee_position'])

        self.solver_type = 'SQP_RTI' if rti else 'SQP'
        self.solver_mode = parameters['solver_mode']
        self.nlp_max_iter = int(parameters['rti_iter']) if rti else int(parameters['nlp_max_iter'])
        self.qp_max_iter = int(parameters['qp_max_iter'])
        self.alpha_reduction = float(parameters['alpha_reduction'])
        self.alpha_min = float(parameters['alpha_min'])
        self.levenberg_marquardt = float(parameters['levenberg_marquardt'])
        self.ext_flag = parameters['ext_flag']

        self.ipopt_opts = dict() 
        for entry in parameters['ipopt_opts']:
            self.ipopt_opts[entry] = parameters['ipopt_opts'][entry]
        
        self.tol_x = float(parameters['tol_x'])
        self.tol_tau = float(parameters['tol_tau'])
        self.tol_dyn = float(parameters['tol_dyn'])
        self.tol_obs = float(parameters['tol_obs'])
        self.tol_safe_set = float(parameters['tol_safe_set'])

        self.Q = np.array(parameters['Q'])
        self.R = float(parameters['R'])         # eye(nu) * R
        self.eps = float(parameters['eps'])
        self.tol_conv = float(parameters['tol_conv'])
        self.tol_cost = float(parameters['tol_cost'])
        self.globalization = 'FIXED_STEP' if rti else 'MERIT_BACKTRACKING'

        self.q_dot_gain = float(parameters['q_dot_gain'])
        self.ws_t = float(parameters['ws_t'])
        self.ws_r = float(parameters['ws_r'])

        # For cartesian constraint
        self.obs_flag = bool(parameters['obs_flag'])
        self.abort_flag = bool(parameters['abort_flag'])
        
        self.frame_name = parameters['frame_ee'] 
        self.ee_radius = float(parameters['ee_radius'])

        self.obs_string = parameters['obs_string']

        self.ddq_max = np.array(parameters['ddq_max'])
        self.ddx_max = np.array(parameters['ddx_max'])

        # collision pairs

        # obstacles
        self.obstacles = []
        for obstacle in parameters['obstacles']:
            obs=dict()
            for entry in obstacle:
                if type(obstacle[entry]) == list: obs[entry] = np.array(obstacle[entry]).astype(float)
                else: obs[entry] = obstacle[entry]
            self.obstacles.append(obs)

        self.use_capsules=bool(parameters['use_capsules'])
        # capsules
        # robot capsules
        self.robot_capsules = []
        if parameters['robot_capsules'] != None:
            for capsule in parameters['robot_capsules']:
                self.robot_capsules.append(self.create_moving_capsule(capsule))
        # fixed capsule
        self.obst_capsules = []
        if parameters['obstacles_capsules'] != None:
            for capsule in parameters['obstacles_capsules']:
                self.obst_capsules.append(self.create_fixed_capsule(capsule))
                    
        self.collisions_pairs = []
        if self.use_capsules:
        # assign pairs 
            if parameters['collision_pairs'] == None:
                for capsule_one in self.robot_capsules:
                    for capsule_two in self.robot_capsules:
                        if capsule_one['name'] != capsule_two['name']:
                            self.collisions_pairs.append(self.assign_pairs(capsule_one['name'],capsule_two['name'], self.obstacles, self.robot_capsules))
                    for capsule_two in self.obst_capsules:
                        self.collisions_pairs.append(self.assign_pairs(capsule_one['name'],capsule_two['name'], self.obstacles, self.obst_capsules))
                    for obst in self.obstacles:
                        self.collisions_pairs.append(self.assign_pairs(capsule_one['name'],obst['name'], self.obstacles, self.obstacles))
            else:
                for pair in parameters['collision_pairs']:
                    self.collisions_pairs.append(self.assign_pairs(pair[0],pair[1],self.obstacles,self.robot_capsules+self.obst_capsules)) 
        
        self.track_traj = bool(parameters['track_traj'])
        if self.track_traj: 
            self.n_steps=int(parameters['n_steps_tracking'])
            self.dim_shape_8 = float(parameters['dim_shape_8'])
            self.offset_traj = np.array(parameters['offset_traj'])
            self.theta_rot_traj = np.array(parameters['theta_rot_traj'])
            self.vel_max_traj = float(parameters['vel_max_traj'])
            self.vel_const = bool(parameters['vel_const'])

        self.noise_mass = float(parameters['noise_mass'])
        self.noise_inertia = float(parameters['noise_inertia'])
        self.noise_cm = float(parameters['noise_cm'])


    def create_moving_capsule(self,capsule):
        """
        Function to create a capsule fixed to a robot link. Assign to the object a name, and supply the name of the URDF link on which the capsule is applied.
        Radius and length define the capsule's dimension, link axis represents on which axis the end point of the capsule is placed, at a distance
        equal to the length of the capsule, from the starting point of the capsule's segment. Usually axis x , for x give 0, for y 1, for z 2.
        Spatial offset permits to select the position of the starting point with respect the frame of link, as defined in the URDF. Rotation offset contains the angles
        for a sequence of rotation z-y-x, w.r.t. to the local link frame, to adjust the capsule orientation. Finally color is required only for visualization,
        the first 3 values represent the RGB triple, the last is the transparency (0 total transparency, 1 no transparency).
        """
        # first point defined by offset from link origin, second length offset from the first one 
        # capsule['end_points'] = [np.hstack((spatial_offset,np.ones(1))), np.hstack((spatial_offset,np.ones(1)))]
        capsule['type'] = 'moving_capsule'
        capsule['end_points'] = [np.hstack((np.zeros(3),np.ones(1))), np.hstack((np.zeros(3),np.ones(1)))]
        capsule['direction'] = np.sign(self.joints[self.links.index(capsule['link_name'])].origin.xyz[capsule['link_axis']])
        capsule['end_points'][1][capsule['link_axis']] += capsule['direction']*capsule['length']
        capsule['end_points_T_fun'] = [None]
        capsule['end_points_fk'] = [None,None]
        capsule['end_points_fk_fun'] = [None]
        return capsule

    def create_fixed_capsule(self,capsule):
        """
        Create a fixed capsule, assigning it a name and a radius for the shape. The length instead is determined by the end points of the capsule's
        segment, the two next arguments. Color works as in the case of the moving capsule. 
        """
        capsule['type'] = 'fixed_capsule'
        capsule['end_points'] = np.array([capsule['point_A'],capsule['point_B']])
        capsule['length'] = np.linalg.norm(capsule['end_points'][0]-capsule['end_points'][1])
        capsule['end_points_fk'] = capsule['end_points']
        capsule['end_points_T_fun'] = align_vectors(np.array([0,1,0]),capsule['end_points'][1]-capsule['end_points'][0])     
        return capsule

    def assign_pairs(self,obj1_name,obj2_name,obstacles_list,capsules_list):
        """
        Assign the collision pairs. The arguments are the name of the capsules or of the obstacle. A capsule must be always present. For the
        moment, in case of obstacle, give it as second argument.
        Based on the types of the pairing objects, a type is assigned.
        """
        args=[obj1_name,obj2_name]
        for obstacle in obstacles_list:
            if obj1_name == obstacle['name']:
                args=args.reverse()
        pair=dict()
        pair['elements'] = [None,None]
        pair['type'] = None
        for capsule in capsules_list:
            if obj1_name == capsule['name']:
                pair['elements'][0] = capsule
                break
        for capsule in capsules_list:
            if obj2_name == capsule['name']:
                pair['elements'][1] = capsule
                if pair['elements'][0] != None:
                    pair['type'] = 0
                break 
        for obstacle in obstacles_list:
            if obj2_name == obstacle['name']:
                pair['elements'][1] = obstacle
                if (pair['elements'][0] != None and obstacle['type'] == 'sphere'): pair['type'] = 1
                elif (pair['elements'][0] != None and obstacle['type'] == 'box'): pair['type'] = 2
                break 
        return pair
