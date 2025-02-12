import os
import yaml
import numpy as np
import argparse


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
    return vars(parser.parse_args())


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
        # temp solution
        if urdf_name == 'ur5':
            self.robot_urdf = f'{self.ROBOTS_DIR}/ur_description/urdf/{urdf_name}_robot.urdf'
        else:
            self.robot_urdf = f'{self.ROBOTS_DIR}/{urdf_name}_description/urdf/{urdf_name}.urdf'

        if filename is None:
            parameters = yaml.load(open(self.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)
        else:
            parameters = yaml.load(open(filename), Loader=yaml.FullLoader)

        self.test_num = int(parameters['test_num'])
        self.n_steps = int(parameters['n_steps'])
        self.cpu_num = int(parameters['cpu_num'])
        self.build = False
        
        self.N = int(parameters['N'])
        self.dt = float(parameters['dt'])
        self.alpha = float(parameters['alpha'])
        self.act = 'tanh' if urdf_name == 'z1' else 'relu'

        self.solver_type = 'SQP_RTI' if rti else 'SQP'
        self.solver_mode = parameters['solver_mode']
        self.nlp_max_iter = int(parameters['rti_iter']) if rti else int(parameters['nlp_max_iter'])
        self.qp_max_iter = int(parameters['qp_max_iter'])
        self.alpha_reduction = float(parameters['alpha_reduction'])
        self.alpha_min = float(parameters['alpha_min'])
        self.levenberg_marquardt = float(parameters['levenberg_marquardt'])
        self.ext_flag = parameters['ext_flag']
        
        self.tol_x = float(parameters['tol_x'])
        self.tol_tau = float(parameters['tol_tau'])
        self.tol_dyn = float(parameters['tol_dyn'])
        self.tol_obs = float(parameters['tol_obs'])
        self.tol_nn = float(parameters['tol_nn'])

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
        self.frame_name = 'gripperMover'       #  TODO: dependence on the robot

        self.box_lb = np.array([0.45, -0.55, 0.])
        self.box_ub = np.array([0.75, -0.25, 0.3])

        # For triple pendulum
        self.g = 9.81
        self.m1 = 0.4
        self.m2 = 0.4
        self.m3 = 0.4
        self.l1 = 0.8
        self.l2 = 0.8
        self.l3 = 0.8

        self.q_min = 3 / 4 * np.pi
        self.q_max = 5 / 4 * np.pi
        self.dq_lim = 10
        self.tau_lim = 10