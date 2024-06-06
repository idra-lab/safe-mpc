import os
import yaml
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='double_pendulum',
                        help='Systems to test. Available: pendulum, double_pendulum, ur5, z1')
    parser.add_argument('--dofs', type=int, default=False, nargs='?',
                        help='Number of desired degrees of freedom of the system')
    parser.add_argument('-c', '--controller', type=str, default='naive',
                        help='Controllers to test. Available: naive, st, stwa, htwa, receding')
    parser.add_argument('-i', '--init-conditions', action='store_true',
                        help='Find the initial conditions for testing all the controller')
    parser.add_argument('-g', '--guess', action='store_true',
                        help='Compute the initial guess of a given controller')
    parser.add_argument('--rti', action='store_true',
                        help='Use SQP-RTI for the MPC solver')
    parser.add_argument('--alpha', type=int, default=2,
                        help='Safety margin for the NN model')
    parser.add_argument('-a', '--abort', type=str, default=None,
                        help='Define the MPC formulation for which the abort controller is tested. '
                             'Available: stwa, htwa, receding')
    parser.add_argument('--repetition', type=int, default=5,
                        help='Number of repetitions for the abort controller')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results')
    return vars(parser.parse_args())


class Parameters:
    def __init__(self, urdf_name, rti=True):
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

        parameters = yaml.load(open(self.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)

        self.test_num = int(parameters['test_num'])
        self.n_steps = int(parameters['n_steps'])
        self.cpu_num = int(parameters['cpu_num'])
        self.regenerate = bool(parameters['regenerate'])
        
        self.T = float(parameters['T'])
        self.dt = float(parameters['dt'])
        self.alpha = float(parameters['alpha'])

        self.solver_type = 'SQP_RTI' if rti else 'SQP'
        self.solver_mode = parameters['solver_mode']
        self.nlp_max_iter = int(parameters['nlp_max_iter'])
        self.qp_max_iter = int(parameters['qp_max_iter'])
        self.qp_tol_stat = float(parameters['qp_tol_stat'])
        self.nlp_tol_stat = float(parameters['nlp_tol_stat'])
        self.alpha_reduction = float(parameters['alpha_reduction'])
        self.alpha_min = float(parameters['alpha_min'])
        self.levenberg_marquardt = float(parameters['levenberg_marquardt'])

        self.state_tol = float(parameters['state_tol'])
        self.conv_tol = float(parameters['conv_tol'])
        self.cost_tol = float(parameters['cost_tol'])
        self.globalization = 'FIXED_STEP' if rti else 'MERIT_BACKTRACKING'

        self.joint_target = int(parameters['joint_target'])
        self.ubound_gap = float(parameters['ubound_gap'])
        self.q_dot_gain = float(parameters['q_dot_gain'])
        self.ws_t = float(parameters['ws_t'])
        self.ws_r = float(parameters['ws_r'])

        # For cartesian constraint
        self.obs_flag = bool(parameters['obs_flag'])
        self.frame_name = 'link2'       #  TODO: dependence on the robot