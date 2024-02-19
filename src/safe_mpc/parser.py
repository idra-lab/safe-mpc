import os
import yaml
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='triple_pendulum',
                        help='Systems to test. Available: double_pendulum, triple_pendulum')
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
                             'Available: stwa, htwa and receding')
    parser.add_argument('--repetition', type=int, default=5,
                        help='Number of repetitions for the abort controller')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results')
    return vars(parser.parse_args())


class Parameters:
    def __init__(self, m_name, cont_type, rti=True):
        # Define all the useful paths
        self.PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = self.PKG_DIR.split('/src/safe_mpc')[0]
        self.CONF_DIR = os.path.join(self.ROOT_DIR, 'config/')
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data/')
        self.GEN_DIR = os.path.join(self.ROOT_DIR, 'generated/')
        self.NN_DIR = os.path.join(self.ROOT_DIR, 'nn_models/' + m_name + '/')

        # Load the parameters from the yaml files
        model = yaml.load(open(self.CONF_DIR + 'models/' + m_name + '.yaml'), Loader=yaml.FullLoader)
        self.l1 = float(model['l1'])
        self.l2 = float(model['l2'])
        self.l3 = float(model['l3'])
        self.g = float(model['g'])
        self.m1 = float(model['m1'])
        self.m2 = float(model['m2'])
        self.m3 = float(model['m3'])
        self.q_min = (1 + float(model['q_min'])) * np.pi
        self.q_max = (1 + float(model['q_max'])) * np.pi
        self.dq_min = float(model['dq_min'])
        self.dq_max = float(model['dq_max'])
        self.u_min = float(model['u_min'])
        self.u_max = float(model['u_max'])
        self.state_tol = float(model['state_tol'])
        self.eps = float(model['eps'])
        self.joint_target = int(model['joint_target'])
        self.ubound_gap = float(model['ubound_gap'])

        simulator = yaml.load(open(self.CONF_DIR + 'simulator.yaml'), Loader=yaml.FullLoader)
        self.dt_s = float(simulator['dt'])
        self.integrator_type = simulator['integrator_type']
        self.num_stages = int(simulator['num_stages'])

        controller = yaml.load(open(self.CONF_DIR + 'controller.yaml'), Loader=yaml.FullLoader)
        self.test_num = int(controller['test_num'])
        self.n_steps = int(controller['n_steps'])
        self.cpu_num = int(controller['cpu_num'])
        self.regenerate = bool(controller['regenerate'])

        self.nlp_max_iter = int(controller['nlp_max_iter'])
        self.qp_max_iter = int(controller['qp_max_iter'])
        self.solver_type = 'SQP_RTI' if rti else 'SQP'
        self.solver_mode = controller['solver_mode']
        self.alpha = int(controller['alpha'])
        self.conv_tol = float(controller['conv_tol'])
        self.globalization = 'FIXED_STEP' if rti else 'MERIT_BACKTRACKING'

        self.dt = float(controller[cont_type]['dt'])
        self.T = float(controller[cont_type]['T'])

        if cont_type not in ['naive', 'abort']:
            self.ws_t = float(controller[cont_type]['ws_t'])
            if cont_type == 'receding':
                self.ws_r = float(controller[cont_type]['ws_r'])

        if cont_type == 'abort':
            self.q_dot_gain = float(controller[cont_type]['q_dot_gain'])
