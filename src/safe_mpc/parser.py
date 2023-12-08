import os
import yaml
import numpy as np


class Parameters:
    def __init__(self, filename):
        # Define all the useful paths
        self.PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = self.PKG_DIR.split('/src/safe_mpc')[0]
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data/')
        self.GEN_DIR = os.path.join(self.ROOT_DIR, 'generated/')
        self.NN_DIR = os.path.join(self.DATA_DIR, 'relu_3dof/')

        # Load the parameters from the yaml file
        data = yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
        self.n_steps = int(data['n_steps'])
        self.test_num = int(data['test_num'])
        self.cpu_num = int(data['cpu_num'])
        self.regenerate = bool(data['regenerate'])

        model = data['model']
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

        simulator = data['simulator']
        self.dt_s = float(simulator['dt'])
        self.integrator_type = int(simulator['integrator_type'])

        controller = data['controller']
        self.dt = float(controller['dt'])
        self.T = float(controller['T'])
        self.N = int(self.T / self.dt)
        self.x_ref = np.array(controller['x_ref'])
        self.solver_type = controller['solver_type']
        self.solver_mode = controller['solver_mode']
        self.nlp_max_iter = int(controller['nlp_max_iter'])
        self.qp_max_iter = int(controller['qp_max_iter'])
        self.globalization = controller['globalization']
        self.alpha = int(controller['safety_margin'])
        self.ws_r = float(controller['ws_r'])
        self.ws_t = float(controller['ws_t'])

        self.Q = np.eye(6) * 1e-4
        self.R = np.eye(3) * 1e-4
        self.Q[0, 0] = 500
