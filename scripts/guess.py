import os 
import pickle
import adam.numpy
import numpy as np
import time
from tqdm import tqdm
from urdf_parser_py.urdf import URDF
import adam
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref
from safe_mpc.controller import NaiveController


args = parse_args()
params = Parameters('z1', False)
params.build = args['build']
model = AdamModel(params, n_dofs=4)

controller = NaiveController(model, obstacles)
controller.setReference(ee_ref)

robot = URDF.from_xml_file(params.robot_urdf)
robot_joints = robot.joints[1:model.nq + 1]
joint_names = [joint.name for joint in robot_joints]
kin_dyn = adam.numpy.KinDynComputations(params.robot_urdf, joint_names, robot.get_root())
kin_dyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
H_b = np.eye(4)

x_min_box = np.array([-1.0, 2., -2.5, model.x_min[3]])
x_max_box = np.array([-0.4, model.x_max[1], -1.4, model.x_max[3]])

num_ics = params.test_num
i = 0
fails = 0
H_b = np.eye(4)
x_guess, u_guess = [], []
progress_bar = tqdm(total=num_ics, desc='Generating initial conditions')
while i < num_ics:
    q0 = np.random.uniform(model.x_min, model.x_max)[:model.nq]
    T_ee = kin_dyn.forward_kinematics(params.frame_name, H_b, q0)
    ee_pos = T_ee[:3, 3] + T_ee[:3, :3] @ model.t_loc
    # Check if ee satisfy our desired initial conditions
    if np.all(np.logical_and(ee_pos >= params.box_lb, ee_pos <= params.box_ub)):
        x0 = np.zeros((model.nx,))
        x0[:model.nq] = q0
        u0 = model.gravity(H_b, q0)[6:]
        if controller.initialize(x0, u0.T):
            i += 1    
            (x_g, u_g) = controller.getGuess() 
            x_guess.append(x_g), u_guess.append(u_g)
            progress_bar.update(1)
        else:
            fails += 1
progress_bar.close()
print(f'Number of failed initializations: {fails}')

with open(params.DATA_DIR + 'guess.pkl', 'wb') as f:
    pickle.dump({'x': np.asarray(x_guess), 'u': np.asarray(u_guess)}, f)