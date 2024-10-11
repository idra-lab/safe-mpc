import adam.numpy
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
from urdf_parser_py.urdf import URDF
import adam
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles, ee_ref, get_controller


args = parse_args()
params = Parameters('z1', rti=False)
params.build = args['build']
model = AdamModel(params, n_dofs=4)

controller = get_controller(args['controller'], model, obstacles)
controller.setReference(ee_ref)

# joint_effort = np.array([2., 23., 10., 4.])
# lh = np.copy(controller.ocp.constraints.lh)
# uh = np.copy(controller.ocp.constraints.uh)
# lh[:model.nq] = -joint_effort
# uh[:model.nq] = joint_effort
# model.tau_min = -joint_effort
# model.tau_max = joint_effort
# for i in range(controller.N):
#     controller.ocp_solver.constraints_set(i, "lh", lh)
#     controller.ocp_solver.constraints_set(i, "uh", uh)

robot = URDF.from_xml_file(params.robot_urdf)
robot_joints = robot.joints[1:model.nq + 1]
joint_names = [joint.name for joint in robot_joints]
kin_dyn = adam.numpy.KinDynComputations(params.robot_urdf, joint_names, robot.get_root())
kin_dyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
H_b = np.eye(4)

num_ics = params.test_num
i = 0
fails = 0
H_b = np.eye(4)
sampler = qmc.Halton(model.nq, scramble=False)
x_guess, u_guess = [], []
progress_bar = tqdm(total=num_ics, desc='Generating initial conditions')
while i < num_ics:
    q0 = qmc.scale(sampler.random(), model.x_min[:model.nq], model.x_max[:model.nq])[0]
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = q0
    if controller.checkCollision(x0):
    # T_ee = kin_dyn.forward_kinematics(params.frame_name, H_b, q0)
    # ee_pos = T_ee[:3, 3] + T_ee[:3, :3] @ model.t_loc
    # # Check if ee satisfy our desired initial conditions
    # if np.all(np.logical_and(ee_pos >= params.box_lb, ee_pos <= params.box_ub)):
        u0 = model.gravity(H_b, q0)[6:]
        if controller.initialize(x0, u0.T):
            i += 1    
            (xg, ug) = controller.getGuess() 
            x_guess.append(xg), u_guess.append(ug)
            progress_bar.update(1)
        else:
            fails += 1
progress_bar.close()
print(f'Number of failed initializations: {fails}')

x_guess = np.asarray(x_guess)
u_guess = np.asarray(u_guess)
x_init = x_guess[:,0,:]
np.save(params.DATA_DIR + 'ics.npy', x_init)   
