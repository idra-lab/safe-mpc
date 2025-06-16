import numpy as np
import torch
import torch.nn as nn
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel, NeuralNetwork
from safe_mpc.controller import NaiveController
from safe_mpc.utils import obstacles, capsules, capsule_pairs, ee_ref
import matplotlib.pyplot as plt
import matplotlib.patches as patches


args = parse_args()
params = Parameters('z1', rti=True)
params.build = args['build']
params.alpha = args['alpha']
model = AdamModel(params, args['dofs'])
model.ee_ref = ee_ref
controller = NaiveController(model, obstacles, capsules, capsule_pairs)

sm = args['alpha']
nq = model.nq
grid = 1e-2

ub = np.sqrt(nq) * max(model.x_max[nq:])
nn_model = NeuralNetwork(model.nx, 128, 1, nn.GELU(approximate='tanh'))
# nn_data = torch.load(f'{params.NN_DIR}{nq}dof_gelu{model.obs_string}.pt',
#                      map_location=torch.device('cpu'))
nn_data = torch.load(f'../nn_models/z1/5dof_gelu_dt10ms_ALPHA_10.pt',
                     map_location=torch.device('cpu'))
nn_model.load_state_dict(nn_data['model'])
mean = nn_data['mean']
std = nn_data['std']

for i in range(nq):
    plt.figure()

    q, v = np.meshgrid(np.arange(model.x_min[i], model.x_max[i] + grid, grid),
                       np.arange(model.x_min[i + nq], model.x_max[i + nq] + grid, grid))
    q_rav, v_rav = q.ravel(), v.ravel()
    n = len(q_rav)

    x_static = (model.x_max + model.x_min) / 2
    x = np.repeat(x_static.reshape(1, len(x_static)), n, axis=0)
    x[:, i] = q_rav
    x[:, nq + i] = v_rav

    # Compute velocity norm
    y = np.linalg.norm(x[:, nq:], axis=1)

    x_in = np.copy(x)
    # Normalize position
    x_in[:, :nq] = (x[:, :nq] - mean) / std
    # Velocity direction
    x_in[:, nq:] /= y.reshape(len(y), 1)

    # Predict
    with torch.no_grad():
        y_pred = nn_model(torch.from_numpy(x_in.astype(np.float32))).cpu().numpy()
    y_pred *= (100 - sm) / 100
    out = np.array([0 if y[j] > y_pred[j] else 1 for j in range(n)])
    z = out.reshape(q.shape)
    plt.contourf(q, v, z, cmap='coolwarm', alpha=0.8)

    # Remove the joint positions s.t. robot collides with obstacles 
    # if params.obs_flag:
    #     pts = np.empty(0)
    #     for j in range(len(x)):
    #         if not controller.checkCollision(x[j]):
    #             pts = np.append(pts, x[j, i])
    #     if len(pts) > 0:
    #         origin = (np.min(pts), model.x_min[i + nq])
    #         width = np.max(pts) - np.min(pts)
    #         height = model.x_max[i + nq] - model.x_min[i + nq]
    #         rect = patches.Rectangle(origin, width, height, linewidth=1, edgecolor='black', facecolor='black')
    #         plt.gca().add_patch(rect)

    if params.obs_flag:
        in_collision = []
        for j in range(len(x)):
            collides = not(controller.checkCollision(x[j]))
            in_collision.append(collides)

        # Find contiguous segments of collisions
        start = None
        for j in range(len(x)):
            if in_collision[j]:
                if start is None:
                    start = j
            else:
                if start is not None:
                    # End of a collision segment
                    pts_segment = x[start:j, i]
                    origin = (np.min(pts_segment), model.x_min[i + nq])
                    width = np.max(pts_segment) - np.min(pts_segment)
                    height = model.x_max[i + nq] - model.x_min[i + nq]
                    rect = patches.Rectangle(origin, width, height, linewidth=1, edgecolor='black', facecolor='black')
                    plt.gca().add_patch(rect)
                    start = None

        # Handle the case where the last points are in collision
        if start is not None:
            pts_segment = x[start:, i]
            origin = (np.min(pts_segment), model.x_min[i + nq])
            width = np.max(pts_segment) - np.min(pts_segment)
            height = model.x_max[i + nq] - model.x_min[i + nq]
            rect = patches.Rectangle(origin, width, height, linewidth=1, edgecolor='black', facecolor='black')
            plt.gca().add_patch(rect)

    plt.xlim([model.x_min[i], model.x_max[i]])
    plt.ylim([model.x_min[i + nq], model.x_max[i + nq]])
    plt.xlabel('q_' + str(i + 1))
    plt.ylabel('dq_' + str(i + 1))
    plt.grid()
    plt.title(f"Joint {i + 1}, Horizon {controller.N}, Safety margin {sm}")
    plt.savefig(f'data/{i + 1}dof_{controller.N}_BRS.png')

plt.show()
