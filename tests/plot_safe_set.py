import numpy as np
import torch
import torch.nn as nn
from safe_mpc.parser import Parameters
from safe_mpc.abstract import AdamModel, NeuralNetwork
from safe_mpc.controller import NaiveController
from safe_mpc.utils import obstacles
import matplotlib.pyplot as plt
import matplotlib.patches as patches


params = Parameters('z1', rti=True)
model = AdamModel(params, n_dofs=4)
controller = NaiveController(model, obstacles)

sm = 15
nq = model.nq
grid = 1e-2

ub = np.sqrt(nq) * max(model.x_max[nq:])
nn_model = NeuralNetwork(model.nx, 256, 1, nn.Tanh(), ub)
nn_data = torch.load(f'{params.NN_DIR}model_{nq}dof{model.obs_add}.pt',
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
    if params.obs_flag:
        pts = np.empty(0)
        for j in range(len(x)):
            if not controller.checkCollision(x[j]):
                pts = np.append(pts, x[j, i])
        if len(pts) > 0:
            origin = (np.min(pts), model.x_min[i + nq])
            width = np.max(pts) - np.min(pts)
            height = model.x_max[i + nq] - model.x_min[i + nq]
            rect = patches.Rectangle(origin, width, height, linewidth=1, edgecolor='black', facecolor='black')
            plt.gca().add_patch(rect)

    plt.xlim([model.x_min[i], model.x_max[i]])
    plt.ylim([model.x_min[i + nq], model.x_max[i + nq]])
    plt.xlabel('q_' + str(i + 1))
    plt.ylabel('dq_' + str(i + 1))
    plt.grid()
    plt.title(f"Classifier section joint {i + 1}, horizon {controller.N}")
    plt.savefig(f'data/{i + 1}dof_{controller.N}_BRS.png')
