import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from safe_mpc.parser import Parameters
from safe_mpc.env_model import AdamModel
from safe_mpc.cost_definition import *

params = Parameters('z1', True)
model = AdamModel(params)
cost = Tracking8NLS(model, params.Q_weight, params.R_weight)
traj = cost.traj[:, :500]

plt.figure()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ball = Circle((0., 0.6), 0.12, color='g')
ax.add_patch(ball)
# ball1 = Circle((-0.2, 0.6), 0.05)
# ball2 = Circle((0.2, 0.6), 0.05)
# ax.add_patch(ball1)
# ax.add_patch(ball2)
ax.scatter(traj[1, 0], traj[0, 0], c='g', marker='o')
ax.plot(traj[1, :], traj[0, :], c='r', ls='--')
ax.set_xlabel('y')
ax.set_ylabel('x')

plt.figure()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ball = Circle((0., 0.12), 0.12, color='g')
ax.add_patch(ball)
# ball1 = Circle((-0.2, 0.12), 0.05)
# ball2 = Circle((0.2, 0.12), 0.05)
# ax.add_patch(ball1)
# ax.add_patch(ball2)
ax.scatter(traj[1, 0], traj[2, 0], c='g', marker='o')
ax.plot(traj[1, :], traj[2, :], c='r', ls='--')
ax.set_xlabel('y')
ax.set_ylabel('z')


plt.show()