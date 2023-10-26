import numpy as np

# MPC parameters
dt = 5e-3
T = 0.18
N = int(T / dt)
n_steps = 100
test_num = 100
cpu_num = 20

# Joint limits
q_min = 3 / 4 * np.pi
q_max = 5 / 4 * np.pi
dq_max = 10
u_max = 10

# Constants
g = 9.81
m1 = 0.4
m2 = 0.4
m3 = 0.4
l1 = 0.8
l2 = 0.8
l3 = 0.8

# LQR parameters
Q = np.eye(6) * 1e-4
R = np.eye(3) * 1e-4
Q[0, 0] = 500

name = "triple_pendulum"
regenerate = True
solver_type = "SQP"
qp_max_iter = 100
