import sys
import numpy as np 
import example_robot_data
import pinocchio as pin 
import matplotlib.pyplot as plt
from safe_mpc.parser import Parameters
from safe_mpc.model import DoublePendulumModel, TriplePendulumModel
from safe_mpc.gravity_compensation import GravityCompensation
from safe_mpc.abstract import SimDynamics


conf = Parameters('../config/params.yaml')
dt = conf.dt_s

model_name = sys.argv[1] if len(sys.argv) > 1 else 'triple'
if model_name == 'double':
    model = DoublePendulumModel(conf)
    x0 = np.array([np.pi + 0.2, np.pi - 0.3, 0., 0.])
    robot = example_robot_data.load('double_pendulum_simple')
    rmodel = robot.model
    rdata = rmodel.createData()
else:
    model = TriplePendulumModel(conf)
    x0 = np.array([np.pi + 0.2, np.pi - 0.3, np.pi + 0.1, 0., 0., 0.])
simulator = SimDynamics(model) 
gc = GravityCompensation(conf, model)

n_step = 100 
x_sim = np.empty((n_step + 1, model.nx)) * np.nan
x_sim[0, :] = x0
q = np.empty((n_step + 1, model.nq)) * np.nan
q[0, :] = x0[:model.nq]
v = np.empty((n_step + 1, model.nv)) * np.nan
v[0, :] = x0[model.nq:]

# Control = gravity compensation + constant
u0 = gc.solve(x0)
u = u0 #+ 0.01

for i in range(n_step):
    # Acados dynamics simulation
    x_sim[i+1, :] = simulator.simulate(x_sim[i, :], u)

    # Pinocchio simulation
    pin.computeAllTerms(rmodel, rdata, q[i], v[i])
    a = pin.aba(rmodel, rdata, q[i], v[i], u)
    v[i+1] = v[i] + dt * a
    q[i+1] = pin.integrate(rmodel, q[i], dt * v[i+1])   #q[i] + v[i] * dt + dt**2 / 2 * a

fig, ax = plt.subplots(model.nq, 1, sharex='col')
for i in range(model.nq):
    ax[i].plot(x_sim[:, i], label='sim')
    ax[i].plot(q[:,i], label='pin', color='r', linestyle='dashed')
    ax[i].legend()
    ax[i].set_ylabel('q' + str(i + 1) + ' (rad)')
    ax[i].grid()
ax[model.nq - 1].set_xlabel('Time (s)')
plt.savefig('test.png')
plt.close()
