import pickle
import numpy as np
import matplotlib.pyplot as plt
from safe_mpc.utils import apply_rc_params


apply_rc_params()
plt.rcParams.update({'axes.grid': True})
data = pickle.load(open("logs.pkl", "rb"))
x, u, r = data["x"], data["u"], data["r"]
bounds, coll, solver = data["bounds"], data["coll"], data["solver"]
nn_r, nn_last = data["nn_r"], data["nn_last"]
timings = [data["solver_time"] * 1e3, data["tot_time"] * 1e3]


t = np.arange(len(x)) * 1e-2

fig, ax = plt.subplots(5, 1, figsize=(24, 8), sharex=True)
ax[0].plot(t, r, label="Receding position", c='r')
ax[0].set_ylabel("r value")
ax[1].plot(t, nn_r, label="NN Receding position", c='b')
ax[1].plot(t, nn_last, label="NN Last position", c='g', ls='--')
ax[1].set_ylabel("NN out")
ax[1].legend()
ax[2].plot(t[:-1], bounds, label="Bounds", c='r')
ax[2].set_ylabel("Bounds")
ax[3].plot(t[:-1], coll, label="Collision", c='b')
ax[3].set_ylabel("Collision")
ax[4].plot(t[:-1], solver, label="Solver", c='g')
ax[4].set_ylabel("Solver")

ax[-1].set_xlabel("Time (s)")

plt.figure(figsize=(10, 8))
plt.boxplot(timings, labels=["Solver", "Total"])
plt.ylabel("Time (ms)")


plt.show()

print(f'99th percentile of solver time: {np.percentile(timings[0], 0.99)} ms')
print(f'99.9th percentile of solver time: {np.percentile(timings[0], 0.999)} ms')
print(f'Max solver time: {np.max(timings[0])} ms')
print(f'99th percentile of total time: {np.percentile(timings[1], 0.99)} ms')
print(f'99.9th percentile of total time: {np.percentile(timings[1], 0.999)} ms')
print(f'Max total time: {np.max(timings[1])} ms')