import numpy as np
import casadi as ca


class GravityCompensation:
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.opti = ca.Opti()
        self.x = self.opti.parameter(model.nx)
        self.u = self.opti.variable(model.nu)
        self.dynamics = ca.Function('dyn', [model.x, model.u], [model.f_expl[model.nq:]])
        self.opti.minimize(1)
        self.opti.subject_to(self.dynamics(self.x, self.u) == 0)    
        self.opti.subject_to(self.opti.bounded(self.model.u_min, self.u, self.model.u_max))
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.linear_solver': 'ma57'}
        self.opti.solver('ipopt', opts)

    def solve(self, x):
        self.opti.set_value(self.x, x)
        sol = self.opti.solve()
        return sol.value(self.u)


if __name__ == "__main__":
    from safe_mpc.parser import Parameters 
    from safe_mpc.model import TriplePendulumModel
    from safe_mpc.abstract import SimDynamics

    params = Parameters('triple_pendulum', 'naive')
    system = TriplePendulumModel(params)
    simulator = SimDynamics(system)
    gc = GravityCompensation(params, system)

    x0 = np.array([3.5, -2.7, 3.6, 0, 0, 0])
    u_static = gc.solve(x0)
    print('Gravity compensation torque: ', u_static)

    # Simulate the system 
    n_step = 100
    x_sim = np.empty((n_step + 1, system.nx)) * np.nan
    x_sim[0] = x0
    for i in range(n_step):
        x_sim[i + 1] = simulator.simulate(x_sim[i], u_static) 

    # Plot the results
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, sharex='col')
    for i in range(3):
        ax[i].plot(x_sim[:, i], label='q' + str(i + 1))
        ax[i].legend()
        ax[i].grid()
    ax[2].set_xlabel('Time (s)')
    plt.savefig('gravity_compensation.png')
    plt.close()
