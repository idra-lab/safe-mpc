import os
import numpy as np
import matplotlib.pyplot as plt


class Debug:
    def __init__(self, conf, controller):
        plt.rcParams['axes.grid'] = True
        self.horizon = np.arange(0, conf.N * conf.dt, conf.dt)
        self.conf = conf
        self.controller = controller
        # Create the temporary directories for debugging purposes
        cwd = os.getcwd()
        self.plot_dir = os.path.join(cwd, 'plots')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def plotBounds(self, t, ax):
        ax.plot(t, self.conf.q_max * np.ones_like(t), color='red', linestyle='--')
        ax.plot(t, self.conf.q_min * np.ones_like(t), color='green', linestyle='--')

    def plotTrajectory(self, test_num, x0, x_guess, x_sim=None):
        fig, ax = plt.subplots(3, 1, sharex='col')
        for i in range(3):
            if x_sim is not None:
                t_sim = np.arange(0, np.shape(x_sim)[0] * self.conf.dt, self.conf.dt)
                self.plotBounds(t_sim, ax[i])
                ax[i].plot(t_sim, x_sim[:, i], label='q' + str(i + 1), color='darkblue',
                           linestyle='--', linewidth=2)
            else:
                self.plotBounds(self.horizon, ax[i])
            ax[i].plot(self.horizon, x_guess[:-1, i], label='q_g' + str(i + 1), color='darkgreen', linewidth=1.5)
            ax[i].scatter(x=0, y=x0[i], c='r')
            ax[i].legend()
            ax[i].set_ylabel('q' + str(i + 1) + ' (rad)')
        ax[2].set_xlabel('Time (s)')
        plt.savefig(self.plot_dir + '/pos' + str(test_num) + '.png')
        plt.close()
