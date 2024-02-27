import os
import yaml
import numpy as np
from functools import reduce
import matplotlib as mpl
import matplotlib.pyplot as plt


class PlotUtils:
    def __init__(self, conf):
        self.conf = conf
        params = yaml.load(open(conf.ROOT_DIR + '/config/plot.yaml'), Loader=yaml.FullLoader)
        mpl.rcdefaults()
        mpl.rcParams['pdf.fonttype'] = int(params['fonttype']['pdf'])
        mpl.rcParams['ps.fonttype'] = int(params['fonttype']['ps'])
        mpl.rcParams['lines.linewidth'] = float(params['lines']['linewidth'])
        mpl.rcParams['lines.markersize'] = float(params['lines']['markersize'])
        mpl.rcParams['patch.linewidth'] = float(params['patch']['linewidth'])
        mpl.rcParams['axes.grid'] = bool(params['axes']['grid'])
        mpl.rcParams['axes.labelsize'] = float(params['axes']['labelsize'])
        mpl.rcParams['font.family'] = params['font']['family']
        mpl.rcParams['font.size'] = float(params['font']['size'])
        mpl.rcParams['text.usetex'] = bool(params['text']['usetex'])
        mpl.rcParams['legend.fontsize'] = float(params['legend']['fontsize'])
        mpl.rcParams['legend.loc'] = params['legend']['loc']
        mpl.rcParams['figure.facecolor'] = params['figure']['facecolor']
        self.plot_dir = os.path.join(conf.ROOT_DIR, 'plots')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def plot_task_abortion(self, k, N, x, x_wa, x_abort, nn_out):
        fig, ax = plt.subplots(4, 1, sharex='col', figsize=(22, 24))
        plt.subplots_adjust(top=0.96, right=0.95, bottom=0.1)
        t = np.arange(0, len(x) * self.conf.dt, self.conf.dt)
        t_wa = np.arange(0, k * self.conf.dt, self.conf.dt)
        t_abort = np.arange(0, self.conf.T + self.conf.dt, self.conf.dt) + t_wa[-1]
        t_longest = reduce(np.union1d, ([t, t_wa, t_abort]))
        nj = x.shape[1] // 2
        for j in range(nj):
            ax[j].plot(t_longest, self.conf.q_max * np.ones_like(t_longest), color='purple',
                       linestyle='dotted', label=r'$q^{\rm{max}}$')
            # ax[j].plot(t, self.conf.q_min * np.ones_like(t), color='purple',
            #            linestyle='dotted', label=r'$q^{\rm{min}}$')
            ax[j].plot(t_wa[:k], x_wa[:k, j], label='STWA', color='b')
            ax[j].plot(t, x[:, j], label=r'ST', color='r', linestyle='dashed')
            ax[j].plot(t_abort, x_abort[:, j], label=r'Abort', color='g')
            ax[j].axvline(x=t_wa[k - 1], color='black', linestyle='-.', linewidth=6)
            ax[j].set_ylabel(fr'$q_{j + 1}$ (rad)')
        ax[1].legend()
        ax[-1].plot(t_wa[:k], nn_out, color='b', label=r'$(1 - \alpha) \phi(x_N) - || \dot q ||^2$')
        ax[-1].plot(t_longest, np.zeros_like(t_longest), color='darkorange', linestyle='dashed', linewidth=8)
        ax[-1].axvline(x=t_wa[k - N], color='black', linestyle='-.', linewidth=6)
        ax[-1].legend()
        ax[-1].set_xlabel('Time (s)')
        ax[-1].set_ylabel('NN output')
