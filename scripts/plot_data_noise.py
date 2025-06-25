import pickle
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def custom_log_formatter(x, pos):
    return f'{int(x):,}'

mpl.rcdefaults()
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['patch.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 25
mpl.rcParams['text.usetex'] = True
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['figure.facecolor'] = 'white'

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True)
hor = args['horizon']
alpha = int(args['alpha'])
model = AdamModel(params)

cont_names = ['naive','zerovel', 'st', 'htwa', 'receding']

colors = ['tomato', 'mediumblue',  'darkorange', 'limegreen', 'darkgreen', 'purple','coral','peru']
markers = ['o', 'x', '*', 's', '^', '>','H','D']
labels = ['Naive', 'Zerovel', 'ST', 'HTWA', 'Receding']

horizons = [45]
noises = [0.1, 1.3, 3.7, 5.0]

margin_joints=[0.1,1.3,2.6,5.0,10.0]
margin_collision=[0.005, 0.008, 0.01, 0.02, 0.03]

failures = {}



def plot_horizons_fails_noise(alpha, noise_level):
    for c in cont_names:
        failures[c] = {}
        for h in horizons:
            failures[c][h] = {}
            for noise in noises:
                data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{hor}hor_{int(alpha)}sm_noise{noise}_control_noise{args["control_noise"]}_q_collision_margins_{args["joint_bounds_margin"]}_{args["collision_margin"]}_scores.pkl', 'rb'))
                failures[c][h][noise] = data_tmp[c]['fails']

    fig, ax = plt.subplots(figsize=(10, 7))
    for i,c in enumerate(cont_names):
        fails = []
        for hor in failures[c]:
            fails.append(failures[c][hor][noise_level])
        ax.plot(horizons, np.array(fails)/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])
    ax.set_xlabel("Horizons (num of nodes)")
    ax.set_ylabel(r"Failures (\%)")
    ax.legend(fancybox=True, framealpha=0.5)
    plt.xticks(horizons)
    plt.tight_layout()
    plt.title(f'Alpha = {alpha}\%, noise = {noise_level}\%')
    plt.savefig(f'{params.DATA_DIR}horizons_vs_fails_noise{noise_level}.svg', bbox_inches='tight',transparent=True)
    plt.savefig(f'{params.DATA_DIR}horizons_vs_fails_noise{noise_level}.png', bbox_inches='tight',transparent=False)

    #plt.show()
    #plt.close()

def plot_noise_level(alpha, horizon):
    for c in cont_names:
        failures[c] = {}
        for h in horizons:
            failures[c][h] = {}
            for noise in noises:
                data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{hor}hor_{int(alpha)}sm_noise{noise}_control_noise{args["control_noise"]}_q_collision_margins_{args["joint_bounds_margin"]}_{args["collision_margin"]}_scores.pkl', 'rb'))
                failures[c][h][noise] = data_tmp[c]['fails']

    fig, ax = plt.subplots(figsize=(10, 7))
    for i,c in enumerate(cont_names):
        fails = []
        for noise in noises:
            fails.append(failures[c][horizon][noise])
        ax.plot(noises, np.array(fails)/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])
    ax.set_xlabel("Noise level [\%]")
    ax.set_ylabel(r"Failures [\%]")
    ax.legend(fancybox=True, framealpha=0.5)
    #plt.xticks(ticks=np.arange(1, len(cont_names) + 1), labels=cont_names)    
    plt.xticks(noises)
    plt.tight_layout()
    plt.title(f'Alpha = {alpha}\%, horizon = {horizon}')
    plt.savefig(f'{params.DATA_DIR}horizons_vs_fails_horizon{horizon}.svg', bbox_inches='tight',transparent=True)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}.png', bbox_inches='tight',transparent=False)

    plt.show()
    plt.close()

def plot_control_noise_level(alpha, horizon,margin_joints_index):
    for c in cont_names:
        failures[c] = {}
        for h in horizons:
            failures[c][h] = {}
            for noise in noises:
                failures[c][h][noise]= {}
                for jj in range(len(margin_joints)):
                    
                    data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{hor}hor_{int(alpha)}sm_noise{0.0}_control_noise{noise}_q_collision_margins_{margin_joints[jj]}_{margin_collision[jj]}_scores.pkl', 'rb'))
                    failures[c][h][noise][margin_joints[jj]] = data_tmp[c]['fails']

    
    fig, ax = plt.subplots(figsize=(10, 7))
    for i,c in enumerate(cont_names):
        fails = []
        for noise in noises:
            fails.append(failures[c][horizon][noise][margin_joints[margin_joints_index]])
        ax.plot(noises, np.array(fails)/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])
    ax.set_xlabel("Noise level [\%]")
    ax.set_ylabel(r"Failures [\%]")
    ax.legend(fancybox=True, framealpha=0.5)
    #plt.xticks(ticks=np.arange(1, len(cont_names) + 1), labels=cont_names)    
    plt.xticks(noises)
    plt.tight_layout()
    plt.title(f'Joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collision[margin_joints_index]} m, horizon = {horizon}', fontsize=20)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}_joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collision[margin_joints_index]} m, horizon = {horizon}.svg', bbox_inches='tight',transparent=True)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}_joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collision[margin_joints_index]} m, horizon = {horizon}.png', bbox_inches='tight',transparent=False)

    plt.show()
    plt.close()


#plot_horizons_fails_noise(20,2.5)
for i in range(len(margin_joints)):
    plot_control_noise_level(20,45,i)




