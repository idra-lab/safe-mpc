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
params = Parameters(args, model_name, rti=True)
hor = args['horizon']
alpha = int(args['alpha'])
model = AdamModel(params)

cont_names = ['naive','zerovel', 'st', 'htwa', 'receding','parallel2', 'constraint_everywhere','receding_parallel']

colors = ['tomato', 'mediumblue',  'darkorange', 'limegreen', 'darkgreen', 'purple','peru','coral']
markers = ['o', 'x', '*', 's', '^', '>','H','D']
labels = ['Naive', 'Zerovel', 'ST', 'HTWA', 'Receding','Parallel','ConstraintEverywhere','Receding2Problems']

horizons = [25]
noises = [0.1, 1.3, 2.5, 3.7, 5.0]
noises = [2.5, 5.0, 10.0, 15.0, 20.0]

margin_joints = [0.1 ,0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 2.6, 5.0]
margin_collisions = [0.001, 0.001, 0.0015, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008]

cont_names = cont_names[:-2]

failures = {}
costs = {}



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
                    
                    data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{h}hor_{int(alpha)}sm_noise{noise}_control_noise{0.0}_q_collision_margins_{margin_joints[jj]}_{margin_collisions[jj]}_scores.pkl', 'rb'))
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
    plt.title(f'Joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collisions[margin_joints_index]} m, horizon = {horizon}', fontsize=20)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}_joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collisions[margin_joints_index]} m, horizon = {horizon}.svg', bbox_inches='tight',transparent=True)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}_joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collisions[margin_joints_index]} m, horizon = {horizon}.png', bbox_inches='tight',transparent=False)

    plt.show()
    plt.close()

def plot__noise_margins(alpha, horizon,noise):
    for c in cont_names:
        failures[c] = {}
        failures[c][horizon] = {}
        
        failures[c][horizon][noise]= {}
        for jj in range(len(margin_joints)):
            
            data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{horizon}hor_{int(alpha)}sm_noise{noise}_control_noise{0.0}_q_collision_margins_{margin_joints[jj]}_{margin_collisions[jj]}_scores.pkl', 'rb'))
            failures[c][horizon][noise][margin_joints[jj]] = data_tmp[c]['fails']

    
    fig, ax = plt.subplots(figsize=(10, 7))
    x_values = np.arange(len(margin_joints))

    labels_margin = [f'{joint_m} \n {collision_m}' for (joint_m,collision_m) in zip(margin_joints,margin_collisions)]
    for i,c in enumerate(cont_names):
        fails = []
        for jj in range(len(margin_joints)):
            fails.append(failures[c][horizon][noise][margin_joints[jj]])
        ax.plot(x_values, np.array(fails)/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])

    plt.xticks(ticks=x_values, labels=labels_margin)
    ax.set_xlabel("Joint margin [\%] and collision margin [m]",labelpad=20)
    ax.tick_params(axis='x', labelsize=22)  
    ax.tick_params(axis='y', labelsize=22)
    ax.set_ylabel(r"Failures [\%]")
    #ax.legend(fancybox=True, framealpha=0.5)
    #plt.xticks(ticks=np.arange(1, len(cont_names) + 1), labels=cont_names)    
    #plt.xticks(margin_joints)
    plt.tight_layout()
    plt.title(f'Noise = {noise}\%, horizon = {horizon}', fontsize=20)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}, varying margin, horizon = {horizon}.svg', bbox_inches='tight',transparent=True)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}, varying margin, horizon = {horizon}.png', bbox_inches='tight',transparent=False)

    plt.show()
    plt.close()

def find_costs(horizon,noise,margin_joints,costs):
    completed = costs[cont_names[0]][horizon][noise][margin_joints][1]
    for c in cont_names[0:]:
        completed = list(set(completed) & set(costs[c][horizon][noise][margin_joints][1]))
        print(f'Controller {c} mean cost: {np.mean(np.array(costs[c][horizon][noise][margin_joints][0])[costs[c][horizon][noise][margin_joints][1]])}')

    min_cost = min([min(np.array(costs[c][horizon][noise][margin_joints][0])[completed]) for c in cont_names])
    for c in cont_names:
        print(f'Controller {c}  min cost = {min(np.array(costs[c][horizon][noise][margin_joints][0])[completed])}')

    return completed, min_cost

def plot_cost_noise_level(alpha, horizon,margin_joints_index):
    for c in cont_names:
        costs[c] = {}
        costs[c][horizon] = {}
        for noise in noises:
            costs[c][horizon][noise]= {}                    
            data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{horizon}hor_{int(alpha)}sm_noise{noise}_control_noise{0.0}_q_collision_margins_{margin_joints[margin_joints_index]}_{margin_collisions[margin_joints_index]}_scores.pkl', 'rb'))
            costs[c][horizon][noise][margin_joints[margin_joints_index]] = [data_tmp[c]['costs'],data_tmp[c]['completed_idx']]
                    
    for noise in noises:
        for c in cont_names:     
            costs[c][horizon][noise][margin_joints[margin_joints_index]].append(0)    
            costs[c][horizon][noise][margin_joints[margin_joints_index]][1], costs[c][horizon][noise][margin_joints[margin_joints_index]][2] = find_costs(horizon,noise,margin_joints[margin_joints_index],costs)
            print(f'Min cost {c} noise {noise} {costs[c][horizon][noise][margin_joints[margin_joints_index]][2]}')
            print(f'std: {np.std(np.array(costs[c][horizon][noise][margin_joints[margin_joints_index]][0])[costs[c][horizon][noise][margin_joints[margin_joints_index]][1]])}')
    
    fig, ax = plt.subplots(figsize=(10, 7))

    for i,c in enumerate(cont_names):
        costs_plot = []
        for noise in noises:
            costs_plot.append(((np.mean(np.array(costs[c][horizon][noise][margin_joints[margin_joints_index]][0])[costs[c][horizon][noise][margin_joints[margin_joints_index]][1]])/costs[c][horizon][noise][margin_joints[margin_joints_index]][2])-1)*100)
        ax.plot(noises, costs_plot, color=colors[i], marker=markers[i], label=labels[i])
    
    ax.set_xlabel("Noise [\%]",labelpad=20)
    ax.set_ylabel("Cost surplus [\%]")
    ax.legend(fancybox=True, framealpha=0.5)
    #plt.xticks(ticks=np.arange(1, len(cont_names) + 1), labels=cont_names)    
    plt.xticks(noises)
    plt.tight_layout()
    plt.title(f'Joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collisions[margin_joints_index]} m, horizon = {horizon}', fontsize=20)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}_joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collisions[margin_joints_index]} m, horizon = {horizon},cost.svg', bbox_inches='tight',transparent=True)
    plt.savefig(f'{params.DATA_DIR}noise_vs_fails_horizon{horizon}_joint margin = {margin_joints[margin_joints_index]}\%, collision margin {margin_collisions[margin_joints_index]} m, horizon = {horizon},cost.png', bbox_inches='tight',transparent=False)

    plt.show()
    plt.close()

def plot_cost_margin_level(alpha, horizon,noise):
    for c in cont_names:
        costs[c] = {}
        costs[c][horizon] = {}
        costs[c][horizon][noise]= {}                    
        for jj in range(len(margin_joints)):
            data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{horizon}hor_{int(alpha)}sm_noise{noise}_control_noise{0.0}_q_collision_margins_{margin_joints[jj]}_{margin_collisions[jj]}_scores.pkl', 'rb'))
            costs[c][horizon][noise][margin_joints[jj]] = [data_tmp[c]['costs'],data_tmp[c]['completed_idx']]
                    
    for c in cont_names:    
        for jj in range(len(margin_joints)): 
            costs[c][horizon][noise][margin_joints[jj]].append(0)    
            costs[c][horizon][noise][margin_joints[jj]][1], costs[c][horizon][noise][margin_joints[jj]][2] = find_costs(horizon,noise,margin_joints[jj],costs)
            print(f'Min cost {c} noise {noise} {costs[c][horizon][noise][margin_joints[jj]][2]}')
            print(f'std: {np.std(np.array(costs[c][horizon][noise][margin_joints[jj]][0])[costs[c][horizon][noise][margin_joints[jj]][1]])}')
    
    fig, ax = plt.subplots(figsize=(10, 7))

    x_values = np.arange(len(margin_joints))
    labels_margin = [f'{joint_m} \n {collision_m}' for (joint_m,collision_m) in zip(margin_joints,margin_collisions)]

    for i,c in enumerate(cont_names):
        costs_plot = []
        for jj in range(len(margin_joints)):
            costs_plot.append(((np.mean(np.array(costs[c][horizon][noise][margin_joints[jj]][0])[costs[c][horizon][noise][margin_joints[jj]][1]])/costs[c][horizon][noise][margin_joints[jj]][2])-1)*100)
        ax.plot(x_values, costs_plot, color=colors[i], marker=markers[i], label=labels[i])

    
    ax.set_xlabel("Joint margin [\%] and collision margin [m]",labelpad=20)
    ax.set_ylabel("Cost surplus [\%]")
    ax.legend(fancybox=True, framealpha=0.5)
    plt.xticks(ticks=x_values, labels=labels_margin)

    #plt.xticks(ticks=np.arange(1, len(cont_names) + 1), labels=cont_names)    
    #plt.xticks(noises)
    plt.tight_layout()
    plt.title(f'Noise = {noise}\%, horizon = {horizon}, cost surplus', fontsize=20)
    plt.savefig(f'{params.DATA_DIR}margin_vs_cost_horizon{horizon} noise = {noise}\%,cost.svg', bbox_inches='tight',transparent=True)
    plt.savefig(f'{params.DATA_DIR}margin_vs_cost_horizon{horizon} noise = {noise}\%,cost.png', bbox_inches='tight',transparent=False)

    plt.show()
    plt.close()









#plot_horizons_fails_noise(20,2.5)
for i in range(len(margin_joints)):
    plot_control_noise_level(20,25,i)
plot_cost_noise_level(20,25,0)

for noise in noises:
    plot__noise_margins(20,25,noise)
plot_cost_margin_level(20,25,10.0)
