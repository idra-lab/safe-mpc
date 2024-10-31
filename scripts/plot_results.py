import pickle
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
import matplotlib as mpl
import matplotlib.pyplot as plt

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
model = AdamModel(params, n_dofs=4)
scores = pickle.load(open(f'{params.DATA_DIR}{model_name}_scores.pkl', 'rb'))

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(scores['naive']['score'], scores['naive']['fails'], color='tomato', marker='o', label='Naive')
ax.scatter(scores['zerovel']['score'], scores['zerovel']['fails'], color='mediumblue', marker='x', label='Zerovel')
ax.scatter(scores['receding']['score'], scores['receding']['fails'], color='navy', marker='o', label='Receding')
ax.scatter(scores['htwa']['score'], scores['htwa']['fails'], color='darkgreen', marker='s', label='HTWA')
ax.scatter(scores['stwa']['score'], scores['stwa']['fails'], color='green', marker='^', label='STWA')
ax.scatter(scores['st']['score'], scores['st']['fails'], color='limegreen', marker='*', label='ST')
ax.set_xlabel(r"Cost surplus (\%)")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.savefig('metrics.pdf', bbox_inches='tight')
plt.close()

horizons = [20, 25, 30, 35, 40, 45]

######### NAIVE #########
# Horizon: 45, Completed task: 61, Collisions: 17                                                                                                                                                          
# Horizon: 40, Completed task: 20, Collisions: 24                                                                                                                                                          
# Horizon: 35, Completed task: 9, Collisions: 28                                                                                                                                                           
# Horizon: 30, Completed task: 6, Collisions: 32                                                                                                                                                           
# Horizon: 25, Completed task: 1, Collisions: 44                                                                                                                                                           
# Horizon: 20, Completed task: 0, Collisions: 57
#########################
naive_comp = [0, 1, 6, 9, 20, 61]
naive_coll = [57, 44, 32, 28, 24, 17]

######### ZEROLVEL #########
# Horizon: 45, Completed task: 69, Collisions: 0
# Horizon: 40, Completed task: 69, Collisions: 0                                                                                                                                                           
# Horizon: 35, Completed task: 68, Collisions: 0                                                                                                                                                           
# Horizon: 30, Completed task: 62, Collisions: 0
# Horizon: 25, Completed task: 12, Collisions: 0
# Horizon: 20, Completed task: 1, Collisions: 0
zerovel_comp = [1, 12, 62, 68, 69, 69]
zerovel_coll = [0, 0, 0, 0, 0, 0]

######### RECEDING #########
# Horizon: 45, Completed task: 65, Collisions: 10
# Horizon: 40, Completed task: 29, Collisions: 10
# Horizon: 35, Completed task: 19, Collisions: 12
# Horizon: 30, Completed task: 6, Collisions: 13
# Horizon: 25, Completed task: 1, Collisions: 22
# Horizon: 20, Completed task: 0, Collisions: 30
receding_comp = [0, 1, 6, 19, 29, 65] 
receding_coll = [30, 22, 13, 12, 10, 10]  

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(horizons, naive_comp, color='red', marker='o', label='Naive')
ax.plot(horizons, zerovel_comp, color='blue', marker='x', label='Zerovel')
ax.plot(horizons, receding_comp, color='green', marker='s', label='Receding')
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Task completed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig('horizons_vs_score.pdf', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(horizons, naive_coll, color='red', marker='o', label='Naive')
ax.plot(horizons, receding_coll, color='green', marker='s', label='Receding')
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig('horizons_vs_failed.pdf', bbox_inches='tight')
plt.close()