import pickle
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
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
# hor = args['horizon']
# alpha = args['alpha']
params = Parameters(model_name, rti=True)
model = AdamModel(params, n_dofs=4)
scores = pickle.load(open(f'{params.DATA_DIR}{model_name}_45hor_10sm_scores.pkl', 'rb'))
cont_names = ['naive', 'zerovel', 'st', 'terminal', 'htwa', 'receding']
colors = ['tomato', 'mediumblue', 'limegreen', 'darkorange', 'darkgreen', 'purple']
markers = ['o', 'x', '*', 's', '^', '>']
labels = ['Naive', 'Zerovel', 'ST', 'Terminal', 'HTWA', 'Receding']

fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names):
    ax.scatter(scores[c]['score'], scores[c]['fails'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel(r"Cost surplus (\%)")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.savefig(f'{params.DATA_DIR}metrics.pdf', bbox_inches='tight')
plt.close()

horizons = [20, 25, 30, 35, 40, 45]
scores_mh = {}
for c in cont_names:
    scores_mh[c] = {}
    scores_mh[c]['score'] = []
    scores_mh[c]['fails'] = []
    for h in horizons:
        data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{h}hor_10sm_scores.pkl', 'rb'))
        scores_mh[c]['score'].append(data_tmp[c]['score'])
        scores_mh[c]['fails'].append(data_tmp[c]['fails'])


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names):
    ax.plot(horizons, scores_mh[c]['score'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Cost surplus (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig(f'{params.DATA_DIR}horizons_vs_score.pdf', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names):
    ax.plot(horizons, scores_mh[c]['fails'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig(f'{params.DATA_DIR}horizons_vs_failed.pdf', bbox_inches='tight')
plt.close()


alphas = [10., 20., 30., 40., 50.]
cont_names, colors, markers, labels = cont_names[2:], colors[2:], markers[2:], labels[2:]
scores_ma = {}
for c in cont_names:
    scores_ma[c] = {}
    scores_ma[c]['score'] = []
    scores_ma[c]['fails'] = []
    for a in alphas:
        data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_45hor_{int(a)}sm_scores.pkl', 'rb'))
        scores_ma[c]['score'].append(data_tmp[c]['score'])
        scores_ma[c]['fails'].append(data_tmp[c]['fails'])

fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names):
    ax.plot(alphas, scores_ma[c]['score'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"Cost surplus (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig(f'{params.DATA_DIR}alphas_vs_score.pdf', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names):
    ax.plot(alphas, scores_ma[c]['fails'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig(f'{params.DATA_DIR}alphas_vs_failed.pdf', bbox_inches='tight')
plt.close()