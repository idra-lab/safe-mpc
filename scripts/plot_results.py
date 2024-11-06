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
# ax.scatter(scores['htwa']['score'], scores['htwa']['fails'], color='darkgreen', marker='s', label='HTWA')
ax.scatter(scores['stwa']['score'], scores['stwa']['fails'], color='green', marker='^', label='STWA')
ax.scatter(scores['st']['score'], scores['st']['fails'], color='limegreen', marker='*', label='ST')
ax.set_xlabel(r"Cost surplus (\%)")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.savefig('metrics.pdf', bbox_inches='tight')
plt.close()

horizons = [20, 25, 30, 35, 40, 45]
cont_names = ['naive', 'zerovel', 'st', 'stwa', 'receding']
scores_mh = {}

for c in cont_names:
    scores_mh[c] = {}
    scores_mh[c]['score'] = []
    scores_mh[c]['fails'] = []
    for h in horizons:
        data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{h}hor_scores.pkl', 'rb'))
        scores_mh[c]['score'].append(data_tmp[c]['score'])
        scores_mh[c]['fails'].append(data_tmp[c]['fails'])


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(horizons, scores_mh['naive']['score'], color='red', marker='o', label='Naive')
ax.plot(horizons, scores_mh['zerovel']['score'], color='blue', marker='x', label='Zerovel')
ax.plot(horizons, scores_mh['st']['score'], color='limegreen', marker='*', label='ST')
ax.plot(horizons, scores_mh['stwa']['score'], color='green', marker='^', label='STWA')
# ax.plot(horizons, scores_mh['htwa']['score'], color='darkgreen', marker='s', label='HTWA')
ax.plot(horizons, scores_mh['receding']['score'], color='purple', marker='<', label='Receding')
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Cost surplus (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig('horizons_vs_score.pdf', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(horizons, scores_mh['naive']['fails'], color='red', marker='o', label='Naive')
ax.plot(horizons, scores_mh['zerovel']['fails'], color='blue', marker='x', label='Zerovel')
ax.plot(horizons, scores_mh['st']['fails'], color='limegreen', marker='*', label='ST')
ax.plot(horizons, scores_mh['stwa']['fails'], color='green', marker='^', label='STWA')
# ax.plot(horizons, scores_mh['htwa']['fails'], color='darkgreen', marker='s', label='HTWA')
ax.plot(horizons, scores_mh['receding']['fails'], color='purple', marker='<', label='Receding')
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig('horizons_vs_failed.pdf', bbox_inches='tight')
plt.close()


alphas = [2., 5., 10., 15., 30., 50.]
scores_ma = {}
for c in ['stwa', 'receding']:
    scores_ma[c] = {}
    scores_ma[c]['score'] = []
    scores_ma[c]['fails'] = []
    for a in alphas:
        data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{int(a)}_scores.pkl', 'rb'))
        scores_ma[c]['score'].append(data_tmp[c]['score'])
        scores_ma[c]['fails'].append(data_tmp[c]['fails'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(alphas, scores_ma['stwa']['fails'], color='green', marker='^', label='STWA')
# ax.plot(alphas, scores_ma['htwa']['fails'], color='darkgreen', marker='s', label='HTWA')
ax.plot(alphas, scores_ma['receding']['fails'], color='purple', marker='s', label='Receding')
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig('alphas_vs_failed.pdf', bbox_inches='tight')
plt.close()