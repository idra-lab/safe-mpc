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
scores = pickle.load(open(f'{params.DATA_DIR}{model_name}_{hor}hor_{int(alpha)}sm_scoresfalse.pkl', 'rb'))

# cont_names = ['naive', 'zerovel', 'st', 'terminal', 'htwa', 'receding']
# cont_names = ['receding','receding_analytic_set']
# cont_names = ['naive', 'zerovel',  'st', 'st_analytic', 'htwa', 'htwa_analytic', 'receding','receding_analytic','parallel','parallel_analytic']
# cont_names = ['naive', 'zerovel',  'st', 'htwa', 'receding','parallel']
cont_names = ['naive','zerovel', 'st', 'htwa','receding','parallel']

#cont_names = cont_names[2:]
colors = ['tomato', 'mediumblue',  'darkorange', 'limegreen', 'darkgreen', 'purple','coral','peru']
markers = ['o', 'x', '*', 's', '^', '>','H','D']
labels = ['Naive', 'Zerovel', 'ST', 'HTWA', 'Receding' ,'Parallel']

# fig, ax = plt.subplots(figsize=(10, 7))
# for i, c in enumerate(cont_names):
#     ax.scatter(scores[c]['score'], scores[c]['fails'], color=colors[i], marker=markers[i], label=labels[i])
# ax.set_xlabel(r"Cost surplus (\%)")
# ax.set_ylabel(r"Task failed (\%)")
# ax.legend(fancybox=True, framealpha=0.8)
# #plt.title(f'Comparison with horizon {hor} and alpha {alpha}\%')
# plt.savefig(f'{params.DATA_DIR}metrics.pdf', bbox_inches='tight',transparent=True)
# plt.savefig(f'{params.DATA_DIR}metrics_transparent.png', bbox_inches='tight',transparent=True)

# plt.show()
plt.close()

horizons = [15, 20, 25, 30, 35, 40, 45, 50]
#horizons = [40]
scores_mh = {}

for c in cont_names:
    scores_mh[c] = {}
    scores_mh[c]['score'] = []
    scores_mh[c]['fails'] = []
    for h in horizons:
        # if not('analytic' in c):
        #     data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{h}hor_{int(alpha)}sm_scoresfalse.pkl', 'rb'))
        # else:
        #     data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{h}hor_{10}sm_scoresfalse.pkl', 'rb'))

        data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{h}hor_{int(alpha)}sm_scoresfalse.pkl', 'rb'))
        
        scores_mh[c]['score'].append(data_tmp[c]['score'])
        scores_mh[c]['fails'].append(data_tmp[c]['fails'])
        if c == 'parallel2':
            print(f'score:{data_tmp["parallel2"]["score"]}')

cont_names_net = cont_names # + [name for name in cont_names[1:] if not('analytic' in name ) ]
cont_names_analytic = cont_names[:1] + [name for name in cont_names[1:] if 'analytic' in name]

print(cont_names_net)
print(cont_names_analytic)

fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_net):
    ax.plot(horizons, scores_mh[c]['score'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Cost surplus (\%)")
ax.legend(fancybox=True, framealpha=0.8)
plt.xticks(horizons)
plt.tight_layout()
#plt.title(f'Comparison alpha={alpha}\%, network set')
plt.savefig(f'{params.DATA_DIR}horizons_vs_score_net.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_score_net_transparent.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_score_net.png', bbox_inches='tight')


plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_analytic):
    ax.plot(horizons, scores_mh[c]['score'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Cost surplus (\%)")
ax.legend(fancybox=True, framealpha=0.8)
plt.xticks(horizons)
plt.tight_layout()
#plt.title(f'Comparison alpha={alpha}\%, analytic set')
plt.savefig(f'{params.DATA_DIR}horizons_vs_score_analytic.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_score_analytic_transparent.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_score_analytic.png', bbox_inches='tight')


plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_net):
    scores = np.array(scores_mh[c]['fails'])
    # scores = np.where(np.abs(scores) < 1e-2 , 1e-1, scores)
    
    ax.plot(horizons, scores/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Task failed (\%)")
custom_ticks = [0,10, 20, 30, 40, 50, 60, 70]
plt.yticks(custom_ticks)
plt.xticks(horizons)
ax.legend(fancybox=True, framealpha=0.8)
# plt.yscale("log")
# plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_log_formatter))
plt.tight_layout()
plt.title(f'Safety margin alpha={alpha}\%')
plt.savefig(f'{params.DATA_DIR}horizons_vs_failed_net_{alpha}.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_failed_net_transparent_{alpha}.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_failed_net_{alpha}.png', bbox_inches='tight')


plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_analytic):
    ax.plot(horizons, np.array(scores_mh[c]['fails'])/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel("Horizons (num of nodes)")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.8)
plt.tight_layout()
plt.xticks(horizons)
#plt.title(f'Comparison alpha={alpha}\%, analytic set')
plt.savefig(f'{params.DATA_DIR}horizons_vs_failed_analytic.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_failed_analytic_transparent.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}horizons_vs_failed_analytic.png', bbox_inches='tight')



plt.show()
plt.close()

alphas = [10., 20., 30., 40., 50.]
cont_names, colors, markers, labels = cont_names[1:], colors[1:], markers[1:], labels[1:]
scores_ma = {}
for c in cont_names:
    scores_ma[c] = {}
    scores_ma[c]['score'] = []
    scores_ma[c]['fails'] = []
    for a in alphas:
        # if not('analytic' in c):
        #     data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{hor}hor_{int(a)}sm_scoresfalse.pkl', 'rb'))
        # else:
        #     data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{hor}hor_{10}sm_scoresfalse.pkl', 'rb'))
        
        data_tmp = pickle.load(open(f'{params.DATA_DIR}{model_name}_{hor}hor_{int(a)}sm_scoresfalse.pkl', 'rb'))

        scores_ma[c]['score'].append(data_tmp[c]['score'])
        scores_ma[c]['fails'].append(data_tmp[c]['fails'])


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_net[1:]):
    ax.plot(alphas, scores_ma[c]['score'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"Cost surplus (\%)")
ax.legend(fancybox=True, framealpha=0.8)
plt.tight_layout()
#plt.title(f'Comparison with horizon {hor}, network set')
plt.savefig(f'{params.DATA_DIR}alphas_vs_score_net.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_score_net_transparent.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_score_net.png', bbox_inches='tight')


plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_net[1:]):
    ax.plot(alphas, np.array(scores_ma[c]['fails'])/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.8)
plt.tight_layout()
#plt.title(f'Comparison with horizon {hor}, network set')
plt.savefig(f'{params.DATA_DIR}alphas_vs_failed_net.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_failed_net_transparent.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_failed_net.png', bbox_inches='tight')

plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_analytic[2:]):
    ax.plot(alphas, scores_ma[c]['score'], color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"Cost surplus (\%)")
ax.legend(fancybox=True, framealpha=0.8)
plt.tight_layout()
plt.xticks(horizons)
#plt.title(f'Comparison with horizon {hor}, analytic set')
plt.savefig(f'{params.DATA_DIR}alphas_vs_score_analytic.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_score_analytic_transparent.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_score_analytic.png', bbox_inches='tight')

plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cont_names_analytic[2:]):
    ax.plot(alphas, np.array(scores_ma[c]['fails'])/(params.test_num/100), color=colors[i], marker=markers[i], label=labels[i])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"Task failed (\%)")
ax.legend(fancybox=True, framealpha=0.8)
plt.tight_layout()
plt.xticks(horizons)
#plt.title(f'Comparison with horizon {hor}, analytic set')
plt.savefig(f'{params.DATA_DIR}alphas_vs_failed_analytic.svg', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_failed_analytic_transparent.png', bbox_inches='tight',transparent=True)
plt.savefig(f'{params.DATA_DIR}alphas_vs_failed_analytic.png', bbox_inches='tight')

plt.show()
plt.close()