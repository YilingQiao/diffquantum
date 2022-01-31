import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.signal import savgol_filter

n_epoch = 200
# data_dir = "logs/text/icml22_qaoa_ablation/"
data_dir = "logs/text/icml22_qaoa/"

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

is_std = True

methods = {}
for file_name in files:
    if file_name.split('.')[-1] != 'txt':
        continue
    parts = file_name.split("_")
    method = ''.join(parts[:-1])

    losses = []
    print(join(data_dir, file_name))
    with open(join(data_dir, file_name), 'r') as f:
        lines = f.readlines()
        flag_start_loss = False
        for line in lines:
            l = line.rstrip()
            if flag_start_loss:
                losses.append(float(l.split()[-1]))
            else:
                if l[0] == '!':
                    flag_start_loss = True
    losses = losses[:n_epoch]
    if method not in methods.keys():
        methods[method] = [losses]
    else:
        methods[method] += [losses]

# plot
colors = [
    [0, 0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330],
    [0.6350, 0.0780, 0.1840]
]
name_color_map = {
    'Ours': 0,
    'Ours720': 0,
    'Ours-Sample': 1,
    'Ours360': 1,
    'Circuit': 2,
    'GRAPE': 2,
    'CMAES': 3,
    'SLSQP': 4,
    'Finite-Diff': 6,
    # 'Finite-Diff-Sample': 6,
}


params = {'legend.fontsize': 25,
          'figure.figsize': (12, 9),
         'axes.labelsize': 30,
         'axes.titlesize': 30,
         'xtick.labelsize':30,
         'ytick.labelsize':30}
plt.rcParams.update(params)
plt.grid(alpha=0.3)

i_m = 0

plt_list = {}
curr_m = 0
for method, num in name_color_map.items():
    if method in methods.keys():
        all_losses = np.array(methods[method])
        mean_ = all_losses.mean(0)
        std_ = all_losses.std(0)
        plt_list[method] = [colors[num], mean_, std_]
        curr_m = num + 1

for method, loss_list in methods.items():
    if method not in plt_list.keys():
        all_losses = np.array(methods[method])
        mean_ = all_losses.mean(0)
        std_ = all_losses.std(0)
        plt_list[method] = [colors[curr_m], mean_, std_]
        curr_m = num + 1

plt.clf()
xx = np.arange(n_epoch)
for method, loss_list in plt_list.items():
    
    mean_ = plt_list[method][1]
    std_ = plt_list[method][2]
    mean_ = savgol_filter(mean_, 51, 3)
    std_ = savgol_filter(std_, 51, 3)
    # if method in ['QAOA']:
    #     mean_ = savgol_filter(mean_, 51, 3)
    #     std_ = savgol_filter(std_, 51, 3)
    # else:
    #     mean_ = savgol_filter(mean_, 51, 3)
    #     std_ = savgol_filter(std_, 51, 3)
    # # if method in ['CMAES', 'SLSQP']:
    # #     for k in range(5):
    # #         mean_ = savgol_filter(mean_, 51, 3)
    # #         std_ = savgol_filter(std_, 51, 3)
    # # else:
    # #     mean_ = savgol_filter(mean_, 51, 3)
    # #     std_ = savgol_filter(std_, 51, 3)
    # print()
    print(method, mean_.shape, mean_[-1])

    color = plt_list[method][0]
    plt.plot(mean_, color=color, label=method, linewidth=4)
    if is_std:
        plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
    # all_mean.append(mean_)
    # all_std.append(std_)

    i_m += 1

# plt.legend(loc="right")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale('log')
plt.grid()


plt.savefig("{}{}.png".format(data_dir, __file__.split('.')[0]))
plt.show()