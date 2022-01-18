import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.signal import savgol_filter

n_epoch = 1000
data_dir = "logs/text/icml22_h2/"

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

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
    [0, 0.45, 0.74],
    [0.85, 0.33, 0.1],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880]
]


params = {'legend.fontsize': 25,
          'figure.figsize': (12, 9),
         'axes.labelsize': 30,
         'axes.titlesize': 30,
         'xtick.labelsize':30,
         'ytick.labelsize':30}
plt.rcParams.update(params)
plt.grid(alpha=0.3)

i_m = 0

plt.clf()
xx = np.arange(n_epoch)
for method, loss_list in methods.items():
    all_losses = np.array(loss_list)
    mean_ = all_losses.mean(0)
    std_ = all_losses.std(0)
    if method in ['CMAES', 'SLSQP']:
        for k in range(5):
            mean_ = savgol_filter(mean_, 51, 3)
            std_ = savgol_filter(std_, 51, 3)
    else:
        mean_ = savgol_filter(mean_, 51, 3)
        std_ = savgol_filter(std_, 51, 3)
    print(method, mean_.shape)

    color = colors[i_m]
    plt.plot(mean_, color=color, label=method, linewidth=4)
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

exit()




mean_ = savgol_filter(all_mean[0], 51, 3)
std_ = savgol_filter(all_std[0], 51, 3) 
color = colors[2]
plt.plot(mean_, color=color, label='Gate model', linewidth=4)
plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)


mean_ = all_mean[1]
std_ = all_std[1] 
color = colors[1]
plt.plot(mean_, color=color, label='Ours Fourier', linewidth=4)
plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)


mean_ = all_mean[2]
std_ = all_std[2] 
color = colors[0]
plt.plot(mean_, color=color, label='Ours Legendre', linewidth=4)
plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)

plt.legend(loc="right")
plt.savefig("{}curve_{}.png".format(ours_legendre.log_dir, 'QAOA'))
plt.yscale('log')
plt.savefig("{}curve_{}.png".format(ours_legendre.log_dir, 'QAOA_log'))
# plt.savefig("{}_{}.png".format(ours_legendre.log_dir, 'losses_control_log'))
