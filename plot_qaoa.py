# import numpy as np
# import matplotlib.pyplot as plt
# from ours_spectral import OurSpectral
# from maxcut_pennylane import qaoa_maxcut

# n_layers = 2
# n_parameters = n_layers * 2
# n_epoch = 30

# n_gate_run = 20
# losses = []
# for i in range(n_gate_run):
#     _, bit_strings, loss = qaoa_maxcut(n_epoch, n_layers=n_layers)
#     losses.append(loss)
# np_loss = np.array(losses)


# ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
# # ours_legendre = OurSpectral(n_basis=n_parameters, basis='Fourier', n_epoch=n_epoch)
# state, prob = ours_legendre.demo_qaoa_max_cut4()
# loss_legendre = ours_legendre.losses_energy

# n = np.array(bit_strings).sum()
# samples = []
# for i in range(len(prob)):
#     samples += [i] * int(prob[i] * n)


# plt.clf()
# plt.plot(losses, label='QAOA')
# plt.plot(loss_legendre, label='Legendre')
# plt.legend(loc="upper right")
# plt.savefig("{}_{}.png".format(ours_legendre.log_dir, 'loss_qaoa'))

# xticks = range(0, 16)
# xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
# bins = np.arange(0, 17) - 0.5
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title("Ours")
# plt.xlabel("bitstrings")
# plt.ylabel("freq.")
# plt.xticks(xticks, xtick_labels, rotation="vertical")
# plt.hist(samples, bins=bins)
# plt.subplot(1, 2, 2)
# plt.title("QAOA")
# plt.xlabel("bitstrings")
# plt.ylabel("freq.")
# plt.xticks(xticks, xtick_labels, rotation="vertical")
# plt.hist(bit_strings, bins=bins)
# plt.tight_layout()
# plt.savefig("{}_{}.png".format(ours_legendre.log_dir, 'qaoa_hist'))



import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral
from grape import Grape
from maxcut_pennylane import qaoa_maxcut

# n_parameters = 4
# n_epoch = 100
# taylor_terms = 20

n_layers = 2
n_parameters = n_layers * 2
n_epoch = 30
n_runs = 3
xx = np.arange(n_epoch)

plt.clf()
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

all_losses = []
all_mean = []
all_std = []

# QAOA

n_gate_run = 20
losses = []
for i in range(n_gate_run):
    _, bit_strings, loss = qaoa_maxcut(n_epoch, n_layers=n_layers)
    losses.append(loss)
np_loss = np.array(losses)
mean_ = np_loss.mean(0)
std_ = np_loss.std(0) 
color = colors[2]
plt.plot(mean_, color=color, label='Gate Model')
plt.fill_between(xx, np.maximum(mean_-std_, -4.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)


# grape
# grape = Grape(taylor_terms=taylor_terms, n_step=n_parameters, n_epoch=n_epoch)
# losses = []
# for i in range(n_runs):
#     grape.demo_TIM()
#     losses.append(grape.losses_energy.copy())
# losses = np.array(losses)
# mean_ = losses.mean(0)
# std_ = losses.std(0) 
# color = colors[2]
# plt.plot(mean_, color=color, label='grape')
# plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
# all_mean.append(mean_)
# all_std.append(std_)

# Fourier
ours_fourier = OurSpectral(n_basis=n_parameters, basis='Fourier', n_epoch=n_epoch)
losses = []
for i in range(n_runs):
    state, prob = ours_fourier.demo_qaoa_max_cut4()
    losses.append(ours_fourier.losses_energy.copy())
losses = np.array(losses)
mean_ = losses.mean(0)
std_ = losses.std(0) 
color = colors[1]
plt.plot(mean_, color=color, label='Ours Fourier')
plt.fill_between(xx, np.maximum(mean_-std_, -4.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)


# Legendre
ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
losses = []
for i in range(n_runs):
    state, prob = ours_legendre.demo_qaoa_max_cut4()
    losses.append(ours_legendre.losses_energy.copy())
losses = np.array(losses)
mean_ = losses.mean(0)
std_ = losses.std(0) 
color = colors[0]
plt.plot(mean_, color=color, label='Ours Legendre')
plt.fill_between(xx, np.maximum(mean_-std_, -4.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)


plt.legend(loc="upper right")
plt.savefig("{}curve_{}.png".format(ours_legendre.log_dir, 'QAOA'))
plt.yscale('log')
plt.savefig("{}curve_{}.png".format(ours_legendre.log_dir, 'QAOA_log'))