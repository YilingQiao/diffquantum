# import numpy as np
# import matplotlib.pyplot as plt
# from ours_spectral import OurSpectral
# from grape import Grape

# n_parameters = 6
# n_epoch = 200
# grape = Grape(taylor_terms=20, n_step=n_parameters, n_epoch=n_epoch)
# grape.demo_TIM()
# loss_grape = grape.losses_energy


# ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch)
# ours_legendre.demo_TIM()
# loss_legendre = ours_legendre.losses_energy


# ours_fourier = OurSpectral(n_basis=n_parameters, basis='Fourier', n_epoch=n_epoch)
# ours_fourier.demo_TIM()
# loss_fourier = ours_fourier.losses_energy

# ground_energy = -2.15882092
# plt.clf()
# plt.plot(np.array(loss_grape) - ground_energy, label='grape')
# plt.plot(np.array(loss_fourier) - ground_energy, label='Fourier')
# plt.plot(np.array(loss_legendre) - ground_energy, label='Legendre')
# plt.legend(loc="upper right")
# plt.savefig("{}_{}.png".format(grape.log_dir, 'losses_TIM'))
# plt.yscale('log')
# plt.savefig("{}_{}.png".format(grape.log_dir, 'losses_TIM_log'))


import numpy as np
import matplotlib.pyplot as plt
from ours_spectral import OurSpectral
from grape import Grape

n_runs = 3
n_parameters = 4
n_epoch = 300
taylor_terms = 20
lr = 5e-2


plt.clf()
xx = np.arange(n_epoch)
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


# Legendre
ours_legendre = OurSpectral(n_basis=n_parameters, basis='Legendre', n_epoch=n_epoch, lr=lr)
ours_legendre.demo_TIM()
min_energy = ours_legendre.min_energy
losses = []
for i in range(n_runs):
    ours_legendre.demo_TIM()
    losses.append(ours_legendre.losses_energy.copy())
losses = np.array(losses) - min_energy
mean_ = losses.mean(0)
std_ = losses.std(0) 
color = colors[0]
plt.plot(mean_, color=color, label='Ours Legendre')
plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)

# grape
grape = Grape(taylor_terms=taylor_terms, n_step=n_parameters, n_epoch=n_epoch, lr=lr)
losses = []
for i in range(n_runs):
    grape.demo_TIM()
    losses.append(grape.losses_energy.copy())
losses = np.array(losses) - min_energy
mean_ = losses.mean(0)
std_ = losses.std(0) 
color = colors[2]
plt.plot(mean_, color=color, label='grape')
plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)

# Fourier
ours_fourier = OurSpectral(n_basis=n_parameters, basis='Fourier', n_epoch=n_epoch, lr=lr)
losses = []
for i in range(n_runs):
    ours_fourier.demo_TIM()
    losses.append(ours_fourier.losses_energy.copy())
losses = np.array(losses) - min_energy
mean_ = losses.mean(0)
std_ = losses.std(0) 
color = colors[1]
plt.plot(mean_, color=color, label='Ours Fourier')
plt.fill_between(xx, np.maximum(mean_-std_, mean_/3.0), mean_+std_, color=color, alpha=0.2)
all_mean.append(mean_)
all_std.append(std_)


plt.legend(loc="upper right")
plt.savefig("{}curve_{}.png".format(grape.log_dir, 'TIM'))
plt.yscale('log')
plt.savefig("{}curve_{}.png".format(grape.log_dir, 'TIM_log'))
# plt.savefig("{}_{}.png".format(grape.log_dir, 'losses_control_log'))